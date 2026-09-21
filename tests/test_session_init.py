from pathlib import Path
import json

import pytest

from local_agent.agent.session_init import init_session
from local_agent.agent.prompts import PromptRenderer
from local_agent.agent.validators import ValidationError
from local_agent.llm.fake import FakeLLMClient


pytestmark = pytest.mark.unit


def test_init_session_writes_llm_draft_files(tmp_path):
    session = tmp_path / "demo"
    _write_user_supplied_data(session)
    llm = FakeLLMClient([_new_session_response()])

    written = init_session(session, llm, PromptRenderer())

    assert session.joinpath("inputs", "user_input.yaml").exists()
    assert session.joinpath("inputs", "data.csv").exists()
    assert session.joinpath("generated", "user_model.py").exists()
    assert session.joinpath("generated").is_dir()
    assert session.joinpath("outputs").is_dir()
    assert len(written) == 3

    user_model = session.joinpath("generated", "user_model.py").read_text()
    assert "k = trainable_parameters['k']" in user_model
    assert "y = y[0]" in user_model
    assert "dydt = -k * y" in user_model


def test_init_session_reports_missing_inputs_without_writing_model(tmp_path):
    session = tmp_path / "demo"
    session.mkdir()
    (session / "user_info.txt").write_text("fit dy/dt = -k*y")
    llm = FakeLLMClient(
        [
            json.dumps(
                {
                    "missing_inputs": ["dataset CSV in inputs/", "parameter range for k"],
                    "review": "",
                    "filename_data": "",
                    "parameters": [],
                    "states": [],
                    "user_info_txt": "",
                }
            )
        ]
    )

    with pytest.raises(ValidationError, match="dataset CSV"):
        init_session(session, llm, PromptRenderer())

    assert not session.joinpath("inputs", "user_input.yaml").exists()
    assert not session.joinpath("generated", "user_model.py").exists()


def test_init_session_repairs_invalid_structured_spec(tmp_path):
    session = tmp_path / "demo"
    _write_user_supplied_data(session)
    llm = FakeLLMClient(
        [
            _new_session_response(rhs="bad_name * y"),
            _new_session_response(),
        ]
    )

    init_session(session, llm, PromptRenderer())

    assert len(llm.requests) == 2
    assert "repair_new_session" in session.joinpath(
        "generated", "agent_logs", "llm_calls.jsonl"
    ).read_text()
    assert "dydt = -k * y" in session.joinpath("generated", "user_model.py").read_text()


def test_init_session_rejects_unsafe_expression(tmp_path):
    session = tmp_path / "demo"
    _write_user_supplied_data(session)
    llm = FakeLLMClient([_new_session_response(rhs="dataset[:, 0]")])

    with pytest.raises(ValidationError, match="scalar formulas"):
        init_session(session, llm, PromptRenderer(), workflow_config=_no_repair_config())


def test_init_session_excludes_generated_outputs_but_keeps_input_yaml_context(tmp_path):
    session = tmp_path / "demo"
    _write_user_supplied_data(session)
    session.joinpath("inputs", "user_input.yaml").write_text("model: old-generated-yaml\n")
    generated = session / "generated"
    generated.mkdir()
    generated.joinpath("user_model.py").write_text("# old generated model")
    logs = generated / "agent_logs"
    logs.mkdir()
    logs.joinpath("llm_calls.jsonl").write_text("old generated log")
    llm = FakeLLMClient([_new_session_response()])

    init_session(session, llm, PromptRenderer(), overwrite=True)

    prompt_context = llm.requests[0][1].content
    assert "old-generated-yaml" in prompt_context
    assert "old generated model" not in prompt_context
    assert "old generated log" not in prompt_context
    assert "FILE: inputs/data.csv" in prompt_context
    assert "FILE: inputs/user_info.txt" in prompt_context
    assert "FILE: inputs/user_input.yaml" in prompt_context
    assert session.joinpath("generated", "agent_logs", "llm_calls.jsonl").read_text().count(
        '"step": "new_session"'
    ) == 1


def test_init_session_summarizes_csv_context(tmp_path):
    session = tmp_path / "demo"
    inputs = session / "inputs"
    inputs.mkdir(parents=True)
    rows = ["time,y"] + [f"{index},{index * 0.5}" for index in range(20)]
    inputs.joinpath("data.csv").write_text("\n".join(rows) + "\n")
    inputs.joinpath("user_info.txt").write_text(
        "Fit dy/dt = -k*y against data.csv. k in [0.001, 10], y0=1.0."
    )
    llm = FakeLLMClient([_new_session_response()])

    init_session(session, llm, PromptRenderer())

    prompt_context = llm.requests[0][1].content
    assert "<csv summary: 20 data rows>" in prompt_context
    assert "time,y" in prompt_context
    assert "4,2.0" in prompt_context
    assert "19,9.5" not in prompt_context


def test_init_session_preserves_existing_detailed_fileset(tmp_path):
    session = tmp_path / "demo"
    inputs = session / "inputs"
    generated = session / "generated"
    inputs.mkdir(parents=True)
    generated.mkdir()
    yaml = Path("tests/vanderpol_session/inputs/user_input.yaml").read_text()
    data = Path("tests/vanderpol_session/inputs/vanderpol_data.csv").read_text()
    model = _valid_user_model()
    inputs.joinpath("user_input.yaml").write_text(yaml)
    inputs.joinpath("vanderpol_data.csv").write_text(data)
    generated.joinpath("user_model.py").write_text(model)
    llm = FakeLLMClient([_vanderpol_session_response()])

    written = init_session(session, llm, PromptRenderer())

    assert inputs.joinpath("user_input.yaml").read_text() == yaml
    assert inputs.joinpath("vanderpol_data.csv").read_text() == data
    assert not inputs.joinpath("data.csv").exists()
    assert written == [inputs / "user_info.txt", generated / "pfit_new_review.txt"]


def test_init_session_renders_fixed_helpers_and_derived_observables(tmp_path):
    session = tmp_path / "derived"
    inputs = session / "inputs"
    inputs.mkdir(parents=True)
    inputs.joinpath("derived.csv").write_text("0.0,2.0\n1.0,1.0\n")
    inputs.joinpath("user_info.txt").write_text(
        "Fit dx/dt = -k*x. Data observes z = scale*x. scale is fixed at 2."
    )
    llm = FakeLLMClient(
        [
            json.dumps(
                {
                    "missing_inputs": [],
                    "review": "Fit a derived observable z = scale*x.",
                    "filename_data": "derived.csv",
                    "parameters": [
                        {
                            "name": "k",
                            "min_value": 0.001,
                            "max_value": 10.0,
                            "logscale": True,
                        }
                    ],
                    "fixed_parameters": [{"name": "scale", "value": 2.0}],
                    "helper_functions": [
                        "def decay(rate, value):\n    return -rate * value"
                    ],
                    "states": [
                        {
                            "name": "x",
                            "initial_value": 1.0,
                            "rhs": "decay(k, x)",
                            "observed_column": None,
                        }
                    ],
                    "observables": [
                        {
                            "name": "z",
                            "expression": "scale * x",
                            "observed_column": 0,
                        }
                    ],
                    "loss_body": "",
                    "user_info_txt": "Local pfit-new draft generated from supplied files.",
                }
            )
        ]
    )

    init_session(session, llm, PromptRenderer())

    user_input = inputs.joinpath("user_input.yaml").read_text()
    user_model = session.joinpath("generated", "user_model.py").read_text()
    assert "- {name: z, observes: z}" in user_input
    assert "- {name: scale, value: 2.0}" in user_input
    assert "- {name: z}" in user_input
    assert "def decay(rate, value):" in user_model
    assert "def _observables(solution, trainable_parameters, fixed_parameters):" in user_model
    assert "'z': scale * x" in user_model
    assert "observables['z'] - dataset[:, 0]" in user_model
    assert "writeout_array = np.zeros([solution_time.shape[0], 4])" in user_model


def _write_user_supplied_data(session: Path) -> None:
    inputs = session / "inputs"
    inputs.mkdir(parents=True)
    inputs.joinpath("data.csv").write_text("0.0,1.0\n1.0,0.5\n2.0,0.25\n")
    inputs.joinpath("user_info.txt").write_text(
        "Fit dy/dt = -k*y against data.csv. k in [0.001, 10], y0=1.0."
    )


def _new_session_response(
    *,
    rhs: str = "-k * y",
) -> str:
    return json.dumps(
        {
            "missing_inputs": [],
            "review": "Fit dy/dt = -k*y to data.csv with squared-error loss.",
            "filename_data": "data.csv",
            "parameters": [
                {
                    "name": "k",
                    "min_value": 0.001,
                    "max_value": 10.0,
                    "logscale": True,
                }
            ],
            "states": [
                {
                    "name": "y",
                    "initial_value": 1.0,
                    "rhs": rhs,
                    "observed_column": 0,
                }
            ],
            "user_info_txt": "Local pfit-new draft generated from supplied files.",
        }
    )


def _vanderpol_session_response() -> str:
    return json.dumps(
        {
            "missing_inputs": [],
            "review": "Fit Van der Pol oscillator to vanderpol_data.csv.",
            "filename_data": "vanderpol_data.csv",
            "parameters": [
                {
                    "name": "mu",
                    "min_value": 0.001,
                    "max_value": 100.0,
                    "logscale": True,
                }
            ],
            "states": [
                {
                    "name": "x1",
                    "initial_value": 2.0,
                    "rhs": "x2",
                    "observed_column": 0,
                },
                {
                    "name": "x2",
                    "initial_value": 0.1,
                    "rhs": "mu * (1 - x1**2) * x2 - x1",
                    "observed_column": 1,
                },
            ],
            "user_info_txt": "Local pfit-new draft generated from supplied files.",
        }
    )


def _valid_yaml() -> str:
    return """<?yaml version="1.0" ?>
<FIT>
    <EXPERIMENT>
        <FILENAME>
            <E> FILENAME_DATA = data.csv </E>
        </FILENAME>
    </EXPERIMENT>
    <PATH>
        <E> USER_INPUT_DIR = inputs </E>
        <E> GENERATED_DIR = generated </E>
        <E> OUTPUT_DIR = outputs </E>
    </PATH>
    <MODEL>
        <TRAINABLE_PARAMETERS>
            <P> N_TRAINABLE_PARAMETERS = 1 </P>
        </TRAINABLE_PARAMETERS>
        <TRAINABLE_PARAMETER_DESCRIPTION>
            <PARAM>
                <P> PARAMETER_NAME = k </P>
                <P> MIN_VAL = 0.001 </P>
                <P> MAX_VAL = 10.0 </P>
                <P> LOGSCALE = Y </P>
            </PARAM>
        </TRAINABLE_PARAMETER_DESCRIPTION>
        <INTEGRATED_SYSTEM_DESCRIPTION>
            <VAR>
                <P> NAME = y </P>
                <P> INIT_VAL = 1.0 </P>
            </VAR>
        </INTEGRATED_SYSTEM_DESCRIPTION>
    </MODEL>
    <POPULATION_OPT>
        <SETTINGS>
            <P> NUM_PARTICLES = 16 </P>
            <P> NUM_ITERS = 5 </P>
            <P> PROCESSORS = 1 </P>
        </SETTINGS>
    </POPULATION_OPT>
    <GRADIENT_OPT>
        <SETTINGS>
            <P> NUM_ITERS = 5 </P>
            <P> STEPSIZE_RTOL = 1e-7 </P>
            <P> STEPSIZE_ATOL = 1e-9 </P>
            <P> INITIAL_TIMESTEP = 1e-6 </P>
            <P> MAX_STEPS = 10000 </P>
            <P> INIT_VALUE_LR = 1e-4 </P>
            <P> END_VALUE_LR = 1e-5 </P>
            <P> TRANSITION_STEPS_LR = 2000 </P>
            <P> DECAY_RATE_LR = 0.9 </P>
        </SETTINGS>
    </GRADIENT_OPT>
    <PLOTTING_INFO>
        <P> WRITE_RESULTS = Y </P>
    </PLOTTING_INFO>
</FIT>
"""


def _valid_user_model() -> str:
    return """import numpy as np

def user_defined_system(t, y, trainable_parameters, fixed_parameters, dataset, t_eval):
    k = trainable_parameters['k']
    y = y[0]
    dy_dt = -k * y
    return np.array([dy_dt])

def _compute_loss_problem(solution_time, solution, dataset, trainable_parameters, fixed_parameters):
    residual = solution[:, 0] - dataset[:, 0]
    return float(np.mean(residual ** 2))

def writeout_description(solution_time, solution, dataset, trainable_parameters, fixed_parameters):
    return np.column_stack((solution_time, solution[:, 0], dataset[:, 0]))
"""


def _no_repair_config():
    from local_agent.agent.config import WorkflowConfig

    return WorkflowConfig(max_repair_attempts=0)
