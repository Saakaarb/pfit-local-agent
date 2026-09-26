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


def test_init_session_assembles_split_new_session_passes(tmp_path):
    session = tmp_path / "demo"
    _write_user_supplied_data(session)
    llm = FakeLLMClient(
        [
            json.dumps(
                {
                    "missing_inputs": [],
                    "review": "Use data.csv.",
                    "filename_data": "data.csv",
                    "user_info_txt": "Dataset selected.",
                }
            ),
            json.dumps(
                {
                    "missing_inputs": [],
                    "review": "Extracted parameter.",
                    "parameters": [
                        {
                            "name": "k",
                            "min_value": 2.0e-19,
                            "max_value": 4.0e-19,
                            "logscale": True,
                        }
                    ],
                    "fixed_parameters": [{"name": "y_0", "value": 1.0}],
                    "user_info_txt": "Parameters selected.",
                }
            ),
            json.dumps(
                {
                    "missing_inputs": [],
                    "review": "Extracted state.",
                    "states": [
                        {
                            "name": "y",
                            "initial_value": 1.0,
                        }
                    ],
                    "user_info_txt": "State selected.",
                }
            ),
            json.dumps(
                {
                    "missing_inputs": [],
                    "review": "Extracted equations.",
                    "formulas": [],
                    "rhs": [{"state": "y", "expression": "-k * y"}],
                    "user_info_txt": "Equations selected.",
                }
            ),
            json.dumps(
                {
                    "missing_inputs": [],
                    "review": "Mapped observations.",
                    "observables": [
                        {"measured": "y", "simulated": "y", "expression": ""}
                    ],
                    "user_info_txt": "Observables selected.",
                }
            ),
            json.dumps(
                {
                    "missing_inputs": [],
                    "review": "Use default squared-error loss.",
                    "custom_loss": False,
                    "data_terms": [],
                    "penalties": [],
                    "user_info_txt": "Local pfit-new draft generated from supplied files.",
                }
            ),
        ]
    )

    init_session(session, llm, PromptRenderer())

    assert len(llm.requests) == 6
    user_input = session.joinpath("inputs", "user_input.yaml").read_text()
    assert "min_val: 2e-19" in user_input
    assert "max_val: 4e-19" in user_input
    assert "y_0" not in user_input
    log_text = session.joinpath("generated", "agent_logs", "llm_calls.jsonl").read_text()
    assert "new_session_dataset" in log_text
    assert "new_session_parameters" in log_text
    assert "new_session_states" in log_text
    assert "new_session_equations" in log_text
    assert "new_session_observables" in log_text
    assert "new_session_loss" in log_text


def test_init_session_split_passes_preserve_uncertainty_columns(tmp_path):
    session = tmp_path / "oregonator"
    inputs = session / "inputs"
    inputs.mkdir(parents=True)
    inputs.joinpath("data.csv").write_text(
        "time,X,Z,X_sd,Z_sd\n"
        "0.0,0.7,0.2,0.01,0.02\n"
        "1.0,0.6,0.3,0.01,0.02\n"
    )
    inputs.joinpath("user_info.txt").write_text(
        "Fit X and Z. The loss should divide residuals by the supplied standard deviations X_sd and Z_sd."
    )
    llm = FakeLLMClient(
        [
            json.dumps(
                {
                    "missing_inputs": [],
                    "review": "Use data.csv.",
                    "filename_data": "data.csv",
                    "user_info_txt": "Dataset selected.",
                }
            ),
            json.dumps(
                {
                    "missing_inputs": [],
                    "review": "Extracted parameter.",
                    "parameters": [
                        {"name": "q", "min_value": 0.001, "max_value": 1.0, "logscale": True}
                    ],
                    "fixed_parameters": [],
                    "user_info_txt": "Parameters selected.",
                }
            ),
            json.dumps(
                {
                    "missing_inputs": [],
                    "review": "Extracted states.",
                    "states": [
                        {"name": "X", "initial_value": 0.7},
                        {"name": "Z", "initial_value": 0.2},
                    ],
                    "user_info_txt": "States selected.",
                }
            ),
            json.dumps(
                {
                    "missing_inputs": [],
                    "review": "Extracted equations.",
                    "formulas": [],
                    "rhs": [
                        {"state": "X", "expression": "-q * X"},
                        {"state": "Z", "expression": "q * X"},
                    ],
                    "user_info_txt": "Equations selected.",
                }
            ),
            json.dumps(
                {
                    "missing_inputs": [],
                    "review": "Mapped observations.",
                    "observables": [
                        {"measured": "X", "simulated": "X", "expression": ""},
                        {"measured": "Z", "simulated": "Z", "expression": ""},
                    ],
                    "user_info_txt": "Observables selected.",
                }
            ),
            json.dumps(
                {
                    "missing_inputs": [],
                    "review": "Use the supplied standard deviations X_sd and Z_sd.",
                    "custom_loss": True,
                    "data_terms": [
                        {"simulated": "X", "measured": "X", "metric": "normalized_mse"},
                        {"simulated": "Z", "measured": "Z", "metric": "normalized_mse"},
                    ],
                    "penalties": [],
                    "user_info_txt": "Use X_sd and Z_sd as uncertainty weights.",
                }
            ),
        ]
    )

    init_session(session, llm, PromptRenderer())

    user_input = session.joinpath("inputs", "user_input.yaml").read_text()
    user_model = session.joinpath("generated", "user_model.py").read_text()
    assert "{name: X_sd, uncertainty_of: X}" in user_input
    assert "{name: Z_sd, uncertainty_of: Z}" in user_input
    assert "dataset[:, 2]" in user_model
    assert "dataset[:, 3]" in user_model
    assert "np.max(dataset[:, 0])" not in user_model


def test_init_session_split_passes_render_derived_loss_and_penalty(tmp_path):
    session = tmp_path / "arc"
    inputs = session / "inputs"
    inputs.mkdir(parents=True)
    inputs.joinpath("arc.csv").write_text("time,T,dTdt\n0.0,354.0,0.1\n1.0,355.0,0.2\n")
    inputs.joinpath("user_info.txt").write_text(
        "Fit c and T. r=-A*c. q=abs(h*r). dc/dt=r, dT/dt=q. "
        "T column observes T and dTdt observes q. Penalize c(tf)>0.02 by 10000."
    )
    llm = FakeLLMClient(
        [
            json.dumps(
                {
                    "missing_inputs": [],
                    "review": "Use arc.csv.",
                    "filename_data": "arc.csv",
                    "user_info_txt": "Dataset selected.",
                }
            ),
            json.dumps(
                {
                    "missing_inputs": [],
                    "review": "Parameters selected.",
                    "parameters": [
                        {"name": "A", "min_value": 0.001, "max_value": 10.0, "logscale": True},
                        {"name": "h", "min_value": 1.0, "max_value": 100.0, "logscale": True},
                    ],
                    "fixed_parameters": [],
                    "user_info_txt": "Parameters selected.",
                }
            ),
            json.dumps(
                {
                    "missing_inputs": [],
                    "review": "States selected.",
                    "states": [
                        {"name": "c", "initial_value": 1.0},
                        {"name": "T", "initial_value": 354.0},
                    ],
                    "user_info_txt": "States selected.",
                }
            ),
            json.dumps(
                {
                    "missing_inputs": [],
                    "review": "Equations selected.",
                    "formulas": [
                        {"name": "r", "expression": "-A * c"},
                        {"name": "q", "expression": "abs(h * r)"},
                    ],
                    "rhs": [
                        {"state": "c", "expression": "r"},
                        {"state": "T", "expression": "q"},
                    ],
                    "user_info_txt": "Equations selected.",
                }
            ),
            json.dumps(
                {
                    "missing_inputs": [],
                    "review": "Observables selected.",
                    "observables": [
                        {"measured": "T", "simulated": "T", "expression": ""},
                        {"measured": "dTdt", "simulated": "q", "expression": ""},
                    ],
                    "user_info_txt": "Observables selected.",
                }
            ),
            json.dumps(
                {
                    "missing_inputs": [],
                    "review": "Custom loss selected.",
                    "custom_loss": True,
                    "data_terms": [
                        {"simulated": "T", "measured": "T", "metric": "normalized_mse"},
                        {"simulated": "log_q", "measured": "log_q_obs", "metric": "normalized_mse"},
                    ],
                    "penalties": [
                        {
                            "left": "c",
                            "left_kind": "simulated",
                            "op": ">",
                            "right": 0.02,
                            "right_kind": "literal",
                            "value": 10000.0,
                            "at": "final",
                        },
                        {
                            "left": "sqrt((T - T_obs)^2 + 1e-12)",
                            "left_kind": "simulated",
                            "op": ">",
                            "right": 50,
                            "right_kind": "literal",
                            "value": 10000.0,
                            "at": "final",
                        }
                    ],
                    "user_info_txt": "Local pfit-new draft generated from supplied files.",
                }
            ),
        ]
    )

    init_session(session, llm, PromptRenderer())

    user_model = session.joinpath("generated", "user_model.py").read_text()
    assert "dcdt = -A * c" in user_model
    assert "dTdt = np.abs(h * (-A * c))" in user_model
    assert "'dTdt': np.abs(h * (-A * c))" in user_model
    assert "loss += np.mean(np.square((solution[:, 1] - dataset[:, 0]) / (np.max(dataset[:, 0]) - np.min(dataset[:, 0]) + 1e-12)))" in user_model
    assert "eps_dTdt = np.min(np.where(dataset[:, 1] > 0.0, dataset[:, 1], np.inf))" in user_model
    assert "log_sim_dTdt = np.log10(observables['dTdt'] + eps_dTdt)" in user_model
    assert "log_measured_dTdt = np.log10(dataset[:, 1] + eps_dTdt)" in user_model
    assert "loss += np.mean(np.square((log_sim_dTdt - log_measured_dTdt) / scale_log_dTdt))" in user_model
    assert "if solution[-1, 0]" not in user_model
    assert "loss += 10000.0 / (1.0 + np.exp(-np.clip(1000.0 * (solution[-1, 0] - 0.02), -60.0, 60.0)))" in user_model
    assert "np.sqrt(np.square(solution[-1, 1] - dataset[-1, 0]) + 1e-12) - 50.0" in user_model


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
            json.dumps({"states": [{"name": "y", "rhs": "-k * y"}]}),
        ]
    )

    init_session(session, llm, PromptRenderer())

    assert len(llm.requests) == 2
    log_text = session.joinpath(
        "generated", "agent_logs", "llm_calls.jsonl"
    ).read_text()
    assert "repair_new_session_expressions" in log_text
    assert "repair_new_session.user.md" not in log_text
    assert "dydt = -k * y" in session.joinpath("generated", "user_model.py").read_text()


def test_init_session_normalizes_common_math_calls_before_llm_repair(tmp_path):
    session = tmp_path / "arc"
    inputs = session / "inputs"
    inputs.mkdir(parents=True)
    inputs.joinpath("arc.csv").write_text("time,T,dTdt\n0.0,354.0,0.001\n1.0,354.001,0.001\n")
    inputs.joinpath("user_info.txt").write_text("Fit ARC-style T and dTdt.")
    response = {
        "missing_inputs": [],
        "review": "ARC-style mixed observed state and derived observable.",
        "filename_data": "arc.csv",
        "parameters": [
            {"name": "A", "min_value": 0.001, "max_value": 10.0, "logscale": True}
        ],
        "fixed_parameters": [{"name": "kb", "value": 1.0}],
        "helper_functions": [],
        "states": [
            {"name": "c", "initial_value": 1.0, "rhs": "-A * exp(-1 / kb) * c", "observed_column": None},
            {"name": "T", "initial_value": 354.0, "rhs": "abs(A * c)", "observed_column": 0},
        ],
        "observables": [
            {"name": "dTdt", "expression": "abs(A * c)", "observed_column": 1}
        ],
        "loss_body": "",
        "user_info_txt": "Local pfit-new draft generated from supplied files.",
    }
    llm = FakeLLMClient([json.dumps(response)])

    init_session(session, llm, PromptRenderer())

    assert len(llm.requests) == 1
    user_model = session.joinpath("generated", "user_model.py").read_text()
    assert "dcdt = -A * np.exp(-1 / kb) * c" in user_model
    assert "dTdt = np.abs(A * c)" in user_model
    assert "'dTdt': np.abs(A * c)" in user_model


def test_init_session_allows_auxiliary_uncertainty_columns(tmp_path):
    session = tmp_path / "oregonator"
    inputs = session / "inputs"
    inputs.mkdir(parents=True)
    inputs.joinpath("data.csv").write_text(
        "time,X,Z,X_sd,Z_sd\n"
        "0.0,0.7,0.2,0.01,0.02\n"
        "1.0,0.6,0.3,0.01,0.02\n"
    )
    inputs.joinpath("user_info.txt").write_text("Fit X and Z using X_sd and Z_sd.")
    response = {
        "missing_inputs": [],
        "review": "Oregonator with sigma columns.",
        "filename_data": "data.csv",
        "parameters": [
            {"name": "q", "min_value": 0.001, "max_value": 1.0, "logscale": True}
        ],
        "fixed_parameters": [],
        "helper_functions": [],
        "states": [
            {"name": "X", "initial_value": 0.7, "rhs": "-q * X", "observed_column": 0},
            {"name": "Z", "initial_value": 0.2, "rhs": "q * X", "observed_column": 1},
        ],
        "observables": [],
        "auxiliary_columns": [
            {"name": "X_sd", "observed_column": 2, "kind": "uncertainty_of", "target": "X"},
            {"name": "Z_sd", "observed_column": 3, "kind": "uncertainty_of", "target": "Z"},
        ],
        "loss_body": (
            "rx = (solution[:, 0] - dataset[:, 0]) / dataset[:, 2]\n"
            "rz = (solution[:, 1] - dataset[:, 1]) / dataset[:, 3]\n"
            "return np.sqrt(np.mean(np.square(rx)) + np.mean(np.square(rz)))"
        ),
        "user_info_txt": "Local pfit-new draft generated from supplied files.",
    }
    llm = FakeLLMClient([json.dumps(response)])

    init_session(session, llm, PromptRenderer())

    assert session.joinpath("inputs", "user_input.yaml").exists()
    assert "dataset[:, 3]" in session.joinpath("generated", "user_model.py").read_text()


@pytest.mark.parametrize("explicit_loss", [False, True])
def test_init_session_automatic_log_loss_respects_user_loss(tmp_path, explicit_loss):
    session = tmp_path / "robertson"
    inputs = session / "inputs"
    inputs.mkdir(parents=True)
    inputs.joinpath("data.csv").write_text(
        "time,y1,y3\n"
        "0.0,1.0,1e-12\n"
        "1.0,0.5,1e-6\n"
        "2.0,0.25,1e-1\n"
    )
    inputs.joinpath("user_info.txt").write_text(
        "Fit y1 and y3.\nLoss:\nCompare y1 and y3 directly with normalized squared error."
        if explicit_loss else "Fit y1 and y3."
    )
    llm = FakeLLMClient(
        [
            json.dumps(
                {
                    "missing_inputs": [],
                    "review": "Use data.csv.",
                    "filename_data": "data.csv",
                    "user_info_txt": "Dataset selected.",
                }
            ),
            json.dumps(
                {
                    "missing_inputs": [],
                    "review": "Extracted parameter.",
                    "parameters": [
                        {"name": "k", "min_value": 0.1, "max_value": 10.0, "logscale": True}
                    ],
                    "fixed_parameters": [],
                    "user_info_txt": "Parameters selected.",
                }
            ),
            json.dumps(
                {
                    "missing_inputs": [],
                    "review": "Extracted states.",
                    "states": [
                        {"name": "y1", "initial_value": 1.0},
                        {"name": "y3", "initial_value": 1e-12},
                    ],
                    "user_info_txt": "States selected.",
                }
            ),
            json.dumps(
                {
                    "missing_inputs": [],
                    "review": "Extracted equations.",
                    "formulas": [],
                    "rhs": [
                        {"state": "y1", "expression": "-k * y1"},
                        {"state": "y3", "expression": "k * y1"},
                    ],
                    "user_info_txt": "Equations selected.",
                }
            ),
            json.dumps(
                {
                    "missing_inputs": [],
                    "review": "Mapped observations.",
                    "observables": [
                        {"measured": "y1", "simulated": "y1", "expression": ""},
                        {"measured": "y3", "simulated": "y3", "expression": ""},
                    ],
                    "user_info_txt": "Observables selected.",
                }
            ),
            json.dumps(
                {
                    "missing_inputs": [],
                    "review": "Use normalized loss.",
                    "custom_loss": True,
                    "data_terms": [
                        {"simulated": "y1", "measured": "y1", "metric": "normalized_mse"},
                        {"simulated": "y3", "measured": "y3", "metric": "normalized_mse"},
                    ],
                    "penalties": [],
                    "user_info_txt": "Use normalized loss.",
                }
            ),
        ]
    )

    init_session(session, llm, PromptRenderer())

    user_model = session.joinpath("generated", "user_model.py").read_text()
    assert ("log_sim_y3" in user_model) is (not explicit_loss)
    assert ("log_measured_y3" in user_model) is (not explicit_loss)
    assert "log_sim_y1" not in user_model


def test_init_session_selects_stiff_integrator_from_problem_description(tmp_path):
    session = tmp_path / "stiff"
    inputs = session / "inputs"
    inputs.mkdir(parents=True)
    inputs.joinpath("data.csv").write_text("time,y\n0.0,1.0\n1.0,0.5\n")
    inputs.joinpath("user_info.txt").write_text("Fit a stiff chemical kinetics model.")
    response = json.loads(_new_session_response())
    response["review"] = "Stiff chemical kinetics model."
    response["parameters"] = [
        {"name": "k", "min_value": 1e-3, "max_value": 1e9, "logscale": True}
    ]
    llm = FakeLLMClient([json.dumps(response)])

    init_session(session, llm, PromptRenderer())

    user_input = session.joinpath("inputs", "user_input.yaml").read_text()
    assert "integrator: Kvaerno5" in user_input


def test_init_session_repairs_only_invalid_loss_body(tmp_path):
    session = tmp_path / "demo"
    _write_user_supplied_data(session)
    bad_response = json.loads(_new_session_response())
    bad_response["loss_body"] = "loss += 1.0"
    repaired_loss = {
        "loss_body": (
            "loss = np.mean(np.square(solution[:, 0] - dataset[:, 0]))\n"
            "return loss"
        )
    }
    llm = FakeLLMClient([json.dumps(bad_response), json.dumps(repaired_loss)])

    init_session(session, llm, PromptRenderer())

    assert len(llm.requests) == 2
    log_text = session.joinpath("generated", "agent_logs", "llm_calls.jsonl").read_text()
    assert "repair_new_session_loss_body" in log_text
    assert "repair_new_session.user.md" not in log_text
    user_model = session.joinpath("generated", "user_model.py").read_text()
    assert "dydt = -k * y" in user_model
    assert "loss = np.mean(np.square(solution[:, 0] - dataset[:, 0]))" in user_model
    assert "return loss" in user_model


def test_init_session_rejects_unsafe_expression(tmp_path):
    session = tmp_path / "demo"
    _write_user_supplied_data(session)
    llm = FakeLLMClient([_new_session_response(rhs="dataset[:, 0]")])

    with pytest.raises(ValidationError, match="scalar formulas"):
        init_session(session, llm, PromptRenderer(), workflow_config=_no_repair_config())


def test_init_session_excludes_generated_outputs_and_prior_generated_inputs(tmp_path):
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
    assert "old-generated-yaml" not in prompt_context
    assert "old generated model" not in prompt_context
    assert "old generated log" not in prompt_context
    assert "FILE: inputs/data.csv" in prompt_context
    assert "FILE: inputs/user_info.txt" in prompt_context
    assert "FILE: inputs/user_input.yaml" not in prompt_context
    assert "FILE: generated/user_model.py" not in prompt_context
    assert session.joinpath("generated", "agent_logs", "llm_calls.jsonl").read_text().count(
        '"step": "new_session_dataset"'
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
    assert "<csv summary: 20 data rows, 2 columns, header present>" in prompt_context
    assert "<csv columns: time, y>" in prompt_context
    assert "time,y" in prompt_context
    assert "4,2.0" in prompt_context
    assert "19,9.5" not in prompt_context


def test_init_session_rejects_headerless_csv(tmp_path):
    session = tmp_path / "demo"
    inputs = session / "inputs"
    inputs.mkdir(parents=True)
    inputs.joinpath("data.csv").write_text("0.0,1.0\n1.0,0.5\n")
    inputs.joinpath("user_info.txt").write_text(
        "Fit dy/dt = -k*y against data.csv. k in [0.001, 10], y0=1.0."
    )
    llm = FakeLLMClient([_new_session_response()])

    with pytest.raises(ValidationError, match="CSV must include a header row"):
        init_session(session, llm, PromptRenderer())

    assert not llm.requests


def test_init_session_rejects_observed_names_that_do_not_match_csv_header(tmp_path):
    session = tmp_path / "demo"
    _write_user_supplied_data(session)
    llm = FakeLLMClient([_new_session_response(observed_state_name="x")])

    with pytest.raises(ValidationError, match="must match the CSV header"):
        init_session(session, llm, PromptRenderer(), workflow_config=_no_repair_config())


def test_init_session_rejects_duplicate_observed_column(tmp_path):
    session = tmp_path / "demo"
    inputs = session / "inputs"
    inputs.mkdir(parents=True)
    inputs.joinpath("data.csv").write_text("time,force,displacement\n0.0,1.0,0.2\n")
    inputs.joinpath("user_info.txt").write_text("Fit dx/dt = -k*x with force and displacement.")
    response = json.loads(_new_session_response())
    response["states"][0]["observed_column"] = 1
    response["observables"] = [
        {"name": "force", "expression": "k * y", "observed_column": 0},
        {"name": "displacement", "expression": "y", "observed_column": 1},
    ]
    llm = FakeLLMClient([json.dumps(response)])

    with pytest.raises(ValidationError, match="assigned more than once"):
        init_session(session, llm, PromptRenderer(), workflow_config=_no_repair_config())


def test_init_session_does_not_use_full_spec_repair_fallback(tmp_path):
    session = tmp_path / "demo"
    inputs = session / "inputs"
    inputs.mkdir(parents=True)
    inputs.joinpath("data.csv").write_text("time,y\n0.0,1.0\n")
    inputs.joinpath("user_info.txt").write_text("Fit y.")
    response = json.loads(_new_session_response())
    response["observables"] = [
        {"name": "duplicate_y", "expression": "y", "observed_column": 0}
    ]
    llm = FakeLLMClient([json.dumps(response), _new_session_response()])

    with pytest.raises(ValidationError, match="No targeted pfit-new repair"):
        init_session(session, llm, PromptRenderer())

    assert len(llm.requests) == 1


def test_init_session_rejects_branchy_rhs_expression(tmp_path):
    session = tmp_path / "demo"
    _write_user_supplied_data(session)
    response = json.loads(_new_session_response(rhs="0 if y > 0 else -k * y"))
    llm = FakeLLMClient([json.dumps(response)])

    with pytest.raises(ValidationError, match="Python conditional expression"):
        init_session(session, llm, PromptRenderer(), workflow_config=_no_repair_config())


def test_init_session_rejects_fixed_parameter_value_drift_from_prompt(tmp_path):
    session = tmp_path / "demo"
    _write_user_supplied_data(session)
    session.joinpath("inputs", "user_info.txt").write_text(
        "Fit dy/dt = -k*y + c.\n\nFixed parameters:\nc = 2.0\n"
    )
    response = json.loads(_new_session_response(rhs="-k * y + c"))
    response["fixed_parameters"] = [{"name": "c", "value": 0.0}]
    llm = FakeLLMClient([json.dumps(response)])

    with pytest.raises(ValidationError, match="does not match user prompt value"):
        init_session(session, llm, PromptRenderer(), workflow_config=_no_repair_config())


def test_init_session_rejects_helper_function_closing_over_unknown_constant(tmp_path):
    session = tmp_path / "demo"
    _write_user_supplied_data(session)
    response = json.loads(_new_session_response(rhs="smooth(y)"))
    response["helper_functions"] = [
        "def smooth(x):\n    return 1.0 / (1.0 + np.exp(-x / width))"
    ]
    llm = FakeLLMClient([json.dumps(response)])

    with pytest.raises(ValidationError, match="helper function smooth uses unknown name: width"):
        init_session(session, llm, PromptRenderer(), workflow_config=_no_repair_config())


def test_init_session_canonicalizes_observed_columns_from_csv_header(tmp_path):
    session = tmp_path / "arc"
    inputs = session / "inputs"
    inputs.mkdir(parents=True)
    inputs.joinpath("arc.csv").write_text("time,T,dTdt\n0.0,354.0,0.001\n1.0,354.001,0.001\n")
    inputs.joinpath("user_info.txt").write_text(
        "Fit c1, c2, T with observed columns T and dTdt."
    )
    response = {
        "missing_inputs": [],
        "review": "ARC-style mixed observed state and derived observable.",
        "filename_data": "arc.csv",
        "parameters": [
            {"name": "k", "min_value": 0.001, "max_value": 10.0, "logscale": True}
        ],
        "fixed_parameters": [],
        "helper_functions": [],
        "states": [
            {"name": "c1", "initial_value": 1.0, "rhs": "-k * c1", "observed_column": None},
            {"name": "c2", "initial_value": 0.0, "rhs": "k * c1", "observed_column": None},
            {"name": "T", "initial_value": 354.0, "rhs": "k * c1", "observed_column": 1},
        ],
        "observables": [
            {"name": "dTdt", "expression": "k * c1", "observed_column": 2}
        ],
        "loss_body": "",
        "user_info_txt": "Local pfit-new draft generated from supplied files.",
    }
    llm = FakeLLMClient([json.dumps(response)])

    init_session(session, llm, PromptRenderer(), workflow_config=_no_repair_config())

    user_input = inputs.joinpath("user_input.yaml").read_text()
    user_model = session.joinpath("generated", "user_model.py").read_text()
    assert "- {name: T, observes: T}" in user_input
    assert "- {name: dTdt, observes: dTdt}" in user_input
    assert "solution[:, 2] - dataset[:, 0]" in user_model
    assert "observables['dTdt'] - dataset[:, 1]" in user_model


def test_init_session_observable_header_does_not_mark_source_state_observed(tmp_path):
    session = tmp_path / "derived_header"
    inputs = session / "inputs"
    inputs.mkdir(parents=True)
    inputs.joinpath("data.csv").write_text("time,displacement\n0.0,0.0\n1.0,1.0\n")
    inputs.joinpath("user_info.txt").write_text("Fit displacement = x1.")
    response = {
        "missing_inputs": [],
        "review": "Derived observable from a state.",
        "filename_data": "data.csv",
        "parameters": [
            {"name": "k", "min_value": 0.001, "max_value": 10.0, "logscale": True}
        ],
        "fixed_parameters": [],
        "helper_functions": [],
        "states": [
            {"name": "x1", "initial_value": 0.0, "rhs": "k", "observed_column": None}
        ],
        "observables": [
            {"name": "displacement", "expression": "x1", "observed_column": 0}
        ],
        "loss_body": "",
        "user_info_txt": "Local pfit-new draft generated from supplied files.",
    }
    llm = FakeLLMClient([json.dumps(response)])

    init_session(session, llm, PromptRenderer())

    user_input = inputs.joinpath("user_input.yaml").read_text()
    user_model = session.joinpath("generated", "user_model.py").read_text()
    assert "- {name: displacement, observes: displacement}" in user_input
    assert "solution[:, 0] - dataset[:, 0]" not in user_model
    assert "observables['displacement'] - dataset[:, 0]" in user_model


def test_init_session_rejects_unassigned_loss_body_names(tmp_path):
    session = tmp_path / "derived"
    inputs = session / "inputs"
    inputs.mkdir(parents=True)
    inputs.joinpath("derived.csv").write_text("time,z\n0.0,2.0\n")
    inputs.joinpath("user_info.txt").write_text("Fit observed z = scale*x.")
    response = {
        "missing_inputs": [],
        "review": "Derived observable with custom loss.",
        "filename_data": "derived.csv",
        "parameters": [
            {"name": "k", "min_value": 0.001, "max_value": 10.0, "logscale": True}
        ],
        "fixed_parameters": [{"name": "scale", "value": 2.0}],
        "helper_functions": [],
        "states": [
            {"name": "x", "initial_value": 1.0, "rhs": "-k * x", "observed_column": None}
        ],
        "observables": [
            {"name": "z", "expression": "scale * x", "observed_column": 0}
        ],
        "loss_body": "return np.mean(np.square(observables['z'] - dataset[:, 0]))",
        "user_info_txt": "Local pfit-new draft generated from supplied files.",
    }
    llm = FakeLLMClient([json.dumps(response)])

    with pytest.raises(ValidationError, match="unknown name: observables"):
        init_session(session, llm, PromptRenderer(), workflow_config=_no_repair_config())


def test_init_session_rejects_loss_body_dataset_index_out_of_bounds(tmp_path):
    session = tmp_path / "demo"
    _write_user_supplied_data(session)
    response = json.loads(_new_session_response())
    response["loss_body"] = "return np.mean(np.square(solution[:, 0] - dataset[:, 1]))"
    llm = FakeLLMClient([json.dumps(response)])

    with pytest.raises(ValidationError, match="dataset column index 1 is out of bounds"):
        init_session(session, llm, PromptRenderer(), workflow_config=_no_repair_config())


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
    inputs.joinpath("derived.csv").write_text("time,z\n0.0,2.0\n1.0,1.0\n")
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
    inputs.joinpath("data.csv").write_text("time,y\n0.0,1.0\n1.0,0.5\n2.0,0.25\n")
    inputs.joinpath("user_info.txt").write_text(
        "Fit dy/dt = -k*y against data.csv. k in [0.001, 10], y0=1.0."
    )


def _new_session_response(
    *,
    rhs: str = "-k * y",
    observed_state_name: str = "y",
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
                    "name": observed_state_name,
                    "initial_value": 1.0,
                    "rhs": rhs if observed_state_name == "y" else rhs.replace("y", observed_state_name),
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
