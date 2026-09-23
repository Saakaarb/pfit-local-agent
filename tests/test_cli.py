from pathlib import Path
import json

import pytest

from local_agent.cli.main import main


pytestmark = pytest.mark.cli


VALID_FRAGMENTS = json.dumps(
    {
        "rhs": ["x2", "-mu * x1"],
        "loss_body": "\n".join(
            [
                "scale = jnp.maximum(jnp.max(jnp.abs(dataset), axis=0), 1e-12)",
                "return jnp.mean(jnp.square((solution - dataset) / scale))",
            ]
        ),
        "writeout_body": "return jnp.concatenate((solution_time[:, None], dataset, solution), axis=1)",
        "review": "basic vanderpol fragments",
    }
)


ONE_STATE_FRAGMENTS = json.dumps(
    {
        "rhs": ["-k * y"],
        "loss_body": "return jnp.mean(jnp.square(solution[:, 0] - dataset[:, 0]))",
        "writeout_body": "return jnp.column_stack((solution_time, dataset[:, 0], solution[:, 0]))",
        "review": "one-state exponential decay fragments",
    }
)

VALID_SPLIT_RESPONSES = json.dumps(
    [
        json.dumps({"helper_functions": [], "review": ""}),
        json.dumps({"rhs": ["x2", "-mu * x1"], "helper_functions": [], "review": ""}),
    ]
)

ONE_STATE_SPLIT_RESPONSES = json.dumps(
    [
        json.dumps({"helper_functions": [], "review": ""}),
        json.dumps({"rhs": ["-k * y"], "helper_functions": [], "review": ""}),
    ]
)


CHECK_RESPONSE = json.dumps(
    {
        "critical_errors": [],
        "warnings": [],
        "recommendations": ["Semantic check found no blocking issues in the fixture."],
        "review": "Session is ready for JAX translation.",
    }
)


def make_session(tmp_path):
    session = tmp_path / "session"
    inputs = session / "inputs"
    inputs.mkdir(parents=True)
    yaml_source = (Path("tests/vanderpol_session/inputs/user_input.yaml")).read_text()
    data_source = (Path("tests/vanderpol_session/inputs/vanderpol_data.csv")).read_text()
    (inputs / "user_input.yaml").write_text(yaml_source)
    (inputs / "vanderpol_data.csv").write_text(data_source)
    generated = session / "generated"
    generated.mkdir()
    (generated / "user_model.py").write_text(
        Path("tests/vanderpol_session/generated/user_model.py").read_text()
    )
    return session


def test_cli_new_writes_starter_files(tmp_path, capsys):
    session = tmp_path / "new_session"
    _write_new_session_context(session)
    response_file = tmp_path / "new_response.json"
    response_file.write_text(_new_session_response())

    exit_code = main(["new", str(session), "--fake-response-file", str(response_file)])

    assert exit_code == 0
    assert "session initialized" in capsys.readouterr().out
    assert (session / "inputs" / "user_input.yaml").exists()
    assert (session / "generated" / "user_model.py").exists()


def test_cli_check_writes_report(tmp_path, capsys):
    session = make_session(tmp_path)
    response_file = tmp_path / "check_response.json"
    response_file.write_text(CHECK_RESPONSE)

    exit_code = main(["check", str(session), "--fake-response-file", str(response_file)])

    assert exit_code == 0
    assert "check report written" in capsys.readouterr().out
    report = (session / "generated" / "user_input_check.txt").read_text()
    assert "Semantic review:" in report
    assert "ready for JAX translation" in report


def test_cli_jax_with_fake_llm(tmp_path, capsys):
    session = make_session(tmp_path)
    response_file = tmp_path / "response.json"
    response_file.write_text(VALID_SPLIT_RESPONSES)

    exit_code = main(
        [
            "jax",
            str(session),
            "--fake-response-file",
            str(response_file),
        ]
    )

    assert exit_code == 0
    assert "validate_generated_script: passed" in capsys.readouterr().out
    assert "def _integrate_system" in (session / "generated" / "generated_script.py").read_text()


def test_cli_run_rejects_extra_mode_argument(tmp_path):
    session = make_session(tmp_path)

    with pytest.raises(SystemExit):
        main(["run", str(session), "gradient-only"])


def test_cli_diagnose_writes_report(tmp_path, capsys):
    session = make_session(tmp_path)
    outputs = session / "outputs"
    outputs.mkdir()
    (outputs / "final_design_point.csv").write_text("1.0\n")

    exit_code = main(["diagnose", str(session)])

    assert exit_code == 0
    assert "diagnosis written" in capsys.readouterr().out
    assert (outputs / "fit_diagnosis.txt").exists()


def test_cli_full_local_workflow_with_fake_llm(tmp_path):
    session = tmp_path / "workflow_session"
    _write_new_session_context(session)
    new_response_file = tmp_path / "new_response.json"
    new_response_file.write_text(_new_session_response())
    response_file = tmp_path / "response.json"
    response_file.write_text(ONE_STATE_SPLIT_RESPONSES)
    check_response_file = tmp_path / "check_response.json"
    check_response_file.write_text(CHECK_RESPONSE)

    assert main(["new", str(session), "--fake-response-file", str(new_response_file)]) == 0
    _make_starter_session_fast_de(session / "inputs" / "user_input.yaml")

    assert main(["check", str(session), "--fake-response-file", str(check_response_file)]) == 0
    assert main(["jax", str(session), "--fake-response-file", str(response_file)]) == 0
    assert main(["run", str(session)]) == 0

    run_dirs = [path for path in (session / "outputs").iterdir() if path.is_dir()]
    assert len(run_dirs) == 1
    run_id = run_dirs[0].name

    assert main(["diagnose", str(session), run_id]) == 0
    assert (run_dirs[0] / "de_fitting.log").exists()
    assert (run_dirs[0] / "NODE_fitting.log").exists()
    assert (run_dirs[0] / "final_design_point.csv").exists()
    assert (run_dirs[0] / "fit_diagnosis.txt").exists()


def _make_starter_session_fast_de(input_yaml: Path) -> None:
    yaml = input_yaml.read_text()
    replacements = {
        "MIN_VAL = 0.001": "MIN_VAL = 0.0",
        "MAX_VAL = 10.0": "MAX_VAL = 4.0",
        "LOGSCALE = Y": "LOGSCALE = N",
        "NUM_PARTICLES = 16": "NUM_PARTICLES = 4",
        "NUM_ITERS = 5": "NUM_ITERS = 1",
        "MAX_STEPS = 10000": "MAX_STEPS = 100",
        "TRANSITION_STEPS_LR = 2000": "TRANSITION_STEPS_LR = 10",
    }
    for old, new in replacements.items():
        yaml = yaml.replace(old, new)
    yaml = yaml.replace(
        "<P> PROCESSORS = 1 </P>",
        "<P> PROCESSORS = 1 </P>\n            <P> ALGORITHM = DE </P>\n            <P> RANDOM_SEED = 7 </P>",
    )
    input_yaml.write_text(yaml)


def _write_new_session_context(session: Path) -> None:
    inputs = session / "inputs"
    inputs.mkdir(parents=True)
    inputs.joinpath("data.csv").write_text("time,y\n0.0,1.0\n1.0,0.5\n2.0,0.25\n")
    inputs.joinpath("user_info.txt").write_text(
        "Fit dy/dt = -k*y against data.csv. k in [0.001, 10], y0=1.0."
    )


def _new_session_response() -> str:
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
                    "rhs": "-k * y",
                    "observed_column": 0,
                }
            ],
            "user_info_txt": "Local pfit-new draft generated from supplied files.",
        }
    )
