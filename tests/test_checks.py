from pathlib import Path
import json

import pytest

from local_agent.agent.checks import check_session, write_check_report
from local_agent.llm.fake import FakeLLMClient


pytestmark = pytest.mark.unit


def test_check_session_passes_existing_fixture(tmp_path):
    session = _make_vanderpol_session(tmp_path)

    report = check_session(session)
    report_path = write_check_report(session, report)

    assert report.passed is True
    assert report_path.name == "user_input_check.txt"
    assert "Critical errors:" in report_path.read_text()


def test_check_session_reports_critical_errors(tmp_path):
    report = check_session(tmp_path / "missing")

    assert report.passed is False
    assert report.critical_errors


def test_check_session_merges_semantic_llm_report(tmp_path):
    session = _make_vanderpol_session(tmp_path)
    client = FakeLLMClient(
        [
            json.dumps(
                {
                    "critical_errors": [],
                    "warnings": ["Check the initial condition mismatch."],
                    "recommendations": ["Use per-column loss scaling."],
                    "review": "Semantic review complete.",
                }
            )
        ]
    )

    report = check_session(session, llm_client=client)

    assert report.passed is True
    assert report.warnings == ["Check the initial condition mismatch."]
    assert report.semantic_review == "Semantic review complete."


def test_check_session_fails_wide_range_linear_loss(tmp_path):
    session = _make_wide_range_session(tmp_path, log_loss=False)

    report = check_session(session)

    assert report.passed is False
    assert any("spans orders of magnitude" in error for error in report.critical_errors)
    assert any("rate" in error for error in report.critical_errors)


def test_check_session_allows_wide_range_log_loss(tmp_path):
    session = _make_wide_range_session(tmp_path, log_loss=True)

    report = check_session(session)

    assert report.passed is True
    assert not any("spans orders of magnitude" in error for error in report.critical_errors)


def test_check_session_fails_declared_uncertainty_not_used(tmp_path):
    session = _make_uncertainty_session(tmp_path)

    report = check_session(session)

    assert report.passed is False
    assert any("uncertainty column X_sd" in error for error in report.critical_errors)
    assert any("dataset[:, 1]" in error for error in report.critical_errors)


def test_check_session_fails_branchy_user_defined_system(tmp_path):
    session = _make_branchy_dynamics_session(tmp_path)

    report = check_session(session)

    assert report.passed is False
    assert any("conditional expression" in error for error in report.critical_errors)


def test_check_session_fails_user_loss_contract_mismatch(tmp_path):
    session = _make_uncertainty_session(tmp_path)
    (session / "inputs" / "user_info.txt").write_text(
        "Loss:\n"
        "- Compare X in log10 space.\n"
        "- Normalize the residuals by the data range.\n"
        "- Use RMSE: sqrt(mean(normalized squared residuals)).\n"
    )

    report = check_session(session)

    assert report.passed is False
    assert any("square root" in error for error in report.critical_errors)
    assert any("log transform" in error for error in report.critical_errors)
    assert any("does not divide by a scale" in error for error in report.critical_errors)


def _make_vanderpol_session(tmp_path):
    session = tmp_path / "session"
    inputs = session / "inputs"
    inputs.mkdir(parents=True)
    (inputs / "user_input.yaml").write_text(
        Path("tests/vanderpol_session/inputs/user_input.yaml").read_text()
    )
    (inputs / "vanderpol_data.csv").write_text(
        Path("tests/vanderpol_session/inputs/vanderpol_data.csv").read_text()
    )
    generated = session / "generated"
    generated.mkdir()
    (generated / "user_model.py").write_text(
        Path("tests/vanderpol_session/generated/user_model.py").read_text()
    )
    return session


def _make_branchy_dynamics_session(tmp_path):
    session = _make_uncertainty_session(tmp_path)
    (session / "generated" / "user_model.py").write_text(
        "import numpy as np\n\n"
        "def user_defined_system(t, y, trainable_parameters, fixed_parameters, dataset, t_eval):\n"
        "    k = trainable_parameters['k']\n"
        "    rhs = -k * y[0] if y[0] > 0 else 0.0\n"
        "    return np.array([rhs])\n\n"
        "def _compute_loss_problem(solution_time, solution, dataset, trainable_parameters, fixed_parameters):\n"
        "    return float(np.mean(np.square((solution[:, 0] - dataset[:, 0]) / dataset[:, 1])))\n\n"
        "def writeout_description(solution_time, solution, dataset, trainable_parameters, fixed_parameters):\n"
        "    return np.column_stack((solution_time, dataset[:, 0], solution[:, 0]))\n"
    )
    return session


def _make_uncertainty_session(tmp_path):
    session = tmp_path / "uncertainty"
    inputs = session / "inputs"
    inputs.mkdir(parents=True)
    (inputs / "data.csv").write_text(
        "time,X,X_sd\n"
        "0.0,1.0,0.1\n"
        "1.0,0.5,0.1\n"
    )
    (inputs / "user_input.yaml").write_text(
        """experiments:
  - data_file: data.csv
    columns:
      - {name: time}
      - {name: X, observes: X}
      - {name: X_sd, uncertainty_of: X}

model:
  trainable_parameters:
    - {name: k, min_val: 0.001, max_val: 10.0, logscale: true}
  fixed_parameters: []
  integrated_variables:
    - {name: X, init_val: 1.0}
  observables: []

population_opt:
  population_size: 4
  num_iters: 1
  processors: 1
  algorithm: DE

gradient_opt:
  num_iters: 1
  stepsize_rtol: [1e-7]
  stepsize_atol: [1e-9]
  initial_timestep: 1e-6
  max_steps: 1000

output:
  write_results: true
"""
    )
    generated = session / "generated"
    generated.mkdir()
    (generated / "user_model.py").write_text(
        "import numpy as np\n\n"
        "def user_defined_system(t, y, trainable_parameters, fixed_parameters, dataset, t_eval):\n"
        "    k = trainable_parameters['k']\n"
        "    return np.array([-k * y[0]])\n\n"
        "def _compute_loss_problem(solution_time, solution, dataset, trainable_parameters, fixed_parameters):\n"
        "    return float(np.mean(np.square(solution[:, 0] - dataset[:, 0])))\n\n"
        "def writeout_description(solution_time, solution, dataset, trainable_parameters, fixed_parameters):\n"
        "    return np.column_stack((solution_time, dataset[:, 0], solution[:, 0]))\n"
    )
    return session


def _make_wide_range_session(tmp_path, *, log_loss: bool):
    session = tmp_path / ("wide_log" if log_loss else "wide_linear")
    inputs = session / "inputs"
    inputs.mkdir(parents=True)
    (inputs / "data.csv").write_text(
        "time,y,rate\n"
        "0.0,1.0,0.0001\n"
        "1.0,0.9,0.01\n"
        "2.0,0.8,1.0\n"
        "3.0,0.7,100.0\n"
    )
    (inputs / "user_input.yaml").write_text(
        """experiments:
  - data_file: data.csv
    columns:
      - {name: time}
      - {name: y, observes: y}
      - {name: rate, observes: rate}

model:
  trainable_parameters:
    - {name: k, min_val: 0.001, max_val: 10.0, logscale: true}
  fixed_parameters: []
  integrated_variables:
    - {name: y, init_val: 1.0}
  observables:
    - {name: rate}

population_opt:
  population_size: 4
  num_iters: 1
  processors: 1
  algorithm: DE

gradient_opt:
  num_iters: 1
  stepsize_rtol: [1e-7]
  stepsize_atol: [1e-9]
  initial_timestep: 1e-6
  max_steps: 1000

output:
  write_results: true
"""
    )
    generated = session / "generated"
    generated.mkdir()
    rate_loss = (
        "    eps = 1e-12\n"
        "    log_sim = np.log10(observables['rate'] + eps)\n"
        "    log_data = np.log10(dataset[:, 1] + eps)\n"
        "    loss += np.mean(np.square(log_sim - log_data))\n"
        if log_loss
        else "    loss += np.mean(np.square(observables['rate'] - dataset[:, 1]))\n"
    )
    (generated / "user_model.py").write_text(
        "import numpy as np\n\n"
        "def user_defined_system(t, y, trainable_parameters, fixed_parameters, dataset, t_eval):\n"
        "    k = trainable_parameters['k']\n"
        "    return np.array([-k * y[0]])\n\n"
        "def _observables(solution, trainable_parameters, fixed_parameters):\n"
        "    k = trainable_parameters['k']\n"
        "    return {'rate': np.abs(k * solution[:, 0])}\n\n"
        "def _compute_loss_problem(solution_time, solution, dataset, trainable_parameters, fixed_parameters):\n"
        "    observables = _observables(solution, trainable_parameters, fixed_parameters)\n"
        "    loss = np.mean(np.square(solution[:, 0] - dataset[:, 0]))\n"
        f"{rate_loss}"
        "    return float(loss)\n\n"
        "def writeout_description(solution_time, solution, dataset, trainable_parameters, fixed_parameters):\n"
        "    observables = _observables(solution, trainable_parameters, fixed_parameters)\n"
        "    return np.column_stack((solution_time, dataset[:, 0], dataset[:, 1], solution[:, 0], observables['rate']))\n"
    )
    return session
