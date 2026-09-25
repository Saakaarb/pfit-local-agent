from pathlib import Path
import json

import pytest

from local_agent.agent.checks import check_session, write_check_report, _loss_expression_evidence
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


def test_unverified_semantic_claim_cannot_veto_valid_session(tmp_path):
    session = _make_vanderpol_session(tmp_path)
    claim = "dataset[:, 0] is time and must be replaced by dataset[:, 0]."
    client = FakeLLMClient([json.dumps({"critical_errors": [claim]})])

    report = check_session(session, llm_client=client)

    assert report.passed
    assert f"Unverified semantic finding: {claim}" in report.warnings


@pytest.mark.parametrize("array", ["dataset", "solution", "solution_time"])
@pytest.mark.parametrize("aliased", [False, True])
def test_measured_normalization_checks_actual_array_even_when_llm_approves(tmp_path, array, aliased):
    session = _make_wide_range_session(tmp_path, log_loss=True)
    (session / "inputs" / "user_info.txt").write_text(
        "Loss:\n- Normalize y residuals by max(abs(measured column)).\n"
    )
    path = session / "generated" / "user_model.py"
    source = path.read_text()
    source = source.replace(
        "    loss = np.mean(np.square(solution[:, 0] - dataset[:, 0]))",
        (f"    values = {array}\n    scale = np.max(np.abs(values))\n" if aliased else "")
        + "    loss = np.mean(np.square((solution[:, 0] - dataset[:, 0]) / "
        + ("scale" if aliased else f"np.max(np.abs({array}))") + "))",
    )
    path.write_text(source)
    client = FakeLLMClient([json.dumps({"critical_errors": []})])

    report = check_session(session, llm_client=client)

    assert report.passed is (array == "dataset")
    if array != "dataset":
        assert any("measured-data normalization" in e for e in report.critical_errors)


def test_semantic_review_receives_runtime_mapping_and_missing_value_counts(tmp_path):
    session = _make_wide_range_session(tmp_path, log_loss=True)
    (session / "inputs" / "data.csv").write_text(
        "time,y,rate\n0,1,0.001\n1,nan,0.01\n2,0.5,1\n3,0.2,10\n"
    )
    client = FakeLLMClient([json.dumps({"critical_errors": []})])

    check_session(session, llm_client=client)

    context = client.requests[0][1].content
    assert "dataset[:, 0]: y" in context
    assert "dataset[:, 1]: rate" in context
    assert "0: y (initial=" in context
    assert "NaN count=1, infinity count=0" in context
    assert "NaN count=0, infinity count=0" in context


@pytest.mark.parametrize("denominator", [
    "np.max(np.abs(dataset[:, 0])) + 1e-12",
    "np.max(np.abs(solution[:, 0])) + 1e-12",
    "scale",
    "scale_for(dataset[:, 0])",
])
def test_loss_evidence_preserves_actual_denominator(denominator):
    source = (
        "def _compute_loss_problem(solution_time, solution, dataset, trainable_parameters, fixed_parameters):\n"
        "    scale = np.max(np.abs(dataset), axis=0)\n"
        f"    return np.mean(np.square((solution - dataset) / ({denominator})))\n"
    )

    evidence = _loss_expression_evidence(source)

    assert f"denominator: {denominator}" in evidence
    assert "line 3:" in evidence


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


@pytest.mark.parametrize(
    ("values", "requires_log", "summary_evidence"),
    [
        ([1e6, 1.1e6, 1.2e6, 1.3e6], False, "automatic log-loss required=no"),
        ([1, 10, 100, 999], False, "automatic log-loss required=no"),
        ([1, 10, 100, 1000], True, "automatic log-loss required=yes"),
        ([-1, 0.001, 1, 1000], False, "contains zero/negative values"),
        ([0, 0.001, 1, 1000], False, "contains zero/negative values"),
        (["nan", 1, 10, 1000], True, "automatic log-loss required=yes"),
        (["nan", "nan", "nan", 1000], False, "fewer than two finite values"),
    ],
)
def test_scale_rule_uses_same_full_column_evidence_for_semantic_review(
    tmp_path, values, requires_log, summary_evidence
):
    session = _make_wide_range_session(tmp_path, log_loss=False)
    (session / "inputs" / "data.csv").write_text(
        "time,y,rate\n"
        + "".join(f"{index},1,{value}\n" for index, value in enumerate(values))
    )
    (session / "inputs" / "user_info.txt").write_text(
        "Loss:\n- Compare y and rate directly in linear space.\n"
    )
    client = FakeLLMClient([json.dumps({"critical_errors": []})])

    report = check_session(session, llm_client=client)

    scale_errors = [e for e in report.critical_errors if "spans orders of magnitude" in e]
    assert bool(scale_errors) is requires_log
    assert all("rate uses dataset[:, 1]" in error for error in scale_errors)
    context = client.requests[0][1].content
    rate_summary = next(line for line in context.splitlines() if line.startswith("- rate: min="))
    assert summary_evidence in rate_summary
    assert "positive log10 range=" not in context
    assert "Compare y and rate directly in linear space." in context


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
