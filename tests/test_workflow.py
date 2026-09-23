from pathlib import Path
import json

import pytest

from local_agent.agent.config import WorkflowConfig
from local_agent.agent.prompts import PromptRenderer
from local_agent.agent.workflow import (
    LocalWorkflow,
    _deterministic_custom_loss_body,
    _helper_function_inventory,
    _inline_referenced_rhs_intermediates,
    _inject_referenced_helper_definitions,
    _pfit_claude_jax_reference,
    _rhs_intermediate_inventory,
    _standard_loss_and_writeout_bodies,
)
from local_agent.agent.session_spec import load_session_spec
from local_agent.llm.fake import FakeLLMClient


pytestmark = pytest.mark.workflow


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

VALID_HELPERS = json.dumps({"helper_functions": [], "review": ""})
VALID_RHS = json.dumps({"rhs": ["x2", "-mu * x1"], "helper_functions": [], "review": ""})
VALID_LOSS = json.dumps(
    {
        "loss_body": "\n".join(
            [
                "scale = jnp.maximum(jnp.max(jnp.abs(dataset), axis=0), 1e-12)",
                "return jnp.mean(jnp.square((solution - dataset) / scale))",
            ]
        ),
        "review": "",
    }
)
VALID_WRITEOUT = json.dumps(
    {
        "writeout_body": "return jnp.concatenate((solution_time[:, None], dataset, solution), axis=1)",
        "review": "",
    }
)
VALID_SPLIT_RESPONSES = [VALID_HELPERS, VALID_RHS]

SMOKE_FAILING_FRAGMENTS = json.dumps(
    {
        "rhs": ["x2", "-mu * x1"],
        "loss_body": "return jnp.array(float('nan'))",
        "writeout_body": "return jnp.concatenate((solution_time[:, None], dataset, solution), axis=1)",
        "review": "intentionally bad loss",
    }
)

SMOKE_FAILING_LOSS = json.dumps(
    {
        "loss_body": "return jnp.array(float('nan'))",
        "review": "intentionally bad loss",
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
    return session


def test_validate_only_workflow_passes(tmp_path):
    session = make_session(tmp_path)
    workflow = LocalWorkflow(FakeLLMClient([]), PromptRenderer())

    result = workflow.validate_only(session)

    assert result.success is True
    assert result.events[-1].step == "validate_session"


def test_deterministic_custom_loss_body_preserves_log_normalized_mse():
    source = """import numpy as np

def _compute_loss_problem(solution_time, solution, dataset, trainable_parameters, fixed_parameters):
    loss = 0.0
    eps = np.min(np.where(dataset[:, 1] > 0.0, dataset[:, 1], np.inf))
    log_sim = np.log10(solution[:, 0] + eps)
    log_data = np.log10(dataset[:, 1] + eps)
    scale = np.max(log_data) - np.min(log_data) + 1e-12
    loss += np.mean(np.square((log_sim - log_data) / scale))
    return float(loss)
"""

    body = _deterministic_custom_loss_body(source)

    assert body is not None
    assert "jnp.log10" in body
    assert "jnp.mean(jnp.square((log_sim - log_data) / scale))" in body
    assert "return loss" in body
    assert "float(loss)" not in body


def test_generate_script_workflow_writes_valid_script(tmp_path):
    session = make_session(tmp_path)
    llm = FakeLLMClient(VALID_SPLIT_RESPONSES)
    workflow = LocalWorkflow(llm, PromptRenderer())

    result = workflow.generate_script(session)

    assert result.success is True
    assert "def _integrate_system" in (session / "generated" / "generated_script.py").read_text()
    assert [event.step for event in result.events] == [
        "validate_session",
        "generate_user_model_skeleton",
        "render_generated_script",
        "validate_generated_script",
        "smoke_test_generated_script",
    ]
    assert (session / "generated" / "user_model.py").exists()
    log_path = session / "generated" / "agent_logs" / "workflow_events.jsonl"
    events = [json.loads(line) for line in log_path.read_text().splitlines()]
    assert events[-1]["step"] == "smoke_test_generated_script"
    assert events[-1]["status"] == "passed"
    llm_log_path = session / "generated" / "agent_logs" / "llm_calls.jsonl"
    llm_calls = [json.loads(line) for line in llm_log_path.read_text().splitlines()]
    assert [call["step"] for call in llm_calls] == [
        "translate_jax_helpers",
        "translate_jax_rhs",
    ]
    user_prompt = llm_calls[0]["messages"][1]["content"]
    assert "Reference contract from pfit-claude" in user_prompt
    assert "NumPy to JAX Mapping" in user_prompt


def test_generate_script_workflow_repairs_invalid_script(tmp_path):
    session = make_session(tmp_path)
    llm = FakeLLMClient(["```json\nnope\n```", VALID_FRAGMENTS])
    workflow = LocalWorkflow(
        llm,
        PromptRenderer(),
        WorkflowConfig(max_repair_attempts=1),
    )

    result = workflow.generate_script(session)

    assert result.success is True
    assert "def _integrate_system" in (session / "generated" / "generated_script.py").read_text()
    assert [event.status for event in result.events].count("failed") == 1
    assert any(event.step == "repair_jax_fragments" for event in result.events)


def test_generate_script_workflow_accepts_simple_fenced_json(tmp_path):
    session = make_session(tmp_path)
    llm = FakeLLMClient(
        [f"```json\n{VALID_HELPERS}\n```", f"```json\n{VALID_RHS}\n```"]
    )
    workflow = LocalWorkflow(llm, PromptRenderer())

    result = workflow.generate_script(session)

    assert result.success is True
    assert "```" not in (session / "generated" / "generated_script.py").read_text()


def test_generate_script_workflow_repairs_invalid_split_output(tmp_path):
    session = make_session(tmp_path)
    bad_rhs = json.dumps({"rhs": ["x2"], "helper_functions": [], "review": ""})
    llm = FakeLLMClient([VALID_HELPERS, bad_rhs, VALID_FRAGMENTS])
    workflow = LocalWorkflow(
        llm,
        PromptRenderer(),
        WorkflowConfig(max_repair_attempts=1),
    )

    result = workflow.generate_script(session)

    assert result.success is True
    assert [event.status for event in result.events].count("failed") == 1
    llm_log_path = session / "generated" / "agent_logs" / "llm_calls.jsonl"
    llm_calls = [json.loads(line) for line in llm_log_path.read_text().splitlines()]
    assert [call["step"] for call in llm_calls] == [
        "translate_jax_helpers",
        "translate_jax_rhs",
        "repair_jax_fragments",
    ]


def test_generate_script_workflow_fails_after_repair_attempts(tmp_path):
    session = make_session(tmp_path)
    llm = FakeLLMClient(["```json\nnope\n```", "def still_bad(): pass"])
    workflow = LocalWorkflow(
        llm,
        PromptRenderer(),
        WorkflowConfig(max_repair_attempts=1),
    )

    result = workflow.generate_script(session)

    assert result.success is False
    assert result.events[-1].status == "failed"


def test_rhs_intermediate_inventory_lists_nonbinding_assignments():
    spec = load_session_spec(Path("tests/robertson_session/inputs/user_input.yaml"))
    source = """
import numpy as np

def user_defined_system(t, y, trainable_parameters, fixed_parameters, dataset, t_eval):
    k1 = trainable_parameters['k1']
    y1 = y[0]
    flux = k1 * y1
    dy1dt = -flux
    return np.array([dy1dt])
"""

    inventory = _rhs_intermediate_inventory(source, spec)

    assert "flux = k1 * y1" in inventory
    assert "k1 = trainable_parameters" not in inventory
    assert "y1 = y[0]" not in inventory


def test_helper_function_inventory_includes_helper_source():
    source = """
def _observables(solution, trainable_parameters, fixed_parameters):
    return {"y": solution[:, 0]}

def user_defined_system(t, y, trainable_parameters, fixed_parameters, dataset, t_eval):
    return y
"""

    inventory = _helper_function_inventory(source)

    assert "def _observables" in inventory
    assert "return {\"y\": solution[:, 0]}" in inventory
    assert "def user_defined_system" not in inventory


def test_pfit_claude_jax_reference_is_packaged():
    reference = _pfit_claude_jax_reference()

    assert "pfit-claude JAX Reference" in reference
    assert "NumPy to JAX Mapping" in reference


def test_standard_loss_and_writeout_handles_derived_observables():
    spec = load_session_spec(Path("sessions/boehm_stat5/inputs/user_input.yaml"))

    bodies = _standard_loss_and_writeout_bodies(spec)

    assert bodies is not None
    loss_body, writeout_body = bodies
    assert "observables = _observables" in loss_body
    assert "observables['pSTAT5A']" in loss_body
    assert "dataset[:, 0]" in writeout_body
    assert "observables['rSTAT5A']" in writeout_body


def test_standard_loss_and_writeout_ignores_uncertainty_columns():
    spec = load_session_spec(Path("sessions/oregonator/inputs/user_input.yaml"))

    bodies = _standard_loss_and_writeout_bodies(spec)

    assert bodies is not None
    loss_body, writeout_body = bodies
    assert "dataset[:, 0]" in loss_body
    assert "dataset[:, 1]" in loss_body
    assert "dataset[:, 2]" not in loss_body
    assert "dataset[:, 3]" not in writeout_body


def test_standard_loss_and_writeout_declines_custom_user_loss():
    spec = load_session_spec(Path("sessions/sliding_basepoint/inputs/user_input.yaml"))
    user_model_source = """
def _compute_loss_problem(solution_time, solution, dataset, trainable_parameters, fixed_parameters):
    return jnp.mean(jnp.abs(solution[:, 0] - dataset[:, 0]))

def writeout_description(solution_time, solution, dataset, trainable_parameters, fixed_parameters):
    return dataset
"""

    assert (
        _standard_loss_and_writeout_bodies(
            spec,
            data_width=2,
            user_model_source=user_model_source,
        )
        is None
    )


def test_standard_loss_and_writeout_allows_framework_default_loss():
    spec = load_session_spec(Path("sessions/oregonator/inputs/user_input.yaml"))
    user_model_source = """
def _compute_loss_problem(solution_time, solution, dataset, trainable_parameters, fixed_parameters):
    residuals = np.column_stack((
        solution[:, 0] - dataset[:, 0],
        solution[:, 1] - dataset[:, 1],
    ))
    return float(np.mean(np.square(residuals)))
"""

    assert (
        _standard_loss_and_writeout_bodies(
            spec,
            user_model_source=user_model_source,
        )
        is not None
    )


def test_standard_loss_and_writeout_declines_mismatched_dataset_width():
    spec = load_session_spec(Path("sessions/sliding_basepoint/inputs/user_input.yaml"))

    assert _standard_loss_and_writeout_bodies(spec, data_width=3) is None


def test_inject_referenced_nested_helper_definitions_from_user_model(tmp_path):
    session = tmp_path / "session"
    generated = session / "generated"
    generated.mkdir(parents=True)
    generated.joinpath("user_model.py").write_text(
        """
def _compute_loss_problem(solution_time, solution, dataset, trainable_parameters, fixed_parameters):
    def F1(Fs, c1, v1):
        return Fs - c1 * np.abs(v1) * np.sign(v1)
    return F1(solution[:, 0], solution[:, 1], solution[:, 2])[0]
"""
    )
    response = json.dumps(
        {
            "rhs": ["-k * y"],
            "helper_functions": [],
            "loss_body": "return F1(solution[:, 0], solution[:, 1], solution[:, 2])[0]",
            "writeout_body": "return solution",
            "review": "",
        }
    )

    injected = json.loads(_inject_referenced_helper_definitions(response, session))

    assert len(injected["helper_functions"]) == 1
    assert injected["helper_functions"][0].startswith("def F1")


def test_inject_referenced_helper_definitions_from_user_model(tmp_path):
    session = tmp_path / "session"
    generated = session / "generated"
    generated.mkdir(parents=True)
    generated.joinpath("user_model.py").write_text(
        """
def _observables(solution, trainable_parameters, fixed_parameters):
    return {"y": solution[:, 0]}

def user_defined_system(t, y, trainable_parameters, fixed_parameters, dataset, t_eval):
    return y
"""
    )
    response = json.dumps(
        {
            "rhs": ["-k * y"],
            "helper_functions": [],
            "loss_body": "observables = _observables(solution, trainable_parameters, fixed_parameters)\nreturn observables['y'][0]",
            "writeout_body": "observables = _observables(solution, trainable_parameters, fixed_parameters)\nreturn solution",
            "review": "",
        }
    )

    injected = json.loads(_inject_referenced_helper_definitions(response, session))

    assert len(injected["helper_functions"]) == 1
    assert "def _observables" in injected["helper_functions"][0]


def test_inject_referenced_helper_replaces_call_entry(tmp_path):
    session = tmp_path / "session"
    generated = session / "generated"
    generated.mkdir(parents=True)
    generated.joinpath("user_model.py").write_text(
        """
def helper(x):
    return x
"""
    )
    response = json.dumps(
        {
            "rhs": ["helper(y)"],
            "helper_functions": ["helper(y)"],
            "loss_body": "return solution[:, 0][0]",
            "writeout_body": "return solution",
            "review": "",
        }
    )

    injected = json.loads(_inject_referenced_helper_definitions(response, session))

    assert injected["helper_functions"] == ["def helper(x):\n    return x"]


def test_inline_referenced_rhs_intermediates_from_user_model(tmp_path):
    session = tmp_path / "session"
    generated = session / "generated"
    generated.mkdir(parents=True)
    generated.joinpath("user_model.py").write_text(
        """
import numpy as np

def user_defined_system(t, y, trainable_parameters, fixed_parameters, dataset, t_eval):
    x1 = y[0]
    x2 = y[1]
    v1 = y[2]
    k = y[3]
    m1 = trainable_parameters['m1']
    Dk = trainable_parameters['Dk']
    c1 = trainable_parameters['c1']
    Fs = k * (x2 - x1)
    dv1dt = (1 / m1) * F1(Fs, c1, v1)
    P = np.abs(m1 * v1 * dv1dt)
    dkdt = Dk * P
    return np.array([dkdt])
"""
    )
    response = json.dumps(
        {
            "rhs": ["Dk * P"],
            "helper_functions": [],
            "loss_body": "return loss",
            "writeout_body": "return writeout",
            "review": "",
        }
    )

    inlined = json.loads(_inline_referenced_rhs_intermediates(response, session))

    assert "P" not in inlined["rhs"][0]
    assert "dv1dt" not in inlined["rhs"][0]
    assert "Fs" not in inlined["rhs"][0]
    assert "k * (x2 - x1)" in inlined["rhs"][0]


def test_inline_referenced_rhs_intermediates_handles_assignment_expression(tmp_path):
    session = tmp_path / "session"
    generated = session / "generated"
    generated.mkdir(parents=True)
    generated.joinpath("user_model.py").write_text(
        """
import numpy as np

def user_defined_system(t, y, trainable_parameters, fixed_parameters, dataset, t_eval):
    x = y[0]
    k = trainable_parameters['k']
    flux = k * x
    dxdt = -flux
    return np.array([dxdt])
"""
    )
    response = json.dumps(
        {
            "rhs": ["dxdt = -flux"],
            "helper_functions": [],
            "loss_body": "return loss",
            "writeout_body": "return writeout",
            "review": "",
        }
    )

    inlined = json.loads(_inline_referenced_rhs_intermediates(response, session))

    assert inlined["rhs"] == ["-(k * x)"]
