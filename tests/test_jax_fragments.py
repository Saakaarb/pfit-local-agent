import json
from pathlib import Path

import pytest

from local_agent.agent.jax_fragments import (
    parse_jax_fragments_response,
    render_generated_script_from_fragments,
)
from local_agent.agent.session_spec import load_session_spec
from local_agent.agent.validators import validate_generated_script_contract
from local_agent.agent.validators import ValidationError


pytestmark = pytest.mark.unit


def test_jax_fragments_accept_helper_functions():
    spec = load_session_spec(Path("tests/robertson_session/inputs/user_input.yaml"))
    response = json.dumps(
        {
            "rhs": [
                "rate1(k1, y1) + k3 * y3 * y2",
                "k1 * y1 - k2 * y2**2 - k3 * y2 * y3",
                "k2 * y2**2",
            ],
            "helper_functions": [
                "def rate1(k1, y1):\n    return -k1 * y1",
            ],
            "loss_body": "return jnp.mean(jnp.square(solution - dataset))",
            "writeout_body": "return jnp.concatenate((solution_time[:, None], dataset, solution), axis=1)",
            "review": "helper function smoke",
        }
    )

    fragments = parse_jax_fragments_response(response, spec)
    script = render_generated_script_from_fragments(fragments, spec)

    assert "def rate1" in script
    script_path = Path("/tmp/generated_script_helper_test.py")
    script_path.write_text(script)
    validate_generated_script_contract(script_path)


def test_jax_fragments_render_configured_integrator():
    spec = load_session_spec(Path("tests/robertson_session/inputs/user_input.yaml"))
    response = json.dumps(
        {
            "rhs": [
                "-k1 * y1 + k3 * y3 * y2",
                "k1 * y1 - k2 * y2**2 - k3 * y2 * y3",
                "k2 * y2**2",
            ],
            "loss_body": "return jnp.mean(jnp.square(solution - dataset))",
            "writeout_body": "return jnp.concatenate((solution_time[:, None], dataset, solution), axis=1)",
            "review": "integrator smoke",
        }
    )

    fragments = parse_jax_fragments_response(response, spec)
    script = render_generated_script_from_fragments(fragments, spec)

    assert "solver = diffrax.Kvaerno5()" in script


def test_jax_fragments_normalize_python_bool_ops_to_jax_logical_ops():
    spec = load_session_spec(Path("sessions/sliding_basepoint/inputs/user_input.yaml"))
    response = json.dumps(
        {
            "rhs": [
                "v1",
                "v2",
                "(k * (x2 - x1) - c1 * jnp.abs(v1) * jnp.sign(v1)) / m1",
                "jnp.where(jnp.abs(-k * (x2 - x1)) < c2 and jnp.abs(v2) < vf, 0, (-k * (x2 - x1) - c2 * jnp.sign(v2)) / m2)",
                "Dk * jnp.abs(v1)",
                "Dc * jnp.abs(v1)",
            ],
            "loss_body": "return jnp.mean(solution[:, 0])",
            "writeout_body": "return solution",
            "review": "bool op normalization",
        }
    )

    fragments = parse_jax_fragments_response(response, spec)

    assert "jnp.logical_and" in fragments.rhs[3]
    assert " and " not in fragments.rhs[3]


def test_jax_fragments_accept_helper_function_object():
    spec = load_session_spec(Path("tests/vanderpol_session/inputs/user_input.yaml"))
    response = json.dumps(
        {
            "rhs": ["x2", "-mu * x1"],
            "helper_functions": [
                {
                    "name": "helper",
                    "body": "value = solution[:, 0]\nreturn value",
                }
            ],
            "loss_body": "value = helper(solution, trainable_parameters, fixed_parameters)\nreturn value[0]",
            "writeout_body": "return solution",
            "review": "",
        }
    )

    fragments = parse_jax_fragments_response(response, spec)

    assert fragments.helper_functions == (
        "def helper(solution, trainable_parameters, fixed_parameters):\n"
        "    value = solution[:, 0]\n"
        "    return value",
    )


def test_jax_fragments_strips_rhs_assignment_prefix():
    spec = load_session_spec(Path("tests/vanderpol_session/inputs/user_input.yaml"))
    response = json.dumps(
        {
            "rhs": ["dx1dt = x2", "dx2dt = -mu * x1"],
            "loss_body": "return jnp.mean(solution[:, 0])",
            "writeout_body": "return solution",
            "review": "",
        }
    )

    fragments = parse_jax_fragments_response(response, spec)

    assert fragments.rhs == ("x2", "-mu * x1")


def test_jax_rhs_response_extracts_returned_array_from_function_lines():
    spec = load_session_spec(Path("tests/vanderpol_session/inputs/user_input.yaml"))
    response = json.dumps(
        {
            "rhs": [
                "def rhs(t, y):\n",
                "    dx1dt = x2\n",
                "    dx2dt = -mu * x1\n",
                "    return jnp.array([dx1dt, dx2dt])\n",
            ],
            "review": "",
        }
    )

    from local_agent.agent.jax_fragments import parse_jax_rhs_response

    rhs = parse_jax_rhs_response(response, spec)

    assert rhs == ("x2", "-mu * x1")


def test_jax_rhs_response_inlines_function_local_derivatives():
    spec = load_session_spec(Path("sessions/sliding_basepoint/inputs/user_input.yaml"))
    response = json.dumps(
        {
            "rhs": [
                "def rhs(t, y, trainable_parameters, fixed_parameters, dataset, t_eval):\n",
                "    c2 = trainable_parameters['c2']\n",
                "    m1 = trainable_parameters['m1']\n",
                "    vf = fixed_parameters['vf']\n",
                "    x1 = y[0]\n",
                "    x2 = y[1]\n",
                "    v1 = y[2]\n",
                "    v2 = y[3]\n",
                "    k = y[4]\n",
                "    c1 = y[5]\n",
                "    def F1(Fs, c1, v1):\n",
                "        return Fs - c1 * jnp.abs(v1) * jnp.sign(v1)\n",
                "    def F2(Fs, c2, v2):\n",
                "        return jnp.where((jnp.abs(Fs) < c2) & (jnp.abs(v2) < vf), 0, Fs - c2 * jnp.sign(v2))\n",
                "    Fs = k * (x2 - x1)\n",
                "    dx1dt = v1\n",
                "    dx2dt = v2\n",
                "    dv1dt = (1 / m1) * F1(Fs, c1, v1)\n",
                "    dv2dt = F2(-Fs, c2, v2)\n",
                "    derivatives = jnp.array([dx1dt, dx2dt, dv1dt, dv2dt, k, c1])\n",
                "    return derivatives\n",
            ],
            "review": "",
        }
    )

    from local_agent.agent.jax_fragments import parse_jax_rhs_response

    rhs = parse_jax_rhs_response(response, spec)

    assert rhs[0] == "v1"
    assert "F1" not in rhs[2]
    assert "k * (x2 - x1)" in rhs[2]
    assert "jnp.where" in rhs[3]


def test_jax_fragments_accept_list_bodies_and_helper_body_lists():
    spec = load_session_spec(Path("tests/vanderpol_session/inputs/user_input.yaml"))
    response = json.dumps(
        {
            "rhs": ["x2", "-mu * x1"],
            "helper_functions": [
                {
                    "name": "helper",
                    "arguments": ["solution"],
                    "body": ["value = solution[:, 0]", "return value"],
                }
            ],
            "loss_body": [
                "value = helper(solution)",
                "return value[0]",
            ],
            "writeout_body": [
                "value = helper(solution)",
                "return jnp.column_stack((solution_time, value))",
            ],
            "review": "",
        }
    )

    fragments = parse_jax_fragments_response(response, spec)

    assert "def helper(solution):" in fragments.helper_functions[0]
    assert "return value[0]" in fragments.loss_body


def test_jax_fragments_unwraps_function_body_fields():
    spec = load_session_spec(Path("tests/vanderpol_session/inputs/user_input.yaml"))
    response = json.dumps(
        {
            "rhs": ["x2", "-mu * x1"],
            "loss_body": (
                "def loss_body(solution_time, solution, dataset, trainable_parameters, fixed_parameters):\n"
                "    value = jnp.mean(solution[:, 0])\n"
                "    return value"
            ),
            "writeout_body": (
                "def writeout_body(solution_time, solution, dataset, trainable_parameters, fixed_parameters):\n"
                "    return solution"
            ),
            "review": "",
        }
    )

    fragments = parse_jax_fragments_response(response, spec)

    assert not fragments.loss_body.startswith("def ")
    assert "return value" in fragments.loss_body


def test_jax_fragments_dedents_body_after_flush_left_first_line():
    spec = load_session_spec(Path("tests/vanderpol_session/inputs/user_input.yaml"))
    response = json.dumps(
        {
            "rhs": ["x2", "-mu * x1"],
            "loss_body": (
                "value = jnp.mean(solution[:, 0])\n"
                "    other = jnp.mean(dataset[:, 0])\n"
                "    return value + other"
            ),
            "writeout_body": (
                "value = solution[:, 0]\n"
                "    return jnp.column_stack((solution_time, value))"
            ),
            "review": "",
        }
    )

    fragments = parse_jax_fragments_response(response, spec)

    assert "    other" not in fragments.loss_body
    assert "other = jnp.mean" in fragments.loss_body
    assert "return value + other" in fragments.loss_body


def test_jax_fragments_accept_underscore_observable_helper():
    spec = load_session_spec(Path("tests/robertson_session/inputs/user_input.yaml"))
    response = json.dumps(
        {
            "rhs": [
                "-k1 * y1 + k3 * y3 * y2",
                "k1 * y1 - k2 * y2**2 - k3 * y2 * y3",
                "k2 * y2**2",
            ],
            "helper_functions": [
                "def _observables(solution):\n    return {'y1': solution[:, 0]}",
            ],
            "loss_body": "observables = _observables(solution)\nreturn jnp.mean(jnp.square(observables['y1'] - dataset[:, 0]))",
            "writeout_body": "observables = _observables(solution)\nreturn jnp.column_stack((solution_time, observables['y1']))",
            "review": "observable helper smoke",
        }
    )

    fragments = parse_jax_fragments_response(response, spec)
    script = render_generated_script_from_fragments(fragments, spec)

    assert "def _observables" in script


def test_jax_fragments_accept_python_literal_response():
    spec = load_session_spec(Path("tests/robertson_session/inputs/user_input.yaml"))
    response = """{
  "rhs": ["-k1 * y1 + k3 * y3 * y2", "k1 * y1 - k2 * y2**2 - k3 * y2 * y3", "k2 * y2**2"],
  "loss_body": "scale = jnp.maximum(jnp.max(jnp.abs(dataset), axis=0), 1e-12)\\n"
               "return jnp.mean(jnp.square((solution - dataset) / scale))",
  "writeout_body": "return jnp.concatenate((solution_time[:, None], dataset, solution), axis=1)",
  "review": "python literal style"
}"""

    fragments = parse_jax_fragments_response(response, spec)

    assert "return jnp.mean" in fragments.loss_body


def test_jax_fragments_accept_writeout_description_alias():
    spec = load_session_spec(Path("tests/robertson_session/inputs/user_input.yaml"))
    response = json.dumps(
        {
            "rhs": [
                "-k1 * y1 + k3 * y3 * y2",
                "k1 * y1 - k2 * y2**2 - k3 * y2 * y3",
                "k2 * y2**2",
            ],
            "loss_body": "return jnp.mean(jnp.square(solution - dataset))",
            "writeout_description": "return jnp.concatenate((solution_time[:, None], dataset, solution), axis=1)",
            "review": "legacy key alias",
        }
    )

    fragments = parse_jax_fragments_response(response, spec)

    assert "solution_time" in fragments.writeout_body


def test_jax_fragments_normalize_double_escaped_body_newlines():
    spec = load_session_spec(Path("tests/robertson_session/inputs/user_input.yaml"))
    response = r'''```json
{
  "rhs": ["-k1 * y1 + k3 * y3 * y2", "k1 * y1 - k2 * y2**2 - k3 * y2 * y3", "k2 * y2**2"],
  "loss_body": "scale = jnp.max(dataset, axis=0)\\nreturn jnp.mean(jnp.square((solution - dataset) / scale))",
  "writeout_body": "Nts = solution_time.shape[0]\\nreturn jnp.concatenate((solution_time[:, None], dataset, solution), axis=1)",
  "review": "double escaped newlines"
}
```'''

    fragments = parse_jax_fragments_response(response, spec)

    assert "\\n" not in fragments.loss_body
    assert "return jnp.mean" in fragments.loss_body


def test_jax_fragments_reject_out_of_range_dataset_index():
    spec = load_session_spec(Path("tests/robertson_session/inputs/user_input.yaml"))
    response = json.dumps(
        {
            "rhs": [
                "-k1 * y1 + k3 * y3 * y2",
                "k1 * y1 - k2 * y2**2 - k3 * y2 * y3",
                "k2 * y2**2",
            ],
            "loss_body": "return jnp.mean(jnp.square(solution[:, 0] - dataset[:, 3]))",
            "writeout_body": "return jnp.concatenate((solution_time[:, None], dataset, solution), axis=1)",
            "review": "bad dataset index",
        }
    )

    with pytest.raises(ValidationError, match="dataset has 3 column"):
        parse_jax_fragments_response(response, spec)


def test_jax_fragments_reject_out_of_range_solution_index():
    spec = load_session_spec(Path("tests/robertson_session/inputs/user_input.yaml"))
    response = json.dumps(
        {
            "rhs": [
                "-k1 * y1 + k3 * y3 * y2",
                "k1 * y1 - k2 * y2**2 - k3 * y2 * y3",
                "k2 * y2**2",
            ],
            "loss_body": "return jnp.mean(jnp.square(solution[:, 3] - dataset[:, 0]))",
            "writeout_body": "return jnp.concatenate((solution_time[:, None], dataset, solution), axis=1)",
            "review": "bad solution index",
        }
    )

    with pytest.raises(ValidationError, match="solution has 3 column"):
        parse_jax_fragments_response(response, spec)


def test_jax_fragments_reports_unknown_rhs_intermediate_with_repair_guidance():
    spec = load_session_spec(Path("tests/robertson_session/inputs/user_input.yaml"))
    response = json.dumps(
        {
            "rhs": [
                "phos_AA - k1 * y1",
                "k1 * y1 - k2 * y2**2 - k3 * y2 * y3",
                "k2 * y2**2",
            ],
            "loss_body": "return jnp.mean(jnp.square(solution - dataset))",
            "writeout_body": "return jnp.concatenate((solution_time[:, None], dataset, solution), axis=1)",
            "review": "bad rhs intermediate",
        }
    )

    with pytest.raises(ValidationError, match="rhs for y1 uses unknown name: phos_AA"):
        parse_jax_fragments_response(response, spec)


def test_jax_fragments_reports_helper_missing_explicit_dependency():
    spec = load_session_spec(Path("tests/robertson_session/inputs/user_input.yaml"))
    response = json.dumps(
        {
            "rhs": [
                "rate(k1, y1)",
                "k1 * y1 - k2 * y2**2 - k3 * y2 * y3",
                "k2 * y2**2",
            ],
            "helper_functions": [
                "def rate(k1, y1):\n    return -k1 * y1 + vf",
            ],
            "loss_body": "return jnp.mean(jnp.square(solution - dataset))",
            "writeout_body": "return jnp.concatenate((solution_time[:, None], dataset, solution), axis=1)",
            "review": "helper closes over vf",
        }
    )

    with pytest.raises(
        ValidationError,
        match="helper function rate uses unknown name: vf",
    ):
        parse_jax_fragments_response(response, spec)


def test_jax_fragments_splits_packed_helper_functions_and_normalizes_np():
    spec = load_session_spec(Path("tests/robertson_session/inputs/user_input.yaml"))
    response = json.dumps(
        {
            "rhs": [
                "rate1(k1, y1)",
                "rate2(k1, y1, k2, y2, k3, y3)",
                "k2 * y2**2",
            ],
            "helper_functions": [
                "def rate1(k1, y1):\n    return -k1 * y1\n\n"
                "def rate2(k1, y1, k2, y2, k3, y3):\n"
                "    return k1 * y1 - k2 * y2**2 - k3 * y2 * y3",
            ],
            "loss_body": "sim = np.stack([solution[:, 0], solution[:, 1], solution[:, 2]], axis=1)\nreturn np.mean(np.square(sim - dataset))",
            "writeout_body": "writeout_array = np.zeros([solution_time.shape[0], 7])\nreturn np.column_stack((solution_time, dataset, solution))",
            "review": "packed helpers",
        }
    )

    fragments = parse_jax_fragments_response(response, spec)

    assert len(fragments.helper_functions) == 2
    assert "def rate1" in fragments.helper_functions[0]
    assert "def rate2" in fragments.helper_functions[1]
    assert "sim = np.stack" not in fragments.loss_body
    assert "return np.mean" not in fragments.loss_body
    assert "jnp.stack" in fragments.loss_body
    assert "np.zeros" in fragments.writeout_body
    assert "np.column_stack" in fragments.writeout_body


def test_jax_fragments_preserves_python_writeout_assignment():
    spec = load_session_spec(Path("tests/robertson_session/inputs/user_input.yaml"))
    response = json.dumps(
        {
            "rhs": [
                "-k1 * y1 + k3 * y3 * y2",
                "k1 * y1 - k2 * y2**2 - k3 * y2 * y3",
                "k2 * y2**2",
            ],
            "loss_body": "return jnp.mean(jnp.square(solution - dataset))",
            "writeout_body": "writeout_array = jnp.zeros([solution_time.shape[0], 7])\nwriteout_array[:, 0] = solution_time\nwriteout_array[:, 1:4] = dataset\nreturn writeout_array",
            "review": "jax assignment normalization",
        }
    )

    fragments = parse_jax_fragments_response(response, spec)

    assert "writeout_array[:, 0] = solution_time" in fragments.writeout_body
    assert "writeout_array[:, 1:4] = dataset" in fragments.writeout_body


def test_jax_fragments_accept_python_writeout_loop_and_nested_helper():
    spec = load_session_spec(Path("tests/robertson_session/inputs/user_input.yaml"))
    response = json.dumps(
        {
            "rhs": [
                "-k1 * y1 + k3 * y3 * y2",
                "k1 * y1 - k2 * y2**2 - k3 * y2 * y3",
                "k2 * y2**2",
            ],
            "loss_body": "return jnp.mean(jnp.square(solution - dataset))",
            "writeout_body": "\n".join(
                [
                    "Nts = solution_time.shape[0]",
                    "out = np.zeros((Nts, 5))",
                    "def force(Fs, c1, v1):",
                    "    return Fs - c1 * np.abs(v1) * np.sign(v1)",
                    "Fs = solution[:, 1] - solution[:, 0]",
                    "for i in range(Nts):",
                    "    out[i, 0] = solution_time[i]",
                    "    out[i, 1] = dataset[i, 0]",
                    "    out[i, 2] = force(Fs[i], solution[i, 2], solution[i, 0])",
                    "return out",
                ]
            ),
            "review": "python writeout",
        }
    )

    fragments = parse_jax_fragments_response(response, spec)
    script = render_generated_script_from_fragments(fragments, spec)

    assert "import numpy as np" in script
    assert "for i in range(Nts)" in fragments.writeout_body
    assert "def force" in fragments.writeout_body


def test_jax_fragments_vectorizes_simple_range_fill_loop():
    spec = load_session_spec(Path("tests/robertson_session/inputs/user_input.yaml"))
    response = json.dumps(
        {
            "rhs": [
                "-k1 * y1 + k3 * y3 * y2",
                "k1 * y1 - k2 * y2**2 - k3 * y2 * y3",
                "k2 * y2**2",
            ],
            "helper_functions": [
                "def force(Fs, c1, v1):\n    return jnp.abs(Fs) + c1 + v1",
            ],
            "loss_body": "Nts = solution_time.shape[0]\nFs = solution[:, 0]\nc1 = solution[:, 1]\nv1 = solution[:, 2]\nF_sim = jnp.zeros(Nts)\ns_sim = jnp.zeros(Nts)\nfor i in range(Nts):\n    F_sim = force(Fs[i], c1[i], v1[i])\n    s_sim = solution[i, 0]\nreturn jnp.mean(F_sim + s_sim)",
            "writeout_body": "return jnp.concatenate((solution_time[:, None], dataset, solution), axis=1)",
            "review": "loop vectorization",
        }
    )

    fragments = parse_jax_fragments_response(response, spec)

    assert "range" not in fragments.loss_body
    assert "F_sim = force(Fs, c1, v1)" in fragments.loss_body
    assert "s_sim = solution[:, 0]" in fragments.loss_body


def test_jax_fragments_rejects_bare_helper_in_rhs():
    spec = load_session_spec(Path("tests/robertson_session/inputs/user_input.yaml"))
    response = json.dumps(
        {
            "rhs": [
                "rate1 + k3 * y3 * y2",
                "k1 * y1 - k2 * y2**2 - k3 * y2 * y3",
                "k2 * y2**2",
            ],
            "helper_functions": [
                "def rate1(k1, y1):\n    return -k1 * y1",
            ],
            "loss_body": "return jnp.mean(jnp.square(solution - dataset))",
            "writeout_body": "return jnp.concatenate((solution_time[:, None], dataset, solution), axis=1)",
            "review": "bare helper",
        }
    )

    with pytest.raises(ValidationError, match="uses helper rate1 as a value"):
        parse_jax_fragments_response(response, spec)


def test_jax_fragments_rejects_helper_function_call_entry_with_guidance():
    spec = load_session_spec(Path("tests/robertson_session/inputs/user_input.yaml"))
    response = json.dumps(
        {
            "rhs": [
                "-k1 * y1 + k3 * y3 * y2",
                "k1 * y1 - k2 * y2**2 - k3 * y2 * y3",
                "k2 * y2**2",
            ],
            "helper_functions": ["_observables(solution)"],
            "loss_body": "return jnp.mean(jnp.square(solution - dataset))",
            "writeout_body": "return jnp.concatenate((solution_time[:, None], dataset, solution), axis=1)",
            "review": "helper call not definition",
        }
    )

    with pytest.raises(ValidationError, match="not call them"):
        parse_jax_fragments_response(response, spec)


def test_jax_fragments_adds_missing_fixed_parameter_to_helper_calls():
    spec = load_session_spec(Path("tests/robertson_session/inputs/user_input.yaml"))
    response = json.dumps(
        {
            "rhs": [
                "rate1(k1, y1)",
                "k1 * y1 - k2 * y2**2 - k3 * y2 * y3",
                "k2 * y2**2",
            ],
            "helper_functions": [
                "def rate1(k1, y1):\n    return -k1 * y1 + unused_constant",
            ],
            "loss_body": "return jnp.mean(jnp.square(solution - dataset))",
            "writeout_body": "return jnp.concatenate((solution_time[:, None], dataset, solution), axis=1)",
            "review": "helper closes over fixed parameter",
        }
    )

    fragments = parse_jax_fragments_response(response, spec)

    assert "def rate1(k1, y1, unused_constant):" in fragments.helper_functions[0]
    assert "rate1(k1, y1, unused_constant)" in fragments.rhs[0]


def test_jax_fragments_adds_missing_state_to_helper_calls():
    spec = load_session_spec(Path("tests/robertson_session/inputs/user_input.yaml"))
    response = json.dumps(
        {
            "rhs": [
                "rate1(k1)",
                "k1 * y1 - k2 * y2**2 - k3 * y2 * y3",
                "k2 * y2**2",
            ],
            "helper_functions": [
                "def rate1(k1):\n    return -k1 * y1",
            ],
            "loss_body": "return jnp.mean(jnp.square(solution - dataset))",
            "writeout_body": "return jnp.concatenate((solution_time[:, None], dataset, solution), axis=1)",
            "review": "helper closes over state",
        }
    )

    fragments = parse_jax_fragments_response(response, spec)

    assert "def rate1(k1, y1):" in fragments.helper_functions[0]
    assert fragments.rhs[0] == "rate1(k1, y1)"


def test_jax_fragments_unknown_helper_free_name_still_fails():
    spec = load_session_spec(Path("tests/robertson_session/inputs/user_input.yaml"))
    response = json.dumps(
        {
            "rhs": [
                "rate1(k1, y1)",
                "k1 * y1 - k2 * y2**2 - k3 * y2 * y3",
                "k2 * y2**2",
            ],
            "helper_functions": [
                "def rate1(k1, y1):\n    return -k1 * y1 + missing_name",
            ],
            "loss_body": "return jnp.mean(jnp.square(solution - dataset))",
            "writeout_body": "return jnp.concatenate((solution_time[:, None], dataset, solution), axis=1)",
            "review": "true unknown",
        }
    )

    with pytest.raises(ValidationError, match="missing_name"):
        parse_jax_fragments_response(response, spec)
