from pathlib import Path

import pytest

from local_agent.agent.validators import (
    ValidationError,
    import_generated_script,
    validate_generated_script_contract,
)


pytestmark = pytest.mark.contract


def test_existing_generated_script_passes_contract():
    script_path = Path("tests/robertson_session/generated/generated_script.py")

    module_ast = validate_generated_script_contract(script_path)

    assert module_ast.body


def test_import_existing_generated_script():
    pytest.importorskip("jax")
    pytest.importorskip("diffrax")
    script_path = Path("tests/vanderpol_session/generated/generated_script.py")

    module = import_generated_script(script_path)

    assert callable(module._compute_loss_problem)
    assert callable(module._write_problem_result)


def test_contract_rejects_markdown_fences(tmp_path):
    script_path = tmp_path / "generated_script.py"
    script_path.write_text("```python\nprint('nope')\n```\n")

    with pytest.raises(ValidationError, match="Markdown fences"):
        validate_generated_script_contract(script_path)


def test_contract_rejects_missing_function(tmp_path):
    script_path = tmp_path / "generated_script.py"
    script_path.write_text(
        """
def user_defined_system(t, y, other_args):
    pass

def _integrate_system(constants, trainable_variables):
    pass

def _compute_loss_problem(constants, trainable_variables):
    pass
"""
    )

    with pytest.raises(ValidationError, match="_write_problem_result"):
        validate_generated_script_contract(script_path)


def test_contract_rejects_wrong_signature(tmp_path):
    script_path = tmp_path / "generated_script.py"
    script_path.write_text(
        """
def user_defined_system(t, y):
    pass

def _integrate_system(constants, trainable_variables):
    pass

def _compute_loss_problem(constants, trainable_variables):
    pass

def _write_problem_result(constants, trainable_variables):
    pass
"""
    )

    with pytest.raises(ValidationError, match="user_defined_system"):
        validate_generated_script_contract(script_path)


def test_contract_rejects_user_model_other_args_keys(tmp_path):
    script_path = tmp_path / "generated_script.py"
    script_path.write_text(
        """
def user_defined_system(t, y, other_args):
    return other_args["trainable_parameters"]

def _integrate_system(constants, trainable_variables):
    return None

def _compute_loss_problem(constants, trainable_variables):
    return 0.0

def _write_problem_result(constants, trainable_variables):
    return constants["dataset"]
"""
    )

    with pytest.raises(ValidationError, match="unsupported other_args key"):
        validate_generated_script_contract(script_path)


def test_contract_rejects_direct_diffrax_tolerances(tmp_path):
    script_path = tmp_path / "generated_script.py"
    script_path.write_text(
        """
import diffrax

def user_defined_system(t, y, other_args):
    return y

def _integrate_system(constants, trainable_variables):
    return diffrax.diffeqsolve(None, None, rtol=1e-6, atol=1e-8)

def _compute_loss_problem(constants, trainable_variables):
    return 0.0

def _write_problem_result(constants, trainable_variables):
    return constants["dataset"]
"""
    )

    with pytest.raises(ValidationError, match="direct keyword"):
        validate_generated_script_contract(script_path)
