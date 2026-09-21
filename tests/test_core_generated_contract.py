from pathlib import Path

import pytest

from local_agent.core.generated_contract import (
    GeneratedContractError,
    REQUIRED_GENERATED_FUNCTIONS,
    validate_generated_script_contract,
)


pytestmark = pytest.mark.unit


def test_required_generated_functions_are_explicit():
    assert REQUIRED_GENERATED_FUNCTIONS["user_defined_system"] == (
        "t",
        "y",
        "other_args",
    )
    assert REQUIRED_GENERATED_FUNCTIONS["_write_problem_result"] == (
        "constants",
        "trainable_variables",
    )


def test_core_contract_reports_missing_script(tmp_path):
    with pytest.raises(GeneratedContractError, match="not found"):
        validate_generated_script_contract(tmp_path / "missing.py")


def test_core_contract_accepts_existing_fixture():
    module_ast = validate_generated_script_contract(
        Path("tests/vanderpol_session/generated/generated_script.py")
    )

    assert module_ast.body
