from pathlib import Path

import pytest


pytestmark = pytest.mark.unit


def test_readme_documents_local_llm_commands():
    readme = Path("README.md").read_text()

    assert "pfit new" in readme
    assert "pfit check" in readme
    assert "pfit jax" in readme
    assert "pfit run" in readme
    assert "pfit diagnose" in readme
    assert "docs/local_llm_orchestration.md" in readme
    assert "docs/user_workflow_compatibility.md" in readme
    assert "python fit_parameters.py <session_name>" in readme
    assert "PFIT_LLM_MODEL" in readme
