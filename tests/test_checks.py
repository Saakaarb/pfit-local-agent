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
