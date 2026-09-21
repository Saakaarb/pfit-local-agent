import pytest

from local_agent.agent.diagnostics import diagnose_run


pytestmark = pytest.mark.unit


def test_diagnose_run_writes_report(tmp_path):
    session = tmp_path / "session"
    outputs = session / "outputs"
    outputs.mkdir(parents=True)
    (outputs / "final_design_point.csv").write_text("1.0\n")

    report_path = diagnose_run(session)

    assert report_path == outputs / "fit_diagnosis.txt"
    assert "final_design_point.csv: present" in report_path.read_text()


def test_diagnose_run_summarizes_de_node_and_workflow_events(tmp_path):
    session = tmp_path / "session"
    outputs = session / "outputs" / "run_1"
    logs = session / "generated" / "agent_logs"
    outputs.mkdir(parents=True)
    logs.mkdir(parents=True)
    (outputs / "final_design_point.csv").write_text("1.0\n")
    (outputs / "result_solution.csv").write_text("0.0,1.0\n")
    (outputs / "de_fitting.log").write_text(
        "Total DE iterations: 2\n"
        "1, 5.0E+00, 0.1\n"
        "2, 2.0E+00, 0.1\n"
    )
    (outputs / "NODE_fitting.log").write_text(
        "Total number of NODE iterations: 2\n"
        "1, 2.0E+00, 0.1\n"
        "2, 1.0E+00, 0.1\n"
    )
    (logs / "workflow_events.jsonl").write_text(
        '{"step": "validate_session", "status": "passed", "message": "ok"}\n'
        '{"step": "repair_generated_script", "status": "attempted", "message": "1"}\n'
    )

    report_path = diagnose_run(session, "run_1")
    report = report_path.read_text()

    assert "- de_fitting.log: present" in report
    assert "- DE: 2 iterations, first loss 5.0000E+00, best loss 2.0000E+00, final loss 2.0000E+00" in report
    assert "- NODE: 2 iterations, first loss 2.0000E+00, best loss 1.0000E+00, final loss 1.0000E+00" in report
    assert "- events: 1 passed, 0 failed, 1 repair attempts" in report
