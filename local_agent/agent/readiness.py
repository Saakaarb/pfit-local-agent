"""Deterministic pre-run gate, adapted from deployed tools/check_ready.py."""
from pathlib import Path
from lib.utils.source_stamp import verify_stamp
from local_agent.agent.checks import check_session
from local_agent.agent.validators import ValidationError, validate_generated_script_contract


def check_ready(session_dir):
    session_dir = Path(session_dir)
    # Recompute checks on current inputs; a missing/stale saved report cannot
    # substitute for validation, and old LLM assertions cannot veto a run.
    report = check_session(session_dir)
    script = session_dir / "generated" / "generated_script.py"
    if not script.is_file():
        report.critical_errors.append("generated_script.py missing; run pfit jax first")
        return report
    try:
        validate_generated_script_contract(script)
    except ValidationError as exc:
        report.critical_errors.append(str(exc))
    stamped, detail = verify_stamp(session_dir)
    if stamped is False:
        report.critical_errors.append(detail + "; run pfit jax again")
    elif stamped is None:
        sources = [session_dir / "inputs" / "user_input.yaml", session_dir / "generated" / "user_model.py"]
        if any(p.is_file() and p.stat().st_mtime_ns > script.stat().st_mtime_ns for p in sources):
            report.critical_errors.append("Generated script is older than its sources; run pfit jax again")
        report.warnings.append(detail + "; using timestamps only, which copies/clones can invalidate. Run pfit jax to stamp it.")
    return report
