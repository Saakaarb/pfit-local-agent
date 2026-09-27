"""Bounded numerical validation; no LLM calls or scientific-expression edits."""
import json
import os
from pathlib import Path
import signal
import subprocess
import sys

from local_agent.agent.validators import SolverValidationError, ValidationError


def smoke_with_deadline(script, session, timeout):
    output = Path(script).parent / 'smoke_worker_result.json'
    output.unlink(missing_ok=True)
    log_path = Path(script).parent / 'solver_recovery_worker.log'
    with log_path.open('a') as log:
        process = subprocess.Popen(
            [sys.executable, '-m', 'local_agent.agent.smoke_worker', str(script), str(session), str(output)],
            stdout=log, stderr=subprocess.STDOUT, start_new_session=True,
        )
        try:
            code = process.wait(timeout=max(timeout, 0.001))
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()
            raise SolverValidationError({'code': 'recovery_timeout', 'result': 'Numerical recovery wall-time limit reached'})
    if code != 0 or not output.exists():
        raise SolverValidationError({'code': 'recovery_worker_failure', 'result': f'Worker exit {code}; see {log_path.name}'})
    try:
        result = json.loads(output.read_text())
    except (OSError, ValueError) as exc:
        raise SolverValidationError({'code': 'recovery_worker_failure', 'result': str(exc)}) from exc
    if result.get('solver_diagnostics'):
        raise SolverValidationError(result['solver_diagnostics'])
    if not result['success']:
        raise ValidationError(result['error'])
