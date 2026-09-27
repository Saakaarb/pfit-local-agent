"""Isolated smoke validation so numerical recovery has an enforceable deadline."""
import json
from pathlib import Path
import sys

from local_agent.agent.validators import SolverValidationError, smoke_test_generated_script


def main():
    script, session, output = map(Path, sys.argv[1:])
    try:
        smoke_test_generated_script(script, session)
        result = {'success': True}
    except SolverValidationError as exc:
        result = {'success': False, 'solver_diagnostics': exc.diagnostics}
    except Exception as exc:
        result = {'success': False, 'error': str(exc)}
    output.write_text(json.dumps(result) + '\n')


if __name__ == '__main__':
    main()
