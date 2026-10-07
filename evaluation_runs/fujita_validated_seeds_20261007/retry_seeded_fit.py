import json
import os
from pathlib import Path
import sys
import time
REPO = Path('/workspace/pfit-local-agent')
sys.path.insert(0, str(REPO / 'scripts'))
from run_model_comparison import stage, save, utc
root = Path(__file__).resolve().parent
session = root / 'session'
result_dir = root / 'seeded_retry'
(result_dir / 'logs').mkdir(parents=True, exist_ok=True)
env = dict(os.environ, PYTHONPATH=str(REPO), OPENBLAS_NUM_THREADS='1', JAX_PLATFORMS='cpu', JAX_ENABLE_X64='true')
entry = dict(case='fujita_egf', status='running', started_utc=utc(), stages={}, purpose='Verify accuracy seeds reach fitting through run snapshot')
started = time.monotonic()
save(result_dir / 'metadata.json', entry)
try:
    result = stage(result_dir, 'run', ['run', str(session), '--no-sloppiness'], 1800, env)
    entry['stages']['run'] = result
    if result['status'] != 'pass':
        raise RuntimeError('Smoke fit failed; inspect run.log')
    run = sorted((session / 'outputs').glob('run_*'))[-1]
    entry['run_id'] = run.name
    entry['fit_summary'] = json.loads((run / 'fit_summary.json').read_text())
    seeds = json.loads((run / 'accuracy_seeds.json').read_text())
    entry['seed_artifact'] = seeds
    assert 'Using 3 accuracy-validated seed points' in (result_dir / 'logs/run.log').read_text()
    assert not entry['fit_summary'].get('refinement_error')
    entry['status'] = 'pass'
except Exception as exc:
    entry.update(status='fail', error=str(exc))
finally:
    entry.update(finished_utc=utc(), total_wall_seconds=round(time.monotonic()-started, 3))
    save(result_dir / 'metadata.json', entry)
    print(json.dumps(entry, indent=2), flush=True)
