"""Isolated manual-source retry; uses the existing instrumented benchmark worker."""
import json
import os
from pathlib import Path
import sys
import time

REPO = Path('/workspace/pfit-local-agent')
sys.path.insert(0, str(REPO/'scripts'))
from run_model_comparison import stage, save, utc

root = Path(sys.argv[1]).resolve()
session = root/'session'
(root/'logs').mkdir(exist_ok=True)
entry = dict(case=root.name, status='running', manual_intervention=False,
             started_utc=utc(), stages={}, protocol='protocol.json')
save(root/'metadata.json', entry)
env = dict(os.environ, PYTHONPATH=str(REPO), OPENBLAS_NUM_THREADS='1', JAX_PLATFORMS='cpu', JAX_ENABLE_X64='true')
flags = ['--model', 'qwen3.8:27b', '--base-url', 'http://127.0.0.1:11434',
         '--timeout-seconds', '600', '--max-tokens', '12000', '--temperature', '0.1']
started = time.monotonic()
try:
    for label in ('check', 'jax', 'run', 'diagnose'):
        command = [label, str(session)]
        if label == 'run': command.append('--no-sloppiness')
        if label == 'diagnose': command.append(entry['run_id'])
        if label != 'run': command += flags
        if label == 'jax': command += ['--max-repair-attempts', '5']
        print(label, flush=True)
        result = stage(root, label, command, 3600 if label == 'jax' else 1800, env)
        entry['stages'][label] = result
        save(root/'metadata.json', entry)
        if result['status'] != 'pass':
            entry.update(status='fail', failed_stage=label)
            break
        if label == 'run':
            run = sorted((session/'outputs').glob('run_*'))[-1]
            entry['run_id'] = run.name
            fit = json.loads((run/'fit_summary.json').read_text())
            entry['fit_summary'] = fit
            seeds = json.loads((run/'accuracy_seeds.json').read_text())
            entry['validated_seed_count'] = len(seeds['validated_seeds'])
        if label == 'diagnose':
            diagnosis = json.loads((run/'ollama_diagnosis.json').read_text())
            entry['diagnosis_status'] = diagnosis['status']
            entry['status'] = 'pass' if diagnosis['status']=='ok' and fit['termination']=='iteration_budget' and not fit.get('refinement_error') else 'degraded'
except Exception as exc:
    entry.update(status='fail', failed_stage=label, error=str(exc))
finally:
    entry.update(finished_utc=utc(), total_wall_seconds=round(time.monotonic()-started, 3))
    save(root/'metadata.json', entry)
    print(entry['status'], entry['total_wall_seconds'], flush=True)
