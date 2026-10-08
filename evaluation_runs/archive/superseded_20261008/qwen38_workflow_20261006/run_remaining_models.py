"""Run the two additional model batches sequentially using the existing runner."""
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[1]
PLAN = ROOT/'extension_20261007.json'

def update(**fields):
    plan = json.loads(PLAN.read_text())
    plan.update(fields)
    temp = PLAN.with_suffix('.tmp')
    temp.write_text(json.dumps(plan, indent=2)+'\n')
    temp.replace(PLAN)

try:
    for model in ('qwen14b', 'qwen32b'):
        update(status='running', active_model=model)
        command = [str(REPO/'.venv/bin/python'), '-u', str(REPO/'scripts/run_model_comparison.py'),
                   '--root', str(ROOT), '--models', model,
                   '--report-models', 'qwen38_27b', 'qwen32b', 'qwen14b', '--rotate-models']
        subprocess.run(command, cwd=REPO, check=True)
        batch = json.loads((ROOT/'manifest.json').read_text())['model_batches'][model]
        if batch['status'] != 'completed':
            raise RuntimeError(f'{model} batch did not complete: {batch}')
    update(status='completed', active_model=None)
except Exception as exc:
    update(status='error', error=str(exc))
    raise
