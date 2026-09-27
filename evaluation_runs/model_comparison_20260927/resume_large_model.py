"""Retry infrastructure-only preflight after RunPod disk accounting catches up."""
import fcntl
import json
from pathlib import Path
import sys
import time
import shutil

REPO = Path('/workspace/pfit-local-agent')
sys.path.insert(0, str(REPO / 'scripts'))
import run_model_comparison as benchmark
root = Path(__file__).resolve().parent
state_path = root / 'recovery.json'
benchmark.save(state_path, dict(status='waiting_for_previous_runner', started_utc=benchmark.utc()))
with (root / '.runner.lock').open('a') as lock:
    fcntl.flock(lock, fcntl.LOCK_EX)
    archive = root / 'infrastructure_attempts' / time.strftime('%Y%m%d_%H%M%S', time.gmtime())
    archive.mkdir(parents=True)
    manifest = json.loads((root / 'manifest.json').read_text())
    benchmark.save(archive / 'manifest.json', manifest)
    for path in (root / 'cases').glob('*/qwen3_coder_next/metadata.json'):
        entry = json.loads(path.read_text())
        if entry['status'] == 'infrastructure_error':
            dest = archive / path.parent.parent.name / 'metadata.json'
            dest.parent.mkdir(parents=True)
            path.rename(dest)
    manifest.update(status='resuming_large_model', infrastructure_recovery=str(archive.relative_to(root)))
    benchmark.save(root / 'manifest.json', manifest)

original = benchmark.ensure_model

def wait_for_disk(*args, **kwargs):
    deadline = time.monotonic() + 600
    while True:
        try:
            result = original(*args, **kwargs)
            benchmark.save(state_path, dict(status='model_ready', model=args[1]['key'], utc=benchmark.utc()))
            return result
        except RuntimeError as exc:
            if 'Insufficient disk:' not in str(exc) or time.monotonic() >= deadline:
                raise
            benchmark.save(state_path, dict(status='waiting_for_disk_accounting', free_bytes=shutil.disk_usage('/workspace').free,
                error=str(exc), utc=benchmark.utc()))
            time.sleep(10)

benchmark.ensure_model = wait_for_disk
sys.argv = [str(REPO / 'scripts/run_model_comparison.py'), '--root', str(root), '--models',
            'qwen3_coder_next', '--rotate-models', '--restore-current-model']
benchmark.save(state_path, dict(status='resuming_large_model', utc=benchmark.utc()))
try:
    benchmark.main()
finally:
    benchmark.save(state_path, dict(status='runner_finished', utc=benchmark.utc()))
