"""Run the prepared external cohort once, using the existing smoke-test harness."""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import run_model_comparison as runner


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    root.mkdir(parents=True, exist_ok=False)
    cohort = runner.REPO / 'benchmarks/external_cases_20261004'
    inventory = json.loads((cohort / 'inventory.json').read_text())
    model = json.loads((runner.REPO / 'benchmarks/model_comparison/models.json').read_text())[0]
    env = dict(os.environ, JAX_PLATFORMS='cpu', JAX_ENABLE_X64='true', PYTHONUNBUFFERED='1')
    manifest = dict(status='running', started_utc=runner.utc(), pid=os.getpid(),
        framework_commit=runner.command_output(['git', 'rev-parse', 'HEAD']),
        model=model, population_size=4, population_iterations=1, gradient_iterations=5,
        sloppiness=False, case_results=[], model_comparison=False)
    runner.save(root / 'manifest.json', manifest)
    try:
        metadata = runner.ensure_model(root, model, False, env)
        for item in sorted(inventory['cases'], key=lambda c: c['priority']):
            name = item['id']
            frozen = root / 'frozen_inputs' / name
            shutil.copytree(cohort / item['input_directory'], frozen)
            case = dict(name=name, status='ready', expected_experiments=item['experiments'],
                expected_data_files=sorted(item['csv_files']), sloppiness=False,
                pre_run_configuration=item['pre_run_configuration'],
                input_hashes={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in frozen.iterdir()})
            runner.run_case(root, case, model, metadata, env)
            result = json.loads((root / 'cases' / name / model['key'] / 'metadata.json').read_text())
            manifest['case_results'].append(dict(case=name, status=result['status'],
                failed_stage=result.get('failed_stage', ''), seconds=result['total_wall_seconds']))
            runner.save(root / 'manifest.json', manifest)
            with (root / 'results.csv').open('w') as handle:
                writer = csv.DictWriter(handle, fieldnames=['case', 'status', 'failed_stage', 'seconds'])
                writer.writeheader()
                writer.writerows(manifest['case_results'])
        manifest['status'] = 'completed'
    except Exception as exc:
        manifest.update(status='infrastructure_error', error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        manifest['finished_utc'] = runner.utc()
        runner.save(root / 'manifest.json', manifest)


if __name__ == '__main__':
    main()
