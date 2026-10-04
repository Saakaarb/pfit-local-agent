"""Smoke-test the external cohort sequentially, with isolated model directories."""
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


def hashes(directory):
    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in directory.iterdir() if p.is_file()}


def reuse_case(source, root, case, model):
    directory = source / 'cases' / case['name'] / model['key']
    path = directory / 'metadata.json'
    if not path.exists():
        return False
    entry = json.loads(path.read_text())
    if entry['status'] not in ('pass', 'fail', 'degraded'):
        raise ValueError(f'Cannot import unfinished attempt: {path}')
    if hashes(source / 'frozen_inputs' / case['name']) != case['input_hashes']:
        raise ValueError(f'Reused inputs differ: {case["name"]}')
    expected = dict(options=runner.OPTIONS, max_repair_attempts=5, request_timeout_seconds=600,
        gradient_iterations=5, population_size=4, population_iterations=1,
        gradient_optimizer='adam', numerical_backend='cpu', jax_x64=True, sloppiness=False)
    if entry['settings'] != expected or entry['model_metadata']['model'] != model['model']:
        raise ValueError(f'Reused model/settings differ: {path}')
    if entry['input_audit'].get('pre_run_configuration', {}) != case['pre_run_configuration']:
        raise ValueError(f'Reused initial-time setup differs: {path}')
    shutil.copytree(directory, root / 'cases' / case['name'] / model['key'])
    runner.save(root / 'cases' / case['name'] / model['key'] / 'import_provenance.json',
        dict(source=str(directory.relative_to(runner.REPO)), imported_utc=runner.utc(),
             original_framework_commit=entry['model_metadata']['framework_commit'],
             note='Original single attempt retained, including failures; paths in its logs refer to the source run.'))
    return True


def report(root, cases, models):
    rows = []
    lines = ['# External-case smoke comparison', '',
        'One attempt per model/case, including the existing repair budget. '
        'DE: population 4, one iteration; Adam: five iterations; no sloppiness analysis. '
        'A workflow pass is not a converged fit or an independent validation of the scientific extraction.', '',
        '| Case | ' + ' | '.join(m['label'] for m in models) + ' |',
        '|---|' + '---|' * len(models)]
    for case in cases:
        cells = []
        for model in models:
            directory = root / 'cases' / case['name'] / model['key']
            path = directory / 'metadata.json'
            entry = json.loads(path.read_text()) if path.exists() else dict(status='pending')
            row = dict(case=case['name'], model=model['key'], status=entry['status'],
                failed_stage=entry.get('failed_stage', ''), total_wall_seconds=entry.get('total_wall_seconds', ''),
                llm_wall_seconds=entry.get('llm_wall_seconds', ''),
                model_size_bytes=entry.get('model_metadata', {}).get('size_bytes', ''),
                gguf_parameter_count=entry.get('model_metadata', {}).get('gguf_parameter_count', ''),
                nominal_parameters_billion=model['nominal_total_parameters_billion'],
                quantization=model['quantization'], thinking_model=model['thinking_model'],
                imported=(directory / 'import_provenance.json').exists())
            row.update({s + '_seconds': entry.get('stages', {}).get(s, {}).get('seconds', '') for s in runner.STAGES})
            rows.append(row)
            label = row['status'] + (f' at {row["failed_stage"]}' if row['failed_stage'] else '')
            cells.append(label + (f' ({row["total_wall_seconds"]} s)' if row['total_wall_seconds'] != '' else ''))
        lines.append('| ' + case['name'] + ' | ' + ' | '.join(cells) + ' |')
    lines += ['', 'The first three 32B attempts may be imported from the earlier external smoke run; '
        'per-case import records preserve their origin. All models receive the same frozen prose/CSV inputs. '
        'Model digests, actual sizes, parameter counts, GPU placement, warm-up and download times are in models/. '
        'Case times include repairs and failed attempts, and exclude model preparation. '
        'Both selected models are dense, non-thinking Qwen2.5-Coder models. '
        'These six cases are separate from the earlier 16-case comparison. '
        'Single attempts do not estimate success probabilities or timing variance.', '']
    (root / 'comparison.md').write_text('\n'.join(lines))
    with (root / 'comparison.csv').open('w') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--models', nargs='+', choices=['qwen32b', 'qwen14b'], default=['qwen32b'])
    parser.add_argument('--reuse-root', type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    root.mkdir(parents=True, exist_ok=False)
    cohort = runner.REPO / 'benchmarks/external_cases_20261004'
    inventory = json.loads((cohort / 'inventory.json').read_text())
    available = json.loads((runner.REPO / 'benchmarks/model_comparison/models.json').read_text())
    models = [next(m for m in available if m['key'] == key) for key in dict.fromkeys(args.models)]
    env = dict(os.environ, JAX_PLATFORMS='cpu', JAX_ENABLE_X64='true', PYTHONUNBUFFERED='1')
    manifest = dict(status='preparing', started_utc=runner.utc(), pid=os.getpid(),
        framework_commit=runner.command_output(['git', 'rev-parse', 'HEAD']),
        models=models, population_size=4, population_iterations=1, gradient_iterations=5,
        sloppiness=False, case_results=[], model_comparison=len(models)>1,
        gpu=runner.command_output(['nvidia-smi', '--query-gpu=name,memory.total,driver_version', '--format=csv']),
        python=sys.version, dependencies=runner.command_output([str(runner.PYTHON), '-m', 'pip', 'freeze']))
    runner.save(root / 'manifest.json', manifest)
    cases = []
    try:
        for item in sorted(inventory['cases'], key=lambda c: c['priority']):
            name = item['id']
            frozen = root / 'frozen_inputs' / name
            shutil.copytree(cohort / item['input_directory'], frozen)
            case = dict(name=name, status='ready', expected_experiments=item['experiments'],
                expected_data_files=sorted(item['csv_files']), sloppiness=False,
                pre_run_configuration=item['pre_run_configuration'], input_hashes=hashes(frozen))
            cases.append(case)
            if args.reuse_root:
                for model in models:
                    reuse_case(args.reuse_root.resolve(), root, case, model)
        runner.save(root / 'input_audit.json', cases)
        runner.save(root / 'cohort_inventory.json', inventory)
        manifest['status'] = 'running'
        manifest['case_results'] = report(root, cases, models)
        runner.save(root / 'manifest.json', manifest)
        for model in models:
            # Unload prior weights before warming up the next model; keep files on disk.
            for loaded in runner.api('ps')['models']:
                if loaded['name'] != model['model']:
                    runner.api('generate', dict(model=loaded['name'], keep_alive=0))
            metadata = runner.ensure_model(root, model, False, env)
            for case in cases:
                path = root / 'cases' / case['name'] / model['key'] / 'metadata.json'
                if path.exists():
                    continue
                runner.run_case(root, case, model, metadata, env)
                manifest['case_results'] = report(root, cases, models)
                runner.save(root / 'manifest.json', manifest)
        manifest['status'] = 'completed'
    except Exception as exc:
        manifest.update(status='infrastructure_error', error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        manifest['finished_utc'] = runner.utc()
        runner.save(root / 'manifest.json', manifest)


if __name__ == '__main__':
    main()
