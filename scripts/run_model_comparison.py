"""Sequential, resumable Ollama comparison using frozen scientist-written inputs.

Run prepare_model_comparison.py first. Model weights may be removed between
batches; all inputs, model identities, timing records and results are retained.
The worker instruments the existing client in-process, leaving production code
unchanged. It fixes the same sampling/context options for every model.
"""
import argparse
import csv
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
import urllib.request

import yaml

REPO = Path(__file__).resolve().parents[1]
PYTHON = REPO / '.venv/bin/python'
OLLAMA = Path('/workspace/ollama/bin/ollama')
BASE = 'http://127.0.0.1:11434'
OPTIONS = dict(seed=7, num_ctx=32768, temperature=0.1, num_predict=12000, top_k=40, top_p=0.9)
STAGES = ('new', 'check', 'jax', 'run', 'diagnose')


def utc():
    return time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2) + '\n')
    tmp.replace(path)


def api(path, body=None, timeout=600):
    request = urllib.request.Request(BASE + '/api/' + path,
        data=None if body is None else json.dumps(body).encode(),
        headers={'Content-Type': 'application/json'})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.load(response)


def command_output(command):
    return subprocess.check_output(command, cwd=REPO, text=True).strip()


def instrumented_worker(directory, stage, arguments):
    sys.path.insert(0, str(REPO))
    from local_agent.llm.ollama import OllamaClient
    from local_agent.cli.main import main
    original = OllamaClient._post_json

    def timed(self, path, payload):
        # Benchmark-only controls, equal across models. No prompt modifications.
        payload['options'].update(OPTIONS)
        payload['keep_alive'] = -1
        record = dict(started_utc=utc(), stage=stage, model=self.model,
                      options=payload['options'].copy(), status='error')
        start = time.monotonic()
        try:
            response = original(self, path, payload)
            record.update(status='ok', **{k: response.get(k) for k in (
                'total_duration', 'load_duration', 'prompt_eval_count', 'prompt_eval_duration',
                'eval_count', 'eval_duration', 'done_reason')})
            record['thinking_characters'] = len((response.get('message') or {}).get('thinking') or '')
            return response
        except Exception as exc:
            record['error'] = f'{type(exc).__name__}: {exc}'
            raise
        finally:
            record['wall_seconds'] = time.monotonic() - start
            with (directory / 'llm_metrics.jsonl').open('a') as handle:
                handle.write(json.dumps(record) + '\n')

    OllamaClient._post_json = timed
    raise SystemExit(main(arguments))


def stage(directory, label, command, timeout, env):
    start = time.monotonic()
    record = dict(started_utc=utc(), command=command)
    with (directory / 'logs' / (label + '.log')).open('w') as log:
        process = subprocess.Popen([str(PYTHON), str(Path(__file__).resolve()), '--worker',
            str(directory), label, *command], cwd=REPO, env=env, stdout=log,
            stderr=subprocess.STDOUT, start_new_session=True)
        record['pid'] = process.pid
        save(directory / 'active_stage.json', dict(stage=label, **record))
        try:
            code = process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
            code = 124
    record.update(exit_code=code, seconds=round(time.monotonic() - start, 3),
                  finished_utc=utc(), status='pass' if code == 0 else 'fail',
                  timed_out=code == 124)
    return record


def report(root, models, inventory):
    rows = []
    lines = ['# Live model comparison', '',
        'Single attempt per model/case, including the existing repair budget. '
        'Times include failed attempts and must be interpreted alongside status. '
        'Downloads and warm-up are recorded separately. These are workflow smoke tests, not converged fits.', '',
        '| Case | ' + ' | '.join(m['key'] + ' (status; seconds)' for m in models) + ' |',
        '|---|' + '---|' * len(models)]
    for case in inventory:
        cells = []
        for model in models:
            path = root / 'cases' / case['name'] / model['key'] / 'metadata.json'
            entry = json.loads(path.read_text()) if path.exists() else dict(status='pending')
            row = dict(case=case['name'], model=model['key'], status=entry['status'],
                failed_stage=entry.get('failed_stage', ''), total_wall_seconds=entry.get('total_wall_seconds', ''),
                llm_wall_seconds=entry.get('llm_wall_seconds', ''), repair_calls=entry.get('repair_calls', ''),
                model_size_bytes=entry.get('model_metadata', {}).get('size_bytes', ''),
                total_parameters_billion=model['nominal_total_parameters_billion'],
                active_parameters_billion=model['nominal_active_parameters_billion'],
                quantization=model['quantization'], thinking_model=model['thinking_model'],
                duplicate_of=case.get('duplicate_of', ''))
            row.update({s + '_seconds': entry.get('stages', {}).get(s, {}).get('seconds', '') for s in STAGES})
            rows.append(row)
            label = entry['status']
            if entry.get('failed_stage'):
                label += ' at ' + entry['failed_stage']
            cells.append(f'{label}; {row["total_wall_seconds"]}' if row['total_wall_seconds'] != '' else label)
        lines.append('| ' + case['name'] + ' | ' + ' | '.join(cells) + ' |')
    lines += ['', '## Counts', '', '| Model | Pass | Fail/degraded | Infrastructure/blocked | Pending/running |',
              '|---|---:|---:|---:|---:|']
    for model in models:
        selected = [r for r in rows if r['model'] == model['key']]
        count = lambda statuses: sum(r['status'] in statuses for r in selected)
        lines.append(f'| {model["key"]} | {count(["pass"])} | {count(["fail", "degraded"])} | '
                     f'{count(["blocked", "infrastructure_error", "interrupted"])} | {count(["pending", "running"])} |')
    lines += ['', 'Two incomplete folders are excluded from the eligible denominator. '
        'test_session duplicates Robertson; sliding_basepoint_headered duplicates sliding_basepoint. '
        'Keep these rows visible but do not treat them as independent scientific problems.', '',
        'A pass requires all five stages, the declared experiment files/count, finite result arrays '
        'and final loss, completed refinement, and accepted Ollama diagnosis. The framework applies '
        'semantic checks and source-to-JAX fidelity checks. This is not an independent proof that '
        'every generated equation matches the original scientific specification.', '',
        'Models differ in architecture, generation, and quantization. Dense total parameters are used '
        'as nominal active counts; the MoE 3B active count is approximate. All are non-thinking. '
        'One trial cannot estimate success probabilities or timing variance.', '']
    (root / 'comparison.md').write_text('\n'.join(lines))
    if rows:
        with (root / 'comparison.csv').open('w') as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


def run_case(root, case, model, metadata, env):
    directory = root / 'cases' / case['name'] / model['key']
    # Session intake recursively reads files under its root. Keep all benchmark
    # metadata and logs in the parent, outside the scientist's session context.
    session = directory / 'session'
    entry = dict(case=case['name'], model_metadata=metadata, input_audit=case,
        status='running', started_utc=utc(), stages={}, settings=dict(options=OPTIONS,
        max_repair_attempts=5, request_timeout_seconds=600, gradient_iterations=5,
        population_size=4, population_iterations=1, gradient_optimizer='adam',
        numerical_backend='cpu', jax_x64=True, sloppiness=True))
    start = time.monotonic()
    save(directory / 'metadata.json', entry)
    if case['status'] == 'blocked':
        entry.update(status='blocked', reason=case['reason'], total_wall_seconds=0)
        save(directory / 'metadata.json', entry)
        return
    (directory / 'logs').mkdir()
    shutil.copytree(root / 'frozen_inputs' / case['name'], session / 'inputs')
    flags = ['--model', model['model'], '--base-url', BASE, '--timeout-seconds', '600',
             '--max-tokens', '12000', '--temperature', '0.1']
    try:
        for label in STAGES:
            command = [label, str(session)]
            if label == 'diagnose':
                command.append(entry['run_id'])
            if label != 'run':
                command += flags
            if label in ('new', 'jax'):
                command += ['--max-repair-attempts', '5']
            # There is no existing YAML/code to preserve or leak into new.
            timeout = {'new': 1800, 'check': 900, 'jax': 1800, 'run': 1800, 'diagnose': 900}[label]
            print(f'{model["key"]}/{case["name"]}: {label}', flush=True)
            outcome = stage(directory, label, command, timeout, env)
            entry['stages'][label] = outcome
            save(directory / 'metadata.json', entry)
            if outcome['status'] != 'pass':
                entry.update(status='fail', failed_stage=label)
                break
            if label == 'new':
                config_path = session / 'inputs/user_input.yaml'
                shutil.copy2(config_path, directory / 'extracted_user_input.yaml')
                config = yaml.safe_load(config_path.read_text())
                files = sorted(e['data_file'] for e in config['experiments'])
                if files != case['expected_data_files']:
                    raise ValueError(f'Wrong experiment set: expected {case["expected_data_files"]}, got {files}')
                config.setdefault('population_opt', {}).update(algorithm='DE', population_size=4,
                    num_iters=1, processors=1, random_seed=7)
                config.setdefault('gradient_opt', {}).update(gradient_optimizer='adam', num_iters=5)
                config_path.write_text(yaml.safe_dump(config, sort_keys=False))
                entry['smoke_budget_override'] = dict(population=config['population_opt'], gradient=config['gradient_opt'])
            elif label == 'run':
                import numpy as np
                run = sorted((session / 'outputs').glob('run_*'))[-1]
                entry['run_id'] = run.name
                fit = entry['fit_summary'] = json.loads((run / 'fit_summary.json').read_text())
                outputs = sorted(run.glob('result_solution*.csv'))
                entry['output_count'] = len(outputs)
                if len(outputs) != case['expected_experiments']:
                    raise ValueError('Wrong number of experiment outputs')
                for path in outputs:
                    values = np.loadtxt(path, delimiter=',', skiprows=1)
                    if not values.size or not np.all(np.isfinite(values)):
                        raise ValueError(f'Empty or non-finite output: {path.name}')
                if not np.isfinite(fit['final_loss']) or fit['final_loss'] >= float(config['gradient_opt'].get('error_loss', 1e10)):
                    raise ValueError('Non-finite/sentinel final loss')
            elif label == 'diagnose':
                diagnosis = json.loads((run / 'ollama_diagnosis.json').read_text())
                entry['diagnosis_status'] = diagnosis['status']
                entry['status'] = ('pass' if diagnosis['status'] == 'ok'
                    and fit['termination'] == 'iteration_budget' and not fit.get('refinement_error') else 'degraded')
    except Exception as exc:
        entry.update(status='fail', failed_stage=label, error=f'{type(exc).__name__}: {exc}')
    finally:
        entry.update(finished_utc=utc(), total_wall_seconds=round(time.monotonic() - start, 3))
        calls = session / 'generated/agent_logs/llm_calls.jsonl'
        if calls.exists():
            entries = [json.loads(line) for line in calls.read_text().splitlines()]
            entry['repair_calls'] = sum(e.get('step', '').startswith('repair_') for e in entries)
            entry['logged_successful_llm_calls'] = len(entries)
        metrics = directory / 'llm_metrics.jsonl'
        if metrics.exists():
            records = [json.loads(line) for line in metrics.read_text().splitlines()]
            entry['llm_calls'] = len(records)
            entry['llm_wall_seconds'] = sum(r['wall_seconds'] for r in records)
            entry['prompt_tokens'] = sum(r.get('prompt_eval_count') or 0 for r in records)
            entry['generated_tokens'] = sum(r.get('eval_count') or 0 for r in records)
            entry['token_counts_incomplete'] = any(r['status'] != 'ok' for r in records)
        save(directory / 'metadata.json', entry)
        print(f'{model["key"]}/{case["name"]}: {entry["status"]} ({entry["total_wall_seconds"]}s)', flush=True)


def ensure_model(root, model, rotate, env):
    directory = root / 'models' / model['key']
    directory.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    installed = api('tags')['models']
    present = any(m['name'] == model['model'] for m in installed)
    if not present:
        # Only remove known comparison models, and only after their completed batch.
        if rotate:
            known = {m['model'] for m in json.loads((REPO / 'benchmarks/model_comparison/models.json').read_text())}
            for item in installed:
                if item['name'] in known and item['name'] != model['model']:
                    subprocess.run([str(OLLAMA), 'stop', item['name']], env=env, check=True, stdout=subprocess.DEVNULL)
                    subprocess.run([str(OLLAMA), 'rm', item['name']], env=env, check=True)
        free = shutil.disk_usage('/workspace').free
        if free < model['expected_download_bytes'] + 2_000_000_000:
            raise RuntimeError(f'Insufficient disk: {free} free, need weights plus 2GB headroom')
        with (directory / 'pull.log').open('a') as log:
            subprocess.run([str(OLLAMA), 'pull', model['model']], env=env, check=True,
                           stdout=log, stderr=subprocess.STDOUT, timeout=7200)
    tags = api('tags')['models']
    tag = next(m for m in tags if m['name'] == model['model'])
    show = api('show', {'model': model['model']})
    save(directory / 'ollama_show.json', show)
    result = dict(model, digest=tag['digest'], size_bytes=tag['size'], details=tag.get('details'),
        installed_at_batch_start=present, preparation_seconds=time.monotonic() - started,
        ollama_version=api('version'), capabilities=show.get('capabilities'),
        gguf_parameter_count=show.get('model_info', {}).get('general.parameter_count'),
        options=OPTIONS, framework_commit=command_output(['git', 'rev-parse', 'HEAD']))
    save(directory / 'metadata.json', result)
    warm_start = time.monotonic()
    warm = api('chat', dict(model=model['model'], messages=[dict(role='user', content='Reply with OK.')],
                           stream=False, options=dict(OPTIONS, num_predict=8), keep_alive=-1))
    result['warmup_seconds'] = time.monotonic() - warm_start
    result['warmup_response'] = warm
    result['loaded_model'] = api('ps')
    result['gpu_after_load'] = command_output(['nvidia-smi', '--query-gpu=name,memory.total,memory.used', '--format=csv'])
    save(directory / 'metadata.json', result)
    # Refuse CPU-offloaded timing comparisons.
    loaded = result['loaded_model']['models']
    if not loaded or any(m.get('size_vram', 0) < m.get('size', 0) for m in loaded):
        raise RuntimeError('Model was not fully loaded on GPU; see model metadata')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--models', nargs='*')
    parser.add_argument('--rotate-models', action='store_true')
    parser.add_argument('--restore-current-model', action='store_true')
    args = parser.parse_args()
    root = args.root.resolve()
    lock = (root / '.runner.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    inventory = json.loads((root / 'input_audit.json').read_text())
    models = json.loads((REPO / 'benchmarks/model_comparison/models.json').read_text())
    selected = [m for m in models if not args.models or m['key'] in args.models]
    if not selected or (args.models and set(args.models) - {m['key'] for m in models}):
        parser.error('Select at least one known comparison model')
    env = dict(os.environ, JAX_PLATFORMS='cpu', JAX_ENABLE_X64='true',
               XLA_PYTHON_CLIENT_PREALLOCATE='false', PYTHONUNBUFFERED='1')
    manifest_path = root / 'manifest.json'
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else dict(
        started_utc=utc(), framework_commit=command_output(['git', 'rev-parse', 'HEAD']),
        gpu=command_output(['nvidia-smi', '--query-gpu=name,memory.total,driver_version', '--format=csv']),
        python=sys.version, dependencies=command_output([str(PYTHON), '-m', 'pip', 'freeze']),
        options=OPTIONS, model_order=[m['key'] for m in selected], model_batches={})
    current_commit = command_output(['git', 'rev-parse', 'HEAD'])
    if manifest['framework_commit'] != current_commit:
        raise RuntimeError('Refusing resume at a different framework commit')
    for path in (root / 'cases').glob('*/*/metadata.json'):
        previous = json.loads(path.read_text())
        if previous['status'] == 'running':
            active = path.parent / 'active_stage.json'
            if active.exists():
                pid = json.loads(active.read_text())['pid']
                try:
                    os.kill(pid, 0)
                except ProcessLookupError:
                    pass
                else:
                    raise RuntimeError(f'Previous worker {pid} is still alive; refusing overlapping run')
            previous.update(status='interrupted', reason='Previous runner stopped; attempt preserved without retry')
            save(path, previous)
    # Detect accidental input edits before consuming GPU time.
    for case in inventory:
        for name, expected in case.get('input_hashes', {}).items():
            path = root / 'frozen_inputs' / case['name'] / name
            if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
                raise ValueError(f'Frozen input changed: {path}')
    manifest.update(status='running', pid=os.getpid())
    save(manifest_path, manifest)
    report(root, selected, inventory)
    for model in selected:
        outstanding = [c for c in inventory if not (root / 'cases' / c['name'] / model['key'] / 'metadata.json').exists()]
        if not outstanding:
            continue
        batch = manifest['model_batches'][model['key']] = dict(status='preparing', started_utc=utc())
        save(manifest_path, manifest)
        print(f'Preparing {model["model"]}', flush=True)
        try:
            metadata = ensure_model(root, model, args.rotate_models, env)
        except Exception as exc:
            batch.update(status='infrastructure_error', error=f'{type(exc).__name__}: {exc}')
            for case in outstanding:
                save(root / 'cases' / case['name'] / model['key'] / 'metadata.json',
                     dict(status='blocked' if case['status'] == 'blocked' else 'infrastructure_error',
                          reason=case.get('reason', batch['error']), model_metadata=model))
            save(manifest_path, manifest)
            report(root, selected, inventory)
            continue
        batch['status'] = 'running'
        save(manifest_path, manifest)
        # A simple case first, then a fixed alphabetical order for every model.
        for case in sorted(outstanding, key=lambda c: (c['name'] != 'theophylline', c['name'])):
            run_case(root, case, model, metadata, env)
            report(root, selected, inventory)
        batch.update(status='completed', finished_utc=utc())
        save(manifest_path, manifest)
    if args.restore_current_model:
        try:
            # Restore the user's configured default after the final batch. Do not
            # overwrite its original benchmark load/preparation metadata.
            restore_root = root / 'restoration'
            ensure_model(restore_root, models[0], args.rotate_models, env)
            manifest['default_model_restored'] = True
        except Exception as exc:
            manifest['default_model_restoration_error'] = str(exc)
    manifest.update(status='completed', finished_utc=utc())
    save(manifest_path, manifest)
    report(root, selected, inventory)
    print(f'Completed: {root / "comparison.md"}', flush=True)


if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == '--worker':
        instrumented_worker(Path(sys.argv[2]), sys.argv[3], sys.argv[4:])
    main()
