"""Live case regression suite. Isolated inputs, stage logs, explicit fit budgets."""
import argparse
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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--gradient-iters', type=int, default=5)
    parser.add_argument('--cases', nargs='*')
    parser.add_argument('--overwrite-config', action='store_true',
                        help='Regenerate YAML as well as model code; default preserves the copied case YAML')
    parser.add_argument('--add-declared-headers', action='store_true',
                        help='Add YAML-declared headers to headerless CSV copies; preserve all numerical rows')
    args = parser.parse_args()
    root = args.root.resolve()
    root.mkdir(parents=True, exist_ok=False)
    env = dict(os.environ, JAX_PLATFORMS='cpu', JAX_ENABLE_X64='true',
               XLA_PYTHON_CLIENT_PREALLOCATE='false', PYTHONUNBUFFERED='1')
    model = 'qwen2.5-coder:32b'
    base = 'http://127.0.0.1:11434'
    flags = ['--model', model, '--base-url', base, '--timeout-seconds', '600',
             '--max-tokens', '12000', '--temperature', '0.1']
    manifest = dict(commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=REPO, text=True).strip(),
        started_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), status='running',
        settings=dict(model=model, max_repair_attempts=5, temperature=.1, max_tokens=12000,
            population_size=4, population_iterations=1, gradient_optimizer='adam',
            gradient_iterations=args.gradient_iters, fitting_timeout_seconds=1800,
            purpose='bounded workflow regression, not convergence',
            config_policy='regenerate' if args.overwrite_config else 'preserve_existing'), results={})
    for endpoint in ('version', 'tags'):
        with urllib.request.urlopen(base + '/api/' + endpoint, timeout=15) as response:
            manifest['ollama_' + endpoint] = json.load(response)

    def save():
        tmp = root / 'manifest.tmp'
        tmp.write_text(json.dumps(manifest, indent=2) + '\n')
        tmp.replace(root / 'manifest.json')

    def step(key, label, command, timeout=7200):
        log = root / 'logs' / key / (label + '.log')
        log.parent.mkdir(parents=True, exist_ok=True)
        print(f'{key}: starting {label}', flush=True)
        start = time.monotonic()
        with log.open('w') as handle:
            proc = subprocess.Popen([str(PYTHON), '-m', 'local_agent.cli.main', *command],
                cwd=REPO, env=env, stdout=handle, stderr=subprocess.STDOUT, start_new_session=True)
            try:
                code = proc.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, signal.SIGTERM)
                try:
                    proc.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    os.killpg(proc.pid, signal.SIGKILL)
                    proc.wait()
                code = 124
        result = dict(status='pass' if code == 0 else 'fail', exit_code=code,
                      seconds=round(time.monotonic()-start, 2), log=str(log.relative_to(root)))
        manifest['results'][key][label] = result
        save()
        print(f'{key}: {label} {result["status"]} ({result["seconds"]}s)', flush=True)
        return code == 0

    cases = [(p.name, p, False) for p in sorted((REPO / 'sessions').iterdir()) if p.is_dir()]
    cases += [(p.name, p, True) for p in sorted((REPO / 'tests/fixtures').iterdir()) if (p / 'inputs').is_dir()]
    if args.cases:
        cases = [case for case in cases if case[0] in args.cases]
    # Run a simple case first as a harness sanity check.
    cases.sort(key=lambda case: (case[0] != 'theophylline', case[0]))
    manifest['inventory'] = [dict(name=n, source=str(p.relative_to(REPO)), fixture=f) for n,p,f in cases]
    save()
    for name, source, fixture in cases:
        key = name
        entry = manifest['results'][key] = dict(source=str(source.relative_to(REPO)), status='running')
        save()
        if not (source / 'inputs/user_info.txt').exists() or not (source / 'inputs/user_input.yaml').exists():
            entry.update(status='blocked', reason='Incomplete existing case: missing user specification or YAML')
            save()
            continue
        session = root / 'sessions' / name
        session.mkdir(parents=True)
        shutil.copytree(source / 'inputs', session / 'inputs')
        if args.add_declared_headers:
            declared = yaml.safe_load((source / 'inputs/user_input.yaml').read_text())
            entry['header_adaptations'] = []
            for experiment in declared['experiments']:
                path = session / 'inputs' / experiment['data_file']
                original = path.read_text()
                try:
                    float(original.splitlines()[0].split(',')[0])
                except ValueError:
                    continue
                names = [column['name'] for column in experiment['columns']]
                if len(original.splitlines()[0].split(',')) != len(names):
                    raise ValueError(f'Cannot add declared header to {path}: width mismatch')
                path.write_text(','.join(names) + '\n' + original)
                entry['header_adaptations'].append(dict(file=experiment['data_file'],
                    names=names, original_sha256=hashlib.sha256(original.encode()).hexdigest(),
                    numerical_rows_unchanged=True))
        # Fixture notes describe the loss only: supply their authoritative equations
        # and declarations explicitly, as in the earlier live fixture evaluations.
        if fixture:
            info = session / 'inputs/user_info.txt'
            info.write_text(info.read_text() + '\n\nReference specification:\n' +
                (source / 'inputs/user_input.yaml').read_text() +
                '\n\nReference Python model (preserve scientific expressions exactly):\n' +
                (source / 'generated/user_model.py').read_text())
        entry['input_hashes'] = {str(p.relative_to(session)): hashlib.sha256(p.read_bytes()).hexdigest()
                                 for p in (session / 'inputs').iterdir() if p.is_file()}
        save()
        try:
            if not step(key, 'new', ['new', str(session), *(['--overwrite'] if args.overwrite_config else []), *flags, '--max-repair-attempts', '5']):
                entry['status'] = 'fail'; continue
            config_path = session / 'inputs/user_input.yaml'
            shutil.copy2(config_path, session / 'pre_smoke_user_input.yaml')
            config = yaml.safe_load(config_path.read_text())
            entry['pre_smoke_gradient_settings'] = config.get('gradient_opt')
            config['population_opt'].update(algorithm='DE', population_size=4, num_iters=1, processors=1, random_seed=7)
            config['gradient_opt'].update(gradient_optimizer='adam', num_iters=args.gradient_iters)
            config_path.write_text(yaml.safe_dump(config, sort_keys=False))
            if not step(key, 'check', ['check', str(session), *flags]):
                entry['status'] = 'fail'; continue
            if not step(key, 'jax', ['jax', str(session), *flags, '--max-repair-attempts', '5']):
                entry['status'] = 'fail'; continue
            if not step(key, 'run', ['run', str(session)], timeout=1800):
                entry['status'] = 'fail'; continue
            run = sorted((session / 'outputs').glob('run_*'))[-1]
            entry['run_id'] = run.name
            fit = json.loads((run / 'fit_summary.json').read_text())
            entry['fit_summary'] = fit
            expected = len(config['experiments'])
            outputs = sorted(run.glob('result_solution*.csv'))
            entry['output_count'] = len(outputs)
            if len(outputs) != expected:
                raise ValueError(f'Expected {expected} result CSVs, found {len(outputs)}')
            import numpy as np
            for path in outputs:
                values = np.loadtxt(path, delimiter=',', skiprows=1)
                if values.size == 0:
                    raise ValueError(f'Empty output: {path.name}')
            if not np.isfinite(fit['final_loss']) or fit['final_loss'] >= float(config['gradient_opt'].get('error_loss', 1e10)):
                raise ValueError('Invalid final fitting loss')
            fit_ok = fit['termination'] == 'iteration_budget' and not fit.get('refinement_error')
            save()
            if not step(key, 'diagnose', ['diagnose', str(session), run.name, *flags], timeout=1200):
                entry['status'] = 'fail'; continue
            interpretation = json.loads((run / 'ollama_diagnosis.json').read_text())
            entry['ollama_diagnosis_status'] = interpretation['status']
            entry['status'] = 'pass' if fit_ok and interpretation['status'] == 'ok' else 'degraded'
        except Exception as exc:
            entry.update(status='fail', error=f'{type(exc).__name__}: {exc}')
        finally:
            calls = session / 'generated/agent_logs/llm_calls.jsonl'
            if calls.exists():
                entry['repair_calls'] = sum('"step": "repair_' in line for line in calls.read_text().splitlines())
            save()
            print(f'{key}: overall {entry["status"]}', flush=True)
    manifest.update(status='completed', finished_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()))
    save()
    print('Completed:', root / 'manifest.json', flush=True)


if __name__ == '__main__':
    main()
