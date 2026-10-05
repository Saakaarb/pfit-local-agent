"""Prepare isolated full-fit sessions, then run two CPU-pinned numerical queues.

Usage: .venv/bin/python scripts/run_qwen38_cpu_fits.py prepare|run
No Ollama calls. Originals and prior smoke outputs are never modified.
"""
import concurrent.futures
import copy
import datetime
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
ROOT = REPO / 'evaluation_runs/qwen38_cpu_fits_20261005'
SOURCE = REPO / 'evaluation_runs/qwen38_comparison_20261005/cases'


def save(path, data):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(data, indent=2) + '\n')
    tmp.replace(path)


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def cpu_slots():
    # Keep SMT siblings together, so concurrent fits do not share physical cores.
    cores = {}
    for cpu in sorted(os.sched_getaffinity(0)):
        topology = Path(f'/sys/devices/system/cpu/cpu{cpu}/topology')
        key = tuple((topology / f).read_text().strip() for f in
                    ('physical_package_id', 'core_id'))
        cores.setdefault(key, []).append(cpu)
    groups = [[], []]
    for i, cpus in enumerate(cores.values()):
        groups[i % 2].extend(cpus)
    if not all(groups):
        raise RuntimeError('Two independent physical CPU groups required')
    return [sorted(g) for g in groups]


def prepare():
    import yaml
    from lib.utils.source_stamp import verify_stamp, write_stamp, STAMP_PREFIX
    from local_agent.agent.readiness import check_ready
    ROOT.mkdir(exist_ok=False)
    slots = cpu_slots()
    entries = []
    for metadata in sorted(SOURCE.glob('*/qwen38_27b/metadata.json')):
        old = json.loads(metadata.read_text())
        name = metadata.parts[-3]
        entry = {'case': name, 'source': str(metadata.parent),
                 'source_status': old['status']}
        entries.append(entry)
        if old['status'] not in ('pass', 'degraded'):
            entry.update(status='blocked', reason=f"Prior workflow: {old['status']}, stage {old.get('failed_stage')}")
            continue
        source = metadata.parent / 'session'
        valid, detail = verify_stamp(source)
        if valid is not True:
            entry.update(status='blocked', reason=detail)
            continue
        session = ROOT / 'cases' / name / 'session'
        session.mkdir(parents=True)
        for directory in ('inputs', 'generated'):
            shutil.copytree(source / directory, session / directory,
                            ignore=shutil.ignore_patterns('__pycache__', 'agent_logs'))
        config_path = session / 'inputs/user_input.yaml'
        original = yaml.safe_load(config_path.read_text())
        config = copy.deepcopy(original)
        dimensions = len(config['model']['trainable_parameters'])
        population = max(64, min(160, math.ceil(8 * dimensions / 16) * 16))
        config['population_opt'].update(algorithm='DE', population_size=population,
                                        num_iters=500, processors=min(map(len, slots)), random_seed=7)
        config['gradient_opt'].update(gradient_optimizer='adam', num_iters=1000)
        # Only runtime optimizer controls change. All equations, data, solver
        # settings, tolerances, parameter order/bounds and loss stay identical.
        allowed = {'population_opt': {'algorithm', 'population_size', 'num_iters', 'processors', 'random_seed'},
                   'gradient_opt': {'gradient_optimizer', 'num_iters'}}
        restored = copy.deepcopy(config)
        for section, keys in allowed.items():
            for key in keys:
                restored[section].pop(key, None)
                if key in original[section]:
                    restored[section][key] = original[section][key]
        assert restored == original
        config_path.write_text(yaml.safe_dump(config, sort_keys=False))
        script = session / 'generated/generated_script.py'
        before = [x for x in script.read_text().splitlines() if not x.startswith(STAMP_PREFIX)]
        write_stamp(session)
        assert before == [x for x in script.read_text().splitlines() if not x.startswith(STAMP_PREFIX)]
        report = check_ready(session)
        entry.update(status='queued' if report.passed else 'blocked',
                     readiness_errors=report.critical_errors, readiness_warnings=report.warnings,
                     parameters=dimensions, session=str(session),
                     original_yaml_sha256=sha(source / 'inputs/user_input.yaml'),
                     fit_yaml_sha256=sha(config_path),
                     generated_source_sha256=sha(source / 'generated/generated_script.py'),
                     stamp_note='Verified original stamp; restamped isolated copy after whitelisted optimizer-only changes; executable code unchanged.',
                     original_population=original['population_opt'], population=config['population_opt'],
                     gradient=config['gradient_opt'])
        save(session.parent / 'status.json', entry)
    plan = {'created_at': now(), 'cpu_slots': slots,
            'memory_limit_bytes': int(Path('/sys/fs/cgroup/memory.max').read_text()),
            'framework_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=REPO, text=True).strip(),
            'sloppiness': False, 'reason': 'Focus CPU time on population search and gradient refinement; curvature analysis deferred.',
            'cases': entries}
    save(ROOT / 'plan.json', plan)
    print(json.dumps({'root': str(ROOT), 'cpu_slots': slots,
                      'queued': sum(x['status'] == 'queued' for x in entries),
                      'blocked': [x['case'] for x in entries if x['status'] == 'blocked']}, indent=2))


def worker(entries, cpus):
    for entry in entries:
        status_path = ROOT / 'cases' / entry['case'] / 'status.json'
        current = json.loads(status_path.read_text())
        if current['status'] != 'queued':
            continue  # Never overwrite or silently repeat an earlier attempt.
        session = Path(entry['session'])
        env = os.environ.copy()
        env.update(JAX_PLATFORMS='cpu', OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
                   MKL_NUM_THREADS='1', NUMEXPR_NUM_THREADS='1', PYTHONUNBUFFERED='1',
                   MPLCONFIGDIR='/workspace/.cache/matplotlib', XDG_CACHE_HOME='/workspace/.cache',
                   XLA_FLAGS=f'--xla_force_host_platform_device_count={len(cpus)} --xla_cpu_multi_thread_eigen=false')
        command = ['taskset', '-c', ','.join(map(str, cpus)), str(REPO / '.venv/bin/python'),
                   '-m', 'local_agent.cli.main', 'run', str(session), '--no-sloppiness']
        started = time.monotonic()
        current.update(status='running', started_at=now(), cpu_affinity=cpus, command=command)
        save(status_path, current)
        print(f"{now()} START {entry['case']} CPUs={cpus}", flush=True)
        try:
            with (session.parent / 'fit.log').open('w') as log:
                process = subprocess.Popen(command, cwd=REPO, env=env, stdout=log, stderr=subprocess.STDOUT,
                                           start_new_session=True)
                current['pid'] = process.pid
                save(status_path, current)
                code = process.wait()
            summaries = sorted(session.glob('outputs/run_*/fit_summary.json'))
            summary = json.loads(summaries[-1].read_text()) if summaries else None
            current.update(returncode=code, fit_summary=summary,
                           status='completed' if code == 0 and summary else 'failed')
            if summary:
                current['run_dir'] = str(summaries[-1].parent)
                current['assessment'] = ('refinement_degraded' if summary.get('refinement_error') or
                                        summary.get('termination') == 'invalid_loss_or_gradient'
                                        else 'budget_completed_not_a_convergence_claim')
        except Exception as exc:
            current.update(status='failed', error=f'{type(exc).__name__}: {exc}')
        current.update(finished_at=now(), elapsed_seconds=time.monotonic() - started)
        save(status_path, current)
        print(f"{now()} {current['status'].upper()} {entry['case']} {current['elapsed_seconds']:.1f}s", flush=True)


def run():
    with (ROOT / 'runner.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        plan = json.loads((ROOT / 'plan.json').read_text())
        if not set(sum(plan['cpu_slots'], [])).issubset(os.sched_getaffinity(0)):
            raise RuntimeError('CPU allocation changed; revise the plan before running')
        (ROOT / 'runner.pid').write_text(str(os.getpid()) + '\n')
        entries = sorted((e for e in plan['cases'] if e['status'] == 'queued'), key=lambda e: e['parameters'])
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            futures = [executor.submit(worker, entries[i::2], cpus) for i, cpus in enumerate(plan['cpu_slots'])]
            for future in futures:
                future.result()
        print(f'{now()} Queue complete', flush=True)


if __name__ == '__main__':
    {'prepare': prepare, 'run': run}[sys.argv[1]]()
