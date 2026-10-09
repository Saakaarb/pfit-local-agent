"""Dataset-only time-scale heuristic; not a stiffness diagnosis of the ODE."""
import json
import math
from pathlib import Path

import numpy as np
import yaml

from lib.utils.experiments import load_experiments
from lib.utils.yamlread import YAMLReader


TIME_SCALE_RATIO = 100.0
MIN_POINTS = 8
STEPS_PER_TIME_SCALE = 10
STEP_HEADROOM_FACTOR = 10


def dataset_solver_selection(records, measured_names):
    """Estimate amplitude/rate time scales using only mapped measured columns."""
    columns = []
    inconclusive = False
    for record in records:
        for j, column in enumerate(record['columns'][1:]):
            name = column.get('observes') or column['name']
            if column.get('uncertainty_of') or column.get('role') in ('forcing', 'uncertainty', 'auxiliary') or name not in measured_names:
                continue
            y = np.asarray(record['dataset'][:, j], dtype=float)
            t = np.asarray(record['t_eval'], dtype=float)
            mask = np.isfinite(y)
            t, y = t[mask], y[mask]
            item = dict(experiment=record['index'], column=column['name'], finite_points=len(t))
            columns.append(item)
            if len(t) < MIN_POINTS:
                item['status'] = 'insufficient_points'
                inconclusive = True
                continue
            # A five-point median suppresses isolated spikes without a fitted model.
            y = np.median(np.lib.stride_tricks.sliding_window_view(y, 5), axis=1)
            t = t[2:-2]
            amplitude = float(np.ptp(y))
            floor = np.finfo(float).eps * max(float(np.max(np.abs(y))), np.finfo(float).tiny) * 32
            if amplitude <= floor:
                item['status'] = 'flat'
                continue
            rates = np.abs(np.diff(y) / np.diff(t))
            rates = rates[rates > floor / (t[-1] - t[0])]
            if len(rates) < 3:
                item['status'] = 'insufficient_changes'
                inconclusive = True
                continue
            slow_rate, fast_rate = np.percentile(rates, [10, 90])
            fast, slow = amplitude / fast_rate, amplitude / slow_rate
            resolved = fast >= 2 * np.median(np.diff(t))
            item.update(status='resolved' if resolved else 'underresolved', amplitude=amplitude,
                        fast_time_scale=float(fast), slow_time_scale=float(slow))
            inconclusive |= not resolved
    usable = [c for c in columns if 'fast_time_scale' in c]
    ratio = (max(c['slow_time_scale'] for c in usable) /
             min(c['fast_time_scale'] for c in usable)) if usable else None
    if ratio is not None and ratio >= TIME_SCALE_RATIO:
        solver, reason = 'Kvaerno5', 'Observed time-scale separation is at least 100-fold'
    elif inconclusive or not usable:
        solver, reason = 'Kvaerno5', 'Dataset time scales are inconclusive; conservative fallback'
    else:
        solver, reason = 'Tsit5', 'Resolved observed time scales have less than 100-fold separation'
    return dict(method='dataset_time_scales', integrator=solver, reason=reason,
                time_scale_ratio=ratio, ratio_threshold=TIME_SCALE_RATIO, columns=columns,
                scope='Observed dynamics only; hidden modes and other parameter values may be stiff')


def estimate_max_steps(records, columns, current, cap, initial_time=None):
    """Seed a common ceiling from observed time scales; validation still owns acceptance."""
    experiments = []
    for record in records:
        start = record['t_eval'][0] if initial_time is None else initial_time
        duration = float(record['t_eval'][-1] - start)
        measured = [c for c in columns if c['experiment'] == record['index']]
        resolved = [c['fast_time_scale'] for c in measured if c['status'] == 'resolved']
        uncertain = any(c['status'] not in ('resolved', 'flat') for c in measured)
        tau = min(resolved) if resolved else None
        usable = tau is not None and not uncertain
        raw = STEPS_PER_TIME_SCALE * STEP_HEADROOM_FACTOR * duration / tau if usable else None
        # Bound before rounding to avoid overflow and preserve an explicit cap.
        proposed = min(cap, max(1000, 1000 * math.ceil(min(raw, cap) / 1000))) if usable else current
        experiments.append(dict(experiment=record['index'], duration=duration,
            fastest_time_scale=tau, raw_estimated_steps=raw if raw is None or math.isfinite(raw) else None,
            proposed_max_steps=proposed, capped=bool(usable and raw > cap),
            method='time_scale_estimate' if usable else 'configured_budget_fallback'))
    selected = max((e['proposed_max_steps'] for e in experiments), default=current)
    return dict(selected_max_steps=selected, configured_max_steps=current, cap=cap,
        steps_per_time_scale=STEPS_PER_TIME_SCALE, headroom_factor=STEP_HEADROOM_FACTOR, rounding_multiple=1000,
        minimum_steps=min(1000, cap), experiments=experiments,
        scope='Initial budget estimate only; sampled integration and accuracy checks remain required')


def select_session_integrator(session_dir):
    session_dir = Path(session_dir)
    path = session_dir / 'inputs/user_input.yaml'
    config = yaml.safe_load(path.read_text())
    gradient = config.get('gradient_opt') or {}
    if not (gradient.get('auto_integrator', False) or gradient.get('auto_max_steps', False)):
        return None
    reader = YAMLReader.from_file(path)
    records = load_experiments(session_dir, reader)
    report = dataset_solver_selection(records,
        set(reader.integrated_variable_names) | set(reader.observable_names))
    if gradient.get('auto_integrator', False):
        gradient['integrator'] = report['integrator']
    else:
        report.update(integrator=reader.integrator, reason='Explicit solver preserved; automatic selection disabled')
    if gradient.get('auto_max_steps', False):
        report['max_steps_estimate'] = estimate_max_steps(records, report['columns'],
            gradient.get('max_steps', 10000),
            gradient.get('solver_recovery_max_steps', gradient.get('max_steps', 10000)), reader.init_time)
        gradient['max_steps'] = report['max_steps_estimate']['selected_max_steps']
    config['gradient_opt'] = gradient
    path.write_text(yaml.safe_dump(config, sort_keys=False))
    (session_dir / 'generated/solver_selection.json').write_text(
        json.dumps(report, indent=2, allow_nan=False) + '\n')
    return report
