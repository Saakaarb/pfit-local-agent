"""Dataset-only time-scale heuristic; not a stiffness diagnosis of the ODE."""
import json
from pathlib import Path

import numpy as np
import yaml

from lib.utils.experiments import load_experiments
from lib.utils.yamlread import YAMLReader


TIME_SCALE_RATIO = 100.0
MIN_POINTS = 8


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


def select_session_integrator(session_dir):
    session_dir = Path(session_dir)
    path = session_dir / 'inputs/user_input.yaml'
    config = yaml.safe_load(path.read_text())
    gradient = config.get('gradient_opt') or {}
    if not gradient.get('auto_integrator', False):
        return None
    reader = YAMLReader.from_file(path)
    report = dataset_solver_selection(load_experiments(session_dir, reader),
        set(reader.integrated_variable_names) | set(reader.observable_names))
    gradient['integrator'] = report['integrator']
    config['gradient_opt'] = gradient
    path.write_text(yaml.safe_dump(config, sort_keys=False))
    (session_dir / 'generated/solver_selection.json').write_text(
        json.dumps(report, indent=2, allow_nan=False) + '\n')
    return report
