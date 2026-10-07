"""Bounded state-scale and DE tolerance checks after solver readiness succeeds."""
import json
from pathlib import Path

import numpy as np
import yaml

LOSS_RTOL = 0.01
LOSS_ATOL = 1e-8
RELAXATION = 10.
STATE_SCALE_FRACTION = 0.01


def stable_losses(reference, proposed):
    """Preserve experiment losses and meaningful ordering of mean losses."""
    reference, proposed = np.asarray(reference), np.asarray(proposed)
    if not np.all(np.isfinite(proposed)):
        return False, 'nonfinite loss'
    if np.any(np.abs(proposed-reference) > LOSS_ATOL + LOSS_RTOL*np.abs(reference)):
        return False, 'experiment loss changed beyond 1% plus 1e-8'
    strict, loose = reference.mean(axis=1), proposed.mean(axis=1)
    for i in range(len(strict)):
        for j in range(i):
            gap = strict[i]-strict[j]
            tie = LOSS_ATOL + LOSS_RTOL*max(abs(strict[i]), abs(strict[j]))
            if abs(gap) > tie and gap*(loose[i]-loose[j]) <= 0:
                return False, 'meaningful candidate ordering changed'
    return True, 'paired losses and candidate ordering stable'


def state_scaled_atol(candidates, rtol, atol):
    """Median over successful candidates, then largest experiment scale per state.

    Each trajectory contributes one scale vector, irrespective of dataset length.
    Missing scale records from older reports cannot support automatic selection.
    """
    if any(not c.get('experiment_state_scales') for c in candidates):
        return None
    values = np.asarray([c['experiment_state_scales'] for c in candidates], dtype=float)
    scales = np.median(values, axis=0).max(axis=0)
    rtol = np.broadcast_to(np.asarray(rtol, dtype=float), scales.shape)
    original = np.broadcast_to(np.asarray(atol, dtype=float), scales.shape)
    with np.errstate(over='ignore', under='ignore', invalid='ignore'):
        proposal = rtol * STATE_SCALE_FRACTION * scales
    usable = np.isfinite(proposal) & (proposal > 0)
    return dict(state_scales=scales.tolist(), proposed_atol=np.where(usable, proposal, original).tolist(),
                fallback_state_indices=np.flatnonzero(~usable).tolist())


def calibrate_tolerances(module, reader, script, records, settings, gradient_settings=None):
    from diffrax import RESULTS
    from lib.utils.experiments import experiment_constants
    from lib.utils.run_artifacts import parameter_axes
    from lib.utils.source_stamp import build_stamp

    gradient_settings = gradient_settings or {}
    folder = Path(script).parent
    report = dict(status='retained', max_steps=reader.max_steps,
                  source_stamp=build_stamp(folder.parent), relaxation_factor=RELAXATION,
                  loss_rtol=LOSS_RTOL, loss_atol=LOSS_ATOL, comparisons=[],
                  state_scaling=dict(status='retained', fraction=STATE_SCALE_FRACTION,
                      state_names=getattr(reader, 'integrated_variable_names', None),
                      method='per-trajectory max(p95(abs(saved states)), abs(initial state)); median across candidates, maximum across experiments',
                      reference_rtol=np.asarray(reader.stepsize_rtol).tolist(),
                      reference_atol=np.asarray(reader.stepsize_atol).tolist()))
    scaling = report['state_scaling']

    def finish(reason):
        report['reason'] = reason
        path = folder/'tolerance_calibration.json'
        temp = path.with_suffix('.tmp')
        temp.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
        temp.replace(path)
        return report

    coverage = json.loads((folder/'solver_coverage.json').read_text())[-1]
    candidates = [c for c in coverage['candidates'] if c['successful']]
    if len(candidates) < 2:
        scaling['reason'] = 'fewer than two feasible candidates'
        return finish('fewer than two feasible candidates; retain configured tolerances')
    reference = [c['experiment_losses'] for c in candidates]
    lo, hi, logs = parameter_axes(reader)

    def compare(rtol, atol):
        result_report = dict(status='retained', comparisons=[])
        proposed = []
        for candidate in candidates:
            point = np.asarray(candidate['normalized_parameters'])
            losses = []
            for record, strict_loss in zip(records, candidate['experiment_losses']):
                constants = experiment_constants(record, reader)
                constants.update(min_limits=lo, max_limits=hi, is_logscale=logs,
                                 stepsize_rtol=np.asarray(rtol), stepsize_atol=np.asarray(atol))
                item = dict(sample=candidate['sample'], experiment=record['index'],
                            reference_loss=strict_loss)
                result_report['comparisons'].append(item)
                try:
                    _, ys, result, stats = module._integrate_system_with_stats(constants, point)
                    item.update(result=str(result), stats={k:int(v) for k,v in stats.items()})
                    if result != RESULTS.successful or not np.all(np.isfinite(ys)):
                        result_report['reason'] = 'proposed solve failed at the fixed step ceiling'
                        return result_report
                    loss = np.asarray(module._compute_loss_problem(constants, point))
                    if loss.shape != () or not np.isfinite(loss) or loss == reader.error_loss:
                        result_report['reason'] = 'proposed solve returned invalid loss'
                        return result_report
                    item['proposed_loss'] = float(loss)
                    losses.append(float(loss))
                except (RuntimeError, FloatingPointError) as exc:
                    item['error'] = str(exc)
                    result_report['reason'] = 'proposed numerical evaluation failed'
                    return result_report
            proposed.append(losses)
        accepted, reason = stable_losses(reference, proposed)
        result_report.update(status='accepted' if accepted else 'retained', reason=reason)
        return result_report

    rtol, atol = np.asarray(reader.stepsize_rtol), np.asarray(reader.stepsize_atol)
    if gradient_settings.get('auto_state_tolerances', True):
        proposal = state_scaled_atol(candidates, rtol, atol)
        if proposal is None:
            scaling['reason'] = 'trajectory scales unavailable'
        else:
            scaling.update(proposal)
            scaling.update(compare(rtol, proposal['proposed_atol']))
            if scaling['status'] == 'accepted':
                atol = np.asarray(proposal['proposed_atol'])
                report.update(status='accepted', gradient_stepsize_atol=atol.tolist())
    else:
        scaling['reason'] = 'automatic state scaling disabled'

    if any(settings.get(key) is not None for key in ('stepsize_rtol', 'stepsize_atol')):
        return finish('explicit population tolerances preserved')
    if not settings.get('auto_tolerances', True):
        return finish('automatic DE tolerance selection disabled')
    de_rtol, de_atol = rtol*RELAXATION, atol*RELAXATION
    de = compare(de_rtol, de_atol)
    report['comparisons'] = de['comparisons']
    report['de_status'] = de['status']
    if de['status'] == 'accepted':
        report.update(status='accepted', stepsize_rtol=de_rtol.tolist(), stepsize_atol=de_atol.tolist())
    return finish(de['reason'])


def apply_calibrated_tolerances(session, script):
    """Parent-only update, after source freshness and every strict gate passed."""
    from lib.utils.source_stamp import build_stamp
    path = Path(script).parent/'tolerance_calibration.json'
    if not path.exists():
        return False
    report = json.loads(path.read_text())
    if report['status'] != 'accepted' or report['source_stamp'] != build_stamp(session):
        return False
    path = Path(session)/'inputs/user_input.yaml'
    raw = yaml.safe_load(path.read_text())
    population, gradient = raw.setdefault('population_opt', {}), raw.setdefault('gradient_opt', {})
    changed = False
    if 'gradient_stepsize_atol' in report and gradient.get('auto_state_tolerances', True):
        gradient['stepsize_atol'] = report['gradient_stepsize_atol']
        changed = True
    if ('stepsize_rtol' in report and population.get('auto_tolerances', True)
            and not any(population.get(k) is not None for k in ('stepsize_rtol', 'stepsize_atol'))):
        for key in ('stepsize_rtol', 'stepsize_atol'):
            population[key] = report[key]
        changed = True
    if changed:
        path.write_text(yaml.safe_dump(raw, sort_keys=False))
    return changed
