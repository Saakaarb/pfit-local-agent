"""One optional DE tolerance comparison after refinement readiness succeeds."""
import json
from pathlib import Path

import numpy as np
import yaml

LOSS_RTOL = 0.01
LOSS_ATOL = 1e-8
RELAXATION = 10.


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


def calibrate_tolerances(module, reader, script, records, settings):
    from diffrax import RESULTS
    from lib.utils.experiments import experiment_constants
    from lib.utils.run_artifacts import parameter_axes
    from lib.utils.source_stamp import build_stamp

    folder = Path(script).parent
    report = dict(status='retained', max_steps=reader.max_steps,
                  source_stamp=build_stamp(folder.parent), relaxation_factor=RELAXATION,
                  loss_rtol=LOSS_RTOL, loss_atol=LOSS_ATOL, comparisons=[])
    def finish(reason):
        report['reason'] = reason
        path = folder/'tolerance_calibration.json'
        temp = path.with_suffix('.tmp')
        temp.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
        temp.replace(path)
        return report

    if any(settings.get(key) is not None for key in ('stepsize_rtol', 'stepsize_atol')):
        return finish('explicit population tolerances preserved')
    if not settings.get('auto_tolerances', True):
        return finish('automatic tolerance selection disabled')
    coverage = json.loads((folder/'solver_coverage.json').read_text())[-1]
    candidates = [c for c in coverage['candidates'] if c['successful']]
    if len(candidates) < 2:
        return finish('fewer than two feasible candidates; retain refinement tolerances')
    rtol = np.asarray(reader.stepsize_rtol)*RELAXATION
    atol = np.asarray(reader.stepsize_atol)*RELAXATION
    lo, hi, logs = parameter_axes(reader)
    reference, proposed = [], []
    for candidate in candidates:
        point = np.asarray(candidate['normalized_parameters'])
        losses = []
        for record, strict_loss in zip(records, candidate['experiment_losses']):
            constants = experiment_constants(record, reader)
            constants.update(min_limits=lo, max_limits=hi, is_logscale=logs,
                             stepsize_rtol=rtol, stepsize_atol=atol)
            item = dict(sample=candidate['sample'], experiment=record['index'],
                        reference_loss=strict_loss)
            report['comparisons'].append(item)
            try:
                _, ys, result, stats = module._integrate_system_with_stats(constants, point)
                item.update(result=str(result), stats={k:int(v) for k,v in stats.items()})
                if result != RESULTS.successful or not np.all(np.isfinite(ys)):
                    return finish('looser solve failed at the fixed step ceiling')
                loss = np.asarray(module._compute_loss_problem(constants, point))
                if loss.shape != () or not np.isfinite(loss) or loss == reader.error_loss:
                    return finish('looser solve returned invalid loss')
                item['proposed_loss'] = float(loss)
                losses.append(float(loss))
            except (RuntimeError, FloatingPointError) as exc:
                item['error'] = str(exc)
                return finish('looser numerical evaluation failed')
        reference.append(candidate['experiment_losses'])
        proposed.append(losses)
    accepted, reason = stable_losses(reference, proposed)
    if accepted:
        report.update(status='accepted', stepsize_rtol=rtol.tolist(), stepsize_atol=atol.tolist())
    return finish(reason)


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
    population = raw.setdefault('population_opt', {})
    if not population.get('auto_tolerances', True) or any(population.get(k) is not None for k in ('stepsize_rtol', 'stepsize_atol')):
        return False
    for key in ('stepsize_rtol', 'stepsize_atol'):
        population[key] = report[key]
    path.write_text(yaml.safe_dump(raw, sort_keys=False))
    return True
