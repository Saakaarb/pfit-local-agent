"""Deterministic, nested parameter coverage for numerical readiness checks."""
import json
import math
from pathlib import Path

import numpy as np

DEFAULT_SAMPLES = 32
DEFAULT_MAX_SAMPLES = 128
DEFAULT_SUCCESS_FRACTION = 0.25
DEFAULT_PATIENCE = 2


def parameter_samples(count, dimensions, seed=7):
    """Midpoint plus seeded Latin-hypercube blocks, stable when count grows.

    The complete growing set is not itself one Latin hypercube. Each added
    block spans every axis; earlier parameter vectors are retained exactly.
    """
    rng = np.random.default_rng(seed)
    blocks = [np.zeros((1, dimensions))]
    total, size = 1, 31
    while total < count:
        block = np.empty((size, dimensions))
        for axis in range(dimensions):
            block[:, axis] = 2 * (rng.permutation(size) + rng.random(size)) / size - 1
        blocks.append(block)
        total += size
        size = 32 if size == 31 else size * 2
    return np.concatenate(blocks)[:count]


def assess_solver_coverage(module, reader, session, script, records, settings):
    from diffrax import RESULTS
    from lib.utils.experiments import experiment_constants
    from lib.utils.run_artifacts import parameter_axes, unscale_parameters
    from local_agent.agent.validators import SolverValidationError, ValidationError

    count = settings.get('solver_validation_samples', DEFAULT_SAMPLES)
    target = float(settings.get('solver_validation_success_fraction', DEFAULT_SUCCESS_FRACTION))
    seed = settings.get('solver_validation_seed', 7)
    points = parameter_samples(count, reader.n_search_axes, seed)
    lo, hi, logs = parameter_axes(reader)
    diagnostics, candidates, successful = [], [], []
    for index, point in enumerate(points):
        complete = True
        failures = []
        for record in records:
            constants = experiment_constants(record, reader)
            constants.update(min_limits=lo, max_limits=hi, is_logscale=logs)
            item = dict(sample=index, experiment=record['index'], filename=record['filename'],
                        solver=reader.integrator, max_steps=reader.max_steps,
                        normalized_parameters=point.tolist(),
                        physical_parameters=dict(zip(reader.trainable_parameter_names,
                                                     unscale_parameters(point, reader).tolist())))
            try:
                ts, ys, result, stats = module._integrate_system_with_stats(constants, point)
                ys = np.asarray(ys)
                finite = np.all(np.isfinite(ys))
                code = ('successful' if result == RESULTS.successful and finite else
                        'step_limit' if result == RESULTS.max_steps_reached else
                        'nonfinite_solution' if result == RESULTS.successful else 'integration_failure')
                item.update(code=code, result=str(result), stats={k: int(v) for k, v in stats.items()},
                            time_start=float(constants['init_time']), time_end=float(record['t_eval'][-1]),
                            requested_rows=len(record['t_eval']),
                            finite_rows=int(np.all(np.isfinite(ys), axis=1).sum()))
                if code == 'successful':
                    loss = np.asarray(module._compute_loss_problem(constants, point))
                    if loss.shape != ():
                        raise ValidationError('_compute_loss_problem must return a scalar')
                    if not np.isfinite(loss) or loss == reader.error_loss:
                        item.update(code='invalid_loss', result='Integration completed but loss was nonfinite or the error-loss sentinel')
            except ValidationError:
                raise
            except (RuntimeError, FloatingPointError) as exc:
                item.update(code='integration_failure', result=f'{type(exc).__name__}: {exc}')
            if item['code'] != 'successful':
                complete = False
                failures.append(item['code'])
            diagnostics.append(item)
        candidates.append(dict(sample=index, successful=complete, failures=failures))
        if complete:
            successful.append(point)
    required = math.ceil(count * target)
    failure_counts = {}
    for item in diagnostics:
        if item['code'] != 'successful':
            failure_counts[item['code']] = failure_counts.get(item['code'], 0) + 1
    summary = dict(code='successful' if len(successful) >= required else 'coverage_below_target',
                   sample_count=count, seed=seed,
                   sampling='midpoint plus nested Latin-hypercube blocks in normalized linear/log coordinates',
                   successful_samples=len(successful), success_fraction=len(successful)/count,
                   required_success_fraction=target, required_successful_samples=required,
                   max_steps=reader.max_steps, failure_counts=failure_counts, candidates=candidates,
                   result=f'{len(successful)}/{count} parameter samples complete all experiments with valid loss; require {required}/{count}.')
    folder = Path(script).parent
    for filename, additions in [('solver_diagnostics.json', diagnostics), ('solver_coverage.json', [summary])]:
        path = folder / filename
        history = json.loads(path.read_text()) if path.exists() else []
        history.extend(additions)
        temp = path.with_suffix('.tmp')
        temp.write_text(json.dumps(history, indent=2, allow_nan=False)+'\n')
        temp.replace(path)
    if len(successful) < required:
        raise SolverValidationError(summary)
    # Source/JAX fidelity and writeout contracts must use feasible points, not
    # require a midpoint that can lie outside the model's feasible region.
    return successful[:2]
