"""Named restart parameters and small, inspectable run artifacts."""
import json
from pathlib import Path

import numpy as np


def parameter_axes(reader):
    lower = np.asarray(reader.min_axis_values, dtype=float)
    upper = np.asarray(reader.max_axis_values, dtype=float)
    logs = np.asarray(reader.axis_logscale, dtype=bool)
    names = list(reader.trainable_parameter_names)
    if not names or len(set(names)) != len(names):
        raise ValueError("Trainable parameter names must be nonempty and unique")
    if lower.shape != (len(names),) or upper.shape != lower.shape or logs.shape != lower.shape:
        raise ValueError("Parameter bounds and logscale flags must match parameter names")
    if not np.all(np.isfinite(lower)) or not np.all(np.isfinite(upper)) or np.any(lower >= upper):
        raise ValueError("Parameter bounds must be finite and strictly increasing")
    if np.any(lower[logs] <= 0):
        raise ValueError("Log-scaled parameter bounds must be positive")
    lo, hi = lower.copy(), upper.copy()
    lo[logs], hi[logs] = np.log10(lower[logs]), np.log10(upper[logs])
    return lo, hi, logs


def scale_parameters(values, reader):
    lo, hi, logs = parameter_axes(reader)
    values = np.asarray(values, dtype=float)
    if values.shape != lo.shape or not np.all(np.isfinite(values)):
        raise ValueError("Seed must contain one finite value per trainable parameter")
    if np.any(values < reader.min_axis_values) or np.any(values > reader.max_axis_values):
        raise ValueError("Seed parameters are outside the current bounds")
    coordinates = values.copy()
    coordinates[logs] = np.log10(coordinates[logs])
    return 2 * (coordinates - lo) / (hi - lo) - 1


def unscale_parameters(position, reader):
    lo, hi, logs = parameter_axes(reader)
    values = lo + (np.asarray(position) + 1) * (hi - lo) / 2
    values[logs] = 10 ** values[logs]
    return values


def load_restart_seed(source_dir, reader, *, allow_legacy=False):
    source_dir = Path(source_dir)
    named_path = source_dir / "final_parameters.json"
    if named_path.exists():
        parameters = json.loads(named_path.read_text())["parameters"]
        names = [item["name"] for item in parameters]
        if len(set(names)) != len(names) or set(names) != set(reader.trainable_parameter_names):
            raise ValueError("Source run parameter names do not match the current model")
        by_name = {item["name"]: item["value"] for item in parameters}
        values = np.asarray([by_name[name] for name in reader.trainable_parameter_names], dtype=float)
    else:
        if not allow_legacy:
            raise ValueError(
                "Source run has no final_parameters.json. For an old unnamed CSV, "
                "use --allow-legacy-seed only if its parameter order matches the current YAML."
            )
        values = np.atleast_1d(np.loadtxt(source_dir / "final_design_point.csv", delimiter=","))
    scale_parameters(values, reader)
    return values


def write_final_parameters(output_dir, reader, values):
    payload = {"parameters": [
        {"name": name, "value": float(value), "logscale": bool(logscale)}
        for name, value, logscale in zip(reader.trainable_parameter_names, values, reader.axis_logscale)
    ]}
    (Path(output_dir) / "final_parameters.json").write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")


def accuracy_seed_context(session, reader):
    """Bind validated seeds to the exact model, configuration and experiment data."""
    import hashlib
    from lib.utils.source_stamp import build_stamp
    session = Path(session)
    context = dict(source_stamp=build_stamp(session), data_hashes={
        experiment['filename']: hashlib.sha256((session / reader.user_input_dirname / experiment['filename']).read_bytes()).hexdigest()
        for experiment in reader.experiments})
    runtime_config = session / 'inputs/run_config.yaml'
    if runtime_config.exists():
        context['runtime_config_hash'] = hashlib.sha256(runtime_config.read_bytes()).hexdigest()
    return context


def load_accuracy_seeds(session, reader):
    path = Path(session) / 'generated/solver_accuracy.json'
    if not path.exists():
        return []
    report = json.loads(path.read_text())
    if 'validated_seeds' not in report:  # legacy reports did not persist seeds
        return []
    if report.get('code') != 'accuracy_passed' or report.get('seed_context') != accuracy_seed_context(session, reader):
        raise ValueError('Accuracy-validated seeds are stale or unvalidated; rerun pfit jax')
    seeds = [seed['normalized_parameters'] for seed in report['validated_seeds']]
    points = np.asarray(seeds, dtype=float)
    if points.ndim != 2 or points.shape[1] != reader.n_search_axes or not len(points) or not np.all(np.isfinite(points)) or np.any(np.abs(points) > 1):
        raise ValueError('Invalid accuracy-validated parameter seeds; rerun pfit jax')
    return points.tolist()


def seed_population(population, seeds):
    """Inject validated normalized points while retaining an exploratory particle."""
    if not len(seeds):
        return population
    seeds = np.asarray(seeds, dtype=float)
    if seeds.ndim != 2 or seeds.shape[1] != population.shape[1] or not np.all(np.isfinite(seeds)) or np.any(np.abs(seeds) > 1):
        raise ValueError('Invalid normalized population seeds')
    count = min(len(seeds), max(1, len(population)-1))
    population[:count] = seeds[:count]
    return population


def refinement_start(make_problem, point, tolerances, population_tolerances, error_loss):
    """Preserve the search winner, retrying with population tolerances if needed.

    Build a fresh problem for each attempt: JIT closures may already have captured
    the first problem's constants, so mutating those constants is not sufficient.
    Explicit gradient-only restarts supply no population tolerances.
    """
    problem = make_problem(*tolerances)
    loss = float(problem._compute_loss(point))
    if np.isfinite(loss) and loss != error_loss:
        return problem, loss, False
    if population_tolerances is not None:
        problem = make_problem(*population_tolerances)
        loss = float(problem._compute_loss(point))
        if np.isfinite(loss) and loss != error_loss:
            return problem, loss, True
    raise ValueError('Starting point has invalid loss or fails integration at available refinement tolerances')


def snapshot_accuracy_seeds(session, reader, snapshot, runtime_reader):
    """Transfer validated seeds only after checking the immutable run copy."""
    if not load_accuracy_seeds(session, reader):
        return
    report = json.loads((Path(session) / 'generated/solver_accuracy.json').read_text())
    original = report['seed_context']
    copied = accuracy_seed_context(snapshot, runtime_reader)
    if copied['source_stamp'] != original['source_stamp']:
        raise ValueError('Model/config changed while snapshotting accuracy seeds')
    for source, target in zip(reader.experiments, runtime_reader.experiments):
        if original['data_hashes'][source['filename']] != copied['data_hashes'][target['filename']]:
            raise ValueError('Data changed while snapshotting accuracy seeds')
    report['source_seed_context'] = original
    report['seed_context'] = copied
    (Path(snapshot) / 'generated/solver_accuracy.json').write_text(json.dumps(report, indent=2) + '\n')
