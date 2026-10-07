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
    return dict(source_stamp=build_stamp(session), data_hashes={
        experiment['filename']: hashlib.sha256((session / reader.user_input_dirname / experiment['filename']).read_bytes()).hexdigest()
        for experiment in reader.experiments})


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


def refinement_start(loss_function, point, validated_seeds, error_loss):
    """Keep a valid search winner, otherwise use the best finite validated seed."""
    loss = float(loss_function(point))
    if np.isfinite(loss) and loss != error_loss:
        return point, loss, False
    feasible = []
    for seed in validated_seeds:
        seed = np.asarray(seed, dtype=float)
        value = float(loss_function(seed))
        if np.isfinite(value) and value != error_loss:
            feasible.append((value, seed))
    if not feasible:
        raise ValueError('Starting point has invalid loss or fails integration at refinement tolerances')
    loss, point = min(feasible, key=lambda item: item[0])
    return point, loss, True
