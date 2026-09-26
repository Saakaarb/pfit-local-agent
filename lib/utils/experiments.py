"""Shared experiment loading and constants, adapted from deployed pfit-claude."""
from pathlib import Path
import numpy as np
from lib.utils.dataset_io import load_dataset, read_header


def load_experiments(session_dir, reader):
    records = []
    for index, experiment in enumerate(reader.experiments):
        path = Path(session_dir) / reader.user_input_dirname / experiment["filename"]
        context = f"Experiment {index + 1} ({experiment['filename']})"
        try:
            if not path.is_file():
                raise ValueError(f"Dataset file not found: {path}")
            data = load_dataset(path)
            header = read_header(path)
            if len(reader.experiments) > 1 and header is not None:
                declared = [column["name"] for column in experiment["columns"]]
                if header != declared:
                    raise ValueError(f"CSV header {header} does not match declared ordered columns {declared}")
            if data.ndim != 2 or data.shape[1] < 2:
                raise ValueError("Dataset must include a time column and at least one data column")
            if data.shape[0] < 2:
                raise ValueError("Dataset must contain at least two time points")
            if np.any(np.isinf(data)):
                raise ValueError("Dataset contains infinite values")
            times = data[:, 0]
            if not np.all(np.isfinite(times)) or not np.all(np.diff(times) > 0):
                raise ValueError("Dataset time values must be finite and strictly increasing")
            if reader.init_time is not None and reader.init_time > times[0]:
                raise ValueError("initial_time must not be after the first dataset time")
            if len(experiment["columns"]) != data.shape[1]:
                raise ValueError("Declared columns must match the dataset column count")
        except (ValueError, OSError) as exc:
            raise ValueError(f"{context}: Could not read valid numeric dataset: {exc}") from exc
        records.append({"index": index + 1, "filename": experiment["filename"], "path": path,
                        "columns": experiment["columns"], "t_eval": times,
                        "dataset": data[:, 1:], "y0": np.asarray(reader.get_y0(index))})
    return records


def experiment_constants(record, reader):
    times = record["t_eval"]
    return {
        "dataset": np.asarray(record["dataset"]), "t_eval": np.asarray(times),
        "init_cond": np.asarray(record["y0"]), "num_steps": len(times),
        "init_time": reader.init_time if reader.init_time is not None else times[0],
        "final_time": times[-1], "stepsize_rtol": np.asarray(reader.stepsize_rtol),
        "stepsize_atol": np.asarray(reader.stepsize_atol), "init_timestep": reader.init_timestep,
        "max_steps": reader.max_steps, "error_loss": reader.error_loss,
        "fixed_parameters": dict(zip(reader.fixed_parameter_names, reader.fixed_parameter_values)),
    }


def mean_experiment_loss(compute_loss, constants_list, parameters):
    # Lazy import keeps input validation and readiness independent of JAX.
    import jax.numpy as jnp
    losses = jnp.stack([compute_loss(c, parameters) for c in constants_list])
    if losses.ndim != 1:
        raise ValueError("Each experiment loss must be scalar")
    sentinels = jnp.asarray([c.get("error_loss", 1e10) for c in constants_list])
    valid = jnp.isfinite(losses) & (losses != sentinels)
    safe_losses = jnp.where(valid, losses, 0.0)
    return jnp.where(jnp.all(valid), jnp.mean(safe_losses), constants_list[0].get("error_loss", 1e10))
