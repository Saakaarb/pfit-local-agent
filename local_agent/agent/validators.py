from dataclasses import dataclass
from pathlib import Path
import yaml
import keyword

import numpy as np

from local_agent.core.generated_contract import (
    GeneratedContractError,
    import_generated_script as core_import_generated_script,
    validate_generated_script_contract as core_validate_generated_script_contract,
)
from lib.utils.yamlread import YAMLReader


class ValidationError(ValueError):
    """Raised when a session or generated file violates the workflow contract."""


@dataclass(frozen=True)
class SessionValidation:
    session_dir: Path
    input_yaml: Path
    dataset_path: Path
    dataset_shape: tuple[int, int]
    n_trainable_parameters: int
    n_integrated_variables: int
    experiments: tuple[dict, ...] = ()


def parse_input_yaml(input_yaml: Path) -> YAMLReader:
    input_yaml = Path(input_yaml)
    if not input_yaml.exists():
        raise ValidationError(f"Input YAML not found: {input_yaml}")

    try:
        raw = yaml.safe_load(input_yaml.read_text())
        _validate_raw_settings(raw)
        reader = YAMLReader.from_file(input_yaml)
        reader.check_name_uniqueness()
    except Exception as exc:
        raise ValidationError(f"Invalid YAML contents in {input_yaml}: {exc}") from exc

    _validate_reader(reader, input_yaml)
    return reader


def validate_session(session_dir: Path) -> SessionValidation:
    session_dir = Path(session_dir)
    input_yaml = session_dir / "inputs" / "user_input.yaml"
    reader = parse_input_yaml(input_yaml)

    from lib.utils.experiments import load_experiments
    try:
        records = load_experiments(session_dir, reader)
    except ValueError as exc:
        raise ValidationError(str(exc)) from exc
    return SessionValidation(
        session_dir=session_dir, input_yaml=input_yaml,
        dataset_path=records[0]["path"],
        dataset_shape=(len(records[0]["t_eval"]), records[0]["dataset"].shape[1] + 1),
        n_trainable_parameters=reader.n_search_axes,
        n_integrated_variables=len(reader.integrated_variable_names),
        experiments=tuple(records),
    )


def smoke_test_generated_script(script_path: Path, session_dir: Path) -> None:
    try:
        module = import_generated_script(script_path)
    except GeneratedContractError as exc:
        raise ValidationError(str(exc)) from exc
    reader = parse_input_yaml(Path(session_dir) / "inputs" / "user_input.yaml")
    from lib.utils.experiments import load_experiments, experiment_constants
    from lib.utils.run_artifacts import parameter_axes
    lo, hi, logs = parameter_axes(reader)
    for record in load_experiments(session_dir, reader):
        constants = experiment_constants(record, reader)
        constants.update(min_limits=lo, max_limits=hi, is_logscale=logs)
        context = f"Experiment {record['index']} ({record['filename']})"
        try:
            loss = np.asarray(module._compute_loss_problem(constants, np.zeros(reader.n_search_axes)))
            if loss.shape != () or not np.isfinite(loss) or loss == reader.error_loss:
                raise ValidationError(f"_compute_loss_problem returned non-finite loss or integration failure: {loss}")
            writeout = np.asarray(module._write_problem_result(constants, np.zeros(reader.n_search_axes)))
            if writeout.ndim != 2:
                raise ValidationError("_write_problem_result must return a 2D array")
            if writeout.shape[0] != len(record["t_eval"]):
                raise ValidationError("_write_problem_result row count must match dataset rows")
        except Exception as exc:
            raise ValidationError(f"{context}: smoke test failed: {exc}") from exc


def validate_generated_script_contract(script_path: Path):
    try:
        return core_validate_generated_script_contract(script_path)
    except GeneratedContractError as exc:
        raise ValidationError(str(exc)) from exc


def import_generated_script(script_path: Path):
    try:
        return core_import_generated_script(script_path)
    except GeneratedContractError as exc:
        raise ValidationError(str(exc)) from exc


def _load_numeric_csv(path: Path) -> np.ndarray:
    from lib.utils.dataset_io import load_dataset
    try:
        return load_dataset(path)
    except Exception as exc:
        raise ValidationError(f"Could not read numeric dataset {path}: {exc}") from exc


def _validate_reader(reader: YAMLReader, input_yaml: Path) -> None:
    if reader.filename_data is None:
        raise ValidationError(f"Missing data_file in {input_yaml}")
    if not reader.n_search_axes:
        raise ValidationError(f"Missing trainable parameters in {input_yaml}")
    if reader.n_search_axes != len(reader.trainable_parameter_names):
        raise ValidationError(
            "Trainable parameter count does not match parameter descriptions: "
            f"{reader.n_search_axes} vs {len(reader.trainable_parameter_names)}"
        )
    if len(reader.min_axis_values) != reader.n_search_axes:
        raise ValidationError("Every trainable parameter must define MIN_VAL")
    if len(reader.max_axis_values) != reader.n_search_axes:
        raise ValidationError("Every trainable parameter must define MAX_VAL")
    if len(reader.axis_logscale) != reader.n_search_axes:
        raise ValidationError("Every trainable parameter must define LOGSCALE")
    if not reader.integrated_variable_names:
        raise ValidationError(f"No integrated variables declared in {input_yaml}")
    if len(reader.integrated_variable_names) != len(reader.integrated_variable_init_values):
        raise ValidationError("Every integrated variable must define INIT_VAL")
    if reader.n_particles is None:
        raise ValidationError(f"Missing population_size in {input_yaml}")
    if reader.n_iters_pop is None:
        raise ValidationError(f"Missing population_opt num_iters in {input_yaml}")
    if reader.processors is None:
        raise ValidationError(f"Missing processors in {input_yaml}")
    if reader.algorithm not in {"PSO", "DE"}:
        raise ValidationError(
            f"Unsupported population_opt algorithm in {input_yaml}: {reader.algorithm}"
        )
    if reader.n_iters_grad is None:
        raise ValidationError(f"Missing gradient_opt num_iters in {input_yaml}")
    if reader.stepsize_rtol is None:
        raise ValidationError(f"Missing stepsize_rtol in {input_yaml}")
    if reader.stepsize_atol is None:
        raise ValidationError(f"Missing stepsize_atol in {input_yaml}")
    if reader.init_timestep is None:
        raise ValidationError(f"Missing initial_timestep in {input_yaml}")
    if reader.max_steps is None:
        raise ValidationError(f"Missing max_steps in {input_yaml}")
    if reader.integrator not in {"Tsit5", "Dopri5", "Dopri8", "Kvaerno5"}:
        raise ValidationError(
            f"Unsupported gradient_opt integrator in {input_yaml}: {reader.integrator}"
        )

    names = reader.trainable_parameter_names + reader.fixed_parameter_names + reader.integrated_variable_names + reader.observable_names
    if any(not name.isidentifier() or keyword.iskeyword(name) for name in names):
        raise ValidationError("Model names must be valid Python identifiers")
    for name, low, high, log in zip(reader.trainable_parameter_names, reader.min_axis_values, reader.max_axis_values, reader.axis_logscale):
        if not np.isfinite(low) or not np.isfinite(high) or low >= high:
            raise ValidationError(f"Parameter {name} requires finite bounds with min_val < max_val")
        if log and low <= 0:
            raise ValidationError(f"Logscale parameter {name} requires positive bounds")
    if not np.all(np.isfinite(reader.fixed_parameter_values + reader.integrated_variable_init_values)):
        raise ValidationError("Fixed parameters and initial conditions must be finite")
    for name in ("stepsize_rtol", "stepsize_atol", "population_stepsize_rtol", "population_stepsize_atol"):
        values = getattr(reader, name)
        if values is not None and (len(values) not in (1, len(reader.integrated_variable_names)) or not np.all(np.isfinite(values)) or np.any(np.asarray(values) <= 0)):
            raise ValidationError(f"{name} must contain positive finite tolerances, scalar or one per state")
    for name in ("init_timestep", "error_loss"):
        value = getattr(reader, name)
        if not np.isfinite(value) or value <= 0:
            raise ValidationError(f"{name} must be positive and finite")
    if reader.init_time is not None and not np.isfinite(reader.init_time):
        raise ValidationError("initial_time must be finite")
    if reader.n_particles < (3 if reader.algorithm == "DE" else 1):
        raise ValidationError("population_size must be at least 3 for DE or 1 for PSO")


def _validate_raw_settings(raw):
    if not isinstance(raw, dict):
        raise ValidationError("YAML input must be a mapping")
    for section, fields in (("population_opt", ("population_size", "num_particles", "num_iters", "processors", "random_seed")), ("gradient_opt", ("num_iters", "max_steps"))):
        for name in fields:
            value = (raw.get(section) or {}).get(name)
            minimum = 0 if name in {"num_iters", "random_seed"} else 1
            if value is not None and (type(value) is not int or value < minimum):
                raise ValidationError(f"{section}.{name} must be an integer >= {minimum}")
    for parameter in (raw.get("model") or {}).get("trainable_parameters") or []:
        if type(parameter.get("logscale", False)) is not bool:
            raise ValidationError("logscale must be a YAML boolean, not a quoted string")
