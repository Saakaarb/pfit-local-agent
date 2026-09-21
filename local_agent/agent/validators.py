from dataclasses import dataclass
from pathlib import Path

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


def parse_input_yaml(input_yaml: Path) -> YAMLReader:
    input_yaml = Path(input_yaml)
    if not input_yaml.exists():
        raise ValidationError(f"Input YAML not found: {input_yaml}")

    try:
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

    dataset_path = session_dir / reader.user_input_dirname / reader.filename_data
    if not dataset_path.exists():
        raise ValidationError(f"Dataset file not found: {dataset_path}")

    try:
        with dataset_path.open("r", encoding="utf-8-sig") as handle:
            dataset = np.genfromtxt(handle, dtype=float, delimiter=",")
    except Exception as exc:
        raise ValidationError(f"Could not read numeric dataset {dataset_path}: {exc}") from exc

    if dataset.size == 0:
        raise ValidationError(f"Dataset is empty: {dataset_path}")
    if dataset.ndim != 2:
        raise ValidationError(
            f"Dataset must be a 2D numeric table with time plus data columns: {dataset_path}"
        )
    if dataset.shape[1] < 2:
        raise ValidationError(
            f"Dataset must include a time column and at least one data column: {dataset_path}"
        )
    if np.any(np.isinf(dataset)):
        raise ValidationError(f"Dataset contains infinite values: {dataset_path}")

    n_integrated_variables = len(reader.integrated_variable_names)

    return SessionValidation(
        session_dir=session_dir,
        input_yaml=input_yaml,
        dataset_path=dataset_path,
        dataset_shape=dataset.shape,
        n_trainable_parameters=reader.n_search_axes,
        n_integrated_variables=n_integrated_variables,
    )


def smoke_test_generated_script(script_path: Path, session_dir: Path) -> None:
    try:
        module = import_generated_script(script_path)
    except GeneratedContractError as exc:
        raise ValidationError(str(exc)) from exc
    reader = parse_input_yaml(Path(session_dir) / "inputs" / "user_input.yaml")
    dataset_path = Path(session_dir) / reader.user_input_dirname / reader.filename_data
    try:
        with dataset_path.open("r", encoding="utf-8-sig") as handle:
            all_data = np.genfromtxt(handle, dtype=float, delimiter=",")
    except Exception as exc:
        raise ValidationError(f"Could not load dataset for smoke test: {exc}") from exc

    if all_data.ndim != 2 or all_data.shape[1] < 2:
        raise ValidationError("Smoke-test dataset must contain time plus data columns")

    t_eval = all_data[:, 0]
    dataset = all_data[:, 1:]
    min_limits = np.array(
        [
            np.log10(value) if reader.axis_logscale[index] else value
            for index, value in enumerate(reader.min_axis_values)
        ]
    )
    max_limits = np.array(
        [
            np.log10(value) if reader.axis_logscale[index] else value
            for index, value in enumerate(reader.max_axis_values)
        ]
    )
    constants = {
        "dataset": dataset,
        "t_eval": t_eval,
        "init_cond": np.array(reader.integrated_variable_init_values),
        "num_steps": dataset.shape[0],
        "init_time": reader.init_time if reader.init_time is not None else t_eval[0],
        "final_time": t_eval[-1],
        "stepsize_rtol": np.array(reader.stepsize_rtol),
        "stepsize_atol": np.array(reader.stepsize_atol),
        "init_timestep": reader.init_timestep,
        "max_steps": reader.max_steps,
        "fixed_parameters": dict(
            zip(reader.fixed_parameter_names, reader.fixed_parameter_values)
        ),
        "error_loss": reader.error_loss,
        "min_limits": min_limits,
        "max_limits": max_limits,
        "is_logscale": np.array(reader.axis_logscale),
    }
    trainable_variables = np.zeros(reader.n_search_axes)

    try:
        loss = module._compute_loss_problem(constants, trainable_variables)
    except Exception as exc:
        raise ValidationError(f"_compute_loss_problem smoke test failed: {exc}") from exc

    loss_array = np.asarray(loss)
    if loss_array.size != 1 or not np.all(np.isfinite(loss_array)):
        raise ValidationError(f"_compute_loss_problem returned non-finite loss: {loss}")

    try:
        writeout = module._write_problem_result(constants, trainable_variables)
    except Exception as exc:
        raise ValidationError(f"_write_problem_result smoke test failed: {exc}") from exc

    writeout_array = np.asarray(writeout)
    if writeout_array.ndim != 2:
        raise ValidationError("_write_problem_result must return a 2D array")
    if writeout_array.shape[0] != dataset.shape[0]:
        raise ValidationError(
            "_write_problem_result row count must match dataset rows: "
            f"{writeout_array.shape[0]} vs {dataset.shape[0]}"
        )


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


def _validate_reader(reader: YAMLReader, input_yaml: Path) -> None:
    if reader.filename_data is None:
        raise ValidationError(f"Missing data_file in {input_yaml}")
    if reader.n_search_axes is None:
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
