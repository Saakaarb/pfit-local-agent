import ast
import copy
import csv
import json
import math
import re
import shutil
import tempfile
from dataclasses import dataclass, replace
from pathlib import Path

from local_agent.agent.checks import user_loss_contract
from local_agent.agent.config import WorkflowConfig
from local_agent.agent.llm_json import parse_llm_json_object
from local_agent.agent.prompts import PromptRenderer
from local_agent.agent.validators import ValidationError, validate_session
from local_agent.llm.base import LLMClient, LLMError, Message


@dataclass(frozen=True)
class NewSessionDraft:
    missing_inputs: tuple[str, ...]
    review: str
    user_input_yaml: str
    user_model_py: str
    user_info_txt: str


@dataclass(frozen=True)
class NewSessionParameter:
    name: str
    min_value: float
    max_value: float
    logscale: bool


@dataclass(frozen=True)
class NewSessionFixedParameter:
    name: str
    value: float


@dataclass(frozen=True)
class NewSessionState:
    name: str
    initial_value: float
    rhs: str
    observed_column: int | None


@dataclass(frozen=True)
class NewSessionObservable:
    name: str
    expression: str
    observed_column: int


@dataclass(frozen=True)
class NewSessionAuxiliaryColumn:
    name: str
    observed_column: int
    kind: str
    target: str


@dataclass(frozen=True)
class NewSessionFormula:
    name: str
    expression: str


@dataclass(frozen=True)
class NewSessionSpec:
    missing_inputs: tuple[str, ...]
    review: str
    filename_data: str
    parameters: tuple[NewSessionParameter, ...]
    fixed_parameters: tuple[NewSessionFixedParameter, ...]
    states: tuple[NewSessionState, ...]
    helper_functions: tuple[str, ...]
    observables: tuple[NewSessionObservable, ...]
    auxiliary_columns: tuple[NewSessionAuxiliaryColumn, ...]
    loss_body: str
    user_info_txt: str
    experiments: tuple[dict, ...] = ()


def init_session(
    session_dir: Path,
    llm_client: LLMClient,
    prompt_renderer: PromptRenderer,
    workflow_config: WorkflowConfig | None = None,
    *,
    overwrite: bool = False,
) -> list[Path]:
    session_dir = Path(session_dir)
    workflow_config = workflow_config or WorkflowConfig()
    generated = session_dir / "generated"
    generated.mkdir(parents=True, exist_ok=True)

    _validate_csv_inputs_have_headers(session_dir)
    context = {"session_context": _collect_session_context(session_dir)}
    if overwrite:
        _clear_previous_new_outputs(session_dir)

    response = _draft_new_session_response(
        session_dir,
        llm_client,
        prompt_renderer,
        workflow_config,
        generated,
        context,
    )
    for attempt in range(0, workflow_config.max_repair_attempts + 1):
        try:
            spec = _parse_new_session_response(response)
            if spec.missing_inputs:
                missing = "\n".join(f"- {item}" for item in spec.missing_inputs)
                raise ValidationError(f"pfit-new missing required inputs:\n{missing}")
            spec = _canonicalize_observed_columns_from_csv_header(session_dir, spec)
            _validate_spec_against_csv_header(session_dir, spec)
            _validate_prompt_declared_values(session_dir, spec)
            draft = _render_new_session_draft(spec)
            _validate_draft(session_dir, draft)
            break
        except ValidationError as exc:
            if str(exc).startswith("pfit-new missing required inputs:"):
                raise
            if attempt >= workflow_config.max_repair_attempts:
                raise
            repaired_response = _try_repair_expressions(
                session_dir,
                response,
                str(exc),
                llm_client,
                prompt_renderer,
                workflow_config,
                generated,
                context,
            )
            if repaired_response is not None:
                response = repaired_response
                continue
            if _is_loss_body_validation_error(exc):
                repaired_response = _try_repair_loss_body(
                    session_dir,
                    response,
                    str(exc),
                    llm_client,
                    prompt_renderer,
                    workflow_config,
                    generated,
                    context,
                )
                if repaired_response is not None:
                    response = repaired_response
                    continue
            raise ValidationError(
                f"{exc}. No targeted pfit-new repair is available for this error."
            ) from exc

    inputs = session_dir / "inputs"
    outputs = session_dir / "outputs"
    inputs.mkdir(parents=True, exist_ok=True)
    outputs.mkdir(exist_ok=True)

    written = [
        _write_if_allowed(inputs / "user_input.yaml", draft.user_input_yaml, overwrite),
        _write_if_allowed(inputs / "user_info.txt", draft.user_info_txt, overwrite=False),
        _write_if_allowed(generated / "user_model.py", draft.user_model_py, overwrite),
    ]
    if draft.review:
        written.append(_write_if_allowed(generated / "pfit_new_review.txt", draft.review, overwrite))
    return [path for path in written if path is not None]


def _draft_new_session_response(
    session_dir: Path,
    llm_client: LLMClient,
    prompt_renderer: PromptRenderer,
    workflow_config: WorkflowConfig,
    generated_dir: Path,
    context: dict[str, str],
) -> str:
    dataset_response = _complete_with_log(
        llm_client,
        generated_dir,
        "new_session_dataset",
        prompt_renderer.render_messages(
            "new_session_dataset.system.md",
            "new_session_dataset.user.md",
            context,
        ),
        temperature=workflow_config.temperature,
        max_tokens=workflow_config.max_tokens,
    )
    dataset_data = parse_llm_json_object(dataset_response, "pfit-new dataset")
    if _looks_like_full_new_session_response(dataset_data):
        return dataset_response
    missing_inputs = _parse_missing_inputs(dataset_data)
    if missing_inputs:
        return json.dumps(
            {
                "missing_inputs": list(missing_inputs),
                "review": _require_string(dataset_data, "review", allow_empty=True),
                "filename_data": "",
                "parameters": [],
                "fixed_parameters": [],
                "states": [],
                "helper_functions": [],
                "observables": [],
                "loss_body": "",
                "user_info_txt": _require_string(dataset_data, "user_info_txt", allow_empty=True),
            }
        )
    experiments = _parse_experiment_selection(dataset_data)
    filename_data = experiments[0]["data_file"]
    for experiment in experiments:
        _validate_dataset_filename(session_dir, experiment["data_file"])
    headers = [_read_csv_header(Path(session_dir) / "inputs" / e["data_file"]) for e in experiments]
    if any(header != headers[0] for header in headers):
        raise ValidationError("pfit-new experiments must have identical ordered CSV headers")

    frozen_dataset = {
        "filename_data": filename_data,
        "csv_header": headers[0] or [],
        "experiments": list(experiments),
    }
    parameter_response = _complete_with_log(
        llm_client,
        generated_dir,
        "new_session_parameters",
        prompt_renderer.render_messages(
            "new_session_parameters.system.md",
            "new_session_parameters.user.md",
            {
                **context,
                "filename_data": filename_data,
                "dataset_context": json.dumps(frozen_dataset, indent=2),
            },
        ),
        temperature=workflow_config.temperature,
        max_tokens=workflow_config.max_tokens,
    )
    parameter_data = parse_llm_json_object(parameter_response, "pfit-new parameters")
    if _looks_like_full_new_session_response(parameter_data):
        return _attach_experiments(parameter_response, experiments)
    missing_inputs = _parse_missing_inputs(parameter_data)
    if missing_inputs:
        return json.dumps(
            {
                "missing_inputs": list(missing_inputs),
                "review": _require_string(parameter_data, "review", allow_empty=True),
                "filename_data": "",
                "parameters": [],
                "fixed_parameters": [],
                "states": [],
                "helper_functions": [],
                "observables": [],
                "loss_body": "",
                "user_info_txt": _require_string(parameter_data, "user_info_txt", allow_empty=True),
            }
        )
    parameters = [_parameter_to_dict(_parse_parameter(item)) for item in _require_list(parameter_data, "parameters")]
    fixed_parameters = [
        _fixed_parameter_to_dict(_parse_fixed_parameter(item))
        for item in parameter_data.get("fixed_parameters", [])
    ]

    frozen_parameters = {
        "filename_data": filename_data,
        "csv_header": frozen_dataset["csv_header"],
        "experiments": list(experiments),
        "parameters": parameters,
        "fixed_parameters": fixed_parameters,
    }
    states_response = _complete_with_log(
        llm_client,
        generated_dir,
        "new_session_states",
        prompt_renderer.render_messages(
            "new_session_states.system.md",
            "new_session_states.user.md",
            {
                **context,
                "frozen_parameters": json.dumps(frozen_parameters, indent=2),
            },
        ),
        temperature=workflow_config.temperature,
        max_tokens=workflow_config.max_tokens,
    )
    states_data = parse_llm_json_object(states_response, "pfit-new states")
    if _looks_like_full_new_session_response(states_data):
        return _attach_experiments(states_response, experiments)
    missing_inputs = _parse_missing_inputs(states_data)
    if missing_inputs:
        return _missing_new_session_response(states_data, missing_inputs)

    frozen_states = {
        **frozen_parameters,
        "states": states_data.get("states", []),
    }
    equations_response = _complete_with_log(
        llm_client,
        generated_dir,
        "new_session_equations",
        prompt_renderer.render_messages(
            "new_session_equations.system.md",
            "new_session_equations.user.md",
            {
                **context,
                "frozen_states": json.dumps(frozen_states, indent=2),
                "large_model_guidance": _large_model_guidance(context, frozen_states),
            },
        ),
        temperature=workflow_config.temperature,
        max_tokens=workflow_config.max_tokens,
    )
    equations_data = parse_llm_json_object(equations_response, "pfit-new equations")
    if _looks_like_full_new_session_response(equations_data):
        return _attach_experiments(equations_response, experiments)
    missing_inputs = _parse_missing_inputs(equations_data)
    if missing_inputs:
        return _missing_new_session_response(equations_data, missing_inputs)

    frozen_equations = {
        **frozen_states,
        "formulas": equations_data.get("formulas", []),
        "rhs": equations_data.get("rhs", []),
    }
    observables_response = _complete_with_log(
        llm_client,
        generated_dir,
        "new_session_observables",
        prompt_renderer.render_messages(
            "new_session_observables.system.md",
            "new_session_observables.user.md",
            {
                **context,
                "frozen_equations": json.dumps(frozen_equations, indent=2),
            },
        ),
        temperature=workflow_config.temperature,
        max_tokens=workflow_config.max_tokens,
    )
    observables_data = parse_llm_json_object(observables_response, "pfit-new observables")
    if _looks_like_full_new_session_response(observables_data):
        return _attach_experiments(observables_response, experiments)
    missing_inputs = _parse_missing_inputs(observables_data)
    if missing_inputs:
        return _missing_new_session_response(observables_data, missing_inputs)

    frozen_observables = {
        **frozen_equations,
        "observables": observables_data.get("observables", []),
    }
    loss_response = _complete_with_log(
        llm_client,
        generated_dir,
        "new_session_loss",
        prompt_renderer.render_messages(
            "new_session_loss.system.md",
            "new_session_loss.user.md",
            {
                **context,
                "frozen_observables": json.dumps(frozen_observables, indent=2),
            },
        ),
        temperature=workflow_config.temperature,
        max_tokens=workflow_config.max_tokens,
    )
    loss_data = parse_llm_json_object(loss_response, "pfit-new loss")
    if _looks_like_full_new_session_response(loss_data):
        return _attach_experiments(loss_response, experiments)
    missing_inputs = _parse_missing_inputs(loss_data)
    if missing_inputs:
        return _missing_new_session_response(loss_data, missing_inputs)
    if len(experiments) == 1 and not user_loss_contract(Path(session_dir)):
        loss_data = _apply_automatic_log_loss(
            loss_data,
            Path(session_dir) / "inputs" / filename_data,
            frozen_dataset["csv_header"],
        )

    return _attach_experiments(json.dumps(
        _assemble_split_new_session_response(
            filename_data=filename_data,
            csv_header=frozen_dataset["csv_header"],
            parameters=parameters,
            fixed_parameters=fixed_parameters,
            states_data=states_data,
            equations_data=equations_data,
            observables_data=observables_data,
            loss_data=loss_data,
            session_context=str(context.get("session_context", "")),
        )
    ), experiments)


def _parse_experiment_selection(data: dict) -> tuple[dict, ...]:
    entries = data.get("experiments")
    if entries is None or entries == []:
        entries = [{"data_file": _require_string(data, "filename_data", allow_empty=False)}]
    if not isinstance(entries, list) or not entries:
        raise ValidationError("pfit-new experiments must be a nonempty list")
    result = []
    for entry in entries:
        if not isinstance(entry, dict) or set(entry) - {"data_file", "initial_conditions"}:
            raise ValidationError("pfit-new experiments require data_file and optional initial_conditions")
        filename = _require_string(entry, "data_file", allow_empty=False)
        overrides = entry.get("initial_conditions", {})
        if not isinstance(overrides, dict):
            raise ValidationError("pfit-new initial_conditions must be a mapping")
        normalized = {}
        for name, value in overrides.items():
            if not isinstance(name, str) or not name.isidentifier() or isinstance(value, bool):
                raise ValidationError("pfit-new invalid initial condition override")
            try:
                number = float(value)
            except (TypeError, ValueError) as exc:
                raise ValidationError("pfit-new initial conditions must be finite numbers") from exc
            if not math.isfinite(number):
                raise ValidationError("pfit-new initial conditions must be finite numbers")
            normalized[name] = number
        result.append({"data_file": filename, "initial_conditions": normalized})
    if data.get("filename_data") and data["filename_data"] != result[0]["data_file"]:
        raise ValidationError("filename_data must match the first experiment")
    return tuple(result)


def _attach_experiments(response: str, experiments) -> str:
    data = parse_llm_json_object(response, "pfit-new")
    data["experiments"] = list(experiments)
    data["filename_data"] = experiments[0]["data_file"]
    return json.dumps(data)


def _looks_like_full_new_session_response(data: dict[str, object]) -> bool:
    return ("filename_data" in data or "experiments" in data) and "parameters" in data and "states" in data


def _missing_new_session_response(
    data: dict[str, object],
    missing_inputs: tuple[str, ...],
) -> str:
    return json.dumps(
        {
            "missing_inputs": list(missing_inputs),
            "review": _require_string(data, "review", allow_empty=True),
            "filename_data": "",
            "parameters": [],
            "fixed_parameters": [],
            "states": [],
            "helper_functions": [],
            "observables": [],
            "loss_body": "",
            "user_info_txt": _require_string(data, "user_info_txt", allow_empty=True),
        }
    )


def _parse_missing_inputs(data: dict[str, object]) -> tuple[str, ...]:
    missing = data.get("missing_inputs", [])
    if not isinstance(missing, list) or not all(isinstance(item, str) for item in missing):
        raise ValidationError("pfit-new missing_inputs must be a list of strings")
    return tuple(item.strip() for item in missing if item.strip())


def _validate_dataset_filename(session_dir: Path, filename_data: str) -> None:
    csv_path = Path(session_dir) / "inputs" / filename_data
    if not csv_path.exists():
        raise ValidationError(f"pfit-new dataset CSV was not found in inputs/: {filename_data}")
    if csv_path.suffix.lower() != ".csv":
        raise ValidationError("pfit-new filename_data must name a CSV file")


def _parameter_to_dict(parameter: NewSessionParameter) -> dict[str, object]:
    return {
        "name": parameter.name,
        "min_value": parameter.min_value,
        "max_value": parameter.max_value,
        "logscale": parameter.logscale,
    }


def _fixed_parameter_to_dict(parameter: NewSessionFixedParameter) -> dict[str, object]:
    return {"name": parameter.name, "value": parameter.value}


def _assemble_split_new_session_response(
    *,
    filename_data: str,
    csv_header: list[str],
    parameters: list[dict[str, object]],
    fixed_parameters: list[dict[str, object]],
    states_data: dict[str, object],
    equations_data: dict[str, object],
    observables_data: dict[str, object],
    loss_data: dict[str, object],
    session_context: str = "",
) -> dict[str, object]:
    measurement_columns = {name.strip(): index for index, name in enumerate(csv_header[1:])}
    state_stubs = _parse_split_state_stubs(states_data)
    formulas = _parse_split_formulas(equations_data)
    rhs_by_state = _parse_split_rhs(equations_data)
    states = []
    for state in state_stubs:
        rhs = rhs_by_state.get(state["name"], "")
        if not rhs:
            raise ValidationError(f"pfit-new equations are missing RHS for state: {state['name']}")
        states.append(
            {
                "name": state["name"],
                "initial_value": state["initial_value"],
                "rhs": _normalize_math_calls(_inline_formulas(rhs, formulas)),
                "observed_column": measurement_columns.get(state["name"]),
            }
        )

    fixed_parameters = _drop_initial_value_fixed_parameters(fixed_parameters, states)
    state_names = {state["name"] for state in states}
    observables = []
    for item in observables_data.get("observables", []):
        if not isinstance(item, dict):
            raise ValidationError("pfit-new observables entries must be objects")
        measured = _clean_identifier(
            _require_string(item, "measured", allow_empty=False),
            "observable",
        )
        expression = (
            _require_string(item, "expression", allow_empty=True).strip()
            or _require_string(item, "simulated", allow_empty=True).strip()
            or measured
        )
        column = measurement_columns.get(measured)
        if column is None:
            raise ValidationError(f"pfit-new observable is not in CSV header: {measured}")
        if measured in state_names:
            if expression != measured:
                raise ValidationError(
                    "pfit-new CSV header matches an integrated state, but observable "
                    f"mapping for {measured} points to {expression}"
                )
            for state in states:
                if state["name"] == measured:
                    state["observed_column"] = column
                    break
            continue
        observables.append(
            {
                "name": measured,
                "expression": _normalize_math_calls(_inline_formulas(expression, formulas)),
                "observed_column": column,
            }
        )

    loss_data = _normalize_split_loss_data(
        loss_data,
        states,
        observables,
        observables_data.get("observables", []),
        csv_header,
    )
    auxiliary_columns = _auxiliary_columns_from_loss_data(loss_data, measurement_columns)
    spec_for_loss = NewSessionSpec(
        missing_inputs=(),
        review="",
        filename_data=filename_data,
        parameters=tuple(_parse_parameter(item) for item in parameters),
        fixed_parameters=tuple(_parse_fixed_parameter(item) for item in fixed_parameters),
        states=tuple(_parse_state(item) for item in states),
        helper_functions=(),
        observables=tuple(_parse_observable(item) for item in observables),
        auxiliary_columns=tuple(_parse_auxiliary_column(item) for item in auxiliary_columns),
        loss_body="",
        user_info_txt="",
    )
    loss_body = _render_structured_loss_body(spec_for_loss, loss_data)
    reviews = [
        _require_string(data, "review", allow_empty=True)
        for data in (states_data, equations_data, observables_data, loss_data)
        if _require_string(data, "review", allow_empty=True)
    ]
    if _text_requests_stiff_integrator(session_context):
        reviews.append("Original user context indicates a stiff model.")
    return {
        "missing_inputs": [],
        "review": "\n".join(reviews),
        "filename_data": filename_data,
        "parameters": parameters,
        "fixed_parameters": fixed_parameters,
        "helper_functions": [],
        "states": states,
        "observables": observables,
        "auxiliary_columns": auxiliary_columns,
        "loss_body": loss_body,
        "user_info_txt": _require_string(loss_data, "user_info_txt", allow_empty=True)
        or _require_string(observables_data, "user_info_txt", allow_empty=True)
        or "Local pfit-new draft generated from supplied files.",
    }


def _drop_initial_value_fixed_parameters(
    fixed_parameters: list[dict[str, object]],
    states: list[dict[str, object]],
) -> list[dict[str, object]]:
    initial_values = {
        f"{state['name']}_0": float(state["initial_value"])
        for state in states
        if "name" in state and "initial_value" in state
    }
    cleaned = []
    for parameter in fixed_parameters:
        name = parameter.get("name")
        if isinstance(name, str) and name in initial_values:
            try:
                value = float(parameter.get("value"))
            except (TypeError, ValueError):
                cleaned.append(parameter)
                continue
            if value == initial_values[name]:
                continue
        cleaned.append(parameter)
    return cleaned


def _apply_automatic_log_loss(
    loss_data: dict[str, object],
    csv_path: Path,
    csv_header: list[str],
) -> dict[str, object]:
    data_terms = loss_data.get("data_terms", [])
    if not isinstance(data_terms, list) or not csv_header:
        return loss_data
    log_range_by_name = _positive_log10_ranges(csv_path, csv_header)
    if not log_range_by_name:
        return loss_data

    changed = False
    normalized_terms = []
    for item in data_terms:
        if not isinstance(item, dict):
            normalized_terms.append(item)
            continue
        term = dict(item)
        measured = str(term.get("measured", ""))
        metric = str(term.get("metric", "")).lower()
        if metric in {"normalized_mse", "range_normalized_mse"}:
            log_range = log_range_by_name.get(measured)
            if log_range is not None and log_range >= 3.0:
                term["metric"] = "log10_normalized_mse"
                changed = True
        normalized_terms.append(term)
    if not changed:
        return loss_data
    normalized = dict(loss_data)
    normalized["data_terms"] = normalized_terms
    review = str(normalized.get("review", "")).strip()
    note = (
        "Automatically using log10_normalized_mse for positive measured columns "
        "that span at least 3 log10 orders of magnitude."
    )
    normalized["review"] = f"{review}\n{note}" if review else note
    return normalized


def _positive_log10_ranges(csv_path: Path, csv_header: list[str]) -> dict[str, float]:
    if len(csv_header) < 2:
        return {}
    columns = [[] for _ in csv_header[1:]]
    try:
        with csv_path.open(newline="", encoding="utf-8-sig") as handle:
            reader = csv.reader(handle)
            next(reader, None)
            for row in reader:
                for index in range(1, min(len(row), len(csv_header))):
                    try:
                        columns[index - 1].append(float(row[index]))
                    except ValueError:
                        continue
    except OSError:
        return {}

    ranges = {}
    for name, values in zip(csv_header[1:], columns):
        finite_values = [value for value in values if value > 0.0]
        if not finite_values or len(finite_values) != len(values):
            continue
        min_value = min(finite_values)
        max_value = max(finite_values)
        if min_value <= 0.0 or max_value <= 0.0:
            continue
        ranges[name.strip()] = math.log10(max_value) - math.log10(min_value)
    return ranges


def _parse_split_state_stubs(data: dict[str, object]) -> list[dict[str, object]]:
    states = []
    for item in _require_list(data, "states"):
        if not isinstance(item, dict):
            raise ValidationError("pfit-new states entries must be objects")
        states.append(
            {
                "name": _clean_identifier(
                    _require_string(item, "name", allow_empty=False),
                    "state",
                ),
                "initial_value": float(item["initial_value"]),
            }
        )
    return states


def _parse_split_formulas(data: dict[str, object]) -> tuple[NewSessionFormula, ...]:
    formulas = []
    for item in data.get("formulas", []):
        if not isinstance(item, dict):
            raise ValidationError("pfit-new formulas entries must be objects")
        formulas.append(
            NewSessionFormula(
                name=_clean_identifier(
                    _require_string(item, "name", allow_empty=False),
                    "formula",
                ),
                expression=_require_string(item, "expression", allow_empty=False).replace("^", "**"),
            )
        )
    return tuple(formulas)


def _parse_split_rhs(data: dict[str, object]) -> dict[str, str]:
    rhs_by_state = {}
    for item in _require_list(data, "rhs"):
        if not isinstance(item, dict):
            raise ValidationError("pfit-new rhs entries must be objects")
        state = _clean_identifier(
            _require_string(item, "state", allow_empty=False),
            "state",
        )
        rhs_by_state[state] = _require_string(item, "expression", allow_empty=False).replace("^", "**")
    return rhs_by_state


def _large_model_guidance(context: dict[str, str], frozen_states: dict[str, object]) -> str:
    state_count = len(frozen_states.get("states", []))
    parameter_count = len(frozen_states.get("parameters", [])) + len(
        frozen_states.get("fixed_parameters", [])
    )
    user_text_size = len(str(context.get("session_context", "")))
    if state_count < 8 and parameter_count < 20 and user_text_size < 12000:
        return "LARGE_MODEL_MODE: off."
    state_names = [
        str(item.get("name"))
        for item in frozen_states.get("states", [])
        if isinstance(item, dict) and item.get("name")
    ]
    fixed_names = [
        str(item.get("name"))
        for item in frozen_states.get("fixed_parameters", [])
        if isinstance(item, dict) and item.get("name")
    ]
    return "\n".join(
        [
            "LARGE_MODEL_MODE: on.",
            "Extract equations conservatively from the user text. Do not summarize, rename, or simplify the system.",
            "Every RHS entry must correspond exactly to one frozen state name.",
            "Use fixed parameter names exactly as frozen; never change fixed parameter values.",
            "If a state equation is not explicitly recoverable, return missing_inputs instead of inventing it.",
            "Frozen states: " + ", ".join(state_names),
            "Frozen fixed parameters: " + (", ".join(fixed_names) if fixed_names else "none"),
        ]
    )


def _normalize_split_loss_data(
    loss_data: dict[str, object],
    states: list[dict[str, object]],
    observables: list[dict[str, object]],
    raw_observables: list[object],
    csv_header: list[str],
) -> dict[str, object]:
    normalized = dict(loss_data)
    state_names = {str(state["name"]) for state in states}
    simulated_to_measured = {name: name for name in state_names}
    for observable in observables:
        simulated_to_measured[str(observable["name"])] = str(observable["name"])
        expression = str(observable["expression"])
        if expression:
            simulated_to_measured[expression] = str(observable["name"])
    for observable in raw_observables:
        if not isinstance(observable, dict):
            continue
        measured = str(observable.get("measured", ""))
        simulated = str(observable.get("simulated", ""))
        expression = str(observable.get("expression", ""))
        if measured and simulated:
            simulated_to_measured[simulated] = measured
        if measured and expression:
            simulated_to_measured[expression] = measured

    data_terms = []
    loss_text = " ".join(
        str(loss_data.get(key, ""))
        for key in ("review", "user_info_txt")
    ).lower()
    measurement_columns = {name.strip(): index for index, name in enumerate(csv_header[1:])}
    for item in loss_data.get("data_terms", []):
        if not isinstance(item, dict):
            data_terms.append(item)
            continue
        term = dict(item)
        simulated = str(term.get("simulated", ""))
        measured = str(term.get("measured", ""))
        if simulated.startswith("log_") and measured.startswith("log_"):
            base_simulated = simulated.removeprefix("log_")
            mapped_measured = simulated_to_measured.get(base_simulated)
            if mapped_measured is not None:
                term["simulated"] = base_simulated
                term["measured"] = mapped_measured
                term["metric"] = "log10_normalized_mse"
        metric = str(term.get("metric", "")).lower()
        measured = str(term.get("measured", ""))
        sigma = str(term.get("sigma", "") or term.get("uncertainty", ""))
        if not sigma and measured:
            candidate = f"{measured}_sd"
            if candidate in measurement_columns and (
                "standard deviation" in loss_text
                or "std" in loss_text
                or "sigma" in loss_text
                or "_sd" in loss_text
            ):
                sigma = candidate
        if sigma:
            term["metric"] = "sigma_weighted_mse"
            term["sigma"] = sigma
        metric = str(term.get("metric", "")).lower()
        if "max(abs" in loss_text or "maximum absolute" in loss_text or "max absolute" in loss_text:
            if metric == "normalized_mse":
                term["metric"] = "max_abs_normalized_mse"
            elif metric == "log10_normalized_mse":
                term["metric"] = "log10_max_abs_normalized_mse"
        if _text_requests_rmse_loss(loss_text):
            metric = str(term.get("metric", "")).lower()
            if metric == "normalized_mse":
                term["metric"] = "normalized_rmse"
            elif metric == "max_abs_normalized_mse":
                term["metric"] = "max_abs_normalized_rmse"
            elif metric == "log10_normalized_mse":
                term["metric"] = "log10_normalized_rmse"
            elif metric == "log10_max_abs_normalized_mse":
                term["metric"] = "log10_max_abs_normalized_rmse"
            elif metric == "sigma_weighted_mse":
                term["metric"] = "sigma_weighted_rmse"
        data_terms.append(term)
    normalized["data_terms"] = data_terms

    penalties = []
    for item in loss_data.get("penalties", []):
        if not isinstance(item, dict):
            penalties.append(item)
            continue
        penalty = dict(item)
        left = str(penalty.get("left", ""))
        op = str(penalty.get("op", ""))
        if "sqrt" in left and op in {">", ">="}:
            for name in sorted(simulated_to_measured, key=len, reverse=True):
                measured_name = simulated_to_measured[name]
                if name in left and f"{measured_name}_obs" in left:
                    threshold = float(penalty.get("right", 0.0))
                    penalty["left"] = name
                    penalty["left_kind"] = "simulated"
                    penalty["op"] = "abs_diff_gt"
                    penalty["right"] = measured_name
                    penalty["right_kind"] = "measured"
                    penalty["threshold"] = threshold
                    break
        penalties.append(penalty)
    normalized["penalties"] = penalties
    return normalized


def _text_requests_rmse_loss(text: str) -> bool:
    return "rmse" in text or "root mean square" in text or "sqrt(mean" in text


def _auxiliary_columns_from_loss_data(
    loss_data: dict[str, object],
    measurement_columns: dict[str, int],
) -> list[dict[str, object]]:
    auxiliary: dict[str, dict[str, object]] = {}
    for item in loss_data.get("data_terms", []):
        if not isinstance(item, dict):
            continue
        sigma = str(item.get("sigma", ""))
        measured = str(item.get("measured", ""))
        if sigma and sigma in measurement_columns:
            auxiliary[sigma] = {
                "name": sigma,
                "observed_column": measurement_columns[sigma],
                "kind": "uncertainty_of",
                "target": measured,
            }
    return [auxiliary[name] for name in sorted(auxiliary, key=lambda item: measurement_columns[item])]


def _inline_formulas(expression: str, formulas: tuple[NewSessionFormula, ...]) -> str:
    if not formulas:
        return expression.replace("^", "**")
    formula_map = {formula.name: formula.expression for formula in formulas}
    try:
        tree = ast.parse(expression.replace("^", "**"), mode="eval")
    except SyntaxError:
        return expression.replace("^", "**")
    for _ in range(len(formula_map) + 1):
        previous = ast.dump(tree)
        inlined = _FormulaInliner(formula_map).visit(tree)
        ast.fix_missing_locations(inlined)
        if ast.dump(inlined) == previous:
            break
        tree = inlined
    return ast.unparse(tree)


class _FormulaInliner(ast.NodeTransformer):
    def __init__(self, formulas: dict[str, str]) -> None:
        self.formulas = formulas

    def visit_Name(self, node: ast.Name) -> ast.AST:
        if not isinstance(node.ctx, ast.Load) or node.id not in self.formulas:
            return node
        try:
            replacement = ast.parse(self.formulas[node.id].replace("^", "**"), mode="eval").body
        except SyntaxError:
            return node
        return copy.deepcopy(replacement)


def _render_structured_loss_body(spec: NewSessionSpec, loss_data: dict[str, object]) -> str:
    custom_loss = bool(loss_data.get("custom_loss", False))
    data_terms = loss_data.get("data_terms", [])
    penalties = loss_data.get("penalties", [])
    if not custom_loss and not data_terms and not penalties:
        return ""
    if not isinstance(data_terms, list) or not isinstance(penalties, list):
        raise ValidationError("pfit-new loss data_terms and penalties must be lists")

    needs_observables = bool(spec.observables)
    lines = []
    if needs_observables:
        lines.append("observables = _observables(solution, trainable_parameters, fixed_parameters)")
    lines.append("loss = 0.0")
    rmse_term_count = 0
    for item in data_terms:
        if not isinstance(item, dict):
            raise ValidationError("pfit-new loss data_terms entries must be objects")
        simulated = _require_string(item, "simulated", allow_empty=False)
        measured = _require_string(item, "measured", allow_empty=False)
        metric = _require_string(item, "metric", allow_empty=True).lower() or "mse"
        simulated = _resolve_simulated_loss_name(spec, simulated, measured)
        measured_ref = _measured_reference(spec, measured)
        residual = f"{_series_reference(spec, simulated)} - {measured_ref}"
        if metric == "mae":
            lines.append(f"loss += np.mean(np.abs({residual}))")
        elif metric == "mse":
            lines.append(f"loss += np.mean(np.square({residual}))")
        elif metric == "rmse":
            lines.append(f"loss += np.mean(np.square({residual}))")
            rmse_term_count += 1
        elif metric in {"normalized_mse", "range_normalized_mse"}:
            scale = f"(np.max({measured_ref}) - np.min({measured_ref}) + 1e-12)"
            lines.append(f"loss += np.mean(np.square(({residual}) / {scale}))")
        elif metric in {"max_abs_normalized_mse"}:
            scale = f"(np.max(np.abs({measured_ref})) + 1e-12)"
            lines.append(f"loss += np.mean(np.square(({residual}) / {scale}))")
        elif metric in {"normalized_rmse", "range_normalized_rmse"}:
            scale = f"(np.max({measured_ref}) - np.min({measured_ref}) + 1e-12)"
            lines.append(f"loss += np.mean(np.square(({residual}) / {scale}))")
            rmse_term_count += 1
        elif metric in {"max_abs_normalized_rmse"}:
            scale = f"(np.max(np.abs({measured_ref})) + 1e-12)"
            lines.append(f"loss += np.mean(np.square(({residual}) / {scale}))")
            rmse_term_count += 1
        elif metric in {"sigma_weighted_mse", "uncertainty_weighted_mse"}:
            sigma = _require_string(item, "sigma", allow_empty=False)
            sigma_ref = _dataset_reference_by_name(spec, sigma)
            lines.append(f"loss += np.mean(np.square(({residual}) / ({sigma_ref} + 1e-12)))")
        elif metric in {"sigma_weighted_rmse", "uncertainty_weighted_rmse"}:
            sigma = _require_string(item, "sigma", allow_empty=False)
            sigma_ref = _dataset_reference_by_name(spec, sigma)
            lines.append(f"loss += np.mean(np.square(({residual}) / ({sigma_ref} + 1e-12)))")
            rmse_term_count += 1
        elif metric in {
            "log10_normalized_mse",
            "log_normalized_mse",
            "log10_normalized_rmse",
            "log_normalized_rmse",
            "log10_max_abs_normalized_mse",
            "log10_max_abs_normalized_rmse",
        }:
            positive_measured = f"np.where({measured_ref} > 0.0, {measured_ref}, np.inf)"
            eps_name = f"eps_{measured}"
            log_sim_name = f"log_sim_{measured}"
            log_measured_name = f"log_measured_{measured}"
            scale_name = f"scale_log_{measured}"
            lines.append(f"{eps_name} = np.min({positive_measured})")
            lines.append(f"{log_sim_name} = np.log10({_series_reference(spec, simulated)} + {eps_name})")
            lines.append(f"{log_measured_name} = np.log10({measured_ref} + {eps_name})")
            if "max_abs" in metric:
                lines.append(f"{scale_name} = np.max(np.abs({log_measured_name})) + 1e-12")
            else:
                lines.append(f"{scale_name} = np.max({log_measured_name}) - np.min({log_measured_name}) + 1e-12")
            lines.append(f"loss += np.mean(np.square(({log_sim_name} - {log_measured_name}) / {scale_name}))")
            if metric.endswith("_rmse"):
                rmse_term_count += 1
        else:
            raise ValidationError(f"pfit-new unsupported loss metric: {metric}")
    if rmse_term_count:
        if rmse_term_count != len(data_terms):
            raise ValidationError("pfit-new cannot mix RMSE and non-RMSE data terms")
        lines.append(f"loss = np.sqrt(loss / {rmse_term_count!r})")
    for item in penalties:
        if not isinstance(item, dict):
            raise ValidationError("pfit-new loss penalties entries must be objects")
        violation = _penalty_violation(spec, item)
        value = float(item.get("value", 0.0))
        sharpness = float(item.get("sharpness", 1000.0))
        lines.append(
            "loss += "
            f"{value!r} / (1.0 + np.exp(-np.clip({sharpness!r} * ({violation}), -60.0, 60.0)))"
        )
    lines.append("return float(loss)")
    return "\n".join(lines)


def _resolve_simulated_loss_name(spec: NewSessionSpec, simulated: str, measured: str) -> str:
    known_names = {state.name for state in spec.states} | {observable.name for observable in spec.observables}
    if simulated in known_names:
        return simulated
    for observable in spec.observables:
        if observable.name == measured:
            return observable.name
    return simulated


def _series_reference(spec: NewSessionSpec, name: str) -> str:
    for index, state in enumerate(spec.states):
        if state.name == name:
            return f"solution[:, {index}]"
    for observable in spec.observables:
        if observable.name == name:
            return f"observables['{observable.name}']"
    raise ValidationError(f"pfit-new loss references unknown simulated quantity: {name}")


def _measured_reference(spec: NewSessionSpec, name: str) -> str:
    for state in spec.states:
        if state.name == name and state.observed_column is not None:
            return f"dataset[:, {state.observed_column}]"
    for observable in spec.observables:
        if observable.name == name:
            return f"dataset[:, {observable.observed_column}]"
    raise ValidationError(f"pfit-new loss references unknown measured quantity: {name}")


def _dataset_reference_by_name(spec: NewSessionSpec, name: str) -> str:
    for state in spec.states:
        if state.name == name and state.observed_column is not None:
            return f"dataset[:, {state.observed_column}]"
    for observable in spec.observables:
        if observable.name == name:
            return f"dataset[:, {observable.observed_column}]"
    for column in spec.auxiliary_columns:
        if column.name == name:
            return f"dataset[:, {column.observed_column}]"
    raise ValidationError(f"pfit-new loss references unknown dataset column: {name}")


def _penalty_violation(spec: NewSessionSpec, item: dict[str, object]) -> str:
    left = _final_reference(
        spec,
        _require_string(item, "left", allow_empty=False),
        kind=_require_string(item, "left_kind", allow_empty=True) or "simulated",
    )
    op = _require_string(item, "op", allow_empty=False)
    right_value = item.get("right", 0.0)
    right = _final_reference(
        spec,
        right_value,
        kind=_require_string(item, "right_kind", allow_empty=True) or "literal",
    )
    if op in {">", ">=", "<", "<=", "=="}:
        if op in {">", ">="}:
            return f"{left} - {right}"
        if op in {"<", "<="}:
            return f"{right} - {left}"
        return f"-np.abs({left} - {right})"
    if op == "abs_diff_gt":
        threshold = float(item["threshold"])
        return f"np.sqrt(np.square({left} - {right}) + 1e-12) - {threshold!r}"
    raise ValidationError(f"pfit-new unsupported penalty operator: {op}")


def _final_reference(spec: NewSessionSpec, value: object, *, kind: str) -> str:
    if kind == "literal":
        return repr(float(value))
    if not isinstance(value, str):
        raise ValidationError("pfit-new non-literal penalty references must be strings")
    if kind == "measured":
        return _measured_reference(spec, value).replace("[:,", "[-1,")
    if kind == "simulated":
        return _series_reference(spec, value).replace("[:,", "[-1,")
    raise ValidationError(f"pfit-new unsupported penalty reference kind: {kind}")


def _is_loss_body_validation_error(exc: ValidationError) -> bool:
    return "loss_body" in str(exc)


def _try_repair_expressions(
    session_dir: Path,
    previous_response: str,
    validation_error: str,
    llm_client: LLMClient,
    prompt_renderer: PromptRenderer,
    workflow_config: WorkflowConfig,
    generated_dir: Path,
    context: dict[str, str],
) -> str | None:
    if "expression" not in validation_error and "np.* functions" not in validation_error:
        return None
    try:
        spec = _parse_new_session_response(previous_response, validate=False)
    except ValidationError:
        return None
    if spec.missing_inputs:
        return None
    spec = _canonicalize_observed_columns_from_csv_header(session_dir, spec)
    normalized_spec = _normalize_math_calls_in_spec(spec)
    if normalized_spec != spec:
        return json.dumps(_spec_to_response_dict(normalized_spec))
    current_spec = _spec_to_response_dict(spec)
    messages = prompt_renderer.render_messages(
        "repair_new_session_expressions.system.md",
        "repair_new_session_expressions.user.md",
        {
            **context,
            "validation_error": validation_error,
            "current_spec": json.dumps(current_spec, indent=2),
        },
    )
    repaired_response = _complete_with_log(
        llm_client,
        generated_dir,
        "repair_new_session_expressions",
        messages,
        temperature=workflow_config.temperature,
        max_tokens=min(workflow_config.max_tokens, 4096),
    )
    data = parse_llm_json_object(repaired_response, "pfit-new expression repair")
    state_rhs = {
        _require_string(item, "name", allow_empty=False): _require_string(
            item,
            "rhs",
            allow_empty=False,
        )
        for item in data.get("states", [])
        if isinstance(item, dict)
    }
    observable_expressions = {
        _require_string(item, "name", allow_empty=False): _require_string(
            item,
            "expression",
            allow_empty=False,
        )
        for item in data.get("observables", [])
        if isinstance(item, dict)
    }
    formulas = _parse_split_formulas(data)
    if not state_rhs and not observable_expressions and not formulas:
        return None
    for state in current_spec["states"]:
        if state["name"] in state_rhs:
            state["rhs"] = state_rhs[state["name"]]
        state["rhs"] = _inline_formulas(state["rhs"], formulas)
    for observable in current_spec["observables"]:
        if observable["name"] in observable_expressions:
            observable["expression"] = observable_expressions[observable["name"]]
        observable["expression"] = _inline_formulas(observable["expression"], formulas)
    return json.dumps(current_spec)


_NUMPY_FUNCTIONS = {
    "abs",
    "arctan",
    "cos",
    "exp",
    "isfinite",
    "log",
    "log10",
    "maximum",
    "mean",
    "minimum",
    "sign",
    "sin",
    "sqrt",
    "square",
    "tan",
    "where",
}


def _normalize_math_calls_in_spec(spec: NewSessionSpec) -> NewSessionSpec:
    states = tuple(
        NewSessionState(
            name=state.name,
            initial_value=state.initial_value,
            rhs=_normalize_math_calls(state.rhs),
            observed_column=state.observed_column,
        )
        for state in spec.states
    )
    observables = tuple(
        NewSessionObservable(
            name=observable.name,
            expression=_normalize_math_calls(observable.expression),
            observed_column=observable.observed_column,
        )
        for observable in spec.observables
    )
    if states == spec.states and observables == spec.observables:
        return spec
    return NewSessionSpec(
        missing_inputs=spec.missing_inputs,
        review=spec.review,
        filename_data=spec.filename_data,
        parameters=spec.parameters,
        fixed_parameters=spec.fixed_parameters,
        states=states,
        helper_functions=spec.helper_functions,
        observables=observables,
        auxiliary_columns=spec.auxiliary_columns,
        loss_body=spec.loss_body,
        user_info_txt=spec.user_info_txt,
        experiments=spec.experiments,
    )


def _normalize_math_calls(expression: str) -> str:
    try:
        tree = ast.parse(expression, mode="eval")
    except SyntaxError:
        return expression
    normalized = _MathCallNormalizer().visit(tree)
    ast.fix_missing_locations(normalized)
    return ast.unparse(normalized)


class _MathCallNormalizer(ast.NodeTransformer):
    def visit_Call(self, node: ast.Call) -> ast.AST:
        self.generic_visit(node)
        if isinstance(node.func, ast.Name) and node.func.id in _NUMPY_FUNCTIONS:
            node.func = ast.Attribute(
                value=ast.Name(id="np", ctx=ast.Load()),
                attr=node.func.id,
                ctx=ast.Load(),
            )
        return node


def _try_repair_loss_body(
    session_dir: Path,
    previous_response: str,
    validation_error: str,
    llm_client: LLMClient,
    prompt_renderer: PromptRenderer,
    workflow_config: WorkflowConfig,
    generated_dir: Path,
    context: dict[str, str],
) -> str | None:
    try:
        spec = _parse_new_session_response(previous_response, validate=False)
    except ValidationError:
        return None
    if spec.missing_inputs:
        return None
    spec = _canonicalize_observed_columns_from_csv_header(session_dir, spec)
    current_spec = _spec_to_response_dict(spec)
    messages = prompt_renderer.render_messages(
        "repair_new_session_loss_body.system.md",
        "repair_new_session_loss_body.user.md",
        {
            **context,
            "validation_error": validation_error,
            "current_spec": json.dumps(current_spec, indent=2),
            "loss_body": spec.loss_body,
        },
    )
    repaired_response = _complete_with_log(
        llm_client,
        generated_dir,
        "repair_new_session_loss_body",
        messages,
        temperature=workflow_config.temperature,
        max_tokens=min(workflow_config.max_tokens, 4096),
    )
    data = parse_llm_json_object(repaired_response, "pfit-new loss_body repair")
    loss_body = _require_string(data, "loss_body", allow_empty=True)
    current_spec["loss_body"] = loss_body
    return json.dumps(current_spec)


def _spec_to_response_dict(spec: NewSessionSpec) -> dict[str, object]:
    return {
        "missing_inputs": list(spec.missing_inputs),
        "review": spec.review,
        "filename_data": spec.filename_data,
        "experiments": list(spec.experiments),
        "parameters": [
            {
                "name": parameter.name,
                "min_value": parameter.min_value,
                "max_value": parameter.max_value,
                "logscale": parameter.logscale,
            }
            for parameter in spec.parameters
        ],
        "fixed_parameters": [
            {"name": parameter.name, "value": parameter.value}
            for parameter in spec.fixed_parameters
        ],
        "helper_functions": list(spec.helper_functions),
        "states": [
            {
                "name": state.name,
                "initial_value": state.initial_value,
                "rhs": state.rhs,
                "observed_column": state.observed_column,
            }
            for state in spec.states
        ],
        "observables": [
            {
                "name": observable.name,
                "expression": observable.expression,
                "observed_column": observable.observed_column,
            }
            for observable in spec.observables
        ],
        "auxiliary_columns": [
            {
                "name": column.name,
                "observed_column": column.observed_column,
                "kind": column.kind,
                "target": column.target,
            }
            for column in spec.auxiliary_columns
        ],
        "loss_body": spec.loss_body,
        "user_info_txt": spec.user_info_txt,
    }


def _write_if_allowed(path: Path, content: str, overwrite: bool) -> Path | None:
    if path.exists() and not overwrite:
        return None
    path.write_text(content)
    return path


def _collect_session_context(session_dir: Path) -> str:
    session_dir = Path(session_dir)
    if not session_dir.exists():
        return "Session directory does not exist yet. No user files were supplied."

    blocks = []
    for path in sorted(item for item in session_dir.rglob("*") if item.is_file()):
        relative = path.relative_to(session_dir)
        if _is_generated_context_file(relative):
            continue
        blocks.append(f"FILE: {relative}\n{_read_context_file(path)}")

    if not blocks:
        return "Session directory exists but contains no user-supplied files."
    return "\n\n---\n\n".join(blocks)


def _is_generated_context_file(relative_path: Path) -> bool:
    if relative_path == Path("inputs/user_input.yaml"):
        return True
    if relative_path.parts[:1] == ("outputs",):
        return True
    if relative_path.parts[:1] == ("generated",):
        return True
    return False


def _read_context_file(path: Path) -> str:
    if path.suffix.lower() == ".csv":
        return _summarize_csv_for_prompt(path)
    if path.stat().st_size > 50_000:
        return "<skipped: file is larger than 50KB>"
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return "<binary or unsupported text encoding>"


def _summarize_csv_for_prompt(path: Path, *, sample_rows: int = 5) -> str:
    try:
        with path.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.reader(handle))
    except UnicodeDecodeError:
        return "<csv file with unsupported text encoding>"
    if not rows:
        return "<empty csv file>"

    header = rows[0]
    has_header = _row_looks_like_header(header)
    data_rows = rows[1:] if has_header else rows
    preview = rows[: sample_rows + 1]
    total_rows = len(data_rows)
    header_status = "present" if has_header else "missing"
    return "\n".join(
        [
            f"<csv summary: {total_rows} data rows, {len(header)} columns, header {header_status}>",
            f"<csv columns: {', '.join(header) if has_header else 'unlabeled'}>",
            "<csv sample>",
            *(",".join(row) for row in preview),
            "</csv sample>",
        ]
    )


def _row_looks_like_header(row: list[str]) -> bool:
    if not row:
        return False
    return not all(_is_float(cell.strip()) for cell in row)


def _is_float(value: str) -> bool:
    if not value:
        return False
    try:
        float(value)
    except ValueError:
        return False
    return True


def _validate_csv_inputs_have_headers(session_dir: Path) -> None:
    inputs = Path(session_dir) / "inputs"
    if not inputs.exists():
        return
    for path in sorted(inputs.glob("*.csv")):
        header = _read_csv_header(path)
        if header is None:
            continue
        if not _row_looks_like_header(header):
            raise ValidationError(
                f"pfit-new CSV must include a header row naming time and observed variables: "
                f"{path.name}"
            )


def _validate_spec_against_csv_header(session_dir: Path, spec: NewSessionSpec) -> None:
    experiments = spec.experiments or ({"data_file": spec.filename_data, "initial_conditions": {}},)
    first_header = None
    for index, experiment in enumerate(experiments):
        filename = experiment["data_file"]
        _validate_dataset_filename(session_dir, filename)
        header = _read_csv_header(Path(session_dir) / "inputs" / filename)
        if first_header is not None and header != first_header:
            raise ValidationError(f"Experiment {index + 1} ({filename}): ordered CSV headers differ")
        first_header = header
        _validate_one_spec_against_csv_header(session_dir, replace(spec, filename_data=filename))


def _validate_one_spec_against_csv_header(session_dir: Path, spec: NewSessionSpec) -> None:
    csv_path = Path(session_dir) / "inputs" / spec.filename_data
    if not csv_path.exists():
        raise ValidationError(f"pfit-new dataset CSV was not found in inputs/: {spec.filename_data}")
    header = _read_csv_header(csv_path)
    if not header or not _row_looks_like_header(header):
        raise ValidationError(
            f"pfit-new CSV must include a header row naming time and observed variables: "
            f"{spec.filename_data}"
        )
    if len(header) < 2:
        raise ValidationError("pfit-new CSV header must include time and at least one observed column")
    if header[0].strip().lower() != "time":
        raise ValidationError("pfit-new CSV first column must be named time")

    measurement_names = [name.strip() for name in header[1:]]
    observed_names = _observed_column_names(spec)
    auxiliary_names = {column.name for column in spec.auxiliary_columns}
    observable_measurement_names = [
        name
        for name in measurement_names
        if name not in auxiliary_names and not _is_auxiliary_measurement_column(name)
    ]
    if observed_names != observable_measurement_names:
        raise ValidationError(
            "pfit-new observed state/observable names must match the CSV header after time: "
            f"expected {observable_measurement_names}, got {observed_names}"
        )
    if spec.loss_body:
        _validate_loss_body_array_indices(
            spec.loss_body,
            dataset_width=len(measurement_names),
            solution_width=len(spec.states),
        )


def _validate_prompt_declared_values(session_dir: Path, spec: NewSessionSpec) -> None:
    user_info = Path(session_dir) / "inputs" / "user_info.txt"
    if not user_info.exists():
        return
    text = user_info.read_text()
    fixed_values = _extract_numeric_assignments_from_sections(
        text,
        ("fixed parameter", "fixed parameters", "fixed constants", "constants"),
    )
    actual_fixed = {parameter.name: parameter.value for parameter in spec.fixed_parameters}
    for name, expected in fixed_values.items():
        if name not in actual_fixed:
            raise ValidationError(
                f"pfit-new missing fixed parameter declared in user prompt: {name}"
            )
        if not math.isclose(actual_fixed[name], expected, rel_tol=1e-12, abs_tol=1e-12):
            raise ValidationError(
                f"pfit-new fixed parameter {name} value {actual_fixed[name]!r} "
                f"does not match user prompt value {expected!r}"
            )

    initial_values = _extract_initial_values_from_prompt(text)
    actual_states = {state.name: state.initial_value for state in spec.states}
    for name, expected in initial_values.items():
        if name not in actual_states:
            continue
        if not math.isclose(actual_states[name], expected, rel_tol=1e-12, abs_tol=1e-12):
            raise ValidationError(
                f"pfit-new state {name} initial value {actual_states[name]!r} "
                f"does not match user prompt value {expected!r}"
            )


def _extract_initial_values_from_prompt(text: str) -> dict[str, float]:
    values = _extract_numeric_assignments_from_sections(
        text,
        ("states", "state initial values", "initial conditions", "integrated variables"),
    )
    normalized: dict[str, float] = {}
    for name, value in values.items():
        normalized[_strip_initial_value_suffix(name)] = value
    for match in re.finditer(
        r"\b([A-Za-z_]\w*)\s*\(\s*0\s*\)\s*=\s*(" + _NUMBER_PATTERN + r")",
        text,
    ):
        normalized[match.group(1)] = float(match.group(2))
    return normalized


_NUMBER_PATTERN = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"


def _extract_numeric_assignments_from_sections(
    text: str,
    headings: tuple[str, ...],
) -> dict[str, float]:
    values: dict[str, float] = {}
    in_section = False
    normalized_headings = {heading.lower().rstrip(":") for heading in headings}
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        heading = line.lstrip("#").strip().lower().rstrip(":")
        if heading in normalized_headings:
            in_section = True
            continue
        if in_section and line.endswith(":") and "=" not in line:
            break
        if not in_section:
            continue
        for match in re.finditer(
            r"\b([A-Za-z_]\w*)\b\s*(?:=|:)\s*(" + _NUMBER_PATTERN + r")",
            line,
        ):
            values[match.group(1)] = float(match.group(2))
    return values


def _strip_initial_value_suffix(name: str) -> str:
    for suffix in ("_0", "0"):
        if name.endswith(suffix) and len(name) > len(suffix):
            return name[: -len(suffix)]
    return name


def _is_auxiliary_measurement_column(name: str) -> bool:
    normalized = name.strip().lower()
    return normalized.endswith(("_sd", "_sigma", "_std", "_stderr", "_se"))


def _canonicalize_observed_columns_from_csv_header(
    session_dir: Path,
    spec: NewSessionSpec,
) -> NewSessionSpec:
    csv_path = Path(session_dir) / "inputs" / spec.filename_data
    if not csv_path.exists():
        return spec
    header = _read_csv_header(csv_path)
    if not header or not _row_looks_like_header(header) or len(header) < 2:
        return spec

    measurement_columns = {
        name.strip(): index for index, name in enumerate(header[1:])
    }
    states = tuple(
        NewSessionState(
            name=state.name,
            initial_value=state.initial_value,
            rhs=state.rhs,
            observed_column=measurement_columns.get(state.name, state.observed_column),
        )
        for state in spec.states
    )
    observables = tuple(
        NewSessionObservable(
            name=observable.name,
            expression=observable.expression,
            observed_column=measurement_columns.get(
                observable.name,
                observable.observed_column,
            ),
        )
        for observable in spec.observables
    )
    auxiliary_columns = tuple(
        NewSessionAuxiliaryColumn(
            name=column.name,
            observed_column=measurement_columns.get(column.name, column.observed_column),
            kind=column.kind,
            target=column.target,
        )
        for column in spec.auxiliary_columns
    )
    return NewSessionSpec(
        missing_inputs=spec.missing_inputs,
        review=spec.review,
        filename_data=spec.filename_data,
        parameters=spec.parameters,
        fixed_parameters=spec.fixed_parameters,
        states=states,
        helper_functions=spec.helper_functions,
        observables=observables,
        auxiliary_columns=auxiliary_columns,
        loss_body=spec.loss_body,
        user_info_txt=spec.user_info_txt,
        experiments=spec.experiments,
    )


def _read_csv_header(path: Path) -> list[str] | None:
    try:
        with path.open(newline="", encoding="utf-8") as handle:
            return next(csv.reader(handle), None)
    except UnicodeDecodeError:
        return None


def _clear_previous_new_outputs(session_dir: Path) -> None:
    targets = [
        session_dir / "inputs" / "user_input.yaml",
        session_dir / "generated" / "user_model.py",
        session_dir / "generated" / "generated_script.py",
        session_dir / "generated" / "pfit_new_review.txt",
        session_dir / "generated" / "user_input_check.txt",
        session_dir / "generated" / "agent_logs",
    ]
    for target in targets:
        if target.is_dir():
            shutil.rmtree(target)
        elif target.exists():
            target.unlink()


def _complete_with_log(
    llm_client: LLMClient,
    generated_dir: Path,
    step: str,
    messages: list[Message],
    *,
    temperature: float,
    max_tokens: int,
) -> str:
    log_dir = generated_dir / "agent_logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    try:
        response = llm_client.complete(
            messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
    except LLMError as exc:
        if exc.partial_response:
            (log_dir / f"partial_{step}.txt").write_text(exc.partial_response)
        raise
    with (log_dir / "llm_calls.jsonl").open("a") as handle:
        handle.write(
            json.dumps(
                {
                    "step": step,
                    "messages": [message.to_dict() for message in messages],
                    "response": response,
                }
            )
        )
        handle.write("\n")
    return response


def _parse_new_session_response(response: str, *, validate: bool = True) -> NewSessionSpec:
    data = parse_llm_json_object(response, "pfit-new")

    missing = data.get("missing_inputs", [])
    if not isinstance(missing, list) or not all(isinstance(item, str) for item in missing):
        raise ValidationError("pfit-new missing_inputs must be a list of strings")

    missing_inputs = tuple(item.strip() for item in missing if item.strip())
    if missing_inputs:
        return NewSessionSpec(
            missing_inputs=missing_inputs,
            review=_require_string(data, "review", allow_empty=True),
            filename_data="",
            parameters=(),
            fixed_parameters=(),
            states=(),
            helper_functions=(),
            observables=(),
            auxiliary_columns=(),
            loss_body="",
            user_info_txt=_require_string(data, "user_info_txt", allow_empty=True),
        )

    parameters = tuple(_parse_parameter(item) for item in _require_list(data, "parameters"))
    fixed_parameters = tuple(
        _parse_fixed_parameter(item)
        for item in data.get("fixed_parameters", [])
    )
    states = _normalize_observed_columns(
        tuple(_parse_state(item) for item in _require_list(data, "states"))
    )
    helper_functions = tuple(_parse_helper_function(item) for item in data.get("helper_functions", []))
    observables = _normalize_observable_columns(
        tuple(_parse_observable(item) for item in data.get("observables", []))
    )
    auxiliary_columns = tuple(
        _parse_auxiliary_column(item) for item in data.get("auxiliary_columns", [])
    )
    spec = NewSessionSpec(
        missing_inputs=(),
        review=_require_string(data, "review", allow_empty=True),
        filename_data=_parse_experiment_selection(data)[0]["data_file"],
        experiments=_parse_experiment_selection(data),
        parameters=parameters,
        fixed_parameters=fixed_parameters,
        states=states,
        helper_functions=helper_functions,
        observables=observables,
        auxiliary_columns=auxiliary_columns,
        loss_body=_require_string(data, "loss_body", allow_empty=True),
        user_info_txt=_require_string(data, "user_info_txt", allow_empty=True),
    )
    if validate:
        _validate_new_session_spec(spec)
    return spec


def _require_string(data: dict[str, object], key: str, *, allow_empty: bool) -> str:
    value = data.get(key, "")
    if not isinstance(value, str):
        raise ValidationError(f"pfit-new {key} must be a string")
    if not allow_empty and not value.strip():
        raise ValidationError(f"pfit-new response is missing {key}")
    return value


def _require_list(data: dict[str, object], key: str) -> list[object]:
    value = data.get(key)
    if not isinstance(value, list):
        raise ValidationError(f"pfit-new {key} must be a list")
    if not value:
        raise ValidationError(f"pfit-new {key} must not be empty")
    return value


def _parse_parameter(value: object) -> NewSessionParameter:
    if not isinstance(value, dict):
        raise ValidationError("pfit-new parameters entries must be objects")
    return NewSessionParameter(
        name=_clean_identifier(_require_string(value, "name", allow_empty=False), "parameter"),
        min_value=float(value["min_value"]),
        max_value=float(value["max_value"]),
        logscale=_require_bool(value, "logscale"),
    )


def _parse_fixed_parameter(value: object) -> NewSessionFixedParameter:
    if not isinstance(value, dict):
        raise ValidationError("pfit-new fixed_parameters entries must be objects")
    return NewSessionFixedParameter(
        name=_clean_identifier(_require_string(value, "name", allow_empty=False), "fixed parameter"),
        value=float(value["value"]),
    )


def _parse_state(value: object) -> NewSessionState:
    if not isinstance(value, dict):
        raise ValidationError("pfit-new states entries must be objects")
    observed_column = value.get("observed_column")
    return NewSessionState(
        name=_clean_identifier(_require_string(value, "name", allow_empty=False), "state"),
        initial_value=float(value["initial_value"]),
        rhs=_require_string(value, "rhs", allow_empty=False).replace("^", "**"),
        observed_column=(
            None
            if observed_column is None or int(observed_column) < 0
            else int(observed_column)
        ),
    )


def _parse_observable(value: object) -> NewSessionObservable:
    if not isinstance(value, dict):
        raise ValidationError("pfit-new observables entries must be objects")
    return NewSessionObservable(
        name=_clean_identifier(_require_string(value, "name", allow_empty=False), "observable"),
        expression=_require_string(value, "expression", allow_empty=False).replace("^", "**"),
        observed_column=int(value["observed_column"]),
    )


def _parse_auxiliary_column(value: object) -> NewSessionAuxiliaryColumn:
    if not isinstance(value, dict):
        raise ValidationError("pfit-new auxiliary_columns entries must be objects")
    return NewSessionAuxiliaryColumn(
        name=_clean_identifier(_require_string(value, "name", allow_empty=False), "auxiliary column"),
        observed_column=int(value["observed_column"]),
        kind=_require_string(value, "kind", allow_empty=False),
        target=_clean_identifier(_require_string(value, "target", allow_empty=False), "auxiliary target"),
    )


def _parse_helper_function(value: object) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValidationError("pfit-new helper_functions entries must be non-empty strings")
    source = value.strip()
    try:
        module_ast = ast.parse(source)
    except SyntaxError as exc:
        raise ValidationError(f"pfit-new helper function has invalid syntax: {exc}") from exc
    functions = [node for node in module_ast.body if isinstance(node, ast.FunctionDef)]
    if len(functions) != 1 or len(module_ast.body) != 1:
        raise ValidationError("pfit-new helper_functions entries must each define one function")
    if not _function_returns_value(functions[0]):
        raise ValidationError(f"pfit-new helper function must return a value: {functions[0].name}")
    for node in ast.walk(module_ast):
        if isinstance(node, (ast.Import, ast.ImportFrom, ast.ClassDef)):
            raise ValidationError("pfit-new helper functions must not import or define classes")
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "print":
            raise ValidationError("pfit-new helper functions must not print")
    _validate_helper_function_names(functions[0])
    return source


def _require_bool(data: dict[str, object], key: str) -> bool:
    value = data.get(key)
    if isinstance(value, bool):
        return value
    if isinstance(value, str) and value.lower() in {"true", "false"}:
        return value.lower() == "true"
    raise ValidationError(f"pfit-new {key} must be a boolean")


def _clean_identifier(name: str, label: str) -> str:
    name = name.strip()
    if not name.isidentifier():
        raise ValidationError(f"pfit-new {label} name is not a Python identifier: {name}")
    return name


def _normalize_observed_columns(states: tuple[NewSessionState, ...]) -> tuple[NewSessionState, ...]:
    columns = [state.observed_column for state in states if state.observed_column is not None]
    if not columns:
        return states
    if min(columns) < 0:
        raise ValidationError("pfit-new observed_column values must be non-negative")
    if 0 not in columns and sorted(columns) == list(range(1, len(columns) + 1)):
        return tuple(
            NewSessionState(
                name=state.name,
                initial_value=state.initial_value,
                rhs=state.rhs,
                observed_column=(
                    None if state.observed_column is None else state.observed_column - 1
                ),
            )
            for state in states
        )
    return states


def _normalize_observable_columns(
    observables: tuple[NewSessionObservable, ...],
) -> tuple[NewSessionObservable, ...]:
    if not observables:
        return observables
    columns = [observable.observed_column for observable in observables]
    if min(columns) < 0:
        raise ValidationError("pfit-new observable observed_column values must be non-negative")
    if 0 not in columns and sorted(columns) == list(range(1, len(columns) + 1)):
        return tuple(
            NewSessionObservable(
                name=observable.name,
                expression=observable.expression,
                observed_column=observable.observed_column - 1,
            )
            for observable in observables
        )
    return observables


def _validate_new_session_spec(spec: NewSessionSpec) -> None:
    if not spec.filename_data.endswith(".csv"):
        raise ValidationError("pfit-new filename_data must name a CSV file")
    if not spec.parameters:
        raise ValidationError("pfit-new requires at least one trainable parameter")
    if not spec.states:
        raise ValidationError("pfit-new requires at least one integrated state")
    if not any(state.observed_column is not None for state in spec.states) and not spec.observables:
        raise ValidationError("pfit-new requires at least one observed state or observable")

    parameter_names = {parameter.name for parameter in spec.parameters}
    fixed_parameter_names = {parameter.name for parameter in spec.fixed_parameters}
    state_names = {state.name for state in spec.states}
    observable_names = {observable.name for observable in spec.observables}
    helper_names = {_helper_function_name(source) for source in spec.helper_functions}
    auxiliary_names = {column.name for column in spec.auxiliary_columns}
    all_names = parameter_names | fixed_parameter_names | state_names | observable_names
    if len(parameter_names) != len(spec.parameters):
        raise ValidationError("pfit-new parameter names must be unique")
    if len(fixed_parameter_names) != len(spec.fixed_parameters):
        raise ValidationError("pfit-new fixed parameter names must be unique")
    if len(state_names) != len(spec.states):
        raise ValidationError("pfit-new state names must be unique")
    if len(observable_names) != len(spec.observables):
        raise ValidationError("pfit-new observable names must be unique")
    if len(helper_names) != len(spec.helper_functions):
        raise ValidationError("pfit-new helper function names must be unique")
    if len(auxiliary_names) != len(spec.auxiliary_columns):
        raise ValidationError("pfit-new auxiliary column names must be unique")
    if len(all_names) != (
        len(spec.parameters)
        + len(spec.fixed_parameters)
        + len(spec.states)
        + len(spec.observables)
    ):
        raise ValidationError("pfit-new model names must be unique")
    if auxiliary_names & all_names:
        raise ValidationError("pfit-new auxiliary column names must not duplicate model names")
    observed_columns = {
        state.observed_column for state in spec.states if state.observed_column is not None
    } | {observable.observed_column for observable in spec.observables}
    for column in spec.auxiliary_columns:
        if column.observed_column in observed_columns:
            raise ValidationError(
                f"pfit-new auxiliary column {column.name} reuses an observed data column"
            )
        if column.kind == "uncertainty_of" and column.target not in state_names | observable_names:
            raise ValidationError(
                f"pfit-new auxiliary column {column.name} targets unknown quantity: {column.target}"
            )

    allowed_names = parameter_names | fixed_parameter_names | state_names | helper_names | {"t", "np"}
    for state in spec.states:
        _validate_expression(state.rhs, allowed_names, helper_names)
    observable_allowed = parameter_names | fixed_parameter_names | state_names | helper_names | {"np"}
    for observable in spec.observables:
        _validate_expression(observable.expression, observable_allowed, helper_names)
    if spec.loss_body:
        _validate_loss_body(spec.loss_body, helper_names | {"_observables"})


def _helper_function_name(source: str) -> str:
    module_ast = ast.parse(source)
    return next(node.name for node in module_ast.body if isinstance(node, ast.FunctionDef))


def _validate_helper_function_names(function: ast.FunctionDef) -> None:
    assigned_names = {arg.arg for arg in function.args.args}
    assigned_names |= {
        target.id
        for node in ast.walk(function)
        for target in getattr(node, "targets", [])
        if isinstance(target, ast.Name)
    }
    allowed_names = assigned_names | {"np", "float"}
    for node in ast.walk(function):
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
            if node.id in _NUMPY_FUNCTIONS:
                continue
            if node.id not in allowed_names:
                raise ValidationError(
                    f"pfit-new helper function {function.name} uses unknown name: {node.id}"
                )
        if isinstance(node, ast.Call):
            np_call = (
                isinstance(node.func, ast.Attribute)
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id == "np"
            )
            builtin_call = isinstance(node.func, ast.Name) and node.func.id in {"abs", "float"}
            if not (np_call or builtin_call):
                raise ValidationError(
                    f"pfit-new helper function {function.name} may only call np.* functions"
                )


def _validate_expression(
    expression: str,
    allowed_names: set[str],
    allowed_functions: set[str] | None = None,
) -> None:
    allowed_functions = allowed_functions or set()
    try:
        expression_ast = ast.parse(expression, mode="eval")
    except SyntaxError as exc:
        raise ValidationError(f"pfit-new expression has invalid syntax: {expression}") from exc

    for node in ast.walk(expression_ast):
        if isinstance(node, ast.IfExp):
            raise ValidationError(
                "pfit-new expression uses a Python conditional expression; "
                "use a smooth np.* switch or np.where-style expression"
            )
        if isinstance(node, ast.BoolOp):
            raise ValidationError(
                "pfit-new expression uses Python and/or; use np.logical_and/np.logical_or "
                "or a smooth np.* switch"
            )
        if isinstance(node, ast.Name) and node.id not in allowed_names:
            raise ValidationError(f"pfit-new expression uses unknown name: {node.id}")
        if isinstance(node, ast.Call):
            np_call = (
                isinstance(node.func, ast.Attribute)
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id == "np"
            )
            helper_call = isinstance(node.func, ast.Name) and node.func.id in allowed_functions
            builtin_call = isinstance(node.func, ast.Name) and node.func.id == "abs"
            if not (np_call or helper_call or builtin_call):
                raise ValidationError("pfit-new expressions may only call np.* functions")
        if isinstance(node, (ast.Subscript, ast.Lambda, ast.Dict, ast.ListComp, ast.GeneratorExp)):
            raise ValidationError("pfit-new expressions must be scalar formulas")


def _validate_loss_body(loss_body: str, allowed_functions: set[str]) -> None:
    try:
        module_ast = ast.parse("def _loss():\n" + _indent_body(loss_body))
    except SyntaxError as exc:
        raise ValidationError(f"pfit-new loss_body has invalid syntax: {exc}") from exc
    function = module_ast.body[0]
    if not isinstance(function, ast.FunctionDef) or not _function_returns_value(function):
        raise ValidationError("pfit-new loss_body must return a value")
    assigned_names = {
        target.id
        for node in ast.walk(function)
        for target in getattr(node, "targets", [])
        if isinstance(target, ast.Name)
    }
    assigned_names |= {
        node.target.id
        for node in ast.walk(function)
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name)
    }
    allowed_names = {
        "solution_time",
        "solution",
        "dataset",
        "trainable_parameters",
        "fixed_parameters",
        "np",
        "float",
    } | allowed_functions | assigned_names
    for node in ast.walk(function):
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
            if node.id not in allowed_names:
                raise ValidationError(f"pfit-new loss_body uses unknown name: {node.id}")


def _validate_loss_body_array_indices(
    loss_body: str,
    *,
    dataset_width: int,
    solution_width: int,
) -> None:
    try:
        module_ast = ast.parse("def _loss():\n" + _indent_body(loss_body))
    except SyntaxError:
        return
    for node in ast.walk(module_ast):
        if not isinstance(node, ast.Subscript) or not isinstance(node.value, ast.Name):
            continue
        if node.value.id not in {"dataset", "solution"}:
            continue
        column_index = _second_axis_constant_index(node.slice)
        if column_index is None:
            continue
        width = dataset_width if node.value.id == "dataset" else solution_width
        if not -width <= column_index < width:
            raise ValidationError(
                f"pfit-new loss_body {node.value.id} column index {column_index} "
                f"is out of bounds for width {width}"
            )


def _second_axis_constant_index(slice_node: ast.AST) -> int | None:
    if not isinstance(slice_node, ast.Tuple) or len(slice_node.elts) < 2:
        return None
    return _constant_int(slice_node.elts[1])


def _constant_int(node: ast.AST) -> int | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, int):
        return node.value
    if (
        isinstance(node, ast.UnaryOp)
        and isinstance(node.op, ast.USub)
        and isinstance(node.operand, ast.Constant)
        and isinstance(node.operand.value, int)
    ):
        return -node.operand.value
    return None


def _indent_body(body: str) -> str:
    return "\n".join(f"    {line}" if line.strip() else line for line in body.splitlines()) + "\n"


def _render_new_session_draft(spec: NewSessionSpec) -> NewSessionDraft:
    return NewSessionDraft(
        missing_inputs=(),
        review=spec.review,
        user_input_yaml=_render_user_input_yaml(spec),
        user_model_py=_render_user_model_from_spec(spec),
        user_info_txt=spec.user_info_txt,
    )


def _render_user_input_yaml(spec: NewSessionSpec) -> str:
    columns = ["      - {name: time}", *_render_experiment_columns(spec)]
    parameters = "\n".join(
        f"    - {{name: {parameter.name}, min_val: {parameter.min_value}, "
        f"max_val: {parameter.max_value}, logscale: {str(parameter.logscale).lower()}}}"
        for parameter in spec.parameters
    )
    fixed_parameters = "\n".join(
        f"    - {{name: {parameter.name}, value: {parameter.value}}}"
        for parameter in spec.fixed_parameters
    )
    states = "\n".join(
        f"    - {{name: {state.name}, init_val: {state.initial_value}}}"
        for state in spec.states
    )
    observables = "\n".join(
        f"    - {{name: {observable.name}}}"
        for observable in spec.observables
    )
    rtol = "[" + ", ".join("1e-7" for _ in spec.states) + "]"
    atol = "[" + ", ".join("1e-9" for _ in spec.states) + "]"
    integrator = _select_integrator(spec)
    experiments = spec.experiments or ({"data_file": spec.filename_data, "initial_conditions": {}},)
    experiment_lines = []
    for experiment in experiments:
        experiment_lines.extend([
            f"  - data_file: {json.dumps(experiment['data_file'])}",
            f"    initial_conditions: {json.dumps(experiment['initial_conditions'])}",
            "    columns:", *columns,
        ])
    return f"""experiments:
{chr(10).join(experiment_lines)}

model:
  trainable_parameters:
{parameters}
  fixed_parameters:{chr(10) + fixed_parameters if fixed_parameters else " []"}
  integrated_variables:
{states}
  observables:{chr(10) + observables if observables else " []"}

population_opt:
  population_size: 16
  num_iters: 5
  processors: 1
  algorithm: DE

gradient_opt:
  num_iters: 5
  stepsize_rtol: {rtol}
  stepsize_atol: {atol}
  initial_timestep: 1e-6
  max_steps: 10000
  integrator: {integrator}
  init_value_lr: 1e-4
  end_value_lr: 1e-5
  transition_steps_lr: 2000
  decay_rate_lr: 0.9

output:
  write_results: true
"""


def _select_integrator(spec: NewSessionSpec) -> str:
    text = " ".join(
        [
            spec.review,
            spec.user_info_txt,
            spec.filename_data,
            " ".join(state.rhs for state in spec.states),
            " ".join(parameter.name for parameter in spec.parameters),
        ]
    ).lower()
    if _text_requests_stiff_integrator(text):
        return "Kvaerno5"
    for parameter in spec.parameters:
        lower = max(abs(parameter.min_value), 1e-300)
        upper = max(abs(parameter.max_value), 1e-300)
        if upper / lower >= 1e8:
            return "Kvaerno5"
    return "Tsit5"


def _text_requests_stiff_integrator(text: str) -> bool:
    lower = text.lower()
    stiff_markers = (
        "stiff",
        "stiffness",
        "fast-slow",
        "fast slow",
        "chemical kinetics",
        "combustion",
        "reaction network",
    )
    return any(marker in lower for marker in stiff_markers)


def _render_experiment_columns(spec: NewSessionSpec) -> list[str]:
    columns: dict[int, str] = {}
    for state in spec.states:
        if state.observed_column is not None:
            columns[state.observed_column] = f"      - {{name: {state.name}, observes: {state.name}}}"
    for observable in spec.observables:
        columns[observable.observed_column] = (
            f"      - {{name: {observable.name}, observes: {observable.name}}}"
        )
    for column in spec.auxiliary_columns:
        if column.kind == "uncertainty_of":
            columns[column.observed_column] = (
                f"      - {{name: {column.name}, uncertainty_of: {column.target}}}"
            )
        else:
            columns[column.observed_column] = f"      - {{name: {column.name}}}"
    return [columns[index] for index in sorted(columns)]


def _render_user_model_from_spec(spec: NewSessionSpec) -> str:
    parameter_bindings = "\n".join(
        f"    {parameter.name} = trainable_parameters['{parameter.name}']"
        for parameter in spec.parameters
    )
    fixed_bindings = "\n".join(
        f"    {parameter.name} = fixed_parameters['{parameter.name}']"
        for parameter in spec.fixed_parameters
    )
    state_bindings = "\n".join(
        f"    {state.name} = y[{index}]"
        for index, state in enumerate(spec.states)
    )
    derivative_lines = "\n".join(
        f"    d{state.name}dt = {state.rhs}"
        for state in spec.states
    )
    derivative_array = ", ".join(f"d{state.name}dt" for state in spec.states)
    helpers = "\n\n".join(spec.helper_functions)
    observables_function = _render_observables_function(spec)
    loss_body = spec.loss_body or _default_loss_body(spec)
    writeout_body = _default_writeout_body(spec)
    prefix = "import numpy as np\n\n"
    if helpers:
        prefix += helpers + "\n\n"
    if observables_function:
        prefix += observables_function + "\n\n"
    return f"""import numpy as np
{chr(10) + helpers + chr(10) if helpers else ""}
{observables_function + chr(10) if observables_function else ""}

def user_defined_system(t, y, trainable_parameters, fixed_parameters, dataset, t_eval):
{parameter_bindings}
{fixed_bindings}
{state_bindings}
{derivative_lines}
    return np.array([{derivative_array}])

def _compute_loss_problem(solution_time, solution, dataset, trainable_parameters, fixed_parameters):
{_indent_body(loss_body).rstrip()}

def writeout_description(solution_time, solution, dataset, trainable_parameters, fixed_parameters):
{_indent_body(writeout_body).rstrip()}
"""


def _render_observables_function(spec: NewSessionSpec) -> str:
    if not spec.observables:
        return ""
    parameter_bindings = "\n".join(
        f"    {parameter.name} = trainable_parameters['{parameter.name}']"
        for parameter in spec.parameters
    )
    fixed_bindings = "\n".join(
        f"    {parameter.name} = fixed_parameters['{parameter.name}']"
        for parameter in spec.fixed_parameters
    )
    state_bindings = "\n".join(
        f"    {state.name} = solution[:, {index}]"
        for index, state in enumerate(spec.states)
    )
    entries = "\n".join(
        f"        '{observable.name}': {observable.expression},"
        for observable in spec.observables
    )
    return f"""def _observables(solution, trainable_parameters, fixed_parameters):
{parameter_bindings}
{fixed_bindings}
{state_bindings}
    return {{
{entries}
    }}"""


def _default_loss_body(spec: NewSessionSpec) -> str:
    lines = []
    if spec.observables:
        lines.append("observables = _observables(solution, trainable_parameters, fixed_parameters)")
    lines.append("residuals = np.column_stack((")
    for index, state in enumerate(spec.states):
        if state.observed_column is not None:
            lines.append(f"    solution[:, {index}] - dataset[:, {state.observed_column}],")
    for observable in spec.observables:
        lines.append(f"    observables['{observable.name}'] - dataset[:, {observable.observed_column}],")
    lines.extend([
        "))",
        "return float(np.mean(np.square(residuals)))",
    ])
    return "\n".join(lines)


def _default_writeout_body(spec: NewSessionSpec) -> str:
    measured = [
        (state.name, state.observed_column)
        for state in spec.states
        if state.observed_column is not None
    ] + [
        (observable.name, observable.observed_column) for observable in spec.observables
    ]
    simulated_state_count = len(spec.states)
    total_columns = 1 + len(measured) + simulated_state_count + len(spec.observables)
    lines = [
        f"writeout_array = np.zeros([solution_time.shape[0], {total_columns}])",
        "writeout_array[:, 0] = solution_time",
    ]
    column = 1
    for _, observed_column in measured:
        lines.append(f"writeout_array[:, {column}] = dataset[:, {observed_column}]")
        column += 1
    for index, _state in enumerate(spec.states):
        lines.append(f"writeout_array[:, {column}] = solution[:, {index}]")
        column += 1
    if spec.observables:
        lines.append("observables = _observables(solution, trainable_parameters, fixed_parameters)")
        for observable in spec.observables:
            lines.append(f"writeout_array[:, {column}] = observables['{observable.name}']")
            column += 1
    lines.append("return writeout_array")
    return "\n".join(lines)


def _observed_column_names(spec: NewSessionSpec) -> list[str]:
    by_column: dict[int, str] = {}
    for state in spec.states:
        if state.observed_column is not None:
            if state.observed_column in by_column:
                raise ValidationError(
                    f"pfit-new observed column {state.observed_column} is assigned more than once"
                )
            by_column[state.observed_column] = state.name
    for observable in spec.observables:
        if observable.observed_column in by_column:
            raise ValidationError(
                f"pfit-new observed column {observable.observed_column} is assigned more than once"
            )
        by_column[observable.observed_column] = observable.name
    if not by_column:
        return []
    return [by_column[index] for index in sorted(by_column)]


def _validate_draft(session_dir: Path, draft: NewSessionDraft) -> None:
    _validate_user_model_source(draft.user_model_py)

    with tempfile.TemporaryDirectory() as tmpdir:
        temp_session = Path(tmpdir) / "session"
        temp_inputs = temp_session / "inputs"
        temp_generated = temp_session / "generated"
        temp_inputs.mkdir(parents=True)
        temp_generated.mkdir()
        temp_inputs.joinpath("user_input.yaml").write_text(draft.user_input_yaml)
        temp_generated.joinpath("user_model.py").write_text(draft.user_model_py)

        source_inputs = Path(session_dir) / "inputs"
        if source_inputs.exists():
            for source in source_inputs.iterdir():
                if source.is_file() and source.name != "user_input.yaml":
                    shutil.copy2(source, temp_inputs / source.name)

        validate_session(temp_session)


def _validate_user_model_source(source: str) -> None:
    if "```" in source:
        raise ValidationError("pfit-new user_model_py contains Markdown fences")
    placeholders = ("TODO", "pass", "Define each derivative")
    for placeholder in placeholders:
        if placeholder in source:
            raise ValidationError(f"pfit-new user_model_py still contains placeholder: {placeholder}")

    try:
        module_ast = ast.parse(source)
    except SyntaxError as exc:
        raise ValidationError(f"pfit-new user_model_py has invalid syntax: {exc}") from exc

    functions = {
        node.name: node
        for node in module_ast.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    required = {
        "user_defined_system": (
            "t",
            "y",
            "trainable_parameters",
            "fixed_parameters",
            "dataset",
            "t_eval",
        ),
        "_compute_loss_problem": (
            "solution_time",
            "solution",
            "dataset",
            "trainable_parameters",
            "fixed_parameters",
        ),
        "writeout_description": (
            "solution_time",
            "solution",
            "dataset",
            "trainable_parameters",
            "fixed_parameters",
        ),
    }
    missing = sorted(set(required) - set(functions))
    if missing:
        raise ValidationError(
            "pfit-new user_model_py is missing required functions: "
            + ", ".join(missing)
        )
    for name, expected_args in required.items():
        function_node = functions[name]
        actual_args = tuple(arg.arg for arg in function_node.args.args)
        if actual_args != expected_args:
            raise ValidationError(
                f"pfit-new {name}() has signature {actual_args}, "
                f"expected {expected_args}"
            )
        if not _function_returns_value(function_node):
            raise ValidationError(f"pfit-new {name}() must return a value")

    banned_terms = ("scipy", "solve_ivp", "open(", "subprocess", "requests", "urllib")
    for term in banned_terms:
        if term in source:
            raise ValidationError(f"pfit-new user_model_py contains unsupported term: {term}")

    for node in ast.walk(module_ast):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            if node.func.id == "print":
                raise ValidationError("pfit-new user_model_py must not print from contract functions")
        if isinstance(node, ast.Subscript) and isinstance(node.value, ast.Name):
            if node.value.id == "trainable_parameters" and isinstance(node.slice, ast.Constant):
                if isinstance(node.slice.value, int):
                    raise ValidationError(
                        "pfit-new user_model_py must access trainable_parameters by name, not index"
                    )
            if node.value.id == "dataset" and isinstance(node.slice, ast.Constant):
                if isinstance(node.slice.value, str):
                    raise ValidationError(
                        "pfit-new user_model_py must treat dataset as an array, not a dataframe"
                    )


def _function_returns_value(function_node: ast.FunctionDef | ast.AsyncFunctionDef) -> bool:
    return any(
        isinstance(node, ast.Return) and node.value is not None
        for node in ast.walk(function_node)
    )
