import ast
import csv
import json
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path

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
class NewSessionSpec:
    missing_inputs: tuple[str, ...]
    review: str
    filename_data: str
    parameters: tuple[NewSessionParameter, ...]
    fixed_parameters: tuple[NewSessionFixedParameter, ...]
    states: tuple[NewSessionState, ...]
    helper_functions: tuple[str, ...]
    observables: tuple[NewSessionObservable, ...]
    loss_body: str
    user_info_txt: str


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

    context = {"session_context": _collect_session_context(session_dir)}
    if overwrite:
        _clear_previous_new_outputs(session_dir)

    messages = prompt_renderer.render_messages(
        "new_session.system.md",
        "new_session.user.md",
        context,
    )
    response = _complete_with_log(
        llm_client,
        generated,
        "new_session",
        messages,
        temperature=workflow_config.temperature,
        max_tokens=workflow_config.max_tokens,
    )
    for attempt in range(0, workflow_config.max_repair_attempts + 1):
        try:
            spec = _parse_new_session_response(response)
            if spec.missing_inputs:
                missing = "\n".join(f"- {item}" for item in spec.missing_inputs)
                raise ValidationError(f"pfit-new missing required inputs:\n{missing}")
            draft = _render_new_session_draft(spec)
            _validate_draft(session_dir, draft)
            break
        except ValidationError as exc:
            if str(exc).startswith("pfit-new missing required inputs:"):
                raise
            if attempt >= workflow_config.max_repair_attempts:
                raise
            repair_messages = prompt_renderer.render_messages(
                "repair_new_session.system.md",
                "repair_new_session.user.md",
                {
                    **context,
                    "attempt": attempt + 1,
                    "validation_error": str(exc),
                    "previous_response": response,
                },
            )
            response = _complete_with_log(
                llm_client,
                generated,
                "repair_new_session",
                repair_messages,
                temperature=workflow_config.temperature,
                max_tokens=workflow_config.max_tokens,
            )

    inputs = session_dir / "inputs"
    outputs = session_dir / "outputs"
    inputs.mkdir(parents=True, exist_ok=True)
    outputs.mkdir(exist_ok=True)

    written = [
        _write_if_allowed(inputs / "user_input.yaml", draft.user_input_yaml, overwrite),
        _write_if_allowed(inputs / "user_info.txt", draft.user_info_txt, overwrite),
        _write_if_allowed(generated / "user_model.py", draft.user_model_py, overwrite),
    ]
    if draft.review:
        written.append(_write_if_allowed(generated / "pfit_new_review.txt", draft.review, overwrite))
    return [path for path in written if path is not None]


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
    if relative_path.parts[:1] in {("generated",), ("outputs",)}:
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

    preview = rows[: sample_rows + 1]
    total_rows = max(len(rows) - 1, 0)
    return "\n".join(
        [
            f"<csv summary: {total_rows} data rows>",
            "<csv sample>",
            *(",".join(row) for row in preview),
            "</csv sample>",
        ]
    )


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


def _parse_new_session_response(response: str) -> NewSessionSpec:
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
    spec = NewSessionSpec(
        missing_inputs=(),
        review=_require_string(data, "review", allow_empty=True),
        filename_data=_require_string(data, "filename_data", allow_empty=False),
        parameters=parameters,
        fixed_parameters=fixed_parameters,
        states=states,
        helper_functions=helper_functions,
        observables=observables,
        loss_body=_require_string(data, "loss_body", allow_empty=True),
        user_info_txt=_require_string(data, "user_info_txt", allow_empty=True),
    )
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
    return NewSessionState(
        name=_clean_identifier(_require_string(value, "name", allow_empty=False), "state"),
        initial_value=float(value["initial_value"]),
        rhs=_require_string(value, "rhs", allow_empty=False).replace("^", "**"),
        observed_column=(
            None
            if value.get("observed_column") is None
            else int(value["observed_column"])
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
    if len(all_names) != (
        len(spec.parameters)
        + len(spec.fixed_parameters)
        + len(spec.states)
        + len(spec.observables)
    ):
        raise ValidationError("pfit-new model names must be unique")

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
        if isinstance(node, ast.Name) and node.id not in allowed_names:
            raise ValidationError(f"pfit-new expression uses unknown name: {node.id}")
        if isinstance(node, ast.Call):
            np_call = (
                isinstance(node.func, ast.Attribute)
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id == "np"
            )
            helper_call = isinstance(node.func, ast.Name) and node.func.id in allowed_functions
            if not (np_call or helper_call):
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
        "residuals",
        "observables",
        "scale",
        "loss",
    } | allowed_functions | assigned_names
    for node in ast.walk(function):
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
            if node.id not in allowed_names:
                raise ValidationError(f"pfit-new loss_body uses unknown name: {node.id}")


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
    columns = ["      - {name: time}"] + [
        f"      - {{name: {name}, observes: {name}}}"
        for name in _observed_column_names(spec)
    ]
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
    return f"""experiments:
  - data_file: {spec.filename_data}
    columns:
{chr(10).join(columns)}

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
  init_value_lr: 1e-4
  end_value_lr: 1e-5
  transition_steps_lr: 2000
  decay_rate_lr: 0.9

output:
  write_results: true
"""


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
            by_column[state.observed_column] = state.name
    for observable in spec.observables:
        by_column[observable.observed_column] = observable.name
    if not by_column:
        return []
    expected = list(range(max(by_column) + 1))
    if sorted(by_column) != expected:
        raise ValidationError("pfit-new observed columns must be contiguous from zero")
    return [by_column[index] for index in expected]


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
    placeholders = ("TODO", "pass", "Define each derivative", "loss = 0.0")
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
