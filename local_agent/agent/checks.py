from dataclasses import dataclass, field
import ast
import csv
import json
from pathlib import Path

import numpy as np
import yaml

from local_agent.agent.config import WorkflowConfig
from local_agent.agent.llm_json import parse_llm_json_object
from local_agent.agent.prompts import PromptRenderer
from local_agent.agent.session_spec import load_session_spec
from local_agent.agent.validators import ValidationError, parse_input_yaml, validate_session
from local_agent.llm.base import LLMClient, LLMError, Message


@dataclass
class CheckReport:
    critical_errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    recommendations: list[str] = field(default_factory=list)
    semantic_review: str = ""

    @property
    def passed(self) -> bool:
        return not self.critical_errors

    def to_text(self) -> str:
        critical_errors = [f"- {item}" for item in self.critical_errors] or ["- none"]
        warnings = [f"- {item}" for item in self.warnings] or ["- none"]
        recommendations = [f"- {item}" for item in self.recommendations] or ["- none"]
        return "\n".join(
            [
                "Critical errors:",
                *critical_errors,
                "",
                "Warnings:",
                *warnings,
                "",
                "Recommendations:",
                *recommendations,
                "",
                "Semantic review:",
                self.semantic_review or "- not run",
                "",
            ]
        )


def check_session(
    session_dir: Path,
    llm_client: LLMClient | None = None,
    prompt_renderer: PromptRenderer | None = None,
    workflow_config: WorkflowConfig | None = None,
) -> CheckReport:
    report = CheckReport()
    session_dir = Path(session_dir)
    try:
        result = validate_session(session_dir)
    except ValidationError as exc:
        report.critical_errors.append(str(exc))
        return report

    user_model = session_dir / "generated" / "user_model.py"
    if not user_model.exists():
        report.critical_errors.append(f"User model not found: {user_model}")
        return report
    _add_branchy_dynamics_checks(user_model, report)
    _add_uncertainty_loss_checks(session_dir, user_model, report)
    _add_dataset_scale_loss_checks(session_dir, user_model, report)
    _add_user_loss_contract_checks(session_dir, user_model, report)

    if result.n_trainable_parameters > 5:
        report.recommendations.append(
            "Large trainable parameter count detected; consider increasing population search budget."
        )
    if result.dataset_shape[0] < 5:
        report.warnings.append(
            "Dataset has very few rows; fitted parameters may be weakly constrained."
        )
    report.recommendations.append(
        "Review generated/user_model.py and parameter bounds before JAX translation."
    )
    if llm_client is not None:
        _add_semantic_review(
            session_dir,
            report,
            llm_client,
            prompt_renderer or PromptRenderer(),
            workflow_config or WorkflowConfig(),
        )
    return report


def write_check_report(session_dir: Path, report: CheckReport) -> Path:
    report_path = Path(session_dir) / "generated" / "user_input_check.txt"
    report_path.parent.mkdir(exist_ok=True)
    report_path.write_text(report.to_text())
    return report_path


def _add_semantic_review(
    session_dir: Path,
    report: CheckReport,
    llm_client: LLMClient,
    prompt_renderer: PromptRenderer,
    workflow_config: WorkflowConfig,
) -> None:
    generated_dir = session_dir / "generated"
    context = _build_check_context(session_dir, report)
    messages = prompt_renderer.render_messages(
        "check_session.system.md",
        "check_session.user.md",
        context,
    )
    response = _complete_with_log(
        llm_client,
        generated_dir,
        "semantic_check",
        messages,
        temperature=workflow_config.temperature,
        max_tokens=workflow_config.max_tokens,
    )
    semantic = _parse_semantic_check_response(response)
    # Model prose is not a validation result. Preserve findings for review without
    # letting an unverified claim override the deterministic checks above.
    report.warnings.extend(
        f"Unverified semantic finding: {finding}" for finding in semantic.critical_errors
    )
    report.warnings.extend(semantic.warnings)
    report.recommendations.extend(semantic.recommendations)
    report.semantic_review = semantic.review


def _build_check_context(session_dir: Path, report: CheckReport) -> dict[str, object]:
    input_yaml = session_dir / "inputs" / "user_input.yaml"
    user_model = session_dir / "generated" / "user_model.py"
    dataset_summary = _dataset_summary(session_dir, input_yaml)
    user_info = session_dir / "inputs" / "user_info.txt"
    return {
        "input_yaml": input_yaml.read_text(),
        "user_model": user_model.read_text(),
        "session_summary": load_session_spec(input_yaml).to_prompt_text(),
        "loss_evidence": _loss_expression_evidence(user_model.read_text()),
        "dataset_summary": dataset_summary,
        "deterministic_report": report.to_text(),
        "user_loss_contract": (
            _extract_loss_contract_text(user_info.read_text()) if user_info.exists() else ""
        ) or "No explicit loss description provided.",
    }


@dataclass(frozen=True)
class SemanticCheck:
    critical_errors: tuple[str, ...]
    warnings: tuple[str, ...]
    recommendations: tuple[str, ...]
    review: str


def _parse_semantic_check_response(response: str) -> SemanticCheck:
    data = parse_llm_json_object(response, "pfit-check")
    return SemanticCheck(
        critical_errors=tuple(_string_list(data, "critical_errors")),
        warnings=tuple(_string_list(data, "warnings")),
        recommendations=tuple(_string_list(data, "recommendations")),
        review=_optional_string(data, "review"),
    )


def _string_list(data: dict[str, object], key: str) -> list[str]:
    value = data.get(key, [])
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise ValidationError(f"pfit-check {key} must be a list of strings")
    return [item.strip() for item in value if item.strip()]


def _optional_string(data: dict[str, object], key: str) -> str:
    value = data.get(key, "")
    if not isinstance(value, str):
        raise ValidationError(f"pfit-check {key} must be a string")
    return value.strip()


def _dataset_summary(session_dir: Path, input_yaml: Path) -> str:
    validation = validate_session(session_dir)
    dataset_path = validation.dataset_path
    reader = parse_input_yaml(input_yaml)
    dataset = _load_numeric_dataset(dataset_path)
    rows: list[list[str]] = []
    with dataset_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.reader(handle)
        for index, row in enumerate(reader):
            if index >= 6:
                break
            rows.append(row)
    preview = "\n".join(",".join(row) for row in rows)
    column_stats = _dataset_column_stats(dataset, parse_input_yaml(input_yaml).data_column_names)
    return "\n".join(
        [
            f"Dataset path: {dataset_path}",
            f"Shape: {validation.dataset_shape[0]} rows x {validation.dataset_shape[1]} columns",
            "Column scale summary:",
            *column_stats,
            "CSV preview:",
            preview,
        ]
    )


def _add_dataset_scale_loss_checks(
    session_dir: Path,
    user_model: Path,
    report: CheckReport,
) -> None:
    input_yaml = session_dir / "inputs" / "user_input.yaml"
    validation = validate_session(session_dir)
    reader = parse_input_yaml(input_yaml)
    dataset = _load_numeric_dataset(validation.dataset_path)
    if dataset.shape[1] < 2:
        return
    logged_dataset_columns = _logged_dataset_columns_in_loss(user_model.read_text())
    measured_names = _measured_column_names(reader)
    for dataset_index, values in enumerate(dataset[:, 1:].T):
        orders = _column_log10_range(values)
        if orders is None or orders < 3.0:
            continue
        finite = values[np.isfinite(values)]
        min_value = float(np.min(finite))
        max_value = float(np.max(finite))
        if dataset_index in logged_dataset_columns:
            continue
        name = measured_names[dataset_index] if dataset_index < len(measured_names) else f"dataset[:, {dataset_index}]"
        report.critical_errors.append(
            "Measured column spans orders of magnitude but the loss does not compare it in log space: "
            f"{name} uses dataset[:, {dataset_index}], positive min={min_value:.6g}, "
            f"max={max_value:.6g}, log10 range={orders:.2f}. "
            "Use a log/log10-transformed residual before normalization."
        )


def _add_uncertainty_loss_checks(
    session_dir: Path,
    user_model: Path,
    report: CheckReport,
) -> None:
    columns = _declared_experiment_columns(session_dir / "inputs" / "user_input.yaml")
    if not columns:
        return
    data_columns = columns[1:] if columns and columns[0].get("name", "").lower() == "time" else columns
    loss_columns = _dataset_columns_in_loss(user_model.read_text())
    for dataset_index, column in enumerate(data_columns):
        if "uncertainty_of" not in column:
            continue
        sigma_name = str(column.get("name", "")).strip()
        target = str(column.get("uncertainty_of", "")).strip()
        if dataset_index in loss_columns:
            continue
        report.critical_errors.append(
            f"Measurement uncertainty column {sigma_name} is declared for {target}, "
            f"but _compute_loss_problem does not use dataset[:, {dataset_index}]."
        )


def _add_user_loss_contract_checks(
    session_dir: Path,
    user_model: Path,
    report: CheckReport,
) -> None:
    user_info = session_dir / "inputs" / "user_info.txt"
    if not user_info.exists():
        return
    loss_text = _extract_loss_contract_text(user_info.read_text())
    if not loss_text:
        return
    source = user_model.read_text()
    loss_source = _loss_function_source(source).lower()
    requested = loss_text.lower()

    if _requests_rmse(requested) and "sqrt" not in loss_source:
        report.critical_errors.append(
            "User prompt specifies an RMSE/sqrt loss, but _compute_loss_problem does not take a square root."
        )
    if _requests_log_loss(requested) and "log" not in loss_source:
        report.critical_errors.append(
            "User prompt specifies log/log10 residuals, but _compute_loss_problem does not use a log transform."
        )
    if _requests_normalized_loss(requested) and "/" not in loss_source:
        report.critical_errors.append(
            "User prompt specifies normalized/scaled residuals, but _compute_loss_problem does not divide by a scale."
        )
    if _requests_normalized_loss(requested) and "measured" in requested:
        for expression in _simulated_scale_denominators(source):
            report.critical_errors.append(
                "User prompt specifies measured-data normalization, but the loss denominator "
                f"reduces simulated states or time instead: {expression}"
            )


def _simulated_scale_denominators(source: str) -> list[str]:
    """Recognize wrong-array scale reductions; leave unknown algebra for review."""
    try:
        module = ast.parse(_loss_function_source(source))
    except SyntaxError:
        return []
    function = next((n for n in module.body if isinstance(n, ast.FunctionDef)), None)
    if function is None:
        return []
    bindings: dict[str, ast.AST] = {}

    def expand(node: ast.AST, seen: frozenset[str] = frozenset()):
        yield node
        if isinstance(node, ast.Name) and node.id in bindings and node.id not in seen:
            yield from expand(bindings[node.id], seen | {node.id})
        else:
            for child in ast.iter_child_nodes(node):
                yield from expand(child, seen)

    errors = []
    # Track only straight-line assignments; don't infer branch/loop semantics.
    for statement in function.body:
        if not isinstance(statement, (ast.Assign, ast.AnnAssign, ast.AugAssign, ast.Return)):
            continue
        value = statement.value
        if value is None:
            continue
        for node in ast.walk(value):
            if not isinstance(node, ast.BinOp) or not isinstance(node.op, ast.Div):
                continue
            if not any(isinstance(n, ast.Name) and n.id == "dataset" for n in expand(node.left)):
                continue
            for term in expand(node.right):
                if not isinstance(term, ast.Call) or not isinstance(term.func, ast.Attribute):
                    continue
                if not isinstance(term.func.value, ast.Name) or term.func.value.id not in {"np", "jnp"}:
                    continue
                if term.func.attr not in {"max", "min", "nanmax", "nanmin", "ptp"} or not term.args:
                    continue
                names = {n.id for n in expand(term.args[0]) if isinstance(n, ast.Name)}
                if names & {"solution", "solution_time"} and "dataset" not in names:
                    errors.append(ast.unparse(node.right))
                    break
        if isinstance(statement, ast.Assign):
            for target in statement.targets:
                if isinstance(target, ast.Name):
                    bindings[target.id] = value
        elif isinstance(statement, ast.AnnAssign) and isinstance(statement.target, ast.Name):
            bindings[statement.target.id] = value
    return list(dict.fromkeys(errors))


def _extract_loss_contract_text(text: str) -> str:
    lines = text.splitlines()
    collected: list[str] = []
    in_loss = False
    for raw_line in lines:
        line = raw_line.strip()
        if not line:
            continue
        heading = line.lstrip("#").strip().lower().rstrip(":")
        if heading == "loss" or heading.startswith("loss "):
            in_loss = True
            collected.append(line)
            continue
        if in_loss and line.endswith(":") and not line.startswith(("-", "*")):
            break
        if in_loss:
            collected.append(line)
    if collected:
        return "\n".join(collected)
    lower = text.lower()
    if "loss" not in lower:
        return ""
    return text


def _loss_function_source(source: str) -> str:
    try:
        module = ast.parse(source)
    except SyntaxError:
        return source
    loss_function = next(
        (
            node
            for node in module.body
            if isinstance(node, ast.FunctionDef) and node.name == "_compute_loss_problem"
        ),
        None,
    )
    if loss_function is None:
        return source
    return ast.get_source_segment(source, loss_function) or source


def _loss_expression_evidence(source: str) -> str:
    """Expose syntax facts, without claiming to prove loss equivalence."""
    try:
        module = ast.parse(source)
    except SyntaxError:
        return "Loss expression evidence unavailable: invalid Python syntax."
    function = next(
        (node for node in module.body
         if isinstance(node, ast.FunctionDef) and node.name == "_compute_loss_problem"),
        None,
    )
    if function is None:
        return "No _compute_loss_problem function found."
    facts = []
    for node in ast.walk(function):
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Div):
            facts.append(
                f"- line {node.lineno}: division {ast.get_source_segment(source, node)}; "
                f"denominator: {ast.get_source_segment(source, node.right)}"
            )
    return "\n".join(facts) or "No division expressions found; inspect calls and helpers for normalization."


def _requests_rmse(text: str) -> bool:
    return "rmse" in text or "root mean square" in text or "sqrt(mean" in text


def _requests_log_loss(text: str) -> bool:
    return "log10" in text or "log scale" in text or "log-space" in text or "log residual" in text


def _requests_normalized_loss(text: str) -> bool:
    markers = (
        "normalized",
        "normalised",
        "normalize",
        "normalise",
        "scaled residual",
        "divide residual",
        "residuals by",
    )
    return any(marker in text for marker in markers)


def _add_branchy_dynamics_checks(user_model: Path, report: CheckReport) -> None:
    try:
        module = ast.parse(user_model.read_text())
    except SyntaxError:
        return
    system = next(
        (
            node
            for node in module.body
            if isinstance(node, ast.FunctionDef) and node.name == "user_defined_system"
        ),
        None,
    )
    if system is None:
        return
    for node in ast.walk(system):
        if isinstance(node, ast.If):
            report.critical_errors.append(
                "user_defined_system contains a Python if statement. "
                "Use smooth equations or np.where-style expressions before pfit-jax."
            )
            return
        if isinstance(node, ast.IfExp):
            report.critical_errors.append(
                "user_defined_system contains a Python conditional expression. "
                "Use smooth equations or np.where-style expressions before pfit-jax."
            )
            return


def _declared_experiment_columns(input_yaml: Path) -> list[dict[str, str]]:
    try:
        data = yaml.safe_load(input_yaml.read_text()) or {}
    except Exception:
        return []
    experiments = data.get("experiments", [])
    if not isinstance(experiments, list) or not experiments:
        return []
    columns = experiments[0].get("columns", [])
    if not isinstance(columns, list):
        return []
    normalized = []
    for column in columns:
        if isinstance(column, dict):
            normalized.append({str(key): str(value) for key, value in column.items()})
    return normalized


def _measured_column_names(reader) -> list[str]:
    names = reader.data_column_names
    if names and names[0].strip().lower() == "time":
        return names[1:]
    return names[1:] if len(names) > 1 else names


def _load_numeric_dataset(path: Path) -> np.ndarray:
    with Path(path).open("r", encoding="utf-8-sig") as handle:
        data = np.genfromtxt(handle, dtype=float, delimiter=",", skip_header=1)
    return np.atleast_2d(data)


def _column_log10_range(values: np.ndarray) -> float | None:
    finite = values[np.isfinite(values)]
    if finite.size < 2 or np.any(finite <= 0.0):
        return None
    return float(np.log10(np.max(finite)) - np.log10(np.min(finite)))


def _dataset_column_stats(dataset: np.ndarray, names: list[str]) -> list[str]:
    stats = []
    for index, values in enumerate(dataset.T):
        name = names[index] if index < len(names) else f"column_{index}"
        finite = values[np.isfinite(values)]
        if finite.size == 0:
            stats.append(
                f"- {name}: no finite values, NaN count={int(np.isnan(values).sum())}, "
                f"infinity count={int(np.isinf(values).sum())}"
            )
            continue
        orders = _column_log10_range(values)
        if index == 0:
            suffix = ", time column; automatic log-loss rule does not apply"
        elif orders is None:
            suffix = ", automatic log-loss rule does not apply (fewer than two finite values or contains zero/negative values)"
        else:
            required = "yes" if orders >= 3.0 else "no"
            suffix = (
                f", all finite values positive, within-column log10(max/min)={orders:.6g}"
                f", automatic log-loss required={required} (threshold: 3 orders)"
            )
        stats.append(
            f"- {name}: min={float(np.min(finite)):.6g}, max={float(np.max(finite)):.6g}{suffix}"
            f", NaN count={int(np.isnan(values).sum())}, infinity count={int(np.isinf(values).sum())}"
        )
    return stats


def _logged_dataset_columns_in_loss(source: str) -> set[int]:
    try:
        module = ast.parse(source)
    except SyntaxError:
        return set()
    loss_function = next(
        (
            node
            for node in module.body
            if isinstance(node, ast.FunctionDef) and node.name == "_compute_loss_problem"
        ),
        None,
    )
    if loss_function is None:
        return set()

    aliases: dict[str, set[int]] = {}
    logged_columns: set[int] = set()
    for statement in loss_function.body:
        for node in ast.walk(statement):
            if not isinstance(node, ast.Call) or not _is_log_call(node.func):
                continue
            for arg in node.args:
                logged_columns.update(_dataset_columns_in_node(arg, aliases))
        if isinstance(statement, ast.Assign):
            columns = _dataset_columns_in_node(statement.value, aliases)
            for target in statement.targets:
                if isinstance(target, ast.Name):
                    aliases[target.id] = columns
    return logged_columns


def _dataset_columns_in_loss(source: str) -> set[int]:
    try:
        module = ast.parse(source)
    except SyntaxError:
        return set()
    loss_function = next(
        (
            node
            for node in module.body
            if isinstance(node, ast.FunctionDef) and node.name == "_compute_loss_problem"
        ),
        None,
    )
    if loss_function is None:
        return set()

    aliases: dict[str, set[int]] = {}
    columns: set[int] = set()
    for statement in loss_function.body:
        columns.update(_dataset_columns_in_node(statement, aliases))
        if isinstance(statement, ast.Assign):
            assigned_columns = _dataset_columns_in_node(statement.value, aliases)
            for target in statement.targets:
                if isinstance(target, ast.Name):
                    aliases[target.id] = assigned_columns
    return columns


def _is_log_call(function: ast.AST) -> bool:
    if isinstance(function, ast.Attribute) and function.attr in {"log", "log10"}:
        return True
    return isinstance(function, ast.Name) and function.id in {"log", "log10"}


def _dataset_columns_in_node(node: ast.AST, aliases: dict[str, set[int]]) -> set[int]:
    columns: set[int] = set()
    for child in ast.walk(node):
        if isinstance(child, ast.Name):
            columns.update(aliases.get(child.id, set()))
        if not isinstance(child, ast.Subscript):
            continue
        if not isinstance(child.value, ast.Name) or child.value.id != "dataset":
            continue
        column_index = _dataset_column_index(child.slice)
        if column_index is not None:
            columns.add(column_index)
    return columns


def _dataset_column_index(slice_node: ast.AST) -> int | None:
    if not isinstance(slice_node, ast.Tuple) or len(slice_node.elts) < 2:
        return None
    column = slice_node.elts[1]
    if isinstance(column, ast.Constant) and isinstance(column.value, int):
        return column.value
    return None


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
