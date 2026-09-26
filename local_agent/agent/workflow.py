import ast
from dataclasses import dataclass, field
from importlib import resources
import json
from pathlib import Path
import re
import textwrap
import traceback

from local_agent.agent.config import WorkflowConfig
from local_agent.agent.jax_fragments import (
    parse_jax_body_response,
    parse_jax_fragments_response,
    parse_jax_helpers_response,
    parse_jax_rhs_response,
    render_generated_script_from_fragments,
)
from local_agent.agent.llm_json import parse_llm_json_object
from local_agent.agent.prompts import PromptRenderer
from local_agent.agent.session_spec import SessionSpec, load_session_spec
from local_agent.agent.user_model import render_user_model_skeleton
from local_agent.agent.validators import (
    ValidationError,
    smoke_test_generated_script,
    validate_generated_script_contract,
    validate_session,
)
from local_agent.llm.base import LLMClient, LLMError, Message


@dataclass(frozen=True)
class WorkflowEvent:
    step: str
    status: str
    message: str


@dataclass
class WorkflowResult:
    success: bool
    events: list[WorkflowEvent] = field(default_factory=list)


class LocalWorkflow:
    def __init__(
        self,
        llm_client: LLMClient,
        prompt_renderer: PromptRenderer,
        workflow_config: WorkflowConfig | None = None,
    ):
        self.llm_client = llm_client
        self.prompt_renderer = prompt_renderer
        self.workflow_config = workflow_config or WorkflowConfig()

    def validate_only(self, session_dir: Path) -> WorkflowResult:
        events: list[WorkflowEvent] = []
        try:
            validate_session(session_dir)
        except ValidationError as exc:
            events.append(WorkflowEvent("validate_session", "failed", str(exc)))
            return WorkflowResult(False, events)

        events.append(WorkflowEvent("validate_session", "passed", str(session_dir)))
        return WorkflowResult(True, events)

    def generate_script(self, session_dir: Path) -> WorkflowResult:
        session_dir = Path(session_dir)
        events: list[WorkflowEvent] = []

        try:
            session_validation = validate_session(session_dir)
        except ValidationError as exc:
            events.append(WorkflowEvent("validate_session", "failed", str(exc)))
            return WorkflowResult(False, events)

        events.append(WorkflowEvent("validate_session", "passed", str(session_dir)))

        generated_dir = session_dir / "generated"
        generated_dir.mkdir(exist_ok=True)
        script_path = generated_dir / "generated_script.py"

        session_spec = load_session_spec(session_validation.input_yaml)
        user_model_path = generated_dir / "user_model.py"
        if not user_model_path.exists():
            user_model_path.write_text(render_user_model_skeleton(session_spec))
            events.append(
                WorkflowEvent("generate_user_model_skeleton", "created", str(user_model_path))
            )

        from lib.utils.source_stamp import build_stamp
        self._generation_source_stamp = build_stamp(session_dir)
        context = self._build_generation_context(
            session_dir,
            session_validation.input_yaml,
            session_spec,
        )
        try:
            response = self._generate_split_jax_response(
                generated_dir,
                context,
                session_spec,
                data_width=max(session_validation.dataset_shape[1] - 1, 0),
            )
        except ValidationError as exc:
            response = "{}"
            events.append(WorkflowEvent("translate_jax_fragments", "failed", str(exc)))
        else:
            if self._render_and_validate_fragments(
                response,
                session_spec,
                script_path,
                session_dir,
                events,
            ):
                return self._finish(True, events, generated_dir)

        for attempt in range(1, self.workflow_config.max_repair_attempts + 1):
            repair_context = {
                **context,
                "attempt": attempt,
                "validation_error": events[-1].message,
                "previous_response": response,
            }
            repair_messages = self.prompt_renderer.render_messages(
                "repair_jax_fragments.system.md",
                "repair_jax_fragments.user.md",
                repair_context,
            )
            response = self._complete_with_log(
                generated_dir,
                "repair_jax_fragments",
                repair_messages,
                temperature=self.workflow_config.temperature,
                max_tokens=self.workflow_config.max_tokens,
            )
            events.append(
                WorkflowEvent("repair_jax_fragments", "attempted", str(attempt))
            )
            if self._render_and_validate_fragments(
                response,
                session_spec,
                script_path,
                session_dir,
                events,
            ):
                return self._finish(True, events, generated_dir)

        return self._finish(False, events, generated_dir)

    def _generate_split_jax_response(
        self,
        generated_dir: Path,
        context: dict[str, object],
        session_spec: SessionSpec,
        data_width: int | None = None,
    ) -> str:
        helper_response = self._complete_prompt(
            generated_dir,
            "translate_jax_helpers",
            "jax_helpers.system.md",
            "jax_helpers.user.md",
            context,
        )
        helper_functions = list(parse_jax_helpers_response(helper_response))

        rhs_context = {
            **context,
            "translated_helper_functions": _format_prompt_helpers(helper_functions),
        }
        rhs_response = self._complete_prompt(
            generated_dir,
            "translate_jax_rhs",
            "jax_rhs.system.md",
            "jax_rhs.user.md",
            rhs_context,
        )
        rhs_data = parse_llm_json_object(rhs_response, "pfit-jax rhs")
        rhs = parse_jax_rhs_response(json.dumps(rhs_data), session_spec)

        body_context = {
            **context,
            "translated_helper_functions": _format_prompt_helpers(helper_functions),
            "translated_rhs": _format_prompt_rhs(rhs, session_spec),
        }
        standard_bodies = _standard_loss_and_writeout_bodies(
            session_spec,
            data_width=data_width,
            user_model_source=str(context.get("user_model", "")),
        )
        if standard_bodies is not None:
            loss_body, writeout_body = standard_bodies
        else:
            loss_body = _deterministic_custom_loss_body(str(context.get("user_model", "")))
            if loss_body is None:
                loss_response = self._complete_prompt(
                    generated_dir,
                    "translate_jax_loss",
                    "jax_loss.system.md",
                    "jax_loss.user.md",
                    body_context,
                )
                loss_body = parse_jax_body_response(loss_response, "loss_body")

            writeout_context = {
                **body_context,
                "translated_loss_body": loss_body,
            }
            writeout_response = self._complete_prompt(
                generated_dir,
                "translate_jax_writeout",
                "jax_writeout.system.md",
                "jax_writeout.user.md",
                writeout_context,
            )
            writeout_body = parse_jax_body_response(writeout_response, "writeout_body")

        return json.dumps(
            {
                "rhs": list(rhs),
                "helper_functions": helper_functions,
                "loss_body": loss_body,
                "writeout_body": writeout_body,
                "review": "assembled from split pfit-jax substeps",
            }
        )

    def _complete_prompt(
        self,
        generated_dir: Path,
        step: str,
        system_template: str,
        user_template: str,
        context: dict[str, object],
    ) -> str:
        messages = self.prompt_renderer.render_messages(
            system_template,
            user_template,
            context,
        )
        return self._complete_with_log(
            generated_dir,
            step,
            messages,
            temperature=self.workflow_config.temperature,
            max_tokens=self.workflow_config.max_tokens,
        )

    def _render_and_validate_fragments(
        self,
        response: str,
        session_spec: SessionSpec,
        script_path: Path,
        session_dir: Path,
        events: list[WorkflowEvent],
    ) -> bool:
        try:
            response = _inline_referenced_rhs_intermediates(response, session_dir)
            response = _inject_referenced_helper_definitions(response, session_dir)
            fragments = parse_jax_fragments_response(response, session_spec)
        except ValidationError as exc:
            events.append(WorkflowEvent("translate_jax_fragments", "failed", str(exc)))
            return False

        script_path.write_text(render_generated_script_from_fragments(fragments, session_spec))
        events.append(WorkflowEvent("render_generated_script", "created", str(script_path)))
        return self._validate_generated_script(script_path, session_dir, events)

    def _validate_generated_script(
        self,
        script_path: Path,
        session_dir: Path,
        events: list[WorkflowEvent],
    ) -> bool:
        # A failed translation must never look like a usable legacy script.
        script_path.write_text("# pfit-sources: pending=true\n" + script_path.read_text())
        try:
            validate_generated_script_contract(script_path)
        except ValidationError as exc:
            events.append(WorkflowEvent("validate_generated_script", "failed", str(exc)))
            return False
        except Exception as exc:
            events.append(
                WorkflowEvent(
                    "validate_generated_script",
                    "failed",
                    "".join(traceback.format_exception_only(type(exc), exc)).strip(),
                )
            )
            return False

        events.append(WorkflowEvent("validate_generated_script", "passed", str(script_path)))
        try:
            smoke_test_generated_script(script_path, session_dir)
        except ValidationError as exc:
            events.append(WorkflowEvent("smoke_test_generated_script", "failed", str(exc)))
            return False

        from lib.utils.source_stamp import build_stamp, write_stamp
        if build_stamp(session_dir) != self._generation_source_stamp:
            events.append(WorkflowEvent("source_freshness", "failed", "Sources changed during translation; rerun pfit jax"))
            return False
        write_stamp(session_dir)
        events.append(WorkflowEvent("smoke_test_generated_script", "passed", str(script_path)))
        return True

    def _build_generation_context(
        self,
        session_dir: Path,
        input_yaml: Path,
        session_spec: SessionSpec,
    ) -> dict[str, object]:
        user_model = session_dir / "generated" / "user_model.py"
        user_model_text = user_model.read_text() if user_model.exists() else ""
        return {
            "session_dir": session_dir,
            "input_yaml": input_yaml.read_text(),
            "session_summary": session_spec.to_prompt_text(),
            "rhs_intermediate_inventory": _rhs_intermediate_inventory(
                user_model_text,
                session_spec,
            ),
            "pfit_claude_jax_reference": _pfit_claude_jax_reference(),
            "user_model": user_model_text,
            "helper_function_inventory": _helper_function_inventory(user_model_text),
        }

    def _finish(
        self,
        success: bool,
        events: list[WorkflowEvent],
        generated_dir: Path,
    ) -> WorkflowResult:
        self._write_event_log(events, generated_dir / "agent_logs")
        return WorkflowResult(success, events)

    def _write_event_log(self, events: list[WorkflowEvent], log_dir: Path) -> None:
        log_dir.mkdir(parents=True, exist_ok=True)
        log_path = log_dir / "workflow_events.jsonl"
        with log_path.open("w") as handle:
            for event in events:
                handle.write(
                    json.dumps(
                        {
                            "step": event.step,
                            "status": event.status,
                            "message": event.message,
                        }
                    )
                )
                handle.write("\n")

    def _complete_with_log(
        self,
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
            response = self.llm_client.complete(
                messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
        except LLMError as exc:
            if exc.partial_response:
                (log_dir / f"partial_{step}.txt").write_text(exc.partial_response)
            raise
        log_path = log_dir / "llm_calls.jsonl"
        with log_path.open("a") as handle:
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


def _pfit_claude_jax_reference() -> str:
    try:
        return (
            resources.files("local_agent.agent.reference")
            .joinpath("pfit_claude_jax_reference.md")
            .read_text()
        )
    except (FileNotFoundError, ModuleNotFoundError):
        return "- pfit-claude JAX reference unavailable."


def _format_prompt_helpers(helper_functions: list[str]) -> str:
    if not helper_functions:
        return "- none"
    return "\n\n".join(f"```python\n{source}\n```" for source in helper_functions)


def _format_prompt_rhs(rhs: tuple[str, ...], session_spec: SessionSpec) -> str:
    lines = []
    for variable, expression in zip(session_spec.integrated_variables, rhs):
        lines.append(f"- d{variable.name}dt = {expression}")
    return "\n".join(lines) if lines else "- none"


def _dedupe_helper_functions(helper_functions: list[str]) -> list[str]:
    deduped: list[str] = []
    seen_names: set[str] = set()
    for source in helper_functions:
        name = _single_function_name(source)
        if name is not None:
            if name in seen_names:
                continue
            seen_names.add(name)
        deduped.append(source)
    return deduped


def _standard_loss_and_writeout_bodies(
    session_spec: SessionSpec,
    *,
    data_width: int | None = None,
    user_model_source: str = "",
) -> tuple[str, str] | None:
    if _user_model_has_custom_loss(user_model_source):
        return None
    declared_width = max(len(session_spec.data_column_names) - 1, 0)
    if data_width is not None and declared_width != data_width:
        return None
    observation_pairs = _standard_observation_pairs(session_spec)
    if not observation_pairs:
        return None

    uses_observables = any(target in session_spec.observable_names for _, target in observation_pairs)
    measured_terms = [f"dataset[:, {data_index}]" for data_index, _ in observation_pairs]
    simulated_terms = [
        _simulated_observation_expression(target, session_spec)
        for _, target in observation_pairs
    ]
    if any(term is None for term in simulated_terms):
        return None
    simulated = [term for term in simulated_terms if term is not None]

    setup_lines = []
    if uses_observables:
        setup_lines.append("observables = _observables(solution, trainable_parameters, fixed_parameters)")
    setup = "\n".join(setup_lines)
    measured_stack = _jnp_stack_expression(measured_terms)
    simulated_stack = _jnp_stack_expression(simulated)

    loss_lines = [
        *setup_lines,
        f"measured = {measured_stack}",
        f"simulated = {simulated_stack}",
        "scale = jnp.maximum(jnp.max(jnp.abs(measured), axis=0), 1.0e-12)",
        "mask = jnp.isfinite(measured) & jnp.isfinite(simulated)",
        "measured_safe = jnp.where(mask, measured, 0.0)",
        "simulated_safe = jnp.where(mask, simulated, 0.0)",
        "residuals = jnp.where(mask, (simulated_safe - measured_safe) / scale, 0.0)",
        "count = jnp.maximum(jnp.sum(mask), 1)",
        "return jnp.sqrt(jnp.sum(residuals * residuals) / count)",
    ]

    writeout_terms = ["solution_time", *measured_terms, *simulated]
    writeout_lines = [
        *setup_lines,
        f"return jnp.column_stack(({', '.join(writeout_terms)}))",
    ]
    return "\n".join(loss_lines), "\n".join(writeout_lines)


def _standard_observation_pairs(session_spec: SessionSpec) -> list[tuple[int, str]]:
    known_targets = (
        {variable.name for variable in session_spec.integrated_variables}
        | set(session_spec.observable_names)
    )
    pairs: list[tuple[int, str]] = []
    for runtime_index, target in enumerate(session_spec.data_column_observes[1:]):
        if target in known_targets:
            pairs.append((runtime_index, target))
    return pairs


def _user_model_has_custom_loss(source: str) -> bool:
    if not source.strip():
        return False
    try:
        module_ast = ast.parse(source)
    except SyntaxError:
        return True
    for node in module_ast.body:
        if isinstance(node, ast.FunctionDef) and node.name == "_compute_loss_problem":
            return not _is_framework_default_loss(node)
    return False


def _deterministic_custom_loss_body(source: str) -> str | None:
    if not source.strip():
        return None
    try:
        module_ast = ast.parse(source)
    except SyntaxError:
        return None
    loss_function = next(
        (
            node
            for node in module_ast.body
            if isinstance(node, ast.FunctionDef) and node.name == "_compute_loss_problem"
        ),
        None,
    )
    if loss_function is None or _is_framework_default_loss(loss_function):
        return None
    banned_nodes = (ast.For, ast.While, ast.If, ast.Try, ast.With, ast.FunctionDef, ast.ClassDef)
    if any(isinstance(node, banned_nodes) and node is not loss_function for node in ast.walk(loss_function)):
        return None
    body_source = ast.get_source_segment(source, loss_function)
    if body_source is None:
        return None
    body_lines = textwrap.dedent(body_source).splitlines()[1:]
    body = textwrap.dedent("\n".join(body_lines)).strip()
    if not body:
        return None
    body = re.sub(r"(?<![A-Za-z0-9_])np\.", "jnp.", body)
    body = re.sub(r"return\s+float\((.+)\)\s*$", r"return \1", body, flags=re.MULTILINE)
    return body


def _is_framework_default_loss(function: ast.FunctionDef) -> bool:
    body = function.body
    while body and _is_framework_binding_assignment(body[0]):
        body = body[1:]
    compact_body = "".join("".join(ast.unparse(statement).split()) for statement in body)
    if compact_body in {"loss=0.0returnloss", "loss=0returnloss", "return0.0", "return0"}:
        return True
    return (
        "residuals=np.column_stack(" in compact_body
        and "returnfloat(np.mean(np.square(residuals)))" in compact_body
    )


def _is_framework_binding_assignment(node: ast.stmt) -> bool:
    if not isinstance(node, ast.Assign) or len(node.targets) != 1:
        return False
    if not isinstance(node.targets[0], ast.Name):
        return False
    value = node.value
    return (
        isinstance(value, ast.Subscript)
        and isinstance(value.value, ast.Name)
        and value.value.id in {"trainable_parameters", "fixed_parameters"}
    )


def _simulated_observation_expression(target: str, session_spec: SessionSpec) -> str | None:
    for index, variable in enumerate(session_spec.integrated_variables):
        if variable.name == target:
            return f"solution[:, {index}]"
    if target in session_spec.observable_names:
        return f"observables['{target}']"
    return None


def _jnp_stack_expression(terms: list[str]) -> str:
    if len(terms) == 1:
        return f"{terms[0]}[:, None]"
    return f"jnp.stack([{', '.join(terms)}], axis=1)"


def _helper_function_inventory(source: str) -> str:
    if not source.strip():
        return "- none"
    contract_functions = {
        "user_defined_system",
        "_compute_loss_problem",
        "writeout_description",
        "write_problem_result",
    }
    try:
        parsed = ast.parse(source)
    except SyntaxError:
        return "- unable to parse user_model.py"

    lines: list[str] = []
    for node in parsed.body:
        if not isinstance(node, ast.FunctionDef):
            continue
        if node.name in contract_functions:
            continue
        source_segment = ast.get_source_segment(source, node)
        if source_segment:
            lines.append(f"- {node.name}:\n```python\n{source_segment}\n```")
        else:
            args = ", ".join(arg.arg for arg in node.args.args)
            lines.append(f"- {node.name}({args})")
    return "\n".join(lines) if lines else "- none"


def _inject_referenced_helper_definitions(response: str, session_dir: Path) -> str:
    user_model = Path(session_dir) / "generated" / "user_model.py"
    if not user_model.exists():
        return response
    user_model_source = user_model.read_text()
    helper_sources = _helper_sources(user_model_source)
    if not helper_sources:
        return response
    try:
        data = parse_llm_json_object(response, "pfit-jax")
    except ValidationError:
        return response

    helper_functions = data.get("helper_functions", [])
    if helper_functions in (None, ""):
        helper_functions = []
    if not isinstance(helper_functions, list):
        return response

    normalized_helpers: list[str] = []
    present_names: set[str] = set()
    for helper in helper_functions:
        if not isinstance(helper, str):
            return response
        replacement = _helper_source_for_call(helper, helper_sources)
        normalized_helpers.append(replacement or helper)
        name = _single_function_name(replacement or helper)
        if name:
            present_names.add(name)

    referenced_names = _referenced_helper_names(data, helper_sources)
    for name in sorted(referenced_names - present_names):
        normalized_helpers.append(helper_sources[name])
    data["helper_functions"] = normalized_helpers
    return json.dumps(data)


def _inline_referenced_rhs_intermediates(response: str, session_dir: Path) -> str:
    user_model = Path(session_dir) / "generated" / "user_model.py"
    if not user_model.exists():
        return response
    try:
        data = parse_llm_json_object(response, "pfit-jax")
    except ValidationError:
        return response
    rhs = data.get("rhs")
    if not isinstance(rhs, list) or not all(isinstance(item, str) for item in rhs):
        return response

    assignment_map = _rhs_intermediate_assignments(user_model.read_text())
    if not assignment_map:
        return response

    data["rhs"] = [_inline_names_in_expression(item, assignment_map) for item in rhs]
    return json.dumps(data)


def _rhs_intermediate_assignments(source: str) -> dict[str, str]:
    try:
        parsed = ast.parse(source)
    except SyntaxError:
        return {}
    function = _find_function(parsed, "user_defined_system")
    if function is None:
        return {}
    assignments: dict[str, str] = {}
    for node in function.body:
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        target = node.targets[0]
        if not isinstance(target, ast.Name):
            continue
        if _is_direct_binding(node.value):
            continue
        assignments[target.id] = ast.unparse(node.value)
    return assignments


class _NameInliner(ast.NodeTransformer):
    def __init__(self, assignment_map: dict[str, str]):
        self.assignment_map = assignment_map
        self._stack: set[str] = set()

    def visit_Name(self, node: ast.Name):
        if not isinstance(node.ctx, ast.Load) or node.id not in self.assignment_map:
            return node
        if node.id in self._stack:
            return node
        self._stack.add(node.id)
        try:
            replacement = ast.parse(self.assignment_map[node.id], mode="eval").body
            replacement = self.visit(replacement)
        finally:
            self._stack.remove(node.id)
        return ast.copy_location(replacement, node)


def _inline_names_in_expression(expression: str, assignment_map: dict[str, str]) -> str:
    try:
        parsed = ast.parse(expression, mode="eval")
    except SyntaxError:
        try:
            parsed_module = ast.parse(expression)
        except SyntaxError:
            return expression
        if len(parsed_module.body) != 1 or not isinstance(parsed_module.body[0], ast.Assign):
            return expression
        parsed = ast.Expression(body=parsed_module.body[0].value)
    parsed = _NameInliner(assignment_map).visit(parsed)
    ast.fix_missing_locations(parsed)
    assert isinstance(parsed, ast.Expression)
    return ast.unparse(parsed.body)


def _helper_sources(source: str) -> dict[str, str]:
    contract_functions = {
        "user_defined_system",
        "_compute_loss_problem",
        "writeout_description",
        "write_problem_result",
    }
    try:
        parsed = ast.parse(source)
    except SyntaxError:
        return {}
    helpers: dict[str, str] = {}
    for node in ast.walk(parsed):
        if not isinstance(node, ast.FunctionDef) or node.name in contract_functions:
            continue
        source_segment = ast.get_source_segment(source, node)
        if source_segment:
            helpers[node.name] = textwrap.dedent(source_segment)
    return helpers


def _helper_source_for_call(source: str, helper_sources: dict[str, str]) -> str | None:
    try:
        parsed = ast.parse(source.strip(), mode="eval")
    except SyntaxError:
        return None
    call = parsed.body
    if isinstance(call, ast.Call) and isinstance(call.func, ast.Name):
        return helper_sources.get(call.func.id)
    return None


def _single_function_name(source: str) -> str | None:
    try:
        parsed = ast.parse(source)
    except SyntaxError:
        return None
    if len(parsed.body) == 1 and isinstance(parsed.body[0], ast.FunctionDef):
        return parsed.body[0].name
    return None


def _referenced_helper_names(data: dict[str, object], helper_sources: dict[str, str]) -> set[str]:
    referenced: set[str] = set()
    for key in ("rhs", "loss_body", "writeout_body", "writeout_description"):
        value = data.get(key)
        values = value if isinstance(value, list) else [value]
        for item in values:
            if isinstance(item, str):
                referenced.update(_called_names(item) & set(helper_sources))
    return referenced


def _called_names(source: str) -> set[str]:
    try:
        parsed = ast.parse(source)
    except SyntaxError:
        try:
            parsed = ast.parse(source, mode="eval")
        except SyntaxError:
            return set()
    names: set[str] = set()
    for node in ast.walk(parsed):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            names.add(node.func.id)
    return names


def _rhs_intermediate_inventory(source: str, session_spec: SessionSpec) -> str:
    if not source.strip():
        return "- none"
    try:
        parsed = ast.parse(source)
    except SyntaxError:
        return "- unable to parse user_model.py"

    function = _find_function(parsed, "user_defined_system")
    if function is None:
        return "- none"

    parameter_names = {parameter.name for parameter in session_spec.trainable_parameters}
    fixed_names = {parameter.name for parameter in session_spec.fixed_parameters}
    state_names = {variable.name for variable in session_spec.integrated_variables}
    skip_names = parameter_names | fixed_names | state_names

    lines: list[str] = []
    for node in function.body:
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        target = node.targets[0]
        if not isinstance(target, ast.Name):
            continue
        if target.id in skip_names or _is_direct_binding(node.value):
            continue
        lines.append(f"- {target.id} = {ast.unparse(node.value)}")
    return "\n".join(lines) if lines else "- none"


def _find_function(parsed: ast.Module, name: str) -> ast.FunctionDef | None:
    for node in parsed.body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    return None


def _is_direct_binding(node: ast.AST) -> bool:
    return (
        isinstance(node, ast.Subscript)
        and isinstance(node.value, ast.Name)
        and node.value.id in {"y", "trainable_parameters", "fixed_parameters"}
    )
