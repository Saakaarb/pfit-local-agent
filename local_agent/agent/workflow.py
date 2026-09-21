import ast
from dataclasses import dataclass, field
from importlib import resources
import json
from pathlib import Path
import traceback

from local_agent.agent.config import WorkflowConfig
from local_agent.agent.jax_fragments import (
    parse_jax_fragments_response,
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

        context = self._build_generation_context(
            session_dir,
            session_validation.input_yaml,
            session_spec,
        )
        messages = self.prompt_renderer.render_messages(
            "jax_fragments.system.md",
            "jax_fragments.user.md",
            context,
        )
        response = self._complete_with_log(
            generated_dir,
            "translate_jax_fragments",
            messages,
            temperature=self.workflow_config.temperature,
            max_tokens=self.workflow_config.max_tokens,
        )
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
                "generated_script": script_path.read_text() if script_path.exists() else "",
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
    for node in parsed.body:
        if not isinstance(node, ast.FunctionDef) or node.name in contract_functions:
            continue
        source_segment = ast.get_source_segment(source, node)
        if source_segment:
            helpers[node.name] = source_segment
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
