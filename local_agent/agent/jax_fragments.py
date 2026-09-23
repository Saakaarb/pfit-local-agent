import ast
from dataclasses import dataclass
import re
import textwrap

from local_agent.agent.llm_json import normalize_code_string, parse_llm_json_object
from local_agent.agent.session_spec import SessionSpec
from local_agent.agent.validators import ValidationError


@dataclass(frozen=True)
class JaxFragments:
    rhs: tuple[str, ...]
    loss_body: str
    writeout_body: str
    helper_functions: tuple[str, ...] = ()
    review: str = ""


def parse_jax_fragments_response(response: str, session_spec: SessionSpec) -> JaxFragments:
    data = parse_llm_json_object(response, "pfit-jax")

    rhs = data.get("rhs")
    if not isinstance(rhs, list) or not all(isinstance(item, str) for item in rhs):
        raise ValidationError("pfit-jax rhs must be a list of strings")
    if len(rhs) != len(session_spec.integrated_variables):
        raise ValidationError(
            "pfit-jax rhs length must match integrated variables: "
            f"{len(rhs)} vs {len(session_spec.integrated_variables)}"
        )

    fragments = JaxFragments(
        rhs=tuple(_normalize_rhs_expression(item) for item in rhs),
        loss_body=_require_string(data, "loss_body", code=True),
        writeout_body=_require_string(
            data,
            "writeout_body",
            fallback_key="writeout_description",
            code=True,
            writeout=True,
        ),
        helper_functions=tuple(_parse_helper_functions(data.get("helper_functions", []))),
        review=_optional_string(data, "review"),
    )
    fragments = _normalize_helper_closures(fragments, session_spec)
    validate_jax_fragments(fragments, session_spec)
    return fragments


def parse_jax_helpers_response(response: str) -> tuple[str, ...]:
    data = parse_llm_json_object(response, "pfit-jax helpers")
    return tuple(_parse_helper_functions(data.get("helper_functions", [])))


def parse_jax_rhs_response(response: str, session_spec: SessionSpec) -> tuple[str, ...]:
    data = parse_llm_json_object(response, "pfit-jax rhs")
    rhs = data.get("rhs")
    if not isinstance(rhs, list) or not all(isinstance(item, str) for item in rhs):
        raise ValidationError("pfit-jax rhs must be a list of strings")
    if len(rhs) != len(session_spec.integrated_variables):
        rhs = _extract_rhs_from_function_lines(rhs)
    if len(rhs) != len(session_spec.integrated_variables):
        raise ValidationError(
            "pfit-jax rhs length must match integrated variables: "
            f"{len(rhs)} vs {len(session_spec.integrated_variables)}"
        )
    return tuple(_normalize_rhs_expression(item) for item in rhs)


def parse_jax_body_response(response: str, key: str) -> str:
    data = parse_llm_json_object(response, f"pfit-jax {key}")
    return _require_string(data, key, code=True, writeout=(key == "writeout_body"))


def validate_jax_fragments(fragments: JaxFragments, session_spec: SessionSpec) -> None:
    helper_names = _validate_helper_functions(fragments.helper_functions)
    allowed_names = (
        {parameter.name for parameter in session_spec.trainable_parameters}
        | {parameter.name for parameter in session_spec.fixed_parameters}
        | {variable.name for variable in session_spec.integrated_variables}
        | helper_names
        | {"t", "jnp"}
    )
    for index, expression in enumerate(fragments.rhs):
        variable_name = session_spec.integrated_variables[index].name
        _validate_expression(
            f"rhs for {variable_name}",
            expression,
            allowed_names,
            helper_names=helper_names,
        )

    body_names = allowed_names | helper_names | {
        "solution_time",
        "solution",
        "dataset",
        "trainable_parameters",
        "fixed_parameters",
        "writeout_array",
        "Nts",
        "scale",
        "scale_factor",
        "residuals",
        "loss",
        "loss_value",
        "jnp",
    }
    _validate_body(
        "loss_body",
        fragments.loss_body,
        body_names,
        helper_names=helper_names,
        require_return=True,
    )
    _validate_body(
        "writeout_body",
        fragments.writeout_body,
        body_names | {"np", "range"},
        helper_names=helper_names,
        require_return=True,
        allow_np=True,
        allow_nested_functions=True,
    )
    data_width = max(len(session_spec.data_column_names) - 1, 0)
    solution_width = len(session_spec.integrated_variables)
    _validate_array_indices("loss_body", fragments.loss_body, data_width, solution_width)
    _validate_array_indices("writeout_body", fragments.writeout_body, data_width, solution_width)


def render_generated_script_from_fragments(
    fragments: JaxFragments,
    session_spec: SessionSpec,
) -> str:
    parameter_names = [parameter.name for parameter in session_spec.trainable_parameters]
    fixed_names = [parameter.name for parameter in session_spec.fixed_parameters]
    variable_names = [variable.name for variable in session_spec.integrated_variables]

    parameter_bindings = _render_parameter_bindings(parameter_names)
    fixed_bindings = _render_fixed_bindings(fixed_names)
    state_bindings = "\n".join(
        f"    {name} = y[{index}]" for index, name in enumerate(variable_names)
    )
    derivatives = "\n".join(
        f"    d{name}dt = {fragments.rhs[index]}"
        for index, name in enumerate(variable_names)
    )
    derivative_array = ", ".join(f"d{name}dt" for name in variable_names)
    helper_functions = _format_helper_functions(fragments.helper_functions)
    loss_body = _indent_body(fragments.loss_body)
    writeout_body = _indent_body(fragments.writeout_body)

    return f"""import jax
import jax.numpy as jnp
import numpy as np
import diffrax
from diffrax import RESULTS

jax.config.update("jax_enable_x64", True)

@jax.jit
def unscale_value(val, min_val, max_val, is_logscale):
    lin_unscaled = ((val + 1.0) / 2.0) * (max_val - min_val) + min_val
    unscaled = jnp.where(is_logscale, 10.0**lin_unscaled, lin_unscaled)
    return unscaled

{helper_functions}

@jax.jit
def user_defined_system(t, y, other_args):
    constants = other_args["constants"]
    trainable_variables = other_args["trainable_variables"]
    dataset = constants["dataset"]
    t_eval = constants["t_eval"]
{parameter_bindings}
{fixed_bindings}
{state_bindings}
{derivatives}
    return jnp.array([{derivative_array}])

@jax.jit
def _integrate_system(constants, trainable_variables):
    term = diffrax.ODETerm(user_defined_system)
    solver = diffrax.{session_spec.integrator}()
    t_eval = constants["t_eval"]
    other_args = {{"constants": constants, "trainable_variables": trainable_variables}}
    sol = diffrax.diffeqsolve(
        term,
        solver,
        t0=constants["init_time"],
        t1=t_eval[-1],
        max_steps={session_spec.max_steps},
        dt0=constants["init_timestep"],
        y0=constants["init_cond"],
        args=other_args,
        saveat=diffrax.SaveAt(ts=t_eval),
        throw=False,
        stepsize_controller=diffrax.PIDController(
            rtol=constants["stepsize_rtol"],
            atol=constants["stepsize_atol"],
        ),
    )
    return sol.ts, sol.ys, sol.result

@jax.jit
def _compute_loss_value(constants, trainable_variables, solution_time, solution):
    dataset = constants["dataset"]
{_indent_block(_render_parameter_bindings(parameter_names), 0)}
{_indent_block(_render_fixed_bindings(fixed_names), 0)}
{loss_body}

@jax.jit
def _compute_loss_problem(constants, trainable_variables):
    solution_time, solution, result = _integrate_system(constants, trainable_variables)
    failed = jnp.logical_or(result == RESULTS.max_steps_reached, result == RESULTS.singular)
    loss_value = _compute_loss_value(constants, trainable_variables, solution_time, solution)
    return jnp.where(failed, constants["error_loss"], loss_value)

def _write_problem_result(constants, trainable_variables):
    dataset = constants["dataset"]
    solution_time, solution, result = _integrate_system(constants, trainable_variables)
{_indent_block(_render_parameter_bindings(parameter_names), 0)}
{_indent_block(_render_fixed_bindings(fixed_names), 0)}
{writeout_body}
"""


def _require_string(
    data: dict[str, object],
    key: str,
    *,
    fallback_key: str | None = None,
    code: bool = False,
    writeout: bool = False,
) -> str:
    value = data.get(key)
    if value is None and fallback_key is not None:
        value = data.get(fallback_key)
    if isinstance(value, list) and all(isinstance(item, str) for item in value):
        value = "\n".join(value)
    if not isinstance(value, str) or not value.strip():
        raise ValidationError(f"pfit-jax response is missing {key}")
    if not code:
        return value.strip()
    if writeout:
        return _normalize_writeout_body_code(value)
    return _normalize_body_code(value)


def _optional_string(data: dict[str, object], key: str) -> str:
    value = data.get(key, "")
    if not isinstance(value, str):
        raise ValidationError(f"pfit-jax {key} must be a string")
    return value.strip()


def _parse_helper_functions(value: object) -> list[str]:
    if value in (None, ""):
        return []
    if not isinstance(value, list):
        raise ValidationError("pfit-jax helper_functions must be a list of strings")
    helpers: list[str] = []
    for item in value:
        item = _coerce_helper_function(item)
        if not isinstance(item, str):
            raise ValidationError("pfit-jax helper_functions must be a list of strings")
        if not item.strip():
            continue
        helpers.extend(_split_helper_functions(_normalize_fragment_code(item)))
    return helpers


def _coerce_helper_function(value: object) -> object:
    if not isinstance(value, dict):
        return value
    name = value.get("name")
    body = value.get("body")
    if body is None:
        body = value.get("source")
    if isinstance(body, list) and all(isinstance(item, str) for item in body):
        body = "\n".join(body)
    if not isinstance(name, str) or not isinstance(body, str):
        return value
    args = value.get("args", value.get("arguments", ["solution", "trainable_parameters", "fixed_parameters"]))
    if not isinstance(args, list) or not all(isinstance(arg, str) for arg in args):
        args = ["solution", "trainable_parameters", "fixed_parameters"]
    normalized_body = normalize_code_string(body).strip()
    if not normalized_body:
        return value
    indented = "\n".join(
        f"    {line}" if line.strip() else line
        for line in normalized_body.splitlines()
    )
    return f"def {name}({', '.join(args)}):\n{indented}"


def _normalize_rhs_expression(source: str) -> str:
    normalized = _normalize_fragment_code(source)
    try:
        parsed = ast.parse(normalized)
    except SyntaxError:
        return normalized
    if len(parsed.body) == 1 and isinstance(parsed.body[0], ast.Assign):
        return _normalize_jax_expression_ast(parsed.body[0].value)
    try:
        expression = ast.parse(normalized, mode="eval")
    except SyntaxError:
        return normalized
    return _normalize_jax_expression_ast(expression.body)


def _normalize_jax_expression_ast(node: ast.AST) -> str:
    normalized = _JaxBooleanNormalizer().visit(node)
    ast.fix_missing_locations(normalized)
    return ast.unparse(normalized)
    return normalized


def _extract_rhs_from_function_lines(lines: list[str]) -> list[str]:
    source = "".join(lines)
    try:
        parsed = ast.parse(source)
    except SyntaxError:
        source = "\n".join(lines)
        try:
            parsed = ast.parse(source)
        except SyntaxError:
            return lines

    for function in ast.walk(parsed):
        if not isinstance(function, ast.FunctionDef):
            continue
        extracted = _extract_rhs_from_function(function)
        if extracted:
            return extracted
    return lines


def _extract_rhs_from_function(function: ast.FunctionDef) -> list[str]:
    assignments: dict[str, ast.AST] = {}
    nested_functions: dict[str, ast.FunctionDef] = {}
    for statement in function.body:
        if isinstance(statement, ast.FunctionDef):
            nested_functions[statement.name] = statement
            continue
        if not isinstance(statement, ast.Assign) or len(statement.targets) != 1:
            continue
        target = statement.targets[0]
        if not isinstance(target, ast.Name) or _is_direct_rhs_binding(statement.value):
            continue
        assignments[target.id] = statement.value

    for statement in function.body:
        if not isinstance(statement, ast.Return):
            continue
        value = statement.value
        if isinstance(value, ast.Name) and value.id in assignments:
            value = assignments[value.id]
        array_arg = _returned_array_argument(value)
        if isinstance(array_arg, (ast.List, ast.Tuple)):
            inliner = _RhsFunctionInliner(assignments, nested_functions)
            return [ast.unparse(inliner.visit(item)) for item in array_arg.elts]
    return []


class _RhsFunctionInliner(ast.NodeTransformer):
    def __init__(
        self,
        assignments: dict[str, ast.AST],
        nested_functions: dict[str, ast.FunctionDef],
    ):
        self.assignments = assignments
        self.nested_functions = nested_functions
        self._stack: set[str] = set()

    def visit_Name(self, node: ast.Name):
        if not isinstance(node.ctx, ast.Load) or node.id not in self.assignments:
            return node
        if node.id in self._stack:
            return node
        self._stack.add(node.id)
        try:
            replacement = self.visit(self.assignments[node.id])
        finally:
            self._stack.remove(node.id)
        return ast.copy_location(replacement, node)

    def visit_Call(self, node: ast.Call):
        self.generic_visit(node)
        if not isinstance(node.func, ast.Name):
            return node
        function = self.nested_functions.get(node.func.id)
        if function is None:
            return node
        return_value = _single_return_value(function)
        if return_value is None:
            return node
        arg_map = {
            arg.arg: value
            for arg, value in zip(function.args.args, node.args)
        }
        replacement = _ArgumentSubstituter(arg_map).visit(return_value)
        return ast.copy_location(self.visit(replacement), node)


class _ArgumentSubstituter(ast.NodeTransformer):
    def __init__(self, arg_map: dict[str, ast.AST]):
        self.arg_map = arg_map

    def visit_Name(self, node: ast.Name):
        if isinstance(node.ctx, ast.Load) and node.id in self.arg_map:
            return ast.copy_location(self.arg_map[node.id], node)
        return node


def _single_return_value(function: ast.FunctionDef) -> ast.AST | None:
    returns = [node.value for node in ast.walk(function) if isinstance(node, ast.Return)]
    if len(returns) == 1:
        return returns[0]
    return None


def _is_direct_rhs_binding(node: ast.AST) -> bool:
    return (
        isinstance(node, ast.Subscript)
        and isinstance(node.value, ast.Name)
        and node.value.id in {"y", "trainable_parameters", "fixed_parameters"}
    )


def _returned_array_argument(node: ast.AST) -> ast.AST | None:
    if not isinstance(node, ast.Call) or not node.args:
        return None
    func = node.func
    if isinstance(func, ast.Attribute) and func.attr == "array":
        return node.args[0]
    if isinstance(func, ast.Name) and func.id in {"array", "jnp_array", "np_array"}:
        return node.args[0]
    return None


def _normalize_fragment_code(source: str) -> str:
    normalized = re.sub(r"(?<![A-Za-z0-9_])np\.", "jnp.", normalize_code_string(source))
    normalized = _normalize_elementwise_range_loops(normalized)
    normalized = _normalize_jax_index_assignments(normalized)
    return _normalize_jax_boolean_code(normalized)


def _normalize_jax_boolean_code(source: str) -> str:
    try:
        parsed = ast.parse(source)
        mode = "exec"
    except SyntaxError:
        try:
            parsed = ast.parse(source, mode="eval")
            mode = "eval"
        except SyntaxError:
            return source
    normalized = _JaxBooleanNormalizer().visit(parsed)
    ast.fix_missing_locations(normalized)
    if mode == "eval":
        assert isinstance(normalized, ast.Expression)
        return ast.unparse(normalized.body)
    return ast.unparse(normalized)


class _JaxBooleanNormalizer(ast.NodeTransformer):
    def visit_BoolOp(self, node: ast.BoolOp) -> ast.AST:
        self.generic_visit(node)
        function_name = "logical_and" if isinstance(node.op, ast.And) else "logical_or"
        values = list(node.values)
        if not values:
            return node
        expression = values[0]
        for value in values[1:]:
            expression = ast.Call(
                func=ast.Attribute(
                    value=ast.Name(id="jnp", ctx=ast.Load()),
                    attr=function_name,
                    ctx=ast.Load(),
                ),
                args=[expression, value],
                keywords=[],
            )
        return expression


def _normalize_body_code(source: str) -> str:
    normalized = _normalize_fragment_code(source)
    unwrapped = _unwrap_single_function_body(normalized)
    if unwrapped != normalized:
        return unwrapped
    return _unwrap_single_function_body(_normalize_fragment_code(_dedent_body_source(source)))


def _normalize_writeout_body_code(source: str) -> str:
    normalized = normalize_code_string(source)
    unwrapped = _unwrap_single_function_body(normalized)
    if unwrapped != normalized:
        return unwrapped
    return _unwrap_single_function_body(_dedent_body_source(source))


def _dedent_body_source(source: str) -> str:
    normalized = normalize_code_string(source)
    lines = normalized.splitlines()
    if len(lines) <= 1:
        return normalized
    first_index = next((index for index, line in enumerate(lines) if line.strip()), None)
    if first_index is None:
        return ""
    first_indent = len(lines[first_index]) - len(lines[first_index].lstrip())
    if first_indent == 0:
        tail_indents = [
            len(line) - len(line.lstrip())
            for line in lines[first_index + 1 :]
            if line.strip()
        ]
        positive_tail = [indent for indent in tail_indents if indent > 0]
        if positive_tail and len(positive_tail) == len(tail_indents):
            trim = min(positive_tail)
            lines = lines[: first_index + 1] + [
                line[trim:] if line.strip() else line
                for line in lines[first_index + 1 :]
            ]
    return textwrap.dedent("\n".join(lines)).strip()


def _unwrap_single_function_body(source: str) -> str:
    try:
        parsed = ast.parse(source)
    except SyntaxError:
        return source
    if len(parsed.body) == 1 and isinstance(parsed.body[0], ast.FunctionDef):
        return "\n".join(ast.unparse(statement) for statement in parsed.body[0].body)
    return source


def _normalize_jax_index_assignments(source: str) -> str:
    lines: list[str] = []
    pattern = re.compile(r"^(\s*)([A-Za-z_][A-Za-z0-9_]*)(\[.+\])\s*=\s*(.+)$")
    for line in source.splitlines():
        match = pattern.match(line)
        if match:
            indent, name, indexer, value = match.groups()
            lines.append(f"{indent}{name} = {name}.at{indexer}.set({value})")
        else:
            lines.append(line)
    return "\n".join(lines)


def _normalize_elementwise_range_loops(source: str) -> str:
    try:
        parsed = ast.parse(source)
    except SyntaxError:
        return source
    parsed = _ElementwiseRangeLoopNormalizer().visit(parsed)
    ast.fix_missing_locations(parsed)
    return ast.unparse(parsed)


class _ElementwiseRangeLoopNormalizer(ast.NodeTransformer):
    def visit_For(self, node: ast.For):
        self.generic_visit(node)
        if not _is_range_loop(node):
            return node
        replacements: list[ast.Assign] = []
        for statement in node.body:
            if not isinstance(statement, ast.Assign) or len(statement.targets) != 1:
                return node
            target = statement.targets[0]
            if not isinstance(target, ast.Name):
                return node
            replacements.append(
                ast.Assign(
                    targets=[target],
                    value=_LoopIndexStripper(node.target.id).visit(statement.value),
                )
            )
        return replacements


def _is_range_loop(node: ast.For) -> bool:
    return (
        isinstance(node.target, ast.Name)
        and isinstance(node.iter, ast.Call)
        and isinstance(node.iter.func, ast.Name)
        and node.iter.func.id == "range"
        and len(node.iter.args) == 1
    )


class _LoopIndexStripper(ast.NodeTransformer):
    def __init__(self, index_name: str):
        self.index_name = index_name

    def visit_Subscript(self, node: ast.Subscript):
        self.generic_visit(node)
        if isinstance(node.slice, ast.Name) and node.slice.id == self.index_name:
            return ast.copy_location(node.value, node)
        if (
            isinstance(node.slice, ast.Tuple)
            and node.slice.elts
            and isinstance(node.slice.elts[0], ast.Name)
            and node.slice.elts[0].id == self.index_name
        ):
            node.slice.elts[0] = ast.Slice()
            return node
        return node


def _split_helper_functions(source: str) -> list[str]:
    try:
        parsed = ast.parse(source)
    except SyntaxError:
        return [source]
    if len(parsed.body) > 1 and all(isinstance(node, ast.FunctionDef) for node in parsed.body):
        return [ast.unparse(node) for node in parsed.body]
    return [source]


def _normalize_helper_closures(
    fragments: JaxFragments,
    session_spec: SessionSpec,
) -> JaxFragments:
    available_names = (
        {parameter.name for parameter in session_spec.trainable_parameters}
        | {parameter.name for parameter in session_spec.fixed_parameters}
        | {variable.name for variable in session_spec.integrated_variables}
        | {"t"}
    )
    helper_sources = list(fragments.helper_functions)
    additions_by_helper: dict[str, list[str]] = {}
    helper_names: set[str] = set()
    for source in helper_sources:
        parsed = _parse_single_helper_or_none(source)
        if parsed is None:
            continue
        function = parsed.body[0]
        assert isinstance(function, ast.FunctionDef)
        helper_names.add(function.name)
        missing = sorted(_helper_free_names(function, helper_names) & available_names)
        if missing:
            additions_by_helper[function.name] = missing

    if not additions_by_helper:
        return fragments

    normalized_helpers = tuple(
        _add_helper_arguments(source, additions_by_helper)
        for source in fragments.helper_functions
    )
    return JaxFragments(
        rhs=tuple(_add_call_arguments(expr, additions_by_helper) for expr in fragments.rhs),
        loss_body=_add_call_arguments(fragments.loss_body, additions_by_helper),
        writeout_body=_add_call_arguments(fragments.writeout_body, additions_by_helper),
        helper_functions=normalized_helpers,
        review=fragments.review,
    )


def _parse_single_helper_or_none(source: str) -> ast.Module | None:
    try:
        parsed = ast.parse(source)
    except SyntaxError:
        return None
    if len(parsed.body) == 1 and isinstance(parsed.body[0], ast.FunctionDef):
        return parsed
    return None


def _helper_free_names(function: ast.FunctionDef, helper_names: set[str]) -> set[str]:
    arg_names = {arg.arg for arg in function.args.args}
    local_names = _assigned_names(function)
    allowed = arg_names | local_names | helper_names | {"jnp"}
    free: set[str] = set()
    for node in ast.walk(function):
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
            if node.id not in allowed:
                free.add(node.id)
    return free


def _add_helper_arguments(
    source: str,
    additions_by_helper: dict[str, list[str]],
) -> str:
    parsed = _parse_single_helper_or_none(source)
    if parsed is None:
        return source
    function = parsed.body[0]
    assert isinstance(function, ast.FunctionDef)
    additions = [
        name
        for name in additions_by_helper.get(function.name, [])
        if name not in {arg.arg for arg in function.args.args}
    ]
    for name in additions:
        function.args.args.append(ast.arg(arg=name))
    ast.fix_missing_locations(parsed)
    return ast.unparse(parsed)


class _CallArgumentAdder(ast.NodeTransformer):
    def __init__(self, additions_by_helper: dict[str, list[str]]):
        self.additions_by_helper = additions_by_helper

    def visit_Call(self, node: ast.Call):
        self.generic_visit(node)
        if isinstance(node.func, ast.Name):
            existing = {
                arg.id
                for arg in node.args
                if isinstance(arg, ast.Name)
            }
            for name in self.additions_by_helper.get(node.func.id, []):
                if name not in existing:
                    node.args.append(ast.Name(id=name, ctx=ast.Load()))
        return node


def _add_call_arguments(source: str, additions_by_helper: dict[str, list[str]]) -> str:
    try:
        parsed = ast.parse(source)
        mode = "exec"
    except SyntaxError:
        try:
            parsed = ast.parse(source, mode="eval")
            mode = "eval"
        except SyntaxError:
            return source
    parsed = _CallArgumentAdder(additions_by_helper).visit(parsed)
    ast.fix_missing_locations(parsed)
    if mode == "eval":
        assert isinstance(parsed, ast.Expression)
        return ast.unparse(parsed.body)
    return ast.unparse(parsed)


def _validate_expression(
    label: str,
    expression: str,
    allowed_names: set[str],
    *,
    helper_names: set[str],
) -> None:
    try:
        parsed = ast.parse(expression, mode="eval")
    except SyntaxError as exc:
        raise ValidationError(f"pfit-jax expression has invalid syntax: {expression}") from exc
    _validate_ast(parsed, allowed_names, label=label, helper_names=helper_names)


def _validate_body(
    label: str,
    body: str,
    allowed_names: set[str],
    *,
    helper_names: set[str],
    require_return: bool,
    allow_np: bool = False,
    allow_nested_functions: bool = False,
) -> None:
    try:
        parsed = ast.parse("def _fragment():\n" + _indent_for_parse(body))
    except SyntaxError as exc:
        raise ValidationError(f"pfit-jax {label} has invalid syntax: {exc}") from exc

    function = parsed.body[0]
    assert isinstance(function, ast.FunctionDef)
    if require_return and not any(isinstance(node, ast.Return) for node in ast.walk(function)):
        raise ValidationError(f"pfit-jax {label} must return a value")
    local_names = _assigned_names(function)
    if allow_nested_functions:
        local_names |= _nested_function_names(function)
    _validate_ast(
        function,
        allowed_names | local_names,
        label=label,
        helper_names=helper_names,
        allow_root_function=True,
        allow_np=allow_np,
        allow_nested_functions=allow_nested_functions,
    )


def _validate_helper_functions(helper_functions: tuple[str, ...]) -> set[str]:
    helper_names: set[str] = set()
    for source in helper_functions:
        try:
            parsed = ast.parse(source)
        except SyntaxError as exc:
            raise ValidationError(f"pfit-jax helper function has invalid syntax: {exc}") from exc
        if len(parsed.body) != 1 or not isinstance(parsed.body[0], ast.FunctionDef):
            if (
                len(parsed.body) == 1
                and isinstance(parsed.body[0], ast.Expr)
                and isinstance(parsed.body[0].value, ast.Call)
            ):
                raise ValidationError(
                    "pfit-jax helper_functions entries must define helper functions, "
                    "not call them. Copy the full helper definition from user_model.py, "
                    "for example def _observables(...): ..."
                )
            raise ValidationError("pfit-jax helper_functions entries must each define one function")
        function = parsed.body[0]
        if function.name in helper_names:
            raise ValidationError(f"pfit-jax duplicate helper function: {function.name}")
        if not any(isinstance(node, ast.Return) for node in ast.walk(function)):
            raise ValidationError(f"pfit-jax helper function must return a value: {function.name}")
        helper_names.add(function.name)

    for source in helper_functions:
        parsed = ast.parse(source)
        function = parsed.body[0]
        assert isinstance(function, ast.FunctionDef)
        arg_names = {arg.arg for arg in function.args.args}
        _validate_ast(
            function,
            arg_names | helper_names | _assigned_names(function) | {"jnp"},
            label=f"helper function {function.name}",
            helper_names=helper_names,
            allow_root_function=True,
        )
    return helper_names


def _validate_array_indices(
    label: str,
    body: str,
    data_width: int,
    solution_width: int,
) -> None:
    try:
        parsed = ast.parse("def _fragment():\n" + _indent_for_parse(body))
    except SyntaxError:
        return
    widths = {"dataset": data_width, "solution": solution_width}
    for node in ast.walk(parsed):
        if not isinstance(node, ast.Subscript):
            continue
        if not isinstance(node.value, ast.Name):
            continue
        width = widths.get(node.value.id)
        if width is None:
            continue
        column = _literal_second_axis(node.slice)
        if column is None:
            continue
        if isinstance(column, int) and (column < 0 or column >= width):
            raise ValidationError(
                f"pfit-jax {label} indexes {node.value.id}[:, {column}], "
                f"but {node.value.id} has {width} column(s)"
            )
        if isinstance(column, slice):
            start = column.start or 0
            stop = column.stop
            if start < 0 or (stop is not None and stop > width):
                raise ValidationError(
                    f"pfit-jax {label} indexes {node.value.id} with an out-of-range column slice"
                )


def _literal_second_axis(node: ast.AST) -> int | slice | None:
    if not isinstance(node, ast.Tuple) or len(node.elts) < 2:
        return None
    column_node = node.elts[1]
    if isinstance(column_node, ast.Constant) and isinstance(column_node.value, int):
        return column_node.value
    if isinstance(column_node, ast.Slice):
        start = _literal_int_or_none(column_node.lower)
        stop = _literal_int_or_none(column_node.upper)
        step = _literal_int_or_none(column_node.step)
        if step not in (None, 1):
            return None
        return slice(start, stop)
    return None


def _literal_int_or_none(node: ast.AST | None) -> int | None:
    if node is None:
        return None
    if isinstance(node, ast.Constant) and isinstance(node.value, int):
        return node.value
    return None


def _validate_ast(
    node: ast.AST,
    allowed_names: set[str],
    *,
    label: str,
    helper_names: set[str],
    allow_root_function: bool = False,
    allow_np: bool = False,
    allow_nested_functions: bool = False,
) -> None:
    banned_nodes = (
        ast.Import,
        ast.ImportFrom,
        ast.FunctionDef,
        ast.AsyncFunctionDef,
        ast.ClassDef,
        ast.With,
        ast.Try,
        ast.Raise,
        ast.Global,
        ast.Nonlocal,
    )
    banned_calls = {"eval", "exec", "open", "__import__"}
    parents = _parent_map(node)
    for child in ast.walk(node):
        if child is node and allow_root_function and isinstance(child, ast.FunctionDef):
            continue
        if child is not node and allow_nested_functions and isinstance(child, ast.FunctionDef):
            continue
        if isinstance(child, banned_nodes):
            raise ValidationError("pfit-jax fragments must not define imports, functions, or classes")
        if isinstance(child, ast.Name):
            if child.id == "np" and not allow_np:
                raise ValidationError("pfit-jax fragments must use jnp, not np")
            if isinstance(child.ctx, ast.Load) and child.id not in allowed_names:
                raise ValidationError(_unknown_name_message(label, child.id))
            if (
                isinstance(child.ctx, ast.Load)
                and child.id in helper_names
                and not _is_call_function_name(child, parents)
            ):
                raise ValidationError(
                    f"pfit-jax {label} uses helper {child.id} as a value. "
                    f"Call {child.id}(...) with explicit arguments, or inline the expression."
                )
        if isinstance(child, ast.Call) and isinstance(child.func, ast.Name):
            if child.func.id in banned_calls:
                raise ValidationError(f"pfit-jax fragment contains banned call: {child.func.id}")


def _unknown_name_message(label: str, name: str) -> str:
    if label.startswith("helper function "):
        helper_name = label.removeprefix("helper function ")
        return (
            f"pfit-jax {label} uses unknown name: {name}. "
            f"Add {name} as an explicit argument to {helper_name} and update every call, "
            f"or inline the value before calling {helper_name}."
        )
    if label.startswith("rhs for "):
        return (
            f"pfit-jax {label} uses unknown name: {name}. "
            "Inline the intermediate expression in rhs, or define a helper function "
            "that takes every needed value as an explicit argument and call that helper."
        )
    return (
        f"pfit-jax {label} uses unknown name: {name}. "
        "Define it in the same body before use, pass it through an explicit helper "
        "argument, or inline the expression from user_model.py."
    )


def _assigned_names(node: ast.AST) -> set[str]:
    names: set[str] = set()
    for child in ast.walk(node):
        if isinstance(child, ast.Name) and isinstance(child.ctx, ast.Store):
            names.add(child.id)
    return names


def _nested_function_names(node: ast.AST) -> set[str]:
    names: set[str] = set()
    for child in ast.walk(node):
        if isinstance(child, ast.FunctionDef):
            names.add(child.name)
            names.update(arg.arg for arg in child.args.args)
            names.update(_assigned_names(child))
    return names


def _parent_map(node: ast.AST) -> dict[ast.AST, ast.AST]:
    return {
        child: parent
        for parent in ast.walk(node)
        for child in ast.iter_child_nodes(parent)
    }


def _is_call_function_name(node: ast.Name, parents: dict[ast.AST, ast.AST]) -> bool:
    parent = parents.get(node)
    return isinstance(parent, ast.Call) and parent.func is node


def _render_parameter_bindings(names: list[str]) -> str:
    lines = [
        '    unscaled_parameters = unscale_value(trainable_variables, constants["min_limits"], constants["max_limits"], constants["is_logscale"])'
    ]
    if names:
        if len(names) == 1:
            lines.append(f"    {names[0]}, = unscaled_parameters")
        else:
            lines.append(f"    {', '.join(names)} = unscaled_parameters")
    dict_items = ", ".join(f'"{name}": {name}' for name in names)
    lines.append(f"    trainable_parameters = {{{dict_items}}}")
    return "\n".join(lines)


def _render_fixed_bindings(names: list[str]) -> str:
    lines = ['    fixed_parameters = constants["fixed_parameters"]']
    lines.extend(f"    {name} = fixed_parameters['{name}']" for name in names)
    return "\n".join(lines)


def _indent_for_parse(body: str) -> str:
    return "\n".join(f"    {line}" if line.strip() else line for line in body.splitlines())


def _indent_body(body: str) -> str:
    return "\n".join(f"    {line}" if line.strip() else line for line in body.splitlines())


def _indent_block(block: str, extra_spaces: int) -> str:
    prefix = " " * extra_spaces
    return "\n".join(prefix + line if line.strip() else line for line in block.splitlines())


def _format_helper_functions(helper_functions: tuple[str, ...]) -> str:
    if not helper_functions:
        return ""
    return "\n\n".join(helper_functions)
