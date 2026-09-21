import ast
import importlib.util
import sys
from pathlib import Path
from types import ModuleType


class GeneratedContractError(ValueError):
    """Raised when generated code violates the pfit script contract."""


REQUIRED_GENERATED_FUNCTIONS = {
    "user_defined_system": ("t", "y", "other_args"),
    "_integrate_system": ("constants", "trainable_variables"),
    "_compute_loss_problem": ("constants", "trainable_variables"),
    "_write_problem_result": ("constants", "trainable_variables"),
}


def validate_generated_script_contract(script_path: Path) -> ast.Module:
    script_path = Path(script_path)
    if not script_path.exists():
        raise GeneratedContractError(f"Generated script not found: {script_path}")

    source = script_path.read_text()
    if "```" in source:
        raise GeneratedContractError(
            f"Generated script contains Markdown fences: {script_path}"
        )

    try:
        module_ast = ast.parse(source, filename=str(script_path))
    except SyntaxError as exc:
        raise GeneratedContractError(
            f"Generated script has invalid Python syntax: {exc}"
        ) from exc

    functions = {
        node.name: node
        for node in module_ast.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }

    for function_name, expected_args in REQUIRED_GENERATED_FUNCTIONS.items():
        function_node = functions.get(function_name)
        if function_node is None:
            raise GeneratedContractError(
                f"Generated script is missing {function_name}()"
            )

        actual_args = tuple(arg.arg for arg in function_node.args.args)
        if actual_args != expected_args:
            raise GeneratedContractError(
                f"{function_name}() has signature {actual_args}, "
                f"expected {expected_args}"
            )

    _validate_generated_runtime_keys(module_ast)
    _validate_diffrax_calls(module_ast)
    return module_ast


def _validate_generated_runtime_keys(module_ast: ast.Module) -> None:
    allowed_other_args = {"constants", "trainable_variables"}
    banned_constants = {"trainable_parameters"}

    for node in ast.walk(module_ast):
        if not isinstance(node, ast.Subscript):
            continue
        if not isinstance(node.value, ast.Name):
            continue
        key = _constant_subscript_key(node)
        if key is None:
            continue
        if node.value.id == "other_args" and key not in allowed_other_args:
            raise GeneratedContractError(
                f'Generated script uses unsupported other_args key "{key}"'
            )
        if node.value.id == "constants" and key in banned_constants:
            raise GeneratedContractError(
                f'Generated script uses unsupported constants key "{key}"'
            )


def _constant_subscript_key(node: ast.Subscript) -> str | None:
    if isinstance(node.slice, ast.Constant) and isinstance(node.slice.value, str):
        return node.slice.value
    return None


def _validate_diffrax_calls(module_ast: ast.Module) -> None:
    for node in ast.walk(module_ast):
        if not isinstance(node, ast.Call):
            continue
        if not _is_diffrax_diffeqsolve(node.func):
            continue
        banned_keywords = sorted(
            keyword.arg for keyword in node.keywords if keyword.arg in {"rtol", "atol"}
        )
        if banned_keywords:
            raise GeneratedContractError(
                "diffrax.diffeqsolve must not receive direct keyword(s): "
                + ", ".join(banned_keywords)
            )


def _is_diffrax_diffeqsolve(node: ast.AST) -> bool:
    return (
        isinstance(node, ast.Attribute)
        and node.attr == "diffeqsolve"
        and isinstance(node.value, ast.Name)
        and node.value.id == "diffrax"
    )


def import_generated_script(script_path: Path) -> ModuleType:
    validate_generated_script_contract(script_path)
    script_path = Path(script_path)
    module_name = f"generated_script_{abs(hash(script_path))}"
    spec = importlib.util.spec_from_file_location(module_name, script_path)
    if spec is None or spec.loader is None:
        raise GeneratedContractError(f"Could not create import spec for {script_path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    except Exception as exc:
        raise GeneratedContractError(f"Generated script failed to import: {exc}") from exc
    return module
