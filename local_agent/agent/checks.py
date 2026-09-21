from dataclasses import dataclass, field
import csv
import json
from pathlib import Path

from local_agent.agent.config import WorkflowConfig
from local_agent.agent.llm_json import parse_llm_json_object
from local_agent.agent.prompts import PromptRenderer
from local_agent.agent.validators import ValidationError, validate_session
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
    report.critical_errors.extend(semantic.critical_errors)
    report.warnings.extend(semantic.warnings)
    report.recommendations.extend(semantic.recommendations)
    report.semantic_review = semantic.review


def _build_check_context(session_dir: Path, report: CheckReport) -> dict[str, object]:
    input_yaml = session_dir / "inputs" / "user_input.yaml"
    user_model = session_dir / "generated" / "user_model.py"
    dataset_summary = _dataset_summary(session_dir, input_yaml)
    return {
        "input_yaml": input_yaml.read_text(),
        "user_model": user_model.read_text(),
        "dataset_summary": dataset_summary,
        "deterministic_report": report.to_text(),
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
    rows: list[list[str]] = []
    with dataset_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.reader(handle)
        for index, row in enumerate(reader):
            if index >= 6:
                break
            rows.append(row)
    preview = "\n".join(",".join(row) for row in rows)
    return "\n".join(
        [
            f"Dataset path: {dataset_path}",
            f"Shape: {validation.dataset_shape[0]} rows x {validation.dataset_shape[1]} columns",
            "CSV preview:",
            preview,
        ]
    )


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
