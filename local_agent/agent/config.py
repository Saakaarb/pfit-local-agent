from dataclasses import dataclass, field
import os
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class LLMConfig:
    model: str | None = None
    base_url: str | None = None
    timeout_seconds: float = 300.0


@dataclass(frozen=True)
class WorkflowConfig:
    max_repair_attempts: int = 5
    temperature: float = 0.1
    max_tokens: int = 12000


@dataclass(frozen=True)
class PfitConfig:
    llm: LLMConfig = field(default_factory=LLMConfig)
    workflow: WorkflowConfig = field(default_factory=WorkflowConfig)


def load_config(repo_dir: Path, session_dir: Path | None = None) -> PfitConfig:
    repo_dir = Path(repo_dir)
    config_data: dict[str, Any] = {}

    repo_config = repo_dir / "pfit.yaml"
    if repo_config.exists():
        config_data = _deep_merge(config_data, _read_yaml(repo_config))

    if session_dir is not None:
        session_config = Path(session_dir) / "pfit.yaml"
        if session_config.exists():
            config_data = _deep_merge(config_data, _read_yaml(session_config))

    return _apply_env_overrides(_build_config(config_data))


def _read_yaml(path: Path) -> dict[str, Any]:
    data = yaml.safe_load(path.read_text()) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Config must be a YAML mapping: {path}")
    return data


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _build_config(data: dict[str, Any]) -> PfitConfig:
    llm_data = data.get("llm", {}) or {}
    workflow_data = data.get("workflow", {}) or {}
    if not isinstance(llm_data, dict):
        raise ValueError("Config section 'llm' must be a mapping")
    if not isinstance(workflow_data, dict):
        raise ValueError("Config section 'workflow' must be a mapping")

    return PfitConfig(
        llm=LLMConfig(
            model=llm_data.get("model"),
            base_url=llm_data.get("base_url"),
            timeout_seconds=float(
                llm_data.get("timeout_seconds", LLMConfig.timeout_seconds)
            ),
        ),
        workflow=WorkflowConfig(
            max_repair_attempts=int(
                workflow_data.get(
                    "max_repair_attempts",
                    WorkflowConfig.max_repair_attempts,
                )
            ),
            temperature=float(
                workflow_data.get("temperature", WorkflowConfig.temperature)
            ),
            max_tokens=int(workflow_data.get("max_tokens", WorkflowConfig.max_tokens)),
        ),
    )


def _apply_env_overrides(config: PfitConfig) -> PfitConfig:
    llm = LLMConfig(
        model=os.environ.get("PFIT_LLM_MODEL", config.llm.model),
        base_url=os.environ.get("PFIT_LLM_BASE_URL", config.llm.base_url),
        timeout_seconds=float(
            os.environ.get("PFIT_LLM_TIMEOUT_SECONDS", config.llm.timeout_seconds)
        ),
    )
    workflow = WorkflowConfig(
        max_repair_attempts=int(
            os.environ.get(
                "PFIT_MAX_REPAIR_ATTEMPTS",
                config.workflow.max_repair_attempts,
            )
        ),
        temperature=float(
            os.environ.get("PFIT_TEMPERATURE", config.workflow.temperature)
        ),
        max_tokens=int(os.environ.get("PFIT_MAX_TOKENS", config.workflow.max_tokens)),
    )
    return PfitConfig(llm=llm, workflow=workflow)
