import pytest

from local_agent.agent.config import load_config


pytestmark = pytest.mark.unit


def test_load_config_defaults(tmp_path):
    config = load_config(tmp_path)

    assert config.llm.model is None
    assert config.llm.timeout_seconds == 300.0
    assert config.workflow.max_repair_attempts == 5


def test_session_config_overrides_repo_config(tmp_path):
    repo = tmp_path
    session = repo / "sessions" / "demo"
    session.mkdir(parents=True)
    (repo / "pfit.yaml").write_text(
        """
llm:
  model: repo-model
  timeout_seconds: 111
workflow:
  temperature: 0.2
  max_tokens: 111
"""
    )
    (session / "pfit.yaml").write_text(
        """
llm:
  model: session-model
workflow:
  max_tokens: 222
"""
    )

    config = load_config(repo, session)

    assert config.llm.model == "session-model"
    assert config.llm.timeout_seconds == 111
    assert config.workflow.temperature == 0.2
    assert config.workflow.max_tokens == 222


def test_invalid_config_shape_raises(tmp_path):
    (tmp_path / "pfit.yaml").write_text("- nope\n")

    with pytest.raises(ValueError, match="YAML mapping"):
        load_config(tmp_path)


def test_environment_overrides_config(tmp_path, monkeypatch):
    (tmp_path / "pfit.yaml").write_text(
        """
llm:
  model: repo-model
workflow:
  max_repair_attempts: 3
"""
    )
    monkeypatch.setenv("PFIT_LLM_MODEL", "env-model")
    monkeypatch.setenv("PFIT_LLM_BASE_URL", "http://localhost:11434")
    monkeypatch.setenv("PFIT_LLM_TIMEOUT_SECONDS", "222")
    monkeypatch.setenv("PFIT_MAX_REPAIR_ATTEMPTS", "7")
    monkeypatch.setenv("PFIT_TEMPERATURE", "0.05")
    monkeypatch.setenv("PFIT_MAX_TOKENS", "999")

    config = load_config(tmp_path)

    assert config.llm.model == "env-model"
    assert config.llm.base_url == "http://localhost:11434"
    assert config.llm.timeout_seconds == 222.0
    assert config.workflow.max_repair_attempts == 7
    assert config.workflow.temperature == 0.05
    assert config.workflow.max_tokens == 999
