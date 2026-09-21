import importlib
from types import SimpleNamespace

import pytest

from local_agent.core.fitting import main, make_run_output_dir, run_driver, select_session_dir


pytestmark = pytest.mark.unit


def test_fit_parameters_import_is_lightweight():
    module = importlib.import_module("fit_parameters")

    assert callable(module.run_driver)


def test_select_session_dir_uses_explicit_session(tmp_path):
    sessions = tmp_path / "sessions"
    session = sessions / "demo"
    session.mkdir(parents=True)

    selected = select_session_dir(sessions, ["fit_parameters.py", "demo"])

    assert selected == session


def test_select_session_dir_rejects_missing_explicit_session(tmp_path):
    sessions = tmp_path / "sessions"
    sessions.mkdir()

    with pytest.raises(ValueError, match="does not exist"):
        select_session_dir(sessions, ["fit_parameters.py", "missing"])


def test_select_session_dir_rejects_empty_sessions_root(tmp_path):
    sessions = tmp_path / "sessions"
    sessions.mkdir()

    with pytest.raises(ValueError, match="sessions/ is empty"):
        select_session_dir(sessions, ["fit_parameters.py"])


def test_main_returns_one_for_empty_sessions_root(tmp_path, monkeypatch, capsys):
    sessions = tmp_path / "sessions"
    sessions.mkdir()
    monkeypatch.chdir(tmp_path)

    exit_code = main(["fit_parameters.py"])

    assert exit_code == 1
    assert "sessions/ is empty" in capsys.readouterr().out


def test_run_driver_checks_generated_script_before_heavy_imports(tmp_path):
    session = tmp_path / "session"
    (session / "inputs").mkdir(parents=True)
    (session / "generated").mkdir()
    input_reader = SimpleNamespace(
        user_input_dirname="inputs",
        output_dirname="outputs",
        generated_dirname="generated",
    )

    with pytest.raises(FileNotFoundError, match="pfit jax"):
        run_driver(session, input_reader)


def test_make_run_output_dir_uses_outputs_run_id(tmp_path):
    session = tmp_path / "session"

    output_dir = make_run_output_dir(session, "run_demo")

    assert output_dir == session / "outputs" / "run_demo"
