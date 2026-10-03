import hashlib
import importlib
from types import SimpleNamespace

import pytest

from local_agent.core.fitting import (
    configure_host_cpu_devices,
    main,
    make_run_output_dir,
    run_driver,
    select_session_dir,
)
from lib.utils.runtime_provenance import collect_runtime_provenance


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


def test_cpu_devices_follow_session_request_and_host_capacity():
    environ = {"XLA_FLAGS": "--xla_cpu_enable_fast_math=false"}

    result = configure_host_cpu_devices(32, environ=environ, detected_cpus=12)

    assert result == {
        "requested_processors": 32,
        "detected_cpus": 12,
        "configured_host_devices": 12,
        "configuration_source": "session/host detection",
    }
    assert environ["XLA_FLAGS"] == (
        "--xla_cpu_enable_fast_math=false --xla_force_host_platform_device_count=12"
    )


def test_cpu_devices_do_not_exceed_smaller_session_request():
    environ = {}

    result = configure_host_cpu_devices(3, environ=environ, detected_cpus=64)

    assert result["configured_host_devices"] == 3
    assert environ["XLA_FLAGS"] == "--xla_force_host_platform_device_count=3"


def test_cpu_devices_preserve_explicit_xla_override():
    environ = {"XLA_FLAGS": "--xla_force_host_platform_device_count=6 --other=value"}

    result = configure_host_cpu_devices(24, environ=environ, detected_cpus=48)

    assert result["configured_host_devices"] == 6
    assert result["configuration_source"] == "XLA_FLAGS override"
    assert environ["XLA_FLAGS"] == "--xla_force_host_platform_device_count=6 --other=value"


def test_runtime_provenance_records_environment_optimizer_and_artifact(tmp_path):
    artifact = tmp_path / "run_config.yaml"
    artifact.write_text("model: demo\n")
    reader = SimpleNamespace(
        algorithm="DE", random_seed=17, n_particles=12, n_iters_pop=8,
        gradient_optimizer="adam", n_iters_grad=100,
    )

    result = collect_runtime_provenance(
        tmp_path,
        artifacts={"run_config": artifact, "missing": tmp_path / "missing"},
        input_reader=reader,
        llm_configuration={"provider": "ollama", "model": "qwen"},
    )

    assert result["artifact_sha256"]["run_config"] == hashlib.sha256(artifact.read_bytes()).hexdigest()
    assert result["artifact_sha256"]["missing"] is None
    assert result["optimizer"]["population_random_seed"] == 17
    assert result["optimizer"]["gradient_optimizer"] == "adam"
    assert result["llm_configuration_at_run"]["model"] == "qwen"
    assert "version" in result["python"]
    assert "commit" in result["git"] or "unavailable" in result["git"]
