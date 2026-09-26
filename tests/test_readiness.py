"""Readiness regressions: reject silent omissions and stale translations."""
import os
from pathlib import Path
import pytest
import yaml
from lib.utils.source_stamp import write_stamp, verify_stamp
from lib.utils.yamlread import YAMLReader
from local_agent.agent.validators import validate_session, parse_input_yaml, ValidationError
from local_agent.agent.readiness import check_ready
from local_agent.core.fitting import run_driver
from local_agent.cli.main import main
from tests.test_session_validation import _minimal_yaml, _write_yaml

pytestmark = pytest.mark.unit


def session(tmp_path, data="0,1\n1,2\n"):
    config = _write_yaml(tmp_path, _minimal_yaml())
    config.with_name("data.csv").write_text(data)
    root = config.parent.parent
    generated = root / "generated"
    generated.mkdir()
    (generated / "user_model.py").write_text("# model\n")
    (generated / "generated_script.py").write_text('''def user_defined_system(t, y, other_args): pass

def _integrate_system(constants, trainable_variables): pass

def _compute_loss_problem(constants, trainable_variables): pass

def _write_problem_result(constants, trainable_variables): pass
''')
    return root


@pytest.mark.parametrize("data,match", [
    ("0,1\n0,2\n", "strictly increasing"),
    ("1,1\n0,2\n", "strictly increasing"),
    ("nan,1\n1,2\n", "finite"),
    ("0,1\n", "two time points"),
    ("0\n1\n", "time column"),
    ("0,1,2\n1,2,3\n", "column count"),
    ("0,1\n1,broken\n", "numeric dataset"),
])
def test_invalid_dataset_rejected(tmp_path, data, match):
    root = session(tmp_path, data)
    with pytest.raises(ValidationError, match=match):
        validate_session(root)


def test_missing_first_measurement_does_not_drop_first_row(tmp_path):
    root = session(tmp_path, "0,\n1,2\n")
    assert validate_session(root).dataset_shape == (2, 2)


@pytest.mark.parametrize("before,after,match", [
    ("min_val: 0.1", "min_val: 0.0", "positive bounds"),
    ("max_val: 10.0", "max_val: 0.01", "finite bounds"),
    ("max_val: 10.0", "max_val: .inf", "finite bounds"),
    ("logscale: true", 'logscale: "false"', "YAML boolean"),
    ("processors: 1", "processors: 0", "integer"),
    ("population_size: 4", "population_size: 2", "at least 3"),
    ("num_iters: 1", "num_iters: 1.5", "integer"),
    ("stepsize_rtol: [1e-7]", "stepsize_rtol: [0]", "positive finite tolerances"),
    ("stepsize_rtol: [1e-7]", "stepsize_rtol: [1e-7, 1e-7]", "one per state"),
    ("initial_timestep: 1e-6", "initial_timestep: -1", "positive and finite"),
])
def test_invalid_numerical_settings(tmp_path, before, after, match):
    config = _write_yaml(tmp_path, _minimal_yaml().replace(before, after))
    with pytest.raises(ValidationError, match=match):
        parse_input_yaml(config)


def test_multiple_experiments_retained_by_low_level_reader(tmp_path):
    config = _write_yaml(tmp_path, _minimal_yaml())
    contents = yaml.safe_load(config.read_text())
    contents["experiments"].append(dict(contents["experiments"][0]))
    config.write_text(yaml.safe_dump(contents))
    reader = YAMLReader.from_file(config)
    assert len(reader.experiments) == 2
    assert reader.get_y0(1) == reader.get_y0(0)


def test_source_change_detected_despite_identical_timestamps(tmp_path):
    root = session(tmp_path)
    write_stamp(root)
    assert verify_stamp(root)[0] is True
    model = root / "generated/user_model.py"
    stat = model.stat()
    model.write_text('label = "value#changed"\n')
    os.utime(model, ns=(stat.st_atime_ns, stat.st_mtime_ns))
    assert verify_stamp(root)[0] is False
    report = check_ready(root)
    assert any("changed since" in e for e in report.critical_errors)
    out = root / "outputs/should_not_exist"
    reader = YAMLReader.from_file(root / "inputs/user_input.yaml")
    with pytest.raises(ValueError, match="not ready"):
        run_driver(root, reader, output_dir_override=out)
    assert not out.exists()


def test_stamping_hashes_text_after_hash_character(tmp_path):
    root = session(tmp_path)
    model = root / "generated/user_model.py"
    model.write_text('label = "value#one"\n')
    write_stamp(root)
    model.write_text('label = "value#two"\n')
    assert verify_stamp(root)[0] is False


def test_ready_requires_translation_but_plain_check_does_not(tmp_path, monkeypatch):
    root = session(tmp_path)
    (root / "generated/generated_script.py").unlink()
    def forbidden(*args, **kwargs):
        raise AssertionError("Deterministic checks must not create an LLM client")
    import importlib
    monkeypatch.setattr(importlib.import_module("local_agent.cli.main"), "create_llm_client", forbidden)
    assert main(["check", str(root), "--deterministic-only"]) == 0
    assert main(["check", str(root), "--ready"]) == 1


def test_legacy_script_reports_timestamp_limitation(tmp_path):
    root = session(tmp_path)
    report = check_ready(root)
    assert report.passed
    assert any("timestamps only" in w for w in report.warnings)
    write_stamp(root)
    assert check_ready(root).passed
    (root / "generated/generated_script.py").write_text("# pfit-sources: pending=true\n" + (root / "generated/generated_script.py").read_text())
    assert not check_ready(root).passed


def test_failed_translation_cannot_pass_as_legacy_script(tmp_path):
    from local_agent.agent.workflow import LocalWorkflow
    from local_agent.agent.prompts import PromptRenderer
    from local_agent.llm.fake import FakeLLMClient
    root = session(tmp_path)
    script = root / "generated/generated_script.py"
    script.write_text("def incomplete(): pass\n")
    workflow = LocalWorkflow(FakeLLMClient([]), PromptRenderer())
    assert not workflow._validate_generated_script(script, root, [])
    assert verify_stamp(root)[0] is False


def test_user_model_out_of_bounds_index_is_critical(tmp_path):
    root = session(tmp_path)
    (root / "generated/user_model.py").write_text("def loss(solution, dataset):\n    return dataset[:, 1]\n")
    from local_agent.agent.checks import check_session
    assert any("column index 1" in e for e in check_session(root).critical_errors)


def test_initial_time_cannot_skip_earlier_data(tmp_path):
    root = session(tmp_path)
    config = root / "inputs/user_input.yaml"
    config.write_text(config.read_text().replace("gradient_opt:", "gradient_opt:\n  initial_time: 0.5"))
    with pytest.raises(ValidationError, match="first dataset time"):
        validate_session(root)
