import pytest
import json
import numpy as np

from fit_parameters import run_driver
from lib.utils.helper_functions import get_input_reader


pytestmark = pytest.mark.slow


GENERATED_SCRIPT = """
import jax.numpy as jnp


def user_defined_system(t, y, other_args):
    return y


def _integrate_system(constants, trainable_variables):
    return constants["t_eval"], constants["dataset"], None


def _unscale_value(value, min_val, max_val):
    return ((value + 1.0) / 2.0) * (max_val - min_val) + min_val


def _compute_loss_problem(constants, trainable_variables):
    target = jnp.array([2.0])
    unscaled = _unscale_value(
        trainable_variables,
        constants["min_limits"],
        constants["max_limits"],
    )
    return jnp.mean((unscaled - target) ** 2)


def _write_problem_result(constants, trainable_variables):
    return jnp.column_stack((constants["t_eval"], constants["dataset"]))
"""


def test_de_global_search_hands_off_to_node(tmp_path, monkeypatch):
    session = tmp_path / "de_session"
    inputs = session / "inputs"
    generated = session / "generated"
    inputs.mkdir(parents=True)
    generated.mkdir()

    (inputs / "data.csv").write_text("0.0,1.0\n1.0,1.0\n")
    (inputs / "user_input.yaml").write_text(
        """experiments:
  - data_file: data.csv
    columns:
      - {name: time}
      - {name: y, observes: y}

model:
  trainable_parameters:
    - {name: k, min_val: 0.0, max_val: 4.0, logscale: false}
  fixed_parameters: []
  integrated_variables:
    - {name: y, init_val: 1.0}

population_opt:
  population_size: 4
  num_iters: 1
  processors: 1
  algorithm: DE
  random_seed: 7

gradient_opt:
  num_iters: 1
  stepsize_rtol: [1e-7]
  stepsize_atol: [1e-9]
  initial_timestep: 1e-6
  max_steps: 100
  init_value_lr: 1e-4
  end_value_lr: 1e-5
  transition_steps_lr: 10
  decay_rate_lr: 0.9

output:
  write_results: true
"""
    )
    (generated / "user_model.py").write_text(GENERATED_SCRIPT)
    (generated / "generated_script.py").write_text(GENERATED_SCRIPT)
    from lib.utils.source_stamp import write_stamp
    write_stamp(session)

    reader = get_input_reader(inputs / "user_input.yaml")
    output_dir = session / "outputs" / "de_regression"

    run_driver(session, reader, output_dir_override=output_dir)

    assert (output_dir / "de_fitting.log").exists()
    assert (output_dir / "NODE_fitting.log").exists()
    assert (output_dir / "final_design_point.csv").exists()
    assert (output_dir / "result_solution.csv").exists()

    assert json.loads((output_dir / "sloppiness.json").read_text())["status"] == "ok"
    before = {p.name: p.read_bytes() for p in output_dir.iterdir() if p.is_file()}
    from lib.algorithms.DE.classes import FitParamsDE
    def forbidden_global(*args, **kwargs):
        raise AssertionError("Global search must not run for a gradient-only restart")
    monkeypatch.setattr(FitParamsDE, "run", forbidden_global)
    restarted = session / "outputs" / "restarted"
    run_driver(session, reader, output_dir_override=restarted, from_run=output_dir.name)
    assert not (restarted / "de_fitting.log").exists()
    assert (restarted / "NODE_fitting.log").exists()
    manifest = json.loads((restarted / "run_manifest.json").read_text())
    assert manifest["mode"] == "gradient-only"
    assert manifest["source_run"] == output_dir.name
    assert manifest["status"] == "completed"
    assert (restarted / "snapshot" / "generated" / "generated_script.py").exists()
    summary = json.loads((restarted / "fit_summary.json").read_text())
    assert summary["final_loss"] <= summary["seed_loss"]
    np.testing.assert_allclose(np.loadtxt(restarted / "final_design_point.csv"), 2.0, atol=1e-5)
    assert before == {p.name: p.read_bytes() for p in output_dir.iterdir() if p.is_file()}
    with pytest.raises(FileExistsError):
        run_driver(session, reader, output_dir_override=output_dir)

    # Standalone analysis uses saved model/data even after the working session changes.
    from analyze_fit import analyze_run
    (generated / "generated_script.py").write_text("raise RuntimeError('working model was edited')\n")
    (inputs / "data.csv").write_text("invalid working data\n")
    analysis = analyze_run(session, restarted.name)
    assert analysis["status"] == "ok"
    assert (restarted / "sloppiness_spectrum.png").exists()
    np.testing.assert_allclose(analysis["hessian"], [[2.0]], atol=1e-7)
