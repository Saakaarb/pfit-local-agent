import json
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from lib.utils.run_artifacts import load_restart_seed, scale_parameters, unscale_parameters
from lib.utils.sloppiness import analyze_sloppiness, write_sloppiness_report

jax.config.update("jax_enable_x64", True)
pytestmark = pytest.mark.unit


def reader():
    return SimpleNamespace(
        trainable_parameter_names=["linear", "rate"],
        min_axis_values=[-4.0, 0.01], max_axis_values=[8.0, 100.0],
        axis_logscale=[False, True], error_loss=1e10,
    )


def test_named_restart_remaps_order_and_uses_physical_log_values(tmp_path):
    (tmp_path / "final_parameters.json").write_text(json.dumps({"parameters": [
        {"name": "rate", "value": 10.0}, {"name": "linear", "value": 2.0},
    ]}))
    values = load_restart_seed(tmp_path, reader())
    np.testing.assert_allclose(values, [2.0, 10.0])
    np.testing.assert_allclose(scale_parameters(values, reader()), [0.0, 0.5])
    np.testing.assert_allclose(unscale_parameters([0.0, 0.5], reader()), values)


@pytest.mark.parametrize("values", [[2, np.nan], [2, -1], [9, 1], [1]])
def test_restart_rejects_invalid_physical_values(values):
    with pytest.raises(ValueError):
        scale_parameters(values, reader())


def test_restart_rejects_parameter_set_change(tmp_path):
    (tmp_path / "final_parameters.json").write_text(json.dumps({"parameters": [
        {"name": "other", "value": 2.0}, {"name": "rate", "value": 1.0},
    ]}))
    with pytest.raises(ValueError, match="names"):
        load_restart_seed(tmp_path, reader())


def test_legacy_restart_requires_explicit_order_assertion(tmp_path):
    (tmp_path / "final_design_point.csv").write_text("2\n10\n")
    with pytest.raises(ValueError, match="allow-legacy-seed"):
        load_restart_seed(tmp_path, reader())
    np.testing.assert_allclose(load_restart_seed(tmp_path, reader(), allow_legacy=True), [2, 10])


def quadratic(x):
    # u=(2+6*x0, 2*x1): curvature in u must not retain normalized-axis factors.
    u = jnp.array([2 + 6 * x[0], 2 * x[1]])
    return 0.5 * ((u[0] - 2) ** 2 * 4 + u[1] ** 2 * 1e-8)


@pytest.mark.parametrize("method", ["ad", "finite-difference"])
def test_curvature_coordinates_and_weak_mode(method):
    report = analyze_sloppiness(quadratic, [0, 0], reader(), method=method)
    np.testing.assert_allclose(report["hessian"], np.diag([4, 1e-8]), rtol=1e-6, atol=1e-12)
    np.testing.assert_allclose(report["eigenvalues"], [4, 1e-8], rtol=1e-6)
    assert report["coordinates"] == ["linear", "log10(rate)"]
    assert report["weak_modes"] == 1
    assert report["negative_modes"] == 0
    vectors = np.asarray(report["eigenvectors"])
    np.testing.assert_allclose(vectors.T @ vectors, np.eye(2))


def test_auto_fallback_is_recorded(monkeypatch):
    def unavailable(*args, **kwargs):
        raise NotImplementedError("second-order solver differentiation unavailable")
    monkeypatch.setattr(jax, "hessian", unavailable)
    report = analyze_sloppiness(quadratic, [0, 0], reader())
    assert report["method"] == "finite-difference"
    assert "NotImplementedError" in report["fallback_reason"]
    np.testing.assert_allclose(report["hessian"], np.diag([4, 1e-8]), atol=1e-9)


def test_boundary_stencils_do_not_evaluate_outside_bounds():
    def bounded_loss(x):
        return jnp.where(jnp.any(jnp.abs(x) > 1), 1e10, quadratic(x))
    report = analyze_sloppiness(bounded_loss, [-1, 1], reader(), method="finite-difference")
    assert report["finite_difference_stencils"] == ["forward", "backward"]
    assert report["parameters_at_bounds"] == ["linear", "rate"]
    np.testing.assert_allclose(report["hessian"], np.diag([4, 1e-8]), atol=1e-8)


def test_saddle_and_flat_curvature_are_not_identifiability_success():
    saddle = analyze_sloppiness(lambda x: x[0] ** 2 - x[1] ** 2, [0, 0], reader())
    assert saddle["negative_modes"] == 1
    assert saddle["verdict"] == "indefinite curvature"
    flat = analyze_sloppiness(lambda x: jnp.sum(x * 0), [0, 0], reader())
    assert flat["verdict"] == "flat curvature"
    assert flat["relative_eigenvalues"] is None


@pytest.mark.parametrize("value", [float("nan"), 1e10])
def test_diagnostic_failure_is_saved_without_raising(tmp_path, value):
    report = write_sloppiness_report(tmp_path, lambda x: jnp.sum(x * 0) + value, [0, 0], reader())
    assert report["status"] == "failed"
    assert json.loads((tmp_path / "sloppiness.json").read_text())["status"] == "failed"
    assert "failed" in (tmp_path / "sloppiness_report.txt").read_text()


def test_skipped_diagnostic_does_not_evaluate_loss(tmp_path):
    def forbidden(x):
        raise AssertionError("must not evaluate")
    assert write_sloppiness_report(tmp_path, forbidden, [0, 0], reader(), enabled=False)["status"] == "skipped"


def test_stiff_solver_curvature_matches_analytic_decay():
    import diffrax
    info = SimpleNamespace(trainable_parameter_names=["k"], min_axis_values=[0.1],
                           max_axis_values=[2.1], axis_logscale=[False], error_loss=1e10)
    def loss(x):
        k = 1.1 + x[0]
        sol = diffrax.diffeqsolve(
            diffrax.ODETerm(lambda t, y, args: -args * y), diffrax.Kvaerno5(),
            t0=0.0, t1=1.0, dt0=0.01, y0=jnp.array(1.0), args=k,
            stepsize_controller=diffrax.PIDController(rtol=1e-9, atol=1e-11),
            max_steps=1000,
        )
        return (sol.ys[-1] - 0.2) ** 2
    report = analyze_sloppiness(loss, [0.0], info, method="auto")
    e = np.exp(-1.1)
    expected = 4 * e ** 2 - 0.4 * e
    np.testing.assert_allclose(report["hessian"], [[expected]], rtol=2e-4, atol=1e-6)


def test_deployed_resolver_selects_latest_completed_seed(tmp_path):
    from lib.utils.run_store import resolve_run
    for name, status in [("run_1", "completed"), ("run_2", "completed"), ("run_3", "failed")]:
        run = tmp_path / "outputs" / name
        run.mkdir(parents=True)
        (run / "run_manifest.json").write_text(json.dumps({"status": status}))
        (run / "final_design_point.csv").write_text("1\n")
    assert resolve_run(tmp_path, require_seed=True).name == "run_2"
    with pytest.raises(FileNotFoundError):
        resolve_run(tmp_path, "run_3", require_seed=True)


def test_deployed_seed_loader_accepts_single_parameter_legacy_csv(tmp_path):
    from fit_gradient_only import load_init_guess
    (tmp_path / "outputs").mkdir()
    (tmp_path / "outputs" / "final_design_point.csv").write_text("2.5\n")
    np.testing.assert_allclose(load_init_guess(tmp_path), [2.5])
