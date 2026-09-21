from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from lib.algorithms.DE.classes import FitParamsDE


pytestmark = pytest.mark.unit


class QuadraticProblem:
    def compute_all_losses(self, population):
        return np.sum((population - 0.25) ** 2, axis=1)


def make_reader(tmp_path, *, seed=123):
    return SimpleNamespace(
        n_search_axes=2,
        axis_logscale=[0, 1],
        min_axis_values=[0.0, 0.1],
        max_axis_values=[10.0, 10.0],
        n_particles=6,
        n_iters_pop=1,
        output_dir=tmp_path,
        random_seed=seed,
    )


def test_de_scales_and_unscales_linear_and_log_axes(tmp_path):
    optimizer = FitParamsDE(make_reader(tmp_path), QuadraticProblem())

    unscaled = optimizer.unscale_design_point(np.array([0.0, 0.0]))

    assert np.allclose(unscaled, np.array([5.0, 1.0]))


def test_de_seed_makes_initial_run_deterministic(tmp_path):
    log_a = tmp_path / "a.log"
    log_b = tmp_path / "b.log"

    a = FitParamsDE(make_reader(tmp_path, seed=77), QuadraticProblem())
    b = FitParamsDE(make_reader(tmp_path, seed=77), QuadraticProblem())

    pos_a, cost_a = a.run(log_a)
    pos_b, cost_b = b.run(log_b)

    assert np.allclose(pos_a, pos_b)
    assert cost_a == pytest.approx(cost_b)


def test_de_writes_iteration_log(tmp_path):
    log_path = tmp_path / "de_fitting.log"
    optimizer = FitParamsDE(make_reader(tmp_path), QuadraticProblem())

    optimizer.run(log_path)

    log = log_path.read_text()
    assert "Population size: 6" in log
    assert "Max iterations: 1" in log
