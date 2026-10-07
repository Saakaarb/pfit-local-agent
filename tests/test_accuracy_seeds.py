import json
from types import SimpleNamespace
import numpy as np
import pytest
from lib.utils.run_artifacts import accuracy_seed_context, load_accuracy_seeds, seed_population


def prepare(tmp_path):
    (tmp_path/'inputs').mkdir()
    (tmp_path/'generated').mkdir()
    (tmp_path/'inputs/user_input.yaml').write_text('config')
    (tmp_path/'generated/user_model.py').write_text('model')
    (tmp_path/'inputs/data.csv').write_text('0,1')
    reader=SimpleNamespace(n_search_axes=2, user_input_dirname='inputs', experiments=[dict(filename='data.csv')])
    report=dict(code='accuracy_passed', validated_seeds=[dict(normalized_parameters=[0., .5])],
                seed_context=accuracy_seed_context(tmp_path, reader))
    (tmp_path/'generated/solver_accuracy.json').write_text(json.dumps(report))
    return reader


def test_seed_report_must_match_data_and_sources(tmp_path):
    reader=prepare(tmp_path)
    assert load_accuracy_seeds(tmp_path, reader) == [[0., .5]]
    (tmp_path/'inputs/data.csv').write_text('0,2')
    with pytest.raises(ValueError, match='stale'):
        load_accuracy_seeds(tmp_path, reader)


def test_seed_report_rejects_changed_model(tmp_path):
    reader=prepare(tmp_path)
    (tmp_path/'generated/user_model.py').write_text('new model')
    with pytest.raises(ValueError, match='stale'):
        load_accuracy_seeds(tmp_path, reader)


def test_seeding_keeps_exploration_and_rejects_out_of_bounds():
    population=np.ones((4,2))*.9
    seeds=[[0.,0.],[.1,.1],[.2,.2],[.3,.3]]
    result=seed_population(population.copy(), seeds)
    np.testing.assert_array_equal(result[:3], seeds[:3])
    np.testing.assert_array_equal(result[-1],population[-1])
    with pytest.raises(ValueError):seed_population(population, [[2.,0.]])


@pytest.mark.parametrize('invalid_loss', [1e10, float('nan'), float('inf')])
def test_invalid_refinement_winner_keeps_parameters_and_uses_population_tolerances(invalid_loss):
    import jax
    import jax.numpy as jnp
    from lib.utils.run_artifacts import refinement_start
    attempts = []
    winner = np.array([.9])
    def make_problem(rtol, atol):
        attempts.append((rtol, atol))
        # Captured tolerance simulates the runtime JIT closure.
        loss = jax.jit(lambda p: jnp.sum(p**2) if rtol >= 1e-6 else jnp.asarray(invalid_loss))
        return SimpleNamespace(_compute_loss=loss)
    problem, value, fallback = refinement_start(make_problem, winner, (1e-7, 1e-9), (1e-6, 1e-8), 1e10)
    assert fallback
    assert value == pytest.approx(.81)
    assert attempts == [(1e-7, 1e-9), (1e-6, 1e-8)]
    np.testing.assert_array_equal(winner, [.9])
    np.testing.assert_allclose(jax.grad(problem._compute_loss)(winner), [1.8])


def test_valid_refinement_winner_retains_requested_tolerances():
    from lib.utils.run_artifacts import refinement_start
    attempts = []
    def make_problem(rtol, atol):
        attempts.append((rtol, atol))
        return SimpleNamespace(_compute_loss=lambda p: p[0]**2)
    _, value, fallback = refinement_start(make_problem, [.9], (1e-7, 1e-9), (1e-6, 1e-8), 1e10)
    assert not fallback
    assert value == pytest.approx(.81)
    assert attempts == [(1e-7, 1e-9)]


@pytest.mark.parametrize('population', [None, (1e-6, 1e-8)])
def test_invalid_winner_is_never_replaced_with_another_seed(population):
    from lib.utils.run_artifacts import refinement_start
    make_problem = lambda rtol, atol: SimpleNamespace(_compute_loss=lambda p: 1e10)
    with pytest.raises(ValueError, match='Starting point'):
        refinement_start(make_problem, [.9], (1e-7, 1e-9), population, 1e10)


def test_run_snapshot_preserves_seeds_across_dataset_renaming(tmp_path):
    import shutil
    from lib.utils.run_artifacts import snapshot_accuracy_seeds
    source=tmp_path/'source';source.mkdir();reader=prepare(source)
    target=tmp_path/'snapshot';shutil.copytree(source,target)
    (target/'inputs/data.csv').rename(target/'inputs/dataset_1.csv')
    (target/'inputs/run_config.yaml').write_text('runtime config')
    runtime=SimpleNamespace(n_search_axes=2,user_input_dirname='inputs',experiments=[dict(filename='dataset_1.csv')])
    snapshot_accuracy_seeds(source,reader,target,runtime)
    assert load_accuracy_seeds(target,runtime)==[[0.,.5]]
    (target/'inputs/run_config.yaml').write_text('changed solver')
    with pytest.raises(ValueError,match='stale'):load_accuracy_seeds(target,runtime)


def test_run_snapshot_rejects_changed_data(tmp_path):
    import shutil
    from lib.utils.run_artifacts import snapshot_accuracy_seeds
    source=tmp_path/'source';source.mkdir();reader=prepare(source)
    target=tmp_path/'snapshot';shutil.copytree(source,target)
    (target/'inputs/data.csv').write_text('different measurements')
    with pytest.raises(ValueError,match='Data changed'):
        snapshot_accuracy_seeds(source,reader,target,reader)
