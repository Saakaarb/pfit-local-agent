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


def test_invalid_refinement_winner_falls_back_to_best_validated_seed():
    from lib.utils.run_artifacts import refinement_start
    def loss(p):return 1e10 if p[0] > .5 else float(p[0]**2)
    point, value, fallback=refinement_start(loss, [.9], [[.3],[.1],[.8]], 1e10)
    np.testing.assert_array_equal(point,[.1])
    assert value == pytest.approx(.01)
    assert fallback
    point, value, fallback=refinement_start(loss, [.2], [[.1]], 1e10)
    assert not fallback  # Preserve a valid global-search winner.
    with pytest.raises(ValueError):refinement_start(loss, [.9], [], 1e10)
