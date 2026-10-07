import numpy as np
from local_agent.agent.solver_selection import dataset_solver_selection


def record(t, *ys, index=1):
    return dict(index=index, t_eval=np.asarray(t), dataset=np.column_stack(ys),
                columns=[dict(name='time')]+[dict(name=f'y{i}', observes=f'y{i}') for i in range(len(ys))])


def select(*records):
    return dataset_solver_selection(records, {'y0', 'y1'})


def test_simple_linear_signals_are_unit_invariant():
    t = np.linspace(0, 10, 101)
    a = select(record(t, t, 1000*t+273))
    assert a['integrator'] == 'Tsit5'
    assert np.isclose(a['time_scale_ratio'], 1)
    assert select(record(t*60, t))['integrator'] == 'Tsit5'


def test_separated_time_scales_across_experiments_choose_implicit():
    t = np.linspace(0, 1, 101)
    a = select(record(t, t), record(t*1000, t, index=2))
    assert a['integrator'] == 'Kvaerno5'
    assert a['time_scale_ratio'] > 999


def test_fast_and_slow_segments_in_one_column():
    t = np.linspace(0, 10, 1001)
    y = np.where(t<5, t, 5+(t-5)*1000)
    assert select(record(t, y))['integrator'] == 'Kvaerno5'


def test_inconclusive_flat_sparse_or_missing_data_defaults_implicit():
    t = np.arange(20.)
    for r in [record(t, t*0), record(t[:5], t[:5]), record(t, np.full(20, np.nan))]:
        assert select(r)['integrator'] == 'Kvaerno5'


def test_forcing_uncertainty_and_auxiliary_are_excluded():
    t = np.arange(100.)
    r = record(t, t, t**10)
    for metadata in [dict(role='forcing'), dict(uncertainty_of='y0'), dict(role='auxiliary')]:
        r['columns'][2] = dict(name='y1', **metadata)
        a = select(r)
        assert a['integrator'] == 'Tsit5'
        assert len(a['columns']) == 1
