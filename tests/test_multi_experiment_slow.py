"""Bounded numerical acceptance tests for multi-experiment fitting."""
import json
from pathlib import Path
import numpy as np
import pytest
import yaml

from tests.test_multi_experiment import copy_fixture, change_config, make_problem
from lib.utils.run_artifacts import scale_parameters
from lib.utils.source_stamp import write_stamp
from local_agent.agent.validators import parse_input_yaml
from local_agent.core.fitting import run_driver
from local_agent.agent.diagnostics import diagnose_run
from analyze_fit import analyze_run

pytestmark = pytest.mark.slow


def test_decay_full_fit_restart_and_snapshot_reanalysis(tmp_path):
    session = copy_fixture(tmp_path)
    def settings(c):
        c['population_opt'].update(algorithm='DE', population_size=20, num_iters=15, processors=1, random_seed=7)
        c['gradient_opt'].update(num_iters=30, gradient_optimizer='lbfgs')
    change_config(session, settings)
    write_stamp(session)
    reader = parse_input_yaml(session / 'inputs/user_input.yaml')
    first = session / 'outputs/run_full'
    fit = run_driver(session, reader, first, sloppiness=False)
    np.testing.assert_allclose(fit, [1., .3], rtol=.02, atol=.003)
    summary = json.loads((first / 'fit_summary.json').read_text())
    assert len(summary['experiment_losses']) == 2
    assert summary['final_loss'] == pytest.approx(np.mean(summary['experiment_losses']))
    second = session / 'outputs/run_restart'
    run_driver(session, reader, second, gradient_only=True, from_run=first.name, sloppiness_method='finite-difference')
    assert not (second / 'de_fitting.log').exists()
    manifest = json.loads((second / 'run_manifest.json').read_text())
    assert len(manifest['experiments']) == 2
    assert manifest['experiments'][1]['initial_conditions'] == {'A':2., 'B':.5}
    for index in (1,2):
        assert (second / f'result_solution_exp{index}.csv').exists()
        assert (second / f'snapshot/inputs/dataset_{index}.csv').exists()
    before = json.loads((second / 'sloppiness.json').read_text())
    assert before['status'] == 'ok'
    # Remove live inputs: historical analysis must still use both saved records.
    (session / 'inputs/decay_run_A.csv').unlink()
    (session / 'inputs/decay_run_B.csv').unlink()
    report = analyze_run(session, second.name, method='finite-difference')
    assert report is not None
    np.testing.assert_allclose(report['hessian'], before['hessian'], rtol=1e-8, atol=1e-8)
    assert report['loss'] == pytest.approx(before['loss'])
    diagnosis = diagnose_run(session, second.name).read_text()
    assert 'Experiments: 2' in diagnosis and 'decay_run_B.csv' in diagnosis
    assert 'result_solution.csv: missing' not in diagnosis
    (second / 'result_solution_exp2.csv').unlink()
    assert 'Missing experiment result: result_solution_exp2.csv' in diagnose_run(session, second.name).read_text()
    assert (first / 'final_parameters.json').exists()


def test_changing_only_second_record_changes_fitted_optimum(tmp_path):
    from tests.test_readiness import session as make_session
    from tests.test_fitting_engine_slow import GENERATED_SCRIPT
    session = make_session(tmp_path, '0,1\n1,1\n')
    (session / 'inputs/second.csv').write_text('0,3\n.5,3\n1,3\n')
    def settings(c):
        import copy
        exp = copy.deepcopy(c['experiments'][0]); exp['data_file']='second.csv'; c['experiments'].append(exp)
        c['model']['trainable_parameters'][0].update(min_val=0.,max_val=6.,logscale=False)
        c['population_opt'].update(num_iters=2, random_seed=8)
        c['gradient_opt']['num_iters']=15
    change_config(session, settings)
    script=GENERATED_SCRIPT.replace('target = jnp.array([2.0])','target = constants["dataset"][:, 0]')
    (session/'generated/user_model.py').write_text(script)
    (session/'generated/generated_script.py').write_text(script)
    write_stamp(session)
    reader=parse_input_yaml(session/'inputs/user_input.yaml')
    first=session/'outputs/first'
    fit=run_driver(session,reader,first,sloppiness_method='ad')
    np.testing.assert_allclose(fit,[2.],atol=1e-6)
    old=json.loads((first/'sloppiness.json').read_text())
    np.testing.assert_allclose(old['hessian'],[[2.]],atol=1e-7)
    (session/'inputs/second.csv').write_text('0,5\n.5,5\n1,5\n')
    second=session/'outputs/second'
    refined=run_driver(session,reader,second,gradient_only=True,from_run=first.name,sloppiness=False)
    np.testing.assert_allclose(refined,[3.],atol=1e-6)
    assert json.loads((second/'fit_summary.json').read_text())['final_loss'] == pytest.approx(4.)
    historic=analyze_run(session,first.name,method='ad')
    assert historic['loss'] == pytest.approx(1.)
    assert json.loads((first/'run_manifest.json').read_text())['experiments'][1]['sha256'] != json.loads((second/'run_manifest.json').read_text())['experiments'][1]['sha256']


def test_sneyd_all_nine_reference_trajectories_and_mean_rmse(tmp_path):
    session=copy_fixture(tmp_path,'sneyd_ipr')
    reader,module,problem=make_problem(session)
    parameters=np.loadtxt(session/'reference_parameters.csv',delimiter=',')
    point=scale_parameters(parameters,reader)
    losses=[]
    for index,c in enumerate(problem.constants_list):
        times,solution,result=module._integrate_system(c,point)
        solution=np.asarray(solution)
        assert np.all(np.isfinite(solution))
        np.testing.assert_allclose(solution[:,-2:],np.tile(reader.get_y0(index)[-2:],(len(times),1)),atol=1e-8)
        probability=(.9*solution[:,4]+.1*solution[:,0])**4
        manual=np.sqrt(np.mean((probability-np.asarray(c['dataset'])[:,0])**2))
        loss=float(module._compute_loss_problem(c,point))
        assert loss == pytest.approx(manual,rel=1e-7,abs=1e-9)
        losses.append(loss)
    assert float(problem._compute_loss(point)) == pytest.approx(np.mean(losses),rel=1e-7)
    reader.output_dir=tmp_path/'sneyd_results';reader.output_dir.mkdir()
    problem.write_problem_result(point,reader,label='result')
    assert len(list(reader.output_dir.glob('result_solution_exp*.csv'))) == 9
    # Bounded optimization smoke: one refinement step from the saved reference
    # point, with every experiment present. This is not a recovery benchmark.
    change_config(session,lambda c:c['gradient_opt'].update(num_iters=1,gradient_optimizer='lbfgs'))
    from lib.utils.helper_functions import fit_gradient_only_system
    output=tmp_path/'sneyd_refinement';output.mkdir()
    fit_gradient_only_system(session/'inputs/user_input.yaml',output,session/'generated',session,parameters,sloppiness=False)
    summary=json.loads((output/'fit_summary.json').read_text())
    assert len(summary['experiment_losses']) == 9
    assert summary['final_loss'] <= summary['seed_loss']
