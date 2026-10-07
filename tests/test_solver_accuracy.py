import json
import threading
from types import SimpleNamespace

import numpy as np
import pytest

from local_agent.agent.solver_accuracy import assess_solver_accuracy, prediction_error, select_probes
from local_agent.agent.validators import SolverValidationError, _validate_raw_settings


def candidates():
    return [dict(sample=i,successful=True,normalized_parameters=[p],experiment_losses=[loss],integration_steps=steps)
            for i,p,loss,steps in [(0,0.,1.,1),(1,.1,3.,100),(2,-1.,2.,2),(3,1.,2.,2),(4,.2,2.,2)]]


def test_probe_selection_best_hardest_then_diverse():
    assert [p['sample'] for p in select_probes(candidates())]==[0,1,2,3]
    assert len(select_probes(candidates()[:1]))==1


def test_measurement_uncertainty_controls_allowance():
    assert prediction_error([1.005],[1.],[1.],[1.],1e-3,1e-6,.01)<1
    assert prediction_error([1.005],[1.],[1.],[.01],1e-3,1e-6,.01)>1
    assert prediction_error([0.],[0.],[0.],None,1e-3,1e-6,.01)==0
    assert prediction_error([float('nan')],[1.],[float('nan')],None,1e-3,1e-6,.01) is None


def run_check(tmp_path, monkeypatch, factor=1., fail_reference=False, derived=False, workers=2, barrier=None, calibration=None, population=None):
    from diffrax import RESULTS
    points=candidates()[:4]
    (tmp_path/'solver_coverage.json').write_text(json.dumps([dict(sample_count=32,candidates=points)]))
    reader=SimpleNamespace(max_steps=10000,integrator='Tsit5',stepsize_rtol=[1e-3],stepsize_atol=[1e-6],
        min_axis_values=[0.],max_axis_values=[1.],axis_logscale=[False],trainable_parameter_names=['k'],
        integrated_variable_names=['y'],observable_names=['z'] if derived else [],error_loss=1e10)
    monkeypatch.setattr('lib.utils.experiments.experiment_constants',lambda r,reader:dict(fixed_parameters={},max_steps=10000))
    records=[dict(index=1,columns=[dict(name='time'),dict(name='value',observes='z' if derived else 'y')],dataset=np.ones((2,1)))]
    threads=set();calls=[];lock=threading.Lock()
    def integrate(c,p):
        with lock:
            first=threading.get_ident() not in threads
            threads.add(threading.get_ident());calls.append(float(c['stepsize_rtol'][0]))
        if barrier is not None and first:barrier.wait(timeout=10)
        rtol=c['stepsize_rtol'][0]
        result=RESULTS.max_steps_reached if fail_reference and rtol<1e-3 else RESULTS.successful
        return np.array([0.,1.]),np.full((2,1),1+rtol*factor),result,dict(num_steps=5)
    module=SimpleNamespace(_integrate_system_with_stats=integrate,_compute_loss_value=lambda c,p,t,y:1.)
    if derived and derived!='missing':module._observables=lambda y,p,f:dict(z=y[:,0]*1000)
    result=assess_solver_accuracy(module,reader,tmp_path,tmp_path/'generated.py',records,
        dict(solver_accuracy_workers=workers),population or {},calibration or {})
    return result,threads,calls


def test_parallel_probes_and_prediction_pass(tmp_path,monkeypatch):
    monkeypatch.setattr('os.sched_getaffinity',lambda pid:{0,1})
    report,threads,calls=run_check(tmp_path,monkeypatch,barrier=threading.Barrier(2))
    assert report['code']=='accuracy_passed'
    assert len(threads)==2
    assert len(calls)==8
    assert report['probe_samples']==[0,1,2,3]


def test_profiles_reuse_solves_without_reintegrating_for_loss(tmp_path,monkeypatch):
    _,_,calls=run_check(tmp_path,monkeypatch,factor=0.,calibration=dict(stepsize_rtol=[.01],stepsize_atol=[1e-5]))
    assert len(calls)==12  # 3 tolerance levels x 4 probes, not 4 levels


@pytest.mark.parametrize('derived',[False,True])
def test_prediction_mismatch_is_rejected_even_with_identical_losses(tmp_path,monkeypatch,derived):
    with pytest.raises(SolverValidationError) as exc:
        run_check(tmp_path,monkeypatch,factor=10.,derived=derived)
    assert exc.value.diagnostics['code']=='accuracy_failed'
    assert json.loads((tmp_path/'solver_accuracy.json').read_text())['code']=='accuracy_failed'


def test_failed_tighter_reference_is_inconclusive(tmp_path,monkeypatch):
    with pytest.raises(SolverValidationError) as exc:
        run_check(tmp_path,monkeypatch,fail_reference=True)
    assert exc.value.diagnostics['code']=='accuracy_inconclusive'
    assert exc.value.diagnostics['failure_cause']=='step_limit'
    assert exc.value.diagnostics['failure']['rtol']==[.0001]
    assert exc.value.diagnostics['failure']['stats']['num_steps']==5


@pytest.mark.parametrize('key,value',[('solver_accuracy_workers',0),('solver_accuracy_workers',5),
    ('solver_accuracy_workers',True),('solver_accuracy_check','true'),('solver_accuracy_rtol',-1),
    ('solver_accuracy_atol_scale',float('nan')),('solver_accuracy_uncertainty_fraction',0)])
def test_bad_accuracy_settings_rejected(key,value):
    with pytest.raises(ValueError):_validate_raw_settings({'gradient_opt':{key:value}})


def test_missing_derived_prediction_interface_blocks_accuracy(tmp_path,monkeypatch):
    with pytest.raises(SolverValidationError) as exc:
        run_check(tmp_path,monkeypatch,derived='missing')
    assert exc.value.diagnostics['code']=='accuracy_interface_error'


def test_worker_count_respects_cpu_affinity(tmp_path,monkeypatch):
    monkeypatch.setattr('os.sched_getaffinity',lambda pid:{0})
    report,threads,_=run_check(tmp_path,monkeypatch,workers=4)
    assert report['workers']==1
    assert len(threads)==1


def test_failed_automatic_de_relaxation_retains_validated_refinement(tmp_path,monkeypatch):
    calibration=dict(status='accepted',stepsize_rtol=[.01],stepsize_atol=[1e-5])
    report,_,_=run_check(tmp_path,monkeypatch,calibration=calibration)
    assert report['code']=='accuracy_passed'
    assert report['population_relaxation_rejected']['code']=='accuracy_failed'
    saved=json.loads((tmp_path/'tolerance_calibration.json').read_text())
    assert saved['de_status']=='retained'
    assert 'stepsize_rtol' not in saved


def test_explicit_population_settings_are_not_silently_overwritten(tmp_path,monkeypatch):
    with pytest.raises(SolverValidationError) as exc:
        run_check(tmp_path,monkeypatch,population=dict(stepsize_rtol=[.01],stepsize_atol=[1e-5]))
    assert exc.value.diagnostics['code']=='accuracy_failed'
    assert not (tmp_path/'tolerance_calibration.json').exists()
