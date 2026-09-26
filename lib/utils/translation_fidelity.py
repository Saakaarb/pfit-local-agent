"""Numerical source/JAX acceptance probes; counterexamples trigger bounded repair.

These tests check representative points, not global scientific equivalence.
The supplied Python model is authoritative; literature extraction is separate.
"""
import importlib.util
import inspect
import json
from pathlib import Path
import numpy as np
from lib.utils.experiments import load_experiments, experiment_constants
from lib.utils.run_artifacts import parameter_axes, unscale_parameters


def compare_source_and_jax(module, reader, session, script):
    from local_agent.agent.workflow import _source_function_is_placeholder
    from local_agent.agent.session_spec import load_session_spec
    session,script=Path(session),Path(script)
    source_path=session/'generated/user_model.py'
    report={'status':'skipped','points':[], 'limitations':[],
            'rtol':1e-6,'atol':1e-8,'scope':'Source Python versus JAX at sampled parameters/states; not paper-equation verification'}
    path=script.parent/'translation_fidelity.json'
    def save():path.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    if not source_path.exists() or not hasattr(module,'_integrate_system'):
        report['limitations'].append('Source model or generated integration interface unavailable.');save();return
    spec=load_session_spec(session/'inputs/user_input.yaml')
    source=source_path.read_text()
    try:
        module_spec=importlib.util.spec_from_file_location('pfit_source_fidelity',source_path)
        original=importlib.util.module_from_spec(module_spec);module_spec.loader.exec_module(original)
        functions={}
        for name,count in [('user_defined_system',6),('_compute_loss_problem',5),('writeout_description',5)]:
            fn=getattr(original,name,None)
            if name == "writeout_description" and fn is None:
                fn=getattr(original,"write_problem_result",None)
            if _source_function_is_placeholder(source,spec,name):
                report['limitations'].append(f'{name}: unchanged framework placeholder has no supplied semantics.')
            elif fn is None:
                report['limitations'].append(f'{name}: not provided in source.')
            elif len(inspect.signature(fn).parameters)!=count:
                raise ValueError(f'Source {name} must accept {count} contract arguments for numerical comparison')
            else: functions[name]=fn
        if not functions:
            save();return
        lo,hi,logs=parameter_axes(reader)
        for record in load_experiments(session,reader):
            c=experiment_constants(record,reader);c.update(min_limits=lo,max_limits=hi,is_logscale=logs)
            for offset in (0.,.1):
                from diffrax import RESULTS
                point=np.full(reader.n_search_axes,offset)
                time,solution,result=module._integrate_system(c,point)
                time,solution=np.asarray(time),np.asarray(solution)
                entry={'experiment':record['index'],'parameter_offset':offset,'components':[]}
                report['points'].append(entry)
                if result != RESULTS.successful or not np.all(np.isfinite(solution)):
                    if offset==0.:raise ValueError(f'Experiment {record["index"]}: integration failed at midpoint')
                    entry['skipped']='Perturbed parameter solve failed; midpoint is still checked.'
                    report['limitations'].append(f'Experiment {record["index"]}: second parameter probe could not be integrated.')
                    continue
                values=unscale_parameters(point,reader)
                parameters=dict(zip(reader.trainable_parameter_names,values))
                args=(time,solution,c['dataset'],parameters,c['fixed_parameters'])
                def compare(label,expected,actual):
                    a,b=np.asarray(expected),np.asarray(actual)
                    if a.shape!=b.shape or not np.all(np.isfinite(a)) or not np.all(np.isfinite(b)) or not np.allclose(a,b,rtol=1e-6,atol=1e-8):
                        delta=float(np.nanmax(np.abs(a-b))) if a.shape==b.shape and a.size and np.any(np.isfinite(a-b)) else None
                        raise ValueError(f'Experiment {record["index"]}, normalized parameter offset {offset}: {label} differs from source Python (shapes {a.shape} vs {b.shape}, max absolute difference {delta}). Preserve the source expression/metric/output columns; do not change source intent.')
                    entry['components'].append(label)
                if 'user_defined_system' in functions:
                    # Include a non-initial state so zero initial values cannot hide coefficient errors.
                    for index in sorted({0,len(time)//2,len(time)-1}):
                        y=solution[index]
                        expected=functions['user_defined_system'](time[index],y,parameters,c['fixed_parameters'],c['dataset'],c['t_eval'])
                        actual=module.user_defined_system(time[index],y,{'constants':c,'trainable_variables':point})
                        expected,actual=np.asarray(expected),np.asarray(actual)
                        if expected.shape != actual.shape:
                            compare(f'RHS shape at row {index}',expected,actual)
                        for axis,name in enumerate(reader.integrated_variable_names):
                            compare(f'RHS for {name} at row {index}',expected[axis],actual[axis])
                if '_compute_loss_problem' in functions:
                    expected=functions['_compute_loss_problem'](*args)
                    compare('loss',expected,module._compute_loss_problem(c,point))
                if 'writeout_description' in functions:
                    expected=functions['writeout_description'](*args)
                    actual=module._write_problem_result(c,point)
                    # NaN observations may be retained for display; positions must agree.
                    a,b=np.asarray(expected),np.asarray(actual)
                    if a.shape==b.shape and np.array_equal(np.isnan(a),np.isnan(b)):
                        expected=np.where(np.isnan(a),0,a);actual=np.where(np.isnan(b),0,b)
                    compare('writeout',expected,actual)
        report['status']='passed'
        save()
    except Exception as exc:
        report['status']='failed';report['error']=f'{type(exc).__name__}: {exc}';save()
        raise ValueError('Numerical source/JAX fidelity check failed: '+str(exc)) from exc
