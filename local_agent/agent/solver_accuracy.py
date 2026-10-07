"""Parallel, bounded forward-accuracy probes against tighter integrations."""
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path

import numpy as np


def select_probes(candidates, limit=4):
    feasible = [c for c in candidates if c['successful']]
    if not feasible:
        return []
    selected = [min(feasible, key=lambda c: np.mean(c['experiment_losses']))]
    hardest = max(feasible, key=lambda c: c.get('integration_steps', 0))
    if hardest['sample'] != selected[0]['sample']:
        selected.append(hardest)
    while len(selected) < min(limit, len(feasible)):
        remaining = [c for c in feasible if c['sample'] not in {p['sample'] for p in selected}]
        selected.append(max(remaining, key=lambda c: min(np.linalg.norm(
            np.asarray(c['normalized_parameters'])-p['normalized_parameters']) for p in selected)))
    return selected


def prediction_error(candidate, reference, observed, sigma, rtol, scale_fraction, sigma_fraction):
    candidate, reference, observed = map(np.asarray, (candidate, reference, observed))
    if candidate.shape != observed.shape or reference.shape != observed.shape:
        raise ValueError('Prediction shape must match measurement rows')
    mask = np.isfinite(observed)
    if not mask.any():return None
    a,b,data = candidate[mask],reference[mask],observed[mask]
    if not np.all(np.isfinite(a)) or not np.all(np.isfinite(b)):return float('inf')
    scale=max(float(np.max(np.abs(data))),float(np.max(np.abs(b))))
    allowance=scale_fraction*scale+rtol*np.abs(b)
    if sigma is not None:
        sigma=np.asarray(sigma)[mask]
        allowance=np.where(np.isfinite(sigma)&(sigma>0),sigma_fraction*sigma,allowance)
    difference=np.abs(a-b)
    ratio=np.divide(difference,allowance,out=np.full_like(difference,np.inf,dtype=float),where=allowance>0)
    return float(np.max(np.where((allowance==0)&(difference==0),0.,ratio)))


def assess_solver_accuracy(module, reader, session, script, records, gradient, population, calibration):
    from diffrax import RESULTS
    from lib.utils.experiments import experiment_constants
    from lib.utils.run_artifacts import parameter_axes, unscale_parameters
    from local_agent.agent.tolerance_calibration import stable_losses
    from local_agent.agent.validators import SolverValidationError

    folder=Path(script).parent
    coverage=json.loads((folder/'solver_coverage.json').read_text())[-1]
    probes=select_probes(coverage['candidates'])
    available=len(os.sched_getaffinity(0)) if hasattr(os,'sched_getaffinity') else (os.cpu_count() or 1)
    workers=min(gradient.get('solver_accuracy_workers',4),available,max(1,len(probes)))
    report=dict(code='accuracy_passed', integrator=reader.integrator, max_steps=reader.max_steps,
        sample_count=coverage['sample_count'],probe_samples=[c['sample'] for c in probes],workers=workers,
        reference_factor=.1,prediction_rtol=gradient.get('solver_accuracy_rtol',1e-3),
        prediction_atol_scale=gradient.get('solver_accuracy_atol_scale',1e-6),
        uncertainty_fraction=gradient.get('solver_accuracy_uncertainty_fraction',.01),comparisons=[],
        scope='Forward predictions and loss stability on observed rows; not gradient accuracy or a global guarantee')
    def finish(code,reason,failure=None):
        report.update(code=code,result=reason)
        if code == 'accuracy_passed':
            approved = [probe for probe, result in zip(probes, results)
                        if 'refinement' in result[1] and probe['sample'] not in excluded]
            approved.sort(key=lambda probe: np.mean(probe['experiment_losses']))
            report['validated_seeds'] = [dict(sample=p['sample'],
                normalized_parameters=p['normalized_parameters'],
                experiment_losses=p['experiment_losses']) for p in approved]
        if failure is not None:
            report['failure_cause'] = failure['code']
            report['failure'] = failure
        path=folder/'solver_accuracy.json';temp=path.with_suffix('.tmp')
        temp.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n');temp.replace(path)
        if code!='accuracy_passed':raise SolverValidationError(report)
        return report
    if not probes:return finish('accuracy_inconclusive','No feasible accuracy probes')
    lo,hi,logs=parameter_axes(reader)
    grad_rtol=np.asarray(calibration.get('gradient_stepsize_rtol',reader.stepsize_rtol),dtype=float)
    grad_atol=np.asarray(calibration.get('gradient_stepsize_atol',reader.stepsize_atol),dtype=float)
    pop_rtol=np.asarray(calibration.get('stepsize_rtol',population.get('stepsize_rtol') if population.get('stepsize_rtol') is not None else grad_rtol),dtype=float)
    pop_atol=np.asarray(calibration.get('stepsize_atol',population.get('stepsize_atol') if population.get('stepsize_atol') is not None else grad_atol),dtype=float)
    profiles=[('refinement',grad_rtol,grad_atol)]
    if not (np.array_equal(grad_rtol,pop_rtol) and np.array_equal(grad_atol,pop_atol)):
        profiles.append(('population',pop_rtol,pop_atol))

    def check_probe(probe):
        # One worker owns this parameter vector, sharing solves between profiles.
        cache={};items=[];losses={}
        def solve(record,rtol,atol):
            # Treat roundoff from multiplying/dividing by ten as the same tolerance.
            key=(record['index'],tuple(format(float(v), '.15g') for v in np.ravel(rtol)),
                 tuple(format(float(v), '.15g') for v in np.ravel(atol)))
            if key in cache:return cache[key]
            c=experiment_constants(record,reader)
            c.update(min_limits=lo,max_limits=hi,is_logscale=logs,stepsize_rtol=rtol,stepsize_atol=atol)
            point=np.asarray(probe['normalized_parameters'])
            ts,ys,result,stats=module._integrate_system_with_stats(c,point)
            if result!=RESULTS.successful or not np.all(np.isfinite(ys)):
                item['failure'] = dict(
                    code='step_limit' if result == RESULTS.max_steps_reached else
                         ('integration_failure' if result != RESULTS.successful else 'nonfinite_solution'),
                    result=str(result), stats={k:int(v) for k,v in stats.items()},
                    sample=probe['sample'], experiment=record['index'],
                    rtol=np.asarray(rtol).tolist(), atol=np.asarray(atol).tolist())
                raise RuntimeError(f'Integration did not complete with finite states: {result}')
            loss=module._compute_loss_value(c,point,ts,ys) if hasattr(module,'_compute_loss_value') else module._compute_loss_problem(c,point)
            loss=np.asarray(loss)
            if loss.shape!=() or not np.isfinite(loss) or loss==reader.error_loss:
                raise RuntimeError('Invalid loss in accuracy evaluation')
            ys=np.asarray(ys)
            predictions={n:ys[:,i] for i,n in enumerate(reader.integrated_variable_names)}
            derived=any((col.get('observes') or col['name']) in reader.observable_names for col in record['columns'][1:])
            if derived:
                if not hasattr(module,'_observables'):
                    raise ValueError('Declared derived observables require the _observables prediction helper')
                parameters=dict(zip(reader.trainable_parameter_names,unscale_parameters(point,reader)))
                predictions.update(module._observables(ys,parameters,c['fixed_parameters']))
            cache[key]=(predictions,float(loss),{k:int(v) for k,v in stats.items()})
            return cache[key]
        try:
            for name,rtol,atol in profiles:
                pairs=[]
                for record in records:
                    item=dict(profile=name,sample=probe['sample'],experiment=record['index'],
                              rtol=rtol.tolist(),atol=atol.tolist(),observables=[])
                    items.append(item)
                    current,loss,stats=solve(record,rtol,atol)
                    reference,ref_loss,ref_stats=solve(record,rtol*.1,atol*.1)
                    item.update(loss=loss,reference_loss=ref_loss,stats=stats,reference_stats=ref_stats)
                    pairs.append((loss,ref_loss));checked=0
                    for i,column in enumerate(record['columns'][1:]):
                        target=column.get('observes') or column['name']
                        if column.get('role')=='forcing' or column.get('uncertainty_of'):continue
                        if target not in reader.integrated_variable_names+reader.observable_names:continue
                        if target not in current or target not in reference:raise ValueError(f'Missing prediction: {target}')
                        sigma_index=next((j for j,col in enumerate(record['columns'][1:]) if col.get('uncertainty_of')==column['name']),None)
                        error=prediction_error(current[target],reference[target],record['dataset'][:,i],
                            None if sigma_index is None else record['dataset'][:,sigma_index],
                            report['prediction_rtol'],report['prediction_atol_scale'],report['uncertainty_fraction'])
                        if error is None:continue
                        checked+=1
                        item['observables'].append(dict(name=target,max_normalized_difference=error if np.isfinite(error) else None))
                        if error>1:return items,losses,'accuracy_failed',f'{name}: {target} disagrees with tighter reference'
                    if not checked:raise ValueError('No mapped finite measurements for accuracy validation')
                losses[name]=pairs
        except (RuntimeError,FloatingPointError) as exc:return items,losses,'accuracy_inconclusive',str(exc)
        except (ValueError,KeyError,TypeError) as exc:return items,losses,'accuracy_interface_error',str(exc)
        return items,losses,'accuracy_passed',''

    with ThreadPoolExecutor(max_workers=workers) as executor:
        results=list(executor.map(check_probe,probes))
    for items,_,_,_ in results:report['comparisons'].extend(items)
    # Refuse a bad refinement profile. A failed automatic DE relaxation may
    # instead use the already-validated refinement profile, without new solves.
    excluded = set()
    fatal=[r for r in results if r[2]!='accuracy_passed' and
           (not r[0] or r[0][-1]['profile']!='population')]
    # Contract errors and demonstrated inaccuracies still block readiness.
    for code in ('accuracy_interface_error', 'accuracy_failed'):
        failed=next((r for r in fatal if r[2]==code),None)
        if failed:return finish(code,failed[3],failed[0][-1].get('failure') if failed[0] else None)
    excluded = {probe['sample'] for probe, result in zip(probes, results)
                if result in fatal and result[2] == 'accuracy_inconclusive'}
    accepted_results = [result for probe, result in zip(probes, results) if probe['sample'] not in excluded]
    if not accepted_results:
        failed = fatal[0]
        return finish(failed[2], failed[3], failed[0][-1].get('failure') if failed[0] else None)
    if excluded:
        report['unresolved_probes'] = [dict(sample=probe['sample'], reason=result[3],
            failure=result[0][-1].get('failure') if result[0] else None)
            for probe,result in zip(probes,results) if probe['sample'] in excluded]
        report['warnings'] = [f'{len(excluded)} accuracy probes unresolved; fit is seeded only from validated points. Accuracy away from those points is unverified.']
    for name,_,_ in profiles:
        failed=next((r for r in accepted_results if r[2]!='accuracy_passed' and r[0][-1]['profile']==name),None)
        if failed:
            code,reason=failed[2],failed[3]
        else:
            current=[[p[0] for p in r[1][name]] for r in accepted_results]
            reference=[[p[1] for p in r[1][name]] for r in accepted_results]
            accepted,reason=stable_losses(reference,current)
            if accepted:continue
            code='accuracy_failed'
        if name=='population' and 'stepsize_rtol' in calibration and code!='accuracy_interface_error':
            calibration.pop('stepsize_rtol',None);calibration.pop('stepsize_atol',None)
            calibration.update(de_status='retained',reason='Automatic DE relaxation rejected by forward-accuracy check: '+reason,
                status='accepted' if 'gradient_stepsize_atol' in calibration else 'retained')
            path=folder/'tolerance_calibration.json';temp=path.with_suffix('.tmp')
            temp.write_text(json.dumps(calibration,indent=2,allow_nan=False)+'\n');temp.replace(path)
            report['population_relaxation_rejected']=dict(code=code,reason=reason)
            return finish('accuracy_passed','Refinement tolerances validated; DE retains those validated tolerances')
        return finish(code,f'{name}: {reason}',failed[0][-1].get('failure') if failed else None)
    return finish('accuracy_passed','Predictions and losses agree with ten-times tighter tolerances on validated probes; see unresolved_probes for exclusions')
