"""Snapshot-based scientific evidence, adapted from Claude's diagnosis rules.

Recommendations never mutate a model, objective, bounds, or optimizer settings.
Residual metrics use measurement units, independently of the fitted objective.
"""
import importlib.util
import json
from pathlib import Path
import numpy as np


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def residual_statistics(time, measured, simulated):
    if np.shape(measured) != np.shape(simulated) or np.shape(time) != np.shape(measured):
        raise ValueError('Time, measurement and simulation shapes must agree')
    mask = np.isfinite(measured)
    if not np.all(np.isfinite(simulated)):
        raise ValueError("Simulation contains nonfinite values; they cannot be masked as missing data")
    if not mask.any():
        return {"count": 0, "status": "no_measurements"}
    residual = simulated[mask] - measured[mask]
    stats = {"status": "ok", "count": int(mask.sum()), "missing": int((~mask).sum()),
             "rmse": float(np.sqrt(np.mean(residual**2))), "mae": float(np.mean(np.abs(residual))),
             "bias": float(np.mean(residual)), "max_abs": float(np.max(np.abs(residual)))}
    t = time[mask]
    stats["time_correlation"] = (float(np.corrcoef(t, residual)[0, 1])
        if len(t) >= 8 and np.std(residual) > 1e-12 and np.std(t) > 0 else None)
    full = simulated - measured
    adjacent = mask[:-1] & mask[1:]
    left, right = full[:-1][adjacent], full[1:][adjacent]
    stats["lag1_correlation"] = (float(np.corrcoef(left, right)[0, 1])
        if len(left) >= 8 and np.std(left) > 1e-12 and np.std(right) > 1e-12 else None)
    stats["time_regions"] = []
    edges = np.linspace(time[0], time[-1], 4)
    for i in range(3):
        select = mask & (time >= edges[i]) & ((time <= edges[i+1]) if i == 2 else (time < edges[i+1]))
        r = full[select]
        stats["time_regions"].append({"start": float(edges[i]), "end": float(edges[i+1]),
            "count": int(select.sum()), "rmse": float(np.sqrt(np.mean(r*r))) if len(r) else None})
    return stats


def probe_gradient(loss_fn, point, error_loss, max_axes=5):
    import jax
    import jax.numpy as jnp
    point = np.asarray(point, dtype=float)
    result = {"status": "unavailable", "coordinates": "normalized [-1, 1] parameter coordinates"}
    try:
        value, gradient = jax.value_and_grad(loss_fn)(jnp.asarray(point))
        value, gradient = float(value), np.asarray(gradient)
        if not np.isfinite(value) or value == error_loss or not np.all(np.isfinite(gradient)):
            raise ValueError("Nonfinite gradient/loss or failed solve at final point")
        result.update(status="ok", loss=value, gradient=gradient.tolist(),
                      infinity_norm=float(np.max(np.abs(gradient))), comparisons=[])
        for axis in np.argsort(-np.abs(gradient))[:max_axes]:
            estimates = []
            for step in (1e-4, 5e-5):
                lower, upper = point.copy(), point.copy()
                lower[axis] = max(-1., point[axis]-step); upper[axis] = min(1., point[axis]+step)
                a, b = float(loss_fn(jnp.asarray(lower))), float(loss_fn(jnp.asarray(upper)))
                if not np.isfinite(a+b) or a == error_loss or b == error_loss:
                    raise ValueError(f"Nearby solve failure on parameter axis {int(axis)}")
                estimates.append((b-a)/(upper[axis]-lower[axis]))
            stable = bool(np.isclose(*estimates, rtol=.02, atol=1e-5))
            agrees = stable and bool(np.isclose(gradient[axis], estimates[-1], rtol=.02, atol=1e-5))
            result['comparisons'].append({'axis': int(axis), 'ad': float(gradient[axis]),
                'finite_difference': estimates, 'stable': stable, 'agrees': agrees,
                'boundary_stencil': bool(abs(point[axis]) > 1-1e-4)})
        if not all(c['agrees'] for c in result['comparisons']):
            result['status'] = 'disagreement'
    except Exception as exc:
        result.update(status='unavailable', reason=f'{type(exc).__name__}: {exc}')
    return result



def optimizer_trace(path):
    rows=[]
    if not path.exists(): return None
    for line in path.read_text().splitlines():
        parts=line.split(',')
        try:
            iteration,value=int(parts[0]),float(parts[1])
            if np.isfinite(value): rows.append((iteration,value))
        except (ValueError,IndexError): continue
    if not rows: return None
    best=rows[0][1];last=rows[0][0]
    for iteration,value in rows[1:]:
        if best-value > max(abs(best)*1e-4,1e-12): last=iteration
        best=min(best,value)
    return {'file':str(path),'rows':len(rows),'first':rows[0][1], 'final':rows[-1][1],
            'best':best,'last_material_improvement_iteration':last,'last_iteration':rows[-1][0],
            'tail':rows[-5:]}

def scientific_diagnosis(session, output, *, probe_gradients=False):
    from lib.utils.run_store import run_config
    from lib.utils.run_artifacts import load_restart_seed, scale_parameters, parameter_axes
    from lib.utils.experiments import load_experiments, experiment_constants, mean_experiment_loss
    from local_agent.agent.validators import parse_input_yaml
    session, output = Path(session), Path(output)
    snapshot = output/'snapshot'
    report = {'status': 'limited', 'run': str(output), 'sources': str(snapshot),
              'findings': [], 'experiments': [], 'parameters': [], 'limitations': [], 'plots': []}
    def finding(code, priority, evidence, action):
        report['findings'].append({'id': code, 'priority': priority, 'evidence': evidence, 'next_step': action})
    def finish():
        report['findings'].sort(key=lambda f:f['priority'])
        report['verdict'] = (report['findings'][0]['evidence'] if report['findings'] else
            'No flagged numerical symptom; inspect residual plots and assess scientific adequacy.')
        (output/'scientific_diagnosis.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
        return report
    if not snapshot.is_dir():
        report['limitations'].append('No run snapshot; historical scientific diagnosis cannot use mutable working inputs.')
        finding('SNAPSHOT', 0, 'Historical model/data provenance is unavailable.',
                'Create a new recorded run before interpreting residuals or gradients.')
        return finish()
    try:
        reader = parse_input_yaml(run_config(output, session))
        physical = load_restart_seed(output, reader, allow_legacy=True)
        point = scale_parameters(physical, reader)
        records = load_experiments(snapshot, reader)
        module = _load(snapshot/'generated/generated_script.py', 'pfit_diagnosis_generated')
        lo, hi, logs = parameter_axes(reader)
        constants = [dict(experiment_constants(r, reader), min_limits=lo, max_limits=hi, is_logscale=logs) for r in records]
    except Exception as exc:
        report['limitations'].append(f'Snapshot reconstruction failed: {type(exc).__name__}: {exc}')
        finding('INPUT', 0, 'The recorded run could not be reconstructed.', 'Inspect saved configuration, final parameters and dataset snapshots.')
        return finish()
    for name, value, position, logscale in zip(reader.trainable_parameter_names, physical, point, logs):
        fraction = float((position+1)/2)
        report['parameters'].append({'name':name,'value':float(value),'range_fraction':fraction,'logscale':bool(logscale)})
        if fraction <= .01 or fraction >= .99:
            finding('BOUND', 2, f'{name}={value:.6g} is at {fraction:.2%} of its {"log10" if logscale else "linear"} search interval.',
                'Check whether this bound is physically justified. If revising bounds, run a full fit; do not widen them automatically.')
    summary_path = output/'fit_summary.json'
    summary = json.loads(summary_path.read_text()) if summary_path.exists() else {}
    report['optimizer'] = summary
    report['optimizer_traces'] = [trace for name in ('de_fitting.log','pso_fitting.log','NODE_fitting.log') if (trace := optimizer_trace(output/name))]
    manifest_path = output / 'run_manifest.json'
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    original_data_files = {
        item.get('index'): item.get('data_file')
        for item in manifest.get('experiments', [])
        if item.get('index') is not None and item.get('data_file')
    }
    # Follow only the recorded restart chain, never unrelated runs.
    report['seed_history'] = []
    cursor=output;seen=set()
    for _ in range(20):
        manifest=cursor/'run_manifest.json'
        if not manifest.exists(): break
        origin=json.loads(manifest.read_text()).get('source_run')
        if not origin: break
        parent=Path(origin)
        if not parent.is_dir(): parent=output.parent/origin
        resolved=str(parent.resolve())
        if resolved in seen: break
        seen.add(resolved)
        report['seed_history'].append({'run':str(parent), 'population_traces':[trace for name in ('de_fitting.log','pso_fitting.log') if (trace := optimizer_trace(parent/name))]})
        cursor=parent
    for trace in report['optimizer_traces']:
        if trace['rows']>=3 and trace['last_material_improvement_iteration']==trace['last_iteration'] and Path(trace['file']).name!='NODE_fitting.log':
            finding('SEARCH_BUDGET',4,f"Population loss was still improving at iteration {trace['last_iteration']}; tail={trace['tail']}.",
                    'If residuals remain unacceptable, increase population_opt.num_iters and run a full fit. Compare seeds before declaring a global optimum.')
    if summary.get('refinement_error'):
        finding('REFINEMENT', 1, f'Gradient refinement reported: {summary["refinement_error"]}',
                'Run diagnose --probe-gradients; compare AD with finite differences before changing the objective or optimizer.')
    try:
        losses = [float(module._compute_loss_problem(c, point)) for c in constants]
        if any(not np.isfinite(v) or v == reader.error_loss for v in losses):
            raise ValueError('At least one final-point solve returned nonfinite loss or the failure sentinel')
        report['loss'] = float(np.mean(losses)); report['experiment_losses'] = losses
        if 'final_loss' in summary and not np.isclose(report['loss'], summary['final_loss'], rtol=1e-5, atol=1e-8):
            finding('REPLAY', 0, f'Replayed loss {report["loss"]:.6g} differs from saved {summary["final_loss"]:.6g}.',
                    'Resolve the replay discrepancy before interpreting fit quality; inspect recorded code, environment and solver settings.')
    except Exception as exc:
        finding('SOLVE', 0, f'Final-point objective failed: {exc}',
                'Inspect equations, solver status, tolerances and max_steps before increasing optimization budgets.')
        return finish()
    gradient_norm = None
    curvature_file = output/'sloppiness.json'
    if curvature_file.exists():
        curvature = json.loads(curvature_file.read_text())
        report['sloppiness'] = {'status':curvature.get('status'), 'weak_modes':curvature.get('weak_modes'), 'negative_modes':curvature.get('negative_modes')}
        if curvature.get('status') == 'ok':
            gradient_norm = curvature.get('normalized_gradient_inf_norm')
            if curvature.get('weak_modes',0):
                finding('WEAK_MODES', 4, f'Saved curvature has {curvature["weak_modes"]} weak parameter combinations.',
                        'Inspect eigenvectors before designing additional experiments or fixing independently known parameters; weak modes are not proof of individual non-identifiability.')
    if probe_gradients:
        report['gradient_probe'] = probe_gradient(lambda x:mean_experiment_loss(module._compute_loss_problem,constants,x), point, reader.error_loss)
        probe=report['gradient_probe']
        if probe['status']=='ok': gradient_norm=probe['infinity_norm']
        else:
            gradient_norm=None
            finding('GRADIENT',1,f'Gradient probe status: {probe["status"]}; {probe.get("reason", "AD/finite-difference checks did not establish agreement; inspect per-axis stability")}',
                    'Compare solver tolerances and inspect non-smooth RHS/loss operations; rerun the probe before trusting gradient refinement.')
    if gradient_norm is not None and np.isfinite(gradient_norm):
        report['normalized_gradient_inf_norm']=float(gradient_norm)
        report['gradient_to_loss_ratio']=float(gradient_norm/max(abs(report['loss']),1e-12))
        if gradient_norm>1e-5:
            finding('STATIONARITY',3,f'Normalized gradient infinity norm is {gradient_norm:.6g}; stationarity is not established.',
                    'If gradient checks pass and bounds are appropriate, increase gradient_opt.num_iters and run a gradient-only restart from this run.')
        else:
            report['stationarity']='Small local gradient; this does not establish a global optimum or scientific fit quality.'
    else:
        report['limitations'].append('No verified final-point gradient is available; a flat optimizer log does not establish convergence.')
    if summary.get('termination')=='iteration_budget':
        finding('BUDGET',4,'Refinement stopped at its iteration budget.',
                'Assess the final gradient (diagnose --probe-gradients) and residual plots before deciding whether a gradient-only restart is worthwhile.')
    # Reconstruct observations instead of assuming a custom writeout column order.
    source_model = None
    try: source_model = _load(snapshot/'generated/user_model.py','pfit_diagnosis_source')
    except Exception as exc: report['limitations'].append(f'Source observable helper unavailable: {exc}')
    for record,c in zip(records,constants):
        exp={'index':record['index'],'data_file':record['filename'],'channels':[]}
        report['experiments'].append(exp)
        try:
            from diffrax import RESULTS
            time, solution, result = module._integrate_system(c,point)
            time,solution=np.asarray(time),np.asarray(solution)
            if result != RESULTS.successful or not np.all(np.isfinite(solution)):
                raise ValueError('Non-success integration or nonfinite trajectory')
            derived={}
            if reader.observable_names:
                if source_model is None or not hasattr(source_model,'_observables'):
                    raise ValueError('Derived observation mapping is unavailable; custom writeout columns will not be guessed')
                derived=source_model._observables(solution,dict(zip(reader.trainable_parameter_names,physical)),c['fixed_parameters'])
            channels=[]
            for j,col in enumerate(record['columns'][1:]):
                if col.get('role')=='forcing' or col.get('uncertainty_of'): continue
                target=col.get('observes') or col['name']
                if target in reader.integrated_variable_names: simulated=solution[:,reader.integrated_variable_names.index(target)]
                elif target in derived: simulated=np.asarray(derived[target])
                else:
                    report['limitations'].append(f'Experiment {record["index"]}: no observation mapping for {col["name"]}.');continue
                measured=record['dataset'][:,j]
                stats=residual_statistics(time,measured,simulated);stats.update(name=col['name'],observes=target)
                exp['channels'].append(stats);channels.append((col['name'],measured,simulated))
                if stats['count']>=8 and stats['rmse']>1e-10 and (abs(stats['bias'])>.25*stats['rmse'] or abs(stats['time_correlation'] or 0)>.5 or abs(stats['lag1_correlation'] or 0)>.5):
                    finding('RESIDUAL',5,f'Experiment {record["index"]}, {col["name"]}: RMSE={stats["rmse"]:.6g}, bias={stats["bias"]:.6g}, time correlation={stats["time_correlation"]}, lag-1={stats["lag1_correlation"]}.',
                            'Inspect the time-region residuals and trajectory plot. Check input timing, calibration, initial conditions and model mismatch before spending more iterations; do not change the user loss automatically.')
            _plot_record(output,record,time,channels,report, original_data_files.get(record['index']))
        except Exception as exc:
            exp['error']=f'{type(exc).__name__}: {exc}'
            report['limitations'].append(f'Experiment {record["index"]} residual/plot diagnosis unavailable: {exc}')
    report['status']='ok' if not report['limitations'] else 'limited'
    return finish()


def _plot_record(output, record, time, channels, report, display_filename=None):
    if not channels: return
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    forcing = [
        (col['name'], record['dataset'][:, j])
        for j, col in enumerate(record['columns'][1:])
        if col.get('role') == 'forcing'
    ]
    title = Path(display_filename or record.get('filename', f"experiment_{record['index']}")).stem.replace('_', ' ').title()
    fig,axes=plt.subplots(len(channels),1,figsize=(7.2,3.1*len(channels)),squeeze=False)
    for i,(name,measured,simulated) in enumerate(channels):
        ax = axes[i,0]
        ax.plot(time, measured, 'o', ms=3.8, color='C0', label='measured')
        ax.plot(time, simulated, '-', lw=1.9, color='C3', label='simulated')
        ax.set_title(f'{title} experiment: {name}')
        ax.set_xlabel('time')
        ax.set_ylabel(name)
        handles, labels = ax.get_legend_handles_labels()
        if forcing:
            twin = ax.twinx()
            for forcing_name, forcing_values in forcing:
                twin.plot(time, forcing_values, '--', lw=1.2, color='0.35', alpha=0.8, label=forcing_name)
            twin.set_ylabel(', '.join(name for name, _ in forcing))
            forcing_handles, forcing_labels = twin.get_legend_handles_labels()
            handles += forcing_handles
            labels += forcing_labels
        ax.legend(handles, labels, loc='best', fontsize=8)
        residual=simulated-measured
        np.savetxt(output/f'residual_exp{record["index"]}_channel{i+1}.csv',np.column_stack([time,measured,simulated,residual]),delimiter=',',header='time,measured,simulated,residual',comments='')
    fig.tight_layout();path=output/f'fit_exp{record["index"]}.png';fig.savefig(path,dpi=160);plt.close(fig)
    report['plots'].append(path.name)
