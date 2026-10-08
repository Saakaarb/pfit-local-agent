"""Plot completed CPU fits using output mappings audited against saved models."""
from pathlib import Path
import os
os.environ['OPENBLAS_NUM_THREADS']='1'
os.environ['MPLCONFIGDIR']='/workspace/.cache/matplotlib'
import json
import argparse
import numpy as np
import yaml
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1]/'evaluation_runs/qwen38_cpu_fits_20261005')
ROOT=parser.parse_args().root.resolve()
OUT=ROOT/'plots';OUT.mkdir(exist_ok=True)
MAPPING={'ARC_fitting':[5,6],'armistead_sphingolipid':[9,10,11],
 'beer_indigoidine':[7,8],'boehm_stat5':[12,13,14],'cascaded_tanks':[4],
 'mapk_cascade':[4],'raia_il13':[10],'schwen_insulin':[13],'sneyd_ipr':[10],
 'oregonator':[3,5],'sliding_basepoint_headered':[9,10],
 'decay_multiexp':[3,4],'lotka_volterra':[3,4],'piezo_bouc_wen':[2],
 'robertson_session':[4,5,6],'test_session':[4,5,6],'theophylline':[4],'vanderpol_session':[3,4]}
entries=[];page=2
for name,cols in MAPPING.items():
 status_path=ROOT/'cases'/name/'status.json'
 if not status_path.exists():continue
 status=json.loads(status_path.read_text())
 if status['status']!='completed':continue
 run=Path(status['run_dir']);config=yaml.safe_load((run/'snapshot/inputs/run_config.yaml').read_text())
 entries.append(dict(case=name,first_page=page,experiments=len(config['experiments']),
                     pdf=f'{name}.pdf',run_dir=str(run.relative_to(ROOT)),
                     fit_summary=status.get('fit_summary'),channels=[],
                     refinement=status.get('assessment'),prediction_columns_zero_based=cols))
 page+=len(config['experiments'])
with PdfPages(OUT/'all_completed_fits.pdf') as combined:
 cover=plt.figure(figsize=(11.7,8.3));cover.text(.08,.91,'Completed Qwen3.8 CPU fits',fontsize=22)
 cover.text(.08,.83,'Measured points and saved fitted trajectories; residual = prediction − measurement.\nLines connect saved measurement times; no additional dense integration.\nCompletion does not imply convergence or parameter identifiability.',fontsize=11)
 for i,e in enumerate(entries):
  note=' — refinement failed; DE fit retained' if e['refinement']=='refinement_degraded' else ''
  cover.text(.08,.71-.033*i,f"Page {e['first_page']:2}: {e['case']} ({e['experiments']} experiment(s)){note}",fontsize=10)
 combined.savefig(cover);plt.close(cover)
 for e in entries:
  name=e['case'];run=ROOT/e['run_dir'];config=yaml.safe_load((run/'snapshot/inputs/run_config.yaml').read_text());cols=MAPPING[name]
  with PdfPages(OUT/e['pdf']) as case_pdf:
   for ei,exp in enumerate(config['experiments']):
    suffix='' if len(config['experiments'])==1 else f'_exp{ei+1}'
    arr=np.loadtxt(run/f'result_solution{suffix}.csv',delimiter=',',ndmin=2)
    data=np.genfromtxt(run/'snapshot/inputs'/exp['data_file'],delimiter=',',names=True)
    obs=[c for c in exp['columns'] if c.get('observes')];assert len(obs)==len(cols)
    np.testing.assert_allclose(arr[:,0],data[data.dtype.names[0]],rtol=1e-10,atol=1e-12)
    fig,axes=plt.subplots(2,len(obs),figsize=(5*len(obs),6),squeeze=False)
    for oi,(col,pc) in enumerate(zip(obs,cols)):
     y=data[col['name']];yp=arr[:,pc];t=arr[:,0]
     np.testing.assert_allclose(arr[:,oi+1],y,rtol=1e-10,atol=1e-12)
     assert np.isfinite(yp).all()
     ax=axes[0,oi];ar=axes[1,oi]
     if name=='ARC_fitting' and col['name']=='dTdt':
      eps=float(y[y>0].min());ax.semilogy(t,y+eps,'o',ms=3,label='Measured + epsilon');ax.semilogy(t,yp+eps,label='Fit + epsilon')
      residual=np.log10(yp+eps)-np.log10(y+eps);ylabel='Log10 residual';metric=f'Log RMSE = {np.sqrt(np.mean(residual**2)):.3g} decades'
     else:
      ax.plot(t,y,'o',ms=3,label='Measured');ax.plot(t,yp,label='Fit')
      residual=yp-y;ylabel='Residual (observable units)';rmse=np.sqrt(np.mean(residual**2));metric=f'RMSE = {rmse:.4g}'
     raw_residual=yp-y
     scale=max(float(np.max(np.abs(y))),1e-12)
     sst=float(np.sum((y-y.mean())**2))
     e['channels'].append(dict(experiment=ei+1,observable=col['name'],n=len(y),
          rmse=float(np.sqrt(np.mean(raw_residual**2))),
          normalized_rmse_percent=float(100*np.sqrt(np.mean(raw_residual**2))/scale),
          r_squared=float(1-np.sum(raw_residual**2)/sst) if sst>0 else None,
          bias=float(np.mean(raw_residual))))
     ar.plot(t,residual,'.-',ms=2,lw=.8);ar.axhline(0,color='black',lw=.6)
     ax.set_title(f"{col['name']}\n{metric}");ax.set_ylabel(col['name']);ar.set_ylabel(ylabel);ax.legend()
     for a in (ax,ar):
      a.set_xlabel('Time (input units)');a.grid(alpha=.2)
      if name in ('robertson_session','test_session'):a.set_xscale('log')
    note=' | DE result retained; Adam failed' if e['refinement']=='refinement_degraded' else ''
    fig.suptitle(f'{name} — experiment {ei+1}' + ('\nDE result retained; Adam failed' if note else ''),fontsize=11)
    fig.tight_layout(rect=(0,0,1,.91));combined.savefig(fig);case_pdf.savefig(fig)
    fig.savefig(OUT/f'{name}{suffix}.png',dpi=130);plt.close(fig)
(OUT/'index.json').write_text(json.dumps(entries,indent=2)+'\n')
print(f'Created {len(entries)} case PDFs and {page-2} experiment plots, plus indexed combined PDF.')
