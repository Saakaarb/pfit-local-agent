"""Review the completed ARC run; optionally wait for the existing run to finish."""
import json
import os
from pathlib import Path
import sys
import time
os.environ['OPENBLAS_NUM_THREADS']='1'
os.environ['MPLCONFIGDIR']='/workspace/.cache/matplotlib'
import numpy as np
import yaml
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]/'evaluation_runs/qwen38_cpu_fits_20261005'
case=ROOT/'cases/ARC_fitting'
while True:
 status=json.loads((case/'status.json').read_text())
 if status['status']!='running':break
 if '--wait' not in sys.argv:raise SystemExit('ARC still running')
 time.sleep(15)
if status['status']!='completed':raise SystemExit(f"ARC status: {status['status']}")
run=Path(status['run_dir']);out=ROOT/'quality_review';out.mkdir(exist_ok=True)
cfg=yaml.safe_load((run/'snapshot/inputs/run_config.yaml').read_text())
a=np.loadtxt(run/'result_solution.csv',delimiter=',')
d=np.genfromtxt(run/'snapshot/inputs/dataset_1.csv',delimiter=',',names=True)
np.testing.assert_allclose(a[:,:3],np.column_stack([d[n] for n in d.dtype.names]),rtol=1e-10,atol=1e-12)
t,T,q,c1,c2,Tp,qp=a.T
assert np.isfinite(a).all()
eps=float(q[q>0].min()); lq=np.log10(q+eps);lp=np.log10(qp+eps)
resT=(Tp-T)/(np.ptp(T)+1e-12);resq=(lp-lq)/(np.ptp(lq)+1e-12)
terms=dict(temperature_mse=float(np.mean(resT**2)),log_heat_rate_mse=float(np.mean(resq**2)))
def sigmoid(z):return float(1/(1+np.exp(-np.clip(1000*z,-60,60))))
penalties=dict(c1=10000*sigmoid(c1[-1]-.02),c2=10000*sigmoid(.98-c2[-1]),
              terminal_temperature=10000*sigmoid(np.sqrt((Tp[-1]-T[-1])**2+1e-12)-50))
loss=sum(terms.values())+sum(penalties.values())
np.testing.assert_allclose(loss,status['fit_summary']['final_loss'],rtol=1e-6,atol=1e-10)
params=json.loads((run/'final_parameters.json').read_text())['parameters'];bounds=[]
for p,b in zip(params,cfg['model']['trainable_parameters']):
 assert p['name']==b['name']
 f=np.log10 if b['logscale'] else lambda x:x
 z=float((f(p['value'])-f(float(b['min_val'])))/(f(float(b['max_val']))-f(float(b['min_val']))))
 bounds.append(dict(name=p['name'],value=p['value'],fraction_of_search_interval=z,near_bound=min(z,1-z)<.01))
metrics=dict(run_dir=str(run),fit_summary=status['fit_summary'],loss_reconstructed=loss,data_loss_terms=terms,penalty_terms=penalties,
 temperature_rmse_K=float(np.sqrt(np.mean((Tp-T)**2))),temperature_range_normalized_rmse_percent=float(100*np.sqrt(terms['temperature_mse'])),
 temperature_max_absolute_error_K=float(np.max(np.abs(Tp-T))),temperature_R2=float(1-np.sum((Tp-T)**2)/np.sum((T-T.mean())**2)),
 log_heat_rate_rmse_decades=float(np.sqrt(np.mean((lp-lq)**2))),log_heat_rate_range_normalized_rmse_percent=float(100*np.sqrt(terms['log_heat_rate_mse'])),
 log_heat_rate_R2=float(1-np.sum((lp-lq)**2)/np.sum((lq-lq.mean())**2)),
 observed_peak_heat_rate=float(q.max()),predicted_peak_heat_rate=float(qp.max()),observed_peak_time=float(t[q.argmax()]),predicted_peak_time=float(t[qp.argmax()]),
 terminal_c1=float(c1[-1]),terminal_c2=float(c2[-1]),terminal_temperature_error_K=float(Tp[-1]-T[-1]),parameters=bounds,
 interpretation_note='Loss is a sum of normalized MSE terms plus penalties; 100*loss is NOT percentage RMSE. Peak metrics are sampled at measurement times.')
(out/'ARC_metrics.json').write_text(json.dumps(metrics,indent=2)+'\n')
fig,axs=plt.subplots(3,2,figsize=(12,11))
axs[0,0].plot(t,T,'.',label='Measured');axs[0,0].plot(t,Tp,label='Fit');axs[0,0].set_ylabel('Temperature (K)')
axs[0,1].semilogy(t,q+eps,'.',label='Measured + epsilon');axs[0,1].semilogy(t,qp+eps,label='Fit + epsilon');axs[0,1].set_ylabel('Heat rate + epsilon (K/s)')
axs[1,0].plot(t,Tp-T);axs[1,0].set_ylabel('Temperature residual (K)');axs[1,0].axhline(0,color='black',lw=.5)
axs[1,1].plot(t,lp-lq);axs[1,1].set_ylabel('Log10 heat-rate residual');axs[1,1].axhline(0,color='black',lw=.5)
axs[2,0].plot(t,c1,label='c1');axs[2,0].plot(t,c2,label='c2');axs[2,0].set_ylabel('Reaction states')
axs[2,1].plot(t,q,label='Measured');axs[2,1].plot(t,qp,label='Fit');axs[2,1].set_ylabel('Heat rate (K/s), linear scale')
for ax in axs.flat:
 ax.set_xlabel('Time (s)')
 if ax.get_legend_handles_labels()[0]:ax.legend()
fig.suptitle('ARC: completed fit and residuals');fig.tight_layout();fig.savefig(out/'ARC_fit.png',dpi=150);fig.savefig(out/'ARC_fit.pdf');plt.close(fig)
print(json.dumps(metrics,indent=2))
