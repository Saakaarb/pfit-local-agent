# Sneyd extended differential-evolution fit

This run uses the accepted local-agent model and all nine experiments, with
320 candidates and 1,000 DE generations (seed 29, eight CPU workers).
The integration settings match the previous 5,000-step rerun. Equations,
parameter bounds, data, loss, and tolerances were unchanged from that rerun.
The complete configuration and model used here are saved under `snapshot/`.

| Result | Mean experiment loss |
| --- | ---: |
| Previously published run (2026-10-08) | 0.02505909149 |
| Previous best rerun (2026-10-09) | 0.02243058078 |
| This run, after DE | 0.02142304944 |
| This run, after refinement | 0.02139618897 |

The final loss is 14.6% below the previously published result and 4.6% below
the previous best rerun. It is the equal-weight mean of nine experiment losses;
individual experiments do not all improve.

All 1,000 DE generations completed. Adam refinement stopped at iteration 290
on a solver failure/error-loss sentinel, retaining the best valid point.
Sloppiness analysis failed because of an invalid loss/gradient or integration.
These results do not establish convergence or identifiability. The fitted k3
and k_2 parameters are at their upper and lower bounds, respectively.

`fit_overview.png` shows all nine datasets and fitted predictions at measurement
times. `fit_exp1.png` through `fit_exp9.png` are the framework's individual
diagnostic plots. The numerical reports and failed sloppiness report are retained.
