Measured forcing support

Reference: pfit-claude origin/deployed_branch at 1b415d8. The cascaded-tanks
fixture has a measured pump-voltage input and lower-tank output. EMPS is a second
reference example with motor-voltage input and position output; it is not yet
part of the local acceptance suite.

The local contract adds role: forcing and linear interpolation to shared column
schemas. Each experiment supplies its own input history. Scalar input bindings
are generated in Python and JAX RHS functions; existing explicit interpolation
in prepared source models is supported. Forcing is excluded from standard loss
observation mappings and automatic log-loss requirements. Loading checks finite
input samples and complete integration-time coverage; all run snapshots carry
the forcing values and roles through restart and historical analysis.

Initial scope: same-grid linear interpolation, with no extrapolation. Independent
forcing grids, step/hold interpolation and missing forcing values are unsupported.
No input is inferred from measured outputs. See INPUT_REQUIREMENTS.md.

Acceptance tests cover two records with unequal time grids and different forcing
histories, between-sample interpolation, analytic trajectories, changing only
forcing, shared parameter recovery, gradient restart, self-contained snapshots,
fresh extraction and invalid roles/coverage/values. The cascaded-tanks fixture
retains provenance and a held-out validation CSV that is never fitted.

Validation: 277 default tests passed, 7 deselected. Live qwen2.5-coder:32b
check/JAX/reference-loss comparison/bounded fit/diagnosis passed on cascaded
tanks. Reference comparisons used two parameter vectors. Mean loss changed
from 0.565812494 to 0.355015351; this is a bounded smoke fit,
not a converged benchmark. Artifacts: evaluation_runs/live_forcing_cascaded_20260926/.
