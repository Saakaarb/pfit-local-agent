Scientific diagnosis

Adapted from pfit-claude deployed_branch 1b415d8, especially diagnosis_rules.md,
pfit-diagnose.md and autodiff_diagnose.py. Local reports use deterministic evidence
and propose actions; they do not edit the user model, objective or settings.

`pfit diagnose SESSION [RUN]` now rebuilds the selected run from its data/model/
configuration snapshot and saved parameters. It replays per-experiment losses,
checks linear/logarithmic parameter-bound proximity, reads optimizer traces and
recorded restart ancestry, and distinguishes budget exhaustion from stationarity.
Historical scientific reconstruction is unavailable without a snapshot; old runs
still receive the existing artifact/log report, with an explicit limitation.

For every mapped observation, the report computes count, missing count, raw-unit
RMSE/MAE/bias, maximum residual, time/lag-one correlations and three time-region
RMSEs. It writes measured-versus-simulated and residual plots and CSVs per record.
Forcing and uncertainty columns are not fitted observations. Simulation NaNs are
never masked as missing measurements. Custom writeout layouts are not guessed:
trajectories come from the recorded integrator and derived observations from the
recorded source helper. Missing/unsupported mappings are reported as limitations.
These residual metrics do not redefine or normalize the fitted objective.

`--probe-gradients` evaluates autodiff and bounded finite differences at two step
sizes on up to five parameter axes (largest AD magnitudes first). Nearby solver
failures, nonfinite gradients and unstable/disagreeing differences produce
specific findings. This optional computation may be expensive. Existing saved
sloppiness supplies local gradient/curvature evidence when available. Small
local gradients, flat best-so-far logs and low loss do not certify global
convergence, parameter identifiability or scientific adequacy.

Artifacts: fit_diagnosis.txt, scientific_diagnosis.json, fit_expN.png and
residual_expN_channelM.csv. Findings contain evidence and a next step: inspect
numerical failures first; investigate bounds/gradients; use a gradient-only
restart for justified extra refinement; use full fitting for changed bounds or
population settings; investigate timing/calibration/model mismatch when residuals
are structured. There is no automatic loss rewriting or configuration mutation.

Thresholds are screening heuristics, not hypothesis tests: a bound is flagged
within 1% of its search interval; gradient infinity norm above 1e-5 does not
establish stationarity; residual structure requires at least eight observations
and bias/RMSE > .25 or absolute correlation > .5. Correlations are only computed
with nonconstant data, and lag-one pairs do not bridge missing measurements.
FD agreement uses 2% relative/1e-5 absolute tolerance and steps 1e-4 and 5e-5.

Limits: diagnosis cannot prove the equations match a paper, infer units or
measurement uncertainty, or decide a model change without scientific judgement.
No automatic causal claim of stiffness or model inadequacy is made from a bad
residual alone. Inspect generated plots; report generation is not visual review.

The CLI now calls the configured Ollama model after computing the evidence
(qwen2.5-coder:32b in the committed configuration). Qwen receives the selected
run's snapshot configuration/source and computed findings, not mutable working
model files. It returns an ordered explanation and proposed next actions, each
referencing available evidence IDs. Unknown evidence citations or malformed
responses are rejected; numerical reports survive Ollama failure. Model prose
is interpretation, not a new validation result or permission to alter the loss.
The text-only model is explicitly told it has not viewed the plot images.

Use --deterministic-only for offline diagnosis. Python callers may omit the
llm_client argument for the same behavior. Standard --model/--base-url/timeout/
temperature/token flags are supported. Interpretation and exact requests are
saved as ollama_diagnosis.json and run-local agent_logs/llm_calls.jsonl; snapshots
are not edited. Repair/recommendation application remains user-controlled.

Live acceptance: cascaded-tanks snapshot replay and plotting succeeded. The
optional probe found unstable FD estimates on two axes at the small-budget fit;
Qwen 32B cited gradient, residual and budget evidence and recommended numerical
checks before further refinement. The plotted trajectory visibly misses much of
the measured variation, consistent with RMSE 3.55015 and strong residual structure.
No model, bounds or objective were modified.

Validation: 283 default tests passed, 7 deselected. Tests cover snapshot isolation,
failed solves, residual masking, bound/gradient findings, figures, AD/FD pathology,
Ollama evidence citations, rejected invented citations and offline CLI behavior.
