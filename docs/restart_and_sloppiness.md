Gradient-only restart and post-fit sloppiness

Current implementation status and remaining work: [feature_parity_status.md](feature_parity_status.md).

The broader audit is recorded in [claude_paper_alignment_review.md](claude_paper_alignment_review.md). This document scopes the first two implementation items from that audit.

Reference: `pfit-claude origin/deployed_branch` at `1b415d8`, particularly `fit_gradient_only.py`, `lib/utils/helper_functions.py::fit_gradient_only_system`, `lib/utils/sloppiness.py`, `lib/utils/run_store.py`, and `analyze_fit.py`. The initial audit accidentally used the older main branch; the comparison document has been corrected. The local port retains the deployed sloppiness entry-point signature, size limit, spectrum plot, run selection, and physical-seed restart semantics.

A restart uses `pfit run sessions/<name> gradient-only --from-run <run-id>`. Omitting `--from-run` selects the latest completed run with a saved point, as in the deployed reference; `--seed-run` is an alias. It loads physical parameter values from that run, maps them by name to the current configuration, checks finiteness and current bounds, and runs only the existing gradient optimizer with fresh optimizer state. Current checked/generated model code and current solver settings apply. This is a warm start, not restoration of L-BFGS history. Every CLI run receives a fresh directory. Source artifacts are preserved.

New runs save named physical parameters, a run manifest, a copied seed, and data/configuration/code snapshots. Fitting executes the snapshot, so later edits cannot change the recorded input. Old runs without a configuration snapshot containing only an unnamed `final_design_point.csv` require `--allow-legacy-seed`, an explicit assertion that its parameter order matches the current YAML. A changed parameter set or out-of-bounds starting point is rejected; bounds are not silently clipped.

Sloppiness runs after the best fitted point has been saved. `--no-sloppiness` skips it. `--sloppiness-method auto|ad|finite-difference` selects automatic fallback, second-order automatic differentiation only, or finite differences of first-order autodiff gradients.

The loss remains the fitted objective at gradient-stage tolerances. Curvature is reported in coordinates u_i = log10(theta_i) for logarithmic search axes and u_i = theta_i otherwise. Normalized search coordinates are transformed explicitly. The Hessian is symmetrized and diagonalized; eigenvectors are columns ordered from largest to smallest eigenvalue. Outputs include a machine-readable JSON report, a text report, CSV matrices/spectrum, and the deployed `sloppiness_spectrum.png`. The deployed limit of 60 parameters is preserved.

Finite differences respect parameter bounds, use central differences in the interior and second-order one-sided differences near boundaries, and compare two step sizes. Integration-failure sentinel losses and nonfinite losses/gradients are rejected. Diagnostic failure is recorded separately and does not erase the fitted result.

The report includes gradient magnitude, bound proximity, negative curvature, and weak modes below a relative eigenvalue floor of 1e-6. Flat, indefinite, unconverged, boundary, or numerically unstable results are qualified. This is a local curvature diagnostic, not a proof of structural/global identifiability or of optimizer convergence. It does not require an LLM call.

Validation covers restart name/order/log-scale handling, rejection of invalid/legacy seeds, preservation of source runs, bypass of global search, known analytic Hessians and eigenspectra, fallback behavior, bound handling, diagnostic failures, and a small actual ODE fit.


Existing runs can be reanalyzed without fitting:

```bash
python analyze_fit.py sessions/<name> --run <run-id>
```

This uses the saved data/model/configuration snapshot. Legacy runs without
snapshots require a new recorded restart rather than silently analyzing changed
working files. `python fit_gradient_only.py <name> --seed-run <run-id>` remains
available as the deployed-style Python entry point.

Local extensions beyond the reference: name-based seed mapping when named
metadata exists; explicit confirmation by flag for otherwise unverifiable legacy
CSV order; JSON/CSV numerical output; invalid-gradient and negative-curvature
checks; bound-aware finite differences and a step-size comparison. These do not
claim to port the deployed multi-experiment engine, live dashboard, or full
framework/environment snapshot machinery.


Validation results: the deployed sloppiness tests and local numerical/restart
checks pass. Full-size Robertson and Van der Pol regression fits completed with
sloppiness reports and plots; Robertson exercised the finite-difference fallback.
Standalone reanalysis was checked after deliberately replacing the working model
and data, confirming that it uses the historical snapshot. An unrelated existing
fake-LLM fixture lacked its writeout response; the failure was reproduced on the
untouched baseline and the fixture was corrected.

Final default suite: **217 passed, 4 deselected**. The three additional numerical
regressions (Robertson, Van der Pol, and full-fit/restart/snapshot reanalysis)
also passed in the expanded run. Local-LLM integration was not run; these features
do not add or change LLM calls.
