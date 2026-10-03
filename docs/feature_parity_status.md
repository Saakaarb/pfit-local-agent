Feature parity status

Updated: 2026-10-03. This is the current implementation tracker for
`pfit-local-agent` against `pfit-claude origin/deployed_branch` at `1b415d8`.
The original repository's checked-out `main` branch is older and is not the
comparison baseline.

Status includes multi-experiment support (`c915677`) and subsequent live Ollama
workflow fixes, after restart, sloppiness and readiness work (`386e061`). “Implemented” means the scoped capability is implemented and
tested; it does not imply identical behavior in every respect. “Partial” identifies
remaining work within an existing capability. “Open” means the gap remains.

Numerical validation follow-up: [solver diagnostics and bounded recovery](solver_recovery.md)
now distinguish step-limit exhaustion from translation defects, preserve
source-derived loss/writeout bodies during field-scoped repair, and persist
accepted solver settings with verified provenance. Live 32B Oregonator now
passes JAX validation after automatic 10k-to-20k recovery, then completes a
five-iteration fit and sloppiness. Other benchmark extraction/check failures
remain open; check-stage automatic repair is still shelved. See the
[three-model comparison](model_comparison_20260927.md) for the unchanged baseline.

**Implemented in the latest port**

| ID | Capability | Evidence and limits |
|---|---|---|
| DONE-01 | Gradient-only restart | `pfit run <session> gradient-only [--from-run <run-id>]`; latest completed seed selection; population search bypassed; physical parameters validated; fresh optimizer state. See [driver](../local_agent/core/fitting.py), [entry point](../fit_gradient_only.py), and [engine](../lib/utils/helper_functions.py). |
| DONE-02 | Post-fit sloppiness | Automatic Hessian/eigenspectrum analysis, physical/log10 coordinates, autodiff-to-finite-difference fallback, spectrum plot, JSON/CSV/text artifacts, and the deployed 60-parameter limit. Failure is nonfatal; nonstationarity, negative curvature and boundary caveats are reported. See [implementation](../lib/utils/sloppiness.py). |
| DONE-03 | Standalone historical sloppiness | [analyze_fit.py](../analyze_fit.py) rebuilds the analysis from a recorded data/model/configuration snapshot without fitting again. Legacy runs without snapshots are deliberately rejected. |
| DONE-04 | Preserved runs and restart provenance | Fresh output directories, named parameters, copied seeds, manifests, data/model/configuration snapshots, and execution from snapshots. Existing run directories are not overwritten. Full provenance parity remains PART-01. |
| DONE-05 | Diagnosis integration for these features | `pfit diagnose` reads saved sloppiness results and recognizes intentional omission of global-search logs on restarts. Generation events come from the run snapshot when available. Broader diagnosis remains PART-02. |
| DONE-06 | PSO packaging | PySwarms 1.3.0 is a declared runtime dependency, its dependency closure is pinned in the Python 3.12 lock file, and import/initialization was smoke-tested against the supported numerical stack. |
| DONE-07 | Host-aware CPU devices | The run driver detects schedulable CPUs, caps JAX host devices by `population_opt.processors`, preserves an explicit `XLA_FLAGS` override, removes the fixed eight-device ceiling, and records requested/detected/configured/actual counts in each run manifest. |
| DONE-08 | Pip-native locked environment | `requirements-lock.txt` pins the complete validated Python 3.12 development/test environment while `pyproject.toml` remains the package definition. The locked install and `pip check` pass. |
| DONE-09 | Run provenance | Run manifests record Git commit/dirty state, Python/platform and core package versions, optimizer and random seed, configured Ollama settings, and SHA-256 hashes for the lock/configuration/model artifacts. Collection is best-effort and cannot break fitting. |
| DONE-10 | Continuous integration | GitHub Actions recreates the locked Python 3.12 environment, runs `pip check`, and executes the deterministic default test suite on every push and pull request. |

Validation recorded for this port: **217 default tests passed, 4 deselected**;
three additional numerical regressions passed (Robertson, Van der Pol, and
full-fit/restart/snapshot reanalysis). Deployed sloppiness tests were ported.
Local-LLM integration was not run. Detailed behavior and test notes:
[restart_and_sloppiness.md](restart_and_sloppiness.md).

Readiness follow-up validation: **243 default tests passed, 4 deselected**;
**47 targeted tests passed**, including the numerical full-fit/restart/reanalysis
regression. See [deterministic_readiness.md](deterministic_readiness.md).

Multi-experiment follow-up: **265 default tests passed, 7 deselected**, plus
**three new numerical acceptance tests** covering fitting, restart, snapshot
reanalysis, sensitivity to experiment 2 and all nine Sneyd conditions. Fake-LLM
Sneyd translation passed. Subsequent live validation: both decay paths and
reference-seeded Sneyd passed; manually corrected fresh Sneyd also completed
nine-record fitting and diagnosis. Autonomous fresh extraction/translation fidelity
remains open. The workflow-fix regression suite passed **270 tests, 7 deselected**.
See [multi_experiment_support.md](multi_experiment_support.md) and
[live evaluation](live_multi_experiment_evaluation.md).

**Remaining divergences and correctness work**

P0 protects scientific correctness; P1 restores broader workflow capability;
P2 is monitoring/UI work. Rows can combine reference parity with a local defect;
they are not all missing features in the same sense.

| ID | Status | Priority | Remaining work |
|---|---|---|---|
| OPEN-01 | Implemented (scoped) | P0 | Shared-parameter fitting across every record, per-record ICs/time grids, equal-mean objective, extraction/translation/checking, per-record outputs, complete snapshots, restart and sloppiness. Common ordered column schemas are required. See [implementation evidence and limits](multi_experiment_support.md). |
| OPEN-02 | Implemented (scoped) | P0 | Explicit forcing roles, per-record linear interpolation, finite-value/time-coverage validation, extraction/JAX support, snapshots and restart. Separate forcing grids and hold interpolation remain unsupported. See [measured_forcing.md](measured_forcing.md). |
| PART-03 | Partial; further work shelved by user | P0 | Missing data and uncertainty: existing loader/loss support needs safe masks before normalization/division/logs and explicit empty-channel policies. The standard generated loss now masks before normalization and rejects empty channels; generated wrappers reject nonfinite simulations. Arbitrary custom/uncertainty masking still needs the broader audit. |
| OPEN-03 | Implemented (scoped) | P0 | User loss/writeout are preserved independently; MSE-pattern default inference is removed. Numerical source/JAX RHS, loss and output probes block acceptance and feed bounded repair on mismatch. This is sampled equivalence, not proof of extracted scientific intent. See [loss_and_translation_fidelity.md](loss_and_translation_fidelity.md). |
| PART-04 | Partial | P0 | Objective preservation: new `user_info.txt` inputs are now required by policy to state the loss mathematically, including mappings, transforms, normalization, reductions, aggregation, weighting and penalties. Explicit loss disables automatic log rewriting; inline declarations and plain RMSE rendering are fixed. Translation has sampled source-equivalence checks. Historical objective mismatches (including Boehm) still require audit; validator enforcement of the new input policy is a separate possible change. |
| PART-05 | Implemented (scoped) | P0 | Deterministic input validation, offline check/ready CLI, source-hash stamping after accepted translation, automatic pre-run checks and existing restart-seed gate. Legacy scripts use a qualified timestamp fallback. See [scope and remaining limits](deterministic_readiness.md); this does not establish numerical translation equivalence. |
| OPEN-04 | Out of scope by user | — | Check-time automatic correction will not be added. `pfit check` remains report-only; new/jax retain their existing bounded repair loops. |
| PART-06 | Implemented; broad convergence study out of scope | P1 | Gradient optimizer selection is implemented: YAML selects Adam (default) or L-BFGS; default gradient budget is 1000, explicit budgets and Adam schedules are preserved. PSO 1.3.0 is now a declared runtime dependency. A broad realistic-budget convergence campaign is explicitly not planned. |
| PART-01 | Implemented (scoped) | P1 | Data/config/model snapshots plus Git, environment, package, optimizer/seed, configured Ollama, and artifact-hash provenance are recorded. The Ollama values are the configuration observed at run time; old runs and CLI overrides used during earlier generation cannot be reconstructed retroactively. |
| PART-02 | Implemented (scoped) | P1 | Snapshot replay, per-record residual/trajectory plots, bounds and budget/stationarity checks, restart history, optional AD/FD probes, and evidence-grounded Ollama recommendations. No automatic edits or scientific-equivalence guarantee. See [scientific_diagnosis.md](scientific_diagnosis.md). |
| OPEN-05 | Partially out of scope by user | P1 | Interactive clarification will not be added. Document/PDF ingestion remains a possible separate feature. |
| OPEN-06 | Out of scope by user | — | Live dashboard, structured runtime monitoring and intervention workflow will not be added. |
| OPEN-07 | Partial | P1 | [All-case live regression](all_case_regression_20260926.md): 12/16 configured cases complete all five stages after documented input/configuration preparation; four generation/check failures remain. Regenerated Sneyd passes all nine records and sampled reference objectives without manual equation changes in this run. Realistic-budget fit quality and broader extraction reliability remain unproven. Earlier [multi-experiment evaluation](live_multi_experiment_evaluation.md) records prior failures and manual corrections. |
| PART-07 | Partial | P1 | Manuscript/documentation alignment: comparison and feature notes are written, but the paper itself has not been edited. Update commands, provider/setup, supported input contracts, numerical claims and examples. Older handoff/evaluation documents remain historical. |

**Explicit scope decisions (2026-10-03)**

The user does not intend to add interactive clarification, check-time automatic
correction, live monitoring/intervention, a broad realistic-budget optimizer
convergence campaign, or broader automatic diagnosis beyond the implemented
snapshot/evidence workflow. These correspond to items 4, 6, 7, 9 and 16 in the
2026-10-03 remaining-features review. They are intentional boundaries, not
unresolved delivery commitments.

PSO packaging, host-aware CPU parallelism and a pip-native locked environment
(items 8, 10 and 12 from that review) were completed on 2026-10-03. Validation:
the locked install resolved successfully, `pip check` found no broken requirements,
the PSO import/initialization smoke test passed, a subprocess exposed the requested
three JAX CPU devices, and the default suite passed 304 tests with 8 deselected.
After adding scoped run provenance, the default suite passed 305 tests with 8
deselected.

Additional scope decisions from the 2026-10-03 broader backlog: exact per-call
LLM generation provenance, heterogeneous per-experiment schemas/overrides,
additional forcing modes/grids, and a GPU execution path will not be pursued.
These were items 3, 7, 8 and 10 in that backlog. Broader masking support for
arbitrary custom and uncertainty-weighted losses (item 6) is deferred because it
is not a small or reliably bounded change. Standard generated-loss NaN masking
remains supported within its documented scope.

**Intentional differences to retain or explicitly document**

| Difference | Current decision |
|---|---|
| Ollama-backed CLI versus Claude slash commands | Keep the local `new -> check -> jax -> run -> diagnose` workflow and YAML-only configuration. |
| Deterministic acceptance versus model assertions | Keep unverified LLM findings as semantic warnings; add deterministic checks rather than granting an LLM-only veto. |
| Named restart parameters | Local named metadata permits safe parameter reordering. Unnamed seeds without configuration snapshots require `--allow-legacy-seed` to assert the current order. |
| Historical analysis | Require a saved snapshot for standalone analysis rather than silently using possibly changed working files. |
| Sloppiness safeguards/output | Retain JSON/CSV results, finite-value checks, bounded finite differences, step-size comparison, and qualified interpretation. These extend the deployed report/plot implementation. |

**How to maintain these notes**

Update the relevant row when implementation work changes its status. Record the
scope actually completed, remaining limitations, validation and date. When
partial work is split, retain its ID so discussion and paper edits can refer to
it. Record intentional departures separately from unfinished parity work.

Use this file for current status; use
[claude_paper_alignment_review.md](claude_paper_alignment_review.md) for the
comparison rationale, reference paths and manuscript mapping; use feature-specific
notes for implementation details. Do not treat the audit's pre-port wording or
older evaluation reports as the current feature checklist.


**Boehm clarification and next priority (2026-09-26)**

Deterministic readiness (PART-05) was moved ahead of multi-experiment fitting and
is now implemented within the scope documented above. Multi-experiment fitting
(OPEN-01) has now been implemented within its common-column/shared-parameter scope. Local
`sessions/boehm_stat5/inputs/user_input.yaml` declares one experiment and one CSV,
with three observable columns (`pSTAT5A`, `pSTAT5B`, `rSTAT5A`), one time grid and
one initial-condition vector. The deployed reference Boehm YAML also declares a
single experiment. This prepared case tests multiple observables, not multiple
independently integrated experiment records. The previous local reader selected
`experiments[0]`, but no later configured Boehm records were present to discard.
The new fitting path processes every declared record.
This inspection does not establish completeness relative to the original
publication or upstream benchmark.

A separate current Boehm discrepancy is objective fidelity: its input note asks
for max-absolute-normalized pooled RMSE, while both saved model and generated JAX
use a sum of range-normalized per-channel MSEs. This is concrete evidence for
OPEN-03/PART-04 and means historical numerical losses should not be assumed
comparable across versions. Historical run artifacts lack snapshots, so the
current generated code alone cannot prove which loss a past run used.

Further multi-experiment evaluation can use the deployed fixtures and examples:
`tests/fixtures/decay_multiexp`, `sneyd_ipr` (9 experiments), and `fujita_egf`
(16 experiments). Record loading, per-record initial conditions/time grids, shared-parameter
aggregation, outputs, snapshots, restart and sloppiness are now implemented.
Regression tests verify that changing experiment 2 changes the fitted result.
Fujita and broader missing-channel/uncertainty cases remain useful evaluation
targets; arbitrary weighting is outside the current shared-schema contract.


**Initial-condition extraction and gradient warnings (2026-09-27)**

State extraction now expands unambiguous group declarations (including zero
values), preserves filename-keyed clamp tables, and makes at most one original-
specification recheck when initial values are reported missing and repairs are
enabled. Complete per-experiment declarations resolve exact state-name missing
flags deterministically, using the first record as the representational default;
records retain their own values. Incomplete declarations still require user input.

Live Qwen2.5-Coder 32B initialization of the unchanged Sneyd benchmark prompt
passed in `evaluation_runs/initial_conditions_live_20260927_v4/sneyd_ipr`.
All eight state defaults and nine IP3/Ca override pairs were verified. Earlier
prompt-only attempts in the corresponding unversioned, v2 and v3 directories
remain preserved: they produced contradictory missing-input claims. This is a
validated initialization fix, not a new end-to-end Sneyd fitting result.

Non-finite gradients remain possible (including cascaded tanks). Estimation now
prints the stopping iteration and affected parameter names, says refinement did
not complete, and retains the best valid point through the existing fallback.
The warning is persisted in `NODE_fitting.log` and `fit_summary.json` includes
`termination_detail`. Equations and gradient mathematics are unchanged.
A regression uses a finite forward objective with a NaN autodiff gradient.

Validation: default suite 329 passed, 8 deselected before the final experiment-
default helper; after that addition, 48 extraction/optimizer/session tests passed.
