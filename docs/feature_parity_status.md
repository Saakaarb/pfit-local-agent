Feature parity status

Updated: 2026-09-26. This is the current implementation tracker for
`pfit-local-agent` against `pfit-claude origin/deployed_branch` at `1b415d8`.
The original repository's checked-out `main` branch is older and is not the
comparison baseline.

Status reflects the current working tree, including the multi-experiment follow-up
after committed restart, sloppiness and readiness work (`386e061`). “Implemented” means the scoped capability is implemented and
tested; it does not imply identical behavior in every respect. “Partial” identifies
remaining work within an existing capability. “Open” means the gap remains.

**Implemented in the latest port**

| ID | Capability | Evidence and limits |
|---|---|---|
| DONE-01 | Gradient-only restart | `pfit run <session> gradient-only [--from-run <run-id>]`; latest completed seed selection; population search bypassed; physical parameters validated; fresh optimizer state. See [driver](../local_agent/core/fitting.py), [entry point](../fit_gradient_only.py), and [engine](../lib/utils/helper_functions.py). |
| DONE-02 | Post-fit sloppiness | Automatic Hessian/eigenspectrum analysis, physical/log10 coordinates, autodiff-to-finite-difference fallback, spectrum plot, JSON/CSV/text artifacts, and the deployed 60-parameter limit. Failure is nonfatal; nonstationarity, negative curvature and boundary caveats are reported. See [implementation](../lib/utils/sloppiness.py). |
| DONE-03 | Standalone historical sloppiness | [analyze_fit.py](../analyze_fit.py) rebuilds the analysis from a recorded data/model/configuration snapshot without fitting again. Legacy runs without snapshots are deliberately rejected. |
| DONE-04 | Preserved runs and restart provenance | Fresh output directories, named parameters, copied seeds, manifests, data/model/configuration snapshots, and execution from snapshots. Existing run directories are not overwritten. Full provenance parity remains PART-01. |
| DONE-05 | Diagnosis integration for these features | `pfit diagnose` reads saved sloppiness results and recognizes intentional omission of global-search logs on restarts. Generation events come from the run snapshot when available. Broader diagnosis remains PART-02. |

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
Sneyd translation passed; live Ollama evaluation remains open. See
[multi_experiment_support.md](multi_experiment_support.md).

**Remaining divergences and correctness work**

P0 protects scientific correctness; P1 restores broader workflow capability;
P2 is monitoring/UI work. Rows can combine reference parity with a local defect;
they are not all missing features in the same sense.

| ID | Status | Priority | Remaining work |
|---|---|---|---|
| OPEN-01 | Implemented (scoped) | P0 | Shared-parameter fitting across every record, per-record ICs/time grids, equal-mean objective, extraction/translation/checking, per-record outputs, complete snapshots, restart and sloppiness. Common ordered column schemas are required. See [implementation evidence and limits](multi_experiment_support.md). |
| OPEN-02 | Open | P0 | Measured forcing histories: explicit input-column roles, interpolation, time coverage and missing-input validation throughout generation and execution. |
| PART-03 | Partial | P0 | Missing data and uncertainty: existing loader/loss support needs safe masks before normalization/division/logs and explicit empty-channel policies. The standard generated loss now masks before normalization and rejects empty channels; generated wrappers reject nonfinite simulations. Arbitrary custom/uncertainty masking still needs the broader audit. |
| OPEN-03 | Open | P0 | Preserve custom loss and writeout independently during translation; add numerical source-versus-JAX comparisons. Current default-loss handling can replace a custom output function. |
| PART-04 | Partial | P0 | Explicit-loss precedence: checker changes preserve user objectives, but new-session automatic log transformations still need alignment. |
| PART-05 | Implemented (scoped) | P0 | Deterministic input validation, offline check/ready CLI, source-hash stamping after accepted translation, automatic pre-run checks and existing restart-seed gate. Legacy scripts use a qualified timestamp fallback. See [scope and remaining limits](deterministic_readiness.md); this does not establish numerical translation equivalence. |
| OPEN-04 | Open | P1 | Check-time automatic correction: bounded mechanical repair and revalidation of generated YAML/model files. New/jax repair loops do not supply this behavior. |
| PART-06 | Partial | P1 | Optimizer configuration: expose the existing Adam branch, declare PSO installation requirements, preserve requested settings, and distinguish tiny smoke-test budgets from fitting budgets. |
| PART-01 | Partial | P1 | Full run provenance: data/config/model snapshots exist, but the deployed framework/tool/environment archive and package/version/hash provenance are not fully ported. |
| PART-02 | Partial | P1 | Scientific diagnosis: add residual/trajectory plots, convergence and parameter-bound checks, autodiff pathology analysis, and evidence-based recommendations. Sloppiness and basic log summaries are implemented. |
| OPEN-05 | Open | P1 | Interactive clarification and document ingestion: local missing-input errors/text intake do not provide the manuscript's conversational/PDF-assisted setup. Distinguish provider-level abilities from code that can be ported. |
| OPEN-06 | Open | P2 | Live dashboard, structured runtime monitoring and intervention workflow from the deployed reference. |
| OPEN-07 | Open | P1 | Consistent local-model evaluation of generation fidelity and fit quality across examples, with failures, repair counts and experimental-data provenance recorded. |
| PART-07 | Partial | P1 | Manuscript/documentation alignment: comparison and feature notes are written, but the paper itself has not been edited. Update commands, provider/setup, supported input contracts, numerical claims and examples. Older handoff/evaluation documents remain historical. |

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

Use the deployed multi-experiment fixtures and examples for parity tests:
`tests/fixtures/decay_multiexp`, `sneyd_ipr` (9 experiments), and `fujita_egf`
(16 experiments). Port record loading, per-record initial conditions/time grids,
shared-parameter loss aggregation, per-record outputs, generation/checking, and
restart/snapshot/sloppiness integration. Test that modifying only experiment 2
changes the aggregate loss and fitted result, so first-record-only regressions
cannot pass unnoticed. Treat missing-channel masks and explicit weighting as
part of that work where those datasets require them.
