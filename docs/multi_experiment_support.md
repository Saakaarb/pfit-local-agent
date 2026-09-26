Multi-experiment support

Implemented 2026-09-26, following the [plan](multi_experiment_plan.md) and
`pfit-claude origin/deployed_branch` at `1b415d8`. OPEN-01 is implemented within
the shared-parameter, common-column contract; see INPUT_REQUIREMENTS.md.

All experiment records now reach YAMLReader/get_y0, shared loading/validation,
session extraction/spec/prompt context, generated-code smoke tests, population
search, gradient refinement and restart, result writing, diagnosis and sloppiness.
Each record has its own initial state and time grid. The generated functions
still process one record per call; the framework averages scalar record losses.
The same aggregation/failure function is used by optimization and sloppiness.

Every run copies all datasets into separate snapshot files and records original
filenames, content hashes, resolved ICs and expected result filenames. The runtime
YAML rewrites every dataset path. Gradient-only restart evaluates the seed against
the current complete experiment list and preserves the old run. Reanalysis loads
all constants from the old snapshot even when working CSVs change or disappear.

Single-record result_solution.csv remains unchanged; multi-record outputs use
result_solution_expN.csv. result_experiments.json maps output filenames to runtime
dataset filenames, while run_manifest.json maps them to original filenames.
fit_summary.json includes the per-record losses and aggregation rule. Diagnosis
reports every expected output and flags missing later-record results.

Reference adaptations and limits

- Match Claude's equal arithmetic mean of per-experiment losses, independent of
  record lengths. No custom record weights, pooled RMSE conversion, per-record
  parameter overrides, arbitrary observation remapping or measured forcing.
- Validate common ordered names, observation/uncertainty meanings and units,
  including actual CSV headers where present. Headerless reference CSVs rely on
  their declared column meanings. Partial IC overrides inherit global values.
- Preserve a failed candidate's full error sentinel; do not average it down.
  Generated wrappers reject any non-success solver result or nonfinite simulated
  trajectory/loss. Standard masking retains missing measurements but cannot hide
  nonfinite simulated values; empty observation channels fail. Custom uncertainty
  and nonlinear transforms still require their own valid masks (PART-03).
- Preserve the existing six extraction stages and local LLM translation step.
  Experiment selection is carried through normalization and targeted repairs.
  The first CSV's automatic log-loss rewrite is disabled for multi-record
  extraction; prompts/checks consider every record and explicit losses take
  precedence. This does not close the broader objective-fidelity work.
- Sneyd's reference YAML requests Adam. Numerical fitting tests explicitly use
  local L-BFGS instead; this port does not add Adam support (PART-06).
- Automated tests use deterministic fake LLM responses for translation acceptance.
  A subsequent [live Ollama evaluation](live_multi_experiment_evaluation.md) passed
  both decay workflows. After workflow fixes, reference-seeded Sneyd also passed
  all nine records through fitting and diagnosis. Fresh Sneyd still fails
  scientific fidelity during extraction. A manually corrected fresh session
  passed all nine records through fitting and diagnosis; OPEN-07 remains open.

Numerical evidence

The reference two-experiment decay fit recovered k1=1.0000000006860064 and
k2=0.29999999976159586 (truth 1 and 0.3). Full fitting, gradient-only restart,
per-record output, finite-difference sloppiness and snapshot-only reanalysis
passed. The reanalysis test deletes both working CSVs before recomputing curvature.

A controlled shared-parameter quadratic fit with record targets 1 and 3 recovered
2. Changing only record 2 to target 5 moved the restarted optimum to 3. Unequal
record lengths did not change equal record weights; analytic gradient/Hessian
checks passed, and the old snapshot retained the original loss and curvature.

Sneyd verification loaded all nine distinct records, checked the IP3/Ca clamp
initial conditions and constant trajectories, independently calculated each
open-probability RMSE, verified their mean, and wrote nine result CSVs. A bounded
one-step refinement from the saved reference parameters reduced mean loss from
0.02138278949932542 to 0.02138229226583555. This is a numerical plumbing check,
not a fresh global-fit or parameter-identifiability benchmark.

Fixtures in tests/fixtures/ include provenance notes. Reference generated scripts
are distinguished from code produced through the local translation workflow.

Validation recorded for this change: **265 default tests passed, 7 deselected**.
The three new bounded numerical tests passed separately (about five minutes):
decay full-fit/restart/reanalysis, second-record optimum sensitivity, and Sneyd
nine-record integration/refinement. The default suite includes 22 focused
multi-experiment tests, including fake-LLM extraction and Sneyd translation,
per-record failure handling, schema/time/IC checks and missing-data safeguards.
`git diff --check` passed. These implementation checks preceded the linked live
Ollama evaluation. No full Sneyd global search was run.

The existing single-experiment full-fit/restart/reanalysis regression also passed
separately (1 test), confirming compatibility with the previous fitting path.
