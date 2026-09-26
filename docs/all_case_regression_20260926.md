# All-case live regression — 2026-09-26

Outcome: 12 complete workflow passes, 4 failures, 0 degraded results, and 2 incomplete case folders.

Tested framework revision: `3f9628fbba042afd7eaf5bfce67b7e8350838745`. Runtime framework code was not changed during the evaluation.
Missing-data extensions and the proposed check repair loop remain shelved. Live monitoring remains deferred.

## Scope and method

- Inventoried every directory under `sessions/` plus the three `tests/fixtures/` cases: 18 folders, 16 with existing specifications/configurations. Both sliding-basepoint variants and the auxiliary test session were included.
- `new → check → jax → run → diagnose` with existing case YAML retained using real Ollama 0.34.4 and `qwen2.5-coder:32b`, temperature 0.1, 12,000 output tokens, 600-second request timeout, and five existing new/JAX repair attempts.
- Each attempt used an isolated `/workspace` copy. Existing generated scripts and old outputs were not reused. Because `new` was called without `--overwrite`, existing YAML was retained while Python model code was regenerated; this does not test fresh YAML extraction. Fixture loss notes were augmented with their authoritative YAML/Python equations, as in earlier reference-backed fixture evaluations. This is not paper/PDF extraction.
- Fitting used DE population 4, one population iteration, seed 7, followed by five Adam iterations on CPU. The pre-smoke YAML was retained before this explicit evaluation-only budget change (the original artifact is named `extracted_user_input.yaml`, but contains the preserved input configuration). These are workflow smoke tests, not convergence or generalization benchmarks.
- The auxiliary `test_session` was additionally rerun with `--overwrite-config` after its retained YAML defaulted to Tsit5 and failed stiff integration. The table uses this separate retry; original failure logs remain intact.
- Sloppiness ran after fitting. A complete pass required the five CLI stages, expected experiment output count, a finite non-sentinel final loss, no failed gradient refinement, and an accepted evidence-grounded Ollama diagnosis.
- Decay and Sneyd fixture CSVs had no headers. Fresh intake rejected them. Separate retries added exactly the YAML-declared labels to copied CSVs without changing numerical rows. Original data and failure artifacts remain intact. No model equations or objectives were manually corrected.

## Per-case outcomes

| Case | new | check | jax | fit | diagnosis | Overall | Notes |
|---|---|---|---|---|---|---|---|
| ARC_fitting | pass | fail | — | — | — | fail | Qwen semantic review repeated a sentence until the 12,000-token limit; truncated invalid JSON aborted check. |
| boehm_stat5 | pass | fail | — | — | — | fail | Fresh loss extraction selected normalized MSE despite explicit pooled RMSE; deterministic check rejected the missing square root. |
| cascaded_tanks | pass | pass | pass | pass | pass | pass |  |
| decay_multiexp | pass | pass | pass | pass | pass | pass | Header-prepared retry: numerical rows unchanged; original fresh intake rejected headerless CSVs. |
| hodgkin_huxley | — | — | — | — | — | blocked | Only a CSV is present; no user specification, YAML or model. |
| lotka_volterra | pass | pass | pass | pass | pass | pass |  |
| mapk_cascade | pass | pass | pass | pass | pass | pass |  |
| nfkb_signaling | fail | — | — | — | — | fail | Dataset extraction returned an invalid initial-condition override; new rejected it. |
| oregonator | pass | fail | — | — | — | fail | Fresh loss extraction selected uncertainty-weighted MSE despite explicit RMSE; deterministic check rejected it. |
| piezo_bouc_wen | pass | pass | pass | pass | pass | pass |  |
| robertson_session | pass | pass | pass | pass | pass | pass |  |
| session1 | — | — | — | — | — | blocked | No input files are present. |
| sliding_basepoint | pass | pass | pass | pass | pass | pass |  |
| sliding_basepoint_headered | pass | pass | pass | pass | pass | pass |  |
| sneyd_ipr | pass | pass | pass | pass | pass | pass | Header-prepared retry: numerical rows unchanged; original fresh intake rejected headerless CSVs. |
| test_session | pass | pass | pass | pass | pass | pass | Separate --overwrite-config rerun. Original retained YAML omitted an integrator and used Tsit5 for stiff Robertson; original JAX smoke solves failed after five repairs. |
| theophylline | pass | pass | pass | pass | pass | pass |  |
| vanderpol_session | pass | pass | pass | pass | pass | pass |  |

## Numerical evidence

Independent reference comparisons passed for cascaded tanks, two-experiment decay, and nine-experiment Sneyd. Parameter identities/bounds/scales, fixed values, experiment ordering and resolved initial conditions matched the fixtures. Per-record objective values matched at a reference parameter vector and a 2% perturbation (`rtol=1e-3`, `atol=2e-6`). These sampled checks do not establish global equivalence.

| Passed case | Experiments | Seed loss | Final loss | Adam evaluations | Sloppiness | JAX/new repairs |
|---|---:|---:|---:|---:|---|---:|
| cascaded_tanks | 1 | 0.565812494 | 0.56180284 | 5 | ok | 1 |
| decay_multiexp | 2 | 0.316666316 | 0.28673177 | 5 | ok | 0 |
| lotka_volterra | 1 | 0.430620317 | 0.430327422 | 5 | ok | 0 |
| mapk_cascade | 1 | 0.308503722 | 0.307246922 | 5 | ok | 0 |
| piezo_bouc_wen | 1 | 0.266267903 | 0.264352022 | 5 | ok | 0 |
| robertson_session | 1 | 0.379247044 | 0.379078196 | 5 | ok | 0 |
| sliding_basepoint | 1 | 0.118806823 | 0.117002775 | 5 | ok | 2 |
| sliding_basepoint_headered | 1 | 0.109227513 | 0.109010499 | 5 | ok | 2 |
| sneyd_ipr | 9 | 0.0857978412 | 0.0853003469 | 5 | ok | 2 |
| test_session | 1 | 0.379247044 | 0.379078196 | 5 | ok | 0 |
| theophylline | 1 | 0.347334694 | 0.346723879 | 5 | ok | 1 |
| vanderpol_session | 1 | 0.990683999 | 0.989333455 | 5 | ok | 0 |

## Automated regression tests

The full non-local-LLM pytest run initially reported **304 passed, 4 failed, 1 deselected**. Two failures were configuration tests run with `PFIT_*` overrides inherited from `pfit-env.sh`; all four configuration tests passed when rerun without those overrides. The reconciled result is **306 passed and two unresolved legacy-fixture failures**.

- `tests/test_framework.py::test_functions[run_driver_robertson]`: old generated script is stale, and the old fixture has no explicit scientific loss contract to override the automatic log-loss requirement for its wide-range channel.
- `tests/test_framework.py::test_functions[run_driver_vanderpol]`: old generated script is stale.

These rejections were not bypassed, and source stamps were not forged to make old artifacts pass. Current live regenerated Robertson/Van der Pol outcomes are shown separately above.

## Interpretation and follow-up

The numerical fitting, forcing and multi-experiment paths work for the cases that reach fitting, including regenerated reference-backed Sneyd in this run. Local-model generation is not yet reliable across the whole case set. The confirmed remaining failures concern semantic-review output validity, faithful loss extraction, and initial-condition schema adherence. The current check stage correctly blocks the demonstrated MSE/RMSE mismatches.

The main retained-config suite, including the separately header-prepared multi-experiment fixtures, had 11 passes and five failures. The extra `test_session` configuration-regeneration result is reported above rather than erasing that initial failure.

An existing-source offline audit also found older source/specification mismatches, unfinished loss bodies, and an NF-κB column-count mismatch. Those pre-existing files were not silently repaired or scored as fresh successes. See `existing_source_checks.json`.

## Reproduction and retained artifacts

Source the persistent environment and use a new output directory:

```bash
source /workspace/pfit-env.sh
.venv/bin/python scripts/run_case_suite.py --root evaluation_runs/my_next_case_suite --gradient-iters 5
.venv/bin/python scripts/run_case_suite.py --root evaluation_runs/my_next_fixture_suite --cases decay_multiexp sneyd_ipr --add-declared-headers --gradient-iters 5
.venv/bin/python scripts/run_case_suite.py --root evaluation_runs/my_next_config_suite --cases test_session --overwrite-config --gradient-iters 5
```

For pytest, use a shell without `PFIT_*` environment overrides when testing configuration defaults.

Raw artifacts are persistent locally and ignored by Git:

- `evaluation_runs/all_cases_20260926/`: primary manifest, per-stage logs, copied sessions, run snapshots/results/plots and model request logs.
- `evaluation_runs/all_cases_headered_fixture_retry_20260926/`: explicit header preparation and both multi-experiment retries.
- `evaluation_runs/all_cases_fresh_config_retry_20260926/`: auxiliary case with fully regenerated YAML.
- `evaluation_runs/all_cases_20260926/fixture_fidelity.json` and `audit_fixtures.py`: independent comparisons and reproducible audit code.
- `evaluation_runs/all_cases_20260926/existing_source_checks.json`: baseline source audit.
- `evaluation_runs/all_cases_20260926.pytest.log`: original full-suite test output.
- `evaluation_runs/all_cases_20260926.config_pytest.log`: clean-environment configuration-test rerun.

The committed summary and reusable runner are reviewable without the raw artifacts; raw model outputs and numerical runs require the persistent RunPod volume.
