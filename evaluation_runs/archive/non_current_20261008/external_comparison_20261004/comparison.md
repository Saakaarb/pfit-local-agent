# External-case smoke comparison

One attempt per model/case, including the existing repair budget. DE: population 4, one iteration; Adam: five iterations; no sloppiness analysis. A workflow pass is not a converged fit or an independent validation of the scientific extraction.

| Case | Qwen2.5-Coder 32B | Qwen2.5-Coder 14B |
|---|---|---|
| beer_indigoidine | pass (159.883 s) | pass (122.038 s) |
| raia_il13 | pass (238.163 s) | fail at new (161.575 s) |
| schwen_insulin | fail at new (71.607 s) | fail at new (67.693 s) |
| armistead_sphingolipid | fail at new (74.189 s) | fail at new (49.376 s) |
| borghans_calcium | pass (167.87 s) | fail at new (53.743 s) |
| fujita_egf | fail at new (76.276 s) | fail at new (133.467 s) |

The first three 32B attempts may be imported from the earlier external smoke run; per-case import records preserve their origin. All models receive the same frozen prose/CSV inputs. Model digests, actual sizes, parameter counts, GPU placement, warm-up and download times are in models/. Case times include repairs and failed attempts, and exclude model preparation. Both selected models are dense, non-thinking Qwen2.5-Coder models. These six cases are separate from the earlier 16-case comparison. Single attempts do not estimate success probabilities or timing variance.

## Recorded extraction failures

| Case | 32B | 14B |
|---|---|---|
| Raia | Passed | Unknown name `IL13_Rec` |
| Schwen | Output also declared as forcing input, causing a model-name collision | Duplicate model names |
| Armistead | Missing RHS for condition indicator `H` | Invalid equations JSON |
| Borghans | Passed | Duplicate model names |
| Fujita | Forcing list contained names absent from CSV headers | Unknown name `L` |

These are workflow/extraction outcomes, not failures to optimize the published
models. No case was manually repaired or retried after seeing these outcomes.
The three successful 32B cases and Beer with 14B completed all five stages.
The extremely small fit budget does not demonstrate convergence.

Hardware: NVIDIA H200, with fitting on CPU/float64. The host exposes 192 logical
CPUs, but the container CPU quota is 20.4 CPU equivalents; smoke tests requested
one population worker. The memory limit is 250,999,996,416 bytes (about 234 GiB).

Full evidence: `evaluation_runs/external_comparison_20261004/`. Imported 32B
artifacts retain their original source paths and framework commit. Only benchmark
assembly and runner files changed between the original and added attempts.
