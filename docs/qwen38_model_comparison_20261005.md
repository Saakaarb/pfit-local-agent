# Completed model comparison

Single attempt per model/case, including the existing repair budget. Times include failed attempts and must be interpreted alongside status. Downloads and warm-up are recorded separately. These are workflow smoke tests, not converged fits.

| Case | qwen32b (status; seconds) | qwen14b (status; seconds) | qwen38_27b (status; seconds) |
|---|---|---|---|
| ARC_fitting | pass; 124.357 | pass; 93.859 | pass; 350.281 |
| boehm_stat5 | pass; 163.913 | pass; 123.887 | pass; 363.489 |
| hodgkin_huxley | blocked; 0 | blocked; 0 | blocked; 0 |
| lotka_volterra | pass; 69.503 | pass; 57.364 | pass; 216.867 |
| mapk_cascade | pass; 129.969 | pass; 104.904 | pass; 276.439 |
| nfkb_signaling | fail at new; 247.685 | fail at new; 100.56 | fail at new; 164.024 |
| oregonator | pass; 97.427 | pass; 88.682 | fail at new; 85.008 |
| piezo_bouc_wen | pass; 110.739 | fail at new; 31.047 | pass; 301.301 |
| robertson_session | pass; 84.429 | pass; 71.083 | pass; 244.685 |
| session1 | blocked; 0 | blocked; 0 | blocked; 0 |
| sliding_basepoint | pass; 105.137 | fail at new; 39.244 | fail at jax; 230.061 |
| sliding_basepoint_headered | pass; 100.882 | pass; 76.934 | fail at jax; 240.266 |
| test_session | pass; 86.539 | pass; 71.166 | pass; 282.858 |
| theophylline | pass; 70.258 | pass; 60.034 | pass; 200.809 |
| vanderpol_session | pass; 72.953 | pass; 54.891 | pass; 213.239 |
| cascaded_tanks | degraded; 82.422 | fail at new; 42.447 | degraded; 223.763 |
| decay_multiexp | pass; 73.886 | pass; 71.732 | pass; 224.104 |
| sneyd_ipr | fail at new; 143.854 | pass; 252.621 | pass; 503.024 |
| beer_indigoidine | pass; 159.883 | pass; 122.038 | pass; 306.116 |
| raia_il13 | pass; 238.163 | fail at new; 161.575 | pass; 373.798 |
| schwen_insulin | fail at new; 71.607 | fail at new; 67.693 | pass; 407.43 |
| armistead_sphingolipid | fail at new; 74.189 | fail at new; 49.376 | pass; 347.834 |
| borghans_calcium | pass; 167.87 | fail at new; 53.743 | pass; 287.245 |
| fujita_egf | fail at new; 76.276 | fail at new; 133.467 | pass; 390.071 |

## Counts

| Model | Pass | Fail/degraded | Infrastructure/blocked | Pending/running |
|---|---:|---:|---:|---:|
| qwen32b | 16 | 6 | 2 | 0 |
| qwen14b | 13 | 9 | 2 | 0 |
| qwen38_27b | 17 | 5 | 2 | 0 |

Two incomplete folders are excluded from the eligible denominator. test_session duplicates Robertson; sliding_basepoint_headered duplicates sliding_basepoint. Keep these rows visible but do not treat them as independent scientific problems.

A pass requires all five stages, the declared experiment files/count, finite result arrays and final loss, completed refinement, and accepted Ollama diagnosis. The framework applies semantic checks and source-to-JAX fidelity checks. This is not an independent proof that every generated equation matches the original scientific specification.

Models differ in architecture, generation, and quantization. Dense total parameters are used as nominal active counts; active counts for MoE models are approximate. Thinking capability and requested mode are recorded in model metadata. For thinking models, the generation budget also includes reasoning tokens. One trial cannot estimate success probabilities or timing variance.

See [batch protocol and baseline provenance](BENCHMARK_NOTES.md).

## Post-run scientific audit: headered sliding-basepoint case

The 32B workflow pass is a scientific false positive for this case: it replaced
the prompt's low-force AND low-speed sticking condition with OR. The archived
trajectory has x2 = v2 = 0 throughout; the sliding mass never moves. Its midpoint
validation completed in 76 steps because it solved this altered system.

Qwen3.8 smoothed the two-condition sticking gate with widths 0.01*c2 and 0.01*vf,
but retained discontinuous sign(v2) friction. It exhausted 50,000 steps (11,395
accepted, 38,605 rejected). At force 0.99*c2, the prompt gives zero acceleration
near zero speed; the extracted model instead switches between about +7501 and
-37.7 m/s² at v2 = -1e-8 and +1e-8.

In isolated diagnostic integrations, changing only sign(v2) to
tanh(v2/(0.01*vf)) completed in 270 steps; tanh(v2/vf) completed in 690 steps.
Neither change is an accepted scientific correction: smoothing alters the model
and its width must be specified deliberately. The original benchmark attempts
and raw workflow statuses remain preserved. No full fit was run for this audit.

The equation-extraction system prompt currently encourages smoothing switches
without a defined width. Source-to-JAX fidelity checks cannot detect an error
already present in the extracted source. The proposed correction is to preserve
the supplied AND logic, prevent silent regularization choices, and check sticking
and release behavior explicitly before treating a numerical pass as success.

Evidence: `evaluation_runs/sliding_diagnosis_20261005/headered_findings.json`,
`point_probes.json`, and `regularization_probe.json`, with reproducible probes.
