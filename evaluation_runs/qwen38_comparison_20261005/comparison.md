# Live model comparison

Single attempt per model/case, including the existing repair budget. Times include failed attempts and must be interpreted alongside status. Downloads and warm-up are recorded separately. These are workflow smoke tests, not converged fits.

| Case | qwen32b (status; seconds) | qwen14b (status; seconds) | qwen38_27b (status; seconds) |
|---|---|---|---|
| ARC_fitting | pass; 124.357 | pass; 93.859 | pending |
| boehm_stat5 | pass; 163.913 | pass; 123.887 | pending |
| hodgkin_huxley | blocked; 0 | blocked; 0 | pending |
| lotka_volterra | pass; 69.503 | pass; 57.364 | pending |
| mapk_cascade | pass; 129.969 | pass; 104.904 | pending |
| nfkb_signaling | fail at new; 247.685 | fail at new; 100.56 | pending |
| oregonator | pass; 97.427 | pass; 88.682 | pending |
| piezo_bouc_wen | pass; 110.739 | fail at new; 31.047 | pending |
| robertson_session | pass; 84.429 | pass; 71.083 | pending |
| session1 | blocked; 0 | blocked; 0 | pending |
| sliding_basepoint | pass; 105.137 | fail at new; 39.244 | pending |
| sliding_basepoint_headered | pass; 100.882 | pass; 76.934 | pending |
| test_session | pass; 86.539 | pass; 71.166 | pending |
| theophylline | pass; 70.258 | pass; 60.034 | pending |
| vanderpol_session | pass; 72.953 | pass; 54.891 | pending |
| cascaded_tanks | degraded; 82.422 | fail at new; 42.447 | pending |
| decay_multiexp | pass; 73.886 | pass; 71.732 | pending |
| sneyd_ipr | fail at new; 143.854 | pass; 252.621 | pending |
| beer_indigoidine | pass; 159.883 | pass; 122.038 | pending |
| raia_il13 | pass; 238.163 | fail at new; 161.575 | pending |
| schwen_insulin | fail at new; 71.607 | fail at new; 67.693 | pending |
| armistead_sphingolipid | fail at new; 74.189 | fail at new; 49.376 | pending |
| borghans_calcium | pass; 167.87 | fail at new; 53.743 | pending |
| fujita_egf | fail at new; 76.276 | fail at new; 133.467 | pending |

## Counts

| Model | Pass | Fail/degraded | Infrastructure/blocked | Pending/running |
|---|---:|---:|---:|---:|
| qwen32b | 16 | 6 | 2 | 0 |
| qwen14b | 13 | 9 | 2 | 0 |
| qwen38_27b | 0 | 0 | 0 | 24 |

Two incomplete folders are excluded from the eligible denominator. test_session duplicates Robertson; sliding_basepoint_headered duplicates sliding_basepoint. Keep these rows visible but do not treat them as independent scientific problems.

A pass requires all five stages, the declared experiment files/count, finite result arrays and final loss, completed refinement, and accepted Ollama diagnosis. The framework applies semantic checks and source-to-JAX fidelity checks. This is not an independent proof that every generated equation matches the original scientific specification.

Models differ in architecture, generation, and quantization. Dense total parameters are used as nominal active counts; active counts for MoE models are approximate. Thinking capability and requested mode are recorded in model metadata. For thinking models, the generation budget also includes reasoning tokens. One trial cannot estimate success probabilities or timing variance.

See [batch protocol and baseline provenance](BENCHMARK_NOTES.md).
