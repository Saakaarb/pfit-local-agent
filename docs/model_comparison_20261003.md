# Live model comparison

Single attempt per model/case, including the existing repair budget. Times include failed attempts and must be interpreted alongside status. Downloads and warm-up are recorded separately. These are workflow smoke tests, not converged fits.

| Case | qwen32b (status; seconds) | qwen14b (status; seconds) |
|---|---|---|
| ARC_fitting | pass; 124.357 | pass; 93.859 |
| boehm_stat5 | pass; 163.913 | pass; 123.887 |
| hodgkin_huxley | blocked; 0 | blocked; 0 |
| lotka_volterra | pass; 69.503 | pass; 57.364 |
| mapk_cascade | pass; 129.969 | pass; 104.904 |
| nfkb_signaling | fail at new; 247.685 | fail at new; 100.56 |
| oregonator | pass; 97.427 | pass; 88.682 |
| piezo_bouc_wen | pass; 110.739 | fail at new; 31.047 |
| robertson_session | pass; 84.429 | pass; 71.083 |
| session1 | blocked; 0 | blocked; 0 |
| sliding_basepoint | pass; 105.137 | fail at new; 39.244 |
| sliding_basepoint_headered | pass; 100.882 | pass; 76.934 |
| test_session | pass; 86.539 | pass; 71.166 |
| theophylline | pass; 70.258 | pass; 60.034 |
| vanderpol_session | pass; 72.953 | pass; 54.891 |
| cascaded_tanks | degraded; 82.422 | fail at new; 42.447 |
| decay_multiexp | pass; 73.886 | pass; 71.732 |
| sneyd_ipr | fail at new; 143.854 | pass; 252.621 |

## Counts

| Model | Pass | Fail/degraded | Infrastructure/blocked | Pending/running |
|---|---:|---:|---:|---:|
| qwen32b | 13 | 3 | 2 | 0 |
| qwen14b | 12 | 4 | 2 | 0 |

Two incomplete folders are excluded from the eligible denominator. test_session duplicates Robertson; sliding_basepoint_headered duplicates sliding_basepoint. Keep these rows visible but do not treat them as independent scientific problems.

A pass requires all five stages, the declared experiment files/count, finite result arrays and final loss, completed refinement, and accepted Ollama diagnosis. The framework applies semantic checks and source-to-JAX fidelity checks. This is not an independent proof that every generated equation matches the original scientific specification.

Models differ in architecture, generation, and quantization. Dense total parameters are used as nominal active counts; the MoE 3B active count is approximate. All are non-thinking. One trial cannot estimate success probabilities or timing variance.
