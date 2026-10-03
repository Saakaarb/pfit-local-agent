# Live model comparison

Single attempt per model/case, including the existing repair budget. Times include failed attempts and must be interpreted alongside status. Downloads and warm-up are recorded separately. These are workflow smoke tests, not converged fits.

| Case | qwen32b (status; seconds) | qwen14b (status; seconds) | qwen3_coder_next (status; seconds) |
|---|---|---|---|
| ARC_fitting | pass; 148.098 | pass; 92.906 | fail at new; 4.204 |
| boehm_stat5 | pass; 220.747 | pass; 125.341 | pass; 109.961 |
| hodgkin_huxley | blocked; 0 | blocked; 0 | blocked |
| lotka_volterra | pass; 66.487 | pass; 55.823 | pass; 60.862 |
| mapk_cascade | pass; 127.194 | pass; 98.516 | pass; 115.301 |
| nfkb_signaling | fail at new; 235.787 | fail at new; 93.06 | fail at new; 73.957 |
| oregonator | fail at jax; 137.916 | fail at jax; 75.929 | fail at jax; 109.21 |
| piezo_bouc_wen | pass; 109.221 | fail at new; 30.641 | pass; 99.383 |
| robertson_session | pass; 82.169 | pass; 75.956 | pass; 77.336 |
| session1 | blocked; 0 | blocked; 0 | blocked |
| sliding_basepoint | pass; 122.226 | fail at new; 38.976 | pass; 86.022 |
| sliding_basepoint_headered | pass; 108.965 | fail at new; 37.201 | pass; 79.533 |
| test_session | pass; 82.451 | pass; 69.479 | pass; 76.709 |
| theophylline | pass; 83.606 | pass; 57.652 | pass; 107.828 |
| vanderpol_session | pass; 74.16 | pass; 54.672 | pass; 63.097 |
| cascaded_tanks | degraded; 90.986 | fail at new; 41.409 | degraded; 80.791 |
| decay_multiexp | pass; 76.105 | pass; 63.094 | pass; 67.282 |
| sneyd_ipr | fail at new; 33.253 | pass; 259.274 | fail at check; 71.364 |

## Counts

| Model | Pass | Fail/degraded | Infrastructure/blocked | Pending/running |
|---|---:|---:|---:|---:|
| qwen32b | 12 | 4 | 2 | 0 |
| qwen14b | 10 | 6 | 2 | 0 |
| qwen3_coder_next | 11 | 5 | 2 | 0 |

Two incomplete folders are excluded from the eligible denominator. test_session duplicates Robertson; sliding_basepoint_headered duplicates sliding_basepoint. Keep these rows visible but do not treat them as independent scientific problems.

A pass requires all five stages, the declared experiment files/count, finite result arrays and final loss, completed refinement, and accepted Ollama diagnosis. The framework applies semantic checks and source-to-JAX fidelity checks. This is not an independent proof that every generated equation matches the original scientific specification.

Models differ in architecture, generation, and quantization. Dense total parameters are used as nominal active counts; the MoE 3B active count is approximate. All are non-thinking. One trial cannot estimate success probabilities or timing variance.
