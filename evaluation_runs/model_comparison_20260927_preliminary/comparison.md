> PRELIMINARY — excluded from scoring: benchmark metadata entered session context.

# Live model comparison

Single attempt per model/case, including the existing repair budget. Times include failed attempts and must be interpreted alongside status. Downloads and warm-up are recorded separately. These are workflow smoke tests, not converged fits.

| Case | qwen32b (status; seconds) | qwen14b (status; seconds) | qwen3_coder_next (status; seconds) |
|---|---|---|---|
| ARC_fitting | pass; 158.104 | pass; 119.688 | pending |
| boehm_stat5 | pass; 213.892 | pass; 124.054 | pending |
| hodgkin_huxley | blocked; 0 | blocked; 0 | pending |
| lotka_volterra | pass; 72.195 | pass; 62.423 | pending |
| mapk_cascade | pass; 135.322 | pass; 103.041 | pending |
| nfkb_signaling | fail at new; 268.253 | fail at new; 352.143 | pending |
| oregonator | fail at new; 58.636 | pending | pending |
| piezo_bouc_wen | pass; 114.413 | pending | pending |
| robertson_session | pass; 88.717 | pending | pending |
| session1 | blocked; 0 | pending | pending |
| sliding_basepoint | pass; 134.319 | pending | pending |
| sliding_basepoint_headered | pass; 124.906 | pending | pending |
| test_session | pass; 92.558 | pending | pending |
| theophylline | pass; 89.047 | fail at new; 26.336 | pending |
| vanderpol_session | pass; 73.442 | pending | pending |
| cascaded_tanks | degraded; 97.84 | fail at new; 34.322 | pending |
| decay_multiexp | pass; 81.996 | pass; 67.884 | pending |
| sneyd_ipr | fail at new; 40.081 | pending | pending |

## Counts

| Model | Pass | Fail/degraded | Infrastructure/blocked | Pending/running |
|---|---:|---:|---:|---:|
| qwen32b | 12 | 4 | 2 | 0 |
| qwen14b | 5 | 3 | 1 | 9 |
| qwen3_coder_next | 0 | 0 | 0 | 18 |

Two incomplete folders are excluded from the eligible denominator. test_session duplicates Robertson; sliding_basepoint_headered duplicates sliding_basepoint. Keep these rows visible but do not treat them as independent scientific problems.

A pass requires all five stages, the declared experiment files/count, finite result arrays and final loss, completed refinement, and accepted Ollama diagnosis. The framework applies semantic checks and source-to-JAX fidelity checks. This is not an independent proof that every generated equation matches the original scientific specification.

Models differ in architecture, generation, and quantization. Dense total parameters are used as nominal active counts; the MoE 3B active count is approximate. All are non-thinking. One trial cannot estimate success probabilities or timing variance.
