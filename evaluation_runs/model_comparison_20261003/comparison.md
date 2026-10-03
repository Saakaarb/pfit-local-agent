Batch stopped at user request for helper-interface correction; not a completed comparison.

# Live model comparison

Single attempt per model/case, including the existing repair budget. Times include failed attempts and must be interpreted alongside status. Downloads and warm-up are recorded separately. These are workflow smoke tests, not converged fits.

| Case | qwen32b (status; seconds) | qwen14b (status; seconds) |
|---|---|---|
| ARC_fitting | fail at jax; 72.503 | pending |
| boehm_stat5 | fail at jax; 110.253 | pending |
| hodgkin_huxley | blocked; 0 | pending |
| lotka_volterra | pass; 68.924 | pending |
| mapk_cascade | pass; 133.082 | pending |
| nfkb_signaling | fail at new; 253.227 | pending |
| oregonator | pass; 96.353 | pending |
| piezo_bouc_wen | pass; 110.609 | pending |
| robertson_session | pass; 84.324 | pending |
| session1 | blocked; 0 | pending |
| sliding_basepoint | fail at jax; 74.278 | pending |
| sliding_basepoint_headered | fail at jax; 62.362 | pending |
| test_session | pass; 81.194 | pending |
| theophylline | fail at jax; 52.542 | pending |
| vanderpol_session | interrupted | pending |
| cascaded_tanks | fail at jax; 48.812 | pending |
| decay_multiexp | pass; 76.76 | pending |
| sneyd_ipr | fail at new; 142.713 | pending |

## Counts

| Model | Pass | Fail/degraded | Infrastructure/blocked | Pending/running |
|---|---:|---:|---:|---:|
| qwen32b | 7 | 8 | 3 | 0 |
| qwen14b | 0 | 0 | 0 | 18 |

Two incomplete folders are excluded from the eligible denominator. test_session duplicates Robertson; sliding_basepoint_headered duplicates sliding_basepoint. Keep these rows visible but do not treat them as independent scientific problems.

A pass requires all five stages, the declared experiment files/count, finite result arrays and final loss, completed refinement, and accepted Ollama diagnosis. The framework applies semantic checks and source-to-JAX fidelity checks. This is not an independent proof that every generated equation matches the original scientific specification.

Models differ in architecture, generation, and quantization. Dense total parameters are used as nominal active counts; the MoE 3B active count is approximate. All are non-thinking. One trial cannot estimate success probabilities or timing variance.
