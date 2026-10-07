# Live model comparison

Fresh runs use one attempt per model/case, including the existing repair budget. Times include failed attempts and must be interpreted alongside status. Downloads and warm-up are recorded separately. These are workflow smoke tests, not converged fits.

| Case | qwen32b (status; seconds) | qwen14b (status; seconds) | qwen38_27b (status; seconds) |
|---|---|---|---|
| ARC_fitting | pass; 144.44 | pass; 108.767 | pending |
| boehm_stat5 | pass; 181.457 | pass; 130.897 | pending |
| hodgkin_huxley | blocked; 0 | blocked; 0 | pending |
| lotka_volterra | pass; 83.323 | pass; 69.787 | pending |
| mapk_cascade | pass; 126.255 | pass; 94.394 | pending |
| nfkb_signaling | fail at new; 256.57 | fail at new; 95.016 | pending |
| oregonator | pass; 90.254 | pass; 77.677 | pending |
| piezo_bouc_wen | pass; 105.702 | fail at new; 30.809 | pending |
| robertson_session | pass; 100.588 | pass; 85.01 | pending |
| session1 | blocked; 0 | blocked; 0 | pending |
| sliding_basepoint_headered | fail at jax; 82.6 | pass; 84.663 | pending |
| test_session | pass; 96.225 | pass; 84.375 | pending |
| theophylline | pass; 82.223 | pass; 66.568 | pending |
| vanderpol_session | pass; 72.629 | pass; 54.391 | pending |
| cascaded_tanks | pass; 87.662 | fail at new; 42.049 | pending |
| decay_multiexp | pass; 85.322 | pass; 82.187 | pending |
| sneyd_ipr | fail at new; 156.673 | pass; 209.931 | pending |
| beer_indigoidine | pass; 82.32 | pass; 70.445 | pending |
| raia_il13 | pass; 195.41 | fail at new; 67.008 | pending |
| schwen_insulin | fail at new; 68.542 | fail at check; 69.829 | pending |
| armistead_sphingolipid | fail at new; 64.696 | fail at new; 42.834 | pending |
| borghans_calcium | pass; 116.434 | fail at new; 41.119 | pending |
| fujita_egf | fail at new; 48.978 | fail at new; 116.522 | pending |

## Counts

| Model | Pass | Fail/degraded | Infrastructure/blocked | Pending/running | Pass / eligible |
|---|---:|---:|---:|---:|---:|
| qwen32b | 15 | 6 | 2 | 0 | 15/21 (71.4%) |
| qwen14b | 13 | 8 | 2 | 0 | 13/21 (61.9%) |
| qwen38_27b | 0 | 0 | 0 | 23 | 0/21 (0.0%) |

Two incomplete folders are excluded from the eligible denominator. test_session duplicates Robertson; sliding_basepoint_headered duplicates sliding_basepoint. Keep these rows visible but do not treat them as independent scientific problems.

A pass requires all five stages, the declared experiment files/count, finite result arrays and final loss, completed refinement, and accepted Ollama diagnosis. The framework applies semantic checks and source-to-JAX fidelity checks. This is not an independent proof that every generated equation matches the original scientific specification.

Models differ in architecture, generation, and quantization. Dense total parameters are used as nominal active counts; active counts for MoE models are approximate. Thinking capability and requested mode are recorded in model metadata. For thinking models, the generation budget also includes reasoning tokens. One trial cannot estimate success probabilities or timing variance.
