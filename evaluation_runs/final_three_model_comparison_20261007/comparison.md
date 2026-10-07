# Live model comparison

Fresh runs use one attempt per model/case, including the existing repair budget. Times include failed attempts and must be interpreted alongside status. Downloads and warm-up are recorded separately. These are workflow smoke tests, not converged fits.

| Case | qwen32b (status; seconds) | qwen14b (status; seconds) | qwen38_27b (status; seconds) |
|---|---|---|---|
Qwen3.8 is an imported historical baseline with retries across workflow revisions; 32B and 14B are fresh attempts at the recorded commit. Retry/manual passes are labelled. These are cumulative workflow outcomes, not a controlled first-attempt model ranking. The CSV retains original baseline statuses and source paths.

| ARC_fitting | pending | pass; 108.767 | pass; 620.018 |
| boehm_stat5 | pending | pass; 130.897 | pass; 463.132 |
| hodgkin_huxley | pending | blocked; 0 | blocked; 0 |
| lotka_volterra | pending | pass; 69.787 | pass; 274.647 |
| mapk_cascade | pending | pass; 94.394 | pass (retry); 385.048 |
| nfkb_signaling | pending | fail at new; 95.016 | fail at new; 168.618 |
| oregonator | pending | pass; 77.677 | pass (retry); 245.199 |
| piezo_bouc_wen | pending | fail at new; 30.809 | pass; 326.306 |
| robertson_session | pending | pass; 85.01 | pass; 285.93 |
| session1 | pending | blocked; 0 | blocked; 0 |
| sliding_basepoint_headered | pending | pass; 84.663 | pass (manual retry); 381.428 |
| test_session | pending | pass; 84.375 | pass; 297.741 |
| theophylline | pending | pass; 66.568 | pass; 223.501 |
| vanderpol_session | pending | pass; 54.391 | pass; 213.881 |
| cascaded_tanks | pending | fail at new; 42.049 | pass; 400.779 |
| decay_multiexp | pending | pass; 82.187 | pass; 272.719 |
| sneyd_ipr | pending | pass; 209.931 | pass; 463.458 |
| beer_indigoidine | pending | pass; 70.445 | pass; 250.156 |
| raia_il13 | pending | fail at new; 67.008 | pass (retry); 68.254 |
| schwen_insulin | pending | fail at check; 69.829 | pass; 530.781 |
| armistead_sphingolipid | pending | fail at new; 42.834 | pass; 387.522 |
| borghans_calcium | pending | fail at new; 41.119 | pass (retry); 83.504 |
| fujita_egf | pending | fail at new; 116.522 | pass (retry); 21.569 |

## Counts

| Model | Pass | Fail/degraded | Infrastructure/blocked | Pending/running | Pass / eligible |
|---|---:|---:|---:|---:|---:|
| qwen32b | 0 | 0 | 0 | 23 | 0/21 (0.0%) |
| qwen14b | 13 | 8 | 2 | 0 | 13/21 (61.9%) |
| qwen38_27b | 20 | 1 | 2 | 0 | 20/21 (95.2%) |

Two incomplete folders are excluded from the eligible denominator. test_session duplicates Robertson; sliding_basepoint_headered duplicates sliding_basepoint. Keep these rows visible but do not treat them as independent scientific problems.

A pass requires all five stages, the declared experiment files/count, finite result arrays and final loss, completed refinement, and accepted Ollama diagnosis. The framework applies semantic checks and source-to-JAX fidelity checks. This is not an independent proof that every generated equation matches the original scientific specification.

Models differ in architecture, generation, and quantization. Dense total parameters are used as nominal active counts; active counts for MoE models are approximate. Thinking capability and requested mode are recorded in model metadata. For thinking models, the generation budget also includes reasoning tokens. One trial cannot estimate success probabilities or timing variance.
