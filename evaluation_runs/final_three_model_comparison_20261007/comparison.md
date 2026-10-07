# Live model comparison

Fresh runs use one attempt per model/case, including the existing repair budget. Times include failed attempts and must be interpreted alongside status. Downloads and warm-up are recorded separately. These are workflow smoke tests, not converged fits.

| Case | qwen32b (status; seconds) | qwen14b (status; seconds) | qwen38_27b (status; seconds) |
|---|---|---|---|
Qwen3.8 is an imported historical baseline with retries across workflow revisions; 32B and 14B are fresh attempts at the recorded commit. Retry/manual passes are labelled. These are cumulative workflow outcomes, not a controlled first-attempt model ranking. The CSV retains original baseline statuses and source paths.

| ARC_fitting | pending | pending | pass; 620.018 |
| boehm_stat5 | pending | pending | pass; 463.132 |
| hodgkin_huxley | pending | pending | blocked; 0 |
| lotka_volterra | pending | pending | pass; 274.647 |
| mapk_cascade | pending | pending | pass (retry); 385.048 |
| nfkb_signaling | pending | pending | fail at new; 168.618 |
| oregonator | pending | pending | pass (retry); 245.199 |
| piezo_bouc_wen | pending | pending | pass; 326.306 |
| robertson_session | pending | pending | pass; 285.93 |
| session1 | pending | pending | blocked; 0 |
| sliding_basepoint_headered | pending | pending | pass (manual retry); 381.428 |
| test_session | pending | pending | pass; 297.741 |
| theophylline | pending | pending | pass; 223.501 |
| vanderpol_session | pending | pending | pass; 213.881 |
| cascaded_tanks | pending | pending | pass; 400.779 |
| decay_multiexp | pending | pending | pass; 272.719 |
| sneyd_ipr | pending | pending | pass; 463.458 |
| beer_indigoidine | pending | pending | pass; 250.156 |
| raia_il13 | pending | pending | pass (retry); 68.254 |
| schwen_insulin | pending | pending | pass; 530.781 |
| armistead_sphingolipid | pending | pending | pass; 387.522 |
| borghans_calcium | pending | pending | pass (retry); 83.504 |
| fujita_egf | pending | pending | pass (retry); 21.569 |

## Counts

| Model | Pass | Fail/degraded | Infrastructure/blocked | Pending/running | Pass / eligible |
|---|---:|---:|---:|---:|---:|
| qwen32b | 0 | 0 | 0 | 23 | 0/21 (0.0%) |
| qwen14b | 0 | 0 | 0 | 23 | 0/21 (0.0%) |
| qwen38_27b | 20 | 1 | 2 | 0 | 20/21 (95.2%) |

Two incomplete folders are excluded from the eligible denominator. test_session duplicates Robertson; sliding_basepoint_headered duplicates sliding_basepoint. Keep these rows visible but do not treat them as independent scientific problems.

A pass requires all five stages, the declared experiment files/count, finite result arrays and final loss, completed refinement, and accepted Ollama diagnosis. The framework applies semantic checks and source-to-JAX fidelity checks. This is not an independent proof that every generated equation matches the original scientific specification.

Models differ in architecture, generation, and quantization. Dense total parameters are used as nominal active counts; active counts for MoE models are approximate. Thinking capability and requested mode are recorded in model metadata. For thinking models, the generation budget also includes reasoning tokens. One trial cannot estimate success probabilities or timing variance.
