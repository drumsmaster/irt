# BRAVE Rasch MML simulation report

**Profile:** `smoke`  
**Monte Carlo fits:** 6  
**Generated:** 2026-08-04T01:53:27

## What this validates

All responses are generated from the Rasch 1PL model used by the estimator. The tests therefore evaluate numerical correctness and parameter recovery, not robustness to 2PL, DIF, local dependence, or careless responding.

The primary administration is random incomplete testing. The adaptive condition is deliberately simple and idealized: it starts adaptively at item 1, uses the true item difficulties for routing, and randomly selects among nearby unused items. There is no warm-up ladder.

Both item and person generating ranges are bounded at approximately -10 to +10 logits. Calibration uses a slightly wider numerical grid so that boundary mass can be diagnosed.

## Internal numerical checks

- Item M-step maximum absolute recovery error: `8.88e-15`
- Item M-step check: `PASS`
- Probability-weight normalization check: `PASS`

## How to read the main metrics

- **Item slope:** regression slope of estimated on true centered item difficulty. 1.0 is correct; above 1 means a stretched scale and below 1 a compressed scale.
- **Item RMSE:** typical item-difficulty error in logits.
- **Tail RMSE:** item RMSE for true difficulties with |b| >= 8.
- **Retained fraction:** fraction of the bank with enough variable responses to calibrate.
- **Latent Wasserstein:** distance in logits between the fitted person distribution and the generating sample distribution; lower is better.
- **Person z SD:** SD of (EAP - true theta) / posterior SD. Around 1 indicates appropriately scaled uncertainty.
- **95% coverage:** fraction of true abilities inside EAP +/- 1.96 posterior SD.
- **Edge mass:** fitted latent probability near the numerical grid boundaries; values above about 0.01 deserve investigation.

## Provisional pass criteria

A row passes when convergence is at least 99%, item slope is within 0.05 of 1, item RMSE is at most 0.40 logits, at least 95% of items are retained, latent edge mass is at most 0.01, and person interval coverage is between 90% and 98%. These are practical validation thresholds, not universal psychometric laws.

## Aggregate results

| scenario               | latent    |   converged_mean |   item_slope_mean |   item_rmse_mean |   item_tail_rmse_mean |   item_retained_fraction_mean |   latent_wasserstein_mean |   person_rmse_mean |   person_z_sd_mean |   person_coverage_95_mean |   latent_edge_mass_mean | overall_pass   |
|:-----------------------|:----------|-----------------:|------------------:|-----------------:|----------------------:|------------------------------:|--------------------------:|-------------------:|-------------------:|--------------------------:|------------------------:|:---------------|
| smoke_normal_random    | normal    |                1 |            1.0132 |           0.5661 |                1.023  |                        0.9024 |                    0.3037 |             0.9963 |             0.9302 |                    0.9733 |                  0.0004 | False          |
| smoke_normal_random    | empirical |                1 |            1.0358 |           0.6007 |                1.0626 |                        0.9024 |                    0.5651 |             1.0936 |             1.9891 |                    0.8267 |                  0      | False          |
| smoke_bimodal_random   | normal    |                1 |            1.0477 |           0.6278 |                0.8778 |                        0.9512 |                    0.9366 |             1.0959 |             1.0441 |                    0.9267 |                  0.0019 | False          |
| smoke_bimodal_random   | empirical |                1 |            1.0761 |           0.7422 |                0.7036 |                        0.9512 |                    0.5962 |             1.1351 |             1.4005 |                    0.8733 |                  0.0033 | False          |
| smoke_bimodal_adaptive | normal    |                1 |            0.9948 |           0.4701 |                0.732  |                        1      |                    1.067  |             0.7768 |             1.0074 |                    0.92   |                  0.001  | False          |
| smoke_bimodal_adaptive | empirical |                1 |            1.0417 |           0.473  |                0.7184 |                        1      |                    0.3761 |             0.7506 |             1.1841 |                    0.9333 |                  0      | False          |

## Scenario-by-scenario interpretation

### smoke_normal_random / normal — REVIEW

Fast code check: wide normal persons, evenly spaced -10 to +10 items, and uniform random incomplete administration.

The recovered item-scale slope was **1.013** and item RMSE was **0.566 logits**. Tail-item RMSE was **1.023 logits**. The estimator retained **90.2%** of items.

Holdout-person RMSE was **0.996 logits**, standardized-error SD was **0.930**, and nominal 95% interval coverage was **97.3%**. Latent-distribution Wasserstein distance was **0.304 logits** and edge mass was **0.0004**.

Criteria requiring review: item_rmse, retention.

### smoke_normal_random / empirical — REVIEW

Fast code check: wide normal persons, evenly spaced -10 to +10 items, and uniform random incomplete administration.

The recovered item-scale slope was **1.036** and item RMSE was **0.601 logits**. Tail-item RMSE was **1.063 logits**. The estimator retained **90.2%** of items.

Holdout-person RMSE was **1.094 logits**, standardized-error SD was **1.989**, and nominal 95% interval coverage was **82.7%**. Latent-distribution Wasserstein distance was **0.565 logits** and edge mass was **0.0000**.

Criteria requiring review: item_rmse, retention, person_coverage.

### smoke_bimodal_random / normal — REVIEW

Fast non-normal check using a wide bimodal person population and random incomplete administration.

The recovered item-scale slope was **1.048** and item RMSE was **0.628 logits**. Tail-item RMSE was **0.878 logits**. The estimator retained **95.1%** of items.

Holdout-person RMSE was **1.096 logits**, standardized-error SD was **1.044**, and nominal 95% interval coverage was **92.7%**. Latent-distribution Wasserstein distance was **0.937 logits** and edge mass was **0.0019**.

Criteria requiring review: item_rmse.

### smoke_bimodal_random / empirical — REVIEW

Fast non-normal check using a wide bimodal person population and random incomplete administration.

The recovered item-scale slope was **1.076** and item RMSE was **0.742 logits**. Tail-item RMSE was **0.704 logits**. The estimator retained **95.1%** of items.

Holdout-person RMSE was **1.135 logits**, standardized-error SD was **1.400**, and nominal 95% interval coverage was **87.3%**. Latent-distribution Wasserstein distance was **0.596 logits** and edge mass was **0.0033**.

Criteria requiring review: scale, item_rmse, person_coverage.

### smoke_bimodal_adaptive / normal — REVIEW

Fast adaptive stress check: no warm-up, broad prior, and random selection among the nearest items at every step.

The recovered item-scale slope was **0.995** and item RMSE was **0.470 logits**. Tail-item RMSE was **0.732 logits**. The estimator retained **100.0%** of items.

Holdout-person RMSE was **0.777 logits**, standardized-error SD was **1.007**, and nominal 95% interval coverage was **92.0%**. Latent-distribution Wasserstein distance was **1.067 logits** and edge mass was **0.0010**.

Criteria requiring review: item_rmse.

### smoke_bimodal_adaptive / empirical — REVIEW

Fast adaptive stress check: no warm-up, broad prior, and random selection among the nearest items at every step.

The recovered item-scale slope was **1.042** and item RMSE was **0.473 logits**. Tail-item RMSE was **0.718 logits**. The estimator retained **100.0%** of items.

Holdout-person RMSE was **0.751 logits**, standardized-error SD was **1.184**, and nominal 95% interval coverage was **93.3%**. Latent-distribution Wasserstein distance was **0.376 logits** and edge mass was **0.0000**.

Criteria requiring review: item_rmse.

## What this report does not establish

This suite intentionally omits model-misspecification scenarios. It does not show that a Rasch model is adequate for real BRAVE data, and it does not evaluate unequal discrimination, DIF, multidimensionality, local dependence, guessing, or disengagement. Those should be investigated separately only after exact-model recovery is satisfactory.

The adaptive router here is an idealized stress test, not a reproduction of BRAVE. Agreement between random and adaptive conditions supports the claim that response-adaptive missingness itself does not distort calibration under these assumptions; disagreement should trigger investigation of exposure and scale connectivity.