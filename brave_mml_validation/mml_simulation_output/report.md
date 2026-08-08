# BRAVE Rasch MML simulation report

**Profile:** `core`  
**Monte Carlo fits:** 400  
**Generated:** 2026-08-03T22:45:05

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

             scenario    latent converged_mean item_slope_mean item_rmse_mean item_tail_rmse_mean item_retained_fraction_mean latent_wasserstein_mean person_rmse_mean person_z_sd_mean person_coverage_95_mean latent_edge_mass_mean  overall_pass
        normal_random    normal         1.0000          1.0119         0.3019              0.5178                      0.9998                  0.1281           0.7180           0.9913                  0.9520                0.0002          True
        normal_random empirical         1.0000          1.0101         0.2877              0.4762                      0.9998                  0.1423           0.7197           1.0227                  0.9443                0.0002          True
        skewed_random    normal         1.0000          0.9968         0.3734              0.6232                      0.9505                  0.5332           0.7295           1.0231                  0.9417                0.0000          True
        skewed_random empirical         1.0000          1.0050         0.3590              0.6598                      0.9505                  0.1269           0.6982           1.0190                  0.9439                0.0001          True
       bimodal_random    normal         1.0000          1.0039         0.3665              0.5745                      0.9993                  0.9450           0.7654           1.0579                  0.9359                0.0007          True
       bimodal_random empirical         1.0000          1.0144         0.2887              0.4945                      0.9993                  0.1299           0.7003           1.0227                  0.9451                0.0001          True
    heavy_tail_random    normal         1.0000          1.0078         0.3040              0.5241                      0.9995                  0.1631           0.7253           1.0032                  0.9480                0.0001          True
    heavy_tail_random empirical         1.0000          1.0117         0.3028              0.5128                      0.9995                  0.1474           0.7257           1.0468                  0.9393                0.0002          True
      normal_adaptive    normal         1.0000          1.0006         0.1604              0.2743                      1.0000                  0.1304           0.3792           1.0087                  0.9473                0.0002          True
      normal_adaptive empirical         1.0000          1.0001         0.1558              0.2645                      1.0000                  0.1020           0.3803           1.0426                  0.9397                0.0000          True
     bimodal_adaptive    normal         1.0000          0.9708         0.3362              0.3619                      0.9953                  0.9831           0.5102           1.3475                  0.8468                0.0004         False
     bimodal_adaptive empirical         1.0000          1.0004         0.1816              0.3412                      0.9953                  0.1178           0.3795           1.1078                  0.9340                0.0000          True
 bimodal_small_sample    normal         1.0000          1.0107         0.5155              0.7916                      0.9604                  0.9323           0.7662           1.0454                  0.9391                0.0008         False
 bimodal_small_sample empirical         1.0000          1.0269         0.4888              0.7364                      0.9604                  0.2637           0.7277           1.1123                  0.9245                0.0007         False
   bimodal_short_test    normal         1.0000          1.0087         0.5505              0.8748                      0.9888                  0.9446           1.1064           1.0765                  0.9298                0.0006         False
   bimodal_short_test empirical         1.0000          1.0153         0.3812              0.6372                      0.9888                  0.1761           0.9788           1.0218                  0.9436                0.0005          True
    bimodal_long_test    normal         1.0000          0.9977         0.2261              0.3361                      1.0000                  0.9367           0.5311           1.0489                  0.9385                0.0007          True
    bimodal_long_test empirical         1.0000          1.0053         0.1850              0.3094                      1.0000                  0.1054           0.4999           1.0294                  0.9432                0.0000          True
bimodal_centered_bank    normal         1.0000          1.0263         0.3519              0.7232                      0.9990                  0.9260           0.7489           1.0158                  0.9458                0.0009          True
bimodal_centered_bank empirical         1.0000          1.0074         0.2521              0.4745                      0.9990                  0.1302           0.6904           1.0287                  0.9445                0.0001          True

## Scenario-by-scenario interpretation

### normal_random / normal — PASS

Primary baseline: wide normal persons and uniform random incomplete administration over a -10 to +10 bank.

The recovered item-scale slope was **1.012** and item RMSE was **0.302 logits**. Tail-item RMSE was **0.518 logits**. The estimator retained **100.0%** of items.

Holdout-person RMSE was **0.718 logits**, standardized-error SD was **0.991**, and nominal 95% interval coverage was **95.2%**. Latent-distribution Wasserstein distance was **0.128 logits** and edge mass was **0.0002**.

### normal_random / empirical — PASS

Primary baseline: wide normal persons and uniform random incomplete administration over a -10 to +10 bank.

The recovered item-scale slope was **1.010** and item RMSE was **0.288 logits**. Tail-item RMSE was **0.476 logits**. The estimator retained **100.0%** of items.

Holdout-person RMSE was **0.720 logits**, standardized-error SD was **1.023**, and nominal 95% interval coverage was **94.4%**. Latent-distribution Wasserstein distance was **0.142 logits** and edge mass was **0.0002**.

### skewed_random / normal — PASS

Strongly skewed but unimodal person distribution; tests whether empirical MML avoids normal-distribution scale distortion.

The recovered item-scale slope was **0.997** and item RMSE was **0.373 logits**. Tail-item RMSE was **0.623 logits**. The estimator retained **95.0%** of items.

Holdout-person RMSE was **0.729 logits**, standardized-error SD was **1.023**, and nominal 95% interval coverage was **94.2%**. Latent-distribution Wasserstein distance was **0.533 logits** and edge mass was **0.0000**.

### skewed_random / empirical — PASS

Strongly skewed but unimodal person distribution; tests whether empirical MML avoids normal-distribution scale distortion.

The recovered item-scale slope was **1.005** and item RMSE was **0.359 logits**. Tail-item RMSE was **0.660 logits**. The estimator retained **95.0%** of items.

Holdout-person RMSE was **0.698 logits**, standardized-error SD was **1.019**, and nominal 95% interval coverage was **94.4%**. Latent-distribution Wasserstein distance was **0.127 logits** and edge mass was **0.0001**.

### bimodal_random / normal — PASS

Wide bimodal population with random incomplete administration; the main test of the empirical latent distribution.

The recovered item-scale slope was **1.004** and item RMSE was **0.366 logits**. Tail-item RMSE was **0.575 logits**. The estimator retained **99.9%** of items.

Holdout-person RMSE was **0.765 logits**, standardized-error SD was **1.058**, and nominal 95% interval coverage was **93.6%**. Latent-distribution Wasserstein distance was **0.945 logits** and edge mass was **0.0007**.

### bimodal_random / empirical — PASS

Wide bimodal population with random incomplete administration; the main test of the empirical latent distribution.

The recovered item-scale slope was **1.014** and item RMSE was **0.289 logits**. Tail-item RMSE was **0.495 logits**. The estimator retained **99.9%** of items.

Holdout-person RMSE was **0.700 logits**, standardized-error SD was **1.023**, and nominal 95% interval coverage was **94.5%**. Latent-distribution Wasserstein distance was **0.130 logits** and edge mass was **0.0001**.

### heavy_tail_random / normal — PASS

Heavy-tailed person population with random incomplete administration; stresses the outer scale without imposing multiple groups.

The recovered item-scale slope was **1.008** and item RMSE was **0.304 logits**. Tail-item RMSE was **0.524 logits**. The estimator retained **100.0%** of items.

Holdout-person RMSE was **0.725 logits**, standardized-error SD was **1.003**, and nominal 95% interval coverage was **94.8%**. Latent-distribution Wasserstein distance was **0.163 logits** and edge mass was **0.0001**.

### heavy_tail_random / empirical — PASS

Heavy-tailed person population with random incomplete administration; stresses the outer scale without imposing multiple groups.

The recovered item-scale slope was **1.012** and item RMSE was **0.303 logits**. Tail-item RMSE was **0.513 logits**. The estimator retained **100.0%** of items.

Holdout-person RMSE was **0.726 logits**, standardized-error SD was **1.047**, and nominal 95% interval coverage was **93.9%**. Latent-distribution Wasserstein distance was **0.147 logits** and edge mass was **0.0002**.

### normal_adaptive / normal — PASS

Idealized fully adaptive randomesque administration with a normal population; isolates adaptive missingness from latent non-normality.

The recovered item-scale slope was **1.001** and item RMSE was **0.160 logits**. Tail-item RMSE was **0.274 logits**. The estimator retained **100.0%** of items.

Holdout-person RMSE was **0.379 logits**, standardized-error SD was **1.009**, and nominal 95% interval coverage was **94.7%**. Latent-distribution Wasserstein distance was **0.130 logits** and edge mass was **0.0002**.

### normal_adaptive / empirical — PASS

Idealized fully adaptive randomesque administration with a normal population; isolates adaptive missingness from latent non-normality.

The recovered item-scale slope was **1.000** and item RMSE was **0.156 logits**. Tail-item RMSE was **0.265 logits**. The estimator retained **100.0%** of items.

Holdout-person RMSE was **0.380 logits**, standardized-error SD was **1.043**, and nominal 95% interval coverage was **94.0%**. Latent-distribution Wasserstein distance was **0.102 logits** and edge mass was **0.0000**.

### bimodal_adaptive / normal — REVIEW

Idealized fully adaptive randomesque administration under a wide bimodal population; the main adaptive stress test.

The recovered item-scale slope was **0.971** and item RMSE was **0.336 logits**. Tail-item RMSE was **0.362 logits**. The estimator retained **99.5%** of items.

Holdout-person RMSE was **0.510 logits**, standardized-error SD was **1.348**, and nominal 95% interval coverage was **84.7%**. Latent-distribution Wasserstein distance was **0.983 logits** and edge mass was **0.0004**.

Criteria requiring review: person_coverage.

### bimodal_adaptive / empirical — PASS

Idealized fully adaptive randomesque administration under a wide bimodal population; the main adaptive stress test.

The recovered item-scale slope was **1.000** and item RMSE was **0.182 logits**. Tail-item RMSE was **0.341 logits**. The estimator retained **99.5%** of items.

Holdout-person RMSE was **0.380 logits**, standardized-error SD was **1.108**, and nominal 95% interval coverage was **93.4%**. Latent-distribution Wasserstein distance was **0.118 logits** and edge mass was **0.0000**.

### bimodal_small_sample / normal — REVIEW

Finite-sample stress test with only 750 calibration persons and a wide bimodal population.

The recovered item-scale slope was **1.011** and item RMSE was **0.515 logits**. Tail-item RMSE was **0.792 logits**. The estimator retained **96.0%** of items.

Holdout-person RMSE was **0.766 logits**, standardized-error SD was **1.045**, and nominal 95% interval coverage was **93.9%**. Latent-distribution Wasserstein distance was **0.932 logits** and edge mass was **0.0008**.

Criteria requiring review: item_rmse.

### bimodal_small_sample / empirical — REVIEW

Finite-sample stress test with only 750 calibration persons and a wide bimodal population.

The recovered item-scale slope was **1.027** and item RMSE was **0.489 logits**. Tail-item RMSE was **0.736 logits**. The estimator retained **96.0%** of items.

Holdout-person RMSE was **0.728 logits**, standardized-error SD was **1.112**, and nominal 95% interval coverage was **92.5%**. Latent-distribution Wasserstein distance was **0.264 logits** and edge mass was **0.0007**.

Criteria requiring review: item_rmse.

### bimodal_short_test / normal — REVIEW

Sparse-person-information condition: only 20 random items per person from a 201-item wide bank.

The recovered item-scale slope was **1.009** and item RMSE was **0.550 logits**. Tail-item RMSE was **0.875 logits**. The estimator retained **98.9%** of items.

Holdout-person RMSE was **1.106 logits**, standardized-error SD was **1.076**, and nominal 95% interval coverage was **93.0%**. Latent-distribution Wasserstein distance was **0.945 logits** and edge mass was **0.0006**.

Criteria requiring review: item_rmse.

### bimodal_short_test / empirical — PASS

Sparse-person-information condition: only 20 random items per person from a 201-item wide bank.

The recovered item-scale slope was **1.015** and item RMSE was **0.381 logits**. Tail-item RMSE was **0.637 logits**. The estimator retained **98.9%** of items.

Holdout-person RMSE was **0.979 logits**, standardized-error SD was **1.022**, and nominal 95% interval coverage was **94.4%**. Latent-distribution Wasserstein distance was **0.176 logits** and edge mass was **0.0005**.

### bimodal_long_test / normal — PASS

High-information comparison: 80 random items per person from the same wide bimodal bank.

The recovered item-scale slope was **0.998** and item RMSE was **0.226 logits**. Tail-item RMSE was **0.336 logits**. The estimator retained **100.0%** of items.

Holdout-person RMSE was **0.531 logits**, standardized-error SD was **1.049**, and nominal 95% interval coverage was **93.8%**. Latent-distribution Wasserstein distance was **0.937 logits** and edge mass was **0.0007**.

### bimodal_long_test / empirical — PASS

High-information comparison: 80 random items per person from the same wide bimodal bank.

The recovered item-scale slope was **1.005** and item RMSE was **0.185 logits**. Tail-item RMSE was **0.309 logits**. The estimator retained **100.0%** of items.

Holdout-person RMSE was **0.500 logits**, standardized-error SD was **1.029**, and nominal 95% interval coverage was **94.3%**. Latent-distribution Wasserstein distance was **0.105 logits** and edge mass was **0.0000**.

### bimodal_centered_bank / normal — PASS

A more BRAVE-like bank with many central items and relatively sparse tails, still spanning -10 to +10 logits.

The recovered item-scale slope was **1.026** and item RMSE was **0.352 logits**. Tail-item RMSE was **0.723 logits**. The estimator retained **99.9%** of items.

Holdout-person RMSE was **0.749 logits**, standardized-error SD was **1.016**, and nominal 95% interval coverage was **94.6%**. Latent-distribution Wasserstein distance was **0.926 logits** and edge mass was **0.0009**.

### bimodal_centered_bank / empirical — PASS

A more BRAVE-like bank with many central items and relatively sparse tails, still spanning -10 to +10 logits.

The recovered item-scale slope was **1.007** and item RMSE was **0.252 logits**. Tail-item RMSE was **0.475 logits**. The estimator retained **99.9%** of items.

Holdout-person RMSE was **0.690 logits**, standardized-error SD was **1.029**, and nominal 95% interval coverage was **94.4%**. Latent-distribution Wasserstein distance was **0.130 logits** and edge mass was **0.0001**.

## What this report does not establish

This suite intentionally omits model-misspecification scenarios. It does not show that a Rasch model is adequate for real BRAVE data, and it does not evaluate unequal discrimination, DIF, multidimensionality, local dependence, guessing, or disengagement. Those should be investigated separately only after exact-model recovery is satisfactory.

The adaptive router here is an idealized stress test, not a reproduction of BRAVE. Agreement between random and adaptive conditions supports the claim that response-adaptive missingness itself does not distort calibration under these assumptions; disagreement should trigger investigation of exposure and scale connectivity.