# BRAVE Rasch MML validation simulations

This harness validates `rasch_mml_improved.py` under data generated exactly from the Rasch 1PL model.

## Scope

The suite focuses on the claims that are necessary before applying MML calibration to BRAVE:

- item and person ranges extend from approximately **-10 to +10 logits**;
- person populations can be normal, skewed, bimodal, or heavy-tailed;
- each person receives only a subset of the bank;
- the primary design is uniform random incomplete administration;
- a secondary condition uses simple fully adaptive randomesque routing;
- there is **no warm-up ladder**;
- both `latent="normal"` and `latent="empirical"` are fitted;
- items are evaluated on the original Rasch scale, without allowing a post-hoc multiplicative linking transformation.

The suite intentionally does **not** simulate 2PL discrimination variation, DIF, multidimensionality, local dependence, guessing, or disengagement. Those are model-fit questions and should be added only after exact-model recovery is satisfactory.

## Why retain one adaptive condition?

Random incomplete administration is the cleanest baseline because it separates sparse data from response-dependent selection. However, BRAVE is fully adaptive, so omitting adaptivity entirely would leave an important claim untested.

The included adaptive router is deliberately idealized:

1. it begins adaptively at the first item;
2. it uses a broad normal routing prior;
3. after every response it calculates a routing EAP;
4. it randomly chooses one of the nearest unused items;
5. it uses the true generating item difficulties for routing.

Thus, it tests whether adaptive missingness itself changes calibration when routing parameters are correct. It is not a simulation of the production BRAVE router.

## Files

- `rasch_mml.py` — MML estimator under test.
- `brave_mml_simulations.py` — simulation, calibration, metrics, report, and plots.
- `scenarios.json` — all simulation scenarios and profiles.

## Run a fast smoke test

From project root folder,

```bash
python -m irt.brave_mml_validation.brave_mml_simulations \
  --profile smoke \
  --output output_smoke \
  --save-level-data
```

The smoke profile contains three small scenarios and one replication each. Its purpose is code verification, not stable Monte Carlo conclusions.

## Run the main study

```bash
python -m irt.brave_mml_validation.brave_mml_simulations \
  --profile core \
  --output output_core
```

The core profile runs ten scenarios with 20 replications each. It compares normal and empirical MML, giving 400 calibrations.

## Run a publication-scale study

```bash
python -m irt.brave_mml_validation.brave_mml_simulations \
  --profile publication \
  --output output_publication
```

The publication profile uses the same scenarios with 200 replications each. Increase this only after the core suite behaves correctly.

## Outputs

Every run creates:

- `report.md` — plain-language explanation of the design, metrics, criteria, and every result row;
- `replication_results.csv` — one row per scenario, replication, and estimator;
- `summary.csv` — Monte Carlo means, SDs, and provisional pass flags;
- `internal_checks.json` — deterministic numerical checks;
- `item_slope.png` — recovered scale slope;
- `item_rmse.png` — item parameter error;
- `person_coverage.png` — uncertainty coverage;
- `latent_wasserstein.png` — person-distribution recovery.

With `--save-level-data`, it also writes compressed item- and person-level files.

## Primary metrics

### Item slope

The true and estimated item difficulties are placed under the same mean-item-zero identification constraint. The slope is

```text
estimated b = slope * true b + residual
```

No multiplicative linking is applied. A slope above 1 means scale stretching; a slope below 1 means compression.

### Item RMSE

Typical absolute calibration error in logits. Tail RMSE is reported separately for items with `|b| >= 8`.

### Person recovery

A separate holdout sample is administered and scored with the fitted item parameters and fitted latent distribution. The report includes RMSE, standardized-error SD, and nominal 95% posterior interval coverage.

### Latent recovery

The fitted latent distribution is compared with the generating calibration sample using mean, SD, quantiles, and one-dimensional Wasserstein distance.

### Exposure and retention

The report records item exposure and the fraction of items surviving the minimum-response and response-variation filters. A correct estimator cannot recover items that receive essentially no useful data.

## Provisional success criteria

For well-exposed, correctly specified scenarios, the report currently requires:

- at least 99% convergence;
- item slope within 0.05 of 1;
- item RMSE no larger than 0.40 logits;
- at least 95% item retention;
- latent edge mass no larger than 0.01;
- person 95% interval coverage between 90% and 98%.

These are configurable study criteria, not universal psychometric thresholds.
