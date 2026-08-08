"""
Simulation validation suite for BRAVE Rasch MML calibration.

The suite deliberately starts with data generated exactly from the Rasch 1PL
model.  Its purpose is to test whether the calibration code recovers the known
item scale and person scores under a wide [-10, 10] logit range, non-normal
person distributions, sparse response matrices, and simple adaptive item
selection.

The adaptive condition is intentionally idealized: routing uses the true item
difficulties, starts adaptively from item 1, and randomly chooses among the
nearest unused items.  It is not intended to reproduce the production BRAVE
router.  Random incomplete administration is the primary baseline.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.special import expit, logsumexp
from scipy.stats import skewnorm, wasserstein_distance

from irt.rasch_mml import (
    _update_items,
    calibrate_mml,
    score_person_mml,
)


# ---------------------------------------------------------------------------
# Data generation
# ---------------------------------------------------------------------------

def _rejection_sample(draw, n: int, low: float, high: float) -> np.ndarray:
    """Draw from an unbounded distribution and reject values outside bounds."""
    accepted: List[np.ndarray] = []
    total = 0
    while total < n:
        candidate = np.asarray(draw(max(1000, 2 * (n - total))), dtype=float)
        candidate = candidate[(candidate >= low) & (candidate <= high)]
        if candidate.size:
            accepted.append(candidate)
            total += candidate.size
    return np.concatenate(accepted)[:n]


def sample_persons(
    rng: np.random.Generator,
    n: int,
    distribution: str,
    theta_range: Tuple[float, float] = (-10.0, 10.0),
) -> np.ndarray:
    """Generate person abilities from several wide, bounded populations."""
    low, high = theta_range

    if distribution == "normal":
        return _rejection_sample(lambda m: rng.normal(0.0, 4.0, m), n, low, high)

    if distribution == "skewed":
        # One continuous, strongly right-skewed population.
        return _rejection_sample(
            lambda m: skewnorm.rvs(7.0, loc=-4.0, scale=5.0, size=m, random_state=rng),
            n,
            low,
            high,
        )

    if distribution == "bimodal":
        def draw(m: int) -> np.ndarray:
            group = rng.random(m) < 0.62
            out = np.empty(m)
            out[group] = rng.normal(-3.8, 2.0, group.sum())
            out[~group] = rng.normal(4.2, 1.8, (~group).sum())
            return out
        return _rejection_sample(draw, n, low, high)

    if distribution == "heavy_tail":
        return _rejection_sample(lambda m: 3.2 * rng.standard_t(3, m), n, low, high)

    raise ValueError(f"Unknown person distribution: {distribution}")


def make_item_bank(
    rng: np.random.Generator,
    n_items: int,
    item_range: Tuple[float, float] = (-10.0, 10.0),
    shape: str = "even",
) -> np.ndarray:
    """Create a known Rasch item bank and enforce mean difficulty zero."""
    low, high = item_range
    u = np.linspace(-1.0, 1.0, n_items)

    if shape == "even":
        b = np.linspace(low, high, n_items)
    elif shape == "centered":
        # More items near the middle while still reaching both endpoints.
        b = np.sign(u) * np.abs(u) ** 1.55
        b = low + (b - b.min()) * (high - low) / (b.max() - b.min())
    elif shape == "irregular":
        b = np.linspace(low, high, n_items)
        b += rng.normal(0.0, 0.18, n_items)
        b[0], b[-1] = low, high
        b.sort()
    else:
        raise ValueError(f"Unknown item bank shape: {shape}")

    return b - b.mean()


# ---------------------------------------------------------------------------
# Administration
# ---------------------------------------------------------------------------

def administer_random(
    rng: np.random.Generator,
    theta: np.ndarray,
    b_true: np.ndarray,
    test_length: int,
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """Uniform random incomplete administration without replacement."""
    J = len(b_true)
    if test_length > J:
        raise ValueError("test_length cannot exceed number of items")

    item_rows: List[np.ndarray] = []
    response_rows: List[np.ndarray] = []
    for ability in theta:
        item_ids = rng.choice(J, size=test_length, replace=False)
        p = expit(ability - b_true[item_ids])
        x = (rng.random(test_length) < p).astype(np.int8)
        item_rows.append(item_ids.astype(np.int32))
        response_rows.append(x)
    return item_rows, response_rows


def administer_adaptive_randomesque(
    rng: np.random.Generator,
    theta: np.ndarray,
    b_true: np.ndarray,
    test_length: int,
    candidate_pool: int = 12,
    routing_grid: Tuple[float, float, int] = (-12.0, 12.0, 97),
    prior_sd: float = 5.0,
    routing_scale: float = 1.0,
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """Simple fully adaptive routing, with no warm-up phase.

    Routing begins with a broad N(0, prior_sd) distribution. At every item,
    it computes the current routing EAP and randomly chooses one of the
    `candidate_pool` nearest unused items. The routing difficulties are the
    true difficulties multiplied by routing_scale.
    """
    J = len(b_true)
    if test_length > J:
        raise ValueError("test_length cannot exceed number of items")

    grid = np.linspace(*routing_grid)
    initial_log_prior = -0.5 * (grid / prior_sd) ** 2
    initial_log_prior -= logsumexp(initial_log_prior)
    b_route = routing_scale * b_true

    item_rows: List[np.ndarray] = []
    response_rows: List[np.ndarray] = []

    for ability in theta:
        log_post = initial_log_prior.copy()
        used = np.zeros(J, dtype=bool)
        selected = np.empty(test_length, dtype=np.int32)
        responses = np.empty(test_length, dtype=np.int8)

        for k in range(test_length):
            post = np.exp(log_post - logsumexp(log_post))
            routing_theta = float(np.sum(post * grid))

            available = np.flatnonzero(~used)
            pool_size = min(candidate_pool, available.size)
            nearest = available[
                np.argpartition(np.abs(b_route[available] - routing_theta), pool_size - 1)[:pool_size]
            ]
            item_id = int(rng.choice(nearest))
            used[item_id] = True

            response = int(rng.random() < expit(ability - b_true[item_id]))
            selected[k] = item_id
            responses[k] = response

            z = grid - b_route[item_id]
            if response:
                log_post += -np.logaddexp(0.0, -z)
            else:
                log_post += -np.logaddexp(0.0, z)
            log_post -= logsumexp(log_post)

        item_rows.append(selected)
        response_rows.append(responses)

    return item_rows, response_rows


def administer(
    rng: np.random.Generator,
    theta: np.ndarray,
    b_true: np.ndarray,
    scenario: dict,
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    design = scenario["administration"]
    if design == "random":
        return administer_random(rng, theta, b_true, scenario["test_length"])
    if design == "adaptive_randomesque":
        return administer_adaptive_randomesque(
            rng,
            theta,
            b_true,
            scenario["test_length"],
            candidate_pool=scenario.get("candidate_pool", 12),
            prior_sd=scenario.get("routing_prior_sd", 5.0),
            routing_scale=scenario.get("routing_scale", 1.0),
        )
    raise ValueError(f"Unknown administration design: {design}")


def to_response_dicts(
    item_rows: List[np.ndarray],
    response_rows: List[np.ndarray],
    n_items: int,
) -> Tuple[Dict[int, Dict[int, int]], Dict[int, Dict[int, int]]]:
    persons: Dict[int, Dict[int, int]] = {}
    items: Dict[int, Dict[int, int]] = {j: {} for j in range(n_items)}

    for person_id, (item_ids, responses) in enumerate(zip(item_rows, response_rows)):
        row = {int(j): int(x) for j, x in zip(item_ids, responses)}
        persons[person_id] = row
        for item_id, response in row.items():
            items[item_id][person_id] = response
    return persons, items


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def weighted_quantile(values: np.ndarray, weights: np.ndarray, q: Iterable[float]) -> np.ndarray:
    order = np.argsort(values)
    values = values[order]
    weights = weights[order]
    cdf = np.cumsum(weights)
    cdf /= cdf[-1]
    return np.interp(np.asarray(list(q), dtype=float), cdf, values)


def item_metrics(
    b_true: np.ndarray,
    items_params: dict,
) -> Tuple[dict, pd.DataFrame, float]:
    retained = np.array(sorted(items_params), dtype=int)
    true_shift = float(np.mean(b_true[retained]))
    true_aligned = b_true[retained] - true_shift
    estimated = np.array([items_params[int(j)]["b"] for j in retained])
    se = np.array([items_params[int(j)]["bSD"] for j in retained])
    exposure = np.array([items_params[int(j)]["totalResponses"] for j in retained])
    error = estimated - true_aligned

    denominator = float(np.dot(true_aligned, true_aligned))
    slope = float(np.dot(true_aligned, estimated) / denominator) if denominator else np.nan
    correlation = float(np.corrcoef(true_aligned, estimated)[0, 1]) if len(retained) > 2 else np.nan

    tail = np.abs(true_aligned) >= 8.0
    center = np.abs(true_aligned) <= 4.0
    metrics = {
        "n_items_retained": int(len(retained)),
        "item_retained_fraction": float(len(retained) / len(b_true)),
        "item_slope": slope,
        "item_span_ratio": float(np.ptp(estimated) / np.ptp(true_aligned)),
        "item_rmse": float(np.sqrt(np.mean(error ** 2))),
        "item_mae": float(np.mean(np.abs(error))),
        "item_correlation": correlation,
        "item_tail_rmse": float(np.sqrt(np.mean(error[tail] ** 2))) if tail.any() else np.nan,
        "item_center_rmse": float(np.sqrt(np.mean(error[center] ** 2))) if center.any() else np.nan,
        "item_se_coverage_95": float(np.mean(np.abs(error) <= 1.96 * se)),
        "exposure_min": int(exposure.min()),
        "exposure_p10": float(np.quantile(exposure, 0.10)),
        "exposure_median": float(np.median(exposure)),
        "exposure_max": int(exposure.max()),
    }

    detail = pd.DataFrame({
        "item_id": retained,
        "b_true": true_aligned,
        "b_est": estimated,
        "error": error,
        "b_se_approx": se,
        "exposure": exposure,
    })
    return metrics, detail, true_shift


def latent_metrics(theta_true: np.ndarray, info: dict, true_shift: float) -> dict:
    true = theta_true - true_shift
    grid = np.asarray(info["latent_grid"], dtype=float)
    weights = np.asarray(info["latent_weights"], dtype=float)
    probs = [0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99]
    q_true = np.quantile(true, probs)
    q_est = weighted_quantile(grid, weights, probs)

    return {
        "latent_mean_error": float(info["latent_mean"] - np.mean(true)),
        "latent_sd_error": float(info["latent_sd"] - np.std(true)),
        "latent_wasserstein": float(wasserstein_distance(true, grid, v_weights=weights)),
        "latent_quantile_mae": float(np.mean(np.abs(q_est - q_true))),
        "latent_edge_mass": float(info["latent_edge_mass"]),
        "latent_effective_nodes": float(info["latent_effective_nodes"]),
    }


def score_holdout(
    theta_true: np.ndarray,
    item_rows: List[np.ndarray],
    response_rows: List[np.ndarray],
    items_params: dict,
    info: dict,
    true_shift: float,
) -> Tuple[dict, pd.DataFrame]:
    theta_aligned = theta_true - true_shift
    estimates, sds, kept_true = [], [], []

    for ability, item_ids, responses in zip(theta_aligned, item_rows, response_rows):
        row = {
            int(j): int(x)
            for j, x in zip(item_ids, responses)
            if int(j) in items_params
        }
        if not row:
            continue
        estimate, sd = score_person_mml(
            row,
            items_params,
            info["latent_grid"],
            info["latent_weights"],
            method="EAP",
        )
        estimates.append(estimate)
        sds.append(sd)
        kept_true.append(float(ability))

    true = np.asarray(kept_true)
    est = np.asarray(estimates)
    sd = np.asarray(sds)
    error = est - true
    z = error / np.maximum(sd, 1e-12)
    tail = np.abs(true) >= 8.0

    metrics = {
        "n_holdout_scored": int(len(true)),
        "person_bias": float(np.mean(error)),
        "person_rmse": float(np.sqrt(np.mean(error ** 2))),
        "person_correlation": float(np.corrcoef(true, est)[0, 1]),
        "person_z_mean": float(np.mean(z)),
        "person_z_sd": float(np.std(z)),
        "person_coverage_95": float(np.mean(np.abs(error) <= 1.96 * sd)),
        "person_tail_rmse": float(np.sqrt(np.mean(error[tail] ** 2))) if tail.any() else np.nan,
        "person_tail_bias": float(np.mean(error[tail])) if tail.any() else np.nan,
    }

    detail = pd.DataFrame({
        "theta_true": true,
        "theta_est": est,
        "theta_sd": sd,
        "error": error,
        "z": z,
    })
    return metrics, detail


# ---------------------------------------------------------------------------
# Internal numerical checks
# ---------------------------------------------------------------------------

def run_internal_checks() -> dict:
    """Small deterministic checks that fail immediately on basic code errors."""
    grid = np.linspace(-12.0, 12.0, 121)
    b_true = np.array([-8.0, -3.0, 0.0, 4.0, 7.0])
    b_true -= b_true.mean()
    weights = np.exp(-0.5 * (grid / 4.0) ** 2)
    weights /= weights.sum()

    n_jq = np.tile(5000.0 * weights, (len(b_true), 1))
    r_jq = n_jq * expit(grid[None, :] - b_true[:, None])
    recovered = _update_items(np.zeros_like(b_true), grid, n_jq, r_jq)
    recovered -= recovered.mean()
    max_error = float(np.max(np.abs(recovered - b_true)))

    checks = {
        "item_m_step_max_abs_error": max_error,
        "item_m_step_pass": bool(max_error < 1e-7),
        "weights_sum": float(weights.sum()),
        "weights_normalization_pass": bool(abs(weights.sum() - 1.0) < 1e-12),
    }
    checks["all_pass"] = bool(all(v for k, v in checks.items() if k.endswith("_pass")))
    return checks


# ---------------------------------------------------------------------------
# Monte Carlo runner
# ---------------------------------------------------------------------------

def run_replication(
    scenario: dict,
    replication: int,
    seed: int,
    save_level_data: bool,
) -> Tuple[List[dict], List[pd.DataFrame], List[pd.DataFrame]]:
    rng = np.random.default_rng(seed)
    b_true = make_item_bank(
        rng,
        scenario["n_items"],
        tuple(scenario.get("item_range", [-10.0, 10.0])),
        scenario.get("item_shape", "even"),
    )

    theta_cal = sample_persons(
        rng,
        scenario["n_persons"],
        scenario["person_distribution"],
        tuple(scenario.get("person_range", [-10.0, 10.0])),
    )
    cal_items, cal_responses = administer(rng, theta_cal, b_true, scenario)
    persons, items = to_response_dicts(cal_items, cal_responses, len(b_true))

    n_holdout = int(scenario.get("n_holdout", min(1500, scenario["n_persons"])))
    theta_holdout = sample_persons(
        rng,
        n_holdout,
        scenario["person_distribution"],
        tuple(scenario.get("person_range", [-10.0, 10.0])),
    )
    hold_items, hold_responses = administer(rng, theta_holdout, b_true, scenario)

    rows: List[dict] = []
    item_details: List[pd.DataFrame] = []
    person_details: List[pd.DataFrame] = []

    for latent in scenario.get("estimators", ["normal", "empirical"]):
        started = time.time()
        items_params, bad_items, info, status = calibrate_mml(
            persons,
            items,
            latent=latent,
            min_item_responses=int(scenario.get("min_item_responses", 20)),
            n_nodes=int(scenario.get("n_nodes", 97)),
            theta_range=tuple(scenario.get("calibration_theta_range", [-12.0, 12.0])),
            max_iter=int(scenario.get("max_iter", 1000)),
            loglik_tol=float(scenario.get("loglik_tol", 1e-6)),
            item_tol=float(scenario.get("item_tol", 1e-4)),
            latent_tol=float(scenario.get("latent_tol", 1e-3)),
            verbose=False,
        )

        im, item_detail, true_shift = item_metrics(b_true, items_params)
        lm = latent_metrics(theta_cal, info, true_shift)
        pm, person_detail = score_holdout(
            theta_holdout,
            hold_items,
            hold_responses,
            items_params,
            info,
            true_shift,
        )

        row = {
            "scenario": scenario["name"],
            "description": scenario["description"],
            "replication": replication,
            "seed": seed,
            "latent": latent,
            "person_distribution": scenario["person_distribution"],
            "administration": scenario["administration"],
            "n_persons": scenario["n_persons"],
            "n_items": scenario["n_items"],
            "test_length": scenario["test_length"],
            "converged": bool(info["converged"]),
            "iterations": int(info["iterations"]),
            "seconds": float(time.time() - started),
            "status": status,
            "n_bad_items": int(len(bad_items)),
            "final_max_step_in_se": float(info["final_max_step_in_SE"]),
            "final_latent_residual": float(info["final_latent_residual"]),
            **im,
            **lm,
            **pm,
        }
        rows.append(row)

        if save_level_data:
            item_detail.insert(0, "latent", latent)
            item_detail.insert(0, "replication", replication)
            item_detail.insert(0, "scenario", scenario["name"])
            item_details.append(item_detail)

            person_detail.insert(0, "latent", latent)
            person_detail.insert(0, "replication", replication)
            person_detail.insert(0, "scenario", scenario["name"])
            person_details.append(person_detail)

    return rows, item_details, person_details


def aggregate_results(results: pd.DataFrame) -> pd.DataFrame:
    numeric_metrics = [
        "converged",
        "item_retained_fraction",
        "item_slope",
        "item_span_ratio",
        "item_rmse",
        "item_tail_rmse",
        "item_correlation",
        "item_se_coverage_95",
        "latent_mean_error",
        "latent_sd_error",
        "latent_wasserstein",
        "latent_quantile_mae",
        "latent_edge_mass",
        "person_bias",
        "person_rmse",
        "person_tail_rmse",
        "person_correlation",
        "person_z_mean",
        "person_z_sd",
        "person_coverage_95",
        "iterations",
        "seconds",
    ]
    grouped = results.groupby(["scenario", "latent"], sort=False)[numeric_metrics]
    mean = grouped.mean().add_suffix("_mean")
    sd = grouped.std(ddof=1).fillna(0.0).add_suffix("_sd")
    summary = pd.concat([mean, sd], axis=1).reset_index()

    # Provisional criteria for correctly specified, adequately exposed runs.
    summary["pass_convergence"] = summary["converged_mean"] >= 0.99
    summary["pass_scale"] = (summary["item_slope_mean"] - 1.0).abs() <= 0.05
    summary["pass_item_rmse"] = summary["item_rmse_mean"] <= 0.40
    summary["pass_retention"] = summary["item_retained_fraction_mean"] >= 0.95
    summary["pass_edge_mass"] = summary["latent_edge_mass_mean"] <= 0.01
    summary["pass_person_coverage"] = summary["person_coverage_95_mean"].between(0.90, 0.98)
    pass_cols = [c for c in summary.columns if c.startswith("pass_")]
    summary["overall_pass"] = summary[pass_cols].all(axis=1)
    return summary


def _format_table(df: pd.DataFrame) -> str:
    try:
        return df.to_markdown(index=False)
    except Exception:
        return df.to_string(index=False)


def write_report(
    output_dir: Path,
    profile: str,
    scenarios: List[dict],
    checks: dict,
    results: pd.DataFrame,
    summary: pd.DataFrame,
) -> None:
    compact = summary[[
        "scenario",
        "latent",
        "converged_mean",
        "item_slope_mean",
        "item_rmse_mean",
        "item_tail_rmse_mean",
        "item_retained_fraction_mean",
        "latent_wasserstein_mean",
        "person_rmse_mean",
        "person_z_sd_mean",
        "person_coverage_95_mean",
        "latent_edge_mass_mean",
        "overall_pass",
    ]].copy()

    for col in compact.columns:
        if col not in {"scenario", "latent", "overall_pass"}:
            compact[col] = compact[col].map(lambda x: f"{x:.4f}")

    lines = [
        "# BRAVE Rasch MML simulation report",
        "",
        f"**Profile:** `{profile}`  ",
        f"**Monte Carlo fits:** {len(results)}  ",
        f"**Generated:** {pd.Timestamp.now().isoformat(timespec='seconds')}",
        "",
        "## What this validates",
        "",
        "All responses are generated from the Rasch 1PL model used by the estimator. "
        "The tests therefore evaluate numerical correctness and parameter recovery, not robustness to 2PL, DIF, local dependence, or careless responding.",
        "",
        "The primary administration is random incomplete testing. The adaptive condition is deliberately simple and idealized: it starts adaptively at item 1, uses the true item difficulties for routing, and randomly selects among nearby unused items. There is no warm-up ladder.",
        "",
        "Both item and person generating ranges are bounded at approximately -10 to +10 logits. Calibration uses a slightly wider numerical grid so that boundary mass can be diagnosed.",
        "",
        "## Internal numerical checks",
        "",
        f"- Item M-step maximum absolute recovery error: `{checks['item_m_step_max_abs_error']:.3g}`",
        f"- Item M-step check: `{'PASS' if checks['item_m_step_pass'] else 'FAIL'}`",
        f"- Probability-weight normalization check: `{'PASS' if checks['weights_normalization_pass'] else 'FAIL'}`",
        "",
        "## How to read the main metrics",
        "",
        "- **Item slope:** regression slope of estimated on true centered item difficulty. 1.0 is correct; above 1 means a stretched scale and below 1 a compressed scale.",
        "- **Item RMSE:** typical item-difficulty error in logits.",
        "- **Tail RMSE:** item RMSE for true difficulties with |b| >= 8.",
        "- **Retained fraction:** fraction of the bank with enough variable responses to calibrate.",
        "- **Latent Wasserstein:** distance in logits between the fitted person distribution and the generating sample distribution; lower is better.",
        "- **Person z SD:** SD of (EAP - true theta) / posterior SD. Around 1 indicates appropriately scaled uncertainty.",
        "- **95% coverage:** fraction of true abilities inside EAP +/- 1.96 posterior SD.",
        "- **Edge mass:** fitted latent probability near the numerical grid boundaries; values above about 0.01 deserve investigation.",
        "",
        "## Provisional pass criteria",
        "",
        "A row passes when convergence is at least 99%, item slope is within 0.05 of 1, item RMSE is at most 0.40 logits, at least 95% of items are retained, latent edge mass is at most 0.01, and person interval coverage is between 90% and 98%. These are practical validation thresholds, not universal psychometric laws.",
        "",
        "## Aggregate results",
        "",
        _format_table(compact),
        "",
        "## Scenario-by-scenario interpretation",
        "",
    ]

    description_by_name = {s["name"]: s["description"] for s in scenarios}
    for _, row in summary.iterrows():
        state = "PASS" if bool(row["overall_pass"]) else "REVIEW"
        lines.extend([
            f"### {row['scenario']} / {row['latent']} — {state}",
            "",
            description_by_name[row["scenario"]],
            "",
            f"The recovered item-scale slope was **{row['item_slope_mean']:.3f}** and item RMSE was **{row['item_rmse_mean']:.3f} logits**. "
            f"Tail-item RMSE was **{row['item_tail_rmse_mean']:.3f} logits**. "
            f"The estimator retained **{100 * row['item_retained_fraction_mean']:.1f}%** of items.",
            "",
            f"Holdout-person RMSE was **{row['person_rmse_mean']:.3f} logits**, standardized-error SD was **{row['person_z_sd_mean']:.3f}**, and nominal 95% interval coverage was **{100 * row['person_coverage_95_mean']:.1f}%**. "
            f"Latent-distribution Wasserstein distance was **{row['latent_wasserstein_mean']:.3f} logits** and edge mass was **{row['latent_edge_mass_mean']:.4f}**.",
            "",
        ])

        failed = [c.replace("pass_", "") for c in summary.columns if c.startswith("pass_") and not bool(row[c])]
        if failed:
            lines.extend([
                "Criteria requiring review: " + ", ".join(failed) + ".",
                "",
            ])

    lines.extend([
        "## What this report does not establish",
        "",
        "This suite intentionally omits model-misspecification scenarios. It does not show that a Rasch model is adequate for real BRAVE data, and it does not evaluate unequal discrimination, DIF, multidimensionality, local dependence, guessing, or disengagement. Those should be investigated separately only after exact-model recovery is satisfactory.",
        "",
        "The adaptive router here is an idealized stress test, not a reproduction of BRAVE. Agreement between random and adaptive conditions supports the claim that response-adaptive missingness itself does not distort calibration under these assumptions; disagreement should trigger investigation of exposure and scale connectivity.",
    ])

    (output_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")


def make_plots(output_dir: Path, summary: pd.DataFrame) -> None:
    labels = [f"{s}\n{l}" for s, l in zip(summary["scenario"], summary["latent"])]
    x = np.arange(len(labels))

    plots = [
        ("item_slope_mean", "item_slope_sd", "Recovered item-scale slope", 1.0, "item_slope.png"),
        ("item_rmse_mean", "item_rmse_sd", "Item RMSE (logits)", None, "item_rmse.png"),
        ("person_coverage_95_mean", "person_coverage_95_sd", "Person 95% interval coverage", 0.95, "person_coverage.png"),
        ("latent_wasserstein_mean", "latent_wasserstein_sd", "Latent Wasserstein distance (logits)", None, "latent_wasserstein.png"),
    ]

    for mean_col, sd_col, ylabel, reference, filename in plots:
        fig, ax = plt.subplots(figsize=(max(8, len(labels) * 0.72), 5))
        ax.errorbar(x, summary[mean_col], yerr=summary[sd_col], fmt="o", capsize=3)
        if reference is not None:
            ax.axhline(reference, linewidth=1)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha="right")
        ax.set_ylabel(ylabel)
        ax.set_title(ylabel + " by scenario and latent model")
        fig.tight_layout()
        fig.savefig(output_dir / filename, dpi=160)
        plt.close(fig)


def load_configuration(path: Path, profile: str) -> Tuple[List[dict], int]:
    config = json.loads(path.read_text(encoding="utf-8"))
    if profile not in config["profiles"]:
        raise ValueError(f"Unknown profile {profile!r}; choose from {list(config['profiles'])}")
    profile_config = config["profiles"][profile]
    names = set(profile_config["scenarios"])
    scenarios = [s for s in config["scenarios"] if s["name"] in names]
    missing = names - {s["name"] for s in scenarios}
    if missing:
        raise ValueError(f"Profile refers to missing scenarios: {sorted(missing)}")
    return scenarios, int(profile_config["replications"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path(__file__).with_name("scenarios.json"))
    parser.add_argument("--profile", choices=["smoke", "core", "publication"], default="smoke")
    parser.add_argument("--output", type=Path, default=Path("mml_simulation_output"))
    parser.add_argument("--seed", type=int, default=20260803)
    parser.add_argument("--save-level-data", action="store_true")
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    scenarios, replications = load_configuration(args.config, args.profile)

    checks = run_internal_checks()
    (args.output / "internal_checks.json").write_text(json.dumps(checks, indent=2), encoding="utf-8")
    if not checks["all_pass"]:
        raise RuntimeError("Internal numerical checks failed; simulation aborted")

    all_rows: List[dict] = []
    all_item_details: List[pd.DataFrame] = []
    all_person_details: List[pd.DataFrame] = []

    print(f"Running profile={args.profile!r}: {len(scenarios)} scenarios x {replications} replications")
    for scenario_index, scenario in enumerate(scenarios):
        print(f"\n[{scenario_index + 1}/{len(scenarios)}] {scenario['name']}: {scenario['description']}")
        for replication in range(replications):
            seed = args.seed + scenario_index * 100_000 + replication
            rows, item_details, person_details = run_replication(
                scenario,
                replication=replication,
                seed=seed,
                save_level_data=args.save_level_data,
            )
            all_rows.extend(rows)
            all_item_details.extend(item_details)
            all_person_details.extend(person_details)
            for row in rows:
                print(
                    f"  rep {replication + 1:3d} {row['latent']:9s} | "
                    f"conv={row['converged']} slope={row['item_slope']:.3f} "
                    f"itemRMSE={row['item_rmse']:.3f} personRMSE={row['person_rmse']:.3f} "
                    f"coverage={row['person_coverage_95']:.3f}"
                )

    results = pd.DataFrame(all_rows)
    summary = aggregate_results(results)
    results.to_csv(args.output / "replication_results.csv", index=False)
    summary.to_csv(args.output / "summary.csv", index=False)

    if args.save_level_data and all_item_details:
        pd.concat(all_item_details, ignore_index=True).to_csv(
            args.output / "item_level_results.csv.gz", index=False, compression="gzip"
        )
        pd.concat(all_person_details, ignore_index=True).to_csv(
            args.output / "person_level_results.csv.gz", index=False, compression="gzip"
        )

    write_report(args.output, args.profile, scenarios, checks, results, summary)
    make_plots(args.output, summary)

    print(f"\nFinished. Explanatory report: {args.output / 'report.md'}")
    print(f"Replication results:       {args.output / 'replication_results.csv'}")
    print(f"Aggregate summary:         {args.output / 'summary.csv'}")


if __name__ == "__main__":
    main()
