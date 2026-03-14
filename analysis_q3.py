"""
======================================================================
Q3 and adjusted Q3 (aQ3) residual correlation analysis
for Rasch 1PL CAT item banks
======================================================================

PURPOSE
-------
This module evaluates *local item dependence* (LID) in a Rasch 1PL CAT
item bank using residual correlations:

    Q3   : raw residual correlation
    aQ3  : adjusted Q3 (item-mean corrected)

It is intended for:
- validation analyses (paper + supplement)
- item-bank quality control
- downstream item filtering decisions

IMPORTANT DESIGN CHOICES
------------------------
• Item difficulties (b) and person abilities (theta) are assumed
  to be pre-calibrated elsewhere.
• This module performs NO estimation and NO imputation.
• Only observed responses are used (CAT-sparse by design).
• The analysis is safe to re-run: old diagnostics are removed before
  new ones are merged.

======================================================================
"""

# ====================================================================
# =============================== IMPORTS =============================
# ====================================================================

import os
import shutil
from typing import Dict, Tuple
import numpy as np
import pandas as pd
import networkx as nx

import math
from typing import Optional, Iterable, Tuple, Union

import matplotlib.pyplot as plt

from irt.utils import readItemsParams, readPersonsParams
from irt.irt import readTable
from irt.models import prob1PL


# ====================================================================
# ========================== HELPER FUNCTIONS =========================
# ====================================================================

def _safe_pearsonr(x: np.ndarray, y: np.ndarray) -> float:
    """
    Compute Pearson correlation with safeguards.

    Returns NaN if:
    - fewer than 3 observations
    - near-zero variance in either vector
    """
    if x.size < 3:
        return np.nan
    if np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return np.nan
    return float(np.corrcoef(x, y)[0, 1])


def _filter_to_calibrated_items(persons, items_params):
    """
    Remove responses to items not present in the calibrated item bank.

    Ensures that residuals are only computed for items with known
    difficulty parameters.
    """
    keep = set(items_params.keys())
    return {
        pid: {iid: u for iid, u in resp.items() if iid in keep}
        for pid, resp in persons.items()
        if any(iid in keep for iid in resp)
    }


def _filter_persons(persons, persons_params, min_items_per_person: int):
    """
    Retain persons eligible for Q3 analysis.

    A person is kept if:
    - they have a valid theta estimate
    - they answered at least MIN_ITEMS_PER_PERSON items
    """
    out = {
        pid: resp
        for pid, resp in persons.items()
        if pid in persons_params
        and not np.isnan(persons_params[pid]["theta"])
        and len(resp) >= min_items_per_person
    }
    print(f"[INFO] persons used for Q3: {len(out)}")
    return out


def _build_item_residual_vectors(persons, persons_params, items_params):
    """
    Build residual vectors per item.

    Residual definition (Rasch 1PL):
        r_pi = u_pi − P_i(theta_p)

    Output:
        dict[itemID] = (person_index_array, residual_array)

    Person indices are used to efficiently compute intersections
    between item response sets.
    """
    pids = list(persons.keys())
    pid_to_idx = {p: i for i, p in enumerate(pids)}

    tmp_idx, tmp_r = {}, {}

    for pid in pids:
        theta = persons_params[pid]["theta"]
        for iid, u in persons[pid].items():
            p = prob1PL(theta, items_params[iid]["b"])
            r = float(u) - float(p)
            tmp_idx.setdefault(iid, []).append(pid_to_idx[pid])
            tmp_r.setdefault(iid, []).append(r)

    return {
        iid: (np.array(tmp_idx[iid]), np.array(tmp_r[iid]))
        for iid in tmp_idx
    }


def _intersect_sorted(a: np.ndarray, b: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Two-pointer intersection of sorted integer arrays.

    Returns indices into arrays a and b corresponding to
    overlapping persons.
    """
    i = j = 0
    pa, pb = [], []
    while i < len(a) and j < len(b):
        if a[i] == b[j]:
            pa.append(i); pb.append(j)
            i += 1; j += 1
        elif a[i] < b[j]:
            i += 1
        else:
            j += 1
    return np.array(pa), np.array(pb)


def _compute_q3_pairs(item_vectors, min_overlap: int):
    """
    Compute raw Q3 residual correlations for all eligible item pairs.

    Q3_ij = corr(r_i, r_j)
    """
    items = sorted(item_vectors.keys())
    rows = []

    for i in range(len(items)):
        a_id = items[i]
        a_idx, a_r = item_vectors[a_id]

        for j in range(i + 1, len(items)):
            b_id = items[j]
            b_idx, b_r = item_vectors[b_id]

            pa, pb = _intersect_sorted(a_idx, b_idx)
            if pa.size < min_overlap:
                continue

            q = _safe_pearsonr(a_r[pa], b_r[pb])
            if not np.isnan(q):
                rows.append((a_id, b_id, q, pa.size))

    return pd.DataFrame(rows, columns=["itemA", "itemB", "q3", "n_overlap"])


def _compute_aq3(q3_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute adjusted Q3 (aQ3).

    aQ3_ij = Q3_ij − mean(Q3_i) − mean(Q3_j)

    Removes item-wise inflation and centers the distribution
    on pair-specific dependence.
    """
    means = pd.concat([
        q3_df[["itemA", "q3"]].rename(columns={"itemA": "itemID"}),
        q3_df[["itemB", "q3"]].rename(columns={"itemB": "itemID"}),
    ]).groupby("itemID")["q3"].mean()

    out = q3_df.copy()
    out["aq3"] = out["q3"] - out["itemA"].map(means) - out["itemB"].map(means)
    return out


def _item_metrics(df, col, all_items, aq3_count_thresholds):
    """
    Compute per-item diagnostics for Q3 or aQ3.

    For each item:
    - number of involved pairs
    - maximum correlation
    - mean correlation
    - maximum overlap
    - counts above selected thresholds
    """
    sym = pd.concat([
        df.rename(columns={"itemA": "itemID"}),
        df.rename(columns={"itemB": "itemID"}),
    ])

    g = sym.groupby("itemID", sort=False)

    res = pd.DataFrame({
        "itemID": g.size().index,
        f"{col}_pairs_n": g.size().values,
        f"{col}_max": g[col].max().values,
        f"{col}_mean": g[col].mean().values,
        f"{col}_overlap_max": g["n_overlap"].max().values,
    })

    for thr in aq3_count_thresholds:
        res[f"{col}_n_ge_{thr:.2f}".replace(".", "_")] = (
            (sym[col] >= thr).groupby(sym["itemID"]).sum().values
        )

    return pd.DataFrame({"itemID": all_items}).merge(res, on="itemID", how="left")


def _cluster_on_threshold(df: pd.DataFrame, col: str, threshold: float, min_overlap: int) -> list[set]:
    """
    Build connected components (clusters) for a given correlation threshold.

    Items are connected if:
    - df[col] >= threshold
    - sufficient overlap
    """
    edges = df[(df[col] >= threshold) & (df["n_overlap"] >= min_overlap)]
    G = nx.Graph()
    for r in edges.itertuples(index=False):
        G.add_edge(r.itemA, r.itemB)
    return list(nx.connected_components(G))


def _clusters_table(
    df: pd.DataFrame,
    *,
    col: str,
    thresholds: Iterable[float],
    min_overlap: int,
) -> pd.DataFrame:
    """
    Create a compact clusters table.

    Output columns:
      - cluster_id  : integer id *within* a given threshold
      - threshold   : threshold that defined the cluster graph
      - items       : comma-separated list of items in the component
    """
    rows = []
    for thr in sorted(set(float(t) for t in thresholds), reverse=False):
        comps = _cluster_on_threshold(df, col=col, threshold=thr, min_overlap=min_overlap)
        # Sort components by size descending for deterministic output
        comps = sorted(comps, key=lambda s: (-len(s), sorted(s)[0] if len(s) else ""))
        for k, comp in enumerate(comps, start=1):
            rows.append({
                "cluster_id": k,
                "threshold": float(thr),
                "items": ",".join(sorted(comp)),
            })
    return pd.DataFrame(rows, columns=["cluster_id", "threshold", "items"])


def _make_bidirectional_pairs(df_pairs: pd.DataFrame) -> pd.DataFrame:
    """
    Duplicate pair rows in both directions (A->B and B->A).

    This is useful for per-item browsing by sorting on itemA.
    """
    a_to_b = df_pairs.copy()
    b_to_a = df_pairs.rename(columns={"itemA": "itemB", "itemB": "itemA"})[
        ["itemA", "itemB", "q3", "aq3", "n_overlap"]
    ].copy()
    out = pd.concat([a_to_b, b_to_a], axis=0, ignore_index=True)
    return out


def _backup_file(path: str, backup_suffix: str):
    """Create a backup of the item bank before overwriting."""
    shutil.copy2(path, path + backup_suffix)


# ====================================================================
# ============================ PUBLIC API =============================
# ====================================================================

def run_q3_analysis(
    *,
    responses_tsv: str,
    # Response log used for residual computation.
    # rows - persons, columns - items, cells - responses

    item_params_csv: str,
    # Item bank CSV with calibrated difficulties.
    # Required columns: itemID, b
    # This file will be augmented with Q3 / aQ3 diagnostics.

    persons_thetas_csv: str,
    # Person ability estimates from calibration.
    # Required columns: personID, theta

    out_dir: str,
    # Output directory for pair-level outputs and logs.

    min_items_per_person: int = 20,
    # Persons with fewer responses are excluded.
    # Rationale: residual correlations are unstable for very short tests.

    min_overlap: int = 30,
    # Minimum number of persons who must have seen BOTH items
    # for Q3 / aQ3 to be computed for that item pair.

    aq3_edge_threshold: float = 0.25,
    # Threshold for defining an aQ3 edge between two items.
    # Used to detect clusters of locally dependent items.

    aq3_count_thresholds=(0.10, 0.20, 0.30, 0.40, 0.50),
    # Per-item diagnostics:
    # counts of item pairs exceeding these aQ3 levels.

    write_item_params_back: bool = True,
    # If True:
    #   - overwrite item_params_csv (after creating a backup)
    # If False:
    #   - write augmented item bank into out_dir

    item_params_backup_suffix: str = ".bak",
    # Suffix for the backup copy of item_params_csv when overwriting.

    diagnostic_prefixes=("q3_", "aq3_", "cluster_id"),
    # Any existing columns starting with these prefixes
    # are removed before merging new diagnostics.
    # This makes the analysis idempotent (safe re-runs).
) -> Dict[str, object]:
    """
    Run Q3 / aQ3 residual correlation analysis and write outputs to disk.

    OUTPUTS
    -------
    out_dir/
      - q3_aq3_pairs.csv     : bidirectional pairs with both q3 and aq3
      - q3_aq3_clusters.csv  : clusters table across thresholds

    item_params_csv (augmented)
      - q3_* columns   : per-item Q3 diagnostics
      - aq3_* columns  : per-item aQ3 diagnostics
      - cluster_id     : aQ3-based dependency cluster (0 = none)

    Returns a dict with key outputs (dataframes + paths) for programmatic use.
    """
    os.makedirs(out_dir, exist_ok=True)

    items_params = readItemsParams(item_params_csv)
    persons, _ = readTable(responses_tsv, correctSymbol="1", incorrectSymbol="0")
    persons = _filter_to_calibrated_items(persons, items_params)

    persons_params = readPersonsParams(persons_thetas_csv)
    persons = _filter_persons(persons, persons_params, min_items_per_person=min_items_per_person)

    item_vectors = _build_item_residual_vectors(persons, persons_params, items_params)
    all_items = sorted(item_vectors.keys())

    # Pair-level computations
    q3 = _compute_q3_pairs(item_vectors, min_overlap=min_overlap)
    aq3 = _compute_aq3(q3)  # contains q3 + aq3 + n_overlap + itemA/itemB

    # ------------------------------------------------
    # Single pair file (no duplication between files)
    # and bidirectional rows for easy per-item browsing
    # ------------------------------------------------
    pairs_bi = _make_bidirectional_pairs(aq3[["itemA", "itemB", "q3", "aq3", "n_overlap"]])

    # Sorting preference: itemA then q3 (descending) then aq3 (descending)
    pairs_bi = pairs_bi.sort_values(["itemA", "q3", "aq3"], ascending=[True, False, False]).reset_index(drop=True)

    pairs_path = os.path.join(out_dir, "q3_aq3_pairs.csv")
    pairs_bi.to_csv(pairs_path, index=False)

    # ------------------------------------------------
    # Clusters file (threshold sweep)
    # Metric is fixed by this pipeline (aQ3), thresholds are:
    #   - aq3_edge_threshold (graph edge threshold used for cluster_id in item bank)
    #   - aq3_count_thresholds (useful to browse cluster emergence vs threshold)
    # ------------------------------------------------
    thresholds_for_clusters = [aq3_edge_threshold, *list(aq3_count_thresholds)]
    clusters_tbl = _clusters_table(aq3, col="aq3", thresholds=thresholds_for_clusters, min_overlap=min_overlap)

    clusters_path = os.path.join(out_dir, "q3_aq3_clusters.csv")
    clusters_tbl.to_csv(clusters_path, index=False)

    # Per-item metrics + cluster_id (at aq3_edge_threshold only)
    item_q3 = _item_metrics(q3, "q3", all_items, aq3_count_thresholds=aq3_count_thresholds)
    item_aq3 = _item_metrics(aq3, "aq3", all_items, aq3_count_thresholds=aq3_count_thresholds)

    # cluster_id for augmentation uses the primary edge threshold only
    comps_primary = _cluster_on_threshold(aq3, col="aq3", threshold=aq3_edge_threshold, min_overlap=min_overlap)
    clusters_primary = pd.DataFrame({
        "itemID": [i for c in comps_primary for i in c],
        "cluster_id": [k + 1 for k, c in enumerate(comps_primary) for _ in c],
    })

    items_df = pd.read_csv(item_params_csv)
    items_df["itemID"] = items_df["itemID"].astype(str)

    # Remove old diagnostics (safe re-runs)
    items_df = items_df[[c for c in items_df.columns if not c.startswith(diagnostic_prefixes)]]

    merged = (
        items_df
        .merge(item_q3, on="itemID", how="left")
        .merge(item_aq3, on="itemID", how="left")
        .merge(clusters_primary, on="itemID", how="left")
    )
    merged["cluster_id"] = merged["cluster_id"].fillna(0).astype(int)

    if write_item_params_back:
        _backup_file(item_params_csv, backup_suffix=item_params_backup_suffix)
        merged.to_csv(item_params_csv, index=False)
        items_out_path = item_params_csv
    else:
        items_out_path = os.path.join(out_dir, "items_en_with_q3.csv")
        merged.to_csv(items_out_path, index=False)

    # ================================================================
    # ======================= SUMMARY PRINT ==========================
    # ================================================================

    def fmt_pct(x):
        return f"{100.0 * x:.2f}%"

    def pct(series, p):
        series = series.dropna()
        return float(np.percentile(series.to_numpy(), p)) if len(series) else np.nan

    print("\n================ Q3 / aQ3 SUMMARY =================")
    print(f"Persons used          : {len(persons)}")
    print(f"Items analysed        : {len(all_items)}")
    print(f"Item pairs (Q3)       : {len(q3)}")
    print(f"Item pairs (aQ3)      : {len(aq3)}")
    print(f"MIN_OVERLAP           : {min_overlap}")
    print(f"AQ3_EDGE_THRESHOLD    : {aq3_edge_threshold:.2f}")
    print("---------------------------------------------------")

    for name, d, col in [("Q3", q3, "q3"), ("aQ3", aq3, "aq3")]:
        s = d[col].dropna()
        n = int(len(s))
        if n == 0:
            print(f"{name}: no eligible pairs after overlap filtering.")
            continue

        p5, p25, p50, p75, p95, p99 = [pct(s, p) for p in (5, 25, 50, 75, 95, 99)]
        mean = float(s.mean())
        sd = float(s.std(ddof=1)) if n > 1 else np.nan
        mx = float(s.max())
        mn = float(s.min())

        print(f"{name} distribution (pairs, n={n})")
        print(f"  mean/SD            : {mean:.4f} / {sd:.4f}")
        print(f"  percentiles        : p5 {p5:.4f}  p25 {p25:.4f}  p50 {p50:.4f}  p75 {p75:.4f}  p95 {p95:.4f}  p99 {p99:.4f}")
        print(f"  min / max          : {mn:.4f} / {mx:.4f}")

        if name == "aQ3":
            for thr in (0.10, 0.20, 0.30, 0.40):
                frac = float((s >= thr).mean())
                print(f"  pairs with {name}>={thr:.2f}: {fmt_pct(frac)}")

        print("")

    ov = aq3["n_overlap"].dropna() if "n_overlap" in aq3.columns else pd.Series([], dtype=float)
    if len(ov):
        ov_p10, ov_p50, ov_p90 = [pct(ov, p) for p in (10, 50, 90)]
        print("Overlap (n_overlap) distribution (aQ3 pairs)")
        print(f"  p10/p50/p90        : {ov_p10:.0f} / {ov_p50:.0f} / {ov_p90:.0f}")
        print(f"  max overlap        : {int(ov.max())}")
        print("")

    n_cluster_items = int((merged["cluster_id"] > 0).sum())
    cluster_sizes = merged.loc[merged["cluster_id"] > 0, "cluster_id"].value_counts()
    n_clusters = int(cluster_sizes.size)
    largest_cluster = int(cluster_sizes.max()) if n_clusters else 0

    print("aQ3 cluster structure (threshold graph)")
    print(f"  items in clusters  : {n_cluster_items} / {len(all_items)} ({fmt_pct(n_cluster_items / max(1, len(all_items)))})")
    print(f"  number of clusters : {n_clusters}")
    print(f"  largest cluster    : {largest_cluster}")
    if n_clusters:
        small = int((cluster_sizes <= 2).sum())
        med = int(((cluster_sizes >= 3) & (cluster_sizes <= 5)).sum())
        large = int((cluster_sizes >= 6).sum())
        print(f"  cluster size counts: <=2: {small}  3-5: {med}  >=6: {large}")
    print("===================================================\n")
    print("DONE.")

    return {
        "q3": q3,
        "aq3": aq3,
        "items_augmented": merged,
        "clusters_table": clusters_tbl,
        "pairs_bidirectional": pairs_bi,
        "paths": {
            # Keep legacy keys but point them to the single consolidated file
            "q3_pairs_csv": pairs_path,
            "aq3_pairs_csv": pairs_path,
            "pairs_csv": pairs_path,
            "clusters_csv": clusters_path,
            "items_out_csv": items_out_path,
        },
        "counts": {
            "persons_used": len(persons),
            "items_analysed": len(all_items),
            "pairs_q3": len(q3),
            "pairs_aq3": len(aq3),
        },
    }



# ============================================================
# ITEM-LEVEL OUTPUTS (Q3 / aQ3) — HOW TO INTERPRET / USE THEM
# ============================================================
# This script computes residual-correlation local dependence diagnostics:
#   Q3  = corr( raw residuals r = x - P(theta,b) )
#   aQ3 = item-mean adjusted Q3:
#           aQ3_ij = Q3_ij - mean(Q3 for item i) - mean(Q3 for item j)
#
# The script merges per-item summaries into ITEM_PARAMS_CSV (items.csv)
# using prefixes "q3_*" and "aq3_*", plus an aQ3-based cluster label.
# These fields are intended for bank filtering and for reporting LID.
#
# --------------------------
# 0) IMPORTANT SCOPE NOTES
# --------------------------
# - Uses fixed person thetas and item bs (no estimation here).
# - Uses ONLY observed responses (CAT sparse by design).
# - Pairwise correlations computed only if n_overlap >= MIN_OVERLAP.
# - Persons are filtered to have >= MIN_ITEMS_PER_PERSON responses.
# - Re-runs are safe: old diagnostic columns are dropped before merge.
#
# --------------------------
# 1) PER-ITEM Q3 DIAGNOSTICS (prefix: q3_)
# --------------------------
# q3_pairs_n (int)
#   Number of eligible item pairs involving this item (after overlap filtering).
#   Higher values = item frequently co-administered with other items (more stable estimates).
#
# q3_max (float)
#   Maximum raw residual correlation (worst-case) involving this item.
#   Large values typically indicate strong local dependence with at least one other item
#   (near-duplicate content, shared cues, format artifacts, etc.).
#
# q3_mean (float)
#   Mean raw residual correlation over all eligible pairs for this item.
#   Elevated mean suggests diffuse dependence with many items (less common than isolated pairs).
#
# q3_overlap_max (int)
#   Maximum overlap count among this item’s eligible pairs (largest number of persons
#   who answered this item and the partner item).
#   Use as a stability indicator: high q3_max with tiny overlaps can be noisy.
#
# --------------------------
# 2) PER-ITEM aQ3 DIAGNOSTICS (prefix: aq3_)
# --------------------------
# aQ3 is "item-mean adjusted" to reduce inflation attributable to items that
# generally correlate with many others. In practice, aq3_max is often more
# diagnostic for pair-specific dependence than q3_max.
#
# aq3_pairs_n (int)
#   Same idea as q3_pairs_n but for the aQ3 table (should usually match q3_pairs_n).
#
# aq3_max (float)
#   Maximum adjusted residual correlation involving this item.
#   Primary "alarm bell" for local dependence clusters in this script.
#   Large aq3_max indicates pair-specific dependence beyond general item tendency.
#
# aq3_mean (float)
#   Mean adjusted residual correlation for this item.
#   Typically near 0; positive mean may indicate broad residual association.
#
# aq3_overlap_max (int)
#   Same as q3_overlap_max but on the aQ3 table (overlap is the same; kept for convenience).
#
# aq3_n_ge_0_10 / _0_20 / _0_30 / _0_40 (int)
#   Counts of this item’s pairs with aQ3 >= the threshold.
#   Interpretation:
#     - few exceedances (e.g., 1–2) -> localized dependence (a couple problematic partners)
#     - many exceedances           -> diffuse dependence or membership in a larger cluster
#   Useful for filtering because it distinguishes “one bad pair” vs “systematic dependence”.
#
# --------------------------
# 3) aQ3-BASED CLUSTER LABEL
# --------------------------
# cluster_id (int)
#   Connected-component ID in a thresholded aQ3 graph:
#     edge if (aq3 >= AQ3_EDGE_THRESHOLD) AND (n_overlap >= MIN_OVERLAP)
#   0 => item not in any dependence cluster at that threshold.
#   >0 => item belongs to a cluster of mutually connected locally dependent items.
#
# IMPORTANT INTERPRETATION NOTE:
#   cluster_id is based on graph connectivity; it does NOT mean all items in the cluster
#   are mutually dependent. Use it as a navigation / grouping tool, not as a single-pass
#   “remove everything” flag.
#
# --------------------------
# 4) PRACTICAL FILTERING LOGIC (RULES OF THUMB)
# --------------------------
# Stronger evidence for review/removal when multiple signals agree, e.g.:
#   - high aq3_max AND
#   - aq3_n_ge_0_20 is non-trivial (many partners) AND/OR
#   - cluster_id > 0 (especially if cluster is large) AND/OR
#   - corroborated by PCAR signals (large |pcar_loading_1|, strong residual clusters).
#
# Avoid removing items based ONLY on:
#   - a single high value from a low-overlap pair (check *_overlap_max / pair overlaps)
#   - cluster_id alone (graph percolation can connect via a few edges)
#   - q3_mean without looking at aq3 (q3 is more susceptible to item-wise inflation)
#
# Recommended workflow:
#   1) Sort by aq3_max and inspect top pairs (aq3_pairs.csv).
#   2) Use aq3_n_ge_* counts to separate isolated pairs vs diffuse clusters.
#   3) Use cluster_id to inspect clusters as groups and prune redundancies.
#   4) Cross-check with PCAR loadings/row-max to confirm structural issues.
# ============================================================



# ============================================================
# BANK-LEVEL OUTPUTS (Q3 / aQ3) — HOW TO INTERPRET / REPORT THEM
# ============================================================
# This script exports pair-level Q3/aQ3 tables and prints a compact
# summary that can be used to describe the bank’s local dependence.
#
# --------------------------
# A) PAIR-LEVEL OUTPUTS (PRIMARY EVIDENCE)
# --------------------------
# OUT_DIR/q3_aq3_pairs.csv
#   Columns:
#     itemA, itemB, q3, aq3, n_overlap
#   Meaning:
#     Residual correlations for eligible item pairs (n_overlap >= MIN_OVERLAP).
#       q3  : raw residual correlation
#       aq3 : item-mean adjusted correlation isolating pair-specific dependence
#     Each pair appears in both directions (A→B and B→A) to enable easy
#     per-item browsing by sorting on itemA.
#   Use:
#     - identify strongest dependence pairs
#     - inspect item neighborhoods
#     - provide illustrative examples / supplements
#
# OUT_DIR/q3_aq3_clusters.csv
#   Columns:
#     cluster_id, threshold, items
#   Meaning:
#     Connected components formed by linking items with
#         aq3 >= threshold  and  n_overlap >= MIN_OVERLAP
#     computed for thresholds:
#         [AQ3_EDGE_THRESHOLD] + AQ3_COUNT_THRESHOLDS
#   Use:
#     - examine group-level dependence structure
#     - detect redundant or bridge items
#     - guide strategic bank refinement
#
# --------------------------
# B) BANK-LEVEL SUMMARIES YOU CAN REPORT
# --------------------------
# The console "Q3 / aQ3 SUMMARY" includes:
#   Persons used
#     -> analysis coverage / stability (after MIN_ITEMS_PER_PERSON filter)
#   Items analysed
#     -> size of item subset with residual vectors (after calibration filter)
#   Item pairs (Q3) / (aQ3)
#     -> how many eligible correlations were computed (after overlap filtering)
#   Q3/aQ3 distribution summaries
#     -> mean/SD, percentiles, min/max
#   Tail mass above key thresholds (aQ3 >= 0.10/0.20/0.30/0.40)
#     -> prevalence of mild/strong dependence
#   Overlap distribution
#     -> stability check under CAT sparsity
#   Cluster structure
#     -> how localized dependence is (items clustered, #clusters, largest size)
#
# Typical reporting language:
#   "Local dependence was evaluated using residual correlations (Q3) and
#    adjusted Q3 (aQ3) computed on observed CAT responses only (min overlap M).
#    The distribution was dominated by small correlations, with a small upper tail;
#    the proportion of pairs exceeding aQ3>=0.20 was X%, and dependence clusters
#    identified at aQ3>=T involved Y items (largest cluster size Z)."
# ============================================================