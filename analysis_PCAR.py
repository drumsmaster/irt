"""
============================================================
Dimensionality check using fixed Rasch parameters (PCAR)
+ item-level diagnostics merged back into item bank
============================================================

PURPOSE
-------
This script performs a residual-based dimensionality check similar to
Winsteps "PCAR" (PCA of residual correlations) using *fixed* Rasch
parameters (theta for persons, b for items).

It also computes useful *item-level* diagnostics and merges them back
into the item parameter file (items.csv), so that you have a single
"master item table" for bank filtering (together with q3/aQ3 outputs
from the other script).

WHAT THIS SCRIPT DOES
---------------------
1) Load a wide response table (persons × items; values 0/1/NA).
2) Merge person thetas and item difficulties b.
3) Compute expected probabilities P_ni = logistic(theta_n - b_i).
4) Compute residuals (observed only):
   - raw residual: r = x - P
   - standardized residual: z = (x - P) / sqrt(P(1-P))  (recommended)
5) Build a sparse residual matrix Z (persons × items; NaN = missing).
6) Compute the item-item residual correlation matrix (Q3-style) using
   pairwise-complete observations with min overlap constraint.
7) Optionally mean-center OFF-DIAGONALS of that correlation matrix
   ("Q3_adj" variant used as the PCA input).
8) Run PCA via eigen-decomposition on the residual correlation matrix.
9) Export outputs + merge item-level outputs back into items.csv.

NOTES ON TERMINOLOGY
--------------------
- In the Rasch/Q3 literature, "Q3" has multiple variants.
  Here, "Q3-style matrix" = item-item correlations of residuals.
- If USE_Q3_ADJ=True, we subtract the *global off-diagonal mean* from
  all off-diagonal entries (not the item-mean adjustment used in the
  dedicated q3/aQ3 script).

OUTPUTS (FILES)
--------------
OUTPUT_DIR/
  - pcar_q3_matrix.csv (optional; large)
      Residual correlation matrix used for PCA (Q3 or Q3_adj).
  - pcar_q3_summary.csv
      Distribution summary of off-diagonal correlations (mean/sd/p95/p99/max).
  - residual_pca_scree.csv
      Eigenvalues and explained variance ratios for the top K contrasts.
  - residual_pca_loadings.csv
      Per-item PCA loadings + b + exposure + stability metrics.
  - top_items_contrast1.csv
      20 items with the largest |Loading_Contrast_1|.

ITEM BANK AUGMENTATION
----------------------
The script also writes item-level PCAR diagnostics into INPUT_ITEMS (items.csv)
(or writes a copy into OUTPUT_DIR if WRITE_ITEM_PARAMS_BACK=False).

New / updated item-level columns use prefix "pcar_". Examples:
  - pcar_included              (0/1): survived PCAR filtering
  - pcar_n_observed            exposure count (responses observed)
  - pcar_valid_corrs_n         # non-NaN correlations in its Q3 row before fill
  - pcar_loading_1..pcar_loading_5   signed PCA loadings on contrasts
  - pcar_abs_loading_1         abs(loading_1) (easy filtering)
  - pcar_communalities_5       sum_{c=1..5} loading_c^2 (how "structural" item is)
  - pcar_side_1                -1/0/+1 based on loading threshold
  - pcar_flag_1                0/1 if abs(loading_1) >= PCAR_LOADING_FLAG
  - pcar_q3adj_row_max         max off-diagonal residual correlation in row
  - pcar_q3adj_row_mean        mean off-diagonal residual correlation in row
  - pcar_q3adj_n_ge_0_20       count of off-diagonal entries >= 0.20
  - pcar_cluster_id_1          cluster id from thresholded graph on Q3-used matrix

The exact "q3" vs "q3adj" prefix depends on USE_Q3_ADJ.

============================================================
"""

from __future__ import annotations

import os
import shutil
import json
from dataclasses import dataclass
from typing import Iterable, Optional, Sequence, Dict, Any, List, Tuple

import numpy as np
import pandas as pd
import networkx as nx


# ============================================================
# HELPERS
# ============================================================

def logistic(x: np.ndarray) -> np.ndarray:
    """Numerically stable enough for typical theta ranges; returns sigmoid(x)."""
    return 1.0 / (1.0 + np.exp(-x))


def ensure_dir(path: str) -> None:
    """Create output directory if it doesn't exist."""
    os.makedirs(path, exist_ok=True)


def safe_id_clean(s: pd.Series) -> pd.Series:
    """Force IDs to string and strip whitespace."""
    return s.astype(str).str.strip()


def backup_file(path: str, suffix: str) -> str:
    """Create backup of a file before overwriting; returns backup path."""
    bak = path + suffix
    shutil.copy2(path, bak)
    return bak


def offdiag_values(M: np.ndarray) -> np.ndarray:
    """Return off-diagonal entries of a square matrix as a 1D array."""
    return M[~np.eye(M.shape[0], dtype=bool)]


# ============================================================
# MAIN FUNCTION
# ============================================================

def run_pcar(
    *,
    input_responses: str,
    input_persons: str,
    input_items: str,
    output_dir: str,
    # Residual choice for Q3/PCA:
    #  - "std": standardized residuals z = (x-P)/sqrt(P(1-P))  (recommended)
    #  - "raw": raw residuals r = x-P
    residual_mode: str = "std",
    # Stability filters:
    min_item_responses: int = 30,
    min_pair_overlap: int = 30,
    min_valid_corrs_per_item_frac: float = 0.10,
    # PCA / PCAR settings
    num_components: int = 5,
    use_q3_adj: bool = True,
    save_q3_matrix: bool = False,
    # Item-level flags / clusters derived from PCAR results
    pcar_loading_flag: float = 0.30,
    pcar_edge_threshold: float = 0.30,
    # Counts of large residual correlations per item (row statistics)
    row_thresholds: Sequence[float] = tuple([0.10, 0.20, 0.30, 0.40, 0.50]),
    # Numerical safety
    eps: float = 1e-9,
    # Merge behavior (like q3 script)
    write_item_params_back: bool = True,
    item_params_backup_suffix: str = ".bak",
    diagnostic_prefixes: Tuple[str, ...] = ("pcar_",),
    # IO details
    responses_sep: str = "\t",
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Run residual-based dimensionality check (PCAR-like) with fixed Rasch parameters.

    Parameters
    ----------
    input_responses
        Wide responses table (persons × items). Must include 'personID' column and 0/1/NA item columns.
    input_persons
        Person parameters CSV with columns: personID, theta
    input_items
        Item parameters CSV with columns: itemID, b
    output_dir
        Directory for outputs.
    residual_mode
        "std" (recommended) or "raw".
    min_item_responses
        Minimum exposure to keep an item for PCAR.
    min_pair_overlap
        Minimum pairwise overlap to compute correlation.
    min_valid_corrs_per_item_frac
        Drop items whose Q3 row is too sparse (too many NaNs before fill).
    num_components
        Number of PCA components (contrasts) to store/loadings to compute.
    use_q3_adj
        If True, subtract global off-diagonal mean from the correlation matrix before PCA.
    save_q3_matrix
        If True, save the full matrix used for PCA (can be huge).
    pcar_loading_flag
        abs(loading_1) >= this => pcar_flag_1 = 1
    pcar_edge_threshold
        Threshold on matrix used for clustering edges.
    row_thresholds
        Per-row counts of off-diagonals >= thresholds.
    eps
        Numerical epsilon for standardization.
    write_item_params_back
        If True, overwrite `input_items` after backup; else write copy into output_dir.
    item_params_backup_suffix
        Suffix for backup file.
    diagnostic_prefixes
        Columns starting with these prefixes are removed from items.csv prior to merge (safe re-runs).
    responses_sep
        Separator for wide response matrix (default TSV).
    verbose
        Print console logs and summary.

    Returns
    -------
    dict with keys:
        - "q3_label": "q3" or "q3adj"
        - "q3_summary": DataFrame
        - "scree": DataFrame
        - "loadings": DataFrame (item-level diagnostics)
        - "items_merged": DataFrame (items.csv with merged pcar columns)
        - "paths": dict of output file paths
        - "run_summary": dict (same content as JSON summary file)
    """
    if verbose:
        print("--- Residual Dimensionality (fixed theta, b; PCAR-like) ---")

    ensure_dir(output_dir)
    if verbose:
        print(f"Output dir: {output_dir}")

    # ------------------------------------------------------------
    # 1) LOAD INPUTS
    # ------------------------------------------------------------
    if verbose:
        print("Loading files...")
    df_resp = pd.read_csv(input_responses, sep=responses_sep, dtype={"personID": str})
    df_p = pd.read_csv(input_persons, dtype={"personID": str})
    df_i = pd.read_csv(input_items, dtype={"itemID": str})

    if "theta" not in df_p.columns:
        raise ValueError("persons.csv must contain column 'theta'")
    if "b" not in df_i.columns:
        raise ValueError("items.csv must contain column 'b'")

    df_resp["personID"] = safe_id_clean(df_resp["personID"])
    df_p["personID"] = safe_id_clean(df_p["personID"])
    df_i["itemID"] = safe_id_clean(df_i["itemID"])

    n_persons_wide = int(df_resp.shape[0])
    n_items_wide = int(df_resp.shape[1] - 1)
    if verbose:
        print(f"Wide response matrix: {n_persons_wide} persons x {n_items_wide} items")

    # ------------------------------------------------------------
    # 2) LONG FORMAT + MERGE PARAMETERS + COMPUTE RESIDUALS
    # ------------------------------------------------------------
    if verbose:
        print("Melting to long format (observed responses only)...")
    value_vars = [c for c in df_resp.columns if c != "personID"]
    df_long = df_resp.melt(
        id_vars=["personID"],
        value_vars=value_vars,
        var_name="itemID",
        value_name="response",
    )
    df_long = df_long.dropna(subset=["response"])
    df_long["itemID"] = safe_id_clean(df_long["itemID"])

    # Ensure 0/1 numeric
    df_long["response"] = pd.to_numeric(df_long["response"], errors="coerce")
    df_long = df_long.dropna(subset=["response"])
    df_long = df_long[df_long["response"].isin([0, 1])].copy()

    if verbose:
        print(f"Observed responses (after cleaning): {len(df_long):,}")

    if verbose:
        print("Merging theta and b...")
    df = df_long.merge(df_p[["personID", "theta"]], on="personID", how="inner")
    df = df.merge(df_i[["itemID", "b"]], on="itemID", how="inner")
    if verbose:
        print(f"Observed responses after merges (theta+b present): {len(df):,}")

    # Expected probabilities P = logistic(theta - b)
    xb = df["theta"].to_numpy(dtype=float) - df["b"].to_numpy(dtype=float)
    p = logistic(xb)

    # Residuals
    x = df["response"].to_numpy(dtype=float)
    r = x - p  # raw residual
    var = p * (1.0 - p)

    rm = residual_mode.lower().strip()
    if rm == "std":
        resid = r / np.sqrt(var + eps)
        resid_name = "std_residual"
    elif rm == "raw":
        resid = r
        resid_name = "raw_residual"
    else:
        raise ValueError("residual_mode must be 'std' or 'raw'")

    df[resid_name] = resid

    # ------------------------------------------------------------
    # 3) PIVOT TO PERSON×ITEM MATRIX (NaN = missing due to CAT)
    # ------------------------------------------------------------
    if verbose:
        print("Pivoting to residual matrix (persons x items; NaN = missing)...")
    Z = df.pivot(index="personID", columns="itemID", values=resid_name)

    # Exposure = number of observed responses per item (before filtering)
    exposure = Z.notna().sum(axis=0).sort_values(ascending=False)

    # Filter out very low-exposure items (unstable correlations / PCA)
    keep_items = exposure[exposure >= min_item_responses].index
    Z = Z.loc[:, keep_items]
    if verbose:
        print(f"Items after exposure filter (>= {min_item_responses}): {Z.shape[1]}")

    if Z.shape[1] < max(3, num_components):
        raise ValueError("Too few items after exposure filtering; lower min_item_responses")

    # ------------------------------------------------------------
    # 4) Q3-STYLE MATRIX: pairwise residual correlations (min overlap)
    # ------------------------------------------------------------
    if verbose:
        print(f"Computing residual correlation matrix (min overlap {min_pair_overlap})...")
    # Pairwise-complete Pearson correlations; entries with < min_pair_overlap become NaN
    Q3 = Z.corr(method="pearson", min_periods=min_pair_overlap)

    # Row-wise number of valid (non-NaN) correlations BEFORE fill (diagonal included currently)
    valid_counts = Q3.notna().sum(axis=1)

    # We want to drop items whose row is too sparse (too few reliable pairwise correlations)
    min_valid = max(10, int(min_valid_corrs_per_item_frac * Q3.shape[0]))
    items_ok = valid_counts[valid_counts >= min_valid].index
    Q3 = Q3.loc[items_ok, items_ok].copy()
    valid_counts = valid_counts.reindex(items_ok)

    # Make diagonal 1.0 (correlation of item with itself)
    np.fill_diagonal(Q3.values, 1.0)

    # Fill remaining NaNs with 0.0 (needed for PCA without NaNs)
    Q3 = Q3.fillna(0.0)
    Q3 = (Q3 + Q3.T) / 2.0  # enforce symmetry

    if verbose:
        print(f"Items retained for Q3/PCA: {Q3.shape[0]} (min valid corrs per item: {min_valid})")
    if Q3.shape[0] < max(3, num_components):
        raise ValueError("Too few items left after Q3 sparsity cleanup; relax thresholds")

    # ------------------------------------------------------------
    # Optional adjusted variant (global mean-centering of off-diagonals)
    # ------------------------------------------------------------
    if use_q3_adj:
        A = Q3.to_numpy(dtype=float).copy()
        mask_off = ~np.eye(A.shape[0], dtype=bool)
        mean_off = float(A[mask_off].mean())
        A[mask_off] = A[mask_off] - mean_off
        np.fill_diagonal(A, 1.0)
        Q3_used = pd.DataFrame(A, index=Q3.index, columns=Q3.columns)
        q3_label = "q3adj"
        if verbose:
            print(f"Using Q3_adj: subtracted global off-diagonal mean = {mean_off:.6f}")
    else:
        Q3_used = Q3
        q3_label = "q3"
        mean_off = float(offdiag_values(Q3_used.to_numpy(dtype=float)).mean())

    paths: Dict[str, str] = {}

    # Save full matrix only if explicitly requested
    if save_q3_matrix:
        q3_path = os.path.join(output_dir, f"pcar_{q3_label}_matrix.csv")
        Q3_used.to_csv(q3_path)
        paths["q3_matrix"] = q3_path
        if verbose:
            print(f"Saved full matrix: {q3_path}")

    # ------------------------------------------------------------
    # Q3 summary (off-diagonal distribution)
    # ------------------------------------------------------------
    A_used = Q3_used.to_numpy(dtype=float)
    off = offdiag_values(A_used)

    q3_summary = pd.DataFrame({
        "metric": ["mean", "sd", "p95", "p99", "max"],
        "value": [
            float(off.mean()),
            float(off.std(ddof=1)),
            float(np.quantile(off, 0.95)),
            float(np.quantile(off, 0.99)),
            float(off.max()),
        ],
    })
    q3_sum_path = os.path.join(output_dir, f"pcar_{q3_label}_summary.csv")
    q3_summary.to_csv(q3_sum_path, index=False)
    paths["q3_summary"] = q3_sum_path
    if verbose:
        print(f"Saved pcar_{q3_label} summary: {q3_sum_path}")

    # ------------------------------------------------------------
    # 5) Residual PCA via eigen-decomposition on Q3_used
    # ------------------------------------------------------------
    if verbose:
        print(f"Running residual PCA on pcar_{q3_label} (top {num_components} contrasts)...")

    M = A_used
    eigvals, eigvecs = np.linalg.eigh(M)  # symmetric => real eigendecomp
    order = np.argsort(eigvals)[::-1]     # descending
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]

    k = min(int(num_components), int(eigvals.shape[0]))
    eigvals_k = eigvals[:k]
    eigvecs_k = eigvecs[:, :k]

    total_var = float(np.trace(M))
    explained = eigvals_k / (total_var + 1e-12)

    # PCA loadings on correlation matrices: v * sqrt(lambda)
    loadings = eigvecs_k * np.sqrt(np.maximum(eigvals_k, 0.0))

    scree = pd.DataFrame({
        "Contrast_ID": np.arange(1, k + 1),
        "Eigenvalue": eigvals_k,
        "Explained_Variance_Ratio": explained,
        "Matrix": f"pcar_{q3_label}",
        "Residual_Mode": residual_mode,
        "Min_Pair_Overlap": min_pair_overlap,
        "Min_Item_Responses": min_item_responses,
    })
    scree_path = os.path.join(output_dir, "residual_pca_scree.csv")
    scree.to_csv(scree_path, index=False)
    paths["scree"] = scree_path
    if verbose:
        print(f"Saved scree: {scree_path}")

    # ------------------------------------------------------------
    # 6) Item-level table: loadings + stability metrics + row stats
    # ------------------------------------------------------------
    items_final = Q3_used.index.to_list()

    load_df = pd.DataFrame(
        loadings,
        index=items_final,
        columns=[f"pcar_loading_{i + 1}" for i in range(k)],
    ).reset_index().rename(columns={"index": "itemID"})

    # Add b (difficulty)
    load_df = load_df.merge(df_i[["itemID", "b"]], on="itemID", how="left")

    # Exposure
    expo_final = exposure.reindex(items_final).fillna(0).astype(int)
    load_df["pcar_n_observed"] = load_df["itemID"].map(expo_final.to_dict()).fillna(0).astype(int)

    # Valid correlations per item row (before fill)
    vc_map = valid_counts.to_dict()
    load_df["pcar_valid_corrs_n"] = load_df["itemID"].map(vc_map).fillna(0).astype(int)

    # Included flag
    load_df["pcar_included"] = 1

    # Abs loading on contrast 1
    load_df["pcar_abs_loading_1"] = load_df["pcar_loading_1"].abs()

    # Side / flag based on loading_1
    load_df["pcar_flag_1"] = (load_df["pcar_abs_loading_1"] >= pcar_loading_flag).astype(int)
    load_df["pcar_side_1"] = np.select(
        [
            load_df["pcar_loading_1"] >= pcar_loading_flag,
            load_df["pcar_loading_1"] <= -pcar_loading_flag,
        ],
        [1, -1],
        default=0,
    ).astype(int)

    # Communalities across K contrasts
    loading_cols = [f"pcar_loading_{i + 1}" for i in range(k)]
    load_df[f"pcar_communalities_{k}"] = (load_df[loading_cols] ** 2).sum(axis=1)

    # ------------------------------------------------------------
    # 7) Row stats from Q3_used (off-diagonals)
    # ------------------------------------------------------------
    row_max: List[float] = []
    row_mean: List[float] = []
    row_counts: Dict[float, List[int]] = {float(thr): [] for thr in row_thresholds}

    for idx, _iid in enumerate(items_final):
        row = A_used[idx, :].copy()
        row[idx] = np.nan
        v = row[~np.isnan(row)]
        row_max.append(float(np.max(v)) if v.size else np.nan)
        row_mean.append(float(np.mean(v)) if v.size else np.nan)
        for thr in row_thresholds:
            thr_f = float(thr)
            row_counts[thr_f].append(int(np.sum(v >= thr_f)) if v.size else 0)

    row_prefix = f"pcar_{q3_label}"
    load_df[f"{row_prefix}_row_max_offdiag"] = row_max
    load_df[f"{row_prefix}_row_mean_offdiag"] = row_mean
    for thr in row_thresholds:
        colname = f"{row_prefix}_n_ge_{float(thr):.2f}".replace(".", "_")
        load_df[colname] = row_counts[float(thr)]

    # ------------------------------------------------------------
    # 8) Clustering on residual correlation matrix (threshold graph)
    # ------------------------------------------------------------
    edges: List[Tuple[str, str]] = []
    n = len(items_final)
    thr_edge = float(pcar_edge_threshold)

    for i in range(n):
        for j in range(i + 1, n):
            if A_used[i, j] >= thr_edge:
                edges.append((items_final[i], items_final[j]))

    G = nx.Graph()
    G.add_edges_from(edges)
    comps = list(nx.connected_components(G))

    cluster_map: Dict[str, int] = {}
    for k_id, comp in enumerate(comps, start=1):
        for iid in comp:
            cluster_map[iid] = k_id

    load_df["pcar_cluster_id_1"] = load_df["itemID"].map(cluster_map).fillna(0).astype(int)
    load_df["pcar_cluster_size_1"] = load_df["pcar_cluster_id_1"].map(
        load_df["pcar_cluster_id_1"].value_counts().to_dict()
    ).fillna(0).astype(int)

    # ------------------------------------------------------------
    # 9) Save PCAR loadings table + top items for contrast 1
    # ------------------------------------------------------------
    load_path = os.path.join(output_dir, "residual_pca_loadings.csv")
    load_df.to_csv(load_path, index=False)
    paths["loadings"] = load_path
    if verbose:
        print(f"Saved loadings + diagnostics: {load_path}")

    top = load_df[["itemID", "b", "pcar_n_observed", "pcar_loading_1", "pcar_abs_loading_1"]].copy()
    top = top.sort_values("pcar_abs_loading_1", ascending=False).head(20)
    top_path = os.path.join(output_dir, "top_items_contrast1.csv")
    top.to_csv(top_path, index=False)
    paths["top_items_contrast1"] = top_path
    if verbose:
        print(f"Saved top items for Contrast 1: {top_path}")

    # ------------------------------------------------------------
    # 10) Merge item-level outputs back into item bank (safe re-runs)
    # ------------------------------------------------------------
    items_df = pd.read_csv(input_items, dtype={"itemID": str})
    items_df["itemID"] = safe_id_clean(items_df["itemID"])

    # Drop old PCAR columns to make re-runs safe and deterministic
    items_df = items_df[[c for c in items_df.columns if not c.startswith(diagnostic_prefixes)]]

    merged = items_df.merge(load_df.drop(columns=["b"], errors="ignore"), on="itemID", how="left")
    merged["pcar_included"] = merged["pcar_included"].fillna(0).astype(int)

    if write_item_params_back:
        bak = backup_file(input_items, item_params_backup_suffix)
        merged.to_csv(input_items, index=False)
        paths["items_out"] = input_items
        paths["items_backup"] = bak
        if verbose:
            print(f"[INFO] Updated item bank written to: {input_items}")
            print(f"[INFO] Backup created at: {bak}")
    else:
        out_items = os.path.join(output_dir, "items_with_pcar.csv")
        merged.to_csv(out_items, index=False)
        paths["items_out"] = out_items
        if verbose:
            print(f"[INFO] Updated item bank written to: {out_items}")

    # ------------------------------------------------------------
    # 11) Write JSON summary for reproducibility
    # ------------------------------------------------------------
    run_summary = {
        "matrix_used": f"pcar_{q3_label}",
        "residual_mode": residual_mode,
        "n_persons_wide": n_persons_wide,
        "n_items_wide": n_items_wide,
        "n_observed_after_merge": int(len(df)),
        "min_item_responses": int(min_item_responses),
        "min_pair_overlap": int(min_pair_overlap),
        "min_valid_corrs_per_item": int(min_valid),
        "items_after_exposure_filter": int(len(keep_items)),
        "items_final_for_pca": int(len(items_final)),
        "num_components": int(k),
        "pcar_loading_flag": float(pcar_loading_flag),
        "pcar_edge_threshold": float(pcar_edge_threshold),
        "q3_offdiag_summary": q3_summary.to_dict(orient="records"),
        "top_eigenvalues": [float(v) for v in eigvals_k],
        "top_explained_ratios": [float(v) for v in explained],
    }
    summary_path = os.path.join(output_dir, "pcar_run_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(run_summary, f, indent=2)
    paths["run_summary_json"] = summary_path
    if verbose:
        print(f"Saved run summary: {summary_path}")

    # ------------------------------------------------------------
    # 12) Print concise console summary (quick sanity check)
    # ------------------------------------------------------------
    if verbose:
        n_clusters = int((load_df["pcar_cluster_id_1"] > 0).sum())
        largest_cluster = (
            int(load_df.loc[load_df["pcar_cluster_id_1"] > 0, "pcar_cluster_size_1"].max())
            if n_clusters
            else 0
        )
        flagged = int(load_df["pcar_flag_1"].sum())

        print("\n" + "-" * 72)
        print("PCAR SUMMARY")
        print(f"Residual mode            : {residual_mode}")
        print(f"Matrix used              : pcar_{q3_label} (use_q3_adj={use_q3_adj})")
        print(f"Persons (wide)           : {n_persons_wide}")
        print(f"Items (wide)             : {n_items_wide}")
        print(f"Observed responses used  : {len(df):,}")
        print(f"Items after exposure     : {len(keep_items)} (>= {min_item_responses} responses)")
        print(f"Items final for PCA      : {len(items_final)} (min valid corrs per item: {min_valid})")
        print(f"Pairs matrix size        : {len(items_final)} x {len(items_final)}")
        print(f"Off-diagonal mean        : {off.mean():.4f}  (sd {off.std(ddof=1):.4f})")
        print(f"Off-diagonal p99 / max   : {np.quantile(off, 0.99):.4f} / {off.max():.4f}")
        print(f"Flag threshold |L1|      : {pcar_loading_flag}   flagged items: {flagged}")
        print(f"Cluster threshold        : {pcar_edge_threshold}  items in clusters: {n_clusters}")
        print(f"Largest cluster size     : {largest_cluster}")
        print("Top eigenvalues          : " + ", ".join([f"{v:.3f}" for v in eigvals_k]))
        print("Explained variance ratios:")
        for i in range(k):
            print(f"  Contrast {i + 1}: {explained[i]:.4f}")
        print("-" * 72)
        print("Interpretation workflow:")
        print(f"1) Check pcar_{q3_label}_summary.csv for p99/max (large values => dependence clusters).")
        print("2) Check residual_pca_scree.csv for a dominant secondary contrast (scree elbow).")
        print("3) Inspect top_items_contrast1.csv and pcar_side_1 for coherent content grouping.")
        print("4) Use items.csv (augmented) to filter based on combined q3/aQ3 + PCAR signals.")
        print("--- Done ---\n")

    return {
        "q3_label": q3_label,
        "q3_summary": q3_summary,
        "scree": scree,
        "loadings": load_df,
        "items_merged": merged,
        "paths": paths,
        "run_summary": run_summary,
    }


# ============================================================
# ITEM-LEVEL OUTPUTS (PCAR) — HOW TO INTERPRET / USE THEM
# ============================================================
# This script writes item-level PCAR diagnostics into the item bank
# (items.csv) with prefix "pcar_*". These are meant to be read together
# with q3/aQ3 diagnostics from the separate Q3 script.
#
# --------------------------
# 1) STABILITY / INCLUSION
# --------------------------
# pcar_included (0/1)
#   1 = item survived PCAR filters (exposure + valid correlation density).
#   0 = excluded; PCAR metrics for the item are NA/unreliable.
#   IMPORTANT: pcar_included=0 does NOT mean "bad item" — only "no PCAR info".
#
# pcar_n_observed (int)
#   Exposure in this dataset (# observed responses for item).
#   Lower exposure => noisier residual correlations/loadings.
#   Heuristic: 30–100 cautious, >200 typically stable (depends on bank/CAT).
#
# pcar_valid_corrs_n (int)
#   How many item-item correlations were actually estimable (met MIN_PAIR_OVERLAP)
#   before NaNs were filled with 0. Lower values => more CAT sparsity around item.
#   Use for trust: large loadings with tiny pcar_valid_corrs_n => be skeptical.
#
# --------------------------
# 2) PCA LOADINGS (DIMENSIONALITY SIGNAL)
# --------------------------
# pcar_loading_1 ... pcar_loading_K (float, signed)
#   Signed loadings on residual contrasts (PCA on residual correlation matrix).
#   Magnitude = strength of participation; sign = which side of contrast.
#   High |loading| suggests structured residual behavior (content cluster, format, LID).
#
# pcar_abs_loading_1 (float)
#   abs(pcar_loading_1); convenient magnitude for filtering.
#   Typical heuristic: <0.20 small, 0.20–0.30 moderate, >=0.30 strong.
#
# pcar_flag_1 (0/1), pcar_side_1 (-1/0/+1)
#   pcar_flag_1 = 1 if abs(loading_1) >= PCAR_LOADING_FLAG (e.g., 0.30).
#   pcar_side_1 indicates direction: +1 (positive), -1 (negative), 0 (small).
#   Use pcar_side_1 to interpret what Contrast 1 represents (which items group together).
#
# pcar_communalities_K (float)
#   Sum of squared loadings over first K contrasts:
#       h2 = sum_{c=1..K} loading_c^2
#   Higher => item is deeply involved in residual structure (stronger “non-unidimensional” signal).
#
# --------------------------------------------
# 3) RESIDUAL-CORRELATION ROW STATS (Q3-LIKE)
# --------------------------------------------
# These are computed from the matrix actually used for PCA:
#   if USE_Q3_ADJ=True => prefix is pcar_q3adj_*
#   else               => prefix is pcar_q3_*
#
# pcar_q3adj_row_max_offdiag (float)
#   Max off-diagonal residual correlation involving this item.
#   “Alarm bell” for near-duplicates / strong local dependence.
#
# pcar_q3adj_row_mean_offdiag (float)
#   Mean off-diagonal residual correlation for this item.
#   Elevated mean => diffuse dependence with many items; near-zero => mostly independent.
#
# pcar_q3adj_n_ge_0_10 / _0_20 / _0_30 / _0_40 (int)
#   Counts of off-diagonal correlations >= threshold.
#   Distinguishes localized dependence (few large) vs diffuse dependence (many moderate).
#
# --------------------------
# 4) PCAR-BASED CLUSTERING
# --------------------------
# pcar_cluster_id_1 (int)
#   Connected-component ID from a threshold graph built on the PCAR matrix used:
#     edge if correlation >= PCAR_EDGE_THRESHOLD (e.g., 0.20).
#   0 => not in a cluster; >0 => part of a residual correlation cluster.
#
# pcar_cluster_size_1 (int)
#   Size of the residual cluster for the item.
#   Size 2–3 => small LID group; >=5 often indicates a coherent secondary dimension/content domain.
#
# --------------------------
# 5) PRACTICAL FILTERING LOGIC (RULES OF THUMB)
# --------------------------
# Stronger evidence for review/removal when multiple signals agree, e.g.:
#   - high aq3_max (from Q3 script) AND
#   - high pcar_q3adj_row_max_offdiag AND/OR
#   - pcar_flag_1==1 (large |loading_1|) AND/OR
#   - membership in a non-trivial cluster (pcar_cluster_size_1 >= 3) AND/OR
#   - high pcar_communalities_K
#
# Avoid removing items based ONLY on:
#   - PCAR metrics alone (especially if exposure is low)
#   - pcar_included==0 (means “insufficient data”, not “bad”)
#
# Recommended workflow:
#   1) Scan for extreme values: aq3_max, pcar_q3adj_row_max_offdiag, abs_loading_1.
#   2) Check clustering: (aq3 cluster_id from Q3 script) and pcar_cluster_id_1 here.
#   3) Inspect top-loading items by pcar_side_1 to interpret the contrast.
#   4) Make decisions using multiple criteria + substantive review of the item content.
# ============================================================




# ============================================================
# BANK-LEVEL OUTPUTS (PCAR) — HOW TO INTERPRET / REPORT THEM
# ============================================================
# The PCAR script produces several *bank-level* diagnostics that
# describe the overall residual structure of the item bank
# (complementing pairwise Q3/aQ3 analyses).
#
# --------------------------
# A) RESIDUAL CORRELATION DISTRIBUTION (Q3-STYLE)
# --------------------------
# Source:
#   - pcar_q3_summary.csv
#   - console summary (mean / sd / p95 / p99 / max)
#
# What it is:
#   Distribution of OFF-DIAGONAL residual correlations between items,
#   computed from standardized (or raw) residuals with overlap filtering.
#
# How to interpret:
#   - Mean ≈ 0      -> globally well-behaved bank
#   - Small SD      -> local dependence is rare and localized
#   - p95 / p99     -> strength of the upper tail (problematic pairs)
#   - Max           -> worst-case local dependence observed
#
# Typical reporting language:
#   "Residual correlations were centered near zero, with a small upper
#    tail (p99 ≈ X), indicating limited local dependence."
#
# --------------------------
# B) RESIDUAL PCA SPECTRUM (DIMENSIONALITY STRENGTH)
# --------------------------
# Source:
#   - residual_pca_scree.csv
#   - console: eigenvalues + explained variance ratios
#
# What it is:
#   PCA on the residual correlation matrix (Q3 or Q3_adj),
#   yielding secondary residual contrasts.
#
# How to interpret:
#   - Large first eigenvalue + rapid decay -> near-unidimensional
#   - Several comparable eigenvalues       -> structured residual dimensions
#   - No sharp elbow                        -> mostly noise-driven residuals
#
# IMPORTANT:
#   Classical eigenvalue>1 or >2 rules do NOT apply here.
#   Interpretation is qualitative (shape + magnitude + item clusters).
#
# Typical reporting language:
#   "Residual PCA revealed one dominant secondary contrast accounting
#    for ~X% of residual variance, with subsequent contrasts contributing
#    substantially less."
#
# --------------------------
# C) RESIDUAL CLUSTER STRUCTURE
# --------------------------
# Source:
#   - pcar_cluster_id_1 / pcar_cluster_size_1 (item-level)
#   - console summary (items in clusters, largest cluster)
#
# What it is:
#   Graph-based clustering of items using a thresholded residual
#   correlation matrix (edge if corr >= PCAR_EDGE_THRESHOLD).
#
# How to interpret:
#   - Few small clusters (size 2–4) -> localized dependence
#   - One large cluster             -> coherent secondary dimension
#   - Many medium clusters          -> heterogeneous content effects
#
# Typical reporting language:
#   "Residual clustering identified several small item groups, with
#    no large cluster spanning the bank, suggesting localized violations
#    of independence."
#
# --------------------------
# D) COVERAGE / STABILITY DIAGNOSTICS
# --------------------------
# Source:
#   - console output
#   - pcar_run_summary.json
#
# What it is:
#   Counts and thresholds documenting how much data supports the analysis:
#     - number of persons
#     - number of items retained
#     - minimum exposure per item
#     - minimum pairwise overlap
#
# How to interpret:
#   These do NOT describe the bank itself, but justify that PCAR results
#   are stable and not artifacts of CAT sparsity.
#
# Typical reporting language:
#   "Dimensionality analyses were restricted to items with at least N
#    observed responses and at least M pairwise overlaps, yielding a
#    final matrix of K items."
#
# --------------------------
# E) HOW THIS FITS WITH Q3 / aQ3 ANALYSES
# --------------------------
# - Q3 / aQ3 script: pairwise local dependence diagnostics
# - PCAR script:    global residual structure / dimensionality
#
# Together they support claims about:
#   (1) overall level of local dependence,
#   (2) presence/absence of secondary residual dimensions,
#   (3) whether violations are localized or widespread.
# ============================================================