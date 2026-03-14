import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

###############################
# helpers
###############################

def _safe_label(s: str) -> str:
    """
    Convert labels to safe strings for column names / filenames.
    Example: "Native speakers" -> "native_speakers"
    """
    s = (s or "").strip().lower()
    s = re.sub(r"\s+", "_", s)
    s = re.sub(r"[^a-z0-9_]+", "", s)
    return s or "group"


def _load_items(path: str, *, word_col: str, b_col: str, min_b_allowed: float) -> pd.DataFrame:
    """
    Load + clean a group calibration file (keep only valid item IDs and b).
    - Strips item IDs
    - Coerces b to numeric
    - Drops NaNs
    - Applies a guard min_b_allowed to remove failed calibrations
    """
    df = pd.read_csv(path, usecols=[word_col, b_col])
    df[word_col] = df[word_col].astype(str).str.strip()
    df[b_col] = pd.to_numeric(df[b_col], errors="coerce")
    df = df.dropna(subset=[word_col, b_col])
    df = df[df[b_col] > float(min_b_allowed)]
    return df


#################################
# main functionality
#################################

def dif(
    GROUP_A_ITEMS_CSV: str,                 # path to item calibrations for group A
    GROUP_B_ITEMS_CSV: str,                 # path to item calibrations for group B
    GROUP_A_LABEL: str,                     # group A label (for reporting + column names)
    GROUP_B_LABEL: str,                     # group B label (for reporting + column names)
    Z_THR: float = 2.0,                     # standardized DIF threshold. Typical: 2 (screening), 2.5 (moderate), 3 (strict)
    OUT_DIR: str | None = None,             # output folder. If provided, writes dif_pairs.csv and dif_bank_summary.csv
    MASTER_ITEMS_CSV: str | None = None,    # master item file to augment with dif_* columns (updated in-place if provided)
    WORD_COL: str = "itemID",               # column with item IDs in all files
    B_COL: str = "b",                       # column with item difficulties in group calibration files
    MIN_B_ALLOWED: float = -12.0,           # guard against clearly failed calibrations (keeps extreme -inf-ish values out)
    DIAGNOSTIC_PREFIXES: tuple[str, ...] = ("dif_",),  # existing columns with these prefixes are removed before merging (safe re-runs)
    verbose: bool = True,                   # print summary + overview to console
) -> dict:
    """
    Simple cross-group DIF screening (SD-based; aligned scales) for Rasch 1PL item calibrations.

    OUTPUTS
    -------
    OUT_DIR/ (if provided)
      - dif_pairs.csv
          Pair-level table for overlapping items (A ∩ B), including:
          b_A, b_B, pred_B_from_A, resid, dif_z, dif_abs_z, dif_flag, dif_direction
      - dif_bank_summary.csv
          One-row bank-level summary (fit params + fit quality, residual SD, counts), including:
          fit_a, fit_c, fit_r2, pearson_r, spearman_r

    MASTER_ITEMS_CSV (if provided; updated in-place)
      Adds DIF item-level fields with prefix "dif_":
        dif_b_<A>                 group A difficulty
        dif_b_<B>                 group B difficulty
        dif_pred_<B>_from_<A>      predicted group B difficulty from aligned group A
        dif_resid_logits           raw DIF residual in logits
        dif_z                      standardized DIF residual (resid / SD(resid))
        dif_abs_z                  |dif_z|
        dif_flag                   1 if |dif_z| > Z_THR else 0
        dif_direction              "<A>_favored" or "<B>_favored"
        dif_overlap_in_pairfile    1 if item existed in BOTH group files else 0

    INTERPRETATION
    --------------
    dif_z > 0  => b_B is larger than expected given b_A after alignment
                 => item is relatively harder for Group B
                 => "<A>_favored"
    dif_z < 0  => b_B is smaller than expected given b_A after alignment
                 => item is relatively harder for Group A
                 => "<B>_favored"
    """
    if verbose:
        print("\n=================== START DIF ANALYSIS ==================\n")

    if OUT_DIR is not None:
        os.makedirs(OUT_DIR, exist_ok=True)

    A = _safe_label(GROUP_A_LABEL)
    B = _safe_label(GROUP_B_LABEL)

    # -------------------------
    # Load + clean group calibrations
    # -------------------------
    dfA = _load_items(GROUP_A_ITEMS_CSV, word_col=WORD_COL, b_col=B_COL, min_b_allowed=MIN_B_ALLOWED)
    dfB = _load_items(GROUP_B_ITEMS_CSV, word_col=WORD_COL, b_col=B_COL, min_b_allowed=MIN_B_ALLOWED)

    # Rename b columns to avoid collisions after merge
    dfA = dfA.rename(columns={B_COL: f"b_{A}"})
    dfB = dfB.rename(columns={B_COL: f"b_{B}"})

    # -------------------------
    # Merge on overlapping items (A ∩ B)
    # -------------------------
    df = dfA.merge(dfB, on=WORD_COL, how="inner")
    if df.empty:
        raise ValueError("No overlapping items between the two group files.")

    x = df[f"b_{A}"].to_numpy(dtype=float)
    y = df[f"b_{B}"].to_numpy(dtype=float)

    # -------------------------
    # Align scales: b_B ~= a*b_A + c
    # -------------------------
    a, c = np.polyfit(x, y, 1)

    pred_col = f"pred_{B}_from_{A}"
    df[pred_col] = a * df[f"b_{A}"] + c
    df["resid"] = df[f"b_{B}"] - df[pred_col]

    sd_resid = float(df["resid"].std(ddof=1))
    if not np.isfinite(sd_resid) or sd_resid < 1e-12:
        raise ValueError("Residual SD is invalid/too small; cannot standardize DIF.")

    # -------------------------
    # Fit quality + association (requested additions)
    # -------------------------
    # R^2 for the alignment regression y ~ a*x + c
    y_hat = a * x + c
    ss_res = float(np.sum((y - y_hat) ** 2))
    ss_tot = float(np.sum((y - float(np.mean(y))) ** 2))
    fit_r2 = float("nan") if ss_tot <= 0 else float(1.0 - ss_res / ss_tot)

    # Pearson correlation between group difficulty estimates (raw b_A vs b_B)
    pearson_r = float("nan")
    if len(x) >= 2 and np.std(x) > 0 and np.std(y) > 0:
        pearson_r = float(np.corrcoef(x, y)[0, 1])

    # Spearman rank correlation (computed via pandas; no extra deps)
    spearman_r = float(pd.Series(x).corr(pd.Series(y), method="spearman"))

    # -------------------------
    # DIF residual standardization + flagging
    # -------------------------
    df["dif_z"] = df["resid"] / sd_resid
    df["dif_abs_z"] = df["dif_z"].abs()
    df["dif_flag"] = (df["dif_abs_z"] > float(Z_THR)).astype(int)
    df["dif_direction"] = np.where(df["dif_z"] > 0, f"{A}_favored", f"{B}_favored")

    thr_logits = float(Z_THR) * sd_resid
    n_out = int(df["dif_flag"].sum())

    # -------------------------
    # Bank-level summary (machine-readable)
    # -------------------------
    summary_df = pd.DataFrame([{
        "group_a_label": GROUP_A_LABEL,
        "group_b_label": GROUP_B_LABEL,
        "group_a_file": GROUP_A_ITEMS_CSV,
        "group_b_file": GROUP_B_ITEMS_CSV,
        "overlap_items_n": int(len(df)),
        "fit_a": float(a),
        "fit_c": float(c),
        "fit_r2": float(fit_r2),
        "pearson_r": float(pearson_r),
        "spearman_r": float(spearman_r),
        "resid_sd": float(sd_resid),
        "z_thr": float(Z_THR),
        "thr_logits": float(thr_logits),
        "outliers_n": int(n_out),
    }])

    summary_path = None
    pairs_path = None

    if OUT_DIR is not None:
        summary_path = os.path.join(OUT_DIR, "dif_bank_summary.csv")
        summary_df.to_csv(summary_path, index=False)

        pairs_path = os.path.join(OUT_DIR, "dif_pairs.csv")
        df.sort_values("dif_abs_z", ascending=False).to_csv(pairs_path, index=False)

        if verbose:
            print(f"[OK] Wrote summary    : {summary_path}")
            print(f"[OK] Wrote pairs table: {pairs_path}")

    # -------------------------
    # Merge item-level DIF diagnostics into MASTER_ITEMS_CSV (optional)
    # -------------------------
    master_path = None

    if MASTER_ITEMS_CSV is not None:
        master = pd.read_csv(MASTER_ITEMS_CSV, dtype={WORD_COL: str})
        master[WORD_COL] = master[WORD_COL].astype(str).str.strip()

        # Drop old dif_* columns (safe re-runs)
        master = master[[c for c in master.columns if not any(c.startswith(p) for p in DIAGNOSTIC_PREFIXES)]]

        # One row per overlapping item
        item_out = pd.DataFrame({
            WORD_COL: df[WORD_COL].astype(str),
            f"dif_b_{A}": df[f"b_{A}"].astype(float),
            f"dif_b_{B}": df[f"b_{B}"].astype(float),
            f"dif_pred_{B}_from_{A}": df[pred_col].astype(float),
            "dif_resid_logits": df["resid"].astype(float),
            "dif_z": df["dif_z"].astype(float),
            "dif_abs_z": df["dif_abs_z"].astype(float),
            "dif_flag": df["dif_flag"].astype(int),
            "dif_direction": df["dif_direction"].astype(str),
            "dif_overlap_in_pairfile": 1,
        })

        merged = master.merge(item_out, on=WORD_COL, how="left")
        merged["dif_overlap_in_pairfile"] = merged["dif_overlap_in_pairfile"].fillna(0).astype(int)

        merged.to_csv(MASTER_ITEMS_CSV, index=False)
        master_path = MASTER_ITEMS_CSV

        if verbose:
            print(f"[OK] Updated master item file: {MASTER_ITEMS_CSV}")

    # -------------------------
    # Console overview (at-a-glance)
    # -------------------------
    if verbose:
        print("\n====================== DIF OVERVIEW ======================")
        print(f"Group A label                 : {GROUP_A_LABEL}")
        print(f"Group B label                 : {GROUP_B_LABEL}")
        print(f"Overlapping items (A ∩ B)      : {len(df)}")
        print(f"Scale alignment               : b_{B} = {a:.4f} * b_{A} + {c:.4f}")
        print(f"Fit R² (alignment)           : {fit_r2:.4f}")
        print(f"Pearson r (b_A vs b_B)       : {pearson_r:.4f}")
        print(f"Spearman ρ (b_A vs b_B)      : {spearman_r:.4f}")
        print(f"Residual SD (logits)          : {sd_resid:.4f}")
        print(f"DIF band threshold (logits)   : ±{thr_logits:.4f}  (Z_THR={Z_THR})")
        print(f"Flagged outliers (|z|>Z_THR)  : {n_out}  ({(n_out/len(df)*100):.2f}% of overlap)")
        print("-" * 58)

        if n_out > 0:
            df_out = df[df["dif_flag"] == 1]
            print("Outlier directions (count):")
            for k, v in df_out["dif_direction"].value_counts().items():
                print(f"  {k:>20s} : {int(v)}")

            print("-" * 58)
            print("Top 10 DIF outliers (by |z|):")
            top10 = (
                df_out.sort_values("dif_abs_z", ascending=False)
                     .head(10)[[WORD_COL, f"b_{A}", f"b_{B}", pred_col, "resid", "dif_z", "dif_direction"]]
            )
            print(top10.to_string(index=False, justify="left", float_format=lambda v: f"{v: .3f}"))
        else:
            print("No outliers at this Z_THR threshold.")

        print("-" * 58)
        print("Files written:")
        if pairs_path is not None:
            print(f"  Pair table   : {pairs_path}")
        if summary_path is not None:
            print(f"  Bank summary : {summary_path}")
        if master_path is not None:
            print(f"  Bank updated : {master_path}")
        print("===========================================================\n")

    return {
        "df_overlap": df,
        "summary_df": summary_df,
        "a": float(a),
        "c": float(c),
        "fit_r2": float(fit_r2),
        "pearson_r": float(pearson_r),
        "spearman_r": float(spearman_r),
        "sd_resid": float(sd_resid),
        "thr_logits": float(thr_logits),
        "outliers_n": int(n_out),
        "summary_path": summary_path,
        "pairs_path": pairs_path,
        "master_path": master_path,
    }


def plot_dif(
    df: pd.DataFrame,                           # output df from dif(): res["df_overlap"]
    a: float,                                   # alignment slope from dif()
    c: float,                                   # alignment intercept from dif()
    sd_resid: float,                            # residual SD from dif()
    GROUP_A_LABEL: str,                         # e.g., "learners"
    GROUP_B_LABEL: str,                         # e.g., "native speakers"
    GROUP_A_LEGEND: str = "learner-favored",    # legend label for z>0 outliers
    GROUP_B_LEGEND: str = "native-favored",     # legend label for z<0 outliers
    Z_THR: float = 2.0,                         # same Z_THR used in dif()
    SAVE_PATH: str | None = None,               # if provided -> save PNG
    show: bool = False,                         # if True -> plt.show()
    # ---- column names (defaults match dif() output) ----
    WORD_COL: str = "itemID",                  # item ID column in df
    B_A_COL: str | None = None,                # if None -> auto-detect b_<safe(A)>
    B_B_COL: str | None = None,                # if None -> auto-detect b_<safe(B)>
    DIF_Z_COL: str = "dif_z",
    DIF_FLAG_COL: str = "dif_flag",
    # ---- plot style (defaults = your current script) ----
    plot_params: dict | None = None,
    xlim: tuple[float, float] = (-5, 12),
    ylim: tuple[float, float] = (-11, 8),
    tick_step: float = 3,
    inlier_marker_size: float = 6,
    outlier_marker_size: float = 8,
    inlier_alpha: float = 0.4,
    outlier_alpha: float = 0.95,
    fit_line_width: float = 2,
    band_line_width: float = 1,
    grid: bool = True,
    grid_alpha: float = 0.4,
    grid_lw: float = 0.5,
    # ---- legend ----
    legend_loc: str = "upper left",
    legend_frameon: bool = True,
    legend_framealpha: float = 0.9,
    legend_handletextpad: float = 0.3,
    # ---- annotation (distance-filtered, by |dif_z|, flagged items only) ----
    annotate: bool = True,
    annotate_only_flagged: bool = True,
    max_labels: int | None = None,
    min_dist: float = 1.174,
    dxdy_pos: tuple[float, float] = (-2, 2),
    dxdy_neg: tuple[float, float] = (2, -2),
    annotation_font_size: float = 8,
    italicize_words: bool = True,
    bbox_kw: dict | None = None,               # default None (as in your current figure)
    # ---- saving ----
    save_dpi: int = 300,
) -> None:
    """
    Plot DIF scatter:
      x = b_A, y = b_B
      line: y = a*x + c
      bands: ± Z_THR * sd_resid (parallel to line)
      outliers colored by dif_z sign among dif_flag==1 items
      annotations: greedy distance filter in data units (logits)

    Saves to SAVE_PATH if provided. Shows if show=True.
    """

    # ---- defaults for rcParams ----
    if plot_params is None:
        plot_params = {
            "figure.figsize": (3.3, 2.8),   # one column figure size
            "figure.dpi": 150,
            "font.size": 8,
            "axes.labelsize": 8,
            "axes.titlesize": 8,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 7,
        }

    # ---- auto-detect b columns if not provided ----
    # Expected in df: b_<safe_label(GROUP_A_LABEL)>, b_<safe_label(GROUP_B_LABEL)>

    if B_A_COL is None:
        B_A_COL = f"b_{_safe_label(GROUP_A_LABEL)}"
    if B_B_COL is None:
        B_B_COL = f"b_{_safe_label(GROUP_B_LABEL)}"

    if B_A_COL not in df.columns or B_B_COL not in df.columns:
        raise KeyError(
            f"Could not find difficulty columns in df. "
            f"Expected '{B_A_COL}' and '{B_B_COL}'. "
            f"Available columns: {list(df.columns)[:20]}..."
        )

    thr_logits = float(Z_THR) * float(sd_resid)

    x = df[B_A_COL].to_numpy(dtype=float)
    y = df[B_B_COL].to_numpy(dtype=float)

    dif_flag = df[DIF_FLAG_COL].to_numpy(dtype=int) if DIF_FLAG_COL in df.columns else np.zeros(len(df), dtype=int)
    dif_z = df[DIF_Z_COL].to_numpy(dtype=float) if DIF_Z_COL in df.columns else np.zeros(len(df), dtype=float)

    inl = (dif_flag == 0)   # inlier items
    out = ~inl              # outlier items
    pos = out & (dif_z > 0) # outlier items favored by group A
    neg = out & (dif_z < 0) # outlier items favored by group B

    with plt.rc_context(plot_params):
        fig, ax = plt.subplots()

        # scatter - inliers
        ax.scatter(
            x[inl], y[inl],
            alpha=inlier_alpha, s=inlier_marker_size, color="gray",
            zorder=1, label="non-flagged items",
        )
        # scatter - outliers group A
        ax.scatter(
            x[pos], y[pos],
            alpha=outlier_alpha, s=outlier_marker_size, color="orange",
            zorder=3, label=GROUP_A_LEGEND,
        )
        # scatter - outliers group B
        ax.scatter(
            x[neg], y[neg],
            alpha=outlier_alpha, s=outlier_marker_size, color="blue",
            zorder=3, label=GROUP_B_LEGEND,
        )

        # fit line + bands
        xs = np.linspace(float(np.min(x)), float(np.max(x)), 200)
        y_line = a * xs + c

        ax.plot(xs, y_line, linewidth=fit_line_width, color="gray", zorder=2, label="alignment fit")
        ax.plot(xs, y_line + thr_logits, linestyle="--", linewidth=band_line_width, color="orange",
                label=f"+{int(Z_THR)} SD band")
        ax.plot(xs, y_line - thr_logits, linestyle="--", linewidth=band_line_width, color="blue",
                label=f"-{int(Z_THR)} SD band")

        # annotations (optional)
        if annotate and WORD_COL in df.columns and DIF_Z_COL in df.columns:
            cand = df.copy()
            if annotate_only_flagged and DIF_FLAG_COL in cand.columns:
                cand = cand[cand[DIF_FLAG_COL].astype(int) == 1].copy()

            cand = cand[np.isfinite(cand[DIF_Z_COL])].copy()
            if not cand.empty:
                cand["_abs_dif_z"] = cand[DIF_Z_COL].abs()
                cand = cand.sort_values("_abs_dif_z", ascending=False)

                chosen_xy: list[tuple[float, float]] = []
                n_labeled = 0

                for _, rrow in cand.iterrows():
                    x0 = float(rrow[B_A_COL])
                    y0 = float(rrow[B_B_COL])

                    if chosen_xy:
                        if any(((x0 - x1) ** 2 + (y0 - y1) ** 2) < (min_dist ** 2) for (x1, y1) in chosen_xy):
                            continue

                    chosen_xy.append((x0, y0))

                    if float(rrow[DIF_Z_COL]) > 0:
                        dx, dy = dxdy_pos
                        ha, va = "right", "bottom"
                    else:
                        dx, dy = dxdy_neg
                        ha, va = "left", "top"

                    ax.annotate(
                        str(rrow[WORD_COL]),
                        (x0, y0),
                        textcoords="offset points",
                        xytext=(dx, dy),
                        ha=ha,
                        va=va,
                        fontsize=annotation_font_size,
                        bbox=bbox_kw,
                        fontstyle=("italic" if italicize_words else "normal"),
                        zorder=5,
                    )

                    n_labeled += 1
                    if (max_labels is not None) and (n_labeled >= max_labels):
                        break

                cand.drop(columns=["_abs_dif_z"], inplace=True, errors="ignore")

        # labels + axes styling
        ax.set_xlabel(f"Item difficulty ({GROUP_A_LABEL}), logit")
        ax.set_ylabel(f"Item difficulty ({GROUP_B_LABEL}), logit")
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)

        ax.tick_params(direction="in")
        ax.xaxis.set_major_locator(MultipleLocator(tick_step))
        ax.yaxis.set_major_locator(MultipleLocator(tick_step))

        if grid:
            ax.grid(True, linewidth=grid_lw, alpha=grid_alpha)

        ax.legend(
            loc=legend_loc,
            frameon=legend_frameon,
            framealpha=legend_framealpha,
            handletextpad=legend_handletextpad,
        )

        fig.tight_layout()

        if SAVE_PATH is not None:
            fig.savefig(SAVE_PATH, dpi=save_dpi, bbox_inches="tight")

        if show:
            plt.show()

        plt.close(fig)


if __name__ == "__main__":

    Z_THR = 2

    res = dif(
        GROUP_A_ITEMS_CSV='/Users/grigorygolovin/Library/CloudStorage/OneDrive-Personal/Projects/word stock estimation/MyVocab stats/items_enlearners.csv',
        GROUP_B_ITEMS_CSV='/Users/grigorygolovin/Library/CloudStorage/OneDrive-Personal/Projects/word stock estimation/MyVocab stats/items_ennatives.csv',
        GROUP_A_LABEL='learners',
        GROUP_B_LABEL='native speakers',
        Z_THR=Z_THR,
        OUT_DIR='/Users/grigorygolovin/Library/CloudStorage/OneDrive-Personal/Projects/word stock estimation/MyVocab stats/dif_out',
        MASTER_ITEMS_CSV='/Users/grigorygolovin/Library/CloudStorage/OneDrive-Personal/Projects/word stock estimation/MyVocab stats/items_en.csv',
    )

    plot_dif(
        df=res["df_overlap"],
        a=res["a"],
        c=res["c"],
        sd_resid=res["sd_resid"],
        Z_THR=Z_THR,
        GROUP_A_LABEL="learners",
        GROUP_B_LABEL="native speakers",
        GROUP_A_LEGEND="learner-favored",
        GROUP_B_LEGEND="native-favored",
        # SAVE_PATH="/path/to/dif_scatter_aligned.png",
        show=True,
    )


# ====================================================================
# INTERPRETATION GUIDANCE (ITEM-LEVEL + BANK-LEVEL)
# ====================================================================
#
# ITEM-LEVEL (in MASTER_ITEMS_CSV)
# -------------------------------
# dif_b_<A> / dif_b_<B>
#   Group-specific calibrated difficulties for the overlapping items.
#
# dif_pred_<B>_from_<A>
#   Predicted Group-B difficulty after aligning the scales:
#       pred_B = a*b_A + c
#   where a,c are estimated on all overlapping items (simple linear fit).
#
# dif_resid_logits
#   Raw DIF residual in logits:
#       resid = b_B - pred_B
#   Positive => item harder for Group B than expected given Group A.
#   Negative => item harder for Group A than expected.
#
# dif_z
#   Standardized DIF residual:
#       z = resid / SD(resid)
#   Comparable across runs; used for flagging.
#
# dif_flag
#   1 if |dif_z| > Z_THR (screening threshold), else 0.
#
# dif_direction
#   Human-readable direction:
#     dif_z > 0 => "A_favored" (item relatively easier for A, harder for B)
#     dif_z < 0 => "B_favored" (item relatively easier for B, harder for A)
#
# dif_overlap_in_pairfile
#   1 if item existed in BOTH group files (i.e., DIF computed), else 0.
#
# Practical use:
#   - Start by sorting items by dif_abs_z (or look at top rows of dif_pairs.csv).
#   - For flagged items, inspect content: cognates, culture knowledge, polysemy,
#     ambiguous acceptability, register, or item-type artifacts.
#   - Cross-check with q3/aQ3 and PCAR: items can show DIF *and* local dependence.
#
# BANK-LEVEL (in OUT_DIR/dif_bank_summary.csv + console)
# ------------------------------------------------------
# overlap_items_n
#   How many items were comparable across groups (intersection).
#
# fit_a / fit_c
#   Linear alignment parameters. Strong deviations from a≈1 or large |c|
#   may indicate scale spread differences between calibrations.
#
# fit_r2
#   R² of the linear alignment b_B ~ a*b_A + c over the overlapping items.
#   Higher values indicate a tighter linear relationship (better alignment).
#
# pearson_r / spearman_r
#   Association between the *raw* difficulty estimates from the two groups
#   (b_A vs b_B) over the overlap set:
#     - Pearson r captures linear association
#     - Spearman ρ captures rank/monotonic association
#   These are complementary to R²: strong correlations can coexist with non-trivial
#   DIF scatter (captured by resid_sd and outlier counts).
#
# resid_sd
#   Overall spread of DIF residuals after alignment (logit units).
#   Larger values => more cross-group item instability.
#
# outliers_n (|z| > Z_THR)
#   How many items show unusually large DIF under this SD-based screen.
#
# Recommended reporting language (screening DIF):
#   "Cross-group item stability was screened by aligning difficulty scales with a
#    linear transformation and examining standardized residuals. Items with
#    |z| > Z_THR were flagged for review as potential DIF candidates."
#
# IMPORTANT LIMITATION
# --------------------
# This is a *screening* method. It does not replace formal DIF analysis
# (e.g., IRT-LR DIF, Mantel–Haenszel / logistic regression DIF, DIF with anchors).
# Use it to prioritize items for deeper investigation.
# ====================================================================
