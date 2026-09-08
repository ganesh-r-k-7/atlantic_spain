"""
content_analysis.py — Content Attribute & Survival Analysis
Atlantic Recording Corporation | Spain Top 50 Analytics
"""

import pandas as pd
import numpy as np
import logging
from scipy import stats

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────
#  SECTION 6 — CONTENT ATTRIBUTE ANALYSIS
# ─────────────────────────────────────────────

def explicit_vs_clean_comparison(lc: pd.DataFrame) -> dict:
    """
    Compare lifecycle metrics between explicit and clean content.
    Returns dict with descriptive stats and t-test results.
    """
    explicit = lc[lc["is_explicit"] == True]
    clean = lc[lc["is_explicit"] == False]

    results = {}
    for metric in ["chart_days", "peak_position", "avg_rank", "rank_volatility",
                   "popularity_growth", "longest_consecutive_run", "reentry_count"]:
        e_vals = explicit[metric].dropna()
        c_vals = clean[metric].dropna()
        t_stat, p_val = stats.ttest_ind(e_vals, c_vals, equal_var=False)
        results[metric] = {
            "explicit_mean": round(e_vals.mean(), 3),
            "clean_mean": round(c_vals.mean(), 3),
            "explicit_median": round(e_vals.median(), 3),
            "clean_median": round(c_vals.median(), 3),
            "t_statistic": round(t_stat, 4),
            "p_value": round(p_val, 6),
            "significant": p_val < 0.05,
        }
    return results


def single_vs_album_comparison(lc: pd.DataFrame) -> dict:
    """Compare lifecycle metrics between singles and album tracks."""
    singles = lc[lc["album_type"] == "single"]
    albums = lc[lc["album_type"] == "album"]

    results = {}
    for metric in ["chart_days", "peak_position", "avg_rank", "rank_volatility",
                   "days_to_peak", "reentry_count", "longest_consecutive_run"]:
        s_vals = singles[metric].dropna()
        a_vals = albums[metric].dropna()
        if len(s_vals) < 2 or len(a_vals) < 2:
            continue
        t_stat, p_val = stats.ttest_ind(s_vals, a_vals, equal_var=False)
        results[metric] = {
            "single_mean": round(s_vals.mean(), 3),
            "album_mean": round(a_vals.mean(), 3),
            "single_median": round(s_vals.median(), 3),
            "album_median": round(a_vals.median(), 3),
            "t_statistic": round(t_stat, 4),
            "p_value": round(p_val, 6),
            "significant": p_val < 0.05,
        }
    return results


def duration_correlation_analysis(lc: pd.DataFrame) -> pd.DataFrame:
    """Pearson correlation between duration and key lifecycle metrics."""
    target_cols = ["chart_days", "peak_position", "avg_rank",
                   "rank_volatility", "popularity_growth", "reentry_count"]
    rows = []
    for col in target_cols:
        valid = lc[["duration_min", col]].dropna()
        if len(valid) < 10:
            continue
        r, p = stats.pearsonr(valid["duration_min"], valid[col])
        rows.append({
            "metric": col,
            "pearson_r": round(r, 4),
            "p_value": round(p, 6),
            "significant": p < 0.05,
        })
    return pd.DataFrame(rows)


def album_size_analysis(lc: pd.DataFrame) -> pd.DataFrame:
    """Spearman correlation between total_tracks and lifecycle stability."""
    target_cols = ["chart_days", "rank_volatility", "longest_consecutive_run",
                   "reentry_count", "avg_rank"]
    rows = []
    for col in target_cols:
        valid = lc[["total_tracks", col]].dropna()
        if len(valid) < 10:
            continue
        r, p = stats.spearmanr(valid["total_tracks"], valid[col])
        rows.append({
            "metric": col,
            "spearman_r": round(r, 4),
            "p_value": round(p, 6),
            "significant": p < 0.05,
        })
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────
#  SECTION 7 — SURVIVAL ANALYSIS
# ─────────────────────────────────────────────

def build_survival_data(lc: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare survival analysis input.

    In survival terms:
    - T  = chart_days (duration on chart)
    - E  = 1 (all songs eventually exited — no censoring in a closed window)
    """
    lc2 = lc.copy()
    if "content_type" not in lc2.columns:
        lc2["content_type"] = lc2["is_explicit"].map({True: "Explicit", False: "Clean"})
    if "release_type" not in lc2.columns:
        lc2["release_type"] = lc2["album_type"].str.title()
    surv = lc2[["chart_days", "is_explicit", "album_type",
                "content_type", "release_type"]].copy()
    surv = surv.rename(columns={"chart_days": "T"})
    surv["E"] = 1  # event = exit from chart
    surv = surv.dropna(subset=["T"])
    surv = surv[surv["T"] > 0]
    return surv


def run_kaplan_meier(surv_df: pd.DataFrame, group_col: str) -> dict:
    """
    Fit Kaplan-Meier survival curves for each level of group_col.

    Returns a dict mapping group label → (timeline, survival_prob arrays).
    """
    try:
        from lifelines import KaplanMeierFitter
    except ImportError:
        logger.error("lifelines not installed. Run: pip install lifelines")
        return {}

    results = {}
    for group, grp in surv_df.groupby(group_col):
        kmf = KaplanMeierFitter()
        kmf.fit(grp["T"], event_observed=grp["E"], label=str(group))
        results[str(group)] = {
            "timeline": kmf.timeline.tolist(),
            "survival_function": kmf.survival_function_[str(group)].tolist(),
            "median_survival": float(kmf.median_survival_time_),
            "n": len(grp),
        }
    return results


def log_rank_test(surv_df: pd.DataFrame, group_col: str) -> dict:
    """
    Perform log-rank test between two groups defined by group_col.
    Returns test statistic and p-value.
    """
    try:
        from lifelines.statistics import logrank_test
    except ImportError:
        return {}

    groups = surv_df[group_col].unique()
    if len(groups) != 2:
        return {"error": f"Expected 2 groups, found {len(groups)}"}

    g1 = surv_df[surv_df[group_col] == groups[0]]
    g2 = surv_df[surv_df[group_col] == groups[1]]
    result = logrank_test(g1["T"], g2["T"], g1["E"], g2["E"])
    return {
        "group1": str(groups[0]),
        "group2": str(groups[1]),
        "test_statistic": round(result.test_statistic, 4),
        "p_value": round(result.p_value, 6),
        "significant": result.p_value < 0.05,
    }


# ─────────────────────────────────────────────
#  SECTION 8 — POPULARITY MATURITY ANALYSIS
# ─────────────────────────────────────────────

def compute_popularity_trajectories(df: pd.DataFrame) -> pd.DataFrame:
    """
    For every song, normalize time-on-chart to [0, 1] and return
    popularity at each normalized time step. Useful for cohort overlays.
    """
    rows = []
    for song_key, grp in df.groupby("song_key"):
        grp = grp.sort_values("date").reset_index(drop=True)
        n = len(grp)
        if n < 3:
            continue
        grp["norm_time"] = np.linspace(0, 1, n)
        grp["song_key"] = song_key
        rows.append(grp[["song_key", "song", "artist", "norm_time",
                          "popularity", "position", "lifecycle_stage"
                          if "lifecycle_stage" in grp.columns else "position"]])
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def popularity_acceleration(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute daily popularity change (first derivative) per song.
    Positive = gaining momentum; Negative = losing momentum.
    """
    df = df.copy().sort_values(["song_key", "date"])
    df["pop_change"] = df.groupby("song_key")["popularity"].diff()
    df["pop_accel"] = df.groupby("song_key")["pop_change"].diff()
    return df


def cohort_popularity_by_stage(df_staged: pd.DataFrame) -> pd.DataFrame:
    """
    Average popularity by lifecycle stage — useful for bar/violin plots.
    """
    return (
        df_staged.groupby("lifecycle_stage")["popularity"]
        .agg(["mean", "median", "std", "count"])
        .round(2)
        .reset_index()
        .rename(columns={"mean": "avg_popularity", "median": "med_popularity",
                          "std": "std_popularity", "count": "n_obs"})
    )
