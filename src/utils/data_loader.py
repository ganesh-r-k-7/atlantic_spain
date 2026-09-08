"""
data_loader.py — Central Analytics Orchestrator
Atlantic Recording Corporation | Spain Top 50 Analytics

Run this once to precompute all derived datasets used by the Streamlit dashboard.
"""

import os
import sys
import pandas as pd

# ── Make src importable ──────────────────────────────────────────────────────
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

from src.preprocessing.cleaner import run_cleaning_pipeline
from src.analytics.lifecycle import (
    build_song_lifecycle,
    classify_daily_stage,
    build_stage_transition_matrix,
    lifecycle_journey_map,
    compute_daily_churn,
    monthly_rotation_summary,
)
from src.analytics.content_analysis import (
    explicit_vs_clean_comparison,
    single_vs_album_comparison,
    duration_correlation_analysis,
    album_size_analysis,
    build_survival_data,
    run_kaplan_meier,
    log_rank_test,
    compute_popularity_trajectories,
    popularity_acceleration,
    cohort_popularity_by_stage,
)


def load_all(data_path: str = None) -> dict:
    """
    Master loader: clean data, compute all analytics, return a dict of DataFrames.

    Returns
    -------
    dict with keys:
        df          — fully cleaned daily observations
        lc          — song lifecycle metrics (one row per song)
        df_staged   — daily observations with lifecycle stage labels
        churn       — daily churn / retention metrics
        monthly     — monthly churn summary
        surv        — survival analysis input
        km_explicit — Kaplan-Meier results by explicit flag
        km_album    — Kaplan-Meier results by album type
        lr_explicit — log-rank test result (explicit vs clean)
        lr_album    — log-rank test result (single vs album)
        transitions — stage transition probability matrix
        journey     — lifecycle journey map per song
        traj        — popularity trajectories (normalized time)
        pop_accel   — daily observations with popularity acceleration
        stage_pop   — popularity by lifecycle stage
        explicit_comp — explicit vs clean comparison stats
        sa_comp     — single vs album comparison stats
        dur_corr    — duration correlation analysis
        album_size  — album size (total_tracks) correlation
    """
    if data_path is None:
        data_path = os.path.join(ROOT, "data", "Atlantic_Spain.csv")

    # ── 1. Clean ──────────────────────────────────────────────────────────
    df = run_cleaning_pipeline(data_path)

    # ── 2. Lifecycle metrics ───────────────────────────────────────────────
    lc = build_song_lifecycle(df)

    # ── 3. Daily stage labels ─────────────────────────────────────────────
    df_staged = classify_daily_stage(df)

    # ── 4. Rotation / churn ───────────────────────────────────────────────
    churn = compute_daily_churn(df)
    monthly = monthly_rotation_summary(churn)

    # ── 5. Survival analysis ──────────────────────────────────────────────
    surv = build_survival_data(lc)
    km_explicit = run_kaplan_meier(surv, "content_type")
    km_album = run_kaplan_meier(surv, "release_type")
    lr_explicit = log_rank_test(surv, "is_explicit")
    lr_album = log_rank_test(surv, "album_type")

    # ── 6. Stage analytics ────────────────────────────────────────────────
    transitions = build_stage_transition_matrix(df_staged)
    journey = lifecycle_journey_map(df_staged)

    # ── 7. Popularity analytics ───────────────────────────────────────────
    traj = compute_popularity_trajectories(df_staged)
    pop_accel = popularity_acceleration(df_staged)
    stage_pop = cohort_popularity_by_stage(df_staged)

    # ── 8. Content attribute analysis ─────────────────────────────────────
    explicit_comp = explicit_vs_clean_comparison(lc)
    sa_comp = single_vs_album_comparison(lc)
    dur_corr = duration_correlation_analysis(lc)
    album_size = album_size_analysis(lc)

    return {
        "df": df,
        "lc": lc.reset_index(),
        "df_staged": df_staged,
        "churn": churn,
        "monthly": monthly,
        "surv": surv.reset_index(),
        "km_explicit": km_explicit,
        "km_album": km_album,
        "lr_explicit": lr_explicit,
        "lr_album": lr_album,
        "transitions": transitions,
        "journey": journey,
        "traj": traj,
        "pop_accel": pop_accel,
        "stage_pop": stage_pop,
        "explicit_comp": explicit_comp,
        "sa_comp": sa_comp,
        "dur_corr": dur_corr,
        "album_size": album_size,
    }


if __name__ == "__main__":
    data = load_all()
    print("=== Dataset Summary ===")
    print(f"  Clean rows     : {len(data['df']):,}")
    print(f"  Unique songs   : {data['lc']['song_key'].nunique()}")
    print(f"  Date range     : {data['df']['date'].min().date()} → {data['df']['date'].max().date()}")
    print(f"  Lifecycle rows : {len(data['lc'])}")
    print(f"  Churn rows     : {len(data['churn'])}")
    print("\n=== Top 10 Songs by Chart Days ===")
    print(data['lc'].nlargest(10, 'chart_days')[['song', 'artist', 'chart_days', 'peak_position']].to_string(index=False))
    print("\n=== Survival Medians ===")
    for grp, info in data['km_explicit'].items():
        print(f"  {grp}: median {info['median_survival']} days")
