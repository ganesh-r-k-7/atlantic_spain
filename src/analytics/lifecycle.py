"""
lifecycle.py — Song Lifecycle Engine & Stage Classification
Atlantic Recording Corporation | Spain Top 50 Analytics
"""

import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
#  SECTION 3 — LIFECYCLE METRICS ENGINE
# ─────────────────────────────────────────────

def build_song_lifecycle(df: pd.DataFrame) -> pd.DataFrame:
    """
    For each unique song, compute full lifecycle metrics.

    Returns a DataFrame indexed by song_key with columns:
    entry_date, exit_date, total_days, peak_position, days_to_peak,
    avg_rank, median_rank, rank_volatility, popularity_growth,
    popularity_decay, longest_consecutive_run, reentry_count, ...
    """
    records = []

    for song_key, grp in df.groupby("song_key"):
        grp = grp.sort_values("date").reset_index(drop=True)

        # — Basic metadata —
        song = grp["song"].iloc[0]
        artist = grp["artist"].iloc[0]
        album_type = grp["album_type"].iloc[0]
        is_explicit = grp["is_explicit"].iloc[0]
        content_type = grp["content_type"].iloc[0]
        duration_min = grp["duration_min"].median()
        total_tracks = grp["total_tracks"].median()

        # — Entry / Exit —
        entry_date = grp["date"].min()
        exit_date = grp["date"].max()
        total_days = (exit_date - entry_date).days + 1
        chart_days = len(grp)  # actual days on chart (may be < total_days if gaps)

        # — Rank metrics —
        peak_position = grp["position"].min()
        avg_rank = grp["position"].mean().round(2)
        median_rank = grp["position"].median()
        rank_volatility = grp["position"].std().round(2) if len(grp) > 1 else 0.0

        # — Days to peak —
        peak_idx = grp["position"].idxmin()
        days_to_peak = (grp.loc[peak_idx, "date"] - entry_date).days

        # — Popularity —
        first_pop = grp["popularity"].iloc[0]
        last_pop = grp["popularity"].iloc[-1]
        peak_pop = grp["popularity"].max()
        popularity_growth = peak_pop - first_pop
        popularity_decay = peak_pop - last_pop

        # — Consecutive run length —
        dates_set = set(grp["date"].dt.date)
        all_dates = pd.date_range(entry_date, exit_date, freq="D")
        max_run = current_run = 0
        for d in all_dates:
            if d.date() in dates_set:
                current_run += 1
                max_run = max(max_run, current_run)
            else:
                current_run = 0
        longest_consecutive_run = max_run

        # — Re-entry count —
        # Count gaps of ≥1 day in chart presence = number of re-entries
        all_chart_dates = sorted(grp["date"].dt.date)
        reentry_count = 0
        for i in range(1, len(all_chart_dates)):
            gap = (all_chart_dates[i] - all_chart_dates[i - 1]).days
            if gap > 1:
                reentry_count += 1

        # — Top10 / Top5 / Top1 days —
        days_top10 = (grp["position"] <= 10).sum()
        days_top5 = (grp["position"] <= 5).sum()
        days_at_one = (grp["position"] == 1).sum()

        # — Rank improvement slope (linear regression on position vs time) —
        if len(grp) > 2:
            x = np.arange(len(grp))
            slope = np.polyfit(x, grp["position"].values, 1)[0]
        else:
            slope = 0.0

        records.append({
            "song_key": song_key,
            "song": song,
            "artist": artist,
            "album_type": album_type,
            "is_explicit": is_explicit,
            "content_type": content_type,
            "duration_min": duration_min,
            "total_tracks": total_tracks,
            "entry_date": entry_date,
            "exit_date": exit_date,
            "total_days": total_days,
            "chart_days": chart_days,
            "peak_position": peak_position,
            "avg_rank": avg_rank,
            "median_rank": median_rank,
            "rank_volatility": rank_volatility,
            "days_to_peak": days_to_peak,
            "first_popularity": first_pop,
            "peak_popularity": peak_pop,
            "last_popularity": last_pop,
            "popularity_growth": popularity_growth,
            "popularity_decay": popularity_decay,
            "longest_consecutive_run": longest_consecutive_run,
            "reentry_count": reentry_count,
            "days_top10": days_top10,
            "days_top5": days_top5,
            "days_at_one": days_at_one,
            "rank_trend_slope": round(slope, 4),
        })

    lc = pd.DataFrame(records).set_index("song_key")
    logger.info(f"Lifecycle engine produced metrics for {len(lc)} songs")
    return lc


# ─────────────────────────────────────────────
#  SECTION 4 — LIFECYCLE STAGE CLASSIFICATION
# ─────────────────────────────────────────────

def classify_daily_stage(df: pd.DataFrame, window: int = 5) -> pd.DataFrame:
    """
    Assign a lifecycle stage to every (song, date) observation.

    Stages
    ------
    1 New Entry   — first 3 days on chart
    2 Growth      — rank improving (rolling avg falling) & not at peak
    3 Peak        — around peak position (within 3 rank points)
    4 Mature      — stable rank after peak, low volatility
    5 Decline     — rank worsening (rolling avg rising) after peak
    """
    df = df.copy().sort_values(["song_key", "date"])
    stage_col = []

    for song_key, grp in df.groupby("song_key", sort=False):
        grp = grp.reset_index(drop=True)
        n = len(grp)
        peak_pos = grp["position"].min()
        peak_day_idx = grp["position"].idxmin()

        # Rolling mean of position (lower = better)
        roll_mean = grp["position"].rolling(window, min_periods=1, center=True).mean()

        stages = []
        for i, row in grp.iterrows():
            if i < 3:
                stage = "New Entry"
            elif abs(row["position"] - peak_pos) <= 3:
                stage = "Peak"
            elif i < peak_day_idx:
                # Before peak: is rank improving?
                delta = roll_mean.iloc[i] - roll_mean.iloc[max(0, i - 1)]
                stage = "Growth" if delta < 0 else "New Entry"
            else:
                # After peak
                volatility = grp["position"].iloc[max(0, i - window):i + 1].std()
                if volatility is not None and not np.isnan(volatility) and volatility < 3.5:
                    stage = "Mature"
                else:
                    stage = "Decline"
            stages.append(stage)

        stage_col.extend(stages)

    df["lifecycle_stage"] = stage_col
    return df


def build_stage_transition_matrix(df_staged: pd.DataFrame) -> pd.DataFrame:
    """
    Build a transition count matrix between lifecycle stages for consecutive
    daily observations of the same song.
    """
    stage_order = ["New Entry", "Growth", "Peak", "Mature", "Decline"]
    transitions = {s: {t: 0 for t in stage_order} for s in stage_order}

    for _, grp in df_staged.groupby("song_key"):
        stages = grp.sort_values("date")["lifecycle_stage"].tolist()
        for a, b in zip(stages[:-1], stages[1:]):
            if a in transitions and b in transitions[a]:
                transitions[a][b] += 1

    matrix = pd.DataFrame(transitions).T[stage_order].loc[stage_order]
    # Normalize to row probabilities
    matrix_pct = matrix.div(matrix.sum(axis=1), axis=0).fillna(0).round(3)
    return matrix_pct


def lifecycle_journey_map(df_staged: pd.DataFrame) -> pd.DataFrame:
    """
    For each song, summarize the ordered stage sequence and time spent in each.
    """
    rows = []
    stage_order = ["New Entry", "Growth", "Peak", "Mature", "Decline"]
    for song_key, grp in df_staged.groupby("song_key"):
        grp = grp.sort_values("date")
        stage_counts = grp["lifecycle_stage"].value_counts()
        journey = " → ".join(
            grp["lifecycle_stage"].drop_duplicates().tolist()
        )
        row = {"song_key": song_key, "journey": journey}
        for s in stage_order:
            row[f"days_{s.lower().replace(' ', '_')}"] = stage_counts.get(s, 0)
        rows.append(row)
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────
#  SECTION 5 — PLAYLIST ROTATION & CHURN
# ─────────────────────────────────────────────

def compute_daily_churn(df: pd.DataFrame) -> pd.DataFrame:
    """
    For each consecutive date pair, compute:
    - entries  : songs new to the chart
    - exits    : songs that left the chart
    - retained : songs present on both dates
    - churn_rate : exits / previous_chart_size
    - retention_rate : retained / previous_chart_size
    - stability_index : retained / 50
    """
    dates = sorted(df["date"].unique())
    records = []
    for prev_d, curr_d in zip(dates[:-1], dates[1:]):
        prev_songs = set(df[df["date"] == prev_d]["song_key"])
        curr_songs = set(df[df["date"] == curr_d]["song_key"])
        retained = prev_songs & curr_songs
        exits = prev_songs - curr_songs
        entries = curr_songs - prev_songs
        n_prev = len(prev_songs)
        churn_rate = len(exits) / n_prev if n_prev else 0
        retention_rate = len(retained) / n_prev if n_prev else 0
        records.append({
            "date": curr_d,
            "entries": len(entries),
            "exits": len(exits),
            "retained": len(retained),
            "chart_size": len(curr_songs),
            "churn_rate": round(churn_rate, 4),
            "retention_rate": round(retention_rate, 4),
            "stability_index": round(len(retained) / 50, 4),
        })
    return pd.DataFrame(records)


def monthly_rotation_summary(churn_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate churn metrics by calendar month."""
    churn_df = churn_df.copy()
    churn_df["month"] = pd.to_datetime(churn_df["date"]).dt.to_period("M")
    monthly = churn_df.groupby("month").agg(
        avg_entries=("entries", "mean"),
        avg_exits=("exits", "mean"),
        avg_churn_rate=("churn_rate", "mean"),
        avg_retention_rate=("retention_rate", "mean"),
        avg_stability_index=("stability_index", "mean"),
        total_unique_entries=("entries", "sum"),
    ).round(4).reset_index()
    monthly["month"] = monthly["month"].astype(str)
    return monthly
