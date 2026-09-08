"""
cleaner.py — Data Cleaning & Validation Pipeline
Atlantic Recording Corporation | Spain Top 50 Analytics
"""

import pandas as pd
import numpy as np
import re
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
#  LOAD
# ─────────────────────────────────────────────

def load_raw(path: str) -> pd.DataFrame:
    """Load raw CSV and perform initial type coercions."""
    df = pd.read_csv(path)
    logger.info(f"Loaded {len(df):,} rows from {path}")
    return df


# ─────────────────────────────────────────────
#  DATE PARSING
# ─────────────────────────────────────────────

def parse_dates(df: pd.DataFrame) -> pd.DataFrame:
    """Robustly parse the 'date' column to datetime."""
    df = df.copy()
    for fmt in ("%d-%m-%Y", "%Y-%m-%d", "%m/%d/%Y", "%d/%m/%Y"):
        try:
            df["date"] = pd.to_datetime(df["date"], format=fmt)
            logger.info(f"Dates parsed with format {fmt}")
            return df
        except (ValueError, TypeError):
            continue
    df["date"] = pd.to_datetime(df["date"], dayfirst=True, infer_datetime_format=True)
    logger.info("Dates parsed with inferred format")
    return df


# ─────────────────────────────────────────────
#  COLUMN TYPES
# ─────────────────────────────────────────────

def cast_types(df: pd.DataFrame) -> pd.DataFrame:
    """Enforce correct dtypes on all columns."""
    df = df.copy()
    df["position"] = pd.to_numeric(df["position"], errors="coerce").astype("Int64")
    df["popularity"] = pd.to_numeric(df["popularity"], errors="coerce").astype("Int64")
    df["duration_ms"] = pd.to_numeric(df["duration_ms"], errors="coerce").astype("Int64")
    df["total_tracks"] = pd.to_numeric(df["total_tracks"], errors="coerce").astype("Int64")
    # is_explicit: coerce to bool
    if df["is_explicit"].dtype != bool:
        df["is_explicit"] = df["is_explicit"].map(
            {"True": True, "False": False, True: True, False: False}
        ).fillna(False).astype(bool)
    df["album_type"] = df["album_type"].str.strip().str.lower()
    return df


# ─────────────────────────────────────────────
#  STRING NORMALIZATION
# ─────────────────────────────────────────────

def _normalize_string(s: str) -> str:
    """Strip extra whitespace and title-case a string."""
    if not isinstance(s, str):
        return ""
    return re.sub(r"\s+", " ", s).strip()


def normalize_text_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize song and artist name columns."""
    df = df.copy()
    df["song"] = df["song"].apply(_normalize_string)
    df["artist"] = df["artist"].apply(_normalize_string)
    # Canonical key for deduplication / grouping
    df["song_key"] = df["song"].str.lower().str.strip()
    df["artist_key"] = df["artist"].str.lower().str.strip()
    return df


# ─────────────────────────────────────────────
#  MISSING VALUE HANDLING
# ─────────────────────────────────────────────

def handle_missing(df: pd.DataFrame) -> pd.DataFrame:
    """
    Drop rows with critical nulls; forward-fill popularity
    within each song's date-sorted sequence.
    """
    df = df.copy()
    critical = ["date", "position", "song", "artist"]
    before = len(df)
    df.dropna(subset=critical, inplace=True)
    dropped = before - len(df)
    if dropped:
        logger.warning(f"Dropped {dropped} rows with null critical fields")

    # Forward-fill popularity within each song
    df = df.sort_values(["song_key", "date"])
    df["popularity"] = df.groupby("song_key")["popularity"].transform(
        lambda x: x.ffill().bfill()
    )
    return df


# ─────────────────────────────────────────────
#  DUPLICATE DETECTION
# ─────────────────────────────────────────────

def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Remove exact duplicate (date, position, song) rows."""
    before = len(df)
    df = df.drop_duplicates(subset=["date", "position", "song_key"])
    removed = before - len(df)
    if removed:
        logger.warning(f"Removed {removed} duplicate rows")
    return df


# ─────────────────────────────────────────────
#  RANK VALIDATION
# ─────────────────────────────────────────────

def validate_ranks(df: pd.DataFrame) -> pd.DataFrame:
    """
    Flag and remove days where the chart has invalid rank coverage.
    Valid: positions 1-50 with at most 50 entries per day.
    """
    invalid_pos = ~df["position"].between(1, 50)
    if invalid_pos.any():
        logger.warning(f"Removing {invalid_pos.sum()} rows with position outside 1-50")
        df = df[~invalid_pos]

    daily_counts = df.groupby("date")["position"].count()
    over_50 = daily_counts[daily_counts > 50].index
    if len(over_50):
        logger.warning(f"{len(over_50)} dates have >50 entries — keeping top 50 by position")
        # Keep only the top-50-ranked entry per (date, position)
        df = df.sort_values(["date", "position"]).groupby("date").head(50)
    return df


# ─────────────────────────────────────────────
#  OUTLIER HANDLING
# ─────────────────────────────────────────────

def clip_outliers(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clamp popularity to [0, 100] and duration_ms to reasonable bounds
    (15 s – 20 min). Duration values outside these bounds are set to NaN.
    """
    df = df.copy()
    df["popularity"] = df["popularity"].clip(0, 100)
    mask_dur = df["duration_ms"].between(15_000, 1_200_000)
    invalid_dur = (~mask_dur).sum()
    if invalid_dur:
        logger.warning(f"Setting {invalid_dur} out-of-range duration values to NaN")
        df.loc[~mask_dur, "duration_ms"] = pd.NA
    return df


# ─────────────────────────────────────────────
#  DERIVED COLUMNS
# ─────────────────────────────────────────────

def add_derived_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Add calendar and convenience columns."""
    df = df.copy()
    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month
    df["month_name"] = df["date"].dt.strftime("%b")
    df["week"] = df["date"].dt.isocalendar().week.astype(int)
    df["day_of_week"] = df["date"].dt.day_name()
    df["duration_sec"] = (df["duration_ms"] / 1000).round(1)
    df["duration_min"] = (df["duration_ms"] / 60_000).round(2)
    df["content_type"] = df["is_explicit"].map({True: "Explicit", False: "Clean"})
    df["release_type"] = df["album_type"].str.title()
    # Inverse rank score: 50 for #1, 1 for #50
    df["rank_score"] = 51 - df["position"]
    return df


# ─────────────────────────────────────────────
#  MASTER PIPELINE
# ─────────────────────────────────────────────

def run_cleaning_pipeline(path: str) -> pd.DataFrame:
    """
    Execute the full cleaning pipeline and return a clean DataFrame.

    Parameters
    ----------
    path : str
        Path to the raw CSV file.

    Returns
    -------
    pd.DataFrame
        Fully cleaned and feature-enriched dataset.
    """
    df = load_raw(path)
    df = parse_dates(df)
    df = cast_types(df)
    df = normalize_text_columns(df)
    df = handle_missing(df)
    df = remove_duplicates(df)
    df = validate_ranks(df)
    df = clip_outliers(df)
    df = add_derived_columns(df)
    df = df.sort_values(["date", "position"]).reset_index(drop=True)
    logger.info(f"Cleaning complete — {len(df):,} rows, {df['date'].nunique()} dates, "
                f"{df['song_key'].nunique()} unique songs")
    return df


if __name__ == "__main__":
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else "data/Atlantic_Spain.csv"
    clean = run_cleaning_pipeline(path)
    print(clean.head(10).to_string())
    print("\nShape:", clean.shape)
    print("Date range:", clean["date"].min().date(), "→", clean["date"].max().date())
