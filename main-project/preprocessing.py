"""
preprocessing.py
----------------
Load, clean, and merge the two datasets used in this version of the project:

1. anime.csv
   Critic-style rating data with title, release date, runtime, genre, and rating
2. popularity/*.csv
   MyAnimeList popularity snapshots with vote totals, favourites, and rank
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

from utilities import log


# Load the anime ratings database and rename its columns to project names.
def load_anime_database(filepath: str | Path) -> pd.DataFrame:
    """Load the anime ratings database and standardize column names."""
    log(f"Loading anime database: {filepath}")
    df = pd.read_csv(filepath)
    df = df.rename(
        columns={
            "Anime": "title",
            "Release_date": "release_date",
            "Length": "length",
            "Genre": "genre",
            "Rating": "critic_rating",
        }
    )

    df["critic_rating"] = pd.to_numeric(df["critic_rating"], errors="coerce")
    df["release_year"] = df["release_date"].apply(_extract_year)
    df["runtime_minutes"] = df["length"].apply(_extract_minutes)
    df["clean_title"] = df["title"].apply(normalize_title)
    return df


# Load all MyAnimeList popularity snapshots and summarize them to one row per title.
def load_popularity_database(popularity_dir: str | Path) -> pd.DataFrame:
    """Load MyAnimeList popularity files and summarize them by title."""
    popularity_dir = Path(popularity_dir)
    files = sorted(popularity_dir.glob("Votes_*.csv"))
    if not files:
        raise ValueError(f"No popularity files found in {popularity_dir}")

    log(f"Loading popularity snapshots from: {popularity_dir}")
    snapshots: list[pd.DataFrame] = []
    for path in files:
        snapshot = pd.read_csv(path)
        snapshot["snapshot_date"] = pd.to_datetime(path.stem.replace("Votes_", ""), format="%Y_%m_%d")
        snapshots.append(snapshot)

    all_votes = pd.concat(snapshots, ignore_index=True)
    all_votes["Votes"] = pd.to_numeric(all_votes["Votes"], errors="coerce").fillna(0)
    all_votes["Score"] = pd.to_numeric(all_votes["Score"], errors="coerce")
    all_votes["Favourites_count"] = pd.to_numeric(all_votes["Favourites_count"], errors="coerce")
    all_votes["Popularity_ranking"] = pd.to_numeric(all_votes["Popularity_ranking"], errors="coerce")
    all_votes["clean_title"] = all_votes["English_Name"].apply(normalize_title)

    weighted_scores = (
        all_votes.groupby(["clean_title", "snapshot_date"])
        .apply(_weighted_mal_score, include_groups=False)
        .reset_index(name="mal_weighted_score")
    )

    grouped = (
        all_votes.groupby(["clean_title", "snapshot_date"], as_index=False)
        .agg(
            mal_votes_total=("Votes", "sum"),
            mal_favourites=("Favourites_count", "max"),
            mal_popularity_rank=("Popularity_ranking", "min"),
            title_popularity=("English_Name", "first"),
        )
        .merge(weighted_scores, on=["clean_title", "snapshot_date"], how="left")
    )

    latest = (
        grouped.sort_values("snapshot_date")
        .groupby("clean_title", as_index=False)
        .tail(1)
        .copy()
    )

    latest["log_mal_votes"] = np.log1p(latest["mal_votes_total"].fillna(0))
    latest["log_mal_favourites"] = np.log1p(latest["mal_favourites"].fillna(0))
    latest["mal_popularity_score"] = 1 / latest["mal_popularity_rank"].replace(0, np.nan)

    return latest[
        [
            "clean_title",
            "title_popularity",
            "snapshot_date",
            "mal_votes_total",
            "mal_weighted_score",
            "mal_favourites",
            "mal_popularity_rank",
            "log_mal_votes",
            "log_mal_favourites",
            "mal_popularity_score",
        ]
    ]


# Combine the anime ratings and MAL popularity data and create clustering features.
def load_and_merge_data(anime_path: str | Path, popularity_dir: str | Path) -> pd.DataFrame:
    """Load both databases, merge them on normalized title, and build features."""
    anime_df = load_anime_database(anime_path)
    popularity_df = load_popularity_database(popularity_dir)

    log("Merging anime.csv with MyAnimeList popularity data")
    merged = anime_df.merge(popularity_df, on="clean_title", how="inner")

    if merged.empty:
        raise ValueError("No matching titles were found between anime.csv and the MyAnimeList files.")

    merged["title"] = merged["title"]
    merged["rating_gap"] = merged["mal_weighted_score"] - merged["critic_rating"]
    merged = merged.dropna(subset=["critic_rating", "mal_weighted_score"])

    log(f"Merged rows: {len(merged)}")
    return merged


# Extract the numeric feature columns and fill missing values before clustering.
def get_feature_matrix(df: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    """Return the numeric feature matrix used for clustering."""
    missing = [feature for feature in features if feature not in df.columns]
    if missing:
        raise ValueError(f"Missing feature columns: {missing}")

    feature_df = df[features].copy()
    for column in feature_df.columns:
        feature_df[column] = pd.to_numeric(feature_df[column], errors="coerce")
        feature_df[column] = feature_df[column].fillna(feature_df[column].median())

    return feature_df


# Normalize title text so matching the two databases is more reliable.
def normalize_title(title: str) -> str:
    """Normalize titles so the two databases can be merged more reliably."""
    text = str(title).lower().strip()
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text


# Pull the first four-digit year out of the release date text.
def _extract_year(value: str) -> float:
    match = re.search(r"(19|20)\d{2}", str(value))
    return float(match.group()) if match else np.nan


# Pull the runtime in minutes out of the length text.
def _extract_minutes(value: str) -> float:
    match = re.search(r"(\d+)", str(value))
    return float(match.group(1)) if match else np.nan


# Compute the weighted MAL score from the per-score vote breakdown.
def _weighted_mal_score(group: pd.DataFrame) -> float:
    total_votes = group["Votes"].sum()
    if total_votes == 0:
        return np.nan
    return float((group["Votes"] * group["Score"]).sum() / total_votes)
