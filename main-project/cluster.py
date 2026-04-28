"""
cluster.py
----------
Core clustering and reporting helpers.
"""

from __future__ import annotations

import pandas as pd
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

from utilities import CONFIG


# Fit the KMeans model using either the provided k or the default from the config.
def fit_kmeans(X_scaled, n_clusters: int | None = None) -> KMeans:
    k = n_clusters or CONFIG["n_clusters"]
    model = KMeans(n_clusters=k, random_state=CONFIG["random_state"], n_init=10)
    model.fit(X_scaled)
    return model


# Fit a Gaussian Mixture Model using the provided or configured cluster count.
def fit_gmm(X_scaled, n_clusters: int | None = None) -> GaussianMixture:
    k = n_clusters or CONFIG["n_clusters"]
    model = GaussianMixture(n_components=k, random_state=CONFIG["random_state"], covariance_type="full")
    model.fit(X_scaled)
    return model


# Add a cluster label column to each row in the merged dataset based on the fitted model.
def assign_clusters(
    df: pd.DataFrame,
    model: KMeans | GaussianMixture,
    scaler: StandardScaler,
    feature_names: list[str],
    cluster_column: str = "cluster",
) -> pd.DataFrame:
    result = df.copy()
    feature_df = result[feature_names].copy()
    for column in feature_df.columns:
        feature_df[column] = pd.to_numeric(feature_df[column], errors="coerce")
        feature_df[column] = feature_df[column].fillna(feature_df[column].median())

    X_scaled = scaler.transform(feature_df)
    result[cluster_column] = model.predict(X_scaled)
    return result


# Summarize each cluster with rating and popularity averages.
def cluster_report(df: pd.DataFrame, cluster_column: str = "cluster") -> pd.DataFrame:
    summary = (
        df.groupby(cluster_column)
        .agg(
            titles=("title", "count"),
            critic_rating=("critic_rating", "mean"),
            mal_weighted_score=("mal_weighted_score", "mean"),
            rating_gap=("rating_gap", "mean"),
            mal_votes_total=("mal_votes_total", "mean"),
            mal_favourites=("mal_favourites", "mean"),
            mal_popularity_rank=("mal_popularity_rank", "mean"),
            release_year=("release_year", "mean"),
        )
        .round(2)
        .sort_index()
    )
    summary.index.name = cluster_column
    return summary


# List representative titles inside each cluster ordered by MAL engagement.
def top_titles_per_cluster(
    df: pd.DataFrame,
    cluster_column: str = "cluster",
    top_n: int = 5,
) -> dict[int, pd.DataFrame]:
    top_titles: dict[int, pd.DataFrame] = {}
    for cluster_id, group in df.groupby(cluster_column):
        top_titles[cluster_id] = (
            group.sort_values(["mal_favourites", "mal_votes_total"], ascending=False)[
                ["title", "critic_rating", "mal_weighted_score", "rating_gap", "mal_favourites", "mal_popularity_rank"]
            ]
            .head(top_n)
            .reset_index(drop=True)
        )
    return top_titles
