"""
cluster.py
----------
Core clustering and reporting helpers.
"""

# Importing the important libraries
from __future__ import annotations
import pandas as pd
from sklearn.cluster import Birch, DBSCAN, HDBSCAN, KMeans
from sklearn.metrics import silhouette_score
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from utilities import CONFIG


# Fit the KMeans model using either the provided k or the default from the config.
# Trains the KMeans clustering model 
def fit_kmeans(X_scaled, n_clusters: int | None = None) -> KMeans:
    k = n_clusters or CONFIG["n_clusters"]
    model = KMeans(n_clusters=k, random_state=CONFIG["random_state"], n_init=10)
    #Learns cluster centers from SCALED DATA
    model.fit(X_scaled)
    return model
    #model should be trained


# Fit a Gaussian Mixture Model using the provided or configured cluster count.
def fit_gmm(X_scaled, n_clusters: int | None = None) -> GaussianMixture:
    k = n_clusters or CONFIG["n_clusters"]
    model = GaussianMixture(n_components=k, random_state=CONFIG["random_state"], covariance_type="full")
    model.fit(X_scaled)
    return model


# Fit DBSCAN on the standardized feature matrix.
def fit_dbscan(X_scaled, eps: float | None = None, min_samples: int | None = None) -> DBSCAN:
    model = DBSCAN(
        eps=eps or CONFIG["dbscan_eps"],
        min_samples=min_samples or CONFIG["dbscan_min_samples"],
    )
    model.fit(X_scaled)
    return model


# Fit HDBSCAN on the standardized feature matrix.
def fit_hdbscan(X_scaled, min_cluster_size: int | None = None, min_samples: int | None = None) -> HDBSCAN:
    model = HDBSCAN(
        min_cluster_size=min_cluster_size or CONFIG["hdbscan_min_cluster_size"],
        min_samples=min_samples or CONFIG["hdbscan_min_samples"],
        allow_single_cluster=True,
        copy=False,
    )
    model.fit(X_scaled)
    return model


# Fit BIRCH using the provided or configured cluster count.
def fit_birch(X_scaled, n_clusters: int | None = None) -> Birch:
    k = n_clusters or CONFIG["n_clusters"]
    model = Birch(
        n_clusters=k,
        threshold=CONFIG["birch_threshold"],
        branching_factor=CONFIG["birch_branching_factor"],
    )
    model.fit(X_scaled)
    return model


# Add a cluster label column to each row in the merged dataset based on the fitted model.
def assign_clusters(
    df: pd.DataFrame,
    model: KMeans | GaussianMixture | DBSCAN | HDBSCAN | Birch,
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
    result[cluster_column] = _predict_cluster_labels(model, X_scaled)
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


# Compare how each clustering model partitioned the dataset.
def clustering_comparison_report(X_scaled, df: pd.DataFrame, cluster_columns: dict[str, str]) -> pd.DataFrame:
    rows = []
    for algorithm_name, cluster_column in cluster_columns.items():
        labels = df[cluster_column]
        non_noise_mask = labels != -1
        non_noise_labels = labels[non_noise_mask]
        non_noise_distinct = sorted(label for label in non_noise_labels.dropna().unique())
        noise_points = int((labels == -1).sum())

        silhouette = float("nan")
        if len(non_noise_distinct) >= 2 and non_noise_mask.sum() > len(non_noise_distinct):
            silhouette = float(silhouette_score(X_scaled[non_noise_mask.to_numpy()], non_noise_labels))

        rows.append(
            {
                "algorithm": algorithm_name,
                "clusters_found": len(non_noise_distinct),
                "noise_points": noise_points,
                "noise_pct": round(noise_points / len(df) * 100, 2),
                "largest_cluster": int(labels.value_counts().max()),
                "silhouette_score": round(silhouette, 4) if pd.notna(silhouette) else float("nan"),
            }
        )

    summary = pd.DataFrame(rows).set_index("algorithm")
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


def _predict_cluster_labels(model, X_scaled) -> pd.Series:
    if hasattr(model, "predict"):
        return pd.Series(model.predict(X_scaled))
    if hasattr(model, "labels_"):
        return pd.Series(model.labels_)
    if hasattr(model, "fit_predict"):
        return pd.Series(model.fit_predict(X_scaled))
    raise AttributeError(f"Unsupported clustering model type: {type(model).__name__}")
