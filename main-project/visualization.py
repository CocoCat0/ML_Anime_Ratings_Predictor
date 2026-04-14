"""
visualization.py
----------------
Simple plots for the merged anime and MyAnimeList data.
"""
# important libraries
from __future__ import annotations
import matplotlib.pyplot as plt
import pandas as pd
from utilities import ensure_output_dir, log

#just making an output directory and outputting all the scatter plots into it
# Create all project plots in one place after clustering is complete.
def plot_all(df: pd.DataFrame, pca_points, cluster_summary: pd.DataFrame) -> None:
    output_dir = ensure_output_dir()

    _plot_pca_scatter(df, pca_points, output_dir / "clusters_pca.png")
    _plot_rating_scatter(df, output_dir / "ratings_scatter.png")
    _plot_cluster_sizes(cluster_summary, output_dir / "cluster_sizes.png")
    _plot_popularity_scatter(df, output_dir / "popularity_scatter.png")


# Plot the PCA view of the clustered titles.
#Each anime is a point
#Uses PCA to reduce features into a 2 Dimensional
#Colors are cluster labels
def _plot_pca_scatter(df: pd.DataFrame, pca_points, path) -> None:
    fig, ax = plt.subplots(figsize=(8, 6))
    scatter = ax.scatter(pca_points[:, 0], pca_points[:, 1], c=df["cluster"], cmap="tab10", alpha=0.7)
    ax.set_title("Anime Clusters")
    ax.set_xlabel("PCA 1")
    ax.set_ylabel("PCA 2")
    ax.legend(*scatter.legend_elements(), title="Cluster")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    log(f"Saved {path.name}")

#X shows critic ratings
#Y shows MyAnimeList Score
#compare professional v user ratings
# Plot critic ratings against MyAnimeList weighted scores.
def _plot_rating_scatter(df: pd.DataFrame, path) -> None:
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(df["critic_rating"], df["mal_weighted_score"], c=df["cluster"], cmap="tab10", alpha=0.7)
    ax.set_title("Critic Rating vs MyAnimeList Score")
    ax.set_xlabel("anime.csv rating")
    ax.set_ylabel("MyAnimeList weighted score")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    log(f"Saved {path.name}")

#X-axis: Log Votes
#Y-axis: Log(Favorites)
#We utilized log since popularity data are usually skewed and it makes it easier to viz
# Plot MyAnimeList votes against favourites to show popularity concentration.
def _plot_popularity_scatter(df: pd.DataFrame, path) -> None:
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(df["log_mal_votes"], df["log_mal_favourites"], c=df["cluster"], cmap="tab10", alpha=0.7)
    ax.set_title("MyAnimeList Votes vs Favourites")
    ax.set_xlabel("log(MyAnimeList votes)")
    ax.set_ylabel("log(MyAnimeList favourites)")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    log(f"Saved {path.name}")

#Number of anime per cluster
#Clusters balanced?
#One cluster Too large?
# Plot how many titles ended up in each cluster.
def _plot_cluster_sizes(cluster_summary: pd.DataFrame, path) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    cluster_summary["titles"].plot(kind="bar", ax=ax, color="#4C78A8")
    ax.set_title("Titles per Cluster")
    ax.set_xlabel("Cluster")
    ax.set_ylabel("Number of Titles")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    log(f"Saved {path.name}")
