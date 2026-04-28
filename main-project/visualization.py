"""
visualization.py
----------------
Simple plots for the clustering and VAE analyses.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd

from utilities import ensure_output_dir, log


# Create all project plots in one place after clustering is complete.
def plot_all(
    df: pd.DataFrame,
    kmeans_summary: pd.DataFrame,
    gmm_summary: pd.DataFrame,
    vae_points,
    vae_history: dict[str, list[float]],
) -> None:
    output_dir = ensure_output_dir()

    _plot_rating_scatter(df, "kmeans_cluster", output_dir / "ratings_scatter.png", "K-Means")
    _plot_cluster_sizes(kmeans_summary, output_dir / "cluster_sizes.png", "K-Means Cluster Sizes")
    _plot_popularity_scatter(df, "kmeans_cluster", output_dir / "popularity_scatter.png", "K-Means")
    _plot_gmm_scatter(df, output_dir / "gmm_clusters.png")
    _plot_cluster_sizes(gmm_summary, output_dir / "gmm_cluster_sizes.png", "GMM Cluster Sizes")
    _plot_vae_latent_space(df, vae_points, output_dir / "vae_latent_space.png")
    _plot_vae_loss(vae_history, output_dir / "vae_loss.png")


# Plot critic ratings against MyAnimeList weighted scores.
def _plot_rating_scatter(df: pd.DataFrame, cluster_column: str, path, label_prefix: str) -> None:
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(df["critic_rating"], df["mal_weighted_score"], c=df[cluster_column], cmap="tab10", alpha=0.7)
    ax.set_title(f"Critic Rating vs MyAnimeList Score ({label_prefix})")
    ax.set_xlabel("anime.csv rating")
    ax.set_ylabel("MyAnimeList weighted score")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    log(f"Saved {path.name}")


# Plot MyAnimeList votes against favourites to show popularity concentration.
def _plot_popularity_scatter(df: pd.DataFrame, cluster_column: str, path, label_prefix: str) -> None:
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(df["log_mal_votes"], df["log_mal_favourites"], c=df[cluster_column], cmap="tab10", alpha=0.7)
    ax.set_title(f"MyAnimeList Votes vs Favourites ({label_prefix})")
    ax.set_xlabel("log(MyAnimeList votes)")
    ax.set_ylabel("log(MyAnimeList favourites)")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    log(f"Saved {path.name}")


# Plot critic ratings against MyAnimeList weighted scores with GMM labels.
def _plot_gmm_scatter(df: pd.DataFrame, path) -> None:
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(df["critic_rating"], df["mal_weighted_score"], c=df["gmm_cluster"], cmap="tab10", alpha=0.7)
    ax.set_title("GMM Clusters on Ratings")
    ax.set_xlabel("anime.csv rating")
    ax.set_ylabel("MyAnimeList weighted score")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    log(f"Saved {path.name}")


# Plot how many titles ended up in each cluster.
def _plot_cluster_sizes(cluster_summary: pd.DataFrame, path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    cluster_summary["titles"].plot(kind="bar", ax=ax, color="#4C78A8")
    ax.set_title(title)
    ax.set_xlabel(cluster_summary.index.name or "Cluster")
    ax.set_ylabel("Number of Titles")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    log(f"Saved {path.name}")


# Plot the VAE latent space colored by GMM cluster labels.
def _plot_vae_latent_space(df: pd.DataFrame, vae_points, path) -> None:
    fig, ax = plt.subplots(figsize=(8, 6))
    scatter = ax.scatter(vae_points[:, 0], vae_points[:, 1], c=df["gmm_cluster"], cmap="tab10", alpha=0.7)
    ax.set_title("VAE Latent Space")
    ax.set_xlabel("Latent Dimension 1")
    ax.set_ylabel("Latent Dimension 2")
    ax.legend(*scatter.legend_elements(), title="GMM Cluster")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    log(f"Saved {path.name}")


# Plot the VAE training losses over time.
def _plot_vae_loss(vae_history: dict[str, list[float]], path) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(vae_history["total_loss"], label="Total loss")
    ax.plot(vae_history["reconstruction_loss"], label="Reconstruction loss")
    ax.plot(vae_history["kl_loss"], label="KL loss")
    ax.set_title("VAE Training Loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    log(f"Saved {path.name}")
