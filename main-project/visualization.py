"""
visualization.py
----------------
Simple plots for the clustering and VAE analyses.
"""
# important libraries
from __future__ import annotations
import matplotlib.pyplot as plt
import pandas as pd
from utilities import ensure_output_dir, log

#mapping algorithms to cluster label col in Dataframe
ALGORITHM_COLUMNS = {
    "K-Means": "kmeans_cluster",
    "GMM": "gmm_cluster",
    "DBSCAN": "dbscan_cluster",
    "HDBSCAN": "hdbscan_cluster",
    "BIRCH": "birch_cluster",
}


#just making an output directory and outputting all the scatter plots into it
# Create all project plots in one place after clustering is complete.
def plot_all(
    df: pd.DataFrame,
    cluster_summaries: dict[str, pd.DataFrame],
    comparison_summary: pd.DataFrame,
    vae_points,
    vae_history: dict[str, list[float]],
) -> None:
    #creating visualizations: rating scatterplots, popularity scatterplots, cluster size bar charts, algorithm comparison grids, vae latent space plots, vae training loss curve
    output_dir = ensure_output_dir()

    _plot_rating_scatter(df, "kmeans_cluster", output_dir / "ratings_scatter.png", "K-Means")
    _plot_cluster_sizes(cluster_summaries["K-Means"], output_dir / "cluster_sizes.png", "K-Means Cluster Sizes")
    _plot_popularity_scatter(df, "kmeans_cluster", output_dir / "popularity_scatter.png", "K-Means")
    _plot_rating_scatter(df, "gmm_cluster", output_dir / "gmm_clusters.png", "GMM")
    _plot_cluster_sizes(cluster_summaries["GMM"], output_dir / "gmm_cluster_sizes.png", "GMM Cluster Sizes")
    _plot_rating_scatter(df, "dbscan_cluster", output_dir / "dbscan_clusters.png", "DBSCAN")
    _plot_cluster_sizes(cluster_summaries["DBSCAN"], output_dir / "dbscan_cluster_sizes.png", "DBSCAN Cluster Sizes")
    _plot_rating_scatter(df, "hdbscan_cluster", output_dir / "hdbscan_clusters.png", "HDBSCAN")
    _plot_cluster_sizes(cluster_summaries["HDBSCAN"], output_dir / "hdbscan_cluster_sizes.png", "HDBSCAN Cluster Sizes")
    _plot_rating_scatter(df, "birch_cluster", output_dir / "birch_clusters.png", "BIRCH")
    _plot_cluster_sizes(cluster_summaries["BIRCH"], output_dir / "birch_cluster_sizes.png", "BIRCH Cluster Sizes")
    #creatinga  multi-algorithm comparison grids
    _plot_algorithm_comparison_grid(
        df,
        x_column="critic_rating",
        y_column="mal_weighted_score",
        path=output_dir / "algorithm_ratings_comparison.png",
        title="Critic Rating vs MyAnimeList Score",
    )
    _plot_algorithm_comparison_grid(
        df,
        x_column="log_mal_votes",
        y_column="log_mal_favourites",
        path=output_dir / "algorithm_popularity_comparison.png",
        title="MyAnimeList Votes vs Favourites",
    )
    #algorithm-lvl metrics
    _plot_cluster_metric_comparison(comparison_summary, output_dir / "algorithm_metrics_comparison.png")
    _plot_cluster_size_comparison(cluster_summaries, output_dir / "algorithm_cluster_size_comparison.png")
    #VAE visualizations
    _plot_vae_latent_space(df, vae_points, output_dir / "vae_latent_space.png")
    _plot_vae_algorithm_grid(df, vae_points, output_dir / "vae_algorithm_comparison.png")
    _plot_vae_loss(vae_history, output_dir / "vae_loss.png")

#X shows critic ratings
#Y shows MyAnimeList Score
#compare professional v user ratings
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

#X-axis: Log Votes
#Y-axis: Log(Favorites)
#We utilized log since popularity data are usually skewed and it makes it easier to viz
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


# Plot how many titles ended up in each cluster.
#plot how many anime titltes fall under each cluster (detecting unbalanced clusters)
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

#creatinga  2x3 grid comparing each clustering algorithm 
def _plot_algorithm_comparison_grid(df: pd.DataFrame, x_column: str, y_column: str, path, title: str) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    axes = axes.flatten()

    for ax, (algorithm_name, cluster_column) in zip(axes, ALGORITHM_COLUMNS.items()):
        ax.scatter(df[x_column], df[y_column], c=df[cluster_column], cmap="tab10", alpha=0.7)
        ax.set_title(algorithm_name)
        ax.set_xlabel(_format_axis_label(x_column))
        ax.set_ylabel(_format_axis_label(y_column))
        ax.grid(alpha=0.3)

    axes[-1].axis("off")
    fig.suptitle(title, fontsize=14)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    log(f"Saved {path.name}")

#plots three metrics per algorithm, # of clusters found, noise percentage (dbscan / hdbscan), silhoutte score
def _plot_cluster_metric_comparison(comparison_summary: pd.DataFrame, path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    summary = comparison_summary.reset_index()

    #clusters found
    axes[0].bar(summary["algorithm"], summary["clusters_found"], color="#4C78A8")
    axes[0].set_title("Clusters Found")
    axes[0].set_ylabel("Count")

    #noise share
    axes[1].bar(summary["algorithm"], summary["noise_pct"], color="#F58518")
    axes[1].set_title("Noise Share")
    axes[1].set_ylabel("Percent of Titles")

    #silhoutte score
    silhouette_values = summary["silhouette_score"].fillna(0.0)
    axes[2].bar(summary["algorithm"], silhouette_values, color="#54A24B")
    axes[2].set_title("Silhouette Score")
    axes[2].set_ylabel("Score")

    #formatting
    for ax in axes:
        ax.tick_params(axis="x", rotation=20)
        ax.grid(alpha=0.25, axis="y")

    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    log(f"Saved {path.name}")

#creatinga  2x3 grid showing cluster size distributions per clustering algorithm
def _plot_cluster_size_comparison(cluster_summaries: dict[str, pd.DataFrame], path) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    axes = axes.flatten()

    for ax, (algorithm_name, summary) in zip(axes, cluster_summaries.items()):
        summary["titles"].plot(kind="bar", ax=ax, color="#4C78A8")
        ax.set_title(f"{algorithm_name} Cluster Sizes")
        ax.set_xlabel(summary.index.name or "Cluster")
        ax.set_ylabel("Number of Titles")
        ax.grid(alpha=0.25, axis="y")

    axes[-1].axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    log(f"Saved {path.name}")


# Plot the VAE latent space colored by GMM cluster labels.
def _plot_vae_latent_space(df: pd.DataFrame, vae_points, path) -> None:
    #Show how vae compresses anime features
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


def _plot_vae_algorithm_grid(df: pd.DataFrame, vae_points, path) -> None:
    #show how clustering algorithm groups titles inside VAE latent space
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    axes = axes.flatten()

    for ax, (algorithm_name, cluster_column) in zip(axes, ALGORITHM_COLUMNS.items()):
        ax.scatter(vae_points[:, 0], vae_points[:, 1], c=df[cluster_column], cmap="tab10", alpha=0.7)
        ax.set_title(f"{algorithm_name} on VAE Space")
        ax.set_xlabel("Latent Dimension 1")
        ax.set_ylabel("Latent Dimension 2")
        ax.grid(alpha=0.3)

    axes[-1].axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    log(f"Saved {path.name}")


# Plot the VAE training losses over time.
def _plot_vae_loss(vae_history: dict[str, list[float]], path) -> None:
    #plot vae training losses, total loss, reconstruction loss, kl divergence loss
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


def _format_axis_label(column_name: str) -> str:
    return column_name.replace("_", " ").title()
