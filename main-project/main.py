"""
main.py
-------
Simple entry point for clustering anime titles using:

- anime.csv for critic-style ratings
- popularity/*.csv for MyAnimeList popularity and score data
"""

#important libraries
from __future__ import annotations
import argparse
import sys

from cluster import assign_clusters, cluster_report, fit_gmm, fit_kmeans, top_titles_per_cluster
from ML_training import find_best_k, fit_scaler, train_vae_projection
from preprocessing import get_feature_matrix, load_and_merge_data
from utilities import CONFIG, check_data_files, ensure_output_dir, log, save_clustered_data
from visualization import plot_all


# Read command-line flags so the pipeline can be configured when run.
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Cluster anime using anime.csv and MyAnimeList data")
    parser.add_argument("--k", type=int, default=None, help="Number of clusters to use")
    parser.add_argument("--find-k", action="store_true", help="Choose k using silhouette score")
    parser.add_argument("--no-vis", action="store_true", help="Skip plotting")
    return parser.parse_args()


# Run the full workflow: load data, cluster titles, save reports, and generate plots.
def main() -> None:
    missing_files = check_data_files()
    if missing_files:
        log("Missing data files:")
        for path in missing_files:
            log(path)
        sys.exit(1)

    ensure_output_dir()

    df = load_and_merge_data(CONFIG["anime_path"], CONFIG["popularity_dir"])
    X = get_feature_matrix(df, CONFIG["features"])
    scaler, X_scaled = fit_scaler(X)
    _, vae_points, vae_history = train_vae_projection(X_scaled)

    k = args.k or CONFIG["n_clusters"]
    if args.find_k:
        k, _ = find_best_k(X_scaled)

    kmeans_model = fit_kmeans(X_scaled, n_clusters=k)
    clustered_df = assign_clusters(df, kmeans_model, scaler, CONFIG["features"], cluster_column="kmeans_cluster")
    kmeans_summary = cluster_report(clustered_df, cluster_column="kmeans_cluster")

    gmm_model = fit_gmm(X_scaled, n_clusters=k)
    clustered_df = assign_clusters(clustered_df, gmm_model, scaler, CONFIG["features"], cluster_column="gmm_cluster")
    gmm_summary = cluster_report(clustered_df, cluster_column="gmm_cluster")

    save_clustered_data(clustered_df)
    kmeans_summary.to_csv(CONFIG["report_path"])
    gmm_summary.to_csv(CONFIG["gmm_report_path"])

    log("K-Means cluster summary:")
    log(kmeans_summary.to_string())
    log("\nGMM cluster summary:")
    log(gmm_summary.to_string())

    for cluster_id, titles in top_titles_per_cluster(clustered_df, cluster_column="kmeans_cluster").items():
        log(f"\nTop titles in K-Means cluster {cluster_id}:")
        log(titles.to_string(index=False))

    if not args.no_vis:
        plot_all(clustered_df, kmeans_summary, gmm_summary, vae_points, vae_history)


if __name__ == "__main__":
    args = parse_args()
    main()
