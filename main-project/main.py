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

from cluster import (
    assign_clusters,
    cluster_report,
    clustering_comparison_report,
    fit_birch,
    fit_dbscan,
    fit_gmm,
    fit_hdbscan,
    fit_kmeans,
    top_titles_per_cluster,
)
from ML_training import find_best_k, fit_scaler, train_vae_projection
from preprocessing import get_feature_matrix, load_and_merge_data
from report_site import build_output_report
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
def main(args: argparse.Namespace) -> None:
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

    dbscan_model = fit_dbscan(X_scaled)
    clustered_df = assign_clusters(clustered_df, dbscan_model, scaler, CONFIG["features"], cluster_column="dbscan_cluster")
    dbscan_summary = cluster_report(clustered_df, cluster_column="dbscan_cluster")

    hdbscan_model = fit_hdbscan(X_scaled)
    clustered_df = assign_clusters(clustered_df, hdbscan_model, scaler, CONFIG["features"], cluster_column="hdbscan_cluster")
    hdbscan_summary = cluster_report(clustered_df, cluster_column="hdbscan_cluster")

    birch_model = fit_birch(X_scaled, n_clusters=k)
    clustered_df = assign_clusters(clustered_df, birch_model, scaler, CONFIG["features"], cluster_column="birch_cluster")
    birch_summary = cluster_report(clustered_df, cluster_column="birch_cluster")

    comparison_summary = clustering_comparison_report(
        X_scaled,
        clustered_df,
        {
            "K-Means": "kmeans_cluster",
            "GMM": "gmm_cluster",
            "DBSCAN": "dbscan_cluster",
            "HDBSCAN": "hdbscan_cluster",
            "BIRCH": "birch_cluster",
        },
    )

    save_clustered_data(clustered_df)
    kmeans_summary.to_csv(CONFIG["report_path"])
    gmm_summary.to_csv(CONFIG["gmm_report_path"])
    dbscan_summary.to_csv(CONFIG["dbscan_report_path"])
    hdbscan_summary.to_csv(CONFIG["hdbscan_report_path"])
    birch_summary.to_csv(CONFIG["birch_report_path"])
    comparison_summary.to_csv(CONFIG["comparison_report_path"])

    log("K-Means cluster summary:")
    log(kmeans_summary.to_string())
    log("\nGMM cluster summary:")
    log(gmm_summary.to_string())
    log("\nDBSCAN cluster summary:")
    log(dbscan_summary.to_string())
    log("\nHDBSCAN cluster summary:")
    log(hdbscan_summary.to_string())
    log("\nBIRCH cluster summary:")
    log(birch_summary.to_string())
    log("\nAlgorithm comparison summary:")
    log(comparison_summary.to_string())

    for cluster_id, titles in top_titles_per_cluster(clustered_df, cluster_column="kmeans_cluster").items():
        log(f"\nTop titles in K-Means cluster {cluster_id}:")
        log(titles.to_string(index=False))

    if not args.no_vis:
        plot_all(
            clustered_df,
            {
                "K-Means": kmeans_summary,
                "GMM": gmm_summary,
                "DBSCAN": dbscan_summary,
                "HDBSCAN": hdbscan_summary,
                "BIRCH": birch_summary,
            },
            comparison_summary,
            vae_points,
            vae_history,
        )

    build_output_report()


if __name__ == "__main__":
    main(parse_args())
