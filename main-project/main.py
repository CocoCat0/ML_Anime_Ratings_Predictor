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

#from cluster.py 
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
#importing important data from python files.
#import ML utilities (scaling, VAE projection, K selection)
from ML_training import find_best_k, fit_scaler, train_vae_projection
#import preprocessing utilities (loading csvs, building feature matrix)
from preprocessing import get_feature_matrix, load_and_merge_data, prev_raw_anime, prev_raw_popularity, prev_clean_anime, prev_clean_popularity, prev_preprocessed_data
#import report builder & general utilities
from report_site import build_output_report
from utilities import CONFIG, check_data_files, ensure_output_dir, log, save_clustered_data
#import viz functions
from visualization import plot_all


# Read command-line flags so the pipeline can be configured when run.
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Cluster anime using anime.csv and MyAnimeList data")
    #'--k' lets user manually choose number of clusters
    parser.add_argument("--k", type=int, default=None, help="Number of clusters to use")
    #'--find-k' auto chooses k through silhoutte sore
    parser.add_argument("--find-k", action="store_true", help="Choose k using silhouette score")
    # '--no-vis' skips making plots
    parser.add_argument("--no-vis", action="store_true", help="Skip plotting")
    #'--raw' to get raw csv data
    parser.add_argument("--raw", action="store_true", help="Print raw datasets before clustering")
    #'--clean' to get cleaned csv data
    parser.add_argument("--clean", action="store_true", help="Print cleaned dataset after normalization")
    #'--preprocess' to get preprocessed data that is loaded into the model
    parser.add_argument("--preprocess", action="store_true", help="Print preprocessed merged dataset before clustering")
    return parser.parse_args()


# Run the full workflow: load data, cluster titles, save reports, and generate plots.
def main(args: argparse.Namespace) -> None:
    #checking if csv files actually exist
    missing_files = check_data_files()
    if missing_files:
        log("Missing data files:")
        for path in missing_files:
            log(path)
        sys.exit(1)
    #make sure directory exists for outputs
    ensure_output_dir()

    #previewing the datasets before clustering
    if args.raw:
        #raw anime dataset
        log("----------- Raw Anime Data -----------")
        raw_anime = prev_raw_anime(CONFIG["anime_path"])
        print(raw_anime.head(20).to_string())
        #raw popularity dataset
        log("----------- Raw Popularity Data -----------")
        raw_popularity = prev_raw_popularity(CONFIG["popularity_dir"])
        print(raw_popularity.head(20).to_string())
        return

    if args.clean:
        #cleaned anime dataset
        log("----------- Cleaned Anime Data -----------")
        clean_anime = prev_clean_anime(CONFIG["anime_path"])
        print(clean_anime.head(20).to_string())
        #cleaned popularity dataset
        log("----------- Cleaned Popularity Data -----------")
        clean_pop = prev_clean_popularity(CONFIG["popularity_dir"])
        print(clean_pop.head(20).to_string())
        return

    if args.preprocess:
        log("----------- Preprocessed & Merged Data -----------")
        processed = prev_preprocessed_data(CONFIG["anime_path"], CONFIG["popularity_dir"])
        print(processed.head(20).to_string())
        return


    #load anime.csv and popularity csv and merge into a dataframe
    df = load_and_merge_data(CONFIG["anime_path"], CONFIG["popularity_dir"])
    #extractng numerical features for clustering
    X = get_feature_matrix(df, CONFIG["features"])
    #scaling features (standardization)
    scaler, X_scaled = fit_scaler(X)
    #train VAE to get data into a 2D latent space (opt visual)
    _, vae_points, vae_history = train_vae_projection(X_scaled)

    #getting the number of clusters from either auto or manual
    k = args.k or CONFIG["n_clusters"]
    if args.find_k:
        k, _ = find_best_k(X_scaled)

    #kmeans
    kmeans_model = fit_kmeans(X_scaled, n_clusters=k)
    clustered_df = assign_clusters(df, kmeans_model, scaler, CONFIG["features"], cluster_column="kmeans_cluster")
    kmeans_summary = cluster_report(clustered_df, cluster_column="kmeans_cluster")

    #Gaussian Mixture Model
    gmm_model = fit_gmm(X_scaled, n_clusters=k)
    clustered_df = assign_clusters(clustered_df, gmm_model, scaler, CONFIG["features"], cluster_column="gmm_cluster")
    gmm_summary = cluster_report(clustered_df, cluster_column="gmm_cluster")

    #DBSCAN
    dbscan_model = fit_dbscan(X_scaled)
    clustered_df = assign_clusters(clustered_df, dbscan_model, scaler, CONFIG["features"], cluster_column="dbscan_cluster")
    dbscan_summary = cluster_report(clustered_df, cluster_column="dbscan_cluster")

    #HDBSCAN
    hdbscan_model = fit_hdbscan(X_scaled)
    clustered_df = assign_clusters(clustered_df, hdbscan_model, scaler, CONFIG["features"], cluster_column="hdbscan_cluster")
    hdbscan_summary = cluster_report(clustered_df, cluster_column="hdbscan_cluster")

    #BIRCH
    birch_model = fit_birch(X_scaled, n_clusters=k)
    clustered_df = assign_clusters(clustered_df, birch_model, scaler, CONFIG["features"], cluster_column="birch_cluster")
    birch_summary = cluster_report(clustered_df, cluster_column="birch_cluster")

    #comparing ALL the clustering algorithms side by side
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

    #summarize in console/log
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

    #printing top anime titles per kmeans cluster
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
    #building html/markdown report site
    build_output_report()


if __name__ == "__main__":
    main(parse_args())
