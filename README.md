# ML Ratings Predictor README

## This project analyzes anime critic ratings and MyAnimeList popularity data using multiple unsupervised machine learning clustering algorithms. The system compares how audience popularity and critic perception relate to one another by grouping anime titles into clusters based on ratings, favourites, votes, popularity rank, release year, and runtime information.

Expected Project Outputs:
- Clustered Datasets
- CSV report
- Data Viusalization
- Auto Generated HTML Website Report : https://cococat0.github.io/ML_Anime_Ratings_Predictor/main-project/output/index.html

Algorithms Included:
- Gausian Mixture Model (GMM)
- Kmeans
- DBSCAN
- HDBSCAN
- BIRCH

The project also includes a lightweight custom Variational Autoencoder (VAE) implementation for latent-space visualization.

---------------------------------------------------------------------------------------------------------------------------------------------------------------------

## Overview of Project Structure:
```bash
main-project/
│
├── data/
│   ├── anime.csv
│   └── popularity/
│       └── Votes_*.csv
│
├── output/
│   ├── *.png
│   ├── *.csv
│   └── index.html
│
├── cluster.py
├── ML_training.py
├── preprocessing.py
├── visualization.py
├── report_site.py
├── utilities.py
├── main.py
│
├── requirements.txt
├── environment.yml
└── README.md
```
---------------------------------------------------------------------------------------------------------------------------------------------------------------------
## Installation

Before running the project, install the required python libraries:
```bash
pip install -r enviornment.txt
```

OR if using conda environment, enter
```bash
conda env create -f environment.yml
conda activate ml_anime_ratings_analysis
```


Open terminal under a main-project directory
```bash
cd main-project
```



To run default clustering pipeline, enter
```bash
python main.py
```
This will:
- Load and preprocess datasets
- Train clustering models
- Generate clustering report
- Save visualization
- Built HTML website Report

## Command Line Options

Find Best K Automatically
```bash
python main.py --find-k
```
Uses silhoutte score analysis to automatically determine the best cluster count


Entering Manual Number of Clusters
```bash
python main.py --k 6
```
Will create 6 clusters. The input number must be non-zero integer


Disable Visualization
```bash
python main.py --no-vis
```
No visualizations will be outputted


Preview Raw Datasets
```bash
python main.py --raw
```
View the raw dataset that is being loaded into the system

Preview Preprocessed Dataset
```bash
python main.py --preprocess
```
View the merged and loaded dataset that will be readable by the machine learning model

---------------------------------------------------------------------------------------------------------------------------------------------------------------------

## Clustering Algorithms
- K-Means : Partitions anime into fixed number of centroid-based clusters.
- Gaussian Mixture Model (GMM) : Uses probabilistic Gaussian distributions for softer cluster boundaries
- DBSCAN : Density-based clustering that detects noise and irregular cluster shapes.
- HDBSCAN : Hierarchical density clustering that adapts to variable-density clusters.
- BIRCH : Scalable clustering using clustering feature trees.

---------------------------------------------------------------------------------------------------------------------------------------------------------------------

## Variational Autoencoder (VAE)
This project includes a custom NumPy-based VAE application for latent-space visualization

The VAE:
- Compresses features into a 2D latent representation
- Learns compressed feature relationships
- Generates latent-space cluster visualization

Outputs include:
- VAE latent space plots
- VAE algorithm comparison plots
- VAE training loss graphs

---------------------------------------------------------------------------------------------------------------------------------------------------------------------

## Outputs
The ouputs are found under the output/ folder in which contains:

CSV Files:
- clustered_anime.csv
- cluster_report.csv
- gmm_cluster_report.csv
- dbscan_cluster_report.csv
- hdbscan_cluster_report.csv
- birch_cluster_report.csv
- algorithm_comparison_report.csv

Visualizations
- Rating comparison plots
- popularity comparison plots
- Cluster size charts
- Algorithm comparison charts
- VAE laten space visualizations
- VAE loss plots

---------------------------------------------------------------------------------------------------------------------------------------------------------------------

## HTML Report
Under output/ folder is the main website index.html

The website showcases the input datasets and clustering algorithms. It incluldes key findings from this project and the outputs created by the system. 

---------------------------------------------------------------------------------------------------------------------------------------------------------------------

## Authors
Christofer Vega 

Alyssa Sombrero
