"""
utilities.py
------------
Shared configuration and small helper functions for the project.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import pandas as pd


BASE_DIR = Path(__file__).resolve().parent

CONFIG = {
    "anime_path": BASE_DIR / "data" / "anime.csv",
    "popularity_dir": BASE_DIR / "data" / "popularity",
    "output_dir": BASE_DIR / "output",
    "clustered_data_path": BASE_DIR / "output" / "clustered_anime.csv",
    "report_path": BASE_DIR / "output" / "cluster_report.csv",
    "gmm_report_path": BASE_DIR / "output" / "gmm_cluster_report.csv",
    "n_clusters": 7,
    "random_state": 42,
    "vae_hidden_dim": 16,
    "vae_latent_dim": 2,
    "vae_epochs": 350,
    "vae_learning_rate": 0.01,
    "vae_beta": 0.08,
    "features": [
        "critic_rating",
        "mal_weighted_score",
        "rating_gap",
        "log_mal_votes",
        "log_mal_favourites",
        "mal_popularity_score",
        "release_year",
        "runtime_minutes",
    ],
}


logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
LOGGER = logging.getLogger("ratings_predictor")


# Send a simple log message to stdout.
def log(message: str) -> None:
    LOGGER.info(message)


# Create the output folder if it does not already exist.
def ensure_output_dir() -> Path:
    output_dir = Path(CONFIG["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


# Check whether the required datasets are present on disk.
def check_data_files() -> list[str]:
    missing = []
    for key in ("anime_path", "popularity_dir"):
        path = Path(CONFIG[key])
        if not path.exists():
            missing.append(str(path))
    return missing


# Save the final clustered dataset to the output directory.
def save_clustered_data(df: pd.DataFrame) -> Path:
    path = Path(CONFIG["clustered_data_path"])
    ensure_output_dir()
    df.to_csv(path, index=False)
    log(f"Saved clustered data to {path}")
    return path
