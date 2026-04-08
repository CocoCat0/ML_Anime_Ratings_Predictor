"""
ML_training.py
--------------
Small training helpers for scaling, k selection, and PCA projection.
"""

from __future__ import annotations

import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

from utilities import CONFIG, log


# Standardize the feature matrix so each feature contributes on a similar scale.
def fit_scaler(X: np.ndarray) -> tuple[StandardScaler, np.ndarray]:
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return scaler, X_scaled


# Try several cluster counts and choose the one with the best silhouette score.
def find_best_k(X_scaled: np.ndarray, min_k: int = 2, max_k: int = 6) -> tuple[int, dict[int, float]]:
    """
    Pick the best k using silhouette score.
    This keeps the model selection step simple and readable.
    """
    upper_k = min(max_k, len(X_scaled) - 1)
    if upper_k < min_k:
        return CONFIG["n_clusters"], {}

    scores: dict[int, float] = {}
    for k in range(min_k, upper_k + 1):
        model = KMeans(n_clusters=k, random_state=CONFIG["random_state"], n_init=10)
        labels = model.fit_predict(X_scaled)
        scores[k] = float(silhouette_score(X_scaled, labels))

    best_k = max(scores, key=scores.get)
    log(f"Silhouette scores: {scores}")
    log(f"Selected k={best_k}")
    return best_k, scores


# Reduce the scaled feature matrix to two dimensions for plotting.
def pca_projection(X_scaled: np.ndarray) -> np.ndarray:
    pca = PCA(n_components=2, random_state=CONFIG["random_state"])
    return pca.fit_transform(X_scaled)
