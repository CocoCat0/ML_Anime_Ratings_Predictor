"""
ML_training.py
--------------
Small training helpers for scaling, k selection, and VAE training.
"""

#important libraries
from __future__ import annotations
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from utilities import CONFIG, log


#Standardize the feature matrix so each feature contributes on a similar scale.
def fit_scaler(X: np.ndarray) -> tuple[StandardScaler, np.ndarray]:
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return scaler, X_scaled


# Try several cluster counts and choose the one with the best silhouette score.
#Playing around with the K configuration to find one that outputs the best silhouttte score.
def find_best_k(X_scaled: np.ndarray, min_k: int = 2, max_k: int = 6) -> tuple[int, dict[int, float]]:
    upper_k = min(max_k, len(X_scaled) - 1)
    if upper_k < min_k:
        return CONFIG["n_clusters"], {}
    #running th model with the potential k's
    scores: dict[int, float] = {}
    for k in range(min_k, upper_k + 1):
        model = KMeans(n_clusters=k, random_state=CONFIG["random_state"], n_init=10)
        labels = model.fit_predict(X_scaled)
        scores[k] = float(silhouette_score(X_scaled, labels))
    #the k that gets the highest silhoutte score is the best k 
    best_k = max(scores, key=scores.get)
    log(f"Silhouette scores: {scores}")
    log(f"Selected k={best_k}")
    return best_k, scores


class SimpleVAE:
    """Small NumPy-only variational autoencoder used for 2D latent projections."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        latent_dim: int,
        learning_rate: float,
        beta: float,
        random_state: int,
    ) -> None:
        rng = np.random.default_rng(random_state)
        self.learning_rate = learning_rate
        self.beta = beta

        self.W1 = rng.normal(0.0, np.sqrt(2 / (input_dim + hidden_dim)), size=(input_dim, hidden_dim))
        self.b1 = np.zeros(hidden_dim)
        self.W_mu = rng.normal(0.0, np.sqrt(2 / (hidden_dim + latent_dim)), size=(hidden_dim, latent_dim))
        self.b_mu = np.zeros(latent_dim)
        self.W_logvar = rng.normal(0.0, np.sqrt(2 / (hidden_dim + latent_dim)), size=(hidden_dim, latent_dim))
        self.b_logvar = np.zeros(latent_dim)
        self.W2 = rng.normal(0.0, np.sqrt(2 / (latent_dim + hidden_dim)), size=(latent_dim, hidden_dim))
        self.b2 = np.zeros(hidden_dim)
        self.W_out = rng.normal(0.0, np.sqrt(2 / (hidden_dim + input_dim)), size=(hidden_dim, input_dim))
        self.b_out = np.zeros(input_dim)

    def _forward(self, X: np.ndarray, rng: np.random.Generator) -> dict[str, np.ndarray]:
        h = np.tanh(X @ self.W1 + self.b1)
        mu = h @ self.W_mu + self.b_mu
        logvar = np.clip(h @ self.W_logvar + self.b_logvar, -6.0, 6.0)
        std = np.exp(0.5 * logvar)
        eps = rng.normal(size=std.shape)
        z = mu + std * eps
        h_dec = np.tanh(z @ self.W2 + self.b2)
        reconstruction = h_dec @ self.W_out + self.b_out
        return {
            "h": h,
            "mu": mu,
            "logvar": logvar,
            "std": std,
            "eps": eps,
            "z": z,
            "h_dec": h_dec,
            "reconstruction": reconstruction,
        }

    def fit(self, X: np.ndarray, epochs: int) -> dict[str, list[float]]:
        rng = np.random.default_rng(CONFIG["random_state"])
        history = {"total_loss": [], "reconstruction_loss": [], "kl_loss": []}

        for _ in range(epochs):
            cache = self._forward(X, rng)
            batch_size = X.shape[0]

            reconstruction_error = cache["reconstruction"] - X
            reconstruction_loss = float(0.5 * np.mean(np.sum(reconstruction_error**2, axis=1)))
            kl_term = 0.5 * np.sum(
                cache["mu"] ** 2 + np.exp(cache["logvar"]) - 1.0 - cache["logvar"],
                axis=1,
            )
            kl_loss = float(np.mean(kl_term))
            total_loss = reconstruction_loss + self.beta * kl_loss

            d_reconstruction = reconstruction_error / batch_size
            dW_out = cache["h_dec"].T @ d_reconstruction
            db_out = d_reconstruction.sum(axis=0)

            d_h_dec = d_reconstruction @ self.W_out.T
            d_pre_h_dec = d_h_dec * (1.0 - cache["h_dec"] ** 2)
            dW2 = cache["z"].T @ d_pre_h_dec
            db2 = d_pre_h_dec.sum(axis=0)

            d_z = d_pre_h_dec @ self.W2.T
            d_mu = d_z + (self.beta / batch_size) * cache["mu"]
            d_logvar = d_z * (0.5 * cache["std"] * cache["eps"])
            d_logvar += (self.beta / batch_size) * 0.5 * (np.exp(cache["logvar"]) - 1.0)

            dW_mu = cache["h"].T @ d_mu
            db_mu = d_mu.sum(axis=0)
            dW_logvar = cache["h"].T @ d_logvar
            db_logvar = d_logvar.sum(axis=0)

            d_h = d_mu @ self.W_mu.T + d_logvar @ self.W_logvar.T
            d_pre_h = d_h * (1.0 - cache["h"] ** 2)
            dW1 = X.T @ d_pre_h
            db1 = d_pre_h.sum(axis=0)

            self.W_out -= self.learning_rate * dW_out
            self.b_out -= self.learning_rate * db_out
            self.W2 -= self.learning_rate * dW2
            self.b2 -= self.learning_rate * db2
            self.W_mu -= self.learning_rate * dW_mu
            self.b_mu -= self.learning_rate * db_mu
            self.W_logvar -= self.learning_rate * dW_logvar
            self.b_logvar -= self.learning_rate * db_logvar
            self.W1 -= self.learning_rate * dW1
            self.b1 -= self.learning_rate * db1

            history["total_loss"].append(total_loss)
            history["reconstruction_loss"].append(reconstruction_loss)
            history["kl_loss"].append(kl_loss)

        return history

    def encode(self, X: np.ndarray) -> np.ndarray:
        h = np.tanh(X @ self.W1 + self.b1)
        return h @ self.W_mu + self.b_mu


def train_vae_projection(X_scaled: np.ndarray) -> tuple[SimpleVAE, np.ndarray, dict[str, list[float]]]:
    vae = SimpleVAE(
        input_dim=X_scaled.shape[1],
        hidden_dim=CONFIG["vae_hidden_dim"],
        latent_dim=CONFIG["vae_latent_dim"],
        learning_rate=CONFIG["vae_learning_rate"],
        beta=CONFIG["vae_beta"],
        random_state=CONFIG["random_state"],
    )
    history = vae.fit(X_scaled, epochs=CONFIG["vae_epochs"])
    latent_points = vae.encode(X_scaled)
    log(f"Finished VAE training for {CONFIG['vae_epochs']} epochs")
    return vae, latent_points, history
