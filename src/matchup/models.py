"""Plate-appearance outcome models with one shared interface.

Every model returns an (n, 7) probability matrix with columns in ``OUTCOMES`` order.
Two baselines set the bar: league average, and log5 (the classic odds-ratio blend of
batter, pitcher and league rates). The machine-learning models have to beat both.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.preprocessing import StandardScaler

from matchup.config import OUTCOMES, RANDOM_STATE
from matchup.features import FEATURES

logger = logging.getLogger(__name__)
N_CLASSES = len(OUTCOMES)
LOG5_COLS = [f"log5_{o}" for o in OUTCOMES]


def _align(proba: np.ndarray, classes) -> np.ndarray:
    """Place predicted columns into OUTCOMES order (a class absent from training gets 0)."""
    out = np.zeros((proba.shape[0], N_CLASSES))
    out[:, np.asarray(classes, dtype=int)] = proba
    out = np.clip(out, 1e-6, None)
    return out / out.sum(axis=1, keepdims=True)


class LeagueAverage:
    name = "League average"

    def fit(self, X, y):
        self.rates = np.bincount(y, minlength=N_CLASSES) / len(y)
        return self

    def predict_proba(self, X):
        return np.tile(self.rates, (len(X), 1))


class Log5:
    name = "Log5"

    def fit(self, X, y):
        return self

    def predict_proba(self, X):
        p = np.clip(X[LOG5_COLS].to_numpy(dtype=float), 1e-6, None)
        return p / p.sum(axis=1, keepdims=True)


class RandomForest:
    name = "Random forest"

    def __init__(self, n_estimators: int = 150, min_samples_leaf: int = 80, features=FEATURES):
        self.features = list(features)
        # Plate-appearance outcomes are mostly noise, so trees need big leaves (80+ PAs) to
        # give smooth probabilities; half-size bootstrap samples add variety and speed.
        self.model = RandomForestClassifier(
            n_estimators=n_estimators, min_samples_leaf=min_samples_leaf, max_samples=0.5,
            max_features="sqrt", random_state=RANDOM_STATE, n_jobs=-1,
        )  # fmt: skip

    def fit(self, X, y):
        self.model.fit(X[self.features], y)
        return self

    def predict_proba(self, X):
        return _align(self.model.predict_proba(X[self.features]), self.model.classes_)


class GradientBoosting:
    name = "Gradient boosting"

    def __init__(self, max_iter: int = 500, features=FEATURES):
        self.features = list(features)
        # Shallow trees, large leaves and a slow learning rate: tuned for noisy outcomes,
        # where a flexible booster otherwise memorizes luck.
        self.model = HistGradientBoostingClassifier(
            max_iter=max_iter, learning_rate=0.03, max_depth=3, min_samples_leaf=400,
            l2_regularization=5.0, early_stopping=True, validation_fraction=0.15,
            n_iter_no_change=30, random_state=RANDOM_STATE,
        )  # fmt: skip

    def fit(self, X, y):
        self.model.fit(X[self.features], y)
        return self

    def predict_proba(self, X):
        return _align(self.model.predict_proba(X[self.features]), self.model.classes_)


class NeuralNetwork:
    name = "Neural network"

    def __init__(self, epochs: int = 30, batch_size: int = 256, features=FEATURES):
        self.features, self.epochs, self.batch_size = list(features), epochs, batch_size
        self.history: dict[str, list[float]] = {}

    def _build(self, n_features: int):
        from tensorflow import keras
        from tensorflow.keras import layers

        model = keras.Sequential([
            layers.Input(shape=(n_features,)),
            layers.Dense(64, activation="relu"),
            layers.Dropout(0.1),
            layers.Dense(32, activation="relu"),
            layers.Dense(N_CLASSES, activation="softmax"),
        ])  # fmt: skip
        model.compile(optimizer=keras.optimizers.Adam(1e-3),
                      loss="sparse_categorical_crossentropy")  # fmt: skip
        return model

    def fit(self, X, y):
        from tensorflow import keras

        keras.utils.set_random_seed(RANDOM_STATE)
        self.scaler = StandardScaler()
        Xs = self.scaler.fit_transform(X[self.features]).astype(np.float32)
        self.model = self._build(Xs.shape[1])
        stop = keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True)
        hist = self.model.fit(Xs, np.asarray(y, dtype=np.int32), epochs=self.epochs,
                              batch_size=self.batch_size, validation_split=0.15,
                              callbacks=[stop], verbose=0)  # fmt: skip
        self.history = {k: [float(v) for v in vals] for k, vals in hist.history.items()}
        return self

    def predict_proba(self, X):
        Xs = self.scaler.transform(X[self.features]).astype(np.float32)
        # Direct calls avoid Keras re-tracing predict() for every new input size.
        parts = [np.asarray(self.model(Xs[i : i + 65536], training=False))
                 for i in range(0, len(Xs), 65536)] or [np.zeros((0, N_CLASSES))]  # fmt: skip
        return _align(np.vstack(parts), range(N_CLASSES))


def make_models(config, include_baselines: bool = True) -> list:
    models = [LeagueAverage(), Log5()] if include_baselines else []
    models += [RandomForest(config.n_estimators, config.rf_min_samples_leaf),
               GradientBoosting(config.hgb_max_iter)]  # fmt: skip
    if config.run_nn:
        models.append(NeuralNetwork(config.nn_epochs, config.nn_batch_size))
    return models


def fit_timed(model, X, y) -> float:
    start = time.time()
    model.fit(X, y)
    return time.time() - start


# ---------------------------------------------------------------------- persistence


def save_model(model, directory: Path, metadata: dict) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    for old in directory.glob("model.*"):
        old.unlink()
    if isinstance(model, NeuralNetwork):
        model.model.save(directory / "model.keras")
        joblib.dump({"scaler": model.scaler, "features": model.features,
                     "history": model.history}, directory / "model.nn.joblib")  # fmt: skip
    else:
        joblib.dump(model, directory / "model.joblib")
    (directory / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def load_model(directory: Path):
    meta_path = directory / "metadata.json"
    if not meta_path.exists():
        raise FileNotFoundError(
            f"No trained model in {directory}. Run the training pipeline first "
            "(PyCharm: 'Train models', or `mlb-matchup train`)."
        )
    metadata = json.loads(meta_path.read_text(encoding="utf-8"))
    if (directory / "model.keras").exists():
        from tensorflow import keras

        extra = joblib.load(directory / "model.nn.joblib")
        model = NeuralNetwork(features=extra["features"])
        model.model = keras.models.load_model(directory / "model.keras")
        model.scaler, model.history = extra["scaler"], extra["history"]
    else:
        model = joblib.load(directory / "model.joblib")
    return model, metadata


def predict_frame(model, X: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(model.predict_proba(X), columns=list(OUTCOMES), index=X.index)
