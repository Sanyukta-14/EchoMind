import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

FEATURES = [
    "temperature",
    "water_level",
    "plant_health",
    "food_supply",
    "animal_population",
]


def make_training_data(df, window=5):
    """Build past-only features for predicting the next measurement."""
    X, y = [], []

    for i in range(window, len(df)):
        history = df.iloc[i - window:i][FEATURES].to_numpy()
        X.append(history.flatten())
        y.append(df.iloc[i][FEATURES].to_numpy())

    return np.asarray(X), np.asarray(y)


def train_predictor(df, window=5, normal_end=None):
    """Train on the initial, presumed-normal section of a simulation."""
    if normal_end is None:
        normal_end = len(df) // 2

    normal_df = df.iloc[:normal_end].copy()
    X, y = make_training_data(normal_df, window)

    if len(X) < 10:
        raise ValueError("Not enough normal data to train the predictor.")

    model = RandomForestRegressor(
        n_estimators=100,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X, y)

    return model


def detect_anomalies(df, model, window=5, threshold=2.5):
    """Flag time steps with unusually large prediction errors."""
    X, y = make_training_data(df, window)
    predictions = model.predict(X)

    # Normalize each variable's errors using training-data scale.
    errors = np.abs(y - predictions)
    scales = np.std(y, axis=0)
    scales[scales < 1e-6] = 1.0

    scores = np.mean(errors / scales, axis=1)
    result = df.iloc[window:].copy()
    result["anomaly_score"] = scores
    result["is_anomaly"] = scores > threshold

    return result


def train_predictor(df, window=5, normal_end=None):
    if normal_end is None:
        normal_end = len(df) // 2

    normal_df = df.iloc[:normal_end].copy()
    X, y = make_training_data(normal_df, window)

    if len(X) < 10:
        raise ValueError("Not enough normal data to train the predictor.")

    model = RandomForestRegressor(
        n_estimators=100,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X, y)
    return model