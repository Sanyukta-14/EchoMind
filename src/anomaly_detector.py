import numpy as np
import pandas as pd

from src.predictor import (
    FEATURES,
    make_training_data,
)


def detect_anomalies(
    df,
    model,
    normal_end,
    window=5,
    threshold=2.5,
):
    """
    Detect anomalies using prediction errors calibrated on normal data.

    Parameters
    ----------
    df : pandas.DataFrame
        Simulated world observations.
    model : trained model
        Model with a predict(X) method.
    normal_end : int
        First step considered abnormal. Only earlier observations
        are used to calibrate the normal error scale.
    window : int
        Number of past observations used for prediction.
    threshold : float
        Minimum anomaly-score threshold.

    Returns
    -------
    result : pandas.DataFrame
        Original observations after the initial window, with
        anomaly_score and is_anomaly columns.
    calibrated_threshold : float
        Threshold used to flag anomalies.
    """

    # Validate the input.
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame.")

    if not FEATURES:
        raise ValueError("FEATURES must contain at least one variable.")

    missing_features = [
        feature for feature in FEATURES
        if feature not in df.columns
    ]

    if missing_features:
        raise ValueError(
            f"Missing required features: {missing_features}"
        )

    if not isinstance(window, (int, np.integer)) or window < 1:
        raise ValueError("window must be a positive integer.")

    if len(df) <= window:
        raise ValueError(
            "The dataset must contain more rows than the window size."
        )

    if not 0 <= normal_end <= len(df):
        raise ValueError(
            "normal_end must be between 0 and the number of rows."
        )

    if not np.isfinite(threshold) or threshold < 0:
        raise ValueError("threshold must be a finite, non-negative number.")

    # Ensure the features contain numeric values.
    numeric_df = df.copy()

    for feature in FEATURES:
        numeric_df[feature] = pd.to_numeric(
            numeric_df[feature],
            errors="coerce",
        )

    feature_values = numeric_df[FEATURES].to_numpy(
        dtype=np.float64,
        copy=True,
    )

    if not np.isfinite(feature_values).all():
        raise ValueError(
            "Feature data contains missing or non-finite values. "
            "Check the simulator output before detecting anomalies."
        )

    # Create prediction examples from historical observations.
    X, actual = make_training_data(numeric_df, window)

    # Convert explicitly to numeric arrays to avoid object-dtype errors.
    X = np.asarray(X, dtype=np.float64)
    actual = np.asarray(actual, dtype=np.float64)

    if actual.ndim == 1:
        actual = actual.reshape(-1, 1)

    # Predict and validate the output.
    predictions = np.asarray(
        model.predict(X),
        dtype=np.float64,
    )

    if predictions.ndim == 1:
        predictions = predictions.reshape(-1, 1)

    if predictions.shape != actual.shape:
        raise ValueError(
            "Prediction shape does not match actual values. "
            f"Predictions: {predictions.shape}; "
            f"actual values: {actual.shape}."
        )

    if not np.isfinite(predictions).all():
        raise ValueError(
            "The prediction model returned non-finite values."
        )

    # Absolute prediction errors for each feature.
    errors = np.abs(actual - predictions)

    if not np.isfinite(errors).all():
        raise ValueError(
            "Prediction errors contain non-finite values."
        )

    # Match prediction rows to their original simulation steps.
    prediction_steps = np.arange(window, len(df))
    normal_mask = prediction_steps < normal_end

    if normal_mask.sum() < 10:
        raise ValueError(
            "Need at least 10 normal prediction examples. "
            "Increase normal_end or reduce window."
        )

    # Calibrate each feature's error scale using normal observations.
    normal_errors = np.asarray(
        errors[normal_mask],
        dtype=np.float64,
    )

    scales = np.asarray(
        np.std(normal_errors, axis=0, dtype=np.float64),
        dtype=np.float64,
    )

    # Avoid division by zero for constant or nearly constant errors.
    scales = np.where(
        np.isfinite(scales) & (scales >= 1e-6),
        scales,
        1.0,
    )

    # Standardized error score for each prediction row.
    normalized_errors = errors / scales
    scores = np.mean(normalized_errors, axis=1)

    if not np.isfinite(scores).all():
        raise ValueError(
            "Anomaly scores contain non-finite values."
        )

    # Calibrate the threshold from normal prediction errors.
    normal_scores = scores[normal_mask]

    calibrated_threshold = max(
        float(threshold),
        float(np.quantile(normal_scores, 0.95)),
    )

    # Preserve the original columns and add anomaly results.
    result = df.iloc[window:].copy()
    result["anomaly_score"] = scores
    result["is_anomaly"] = scores > calibrated_threshold

    return result, calibrated_threshold