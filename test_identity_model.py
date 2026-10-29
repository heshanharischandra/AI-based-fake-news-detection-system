
from pathlib import Path

import joblib
import pandas as pd


# ============================================================
# PROJECT PATHS
# ============================================================

ROOT = Path(__file__).resolve().parent

MODEL_PATH = (
    ROOT / "models" / "identity_matcher_pilot.joblib"
)

DATA_PATH = (
    ROOT / "results" / "identity_model_training_data.csv"
)

DEFAULT_FEATURES = [
    "name_handle_similarity",
    "domain_handle_similarity",
    "name_in_handle",
    "domain_in_handle",
]


# ============================================================
# EXTRACT MODEL FROM SAVED FILE
# ============================================================

def extract_model(saved):

    if hasattr(saved, "predict"):
        print("Saved format: Direct model")
        return saved

    if not isinstance(saved, dict):
        raise TypeError(
            "Unsupported saved model type: "
            + type(saved).__name__
        )

    print("Saved format: Dictionary")
    print("Available dictionary keys:", list(saved.keys()))

    possible_keys = [
        "model",
        "pipeline",
        "classifier",
        "estimator",
        "trained_model",
        "model_pipeline",
        "identity_model",
    ]

    for key in possible_keys:
        candidate = saved.get(key)

        if hasattr(candidate, "predict"):
            print("Model found under key:", key)
            return candidate

    # Automatically find a model if another key was used.
    matches = [
        (key, value)
        for key, value in saved.items()
        if hasattr(value, "predict")
    ]

    if len(matches) == 1:
        key, model = matches[0]
        print("Model found under key:", key)
        return model

    raise ValueError(
        "No trained estimator found in dictionary. "
        "Check the printed dictionary keys."
    )


# ============================================================
# IDENTIFY THE MODEL'S INPUT FEATURES
# ============================================================

def get_feature_names(saved, model):

    if hasattr(model, "feature_names_in_"):
        return list(model.feature_names_in_)

    if isinstance(saved, dict):

        for key in [
            "feature_columns",
            "feature_names",
            "features",
            "input_features",
        ]:
            value = saved.get(key)

            if (
                isinstance(value, (list, tuple))
                and all(
                    isinstance(name, str)
                    for name in value
                )
            ):
                return list(value)

    print(
        "Feature metadata not found. "
        "Using the known pilot feature columns."
    )

    return DEFAULT_FEATURES


# ============================================================
# CONVERT PREDICTIONS TO READABLE LABELS
# ============================================================

def readable_label(value):

    text = str(value).strip()

    if text in {"1", "1.0", "same_publisher"}:
        return "same_publisher"

    if text in {"0", "0.0", "different_publisher"}:
        return "different_publisher"

    return text


# ============================================================
# MAIN TEST
# ============================================================

def main():

    print("=" * 60)
    print("IDENTITY MATCHING MODEL - PREDICTION SMOKE TEST")
    print("=" * 60)

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found: {MODEL_PATH}"
        )

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Training CSV not found: {DATA_PATH}"
        )

    # Load only a locally created, trusted model artifact.
    saved = joblib.load(MODEL_PATH)

    # FIX: Extract estimator from dictionary.
    model = extract_model(saved)

    # Find the correct model features.
    features = get_feature_names(saved, model)

    # Load training feature data.
    data = pd.read_csv(DATA_PATH)

    missing = [
        name for name in features
        if name not in data.columns
    ]

    if missing:
        raise ValueError(
            f"Missing CSV feature columns: {missing}\n"
            f"Available columns: {list(data.columns)}"
        )

    if data.empty:
        raise ValueError("Training data CSV is empty.")

    samples = data[features].head(5).copy()

    samples = samples.apply(
        pd.to_numeric,
        errors="raise"
    )

    if samples.isna().any().any():
        raise ValueError(
            "The selected samples contain missing feature values."
        )

    # Run model predictions.
    predictions = model.predict(samples)

    print()
    print("MODEL LOADED SUCCESSFULLY")
    print("-" * 60)

    print("Model type:", type(model).__name__)
    print("Training data rows:", len(data))
    print("Input features:", features)
    print("Samples tested:", len(samples))

    print()
    print("EXPERIMENTAL PREDICTIONS")
    print("-" * 60)

    for index, prediction in enumerate(
        predictions,
        start=1
    ):
        label = readable_label(prediction)

        print(
            f"Sample {index}: {label}"
        )

    print()
    print("=" * 60)
    print("MODEL PREDICTION TEST COMPLETED")
    print("=" * 60)

    print(
        "WARNING: Predictions above use training examples."
    )
    print(
        "This only confirms that the saved model works."
    )
    print(
        "It does not establish model accuracy."
    )


if __name__ == "__main__":
    main()
