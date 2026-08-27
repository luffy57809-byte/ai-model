"""
Classify-then-regress model for sag_deg specifically (torque predictions
from train_failure_model.py are already solid on their own - see
validate_predictions.py results - this only replaces the sag models).

Rationale (verified against real data, not assumed): sag_deg is bimodal -
56% of designs sag under 5 degrees, 25% sag over 45 degrees, only ~19%
fall in between. A single continuous regressor wastes capacity trying to
smoothly interpolate across that mostly-empty middle, and can output
physically impossible values (negative sag) on comfortably-passing
designs, as validate_predictions.py caught directly.

Fix: (1) a classifier predicts pass/fail (using the same sag_tolerance_deg
threshold as run_lift_test itself), (2) a regressor trained ONLY on the
failing subset predicts magnitude - it never has to represent the flat
near-zero region at all, so its output is a real, physically meaningful
"how bad is this failure" estimate rather than an interpolation artifact.
"""

import joblib
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, r2_score, mean_absolute_error

from src.ml.generate_dataset import PARAM_RANGES

FEATURE_COLUMNS = list(PARAM_RANGES.keys())
JOINTS = ["shoulder", "elbow"]


def train_sag_models(dataset_path: str = "data/lift_test_dataset.csv", seed: int = 0):
    df = pd.read_csv(dataset_path)
    X = df[FEATURE_COLUMNS]

    models = {}
    metrics = {}

    for joint in JOINTS:
        passes_col = f"{joint}_passes"
        sag_col = f"{joint}_sag_deg"

        # 1. Classifier: predicts pass/fail.
        y_class = df[passes_col].astype(int)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_class, test_size=0.2, random_state=seed, stratify=y_class
        )
        classifier = GradientBoostingClassifier(
            n_estimators=200, max_depth=3, learning_rate=0.05, random_state=seed
        )
        classifier.fit(X_train, y_train)
        class_acc = accuracy_score(y_test, classifier.predict(X_test))

        # 2. Regressor: trained ONLY on the failing subset, to predict
        # magnitude - never has to represent the flat near-zero region.
        failing = df[~df[passes_col].astype(bool)]
        X_fail = failing[FEATURE_COLUMNS]
        y_fail = failing[sag_col]
        Xf_train, Xf_test, yf_train, yf_test = train_test_split(
            X_fail, y_fail, test_size=0.2, random_state=seed
        )
        regressor = GradientBoostingRegressor(
            n_estimators=200, max_depth=3, learning_rate=0.05, random_state=seed
        )
        regressor.fit(Xf_train, yf_train)
        reg_r2 = r2_score(yf_test, regressor.predict(Xf_test))
        reg_mae = mean_absolute_error(yf_test, regressor.predict(Xf_test))

        models[joint] = {"classifier": classifier, "regressor": regressor}
        metrics[joint] = {
            "classifier_accuracy": class_acc,
            "regressor_r2_on_failures_only": reg_r2,
            "regressor_mae_on_failures_only": reg_mae,
            "n_failing_examples": len(failing),
        }

        print(f"{joint}: classifier_accuracy={class_acc:.4f}  "
              f"(regressor, failures-only subset [n={len(failing)}]) "
              f"R^2={reg_r2:.4f} MAE={reg_mae:.4f}")

    return models, metrics


def predict_sag(joint: str, models: dict, X: pd.DataFrame) -> float:
    """The actual inference-time combination: classify first, only regress
    magnitude if predicted to fail. Always returns a physically valid,
    non-negative value."""
    will_pass = models[joint]["classifier"].predict(X)[0]
    if will_pass:
        return 0.0
    predicted_magnitude = models[joint]["regressor"].predict(X)[0]
    return max(0.0, predicted_magnitude)


if __name__ == "__main__":
    models, metrics = train_sag_models()

    import os
    os.makedirs("data/models", exist_ok=True)
    for joint, joint_models in models.items():
        joblib.dump(joint_models["classifier"], f"data/models/{joint}_sag_classifier.joblib")
        joblib.dump(joint_models["regressor"], f"data/models/{joint}_sag_regressor.joblib")

    print("\nSag models saved to data/models/")
