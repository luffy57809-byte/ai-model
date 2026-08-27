"""
Trains per-joint failure-prediction models on the N-link dataset - ONE
set of models, usable for any joint in any arm with 2-5 links, unlike
the original 2-link-only models that only understood "shoulder"/"elbow"
as fixed named slots.

Same modeling choices as before, now applied to the joint-indexed
features: torque via a straightforward GradientBoostingRegressor
(worked well previously - R^2=0.96/0.78 on the 2-link data), sag via
classify-then-regress (the 2-link experiments found sag is bimodal and
the magnitude regressor stays a rough estimate even after that split -
same expected limitation here, not fixed by generalizing).
"""

import joblib
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, accuracy_score

from src.ml.generate_dataset_nlink import PER_JOINT_FEATURE_COLUMNS


def train_torque_model(df: pd.DataFrame, seed: int = 0):
    X = df[PER_JOINT_FEATURE_COLUMNS]
    y = df["max_applied_torque_nm"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed)

    model = GradientBoostingRegressor(n_estimators=200, max_depth=3, learning_rate=0.05, random_state=seed)
    model.fit(X_train, y_train)

    r2 = r2_score(y_test, model.predict(X_test))
    mae = mean_absolute_error(y_test, model.predict(X_test))
    print(f"torque model: R^2={r2:.4f}  MAE={mae:.4f}")
    return model


def train_sag_models(df: pd.DataFrame, seed: int = 0):
    X = df[PER_JOINT_FEATURE_COLUMNS]
    y_class = df["passes"].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_class, test_size=0.2, random_state=seed, stratify=y_class
    )
    classifier = GradientBoostingClassifier(n_estimators=200, max_depth=3, learning_rate=0.05, random_state=seed)
    classifier.fit(X_train, y_train)
    class_acc = accuracy_score(y_test, classifier.predict(X_test))

    failing = df[~df["passes"].astype(bool)]
    X_fail = failing[PER_JOINT_FEATURE_COLUMNS]
    y_fail = failing["sag_deg"]
    Xf_train, Xf_test, yf_train, yf_test = train_test_split(X_fail, y_fail, test_size=0.2, random_state=seed)

    regressor = GradientBoostingRegressor(n_estimators=200, max_depth=3, learning_rate=0.05, random_state=seed)
    regressor.fit(Xf_train, yf_train)
    reg_r2 = r2_score(yf_test, regressor.predict(Xf_test))
    reg_mae = mean_absolute_error(yf_test, regressor.predict(Xf_test))

    print(f"sag classifier: accuracy={class_acc:.4f}  "
          f"(regressor, failures-only [n={len(failing)}]) R^2={reg_r2:.4f} MAE={reg_mae:.4f}")

    return classifier, regressor


if __name__ == "__main__":
    df = pd.read_csv("data/lift_test_dataset_nlink.csv")

    torque_model = train_torque_model(df)
    sag_classifier, sag_regressor = train_sag_models(df)

    import os
    os.makedirs("data/models", exist_ok=True)
    joblib.dump(torque_model, "data/models/nlink_torque_model.joblib")
    joblib.dump(sag_classifier, "data/models/nlink_sag_classifier.joblib")
    joblib.dump(sag_regressor, "data/models/nlink_sag_regressor.joblib")

    print("\nModels saved to data/models/")
