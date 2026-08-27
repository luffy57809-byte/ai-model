"""
Loads the N-link failure-prediction models and applies them to an
ArmConfig with ANY number of links (2-5, the range the models were
trained on) - supersedes the old fixed 2-link-only predict.py.

CONFIDENCE LEVELS (from validate_nlink_predictions.py findings): torque
predictions and pass/fail classification are generally solid (92%+
classifier accuracy) but NOT perfect - validation caught one real false
failure prediction (4-link case), so pass/fail should be treated as a
strong but imperfect signal, not a guarantee. Sag MAGNITUDE remains a
low-confidence estimate, same limitation as the original 2-link models.
"""

import joblib
import pandas as pd

from src.urdf_generator.schema import ArmConfig

FEATURE_COLUMNS = [
    "link_length_m", "link_mass_kg", "link_radius_m", "max_torque_nm",
    "joint_index", "num_joints_total",
    "cumulative_length_before", "cumulative_mass_before",
    "total_arm_length", "total_arm_mass", "payload_mass_kg",
]

MIN_LINKS, MAX_LINKS = 2, 5

_models_cache = None


class PredictionError(ValueError):
    """Raised when a config is outside the trained models' supported range."""


def _load_models() -> dict:
    global _models_cache
    if _models_cache is not None:
        return _models_cache

    _models_cache = {
        "torque": joblib.load("data/models/nlink_torque_model.joblib"),
        "sag_classifier": joblib.load("data/models/nlink_sag_classifier.joblib"),
        "sag_regressor": joblib.load("data/models/nlink_sag_regressor.joblib"),
    }
    return _models_cache


def _extract_features(config: ArmConfig) -> pd.DataFrame:
    n = len(config.links)
    if not (MIN_LINKS <= n <= MAX_LINKS):
        raise PredictionError(
            f"Prediction models only support arms with {MIN_LINKS}-{MAX_LINKS} "
            f"links (a strict serial chain, one joint per link) - this config "
            f"has {n} links."
        )
    if len(config.joints) != n:
        raise PredictionError(
            f"Expected exactly one joint per link ({n} links need {n} joints) - "
            f"got {len(config.joints)} joints."
        )

    total_length = sum(l.length_m for l in config.links)
    total_mass = sum(l.mass_kg for l in config.links)

    rows = []
    cumulative_length, cumulative_mass = 0.0, 0.0
    for link, joint in zip(config.links, config.joints):
        rows.append({
            "link_length_m": link.length_m,
            "link_mass_kg": link.mass_kg,
            "link_radius_m": link.radius_m,
            "max_torque_nm": joint.max_torque_nm,
            "joint_index": len(rows),
            "num_joints_total": n,
            "cumulative_length_before": cumulative_length,
            "cumulative_mass_before": cumulative_mass,
            "total_arm_length": total_length,
            "total_arm_mass": total_mass,
            "payload_mass_kg": config.payload_mass_kg,
            "joint_name": joint.name,
        })
        cumulative_length += link.length_m
        cumulative_mass += link.mass_kg

    return pd.DataFrame(rows)


def predict_lift_test(config: ArmConfig) -> dict:
    models = _load_models()
    rows_df = _extract_features(config)
    X = rows_df[FEATURE_COLUMNS]

    result = {}
    overall_passes = True
    for i, row in rows_df.iterrows():
        joint_name = row["joint_name"]
        features = X.iloc[[i]]

        predicted_torque = float(models["torque"].predict(features)[0])
        will_pass = bool(models["sag_classifier"].predict(features)[0])
        if will_pass:
            predicted_sag = 0.0
        else:
            raw = float(models["sag_regressor"].predict(features)[0])
            predicted_sag = max(0.0, raw)

        overall_passes = overall_passes and will_pass
        result[joint_name] = {
            "predicted_max_applied_torque_nm": round(predicted_torque, 3),
            "predicted_passes": will_pass,
            "predicted_sag_deg": round(predicted_sag, 2),
            "sag_confidence": "low" if not will_pass else "high",
        }

    result["overall_predicted_passes"] = overall_passes
    result["note"] = (
        "Torque and pass/fail predictions are generally reliable (92%+ "
        "classifier accuracy) but not perfect - occasional false failure "
        "predictions are possible, especially for middle joints in longer "
        "chains. Sag magnitude for failing joints is a rough, low-confidence "
        "estimate. This is a fast approximation, not a substitute for "
        "/analyze/arm's real simulation."
    )
    return result
