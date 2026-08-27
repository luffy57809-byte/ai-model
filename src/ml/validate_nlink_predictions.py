"""
Validates the N-link models against real run_lift_test() calls on
several genuinely varied arm shapes (2, 3, 4, and 5 links) - not just
the held-out test split, same discipline as validate_predictions.py for
the original 2-link models (which caught a real negative-sag bug that
R^2/MAE alone didn't reveal).
"""

import joblib
import numpy as np
import pandas as pd

from src.urdf_generator.schema import ArmConfig, Link, Joint, JointType
from src.simulation.lift_test import run_lift_test
from src.ml.generate_dataset_nlink import PER_JOINT_FEATURE_COLUMNS


def load_models():
    return {
        "torque": joblib.load("data/models/nlink_torque_model.joblib"),
        "sag_classifier": joblib.load("data/models/nlink_sag_classifier.joblib"),
        "sag_regressor": joblib.load("data/models/nlink_sag_regressor.joblib"),
    }


def build_arm(n_links: int, link_length: float, link_mass: float, torque: float, payload: float) -> ArmConfig:
    links, joints = [], []
    parent = "base_link"
    for i in range(n_links):
        links.append(Link(name=f"link{i}", length_m=link_length, mass_kg=link_mass, radius_m=0.03))
        joints.append(Joint(
            name=f"joint{i}", joint_type=JointType.REVOLUTE, parent_link=parent,
            child_link=f"link{i}", axis=(0, 1, 0), lower_limit_rad=-1.57, upper_limit_rad=1.57,
            max_torque_nm=torque,
        ))
        parent = f"link{i}"
    return ArmConfig(name="validation_case", links=links, joints=joints, payload_mass_kg=payload)


def predict_and_compare(config: ArmConfig, models: dict, label: str):
    real_result = run_lift_test(config)
    real_by_joint = {j["joint_name"]: j for j in real_result["joint_results"]}

    total_length = sum(l.length_m for l in config.links)
    total_mass = sum(l.mass_kg for l in config.links)
    n = len(config.links)

    print(f"\n=== {label} ({n} links) ===")
    print(f"{'joint':<10} {'pred_torque':>12} {'real_torque':>12} {'pred_pass':>10} {'real_pass':>10} {'pred_sag':>10} {'real_sag':>10}")

    cumulative_length, cumulative_mass = 0.0, 0.0
    for i, (link, joint) in enumerate(zip(config.links, config.joints)):
        features = pd.DataFrame([{
            "link_length_m": link.length_m, "link_mass_kg": link.mass_kg,
            "link_radius_m": link.radius_m, "max_torque_nm": joint.max_torque_nm,
            "joint_index": i, "num_joints_total": n,
            "cumulative_length_before": cumulative_length, "cumulative_mass_before": cumulative_mass,
            "total_arm_length": total_length, "total_arm_mass": total_mass,
            "payload_mass_kg": config.payload_mass_kg,
        }])[PER_JOINT_FEATURE_COLUMNS]

        pred_torque = models["torque"].predict(features)[0]
        pred_passes = bool(models["sag_classifier"].predict(features)[0])
        pred_sag = 0.0 if pred_passes else max(0.0, models["sag_regressor"].predict(features)[0])

        real = real_by_joint[joint.name]

        print(f"{joint.name:<10} {pred_torque:>12.3f} {real['max_applied_torque_nm']:>12.3f} "
              f"{str(pred_passes):>10} {str(real['passes']):>10} {pred_sag:>10.2f} {real['sag_deg']:>10.2f}")

        cumulative_length += link.length_m
        cumulative_mass += link.mass_kg


if __name__ == "__main__":
    models = load_models()

    predict_and_compare(build_arm(2, 0.3, 1.5, 15.0, 0.5), models, "2-link, well-sized motors")
    predict_and_compare(build_arm(3, 0.25, 1.2, 6.0, 0.3), models, "3-link, moderate motors")
    predict_and_compare(build_arm(4, 0.2, 1.0, 3.0, 0.2), models, "4-link, undersized base motors")
    predict_and_compare(build_arm(5, 0.15, 0.6, 10.0, 0.1), models, "5-link, generously sized motors")
