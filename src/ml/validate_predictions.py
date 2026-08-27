"""
Sanity-checks the trained failure-prediction models: for a handful of
real, meaningful arm designs (not random dataset samples), compares each
model's prediction against a FRESH real run_lift_test() call - the honest
test of whether these predictions are trustworthy enough to build on top
of, rather than just trusting the held-out R^2/MAE numbers in isolation.
"""

import joblib
import pandas as pd

from src.urdf_generator.schema import ArmConfig, Link, Joint, JointType
from src.simulation.lift_test import run_lift_test
from src.ml.generate_dataset import PARAM_RANGES

FEATURE_COLUMNS = list(PARAM_RANGES.keys())
TARGET_COLUMNS = [
    "shoulder_sag_deg",
    "shoulder_max_applied_torque_nm",
    "elbow_sag_deg",
    "elbow_max_applied_torque_nm",
]


def load_models():
    return {
        target: joblib.load(f"data/models/{target}_model.joblib")
        for target in TARGET_COLUMNS
    }


# Real, meaningful test cases - not random dataset samples:
TEST_CASES = {
    "reach_test (known-good baseline used throughout this project)": dict(
        upper_arm_length_m=0.3, forearm_length_m=0.25,
        upper_arm_mass_kg=1.5, forearm_mass_kg=1.0,
        upper_arm_radius_m=0.03, forearm_radius_m=0.03,
        shoulder_max_torque_nm=15.0, elbow_max_torque_nm=8.0,
        payload_mass_kg=0.5,
    ),
    "deliberately undersized shoulder motor": dict(
        upper_arm_length_m=0.3, forearm_length_m=0.25,
        upper_arm_mass_kg=1.5, forearm_mass_kg=1.0,
        upper_arm_radius_m=0.03, forearm_radius_m=0.03,
        shoulder_max_torque_nm=2.5, elbow_max_torque_nm=8.0,
        payload_mass_kg=0.5,
    ),
    "deliberately oversized motors (should pass comfortably)": dict(
        upper_arm_length_m=0.3, forearm_length_m=0.25,
        upper_arm_mass_kg=1.5, forearm_mass_kg=1.0,
        upper_arm_radius_m=0.03, forearm_radius_m=0.03,
        shoulder_max_torque_nm=20.0, elbow_max_torque_nm=12.0,
        payload_mass_kg=0.2,
    ),
    "long, heavy arm with heavy payload": dict(
        upper_arm_length_m=0.48, forearm_length_m=0.38,
        upper_arm_mass_kg=2.8, forearm_mass_kg=1.8,
        upper_arm_radius_m=0.045, forearm_radius_m=0.045,
        shoulder_max_torque_nm=10.0, elbow_max_torque_nm=5.0,
        payload_mass_kg=0.9,
    ),
}


def build_config(params: dict) -> ArmConfig:
    return ArmConfig(
        name="validation_case",
        links=[
            Link(name="upper_arm", length_m=params["upper_arm_length_m"],
                 mass_kg=params["upper_arm_mass_kg"], radius_m=params["upper_arm_radius_m"]),
            Link(name="forearm", length_m=params["forearm_length_m"],
                 mass_kg=params["forearm_mass_kg"], radius_m=params["forearm_radius_m"]),
        ],
        joints=[
            Joint(name="shoulder", joint_type=JointType.REVOLUTE, parent_link="base_link",
                  child_link="upper_arm", axis=(0, 1, 0), lower_limit_rad=-1.57,
                  upper_limit_rad=1.57, max_torque_nm=params["shoulder_max_torque_nm"]),
            Joint(name="elbow", joint_type=JointType.REVOLUTE, parent_link="upper_arm",
                  child_link="forearm", axis=(0, 1, 0), lower_limit_rad=-2.5,
                  upper_limit_rad=2.5, max_torque_nm=params["elbow_max_torque_nm"]),
        ],
        payload_mass_kg=params["payload_mass_kg"],
    )


if __name__ == "__main__":
    models = load_models()

    for label, params in TEST_CASES.items():
        config = build_config(params)
        real_result = run_lift_test(config)
        real_by_joint = {j["joint_name"]: j for j in real_result["joint_results"]}

        X = pd.DataFrame([params])[FEATURE_COLUMNS]
        predicted = {target: models[target].predict(X)[0] for target in TARGET_COLUMNS}

        print(f"\n=== {label} ===")
        print(f"{'metric':<35} {'predicted':>12} {'real':>12} {'abs error':>12}")
        for target in TARGET_COLUMNS:
            real_key = target.replace("shoulder_", "").replace("elbow_", "")
            joint_name = "shoulder" if target.startswith("shoulder") else "elbow"
            real_value = real_by_joint[joint_name][real_key]
            pred_value = predicted[target]
            print(f"{target:<35} {pred_value:>12.3f} {real_value:>12.3f} {abs(pred_value - real_value):>12.3f}")
