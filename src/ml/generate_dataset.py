"""
Generates a labeled dataset for ML failure prediction: randomizes arm
design parameters within realistic-but-broad ranges, runs each through
the real dynamic lift test (run_lift_test), and records both inputs
(design parameters) and outputs (per-joint sag/torque from the real
physics sim) as one row per example.

Deliberately uses plain cylinder links (no mesh_id) - full control over
exact parameter ranges without needing a library of pre-varied meshes.

Deliberately spans clearly-undersized to clearly-oversized torque ratings,
so the dataset contains real failures, not just passing designs - a
dataset of all-passes teaches a model nothing about the failure boundary.

Joint axis fixed at (0,1,0) - a real lifting axis. Randomizing into
(0,0,1) roll axes would inject known-degenerate cases (see
motion_planner.py's docstring on this exact distinction) that aren't
interesting failures, just a different, uninteresting problem.
"""

import csv
import time

import numpy as np

from src.urdf_generator.schema import ArmConfig, Link, Joint, JointType
from src.simulation.lift_test import run_lift_test


PARAM_RANGES = {
    "upper_arm_length_m": (0.15, 0.5),
    "forearm_length_m": (0.1, 0.4),
    "upper_arm_mass_kg": (0.3, 3.0),
    "forearm_mass_kg": (0.2, 2.0),
    "upper_arm_radius_m": (0.02, 0.05),
    "forearm_radius_m": (0.02, 0.05),
    "shoulder_max_torque_nm": (2.0, 20.0),
    "elbow_max_torque_nm": (1.0, 12.0),
    "payload_mass_kg": (0.0, 1.0),
}


def sample_config(rng: np.random.Generator) -> tuple[ArmConfig, dict]:
    params = {
        key: float(rng.uniform(lo, hi))
        for key, (lo, hi) in PARAM_RANGES.items()
    }

    config = ArmConfig(
        name="dataset_sample",
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
    return config, params


def generate_dataset(n_samples: int, seed: int = 0, output_path: str = "data/lift_test_dataset.csv"):
    rng = np.random.default_rng(seed)

    fieldnames = list(PARAM_RANGES.keys()) + [
        "shoulder_sag_deg", "shoulder_max_applied_torque_nm", "shoulder_passes",
        "elbow_sag_deg", "elbow_max_applied_torque_nm", "elbow_passes",
        "overall_passes",
    ]

    import os
    os.makedirs("data", exist_ok=True)

    start = time.time()
    errors = 0
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for i in range(n_samples):
            config, params = sample_config(rng)
            try:
                result = run_lift_test(config)
            except Exception as exc:
                errors += 1
                continue

            joints_by_name = {j["joint_name"]: j for j in result["joint_results"]}
            row = dict(params)
            row["shoulder_sag_deg"] = joints_by_name["shoulder"]["sag_deg"]
            row["shoulder_max_applied_torque_nm"] = joints_by_name["shoulder"]["max_applied_torque_nm"]
            row["shoulder_passes"] = joints_by_name["shoulder"]["passes"]
            row["elbow_sag_deg"] = joints_by_name["elbow"]["sag_deg"]
            row["elbow_max_applied_torque_nm"] = joints_by_name["elbow"]["max_applied_torque_nm"]
            row["elbow_passes"] = joints_by_name["elbow"]["passes"]
            row["overall_passes"] = result["overall_passes"]
            writer.writerow(row)

            if (i + 1) % 500 == 0:
                elapsed = time.time() - start
                print(f"{i + 1}/{n_samples} done ({elapsed:.1f}s elapsed, {errors} errors so far)")

    elapsed = time.time() - start
    print(f"\nDone: {n_samples} samples attempted, {errors} errors, "
          f"{n_samples - errors} written to {output_path}, {elapsed:.1f}s total")


if __name__ == "__main__":
    generate_dataset(n_samples=5000, seed=42)
