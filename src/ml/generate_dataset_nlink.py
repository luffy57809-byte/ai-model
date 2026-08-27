"""
Dataset generator for N-link arms (2-5 links), producing ONE ROW PER
JOINT rather than one row per arm - this is what makes the model
architecture generalize to any link count. Each joint's row includes its
own properties plus chain-position and cumulative-load context, so the
model learns a genuinely reusable notion of "a joint under this much
load, this far from the base" rather than fixed named slots
(shoulder/elbow) like the original 2-link-only model.

Same principles as generate_dataset.py: plain cylinder links (full
control over parameter ranges), fixed lifting axis (0,1,0), wide-enough
torque ranges to produce real failures, not just passes.
"""

import csv
import os
import time

import numpy as np

from src.urdf_generator.schema import ArmConfig, Link, Joint, JointType
from src.simulation.lift_test import run_lift_test


LINK_COUNT_RANGE = (2, 5)
LENGTH_RANGE = (0.1, 0.4)
MASS_RANGE = (0.2, 2.5)
RADIUS_RANGE = (0.02, 0.05)
TORQUE_RANGE = (1.0, 20.0)
PAYLOAD_RANGE = (0.0, 1.0)

PER_JOINT_FEATURE_COLUMNS = [
    "link_length_m", "link_mass_kg", "link_radius_m", "max_torque_nm",
    "joint_index", "num_joints_total",
    "cumulative_length_before", "cumulative_mass_before",
    "total_arm_length", "total_arm_mass", "payload_mass_kg",
]
PER_JOINT_TARGET_COLUMNS = ["sag_deg", "max_applied_torque_nm", "passes"]


def sample_config(rng: np.random.Generator) -> ArmConfig:
    n_links = int(rng.integers(LINK_COUNT_RANGE[0], LINK_COUNT_RANGE[1] + 1))

    links, joints = [], []
    parent_name = "base_link"
    for i in range(n_links):
        link_name = f"link{i}"
        joint_name = f"joint{i}"

        length_m = float(rng.uniform(*LENGTH_RANGE))
        mass_kg = float(rng.uniform(*MASS_RANGE))
        radius_m = float(rng.uniform(*RADIUS_RANGE))
        max_torque_nm = float(rng.uniform(*TORQUE_RANGE))

        links.append(Link(name=link_name, length_m=length_m, mass_kg=mass_kg, radius_m=radius_m))
        joints.append(Joint(
            name=joint_name, joint_type=JointType.REVOLUTE, parent_link=parent_name,
            child_link=link_name, axis=(0, 1, 0), lower_limit_rad=-1.57, upper_limit_rad=1.57,
            max_torque_nm=max_torque_nm,
        ))
        parent_name = link_name

    payload_mass_kg = float(rng.uniform(*PAYLOAD_RANGE))

    return ArmConfig(
        name="dataset_sample_nlink", links=links, joints=joints, payload_mass_kg=payload_mass_kg
    )


def config_to_rows(config: ArmConfig, lift_result: dict) -> list[dict]:
    """One row per joint, with cumulative-load context computed relative
    to that joint's position in the chain."""
    n = len(config.links)
    total_length = sum(l.length_m for l in config.links)
    total_mass = sum(l.mass_kg for l in config.links)

    joint_results_by_name = {jr["joint_name"]: jr for jr in lift_result["joint_results"]}

    rows = []
    cumulative_length = 0.0
    cumulative_mass = 0.0
    for i, (link, joint) in enumerate(zip(config.links, config.joints)):
        jr = joint_results_by_name[joint.name]
        row = {
            "link_length_m": link.length_m,
            "link_mass_kg": link.mass_kg,
            "link_radius_m": link.radius_m,
            "max_torque_nm": joint.max_torque_nm,
            "joint_index": i,
            "num_joints_total": n,
            "cumulative_length_before": cumulative_length,
            "cumulative_mass_before": cumulative_mass,
            "total_arm_length": total_length,
            "total_arm_mass": total_mass,
            "payload_mass_kg": config.payload_mass_kg,
            "sag_deg": jr["sag_deg"],
            "max_applied_torque_nm": jr["max_applied_torque_nm"],
            "passes": jr["passes"],
        }
        rows.append(row)
        cumulative_length += link.length_m
        cumulative_mass += link.mass_kg

    return rows


def generate_dataset(n_arms: int, seed: int = 0, output_path: str = "data/lift_test_dataset_nlink.csv"):
    rng = np.random.default_rng(seed)
    fieldnames = PER_JOINT_FEATURE_COLUMNS + PER_JOINT_TARGET_COLUMNS

    os.makedirs("data", exist_ok=True)

    start = time.time()
    errors = 0
    total_joint_rows = 0
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for i in range(n_arms):
            config = sample_config(rng)
            try:
                result = run_lift_test(config)
            except Exception:
                errors += 1
                continue

            rows = config_to_rows(config, result)
            for row in rows:
                writer.writerow(row)
            total_joint_rows += len(rows)

            if (i + 1) % 1000 == 0:
                elapsed = time.time() - start
                print(f"{i + 1}/{n_arms} arms done ({elapsed:.1f}s elapsed, "
                      f"{total_joint_rows} joint-rows so far, {errors} errors)")

    elapsed = time.time() - start
    print(f"\nDone: {n_arms} arms attempted, {errors} errors, "
          f"{total_joint_rows} joint-rows written to {output_path}, {elapsed:.1f}s total")


if __name__ == "__main__":
    generate_dataset(n_arms=3000, seed=42)
