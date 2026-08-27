"""
End-to-end test: a mesh link with mass concentrated at one end (like a real
link with a motor housing) should produce a DIFFERENT, more accurate torque
result than the cylinder approximation would.
"""

import trimesh
import pytest

from src.urdf_generator.schema import ArmConfig, Link, Joint, JointType
from src.analysis.mesh_processor import compute_mesh_properties
from src.analysis.torque_check import compute_static_torques


def test_mesh_link_with_offset_mass_gives_different_torque_than_cylinder_assumption():
    shaft = trimesh.creation.box(extents=[0.02, 0.02, 0.3])
    shaft.apply_translation([0, 0, 0.15])
    housing = trimesh.creation.box(extents=[0.06, 0.06, 0.06])
    housing.apply_translation([0, 0, 0.27])
    combined = trimesh.util.concatenate([shaft, housing])

    target_mass = 1.0
    mesh_props = compute_mesh_properties(combined.export(file_type="stl"), file_type="stl", target_mass_kg=target_mass)

    geometric_midpoint = 0.15
    assert mesh_props["com_offset_z_m"] > geometric_midpoint + 0.01

    mesh_link = Link(
        name="mesh_link", length_m=mesh_props["effective_length_m"], mass_kg=target_mass,
        mesh_id="test_mesh", com_offset_m=mesh_props["com_offset_z_m"],
        inertia_ixx=mesh_props["inertia_tensor"]["ixx"], inertia_iyy=mesh_props["inertia_tensor"]["iyy"],
        inertia_izz=mesh_props["inertia_tensor"]["izz"], inertia_ixy=mesh_props["inertia_tensor"]["ixy"],
        inertia_ixz=mesh_props["inertia_tensor"]["ixz"], inertia_iyz=mesh_props["inertia_tensor"]["iyz"],
    )
    cylinder_equivalent_link = Link(
        name="cylinder_link", length_m=mesh_props["effective_length_m"], mass_kg=target_mass, radius_m=0.03,
    )
    joint = Joint(
        name="j1", joint_type=JointType.REVOLUTE, parent_link="base_link",
        child_link="mesh_link", axis=(0, 1, 0), lower_limit_rad=-1.57, upper_limit_rad=1.57, max_torque_nm=100.0,
    )

    mesh_config = ArmConfig(name="mesh_test", links=[mesh_link], joints=[joint], payload_mass_kg=0.0)
    cylinder_config = ArmConfig(
        name="cyl_test", links=[cylinder_equivalent_link],
        joints=[Joint(**{**joint.model_dump(), "child_link": "cylinder_link"})], payload_mass_kg=0.0,
    )

    mesh_result = compute_static_torques(mesh_config)[0]
    cylinder_result = compute_static_torques(cylinder_config)[0]

    expected_mesh_torque = 9.81 * target_mass * mesh_props["com_offset_z_m"]
    assert mesh_result["required_torque_nm"] == pytest.approx(expected_mesh_torque, abs=0.001)
    assert mesh_result["required_torque_nm"] != cylinder_result["required_torque_nm"]
    assert mesh_result["required_torque_nm"] > cylinder_result["required_torque_nm"]
