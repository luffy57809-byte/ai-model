import pytest
import trimesh

from src.urdf_generator.schema import ArmConfig, Link, Joint, JointType
from src.analysis.mesh_processor import compute_mesh_properties
from src.analysis.mass_optimizer import optimize_link_masses
from src.analysis.torque_check import compute_static_torques


def _dumbbell_mesh_bytes():
    shaft = trimesh.creation.box(extents=[0.02, 0.02, 0.3])
    shaft.apply_translation([0, 0, 0.15])
    housing = trimesh.creation.box(extents=[0.06, 0.06, 0.06])
    housing.apply_translation([0, 0, 0.27])
    combined = trimesh.util.concatenate([shaft, housing])
    return combined.export(file_type="stl")


def test_optimizer_uses_real_mesh_com_not_length_over_two(tmp_path, monkeypatch):
    import src.urdf_generator.generator as generator_module
    mesh_dir = tmp_path / "meshes"
    mesh_dir.mkdir()
    monkeypatch.setattr(generator_module, "MESH_STORAGE_DIR", mesh_dir)

    mesh_bytes = _dumbbell_mesh_bytes()
    (mesh_dir / "dumbbell.stl").write_bytes(mesh_bytes)
    props = compute_mesh_properties(mesh_bytes, file_type="stl", target_mass_kg=1.0)

    geometric_midpoint = 0.15
    real_com = props["com_offset_z_m"]
    assert real_com > geometric_midpoint + 0.05

    mesh_link = Link(
        name="mesh_link", length_m=props["effective_length_m"], mass_kg=1.0,
        mesh_id="dumbbell", com_offset_m=real_com,
        inertia_ixx=props["inertia_tensor"]["ixx"], inertia_iyy=props["inertia_tensor"]["iyy"],
        inertia_izz=props["inertia_tensor"]["izz"], inertia_ixy=props["inertia_tensor"]["ixy"],
        inertia_ixz=props["inertia_tensor"]["ixz"], inertia_iyz=props["inertia_tensor"]["iyz"],
    )
    joint = Joint(
        name="j1", joint_type=JointType.REVOLUTE, parent_link="base_link",
        child_link="mesh_link", axis=(0, 1, 0),
        lower_limit_rad=-1.57, upper_limit_rad=1.57, max_torque_nm=5.0,
    )
    config = ArmConfig(name="mesh_opt_test", links=[mesh_link], joints=[joint], payload_mass_kg=0.0)

    safety_margin = 0.2
    min_mass_kg = 0.1
    result = optimize_link_masses(config, safety_margin=safety_margin, min_mass_kg=min_mass_kg)

    assert result["feasible"] is True
    optimized_mass = result["links"][0]["optimized_mass_kg"]

    max_mass_allowed = (joint.max_torque_nm * (1 - safety_margin)) / (9.81 * real_com)
    assert optimized_mass <= max_mass_allowed + 1e-6
    assert optimized_mass == pytest.approx(min_mass_kg, abs=1e-4)

    config.links[0].mass_kg = optimized_mass
    verification = compute_static_torques(config)
    assert verification[0]["passes"] is True
    assert verification[0]["margin_percent"] >= safety_margin * 100 - 0.1


def test_mesh_link_result_includes_reupload_note(tmp_path, monkeypatch):
    import src.urdf_generator.generator as generator_module
    mesh_dir = tmp_path / "meshes"
    mesh_dir.mkdir()
    monkeypatch.setattr(generator_module, "MESH_STORAGE_DIR", mesh_dir)

    mesh_bytes = _dumbbell_mesh_bytes()
    (mesh_dir / "dumbbell.stl").write_bytes(mesh_bytes)
    props = compute_mesh_properties(mesh_bytes, file_type="stl", target_mass_kg=1.0)

    mesh_link = Link(
        name="mesh_link", length_m=props["effective_length_m"], mass_kg=1.0,
        mesh_id="dumbbell", com_offset_m=props["com_offset_z_m"],
        inertia_ixx=props["inertia_tensor"]["ixx"], inertia_iyy=props["inertia_tensor"]["iyy"],
        inertia_izz=props["inertia_tensor"]["izz"], inertia_ixy=props["inertia_tensor"]["ixy"],
        inertia_ixz=props["inertia_tensor"]["ixz"], inertia_iyz=props["inertia_tensor"]["iyz"],
    )
    joint = Joint(
        name="j1", joint_type=JointType.REVOLUTE, parent_link="base_link",
        child_link="mesh_link", axis=(0, 1, 0),
        lower_limit_rad=-1.57, upper_limit_rad=1.57, max_torque_nm=100.0,
    )
    config = ArmConfig(name="mesh_opt_note_test", links=[mesh_link], joints=[joint], payload_mass_kg=0.0)

    result = optimize_link_masses(config)
    assert "note" in result["links"][0]
    assert "re-upload" in result["links"][0]["note"].lower()
