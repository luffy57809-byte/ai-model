import trimesh
import pytest
from src.urdf_generator.schema import ArmConfig, Link, Joint, JointType
from src.analysis.mesh_processor import compute_mesh_properties
from src.simulation.lift_test import run_lift_test, MeshDynamicsError


def _build_single_mesh_link_config(mesh_bytes, mesh_id, target_mass, max_torque=100.0):
    props = compute_mesh_properties(mesh_bytes, file_type="stl", target_mass_kg=target_mass)
    link = Link(
        name="mesh_link", length_m=props["effective_length_m"], mass_kg=target_mass,
        mesh_id=mesh_id, com_offset_m=props["com_offset_z_m"],
        inertia_ixx=props["inertia_tensor"]["ixx"], inertia_iyy=props["inertia_tensor"]["iyy"],
        inertia_izz=props["inertia_tensor"]["izz"], inertia_ixy=props["inertia_tensor"]["ixy"],
        inertia_ixz=props["inertia_tensor"]["ixz"], inertia_iyz=props["inertia_tensor"]["iyz"],
    )
    joint = Joint(
        name="j1", joint_type=JointType.REVOLUTE, parent_link="base_link",
        child_link="mesh_link", axis=(0, 1, 0),
        lower_limit_rad=-1.57, upper_limit_rad=1.57, max_torque_nm=max_torque,
    )
    return ArmConfig(name="mesh_test_arm", links=[link], joints=[joint], payload_mass_kg=0.0), props


@pytest.fixture
def dumbbell_mesh_bytes(tmp_path, monkeypatch):
    import src.urdf_generator.generator as generator_module
    mesh_dir = tmp_path / "meshes"
    mesh_dir.mkdir()
    monkeypatch.setattr(generator_module, "MESH_STORAGE_DIR", mesh_dir)

    shaft = trimesh.creation.box(extents=[0.02, 0.02, 0.3])
    shaft.apply_translation([0, 0, 0.15])
    housing = trimesh.creation.box(extents=[0.06, 0.06, 0.06])
    housing.apply_translation([0, 0, 0.27])
    combined = trimesh.util.concatenate([shaft, housing])
    assert combined.is_convex is False

    stl_bytes = combined.export(file_type="stl")
    (mesh_dir / "dumbbell_test.stl").write_bytes(stl_bytes)
    return stl_bytes


def test_dynamic_torque_matches_static_calculation_for_mesh_link(dumbbell_mesh_bytes):
    target_mass = 0.5
    config, props = _build_single_mesh_link_config(dumbbell_mesh_bytes, "dumbbell_test", target_mass)
    result = run_lift_test(config, sim_seconds=2.0)
    applied_torque = result["joint_results"][0]["max_applied_torque_nm"]
    expected_torque = 9.81 * target_mass * props["com_offset_z_m"]
    assert applied_torque == pytest.approx(expected_torque, rel=0.02)


def test_significantly_tilted_inertia_raises_rather_than_silently_misapplying():
    import tempfile
    import src.urdf_generator.generator as generator_module
    from pathlib import Path

    box = trimesh.creation.box(extents=[0.1, 0.2, 0.3])
    box.apply_transform(trimesh.transformations.rotation_matrix(0.5, [0, 0, 1]))
    stl_bytes = box.export(file_type="stl")

    with tempfile.TemporaryDirectory() as tmp_dir:
        mesh_dir = Path(tmp_dir)
        original = generator_module.MESH_STORAGE_DIR
        generator_module.MESH_STORAGE_DIR = mesh_dir
        try:
            (mesh_dir / "tilted_test.stl").write_bytes(stl_bytes)
            config, _ = _build_single_mesh_link_config(stl_bytes, "tilted_test", 1.0)
            with pytest.raises(MeshDynamicsError, match="tilted principal-axis"):
                run_lift_test(config, sim_seconds=0.5)
        finally:
            generator_module.MESH_STORAGE_DIR = original
