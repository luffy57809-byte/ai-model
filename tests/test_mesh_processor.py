import trimesh
import pytest
from src.analysis.mesh_processor import compute_mesh_properties, MeshProcessingError


def _mesh_to_stl_bytes(mesh):
    return mesh.export(file_type="stl")


def test_box_mass_properties_match_analytical_formula():
    box = trimesh.creation.box(extents=[0.1, 0.1, 0.1])
    result = compute_mesh_properties(_mesh_to_stl_bytes(box), file_type="stl", target_mass_kg=1.0)
    assert result["mass_kg"] == pytest.approx(1.0, abs=1e-6)
    assert result["is_watertight"] is True
    expected = (1 / 12) * 1.0 * (0.1**2 + 0.1**2)
    assert result["inertia_tensor"]["ixx"] == pytest.approx(expected, rel=1e-4)
    assert result["inertia_tensor"]["ixy"] == pytest.approx(0.0, abs=1e-6)


def test_cylinder_mass_properties_match_existing_analytical_formula():
    radius, length, mass = 0.03, 0.3, 1.5
    cyl = trimesh.creation.cylinder(radius=radius, height=length, sections=64)
    result = compute_mesh_properties(_mesh_to_stl_bytes(cyl), file_type="stl", target_mass_kg=mass)
    ixx_expected = (1 / 12) * mass * (3 * radius**2 + length**2)
    izz_expected = 0.5 * mass * radius**2
    assert result["inertia_tensor"]["ixx"] == pytest.approx(ixx_expected, rel=1e-3)
    assert result["inertia_tensor"]["izz"] == pytest.approx(izz_expected, rel=3e-3)


def test_asymmetric_shape_produces_nonzero_off_diagonal_terms_with_correct_sign():
    box = trimesh.creation.box(extents=[0.1, 0.2, 0.3])
    box.apply_transform(trimesh.transformations.rotation_matrix(0.3, [0, 0, 1]))
    result = compute_mesh_properties(_mesh_to_stl_bytes(box), file_type="stl", target_mass_kg=1.0)
    assert abs(result["inertia_tensor"]["ixy"]) > 1e-6
    assert result["inertia_tensor"]["ixz"] == pytest.approx(0.0, abs=1e-6)
    assert result["inertia_tensor"]["iyz"] == pytest.approx(0.0, abs=1e-6)


def test_open_surface_mesh_is_rejected():
    flat_mesh = trimesh.Trimesh(vertices=[[0,0,0],[1,0,0],[0,1,0]], faces=[[0,1,2]])
    with pytest.raises(MeshProcessingError, match="non-positive volume"):
        compute_mesh_properties(flat_mesh.export(file_type="stl"), file_type="stl", target_mass_kg=1.0)


def test_garbage_bytes_are_rejected():
    with pytest.raises(MeshProcessingError):
        compute_mesh_properties(b"not an stl file", file_type="stl", target_mass_kg=1.0)


def test_zero_or_negative_target_mass_rejected():
    box = trimesh.creation.box(extents=[0.1, 0.1, 0.1])
    with pytest.raises(MeshProcessingError, match="must be positive"):
        compute_mesh_properties(_mesh_to_stl_bytes(box), file_type="stl", target_mass_kg=0)


def test_effective_length_matches_bounding_box_z_extent():
    box = trimesh.creation.box(extents=[0.05, 0.05, 0.4])
    result = compute_mesh_properties(_mesh_to_stl_bytes(box), file_type="stl", target_mass_kg=1.0)
    assert result["effective_length_m"] == pytest.approx(0.4, rel=1e-3)
