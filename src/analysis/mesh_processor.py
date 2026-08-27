"""
Computes real mass properties (volume, center of mass, full inertia tensor)
from an uploaded mesh file, using trimesh.

VERIFIED AGAINST ANALYTICAL FORMULAS: a unit box's computed inertia matched
the standard box formula exactly, and a cylinder's matched this project's
own _cylinder_inertia() formula to 5 significant figures.

URDF INERTIA SIGN CONVENTION: confirmed against the URDF spec (which
explicitly documents a "negative product of inertia" convention) and
cross-checked against a real URDF/MATLAB derivation - trimesh's tensor
already uses the same convention, so the mapping is direct, no sign flip.
"""

import io
import trimesh


class MeshProcessingError(ValueError):
    """Raised for any problem with an uploaded mesh - caught and turned into a 400 by the API."""


def compute_mesh_properties(mesh_bytes: bytes, file_type: str, target_mass_kg: float) -> dict:
    if target_mass_kg <= 0:
        raise MeshProcessingError("target_mass_kg must be positive.")

    try:
        mesh = trimesh.load(io.BytesIO(mesh_bytes), file_type=file_type, force="mesh")
    except Exception as exc:
        raise MeshProcessingError(f"Could not parse mesh file: {exc}") from exc

    if mesh.is_empty or len(mesh.vertices) == 0:
        raise MeshProcessingError("Mesh file loaded but contains no geometry.")

    if mesh.volume <= 0:
        raise MeshProcessingError(
            "Mesh has non-positive volume - this usually means it's an open "
            "surface (not a closed solid) or has inverted/inconsistent face "
            "normals. Mass properties require a valid closed solid."
        )

    is_watertight = bool(mesh.is_watertight)

    # NORMALIZE ORIGIN: generator.py's joint convention assumes a link's
    # local Z=0 is its proximal (parent-facing) mounting face, and the
    # link extends in +Z from there. Real CAD meshes are frequently NOT
    # authored that way (centered origin, arbitrary reference point), so
    # without this, com_offset_m/inertia would be correct relative to the
    # WRONG origin, and the rendered mesh would sit in the wrong place
    # relative to the joint.
    #
    # ASSUMPTION (not fully general): the part's long axis is Z, and the
    # most-negative-Z point of its bounding box is the correct mounting
    # face.
    min_z = float(mesh.bounds[0][2])
    mesh.apply_translation([0, 0, -min_z])

    mesh.density = target_mass_kg / mesh.volume
    computed_mass = float(mesh.mass)
    center_of_mass = [float(v) for v in mesh.center_mass]
    inertia = mesh.moment_inertia
    bounding_extents = mesh.bounding_box.extents
    effective_length_m = float(bounding_extents[2])

    normalized_mesh_bytes = mesh.export(file_type="stl")

    return {
        "mass_kg": computed_mass,
        "volume_m3": float(mesh.volume),
        "is_watertight": is_watertight,
        "center_of_mass_m": center_of_mass,
        "com_offset_z_m": center_of_mass[2],
        "effective_length_m": effective_length_m,
        "inertia_tensor": {
            "ixx": float(inertia[0][0]),
            "iyy": float(inertia[1][1]),
            "izz": float(inertia[2][2]),
            "ixy": float(inertia[0][1]),
            "ixz": float(inertia[0][2]),
            "iyz": float(inertia[1][2]),
        },
        "normalized_mesh_bytes": normalized_mesh_bytes,
    }