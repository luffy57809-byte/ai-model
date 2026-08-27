"""
Computes real mass properties from an uploaded STEP file, using cadquery
(OpenCASCADE bindings) - the true CAD-format counterpart to
mesh_processor.py's STL/OBJ handling via trimesh.

KEY DIFFERENCE FROM mesh_processor.py: volume/center-of-mass/inertia are
computed ANALYTICALLY from the real CAD solid geometry (via OpenCASCADE's
own mass-property routines), not approximated from a triangulated
surface - genuinely more accurate than the trimesh path, since STEP
encodes exact parametric surfaces rather than a mesh approximation.

Converts to a mesh (tessellation) only at the end, to bridge into the
existing mesh-based storage/simulation pipeline (mesh_store.py,
generator.py) - everything downstream of this module stays unchanged.

Applies the SAME origin-normalization convention as mesh_processor.py
(local Z=0 = the assumed mounting/proximal face) for consistency with
the rest of the pipeline - see that module's docstring for the full
rationale and its stated limitation (assumes the long axis is Z).
"""

import io

import cadquery as cq
import numpy as np
import trimesh


class StepProcessingError(ValueError):
    """Raised for any problem with an uploaded STEP file - caught and turned into a 400 by the API."""


def compute_step_properties(step_bytes: bytes, target_mass_kg: float) -> dict:
    if target_mass_kg <= 0:
        raise StepProcessingError("target_mass_kg must be positive.")

    try:
        with io.BytesIO(step_bytes) as buf:
            # cadquery's importStep needs a real file path, not bytes -
            # write to a temp file for the import step only.
            import tempfile, os
            with tempfile.NamedTemporaryFile(suffix=".step", delete=False) as f:
                f.write(step_bytes)
                tmp_path = f.name
            try:
                loaded = cq.importers.importStep(tmp_path)
            finally:
                os.unlink(tmp_path)
    except Exception as exc:
        raise StepProcessingError(f"Could not parse STEP file: {exc}") from exc

    solids = loaded.solids().vals()
    if len(solids) == 0:
        raise StepProcessingError("STEP file loaded but contains no solid geometry.")
    if len(solids) > 1:
        raise StepProcessingError(
            f"STEP file contains {len(solids)} separate solids - this pipeline "
            f"expects one solid per uploaded part (same convention as one mesh "
            f"per link). Split multi-part assemblies into individual STEP files "
            f"before uploading, or use /designs/from_meshes to assemble them."
        )
    solid = solids[0]

    volume_m3 = solid.Volume()
    if volume_m3 <= 0:
        raise StepProcessingError(
            "Solid has non-positive volume - this usually means invalid or "
            "degenerate geometry in the STEP file."
        )

    # Tessellate to bridge into the existing mesh-based pipeline.
    try:
        vertices, faces = solid.tessellate(tolerance=0.0005)
    except Exception as exc:
        raise StepProcessingError(f"Could not tessellate STEP geometry: {exc}") from exc

    vertices_arr = np.array([[v.x, v.y, v.z] for v in vertices])
    mesh = trimesh.Trimesh(vertices=vertices_arr, faces=np.array(faces), process=True)

    if not mesh.is_watertight:
        raise StepProcessingError(
            "Tessellated STEP geometry is not watertight - this can happen with "
            "very thin features or tessellation tolerance issues. Try simplifying "
            "the part or increasing wall thickness."
        )

    # Same origin-normalization convention as mesh_processor.py: shift so
    # local Z=0 is the assumed proximal/mounting face.
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
        "is_watertight": True,
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
        "source_format": "step",
    }
