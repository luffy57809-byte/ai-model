"""
Builds a full ArmConfig from a list of already-uploaded mesh_ids plus
joint connectivity metadata - so a person assembling a robot from real
CAD-derived parts doesn't have to hand-type length_m, mass_kg,
com_offset_m, or inertia values (mesh_processor.py already computed all
of that at upload time; mesh_store.py now persists it).

Still assumes a strictly serial chain (base_link -> link1 -> link2 -> ...),
matching generator.py's existing convention - no branching support.
"""

from src.urdf_generator.schema import ArmConfig, Link, Joint, JointType
from src.storage.mesh_store import load_mesh_properties


class AssemblyError(ValueError):
    """Raised when the requested assembly can't be built - bad mesh_id,
    mismatched link/joint counts, etc. Caught and turned into a 400 by the API."""


def build_config_from_meshes(
    name: str,
    link_specs: list[dict],
    joint_specs: list[dict],
    user_id: str,
    payload_mass_kg: float = 0.0,
) -> ArmConfig:
    """
    link_specs: one dict per link, in chain order, each:
        {"name": str, "mesh_id": str}
    joint_specs: one dict per joint connecting consecutive links, each:
        {"name": str, "joint_type": "revolute"|"continuous"|"prismatic",
         "axis": [x, y, z] (optional, default (0,1,0) - a lift axis, not
                  (0,0,1) which is a roll axis; see generator.py's demo
                  arm note on this exact distinction),
         "lower_limit_rad": float, "upper_limit_rad": float,
         "max_torque_nm": float, "max_velocity_rad_s": float (optional)}
    There must be exactly one fewer joint than links (joint[i] connects
    link[i] to link[i+1], with joint[0] connecting base_link to link[0]).
    """
    if len(joint_specs) != len(link_specs):
        raise AssemblyError(
            f"Expected exactly one joint per link ({len(link_specs)} links "
            f"need {len(link_specs)} joints - one per link, connecting it "
            f"to its parent), got {len(joint_specs)} joint(s)."
        )

    links = []
    for spec in link_specs:
        mesh_id = spec["mesh_id"]
        try:
            props = load_mesh_properties(mesh_id, user_id)
        except FileNotFoundError as exc:
            raise AssemblyError(str(exc)) from exc

        links.append(Link(
            name=spec["name"],
            length_m=props["length_m"],
            mass_kg=props["mass_kg"],
            mesh_id=mesh_id,
            com_offset_m=props["com_offset_m"],
            inertia_ixx=props["inertia_ixx"],
            inertia_iyy=props["inertia_iyy"],
            inertia_izz=props["inertia_izz"],
            inertia_ixy=props["inertia_ixy"],
            inertia_ixz=props["inertia_ixz"],
            inertia_iyz=props["inertia_iyz"],
        ))

    parent_names = ["base_link"] + [spec["name"] for spec in link_specs[:-1]]
    joints = []
    for joint_spec, parent_name, link_spec in zip(joint_specs, parent_names, link_specs):
        joints.append(Joint(
            name=joint_spec["name"],
            joint_type=JointType(joint_spec["joint_type"]),
            parent_link=parent_name,
            child_link=link_spec["name"],
            axis=tuple(joint_spec.get("axis", (0, 1, 0))),
            lower_limit_rad=joint_spec.get("lower_limit_rad"),
            upper_limit_rad=joint_spec.get("upper_limit_rad"),
            max_torque_nm=joint_spec["max_torque_nm"],
            max_velocity_rad_s=joint_spec.get("max_velocity_rad_s", 3.0),
        ))

    return ArmConfig(
        name=name,
        links=links,
        joints=joints,
        payload_mass_kg=payload_mass_kg,
    )
