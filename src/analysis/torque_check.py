"""
Static torque capability check.

No PyBullet, no simulation - just the free-body-diagram physics you'd do
by hand on paper. For each joint, computes the torque required to hold the
arm fully extended horizontally (the worst-case pose for gravity loading),
and compares it to that joint's max_torque_nm.

MESH LINKS: for a plain cylinder link, the center of mass is assumed to
sit at length_m/2 (the geometric center of a uniform cylinder). For a
mesh-based link, we use the REAL computed com_offset_m instead - a mesh's
mass isn't necessarily centered on its bounding box (e.g. a link with a
motor housing lump at one end has its real COM shifted toward that end).
This is one of the concrete accuracy improvements mesh support buys you
over the cylinder approximation.
"""

from src.urdf_generator.schema import ArmConfig, Link

GRAVITY = 9.81


def _distance_to_com(link: Link) -> float:
    """Distance from the link's own proximal joint to its center of mass."""
    if link.is_mesh_based():
        return link.com_offset_m
    return link.length_m / 2.0


def compute_static_torques(config: ArmConfig) -> list[dict]:
    if len(config.links) != len(config.joints):
        raise ValueError(
            "This check assumes one link per joint in serial order "
            "(joint[i] drives link[i]). Config has "
            f"{len(config.joints)} joints but {len(config.links)} links."
        )

    n = len(config.links)
    results = []

    for i in range(n):
        joint = config.joints[i]

        required_torque = 0.0
        cumulative_distance = 0.0

        for k in range(i, n):
            link_k = config.links[k]
            distance_to_com = cumulative_distance + _distance_to_com(link_k)
            required_torque += GRAVITY * link_k.mass_kg * distance_to_com
            cumulative_distance += link_k.length_m

        required_torque += GRAVITY * config.payload_mass_kg * cumulative_distance

        max_torque = joint.max_torque_nm
        margin_percent = ((max_torque - required_torque) / max_torque) * 100.0

        results.append({
            "joint_name": joint.name,
            "required_torque_nm": round(required_torque, 4),
            "max_torque_nm": max_torque,
            "margin_percent": round(margin_percent, 1),
            "passes": required_torque <= max_torque,
        })

    return results
