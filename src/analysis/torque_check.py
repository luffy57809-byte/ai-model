"""
Static torque capability check.

No PyBullet, no simulation - just the free-body-diagram physics you'd do
by hand on paper. For each joint, computes the torque required to hold the
arm fully extended horizontally (the worst-case pose for gravity loading -
maximum lever arm on every downstream mass), and compares it to that
joint's max_torque_nm.

ASSUMPTION (important, document this to the user): this treats the arm as
a planar serial chain laid out flat and horizontal, base to end effector,
in a straight line. That's the correct worst case for a simple planar arm
like the two-link sample. It is a simplification for arms with joints on
different rotation axes (e.g. a joint that rotates about a vertical axis
doesn't actually lift anything against gravity in that configuration) -
a full 3D worst-case pose search is future work, not this check.

config.joints and config.links are assumed to be listed in serial order,
base to end effector (joint[i] drives link[i], which sits between
joint[i] and joint[i+1]).
"""

from src.urdf_generator.schema import ArmConfig

GRAVITY = 9.81


def compute_static_torques(config: ArmConfig) -> list[dict]:
    """
    Returns one result dict per joint, in the same order as config.joints:
      {
        "joint_name": str,
        "required_torque_nm": float,   # to hold the fully-extended horizontal pose
        "max_torque_nm": float,        # from the joint's spec
        "margin_percent": float,       # positive = spare capacity, negative = undersized
        "passes": bool,
      }
    """
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
            distance_to_com = cumulative_distance + (link_k.length_m / 2.0)
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
