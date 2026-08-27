"""
Mass-minimization optimizer: finds the lightest possible link masses that
still keep every joint within a safety margin of its rated torque.

MESH LINKS: mass is treated as a free variable, but the distance-to-COM
factor is fixed at the mesh's originally-computed com_offset_m. This is
actually correct, not an approximation - a rigid body's COM location is a
purely geometric property, independent of its uniform density/mass. What
IS a real limitation: the mesh's original inertia_* values (used only by
the DYNAMIC simulation, not this static optimizer) won't automatically
rescale - re-upload with the new target mass before running a dynamic
lift test on an optimized design.
"""

from scipy.optimize import linprog

from src.urdf_generator.schema import ArmConfig, Link

GRAVITY = 9.81


def _distance_to_com(link: Link) -> float:
    """Kept in sync with torque_check.py's identical logic - see that
    module's docstring for the mesh-vs-cylinder distinction."""
    if link.is_mesh_based():
        return link.com_offset_m
    return link.length_m / 2.0


def optimize_link_masses(
    config: ArmConfig,
    safety_margin: float = 0.2,
    min_mass_kg: float = 0.1,
) -> dict:
    n = len(config.links)
    original_total_mass = sum(link.mass_kg for link in config.links)

    if n != len(config.joints):
        return {
            "feasible": False,
            "original_total_mass_kg": original_total_mass,
            "optimized_total_mass_kg": None,
            "mass_reduction_kg": None,
            "mass_reduction_percent": None,
            "links": None,
            "message": (
                f"Cannot optimize: this model assumes one link per joint "
                f"(joint[i] drives link[i]), but got {n} links and "
                f"{len(config.joints)} joints."
            ),
        }

    c = [1.0] * n
    A_ub = []
    b_ub = []

    for i in range(n):
        row = [0.0] * n
        cumulative_distance = 0.0
        for k in range(i, n):
            link_k = config.links[k]
            distance_to_com = cumulative_distance + _distance_to_com(link_k)
            row[k] = GRAVITY * distance_to_com
            cumulative_distance += link_k.length_m

        payload_term = GRAVITY * config.payload_mass_kg * cumulative_distance
        rhs = config.joints[i].max_torque_nm * (1 - safety_margin) - payload_term

        A_ub.append(row)
        b_ub.append(rhs)

    bounds = [(min_mass_kg, None)] * n

    result = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method="highs")

    if not result.success:
        return {
            "feasible": False,
            "original_total_mass_kg": original_total_mass,
            "optimized_total_mass_kg": None,
            "mass_reduction_kg": None,
            "mass_reduction_percent": None,
            "links": None,
            "message": (
                "No feasible mass assignment found at this safety margin. "
                "This usually means the payload alone (or the min_mass_kg "
                "floor) already exceeds a joint's torque budget - consider "
                "a stronger motor, a lighter payload, or a lower safety_margin."
            ),
        }

    optimized_masses = result.x
    optimized_total_mass = float(sum(optimized_masses))

    links_report = []
    for i, link in enumerate(config.links):
        entry = {
            "name": link.name,
            "original_mass_kg": link.mass_kg,
            "optimized_mass_kg": round(float(optimized_masses[i]), 4),
        }
        if link.is_mesh_based():
            entry["note"] = (
                "Mesh-based link: re-upload this mesh with the optimized "
                "target mass to get correct inertia values before running "
                "a dynamic lift test on this result."
            )
        links_report.append(entry)

    return {
        "feasible": True,
        "original_total_mass_kg": round(original_total_mass, 4),
        "optimized_total_mass_kg": round(optimized_total_mass, 4),
        "mass_reduction_kg": round(original_total_mass - optimized_total_mass, 4),
        "mass_reduction_percent": round(
            (original_total_mass - optimized_total_mass) / original_total_mass * 100, 1
        ) if original_total_mass > 0 else 0.0,
        "links": links_report,
        "message": "Optimization succeeded.",
    }
