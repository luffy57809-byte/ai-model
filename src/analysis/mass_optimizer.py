"""
Mass-minimization optimizer: finds the lightest possible link masses that
still keep every joint within a safety margin of its rated torque.

WHY THIS IS A REAL LINEAR PROGRAM, NOT A HEURISTIC:
torque_check.py's required-torque equation is linear in link mass - each
link's mass is multiplied by a fixed distance factor (which depends only
on link lengths, which we're NOT optimizing here). So:

    minimize   sum(link masses)
    subject to  required_torque(joint_i) <= max_torque_nm(joint_i) * (1 - safety_margin)   for every joint i
                link_mass >= min_mass_kg                                                    for every link

...is an exact linear program, solvable with scipy.optimize.linprog. There's
no approximation or guessing involved in the math itself.

IMPORTANT MODELING LIMITATION - read this before trusting the output:
this model has no relationship between a link's mass and its physical
size (length/radius) or structural strength. Nothing stops the optimizer
from suggesting a link be built at min_mass_kg even if that's not
physically realizable for that link's dimensions and material. Treat the
output as "how much mass could theoretically be removed given only the
motors' torque limits", not as a manufacturable final design. min_mass_kg
is your one dial for encoding "don't go lighter than this is physically
sane" - set it thoughtfully.
"""

from scipy.optimize import linprog

from src.urdf_generator.schema import ArmConfig

GRAVITY = 9.81


def optimize_link_masses(
    config: ArmConfig,
    safety_margin: float = 0.2,
    min_mass_kg: float = 0.1,
) -> dict:
    """
    safety_margin: e.g. 0.2 means every joint must have at least 20% torque
                   margin after optimization (required <= 80% of rated).
    min_mass_kg: floor on any individual link's mass - see the module
                 docstring's limitation note before relying on this.
    """
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
            distance_to_com = cumulative_distance + (link_k.length_m / 2.0)
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
        links_report.append({
            "name": link.name,
            "original_mass_kg": link.mass_kg,
            "optimized_mass_kg": round(float(optimized_masses[i]), 4),
        })

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
