"""
Converts an ArmConfig into a valid URDF string.

Everything downstream (simulation, AI reports) depends on this producing
physically valid URDF - wrong inertia values here will silently produce
wrong simulation results later.

MESH LINKS: if a Link has mesh_id set, this generator emits a <mesh>
geometry (referencing the stored file on disk) instead of a <cylinder>,
and uses the link's real computed inertia_* fields and com_offset_m
instead of the analytic cylinder formula and length/2 assumption. See
schema.py's Link docstring and analysis/mesh_processor.py for how those
fields get populated.
"""

from pathlib import Path

from src.urdf_generator.schema import ArmConfig, Link, Joint

MESH_STORAGE_DIR = Path("data/meshes")


def _cylinder_inertia(mass: float, radius: float, length: float) -> tuple[float, float, float]:
    """
    Moment of inertia for a solid cylinder about its own center of mass,
    aligned along its length (the z-axis of the link frame).
    """
    ixx = iyy = (1.0 / 12.0) * mass * (3 * radius**2 + length**2)
    izz = 0.5 * mass * radius**2
    return ixx, iyy, izz


def _mesh_file_path(mesh_id: str) -> str:
    """Absolute path to a stored mesh file - avoids relative-path resolution
    headaches when PyBullet loads the URDF from a temp directory."""
    return str((MESH_STORAGE_DIR / f"{mesh_id}.stl").resolve())


def _link_xml(link: Link) -> str:
    if link.is_mesh_based():
        return _mesh_link_xml(link)
    return _cylinder_link_xml(link)


def _cylinder_link_xml(link: Link) -> str:
    ixx, iyy, izz = _cylinder_inertia(link.mass_kg, link.radius_m, link.length_m)
    half = link.length_m / 2.0
    return f"""  <link name="{link.name}">
    <visual>
      <origin xyz="0 0 {half}" rpy="0 0 0"/>
      <geometry>
        <cylinder radius="{link.radius_m}" length="{link.length_m}"/>
      </geometry>
    </visual>
    <collision>
      <origin xyz="0 0 {half}" rpy="0 0 0"/>
      <geometry>
        <cylinder radius="{link.radius_m}" length="{link.length_m}"/>
      </geometry>
    </collision>
    <inertial>
      <origin xyz="0 0 {half}" rpy="0 0 0"/>
      <mass value="{link.mass_kg}"/>
      <inertia ixx="{ixx:.8f}" ixy="0" ixz="0" iyy="{iyy:.8f}" iyz="0" izz="{izz:.8f}"/>
    </inertial>
  </link>
"""


def _mesh_link_xml(link: Link) -> str:
    mesh_path = _mesh_file_path(link.mesh_id)
    com_z = link.com_offset_m
    return f"""  <link name="{link.name}">
    <visual>
      <geometry>
        <mesh filename="{mesh_path}"/>
      </geometry>
    </visual>
    <collision>
      <geometry>
        <mesh filename="{mesh_path}"/>
      </geometry>
    </collision>
    <inertial>
      <origin xyz="0 0 {com_z}" rpy="0 0 0"/>
      <mass value="{link.mass_kg}"/>
      <inertia ixx="{link.inertia_ixx:.8f}" ixy="{link.inertia_ixy:.8f}" ixz="{link.inertia_ixz:.8f}" iyy="{link.inertia_iyy:.8f}" iyz="{link.inertia_iyz:.8f}" izz="{link.inertia_izz:.8f}"/>
    </inertial>
  </link>
"""


def _joint_xml(joint: Joint, parent_length: float) -> str:
    limit_xml = ""
    if joint.joint_type.value in ("revolute", "prismatic"):
        lower = joint.lower_limit_rad if joint.lower_limit_rad is not None else -3.14159
        upper = joint.upper_limit_rad if joint.upper_limit_rad is not None else 3.14159
        limit_xml = (
            f'<limit lower="{lower}" upper="{upper}" '
            f'effort="{joint.max_torque_nm}" velocity="{joint.max_velocity_rad_s}"/>'
        )
    return f"""  <joint name="{joint.name}" type="{joint.joint_type.value}">
    <parent link="{joint.parent_link}"/>
    <child link="{joint.child_link}"/>
    <origin xyz="0 0 {parent_length}" rpy="0 0 0"/>
    <axis xyz="{joint.axis[0]} {joint.axis[1]} {joint.axis[2]}"/>
    {limit_xml}
  </joint>
"""


def generate_urdf(config: ArmConfig) -> str:
    """Build a complete URDF document string from a validated ArmConfig."""
    errors = validate_config(config)
    if errors:
        raise ValueError(f"Invalid ArmConfig: {errors}")

    link_by_name = {link.name: link for link in config.links}

    parts = [f'<?xml version="1.0"?>\n<robot name="{config.name}">\n']

    parts.append(
        '  <link name="base_link">\n'
        '    <visual><geometry><box size="0.15 0.15 0.05"/></geometry></visual>\n'
        '    <collision><geometry><box size="0.15 0.15 0.05"/></geometry></collision>\n'
        '    <inertial><mass value="1.0"/>'
        '<inertia ixx="0.001" ixy="0" ixz="0" iyy="0.001" iyz="0" izz="0.001"/></inertial>\n'
        "  </link>\n"
    )

    for link in config.links:
        parts.append(_link_xml(link))

    for joint in config.joints:
        parent_link = link_by_name.get(joint.parent_link)
        parent_length = parent_link.length_m if parent_link else 0.0
        parts.append(_joint_xml(joint, parent_length))

    if config.payload_mass_kg > 0 and config.links:
        last_link = config.links[-1]
        payload_radius = 0.025
        parts.append(
            f'  <link name="payload">\n'
            f'    <visual><geometry><sphere radius="{payload_radius}"/></geometry></visual>\n'
            f'    <collision><geometry><sphere radius="{payload_radius}"/></geometry></collision>\n'
            f'    <inertial><mass value="{config.payload_mass_kg}"/>'
            f'<inertia ixx="0.0001" ixy="0" ixz="0" iyy="0.0001" iyz="0" izz="0.0001"/></inertial>\n'
            f"  </link>\n"
            f'  <joint name="payload_mount" type="fixed">\n'
            f'    <parent link="{last_link.name}"/>\n'
            f'    <child link="payload"/>\n'
            f'    <origin xyz="0 0 {last_link.length_m}" rpy="0 0 0"/>\n'
            f"  </joint>\n"
        )

    parts.append("</robot>\n")
    return "".join(parts)


def validate_config(config: ArmConfig) -> list[str]:
    """Return a list of human-readable problems. Empty list means valid."""
    errors = []
    names = config.link_names()

    if len(config.links) != len(names):
        errors.append("Duplicate link names found.")

    for joint in config.joints:
        if joint.parent_link not in names and joint.parent_link != "base_link":
            errors.append(f"Joint '{joint.name}' references unknown parent_link '{joint.parent_link}'.")
        if joint.child_link not in names:
            errors.append(f"Joint '{joint.name}' references unknown child_link '{joint.child_link}'.")
        if joint.joint_type.value in ("revolute", "prismatic"):
            if joint.lower_limit_rad is None or joint.upper_limit_rad is None:
                errors.append(f"Joint '{joint.name}' is {joint.joint_type.value} but missing limits.")

    for link in config.links:
        if link.is_mesh_based():
            mesh_path = MESH_STORAGE_DIR / f"{link.mesh_id}.stl"
            if not mesh_path.exists():
                errors.append(f"Link '{link.name}' references mesh_id '{link.mesh_id}' but no such mesh file exists.")

    return errors
