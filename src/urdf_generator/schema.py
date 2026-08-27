"""
Structured, parametric description of a robotic arm.

This is the v1 input format. Instead of parsing arbitrary CAD, the user
(or a future CAD-parsing layer) fills out this schema, and the generator
turns it directly into a valid URDF.

MESH SUPPORT: a Link can optionally reference an uploaded mesh (mesh_id)
instead of being a plain cylinder. When mesh_id is set, com_offset_m and
the six inertia_* fields must also be set (they're computed by
analysis/mesh_processor.py at upload time) - generator.py and
torque_check.py both check for mesh_id and use these real, mesh-derived
values instead of the analytic cylinder formulas. length_m and radius_m
are still required even for mesh links: length_m becomes the mesh's
bounding-box Z-extent (used to place the next joint in the chain) and
radius_m is simply unused/ignored for mesh links, kept only so the schema
doesn't need two different Link shapes.
"""

from enum import Enum
from pydantic import BaseModel, Field, model_validator


class JointType(str, Enum):
    REVOLUTE = "revolute"
    CONTINUOUS = "continuous"
    PRISMATIC = "prismatic"


class Link(BaseModel):
    """A single rigid segment of the arm (e.g. 'upper_arm', 'forearm')."""
    name: str
    length_m: float = Field(..., gt=0, description="Length along the link's primary axis")
    radius_m: float = Field(0.03, gt=0, description="Cylinder radius. Ignored if mesh_id is set.")
    mass_kg: float = Field(..., gt=0)

    # Mesh support - all None for a plain cylinder link (the original,
    # still fully supported, default case).
    mesh_id: str | None = Field(None, description="References an uploaded mesh via /meshes/upload.")
    com_offset_m: float | None = Field(None, description="Required if mesh_id is set - distance from the proximal joint to the mesh's real center of mass, along local Z.")
    inertia_ixx: float | None = None
    inertia_iyy: float | None = None
    inertia_izz: float | None = None
    inertia_ixy: float | None = None
    inertia_ixz: float | None = None
    inertia_iyz: float | None = None

    @model_validator(mode="after")
    def check_mesh_fields_complete(self):
        if self.mesh_id is not None:
            required = [self.com_offset_m, self.inertia_ixx, self.inertia_iyy, self.inertia_izz]
            if any(v is None for v in required):
                raise ValueError(
                    f"Link '{self.name}' has mesh_id set but is missing com_offset_m or "
                    f"inertia values. These are computed by /meshes/upload - don't set "
                    f"mesh_id manually without them."
                )
        return self

    def is_mesh_based(self) -> bool:
        return self.mesh_id is not None


class Joint(BaseModel):
    """Connects parent_link to child_link."""
    name: str
    joint_type: JointType
    parent_link: str
    child_link: str
    axis: tuple[float, float, float] = (0, 0, 1)
    lower_limit_rad: float | None = None
    upper_limit_rad: float | None = None
    max_torque_nm: float = Field(..., gt=0)
    max_velocity_rad_s: float = Field(3.0, gt=0)


class ArmConfig(BaseModel):
    """Full parametric description of a serial robotic arm, base to end effector."""
    name: str
    # max_length=20: generous for any real robot arm design, bounds
    # worst-case PyBullet simulation cost per request (each additional
    # link adds real simulation time) - a DoS-prevention measure, not a
    # meaningful limit on legitimate designs.
    links: list[Link] = Field(..., max_length=20)
    joints: list[Joint] = Field(..., max_length=20)
    payload_mass_kg: float = Field(0.0, ge=0)

    def link_names(self) -> set[str]:
        return {link.name for link in self.links}
