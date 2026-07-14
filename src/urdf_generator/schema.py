"""
Structured, parametric description of a robotic arm.

This is the v1 input format. Instead of parsing arbitrary CAD, the user
(or a future CAD-parsing layer) fills out this schema, and the generator
turns it directly into a valid URDF.

Design note: every field here maps to something a real URDF <link> or
<joint> needs. If you add a field, make sure generator.py actually uses it -
we don't want "decorative" fields that look configurable but do nothing.
"""

from enum import Enum
from pydantic import BaseModel, Field


class JointType(str, Enum):
    REVOLUTE = "revolute"   # rotates within limits (most arm joints)
    CONTINUOUS = "continuous"  # rotates freely, no limits
    PRISMATIC = "prismatic"  # slides linearly


class Link(BaseModel):
    """A single rigid segment of the arm (e.g. 'upper_arm', 'forearm')."""
    name: str
    length_m: float = Field(..., gt=0, description="Length along the link's primary axis")
    radius_m: float = Field(0.03, gt=0, description="Approximate radius, for a cylinder collision shape")
    mass_kg: float = Field(..., gt=0)


class Joint(BaseModel):
    """Connects parent_link to child_link."""
    name: str
    joint_type: JointType
    parent_link: str
    child_link: str
    axis: tuple[float, float, float] = (0, 0, 1)
    lower_limit_rad: float | None = Field(None, description="Required for revolute/prismatic")
    upper_limit_rad: float | None = None
    max_torque_nm: float = Field(..., gt=0, description="Motor stall torque - used for capability checks")
    max_velocity_rad_s: float = Field(3.0, gt=0)


class ArmConfig(BaseModel):
    """Full parametric description of a serial robotic arm, base to end effector."""
    name: str
    links: list[Link]
    joints: list[Joint]
    payload_mass_kg: float = Field(0.0, ge=0, description="Mass to attach at the end effector for testing")

    def link_names(self) -> set[str]:
        return {link.name for link in self.links}
