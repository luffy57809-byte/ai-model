"""Reference configs for testing. Not production data - just known-good examples."""

from src.urdf_generator.schema import ArmConfig, Link, Joint, JointType


def two_link_arm() -> ArmConfig:
    """A simple 2-joint arm: shoulder (rotates about z) + elbow (rotates about y)."""
    return ArmConfig(
        name="two_link_arm",
        links=[
            Link(name="upper_arm", length_m=0.3, radius_m=0.03, mass_kg=1.5),
            Link(name="forearm", length_m=0.25, radius_m=0.025, mass_kg=1.0),
        ],
        joints=[
            Joint(
                name="shoulder",
                joint_type=JointType.REVOLUTE,
                parent_link="base_link",
                child_link="upper_arm",
                axis=(0, 0, 1),
                lower_limit_rad=-3.14,
                upper_limit_rad=3.14,
                max_torque_nm=15.0,
                max_velocity_rad_s=2.0,
            ),
            Joint(
                name="elbow",
                joint_type=JointType.REVOLUTE,
                parent_link="upper_arm",
                child_link="forearm",
                axis=(0, 1, 0),
                lower_limit_rad=-2.5,
                upper_limit_rad=2.5,
                max_torque_nm=8.0,
                max_velocity_rad_s=2.0,
            ),
        ],
        payload_mass_kg=0.5,
    )
