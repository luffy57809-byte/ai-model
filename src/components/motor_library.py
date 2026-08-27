"""
A small library of real, commercially available servo motors, for picking
realistic torque/speed/mass values instead of typing arbitrary numbers.

SOURCING: every spec below comes from a manufacturer datasheet or official
product page, cross-checked against at least one independent retailer
listing where possible. Unit conversions computed programmatically:
    1 kgf*cm = 0.0980665 N*m
    rad/s = RPM * 2*pi / 60

HONEST CAVEATS:
- "Stall torque" is the maximum momentary torque before stalling - real
  sustained torque is typically much lower (ROBOTIS notes continuous
  torque is roughly 20% of stall torque). Treat these as ceilings, not
  safe sustained operating points.
- Prices are a single snapshot, volatile by retailer/region/time - always
  check current listings before a real purchasing decision.
- Torque/speed vary with supply voltage; values below are at one specific
  voltage (noted per motor).
"""

MOTOR_LIBRARY = [
    {
        "id": "sg90",
        "name": "TowerPro SG90 (micro hobby servo)",
        "mass_kg": 0.009,
        "stall_torque_nm": 0.1765,
        "max_velocity_rad_s": 10.472,
        "approx_price_usd": 3.5,
        "notes": "Very small/cheap hobby servo. Real-world torque is often "
                 "noticeably lower than the datasheet stall figure - treat "
                 "this as an optimistic ceiling, not a working number.",
        "source": "TowerPro SG90 datasheet (4.8V)",
    },
    {
        "id": "mg996r",
        "name": "TowerPro MG996R (metal-gear hobby servo)",
        "mass_kg": 0.055,
        "stall_torque_nm": 1.0787,
        "max_velocity_rad_s": 6.981,
        "approx_price_usd": 7.0,
        "notes": "Common cheap high-torque hobby servo, popular in small "
                 "arm builds. Metal gears, but positional accuracy is "
                 "mediocre compared to smart servos below.",
        "source": "TowerPro MG996R official datasheet (6V)",
    },
    {
        "id": "dynamixel_xl430",
        "name": "ROBOTIS Dynamixel XL430-W250-T",
        "mass_kg": 0.0572,
        "stall_torque_nm": 1.5,
        "max_velocity_rad_s": 6.388,
        "approx_price_usd": 45.0,
        "notes": "Smart servo: real position/velocity feedback, proper "
                 "digital control, much better repeatability than hobby "
                 "servos. Reasonable choice for a small arm's wrist/elbow.",
        "source": "ROBOTIS official e-manual + robotis.us listing",
    },
    {
        "id": "dynamixel_xm430",
        "name": "ROBOTIS Dynamixel XM430-W350-T",
        "mass_kg": 0.082,
        "stall_torque_nm": 4.1,
        "max_velocity_rad_s": 4.817,
        "approx_price_usd": 180.0,
        "notes": "Significantly stronger smart servo, current-based torque "
                 "control. A realistic choice for a shoulder joint. Price "
                 "varies a lot by retailer - rough order of magnitude only.",
        "source": "ROBOTIS official e-manual + RobotShop/robot-advance listings",
    },
]


def list_motors() -> list[dict]:
    return MOTOR_LIBRARY


def get_motor(motor_id: str) -> dict | None:
    for motor in MOTOR_LIBRARY:
        if motor["id"] == motor_id:
            return motor
    return None
