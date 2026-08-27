import math
import pytest
from src.components.motor_library import list_motors, get_motor, MOTOR_LIBRARY


def test_library_has_expected_motors():
    motors = list_motors()
    ids = {m["id"] for m in motors}
    assert ids == {"sg90", "mg996r", "dynamixel_xl430", "dynamixel_xm430"}


def test_every_motor_has_required_fields():
    required_fields = {
        "id", "name", "mass_kg", "stall_torque_nm",
        "max_velocity_rad_s", "approx_price_usd", "notes", "source",
    }
    for motor in MOTOR_LIBRARY:
        missing = required_fields - motor.keys()
        assert not missing, f"{motor['id']} is missing fields: {missing}"


def test_every_motor_has_physically_sane_positive_values():
    for motor in MOTOR_LIBRARY:
        assert motor["mass_kg"] > 0
        assert motor["stall_torque_nm"] > 0
        assert motor["max_velocity_rad_s"] > 0
        assert motor["approx_price_usd"] > 0


def test_get_motor_returns_correct_entry():
    motor = get_motor("dynamixel_xl430")
    assert motor is not None
    assert motor["name"] == "ROBOTIS Dynamixel XL430-W250-T"


def test_get_motor_returns_none_for_unknown_id():
    assert get_motor("nonexistent_motor") is None


def test_sg90_torque_conversion_matches_independent_calculation():
    expected_nm = 1.8 * 0.0980665
    motor = get_motor("sg90")
    assert motor["stall_torque_nm"] == pytest.approx(expected_nm, abs=0.0001)


def test_sg90_speed_conversion_matches_independent_calculation():
    expected_rad_s = math.radians(60) / 0.1
    motor = get_motor("sg90")
    assert motor["max_velocity_rad_s"] == pytest.approx(expected_rad_s, abs=0.001)


def test_mg996r_torque_conversion_matches_independent_calculation():
    expected_nm = 11 * 0.0980665
    motor = get_motor("mg996r")
    assert motor["stall_torque_nm"] == pytest.approx(expected_nm, abs=0.0001)


def test_dynamixel_xl430_speed_conversion_matches_independent_calculation():
    expected_rad_s = 61 * 2 * math.pi / 60
    motor = get_motor("dynamixel_xl430")
    assert motor["max_velocity_rad_s"] == pytest.approx(expected_rad_s, abs=0.001)


def test_dynamixel_xm430_speed_conversion_matches_independent_calculation():
    expected_rad_s = 46 * 2 * math.pi / 60
    motor = get_motor("dynamixel_xm430")
    assert motor["max_velocity_rad_s"] == pytest.approx(expected_rad_s, abs=0.001)


def test_xm430_is_stronger_than_xl430_which_is_stronger_than_hobby_servos():
    sg90 = get_motor("sg90")
    mg996r = get_motor("mg996r")
    xl430 = get_motor("dynamixel_xl430")
    xm430 = get_motor("dynamixel_xm430")

    assert sg90["stall_torque_nm"] < mg996r["stall_torque_nm"]
    assert mg996r["stall_torque_nm"] < xl430["stall_torque_nm"]
    assert xl430["stall_torque_nm"] < xm430["stall_torque_nm"]
