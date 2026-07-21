from src.urdf_generator.samples import two_link_arm
from src.analysis.torque_check import compute_static_torques
from src.simulation.lift_test import run_lift_test
from src.ai_layer.grounding_check import check_report_grounded


def test_grounded_report_passes():
    config = two_link_arm()
    torque_results = compute_static_torques(config)
    lift_results = run_lift_test(config)

    honest_report = (
        "The shoulder joint requires 9.07 Nm and is rated for 15.0 Nm, "
        "a 39.5% margin. The elbow requires 2.45 Nm against a rating of "
        "8.0 Nm, a 69.3% margin. Both joints pass."
    )

    result = check_report_grounded(honest_report, torque_results, lift_results)
    assert result["is_fully_grounded"] is True
    assert result["ungrounded_numbers"] == []


def test_fabricated_number_is_caught():
    config = two_link_arm()
    torque_results = compute_static_torques(config)

    fabricated_report = (
        "The shoulder joint requires 9.07 Nm and is rated for 15.0 Nm. "
        "The motor will overheat after approximately 4200 duty cycles."
    )

    result = check_report_grounded(fabricated_report, torque_results)
    assert result["is_fully_grounded"] is False
    assert 4200.0 in result["ungrounded_numbers"]


def test_harmless_small_integers_are_not_flagged():
    config = two_link_arm()
    torque_results = compute_static_torques(config)

    report = "This is a 2-joint arm. Joint 1 (shoulder) and joint 2 (elbow) both pass."

    result = check_report_grounded(report, torque_results)
    assert result["is_fully_grounded"] is True
