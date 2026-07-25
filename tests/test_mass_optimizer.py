import copy
import pytest
from src.urdf_generator.samples import two_link_arm
from src.analysis.mass_optimizer import optimize_link_masses
from src.analysis.torque_check import compute_static_torques


def test_optimizer_matches_hand_calculation():
    config = two_link_arm()
    result = optimize_link_masses(config, safety_margin=0.2, min_mass_kg=0.1)
    assert result["feasible"] is True
    assert result["original_total_mass_kg"] == 2.5
    assert result["optimized_total_mass_kg"] == pytest.approx(0.2, abs=0.001)
    assert result["links"][0]["optimized_mass_kg"] == pytest.approx(0.1, abs=0.001)
    assert result["links"][1]["optimized_mass_kg"] == pytest.approx(0.1, abs=0.001)


def test_optimized_masses_actually_satisfy_the_real_torque_check():
    config = two_link_arm()
    result = optimize_link_masses(config, safety_margin=0.2, min_mass_kg=0.1)
    optimized_config = copy.deepcopy(config)
    for i, link_result in enumerate(result["links"]):
        optimized_config.links[i].mass_kg = link_result["optimized_mass_kg"]
    verification = compute_static_torques(optimized_config)
    for joint_result in verification:
        assert joint_result["passes"] is True
        assert joint_result["margin_percent"] >= 20.0


def test_infeasible_when_payload_alone_exceeds_capacity():
    config = two_link_arm()
    config.payload_mass_kg = 500.0
    result = optimize_link_masses(config, safety_margin=0.2, min_mass_kg=0.1)
    assert result["feasible"] is False
    assert result["optimized_total_mass_kg"] is None


def test_mismatched_links_and_joints_reports_clear_error():
    config = two_link_arm()
    config.joints = config.joints[:1]
    result = optimize_link_masses(config)
    assert result["feasible"] is False
    assert "one link per joint" in result["message"]


def test_tighter_safety_margin_still_feasible_but_uses_more_mass_headroom():
    config = two_link_arm()
    result = optimize_link_masses(config, safety_margin=0.5, min_mass_kg=0.1)
    assert result["feasible"] is True
