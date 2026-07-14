from src.urdf_generator.generator import generate_urdf, validate_config
from src.urdf_generator.samples import two_link_arm
from src.simulation.runner import run_smoke_test


def test_two_link_arm_validates():
    config = two_link_arm()
    assert validate_config(config) == []


def test_generate_urdf_produces_xml():
    config = two_link_arm()
    urdf = generate_urdf(config)
    assert urdf.startswith('<?xml version="1.0"?>')
    assert "<robot" in urdf
    assert "shoulder" in urdf
    assert "elbow" in urdf


def test_full_pipeline_runs_in_pybullet():
    config = two_link_arm()
    urdf = generate_urdf(config)
    result = run_smoke_test(urdf, sim_seconds=0.5)
    assert result["num_joints"] == 3  # shoulder + elbow + fixed payload_mount joint
    assert len(result["final_joint_positions"]) == 3
