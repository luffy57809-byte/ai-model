"""
Real integration test against the live Anthropic API.
Skipped automatically if ANTHROPIC_API_KEY isn't set.
"""

import os
import pytest

from src.urdf_generator.samples import two_link_arm
from src.analysis.torque_check import compute_static_torques
from src.simulation.lift_test import run_lift_test
from src.ai_layer.report_generator import generate_report
from src.ai_layer.grounding_check import check_report_grounded

requires_api_key = pytest.mark.skipif(
    not os.environ.get("GEMINI_API_KEY"),
    reason="GEMINI_API_KEY not set - skipping real API call",
)


@requires_api_key
def test_real_report_is_grounded_in_real_data():
    config = two_link_arm()
    torque_results = compute_static_torques(config)
    lift_results = run_lift_test(config)

    report_text = generate_report(config.name, torque_results, lift_results)
    assert isinstance(report_text, str)
    assert len(report_text) > 0

    grounding = check_report_grounded(report_text, torque_results, lift_results)

    if not grounding["is_fully_grounded"]:
        print("\n--- REPORT TEXT ---")
        print(report_text)
        print("--- UNGROUNDED NUMBERS ---")
        print(grounding["ungrounded_numbers"])

    assert grounding["is_fully_grounded"], (
        f"Report referenced numbers not traceable to input data: "
        f"{grounding['ungrounded_numbers']}. See printed report above."
    )
