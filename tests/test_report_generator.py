import os
import json
import pytest
from unittest.mock import patch, MagicMock

from src.ai_layer.report_generator import generate_report, _build_user_message
from src.urdf_generator.samples import two_link_arm
from src.analysis.torque_check import compute_static_torques
from src.simulation.lift_test import run_lift_test


def test_missing_api_key_raises_clear_error(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    config = two_link_arm()
    torque_results = compute_static_torques(config)

    with pytest.raises(RuntimeError, match="GEMINI_API_KEY is not set"):
        generate_report(config.name, torque_results)


def test_user_message_contains_real_numbers_not_invented_ones():
    config = two_link_arm()
    torque_results = compute_static_torques(config)
    lift_results = run_lift_test(config)

    message = _build_user_message(config.name, torque_results, lift_results)
    parsed = json.loads(message.split("\n\n", 1)[1])

    assert parsed["robot_name"] == "two_link_arm"
    assert parsed["static_torque_check"] == torque_results
    assert parsed["dynamic_lift_test"] == lift_results


def test_generate_report_calls_api_with_expected_shape(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "fake-key-for-testing")

    fake_response = MagicMock()
    fake_response.text = "Mock report: all joints pass with healthy margins."

    with patch("src.ai_layer.report_generator.genai.Client") as MockClient:
        mock_instance = MockClient.return_value
        mock_instance.models.generate_content.return_value = fake_response

        config = two_link_arm()
        torque_results = compute_static_torques(config)
        report = generate_report(config.name, torque_results)

        assert report == "Mock report: all joints pass with healthy margins."
        mock_instance.models.generate_content.assert_called_once()
        call_kwargs = mock_instance.models.generate_content.call_args.kwargs
        assert call_kwargs["model"] == "gemini-3.5-flash"
        assert "two_link_arm" in call_kwargs["contents"]
