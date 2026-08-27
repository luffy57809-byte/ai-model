import json
import pytest
import requests
from unittest.mock import patch, MagicMock

from src.ai_layer.report_generator import generate_report, _build_user_message
from src.urdf_generator.samples import two_link_arm
from src.analysis.torque_check import compute_static_torques
from src.simulation.lift_test import run_lift_test


def test_unreachable_ollama_raises_clear_error(monkeypatch):
    monkeypatch.setattr("src.ai_layer.report_generator.OLLAMA_URL", "http://localhost:1")
    config = two_link_arm()
    torque_results = compute_static_torques(config)

    with pytest.raises(RuntimeError, match="Could not reach Ollama"):
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


def test_generate_report_calls_ollama_with_expected_shape(monkeypatch):
    config = two_link_arm()
    torque_results = compute_static_torques(config)

    fake_health_response = MagicMock()
    fake_health_response.raise_for_status = MagicMock()

    fake_generate_response = MagicMock()
    fake_generate_response.raise_for_status = MagicMock()
    fake_generate_response.json.return_value = {
        "response": "Mock report: all joints pass with healthy margins."
    }

    with patch("requests.get", return_value=fake_health_response) as mock_get, \
         patch("requests.post", return_value=fake_generate_response) as mock_post:
        report = generate_report(config.name, torque_results)

        assert report == "Mock report: all joints pass with healthy margins."
        mock_get.assert_called_once()
        mock_post.assert_called_once()

        call_kwargs = mock_post.call_args.kwargs
        assert call_kwargs["json"]["model"] == "llama3.2"
        assert "two_link_arm" in call_kwargs["json"]["prompt"]
