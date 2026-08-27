"""
Turns the real numeric output of the torque check + lift test into a
plain-English engineering report.

Uses a local Ollama model (llama3.2) rather than a cloud API - no
account/API key required. Same grounding rules as before: only reference
provided numbers, never fabricate, explain failures concretely.
"""

import json
import os

OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")
MODEL = os.environ.get("REPORT_MODEL", "llama3.2")

SYSTEM_PROMPT = """You are a robotics engineering assistant writing a design
review for a robotic arm. You will be given real, already-computed physics
results (a static torque analysis and a dynamic simulation result) as JSON.

STRICT RULES:
- Only ever reference numbers that appear in the provided JSON data.
- Never invent, estimate, or round-trip a number that isn't given to you.
- If asked to explain a discrepancy between the static and dynamic results
  (e.g. a joint whose static estimate looks wrong), explain it using the
  joint's rotation axis and geometry logic, not by fabricating a new figure.
- If every joint passes, say so plainly - don't manufacture concerns to
  sound thorough.
- If any joint fails, explain concretely why (compare required vs rated
  torque, and dynamic sag if available) and give one specific, actionable
  recommendation.
- Write for a design engineer: concise, technical, no marketing language.
"""


def _build_user_message(config_name: str, torque_check: list[dict], lift_test: dict | None) -> str:
    data = {
        "robot_name": config_name,
        "static_torque_check": torque_check,
        "dynamic_lift_test": lift_test,
    }
    return (
        "Here are the real computed results for this robot arm design. "
        "Write a short design review based only on this data:\n\n"
        f"{json.dumps(data, indent=2)}"
    )


def generate_report(config_name: str, torque_check: list[dict], lift_test: dict | None = None) -> str:
    try:
        import requests
    except ImportError as exc:
        raise RuntimeError(
            "The 'requests' package is not installed - report generation "
            "is disabled in this environment (it's an optional dependency, "
            "left out of the production image to keep it lean)."
        ) from exc

    try:
        health_check = requests.get(f"{OLLAMA_URL}/api/tags", timeout=3)
        health_check.raise_for_status()
    except requests.RequestException as exc:
        raise RuntimeError(
            f"Could not reach Ollama at {OLLAMA_URL} - is `ollama serve` running? "
            f"Original error: {exc}"
        ) from exc

    user_message = _build_user_message(config_name, torque_check, lift_test)

    try:
        response = requests.post(
            f"{OLLAMA_URL}/api/generate",
            json={
                "model": MODEL,
                "system": SYSTEM_PROMPT,
                "prompt": user_message,
                "stream": False,
            },
            timeout=120,
        )
        response.raise_for_status()
    except requests.RequestException as exc:
        raise RuntimeError(f"Ollama request failed: {exc}") from exc

    return response.json()["response"]
