"""
Turns the real numeric output of the torque check + lift test into a
plain-English engineering report using the Gemini API (free tier - no
billing required).
"""

import json
import os
from google import genai
from google.genai import types

MODEL = "gemini-3.5-flash"

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
    if not os.environ.get("GEMINI_API_KEY"):
        raise RuntimeError(
            "GEMINI_API_KEY is not set. Get a free key (no credit card) at "
            "https://aistudio.google.com/apikey, then run "
            "`export GEMINI_API_KEY=...` in your terminal and restart the server."
        )

    client = genai.Client()
    user_message = _build_user_message(config_name, torque_check, lift_test)

    response = client.models.generate_content(
        model=MODEL,
        contents=user_message,
        config=types.GenerateContentConfig(system_instruction=SYSTEM_PROMPT),
    )

    return response.text
