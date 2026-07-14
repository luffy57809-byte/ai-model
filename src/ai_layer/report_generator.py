"""
Turns the real numeric output of the torque check + lift test into a
plain-English engineering report using the Claude API.

CORE RULE: the LLM explains and organizes; it never invents a number.
Every figure it's allowed to discuss is embedded directly in the prompt as
data. The system prompt explicitly forbids introducing any number that
isn't in that data block.

Requires the ANTHROPIC_API_KEY environment variable to be set. In a
Codespace: `export ANTHROPIC_API_KEY=sk-ant-...` in the terminal (or add it
as a Codespaces secret so it persists - see README).
"""

import json
import os
import anthropic

MODEL = "claude-sonnet-5"

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
  recommendation (e.g. "increase the elbow motor's rated torque from X to
  at least Y Nm" - filling in X and Y only from the given data).
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
    """
    Raises RuntimeError with a clear message if ANTHROPIC_API_KEY isn't set,
    rather than failing with a cryptic SDK error.
    """
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise RuntimeError(
            "ANTHROPIC_API_KEY is not set. Run `export ANTHROPIC_API_KEY=sk-ant-...` "
            "in your terminal, or add it as a Codespaces secret, then restart the server."
        )

    client = anthropic.Anthropic()
    user_message = _build_user_message(config_name, torque_check, lift_test)

    response = client.messages.create(
        model=MODEL,
        max_tokens=1000,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_message}],
    )

    return "".join(block.text for block in response.content if block.type == "text")
