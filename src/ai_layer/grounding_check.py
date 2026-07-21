"""
Checks whether an LLM-generated report actually stuck to the numbers it was
given, instead of just trusting the system prompt's instructions.

Approach: extract every numeric token from the report text, build the set of
"allowed" numbers from the real input data (plus a small set of harmless
values like 0/1/2 that show up as counts, degrees-of-freedom labels, etc.),
and flag any report number that isn't within tolerance of anything allowed.

This is a heuristic, not a proof - a model could still combine two real
numbers into a third "real" one that looks ungrounded but is actually valid.
Treat flagged numbers as "worth a human look", not automatic failures.
"""

import re

NUMBER_PATTERN = re.compile(r"-?\d+\.?\d*")
HARMLESS_INTEGERS = {0, 1, 2, 3, 4, 5}


def _collect_numbers_from_data(data) -> set[float]:
    found = set()
    if isinstance(data, dict):
        for value in data.values():
            found |= _collect_numbers_from_data(value)
    elif isinstance(data, list):
        for item in data:
            found |= _collect_numbers_from_data(item)
    elif isinstance(data, (int, float)) and not isinstance(data, bool):
        found.add(round(float(data), 2))
    return found


def _extract_numbers_from_text(text: str) -> list[float]:
    matches = NUMBER_PATTERN.findall(text)
    numbers = []
    for m in matches:
        try:
            numbers.append(round(float(m), 2))
        except ValueError:
            continue
    return numbers


def check_report_grounded(
    report_text: str,
    torque_check: list[dict],
    lift_test: dict | None = None,
    tolerance: float = 0.15,
) -> dict:
    allowed_numbers = _collect_numbers_from_data(torque_check)
    if lift_test is not None:
        allowed_numbers |= _collect_numbers_from_data(lift_test)
    allowed_numbers |= {float(i) for i in HARMLESS_INTEGERS}

    report_numbers = _extract_numbers_from_text(report_text)

    ungrounded = []
    for num in report_numbers:
        if any(abs(num - allowed) <= tolerance for allowed in allowed_numbers):
            continue
        ungrounded.append(num)

    return {
        "ungrounded_numbers": ungrounded,
        "is_fully_grounded": len(ungrounded) == 0,
    }
