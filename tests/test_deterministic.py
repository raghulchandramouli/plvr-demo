import pytest

from plvr.llm import Completion
from plvr.primitive import Context, Runner, verbatim
from plvr.telecom import primitives as P
from plvr.telecom.policy import Area, Compiled, Rule, device_functions, parse_sections

POLICY = """<main_policy>
# Agent Policy
The current time is 2025-02-25 12:08:00 EST.
## Data Refueling
- Ask how much data
</main_policy>
<tech_support_policy>
# Service
## Airplane Mode
If it is ON, guide the user to use `toggle_airplane_mode()` to turn it OFF.
# Data
## Prerequisites
Refer to the "Service" section.
1. **check_status_bar** - Shows the status bar.
</tech_support_policy>"""


def test_verbatim_ignores_markdown_case_and_spacing():
    assert verbatim("guide the user to use toggle_airplane_mode() to turn it off", POLICY)
    assert not verbatim("guide the user to reboot", POLICY)
    assert not verbatim(None, POLICY)


def test_evaluate_allows_numbers_and_iso_dates_only():
    assert P.evaluate("15.2 > 15") is True
    assert P.evaluate("2 * 5.0") == 10.0
    assert P.evaluate("'2025-01-31' < '2025-02-25'") is True
    with pytest.raises(ValueError):
        P.evaluate("__import__('os')")
    with pytest.raises(ValueError):
        P.evaluate("'yesterday' < '2025-02-25'")


def test_sections_nest_by_heading_and_split_policy_parts():
    sections = parse_sections(POLICY)
    by_title = {s.title: s for s in sections}
    assert by_title["Data Refueling"].part == "main"
    assert by_title["Airplane Mode"].part == "tech"
    assert by_title["Airplane Mode"].parent == by_title["Service"].id
    assert by_title["Prerequisites"].parent == by_title["Data"].id
    assert "2025-02-25" in by_title["Agent Policy"].body


def test_device_functions_come_from_call_and_bold_spans():
    found = device_functions(POLICY, agent_functions=["get_customer_by_phone"])
    assert found == {"check_status_bar": "Shows the status bar.", "toggle_airplane_mode": ""}


def _rule(area: str, condition: str) -> Rule:
    return Rule(area, area, condition, None, "fix", {}, False)


def test_checklist_puts_prerequisites_first_and_survives_cycles():
    compiled = Compiled(
        areas={
            "Service": Area("Service", "check", "", ["Data"], [_rule("Service", "s")]),
            "Data": Area("Data", "check", "", ["Service"], [_rule("Data", "d")]),
        },
        procedures={}, device_functions=[], agent_functions=[], descriptions={}, rejected=[], preamble="",
    )
    assert [r.condition for r in compiled.checklist("Data")] == ["s", "d"]


class FakeServer:
    def __init__(self, text: str):
        self.text = text

    def render(self, messages, tools=None, *, thinking):
        return messages[-1]["content"]

    def complete(self, prompt, **kwargs):
        return Completion(self.text, "", [], 10, 0, 5, 0.0, "stop")


def test_rejected_output_falls_back_and_is_logged():
    run = Runner(FakeServer('{"value": "555-000-0000", "quote": "my number is 555-000-0000"}'))
    value = run(P.WHERE_VALUE, Context(), parameter="phone", source="my number is 555-123-2002")
    assert value.value is None
    assert run.calls[-1].accepted is False and "not verbatim" in run.calls[-1].rejection


def test_expression_must_use_numbers_from_the_record():
    run = Runner(FakeServer('{"expression": "16 > 15", "answer": "yes"}'))
    met = run(P.IS_CONDITION_MET, Context(), condition="usage exceeds limit", record='{"data_used_gb": 15.2, "data_limit_gb": 15}')
    assert met.answer == "cannot tell"
    assert "16" in run.calls[-1].rejection
