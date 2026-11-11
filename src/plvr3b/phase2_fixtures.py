from copy import deepcopy
from dataclasses import asdict

from .baseline_tasks import build_smoke_cases, initial_state
from .types import Fixture


def grounded(text, value):
    spelling = str(value).lower() if type(value) is bool else str(value)
    start = text.find(spelling)
    if start < 0:
        raise ValueError(f"Fixture literal {value!r} has no request source")
    return {"value": value, "source": {"start": start, "end": start + len(spelling)}}


def payload_for(case, reverse=False):
    tasks = []
    for call in case.oracle["calls"]:
        arguments = {key: {"ref": value["$ref"]} if isinstance(value, dict) else grounded(case.request, value) for key, value in call["arguments"].items()}
        tasks.append({"id": call["id"], "intent": call["tool"], "arguments": arguments})
    return {
        "id": case.id.replace("-", "_"), "text": case.request,
        "tasks": list(reversed(tasks)) if reverse else tasks,
        "return": {"ref": case.oracle["return"]["$ref"]},
    }


def build_fixtures():
    cases = build_smoke_cases()
    fixtures = [Fixture(case.id, payload_for(case), deepcopy(case.initial_state), deepcopy(case.expected_result), deepcopy(case.expected_state), deepcopy(case.expected_actions)) for case in cases]
    for index in (18, 22):
        case = cases[index]
        fixtures.append(Fixture(case.id + "-reversed", payload_for(case, reverse=True), deepcopy(case.initial_state), deepcopy(case.expected_result), deepcopy(case.expected_state), deepcopy(case.expected_actions)))
    fixtures.append(Fixture("empty", {"id": "empty", "text": "No actions.", "tasks": [], "return": None}, initial_state(), None, initial_state(), ()))
    text = "Reserve 03 units of P1 and return its inventory."
    payload = {
        "id": "integer_transform", "text": text,
        "tasks": [{"id": "reserve", "intent": "reserve_inventory", "arguments": {
            "product_id": grounded(text, "P1"),
            "quantity": {"transform": "parse_integer", "input": grounded(text, "03")},
        }}], "return": {"ref": "reserve"},
    }
    expected_state = initial_state()
    expected_state["products"]["P1"].update(available=5, reserved=5)
    fixtures.append(Fixture("integer-transform", payload, initial_state(), deepcopy(expected_state["products"]["P1"]), expected_state, ({"tool": "reserve_inventory", "arguments": {"product_id": "P1", "quantity": 3}},)))
    return tuple(fixtures)


def fixture_data(fixtures):
    return {"version": "phase2-oracle-v1", "split": "development-oracle", "fixtures": [asdict(f) for f in fixtures]}
