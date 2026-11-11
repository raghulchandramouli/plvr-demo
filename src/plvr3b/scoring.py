from collections import Counter
import json


def action_key(action):
    return json.dumps(action, sort_keys=True)


def score_fixture(execution, fixture):
    violations = []
    if execution.status != "executed":
        return {"success": False, "violations": [v.code for v in execution.violations]}
    if execution.result != fixture.expected_result:
        violations.append("wrong_result")
    if execution.final_state != fixture.expected_state:
        violations.append("wrong_final_state")
    actual = Counter(action_key({"tool": e["tool"], "arguments": e["arguments"]}) for e in execution.events)
    expected = Counter(action_key(a) for a in fixture.expected_actions)
    if actual != expected:
        violations.append("unwanted_or_missing_action")
    return {"success": not violations, "violations": violations}
