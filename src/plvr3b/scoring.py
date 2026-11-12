from collections import Counter
from functools import lru_cache
import json
import math


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


def value_terms(predicted, expected):
    if type(predicted) is not type(expected):
        return 1, [1.0]
    if isinstance(expected, dict):
        errors, terms = 0, []
        for key in sorted(set(predicted) | set(expected)):
            if key not in predicted or key not in expected:
                errors += 1
                terms.append(1.0)
            else:
                count, values = value_terms(predicted[key], expected[key])
                errors += count
                terms.extend(values)
        return errors, terms
    if isinstance(expected, list):
        predicted = list(predicted)
        expected = list(expected)
        identity = next((key for key in ("customer_id", "order_id", "product_id", "return_id", "booking_id", "option_id")
                         if predicted and expected and all(isinstance(v, dict) and key in v and type(v[key]) in (str, int) for v in predicted + expected)
                         and len({v[key] for v in predicted}) == len(predicted)
                         and len({v[key] for v in expected}) == len(expected)), None)
        if identity:
            return value_terms({v[identity]: v for v in predicted}, {v[identity]: v for v in expected})
        errors, terms = 0, []
        for index in range(max(len(predicted), len(expected))):
            if index >= len(predicted) or index >= len(expected):
                errors += 1
                terms.append(1.0)
            else:
                count, values = value_terms(predicted[index], expected[index])
                errors += count
                terms.extend(values)
        return errors, terms
    return 0, [0.0 if predicted == expected else 1.0]


def call_terms(predicted, expected):
    if predicted is None or expected is None:
        return 1, [1.0]
    if type(predicted.get("tool")) is not str or type(predicted.get("arguments")) is not dict:
        return 1, [1.0]
    if predicted["tool"] != expected["tool"]:
        return 0, [1.0]
    left, right = predicted["arguments"], expected["arguments"]
    keys = set(left) | set(right)
    correct = sum(k in left and k in right and type(left[k]) is type(right[k]) and left[k] == right[k] for k in keys)
    agreement = 0.5 + 0.5 * (correct / len(keys) if keys else 1.0)
    type_errors = sum(k not in left or k not in right or type(left[k]) is not type(right[k]) for k in keys)
    return type_errors, [1.0 - agreement]


def align_calls(predicted, expected):
    size = max(len(predicted), len(expected))
    left = list(predicted) + [None] * (size - len(predicted))
    right = list(expected) + [None] * (size - len(expected))

    @lru_cache(None)
    def best(index, mask):
        if index == size:
            return 0, 0.0, ()
        choices = []
        for j in range(size):
            if mask & (1 << j):
                continue
            same_tool = int(left[index] is not None and right[j] is not None and left[index].get("tool") == right[j].get("tool"))
            agreement = 0.0
            if same_tool:
                a, b = left[index].get("arguments", {}), right[j].get("arguments", {})
                keys = set(a) | set(b)
                agreement = sum(k in a and k in b and type(a[k]) is type(b[k]) and a[k] == b[k] for k in keys) / len(keys) if keys else 1.0
            rest_same, rest_agreement, rest_pairs = best(index + 1, mask | (1 << j))
            choices.append((rest_same - same_tool, rest_agreement - agreement, ((index, j),) + rest_pairs))
        return min(choices)

    _, _, pairs = best(0, 0)
    errors = sum(call_terms(left[i], right[j])[0] for i, j in pairs)
    atoms = [v for i, j in pairs for v in call_terms(left[i], right[j])[1]]
    return errors, atoms, [{"predicted": i if i < len(predicted) else None, "expected": j if j < len(expected) else None} for i, j in pairs]


def ranking_terms(execution, fixture):
    actual = [{"tool": event["tool"], "arguments": event["arguments"]} for event in execution.events]
    type_errors, atoms, alignment = align_calls(actual, fixture.expected_actions)
    for predicted, expected in ((execution.result, fixture.expected_result), (execution.final_state, fixture.expected_state)):
        errors, terms = value_terms(predicted, expected)
        type_errors += errors
        atoms.extend(terms)
    unmet = int(execution.status != "executed" or (fixture.expected_result is not None and execution.result is None))
    if unmet and not any(atoms):
        atoms.append(1.0)
    return {"unmet_requirements": unmet, "type_errors": type_errors, "value_error_squared": sum(v * v for v in atoms), "atoms": atoms, "alignment": alignment}


def aggregate_rank(contributions, required_inputs, incumbent_inputs):
    return (
        int(any(c["unmet_requirements"] for c in contributions)),
        sum(c["type_errors"] for c in contributions),
        math.sqrt(sum(c["value_error_squared"] for c in contributions)),
        len(set(required_inputs) ^ set(incumbent_inputs)),
    )


def tree_size(program):
    return 1 + sum(tree_size(child) for child in program.children)


def selection_key(candidate):
    return tuple(candidate["rank"]) + (candidate["size"], candidate["label"])
