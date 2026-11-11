from collections import Counter
from copy import deepcopy
from dataclasses import asdict, dataclass
import json
import math
from typing import Callable


@dataclass(frozen=True)
class ToolSpec:
    arguments: dict[str, str]
    returns: tuple[str, ...]
    function: Callable


@dataclass(frozen=True)
class SmokeCase:
    id: str
    request: str
    available_tools: tuple[str, ...]
    initial_state: dict
    oracle: dict
    expected_result: object
    expected_state: dict
    expected_actions: tuple[dict, ...]
    split: str = "smoke-development"


def update_customer_email(state, customer_id, email):
    state["customers"][customer_id]["email"] = email
    return deepcopy(state["customers"][customer_id])


def reserve_inventory(state, product_id, quantity):
    product = state["products"][product_id]
    if quantity < 1 or quantity > product["available"]:
        raise ValueError("Quantity must be positive and available")
    product["available"] -= quantity
    product["reserved"] += quantity
    return deepcopy(product)


def release_inventory(state, product_id, quantity):
    product = state["products"][product_id]
    if quantity < 1 or quantity > product["reserved"]:
        raise ValueError("Quantity must be positive and reserved")
    product["available"] += quantity
    product["reserved"] -= quantity
    return deepcopy(product)


def update_order_address(state, order_id, address):
    state["orders"][order_id]["address"] = address
    return deepcopy(state["orders"][order_id])


def create_return(state, order_id, reason):
    if not state["orders"][order_id]["eligible"]:
        raise ValueError("Order is not eligible for return")
    return_id = f"R{len(state['returns']) + 1}"
    state["returns"][return_id] = {"return_id": return_id, "order_id": order_id, "reason": reason}
    return deepcopy(state["returns"][return_id])


def create_booking(state, customer_id, destination, date):
    state["customers"][customer_id]
    booking_id = f"B{len(state['bookings']) + 1}"
    state["bookings"][booking_id] = {
        "booking_id": booking_id, "customer_id": customer_id,
        "destination": destination, "date": date, "status": "confirmed",
    }
    return deepcopy(state["bookings"][booking_id])


def cancel_booking(state, booking_id):
    state["bookings"][booking_id]["status"] = "cancelled"
    return deepcopy(state["bookings"][booking_id])


def change_booking_date(state, booking_id, date):
    state["bookings"][booking_id]["date"] = date
    return deepcopy(state["bookings"][booking_id])


TOOLS = {
    "lookup_customer": ToolSpec({"customer_id": "string"}, ("customer_id", "name", "email"), lambda s, customer_id: deepcopy(s["customers"][customer_id])),
    "lookup_customer_orders": ToolSpec({"customer_id": "string"}, ("order_ids",), lambda s, customer_id: {"order_ids": [k for k, v in s["orders"].items() if v["customer_id"] == customer_id]}),
    "lookup_customer_bookings": ToolSpec({"customer_id": "string"}, ("booking_ids",), lambda s, customer_id: {"booking_ids": [k for k, v in s["bookings"].items() if v["customer_id"] == customer_id]}),
    "update_customer_email": ToolSpec({"customer_id": "string", "email": "string"}, ("customer_id", "name", "email"), update_customer_email),
    "lookup_order": ToolSpec({"order_id": "string"}, ("order_id", "customer_id", "product_id", "address", "eligible"), lambda s, order_id: deepcopy(s["orders"][order_id])),
    "check_inventory": ToolSpec({"product_id": "string"}, ("product_id", "name", "available", "reserved"), lambda s, product_id: deepcopy(s["products"][product_id])),
    "lookup_product": ToolSpec({"product_id": "string"}, ("product_id", "name", "available", "reserved"), lambda s, product_id: deepcopy(s["products"][product_id])),
    "reserve_inventory": ToolSpec({"product_id": "string", "quantity": "integer"}, ("product_id", "name", "available", "reserved"), reserve_inventory),
    "release_inventory": ToolSpec({"product_id": "string", "quantity": "integer"}, ("product_id", "name", "available", "reserved"), release_inventory),
    "update_order_address": ToolSpec({"order_id": "string", "address": "string"}, ("order_id", "customer_id", "product_id", "address", "eligible"), update_order_address),
    "check_return_eligibility": ToolSpec({"order_id": "string"}, ("order_id", "eligible"), lambda s, order_id: {"order_id": order_id, "eligible": s["orders"][order_id]["eligible"]}),
    "create_return": ToolSpec({"order_id": "string", "reason": "string"}, ("return_id", "order_id", "reason"), create_return),
    "lookup_return": ToolSpec({"return_id": "string"}, ("return_id", "order_id", "reason"), lambda s, return_id: deepcopy(s["returns"][return_id])),
    "lookup_booking": ToolSpec({"booking_id": "string"}, ("booking_id", "customer_id", "destination", "date", "status"), lambda s, booking_id: deepcopy(s["bookings"][booking_id])),
    "search_alternatives": ToolSpec({"destination": "string", "date": "string"}, ("options",), lambda s, destination, date: {"options": [{"destination": destination, "date": date, "option_id": "ALT1"}]}),
    "create_booking": ToolSpec({"customer_id": "string", "destination": "string", "date": "string"}, ("booking_id", "customer_id", "destination", "date", "status"), create_booking),
    "cancel_booking": ToolSpec({"booking_id": "string"}, ("booking_id", "customer_id", "destination", "date", "status"), cancel_booking),
    "change_booking_date": ToolSpec({"booking_id": "string", "date": "string"}, ("booking_id", "customer_id", "destination", "date", "status"), change_booking_date),
}


def initial_state():
    return {
        "customers": {
            "C1": {"customer_id": "C1", "name": "Asha", "email": "asha@example.test"},
            "C2": {"customer_id": "C2", "name": "Ben", "email": "ben@example.test"},
        },
        "orders": {
            "O1": {"order_id": "O1", "customer_id": "C1", "product_id": "P1", "address": "10 Lake Road", "eligible": True},
            "O2": {"order_id": "O2", "customer_id": "C2", "product_id": "P2", "address": "20 Hill Road", "eligible": False},
        },
        "products": {
            "P1": {"product_id": "P1", "name": "Mug", "available": 8, "reserved": 2},
            "P2": {"product_id": "P2", "name": "Lamp", "available": 4, "reserved": 1},
        },
        "returns": {"R1": {"return_id": "R1", "order_id": "O1", "reason": "wrong color"}},
        "bookings": {
            "B1": {"booking_id": "B1", "customer_id": "C1", "destination": "Chennai", "date": "2026-12-01", "status": "confirmed"},
            "B2": {"booking_id": "B2", "customer_id": "C2", "destination": "Kochi", "date": "2026-12-03", "status": "confirmed"},
        },
    }


def ref(value):
    return {"$ref": value}


def sequence(*actions):
    return {"op": "Sequence", "calls": [
        {"id": f"c{i}", "tool": tool, "arguments": args} for i, (tool, args) in enumerate(actions)
    ], "return": ref(f"c{len(actions)-1}")}


def build_smoke_cases():
    base = initial_state()
    rows = [
        ("Look up customer C1 and return the customer record.", [("lookup_customer", {"customer_id": "C1"})], base["customers"]["C1"], None),
        ("List the order IDs belonging to customer C1.", [("lookup_customer_orders", {"customer_id": "C1"})], {"order_ids": ["O1"]}, None),
        ("List the booking IDs belonging to customer C2.", [("lookup_customer_bookings", {"customer_id": "C2"})], {"booking_ids": ["B2"]}, None),
        ("Set customer C1's email to new@example.test and return the updated record.", [("update_customer_email", {"customer_id": "C1", "email": "new@example.test"})], {**base["customers"]["C1"], "email": "new@example.test"}, ("customers", "C1", "email", "new@example.test")),
        ("Look up order O1 and return its record.", [("lookup_order", {"order_id": "O1"})], base["orders"]["O1"], None),
        ("Check inventory for product P1 and return its inventory record.", [("check_inventory", {"product_id": "P1"})], base["products"]["P1"], None),
        ("Look up product P2 and return its record.", [("lookup_product", {"product_id": "P2"})], base["products"]["P2"], None),
        ("Reserve 3 units of P1 and return its updated inventory.", [("reserve_inventory", {"product_id": "P1", "quantity": 3})], {**base["products"]["P1"], "available": 5, "reserved": 5}, ("products", "P1", None, {**base["products"]["P1"], "available": 5, "reserved": 5})),
        ("Release 1 reserved unit of P1 and return its updated inventory.", [("release_inventory", {"product_id": "P1", "quantity": 1})], {**base["products"]["P1"], "available": 9, "reserved": 1}, ("products", "P1", None, {**base["products"]["P1"], "available": 9, "reserved": 1})),
        ("Change order O1's address to 5 River Street and return the order.", [("update_order_address", {"order_id": "O1", "address": "5 River Street"})], {**base["orders"]["O1"], "address": "5 River Street"}, ("orders", "O1", "address", "5 River Street")),
        ("Check return eligibility for order O2.", [("check_return_eligibility", {"order_id": "O2"})], {"order_id": "O2", "eligible": False}, None),
        ("Create a return for eligible order O1 with reason damaged and return its record.", [("create_return", {"order_id": "O1", "reason": "damaged"})], {"return_id": "R2", "order_id": "O1", "reason": "damaged"}, ("returns", "R2", None, {"return_id": "R2", "order_id": "O1", "reason": "damaged"})),
        ("Look up return R1.", [("lookup_return", {"return_id": "R1"})], base["returns"]["R1"], None),
        ("Look up booking B1.", [("lookup_booking", {"booking_id": "B1"})], base["bookings"]["B1"], None),
        ("Search alternatives for Madurai on 2026-12-10.", [("search_alternatives", {"destination": "Madurai", "date": "2026-12-10"})], {"options": [{"destination": "Madurai", "date": "2026-12-10", "option_id": "ALT1"}]}, None),
        ("Create a booking for customer C1 to Madurai on 2026-12-10 and return it.", [("create_booking", {"customer_id": "C1", "destination": "Madurai", "date": "2026-12-10"})], {"booking_id": "B3", "customer_id": "C1", "destination": "Madurai", "date": "2026-12-10", "status": "confirmed"}, ("bookings", "B3", None, {"booking_id": "B3", "customer_id": "C1", "destination": "Madurai", "date": "2026-12-10", "status": "confirmed"})),
        ("Cancel booking B2 and return its updated record.", [("cancel_booking", {"booking_id": "B2"})], {**base["bookings"]["B2"], "status": "cancelled"}, ("bookings", "B2", "status", "cancelled")),
        ("Move booking B1 to 2026-12-12 and return it.", [("change_booking_date", {"booking_id": "B1", "date": "2026-12-12"})], {**base["bookings"]["B1"], "date": "2026-12-12"}, ("bookings", "B1", "date", "2026-12-12")),
    ]
    cases = []

    def add(request, actions, result, mutation=None, resolved=None):
        expected_state = deepcopy(base)
        if mutation:
            table, key, field, value = mutation
            if field is None:
                expected_state[table][key] = deepcopy(value)
            else:
                expected_state[table][key][field] = value
        cases.append(SmokeCase(
            f"smoke-{len(cases)+1:03d}", request, tuple(TOOLS),
            deepcopy(base), sequence(*actions), deepcopy(result), expected_state,
            tuple({"tool": tool, "arguments": deepcopy(args)} for tool, args in (resolved or actions)),
        ))

    for row in rows:
        add(*row)
    add("Look up order O1, then look up its customer using the returned customer_id. Return the customer.", [("lookup_order", {"order_id": "O1"}), ("lookup_customer", {"customer_id": ref("c0.customer_id")})], base["customers"]["C1"], resolved=[("lookup_order", {"order_id": "O1"}), ("lookup_customer", {"customer_id": "C1"})])
    add("Look up order O2, then check inventory of its returned product_id. Return the inventory.", [("lookup_order", {"order_id": "O2"}), ("check_inventory", {"product_id": ref("c0.product_id")})], base["products"]["P2"], resolved=[("lookup_order", {"order_id": "O2"}), ("check_inventory", {"product_id": "P2"})])
    add("Look up booking B2, then look up its customer using the returned customer_id. Return the customer.", [("lookup_booking", {"booking_id": "B2"}), ("lookup_customer", {"customer_id": ref("c0.customer_id")})], base["customers"]["C2"], resolved=[("lookup_booking", {"booking_id": "B2"}), ("lookup_customer", {"customer_id": "C2"})])
    add("Look up return R1, then look up its order using the returned order_id. Return the order.", [("lookup_return", {"return_id": "R1"}), ("lookup_order", {"order_id": ref("c0.order_id")})], base["orders"]["O1"], resolved=[("lookup_return", {"return_id": "R1"}), ("lookup_order", {"order_id": "O1"})])
    add("Create a return for eligible order O1 with reason broken, then look up the return using its returned return_id. Return that lookup.", [("create_return", {"order_id": "O1", "reason": "broken"}), ("lookup_return", {"return_id": ref("c0.return_id")})], {"return_id": "R2", "order_id": "O1", "reason": "broken"}, ("returns", "R2", None, {"return_id": "R2", "order_id": "O1", "reason": "broken"}), [("create_return", {"order_id": "O1", "reason": "broken"}), ("lookup_return", {"return_id": "R2"})])
    add("Create a C2 booking to Salem on 2026-12-20, then look it up using its returned booking_id. Return the lookup.", [("create_booking", {"customer_id": "C2", "destination": "Salem", "date": "2026-12-20"}), ("lookup_booking", {"booking_id": ref("c0.booking_id")})], {"booking_id": "B3", "customer_id": "C2", "destination": "Salem", "date": "2026-12-20", "status": "confirmed"}, ("bookings", "B3", None, {"booking_id": "B3", "customer_id": "C2", "destination": "Salem", "date": "2026-12-20", "status": "confirmed"}), [("create_booking", {"customer_id": "C2", "destination": "Salem", "date": "2026-12-20"}), ("lookup_booking", {"booking_id": "B3"})])
    add("Look up booking B1, then search alternatives for its returned destination and date. Return the options.", [("lookup_booking", {"booking_id": "B1"}), ("search_alternatives", {"destination": ref("c0.destination"), "date": ref("c0.date")})], {"options": [{"destination": "Chennai", "date": "2026-12-01", "option_id": "ALT1"}]}, resolved=[("lookup_booking", {"booking_id": "B1"}), ("search_alternatives", {"destination": "Chennai", "date": "2026-12-01"})])
    add("Look up order O1, then look up its product using the returned product_id. Return the product.", [("lookup_order", {"order_id": "O1"}), ("lookup_product", {"product_id": ref("c0.product_id")})], base["products"]["P1"], resolved=[("lookup_order", {"order_id": "O1"}), ("lookup_product", {"product_id": "P1"})])
    add("Update C2's email to hello@example.test, then look up that customer using the returned customer_id. Return the lookup.", [("update_customer_email", {"customer_id": "C2", "email": "hello@example.test"}), ("lookup_customer", {"customer_id": ref("c0.customer_id")})], {**base["customers"]["C2"], "email": "hello@example.test"}, ("customers", "C2", "email", "hello@example.test"), [("update_customer_email", {"customer_id": "C2", "email": "hello@example.test"}), ("lookup_customer", {"customer_id": "C2"})])
    add("Look up customer C1, then list orders using that returned customer_id. Return the order IDs.", [("lookup_customer", {"customer_id": "C1"}), ("lookup_customer_orders", {"customer_id": ref("c0.customer_id")})], {"order_ids": ["O1"]}, resolved=[("lookup_customer", {"customer_id": "C1"}), ("lookup_customer_orders", {"customer_id": "C1"})])
    add("Look up customer C2 and independently look up order O1. Return the order record.", [("lookup_customer", {"customer_id": "C2"}), ("lookup_order", {"order_id": "O1"})], base["orders"]["O1"])
    add("Look up product P2 and independently look up booking B1. Return the booking record.", [("lookup_product", {"product_id": "P2"}), ("lookup_booking", {"booking_id": "B1"})], base["bookings"]["B1"])
    return tuple(cases)


def strict_json(raw):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError(f"Duplicate JSON key {key}")
            result[key] = value
        return result

    def constant(value):
        raise ValueError(f"Non-finite JSON value {value}")

    def finite(value):
        result = float(value)
        if not math.isfinite(result):
            raise ValueError(f"Non-finite JSON number {value}")
        return result

    return json.loads(raw, object_pairs_hook=pairs, parse_constant=constant, parse_float=finite)


def resolve(value, results):
    if isinstance(value, dict):
        if set(value) != {"$ref"} or not isinstance(value["$ref"], str):
            raise ValueError("Only a single $ref key is a valid reference expression")
        parts = value["$ref"].split(".")
        if len(parts) > 2 or parts[0] not in results:
            raise ValueError("Reference must name an earlier call, optionally one result field")
        result = results[parts[0]]
        if len(parts) == 2:
            result = result[parts[1]]
        return deepcopy(result)
    if type(value) not in (str, int, float, bool, type(None), list):
        raise ValueError("Unsupported literal")
    return deepcopy(value)


def action_key(action):
    return json.dumps(action, sort_keys=True)


def dependency_violations(case, events):
    observed = {
        action_key({"tool": event["tool"], "arguments": event["resolved_arguments"]}): event
        for event in events
    }
    required = {
        call["id"]: action_key(action)
        for call, action in zip(case.oracle["calls"], case.expected_actions)
    }
    violations = []
    for call in case.oracle["calls"]:
        event = observed[required[call["id"]]]
        for argument, value in call["arguments"].items():
            if not isinstance(value, dict):
                continue
            source_id, separator, field = value["$ref"].partition(".")
            source = observed[required[source_id]]
            binding = source["id"] + (separator + field if separator else "")
            if event["arguments"][argument] != ref(binding):
                violations.append(f"Required result binding missing for {event['id']}.{argument}")
    return violations


def evaluate(raw, case):
    state = deepcopy(case.initial_state)
    evaluation = {"status": "format_failure", "parsed": None, "violations": [], "events": [], "result": None, "final_state": state}
    try:
        parsed = strict_json(raw)
    except (ValueError, TypeError, RecursionError) as error:
        evaluation["violations"].append(str(error))
        return evaluation
    evaluation["parsed"] = parsed
    evaluation["status"] = "contract_failure"
    results = {}
    current = "root"
    try:
        if not isinstance(parsed, dict) or set(parsed) != {"op", "calls", "return"} or parsed["op"] != "Sequence":
            raise ValueError("Expected exactly op=Sequence, calls, and return")
        if not isinstance(parsed["calls"], list) or not 1 <= len(parsed["calls"]) <= 6:
            raise ValueError("Sequence requires 1 to 6 calls")
        for index, call in enumerate(parsed["calls"]):
            current = f"calls[{index}]"
            if not isinstance(call, dict) or set(call) != {"id", "tool", "arguments"}:
                raise ValueError("A call requires exactly id, tool, and arguments")
            call_id, name, arguments = call["id"], call["tool"], call["arguments"]
            if not isinstance(call_id, str) or not call_id.isidentifier() or call_id in results:
                raise ValueError("Call IDs must be unique identifiers")
            if not isinstance(name, str) or name not in case.available_tools:
                raise ValueError("Tool is not available for this case")
            spec = TOOLS[name]
            if not isinstance(arguments, dict) or set(arguments) != set(spec.arguments):
                raise ValueError("Arguments must match the tool schema exactly")
            args = {key: resolve(value, results) for key, value in arguments.items()}
            for key, kind in spec.arguments.items():
                value = args[key]
                if kind == "string" and (type(value) is not str or not value or len(value) > 256):
                    raise ValueError(f"{key} must be a nonempty string of at most 256 characters")
                if kind == "integer" and (type(value) is not int or not 1 <= value <= 100):
                    raise ValueError(f"{key} must be an integer in [1,100]")
            result = spec.function(state, **args)
            results[call_id] = deepcopy(result)
            evaluation["events"].append({"id": call_id, "tool": name, "arguments": arguments, "resolved_arguments": args, "result": deepcopy(result), "state_after": deepcopy(state)})
        current = "return"
        evaluation["result"] = resolve(parsed["return"], results)
    except (ValueError, KeyError, TypeError, IndexError, RecursionError) as error:
        evaluation["violations"].append(f"{current}: {type(error).__name__}: {error}")
        return evaluation
    actions = Counter(action_key({"tool": event["tool"], "arguments": event["resolved_arguments"]}) for event in evaluation["events"])
    expected_actions = Counter(action_key(action) for action in case.expected_actions)
    if evaluation["result"] != case.expected_result:
        evaluation["violations"].append("Returned value differs from the task requirement")
    if state != case.expected_state:
        evaluation["violations"].append("Final mock state differs from the task requirement")
    if actions != expected_actions:
        evaluation["violations"].append("Missing, extra, or incorrect tool actions")
    else:
        evaluation["violations"].extend(dependency_violations(case, evaluation["events"]))
    evaluation["status"] = "semantic_failure" if evaluation["violations"] else "success"
    return evaluation


def validate_oracles(cases):
    reports = []
    for case in cases:
        report = evaluate(json.dumps(case.oracle), case)
        if report["status"] != "success":
            raise ValueError(f"Invalid oracle {case.id}: {report['violations']}")
        reports.append({"id": case.id, "status": "success", "tool_calls": len(report["events"])})
    return reports


def public_input(case):
    return {"request": case.request, "available_tools": [
        {"name": name, "arguments": TOOLS[name].arguments, "returns": TOOLS[name].returns}
        for name in case.available_tools
    ]}


def prompt_messages(public):
    system = (
        'You control a deterministic mock tool environment. Output exactly one JSON object, no markdown or explanation. '
        'The grammar is {"op":"Sequence","calls":[{"id":"c0","tool":"TOOL_NAME","arguments":{"arg":"literal"}}],"return":{"$ref":"c0"}}. '
        'Use 1 to 6 calls, only the available tools, and exactly their required arguments. '
        'IDs must be unique. To use an earlier result field write {"$ref":"c0.field"}; '
        'to return its whole result write {"$ref":"c0"}. Never guess dynamically created IDs. '
        'Return the requested record or result. Do every requested action and no extra actions. '
        'Tools read or mutate local mock state. All requests have complete inputs and require no conditional branches.'
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": json.dumps(public, separators=(",", ":"))}]


def manifest(cases):
    return {"version": "smoke-v1", "split": "smoke-development", "generator": "src/plvr3b/baseline_tasks.py", "cases": [asdict(case) for case in cases]}
