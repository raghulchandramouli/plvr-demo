from datetime import date
from types import MappingProxyType

from .baseline_tasks import TOOLS
from .types import (
    Budget, Context, ContractError, DependencyGraph, Foreach, Literal, PrimitiveNode,
    Request, ResultRef, Sequence, SourceSpan, Task, TaskList, ToolDefinition,
    ToolRegistry, Transform,
)


FIELD_KINDS = {
    "customer_id": "customer_id", "order_id": "order_id", "product_id": "product_id",
    "return_id": "return_id", "booking_id": "booking_id", "name": "string",
    "email": "string", "address": "string", "reason": "string", "destination": "string",
    "date": "date", "status": "status", "available": "integer", "reserved": "integer",
    "eligible": "boolean", "order_ids": "order_ids", "booking_ids": "booking_ids",
    "options": "options",
}
MUTATIONS = frozenset({
    "update_customer_email", "reserve_inventory", "release_inventory", "update_order_address",
    "create_return", "create_booking", "cancel_booking", "change_booking_date",
})


def registry():
    return ToolRegistry(MappingProxyType({
        name: ToolDefinition(
            name,
            tuple((key, FIELD_KINDS.get(key, kind)) for key, kind in spec.arguments.items()),
            tuple((key, FIELD_KINDS[key]) for key in spec.returns),
            name in MUTATIONS, spec.function,
        ) for name, spec in TOOLS.items()
    }))


def require(condition, code, path, message):
    if not condition:
        raise ContractError(code, path, message)


def fields(value, required, optional, path):
    require(type(value) is dict, "wrong_type", path, "Expected an object")
    require(set(value) >= required, "missing_field", path, f"Required fields {sorted(required)}")
    require(set(value) <= required | optional, "extra_field", path, "Unexpected fields")


def identifier(value, path):
    require(type(value) is str and value.isidentifier(), "invalid_identifier", path, "Expected an identifier")
    return value


def parse_ref(raw, path):
    fields(raw, {"ref"}, set(), path)
    require(type(raw["ref"]) is str, "wrong_type", path, "Reference must be a string")
    parts = raw["ref"].split(".")
    require(1 <= len(parts) <= 2 and all(p.isidentifier() for p in parts), "invalid_reference", path, "Expected task or task.field")
    return ResultRef(parts[0], parts[1] if len(parts) == 2 else None)


def validate_integer_operand(operand, path):
    require(isinstance(operand, Literal) and type(operand.value) is str and 1 <= len(operand.value) <= 256 and operand.value.isascii() and operand.value.isdigit(), "invalid_transform", path, "parse_integer needs a grounded ASCII digit string of at most 256 characters")


def literal_spelling(value, path):
    require(type(value) in (str, int, bool), "wrong_type", path, "Only string, integer, and boolean literals are supported")
    if type(value) is int:
        require(-(10 ** 256) < value < 10 ** 256, "invalid_value", path, "Integer literals are limited to 256 digits")
    return str(value).lower() if type(value) is bool else str(value)


def parse_value(raw, request_id, text, path):
    require(type(raw) is dict, "missing_provenance", path, "Every argument needs source or result provenance")
    if "ref" in raw:
        return parse_ref(raw, path)
    if "transform" in raw:
        fields(raw, {"transform", "input"}, set(), path)
        require(raw["transform"] == "parse_integer", "unsupported_transform", path, "Only parse_integer is declared")
        require(type(raw["input"]) is dict and set(raw["input"]) == {"value", "source"}, "invalid_transform", path, "Transformation input must be a grounded literal")
        operand = parse_value(raw["input"], request_id, text, path + ".input")
        validate_integer_operand(operand, path)
        return Transform("parse_integer", operand)
    fields(raw, {"value", "source"}, set(), path)
    fields(raw["source"], {"start", "end"}, set(), path + ".source")
    start, end = raw["source"]["start"], raw["source"]["end"]
    require(type(start) is int and type(end) is int and 0 <= start < end <= len(text), "invalid_span", path, "Source range is outside request text")
    value = raw["value"]
    spelling = literal_spelling(value, path)
    require(text[start:end] == spelling, "ungrounded_value", path, "Literal differs from its source text")
    return Literal(value, SourceSpan(request_id, start, end))


def scalar_value(value):
    if isinstance(value, Literal):
        return value.value
    if isinstance(value, Transform):
        return int(value.operand.value)
    raise TypeError("A result reference is not yet a scalar")


def valid_scalar(value, kind, path):
    if kind == "integer":
        require(type(value) is int, "wrong_argument_type", path, "Expected integer, not boolean")
    elif kind == "boolean":
        require(type(value) is bool, "wrong_argument_type", path, "Expected boolean")
    else:
        require(type(value) is str and 0 < len(value) <= 256, "wrong_argument_type", path, "Expected a nonempty string of at most 256 characters")
        if kind.endswith("_id"):
            prefix = {"customer_id": "C", "order_id": "O", "product_id": "P", "return_id": "R", "booking_id": "B"}[kind]
            require(value.startswith(prefix) and value[1:].isdigit(), "invalid_value", path, "Identifier has the wrong semantic kind")
        if kind == "date":
            try:
                require(date.fromisoformat(value).isoformat() == value, "invalid_value", path, "Expected ISO date")
            except ValueError:
                raise ContractError("invalid_value", path, "Expected a valid ISO date")
        if kind == "status":
            require(value in ("confirmed", "cancelled"), "invalid_value", path, "Unknown booking status")


def validate_tool_result(value, spec, path):
    require(type(value) is dict and set(value) == set(dict(spec.result_fields)), "tool_result_contract", path, "Tool result fields differ from its declared schema")
    for field, kind in spec.result_fields:
        item = value[field]
        if kind in ("order_ids", "booking_ids"):
            require(type(item) is list, "tool_result_contract", path, "Expected an identifier list")
            for identifier_value in item:
                valid_scalar(identifier_value, kind[:-1], path + "." + field)
        elif kind == "options":
            require(type(item) is list and all(type(option) is dict and set(option) == {"destination", "date", "option_id"} for option in item), "tool_result_contract", path, "Expected alternative option records")
            for option in item:
                valid_scalar(option["destination"], "string", path)
                valid_scalar(option["date"], "date", path)
                valid_scalar(option["option_id"], "string", path)
        else:
            valid_scalar(item, kind, path + "." + field)


def value_kind(value, tasks, tools, path):
    if isinstance(value, ResultRef):
        require(value.task_id in tasks, "unknown_reference", path, "Referenced task does not exist")
        source = tasks[value.task_id]
        require(value.field is not None, "wrong_argument_type", path, "An argument needs a scalar result field")
        outputs = dict(tools.specs[source.intent].result_fields)
        require(value.field in outputs, "unknown_result_field", path, "Referenced output field does not exist")
        return outputs[value.field]
    scalar = scalar_value(value)
    return "boolean" if type(scalar) is bool else "integer" if type(scalar) is int else "string"


def graph(tasks):
    nodes = tuple(t.id for t in tasks.items)
    edges = set()
    for task in tasks.items:
        sources = set(task.depends_on) | {v.task_id for _, v in task.arguments if isinstance(v, ResultRef)}
        for source in sources:
            require(source in nodes, "unknown_dependency", task.id, "Dependency does not exist")
            edges.add((source, task.id))
    result = DependencyGraph(nodes, tuple(sorted(edges)))
    topological(result)
    return result


def topological(dependencies):
    remaining = set(dependencies.nodes)
    ordered = []
    while remaining:
        ready = [n for n in dependencies.nodes if n in remaining and not any(target == n and source in remaining for source, target in dependencies.edges)]
        require(bool(ready), "dependency_cycle", "tasks", "Dependency graph contains a cycle")
        chosen = ready[0]
        ordered.append(chosen)
        remaining.remove(chosen)
    return tuple(ordered)


def validate_request(request, tools, budget=Budget()):
    require(isinstance(request, Request), "wrong_input_type", "request", "Expected Request")
    require(len(request.tasks.items) <= budget.max_tasks, "task_budget", "tasks", "Too many tasks")
    tasks = {t.id: t for t in request.tasks.items}
    require(len(tasks) == len(request.tasks.items), "duplicate_task", "tasks", "Task IDs must be unique")
    for task in request.tasks.items:
        require(task.intent in tools.specs, "unsupported_intent", task.id, "Intent is not in the tool registry")
    for task in request.tasks.items:
        spec = tools.specs[task.intent]
        args = dict(task.arguments)
        require(len(args) == len(task.arguments) and set(args) == set(dict(spec.arguments)), "argument_fields", task.id, "Argument names must match the tool exactly")
        for key, kind in spec.arguments:
            value = args[key]
            path = f"{task.id}.{key}"
            if isinstance(value, ResultRef):
                require(value_kind(value, tasks, tools, path) == kind, "incompatible_reference_type", path, "Referenced field has a different semantic type")
            else:
                literal = value.operand if isinstance(value, Transform) else value
                require(isinstance(literal, Literal) and literal.source.request_id == request.id, "invalid_provenance", path, "Literal must belong to this request")
                source = literal.source
                require(type(source.start) is int and type(source.end) is int and 0 <= source.start < source.end <= len(request.text), "invalid_span", path, "Source range is invalid")
                spelling = literal_spelling(literal.value, path)
                require(request.text[source.start:source.end] == spelling, "ungrounded_value", path, "Literal differs from request source")
                if isinstance(value, Transform):
                    require(value.name == "parse_integer", "invalid_transform", path, "Unsupported transformation")
                    validate_integer_operand(literal, path)
                scalar = scalar_value(value)
                valid_scalar(scalar, kind, path)
                if key == "quantity":
                    require(1 <= scalar <= 100, "invalid_value", path, "Quantity must be in [1,100]")
    graph(request.tasks)
    if request.return_value is not None:
        result = request.return_value
        require(result.task_id in tasks, "unknown_reference", "return", "Return task does not exist")
        if result.field is not None:
            require(result.field in dict(tools.specs[tasks[result.task_id].intent].result_fields), "unknown_result_field", "return", "Unknown return field")
    require(bool(request.tasks.items) == (request.return_value is not None), "missing_return", "return", "Nonempty requests require a return reference; empty requests require null")


def parse_request(raw, tools=None, budget=Budget()):
    tools = tools or registry()
    require(type(raw) is dict, "wrong_type", "request", "Expected a structured request object")
    require("tasks" in raw, "needs_semantic_extraction", "tasks", "Free-form requests need a separately implemented extractor")
    fields(raw, {"id", "text", "tasks", "return"}, set(), "request")
    request_id = identifier(raw["id"], "id")
    require(type(raw["text"]) is str, "wrong_type", "text", "Request text must be a string")
    require(type(raw["tasks"]) is list, "wrong_type", "tasks", "Tasks must be a list")
    tasks = []
    for index, item in enumerate(raw["tasks"]):
        path = f"tasks[{index}]"
        fields(item, {"id", "intent", "arguments"}, {"depends_on"}, path)
        task_id = identifier(item["id"], path)
        require(type(item["intent"]) is str, "wrong_type", path, "Intent must be a string")
        require(type(item["arguments"]) is dict, "wrong_type", path, "Arguments must be an object")
        dependencies = item.get("depends_on", [])
        require(type(dependencies) is list and all(type(d) is str and d.isidentifier() for d in dependencies), "wrong_type", path, "Dependencies must be task identifiers")
        arguments = tuple((key, parse_value(value, request_id, raw["text"], f"{path}.{key}")) for key, value in item["arguments"].items())
        tasks.append(Task(task_id, item["intent"], arguments, tuple(dependencies)))
    result = None if raw["return"] is None else parse_ref(raw["return"], "return")
    request = Request(request_id, raw["text"], TaskList(tuple(tasks)), result)
    validate_request(request, tools, budget)
    return request


def parse_program(raw):
    return _parse_program(raw, 0)


def _parse_program(raw, depth):
    fields(raw, {"op"}, {"name", "children", "child"}, "program")
    if raw["op"] == "Primitive":
        fields(raw, {"op", "name"}, set(), "program")
        require(raw["name"] in ("Decompose", "Get-Order", "Match"), "unsupported_primitive", "program", "Unknown primitive")
        return PrimitiveNode(raw["name"])
    if raw["op"] == "Sequence":
        fields(raw, {"op", "children"}, set(), "program")
        require(depth < 2, "program_depth", "program", "Operator nesting exceeds the declared bound")
        require(type(raw["children"]) is list and bool(raw["children"]), "invalid_program", "program", "Sequence requires children")
        require(len(raw["children"]) <= 4, "program_width", "program", "Sequence width exceeds the declared bound")
        return Sequence(tuple(_parse_program(child, depth + 1) for child in raw["children"]))
    if raw["op"] == "Foreach":
        fields(raw, {"op", "child"}, set(), "program")
        require(depth < 2, "program_depth", "program", "Operator nesting exceeds the declared bound")
        return Foreach(_parse_program(raw["child"], depth + 1))
    raise ContractError("unsupported_operator", "program", "Only sequential Sequence and bounded Foreach are supported; Parallel and Loop are rejected")
