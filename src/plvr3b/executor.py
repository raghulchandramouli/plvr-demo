from copy import deepcopy
from dataclasses import asdict

from .contracts import graph, require, scalar_value, valid_scalar, validate_request, validate_tool_result
from .primitives import PRIMITIVES
from .types import (
    Budget, Context, ContractError, Execution, Foreach, PrimitiveNode, ResultRef,
    Scratchpad, Sequence, TaskList, ToolCall, ToolCallList, OPERATION_SPECS,
)


SIGNATURES = {name: (spec.data_input, spec.output_type) for name, spec in OPERATION_SPECS.items()}


def program_data(program):
    if isinstance(program, PrimitiveNode):
        return {"op": "Primitive", "name": program.name}
    if isinstance(program, Foreach):
        return {"op": "Foreach", "child": program_data(program.child)}
    if isinstance(program, Sequence):
        return {"op": "Sequence", "children": [program_data(child) for child in program.children]}
    raise ContractError("unsupported_operator", "program", "Unknown AST node")


def canonicalize(program):
    if isinstance(program, Sequence):
        children = []
        for child in program.children:
            normalized = canonicalize(child)
            nodes = normalized.children if isinstance(normalized, Sequence) else (normalized,)
            for node in nodes:
                if node == PrimitiveNode("Get-Order") and children and children[-1] == node:
                    continue
                children.append(node)
        return children[0] if len(children) == 1 else Sequence(tuple(children))
    if isinstance(program, Foreach):
        return Foreach(canonicalize(program.child))
    return program


def signature(program, max_depth=2, max_width=4, depth=0):
    require(not program.parameters, "unsupported_parameters", "program", "Current parameter domains contain only empty records")
    if isinstance(program, PrimitiveNode):
        require(program.name in SIGNATURES, "unsupported_primitive", "program", "Unknown primitive")
        return SIGNATURES[program.name]
    require(depth < max_depth, "program_depth", "program", "Operator nesting exceeds the declared bound")
    if isinstance(program, Foreach):
        require(isinstance(program.child, PrimitiveNode), "incompatible_program", "Foreach", "Nested loops and composed loop bodies are unsupported")
        require(signature(program.child) == ("Task", "ToolCall"), "incompatible_program", "Foreach", "Foreach requires Task to ToolCall")
        return "TaskList", "ToolCallList"
    require(isinstance(program, Sequence) and 1 <= len(program.children) <= max_width, "program_width", "Sequence", "Sequence width is outside the declared bound")
    inputs, current = signature(program.children[0], max_depth, max_width, depth + 1)
    for child in program.children[1:]:
        needed, produced = signature(child, max_depth, max_width, depth + 1)
        require(current == needed, "incompatible_program", "Sequence", f"Cannot pass {current} to {needed}")
        current = produced
    return inputs, current


def ordered_program():
    return Sequence((PrimitiveNode("Decompose"), PrimitiveNode("Get-Order"), Foreach(PrimitiveNode("Match"))))


def unordered_program():
    return Sequence((PrimitiveNode("Decompose"), Foreach(PrimitiveNode("Match"))))


def invoke(name, value, context, scratch):
    require(len(scratch.primitive_events) < context.budget.max_primitive_calls, "primitive_budget", name, "Primitive invocation budget exhausted")
    primitive = PRIMITIVES[name]
    event = {"name": name, "input_type": type(value).__name__, "status": "attempted"}
    scratch.primitive_events.append(event)
    try:
        output = primitive.run(value, context)
    except ContractError:
        event["status"] = "failed"
        raise
    event.update(output_type=type(output).__name__, status="executed")
    violations = primitive.validate(value, output, context)
    if violations:
        event["status"] = "failed"
        error = ContractError(violations[0].code, violations[0].path, violations[0].message)
        raise error
    return output


def compose(program, value, context, scratch):
    if isinstance(program, PrimitiveNode):
        return invoke(program.name, value, context, scratch)
    if isinstance(program, Sequence):
        for child in program.children:
            value = compose(child, value, context, scratch)
        return value
    require(isinstance(value, TaskList), "wrong_input_type", "Foreach", "Expected TaskList")
    require(len(value.items) <= context.budget.max_iterations, "iteration_budget", "Foreach", "Foreach iteration limit exceeded")
    return ToolCallList(tuple(compose(program.child, task, context, scratch) for task in value.items))


def resolve(value, scratch):
    if isinstance(value, ResultRef):
        require(value.task_id in scratch.results, "unresolved_reference", value.task_id, "Referenced result has not executed")
        record = scratch.results[value.task_id]
        if value.field is None:
            return deepcopy(record["value"]), asdict(value)
        require(value.field in record["value"], "unknown_result_field", value.task_id, "Result field is missing")
        return deepcopy(record["value"][value.field]), asdict(value)
    return scalar_value(value), asdict(value)


def execute_calls(calls, context, scratch):
    require(isinstance(calls, ToolCallList), "wrong_input_type", "calls", "Expected ToolCallList")
    require(len(calls.items) <= context.budget.max_tool_calls, "tool_budget", "calls", "Tool call budget exceeded")
    ids = [call.id for call in calls.items]
    require(len(ids) == len(set(ids)), "duplicate_call", "calls", "Call IDs must be unique")
    tasks = {task.id: task for task in context.request.tasks.items}
    require(set(ids) <= set(tasks), "unrequested_call", "calls", "An extra call was not requested")
    require(set(ids) == set(tasks), "missing_call", "calls", "A requested action is missing")
    edges = graph(context.request.tasks).edges
    for call in calls.items:
        task = tasks[call.id]
        require(call == ToolCall(task.id, task.intent, task.arguments), "binding_contract", call.id, "Call differs from requested tool or grounded bindings")
        for source, target in edges:
            if target == call.id:
                require(source in scratch.results, "unresolved_dependency", call.id, "A required earlier task has not executed")
        spec = context.registry.specs[call.tool]
        args, origins = {}, {}
        for key, value in call.arguments:
            args[key], origins[key] = resolve(value, scratch)
            valid_scalar(args[key], dict(spec.arguments)[key], f"{call.id}.{key}")
            if key == "quantity":
                require(1 <= args[key] <= 100, "invalid_value", key, "Quantity must be in [1,100]")
        try:
            result = spec.function(scratch.state, **args)
        except (ValueError, KeyError, TypeError) as error:
            raise ContractError("tool_precondition", call.id, str(error)) from error
        validate_tool_result(result, spec, call.id)
        provenance = {field: asdict(ResultRef(call.id, field)) for field in result}
        scratch.results[call.id] = {"value": deepcopy(result), "provenance": provenance}
        scratch.events.append({"id": call.id, "tool": call.tool, "arguments": deepcopy(args), "argument_provenance": origins, "result": deepcopy(result), "result_provenance": provenance, "state_after": deepcopy(scratch.state)})
    return resolve(context.request.return_value, scratch)[0] if context.request.return_value else None


def run_program(program, request, tools, initial_state, budget=Budget()):
    scratch = Scratchpad(request, tools, budget, deepcopy(initial_state))
    context = Context(request, tools, budget)
    try:
        validate_request(request, tools, budget)
        require(signature(program) == ("Request", "ToolCallList"), "incompatible_program", "program", "Complete programs require Request and produce ToolCallList")
        calls = compose(program, request, context, scratch)
        result = execute_calls(calls, context, scratch)
        return Execution("executed", result, scratch.state, scratch.events, scratch.primitive_events, (), scratch.results)
    except ContractError as error:
        return Execution("contract_failure", None, scratch.state, scratch.events, scratch.primitive_events, error.violations, scratch.results)
