from ..contracts import graph, require, topological
from ..types import ContractError, Request, Task, TaskList, ToolCall


class Primitive:
    input_type = object

    def validate(self, value, output, context):
        try:
            require(isinstance(value, self.input_type), "wrong_input_type", self.name, f"Expected {self.input_type.__name__}")
            self.check(value, output, context)
        except ContractError as error:
            return error.violations
        return ()


class Decompose(Primitive):
    name = "Decompose"
    input_type = Request

    def run(self, value, context):
        require(isinstance(value, Request) and value == context.request, "wrong_input_type", self.name, "Expected the preserved Request")
        return value.tasks

    def check(self, value, output, context):
        require(value == context.request and isinstance(output, TaskList) and output == value.tasks, "decompose_contract", self.name, "All task identities, bindings, order, and provenance must be preserved")


class GetOrder(Primitive):
    name = "Get-Order"
    input_type = TaskList

    def run(self, value, context):
        require(isinstance(value, TaskList), "wrong_input_type", self.name, "Expected TaskList")
        require({t.id: t for t in value.items} == {t.id: t for t in context.request.tasks.items} and len(value.items) == len(context.request.tasks.items), "get_order_contract", self.name, "Input must preserve every request task")
        order = topological(graph(value))
        tasks = {task.id: task for task in value.items}
        return TaskList(tuple(tasks[task_id] for task_id in order))

    def check(self, value, output, context):
        require(isinstance(output, TaskList), "get_order_contract", self.name, "Expected TaskList")
        expected = self.run(value, context)
        require(output == expected, "get_order_contract", self.name, "Stable topological order must preserve every task and binding")


class Match(Primitive):
    name = "Match"
    input_type = Task

    def run(self, value, context):
        require(isinstance(value, Task) and value in context.request.tasks.items, "wrong_input_type", self.name, "Expected a task from the preserved Request")
        require(value.intent in context.registry.specs, "unsupported_intent", value.id, "Unknown intent")
        return ToolCall(value.id, value.intent, value.arguments)

    def check(self, value, output, context):
        require(value in context.request.tasks.items, "match_contract", self.name, "Task is not part of this request")
        require(isinstance(output, ToolCall) and output == ToolCall(value.id, value.intent, value.arguments), "match_contract", self.name, "Tool identity, arguments, and provenance must match the task")


PRIMITIVES = {p.name: p for p in (Decompose(), GetOrder(), Match())}
