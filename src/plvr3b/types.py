from dataclasses import dataclass, field
from typing import Callable, Mapping
from types import MappingProxyType


@dataclass(frozen=True)
class ContractViolation:
    code: str
    path: str
    message: str


class ContractError(ValueError):
    def __init__(self, code, path, message):
        self.violations = (ContractViolation(code, path, message),)
        super().__init__(f"{code} at {path}: {message}")


@dataclass(frozen=True)
class SourceSpan:
    request_id: str
    start: int
    end: int


@dataclass(frozen=True)
class Literal:
    value: str | int | bool
    source: SourceSpan


@dataclass(frozen=True)
class ResultRef:
    task_id: str
    field: str | None = None


@dataclass(frozen=True)
class Transform:
    name: str
    operand: Literal


Value = Literal | ResultRef | Transform


@dataclass(frozen=True)
class Task:
    id: str
    intent: str
    arguments: tuple[tuple[str, Value], ...]
    depends_on: tuple[str, ...] = ()


@dataclass(frozen=True)
class TaskList:
    items: tuple[Task, ...]


@dataclass(frozen=True)
class Request:
    id: str
    text: str
    tasks: TaskList
    return_value: ResultRef | None


@dataclass(frozen=True)
class DependencyGraph:
    nodes: tuple[str, ...]
    edges: tuple[tuple[str, str], ...]


@dataclass(frozen=True)
class ToolDefinition:
    name: str
    arguments: tuple[tuple[str, str], ...]
    result_fields: tuple[tuple[str, str], ...]
    mutates: bool
    function: Callable


@dataclass(frozen=True)
class ToolRegistry:
    specs: Mapping[str, ToolDefinition]


@dataclass(frozen=True)
class ToolCall:
    id: str
    tool: str
    arguments: tuple[tuple[str, Value], ...]


@dataclass(frozen=True)
class ToolCallList:
    items: tuple[ToolCall, ...]


@dataclass(frozen=True)
class OperationSpec:
    data_input: str
    output_type: str
    context_inputs: tuple[str, ...] = ()
    version: str = "oracle-v1"


OPERATION_SPECS = MappingProxyType({
    "Decompose": OperationSpec("Request", "TaskList"),
    "Get-Order": OperationSpec("TaskList", "TaskList"),
    "Match": OperationSpec("Task", "ToolCall", ("ToolRegistry",)),
})


def node_metadata(node, operation, data_input, output_type, requirements, constraints):
    for name, value in (
        ("operation", operation), ("data_input", data_input),
        ("output_type", output_type), ("required_inputs", tuple(sorted(requirements))),
        ("constraints", constraints),
    ):
        object.__setattr__(node, name, value)


@dataclass(frozen=True)
class PrimitiveNode:
    name: str
    parameters: tuple[tuple[str, object], ...] = ()
    operation: str = field(init=False)
    children: tuple = field(init=False, default=())
    data_input: str = field(init=False)
    required_inputs: tuple[str, ...] = field(init=False)
    output_type: str = field(init=False)
    constraints: tuple[str, ...] = field(init=False)

    def __post_init__(self):
        spec = OPERATION_SPECS.get(self.name, OperationSpec("<unknown>", "<unknown>"))
        node_metadata(self, "Primitive", spec.data_input, spec.output_type,
                      {spec.data_input, *spec.context_inputs}, ("declared_parameters_only",))


@dataclass(frozen=True)
class Sequence:
    children: tuple
    parameters: tuple[tuple[str, object], ...] = ()
    operation: str = field(init=False)
    data_input: str = field(init=False)
    required_inputs: tuple[str, ...] = field(init=False)
    output_type: str = field(init=False)
    constraints: tuple[str, ...] = field(init=False)

    def __post_init__(self):
        first = self.children[0] if self.children else None
        data_input = getattr(first, "data_input", "<unknown>")
        current = data_input
        requirements = {data_input}
        compatible = bool(self.children)
        for child in self.children:
            needed = getattr(child, "data_input", "<unknown>")
            compatible = compatible and current == needed
            requirements.update(set(getattr(child, "required_inputs", ())) - {needed})
            if current != needed:
                requirements.add(needed)
            current = getattr(child, "output_type", "<unknown>")
        node_metadata(self, "Sequence", data_input, current, requirements,
                      ("operator_depth<=2", "width<=4", "compatible" if compatible else "incompatible"))


@dataclass(frozen=True)
class Foreach:
    child: PrimitiveNode
    parameters: tuple[tuple[str, object], ...] = ()
    operation: str = field(init=False)
    children: tuple = field(init=False)
    data_input: str = field(init=False)
    required_inputs: tuple[str, ...] = field(init=False)
    output_type: str = field(init=False)
    constraints: tuple[str, ...] = field(init=False)

    def __post_init__(self):
        object.__setattr__(self, "children", (self.child,))
        input_type = {"Task": "TaskList"}.get(getattr(self.child, "data_input", None), "<unknown>")
        output = {"ToolCall": "ToolCallList"}.get(getattr(self.child, "output_type", None), "<unknown>")
        requirements = (set(getattr(self.child, "required_inputs", ())) - {getattr(self.child, "data_input", None)}) | {input_type}
        node_metadata(self, "Foreach", input_type, output, requirements,
                      ("single_primitive_body", "sequential", "iterations<=budget.max_iterations"))


Program = PrimitiveNode | Sequence | Foreach


@dataclass(frozen=True)
class Budget:
    max_tasks: int = 6
    max_tool_calls: int = 6
    max_iterations: int = 6
    max_primitive_calls: int = 16


@dataclass
class Scratchpad:
    request: Request
    registry: ToolRegistry
    budget: Budget
    state: dict
    results: dict = field(default_factory=dict)
    events: list = field(default_factory=list)
    primitive_events: list = field(default_factory=list)


@dataclass(frozen=True)
class Context:
    request: Request
    registry: ToolRegistry
    budget: Budget


@dataclass
class Execution:
    status: str
    result: object
    final_state: dict
    events: list
    primitive_events: list
    violations: tuple[ContractViolation, ...]
    result_provenance: dict


@dataclass(frozen=True)
class Fixture:
    id: str
    payload: dict
    initial_state: dict
    expected_result: object
    expected_state: dict
    expected_actions: tuple[dict, ...]
