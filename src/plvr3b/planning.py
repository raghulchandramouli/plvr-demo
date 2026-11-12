from dataclasses import asdict, dataclass
from collections.abc import Mapping
from itertools import product
import json

from .contracts import require
from .executor import canonicalize, program_data, signature
from .types import Foreach, OPERATION_SPECS, PrimitiveNode, Sequence


@dataclass(frozen=True)
class SearchLimits:
    layers: int = 2
    operator_depth: int = 2
    beam_width: int = 4
    per_key_cap: int = 4
    max_layer_plans: int = 20
    max_candidates: int = 20

    def validate(self):
        for name, value in asdict(self).items():
            require(type(value) is int and value > 0, "invalid_search_limit", name, "Expected a positive integer")
        require(self.layers <= 2 and self.operator_depth == 2, "unsupported_search_bounds", "limits", "Current grammar supports at most two backward layers and operator depth two")
        require(self.beam_width <= 4 and self.per_key_cap <= 4 and self.max_layer_plans <= 20 and self.max_candidates <= 20,
                "unsupported_search_bounds", "limits", "Requested limits exceed the declared debugging envelope")


def structural_label(program):
    return json.dumps(program_data(program), sort_keys=True, separators=(",", ":"))


def typed_tree(program):
    return {
        "operation": program.operation,
        "name": program.name if isinstance(program, PrimitiveNode) else None,
        "parameters": dict(program.parameters),
        "required_inputs": list(program.required_inputs),
        "output_type": program.output_type,
        "constraints": list(program.constraints),
        "children": [typed_tree(child) for child in program.children],
    }


def stage_catalog(catalog=OPERATION_SPECS):
    require(isinstance(catalog, Mapping) and all(name in OPERATION_SPECS and spec == OPERATION_SPECS[name] for name, spec in catalog.items()),
            "unsupported_catalog", "catalog", "The oracle planner supports only unchanged subsets of its authoritative operation catalog")
    leaves = tuple(PrimitiveNode(name) for name in sorted(catalog))
    stages = list(leaves)
    for leaf in leaves:
        if (catalog[leaf.name].data_input, catalog[leaf.name].output_type) == ("Task", "ToolCall"):
            stages.append(Foreach(leaf))
    return tuple(stages)


def resolve_types(goal="ToolCallList", supplied=frozenset({"Request", "ToolRegistry"}), limits=SearchLimits(), catalog=OPERATION_SPECS, guided=False):
    limits.validate()
    supplied = frozenset(supplied)
    stages = stage_catalog(catalog)
    records = []
    suffixes = [stage for stage in stages if stage.output_type == goal]
    suffixes.sort(key=structural_label)
    if guided:
        suffixes = retain_frontier(suffixes, limits)
    plans, seen = [], set()
    emitted = 0

    def prefix(required, remaining):
        if required in supplied:
            yield ()
        if remaining == 0:
            return
        for stage in stages:
            if stage.output_type != required:
                continue
            if set(stage.required_inputs) - {stage.data_input} - supplied:
                continue
            for earlier in prefix(stage.data_input, remaining - 1):
                yield earlier + (stage,)

    for suffix in suffixes:
        if emitted >= limits.max_layer_plans:
            break
        emitted += 1
        missing = set(suffix.required_inputs) - supplied
        records.append({"layer": 1, "kind": "open" if missing else "closed", "required_inputs": sorted(missing), "tree": typed_tree(suffix), "emitted": emitted})
        if set(suffix.required_inputs) - {suffix.data_input} - supplied:
            continue
        if missing and limits.layers < 2:
            continue
        prefixes = prefix(suffix.data_input, 3) if missing else ((),)
        for before in prefixes:
            if emitted >= limits.max_layer_plans:
                break
            tree = canonicalize(Sequence(before + (suffix,)))
            key = structural_label(tree)
            if key in seen:
                continue
            _, produced = signature(tree)
            if produced != goal or set(tree.required_inputs) - supplied:
                continue
            seen.add(key)
            emitted += 1
            plans.append(tree)
            records.append({"layer": 2 if before else 1, "kind": "closed", "required_inputs": list(tree.required_inputs), "tree": typed_tree(tree), "emitted": emitted})
    return tuple(plans), {
        "goal": goal, "supplied": sorted(supplied), "layers": records,
        "emitted_layer_plans": emitted,
        "stop_reason": "plan_budget" if emitted >= limits.max_layer_plans else "type_space_exhausted",
    }


def retain_frontier(plans, limits, key=None):
    counts, kept = {}, []
    ordered = sorted(plans, key=key or structural_label)
    for plan in ordered:
        group = (plan.required_inputs, plan.output_type)
        if counts.get(group, 0) >= limits.per_key_cap:
            continue
        kept.append(plan)
        counts[group] = counts.get(group, 0) + 1
        if len(kept) >= limits.beam_width:
            break
    return tuple(kept)


def bind_parameters(plans, domains):
    result = []
    for plan in plans:
        names = []

        def visit(node):
            if isinstance(node, PrimitiveNode):
                names.append(node.name)
            for child in node.children:
                visit(child)

        visit(plan)
        for name in names:
            require(name in domains and domains[name] == [{}], "unsupported_parameter_domain", name, "Current oracle primitives declare only the empty parameter record")
        for bindings in product(*(domains[name] for name in names)):
            result.append((plan, {str(i): {"primitive": name, "parameters": binding} for i, (name, binding) in enumerate(zip(names, bindings))}))
    return tuple(result)
