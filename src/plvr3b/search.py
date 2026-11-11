from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
from itertools import product
import json
from pathlib import Path
import platform
import sys

import yaml

from .contracts import parse_request, registry, require
from .executor import canonicalize, ordered_program, program_data, run_program, signature
from .phase2_fixtures import build_fixtures, fixture_data
from .scoring import score_fixture
from .types import Budget, ContractError, Foreach, PrimitiveNode, Sequence


def encoded(value):
    return json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"


def fingerprint(value):
    return hashlib.sha256(encoded(value).encode()).hexdigest()


def enumerate_programs(max_depth=2, max_width=4):
    require(max_depth == 2 and type(max_width) is int and 1 <= max_width <= 4, "unsupported_search_bounds", "search", "This finite grammar requires depth 2 and sequence width in [1,4]")
    primitives = tuple(PrimitiveNode(name) for name in ("Decompose", "Get-Order", "Match"))
    stages = primitives + tuple(Foreach(p) for p in primitives)
    records, unique = [], {}
    for width in range(1, max_width + 1):
        for children in product(stages, repeat=width):
            tree = Sequence(children)
            record = {"program": program_data(tree), "type_valid": False}
            try:
                require(signature(tree, max_depth, max_width) == ("Request", "ToolCallList"), "incompatible_program", "program", "Complete signature must be Request to ToolCallList")
                canonical = canonicalize(tree)
                key = fingerprint(program_data(canonical))
                record.update(type_valid=True, canonical_id=key)
                unique[key] = canonical
            except ContractError as error:
                record["rejection"] = asdict(error.violations[0])
            records.append(record)
    return tuple(unique.items()), records


def behavioral_record(execution, score, include_cost):
    record = {
        "success": score["success"], "status": execution.status,
        "violations": [v.code for v in execution.violations], "result": execution.result,
        "final_state": execution.final_state,
        "tool_events": [{"tool": e["tool"], "arguments": e["arguments"], "result": e["result"]} for e in execution.events],
    }
    if include_cost:
        record["primitive_invocations"] = len(execution.primitive_events)
    return record


def run_gate(root, output):
    root, output = Path(root), Path(output)
    config = yaml.safe_load((root / "configs/search.yaml").read_text())
    require(config["version"] == "phase2-oracle-v1" and config["primitive_names"] == ["Decompose", "Get-Order", "Match"] and all(config["primitive_parameters"][name] == [{}] for name in config["primitive_names"]), "unsupported_search_policy", "search", "Primitive domains are fixed singleton parameter records")
    require(config["useful_definition"] == "success_on_every_admitted_fixture" and config["cost_metric"] == "executed_oracle_primitive_invocations", "unsupported_search_policy", "search", "Gate definitions are fixed before execution")
    require(config["max_operator_depth"] == 2 and config["max_sequence_width"] == 4, "unsupported_search_policy", "search", "The gate freezes depth 2 and width 4")
    budget = Budget(**{key: config[key] for key in ("max_tasks", "max_tool_calls", "max_iterations", "max_primitive_calls")})
    tools, fixtures = registry(), build_fixtures()
    oracle_checks = []
    for fixture in fixtures:
        request = parse_request(fixture.payload, tools, budget)
        execution = run_program(ordered_program(), request, tools, fixture.initial_state, budget)
        score = score_fixture(execution, fixture)
        require(score["success"], "invalid_oracle", fixture.id, f"Oracle cannot solve admitted fixture: {score['violations']}")
        oracle_checks.append({"fixture_id": fixture.id, "success": True})
    programs, enumeration = enumerate_programs(config["max_operator_depth"], config["max_sequence_width"])
    output.mkdir(parents=True, exist_ok=False)
    (output / "traces").mkdir()
    data = fixture_data(fixtures)
    (output / "fixtures.json").write_text(encoded(data))
    (output / "oracle_checks.json").write_text(encoded(oracle_checks))
    (output / "enumeration.json").write_text(encoded(enumeration))
    source_paths = [root / "configs/search.yaml", root / "scripts/run_search.py"] + [root / "src/plvr3b" / name for name in ("types.py", "contracts.py", "executor.py", "scoring.py", "search.py", "phase2_fixtures.py", "primitives/__init__.py", "baseline_tasks.py")]
    metadata = {
        "version": config["version"], "created_at": datetime.now(timezone.utc).isoformat(),
        "python": sys.version, "platform": platform.platform(), "config": config,
        "source_hashes": {str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest() for path in source_paths},
        "fixtures_sha256": fingerprint(data), "model_calls": 0,
        "cost_notes": "Counts attempted oracle run invocations only; validation and contract-check CPU are excluded. Not neural calls, tokens, latency, or demonstrated compute savings.",
    }
    (output / "metadata.json").write_text(encoded(metadata))
    candidates = []
    for candidate_id, program in programs:
        outcomes, vector, semantic_vector = [], [], []
        for fixture in fixtures:
            request = parse_request(fixture.payload, tools, budget)
            execution = run_program(program, request, tools, fixture.initial_state, budget)
            score = score_fixture(execution, fixture)
            trace = {"fixture_id": fixture.id, "candidate_id": candidate_id, "program": program_data(program), "execution": asdict(execution), "score": score}
            (output / "traces" / f"{candidate_id[:12]}-{fixture.id}.json").write_text(encoded(trace))
            outcomes.append({"fixture_id": fixture.id, "success": score["success"], "primitive_calls": len(execution.primitive_events), "tool_calls": len(execution.events), "violation_codes": [v.code for v in execution.violations]})
            vector.append(behavioral_record(execution, score, True))
            semantic_vector.append(behavioral_record(execution, score, False))
        candidates.append({
            "id": candidate_id, "program": program_data(program), "outcomes": outcomes,
            "successes": sum(o["success"] for o in outcomes), "fixtures": len(fixtures),
            "primitive_calls": sum(o["primitive_calls"] for o in outcomes),
            "tool_calls": sum(o["tool_calls"] for o in outcomes),
            "behavior_id": fingerprint(vector), "semantic_behavior_id": fingerprint(semantic_vector),
        })
    useful = [c["id"] for c in candidates if c["successes"] == len(fixtures)]
    pareto = [c["id"] for c in candidates if not any(other["successes"] >= c["successes"] and other["primitive_calls"] <= c["primitive_calls"] and (other["successes"] > c["successes"] or other["primitive_calls"] < c["primitive_calls"]) for other in candidates)]
    summary = {
        "status": "verified", "gate": "limited", "verdict": "exhaustive_only", "guided_search_justified": False,
        "reason": f"The declared grammar has {len(candidates)} canonical programs and {len(useful)} workflows succeeding on every admitted fixture. This trivially small space requires exhaustive evaluation, not guided-search claims.",
        "raw_syntax_count": len(enumeration), "type_valid_syntax_count": sum(e["type_valid"] for e in enumeration),
        "canonical_program_count": len(candidates), "distinct_behavior_count": len({c["behavior_id"] for c in candidates}),
        "distinct_semantic_behavior_count": len({c["semantic_behavior_id"] for c in candidates}),
        "useful_workflows": useful, "useful_workflow_count": len(useful), "pareto_programs": pareto,
        "fixture_count": len(fixtures), "oracle_checks_passed": len(oracle_checks), "candidates": candidates,
        "model_calls": 0, "cost_metric": config["cost_metric"],
    }
    (output / "gate_summary.json").write_text(encoded(summary))
    return summary
