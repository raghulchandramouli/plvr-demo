from dataclasses import asdict, replace
from pathlib import Path
import hashlib
import json
import platform
import random

import yaml

from .cache import RuntimeCache, VersionStamp
from .contracts import parse_request, registry, require
from .executor import program_data, run_program
from .phase2_fixtures import build_fixtures, fixture_data
from .planning import SearchLimits, bind_parameters, resolve_types, retain_frontier, structural_label, typed_tree
from .scoring import aggregate_rank, ranking_terms, score_fixture, selection_key, tree_size
from .search import behavioral_record, encoded, enumerate_programs, fingerprint
from .types import Budget


def evaluate_candidate(program, fixtures, tools, budget, cache, incumbent_inputs):
    outcomes, contributions, traces, behaviors = [], [], [], []
    before = dict(cache.stats) if cache else {}
    for fixture in fixtures:
        request = parse_request(fixture.payload, tools, budget)
        execution = run_program(program, request, tools, fixture.initial_state, budget, cache)
        score = score_fixture(execution, fixture)
        contribution = ranking_terms(execution, fixture)
        contributions.append(contribution)
        traces.append({"fixture_id": fixture.id, "execution": asdict(execution), "score": score, "rank_terms": contribution})
        outcomes.append({"fixture_id": fixture.id, "success": score["success"]})
        behaviors.append(behavioral_record(execution, score, False))
    after = dict(cache.stats) if cache else {}
    return {
        "id": fingerprint(program_data(program)), "program": program_data(program), "typed_tree": typed_tree(program),
        "label": structural_label(program), "size": tree_size(program),
        "rank": aggregate_rank(contributions, program.required_inputs, incumbent_inputs),
        "successes": sum(o["success"] for o in outcomes), "fixtures": len(fixtures),
        "outcomes": outcomes, "traces": traces, "behavior_id": fingerprint(behaviors),
        "logical_primitive_calls": sum(len(t["execution"]["primitive_events"]) for t in traces),
        "logical_tool_calls": sum(len(t["execution"]["events"]) for t in traces),
        "cache_delta": {key: after[key] - before[key] for key in before},
    }


def run_arm(method, fixtures, tools, budget, limits, versions, domains, reference, seed=None):
    limits.validate()
    require(limits.layers == 2, "unsupported_comparison_layers", "limits", "The shared comparison admits only the declared two-layer grammar")
    cache = RuntimeCache(versions)
    derivation = None
    if method == "exhaustive":
        plans = tuple(program for _, program in reference)
    else:
        construction_limits = limits if method == "guided" else replace(limits, max_layer_plans=20)
        plans, derivation = resolve_types(limits=construction_limits, guided=method == "guided")
        if method == "random":
            derivation["purpose"] = "Full admissible pool construction before seeded sampling and arm emission caps"
    bound = bind_parameters(plans, domains)
    plans = [plan for plan, _ in bound]
    proposed_count = len(plans)
    if method == "random":
        random.Random(seed).shuffle(plans)
    incumbent = frozenset({"Request", "ToolRegistry"})
    results = []
    closed_cap = min(limits.max_candidates, limits.max_layer_plans)
    max_fixture_evaluations = closed_cap * len(fixtures)
    for program in plans:
        if len(results) >= closed_cap:
            break
        results.append(evaluate_candidate(program, fixtures, tools, budget, cache, incumbent))
    frontier = results
    if method == "guided":
        keyed = {c["label"]: c for c in results}
        retained = retain_frontier([p for p in plans[:len(results)]], limits, key=lambda p: selection_key(keyed[structural_label(p)]))
        frontier = [keyed[structural_label(p)] for p in retained]
    selected = min(frontier, key=selection_key) if frontier else None
    return {
        "method": method, "seed": seed, "distribution": "uniform permutation of canonical type-valid bound programs without replacement" if method == "random" else None,
        "cache_start": "cold", "fixture_ids": [f.id for f in fixtures], "fixture_manifest_hash": fingerprint(fixture_data(fixtures)),
        "limits": asdict(limits), "runtime_budget": asdict(budget), "max_fixture_evaluations": max_fixture_evaluations,
        "evaluated_candidates": len(results), "fixture_evaluations": len(results) * len(fixtures),
        "proposed_canonical_candidates": proposed_count,
        "emitted_layer_plans": derivation["emitted_layer_plans"] if method == "guided" else len(results),
        "emitted_closed_plans": len(results), "max_closed_candidates": closed_cap, "candidate_order": [c["id"] for c in results],
        "stop_reason": ("candidate_budget" if limits.max_candidates <= limits.max_layer_plans else "plan_budget") if len(results) < len(plans) else (derivation["stop_reason"] if derivation else "type_space_exhausted"),
        "derivation": derivation, "parameter_bindings": [bindings for _, bindings in bound],
        "beam_before": len(results) if method == "guided" else None,
        "beam_after": len(frontier) if method == "guided" else None,
        "selected_id": selected["id"] if selected else None,
        "selected_rank": selected["rank"] if selected else None,
        "selected_successes": selected["successes"] if selected else None,
        "retained_ids": [c["id"] for c in frontier], "cache_stats": dict(cache.stats),
        "cache_events": cache.events, "candidates": results,
    }


def run_comparison(root, output):
    root, output = Path(root), Path(output)
    config = yaml.safe_load((root / "configs/search.yaml").read_text())
    settings = config["phase3"]
    limits = SearchLimits(**settings["limits"])
    limits.validate()
    require(settings["random_seeds"] == [0, 1, 2], "unsupported_search_policy", "random_seeds", "Declared comparison uses three search seeds")
    budget = Budget(**{key: config[key] for key in ("max_tasks", "max_tool_calls", "max_iterations", "max_primitive_calls")})
    model = yaml.safe_load((root / "configs/model.yaml").read_text())
    versions = VersionStamp.from_root(root, model)
    fixtures, tools = build_fixtures(), registry()
    reference, enumeration = enumerate_programs()
    backward, derivation = resolve_types(limits=SearchLimits())
    require({fingerprint(program_data(p)) for p in backward} == {key for key, _ in reference}, "reference_mismatch", "search", "Backward construction and independent exhaustive reference disagree")
    oracle_checks = [evaluate_candidate(p, fixtures, tools, budget, None, {"Request", "ToolRegistry"}) for _, p in reference]
    require(any(c["successes"] == len(fixtures) for c in oracle_checks), "unsolvable_fixtures", "search", "No reference candidate solves all admitted fixtures")
    arms = [run_arm(method, fixtures, tools, budget, limits, versions, config["primitive_parameters"], reference if method == "exhaustive" else ())
            for method in ("exhaustive", "guided")]
    arms.extend(run_arm("random", fixtures, tools, budget, limits, versions, config["primitive_parameters"], (), seed) for seed in settings["random_seeds"])
    output.mkdir(parents=True, exist_ok=False)
    (output / "arms").mkdir()
    (output / "fixtures.json").write_text(encoded(fixture_data(fixtures)))
    (output / "enumeration.json").write_text(encoded(enumeration))
    (output / "backward_resolution.json").write_text(encoded(derivation))
    (output / "oracle_checks.json").write_text(encoded(oracle_checks))
    arm_summaries = []
    for arm in arms:
        label = arm["method"] + (f"-{arm['seed']}" if arm["seed"] is not None else "")
        directory = output / "arms" / label
        directory.mkdir()
        for candidate in arm["candidates"]:
            (directory / f"{candidate['id']}.json").write_text(encoded(candidate))
        summary = {key: value for key, value in arm.items() if key != "candidates"}
        (directory / "summary.json").write_text(encoded(summary))
        arm_summaries.append(summary)
    source_names = ["types.py", "planning.py", "cache.py", "executor.py", "scoring.py", "search.py", "comparison.py", "contracts.py", "phase2_fixtures.py", "primitives/__init__.py", "baseline_tasks.py"]
    paths = [root / "src/plvr3b" / n for n in source_names] + [root / n for n in ("configs/search.yaml", "configs/model.yaml", "pyproject.toml", "uv.lock", "scripts/run_search.py")]
    metadata = {"python": platform.python_version(), "platform": platform.platform(), "config": config, "versions": asdict(versions),
                "source_hashes": {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
                "fixtures_sha256": fingerprint(fixture_data(fixtures)), "model_calls": 0,
                "preflight_fixture_evaluations": len(oracle_checks) * len(fixtures), "preflight_cache": "disabled",
                "selection": "Aggregate four-field rank, then canonical AST size, then canonical label; fixed incumbent inputs Request+ToolRegistry.",
                "cost_notes": "Logical requests include cache replays. Main-path primitive run attempts equal primitive cache misses; validation recomputation is excluded. Execution cache hits replay trace evidence without rerunning primitives. Validation CPU excluded.",
                "paper_matched": False}
    (output / "metadata.json").write_text(encoded(metadata))
    success = all(a["selected_successes"] == len(fixtures) for a in arms)
    summary = {"status": "verified" if success else "not_verified", "gate": "mechanics_only", "guided_search_advantage": False,
               "raw_syntax_count": len(enumeration), "type_valid_syntax_count": sum(r["type_valid"] for r in enumeration),
               "canonical_program_count": len(reference), "distinct_semantic_behavior_count": len({c["behavior_id"] for c in oracle_checks}),
               "fixture_count": len(fixtures), "discovery_without_program_template": True, "model_calls": 0,
               "arms": [{k: a[k] for k in ("method", "seed", "selected_id", "selected_rank", "selected_successes", "evaluated_candidates", "fixture_evaluations", "cache_stats", "stop_reason")} for a in arms]}
    (output / "comparison_summary.json").write_text(encoded(summary))
    require(success, "discovery_gate_failed", "search", "At least one arm failed to discover a fully solving program within the declared budget")
    return summary
