"""Score the test arms for one student against the pre-registered claim C1, and audit every record.

Usage: python -m plvr.score --student qwen3-8b
"""

import argparse
import json
import random
from math import comb
from pathlib import Path

C1_MIN_GAIN = 20.0
C1_MAX_P = 0.01
BOOTSTRAP = 10_000


def load(arm: str, student: str) -> dict[str, dict]:
    return {r["task_id"]: r for r in (json.loads(p.read_text()) for p in Path(f"runs/test/{arm}-{student}").glob("*.json"))}


def mcnemar_exact(program_only: int, vanilla_only: int) -> float:
    n = program_only + vanilla_only
    if n == 0:
        return 1.0
    tail = sum(comb(n, k) for k in range(min(program_only, vanilla_only) + 1)) / 2**n
    return min(1.0, 2 * tail)


def bootstrap_gain(pairs: list[tuple[int, int]], seed: int = 0) -> tuple[float, float]:
    rng = random.Random(seed)
    gains = sorted(
        100 * sum(p - v for p, v in (rng.choice(pairs) for _ in pairs)) / len(pairs) for _ in range(BOOTSTRAP)
    )
    return gains[int(0.025 * BOOTSTRAP)], gains[int(0.975 * BOOTSTRAP) - 1]


def verdict(gain: float, p: float, upper: float) -> str:
    if gain >= C1_MIN_GAIN and p < C1_MAX_P:
        return "VERIFIED"
    if upper < C1_MIN_GAIN:
        return "NOT VERIFIED"
    return "INCONCLUSIVE"


def audit(name: str, records: dict[str, dict]) -> None:
    retried = [r for r in records.values() if r.get("failed_attempts")]
    infra = [r for r in records.values() if r.get("termination_reason") == "infrastructure_error"]
    memory = [r for r in retried if any("Insufficient Memory" in e or "OutOfMemory" in e for e in r["failed_attempts"])]
    print(f"  {name}: {len(records)} records, {len(retried)} retried, {len(memory)} retried after a GPU memory error, "
          f"{len(infra)} scored 0 as infrastructure errors")
    for r in infra + memory:
        print(f"    rerun candidate: {r['task_id']}")


def usage(records: dict[str, dict]) -> str:
    minutes = sorted(r["seconds"] / 60 for r in records.values())
    student_out = [sum(t["completion_tokens"] for t in r["agent_log"]["turns"]) for r in records.values()]
    runner = [r["program_log"]["calls"] for r in records.values() if r.get("program_log")]
    text = (f"median {minutes[len(minutes) // 2]:.1f} min/task, student output {sum(student_out) / len(student_out):,.0f} tokens/task")
    if runner:
        calls = sum(len(c) for c in runner) / len(runner)
        tokens = sum(x["prompt_tokens"] + x["completion_tokens"] for c in runner for x in c) / len(runner)
        text += f", runner {calls:.0f} primitive calls and {tokens:,.0f} tokens/task"
    return text


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--student", default="qwen3-8b")
    args = parser.parse_args()

    program, vanilla = load("program", args.student), load("vanilla", args.student)
    shared = sorted(set(program) & set(vanilla))
    pairs = [(int(program[t]["reward"] == 1.0), int(vanilla[t]["reward"] == 1.0)) for t in shared]
    program_only = sum(p and not v for p, v in pairs)
    vanilla_only = sum(v and not p for p, v in pairs)
    p_rate = 100 * sum(p for p, _ in pairs) / len(pairs)
    v_rate = 100 * sum(v for _, v in pairs) / len(pairs)
    gain = p_rate - v_rate
    p_value = mcnemar_exact(program_only, vanilla_only)
    lower, upper = bootstrap_gain(pairs)

    print(f"Student {args.student}, {len(shared)} test tasks run by both arms")
    print(f"  vanilla pass@1  {v_rate:5.1f}%")
    print(f"  program pass@1  {p_rate:5.1f}%")
    print(f"  gain            {gain:+5.1f} points, 95% bootstrap CI [{lower:+.1f}, {upper:+.1f}]")
    print(f"  discordant      program only {program_only}, vanilla only {vanilla_only}; exact McNemar p = {p_value:.4f}")
    print(f"  C1 (gain >= {C1_MIN_GAIN:g} and p < {C1_MAX_P}): {verdict(gain, p_value, upper)}")
    print("Usage")
    print(f"  vanilla: {usage(vanilla)}")
    print(f"  program: {usage(program)}")
    print("Audit")
    audit("vanilla", vanilla)
    audit("program", program)
    by_kind: dict[str, list[tuple[int, int]]] = {}
    for task_id, pair in zip(shared, pairs):
        by_kind.setdefault(task_id.split("]")[0] + "]", []).append(pair)
    print("By issue")
    for kind, kind_pairs in sorted(by_kind.items()):
        n = len(kind_pairs)
        print(f"  {kind:22} n={n:2}  vanilla {sum(v for _, v in kind_pairs)}/{n}  program {sum(p for p, _ in kind_pairs)}/{n}")


if __name__ == "__main__":
    main()
