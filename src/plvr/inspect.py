"""Print a run directory: score per task, and for one task the conversation with the program's step at each agent turn."""

import argparse
import json
from pathlib import Path


def summary(run_dir: Path) -> None:
    records = [json.loads(p.read_text()) for p in sorted(run_dir.glob("*.json"))]
    solved = sum(1 for r in records if r["reward"] == 1.0)
    print(f"{run_dir}: {solved}/{len(records)} solved")
    for r in records:
        turns = r["agent_log"]["turns"] if r.get("agent_log") else []
        calls = len((r.get("program_log") or {}).get("calls", []))
        print(
            f"  reward={r['reward']} {r.get('termination_reason', '')[:28]:28} {r['seconds'] / 60:5.1f}min "
            f"student_turns={len(turns):3} primitive_calls={calls:4} retries={len(r.get('failed_attempts', []))} {r['task_id'][:90]}"
        )


def conversation(record: dict) -> None:
    steps = iter((record.get("program_log") or {}).get("trace", []))
    user_spoke = False
    for m in record.get("simulation", {}).get("messages", []):
        user_spoke = user_spoke or m["role"] == "user"
        if m["role"] == "assistant" and user_spoke and record.get("program_log"):
            step = next(steps, None)
            if step:
                s = step["step"]
                print(f"    >> program: [{s['action']}] {s['instruction'][:200]} | why: {s['why'][:160]}")
        text = (m.get("content") or "").replace("\n", " ")[:300]
        calls = " ".join(f"{c['name']}({json.dumps(c['arguments'])})" for c in m.get("tool_calls") or [])
        print(f"  {m['role'][:4]:4} {text} {calls}")
    info = record.get("simulation", {}).get("reward_info") or {}
    print("  env assertions:", [(a["env_assertion"]["func_name"], a["met"]) for a in info.get("env_assertions") or []])
    if record.get("program_log"):
        calls = record["program_log"]["calls"]
        rejected = [c for c in calls if not c["accepted"]]
        print(f"  primitive calls: {len(calls)}, rejected: {len(rejected)}")
        for c in rejected[:15]:
            print(f"    rejected {c['primitive']}: {c['rejection']}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--task", help="substring of a task id to print in full")
    args = parser.parse_args()
    summary(args.run_dir)
    if args.task:
        for p in sorted(args.run_dir.glob("*.json")):
            record = json.loads(p.read_text())
            if args.task in record["task_id"]:
                print(f"\n== {record['task_id']}")
                conversation(record)


if __name__ == "__main__":
    main()
