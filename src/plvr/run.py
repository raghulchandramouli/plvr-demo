"""Run one arm with one student over a τ² Telecom task set. Writes one JSON file per task and skips tasks already done."""

import argparse
import hashlib
import json
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from tau2.data_model.tasks import Task
from tau2.evaluator.evaluator import EvaluationType
from tau2.orchestrator.orchestrator import Orchestrator
from tau2.runner import build_environment, build_user, run_simulation

from plvr import tasks
from plvr.agent import StudentAgent
from plvr.llm import Server
from plvr.models import MODELS, PRIMITIVE_RUNNER, STUDENTS, USER_SIMULATOR

MAX_STEPS = 200
MAX_ERRORS = 10
SEED = 300
MAX_ATTEMPTS = 4  # tau2's run_with_retry: the first attempt plus 3 retries, then an infrastructure error scored 0


def task_file(out_dir: Path, task_id: str) -> Path:
    return out_dir / f"{hashlib.sha1(task_id.encode()).hexdigest()[:12]}.json"


def run_task(task: Task, arm: str, student: Server, runner: Server, user_key: str, out_dir: Path) -> float:
    errors: list[str] = []
    for _ in range(MAX_ATTEMPTS):
        record = _attempt(task, arm, student, runner, user_key)
        if "error" not in record:
            break
        errors.append(record.pop("error"))
    else:
        record |= {"reward": 0.0, "termination_reason": "infrastructure_error"}
    record["failed_attempts"] = errors
    task_file(out_dir, task.id).write_text(json.dumps(record, indent=1, default=str))
    return record["reward"]


def _attempt(task: Task, arm: str, student: Server, runner: Server, user_key: str) -> dict:
    env = build_environment(tasks.DOMAIN)
    thinker = None
    if arm == "program":
        from plvr.telecom.program import TelecomProgram

        thinker = TelecomProgram(policy=env.get_policy(), tools=env.get_tools(), runner=runner)
    agent = StudentAgent(tools=env.get_tools(), domain_policy=env.get_policy(), student=student, thinker=thinker)
    user_model = MODELS[user_key]
    user = build_user(
        "user_simulator",
        env,
        task,
        llm=f"openai/{user_model.path()}",
        llm_args={
            "api_base": user_model.api_base,
            "api_key": "local",
            "temperature": 0.0,
            "extra_body": {"chat_template_kwargs": {"enable_thinking": False}},
        },
    )
    orchestrator = Orchestrator(
        domain=tasks.DOMAIN, agent=agent, user=user, environment=env, task=task,
        max_steps=MAX_STEPS, max_errors=MAX_ERRORS, seed=SEED,
    )
    started = time.time()
    record = {"task_id": task.id, "arm": arm, "student": student.model.key, "user": user_key}
    try:
        result = run_simulation(orchestrator, evaluation_type=EvaluationType.ALL)
        record |= {
            "reward": result.reward_info.reward if result.reward_info else None,
            "termination_reason": str(result.termination_reason),
            "simulation": json.loads(result.model_dump_json()),
        }
    except Exception:
        record |= {"reward": None, "error": traceback.format_exc()[-3000:]}
    return record | {
        "seconds": time.time() - started,
        "agent_log": agent.log.as_dict(),
        "program_log": thinker.log_dict() if thinker else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", choices=["vanilla", "program"], required=True)
    parser.add_argument("--student", choices=STUDENTS, required=True)
    parser.add_argument("--task-set", choices=tasks.TASK_SETS, required=True)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--user", default=USER_SIMULATOR, choices=sorted(MODELS))
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    student, runner = Server(MODELS[args.student]), Server(MODELS[PRIMITIVE_RUNNER])
    todo = [t for t in tasks.load(args.task_set, args.limit) if not task_file(args.out, t.id).exists()]
    print(f"{len(todo)} tasks to run into {args.out}", flush=True)
    with ThreadPoolExecutor(args.concurrency) as pool:
        futures = {pool.submit(run_task, t, args.arm, student, runner, args.user, args.out): t for t in todo}
        for done, future in enumerate(as_completed(futures), 1):
            print(f"[{done}/{len(todo)}] reward={future.result()} {futures[future].id}", flush=True)


if __name__ == "__main__":
    main()
