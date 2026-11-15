"""Live check of the guards on chosen dev tasks: does each guard fire, and are its blocks right?

  vanilla + tool-name guard on tasks where plain Qwen3-8B calls phone-only functions;
  program + full guard on the suspension tasks, where the frozen program resumed lines before payment.
Records go to runs/guard-check/<arm>-<guard>/, one file per task, as plvr.run writes them.
"""

import sys
from pathlib import Path

from plvr import tasks
from plvr.llm import Server
from plvr.models import MODELS, PRIMITIVE_RUNNER, USER_SIMULATOR
from plvr.run import check_servers, run_task, task_file

PLAN = [
    ("vanilla", "tools", [
        "[mobile_data_issue]airplane_mode_on|bad_vpn|user_abroad_roaming_disabled_on[PERSONA:Hard]",
        "[service_issue]unseat_sim_card[PERSONA:Easy]",
        "[mms_issue]airplane_mode_on|bad_network_preference|break_app_both_permissions[PERSONA:Easy]",
    ]),
    ("program", "full", [
        "[service_issue]overdue_bill_suspension[PERSONA:Easy]",
        "[service_issue]break_apn_settings|overdue_bill_suspension[PERSONA:None]",
        "[service_issue]airplane_mode_on|contract_end_suspension|lock_sim_card_pin[PERSONA:Hard]",
        "[service_issue]lock_sim_card_pin|overdue_bill_suspension|unseat_sim_card[PERSONA:None]",
    ]),
]


def main() -> None:
    by_id = {t.id: t for t in tasks.load("dev")}
    student = runner = Server(MODELS[PRIMITIVE_RUNNER])
    check_servers(student, Server(MODELS[USER_SIMULATOR]))
    student.tokenizer
    for arm, guard, ids in PLAN:
        out = Path("runs/guard-check") / f"{arm}-{guard}"
        out.mkdir(parents=True, exist_ok=True)
        for task_id in ids:
            if task_file(out, task_id).exists():
                continue
            reward = run_task(by_id[task_id], arm, student, runner, USER_SIMULATOR, out, guard)
            print(f"{arm}-{guard} reward={reward} {task_id}", flush=True)


if __name__ == "__main__":
    sys.exit(main())
