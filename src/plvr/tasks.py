"""The τ² Telecom task sets: `test` (τ²'s 40-task test split, the sample we evaluate), `base` (the paper's 114), and a dev set disjoint from both."""

import json
import random
from collections import Counter
from functools import cache
from pathlib import Path

from tau2.data_model.tasks import Task
from tau2.runner import get_tasks

DOMAIN = "telecom"
DEV_SIZE = 40
DEV_MIN_PER_FAULT = 3
DEV_SEED = 0


def faults(task_id: str) -> list[str]:
    return task_id.split("]", 1)[1].split("[PERSONA")[0].split("|")


@cache
def splits() -> dict[str, list[str]]:
    from tau2.utils.utils import DATA_DIR

    return json.loads(Path(DATA_DIR, "tau2", "domains", DOMAIN, "split_tasks.json").read_text())


def dev_ids() -> list[str]:
    """Sample `full` minus `base` so every fault in `base` appears at least DEV_MIN_PER_FAULT times."""
    base = set(splits()["base"])
    pool = sorted(i for i in splits()["full"] if i not in base)
    rng = random.Random(DEV_SEED)
    rng.shuffle(pool)
    chosen: list[str] = []
    seen: Counter[str] = Counter()
    for fault in sorted({f for i in base for f in faults(i)}):
        for task_id in pool:
            if seen[fault] >= DEV_MIN_PER_FAULT:
                break
            if fault in faults(task_id) and task_id not in chosen:
                chosen.append(task_id)
                seen.update(faults(task_id))
    for task_id in pool:
        if len(chosen) >= DEV_SIZE:
            break
        if task_id not in chosen:
            chosen.append(task_id)
    return chosen


TASK_SETS = ("dev", "test", "base")


def load(task_set: str, limit: int | None = None) -> list[Task]:
    ids = dev_ids() if task_set == "dev" else splits()[task_set]
    return get_tasks(DOMAIN, task_ids=ids[:limit] if limit else ids)
