# Status: paused on 2026-11-15

The user paused all work. Nothing is running.

## Where things stand

- **Plain Qwen3-8B on the 40 test tasks.** 3 of 40 finished (1 solved), saved in `runs/test/vanilla-qwen3-8b/`. The 2 tasks in flight were lost and rerun on resume. Their conversations have not been read, as the pre-registration requires.
- **Program.** Not frozen. On dev tasks with the real 30B customer it solved 1 of 2 before the pause. The failure was Qwen3-8B judging "The SIM card is active" as meeting "shows Locked", which triggered a wrong transfer.
- **Last change, untested.** The condition primitives now copy the state the rule looks for and the state the reading shows, then compare them (`Compare` in `src/plvr/telecom/primitives.py`). It has not run yet. The probe that was about to test it failed on a bug in the probe script itself (a section-name lookup), not in the primitive.
- **Downloads.** Qwen3-8B and Qwen3-30B are complete. Qwen3-4B and Qwen3-14B are not.

## How to resume

Start both servers. Each needs its own terminal or background job.

```bash
.venv/bin/python -m plvr.models serve qwen3-8b --concurrency 3
```

```bash
.venv/bin/python -m plvr.models serve qwen3-30b-a3b-instruct --concurrency 3
```

Continue the plain run. It skips the tasks already finished.

```bash
.venv/bin/python -m plvr.run --arm vanilla --student qwen3-8b --task-set test --concurrency 2 --out runs/test/vanilla-qwen3-8b
```

Then test the new condition primitive on the cases that failed, rerun the program on dev tasks, and freeze it with a git tag before its test run.

## Known limits on this Mac

Two servers plus 3 conversations fit in memory only with the per-server budgets in `src/plvr/serve.py`. Keep the total at 3 conversations or fewer.
