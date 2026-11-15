# Status

## Program frozen on 2026-11-15

The program and every primitive prompt are frozen at the git tag `program-frozen-v1`.
No program or primitive code changes after this tag for the test runs.
`compiled-checklist.txt` is the checklist the frozen program compiled from the τ² policy.

## Development record before the freeze

All development used dev tasks (`full` minus `base`), never test tasks.

| Round | Customer model | Solved | What the failures taught |
| --- | --- | --- | --- |
| Stand-in runs 1–4 | Qwen3-8B (stand-in) | 0 of 6 | Harness and program bugs; the stand-in customer confused its role. |
| Round 1 | Qwen3-30B | 1 of 2 | A flipped yes/no judgement escalated an active SIM as "Locked". |
| Round 2 | Qwen3-30B | 1 of 2 | Healthy-state check misread "Mobile Data Disabled"; "2g_only" not matched to "2G"; transfer when the checklist ran out. |
| Round 3 | Qwen3-30B | 1 of 4 (5th stopped at the freeze) | A section heading taken as a state; the customer invented a test result; toggles fired on readings that did not mention the setting. |

The condition probe (`scripts/probe_conditions.py`) scores 16 of 19 at the freeze.
Its three errors either cost extra turns or are caught by the second pass.

## Test runs

Run in this order, 2 conversations at a time:

1. The program with Qwen3-8B as student, 40 test tasks: `runs/test/program-qwen3-8b/`.
2. Plain Qwen3-8B, the 36 test tasks not yet done: `runs/test/vanilla-qwen3-8b/` (4 finished before the freeze).

## How to resume after a stop

Start both servers, each in its own terminal or background job.

```bash
.venv/bin/python -m plvr.models serve qwen3-8b
```

```bash
.venv/bin/python -m plvr.models serve qwen3-30b-a3b-instruct
```

Rerun the same `plvr.run` command. It skips tasks already finished.

```bash
.venv/bin/python -m plvr.run --arm program --student qwen3-8b --task-set test --concurrency 2 --out runs/test/program-qwen3-8b
```

```bash
.venv/bin/python -m plvr.run --arm vanilla --student qwen3-8b --task-set test --concurrency 2 --out runs/test/vanilla-qwen3-8b
```

## Known limits on this Mac

Two servers plus 2 conversations fit in GPU memory with the per-server budgets in `src/plvr/serve.py`.
Three conversations ran out of Metal memory twice.
