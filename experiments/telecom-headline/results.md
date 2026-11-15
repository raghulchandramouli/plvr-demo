# Results: the PLVR telecom program on Qwen3-8B

On τ²-bench Telecom's 40-task test split, the paper's final telecom program raised Qwen3-8B from 17.5% to 50.0% pass@1, a gain of 32.5 points.
The pre-registered claim C1 is VERIFIED for this student.
The gain is about two thirds of the paper's 48.2 points for Granite-4.2-8B, and both absolute scores are lower than the paper's.

## Headline

| | Qwen3-8B, this run | Granite-4.2-8B, paper Table 2 |
| --- | --- | --- |
| Vanilla (thinking mode) | 17.5% (7/40) | 43.0% |
| Final program | 50.0% (20/40) | 91.2% |
| Gain | +32.5 points | +48.2 points |

The paper's figures are on the 114-task `base` split with GPT-4.1 as the user. This run uses τ²'s 40-task `test` split, a subset of `base`, with Qwen3-30B-A3B as the user.
Absolute scores are not comparable across the two setups. The vanilla-to-program gain is the quantity under test.

## Pre-registered claims

| Claim | Verdict | Evidence |
| --- | --- | --- |
| C1. The program raises every student by at least 20 points. | VERIFIED for Qwen3-8B | Gain +32.5 points. Exact McNemar test on the 40 paired tasks: 14 solved only by the program, 1 only by vanilla, p = 0.001. |
| C2. The spread between students shrinks under the program. | NOT TESTED | Needs Qwen3-4B and Qwen3-14B runs, which were optional. |
| C3. Under the program an 8B student is enough. | NOT TESTED | Needs the Qwen3-14B run. |

Two caveats on C1:

- **The interval is wide.** The 95% bootstrap interval on the gain is [+17.5, +50.0]. Its lower end sits below the 20-point threshold, so the data do not rule out a gain under 20 points. The pre-registered rule uses the point estimate and the McNemar test, and both pass.
- **One student only.** C1 says "every student". Only the 8B student was run, so the verdict covers that student alone.

## By issue

| Issue | Tasks | Vanilla | Program |
| --- | --- | --- | --- |
| Service (no signal) | 15 | 6 | 11 |
| Mobile data | 9 | 1 | 4 |
| MMS | 16 | 0 | 5 |

The program helps most where vanilla does worst.
Vanilla solved no MMS task, the issue with the longest chain of prerequisites (service, then data, then MMS settings).

## What the program changes

- **It keeps troubleshooting.** Vanilla's median task is 11 agent turns, the program's 31. In 32 of vanilla's 33 failures the simulated user ended the call with the problem unfixed. The program walks the policy checklist instead.
- **The student writes less.** The student's own output falls from 8,785 to 1,584 tokens per task, because the program writes its reasoning.
- **It does not stop the student calling phone functions.** The student tried to call a user-side function itself in 29 vanilla tasks and 24 program tasks. The program states which tools the student may call but does not filter its actions. The paper reserves filtering for the retail block.

## Why the program still fails half the time

Of the program's 20 failures:

| Ending | Count |
| --- | --- |
| Transfer after both passes over the checklist left the problem in place | 15 |
| ...of which a policy procedure also hit its 8-turn cap | 7 |
| τ²'s 200-step limit, on tasks with 9 faults | 2 |
| Other endings, including one where the user wrongly reported the problem fixed | 3 |

The most common single cause is procedure completion.
After `enable_roaming` succeeds, the "Is the request achieved" primitive often does not confirm the roaming procedure is done, so the procedure runs to its turn cap and the program never reaches Data Refueling.
Data refueling was required and missed in 9 of the 20 failures. The program called `refuel_data` in only 2 tasks.

The other cause is Qwen3-8B's judgement of free-text readings.
At the freeze the condition probe scored 16 of 19 (`scripts/probe_conditions.py`). The guards added during development make most of its errors cost turns rather than tasks, but a missed fix still ends in a transfer.

## Cost per task

| | Vanilla | Program |
| --- | --- | --- |
| Median wall time, one task at a time | 3.7 min | 8.6 min |
| Student output tokens | 8,785 | 1,584 |
| Runner (Qwen3-8B) primitive calls | none | 61 |
| Runner tokens, prompt plus output | none | 66,475 |

The paper reports 100 to 117 primitive calls per task on Granite-4.2-8B. This program makes fewer calls because it compiles the policy's checklist once and caches it.

## Threats to validity

- **The user simulator is weaker than the paper's.** Qwen3-30B-A3B replaced GPT-4.1. In development it once reported a test result it had not run. Both arms meet the same simulator, so the gain stays comparable, but absolute scores are lower.
- **The primitives were not written from teacher traces.** Claude Opus 5.5 wrote them from the primitive names and the published program, then tuned them in three rounds on development tasks that share τ²'s policy. The paper's primitives were authored from clusters of Kimi-K3 traces.
- **No search was run.** The program is the paper's published final telecom block, interpreted in code. The A* search over the rule library was not reproduced.
- **The sample is small.** 40 tasks with one run per arm. A single cell's standard error is about 8 points.
- **Run conditions changed mid-run.** Server memory limits and concurrency changed during the program arm, after GPU out-of-memory stops. The program and its prompts did not change after the `program-frozen-v1` tag; only server code did. Generation runs at temperature 0.6, so batching differences only reorder sampling noise. The audit found no task scored 0 for an infrastructure reason. One vanilla task was retried after a tokenizer import race, and the retry ran normally.

## What a second version would change

1. **Procedure completion.** End a procedure on a successful write that the procedure names, rather than on a judgement of whether the request is achieved.
2. **Procedure execution.** Run the steps of Data Refueling and Overdue Bill Payment in the program, one at a time. Today the program hands the student the policy text, and in one task the student resumed a suspended line without collecting the overdue bill.
3. **The student's tool calls.** Decide whether the telecom block should reject calls to user-side functions. That would be a departure from the paper's telecom form.

Each of these would be developed on dev tasks and evaluated as a separate, newly frozen program.

## How to reproduce

```bash
scripts/setup.sh
```

```bash
scripts/run_test_arms.sh
```

```bash
.venv/bin/python -m plvr.score --student qwen3-8b
```

The program is frozen at the git tag `program-frozen-v1`. Per-task records are written to `runs/test/`, which is not committed.
