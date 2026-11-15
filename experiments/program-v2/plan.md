# Program v2: plan

Version 1 of the telecom program is frozen at `program-frozen-v1` and keeps its reported results.
Version 2 targets the three largest causes of version 1's failures on the 40-task test split, found in `experiments/telecom-headline/results.md` and `experiments/test-time-verification/exploration.md`.

| Item | Change | Evidence it targets |
| --- | --- | --- |
| 1 | A policy procedure ends when the state change it exists for succeeds, and the program runs its steps one at a time. | 7 of 20 failures had a procedure hit its 8-turn cap; refuel was needed and missed in 9 failures; "Is the request achieved" rejected 41% of its outputs. |
| 2 | Deterministic guards around the student's actions: a tool-name guard, and a precondition guard on state changes read from the policy. | The student called phone-only functions in 5% of program turns and 24% of vanilla turns; 4 tasks resumed a line before any payment. |
| 3 | Fix-order search: try the policy's candidate fixes in order and score them with the real diagnostic, instead of trusting a misjudged reading. | 15 of 20 failures ended in a transfer after the checklist ran out; Qwen3-8B's condition judgements are systematically wrong on about 3 in 19 probe cases. |

Items 1 and 2 are built in parallel; item 3 builds on item 1.
Development and checks use only dev tasks (`full` minus `base`).
The evaluation set is τ²'s 74-task `train` split, which no run has touched.
A pre-registration is written and committed before any evaluation run.
Items 2 and 3 go beyond the paper's telecom block, so results that include them are reported as extensions of the paper's method.

## Evaluation arms

All arms run Qwen3-8B as the student on the same 74 `train` tasks, so every comparison is on the same tasks.

| Arm | Why | Time, one task at a time |
| --- | --- | --- |
| Plain Qwen3-8B | The baseline. It has never run on these tasks. | about 4.5 h |
| Program v1, frozen | v1 has been scored only on the 40 `test` tasks. v1 against v2 needs both on the same 74. | about 10.5 h |
| Program v2, no guards | What fixing the program itself is worth. Stays inside the paper's telecom form. | about 10.5 h |
| Program v2 with guards | What the guards add on top. They are the paper's retail form, so they are reported separately from v2 alone. | about 10.5 h |

The minimal set is the first three arms, about 25 hours; with the fourth it is about 36 hours.
Running a fixed half of the 74 tasks halves both, with wider confidence intervals.
Which set runs is decided with the user when the pre-registration is written.

## Dev pass bars

Each item must clear its bar on dev tasks before the long evaluation starts.
An item that fails its bar is fixed or left out, and the pre-registration says which.

| Item | Bar on dev tasks |
| --- | --- |
| 1, procedures | Every dev task whose fault needs a refuel gets `refuel_data` called. No procedure hits a turn cap. On the overdue-bill tasks, the payment is made before `resume_line`. |
| 2, guards | Every block the guards make is right when read against the records, so there are no false blocks. No phone-only call reaches the environment. |
| 3, fix-order search | Fewer runs end in a transfer after the checklist ran out than with item 1 alone, on the same dev tasks, and no dev task that passed with item 1 alone fails. |

## Model servers

Only the main session starts model servers.
The GPU holds at most two conversations at a time, so agents and other sessions write code and run unit tests only.
