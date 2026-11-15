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
