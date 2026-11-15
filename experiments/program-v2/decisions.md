# Program v2: decisions

## 2026-11-15: guard preconditions on the real policy

The first compile of the precondition guard on the real telecom policy read each function against the whole policy.
It kept six preconditions and missed the one the guard exists for:

| Function | Kept | Problem |
| --- | --- | --- |
| `resume_line` | "Line status must be Suspended or Pending Activation" | From the tool description. Missed the overdue-bill and contract-end rules. |
| `enable_roaming` | forbidden if the contract end date is in the past | That rule is about lifting a suspension. |
| `disable_roaming` | "The current time is 2025-02-25 12:08:00 EST." | Not a condition. |
| `refuel_data`, `send_payment_request`, `suspend_line` | conditions copied from tool descriptions | The tools check these themselves. |

Two changes fixed it:

1. A new primitive, "Sections that govern a function", picks the policy sections that say when the agent may do what the function does.
   The precondition primitive then reads only those sections, plus any tech-support section that names the function.
2. A kept precondition must come from a sentence that permits or forbids: one with "allowed", "only", "must", "cannot", "never", "may not", "should not", "do not", "not permitted" or "required".
   Sentences that describe how things work, such as "If a line is roaming enabled, the user can use their phone's data connection abroad", are dropped and logged.

The second compile keeps three preconditions and drops five, all correctly:

| Function | Precondition |
| --- | --- |
| `resume_line` | Required: "You are allowed to lift the suspension after the user has paid all their overdue bills." |
| `resume_line` | Forbidden: "You are not allowed to lift the suspension if the line's contract end date is in the past, even if the user has paid all their overdue bills." |
| `send_payment_request` | Required: "A user can only have one bill in the AWAITING PAYMENT status at a time." |

The word list is a hand-written verifier, so it is part of the program, like the primitive verifiers.
It was written after reading the telecom policy, so whether it transfers to another domain's policy is untested.

## 2026-11-15: the precondition guard must not ask the model at run time

Live dev check, program v1 with `--guard full`, task `[service_issue]overdue_bill_suspension[PERSONA:Easy]`.
The student called `resume_line` while bill B1234321 was overdue, and the task failed on `assert_no_overdue_bill`.
The guard asked Qwen3-8B "Is the condition met" for both `resume_line` preconditions:

| Precondition | Raw reply | Verifier | Guard |
| --- | --- | --- | --- |
| Required: paid all overdue bills | answer "no", expression `'B1234321' < '2025-02-25'` | rejected: expression does not evaluate | "cannot tell", call sent |
| Forbidden: contract end date in the past | answer "no", expression `'2026-12-31' >= '2025-02-25'` | rejected: answer disagrees with expression | "cannot tell", call sent |

Both answers were right, and both were thrown away, because the expression the model attached was wrong.
The evidence it copied was the policy sentence itself, which the record included, so evidence alone could not settle it either.

Decision: the model maps each precondition to a record check once, at compile time.
The check names a read function, a field, a test and a value.
Code evaluates it at run time against the latest matching read.
When no read is there yet, the guard blocks and tells the student which read to make.
A separate agent builds this in guards.py, so it does not touch the files the procedure agent is changing.

The tool-name guard on vanilla, 3 dev tasks: 7 phone-only calls blocked, all 7 recovered on the first resample, none reached the environment.
