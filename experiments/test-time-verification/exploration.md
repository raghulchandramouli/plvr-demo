# Exploration: test-time verification for the telecom agent

The question is whether checking work while the agent runs can raise τ² Telecom scores beyond the frozen program's 50% and plain Qwen3-8B's 17.5%.
Here, checking work means verifying an action before it goes out, or verifying a judgement before the program acts on it.
This note covers what methods exist, where our own runs show room for them, one measured experiment, and a plan to test the rest fairly.

## What exists

Recent work on verifying tool-using agents at test time falls into four families.
Details come from abstracts and search summaries, not full reads.

| Family | How it checks | Examples |
| --- | --- | --- |
| Policy guards | Policy text becomes checks that run before each action. A violating action is blocked, and the agent revises it. | ToolGuard (EMNLP 2025, τ-bench airline); interwhen (Microsoft Research 2026, verifiers run alongside the reasoning trace); deterministic gates on τ²-bench (arXiv 2607.07405); PolicyGuard (a sub-agent verifier) |
| Simulate before acting | A simulated environment previews a tool call's effect before the real call. | GATS / Gecko (arXiv 2602.19218), with reported gains on τ²-bench |
| Best-of-N with a verifier | Sample several candidate steps and let a reward model or judge pick one. | ToolPRM, PRInTS, AgentProcessBench. The paper's sample-and-judge baseline is this kind. |
| Check after acting | Postcondition checks, verify-before-retry, idempotency keys around tool calls. | Verified tool calls under non-atomic failures (arXiv 2608.02645) |

The deterministic-gates paper adds a caution.
Blocking a bad action does not mean the agent then recovers, and on τ² retail its gates made results worse.

The PLVR paper itself uses two of these.
Each of its primitives carries a verifier that rejects malformed or ungrounded outputs, and its retail block is a policy guard around the student's actions.
Our telecom program keeps the first and, following the paper's telecom block, leaves out the second.

## Where verification would act in our runs

All counts come from the 40-task test runs of Qwen3-8B, read in aggregate.

| Signal | Vanilla | Program |
| --- | --- | --- |
| Student calls a function that only the user's phone has | 121 calls, 24% of agent turns, in 29 tasks (26 of them failed) | 64 calls, 5% of agent turns, in 24 tasks (12 of them failed) |
| State change before its policy precondition: `resume_line` before any payment | 0 tasks | 4 tasks |
| Failures that end in a transfer after the checklist ran out | not applicable | 15 of 20 |

The program's own primitive verifiers rejected 383 of 2,442 outputs (16%).
On rejection the program uses the primitive's fallback, which is almost always "cannot tell".
Four primitives account for most rejections, and three of them sit on the paths that failed most:

| Primitive | Rejected | Where it bites |
| --- | --- | --- |
| Is the condition met | 88 of 96 (92%) | Record-based conditions such as suspensions and data limits |
| Price terms | 19 of 22 (86%) | The data refuel price, part of the missed refuels |
| Is the request achieved | 16 of 39 (41%) | Ending a policy procedure, the main cause of the missed refuels |
| Does the condition hold now | 118 of 407 (29%) | Every checklist rule |

So the verifiers mostly do catch bad outputs.
What they lack is a second chance: a rejected output is replaced by "cannot tell" instead of being tried again.

## Experiment 1: sample the judgement more than once

The judgement behind each checklist rule is two primitives.
"Does the condition hold now" and "Is the setting already as wanted" are combined by the frozen program's decision rule.
`scripts/probe_self_consistency.py` scores three ways of calling them on the 19 labelled probe cases:

- **greedy:** one call at temperature 0. This is what the frozen program does.
- **resample:** up to 3 samples at temperature 0.6. It keeps the first one its verifier accepts.
- **vote:** 5 samples at temperature 0.6. It takes the majority answer among those the verifier accepts.

| Strategy | Correct | Missed fixes | Model calls |
| --- | --- | --- | --- |
| greedy | 16 of 19 | 1 | 36 |
| resample | 16 of 19 | 1 | 46 |
| vote | 16 of 19 | 1 | 180 |

Sampling more did not change a single decision.
Qwen3-8B's three errors on these cases are systematic: it makes the same wrong judgement on every sample, so neither another try nor a majority recovers the right one.
Voting cost five times the calls for nothing.

This is a negative result for verification by re-asking the same model.
A verifier helps only when it brings something the first call lacked.
That can be a deterministic check, different information, or a stronger judge.
The probe cannot say whether resampling recovers the outputs that verifiers *reject* on real tasks, because almost all of its errors are accepted outputs that are wrong.

## Options for our setup

| Option | What it checks | Targets | Extra cost per task (estimate) |
| --- | --- | --- | --- |
| A. Tool-name guard on the student | Before an action goes out, its function must be one of the agent's tools; otherwise the student is told why and asked again, at most twice. | Phone-only calls: 24% of vanilla turns, 5% of program turns | One student call per rejected action |
| B. Resample on rejection | When a primitive's verifier rejects an output, sample again (up to 3) before using the fallback. | The 16% of primitive outputs now replaced by "cannot tell", including the refuel and procedure-ending primitives | Up to 2 runner calls per rejection, about 770 per task at today's rejection rate, many of them short |
| C. Vote on costly decisions | Sample toggles and escalations 5 times and take the majority. | Wrong toggles and wrong transfers. **Dropped: Experiment 1 shows no gain at 5x the cost.** | About 8 extra runner calls per checklist rule |
| D. Precondition guard on state changes | Before a state-changing call, check the precondition the policy states, read from the records, such as a paid bill before `resume_line`. Like ToolGuard, the checks come from the policy text through primitives. | The 4 tasks that resumed a line before payment, and refuels without an amount or confirmation | One primitive call per state change |

A and D are guards around the student, the paper's retail form.
Adding them to the telecom block departs from the paper's telecom design, so results with them would be reported as an extension, not as the paper's method.
B and C only make the existing primitives more reliable, which stays inside the paper's telecom form.

After Experiment 1, the order of promise is:

1. **A and D, the deterministic guards.** They add information the student and the primitives lack: the agent's real tool list, and the policy's preconditions read from the records. They target failures counted in our logs. They cannot repeat a model's mistake, because no model decides them.
2. **B, resampling on rejection.** It is cheap to add, and it recovers outputs only when the rejection was bad luck rather than a systematic format error. Whether it does is unknown until it runs on tasks.
3. **A stronger or differently prompted judge for checklist conditions.** This is the only option aimed at the systematic misjudgements. A larger model would also run on every rule, so it moves the study away from "a small model is enough".

## How to test it fairly

- **Fresh evaluation tasks.** The 40 test-split trajectories were read closely to analyse failures, so they cannot be a clean test of methods designed afterwards. τ²'s `train` split has 74 tasks; they are part of the paper's 114, and nothing has run on them. They become the evaluation set. Development stays on the `full`-minus-`base` dev tasks.
- **Pre-registration.** Write down the arms, the thresholds, and the verdict rules before any run, as for the first study.
- **Arms.** The frozen program; the program with B (and C if Experiment 1 supports it); the program with B plus A and D; and vanilla with A alone, a cheap check of whether a guard by itself helps plain Qwen3-8B.
- **Cost.** At one task at a time, the program takes about 8.6 minutes per task, about 10.5 hours for 74 tasks per arm. Running all four arms is about 1.5 days of machine time, so the first pass could use a fixed half of the 74 tasks.

## Sources

- [interwhen, arXiv 2602.11202](https://arxiv.org/pdf/2602.11202)
- [GATS / Gecko, arXiv 2602.19218](https://arxiv.org/pdf/2602.19218)
- [Self-verification sampling with dynamics modelling, arXiv 2506.02918](https://arxiv.org/pdf/2506.02918)
- [Verified tool calls under non-atomic failures, arXiv 2608.02645](https://huggingface.co/papers/2608.02645.md)
- [Agentic verifier, arXiv 2604.16004](https://arxiv.org/html/2604.16004)
- [ToolGuard: enforcing company policy adherence in agentic workflows, arXiv 2507.16459](https://arxiv.org/abs/2507.16459v2)
- [Deterministic gates on τ²-bench, arXiv 2607.07405](https://huggingface.co/papers/2607.07405.md)
- [Policy violations in τ²-bench airline, arXiv 2603.29665](https://arxiv.org/html/2603.29665v1)
- [ToolPRMBench](https://www.iclr.cc/virtual/2026/10012620)
- [ToolPRM, arXiv 2510.14703](https://www.arxiv.org/pdf/2510.14703)
- [AgentProcessBench, arXiv 2603.14465](https://arxiv.org/pdf/2603.14465)
