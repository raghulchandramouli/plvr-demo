# Pre-registration: the τ² Telecom result with local Qwen3

Written on 2026-11-14, before any program code exists and before any run on the evaluation tasks.

## The claim under test

The paper is "Compiling Expert Procedures into Verified Programs: When a Small Model Is Enough" (CoreThink AI Team).
On τ²-bench Telecom it reports these pass@1 scores (Table 2):

| Student | Vanilla | Final program |
| --- | --- | --- |
| Granite-4.2-8B | 43.0 | 91.2 |
| Nemotron 3.5 Lightning | 47.4 | 94.4 |
| Nemotron 3 Nano | 36.8 | 93.3 |
| Nemotron 3 Super | 51.8 | 93.3 |

The spread between the three Nemotron students falls from 15.0 points to 1.1 points under the program.

## What this run changes

| Paper | This run | Why |
| --- | --- | --- |
| Granite-4.2-8B runs the primitives | Qwen3-8B (MLX 4-bit), thinking off, temperature 0 | The user chose Qwen. Same size class. |
| Students are three Nemotron models | Qwen3-4B, Qwen3-8B, Qwen3-14B (MLX 4-bit) | Three sizes of one family test whether the spread shrinks. |
| GPT-4.1 simulates the user | Qwen3-30B-A3B-Instruct-2507 (MLX 4-bit), temperature 0 | No API access. Absolute scores will differ from the paper. |
| Claude 5 Opus writes the primitives from clusters of Kimi-K3 traces | Claude Opus 5.5 writes the primitives from the names, groups and signatures in Appendix D.1 | No teacher traces exist in this phase. The primitives are not trace-grounded. |
| A* search builds the program | The final telecom program of Appendix B.1 is used as published | Tests the published program, not the search. |
| Vanilla is chain-of-thought prompting | Qwen3 thinking mode | Qwen3's native chain of thought. |

These are the largest threats to validity.
The user simulator is weaker than GPT-4.1, so some failures will be the simulator's.
Both arms face the same simulator, so the vanilla-to-program difference stays comparable.
The primitives are written by the same model family the paper used for authoring, but without the traces.

## Rules that keep the test honest

1. The program code holds no telecom troubleshooting rule, fix, or tool order of its own.
   It reads the checklist out of the policy at run time through primitives.
   The lines of Appendix B.1 are the only domain behaviour written into the program, because the paper publishes them.
2. Primitive prompts are developed only on dev tasks: τ² Telecom tasks in the `full` split that are not in the `base` split.
3. The program and every primitive prompt are frozen in a tagged commit before any evaluation run.
4. No trajectory from an evaluation task is read before that tag exists.
5. Every arm uses the same user simulator, the same sampling settings, and the τ² defaults of 200 steps and 10 errors.

## Evaluation set

The τ² Telecom `base` split at tau2-bench commit `c06b07dd7c4d4ec61ba6ff9bcdf0b15dd7ef36d4`: 114 tasks, 49 MMS, 36 mobile data, 29 service.
This matches the paper's 114 Telecom tasks.
Reward is τ²'s environment-assertion check.

## Arms

For each student in {Qwen3-4B, Qwen3-8B, Qwen3-14B}:

- **Vanilla.** τ²'s standard agent prompt. The student thinks in Qwen3 thinking mode, then acts.
- **Program.** The final telecom program runs its primitives on Qwen3-8B and writes a chain of thought.
  That text goes inside the student's `<think>` block, where its own reasoning would go.
  The student then writes the next message or tool call.

Both arms sample the student at temperature 0.6, top-p 0.95, top-k 20, Qwen's recommended thinking-mode settings.
Each cell runs once. Program cells get two more runs if measured throughput allows, as in the paper.
That decision is recorded before the runs start.

## Claims and verdict rules

Each verdict is VERIFIED, NOT VERIFIED, or INCONCLUSIVE.

- **C1. The program raises every student by at least 20 points.**
  VERIFIED if every student gains at least 20 points and a paired exact McNemar test over the 114 tasks gives p < 0.01.
  NOT VERIFIED if any student's 95% confidence interval on the paired gain lies entirely below 20.
  INCONCLUSIVE otherwise.
- **C2. The spread between students shrinks under the program.**
  VERIFIED if the max-minus-min pass@1 across the three students is smaller under the program than under vanilla by at least 5 points.
  NOT VERIFIED if it is not smaller.
  INCONCLUSIVE if it is smaller by less than 5 points.
- **C3. Under the program an 8B student is enough.**
  VERIFIED if Qwen3-8B under the program is within 5 points of Qwen3-14B under the program.
  NOT VERIFIED if it trails by more than 10 points.
  INCONCLUSIVE between 5 and 10.

A single run of 114 tasks has a standard error near 4.5 points at 50% accuracy.
C2 and C3 rest on differences close to that resolution, so their verdicts carry that caveat.
