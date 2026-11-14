# Decision log

One row per decision, newest last. Each row says what was decided, why, and what evidence it rests on.

| Date | Decision | Why | Evidence |
| --- | --- | --- | --- |
| 2026-11-14 | Reproduce only the τ² Telecom headline first. | The user chose it over the full pipeline and over the full paper. It is the paper's largest effect (+48 points for the 8B model), so it is the cheapest claim to falsify. | User answer in session. |
| 2026-11-14 | Qwen3-8B runs the primitives. Qwen3-4B, 8B and 14B are the students. Qwen3-30B-A3B-Instruct-2507 is the user simulator. | The user chose local Qwen and no API spend. | User answer in session. No API keys on the machine. |
| 2026-11-14 | Evaluate on the τ² `base` split (114 tasks). Develop on `full` minus `base`. | `base` has the paper's 114 tasks. `train` (74) plus `test` (40) equals `base`, so neither half alone is the paper's set. `full` minus `base` gives 2,171 disjoint tasks of the same generator for development. | `split_tasks.json` at tau2 commit `c06b07d`, checked by script in session. |
| 2026-11-14 | Use the published final program (Appendix B.1). No search. | The paper publishes the final program. Running the search needs teacher traces and about 1.9B primitive tokens, roughly a week on this Mac. | Paper §3.4, Appendix A. |
| 2026-11-14 | The program reads the troubleshooting checklist out of the policy through primitives. No rule is hard-coded. | Hard-coding the τ² manual into code would move the primitives' work into the author's head and inflate the result. | Pre-registration rule 1. |
| 2026-11-14 | Put the program's chain of thought inside the student's `<think>` block. Use Qwen3 thinking mode for vanilla. | The paper places the program's text "where its own reasoning would go" (§4.3). Qwen3's thinking block is that place. | Paper §1, §4.3. |
| 2026-11-14 | Skip a design arena for the program's structure. | Appendix B.1 fixes the program's structure line by line, so an arena would only explore shapes the reproduction must not take. | Paper Appendix B.1. |
