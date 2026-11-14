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
| 2026-11-14 | Evaluate on τ²'s 40-task `test` split instead of all 114 `base` tasks. Run Qwen3-8B first. | The user asked for a sample to cut run time. τ²'s own split avoids hand-picking. 40 paired tasks resolve the paper's 40+ point gain, but not its few-point spread claims. | User message in session. One vanilla task measured at 14.5 min. |
| 2026-11-14 | Retry a crashed simulation up to 3 times, then score it 0. | τ²'s `run_with_retry` does the same, so both arms meet failures the way τ² scores them. | First smoke run crashed when the stand-in user sent an empty message. tau2 `runner/progress.py`. |
| 2026-11-14 | Patch mlx_lm's server so text completions return the tool calls and reasoning it parses. | Stock 0.32.2 drops both from `/v1/completions`. The program arm needs raw completions to place its text in the think block. | Probe returned `''` with no tool call before the patch, and the call after it. |
| 2026-11-14 | Keep rule extraction at section level, drop only the bad rules, clear a check that equals its fix, keep only quoted fix arguments. | One bad rule was discarding whole sections (Data Roaming, 2G). Rejecting per rule keeps the verifier's guarantee without losing the rest. | `python -m plvr.telecom.policy` before and after: roaming and 2G rules now present, APN reboot follow-up restored. |
