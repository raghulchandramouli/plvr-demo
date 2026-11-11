# Local baseline smoke result

Recorded on 2026-11-11. The recording gate is complete. The corrected baseline
passes 2 of 30 smoke tasks under the declared strict task contract. All 28
failures remain inspectable. This is a development smoke result, not a locked
evaluation or a claim of general model accuracy.

## Measured run

The primary run is [baseline-smoke-20261111-02](../runs/baseline-smoke-20261111-02/summary.json).
It ran locally on the Apple M5 Pro with Python 3.12.14, MLX 0.32.4, and MLX LM
0.32.2. One process loaded the unquantized Qwen2.5-3B-Instruct model once and
made 30 sequential calls. Every prompt included all 18 mock-tool schemas.

| Measurement | Recorded value |
| --- | ---: |
| Planned and attempted examples | 30 and 30 |
| Model loads and model calls | 1 and 30 |
| Completed mock-tool calls | 23 |
| Successes | 2 |
| Formatting failures | 5 |
| Contract failures | 18 |
| Semantic failures | 5 |
| Operational failures and timeouts | 0 and 0 |
| Whole-run wall time | 104.308828 seconds |
| Model loading | 4.025033 seconds |
| Generation loops including trace recording | 99.654372 seconds |
| Evaluation time | 0.009948 seconds |
| Submitted prompt tokens | 19,638 |
| Emitted generation tokens including EOS | 2,586 |
| Prompt tokens per example | 644 to 672 |
| Process peak RSS | 6,458,277,888 bytes |
| MLX allocator peak | 6,397,980,784 bytes |
| Logical bytes of run artifacts | 936,643 bytes |
| Free disk after the run | 795,688,214,528 bytes |

RSS and MLX allocations overlap on unified memory. Do not add their peaks.
The native swap report read `used = 1707.94M` both before and after this run. Resource numbers and
times describe this run only. No throughput or speedup comparison is claimed.
The final summary rewrite and lock release are outside the wall-time interval.

All 434 loaded parameter tensors were BF16. All 30 traces recorded BF16 KV-cache
arrays and emitted log-probability arrays. Other forward-pass intermediates were
not measured. The 120-second deadline is cooperative and can overrun a single
GPU operation. See [assumptions.md](../assumptions.md) for measurement definitions.

## Inspect successes and failures

- [smoke-015](../runs/baseline-smoke-20261111-02/traces/smoke-015.json) generates and executes a valid `search_alternatives` call.
- [smoke-023](../runs/baseline-smoke-20261111-02/traces/smoke-023.json) creates a return and looks it up through the returned identifier.
- [smoke-005](../runs/baseline-smoke-20261111-02/traces/smoke-005.json) reaches 512 generated tokens with invalid, incomplete JSON. It remains a formatting failure.
- [smoke-001](../runs/baseline-smoke-20261111-02/traces/smoke-001.json) omits the required `return` field and fails the program contract.
- [smoke-004](../runs/baseline-smoke-20261111-02/traces/smoke-004.json) executes extra customer lookups and fails the required-action check.

The scorer checks exact required results, state changes, tool-action
multiplicity, and requested dependency bindings. It can reject an otherwise
equivalent tool choice, such as `lookup_product` in place of `check_inventory`.
Interpret the success count under that declared synthetic contract.

## Verification and preserved development run

All 30 independently specified oracles pass before model loading. All 21 unit
tests pass. The offline replay command independently reproduced every saved
evaluation and reconciled all example, token, call, timing, and artifact totals.

The [first development run](../runs/baseline-smoke-20261111-01/summary.json) is
also retained. An independent review found that the scorer accepted a literal
guess and reversed dependency order when the final values matched. Two new
regressions failed before the fix and pass afterward. Required references are
now checked against actual source actions while allowing renamed call IDs and
reordered independent calls.

The model prompts, task corpus, and expected answers did not change. The
corrected run repeated all 30 calls after the scorer fix. Both runs have exactly
the same prompts, raw outputs, token IDs, and classifications. Across both
recorded model runs, the work comprised 2 loads, 60 calls, 39,276 submitted prompt
tokens, and 5,172 generated tokens. The first run took 102.192386 seconds; it is
not hidden warmup work or part of the primary run's timing.

The [frozen descriptor](../data/manifests/smoke-v1.json) records the full-corpus
SHA-256 as `326f7c01653406fffeda4e2f90294feced61b736465897e655ef9dbfc6a6b421`.
These 30 examples are development smoke data. No locked evaluation was created.

## Repeat the checks

Use a new output directory for every model run. Run from the repository root:

```sh
uv sync --locked
uv run --locked env PYTHONPATH=src python -m unittest discover -s tests -v
uv run --locked python scripts/evaluate.py runs/baseline-smoke-20261111-02
uv run --locked python scripts/run_baseline.py --output runs/baseline-smoke-003
```

The run artifacts are local and ignored by Git. Keep the source generator,
frozen descriptor, runner, tests, and this concise report in version control.

## Attention

Reviewed by an independent Codex agent in the available model family. The
review was not model-diverse. No blocking findings remain. The low success
count and the measurement qualifications above remain material limitations.
