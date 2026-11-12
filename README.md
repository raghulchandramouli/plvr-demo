# PLVR-3B

Local Qwen2.5-3B-Instruct experiments on Apple silicon. The smoke baseline runs
against deterministic mock tools. Program search runs on deterministic oracle
primitives only; Qwen-backed primitives and training are not implemented yet.

## Prepare the runtime

The environment uses Python 3.12.14, MLX 0.32.4, and MLX LM 0.32.2.
`uv.lock` pins transitive dependencies and distribution hashes. Run from the
repository root with uv 0.12.5.

```sh
uv sync --locked
uv run --locked python scripts/verify_runtime.py
```

The verifier requires the source weights and tokenizer in `model/` at the
revision recorded in [configs/model.yaml](configs/model.yaml). It loads them
locally, checks parameter dtypes and the greedy sampler, and prints a JSON
report. It does not generate text or run an evaluation.

The baseline reads the experiment policy from `configs/model.yaml` and the
resource limits from `configs/evaluation.yaml`.

## Run the smoke baseline

Run these commands from the repository root. Use a new output directory for
each run. Existing directories are never overwritten.

```sh
uv run --locked env PYTHONPATH=src python -m unittest discover -s tests -v
uv run --locked python scripts/run_baseline.py --oracle-only --output runs/oracle-check-001
uv run --locked python scripts/run_baseline.py --output runs/baseline-smoke-001
uv run --locked python scripts/evaluate.py runs/baseline-smoke-001
```

The model command loads Qwen once and makes 30 sequential calls. Every example
receives the same catalog of 18 customer, order, inventory, return, and booking
tools. All tools operate on fresh in-memory mock state. No real accounts or
external services are involved.

The deterministic generator produces 30 development smoke tasks. It validates
every oracle before loading the model. These examples are separate from locked
evaluation. Oracle programs and expected results never enter model prompts.

Each run saves `manifest.json`, `oracle_checks.json`, `metadata.json`,
`summary.json`, and per-example files under `traces/`. JSON traces contain the
exact prompt, raw output, parsed program, violations, execution events, observed
results, timing, tokens, and memory counters. The manifest records each example's
oracle and expected results. Streamed token events
are flushed to JSONL files. Formatting, contract, and semantic failures remain
in the results alongside operational errors and timeouts.

`scripts/evaluate.py` replays saved outputs without loading the model. It checks
the classifications and reconciles token, call, timing, and artifact totals.
The run command exits successfully when the run finishes, even if individual
examples fail. Read `summary.json` for task outcomes.

The runner enforces the full 512-token output reservation within a 2,048-token
total sequence limit. It checks a 120-second cooperative deadline during prefill
and streaming. A single GPU operation can overrun that deadline. Memory reports
keep process peak RSS and MLX allocator peak separate because their accounting
overlaps on unified memory. Generation-loop timing includes streaming and trace
recording. Whole-run timing also includes model loading and evaluation.

See [assumptions.md](assumptions.md) for provenance, resource limits, and the
verified software status.

## Check the typed runtime and finite program space

Phase 2 uses deterministic oracle primitives against the same 18 mock tools.
It does not load Qwen. Run from the repository root with a new output directory:

```sh
uv run --locked env PYTHONPATH=src python -m unittest discover -s tests -v
uv run --locked python scripts/run_search.py --oracle-only --output runs/phase2-oracle-001
```

The command validates the oracle fixtures before enumerating the declared
grammar. It saves `fixtures.json`, `oracle_checks.json`, `enumeration.json`,
`metadata.json`, `gate_summary.json`, and a trace for each candidate/fixture pair.

The grammar composes `Decompose`, optional `Get-Order`, and `Foreach(Match)`.
`Match` handles one task. Literal arguments carry checked request spans;
references retain their source task until execution. Every candidate gets fresh
mock state. Primitive validation and task scoring are separate.

A completed command means the exhaustive experiment ran. Inspect `gate`,
`verdict`, and `guided_search_justified` in `gate_summary.json` for the search-space
decision. A tiny space or one universally correct workflow calls for exhaustive
evaluation. It does not justify a guided-search comparison.

The [Phase 2 report](experiments/phase2-oracle-v1.md) records the current result.
The ordered oracle workflow passes 34/34 fixtures; preserving request order
passes 32/34. Only one workflow succeeds universally, so the verdict is
`exhaustive_only`. This does not improve or replace the measured Qwen baseline.

## Compare search mechanics

```sh
uv run --locked python scripts/run_search.py --oracle-only --mode compare --output runs/phase3-search-001
```

This mode constructs programs backward from the desired output type and compares
exhaustive, bounded guided, and seeded random search with shared scoring and
budgets. Each arm uses its own cold versioned caches. The default `--mode gate`
preserves the Phase 2 command.

All five arms currently select the 34/34 oracle workflow from two canonical
programs. This verifies mechanics, not a guided-search advantage or model accuracy.
See the [Phase 3 report](experiments/phase3-search-v1.md) for evidence, cache
accounting, ranking rules, and the target paper's withdrawal status.
