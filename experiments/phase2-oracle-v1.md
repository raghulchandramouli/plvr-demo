# Typed runtime and search-space result

Recorded on 2026-11-11. The typed runtime checks pass. The search-space verdict
is **limited, exhaustive evaluation only**. One workflow succeeds across all
admitted fixtures. No extra operators were added to create artificial choices.

This phase uses deterministic oracle primitives. It does not measure Qwen
accuracy, run inference, or change the earlier 2/30 model baseline result.

## Measured program space

The [saved gate summary](../runs/phase2-oracle-20261111-01/gate_summary.json)
records these results:

| Check | Result |
| --- | ---: |
| Raw syntax trees | 1,554 |
| Type-valid complete trees | 3 |
| Canonical programs | 2 |
| Distinct execution behavior classes, with or without cost | 2 |
| Oracle fixtures validated before enumeration | 34 |
| Candidate/fixture traces | 68 |
| Universally correct workflows | 1 |
| Unit tests passing, including Phase 1 | 49 |
| Model calls | 0 |

Enumeration constructs Sequence words of lengths one through four from six
stage forms. The count is `6 + 36 + 216 + 1296 = 1554`. Signature checks reject
incompatible compositions. The third type-valid tree repeats `Get-Order` and
canonicalizes to the ordered workflow. It is not counted as another choice.

| Canonical workflow | Fixtures passed | Logical oracle run attempts | Completed tool calls |
| --- | ---: | ---: | ---: |
| `Decompose -> Foreach(Match)` | 32/34 | 81 | 43 |
| `Decompose -> Get-Order -> Foreach(Match)` | 34/34 | 115 | 47 |

The 34 fixtures comprise the 30 existing smoke tasks, two reversed-dependency
variants, an explicit empty request, and a grounded `03` integer transformation.
They are development oracle fixtures, separate from locked evaluation.

On each reversed fixture, preserving request order attempts a consumer before
its dependency and fails with `unresolved_dependency`. Stable topological
ordering executes the producer first and resolves the actual returned value.
For example, compare the [unordered trace](../runs/phase2-oracle-20261111-01/traces/2362d89fc1d2-smoke-019-reversed.json)
with the [ordered trace](../runs/phase2-oracle-20261111-01/traces/17dd6d7f7d5d-smoke-019-reversed.json).

Both candidates are Pareto choices under the declared coverage/count metric.
The unordered workflow uses fewer logical oracle run attempts but fails two
fixtures. Only the ordering workflow solves every fixture. Two candidates form
a trivial exhaustive space, which does not justify guided search. Phase 3 later
ran guided and random search on this space as a mechanics check only. See the
[Phase 3 report](phase3-search-v1.md).

Invocation counts exclude validation CPU and operator dispatch. They include
failed logical run attempts. They are not neural calls, token costs, latency,
or demonstrated compute savings. Semantic behavior fingerprints also exclude
invocation counts, and still distinguish the two candidates on reversed inputs.

## Verified runtime behavior

The requested domain types, provenance expressions, primitive contracts, and
budgets are documented in [assumptions.md](../assumptions.md). The registry
reuses all 18 existing tool implementations. Every candidate starts with a fresh
state copy and scratchpad. Primitive context contains request, registry, and
budget, with no expected answers.

Tests cover valid oracle execution, source-span equality, semantic reference
types, required and unsupported inputs, argument values, duplicate tasks/calls,
dependency cycles, dynamic references, preserved partial events, output schemas,
empty collections, bounded Foreach iterations, exhausted budgets, extra actions,
and rejection of Parallel, Loop, and incompatible compositions. Primitive
`validate` methods reject dropped tasks, incorrect order, and fabricated calls.

Independent review exposed two parser failures that existing tests missed.
Oversized integer transformations and deeply nested programs escaped as raw
Python exceptions. Regression tests failed before the fixes and now pass.
Related tests also bound nested transformation operands and oversized integer
literals. Invalid input now fails with structured contract violations before
conversion or recursion exceeds the declared bounds.

The final run's 68 saved executions were independently recomputed by the root
agent. Source hashes, fixture hash, classifications, primitive/tool counts, and
behavior fingerprints all matched. The audit is saved locally at
[artifact-audit.json](../runs/phase2-development/artifact-audit.json).
The Phase 1 source and frozen corpus are unchanged, and its saved-run replay
still passes.

## Repeat the experiment

Run from the repository root with a new output directory:

```sh
uv sync --locked
uv run --locked env PYTHONPATH=src python -m unittest discover -s tests -v
uv run --locked python scripts/run_search.py --oracle-only --output runs/phase2-oracle-002
```

Each run saves the fixture manifest, oracle checks, all syntax/type decisions,
configuration/source hashes, candidate traces, and the gate summary. The command
completes successfully when the experiment runs. Its `exhaustive_only` verdict
does not mean that the richer search-space gate passed.

Model the Domain keeps values and their origins in explicit records. Boundary
Discipline rejects invalid data during parsing. Build the Lever provides the
rerunnable exhaustive command. Prove It Works checks actual executions and saved
artifacts. The decision trail is local at
[decisions.tsv](../runs/phase2-development/decisions.tsv).

## Attention

Only one workflow is correct across all admitted fixtures. This is a runtime
milestone and a limited search-space result. Free-form neural extraction,
general loops, parallel execution, fine-tuning, and guided search were not
implemented. Independent review used the available Codex model family.
