# Search mechanics result

Recorded on 2026-11-11. The Phase 3 discovery gate passes on the bounded oracle
grammar. Exhaustive search, guided search, and random seeds 0, 1, and 2 all
discover the same program that solves 34/34 development fixtures.

There are only two canonical programs. These results verify mechanics and
controls, not a guided-search advantage or improved Qwen accuracy. No model
inference or fine-tuning ran.

## Recorded comparison

The [comparison summary](../runs/phase3-search-20261111-01/comparison_summary.json)
records 1,554 raw trees, 3 type-valid trees, 2 canonical programs, and 2 semantic
behavior classes. The repeated `Get-Order` tree is canonicalized away.

| Arm | Candidates evaluated | Fixture evaluations | Selected successes | Primitive misses | Primitive hits |
| --- | ---: | ---: | ---: | ---: | ---: |
| Exhaustive | 2 | 68 | 34/34 | 115 | 81 |
| Guided | 2 | 68 | 34/34 | 115 | 81 |
| Random seed 0 | 2 | 68 | 34/34 | 115 | 81 |
| Random seed 1 | 2 | 68 | 34/34 | 115 | 81 |
| Random seed 2 | 2 | 68 | 34/34 | 115 | 81 |

Every selected rank is `(0, 0, 0, 0)`. Every arm starts with an isolated cold
cache and uses the same fixture order, runtime budgets, and development selector.
Each arm exhausts the pool. Random sampling is uniform over canonical admissible
programs without replacement, rather than raw trees with redundant variants.

Primitive misses count main-path `run` attempts. Validation recomputation is
excluded. The two candidates request 196 logical primitive invocations per arm;
81 use shared cached outputs and 115 miss. Each cold arm has 68 execution-cache
misses and no execution-cache hits because candidate/environment pairs are unique.
Warm execution hits, defensive copying, and version invalidation are verified
separately by execution-level unit tests.

There are 340 saved arm/fixture executions. Reference preflight adds 68 uncached
fixture evaluations, recorded separately in metadata. It does not warm any arm.
These work counts are not latency, neural-call, or compute-savings measurements.

## Discovery and bounded search

The existing AST now stores operation, children, parameters, required inputs,
output type, and constraints. One catalog supplies primitive signatures.
Backward construction begins at `ToolCallList`, derives its producer and remaining
input types, and closes those inputs against supplied `Request` and `ToolRegistry`.
Intermediate `TaskList` and `Task` values are not external user requirements.

Discovery does not receive `ordered_program()` or the exhaustive candidate pool.
Tests disable the known-program helpers and compare independently constructed
backward and exhaustive canonical sets. Removing `Get-Order` removes that
candidate. Unsupported signature/context overrides fail explicitly; the current
oracle catalog supports unchanged subsets only.

Type construction precedes parameter binding. The three current primitive
domains each contain one empty record. The implementation rejects invented
parameters rather than increasing the candidate count.

The debugging limits are two backward construction layers, operator depth two,
beam width four, per-key cap four, and twenty emitted layer plans. Closed
candidate evaluation is capped at twenty candidates, each on all 34 fixtures.
Guided construction counts open and closed derivation plans; it scores closed
plans and retains at most four per input/output key and four globally. Open plans
have deterministic structural ordering, not fabricated execution rewards.

Random constructs its full admissible pool before shuffling and applying the
shared closed-candidate cap. A tighter emission budget cannot silently truncate
the sampling population. Comparison runs require the declared two-layer grammar;
the type resolver separately tests failure to close with one layer. Binding-cap
tests verify plan, candidate, beam, and per-key behavior.

## Shared ranking and caches

All arms use the same lexicographic rank:

```text
(unmet_requirements, type_errors, sqrt(sum(value_error_atoms_squared)), deviation)
```

Canonical AST size and canonical label break remaining ties. Primitive invocation
cost is reported separately. See [assumptions.md](../assumptions.md) for exact
alignment, partial-credit, missing/extra-object, and deviation rules.

Primitive cache keys include the typed input, preserved request/registry context,
budgets, parameters, and implementation/model/adapter/prompt/decoding/runtime
identities. Execution keys additionally include the canonical program and full
initial environment. Cached executions contain no expected answers or scores.
Scoring is recomputed when gold answers change.

Cache outputs are validated and copied. Adapter or other version updates clear
both stores. Callable identity includes code, constants, positional defaults,
keyword-only defaults, and closure values. An executed regression caught stale
reuse after a keyword-default change; it failed before that field was added and
passes afterward. Caching is scoped to the deterministic local mock runtime.

## Verification and paper status

All 70 tests pass. Root independently replayed all 340 saved arm executions and
recomputed contributions, rankings, selections, and cache ledgers. Source and
fixture hashes match. The [audit](../runs/phase3-development/artifact-audit.json)
records the checks. Phase 1 replay and the Phase 2 exhaustive gate also pass.

On 2026-11-12, imports in `scoring.py` and `planning.py` moved to module level.
Behavior did not change. The rerun `runs/phase3-search-20261112-02` matches the
recorded run in every saved file except those two source hashes. That covers the
summary, backward resolution, enumeration, fixtures, oracle checks, and all 15
arm files. Every arm stops with `type_space_exhausted`.

The [current arXiv record](https://arxiv.org/abs/2608.28421) reports withdrawal
after benchmark baseline errors and discarded experiments. The local audit uses
[pinned v1](https://arxiv.org/html/2608.28421v1) only as a mechanism reference.
Its final settings differ from these debugging limits. This is not paper-matched
search or reproduction of validated experimental claims.

## Repeat

Use a new output directory:

```sh
uv sync --locked
uv run --locked env PYTHONPATH=src python -m unittest discover -s tests -v
uv run --locked python scripts/run_search.py --oracle-only --mode compare --output runs/phase3-search-002
```

Build the Lever provides the comparison runner. Model the Domain keeps tree
metadata and provenance explicit. Prove It Works validates actual saved outcomes.
The local [decision trail](../runs/phase3-development/decisions.tsv) records review
findings and fixes.

## Attention

The tiny grammar still supports exhaustive evaluation only as an experimental
conclusion. Guided/random machinery is tested, but its presence is not evidence
of an advantage. Neural primitive quality and training remain untested here.
Independent review used the available Codex model family.
