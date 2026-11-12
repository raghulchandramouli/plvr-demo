# PLVR-3B assumptions

Recorded on 2026-11-11. This repository has downloaded source weights, a pinned MLX LM environment, a local smoke baseline, and verified oracle runtime/search mechanics. Training and model-based comparative benchmark runs have not started.

## Hardware and software

| Item | Observed value |
| --- | --- |
| Machine | MacBook Pro, Mac17,8, Apple M5 Pro |
| CPU | 18 cores, as reported by `system_profiler` |
| Unified memory | 48 GB |
| Architecture | arm64 |
| macOS | 26.6.2, build 25G83 |
| System Python | 3.9.6 |
| uv | 0.12.5 |
| Git | 2.50.1, Apple Git-155 |
| Project Python | CPython 3.12.14, pinned in `.python-version` and `pyproject.toml` |
| MLX and MLX LM | MLX 0.32.4, MLX Metal 0.32.4, MLX LM 0.32.2, installed in `.venv/` |
| Tokenizer environment | Transformers 5.22.2 and tokenizers 0.23.3, pinned in `uv.lock` |
| Configuration parser | PyYAML 6.0.3 |

Hardware comes from `system_profiler SPHardwareDataType`; system software comes from `sw_vers`, `python3 -V`, `uv --version`, and `git --version`. Project package versions come from installed distribution metadata. `uv pip check` passes for all 38 installed packages. `uv.lock` records all transitive package versions and distribution hashes. Recreate the project environment with `uv sync --locked`. The `transformers_version` in the downloaded model configuration describes its source export, not an installed dependency.

## Backbone and precision

[configs/model.yaml](configs/model.yaml) pins the model and tokenizer to `aa8e72537993ba99e69dfaafa59ed015b17504d1`. The downloaded files in `model/` all report that revision in their Hugging Face metadata. MLX LM 0.32.2 loads these source Safetensors directly. No conversion command is required, and no conversion or quantization has been performed.

Both Safetensors headers were inspected: all 434 tensors are BF16, every tensor in the weight index is present, and the shard sizes match their tensor offsets. The two SHA-256 checksums match the download metadata:

| Shard | SHA-256 |
| --- | --- |
| `model-00001-of-00002.safetensors` | `67347b23fb4165b652eb6611f5e1f2a06dfcddba8e909df1b2b0b1857bee06c2` |
| `model-00002-of-00002.safetensors` | `a40d941d0e7e0b966ad8b62bb6d6b7c88cce1299197b599d9d0a4ce59aabfc1d` |

Start with this unquantized BF16 backbone. Preserve BF16 during any future format conversion and verify the resulting tensor dtypes before running it.

### Pinned runtime and decoding

On 2026-11-11, `uv run --locked python scripts/verify_runtime.py` loaded the model on `Device(gpu, 0)` without network access or remote code execution. All 434 loaded parameter tensors were `mlx.core.bfloat16`. The check rendered the pinned tokenizer's default chat template with an assistant generation prefix and verified greedy selection on synthetic logits. It did not execute a model forward pass or generate text.

`configs/model.yaml` records the initial experimental policy:

- Greedy decoding with temperature 0, seed 0, top-p 1, min-p 0, and top-k 0. XTC is disabled. Sampling filters do not affect the temperature-zero argmax path.
- At most 512 generated tokens and 2,048 total input and output tokens. Future runners must check the rendered prompt length and enforce both limits before generation.
- Stop at either EOS token, 151643 or 151645. No custom stop strings or logits processors. Repetition, presence, and frequency penalties are disabled.
- Use the tokenizer's default chat template with `add_generation_prompt=true` and no template overrides.
- No weight or KV-cache quantization, no adapters, and a fresh prompt cache for each independent call.
- Use the Metal GPU. Set the MLX allocator guideline to 32 GiB. The verifier checks MLX peak allocation after loading. This is not a strict total-process memory cap.

The setup verifier's BF16 observation applies to loaded model parameters. The baseline now records KV-cache and emitted log-probability dtypes during real generation. It does not measure every forward-pass intermediate. The upstream sampling defaults in `model/generation_config.json` are not the experimental settings. Prefill chunks are explicitly pinned to 256 tokens in `configs/model.yaml`.

Recreate and check the environment from the repository root:

```sh
uv sync --locked
uv run --locked python scripts/verify_runtime.py
```

The verifier checks installed direct dependency versions against `pyproject.toml`. `uv sync --locked` installs the complete transitive environment from `uv.lock`. The [MLX LM package documentation](https://pypi.org/project/mlx-lm/0.32.2/) describes its loading API. The local installed source and the runtime check establish compatibility with these particular source weights.

If quantization becomes necessary, record its reason, method, bit width, group size, conversion command, source revision, and resulting artifact hashes before running it. Give that configuration a distinct experiment identifier. Rerun every affected baseline and search comparison using the same quantized representation; report those results separately from BF16 runs.

## Starting resource limits

These are declared starting budgets. The baseline enforces token limits, checks process RSS and MLX allocator peaks, checks disk budgets, and uses cooperative example deadlines. Search and training runners remain placeholders. The memory checks are boundary checks rather than operating-system limits, and a GPU operation can exceed a time or memory budget before control returns to Python.

| Resource | Starting limit |
| --- | --- |
| Location | This M5 Pro Mac for every model, search, training, evaluator, and simulator call |
| Concurrency | One model process, one active model call; no simultaneous training and inference |
| Memory | 32 GiB total experiment footprint, reserving the remaining memory for macOS and other applications; stop a pilot if it exceeds the budget or causes sustained swap growth |
| Initial sequence | At most 2,048 tokens including input and output; record truncation and reject examples that lose essential content |
| Training microbatch | 1 initially; profile before increasing |
| Smoke evaluation | 30 examples, at most 512 generated tokens per call and 120 seconds per example; failed or timed-out examples remain in results |
| Search structure | 2 layers, depth 2, beam width 4, per-key cap 4, at most 20 emitted layer plans |
| Search pilot | At most 1 hour and 200,000 generated tokens per run; stop at the first limit and report actual usage |
| Local artifacts | At most 50 GiB for generated data, adapters, caches, and traces in addition to the source model; check free space before each run |

Declare larger budgets for the full comparison after profiling on development data. Use comparable budgets across arms and freeze them before locked evaluation. Do not reduce a test set after seeing its results to fit a budget.

## Baseline smoke protocol

The deterministic generator in `src/plvr3b/baseline_tasks.py` constructs 30 development smoke examples, comprising 18 single-tool tasks and 12 two-call tasks. Ten two-call tasks require an earlier result field. Every example receives the complete 18-tool catalog. No locked evaluation set has been created, and these examples must not be copied into it.

The frozen descriptor in `data/manifests/smoke-v1.json` pins the canonical full-corpus SHA-256 to `326f7c01653406fffeda4e2f90294feced61b736465897e655ef9dbfc6a6b421`. The runner checks that hash and executes all 30 oracle programs before loading the model. Expected results, expected state changes, and required actions are specified separately from the executor's outputs. Only the request and tool catalog enter model prompts. Oracles and expected answers remain in the evaluator and saved manifest.

The initial grammar is a bounded `Sequence` with one to six calls and an explicit return expression:

```json
{"op":"Sequence","calls":[{"id":"c0","tool":"lookup_order","arguments":{"order_id":"O1"}},{"id":"c1","tool":"lookup_customer","arguments":{"customer_id":{"$ref":"c0.customer_id"}}}],"return":{"$ref":"c1"}}
```

Call IDs are unique identifiers. Arguments are literals or references to an earlier call, optionally selecting one result field. References resolve at execution time. Arbitrary expressions, forward references, nested field traversal, `Foreach`, branches, and search are not implemented in this phase. Each execution starts from a deep copy of its example's mock state.

The model output is parsed as exactly one JSON value. No Markdown extraction, repair, retry, or constrained decoding is applied. Duplicate keys and non-finite numbers are rejected. Outcomes are classified as follows:

- `format_failure` means JSON could not be parsed under that policy.
- `contract_failure` means the parsed program violates the grammar, tool schema, argument bounds, reference rules, or tool preconditions.
- `semantic_failure` means an executable program fails the declared result, final state, required tool-action multiset, or required result-binding checks.
- `success` requires all those checks. Renaming call IDs and reordering independent calls are allowed. Required dependent arguments must reference the requested earlier result rather than guess its value.
- Operational failures and cooperative timeouts have their own statuses. Their raw or partial output, counters, exceptions, and any evaluable program are retained. A resource abort retains unattempted examples as `not_run`.

This is a strict synthetic task contract. Tool identity and action multiplicity are part of the score. A behaviorally similar tool choice can therefore fail, such as using `lookup_product` when the task contract requires `check_inventory`, even though these mock tools currently expose the same fields. These results do not establish general task-solving ability or a search-space gate.

The runner uses one model process, one load, one active call, and a fresh KV cache per example. It holds an exclusive baseline lock and never overwrites a run directory. The saved metadata includes the model policy, environment, Git state, source/configuration hashes, and corpus hash. Per-example JSON traces and flushed token-event JSONL files retain exact prompts, outputs, parsed programs, violations, execution events, timing, and counters. Full expected inputs and oracle results are in the manifest.

Whole-run wall time starts at runner entry before oracle checks and model loading. It includes generation, scoring, and most artifact work. The final summary rewrite and lock release are excluded. Generation-loop time includes streaming, token-event writes, and resource checks. Submitted prompt tokens are counted for attempted calls; an interrupted prefill does not prove that every submitted token was processed. Generated-token counts use MLX's final cumulative response count, including terminal EOS when emitted, rather than summing cumulative chunks. Internal asynchronous lookahead computations are not included in that emitted-token count.

Memory reports keep process high-water RSS and cumulative MLX allocator peak separate. Darwin `ru_maxrss` is measured in bytes. The metrics overlap on unified memory and must not be added. Disk use is the logical byte count of all run artifact files, including the final summary. Free disk and swap usage are also captured before and after the run. `scripts/evaluate.py` replays outputs without MLX and reconciles saved classifications and aggregate counters.

The corrected run `runs/baseline-smoke-20261111-02` completed all 30 calls with one model load. It recorded 2 successes, 18 contract failures, 5 formatting failures, and 5 semantic failures. Wall time was 104.308828 seconds, with 19,638 submitted prompt tokens and 2,586 generated tokens. Process peak RSS was 6,458,277,888 bytes, MLX peak allocation was 6,397,980,784 bytes, and run artifacts occupied 936,643 logical bytes. Every trace recorded BF16 KV-cache and emitted log-probability arrays. The first development run is preserved separately. All 21 tests and the corrected run's offline replay audit pass. [The baseline report](experiments/baseline-smoke-v1.md) records the scorer correction, both runs, failure examples, and repeat commands.

## Phase 2 typed runtime and finite grammar

This phase implements deterministic oracle primitives, not a neural extractor or a trained primitive library. A structured request explicitly supplies tasks, intents, grounded arguments, dependencies, and a return reference. Free-form input without structured tasks fails with `needs_semantic_extraction`. Expected answers and fixture scores never enter primitive context.

### Types and provenance

`src/plvr3b/types.py` defines `Request`, `SourceSpan`, `Task`, `TaskList`, `DependencyGraph`, `ToolRegistry`, `ToolCall`, `ToolCallList`, `Scratchpad`, and `ContractViolation`. The domain records are frozen, except the per-execution scratchpad and result container. Task and call bindings are tuples. The registry is a read-only typed view over the existing 18 `baseline_tasks.TOOLS` functions and argument schemas. Phase 1's model, prompts, corpus, and saved traces are unchanged.

`Context` preserves exactly the request, tool registry, and budget. `Scratchpad` adds a fresh deep copy of the initial mock state, completed results, execution events, and primitive invocation events. No mutable scratchpad is shared across candidates.

Every argument is one of three expressions:

- `Literal(value, SourceSpan(request_id, start, end))` uses a zero-based, half-open Python character range in the original request text. The value's spelling must match that exact range, and the request identity must match. Booleans use `true` or `false` spelling.
- `ResultRef(task_id, field)` keeps a symbolic earlier-result source until execution. References are checked against declared result fields and semantic kinds before execution. A customer-ID input cannot bind to a product-ID output merely because both are strings. A whole-record reference is allowed as a return value, but not as a scalar argument.
- `Transform("parse_integer", grounded_string)` explicitly converts a grounded ASCII digit string of at most 256 characters. The operand must be a literal, not another transform or reference. The `03` fixture demonstrates normalization while retaining its operand and source span. Arbitrary expressions and transformation names are rejected. Integer literal representations are bounded to 256 digits before string conversion; tool-specific quantity limits remain tighter.

Resolved arguments retain their original expression provenance in execution events. Each tool-result field is tagged with its producing task and field name. Downstream values are read from completed results rather than guessed from expected answers or initial mock identifiers.

### Primitive signatures and contracts

Every oracle primitive implements `run(value, context)` and `validate(value, output, context)`. Validation returns a tuple of structured `ContractViolation` records. The executor invokes validation after each primitive run. Task success is computed separately by `score_fixture`.

| Primitive | Signature | Required behavior |
| --- | --- | --- |
| `Decompose` | `Request -> TaskList` | Preserve all supplied structured tasks, request order, arguments, identities, dependencies, and provenance. |
| `Get-Order` | `TaskList -> TaskList` | Stable topological ordering of explicit dependencies and result-reference edges. Preserve every task and binding. Independent ties follow request order. Reject cycles. |
| `Match` | `Task × ToolRegistry -> ToolCall` | Preserve the task identity and grounded arguments, select its declared supported intent, and retain unresolved result references. Do not execute tools. |
| `Foreach(Match)` | `TaskList × ToolRegistry -> ToolCallList` | Invoke single-task `Match` sequentially once per item. An explicitly empty collection produces no calls. |

The registry, request, and budgets remain shared context across these signatures. `ToolRegistry` is an external requirement; intermediate `TaskList` values produced by `Decompose` are not additional user inputs. Every primitive has the singleton parameter domain `{ { } }`. The only ordering algorithm is stable topological order. There is no identity variant or bulk `Match` signature.

Contracts check required and unexpected fields, argument names, scalar types, semantic reference types, source spans, result fields, known intents, unique task/call IDs, dependency existence and cycles, return presence, requested action coverage, and budgets. IDs have their declared customer/order/product/return/booking prefixes. Quantities are integers in `[1,100]`, with booleans excluded. Strings are nonempty and bounded to 256 characters at tool boundaries; dates must be valid canonical ISO dates. Booking statuses are `confirmed` or `cancelled`. Tool preconditions, result fields, and returned value types are checked during execution.

An explicit empty request has `tasks=[]` and `return=null`. Missing structured input is not treated as an empty request. Execution requires every dependency to have completed before its consumer. It rejects extra, duplicate, missing, or fabricated calls, and preserves completed events on a later contract failure. All `Parallel`, general `Loop`, and `Identity` operators are unsupported. Rejecting all parallel execution also rejects unsafe parallel mutation. Bounded-loop checks cover `Foreach` iteration and invocation limits.

### Grammar, bounds, and canonicalization

Phase 2's AST composes primitives. It is distinct from Phase 1's historical JSON `Sequence` of tool calls. Phase 1's wire grammar and replay behavior remain intact.

`configs/search.yaml` fixes maximum operator nesting to 2, Sequence width to 4, tasks/tool calls/Foreach iterations to 6 each, and oracle primitive run attempts to 16 per execution. Primitive leaves do not add operator depth. `Foreach` admits only the single `Match` primitive; nested loops and composed loop bodies are rejected. No conditionals or arbitrary executable expressions are admitted.

Enumeration forms Sequence words of lengths 1 through 4 from the three primitive leaves and the three syntactic Foreach-wrapped leaves. It checks actual signatures and retains only complete `Request -> ToolCallList` programs. Incompatible Foreach bodies fail type checks. Canonicalization flattens nested Sequences, removes singleton Sequences, and collapses adjacent repeated `Get-Order`. Repeating stable topological sorting does not create another useful behavior.

The resulting complete shapes are `Sequence(Decompose, Foreach(Match))` and `Sequence(Decompose, Get-Order, Foreach(Match))`. Actual execution, rather than syntax-tree counts, determines their fixture outcomes and behavior classes. Behavior fingerprints exclude call-ID spelling and timing. They include observed tool actions/results, final state, outcome, and contract failures. Cost-inclusive fingerprints additionally include the primitive invocation count.

The cost metric counts logical oracle `run` attempts, including failed attempts. It excludes validation CPU and operator dispatch. In particular, `Get-Order` validation recomputes its expected order without adding an invocation event. These counts are not neural calls, token costs, latency, or demonstrated compute savings.

The gate defines a generally useful workflow as one that succeeds on every admitted development fixture. It also reports coverage/cost Pareto candidates. Multiple Pareto candidates alone do not establish a rich program search space. A tiny space or one universally correct workflow requires exhaustive evaluation and does not justify guided search.

The final run `runs/phase2-oracle-20261111-01` enumerated 1,554 syntax trees, 3 type-valid trees, and 2 canonical programs. It saved 68 candidate/fixture traces across 34 oracle-validated development fixtures. Without `Get-Order`, the program solved 32/34 fixtures with 81 logical primitive run attempts and 43 completed tool calls. With `Get-Order`, it solved 34/34 with 115 run attempts and 47 tool calls. Both semantic and cost-inclusive behavior counts were 2. Only one workflow solved every fixture. The verdict is `limited` and `exhaustive_only`, with `guided_search_justified=false`.

All 49 tests pass. The saved executions, counters, hashes, and behavior fingerprints were recomputed and matched. Parser regression checks cover oversized integer operands/literals, nested transformation operands, program depth, and Sequence width. Phase 1's source, corpus, and replay remain unchanged. No model inference or fine-tuning was used for Phase 2. [The Phase 2 report](experiments/phase2-oracle-v1.md) records the exact workflows, limits, evidence, and repeat command. The richer search-space gate remains open; this result does not authorize a guided-search advantage claim.

## Phase 3 search, ranking, and caches

The comparison mode is `scripts/run_search.py --oracle-only --mode compare`. The default gate mode remains Phase 2. The operation catalog in `types.py` supplies one authoritative set of primitive signatures and versions. Every runtime tree node stores its operation, children, parameter record, free input requirements, output type, and constraints. Historical tool-program serialization is unchanged.

`resolve_types` starts from `ToolCallList` and the supplied types `{Request, ToolRegistry}`. It finds producers by declared signatures and the admitted single-task Foreach lift, then derives predecessor requirements until they close against supplied input. The bounded self-loop for `Get-Order` can be traversed before `Decompose` closes `TaskList` against `Request`. No target program, oracle action tree, or exhaustive candidate pool is supplied to discovery. The independent raw enumerator remains the reference. Catalog subsets may remove operators; changed signatures, versions, or context requirements fail explicitly as `unsupported_catalog` until a new library is implemented.

Type construction and `bind_parameters` are separate. The present domains are `[{}]` for each primitive, so there is one assignment per type plan. Nonempty or unknown parameter domains are rejected. Intermediate values produced by the composition do not become external requirements. Complete programs require exactly `Request` and `ToolRegistry`.

### Shared budgets and sampling

The declared comparison limits are 2 backward construction layers, operator depth 2, beam width 4, per-key cap 4, and at most 20 emitted layer plans. Layer 1 derives a goal-producing suffix and its unmet inputs. Layer 2 closes the remaining input through bounded predecessor compositions. Guided construction counts both open and closed derivation plans against the emission cap. Open plans are ordered structurally; only closed plans receive measured execution scores. The closed beam retains at most four candidates per required-input/output key and four globally.

All arms additionally share a maximum of 20 closed candidate evaluations and `20 × 34 = 680` candidate/fixture evaluations. A candidate is scored on the entire identical ordered fixture manifest; no arm compares a partial subset. The runtime's six-task/call/iteration and sixteen-primitive-invocation budgets remain in effect. Comparison mode requires the declared two-layer grammar. One-layer closure failure is separately tested by the type resolver.

Exhaustive uses the independent canonical reference. Guided constructs candidates backward and retains scored closed candidates. Random constructs the full admissible canonical pool before shuffling, then samples without replacement until the pool or shared evaluation cap is exhausted. It is uniform over canonical programs, not raw syntax trees. Seeds are 0, 1, and 2. A smaller emission cap is applied after pool construction and cannot bias the sampling population to the first discovered program.

Search stops at the first of three conditions, recorded per arm as `stop_reason`. `type_space_exhausted` means every admissible canonical program was evaluated. `plan_budget` means guided construction emitted 20 layer plans. `candidate_budget` means 20 closed candidates were evaluated. No arm stops early on a perfect score. The recorded runs stop with `type_space_exhausted`, because the pool has two programs.

Every arm and random seed gets an isolated cold cache. Oracle solvability checks execute the independent reference without caching before arm evaluation. Reference work is recorded separately and cannot warm a method's cache. All methods use the same final development-set selector.

### Exact local ranking policy

The four-field rank is `(unmet_requirements, type_errors, value_error, deviation)`, compared lexicographically. Canonical node count and canonical serialized label break remaining ties. Primitive call cost is not substituted for deviation or a rank component.

- `unmet_requirements` is 1 if any fixture fails execution or lacks a required result, otherwise 0. A legitimate empty request returning `None` is not a failure.
- Tool calls align one-to-one by maximizing matching function names first, then exact typed argument agreement. Stable index pairs break alignment ties. Alignment occurs before computing the rank; it cannot hide a type error by pairing two different functions.
- For a matched function, agreement is `0.5 + 0.5 × correct_typed_bindings / argument_key_union_size`. A zero-argument match has agreement 1. The error atom is `1 - agreement`. A wrong function gets no binding credit and contributes atom 1. Missing or extra calls each add one type error and atom 1. Missing, extra, or differently typed bindings add type errors. Booleans and integers are distinct.
- Returned values and full expected mock state also contribute atoms. Dictionaries align by field/entity key. Lists of records align by an unambiguous unique identifier using the declared priority `customer_id`, `order_id`, `product_id`, `return_id`, `booking_id`, `option_id`; other lists compare by position. Missing/extra fields or items and type mismatches add one type error and atom 1. Same-typed scalar differences add atom 1, with exact scalar matches adding 0. Numeric differences use exact agreement, not an invented continuous scale.
- Value error is `sqrt(sum(atom²))` across all scored objects and all examples. Per-example magnitudes are not summed. Action and output/state checks intentionally provide cumulative penalties; they do not normalize by the number of correct unchanged fields. A failed execution always has a nonzero error contribution.
- Deviation is the symmetric difference of free required input types and the shared incumbent input set `{Request, ToolRegistry}`. It is 0 for both current complete candidates. The local incumbent set is fixed; no AST-edit distance is used.

Strict `score_fixture` success remains separate from partial-credit ranking. This alignment, atom policy, full-fixture closed-plan scoring, and fixed incumbent set are explicit local choices. They are not claims of exact paper scoring fidelity.

### Cache identities and invalidation

`RuntimeCache` is integrated into `executor.invoke` and `run_program`. Primitive keys include the typed input, parameters, preserved request and read-only registry descriptors, budgets, and `VersionStamp`. The version stamp includes primitive/type/contract implementation hashes, model revision, adapter revision, prompt/template identity, full decoding digest, full model/runtime policy digest, and executor/tool implementation hashes. Registry callable identity includes bytecode/constants, positional defaults, keyword-only defaults, and closure values. Request IDs and source spans remain part of the input identity.

Execution keys additionally include the canonical program and full initial mock environment state. Caches store executions, not gold answers or scores. A changed expected answer is rescored even when execution is reused. Outputs are defensively copied on storage and retrieval. Primitive hits are contract-validated. Updating a version stamp, including after an adapter update, clears both stores. Tests verify actual misses after version, schema, implementation, environment, request/provenance, budget, and runtime-representation changes.

Logical primitive requests include cache-served requests. `primitive_misses` counts main-path primitive `run` attempts; validation recomputation is excluded. In particular, `Get-Order` validation recomputes ordering and is not counted as a main-path call. Execution-cache hits mark their primitive evidence as replayed rather than rerunning it. This caching applies only to the deterministic local mock runtime, with all environment state supplied explicitly.

### Recorded Phase 3 result

`runs/phase3-search-20261111-01` records discovery without a supplied program template. Exhaustive, guided, and random seeds 0/1/2 all select the 34/34 oracle workflow. Each cold arm evaluates 2 candidates on 34 fixtures, producing 68 fixture evaluations, 115 main-path primitive misses, 81 primitive hits, 68 execution-cache misses, and 0 execution-cache hits. Across the five arms there are 340 saved fixture executions; uncached reference preflight adds 68 separately recorded evaluations. Warm execution hits and adapter invalidation are exercised by tests, not claimed as observed cold-arm hits.

All 70 tests pass. Root independently replayed the 340 arm executions and reconciled rankings, selections, source/fixture hashes, and cache ledgers. Phase 1 replay and the Phase 2 gate still pass. The grammar remains 1,554 raw trees, 3 type-valid trees, 2 canonical programs, and 2 semantic behavior classes. This is a mechanics-only result with `guided_search_advantage=false` and `paper_matched=false`. No model inference or fine-tuning ran. See [the Phase 3 report](experiments/phase3-search-v1.md).

## License review

The current inventory contains the Qwen model and only `.gitkeep` files under `data/`. No third-party datasets or generated training examples are present. No checkpoint or dataset distribution is authorized by this review.

The local `model/LICENSE` is the [Qwen Research License at the pinned revision](https://huggingface.co/Qwen/Qwen2.5-3B-Instruct/blob/aa8e72537993ba99e69dfaafa59ed015b17504d1/LICENSE). It permits noncommercial research and evaluation; commercial use requires a separate license. Redistribution of model materials or derivatives requires the license copy, prominent modification notices, and the prescribed attribution in a `Notice` file. Documentation for a distributed model trained or improved using Qwen materials or outputs must display the Qwen attribution required by section 4(b).

Planned external sources were reviewed separately from the model:

| Source | Current finding and release condition |
| --- | --- |
| [BFCL dataset](https://huggingface.co/datasets/gorilla-llm/Berkeley-Function-Calling-Leaderboard) | The dataset card declares Apache-2.0. Pin the actual dataset revision and retain its license, attribution, applicable NOTICE, and modification notices before distributing any included data. |
| [LiveCodeBench dataset card](https://huggingface.co/datasets/livecodebench/code_generation_lite/raw/main/README.md) and [loader](https://huggingface.co/datasets/livecodebench/code_generation_lite/raw/main/code_generation_lite.py) | The card declares generic `cc`, while the loader declares MIT. Rights to sourced competition problems remain unresolved. Do not redistribute these data or train a distributed checkpoint on them until the exact selected content and terms are resolved. |
| [tau2-bench repository license](https://github.com/sierra-research/tau2-bench/blob/main/LICENSE) | The repository is MIT-licensed. Review the exact selected domain and any separately sourced assets at the pinned revision before distribution. Preserve applicable notices. |

For synthetic data, record generator provenance and any source content or local Qwen outputs in its manifest. Review those terms before distributing the data or a checkpoint trained on them. A code repository's license alone does not establish rights to every dataset asset. Repeat the review when an actual dataset revision or release artifact is selected.

## Differences from the target paper

Paper status checked on 2026-11-11. The [current arXiv record](https://arxiv.org/abs/2608.28421) reports that the authors withdrew v2 on 2026-09-30 because of benchmark baseline errors and discarded the experiments. The pinned v1 remains a mechanism reference for this local study. Its benchmark claims are not treated as validated reproduction targets.

Target: [PLVR, arXiv 2608.28421v1](https://arxiv.org/html/2608.28421v1). These are planned adaptations, not reproduced results:

- One shared Qwen2.5-3B-Instruct backbone with prompts first and separate LoRA adapters later replaces the paper's multi-model primitive library.
- The first study uses only `Decompose`, `Get-Order`, and `Match`, with `Sequence` and `Foreach` on synthetic straight-line tasks. The broader primitive library, conditional control flow, and external benchmarks come later.
- MLX LM on one M5 Pro replaces the paper's execution and training setup. Local resource caps, data allocations, and proposed LoRA settings are project choices.
- Exact primitive signatures, operator semantics, loss, alignment, ranking, and search-limit correspondence still require an audit against the paper before claiming mechanism fidelity.
- Future tau2 evaluation uses a local Qwen user simulator. Record that simulator change as a benchmark adaptation.

The Phase 3 audit checked v1 sections 3.5 and 3.6 and Appendix A.1/A.3. Its final search settings are 4 layers, composition depth 3, beam width 8, per-key cap 8, and 60 plans. The requested initial settings of 2, 2, 4, 4, and 20 are local debugging adaptations. The ranking uses unmet execution requirements, type errors, cumulative L2 value error, and required-input-set deviation, followed by size and label. The local closed-plan evaluator and exact partial-credit rules must be recorded separately; this project does not claim paper-matched search.

No author implementation was available on 2026-11-12. The v1 abstract says the authors release a symbolic backpropagation library and a conformance checker, but v1 gives no URL. Its reproducibility statement says both will be released after peer review. The current arXiv record links no code. Every primitive, grammar, ranking, and search component here is a local implementation written from the v1 text.

## Files kept out of Git

`.gitignore` excludes `model/`, generated contents of `data/raw/`, `data/generated/`, `data/processed/`, checkpoint and adapter directories, common weight formats, caches, runs, traces, logs, and credential files. Put generated datasets and full traces in those directories. Keep generators, manifests, configuration, and concise reports in Git. Use `.env.example` only for sanitized placeholders.

Ignore rules do not remove already tracked files. The index was checked during this setup; it contained no model weights, generated datasets, credentials, or execution traces. Check the staged paths again before a future commit or release.
