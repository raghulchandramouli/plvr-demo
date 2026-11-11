# PLVR-3B assumptions

Recorded on 2026-11-11. This repository has downloaded source weights, a pinned MLX LM environment, and a working local smoke baseline. Training, search, and comparative benchmark runs have not started.

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

Target: [PLVR, arXiv 2608.28421v1](https://arxiv.org/html/2608.28421v1). These are planned adaptations, not reproduced results:

- One shared Qwen2.5-3B-Instruct backbone with prompts first and separate LoRA adapters later replaces the paper's multi-model primitive library.
- The first study uses only `Decompose`, `Get-Order`, and `Match`, with `Sequence` and `Foreach` on synthetic straight-line tasks. The broader primitive library, conditional control flow, and external benchmarks come later.
- MLX LM on one M5 Pro replaces the paper's execution and training setup. Local resource caps, data allocations, and proposed LoRA settings are project choices.
- Exact primitive signatures, operator semantics, loss, alignment, ranking, and search-limit correspondence still require an audit against the paper before claiming mechanism fidelity.
- Future tau2 evaluation uses a local Qwen user simulator. Record that simulator change as a benchmark adaptation.

## Files kept out of Git

`.gitignore` excludes `model/`, generated contents of `data/raw/`, `data/generated/`, `data/processed/`, checkpoint and adapter directories, common weight formats, caches, runs, traces, logs, and credential files. Put generated datasets and full traces in those directories. Keep generators, manifests, configuration, and concise reports in Git. Use `.env.example` only for sanitized placeholders.

Ignore rules do not remove already tracked files. The index was checked during this setup; it contained no model weights, generated datasets, credentials, or execution traces. Check the staged paths again before a future commit or release.
