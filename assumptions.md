# PLVR-3B assumptions

Recorded on 2026-11-11. This repository has downloaded source weights, a pinned MLX LM environment, and an implementation plan. Text generation, training, and benchmark runs have not started.

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

[configs/model.yaml](configs/model.yaml) pins the model and tokenizer to `aa8e72537993ba99e69dfaafa59ed015b17504d1`. The downloaded files in `model/` all report that revision in their Hugging Face metadata. No MLX conversion or quantization has been performed.

Both Safetensors headers were inspected: all 434 tensors are BF16, every tensor in the weight index is present, and the shard sizes match their tensor offsets. The two SHA-256 checksums match the download metadata:

| Shard | SHA-256 |
| --- | --- |
| `model-00001-of-00002.safetensors` | `67347b23fb4165b652eb6611f5e1f2a06dfcddba8e909df1b2b0b1857bee06c2` |
| `model-00002-of-00002.safetensors` | `a40d941d0e7e0b966ad8b62bb6d6b7c88cce1299197b599d9d0a4ce59aabfc1d` |

Start with this unquantized BF16 backbone. Preserve BF16 during any initial format conversion and verify the resulting tensor dtypes before running it. Runtime dtype and decoding settings still require validation in MLX LM. The upstream sampling defaults in `model/generation_config.json` have not been selected as experimental settings.

If quantization becomes necessary, record its reason, method, bit width, group size, conversion command, source revision, and resulting artifact hashes before running it. Give that configuration a distinct experiment identifier. Rerun every affected baseline and search comparison using the same quantized representation; report those results separately from BF16 runs.

## Starting resource limits

These are declared starting budgets, not measured performance or enforced runner limits. The runners are still placeholders and must enforce the applicable limits before experiments.

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
