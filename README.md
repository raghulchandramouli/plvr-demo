# PLVR-3B

Local Qwen2.5-3B-Instruct experiments on Apple silicon. The baseline, search,
training, and evaluation runners are placeholders.

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

The experiment policy lives in `configs/model.yaml`. Future runners must read
that policy, apply the pinned chat template and decoding settings, and enforce
the total sequence budget before model calls. The verifier does not turn the
placeholder runners into working inference commands.

See [assumptions.md](assumptions.md) for provenance, resource limits, and the
verified software status.
