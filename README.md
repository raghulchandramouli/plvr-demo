# plvr-demo

This repository reproduces the τ²-bench Telecom result of CoreThink's paper "Compiling Expert Procedures into Verified Programs: When a Small Model Is Enough". It uses local Qwen3 models on an Apple-silicon Mac.

The paper reports that on τ² Telecom, an 8B model under its program goes from 43.0% to 91.2%. It also reports that the gap between student models shrinks from 15.0 points to 1.1. This repository tests that claim with Qwen3-4B, Qwen3-8B and Qwen3-14B as students. Qwen3-8B runs the primitives, and Qwen3-30B-A3B plays the τ² user.

Status: in progress. No results yet.

- [Pre-registration](experiments/telecom-headline/preregistration.md) lists the claims, the verdict rules, and every deviation from the paper.
- [Decision log](experiments/telecom-headline/decisions.md) records each decision as it was made.

## Setup

You need macOS on Apple silicon, about 32 GB of free unified memory, and [uv](https://docs.astral.sh/uv/).

```bash
scripts/setup.sh
```

The script clones τ²-bench at a pinned commit into `vendor/`. It then installs the project into `.venv/` and downloads the four MLX models, about 32 GB in total.
