"""Check the pinned local runtime without generating model output."""

import importlib.metadata
import json
import os
from pathlib import Path
import platform
import tomllib

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import mlx.core as mx
from mlx.utils import tree_flatten
from mlx_lm import load
from mlx_lm.sample_utils import make_logits_processors, make_sampler
import yaml

root = Path(__file__).resolve().parents[1]
project = tomllib.loads((root / "pyproject.toml").read_text())
config = yaml.safe_load((root / "configs/model.yaml").read_text())
runtime = config["runtime"]
decoding = config["decoding"]
template = config["chat_template"]
python_version = (root / ".python-version").read_text().strip()
assert platform.python_version() == python_version, "Python differs from .python-version"
assert project["project"]["requires-python"] == f"=={python_version}"
versions = {}
for requirement in project["project"]["dependencies"]:
    name, version = requirement.split("==")
    versions[name] = importlib.metadata.version(name)
    assert versions[name] == version, f"Installed {name} differs from the project pin"

assert runtime["device"] == "gpu" and mx.is_available(mx.gpu), "Metal GPU unavailable"
assert config["dtype"] == runtime["weight_dtype"] == "bfloat16"
assert config["quantization"] is None and runtime["kv_bits"] is None
assert runtime["trust_remote_code"] is False
assert runtime["prompt_cache_between_calls"] is False
assert 0 < decoding["max_tokens"] < runtime["max_sequence_tokens"]
mx.set_default_device(mx.gpu)
mx.set_memory_limit(runtime["mlx_memory_limit_bytes"])
mx.random.seed(decoding["seed"])

model_path = root / config["path"]
assert model_path.is_dir(), "Local model directory missing"
model, tokenizer = load(
    str(model_path),
    tokenizer_config={"trust_remote_code": False, "local_files_only": True},
    trust_remote_code=False,
)
parameters = tree_flatten(model.parameters())
assert parameters, "Model has no parameters"
assert all(value.dtype == mx.bfloat16 for _, value in parameters), "Non-BF16 weights loaded"
assert mx.get_peak_memory() <= runtime["mlx_memory_limit_bytes"], "Load exceeded the MLX memory budget"
assert template["source"] == "tokenizer_default" and tokenizer.chat_template
assert template["add_generation_prompt"] is True and template["kwargs"] == {}
prompt = tokenizer.apply_chat_template(
    [{"role": "user", "content": "Runtime check."}],
    tokenize=False,
    add_generation_prompt=template["add_generation_prompt"],
    **template["kwargs"],
)
assert prompt.endswith("<|im_start|>assistant\n"), "Qwen assistant prefix missing"
assert decoding["stop"] == "model_eos_token_ids" and tokenizer.eos_token_ids
sampler = make_sampler(**{
    name: decoding[name] for name in (
        "temp", "top_p", "min_p", "top_k", "min_tokens_to_keep",
        "xtc_probability", "xtc_threshold",
    )
})
assert decoding["temp"] == 0.0, "Initial policy requires greedy decoding"
assert sampler(mx.array([[0.0, 2.0, 1.0]])).item() == 1, "Sampler is not greedy"
processors = make_logits_processors(**{
    name: decoding[name] for name in (
        "repetition_penalty", "presence_penalty", "frequency_penalty", "logit_bias",
    )
})
assert processors == [], "Initial policy requires no logits processors"
print(json.dumps({
    "python": python_version,
    "packages": versions,
    "device": str(mx.default_device()),
    "model_path": str(model_path),
    "parameter_tensors": len(parameters),
    "weight_dtype": "bfloat16",
    "conversion": config["conversion"],
    "eos_token_ids": sorted(tokenizer.eos_token_ids),
    "runtime": runtime,
    "decoding": decoding,
    "chat_template": template,
    "generation_performed": False,
}, indent=2))
