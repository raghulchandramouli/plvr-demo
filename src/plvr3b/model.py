import importlib.metadata
import os
from pathlib import Path
import platform
import time
import tomllib

from .baseline_tasks import prompt_messages


class MLXModel:
    def __init__(self, root: Path, config: dict):
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        import mlx.core as mx
        from mlx.utils import tree_flatten
        from mlx_lm import load
        from mlx_lm.sample_utils import make_logits_processors, make_sampler

        self.mx = mx
        self.config = config
        self.last_observations = {}
        python = (root / ".python-version").read_text().strip()
        project = tomllib.loads((root / "pyproject.toml").read_text())["project"]
        if platform.python_version() != python or project["requires-python"] != f"=={python}":
            raise ValueError("Python differs from the project pin")
        for requirement in project["dependencies"]:
            name, version = requirement.split("==")
            if importlib.metadata.version(name) != version:
                raise ValueError(f"Installed {name} differs from project pin")
        runtime, decoding = config["runtime"], config["decoding"]
        if not (runtime["device"] == "gpu" and mx.is_available(mx.gpu)):
            raise ValueError("Pinned Metal GPU unavailable")
        if config["dtype"] != "bfloat16" or runtime["weight_dtype"] != "bfloat16" or config["quantization"] is not None:
            raise ValueError("Initial baseline requires unquantized BF16 weights")
        if runtime["kv_bits"] is not None or runtime["prompt_cache_between_calls"] or runtime["trust_remote_code"]:
            raise ValueError("Initial baseline requires fresh unquantized cache and remote code disabled")
        if decoding["temp"] != 0 or decoding["stop"] != "model_eos_token_ids":
            raise ValueError("Initial baseline requires greedy decoding with model EOS stops")
        mx.set_default_device(mx.gpu)
        mx.set_memory_limit(runtime["mlx_memory_limit_bytes"])
        self.model, self.tokenizer = load(
            str(root / config["path"]), tokenizer_config={"trust_remote_code": False, "local_files_only": True},
            trust_remote_code=False,
        )
        parameters = tree_flatten(self.model.parameters())
        if not parameters or any(value.dtype != mx.bfloat16 for _, value in parameters):
            raise ValueError("Loaded model has non-BF16 parameters")
        self.sampler = make_sampler(**{name: decoding[name] for name in (
            "temp", "top_p", "min_p", "top_k", "min_tokens_to_keep", "xtc_probability", "xtc_threshold",
        )})
        self.processors = make_logits_processors(**{name: decoding[name] for name in (
            "repetition_penalty", "presence_penalty", "frequency_penalty", "logit_bias",
        )})
        if self.processors:
            raise ValueError("Initial baseline requires no logits processors")
        self.metadata = {
            "model_load_count": 1, "parameter_tensors": len(parameters), "weight_dtype": "bfloat16",
            "device": mx.device_info(), "eos_token_ids": sorted(self.tokenizer.eos_token_ids),
        }

    def prepare(self, public):
        template = self.config["chat_template"]
        if template["source"] != "tokenizer_default" or not template["add_generation_prompt"] or template["kwargs"]:
            raise ValueError("Initial baseline requires the pinned default chat template")
        prompt = self.tokenizer.apply_chat_template(
            prompt_messages(public), tokenize=False, add_generation_prompt=True,
        )
        tokens = self.tokenizer.encode(prompt, add_special_tokens=False)
        if len(tokens) + self.config["decoding"]["max_tokens"] > self.config["runtime"]["max_sequence_tokens"]:
            raise ValueError("Rendered prompt plus full output allowance exceeds sequence budget; no truncation")
        return prompt, tokens

    def peak_memory(self):
        return self.mx.get_peak_memory()

    def generate(self, tokens, deadline):
        from mlx.utils import tree_flatten
        from mlx_lm import stream_generate
        from mlx_lm.models.cache import make_prompt_cache

        mx = self.mx
        decoding, runtime = self.config["decoding"], self.config["runtime"]
        mx.random.seed(decoding["seed"])
        cache = make_prompt_cache(self.model)
        self.last_observations = {}

        def progress(*_):
            if time.perf_counter() > deadline:
                raise TimeoutError("Cooperative example deadline exceeded")
            if mx.get_peak_memory() > runtime["mlx_memory_limit_bytes"]:
                raise MemoryError("MLX allocator peak exceeded the memory budget")

        stream = stream_generate(
            self.model, self.tokenizer, tokens, max_tokens=decoding["max_tokens"],
            sampler=self.sampler, logits_processors=self.processors, prompt_cache=cache,
            kv_bits=None, prefill_step_size=runtime["prefill_step_size"], prompt_progress_callback=progress,
        )
        try:
            for response in stream:
                self.last_observations["logprobs_dtype"] = str(response.logprobs.dtype)
                yield {
                    "text": response.text, "token_id": response.token,
                    "prompt_tokens": response.prompt_tokens, "generation_tokens": response.generation_tokens,
                    "finish_reason": response.finish_reason,
                }
                progress()
        finally:
            stream.close()
            mx.synchronize()
            dtypes = {str(array.dtype) for layer in cache for _, array in tree_flatten(layer.state) if hasattr(array, "dtype")}
            self.last_observations["kv_cache_dtypes"] = sorted(dtypes)
            self.last_observations["mlx_peak_bytes"] = mx.get_peak_memory()
