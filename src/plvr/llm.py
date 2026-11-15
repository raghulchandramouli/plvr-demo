"""Talk to a local mlx_lm server: render prompts with the model's own chat template, then complete them as raw text."""

import json
import time
from dataclasses import dataclass
from functools import cached_property

import httpx

from plvr.models import LocalModel


@dataclass(frozen=True)
class ToolCall:
    name: str
    arguments: dict


@dataclass(frozen=True)
class Completion:
    text: str
    reasoning: str
    tool_calls: list[ToolCall]
    prompt_tokens: int
    cached_tokens: int
    completion_tokens: int
    seconds: float
    finish_reason: str | None


class Server:
    def __init__(self, model: LocalModel):
        self.model = model
        self._client = httpx.Client(base_url=model.api_base, timeout=httpx.Timeout(3600.0))

    @cached_property
    def tokenizer(self):
        from transformers import AutoTokenizer

        return AutoTokenizer.from_pretrained(self.model.path())

    def render(self, messages: list[dict], tools: list[dict] | None = None, *, thinking: bool) -> str:
        return self.tokenizer.apply_chat_template(
            [_template_ready(m) for m in messages],
            tools=tools,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=thinking,
        )

    def complete(
        self,
        prompt: str,
        *,
        max_tokens: int,
        temperature: float,
        top_p: float = 1.0,
        top_k: int = 0,
        stop: list[str] | None = None,
    ) -> Completion:
        started = time.perf_counter()
        response = self._client.post(
            "/completions",
            json={
                "model": self.model.path(),
                "prompt": prompt,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "top_p": top_p,
                "top_k": top_k,
                "stop": stop or [],
            },
        )
        response.raise_for_status()
        body = response.json()
        usage = body.get("usage") or {}
        choice = body["choices"][0]
        return Completion(
            text=choice["text"].strip(),
            reasoning=choice.get("reasoning", "").strip(),
            tool_calls=[_parse_tool_call(c) for c in choice.get("tool_calls", [])],
            prompt_tokens=usage.get("prompt_tokens", 0),
            cached_tokens=(usage.get("prompt_tokens_details") or {}).get("cached_tokens", 0),
            completion_tokens=usage.get("completion_tokens", 0),
            seconds=time.perf_counter() - started,
            finish_reason=choice.get("finish_reason"),
        )


def _template_ready(message: dict) -> dict:
    """Qwen3's template needs string content and dict tool arguments; OpenAI-style messages carry None and JSON text."""
    message = {**message, "content": message.get("content") or ""}
    if message.get("tool_calls"):
        message["tool_calls"] = [
            {
                **call,
                "function": {
                    **call["function"],
                    "arguments": json.loads(call["function"]["arguments"])
                    if isinstance(call["function"]["arguments"], str)
                    else call["function"]["arguments"],
                },
            }
            for call in message["tool_calls"]
        ]
    return message


def _parse_tool_call(call: dict) -> ToolCall:
    function = call["function"]
    arguments = function.get("arguments") or {}
    return ToolCall(function["name"], json.loads(arguments) if isinstance(arguments, str) else arguments)
