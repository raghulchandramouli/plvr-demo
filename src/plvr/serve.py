"""Run mlx_lm's server with two changes for running two model servers side by side on one Mac.

1. Text completions also return the tool calls and reasoning the server parsed out of the text. Stock mlx_lm 0.32.2
   drops both, and the student agents need raw-prompt completions to place the program's text in the think block.
2. Each server wires and caches only its own budget. Stock mlx_lm wires up to the whole recommended working set and
   never caps the buffer cache, so two servers starved each other until Metal ran out of memory.
"""

import argparse
import inspect
import sys

import mlx.core as mx
from mlx_lm import server


def _keep_parsed_parts(generate_response):
    signature = inspect.signature(generate_response)

    def patched(self, *args, **kwargs):
        response = generate_response(self, *args, **kwargs)
        if self.object_type == "text_completion":
            bound = signature.bind(self, *args, **kwargs)
            choice = response["choices"][0]
            choice["tool_calls"] = bound.arguments.get("tool_calls") or []
            choice["reasoning"] = bound.arguments.get("reasoning_text") or ""
        return response

    return patched


def main(argv: list[str]) -> None:
    budget = argparse.ArgumentParser()
    budget.add_argument("--wired-gb", type=float, required=True)
    budget.add_argument("--buffer-cache-gb", type=float, default=1.0)
    limits, rest = budget.parse_known_args(argv)

    def set_budget() -> int:
        mx.set_cache_limit(int(limits.buffer_cache_gb * 2**30))
        return mx.set_wired_limit(int(limits.wired_gb * 2**30))

    server.maybe_set_recommended_wired_limit = set_budget
    server.APIHandler.generate_response = _keep_parsed_parts(server.APIHandler.generate_response)
    sys.argv = ["mlx_lm.server", *rest]
    server.main()


if __name__ == "__main__":
    main(sys.argv[1:])
