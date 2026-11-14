"""Run mlx_lm's server with text completions that also return the tool calls and reasoning it parsed out of the text.

Stock mlx_lm 0.32.2 parses Qwen3 tool calls and the think block out of a text completion and then drops both.
The student agents need raw-prompt completions (to place the program's text inside the think block) and the tool calls.
"""

import inspect
import sys

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
    server.APIHandler.generate_response = _keep_parsed_parts(server.APIHandler.generate_response)
    sys.argv = ["mlx_lm.server", *argv]
    server.main()


if __name__ == "__main__":
    main(sys.argv[1:])
