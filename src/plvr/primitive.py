"""A primitive: one call to the frozen runner model, scoped to one operation, with a JSON output, a verifier and a fallback.

Paper §3.2: "an output that violates its schema, quotes text not verbatim in its source, cites a fact that does not
exist, or gives an expression that does not evaluate is rejected and the fallback is used."
"""

import json
import re
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Generic, TypeVar

from pydantic import BaseModel, ValidationError

from plvr.llm import Server

Out = TypeVar("Out", bound=BaseModel)

SYSTEM = (
    "You perform exactly one step of a customer-support procedure. "
    "Read the inputs, do only what the step asks, and answer with a single JSON object in the requested form. "
    "When you quote, copy the text character for character from the input named."
)
MAX_INPUT_CHARS = 12000


@dataclass(frozen=True)
class Context:
    """The shared context every primitive sees (paper §3.2)."""

    facts: str = ""
    latest_user_message: str = ""
    previous_agent_message: str = ""


@dataclass(frozen=True)
class Primitive(Generic[Out]):
    name: str
    group: str
    instruction: str
    form: str
    output: type[Out]
    verify: Callable[[Out, dict[str, Any]], str | None]
    fallback: Callable[[dict[str, Any]], Out]
    max_tokens: int = 400


@dataclass
class CallLog:
    primitive: str
    accepted: bool
    rejection: str | None
    output: dict
    prompt_tokens: int
    cached_tokens: int
    completion_tokens: int
    seconds: float


@dataclass
class Runner:
    server: Server
    calls: list[CallLog] = field(default_factory=list)

    def __call__(self, primitive: Primitive[Out], context: Context, **inputs: Any) -> Out:
        prompt = self.server.render(
            [{"role": "system", "content": SYSTEM}, {"role": "user", "content": _render(primitive, context, inputs)}],
            thinking=False,
        )
        out = self.server.complete(prompt, max_tokens=primitive.max_tokens, temperature=0.0)
        rejection, value = None, None
        try:
            value = primitive.output.model_validate(_first_json_object(out.text))
            rejection = primitive.verify(value, inputs)
        except (ValueError, ValidationError) as error:
            rejection = f"schema: {str(error)[:200]}"
        if rejection is not None:
            value = primitive.fallback(inputs)
        self.calls.append(
            CallLog(
                primitive=primitive.name,
                accepted=rejection is None,
                rejection=rejection,
                output=value.model_dump(),
                prompt_tokens=out.prompt_tokens,
                cached_tokens=out.cached_tokens,
                completion_tokens=out.completion_tokens,
                seconds=out.seconds,
            )
        )
        return value

    def log_dict(self) -> list[dict]:
        return [asdict(c) for c in self.calls]


def _render(primitive: Primitive, context: Context, inputs: dict[str, Any]) -> str:
    parts = [f"STEP: {primitive.name}", primitive.instruction, ""]
    for label, value in (
        ("FACTS GATHERED SO FAR", context.facts),
        ("LATEST USER MESSAGE", context.latest_user_message),
        ("AGENT'S PREVIOUS MESSAGE", context.previous_agent_message),
    ):
        if value:
            parts.append(f"{label}:\n{value}\n")
    for name, value in inputs.items():
        text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, indent=1)
        parts.append(f"{name.upper()}:\n{text[:MAX_INPUT_CHARS]}\n")
    parts.append(f"Answer with one JSON object of this form and nothing else:\n{primitive.form}")
    return "\n".join(parts)


def _first_json_object(text: str) -> Any:
    text = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.M).strip()
    start = text.find("{")
    if start < 0:
        raise ValueError("no JSON object in output")
    value, _ = json.JSONDecoder().raw_decode(text[start:])
    return value


def verbatim(quote: str | None, source: str) -> bool:
    """True when the quote appears in the source, ignoring case, whitespace runs, quote and emphasis marks, and end punctuation."""
    if not quote:
        return False
    squash = lambda s: re.sub(r"\s+", " ", re.sub(r"[*`\"'“”‘’]", "", s)).strip().lower()
    return squash(quote).rstrip(".,;:!? ") in squash(source)
