"""Does verifying a primitive by sampling it more than once beat a single greedy call?

Three strategies on the labelled probe cases, each scored on the program's fix decision:
  greedy      one call at temperature 0, the frozen program's behaviour;
  resample    up to 3 samples at temperature 0.6, keep the first one its verifier accepts;
  vote        5 samples at temperature 0.6, majority answer among those the verifier accepts.
Run against a live Qwen3-8B server: `python -I scripts/probe_self_consistency.py`.
"""

import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from probe_conditions import CASES, decide, load, rule_for  # noqa: E402

from plvr.primitive import SYSTEM, Context, Primitive, _first_json_object, _render  # noqa: E402

SAMPLING = {"temperature": 0.6, "top_p": 0.95, "top_k": 20}


class Sampler:
    def __init__(self, server):
        self.server = server
        self.calls = 0

    def one(self, primitive: Primitive, inputs: dict, greedy: bool) -> str | None:
        """One call; the answer if the primitive's own verifier accepts it, else None."""
        self.calls += 1
        prompt = self.server.render(
            [{"role": "system", "content": SYSTEM}, {"role": "user", "content": _render(primitive, Context(), inputs)}],
            thinking=False,
        )
        out = self.server.complete(prompt, max_tokens=primitive.max_tokens, **({"temperature": 0.0} if greedy else SAMPLING))
        try:
            value = primitive.output.model_validate(_first_json_object(out.text))
        except Exception:
            return None
        return value.answer if primitive.verify(value, inputs) is None else None

    def greedy(self, primitive: Primitive, **inputs) -> str:
        return self.one(primitive, inputs, greedy=True) or "cannot tell"

    def resample(self, primitive: Primitive, **inputs) -> str:
        for _ in range(3):
            if (answer := self.one(primitive, inputs, greedy=False)) is not None:
                return answer
        return "cannot tell"

    def vote(self, primitive: Primitive, **inputs) -> str:
        answers = Counter(a for a in (self.one(primitive, inputs, greedy=False) for _ in range(5)) if a is not None)
        if not answers:
            return "cannot tell"
        (top, n), *rest = answers.most_common()
        return "cannot tell" if rest and rest[0][1] == n else top


def main() -> None:
    compiled, run = load()
    for name in ("greedy", "resample", "vote"):
        sampler = Sampler(run.server)
        ask = getattr(sampler, name)
        correct, harmful = 0, 0
        for section, n, reading, want in CASES:
            got = decide(compiled, ask, rule_for(compiled, section, n), reading)
            correct += got == want
            harmful += want == "yes" and got != "yes"
        print(f"{name:9} {correct}/{len(CASES)} correct, {harmful} missed fixes, {sampler.calls} model calls", flush=True)


if __name__ == "__main__":
    main()
