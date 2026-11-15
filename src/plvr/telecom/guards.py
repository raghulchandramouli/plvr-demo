"""Deterministic guards that check a student action before it goes out to τ².

Tool-name guard: an action may only call one of the agent's tools. A function that only the user's phone has gets a note
saying so; the student is asked again.

Precondition guard: before a state-changing call, the preconditions the policy states for it are checked against the
records the agent has seen. Which tools change state, and what their preconditions are, are read from the policy and the
tool descriptions once per policy through the primitives below (each with a verbatim verifier). At run time every
precondition goes through the existing "Is the condition met" primitive, so numbers and dates are computed in code.
Nothing here names a τ² tool, fault or policy rule.
"""

import hashlib
import json
import re
import threading
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel
from tau2.data_model.message import AssistantMessage, ToolMessage

from plvr.primitive import Context, Primitive, Runner, verbatim
from plvr.telecom import primitives as P
from plvr.telecom.policy import device_functions, parse_sections, sentence_of

GUARD_VERSION = 3
CACHE_DIR = Path("runs/cache")
MAX_RESAMPLES = 2
_lock = threading.Lock()


# ---------------------------------------------------------------- primitives (compile time)


class Titles(BaseModel):
    titles: list[str]


class Requirement(BaseModel):
    condition: str
    must_hold: bool
    about: Literal["records", "conversation"]


class Requirements(BaseModel):
    requirements: list[Requirement]


GOVERNING_SECTIONS = Primitive(
    name="Sections that govern a function",
    group="checking the policy",
    instruction=(
        "The FUNCTION does what its DESCRIPTION says. List the titles from SECTION_TITLES of the policy sections whose rules say "
        "when the agent may or may not do what this FUNCTION does. The section may describe the action in its own words without "
        "naming the FUNCTION. Return [] if none does."
    ),
    form='{"titles": ["<title>", ...]}',
    output=Titles,
    verify=lambda o, i: next((f"unknown title {t!r}" for t in o.titles if t not in i["section_titles"]), None),
    fallback=lambda i: Titles(titles=[]),
    max_tokens=150,
)

PRECONDITIONS = Primitive(
    name="What must hold before the function is called",
    group="checking the policy",
    instruction=(
        "List what the POLICY requires to be true before the agent does what the FUNCTION does, or what makes the POLICY forbid it. "
        "CONDITION: copy the words that state it, as a statement that is true or false. "
        "MUST_HOLD: true if the call is allowed only when the condition holds; false if the call is forbidden when the condition holds. "
        'ABOUT: "records" if the condition can be decided from stored data (statuses, dates, amounts, quantities); '
        '"conversation" if it depends on what the user said, asked or agreed to. '
        "Leave out steps the agent takes, checks the DESCRIPTION says the function already makes, and anything that does not "
        "concern this FUNCTION. Return an empty list if there is none."
    ),
    form='{"requirements": [{"condition": "<copied words>", "must_hold": true | false, "about": "records" | "conversation"}]}',
    output=Requirements,
    verify=lambda o, i: (
        "no condition is verbatim" if o.requirements and not any(verbatim(r.condition, i["policy"]) for r in o.requirements) else None
    ),
    fallback=lambda i: Requirements(requirements=[]),
    max_tokens=500,
)


# ---------------------------------------------------------------- the compiled plan


@dataclass
class Precondition:
    function: str
    condition: str
    must_hold: bool
    sentence: str


@dataclass
class GuardPlan:
    preamble: str
    state_changing: dict[str, str]
    preconditions: dict[str, list[Precondition]]
    rejected: list[str] = field(default_factory=list)


NORMATIVE = re.compile(r"\b(allowed|only|must|cannot|can not|never|may not|should not|do not|not permitted|required)\b", re.I)


def _permits_or_forbids(sentence: str) -> bool:
    """A precondition has to come from a sentence that grants or forbids something, not one that describes how things work."""
    return NORMATIVE.search(sentence) is not None


def compile_plan(policy: str, tools: list, runner: Runner) -> GuardPlan:
    """Which agent tools change state, and the record-checkable preconditions the policy gives each of them."""
    key = hashlib.sha256(
        (policy + Path(P.__file__).read_text() + Path(__file__).read_text() + str(GUARD_VERSION)).encode()
    ).hexdigest()[:16]
    path = CACHE_DIR / f"telecom-guards-{key}.json"
    with _lock:
        if path.exists():
            data = json.loads(path.read_text())
            data["preconditions"] = {f: [Precondition(**p) for p in ps] for f, ps in data["preconditions"].items()}
            return GuardPlan(**data)
        plan = _compile(policy, tools, runner)
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(plan), indent=1))
        return plan


def _compile(policy: str, tools: list, run: Runner) -> GuardPlan:
    ctx = Context()
    sections = parse_sections(policy)
    main = "\n\n".join(f"{'#' * s.level} {s.title}\n{s.body}" for s in sections if s.part == "main")
    rule_sections = {s.title: f"## {s.title}\n{s.body}" for s in sections if s.part == "main" and s.level >= 2 and s.body}
    preamble = next((s.body for s in sections if s.part == "main" and s.level == 1), "")
    plan = GuardPlan(preamble, {}, {})
    for tool in tools:
        name, description = tool.name, tool.openai_schema["function"]["description"]
        tech = "\n\n".join(f"## {s.title}\n{s.body}" for s in sections if s.part == "tech" and name in s.body)
        change = run(P.CHANGES_STATE, ctx, function=name, description=description, policy=f"{main}\n\n{tech}".strip())
        if not change.changes_state:
            continue
        plan.state_changing[name] = change.evidence or ""
        governing = run(
            GOVERNING_SECTIONS, ctx, function=name, description=description, section_titles=list(rule_sections)
        ).titles
        text = "\n\n".join([*(rule_sections[t] for t in governing), tech]).strip()
        plan.preconditions[name] = []
        if not text:
            plan.rejected.append(f"{name}: no policy section governs it")
            continue
        found = run(PRECONDITIONS, ctx, function=name, description=description, policy=text).requirements
        for r in found:
            if r.about != "records":
                plan.rejected.append(f"{name}: '{r.condition}' depends on the conversation, not enforced")
            elif not verbatim(r.condition, text):
                plan.rejected.append(f"{name}: '{r.condition}' not verbatim in the policy, dropped")
            elif not _permits_or_forbids(sentence := sentence_of(text, r.condition)):
                plan.rejected.append(f"{name}: '{sentence}' describes, does not permit or forbid, dropped")
            else:
                plan.preconditions[name].append(Precondition(name, r.condition, r.must_hold, sentence))
    return plan


# ---------------------------------------------------------------- run time


@dataclass
class Violation:
    kind: Literal["tool_name", "precondition"]
    call: str
    arguments: dict
    reason: str
    note: str
    fallback_text: str


@dataclass
class Verdict:
    violation: Violation | None = None
    unverified: list[dict] = field(default_factory=list)


class Guard:
    """`tool_names` always applies. `plan` and `run` switch the precondition guard on."""

    def __init__(
        self,
        tools: list,
        policy: str,
        plan: GuardPlan | None = None,
        run: Runner | None = None,
        block_unverified: bool = False,
    ):
        self.tool_names = [t.name for t in tools]
        self.phone = device_functions(policy, self.tool_names)
        self.plan, self.run, self.block_unverified = plan, run, block_unverified

    def check(self, calls: list, messages: list) -> Verdict:
        for call in calls:
            if call.name not in self.tool_names:
                return Verdict(self._unknown_tool(call))
        verdict = Verdict()
        if self.plan is None or self.run is None:
            return verdict
        records = self._records(messages)
        for call in calls:
            for pre in self.plan.preconditions.get(call.name, []):
                met = self._met(pre, call, records)
                breach = (pre.must_hold and met.answer == "no") or (not pre.must_hold and met.answer == "yes")
                if met.answer == "cannot tell" and not self.block_unverified:
                    verdict.unverified.append({"call": call.name, "condition": pre.condition})
                elif breach or met.answer == "cannot tell":
                    verdict.violation = self._breach(pre, call, met)
                    return verdict
        return verdict

    def _unknown_tool(self, call) -> Violation:
        if call.name in self.phone:
            note = (
                f"{call.name} is not one of your tools: it runs on the user's phone, so only the user can run it. "
                f"Ask the user to run {call.name}() and tell you what it shows. Your tools are: {', '.join(self.tool_names)}."
            )
            fallback = f"Could you please run {call.name}() on your phone and tell me exactly what it shows?"
        else:
            note = f"{call.name} is not one of your tools. Your tools are: {', '.join(self.tool_names)}."
            fallback = "Could you tell me a bit more about what you need help with?"
        return Violation("tool_name", call.name, dict(call.arguments), "not one of the agent's tools", note, fallback)

    def _records(self, messages: list) -> str:
        asked: dict[str, tuple[str, dict]] = {}
        latest: dict[str, str] = {}
        for m in messages:
            if isinstance(m, AssistantMessage):
                for c in m.tool_calls or []:
                    asked[c.id] = (c.name, c.arguments)
            elif isinstance(m, ToolMessage) and not m.error and m.id in asked:
                name, arguments = asked[m.id]
                call = f"{name}({P.dumps(arguments)})"
                latest.pop(call, None)
                latest[call] = f"{call} -> {m.content or ''}"
        return "\n".join(latest.values())

    def _met(self, pre: Precondition, call, records: str) -> P.Met:
        record = "\n".join(
            [
                self.plan.preamble,
                f"policy: {pre.sentence}",
                records[-9000:],
                f"call about to be made: {call.name}({P.dumps(call.arguments)})",
            ]
        )
        return self.run(P.IS_CONDITION_MET, Context(), condition=pre.condition, record=record)

    def _breach(self, pre: Precondition, call, met: P.Met) -> Violation:
        shown = f"the records show {met.evidence!r}" if met.evidence else f"the records give {met.expression!r}" if met.expression else "the records you have do not show it"
        required = "required" if pre.must_hold else "forbidden"
        reason = f"{required}: {pre.condition} ({met.answer})"
        note = (
            f"Your call {call.name}({P.dumps(call.arguments)}) was not sent. The policy says: \"{pre.sentence}\". "
            f"That is not met yet, because {shown}. Read the records that settle it, or do what the policy requires first."
        )
        fallback = (
            f"I can't do that yet. Our policy says: \"{pre.sentence}\" "
            "I don't have a record showing that is the case."
        )
        return Violation("precondition", call.name, dict(call.arguments), reason, note, fallback)


def build_guard(mode: str, policy: str, tools: list, runner: Any) -> Guard | None:
    """`off` returns no guard, `tools` the tool-name guard alone, `full` adds the precondition guard."""
    if mode == "off":
        return None
    if mode == "tools":
        return Guard(tools, policy)
    return Guard(tools, policy, compile_plan(policy, tools, Runner(runner)), Runner(runner))
