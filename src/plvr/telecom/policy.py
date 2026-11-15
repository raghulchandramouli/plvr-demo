"""Read the troubleshooting checklist out of the policy text, through primitives, once per policy.

Code here only parses Markdown structure (headings, bullets, `name(` and **name** spans). Which sections are problem
areas, which rules they hold, what each rule checks and fixes, and which procedure a rule defers to are decided by
primitives running on the frozen runner model.
"""

import hashlib
import json
import re
import threading
from dataclasses import asdict, dataclass, field
from pathlib import Path

from plvr.primitive import Context, Runner, verbatim
from plvr.telecom import primitives as P

COMPILE_VERSION = 4
CACHE_DIR = Path("runs/cache")
_lock = threading.Lock()


@dataclass(frozen=True)
class Section:
    id: int
    part: str
    level: int
    title: str
    parent: int | None
    body: str


@dataclass
class Rule:
    area: str
    section: str
    condition: str
    check: str | None
    fix: str | None
    fix_arguments: dict
    escalate: bool
    procedures: list[str] = field(default_factory=list)
    then: list[str] = field(default_factory=list)
    reader: str | None = None
    policy: str = ""
    fixed_state: str | None = None


@dataclass
class Area:
    title: str
    diagnostic: str
    criterion: str
    before: list[str]
    rules: list[Rule]


@dataclass
class Procedure:
    """A main-policy section that tells the agent to make a change, read as what to do around that change."""

    title: str
    effects: dict[str, str]  # agent function -> the sentence that tells the agent to make its change, in written order
    reads: str | None = None  # read-only function for the records the section says to check before the change
    check: str | None = None  # the sentence that says what the agent checks once the change is made
    check_reads: str | None = None  # read-only function whose result shows whether `check` holds
    after: str | None = None  # the sentence that says what the user has to do after the change
    requires: list[str] = field(default_factory=list)


@dataclass
class Compiled:
    areas: dict[str, Area]
    procedures: dict[str, str]
    device_functions: list[str]
    agent_functions: list[str]
    descriptions: dict[str, str]
    rejected: list[str]
    preamble: str
    plans: dict[str, Procedure] = field(default_factory=dict)
    state_changing: list[str] = field(default_factory=list)

    def checklist(self, area_title: str) -> list[Rule]:
        """Rules of the area's prerequisites first, then its own, in written order, without repeats."""
        order: list[str] = []
        entered: set[str] = set()

        def visit(title: str) -> None:
            entered.add(title)
            for before in self.areas[title].before:
                if before in self.areas and before not in entered:
                    visit(before)
            order.append(title)

        visit(area_title)
        seen, rules = set(), []
        for title in order:
            for rule in self.areas[title].rules:
                key = (rule.check, rule.fix, tuple(rule.procedures), rule.escalate, rule.condition.lower())
                if key not in seen:
                    seen.add(key)
                    rules.append(rule)
        return rules


def parse_sections(policy: str) -> list[Section]:
    sections: list[Section] = []
    stack: list[Section] = []
    part, body = "main", []

    def close() -> None:
        if sections:
            s = sections[-1]
            sections[-1] = Section(s.id, s.part, s.level, s.title, s.parent, "\n".join(body).strip())

    for line in policy.splitlines():
        if line.strip().startswith("<tech_support_policy>"):
            part = "tech"
            continue
        if re.match(r"^\s*</?\w+_policy>\s*$", line):
            continue
        heading = re.match(r"^(#{1,6})\s+(.*)$", line)
        if not heading:
            body.append(line)
            continue
        close()
        body = []
        level = len(heading.group(1))
        while stack and (stack[-1].level >= level or stack[-1].part != part):
            stack.pop()
        section = Section(len(sections), part, level, heading.group(2).strip(), stack[-1].id if stack else None, "")
        sections.append(section)
        stack.append(section)
    close()
    return sections


def descendants(sections: list[Section], root: int) -> list[Section]:
    out, frontier = [], {root}
    for s in sections:
        if s.parent in frontier:
            out.append(s)
            frontier.add(s.id)
    return out


def full_text(sections: list[Section], root: int) -> str:
    return "\n\n".join(f"{'#' * s.level} {s.title}\n{s.body}" for s in [sections[root], *descendants(sections, root)])


def device_functions(policy: str, agent_functions: list[str]) -> dict[str, str]:
    """Names the policy writes as `name(` or **name**, with the description written next to a **name** entry."""
    names = set(re.findall(r"\b([a-z][a-z]*(?:_[a-z]+)+)\(", policy)) | set(re.findall(r"\*\*([a-z][a-z]*(?:_[a-z]+)+)\*\*", policy))
    found = {n: "" for n in sorted(names - set(agent_functions))}
    for name, description in re.findall(r"\*\*([a-z][a-z_]+)\*\*\s*-\s*(.+)", policy):
        if name in found:
            found[name] = description.strip()
    return found


def compile_policy(policy: str, tools: list, runner: Runner) -> Compiled:
    key = hashlib.sha256(
        (policy + Path(P.__file__).read_text() + Path(__file__).read_text() + str(COMPILE_VERSION)).encode()
    ).hexdigest()[:16]
    path = CACHE_DIR / f"telecom-policy-{key}.json"
    with _lock:
        if path.exists():
            return _load(json.loads(path.read_text()))
        compiled = _compile(policy, tools, runner)
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(compiled), indent=1))
        return compiled


def _load(data: dict) -> Compiled:
    data["areas"] = {
        t: Area(**{**a, "rules": [Rule(**r) for r in a["rules"]]}) for t, a in data["areas"].items()
    }
    data["plans"] = {t: Procedure(**p) for t, p in data.get("plans", {}).items()}
    return Compiled(**data)


def _compile(policy: str, tools: list, run: Runner) -> Compiled:
    ctx = Context()
    sections = parse_sections(policy)
    agent = {t.name: t.openai_schema["function"]["description"] for t in tools}
    device = device_functions(policy, list(agent))
    descriptions = {**agent, **device}
    known = sorted(descriptions)
    rejected: list[str] = []

    procedures = {s.title: s.body for s in sections if s.part == "main" and s.level >= 2 and s.body}
    preamble = next((s.body for s in sections if s.part == "main" and s.level == 1), "")
    tops = [s for s in sections if s.part == "tech" and s.level == 1]

    areas: dict[str, Area] = {}
    for top in tops:
        subs = [s for s in descendants(sections, top.id) if s.body]
        governed: dict[int, list[str]] = {}
        for s in subs:
            governed[s.id] = run(
                P.FUNCTIONS_GOVERNED, ctx, section=f"## {s.title}\n{s.body}", known_functions=known
            ).functions
        rules: list[Rule] = []
        for s in subs:
            rules.extend(_rules_of(run, ctx, top, s, governed[s.id], agent, procedures, rejected))
        if not rules:
            continue
        area_functions = sorted({f for fs in governed.values() for f in fs})
        diagnostic = run(
            P.PICK_FUNCTION,
            ctx,
            goal="Read-only check that shows whether the user still has the problem this AREA troubleshoots.",
            area=full_text(sections, top.id),
            functions=area_functions,
        ).function
        criterion = next((f"## {s.title}\n{s.body}" for s in subs if diagnostic in s.body), "")
        areas[top.title] = Area(top.title, diagnostic, criterion, [], rules)

    titles = list(areas)
    for top in tops:
        if top.title in areas:
            others = [t for t in titles if t != top.title]
            areas[top.title].before = run(
                P.ORDER_REQUESTS, ctx, area=full_text(sections, top.id)[:6000], area_titles=others
            ).before

    reads = descriptions
    for area in areas.values():
        for rule in area.rules:
            if rule.check is None and not rule.escalate:
                rule.check = _pick(run, ctx, f"Read, without changing anything, whether this holds: {rule.condition}", reads, rule.fix)
            if rule.fix in device and rule.fix not in (rule.check or ""):
                rule.reader = rule.check or _pick(
                    run, ctx, f"Read the current state of the setting that {rule.fix} changes.", reads, rule.fix
                )
    compiled = Compiled(areas, procedures, sorted(device), sorted(agent), descriptions, rejected, preamble)
    _add_procedure_plans(run, ctx, compiled, sections)
    return compiled


def _add_procedure_plans(run: Runner, ctx: Context, compiled: Compiled, sections: list[Section]) -> None:
    """Runs after the checklist is compiled, so the functions it adds do not change what the checklist was read with."""
    named = named_functions("\n".join(compiled.procedures.values()), compiled.descriptions)
    compiled.device_functions = sorted({*compiled.device_functions, *named})
    compiled.descriptions.update(dict.fromkeys(named, ""))
    agent = {n: compiled.descriptions[n] for n in compiled.agent_functions}
    writes = changing_functions(run, ctx, agent, sections)
    compiled.state_changing = sorted(writes)
    reads = {n: d for n, d in agent.items() if n not in writes}
    compiled.plans = plan_procedures(run, ctx, compiled.procedures, writes, reads, list(compiled.descriptions), compiled.rejected)


def named_functions(text: str, known: dict[str, str]) -> list[str]:
    """Names a procedure writes in snake_case that are not among the known functions: the agent has no such tool."""
    return sorted(set(re.findall(r"\b[a-z]+(?:_[a-z]+)+\b", text)) - set(known))


def changing_functions(run: Runner, ctx: Context, agent: dict[str, str], sections: list[Section]) -> dict[str, str]:
    main = "\n\n".join(f"{'#' * s.level} {s.title}\n{s.body}" for s in sections if s.part == "main")
    changing = {}
    for name, description in agent.items():
        tech = "\n\n".join(f"## {s.title}\n{s.body}" for s in sections if s.part == "tech" and name in s.body)
        if run(P.CHANGES_STATE, ctx, function=name, description=description, policy=f"{main}\n\n{tech}".strip()).changes_state:
            changing[name] = description
    return changing


# ---------------------------------------------------------------- what a quoted policy sentence may be used for

ADDRESSES_THE_AGENT = re.compile(r"\b(you|your|should|must|need to|allowed|always)\b", re.I)
FORBIDS = re.compile(r"\b(not allowed|not permitted|may not|must not|should not|cannot|can not|never|do not|don't)\b", re.I)
ORDERS = re.compile(r"\b(after|before|once|until|first)\b", re.I)
CHECKS = re.compile(r"\b(check|verify|make sure|ensure)\b", re.I)
_LIST_ITEM = re.compile(r"\s*(?:[-*+]|\d+[.)])\s")


def _segments(source: str) -> list[str]:
    return [segment for segment in re.split(r"(?<=[.!?])\s+|\n+", source) if segment.strip()]


def sentence_of(source: str, quote: str) -> str:
    """The sentence or list item of the source that holds the quote, or the quote itself when none does."""
    return next((segment.strip(" -*") for segment in _segments(source) if verbatim(quote, segment)), quote)


def _line_of(source: str, text: str) -> int:
    return next((n for n, line in enumerate(source.splitlines()) if verbatim(text, line)), -1)


def lead_in(source: str, sentence: str) -> str:
    """For a list item, the lines that introduce it: each line above it with less indentation, up to the first non-item line."""
    lines = source.splitlines()
    at = next((n for n, line in enumerate(lines) if _LIST_ITEM.match(line) and verbatim(sentence, line)), None)
    if at is None:
        return ""
    indent = lambda line: len(line) - len(line.lstrip())
    depth, above = indent(lines[at]), []
    for line in reversed(lines[:at]):
        if not line.strip() or (_LIST_ITEM.match(line) and indent(line) >= depth):
            continue
        above.append(line.strip())
        if not _LIST_ITEM.match(line):
            break
        depth = indent(line)
    return " ".join(above)


def instruction_problem(source: str, line: str, other_functions: list[str]) -> str | None:
    """Why this line cannot be the policy telling the agent to make one function's change, or None if it can."""
    if _line_of(source, line) < 0:
        return "is not in the section"
    if FORBIDS.search(line):
        return "forbids, does not instruct"
    if not ADDRESSES_THE_AGENT.search(f"{line} {lead_in(source, line)}"):
        return "describes, does not instruct"
    if named := [f for f in other_functions if re.search(rf"\b{f}\b", line)]:
        return f"names another function, {named[0]}"
    return None


def instructing_lines(sections: dict[str, str], other_functions: list[str]) -> list[tuple[str, str]]:
    """Every (title, line) of the sections that can tell the agent to make a change: the only lines the runner may pick."""
    return [
        (title, text)
        for title, body in sections.items()
        for line in body.splitlines()
        if (text := _LIST_ITEM.sub("", line, count=1).strip()) and instruction_problem(body, text, other_functions) is None
    ]


def _shares_a_word(title: str, sentence: str) -> bool:
    stems = lambda text: {word[:4] for word in re.findall(r"[a-z]{4,}", text.lower())}
    return bool(stems(title) & stems(sentence))


def plan_procedures(
    run: Runner,
    ctx: Context,
    sections: dict[str, str],
    writes: dict[str, str],
    reads: dict[str, str],
    functions: list[str],
    rejected: list[str],
) -> dict[str, Procedure]:
    """A main-policy section becomes a procedure when one of its sentences tells the agent to make a changing function's change.

    The runner model picks among the lines that instruct, and must read its pick back as the same function.
    """
    effects: dict[str, dict[str, str]] = {}
    for function, description in writes.items():
        lines = instructing_lines(sections, [f for f in functions if f != function])
        numbered = [f"{n}. [{title}] {line}" for n, (title, line) in enumerate(lines, 1)]
        number = run(P.WHICH_LINE_INSTRUCTS, ctx, function=function, description=description, lines=numbered).number
        if number is None:
            continue
        title, line = lines[number - 1]
        back = _pick(run, ctx, f"Do what this policy sentence tells the agent to do: {line}", writes, None)
        if back == function:
            effects.setdefault(title, {})[function] = line
        else:
            rejected.append(f"{title}: {function} from {line!r} dropped (reads back as {back}, not {function})")

    plans: dict[str, Procedure] = {}
    for title, body in sections.items():
        if title not in effects:
            continue
        text = f"## {title}\n{body}"
        in_order = sorted(effects[title].items(), key=lambda item: _line_of(body, item[1]))
        plan = plans[title] = Procedure(title, dict(in_order))
        plan.reads = _pick(
            run, ctx, f"Read the records this policy section tells the agent to check before it makes the change.\n{text}", reads, None
        )
        asked = run(P.COPY_SENTENCE, ctx, section=text, question=CHECK_QUESTION).quote
        if asked:
            check = sentence_of(body, asked)
            problem = instruction_problem(body, check, []) or (
                "names no check"
                if not CHECKS.search(check)
                else "comes before the change"
                if _line_of(body, check) <= _line_of(body, in_order[-1][1])
                else None
            )
            if problem:
                rejected.append(f"{title}: completion check {check!r} dropped ({problem})")
            else:
                plan.check = check
                goal = f"Read the record this policy sentence tells the agent to check: {check}"
                plan.check_reads = _pick(run, ctx, goal, reads, None) or plan.reads
        asked = run(P.COPY_SENTENCE, ctx, section=text, question=AFTER_QUESTION).quote
        if asked:
            after = sentence_of(body, asked)
            if _line_of(body, after) < 0 or FORBIDS.search(after) or not ORDERS.search(after) or not re.search(r"\buser\b", after, re.I):
                rejected.append(f"{title}: follow-up {after!r} dropped (does not say what the user does after the change)")
            else:
                plan.after = after

    summaries = {title: sections[title][:400] for title in plans}
    for title, plan in plans.items():
        others = {t: body for t, body in summaries.items() if t != title}
        if not others:
            continue
        for first in run(P.WHAT_COMES_FIRST, ctx, section=f"## {title}\n{sections[title]}", procedures=others).first:
            sentence = sentence_of(sections[title], first.quote)
            if ORDERS.search(sentence) and _shares_a_word(first.title, sentence):
                plan.requires.append(first.title)
            else:
                rejected.append(f"{title}: prerequisite {first.title!r} from {sentence!r} dropped (the sentence does not put it first)")
    return plans


CHECK_QUESTION = "What does the SECTION tell the agent to check after the change is made, before it counts the change as complete?"
AFTER_QUESTION = "What does the SECTION say the user has to do after the agent has made the change?"


def _pick(run: Runner, ctx: Context, goal: str, functions: dict[str, str], exclude: str | None) -> str | None:
    choice = run(P.PICK_FUNCTION, ctx, goal=goal, functions={f: d for f, d in functions.items() if f != exclude}).function
    return None if choice == "none" else choice


def _rules_of(run, ctx, top, s, governed, agent, procedures, rejected) -> list[Rule]:
    text = f"## {s.title}\n{s.body}"
    if not governed and not re.search(r"\b(if|when|after)\b", s.body, re.I):
        return []
    parts = []
    for part in run(P.COPY_RULE_CONDITION, ctx, section=text, functions=governed).rules:
        if part.check and (part.check == part.fix or part.check not in governed):
            part = part.model_copy(update={"check": None})
        part = part.model_copy(update={"fix_arguments": P.quoted_arguments(part.fix_arguments, text)})
        if part.fixed_state and not verbatim(part.fixed_state, text):
            part = part.model_copy(update={"fixed_state": None})
        if problem := P.rule_problem(part, text, governed):
            rejected.append(f"{s.title}: dropped rule ({problem})")
        else:
            parts.append(part)
    if not parts:
        rejected.append(f"{s.title}: no rules")
    defers_to_policy = bool(re.search(r"\brefer\b[^.]*\bpolicy\b", s.body, re.I))
    rules: list[Rule] = []
    for part in parts:
        if rules and part.check is None and part.fix and not part.escalate and run(
            P.IS_RULE_ABOUT_ACTION, ctx, rule=part.condition, action=rules[-1].fix or ""
        ).about_this_action:
            rules[-1].then.append(part.fix)
            continue
        rule = Rule(
            top.title, s.title, part.condition, part.check, part.fix, part.fix_arguments, part.escalate,
            policy=text, fixed_state=part.fixed_state,
        )
        if defers_to_policy and not rule.escalate:
            rule.procedures = run(P.PLAN_STEPS, ctx, rule=text, procedure_titles=list(procedures)).procedures
        if rule.fix is None and not rule.escalate:
            if not rule.procedures:
                rejected.append(f"{s.title}: '{part.condition}' names no fix and defers to no procedure")
                continue
        rules.append(rule)
    return rules


def main() -> None:
    """Compile the τ² Telecom policy and print each area's checklist, so the compiled rules can be audited."""
    from tau2.runner import build_environment

    from plvr.llm import Server
    from plvr.models import MODELS, PRIMITIVE_RUNNER

    env = build_environment("telecom")
    run = Runner(Server(MODELS[PRIMITIVE_RUNNER]))
    compiled = compile_policy(env.get_policy(), env.get_tools(), run)
    for area in compiled.areas.values():
        print(f"\n== {area.title}\n   diagnostic={area.diagnostic} before={area.before}")
        for i, r in enumerate(compiled.checklist(area.title)):
            print(
                f"   {i:2} [{r.section}] if {r.condition!r}: check={r.check} fix={r.fix}{r.fix_arguments or ''}"
                f" then={r.then} reader={r.reader} escalate={r.escalate} procedures={r.procedures} fixed_state={r.fixed_state!r}"
            )
    print(f"\n== Procedures\n   state-changing functions: {compiled.state_changing}")
    for plan in compiled.plans.values():
        print(f"   [{plan.title}] requires={plan.requires} reads={plan.reads}")
        for function, sentence in plan.effects.items():
            print(f"      effect {function}: {sentence!r}")
        print(f"      check ({plan.check_reads}): {plan.check!r}")
        print(f"      after: {plan.after!r}")
    print("\nrejected:", *compiled.rejected, sep="\n  ")
    rejected_calls = [c for c in run.calls if not c.accepted]
    print(f"\n{len(run.calls)} primitive calls, {len(rejected_calls)} rejected by verifiers")
    for c in rejected_calls:
        print("  ", c.primitive, "|", c.rejection)


if __name__ == "__main__":
    main()
