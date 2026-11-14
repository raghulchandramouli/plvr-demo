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

from plvr.primitive import Context, Runner
from plvr.telecom import primitives as P

COMPILE_VERSION = 3
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


@dataclass
class Area:
    title: str
    diagnostic: str
    criterion: str
    before: list[str]
    rules: list[Rule]


@dataclass
class Compiled:
    areas: dict[str, Area]
    procedures: dict[str, str]
    device_functions: list[str]
    agent_functions: list[str]
    descriptions: dict[str, str]
    rejected: list[str]
    preamble: str

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
    return Compiled(areas, procedures, sorted(device), sorted(agent), descriptions, rejected, preamble)


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
        rule = Rule(top.title, s.title, part.condition, part.check, part.fix, part.fix_arguments, part.escalate, policy=text)
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
                f" then={r.then} reader={r.reader} escalate={r.escalate} procedures={r.procedures}"
            )
    print("\nrejected:", *compiled.rejected, sep="\n  ")
    rejected_calls = [c for c in run.calls if not c.accepted]
    print(f"\n{len(run.calls)} primitive calls, {len(rejected_calls)} rejected by verifiers")
    for c in rejected_calls:
        print("  ", c.primitive, "|", c.rejection)


if __name__ == "__main__":
    main()
