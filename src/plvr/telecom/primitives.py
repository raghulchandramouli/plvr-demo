"""The primitives the telecom block uses, named and grouped as in the paper's Appendix D.1.

Each prompt names no τ² tool, fault, or fix. Everything domain-specific arrives through the inputs at run time:
the policy text, the tool list, the conversation, and the readings gathered so far.
"""

import ast
import json
import operator
import re
from typing import Any, Literal

from pydantic import BaseModel, field_validator

from plvr.primitive import Primitive, verbatim

# ---------------------------------------------------------------- outputs


class Functions(BaseModel):
    functions: list[str]


class RuleParts(BaseModel):
    condition: str
    check: str | None = None
    fix: str | None = None
    fix_arguments: dict[str, Any] = {}
    escalate: bool = False
    fixed_state: str | None = None

    @field_validator("check", "fix", "fixed_state", mode="before")
    @classmethod
    def _null_word(cls, value: Any) -> Any:
        return None if isinstance(value, str) and value.strip().lower() in ("", "null", "none") else value


class SectionRules(BaseModel):
    rules: list[RuleParts]


class AboutAction(BaseModel):
    about_this_action: bool


class Picked(BaseModel):
    function: str
    alternates: list[str] = []


class Procedures(BaseModel):
    procedures: list[str]


class Before(BaseModel):
    before: list[str]


class Requests(BaseModel):
    requests: list[str]


class Area(BaseModel):
    area: str


class Value(BaseModel):
    value: str | None
    quote: str | None = None


class Report(BaseModel):
    function: str
    result: str | None = None


class DeviceReply(BaseModel):
    reports: list[Report]


Tri = Literal["yes", "no", "cannot tell"]


class Shows(BaseModel):
    answer: Tri
    evidence: str | None = None


class Compare(BaseModel):
    """Two short copied states and whether they match; the yes/no answer is derived, not asked for."""

    state_rule_names: str | None = None
    state_in_reading: str | None = None
    same: bool | None = None

    @property
    def answer(self) -> Tri:
        if not self.state_in_reading:
            return "cannot tell"
        if self.same or _names_one_token_of(self.state_rule_names, self.state_in_reading):
            return "yes"
        return "cannot tell" if self.same is None else "no"


_TOKEN = re.compile(r"[a-z0-9]+")
_NEGATION = {"not", "no", "never", "nt", "off"}


def _names_one_token_of(rule_state: str | None, reading_state: str) -> bool:
    """A one-word state the rule names (\"2G\", \"Locked\") matches a reading that contains that word and no negation."""
    rule_tokens = _TOKEN.findall((rule_state or "").lower())
    reading_tokens = _TOKEN.findall(reading_state.lower().replace("n't", " nt"))
    return len(rule_tokens) == 1 and rule_tokens[0] in reading_tokens and not _NEGATION & set(reading_tokens)


class Met(BaseModel):
    expression: str | None = None
    answer: Tri
    evidence: str | None = None


class Fixed(BaseModel):
    fixed: Tri
    evidence: str | None = None


class Arguments(BaseModel):
    arguments: dict[str, Any]


class Price(BaseModel):
    quantity: float | None
    unit_price: float | None
    quantity_quote: str | None = None
    unit_price_quote: str | None = None


class Confirmation(BaseModel):
    answer: Literal["yes", "no", "unclear"]


class Steps(BaseModel):
    steps: list[str]


class Changes(BaseModel):
    evidence: str | None = None
    changes: bool


class Prerequisite(BaseModel):
    title: str
    quote: str | None = None


class Prerequisites(BaseModel):
    first: list[Prerequisite]


# ---------------------------------------------------------------- verifier helpers


def _all_in(names: list[str], allowed: list[str] | set[str]) -> str | None:
    bad = [n for n in names if n not in allowed]
    return f"unknown functions {bad}" if bad else None


def _evidence(answer: str, evidence: str | None, source: str) -> str | None:
    if answer == "cannot tell":
        return None
    return None if verbatim(evidence, source) else "evidence not verbatim in source"


def _flat_values(value: Any) -> list[Any]:
    if isinstance(value, dict):
        return [v for x in value.values() for v in _flat_values(x)]
    if isinstance(value, list):
        return [v for x in value for v in _flat_values(x)]
    return [value]


_COMPARE = {ast.Gt: operator.gt, ast.GtE: operator.ge, ast.Lt: operator.lt, ast.LtE: operator.le, ast.Eq: operator.eq, ast.NotEq: operator.ne}
_ARITH = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv}


_ISO_DATE = re.compile(r"\d{4}-\d{2}-\d{2}")


def evaluate(expression: str) -> float | bool:
    """Evaluate arithmetic and comparisons over number and 'YYYY-MM-DD' literals only; anything else raises ValueError."""

    def ev(node: ast.AST) -> Any:
        if isinstance(node, ast.Expression):
            return ev(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return node.value
        if isinstance(node, ast.Constant) and isinstance(node.value, str) and _ISO_DATE.fullmatch(node.value):
            return node.value
        if isinstance(node, ast.BinOp) and type(node.op) in _ARITH:
            return _ARITH[type(node.op)](ev(node.left), ev(node.right))
        if isinstance(node, ast.Compare) and len(node.ops) == 1 and type(node.ops[0]) in _COMPARE:
            return _COMPARE[type(node.ops[0])](ev(node.left), ev(node.comparators[0]))
        if isinstance(node, ast.BoolOp):
            values = [ev(v) for v in node.values]
            return all(values) if isinstance(node.op, ast.And) else any(values)
        raise ValueError(f"not allowed: {ast.dump(node)[:80]}")

    return ev(ast.parse(expression, mode="eval"))


_NUMBER = re.compile(r"\d+(?:\.\d+)?")


def _numbers_from(expression: str, source: str) -> str | None:
    available = {float(n) for n in _NUMBER.findall(source)}
    for number in _NUMBER.findall(expression):
        if float(number) not in available and float(number) not in (0, 1):
            return f"number {number} not in source"
    return None


# ---------------------------------------------------------------- checking the policy (compile time)

FUNCTIONS_GOVERNED = Primitive(
    name="Functions a policy section governs",
    group="checking the policy",
    instruction=(
        "List the functions that this policy SECTION tells the agent to use, or to have the user use. "
        "Only list names from KNOWN_FUNCTIONS that are written in the SECTION."
    ),
    form='{"functions": ["<name>", ...]}',
    output=Functions,
    verify=lambda o, i: _all_in(o.functions, i["known_functions"])
    or next((f"{f} not in section" for f in o.functions if f not in i["section"]), None),
    fallback=lambda i: Functions(functions=[]),
)

def rule_problem(rule: RuleParts, section: str, functions: list[str]) -> str | None:
    """A rule needs a verbatim condition and a real fix. An unknown check is cleared at compile time, not rejected."""
    if not verbatim(rule.condition, section):
        return f"condition {rule.condition!r} not verbatim in section"
    return _all_in([rule.fix] if rule.fix else [], functions)


def quoted_arguments(arguments: dict[str, Any], section: str) -> dict[str, Any]:
    """Keep only argument values the section writes in quotes; anything else is not a value the rule gives."""
    return {k: v for k, v in arguments.items() if isinstance(v, str) and re.search(rf"[\"']{re.escape(v)}[\"']", section)}


def _verify_rules(o: SectionRules, i: dict) -> str | None:
    """Reject only when no rule survives; compile drops the individual rules that fail `rule_problem`."""
    problems = [rule_problem(r, i["section"], i["functions"]) for r in o.rules]
    return problems[0] if problems and all(problems) else None


COPY_RULE_CONDITION = Primitive(
    name="Copy the rule's condition",
    group="troubleshooting conditions",
    instruction=(
        "List every rule in this policy SECTION of the form \"if <condition>, do <action>\", in the order written. "
        "A sentence of the form \"after <action>, do <action>\" is a rule of its own: list it with the earlier action as its condition. "
        "A rule must tell the agent what to do. Skip sentences that only list possible causes or describe what a check returns. "
        "When the section tells the agent to check something and refers to the general policy for what to do if it is not as it "
        "should be, that is a rule too: copy what to check as its CONDITION and leave FIX null. "
        "For each rule: "
        "CONDITION: copy the words that state when it applies. "
        "CHECK: the read-only function from FUNCTIONS whose result shows the condition, or null if the rule names none. "
        "It is never the same function as the FIX. "
        "FIX: the function from FUNCTIONS the rule says to use, or have the user use, when the condition holds, or null if it names none. "
        "FIX_ARGUMENTS: argument values the rule writes for the fix, copied exactly, or {}. "
        "ESCALATE: true only if the rule says to escalate or transfer the user. "
        "FIXED_STATE: copy the words that say what state the fix leaves the setting in (for example \"to turn it OFF\"), or null. "
        "Return {\"rules\": []} if the section has no such rule."
    ),
    form='{"rules": [{"condition": "<copied words>", "check": "<name>" | null, "fix": "<name>" | null, "fix_arguments": {}, "escalate": false, "fixed_state": "<copied words>" | null}]}',
    output=SectionRules,
    verify=_verify_rules,
    fallback=lambda i: SectionRules(rules=[]),
    max_tokens=900,
)

IS_RULE_ABOUT_ACTION = Primitive(
    name="Is the rule about this action",
    group="checking the policy",
    instruction=(
        "Decide whether this RULE applies right after the ACTION has been done, so that it is a follow-up to that action."
    ),
    form='{"about_this_action": true | false}',
    output=AboutAction,
    verify=lambda o, i: None,
    fallback=lambda i: AboutAction(about_this_action=False),
    max_tokens=30,
)

ORDER_REQUESTS = Primitive(
    name="Order requests",
    group="understanding the message",
    instruction=(
        "This policy AREA may say that other areas must work before it can. "
        "List the titles from AREA_TITLES that the AREA says must be checked first, in the order it gives."
    ),
    form='{"before": ["<title>", ...]}',
    output=Before,
    verify=lambda o, i: _all_in(o.before, i["area_titles"])
    or next((f"{t!r} is not named in the area" for t in o.before if not verbatim(t, i["area"])), None),
    fallback=lambda i: Before(before=[]),
)

PLAN_STEPS = Primitive(
    name="Plan steps",
    group="finding the tools",
    instruction=(
        "This RULE refers the agent to other parts of the policy for how to act. "
        "List the titles from PROCEDURE_TITLES whose steps the agent should follow for this rule. Return [] if none."
    ),
    form='{"procedures": ["<title>", ...]}',
    output=Procedures,
    verify=lambda o, i: _all_in(o.procedures, i["procedure_titles"]),
    fallback=lambda i: Procedures(procedures=[]),
)

COPY_PROCEDURE_STEPS = Primitive(
    name="Copy the procedure's steps",
    group="checking the policy",
    instruction=(
        "This policy SECTION may tell the agent how to carry out a request. "
        "List, in the order they must happen, each step the SECTION tells the agent or the user to take. "
        "Copy each step's words from the SECTION, one action per step. "
        "Skip sentences that only describe or explain, and skip lists of reasons or attributes. "
        'Return {"steps": []} if the SECTION gives no steps.'
    ),
    form='{"steps": ["<copied words>", ...]}',
    output=Steps,
    verify=lambda o, i: "no step is verbatim in the section"
    if o.steps and not any(verbatim(s, i["section"]) for s in o.steps)
    else None,
    fallback=lambda i: Steps(steps=[]),
    max_tokens=600,
)

DOES_FUNCTION_CHANGE_RECORD = Primitive(
    name="Does the function change a record",
    group="finding the tools",
    instruction=(
        "Decide from its DESCRIPTION whether calling this FUNCTION changes a record, or only reads. "
        "Copy the words of the DESCRIPTION that show it."
    ),
    form='{"evidence": "<copied words>", "changes": true | false}',
    output=Changes,
    verify=lambda o, i: None if verbatim(o.evidence, i["description"]) else "evidence not verbatim in description",
    fallback=lambda i: Changes(changes=False),
    max_tokens=80,
)

WHAT_COMES_FIRST = Primitive(
    name="What must be done first",
    group="checking the policy",
    instruction=(
        "This policy SECTION makes a change for the user. It may say that something else must have happened before "
        "the agent may make that change. PROCEDURES lists other parts of the policy that make a change. "
        "List the titles from PROCEDURES whose change the SECTION says must come first, and for each copy the words "
        'of the SECTION that say so. Return {"first": []} if the SECTION names none.'
    ),
    form='{"first": [{"title": "<title from PROCEDURES>", "quote": "<copied words of SECTION>"}]}',
    output=Prerequisites,
    verify=lambda o, i: _all_in([f.title for f in o.first], i["procedures"])
    or next((f"quote for {f.title!r} not verbatim in section" for f in o.first if not verbatim(f.quote, i["section"])), None),
    fallback=lambda i: Prerequisites(first=[]),
    max_tokens=200,
)

PICK_FUNCTION = Primitive(
    name="Pick the function",
    group="finding the tools",
    instruction="Pick the one function from FUNCTIONS whose effect is what the GOAL asks for. Use \"none\" if no function fits.",
    form='{"function": "<name>" | "none", "alternates": ["<name>", ...]}',
    output=Picked,
    verify=lambda o, i: _all_in([o.function, *o.alternates], [*i["functions"], "none"]),
    fallback=lambda i: Picked(function="none"),
    max_tokens=80,
)

# ---------------------------------------------------------------- understanding the message (run time)

SPLIT_REQUESTS = Primitive(
    name="Split into requests",
    group="understanding the message",
    instruction="Split the MESSAGE into the separate things the user asks for. Copy each request's words from the MESSAGE.",
    form='{"requests": ["<copied words>", ...]}',
    output=Requests,
    verify=lambda o, i: None if o.requests and all(verbatim(r, i["message"]) for r in o.requests) else "requests not verbatim",
    fallback=lambda i: Requests(requests=[i["message"]]),
)

CLASSIFY_REQUEST = Primitive(
    name="Classify one request",
    group="understanding the message",
    instruction=(
        "Pick the policy AREA that covers the problem this REQUEST reports. "
        'Answer with one title from AREA_TITLES, or "none" if the request reports no problem those areas cover.'
    ),
    form='{"area": "<title>" | "none"}',
    output=Area,
    verify=lambda o, i: None if o.area in [*i["area_titles"], "none"] else "unknown area",
    fallback=lambda i: Area(area="none"),
    max_tokens=60,
)

DEVICE_REPLY = Primitive(
    name="Device reply",
    group="understanding the message",
    instruction=(
        "The agent asked the user to use the functions in ASKED on their phone, and the USER_MESSAGE is the user's answer. "
        "For each function in ASKED, copy the words of the USER_MESSAGE that tell what the user saw or did with it. "
        "The user usually does not name the function; match by what the function does. Use null if the message does not say."
    ),
    form='{"reports": [{"function": "<name from ASKED>", "result": "<copied words>" | null}]}',
    output=DeviceReply,
    verify=lambda o, i: _all_in([r.function for r in o.reports], i["asked"])
    or (
        "no result is verbatim"
        if o.reports and all(r.result and not verbatim(r.result, i["user_message"]) for r in o.reports)
        else None
    ),
    fallback=lambda i: DeviceReply(reports=[]),
)

USER_PREMISE = Primitive(
    name="User's own premise or condition",
    group="checking the policy",
    instruction=(
        "Decide whether the user's own words in USER_MESSAGES show that the CONDITION is true of them. "
        "Copy the words that show it as evidence."
    ),
    form='{"evidence": "<copied words>" | null, "answer": "yes" | "no" | "cannot tell"}',
    output=Shows,
    verify=lambda o, i: _evidence(o.answer, o.evidence, i["user_messages"]),
    fallback=lambda i: Shows(answer="cannot tell"),
)

WHERE_VALUE = Primitive(
    name="Where a value comes from",
    group="building the call",
    instruction=(
        "Find the value of the PARAMETER in the SOURCE. Copy the value exactly as written, and copy the words around it as the quote. "
        "Use null if the SOURCE does not give it."
    ),
    form='{"value": "<copied value>" | null, "quote": "<copied words>" | null}',
    output=Value,
    verify=lambda o, i: None if o.value is None or (verbatim(o.value, i["source"]) and verbatim(o.quote, i["source"])) else "value not verbatim",
    fallback=lambda i: Value(value=None),
    max_tokens=120,
)

# ---------------------------------------------------------------- troubleshooting conditions (run time)

def _verify_compare(o: Compare, i: dict) -> str | None:
    if o.state_in_reading and not verbatim(o.state_in_reading, i["reading"]):
        return "state_in_reading not verbatim in reading"
    if o.state_rule_names and not verbatim(o.state_rule_names, i["rule"]):
        return "state_rule_names not verbatim in the rule's own words"
    return None


_COMPARE_STEPS = (
    "STATE_RULE_NAMES: copy from RULE the state it looks for (for example \"Missing\" or \"Airplane Mode is ON\"). "
    "STATE_IN_READING: copy from READING what it shows for that same setting, or null if READING does not mention it. "
    "SAME: true if the two name the same state, false if they name different states. Judge the meaning, not the wording: "
    "\"not enabled\" and \"disabled\" are the same state, and a value that includes the named state counts as that state "
    "(for example \"abc_only\" includes \"abc\"). RULE_TEXT is the policy section, for context."
)
_COMPARE_FORM = '{"state_rule_names": "<copied words>", "state_in_reading": "<copied words>" | null, "same": true | false | null}'

DOES_READING_SHOW = Primitive(
    name="Does one reading show the condition",
    group="troubleshooting conditions",
    instruction="A READING from one function may also show the state a RULE looks for. " + _COMPARE_STEPS,
    form=_COMPARE_FORM,
    output=Compare,
    verify=_verify_compare,
    fallback=lambda i: Compare(),
)

DOES_CONDITION_HOLD = Primitive(
    name="Does the condition hold now",
    group="troubleshooting conditions",
    instruction="The READING was taken with the function the RULE names. " + _COMPARE_STEPS,
    form=_COMPARE_FORM,
    output=Compare,
    verify=_verify_compare,
    fallback=lambda i: Compare(),
)


def _verify_met(o: Met, i: dict) -> str | None:
    if o.expression:
        try:
            value = evaluate(o.expression)
        except (ValueError, SyntaxError, ZeroDivisionError) as error:
            return f"expression does not evaluate: {error}"
        if (o.answer == "yes") != bool(value) and o.answer != "cannot tell":
            return "answer disagrees with expression"
        return _numbers_from(o.expression, i["record"])
    return _evidence(o.answer, o.evidence, i["record"])


IS_CONDITION_MET = Primitive(
    name="Is the condition met",
    group="checking the policy",
    instruction=(
        "Decide whether the CONDITION is met by the RECORD. If deciding needs a comparison of numbers or dates, write it as an "
        'EXPRESSION using values copied from the RECORD (for example "12.5 > 10" or "\'2025-01-31\' < \'2025-02-25\'"). '
        "Otherwise copy the deciding words as evidence."
    ),
    form='{"evidence": "<copied words>" | null, "expression": "<values and > < >= <= == + - * />" | null, "answer": "yes" | "no" | "cannot tell"}',
    output=Met,
    verify=_verify_met,
    fallback=lambda i: Met(answer="cannot tell"),
)

ALREADY_AS_WANTED = Primitive(
    name="Is the setting already as wanted",
    group="troubleshooting conditions",
    instruction=(
        "The fix in RULE changes a setting on the user's phone. "
        "STATE_RULE_NAMES: copy from RULE the state the setting is in after the fix has run, the healthy state "
        "(for example \"OFF\" after turning a feature off, or \"granted\" after granting a permission). "
        "STATE_IN_READING: copy from READING what it shows for that setting now, or null if READING does not mention it. "
        "SAME: true only if READING shows the setting already fully in that healthy state. Judge the meaning, not the wording."
    ),
    form=_COMPARE_FORM,
    output=Compare,
    verify=_verify_compare,
    fallback=lambda i: Compare(),
)

PROBLEM_FIXED = Primitive(
    name="Is the problem fixed",
    group="troubleshooting conditions",
    instruction=(
        "Decide whether the READING of the problem's check shows the problem is solved, judged by the POLICY_CRITERION "
        "and by what the user said they need in USER_GOAL. Copy the deciding words of the READING."
    ),
    form='{"evidence": "<copied words>" | null, "fixed": "yes" | "no" | "cannot tell"}',
    output=Fixed,
    verify=lambda o, i: _evidence(o.fixed, o.evidence, i["reading"]),
    fallback=lambda i: Fixed(fixed="cannot tell"),
)

# ---------------------------------------------------------------- building the call (run time)


def _verify_arguments(o: Arguments, i: dict) -> str | None:
    required = set(i["parameters"].get("required", []))
    if missing := required - set(o.arguments):
        return f"missing {sorted(missing)}"
    if extra := set(o.arguments) - set(i["parameters"].get("properties", {})):
        return f"unknown parameters {sorted(extra)}"
    for value in _flat_values(o.arguments):
        if isinstance(value, str) and not verbatim(value, i["facts_and_words"]):
            return f"value {value!r} not in records or the user's words"
        if isinstance(value, (int, float)) and not isinstance(value, bool) and _numbers_from(str(value), i["facts_and_words"]):
            return f"number {value} not in records or the user's words"
    return None


FILL_PARAMETERS = Primitive(
    name="Fill the parameters",
    group="building the call",
    instruction=(
        "Fill the arguments of FUNCTION for the GOAL. Every value must be copied from FACTS_AND_WORDS. "
        "An identifying argument names the target record, not the change to make."
    ),
    form='{"arguments": {"<parameter>": <value>, ...}}',
    output=Arguments,
    verify=_verify_arguments,
    fallback=lambda i: Arguments(arguments={}),
    max_tokens=200,
)


def _verify_price(o: Price, i: dict) -> str | None:
    if o.quantity is None or o.unit_price is None:
        return "missing quantity or price"
    if not verbatim(o.quantity_quote, i["source"]) or not verbatim(o.unit_price_quote, i["source"]):
        return "quotes not verbatim"
    return _numbers_from(f"{o.quantity} {o.unit_price}", i["source"])


PRICE_TERMS = Primitive(
    name="Price terms",
    group="building the call",
    instruction=(
        "Find the QUANTITY the user asked for and the UNIT_PRICE that applies to it in the SOURCE. "
        "Copy the words that state each one. Do not compute the total."
    ),
    form='{"quantity": <number> | null, "unit_price": <number> | null, "quantity_quote": "<copied words>", "unit_price_quote": "<copied words>"}',
    output=Price,
    verify=_verify_price,
    fallback=lambda i: Price(quantity=None, unit_price=None),
    max_tokens=150,
)

REPLY_TO_CONFIRMATION = Primitive(
    name="Reply to a confirmation",
    group="checking the policy",
    instruction=(
        "The agent asked the QUESTION. Decide whether the USER_MESSAGE agrees to it explicitly. "
        'A choice among options or a qualified yes is "unclear".'
    ),
    form='{"answer": "yes" | "no" | "unclear"}',
    output=Confirmation,
    verify=lambda o, i: None,
    fallback=lambda i: Confirmation(answer="unclear"),
    max_tokens=30,
)


def dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, default=str)
