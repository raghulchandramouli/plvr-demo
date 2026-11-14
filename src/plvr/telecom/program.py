"""The final telecom block of Appendix B.1, run before each student turn.

It reads the conversation so far, runs primitives on the frozen runner model, decides the next step, and writes it as
the reasoning the student acts on. Method docstrings quote the listing's lines they carry out.
"""

import json
import re
from collections import Counter
from dataclasses import asdict, dataclass
from typing import Any, Literal

from tau2.data_model.message import AssistantMessage, ToolMessage, UserMessage

from plvr.llm import Server
from plvr.primitive import MAX_INPUT_CHARS, Context, Primitive, Runner
from plvr.telecom import primitives as P
from plvr.telecom.policy import Rule, compile_policy

MAX_FIX_ATTEMPTS = 2
MAX_PROCEDURE_TURNS = 8
MAX_REPEATS = 2
TRANSFER_MESSAGE = "YOU ARE BEING TRANSFERRED TO A HUMAN AGENT. PLEASE HOLD ON."

Action = Literal["call", "ask", "procedure", "transfer", "tell", "close", "open"]

ACTION_FORM = {
    "call": "Make exactly this one tool call now and write no message.",
    "transfer": "Make exactly this one tool call now and write no message.",
    "ask": "Send one message to the user and make no tool call.",
    "tell": "Send one message to the user and make no tool call.",
    "close": "Send one message to the user and make no tool call.",
    "open": "Send one message to the user and make no tool call.",
    "procedure": "Make the one tool call or send the one message that the next step needs, not both.",
}


@dataclass
class Reading:
    function: str
    text: str
    turn: int


@dataclass
class Step:
    action: Action
    instruction: str
    why: str
    function: str | None = None
    arguments: dict | None = None
    rule: int | None = None
    policy: str = ""


class TelecomProgram:
    def __init__(self, policy: str, tools: list, runner: Server):
        self.compiled = compile_policy(policy, tools, Runner(runner))
        self.run = Runner(runner)
        self.schemas = {t.name: t.openai_schema["function"] for t in tools}
        self.agent_functions = {n: self.compiled.descriptions[n] for n in self.compiled.agent_functions}
        self.seen = 0
        self.turn = 0
        self.user_words: list[str] = []
        self.latest_user = ""
        self.previous_agent = ""
        self.requests: list[str] = []
        self.area: str | None = None
        self.checklist: list[Rule] = []
        self.phone: str | None = None
        self.customer: dict | None = None
        self.line: dict | None = None
        self.lines_read: list[dict] = []
        self.readings: dict[str, Reading] = {}
        self.records: list[str] = []
        self.pending: dict[str, tuple[str, dict]] = {}
        self.executed: list[tuple[str, dict, bool]] = []
        self.asked: list[str] = []
        self.status: dict[int, str] = {}
        self.attempts: Counter[int] = Counter()
        self.procedures_done: set[int] = set()
        self.last_fix_turn = 0
        self.proc: dict | None = None
        self.closed = False
        self.last_step: Step | None = None
        self.trace: list[dict] = []
        self._memo: dict[str, Any] = {}
        self.repeats = 0
        self.unreadable: set[str] = set()

    # ------------------------------------------------------------------ interface

    def think(self, history: list) -> str:
        for message in history[self.seen :]:
            self._ingest(message)
        self.seen = len(history)
        step = self._decide()
        self.repeats = self.repeats + 1 if step == self.last_step else 0
        if self.repeats >= MAX_REPEATS and step.action == "ask" and step.function:
            self.unreadable.add(step.function)
            self.repeats = 0
            step = self._decide()
        self.last_step = step
        self.trace.append({"turn": self.turn, "primitive_calls": len(self.run.calls), "step": asdict(step), "status": dict(self.status)})
        return self._write(step)

    def log_dict(self) -> dict:
        return {
            "area": self.area,
            "checklist": [f"[{r.section}] {r.condition}" for r in self.checklist],
            "trace": self.trace,
            "calls": self.run.log_dict(),
        }

    # ------------------------------------------------------------------ reading the conversation

    def _ingest(self, message: Any) -> None:
        if isinstance(message, AssistantMessage):
            self.previous_agent = message.content or ""
            for call in message.tool_calls or []:
                self.pending[call.id] = (call.name, call.arguments)
            self.asked = [f for f in self.compiled.device_functions if re.search(rf"\b{f}\b", self.previous_agent)]
        elif isinstance(message, ToolMessage):
            name, arguments = self.pending.pop(message.id, ("unknown", {}))
            text = message.content or ""
            self.executed.append((name, arguments, bool(message.error)))
            if message.error or name not in self.agent_functions:
                return
            self.records.append(f"{name}({P.dumps(arguments)}) -> {text}")
            self.readings[name] = Reading(name, text, self.turn)
            self._take_record(text)
        elif isinstance(message, UserMessage) and message.content:
            self.turn += 1
            self.latest_user = message.content
            self.user_words.append(message.content)
            self._understand(message.content)

    def _take_record(self, text: str) -> None:
        try:
            record = json.loads(text)
        except json.JSONDecodeError:
            return
        if not isinstance(record, dict):
            return
        if "customer_id" in record and "line_ids" in record:
            self.customer = record
        if "line_id" in record and "phone_number" in record:
            self.lines_read.append(record)
            if self.phone and _digits(record["phone_number"]) == _digits(self.phone):
                self.line = record

    def _understand(self, message: str) -> None:
        """split the message into requests"""
        if self.area is None:
            for request in self._call(P.SPLIT_REQUESTS, message=message).requests:
                area = self._call(P.CLASSIFY_REQUEST, request=request, area_titles=list(self.compiled.areas)).area
                if area != "none":
                    self.requests.append(request)
                    self.area = area
                    self.checklist = self.compiled.checklist(area)
                    break
        if self.phone is None:
            self.phone = self._call(
                P.WHERE_VALUE, parameter="the phone number of the user's line", source="\n".join(self.user_words)
            ).value
        if self.area is not None:
            self._device_reply(message)

    def _device_reply(self, message: str) -> None:
        """customer acts on their device: the user's report of each device function becomes a reading"""
        if not self.asked:
            return
        asked = {f: self.compiled.descriptions.get(f, "") for f in self.asked}
        reports = self._call(P.DEVICE_REPLY, asked=asked, user_message=message).reports
        step = self.last_step
        fix = self.checklist[step.rule].fix if step and step.rule is not None else None
        for report in reports:
            if report.result:
                self.readings[report.function] = Reading(report.function, report.result, self.turn)
                self.unreadable.discard(report.function)
            if report.result and fix and report.function == fix and step.function == fix:
                self.status[step.rule] = "fixed"
                self.last_fix_turn = self.turn

    # ------------------------------------------------------------------ deciding the next step

    def _decide(self) -> Step:
        if self.closed:
            return Step("open", "Reply briefly to the user's last message. Do not start new troubleshooting.", "The problem is solved.")
        if self._transferred():
            self.closed = True
            return Step("tell", f"Send the user exactly: '{TRANSFER_MESSAGE}'", "The transfer call is made; the policy says to send this message next.")
        if self.area is None:
            return Step("open", "Ask the user to describe the problem with their phone or line.", "No problem the policy covers is known yet.")
        if self.customer is None:
            return self._identify()
        if self.line is None:
            step = self._find_line()
            if step:
                return step
        area = self.compiled.areas[self.area]
        diagnostic = self._fresh(area.diagnostic)
        if diagnostic is None:
            return self._ask_device(area.diagnostic, "Check the problem early: the policy's check shows whether the problem is there.")
        fixed = self._call(
            P.PROBLEM_FIXED,
            policy_criterion=area.criterion,
            user_goal="\n".join(self.user_words[:3]),
            reading=f"{area.diagnostic}: {diagnostic.text}",
        ).fixed
        if fixed == "yes":
            self.closed = True
            return Step(
                "close",
                "Tell the user the problem is solved and ask whether they need anything else.",
                f"Re-run the rules' success check; stop when it passes: {area.diagnostic}() shows {diagnostic.text!r}.",
            )
        if self.proc is not None:
            step = self._procedure_step()
            if step:
                return step
        for index, rule in enumerate(self.checklist):
            if index not in self.status:
                step = self._rule_step(index, rule)
                if step:
                    return step
        return self._transfer("Escalate after a failed attempt: every rule in the policy was tried and the problem remains.")

    def _identify(self) -> Step:
        """identify first: looks up the user before anything else"""
        if self.phone is None or self._failed_lookup():
            self.phone = None
            return Step(
                "ask",
                "Ask the user for the phone number of the line that has the problem.",
                "Identify first: the customer must be looked up before anything else.",
            )
        function = self._pick("Find the customer's account from the phone number the user gave.")
        return self._agent_call(function, "Look up the customer by the phone number the user gave.", "Identify first: look up the user before anything else.")

    def _find_line(self) -> Step | None:
        """find the right record: reads all records to match the user's identifier"""
        read = {line["line_id"] for line in self.lines_read}
        unread = [line_id for line_id in self.customer.get("line_ids", []) if line_id not in read]
        if unread:
            function = self._pick(f"Read the details of the line record with ID {unread[0]}.")
            return self._agent_call(
                function,
                f"Read the details of line {unread[0]}.",
                f"Find the right record: read every line on the account to find the one with phone number {self.phone}.",
            )
        self.line = self.lines_read[0] if self.lines_read else {}
        return None

    def _rule_step(self, index: int, rule: Rule) -> Step | None:
        """for each rule of the form "if <condition>, apply <fix>"; go through them in their written order"""
        holds = self._holds(index, rule)
        if isinstance(holds, Step):
            return holds
        if not holds:
            self.status[index] = "not needed"
            return None
        if rule.escalate:
            self.status[index] = "escalated"
            return self._transfer(f"Refusal cites the rule: the policy says to escalate when {rule.condition!r}.", rule.policy)
        if rule.procedures and index not in self.procedures_done:
            self.proc = {"rule": index, "title": self._choose_procedure(rule), "turns": 0, "start": len(self.executed), "after": 0}
            return self._procedure_step()
        if rule.fix is None:
            self.status[index] = "fixed"
            return None
        return self._device_fix(index, rule)

    def _holds(self, index: int, rule: Rule) -> bool | Step:
        """read the current state; check the problem; dates computed in code"""
        text = f"[{rule.section}] {rule.condition}"
        if rule.check:
            reading = self._fresh(rule.check)
            if reading is None:
                shown = self._shown_by_other_readings(rule, text)
                if shown is not None:
                    return shown
                if rule.check in self.compiled.device_functions:
                    return self._ask_device(rule.check, f"Read the current state for the rule {text!r}.", index)
                return self._agent_call(
                    rule.check, f"Read {rule.check} for the user's line.", f"Read the current state for the rule {text!r}.", index
                )
            answer = self._call(P.DOES_CONDITION_HOLD, rule=text, reading=f"{rule.check}: {reading.text}").answer
        else:
            met = self._call(P.IS_CONDITION_MET, condition=text, record=self._record_text(rule, with_user_words=False))
            answer = met.answer
            if met.expression:
                answer = "yes" if P.evaluate(met.expression) else "no"
        if answer == "cannot tell" and not rule.escalate:
            answer = self._call(P.USER_PREMISE, condition=text, user_messages="\n".join(self.user_words)).answer
        return answer == "yes"

    def _shown_by_other_readings(self, rule: Rule, condition: str) -> bool | None:
        """A fresh reading of another function counts only when the rule's own policy section names that function."""
        fresh = [
            r for r in self.readings.values()
            if r.function != rule.check and r.turn >= self.last_fix_turn and re.search(rf"\b{r.function}\b", rule.policy)
        ]
        for reading in sorted(fresh, key=lambda r: -r.turn)[:2]:
            answer = self._call(P.DOES_READING_SHOW, condition=condition, reading=f"{reading.function}: {reading.text}").answer
            if answer != "cannot tell":
                return answer == "yes"
        return None

    def _device_fix(self, index: int, rule: Rule) -> Step | None:
        """skip a fix whose effect is already in place; after a state-changing action, request the follow-up step at once"""
        if self.attempts[index] >= MAX_FIX_ATTEMPTS:
            self.status[index] = "failed"
            return None
        reader = rule.reader or rule.check
        current = self._fresh(reader) if reader else None
        if current is not None:
            already = self._call(
                P.ALREADY_AS_WANTED, fix=f"{rule.fix}: [{rule.section}] {rule.condition}", reading=f"{reader}: {current.text}"
            ).already
            if already == "yes":
                self.status[index] = "already in place"
                return None
        elif reader and reader != rule.check:
            return self._ask_device(reader, f"Read the current state before {rule.fix} so a fix already in place is not undone.", index)
        self.attempts[index] += 1
        diagnostic = self.compiled.areas[self.area].diagnostic
        then = "".join(f", then {f}()" for f in rule.then)
        return Step(
            "ask",
            f"Ask the user to use {_call_text(rule.fix, rule.fix_arguments)}{then} as the policy below says, "
            f"and then to run {diagnostic}() again and tell you exactly what it shows. "
            "Do not call these functions yourself: they run on the user's phone.",
            f"The rule {rule.condition!r} applies and its fix is not in place.",
            function=rule.fix,
            rule=index,
            policy=rule.policy,
        )

    def _choose_procedure(self, rule: Rule) -> str:
        if len(rule.procedures) == 1:
            return rule.procedures[0]
        for title in rule.procedures:
            premise = self._call(
                P.USER_PREMISE, condition=f"The user accepts this option: {title}", user_messages="\n".join(self.user_words)
            ).answer
            if premise == "yes":
                return title
        return " or ".join(rule.procedures)

    def _procedure_step(self) -> Step | None:
        """one step at a time: executes a procedure written as a list of steps one step at a time"""
        p = self.proc
        index, title = p["rule"], p["title"]
        text = "\n\n".join(f"## {t}\n{self.compiled.procedures[t]}" for t in title.split(" or "))
        actions = [f"{n}({P.dumps(a)})" for n, a, error in self.executed[p["start"] :] if not error]
        made = bool(actions) and self._call(
            P.REQUEST_ACHIEVED, policy_section=text, actions="\n".join(r for r in self.records if any(r.startswith(a) for a in actions))
        ).achieved == "yes"
        if made:
            p["after"] += 1
        finished = made and p["after"] > 1
        if finished or self._declined() or p["turns"] >= MAX_PROCEDURE_TURNS:
            self.procedures_done.add(index)
            self.proc = None
            if made:
                self.last_fix_turn = self.turn
            rule = self.checklist[index]
            if rule.fix is None or not made:
                self.status[index] = "fixed" if made else "procedure not completed"
            return None
        p["turns"] += 1
        why = f"The rule {self.checklist[index].condition!r} applies and the policy defers to the section below."
        if made:
            diagnostic = self.compiled.areas[self.area].diagnostic
            instruction = f"The change is made. Do the step the section says comes after it, if any; otherwise ask the user to run {diagnostic}() again."
        else:
            instruction = "Follow the policy section below one step at a time. Do only the next step not yet done in this conversation."
        return Step("procedure", instruction, why, rule=index, policy=text + self._price_fact(text))

    def _price_fact(self, procedure: str) -> str:
        """quantities from records: every quantity is user-stated or computed from records"""
        if "price" not in procedure.lower():
            return ""
        price = self._call(P.PRICE_TERMS, source=self._record_text())
        if price.quantity is None or price.unit_price is None:
            return ""
        return f"\n\nComputed: {price.quantity:g} at {price.unit_price:g} each is {price.quantity * price.unit_price:.2f} in total."

    def _transfer(self, why: str, policy: str = "") -> Step:
        """escalate after a failed attempt; ended contract to a human"""
        function = self._pick("Transfer the user to a human agent.")
        return Step(
            "transfer",
            f"Call {function} with a one-sentence summary of the user's issue. The next message will tell the user they are being transferred.",
            why,
            function=function,
            policy=policy,
        )

    # ------------------------------------------------------------------ helpers

    def _agent_call(self, function: str, goal: str, why: str, rule: int | None = None) -> Step:
        """argument names the target: an identifying argument names the target, not the change to make"""
        arguments = self._call(
            P.FILL_PARAMETERS,
            function=function,
            parameters=self.schemas[function]["parameters"],
            goal=goal,
            facts_and_words=self._record_text(),
        ).arguments
        if not arguments and self.schemas[function]["parameters"].get("required"):
            return Step("ask", f"Ask the user for the information needed to {goal[0].lower()}{goal[1:]}", why, rule=rule)
        return Step("call", f"Call {_call_text(function, arguments)}.", why, function=function, arguments=arguments, rule=rule)

    def _ask_device(self, function: str, why: str, rule: int | None = None) -> Step:
        return Step(
            "ask",
            f"Ask the user to run {function}() on their phone and tell you exactly what it shows. "
            f"Do not call {function} yourself: it runs on the user's phone, and only the user can run it.",
            why,
            function=function,
            rule=rule,
        )

    def _pick(self, goal: str) -> str:
        return self._call(P.PICK_FUNCTION, goal=goal, functions=self.agent_functions).function

    def _fresh(self, function: str) -> Reading | None:
        reading = self.readings.get(function)
        if function in self.unreadable:
            return reading or Reading(function, "(the user could not report this)", self.turn)
        return reading if reading and reading.turn >= self.last_fix_turn else None

    def _transferred(self) -> bool:
        return bool(self.last_step and self.last_step.action == "transfer") and any(
            name == self.last_step.function and not error for name, _, error in self.executed
        )

    def _failed_lookup(self) -> bool:
        return bool(self.executed) and self.executed[-1][2] and self.customer is None

    def _declined(self) -> bool:
        if not (self.last_step and self.last_step.action == "procedure" and self.previous_agent.rstrip().endswith("?")):
            return False
        return self._call(P.REPLY_TO_CONFIRMATION, question=self.previous_agent, user_message=self.latest_user).answer == "no"

    def _record_text(self, rule: Rule | None = None, with_user_words: bool = True) -> str:
        parts = [self.compiled.preamble]
        if rule is not None:
            parts += [f"## {t}\n{self.compiled.procedures[t]}" for t in rule.procedures]
        if self.customer:
            parts.append(f"customer: {P.dumps(self.customer)}")
        if self.line:
            parts.append(f"line: {P.dumps(self.line)}")
        parts += self.records[-6:]
        if with_user_words:
            parts.append("user said: " + " | ".join(self.user_words))
        return "\n".join(parts)[-MAX_INPUT_CHARS:]

    def _facts(self) -> str:
        facts = []
        if self.area:
            facts.append(f"Problem area: {self.area}. Request: {' / '.join(self.requests)}")
        if self.customer:
            facts.append(f"Customer: {self.customer.get('customer_id')} {self.customer.get('full_name', '')}")
        if self.line:
            facts.append(f"Line: {self.line.get('line_id')} phone {self.line.get('phone_number')} status {self.line.get('status')}")
        for reading in self.readings.values():
            if reading.function in self.compiled.device_functions:
                stale = "" if reading.turn >= self.last_fix_turn else " (before the last fix)"
                facts.append(f"{reading.function}(): {reading.text[:300]}{stale}")
        for index, state in self.status.items():
            facts.append(f"Rule [{self.checklist[index].section}]: {state}")
        return "\n".join(facts)

    def _call(self, primitive: Primitive, **inputs: Any) -> Any:
        key = primitive.name + json.dumps(inputs, sort_keys=True, default=str)
        if key not in self._memo:
            self._memo[key] = self.run(primitive, Context(self._facts(), self.latest_user, self.previous_agent), **inputs)
        return self._memo[key]

    def _write(self, step: Step) -> str:
        parts = [f"What I know so far:\n{self._facts() or '(nothing yet)'}"]
        if step.policy:
            parts.append(f"The policy for this step:\n{step.policy.strip()}")
        parts.append(f"Why: {step.why}")
        parts.append(f"Next: {step.instruction}")
        parts.append(ACTION_FORM[step.action])
        parts.append(
            f"The only tools I can call are: {', '.join(self.compiled.agent_functions)}. "
            "Every other function runs on the user's phone, so I must ask the user to run it and never call it myself."
        )
        return "\n\n".join(parts)


def _digits(text: str) -> str:
    return re.sub(r"\D", "", str(text))


def _call_text(function: str, arguments: dict | None) -> str:
    args = ", ".join(f"{k}={json.dumps(v)}" for k, v in (arguments or {}).items())
    return f"{function}({args})"
