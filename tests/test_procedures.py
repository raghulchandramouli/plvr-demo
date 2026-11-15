"""Procedures: what the compile step accepts from a policy section, and how the program carries one out turn by turn.

The runner model is a fake that answers each primitive from a script, so every verifier and fallback runs for real.
The compile-time scripts replay what Qwen3-8B answered on the real τ² policy, where that was wrong.
"""

import json
from dataclasses import asdict
from types import SimpleNamespace

import pytest
from tau2.data_model.message import AssistantMessage, ToolCall, ToolMessage, UserMessage

from plvr.llm import Completion
from plvr.primitive import Context, Runner
from plvr.telecom import program as program_module
from plvr.telecom.policy import (
    Area,
    Compiled,
    Procedure,
    Rule,
    _load,
    changing_functions,
    instructing_lines,
    instruction_problem,
    named_functions,
    parse_sections,
    plan_procedures,
)
from plvr.telecom.program import MAX_PROCEDURE_TURNS, Reading, TelecomProgram


class ScriptedServer:
    """Answers a primitive by its name. An unscripted primitive answers `{}`, which fails its schema and falls back."""

    def __init__(self, answers: dict):
        self.answers = answers

    def render(self, messages, tools=None, *, thinking):
        return messages[-1]["content"]

    def complete(self, prompt, **kwargs):
        answer = self.answers.get(prompt.splitlines()[0].removeprefix("STEP: "), {})
        if callable(answer):
            answer = answer(prompt)
        return Completion(json.dumps(answer), "", [], 10, 0, 5, 0.0, "stop")


def last_input(prompt: str, name: str) -> str:
    """The text of the last named input of a rendered primitive prompt, which may hold blank lines."""
    return prompt.split(f"\n{name}:\n", 1)[1].split("\n\nAnswer with one JSON", 1)[0]


def by_input(name: str, table: dict, default=None):
    """Answer by the first key of `table` found in the named input."""
    return lambda prompt: next((answer for key, answer in table.items() if key in first_block(prompt, name)), default or {})


def first_block(prompt: str, name: str) -> str:
    """A named input up to the next blank line: enough for one-line inputs such as FUNCTION, GOAL or QUESTION."""
    return prompt.split(f"\n{name}:\n", 1)[1].split("\n\n", 1)[0]


# ---------------------------------------------------------------- compile time

SECTIONS = {
    "Line": """Each line has the following attributes:
- line ID
- status
- roaming status
- suspension start date (if applicable)

There are four line status types: **Active**, **Suspended**, **Pending Activation**, and **Closed**.""",
    "Overdue Bill Payment": """You can help the user make a payment for an overdue bill.
To do so you need to follow these steps:
- Check the bill status to make sure it is overdue.
- Check the bill amount due
- Send the user a payment request for the overdue bill.
    - This will change the status of the bill to AWAITING PAYMENT.
- Inform the user that a payment request has been sent. They should:
    - Check their payment requests using the check_payment_request tool.
- If the user accepts the payment request, use the make_payment tool to make the payment.
- After the payment is made, the bill status will be updated to PAID.
- Always check that the bill status is updated to PAID before informing the user that the bill has been paid.

Important:
- A user can only have one bill in the AWAITING PAYMENT status at a time.""",
    "Line Suspension": """When a line is suspended, the user will not have service.
A line can be suspended for the following reasons:
- The user has an overdue bill.
- The line's contract end date is in the past.

You are allowed to lift the suspension after the user has paid all their overdue bills.
You are not allowed to lift the suspension if the line's contract end date is in the past, even if the user has paid all their overdue bills.

After you resume the line, the user will have to reboot their device to get service.""",
    "Data Refueling": """Each plan specify the maxium data usage per month.
You can add more data to the line by "refueling" data at a price per GB specified by the plan.
The maximum amount of data that can be refueled is 2GB.
To refuel data you should:
- Ask them how much data they want to refuel
- Confirm the price
- Apply the refueled data to the line associated with the phone number the user provided.""",
    "Data Roaming": """If a line is roaming enabled, the user can use their phone's data connection in areas outside their home network.
We offer data roaming to users who are traveling outside their home network.
If a user is traveling outside their home network, you should check if the line is roaming enabled. If it is not, you should enable it at no cost for the user.""",
}
WRITES = {
    "disable_roaming": "Disables international roaming on a line.",
    "enable_roaming": "Enables international roaming on a line.",
    "refuel_data": "Refuels data for a specific line, adding to the customer's bill.",
    "resume_line": "Resumes a suspended line.",
    "send_payment_request": "Sends a payment request to the customer for a specific bill.",
    "suspend_line": "Suspends a specific line (max 6 months).",
}
READS = {"get_details_by_id": "Retrieves the details for a given ID.", "get_bills_for_customer": "Retrieves a list of the customer's bills."}
FUNCTIONS = [*WRITES, *READS, "make_payment", "check_payment_request", "reboot_device"]

APPLY = "Apply the refueled data to the line associated with the phone number the user provided."
SEND = "Send the user a payment request for the overdue bill."
ENABLE = "If a user is traveling outside their home network, you should check if the line is roaming enabled. If it is not, you should enable it at no cost for the user."
LIFT = "You are allowed to lift the suspension after the user has paid all their overdue bills."
REBOOT = "After you resume the line, the user will have to reboot their device to get service."
PAID = "Always check that the bill status is updated to PAID before informing the user that the bill has been paid."
TRUE_EFFECTS = {
    "enable_roaming": {"title": "Data Roaming", "quote": ENABLE},
    "refuel_data": {"title": "Data Refueling", "quote": APPLY},
    "resume_line": {"title": "Line Suspension", "quote": LIFT},
    "send_payment_request": {"title": "Overdue Bill Payment", "quote": SEND},
}
READ_BACK = by_input(
    "GOAL",
    {
        "enable it": {"function": "enable_roaming"},
        "Apply the refueled": {"function": "refuel_data"},
        "lift the suspension": {"function": "resume_line"},
        "payment request": {"function": "send_payment_request"},
    },
    {"function": "none"},
)


def plan(answers: dict) -> tuple[dict[str, Procedure], list[str]]:
    rejected: list[str] = []
    run = Runner(ScriptedServer({"Pick the function": READ_BACK, **answers}))
    return plan_procedures(run, Context(), SECTIONS, WRITES, READS, FUNCTIONS, rejected), rejected


def effects(table: dict) -> dict:
    """The runner names, per function, the offered line that holds the quote; a quote that is not offered gets a number no line has."""

    def number(prompt: str) -> dict:
        quote = by_input("FUNCTION", table, {"quote": None})(prompt)["quote"]
        offered = last_input(prompt, "LINES")
        return {"number": next((n for n, line in enumerate(json.loads(offered), 1) if quote and quote in line), 999 if quote else None)}

    return {"Which line tells the agent to make the change": number}


@pytest.mark.parametrize(
    "title, sentence, problem",
    [
        ("Line", "roaming status", "describes"),
        ("Line", "suspension start date (if applicable)", "describes"),
        ("Line Suspension", "The user has an overdue bill.", "describes"),
        ("Line Suspension", "When a line is suspended, the user will not have service.", "describes"),
        ("Data Roaming", "We offer data roaming to users who are traveling outside their home network.", "describes"),
        ("Overdue Bill Payment", "A user can only have one bill in the AWAITING PAYMENT status at a time.", "describes"),
        ("Line Suspension", "You are not allowed to lift the suspension if the line's contract end date is in the past, even if the user has paid all their overdue bills.", "forbids"),
        ("Overdue Bill Payment", "If the user accepts the payment request, use the make_payment tool to make the payment.", "names another function"),
        ("Data Roaming", "You should disable roaming for the user.", "is not in the section"),
    ],
)
def test_a_sentence_that_does_not_instruct_cannot_carry_a_change(title, sentence, problem):
    assert problem in instruction_problem(SECTIONS[title], sentence, FUNCTIONS)


@pytest.mark.parametrize("title, sentence", [("Data Refueling", APPLY), ("Overdue Bill Payment", SEND), ("Data Roaming", ENABLE), ("Line Suspension", LIFT)])
def test_a_sentence_that_instructs_the_agent_can_carry_a_change(title, sentence):
    assert instruction_problem(SECTIONS[title], sentence, FUNCTIONS) is None


def test_only_sections_that_instruct_a_change_become_procedures():
    plans, rejected = plan(effects(TRUE_EFFECTS))
    assert {title: list(p.effects) for title, p in plans.items()} == {
        "Overdue Bill Payment": ["send_payment_request"],
        "Line Suspension": ["resume_line"],
        "Data Refueling": ["refuel_data"],
        "Data Roaming": ["enable_roaming"],
    }
    assert plans["Data Refueling"].effects["refuel_data"] == APPLY
    assert rejected == []


def test_what_qwen3_8b_read_from_lists_and_descriptions_is_never_offered_as_an_instruction():
    read_by_qwen = {
        "disable_roaming": {"title": "Line", "quote": "roaming status"},
        "suspend_line": {"title": "Line Suspension", "quote": "The user has an overdue bill."},
        "send_payment_request": {
            "title": "Overdue Bill Payment",
            "quote": "If the user accepts the payment request, use the make_payment tool to make the payment.",
        },
        "enable_roaming": {"title": "Data Roaming", "quote": "If a line is roaming enabled, the user can use their phone's data connection"},
    }
    offered = [line for _, line in instructing_lines(SECTIONS, FUNCTIONS)]
    assert not any(read["quote"] in line for read in read_by_qwen.values() for line in offered)
    assert {APPLY, SEND, ENABLE, LIFT} <= set(offered)

    plans, _ = plan(effects({**read_by_qwen, "refuel_data": {"title": "Data Refueling", "quote": APPLY}}))
    assert list(plans) == ["Data Refueling"]


def test_a_change_the_model_reads_back_as_another_function_is_dropped():
    plans, rejected = plan(effects({"suspend_line": {"title": "Line Suspension", "quote": LIFT}, "disable_roaming": {"title": "Data Roaming", "quote": ENABLE}}))
    assert plans == {}
    assert "reads back as enable_roaming, not disable_roaming" in rejected[0] and "reads back as resume_line, not suspend_line" in rejected[1]


def test_a_prerequisite_needs_a_sentence_that_puts_it_first():
    first = by_input(
        "SECTION",
        {
            "## Line Suspension": {"first": [{"title": "Overdue Bill Payment", "quote": "after the user has paid all their overdue bills"}]},
            # Qwen3-8B put another section before roaming on the strength of this clause.
            "## Data Roaming": {"first": [{"title": "Line Suspension", "quote": "you should check if the line is roaming enabled"}]},
            "## Data Refueling": {"first": [{"title": "Data Roaming", "quote": "To refuel data you should"}]},
        },
        {"first": []},
    )
    plans, rejected = plan({**effects(TRUE_EFFECTS), "What must be done first": first})
    assert {title: p.requires for title, p in plans.items()} == {
        "Overdue Bill Payment": [],
        "Line Suspension": ["Overdue Bill Payment"],
        "Data Refueling": [],
        "Data Roaming": [],
    }
    assert len(rejected) == 2


def copied(table: dict):
    """Answers "Copy the sentence that answers" by section title, for one of its two questions."""
    return lambda question: lambda prompt: (
        by_input("SECTION", {f"## {t}": {"quote": q} for t, q in table.items()}, {"quote": None})(prompt)
        if question in first_block(prompt, "QUESTION")
        else {"quote": None}
    )


def test_a_completion_check_must_be_a_check_the_policy_puts_after_the_change():
    wrong = copied({"Data Refueling": "Confirm the price", "Overdue Bill Payment": "Check the bill status to make sure it is overdue."})
    plans, rejected = plan({**effects(TRUE_EFFECTS), "Copy the sentence that answers": wrong("check after the change")})
    assert all(p.check is None for p in plans.values())
    assert "names no check" in rejected[1] and "comes before the change" in rejected[0]

    right = copied({"Overdue Bill Payment": PAID})
    reader = by_input("GOAL", {"Read the record this policy sentence": {"function": "get_details_by_id"}})
    plans, rejected = plan(
        {
            **effects(TRUE_EFFECTS),
            "Copy the sentence that answers": right("check after the change"),
            "Pick the function": lambda prompt: reader(prompt) or READ_BACK(prompt),
        }
    )
    assert plans["Overdue Bill Payment"].check == PAID
    assert plans["Overdue Bill Payment"].check_reads == "get_details_by_id"
    assert rejected == []


def test_a_follow_up_must_say_what_the_user_does_after_the_change():
    answers = copied({"Line Suspension": REBOOT, "Overdue Bill Payment": "After the payment is made, the bill status will be updated to PAID."})
    plans, rejected = plan({**effects(TRUE_EFFECTS), "Copy the sentence that answers": answers("user has to do after")})
    assert plans["Line Suspension"].after.startswith("After you resume the line")
    assert plans["Overdue Bill Payment"].after is None and len(rejected) == 1


def test_a_function_changes_state_only_with_evidence_from_its_description_or_the_policy():
    sections = parse_sections("# Policy\n## Data Roaming\n" + SECTIONS["Data Roaming"])
    agent = {**WRITES, **READS}
    says = lambda evidence: Runner(ScriptedServer({"Does the function change stored data": {"changes_state": True, "evidence": evidence}}))
    assert changing_functions(says("Deletes the account"), Context(), agent, sections) == {}
    assert list(changing_functions(says("roaming on a line"), Context(), agent, sections)) == ["disable_roaming", "enable_roaming"]


def test_names_a_procedure_writes_that_are_not_agent_tools_run_on_the_users_side():
    assert named_functions(SECTIONS["Overdue Bill Payment"], {"transfer_to_human_agents": ""}) == ["check_payment_request", "make_payment"]


def test_compiled_plans_survive_the_cache():
    procedure = Procedure("T", {"f": "do it"}, "r", "check it", "r", "then the user does this", ["U"])
    compiled = Compiled({}, {}, [], [], {}, [], "", plans={"T": procedure}, state_changing=["f"])
    assert _load(json.loads(json.dumps(asdict(compiled)))) == compiled


# ---------------------------------------------------------------- run time


def tool(name: str, *parameters: str):
    schema = {"name": name, "description": "", "parameters": {"properties": {p: {} for p in parameters}, "required": list(parameters)}}
    return SimpleNamespace(name=name, openai_schema={"function": schema})


TOOLS = [
    tool("refuel_data", "customer_id", "line_id", "gb_amount"),
    tool("resume_line", "customer_id", "line_id"),
    tool("send_payment_request", "customer_id", "bill_id"),
    tool("get_details_by_id", "id"),
    tool("get_bills_for_customer", "customer_id"),
]
PLANS = {
    "Data Refueling": Procedure("Data Refueling", {"refuel_data": APPLY}),
    "Line Suspension": Procedure("Line Suspension", {"resume_line": LIFT}, after=REBOOT, requires=["Overdue Bill Payment"]),
    "Overdue Bill Payment": Procedure(
        "Overdue Bill Payment", {"send_payment_request": SEND}, reads="get_bills_for_customer", check=PAID, check_reads="get_details_by_id"
    ),
}
CONDITION = "the rule applies"
FILL = by_input(
    "FUNCTION",
    {
        "refuel_data": {"arguments": {"customer_id": "CA", "line_id": "LA", "gb_amount": 3}},
        "resume_line": {"arguments": {"customer_id": "CA", "line_id": "LA"}},
        "send_payment_request": {"arguments": {"customer_id": "CA", "bill_id": "BA"}},
        "get_bills_for_customer": {"arguments": {"customer_id": "CA"}},
        "get_details_by_id": {"arguments": {"id": "BA"}},
    },
)


def condition_met(prompt: str) -> dict:
    if CONDITION in first_block(prompt, "CONDITION"):
        return {"evidence": "customer", "answer": "yes"}
    record = last_input(prompt, "RECORD")
    return {"evidence": "Paid", "answer": "yes"} if '"status": "Paid"' in record else {"evidence": "Awaiting Payment", "answer": "no"}


ANSWERS = {
    "Is the condition met": condition_met,
    "Fill the parameters": FILL,
    "Reply to a confirmation": by_input("USER_MESSAGE", {"No thanks": {"answer": "no"}}, {"answer": "unclear"}),
    "Device reply": lambda prompt: {"reports": [{"function": "run_speed_test", "result": first_block(prompt, "USER_MESSAGE")}]},
}


class Chat:
    """Plays the student and the user against the program, and returns the step the program decides after each."""

    def __init__(self, monkeypatch, procedures: list[str], answers: dict | None = None, plans: dict | None = None):
        rule = Rule("Data", "Section", CONDITION, None, None, {}, False, procedures=procedures)
        compiled = Compiled(
            areas={"Data": Area("Data", "run_speed_test", "", [], [rule])},
            procedures={**SECTIONS, "Change Plan": "Apply the plan to the line."},
            device_functions=["make_payment", "reboot_device", "run_speed_test"],
            agent_functions=[t.name for t in TOOLS],
            descriptions={t.name: "" for t in TOOLS},
            rejected=[],
            preamble="",
            plans=plans or PLANS,
            state_changing=["refuel_data", "resume_line", "send_payment_request"],
        )
        monkeypatch.setattr(program_module, "compile_policy", lambda *_: compiled)
        self.program = program = TelecomProgram("", TOOLS, ScriptedServer({**ANSWERS, **(answers or {})}))
        program.area, program.checklist, program.phone = "Data", compiled.checklist("Data"), "555"
        program.customer = {"customer_id": "CA", "line_ids": ["LA"], "bill_ids": ["BA", "BB"]}
        program.line = {"line_id": "LA", "phone_number": "555", "plan_id": "PA"}
        program.readings["run_speed_test"] = Reading("run_speed_test", "no connection", 0)
        self.history: list = []
        self.calls = 0

    def step(self) -> SimpleNamespace:
        self.program.think(self.history)
        return SimpleNamespace(**self.program.trace[-1]["step"], status=self.program.status.get(0))

    def user(self, text: str, after: str = "Shall I go on?") -> SimpleNamespace:
        if self.history:
            self.history.append(AssistantMessage(role="assistant", content=after))
        self.history.append(UserMessage(role="user", content=text))
        return self.step()

    def tool(self, name: str, arguments: dict, result: str, error: bool = False) -> SimpleNamespace:
        self.calls += 1
        call = ToolCall(id=f"call{self.calls}", name=name, arguments=arguments)
        self.history.append(AssistantMessage(role="assistant", tool_calls=[call]))
        self.history.append(ToolMessage(id=call.id, role="tool", content=result, error=error))
        return self.step()

    def called(self) -> list[str]:
        return [s["step"]["function"] for s in self.program.trace if s["step"]["action"] == "call"]


def test_a_procedure_gives_one_instruction_a_turn_and_ends_when_its_change_succeeds(monkeypatch):
    price = {"quantity": 3, "unit_price": 2.5, "quantity_quote": "3 GB", "unit_price_quote": "price_per_gb: 2.5"}
    chat = Chat(
        monkeypatch,
        ["Data Refueling"],
        {
            "Price terms": price,
            "Where a value comes from": {"value": "PA", "quote": "plan_id: PA"},
            "Pick the function": {"function": "get_details_by_id"},
            "Fill the parameters": lambda prompt: {"arguments": {"id": "PA"}} if "get_details_by_id" in first_block(prompt, "FUNCTION") else FILL(prompt),
        },
    )

    step = chat.user("My data stopped working.")
    assert (step.action, step.function) == ("ask", None) and "information needed" in step.instruction, "no amount is stated yet"

    step = chat.user("3 GB please.", after="How much data would you like to add?")
    assert (step.action, step.function, step.arguments) == ("call", "get_details_by_id", {"id": "PA"}), "the price is read before it is quoted"

    step = chat.tool("get_details_by_id", {"id": "PA"}, '{"plan_id": "PA", "price_per_gb": 2.5}')
    assert step.action == "ask" and "confirm" in step.instruction
    assert "3 at 2.5 each is 7.50 in total" in step.policy

    step = chat.user("Yes, go ahead.", after="That is 7.50 in total. Do you confirm?")
    assert (step.action, step.function) == ("call", "refuel_data")
    assert step.arguments == {"customer_id": "CA", "line_id": "LA", "gb_amount": 3}
    assert chat.program.proc is not None

    step = chat.tool("refuel_data", step.arguments, '{"message": "added"}')
    assert chat.program.proc is None and step.status == "fixed"
    assert step.function == "run_speed_test", "the change is made, so the program goes back to the problem's own check"


def test_the_price_is_confirmed_before_the_change_even_when_no_total_can_be_computed(monkeypatch):
    chat = Chat(monkeypatch, ["Data Refueling"])
    step = chat.user("My data stopped working, add 3 GB.")
    assert step.action == "ask" and "confirm the price" in step.instruction
    step = chat.user("Yes.", after="Do you confirm the price?")
    assert (step.action, step.function) == ("call", "refuel_data")


def test_a_change_the_student_makes_early_ends_the_procedure(monkeypatch):
    chat = Chat(monkeypatch, ["Data Refueling"])
    chat.user("My data stopped working, add 3 GB.")
    step = chat.tool("refuel_data", {"customer_id": "CA", "line_id": "LA", "gb_amount": 3}, '{"message": "added"}')
    assert chat.program.proc is None and step.status == "fixed"
    assert chat.called() == []


def test_a_user_who_declines_ends_the_procedure_without_the_change(monkeypatch):
    chat = Chat(monkeypatch, ["Data Refueling"])
    chat.user("My data stopped working.")
    step = chat.user("No thanks, I do not want to pay for data.", after="How much data would you like to add?")
    assert chat.program.proc is None and step.status == "declined by the user"
    assert chat.called() == []


def test_a_change_that_keeps_failing_ends_the_procedure(monkeypatch):
    chat = Chat(monkeypatch, ["Data Refueling"])
    chat.user("My data stopped working, add 3 GB.")
    step = chat.user("Yes.", after="Do you confirm?")
    assert step.function == "refuel_data"
    step = chat.tool("refuel_data", step.arguments, "Error: line not found", error=True)
    assert step.function == "refuel_data", "one failed call is retried"
    step = chat.tool("refuel_data", step.arguments, "Error: line not found", error=True)
    assert chat.program.proc is None and step.status == "procedure not completed"


def test_a_procedure_the_agent_has_no_function_for_is_not_started(monkeypatch):
    chat = Chat(monkeypatch, ["Change Plan"])
    step = chat.user("My data stopped working.")
    assert step.status == "procedure not completed" and chat.program.proc is None

    chat = Chat(monkeypatch, ["Change Plan", "Data Refueling"])
    step = chat.user("My data stopped working.")
    assert "information needed" in step.instruction, "the one procedure with a function is taken without asking the user to choose"


BILLS = '[{"bill_id": "BA", "status": "%s"}, {"bill_id": "BB", "status": "Paid"}]'


def request_payment(chat: Chat) -> SimpleNamespace:
    step = chat.user("I have no service.")
    assert (step.action, step.function) == ("call", "get_bills_for_customer"), "the bills are read before one is picked"
    step = chat.tool("get_bills_for_customer", step.arguments, BILLS % "Overdue")
    assert (step.function, step.arguments) == ("send_payment_request", {"customer_id": "CA", "bill_id": "BA"})
    step = chat.tool("send_payment_request", step.arguments, "Payment request sent to the customer for bill BA")
    assert step.action == "ask" and "not count it as complete" in step.instruction
    return chat.user("I see a payment request for 50 USD.", after="Please check your payment requests.")


def test_what_the_policy_puts_first_runs_first_and_must_be_read_as_complete(monkeypatch):
    chat = Chat(monkeypatch, ["Line Suspension"])
    step = request_payment(chat)
    assert (step.action, step.function) == ("call", "get_details_by_id")

    step = chat.tool("get_details_by_id", {"id": "BA"}, '{"bill_id": "BA", "status": "Awaiting Payment"}')
    assert step.action == "ask" and "resume_line" not in chat.called(), "the request is sent but the bill is not paid"

    step = chat.user("Payment of 50 USD has been made for bill BA.", after="Please make the payment.")
    assert (step.action, step.function) == ("call", "get_details_by_id")

    step = chat.tool("get_details_by_id", {"id": "BA"}, '{"bill_id": "BA", "status": "Paid"}')
    assert (step.action, step.function) == ("call", "resume_line")

    step = chat.tool("resume_line", step.arguments, '{"message": "resumed"}')
    assert step.action == "ask" and step.policy == REBOOT, "the user is told what follows the change"
    assert chat.program.proc is not None

    step = chat.user("I restarted the phone and the speed test says excellent.", after="Please restart your phone and run run_speed_test().")
    assert chat.program.proc is None and step.status == "fixed"


def test_an_unpaid_bill_never_lets_the_line_be_resumed(monkeypatch):
    chat = Chat(monkeypatch, ["Line Suspension"])
    step = request_payment(chat)
    for _ in range(MAX_PROCEDURE_TURNS):
        if chat.program.proc is None:
            break
        if step.action == "call":
            step = chat.tool(step.function, step.arguments, '{"bill_id": "BA", "status": "Awaiting Payment"}')
        else:
            step = chat.user("I will pay later.", after="Please make the payment.")
    assert chat.program.proc is None and chat.program.status[0] == "procedure not completed"
    assert "resume_line" not in chat.called()


def test_another_paid_bill_in_a_list_does_not_complete_the_payment(monkeypatch):
    listed = {**PLANS, "Overdue Bill Payment": Procedure("Overdue Bill Payment", {"send_payment_request": SEND}, check=PAID, check_reads="get_bills_for_customer")}
    chat = Chat(monkeypatch, ["Line Suspension"], plans=listed)
    step = chat.user("I have no service.")
    chat.tool("send_payment_request", step.arguments, "Payment request sent to the customer for bill BA")
    step = chat.user("I see a payment request.", after="Please check your payment requests.")
    step = chat.tool("get_bills_for_customer", step.arguments, BILLS % "Awaiting Payment")
    assert step.action == "ask" and "resume_line" not in chat.called(), "bill BB is paid, but the request was for bill BA"

    step = chat.user("Paid.", after="Please make the payment.")
    step = chat.tool("get_bills_for_customer", step.arguments, BILLS % "Paid")
    assert step.function == "resume_line"


def test_the_turn_cap_ends_a_procedure_that_makes_no_progress(monkeypatch):
    chat = Chat(monkeypatch, ["Data Refueling"])
    chat.user("My data stopped working, add 3 GB.")
    step = chat.user("Yes.", after="Do you confirm?")
    for _ in range(MAX_PROCEDURE_TURNS):
        if chat.program.proc is None:
            break
        step = chat.user("Is it done yet?", after="One moment.")
    assert chat.program.proc is None and step.status == "procedure not completed"
