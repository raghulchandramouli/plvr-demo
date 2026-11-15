"""Procedures: what the compile step reads from a policy section, and how the program carries one out turn by turn.

The runner model is a fake that answers each primitive from a script, so every verifier and fallback runs for real.
"""

import json
from dataclasses import asdict
from types import SimpleNamespace

import pytest
from tau2.data_model.message import AssistantMessage, ToolCall, ToolMessage, UserMessage

from plvr.llm import Completion
from plvr.primitive import Context, Runner
from plvr.telecom import program as program_module
from plvr.telecom.policy import Area, Compiled, Procedure, ProcedureStep, Rule, _load, changing_functions, named_functions, plan_procedures
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


def given(prompt: str, name: str) -> str:
    """The text of one named input of a rendered primitive prompt."""
    return prompt.split(f"{name}:\n", 1)[1].split("\n\n", 1)[0]


def by_input(name: str, table: dict, default=None):
    """Answer by the first key of `table` found in the named input."""
    return lambda prompt: next((answer for key, answer in table.items() if key in given(prompt, name)), default or {})


# ---------------------------------------------------------------- compile time

REFUELING = """You can add more data to the line by "refueling" data at a price per GB specified by the plan.
To refuel data you should:
- Ask them how much data they want to refuel
- Confirm the price
- Apply the refueled data to the line associated with the phone number the user provided."""

SUSPENSION = """A line can be suspended for the following reasons:
- The user has an overdue bill.
You are allowed to lift the suspension after the user has paid all their overdue bills.
After you resume the line, the user will have to reboot their device to get service."""

PAYMENT = """- Send the user a payment request for the overdue bill.
- If the user accepts the payment request, use the make_payment tool to make the payment.
- Always check that the bill status is updated to PAID before informing the user that the bill has been paid."""

AGENT = {
    "refuel_data": "Refuels data for a specific line, adding to the customer's bill.",
    "resume_line": "Resumes a suspended line.",
    "send_payment_request": "Sends a payment request to the customer for a specific bill.",
    "get_details_by_id": "Retrieves the details for a given ID.",
}
CHANGES = {
    "Does the function change a record": by_input(
        "DESCRIPTION",
        {
            "Refuels": {"evidence": "Refuels data", "changes": True},
            "Resumes": {"evidence": "Resumes a suspended line", "changes": True},
            "Sends": {"evidence": "Sends a payment request", "changes": True},
            "Retrieves": {"evidence": "Retrieves the details", "changes": False},
        },
    )
}


def plan(sections: dict, answers: dict, deferred_to=frozenset()) -> dict[str, Procedure]:
    run = Runner(ScriptedServer({**CHANGES, **answers}))
    functions = {**AGENT, "make_payment": "", "reboot_device": ""}
    return plan_procedures(run, Context(), sections, changing_functions(run, Context(), AGENT), functions, set(deferred_to))


def test_a_function_is_state_changing_only_with_evidence_from_its_description():
    run = Runner(ScriptedServer({"Does the function change a record": {"evidence": "Deletes the line", "changes": True}}))
    assert changing_functions(run, Context(), AGENT) == {}
    assert sorted(changing_functions(Runner(ScriptedServer(CHANGES)), Context(), AGENT)) == ["refuel_data", "resume_line", "send_payment_request"]


def test_a_procedures_effects_are_the_changing_functions_its_steps_use():
    plans = plan(
        {"Data Refueling": REFUELING},
        {
            "Copy the procedure's steps": {
                "steps": [
                    "Ask them how much data they want to refuel",
                    "Confirm the price",
                    "Offer a discount",
                    "Apply the refueled data to the line associated with the phone number the user provided.",
                ]
            },
            # The picker wrongly gives the asking step the changing function too.
            "Pick the function": by_input("GOAL", {"Ask them": {"function": "refuel_data"}, "Apply": {"function": "refuel_data"}}, {"function": "none"}),
        },
    )
    refueling = plans["Data Refueling"]
    assert [(s.text[:5], s.function) for s in refueling.steps] == [("Ask t", None), ("Confi", None), ("Apply", "refuel_data")]
    assert refueling.effects == ["refuel_data"]


def test_a_read_only_function_is_never_an_effect():
    plans = plan(
        {"Data Refueling": REFUELING},
        {"Copy the procedure's steps": {"steps": ["Confirm the price"]}, "Pick the function": {"function": "get_details_by_id"}},
    )
    assert plans == {}


def test_a_section_without_steps_gets_its_change_only_when_a_rule_defers_to_it():
    answers = {"Pick the function": {"function": "resume_line"}}
    assert plan({"Line Suspension": SUSPENSION}, answers) == {}
    deferred = plan({"Line Suspension": SUSPENSION}, answers, {"Line Suspension"})["Line Suspension"]
    assert deferred.effects == ["resume_line"]
    assert [(s.text, s.function) for s in deferred.steps] == [("Resumes a suspended line.", "resume_line")]


def test_what_comes_first_needs_the_sections_own_words():
    steps = by_input(
        "SECTION",
        {
            "## Line Suspension": {"steps": ["You are allowed to lift the suspension after the user has paid all their overdue bills."]},
            "## Overdue Bill Payment": {"steps": ["Send the user a payment request for the overdue bill."]},
        },
    )
    pick = by_input("GOAL", {"lift the suspension": {"function": "resume_line"}, "payment request": {"function": "send_payment_request"}})
    sections = {"Line Suspension": SUSPENSION, "Overdue Bill Payment": PAYMENT}
    quoted = by_input(
        "SECTION", {"## Line Suspension": {"first": [{"title": "Overdue Bill Payment", "quote": "after the user has paid all their overdue bills"}]}}, {"first": []}
    )
    plans = plan(sections, {"Copy the procedure's steps": steps, "Pick the function": pick, "What must be done first": quoted})
    assert plans["Line Suspension"].requires == ["Overdue Bill Payment"]
    assert plans["Overdue Bill Payment"].requires == []

    invented = {"first": [{"title": "Overdue Bill Payment", "quote": "the bill must be settled first"}]}
    plans = plan(sections, {"Copy the procedure's steps": steps, "Pick the function": pick, "What must be done first": invented})
    assert plans["Line Suspension"].requires == []


def test_names_a_procedure_writes_that_are_not_agent_tools_run_on_the_users_side():
    assert named_functions(PAYMENT + " transfer_to_human_agents", {"transfer_to_human_agents": ""}) == ["make_payment"]


def test_compiled_plans_survive_the_cache():
    compiled = Compiled({}, {}, [], [], {}, [], "", plans={"T": Procedure("T", [ProcedureStep("do it", "f")], ["f"], ["U"])}, state_changing=["f"])
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
]
PLANS = {
    "Data Refueling": Procedure(
        "Data Refueling",
        [
            ProcedureStep("Ask them how much data they want to refuel"),
            ProcedureStep("Confirm the price"),
            ProcedureStep("Apply the refueled data to the line", "refuel_data"),
        ],
        ["refuel_data"],
    ),
    "Line Suspension": Procedure(
        "Line Suspension",
        [
            ProcedureStep("lift the suspension after the user has paid all their overdue bills", "resume_line"),
            ProcedureStep("the user will have to reboot their device", "reboot_device"),
        ],
        ["resume_line"],
        ["Overdue Bill Payment"],
    ),
    "Overdue Bill Payment": Procedure(
        "Overdue Bill Payment",
        [
            ProcedureStep("Send the user a payment request for the overdue bill.", "send_payment_request"),
            ProcedureStep("use the make_payment tool to make the payment", "make_payment"),
            ProcedureStep("Always check that the bill status is updated to PAID", "get_details_by_id"),
        ],
        ["send_payment_request"],
    ),
}
CONDITION = "the rule applies"
FILL = by_input(
    "FUNCTION",
    {
        "refuel_data": {"arguments": {"customer_id": "CA", "line_id": "LA", "gb_amount": 3}},
        "resume_line": {"arguments": {"customer_id": "CA", "line_id": "LA"}},
        "send_payment_request": {"arguments": {"customer_id": "CA", "bill_id": "BA"}},
    },
)
ANSWERS = {
    "Is the condition met": by_input("CONDITION", {CONDITION: {"evidence": "customer", "answer": "yes"}}),
    "Fill the parameters": FILL,
    "Reply to a confirmation": by_input("USER_MESSAGE", {"No thanks": {"answer": "no"}}, {"answer": "unclear"}),
    "Device reply": lambda prompt: {
        "reports": [{"function": f, "result": given(prompt, "USER_MESSAGE")} for f in ("make_payment", "reboot_device") if f in given(prompt, "ASKED")]
    },
}


class Chat:
    """Plays the student and the user against the program, and returns the step the program decides after each."""

    def __init__(self, monkeypatch, procedures: list[str], answers: dict | None = None):
        rule = Rule("Data", "Section", CONDITION, None, None, {}, False, procedures=procedures)
        compiled = Compiled(
            areas={"Data": Area("Data", "run_speed_test", "", [], [rule])},
            procedures={title: f"{title} at a price per GB" for title in [*PLANS, "Change Plan"]},
            device_functions=["make_payment", "reboot_device", "run_speed_test"],
            agent_functions=[t.name for t in TOOLS],
            descriptions={t.name: "" for t in TOOLS},
            rejected=[],
            preamble="",
            plans=PLANS,
            state_changing=["refuel_data", "resume_line", "send_payment_request"],
        )
        monkeypatch.setattr(program_module, "compile_policy", lambda *_: compiled)
        self.program = program = TelecomProgram("", TOOLS, ScriptedServer({**ANSWERS, **(answers or {})}))
        program.area, program.checklist, program.phone = "Data", compiled.checklist("Data"), "555"
        program.customer = {"customer_id": "CA", "line_ids": ["LA"], "bill_ids": ["BA"]}
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


def test_a_procedure_runs_one_step_a_turn_and_ends_when_its_change_succeeds(monkeypatch):
    price = {"quantity": 3, "unit_price": 2.5, "quantity_quote": "3 GB", "unit_price_quote": "price_per_gb: 2.5"}
    chat = Chat(
        monkeypatch,
        ["Data Refueling"],
        {
            "Price terms": price,
            "Where a value comes from": {"value": "PA", "quote": "plan_id: PA"},
            "Pick the function": {"function": "get_details_by_id"},
            "Fill the parameters": lambda prompt: {"arguments": {"id": "PA"}} if "get_details_by_id" in given(prompt, "FUNCTION") else FILL(prompt),
        },
    )

    step = chat.user("My data stopped working.")
    assert (step.action, step.function) == ("ask", None) and "how much data" in step.instruction

    step = chat.user("3 GB please.", after="How much data would you like to add?")
    assert (step.action, step.function, step.arguments) == ("call", "get_details_by_id", {"id": "PA"})

    step = chat.tool("get_details_by_id", {"id": "PA"}, '{"plan_id": "PA", "price_per_gb": 2.5}')
    assert step.action == "ask" and "Confirm the price" in step.instruction
    assert "3 at 2.5 each is 7.50 in total" in step.policy

    step = chat.user("Yes, go ahead.", after="That is 7.50 in total. Do you confirm?")
    assert (step.action, step.function) == ("call", "refuel_data")
    assert step.arguments == {"customer_id": "CA", "line_id": "LA", "gb_amount": 3}
    assert chat.program.proc is not None

    step = chat.tool("refuel_data", step.arguments, '{"message": "added"}')
    assert chat.program.proc is None and step.status == "fixed"
    assert step.function == "run_speed_test", "the change is made, so the program goes back to the problem's own check"


def test_a_missing_value_is_asked_for_before_the_change_is_called(monkeypatch):
    chat = Chat(monkeypatch, ["Data Refueling"])
    chat.user("My data stopped working.")
    chat.user("Some more data please.", after="How much data would you like to add?")
    step = chat.user("OK.", after="I cannot quote a price yet. Shall I continue?")
    assert (step.action, step.function) == ("ask", None) and "information needed" in step.instruction, "no amount was stated"

    step = chat.user("3 GB.", after="How many GB?")
    assert (step.action, step.arguments["gb_amount"]) == ("call", 3)


def test_a_change_the_student_makes_early_ends_the_procedure(monkeypatch):
    chat = Chat(monkeypatch, ["Data Refueling"])
    chat.user("My data stopped working, add 3 GB.")
    step = chat.tool("refuel_data", {"customer_id": "CA", "line_id": "LA", "gb_amount": 3}, '{"message": "added"}')
    assert chat.program.proc is None and step.status == "fixed"
    assert all(s["step"]["function"] != "refuel_data" for s in chat.program.trace)


def test_a_user_who_declines_ends_the_procedure_without_the_change(monkeypatch):
    chat = Chat(monkeypatch, ["Data Refueling"])
    chat.user("My data stopped working.")
    step = chat.user("No thanks, I do not want to pay for data.", after="How much data would you like to add?")
    assert chat.program.proc is None and step.status == "declined by the user"
    assert step.function == "run_speed_test", "the program moves on to the rest of the checklist"


def test_a_change_that_keeps_failing_ends_the_procedure(monkeypatch):
    chat = Chat(monkeypatch, ["Data Refueling"])
    chat.user("My data stopped working, add 3 GB.")
    chat.user("3 GB.", after="How much data would you like to add?")
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
    assert "how much data" in step.instruction, "the one procedure with a function is taken without asking the user to choose"


def suspension(monkeypatch, bill_after_payment: str) -> tuple[Chat, list[str]]:
    paid = {"evidence": "Paid", "answer": "yes"} if "Paid" in bill_after_payment else {"evidence": "Awaiting Payment", "answer": "no"}
    chat = Chat(
        monkeypatch,
        ["Line Suspension"],
        {
            "Is the condition met": by_input("CONDITION", {CONDITION: {"evidence": "customer", "answer": "yes"}, "updated to PAID": paid}),
            "Fill the parameters": lambda prompt: {"arguments": {"id": "BA"}} if "get_details_by_id" in given(prompt, "FUNCTION") else FILL(prompt),
        },
    )
    asked = []
    step = chat.user("I have no service.")
    asked.append(step.function)
    step = chat.tool("send_payment_request", step.arguments, "Payment request sent to the customer for bill BA")
    asked.append(step.function)
    step = chat.user("Payment of 50 USD has been made for bill BA.", after="Please use make_payment() and tell me what it shows.")
    asked.append(step.function)
    step = chat.tool("get_details_by_id", step.arguments, bill_after_payment)
    asked.append(step.function)
    return chat, asked


def test_what_the_policy_puts_first_runs_first_and_is_verified(monkeypatch):
    chat, asked = suspension(monkeypatch, '{"bill_id": "BA", "status": "Paid"}')
    assert asked == ["send_payment_request", "make_payment", "get_details_by_id", "resume_line"]

    step = chat.tool("resume_line", {"customer_id": "CA", "line_id": "LA"}, '{"message": "resumed"}')
    assert (step.action, step.function) == ("ask", "reboot_device"), "the step after the change is still carried out"
    assert chat.program.proc is not None

    step = chat.user("I restarted the phone.", after="Please restart your phone.")
    assert chat.program.proc is None and step.status == "fixed"


def test_an_unverified_payment_stops_before_the_line_is_resumed(monkeypatch):
    chat, asked = suspension(monkeypatch, '{"bill_id": "BA", "status": "Awaiting Payment"}')
    assert "resume_line" not in asked
    assert chat.program.proc is None and chat.program.status[0] == "procedure not completed"


def test_the_turn_cap_ends_a_procedure_that_makes_no_progress(monkeypatch):
    chat = Chat(monkeypatch, ["Data Refueling"])
    chat.user("My data stopped working.")
    chat.user("3 GB.", after="How much data would you like to add?")
    step = chat.user("Yes.", after="Do you confirm?")
    for _ in range(MAX_PROCEDURE_TURNS):
        if chat.program.proc is None:
            break
        step = chat.user("Is it done yet?", after="One moment.")
    assert chat.program.proc is None and step.status == "procedure not completed"
