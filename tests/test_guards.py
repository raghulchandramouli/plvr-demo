import json
from types import SimpleNamespace

from tau2.data_model.message import AssistantMessage, ToolCall, ToolMessage, UserMessage

from plvr.agent import StudentAgent
from plvr.llm import Completion, ToolCall as Call
from plvr.primitive import Runner
from plvr.telecom import guards as G

POLICY = """<main_policy>
# Agent Policy
The current time is 2025-02-25 12:08:00 EST.
## Line Suspension
You are allowed to lift the suspension after the user has paid all their overdue bills.
## Overdue Bill Payment
The send payment request tool will not check if the bill is overdue.
</main_policy>
<tech_support_policy>
# Service
## Airplane Mode
If it is ON, guide the user to use `toggle_airplane_mode()` to turn it OFF.
</tech_support_policy>"""


def tool(name: str, description: str):
    return SimpleNamespace(name=name, openai_schema={"function": {"description": description}})


TOOLS = [
    tool("get_bills_for_customer", "Retrieves a list of the customer's bills."),
    tool("resume_line", "Resumes a suspended line."),
]


class Brain:
    """Answers each primitive from a table keyed by the STEP line, and keeps the prompts."""

    def __init__(self, answers):
        self.answers, self.prompts = answers, []

    def render(self, messages, tools=None, *, thinking):
        return messages[-1]["content"]

    def complete(self, prompt, **kwargs):
        self.prompts.append(prompt)
        step = prompt.splitlines()[0].removeprefix("STEP: ")
        reply = self.answers[step]
        text = reply(prompt) if callable(reply) else reply
        return Completion(text, "", [], 10, 0, 5, 0.0, "stop")


CHANGES = '{"changes_state": true, "evidence": "Resumes a suspended line."}'
READS = '{"changes_state": false}'
PRECONDITION = (
    '{"requirements": [{"condition": "the user has paid all their overdue bills", "must_hold": true, "about": "records"},'
    ' {"condition": "the user agrees", "must_hold": true, "about": "conversation"}]}'
)


def plan_with(tmp_path, monkeypatch, met):
    monkeypatch.setattr(G, "CACHE_DIR", tmp_path)
    brain = Brain(
        {
            "Does the function change stored data": lambda p: CHANGES if "FUNCTION:\nresume_line" in p else READS,
            "Sections that govern a function": '{"titles": ["Line Suspension"]}',
            "What must hold before the function is called": PRECONDITION,
            "Is the condition met": met,
        }
    )
    return G.compile_plan(POLICY, TOOLS, Runner(brain)), brain


def test_plan_reads_write_tools_and_record_preconditions_from_the_policy(tmp_path, monkeypatch):
    plan, _ = plan_with(tmp_path, monkeypatch, '{"answer": "cannot tell"}')
    assert list(plan.state_changing) == ["resume_line"]
    [pre] = plan.preconditions["resume_line"]
    assert pre.condition == "the user has paid all their overdue bills"
    assert pre.sentence.startswith("You are allowed to lift the suspension after")
    assert any("conversation" in r for r in plan.rejected)
    assert list(tmp_path.glob("telecom-guards-*.json"))


def test_a_sentence_that_only_describes_is_not_a_precondition():
    assert G._permits_or_forbids("You are allowed to lift the suspension after the user has paid all their overdue bills.")
    assert G._permits_or_forbids("A user can only have one bill in the AWAITING PAYMENT status at a time.")
    assert not G._permits_or_forbids("If a line is roaming enabled, the user can use their phone's data connection abroad.")
    assert not G._permits_or_forbids("The user has an overdue bill.")


def test_preconditions_are_read_only_from_the_sections_that_govern_the_function(tmp_path, monkeypatch):
    _, brain = plan_with(tmp_path, monkeypatch, '{"answer": "cannot tell"}')
    [asked] = [p for p in brain.prompts if p.startswith("STEP: What must hold before the function is called")]
    assert "lift the suspension" in asked
    assert "send payment request tool" not in asked and "The current time is" not in asked


def history(bill_status: str):
    bills = ToolCall(id="c1", name="get_bills_for_customer", arguments={"customer_id": "C1"}, requestor="assistant")
    result = ToolMessage(id="c1", role="tool", content=json.dumps([{"bill_id": "B1", "status": bill_status}]), requestor="assistant")
    return [UserMessage(role="user", content="Please restore my line"), AssistantMessage(role="assistant", tool_calls=[bills]), result]


def guard(tmp_path, monkeypatch, met):
    plan, brain = plan_with(tmp_path, monkeypatch, met)
    return G.Guard(TOOLS, POLICY, plan, Runner(brain)), brain


RESUME = Call("resume_line", {"line_id": "L1"})
UNPAID = '{"evidence": "\\"status\\": \\"Overdue\\"", "answer": "no"}'
PAID = '{"evidence": "\\"status\\": \\"Paid\\"", "answer": "yes"}'


def test_unmet_precondition_is_blocked_with_the_policys_words(tmp_path, monkeypatch):
    g, brain = guard(tmp_path, monkeypatch, UNPAID)
    verdict = g.check([RESUME], history("Overdue"))
    assert verdict.violation.kind == "precondition"
    assert "You are allowed to lift the suspension after the user has paid all their overdue bills" in verdict.violation.note
    record = brain.prompts[-1]
    assert "2025-02-25" in record and "Overdue" in record and "resume_line" in record


def test_met_precondition_and_read_calls_pass(tmp_path, monkeypatch):
    g, _ = guard(tmp_path, monkeypatch, PAID)
    assert g.check([RESUME], history("Paid")).violation is None
    assert g.check([Call("get_bills_for_customer", {})], history("Overdue")).violation is None


def test_cannot_tell_passes_and_is_reported_unless_unverified_blocks(tmp_path, monkeypatch):
    g, _ = guard(tmp_path, monkeypatch, '{"answer": "cannot tell"}')
    verdict = g.check([RESUME], [])
    assert verdict.violation is None and verdict.unverified[0]["call"] == "resume_line"
    g.block_unverified = True
    assert g.check([RESUME], []).violation is not None


def test_phone_function_is_not_a_tool_and_is_told_so():
    g = G.Guard(TOOLS, POLICY)
    v = g.check([Call("toggle_airplane_mode", {})], []).violation
    assert v.kind == "tool_name" and "runs on the user's phone" in v.note
    assert "toggle_airplane_mode()" in v.fallback_text
    other = g.check([Call("no_such_thing", {})], []).violation
    assert "runs on the user's phone" not in other.note
    assert g.check([Call("resume_line", {})], []).violation is None


class Student:
    """Plays scripted completions and keeps each prompt it was given."""

    def __init__(self, outputs):
        self.outputs, self.prompts = list(outputs), []

    def render(self, messages, tools=None, *, thinking):
        return "\n".join(f"{m['role']}: {m['content']}" for m in messages)

    def complete(self, prompt, **kwargs):
        self.prompts.append(prompt)
        calls, text = self.outputs.pop(0)
        return Completion(text, "r", calls, 10, 0, 5, 0.0, "stop")


PHONE = [Call("toggle_airplane_mode", {})]


def agent_with(outputs, g):
    student = Student(outputs)
    agent = StudentAgent(TOOLS, POLICY, student, None, guard=g)
    state = agent.get_init_state()
    reply, state = agent.generate_next_message(UserMessage(role="user", content="My phone has no service"), state)
    return agent, student, reply


def test_phone_call_is_resampled_with_a_reason_then_sent_when_valid():
    agent, student, reply = agent_with([(PHONE, ""), ([], "Please run toggle_airplane_mode().")], G.Guard(TOOLS, POLICY))
    assert reply.tool_calls is None and "Please run" in reply.content
    assert len(student.prompts) == 2
    assert "runs on the user's phone" in student.prompts[1] and "runs on the user's phone" not in student.prompts[0]
    assert [e["outcome"] for e in agent.log.guard_events] == ["resampled", "sent"]
    assert agent.log.guard_events[0]["call"] == "toggle_airplane_mode"
    assert len(agent.log.turns) == 1
    assert len(agent.log.as_dict()["guard_events"]) == 2


def test_after_two_resamples_the_fallback_asks_the_user_instead_of_calling():
    agent, student, reply = agent_with([(PHONE, "")] * 3, G.Guard(TOOLS, POLICY))
    assert len(student.prompts) == 3
    assert reply.tool_calls is None
    assert reply.content == "Could you please run toggle_airplane_mode() on your phone and tell me exactly what it shows?"
    assert [e["outcome"] for e in agent.log.guard_events] == ["resampled", "resampled", "fallback"]


def test_precondition_violation_resamples_and_falls_back(tmp_path, monkeypatch):
    g, _ = guard(tmp_path, monkeypatch, UNPAID)
    agent = StudentAgent(TOOLS, POLICY, Student([([RESUME], "")] * 3), None, guard=g)
    state = agent.get_init_state(history("Overdue"))
    reply, _ = agent.generate_next_message(UserMessage(role="user", content="Restore it"), state)
    assert reply.tool_calls is None and "Our policy says" in reply.content
    assert [e["guard"] for e in agent.log.guard_events] == ["precondition"] * 3
    assert agent.log.guard_events[-1]["outcome"] == "fallback"


def test_guard_off_sends_the_call_unchanged_and_logs_nothing_new():
    agent, student, reply = agent_with([(PHONE, "")], None)
    assert reply.tool_calls[0].name == "toggle_airplane_mode"
    assert len(student.prompts) == 1 and "Automatic check" not in student.prompts[0]
    assert "guard_events" not in agent.log.as_dict()


def test_build_guard_modes():
    assert G.build_guard("off", POLICY, TOOLS, None) is None
    tools_only = G.build_guard("tools", POLICY, TOOLS, None)
    assert tools_only.plan is None and tools_only.tool_names == ["get_bills_for_customer", "resume_line"]
