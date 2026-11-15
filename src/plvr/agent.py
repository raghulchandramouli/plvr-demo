"""The τ² agent for both arms: the student acts after its own reasoning (vanilla) or after the program's (program)."""

import uuid
from dataclasses import asdict, dataclass, field
from typing import Optional, Protocol

from pydantic import BaseModel
from tau2.agent.base_agent import HalfDuplexAgent, ValidAgentInputMessage
from tau2.agent.llm_agent import AGENT_INSTRUCTION, SYSTEM_PROMPT
from tau2.data_model.message import (
    APICompatibleMessage,
    AssistantMessage,
    Message,
    MultiToolMessage,
    SystemMessage,
    ToolCall,
)
from tau2.environment.tool import Tool
from tau2.utils.llm_utils import to_litellm_messages

from plvr.llm import Server

SAMPLING = {"temperature": 0.6, "top_p": 0.95, "top_k": 20}
MAX_STUDENT_TOKENS = 8192


class Thinker(Protocol):
    """Writes the reasoning the student acts on, given the conversation so far."""

    def think(self, history: list[APICompatibleMessage]) -> str: ...


@dataclass
class TurnLog:
    reasoning: str
    injected: bool
    prompt_tokens: int
    cached_tokens: int
    completion_tokens: int
    seconds: float
    finish_reason: str | None


@dataclass
class TaskLog:
    turns: list[TurnLog] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {"turns": [asdict(t) for t in self.turns]}


class StudentState(BaseModel):
    messages: list[APICompatibleMessage]


class StudentAgent(HalfDuplexAgent[StudentState]):
    def __init__(self, tools: list[Tool], domain_policy: str, student: Server, thinker: Thinker | None):
        super().__init__(tools=tools, domain_policy=domain_policy)
        self.student = student
        self.thinker = thinker
        self.log = TaskLog()
        self._system = SystemMessage(
            role="system", content=SYSTEM_PROMPT.format(agent_instruction=AGENT_INSTRUCTION, domain_policy=domain_policy)
        )
        self._tool_schemas = [t.openai_schema for t in tools]

    def get_init_state(self, message_history: Optional[list[Message]] = None) -> StudentState:
        return StudentState(messages=list(message_history or []))

    def generate_next_message(
        self, message: ValidAgentInputMessage, state: StudentState
    ) -> tuple[AssistantMessage, StudentState]:
        if isinstance(message, MultiToolMessage):
            state.messages.extend(message.tool_messages)
        else:
            state.messages.append(message)

        prompt = self.student.render(
            to_litellm_messages([self._system, *state.messages]), self._tool_schemas, thinking=True
        )
        reasoning = None
        if self.thinker is not None:
            reasoning = self.thinker.think(state.messages)
            prompt += f"<think>\n{reasoning}\n</think>\n\n"

        out = self.student.complete(prompt, max_tokens=MAX_STUDENT_TOKENS, **SAMPLING)
        self.log.turns.append(
            TurnLog(
                reasoning=reasoning if reasoning is not None else out.reasoning,
                injected=reasoning is not None,
                prompt_tokens=out.prompt_tokens,
                cached_tokens=out.cached_tokens,
                completion_tokens=out.completion_tokens,
                seconds=out.seconds,
                finish_reason=out.finish_reason,
            )
        )
        tool_calls = [
            ToolCall(id=f"call_{uuid.uuid4().hex[:12]}", name=c.name, arguments=c.arguments, requestor="assistant")
            for c in out.tool_calls
        ]
        reply = AssistantMessage(
            role="assistant",
            content=out.text or None,
            tool_calls=tool_calls or None,
            cost=0.0,
            usage={"prompt_tokens": out.prompt_tokens, "completion_tokens": out.completion_tokens},
        )
        state.messages.append(reply)
        return reply, state
