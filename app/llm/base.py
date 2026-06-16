"""Common LLM interface for multi-step tool calling.

Both evaluated models implement the same conversational interface so the
chatbot and the evaluation runner are model-agnostic. The chatbot drives a
multi-step loop: it calls :meth:`LLMClient.chat`, runs any requested tools,
appends their results to the message history, and calls ``chat`` again until the
model returns a final answer instead of tool calls.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from app.tools.models import ToolCall


@dataclass
class PlannedToolCall:
    """A single tool call the model wants to make, with its provider-side id."""

    id: str
    call: ToolCall


@dataclass
class AssistantTurn:
    """One assistant step: either tool calls to run, or a final answer."""

    planned_calls: list[PlannedToolCall] = field(default_factory=list)
    content: str | None = None
    latency_ms: float = 0.0
    # The raw assistant message to append to the running history (so tool
    # results can reference the tool_call ids on the next turn).
    assistant_message: dict[str, Any] = field(default_factory=dict)
    request_error: str | None = None

    @property
    def wants_tools(self) -> bool:
        return bool(self.planned_calls)


class LLMClient(ABC):
    """Abstract multi-step function-calling client."""

    def __init__(self, model_name: str, temperature: float = 0.0) -> None:
        self.model_name = model_name
        # Hard-coded to 0.0 by concrete clients for reproducible decoding.
        self.temperature = temperature

    @abstractmethod
    def chat(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> AssistantTurn:
        """One turn: given the history + tools, return tool calls or an answer."""
