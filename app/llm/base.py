"""Common LLM interface for the function-calling layer (Phase 8).

Both evaluated models implement the same interface so the chatbot and the
evaluation runner are model-agnostic. Concrete clients live alongside this file
(e.g. :mod:`app.llm.groq_client`).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

from app.tools.models import ToolCall


@dataclass
class ToolCallDecision:
    """The model's tool-selection step output."""

    tool_call: ToolCall | None
    made_tool_call: bool
    raw_content: str | None = None
    latency_ms: float = 0.0
    parse_error: str | None = None


class LLMClient(ABC):
    """Abstract function-calling client."""

    def __init__(self, model_name: str, temperature: float = 0.0) -> None:
        self.model_name = model_name
        # Hard-coded to 0.0 by concrete clients for reproducible decoding.
        self.temperature = temperature

    @abstractmethod
    def generate_tool_call(self, user_query: str, tools: list[dict[str, Any]]) -> ToolCallDecision:
        """Ask the model to pick ONE tool and its arguments for ``user_query``.

        ``tools`` are OpenAI/Ollama-format tool schemas. Implementations should
        prefer native tool calling and fall back to strict JSON parsing.
        """

    @abstractmethod
    def generate_final_answer(
        self, user_query: str, tool_name: str, tool_result_json: str
    ) -> tuple[str, float]:
        """Produce the grounded final answer. Returns (answer, latency_ms)."""
