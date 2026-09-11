"""Reasoning delegation backends for full-duplex providers (RFC §12.4.1).

A full-duplex model (OpenAI GPT-Live) holds the conversation and hands
reasoning and tool use to a backend. In the integrator mode that backend is
the application's: :class:`~roomkit.channels.realtime_voice.RealtimeVoiceChannel`
serves each delegation through a :class:`ReasoningBackend`, hands it the
transcript recorded since the previous one — the model sends no task text —
and returns every output to the model as spoken or silent context.

:class:`AIProviderReasoningBackend` is the default: any
:class:`~roomkit.providers.ai.base.AIProvider` run through a small
generate → tools → generate loop, its tool calls executed through the
channel's own gate.
"""

from __future__ import annotations

import json
import logging
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator, Awaitable, Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

from roomkit.providers.ai.base import (
    AIContext,
    AIMessage,
    AITextPart,
    AIThinkingPart,
    AITool,
    AIToolCallPart,
    AIToolResultPart,
)

if TYPE_CHECKING:
    from roomkit.providers.ai.base import AIProvider
    from roomkit.voice.base import VoiceSession

logger = logging.getLogger("roomkit.voice.realtime.reasoning")

ToolExecutor = Callable[[str, dict[str, Any]], Awaitable[str]]
"""``(name, arguments) -> result`` — runs one tool call through the channel's gate."""

DEFAULT_TRANSCRIPT_INSTRUCTION = "Act on the user's most recent request in the conversation above."


@dataclass(frozen=True)
class TranscriptLine:
    """One speaker's contribution in the transcript handed to a backend."""

    role: Literal["user", "assistant"]
    text: str


@dataclass
class ReasoningRequest:
    """What a backend receives for one delegation (RFC §12.4.1).

    Attributes:
        session: The voice session whose model delegated.
        delegation_id: Opaque provider identifier; returned unchanged with
            every output.
        transcript: User and assistant lines recorded since the previous
            request — the backend's only account of what was asked.
        first: Whether this is the session's first request, in which case
            the transcript is the whole conversation so far.
        tools: The channel's declared tool catalogue, as tool dicts. The
            backend offers these to its model.
        execute_tool: Runs one tool call through the channel's pre-execution
            gate (declared catalogue, argument schema, skill gating,
            ``BEFORE_TOOL_USE``, ``ON_TOOL_CALL``) and returns the result
            text. A backend MUST route its tool calls through it.
    """

    session: VoiceSession
    delegation_id: str
    transcript: list[TranscriptLine]
    first: bool
    tools: list[dict[str, Any]] = field(default_factory=list)
    execute_tool: ToolExecutor | None = None


@dataclass(frozen=True)
class ReasoningOutput:
    """One piece of a backend's output, on its way to the model.

    Attributes:
        text: What the backend produced.
        spoken: ``True`` asks the model to relay it to the user in its own
            words; ``False`` adds it as silent context the model may draw on.
        is_final: The answer to the request, as opposed to progress toward
            it.
    """

    text: str
    spoken: bool = True
    is_final: bool = False


class ReasoningBackend(ABC):
    """The backend a full-duplex model's integrator-side delegation goes to.

    Implementations work out the request from ``request.transcript``, answer
    it with their own model, context and tools, and yield
    :class:`ReasoningOutput` as they go. Tool calls MUST go through
    ``request.execute_tool`` so the channel's gate applies (RFC §12.4.1).
    """

    @abstractmethod
    def run(self, request: ReasoningRequest) -> AsyncIterator[ReasoningOutput]:
        """Serve one delegation, yielding outputs as they are produced."""
        ...

    async def session_ended(self, session_id: str) -> None:  # noqa: B027
        """Release any state kept for a session that just ended."""

    async def close(self) -> None:  # noqa: B027
        """Release all resources."""


def render_transcript_request(
    transcript: list[TranscriptLine],
    *,
    first: bool,
    instruction: str = DEFAULT_TRANSCRIPT_INSTRUCTION,
) -> str:
    """Render the transcript as one labelled message for a text model.

    Flattening the conversation into one user message keeps the two
    conversations apart: the backend's own context holds only what the
    backend itself said as assistant messages, so it never mistakes the
    voice model's speech for its own.
    """
    lines: list[str] = []
    if transcript:
        lines.append(
            "Voice conversation so far:"
            if first
            else "Voice conversation since the previous delegation:"
        )
        lines.extend(f"{line.role.upper()}: {line.text}" for line in transcript if line.text)
        lines.append("")
    lines.append(instruction)
    return "\n".join(lines)


class AIProviderReasoningBackend(ReasoningBackend):
    """Default backend: an :class:`~roomkit.providers.ai.base.AIProvider` in a tool loop.

    Each request becomes one user message carrying the transcript; the loop
    runs the model, executes the tool calls it makes through
    ``request.execute_tool``, and runs it again until it answers in text or
    the round cap is reached. Text the model produces before a tool round is
    yielded as progress — silent by default, spoken with
    ``spoken_progress=True`` — and the final answer is yielded spoken.

    The backend keeps its own conversation per voice session, so a later
    delegation sees what it worked out for an earlier one; the channel
    releases it when the session ends.

    Example:
        backend = AIProviderReasoningBackend(
            AnthropicAIProvider(AnthropicConfig(api_key="...")),
            system_prompt=BACKEND_PROMPT,
        )
        channel = RealtimeVoiceChannel(
            "voice", provider=live, transport=transport,
            tools=[check_flight, rebook_flight], reasoning_backend=backend,
        )
    """

    def __init__(
        self,
        provider: AIProvider,
        *,
        system_prompt: str | None = None,
        max_tool_rounds: int = 5,
        temperature: float | None = None,
        spoken_progress: bool = False,
    ) -> None:
        if max_tool_rounds < 0:
            raise ValueError("max_tool_rounds must not be negative")
        self._provider = provider
        self._system_prompt = system_prompt
        self._max_tool_rounds = max_tool_rounds
        self._temperature = temperature
        self._spoken_progress = spoken_progress
        self._histories: dict[str, list[AIMessage]] = {}

    @property
    def provider(self) -> AIProvider:
        """The model behind this backend."""
        return self._provider

    async def run(self, request: ReasoningRequest) -> AsyncIterator[ReasoningOutput]:
        history = self._histories.setdefault(request.session.id, [])
        history.append(
            AIMessage(
                role="user",
                content=render_transcript_request(request.transcript, first=request.first),
            )
        )
        tools = [
            AITool(
                name=t["name"],
                description=str(t.get("description", "")),
                parameters=dict(t.get("parameters") or {}),
            )
            for t in request.tools
            if isinstance(t, dict) and t.get("name")
        ]

        for round_idx in range(self._max_tool_rounds + 1):
            context = AIContext(
                messages=list(history),
                system_prompt=self._system_prompt,
                tools=tools,
            )
            if self._temperature is not None:
                context.temperature = self._temperature
            response = await self._provider.generate(context)
            text = (response.content or "").strip()

            if not response.tool_calls or round_idx == self._max_tool_rounds:
                if response.tool_calls:
                    logger.warning(
                        "Reasoning backend hit the %d-round tool cap for delegation %s",
                        self._max_tool_rounds,
                        request.delegation_id,
                    )
                history.append(AIMessage(role="assistant", content=text or "(no answer)"))
                if text:
                    yield ReasoningOutput(text=text, spoken=True, is_final=True)
                return

            parts: list[Any] = []
            if response.thinking:
                parts.append(
                    AIThinkingPart(
                        thinking=response.thinking, signature=response.thinking_signature
                    )
                )
            if text:
                parts.append(AITextPart(text=text))
            parts.extend(
                AIToolCallPart(id=tc.id, name=tc.name, arguments=tc.arguments)
                for tc in response.tool_calls
            )
            history.append(AIMessage(role="assistant", content=parts))
            if text:
                yield ReasoningOutput(text=text, spoken=self._spoken_progress, is_final=False)

            results: list[Any] = []
            for tc in response.tool_calls:
                result = await self._execute(request, tc.name, tc.arguments)
                results.append(AIToolResultPart(tool_call_id=tc.id, name=tc.name, result=result))
            history.append(AIMessage(role="tool", content=results))

    async def _execute(
        self, request: ReasoningRequest, name: str, arguments: dict[str, Any]
    ) -> str:
        if request.execute_tool is None:
            return json.dumps({"error": f"No tool executor available for {name}"})
        try:
            return await request.execute_tool(name, arguments)
        except Exception as exc:
            logger.exception("Reasoning backend tool %s failed", name)
            return json.dumps({"error": f"Tool {name} failed: {exc}"})

    async def session_ended(self, session_id: str) -> None:
        self._histories.pop(session_id, None)

    async def close(self) -> None:
        self._histories.clear()
