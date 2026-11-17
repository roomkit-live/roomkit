"""Tests for the Google Gemini AI provider."""

from __future__ import annotations

import base64
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from roomkit.channels.ai import AIChannel
from roomkit.providers.ai.base import (
    AIContext,
    AIImagePart,
    AIMessage,
    AITextPart,
    AITool,
    ProviderError,
    StreamDone,
    StreamTextDelta,
    StreamToolCall,
    StreamUsage,
)
from roomkit.providers.ai.response_schema import ResponseSchemaError
from roomkit.providers.ai.tool_calls import MALFORMED_CALL_NUDGE
from roomkit.providers.gemini.config import GeminiConfig
from roomkit.providers.gemini.request import format_content, format_messages
from tests.private_module import load_private_copy
from tests.tool_loop_modes import run_tool_loop


class _FakeStreamIterator:
    """Simulates an async iterator returned by generate_content_stream."""

    def __init__(self, chunks: list[SimpleNamespace]) -> None:
        self._chunks = chunks
        self._index = 0

    def __aiter__(self) -> _FakeStreamIterator:
        return self

    async def __anext__(self) -> SimpleNamespace:
        if self._index >= len(self._chunks):
            raise StopAsyncIteration
        chunk = self._chunks[self._index]
        self._index += 1
        return chunk


def _mock_genai_module() -> MagicMock:
    """Return a MagicMock that behaves like the google.genai module."""
    mod = MagicMock()

    # Mock types
    types = MagicMock()
    types.Content = MagicMock(side_effect=lambda **kw: SimpleNamespace(**kw))
    types.Part.from_text = MagicMock(side_effect=lambda text: SimpleNamespace(text=text))
    types.Part.from_uri = MagicMock(
        side_effect=lambda file_uri, mime_type: SimpleNamespace(uri=file_uri, mime_type=mime_type)
    )
    types.Part.from_bytes = MagicMock(
        side_effect=lambda data, mime_type: SimpleNamespace(data=data, mime_type=mime_type)
    )
    types.GenerateContentConfig = MagicMock(side_effect=lambda **kw: SimpleNamespace(**kw))
    types.Tool = MagicMock(side_effect=lambda **kw: SimpleNamespace(**kw))
    mod.types = types

    # Mock Client with async generate_content and generate_content_stream
    client_instance = MagicMock()
    client_instance.aio.models.generate_content = AsyncMock()
    client_instance.aio.models.generate_content_stream = AsyncMock()
    mod.Client.return_value = client_instance

    # Attach types to the module so 'from google.genai import types' works
    mod.types = types

    return mod


def _config(**overrides: Any) -> GeminiConfig:
    defaults: dict[str, Any] = {"api_key": "test-api-key"}
    defaults.update(overrides)
    return GeminiConfig(**defaults)


def _mock_response(
    text: str = "Hello!",
    prompt_tokens: int = 10,
    completion_tokens: int = 25,
    tool_calls: list[dict[str, Any]] | None = None,
) -> SimpleNamespace:
    """Build a fake Gemini response."""
    parts = [SimpleNamespace(text=text, function_call=None)]

    # Add tool call parts if provided
    if tool_calls:
        for tc in tool_calls:
            parts.append(
                SimpleNamespace(
                    text=None,
                    function_call=SimpleNamespace(
                        name=tc["name"],
                        args=tc.get("args", {}),
                    ),
                )
            )

    return SimpleNamespace(
        text=text,
        candidates=[
            SimpleNamespace(
                content=SimpleNamespace(parts=parts),
            )
        ],
        usage_metadata=SimpleNamespace(
            prompt_token_count=prompt_tokens,
            candidates_token_count=completion_tokens,
        ),
    )


def _context(**overrides: Any) -> AIContext:
    defaults: dict[str, Any] = {
        "messages": [AIMessage(role="user", content="Hi")],
    }
    defaults.update(overrides)
    return AIContext(**defaults)


def _genai_modules(mock_genai: MagicMock) -> dict[str, Any]:
    """Build sys.modules patch dict for Gemini tests."""
    return {
        "google": MagicMock(genai=mock_genai),
        "google.genai": mock_genai,
    }


def _stream_chunks(
    text_parts: list[str] | None = None,
    tool_calls: list[dict[str, Any]] | None = None,
    prompt_tokens: int = 10,
    completion_tokens: int = 25,
) -> _FakeStreamIterator:
    """Build a fake stream iterator from text parts and/or tool calls."""
    chunks: list[SimpleNamespace] = []

    for text in text_parts or []:
        chunks.append(
            SimpleNamespace(
                candidates=[
                    SimpleNamespace(
                        content=SimpleNamespace(
                            parts=[SimpleNamespace(text=text, function_call=None)]
                        )
                    )
                ],
                usage_metadata=None,
            )
        )

    for tc in tool_calls or []:
        chunks.append(
            SimpleNamespace(
                candidates=[
                    SimpleNamespace(
                        content=SimpleNamespace(
                            parts=[
                                SimpleNamespace(
                                    text=None,
                                    function_call=SimpleNamespace(
                                        name=tc["name"],
                                        args=tc.get("args", {}),
                                    ),
                                )
                            ]
                        )
                    )
                ],
                usage_metadata=None,
            )
        )

    # Final chunk with usage
    chunks.append(
        SimpleNamespace(
            candidates=None,
            usage_metadata=SimpleNamespace(
                prompt_token_count=prompt_tokens,
                candidates_token_count=completion_tokens,
            ),
        )
    )

    return _FakeStreamIterator(chunks)


class TestGeminiAIProvider:
    @pytest.mark.asyncio
    async def test_generate_success(self) -> None:
        mock_genai = _mock_genai_module()
        with patch.dict("sys.modules", _genai_modules(mock_genai)):
            from roomkit.providers.gemini.ai import GeminiAIProvider

            provider = GeminiAIProvider(_config())
            provider._client.aio.models.generate_content_stream.return_value = _stream_chunks(
                text_parts=["Hi there!"]
            )
            result = await provider.generate(_context())

            assert result.content == "Hi there!"
            assert result.metadata["model"] == "gemini-3.8-flash"

    @pytest.mark.asyncio
    async def test_data_uri_image_decoded_to_inline_bytes(self) -> None:
        """A ``data:`` URI image must become inline bytes, not a file URI.

        Gemini's ``from_uri`` expects a fetchable URI; a data URI handed
        to it ships a broken reference and the model never sees the image.
        The provider routes data URIs through ``from_bytes`` with the
        decoded payload.
        """
        mock_genai = _mock_genai_module()
        with patch.dict("sys.modules", _genai_modules(mock_genai)):
            from roomkit.providers.gemini.ai import GeminiAIProvider

            provider = GeminiAIProvider(_config())
            raw = b"\x89PNG\r\n\x1a\n fake image bytes"
            b64 = base64.b64encode(raw).decode()
            parts = format_content(
                provider._types,
                [
                    AITextPart(text="what is this?"),
                    AIImagePart(url=f"data:image/png;base64,{b64}", mime_type="image/png"),
                ],
            )

            mock_genai.types.Part.from_uri.assert_not_called()
            mock_genai.types.Part.from_bytes.assert_called_once_with(
                data=raw, mime_type="image/png"
            )
            assert parts[-1].data == raw
            assert parts[-1].mime_type == "image/png"

    @pytest.mark.asyncio
    async def test_real_uri_image_passed_through_from_uri(self) -> None:
        """A genuine fetchable URI still goes through ``from_uri``."""
        mock_genai = _mock_genai_module()
        with patch.dict("sys.modules", _genai_modules(mock_genai)):
            from roomkit.providers.gemini.ai import GeminiAIProvider

            provider = GeminiAIProvider(_config())
            parts = format_content(
                provider._types,
                [AIImagePart(url="https://example.com/cat.png", mime_type="image/png")],
            )

            mock_genai.types.Part.from_bytes.assert_not_called()
            mock_genai.types.Part.from_uri.assert_called_once_with(
                file_uri="https://example.com/cat.png", mime_type="image/png"
            )
            assert parts[-1].uri == "https://example.com/cat.png"

    @pytest.mark.asyncio
    async def test_tool_result_string_is_single_function_response(self) -> None:
        """A string tool result stays a single text function response — the
        unchanged path for every existing text tool."""
        mock_genai = _mock_genai_module()
        with patch.dict("sys.modules", _genai_modules(mock_genai)):
            from roomkit.providers.ai.base import AIToolResultPart
            from roomkit.providers.gemini.ai import GeminiAIProvider

            provider = GeminiAIProvider(_config())
            contents = format_messages(
                provider._types,
                [
                    AIMessage(
                        role="tool",
                        content=[AIToolResultPart(tool_call_id="t1", name="foo", result="hello")],
                    )
                ],
            )

            assert len(contents) == 1
            assert contents[0].role == "user"
            mock_genai.types.Part.from_function_response.assert_called_once_with(
                name="foo", response={"result": "hello"}
            )
            mock_genai.types.Part.from_bytes.assert_not_called()

    @pytest.mark.asyncio
    async def test_tool_result_image_appended_as_user_content(self) -> None:
        """An image tool result keeps the function response text-only and puts
        the decoded image on a following user Content (inline bytes)."""
        mock_genai = _mock_genai_module()
        with patch.dict("sys.modules", _genai_modules(mock_genai)):
            from roomkit.providers.ai.base import AIToolResultPart
            from roomkit.providers.gemini.ai import GeminiAIProvider

            provider = GeminiAIProvider(_config())
            raw = b"\x89PNG\r\n\x1a\n screenshot bytes"
            b64 = base64.b64encode(raw).decode()
            contents = format_messages(
                provider._types,
                [
                    AIMessage(
                        role="tool",
                        content=[
                            AIToolResultPart(
                                tool_call_id="t1",
                                name="screenshot",
                                result=[
                                    AITextPart(text="the screen"),
                                    AIImagePart(
                                        url=f"data:image/png;base64,{b64}", mime_type="image/png"
                                    ),
                                ],
                            )
                        ],
                    )
                ],
            )

            # Function response carries the text only; image is a second user
            # Content built from inline bytes (from_bytes, not from_uri).
            mock_genai.types.Part.from_function_response.assert_called_once_with(
                name="screenshot", response={"result": "the screen"}
            )
            mock_genai.types.Part.from_bytes.assert_called_once_with(
                data=raw, mime_type="image/png"
            )
            assert len(contents) == 2
            assert contents[1].role == "user"
            assert contents[1].parts[-1].data == raw

    @pytest.mark.asyncio
    async def test_generate_with_system_prompt(self) -> None:
        mock_genai = _mock_genai_module()
        with patch.dict("sys.modules", _genai_modules(mock_genai)):
            from roomkit.providers.gemini.ai import GeminiAIProvider

            provider = GeminiAIProvider(_config())
            provider._client.aio.models.generate_content_stream.return_value = _stream_chunks(
                text_parts=["Hello"]
            )
            ctx = _context(system_prompt="You are helpful.")
            await provider.generate(ctx)

            assert provider._client.aio.models.generate_content_stream.called

    @pytest.mark.asyncio
    async def test_generate_maps_usage(self) -> None:
        mock_genai = _mock_genai_module()
        with patch.dict("sys.modules", _genai_modules(mock_genai)):
            from roomkit.providers.gemini.ai import GeminiAIProvider

            provider = GeminiAIProvider(_config())
            provider._client.aio.models.generate_content_stream.return_value = _stream_chunks(
                text_parts=["hi"], prompt_tokens=42, completion_tokens=7
            )
            result = await provider.generate(_context())

            assert result.usage == {"input_tokens": 42, "output_tokens": 7}

    async def test_thinking_is_counted_and_priced_as_output(self) -> None:
        """Gemini bills thinking as output but counts it outside candidates: a
        turn that thinks 900 tokens to answer in 10 is priced for 910 (RMK-312)."""
        mock_genai = _mock_genai_module()
        with patch.dict("sys.modules", _genai_modules(mock_genai)):
            from roomkit.providers.gemini.ai import GeminiAIProvider

            provider = GeminiAIProvider(_config())
            usage = SimpleNamespace(
                prompt_token_count=100,
                candidates_token_count=10,
                thoughts_token_count=900,
                cached_content_token_count=40,
            )
            text = SimpleNamespace(
                candidates=[
                    SimpleNamespace(
                        content=SimpleNamespace(
                            parts=[SimpleNamespace(text="hi", function_call=None)]
                        )
                    )
                ],
                usage_metadata=None,
            )
            provider._client.aio.models.generate_content_stream.return_value = _FakeStreamIterator(
                [text, SimpleNamespace(candidates=None, usage_metadata=usage)]
            )
            result = await provider.generate(_context())

            assert result.usage == {
                "input_tokens": 60,
                "output_tokens": 910,
                "cache_read_input_tokens": 40,
                "reasoning_tokens": 900,
            }
            pricing = provider.catalog_entry().pricing
            without_thinking = {**result.usage, "output_tokens": 10}
            assert pricing.cost_for(result.usage) > pricing.cost_for(without_thinking)

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "raw_reason",
        [
            # The SDK hands back a FinishReason enum; the wire spelling is its
            # ``.name``. A plain string must pass through untouched.
            pytest.param(SimpleNamespace(name="MAX_TOKENS"), id="enum"),
            pytest.param("MAX_TOKENS", id="string"),
        ],
    )
    async def test_generate_reports_truncation(self, raw_reason: Any) -> None:
        """A truncated round must be distinguishable from a silent one.

        Gemini reports MAX_TOKENS on the candidate of a chunk that carries no
        parts at all — exactly the round where the model spent its whole budget
        thinking. Without this the tool loop reads the empty content as a model
        that failed to verbalize, and re-prompts under the same cap.
        """
        mock_genai = _mock_genai_module()
        with patch.dict("sys.modules", _genai_modules(mock_genai)):
            from roomkit.providers.gemini.ai import GeminiAIProvider

            provider = GeminiAIProvider(_config())
            provider._client.aio.models.generate_content_stream.return_value = _FakeStreamIterator(
                [
                    SimpleNamespace(
                        candidates=[SimpleNamespace(content=None, finish_reason=raw_reason)],
                        usage_metadata=SimpleNamespace(
                            prompt_token_count=10, candidates_token_count=4096
                        ),
                    )
                ]
            )
            result = await provider.generate(_context())

            assert result.content == ""
            assert result.finish_reason == "MAX_TOKENS"

    async def test_a_malformed_call_is_retried_by_the_tool_loop(self, streaming: bool) -> None:
        """Gemini ends a round on a call it could not parse with no part at all:
        the loop tells the model its call did not run, and it calls again (RMK-314)."""
        mock_genai = _mock_genai_module()
        with patch.dict("sys.modules", _genai_modules(mock_genai)):
            from roomkit.providers.gemini.ai import GeminiAIProvider

            provider = GeminiAIProvider(_config())
            malformed = SimpleNamespace(
                candidates=[
                    SimpleNamespace(
                        content=None, finish_reason=SimpleNamespace(name="MALFORMED_FUNCTION_CALL")
                    )
                ],
                usage_metadata=None,
            )
            provider._client.aio.models.generate_content_stream.side_effect = [
                _FakeStreamIterator([malformed]),
                _stream_chunks(tool_calls=[{"name": "search", "args": {"query": "x"}}]),
                _stream_chunks(text_parts=["Found it."]),
            ]
            handler = AsyncMock(return_value="ok")
            channel = AIChannel(
                "ai", provider=provider, tool_handler=handler, tool_loop_timeout_seconds=None
            )
            context = _context(
                tools=[AITool(name="search", description="Search", parameters={"type": "object"})]
            )
            run = await run_tool_loop(channel, context)

            assert run.text == "Found it."
            assert handler.await_count == 1
            assert [m.content for m in context.messages].count(MALFORMED_CALL_NUDGE) == 1

    @pytest.mark.asyncio
    async def test_generate_with_tools(self) -> None:
        mock_genai = _mock_genai_module()
        with patch.dict("sys.modules", _genai_modules(mock_genai)):
            from roomkit.providers.gemini.ai import GeminiAIProvider

            provider = GeminiAIProvider(_config())
            provider._client.aio.models.generate_content_stream.return_value = _stream_chunks(
                tool_calls=[{"name": "search", "args": {"query": "test"}}],
            )
            ctx = _context(
                tools=[
                    AITool(
                        name="search",
                        description="Search for info",
                        parameters={"type": "object", "properties": {"query": {"type": "string"}}},
                    )
                ]
            )
            result = await provider.generate(ctx)

            assert len(result.tool_calls) == 1
            assert result.tool_calls[0].name == "search"
            assert result.tool_calls[0].arguments == {"query": "test"}

    @pytest.mark.asyncio
    async def test_generate_api_error(self) -> None:
        mock_genai = _mock_genai_module()
        with patch.dict("sys.modules", _genai_modules(mock_genai)):
            from roomkit.providers.ai.base import ProviderError
            from roomkit.providers.gemini.ai import GeminiAIProvider

            provider = GeminiAIProvider(_config())
            provider._client.aio.models.generate_content_stream.side_effect = Exception(
                "API error"
            )

            with pytest.raises(ProviderError) as exc_info:
                await provider.generate(_context())

            assert "API error" in str(exc_info.value)
            assert exc_info.value.provider == "gemini"

    @pytest.mark.asyncio
    async def test_rate_limit_error_is_retryable(self) -> None:
        mock_genai = _mock_genai_module()
        with patch.dict("sys.modules", _genai_modules(mock_genai)):
            from roomkit.providers.ai.base import ProviderError
            from roomkit.providers.gemini.ai import GeminiAIProvider

            provider = GeminiAIProvider(_config())
            provider._client.aio.models.generate_content_stream.side_effect = Exception(
                "Rate limit exceeded 429"
            )

            with pytest.raises(ProviderError) as exc_info:
                await provider.generate(_context())

            assert exc_info.value.retryable is True

    def test_config_defaults(self) -> None:
        cfg = _config()
        assert cfg.model == "gemini-3.8-flash"
        assert cfg.max_tokens == 1024

    def test_supports_vision(self) -> None:
        mock_genai = _mock_genai_module()
        with patch.dict("sys.modules", _genai_modules(mock_genai)):
            from roomkit.providers.gemini.ai import GeminiAIProvider

            provider = GeminiAIProvider(_config())
            assert provider.supports_vision is True

    def test_supports_streaming(self) -> None:
        mock_genai = _mock_genai_module()
        with patch.dict("sys.modules", _genai_modules(mock_genai)):
            from roomkit.providers.gemini.ai import GeminiAIProvider

            provider = GeminiAIProvider(_config())
            assert provider.supports_streaming is True
            assert provider.supports_structured_streaming is True

    def test_model_name(self) -> None:
        mock_genai = _mock_genai_module()
        with patch.dict("sys.modules", _genai_modules(mock_genai)):
            from roomkit.providers.gemini.ai import GeminiAIProvider

            provider = GeminiAIProvider(_config(model="gemini-1.5-pro"))
            assert provider.model_name == "gemini-1.5-pro"

    def test_lazy_import_error(self) -> None:
        missing = {"google": None, "google.genai": None}
        with patch.dict("sys.modules", missing):
            mod = load_private_copy("roomkit.providers.gemini.ai", missing)

            with pytest.raises(ImportError, match="google-genai is required"):
                mod.GeminiAIProvider(_config())

    @pytest.mark.asyncio
    async def test_structured_stream_yields_events(self) -> None:
        mock_genai = _mock_genai_module()
        with patch.dict("sys.modules", _genai_modules(mock_genai)):
            from roomkit.providers.gemini.ai import GeminiAIProvider

            provider = GeminiAIProvider(_config())
            provider._client.aio.models.generate_content_stream.return_value = _stream_chunks(
                text_parts=["Hello", " world"],
                prompt_tokens=5,
                completion_tokens=10,
            )

            events = []
            async for event in provider.generate_structured_stream(_context()):
                events.append(event)

            assert len(events) == 4
            assert isinstance(events[0], StreamTextDelta)
            assert events[0].text == "Hello"
            assert isinstance(events[1], StreamTextDelta)
            assert events[1].text == " world"
            # The input, told as the chunk that carries it comes, which a cut
            # generation counts (RFC §6.4, RMK-369).
            assert events[2] == StreamUsage(usage={"input_tokens": 5})
            assert isinstance(events[3], StreamDone)
            assert events[3].usage == {"input_tokens": 5, "output_tokens": 10}
            assert events[3].metadata["model"] == "gemini-3.8-flash"

    @pytest.mark.asyncio
    async def test_structured_stream_surfaces_thought_parts(self) -> None:
        # Gemini flags reasoning summaries with thought=True on the part; the
        # provider must surface those as thinking, plain text as text.
        from roomkit.providers.ai.base import StreamThinkingDelta

        mock_genai = _mock_genai_module()
        with patch.dict("sys.modules", _genai_modules(mock_genai)):
            from roomkit.providers.gemini.ai import GeminiAIProvider

            provider = GeminiAIProvider(_config())
            thought = SimpleNamespace(text="reasoning", thought=True, function_call=None)
            answer = SimpleNamespace(text="answer", thought=False, function_call=None)
            chunk = SimpleNamespace(
                candidates=[SimpleNamespace(content=SimpleNamespace(parts=[thought, answer]))],
                usage_metadata=SimpleNamespace(prompt_token_count=3, candidates_token_count=4),
            )
            provider._client.aio.models.generate_content_stream.return_value = _FakeStreamIterator(
                [chunk]
            )

            events = [
                e async for e in provider.generate_structured_stream(_context(thinking_budget=512))
            ]
            thinking = [e.thinking for e in events if isinstance(e, StreamThinkingDelta)]
            text = [e.text for e in events if isinstance(e, StreamTextDelta)]
            assert thinking == ["reasoning"]
            assert text == ["answer"]

    @pytest.mark.asyncio
    async def test_duplicate_streamed_function_call_keeps_signature(self) -> None:
        # Gemini streams the same call across chunks: the first carries its
        # thought_signature, a later one re-emits it without. The provider must
        # collapse them into ONE tool call that retains the signature — else
        # Gemini 3 rejects the next turn with HTTP 400 "missing thought_signature".
        mock_genai = _mock_genai_module()
        with patch.dict("sys.modules", _genai_modules(mock_genai)):
            from roomkit.providers.gemini.ai import GeminiAIProvider

            def _fc_chunk(sig: bytes | None) -> SimpleNamespace:
                part = SimpleNamespace(
                    text=None,
                    function_call=SimpleNamespace(name="web_search", args={"q": "x"}),
                    thought_signature=sig,
                )
                return SimpleNamespace(
                    candidates=[SimpleNamespace(content=SimpleNamespace(parts=[part]))],
                    usage_metadata=None,
                )

            final = SimpleNamespace(
                candidates=None,
                usage_metadata=SimpleNamespace(prompt_token_count=1, candidates_token_count=1),
            )
            provider = GeminiAIProvider(_config())
            provider._client.aio.models.generate_content_stream.return_value = _FakeStreamIterator(
                [_fc_chunk(b"sigbytes"), _fc_chunk(None), final]
            )

            events = [
                e async for e in provider.generate_structured_stream(_context(thinking_budget=512))
            ]
            tool_calls = [e for e in events if isinstance(e, StreamToolCall)]
            assert len(tool_calls) == 1
            assert tool_calls[0].name == "web_search"
            assert tool_calls[0].metadata["thought_signature"] == base64.b64encode(
                b"sigbytes"
            ).decode("ascii")

    @pytest.mark.asyncio
    async def test_unsigned_call_before_signed_one_still_replays_signed(self) -> None:
        # Gemini signs one functionCall part of a parallel round, and its
        # validator then demands a signature on EVERY history functionCall.
        # The borrowed signature must reach a call that comes BEFORE the signed
        # one — the round is scanned up front, not carried forward.
        mock_genai = _mock_genai_module()
        with patch.dict("sys.modules", _genai_modules(mock_genai)):
            from roomkit.providers.ai.base import AIToolCallPart
            from roomkit.providers.gemini.ai import GeminiAIProvider

            sig = base64.b64encode(b"sigbytes").decode("ascii")
            provider = GeminiAIProvider(_config())
            format_messages(
                provider._types,
                [
                    AIMessage(
                        role="assistant",
                        content=[
                            AIToolCallPart(id="c1", name="find_tools", arguments={"q": "x"}),
                            AIToolCallPart(
                                id="c2",
                                name="get_status",
                                arguments={},
                                metadata={"thought_signature": sig},
                            ),
                        ],
                    )
                ],
            )

            signed = [c.kwargs for c in mock_genai.types.Part.call_args_list]
            assert len(signed) == 2
            assert all(kw["thought_signature"] == b"sigbytes" for kw in signed)
            # An unsigned Part would have gone through from_function_call.
            mock_genai.types.Part.from_function_call.assert_not_called()

    @pytest.mark.asyncio
    async def test_each_signed_call_keeps_its_own_signature(self) -> None:
        # Borrowing is a fallback, not a rewrite: a call that carries its own
        # signature replays with that one, not the round's first.
        mock_genai = _mock_genai_module()
        with patch.dict("sys.modules", _genai_modules(mock_genai)):
            from roomkit.providers.ai.base import AIToolCallPart
            from roomkit.providers.gemini.ai import GeminiAIProvider

            def _signed(call_id: str, name: str, sig: bytes) -> AIToolCallPart:
                return AIToolCallPart(
                    id=call_id,
                    name=name,
                    metadata={"thought_signature": base64.b64encode(sig).decode("ascii")},
                )

            provider = GeminiAIProvider(_config())
            format_messages(
                provider._types,
                [
                    AIMessage(
                        role="assistant",
                        content=[
                            _signed("c1", "a", b"first"),
                            _signed("c2", "b", b"second"),
                        ],
                    )
                ],
            )

            sigs = [c.kwargs["thought_signature"] for c in mock_genai.types.Part.call_args_list]
            assert sigs == [b"first", b"second"]

    @pytest.mark.asyncio
    async def test_round_without_any_signature_warns_once(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        # Nothing to borrow: the round is the one Gemini 3 rejects on the next
        # turn, and it is worth exactly one warning — not one per call.
        mock_genai = _mock_genai_module()
        with patch.dict("sys.modules", _genai_modules(mock_genai)):
            from roomkit.providers.gemini.ai import GeminiAIProvider

            provider = GeminiAIProvider(_config())
            provider._client.aio.models.generate_content_stream.return_value = _stream_chunks(
                tool_calls=[{"name": "alpha"}, {"name": "beta"}],
            )

            with caplog.at_level("WARNING", logger="roomkit.providers.gemini.ai"):
                calls = [
                    e
                    async for e in provider.generate_structured_stream(
                        _context(thinking_budget=512)
                    )
                    if isinstance(e, StreamToolCall)
                ]

            assert len(calls) == 2
            assert all("thought_signature" not in c.metadata for c in calls)
            warnings = [r for r in caplog.records if r.levelname == "WARNING"]
            assert len(warnings) == 1
            assert "no thought_signature" in warnings[0].getMessage()

    @pytest.mark.asyncio
    async def test_partially_signed_round_does_not_warn(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        # One signature anywhere in the round is enough: format_messages lends
        # it to the others, so the replay is valid and there is nothing to say.
        mock_genai = _mock_genai_module()
        with patch.dict("sys.modules", _genai_modules(mock_genai)):
            from roomkit.providers.gemini.ai import GeminiAIProvider

            def _fc_chunk(name: str, sig: bytes | None) -> SimpleNamespace:
                part = SimpleNamespace(
                    text=None,
                    function_call=SimpleNamespace(name=name, args={}),
                    thought_signature=sig,
                )
                return SimpleNamespace(
                    candidates=[SimpleNamespace(content=SimpleNamespace(parts=[part]))],
                    usage_metadata=None,
                )

            provider = GeminiAIProvider(_config())
            provider._client.aio.models.generate_content_stream.return_value = _FakeStreamIterator(
                [_fc_chunk("alpha", b"sig"), _fc_chunk("beta", None)]
            )

            with caplog.at_level("WARNING", logger="roomkit.providers.gemini.ai"):
                calls = [
                    e
                    async for e in provider.generate_structured_stream(
                        _context(thinking_budget=512)
                    )
                    if isinstance(e, StreamToolCall)
                ]

            assert len(calls) == 2
            assert [r for r in caplog.records if r.levelname == "WARNING"] == []

    @pytest.mark.asyncio
    async def test_distinct_parallel_function_calls_not_merged(self) -> None:
        # Two web_search calls with different args are distinct — they must NOT
        # be collapsed by the duplicate-merge logic.
        mock_genai = _mock_genai_module()
        with patch.dict("sys.modules", _genai_modules(mock_genai)):
            from roomkit.providers.gemini.ai import GeminiAIProvider

            def _fc_chunk(query: str) -> SimpleNamespace:
                part = SimpleNamespace(
                    text=None,
                    function_call=SimpleNamespace(name="web_search", args={"q": query}),
                    thought_signature=b"s",
                )
                return SimpleNamespace(
                    candidates=[SimpleNamespace(content=SimpleNamespace(parts=[part]))],
                    usage_metadata=None,
                )

            final = SimpleNamespace(
                candidates=None,
                usage_metadata=SimpleNamespace(prompt_token_count=1, candidates_token_count=1),
            )
            provider = GeminiAIProvider(_config())
            provider._client.aio.models.generate_content_stream.return_value = _FakeStreamIterator(
                [_fc_chunk("alpha"), _fc_chunk("beta"), final]
            )

            events = [
                e async for e in provider.generate_structured_stream(_context(thinking_budget=512))
            ]
            tool_calls = [e for e in events if isinstance(e, StreamToolCall)]
            assert len(tool_calls) == 2
            assert {tc.arguments["q"] for tc in tool_calls} == {"alpha", "beta"}

    @pytest.mark.asyncio
    async def test_history_ending_with_model_turn_is_refused(self) -> None:
        # Gemini answers a user turn; a history ending on a model one comes
        # back as an opaque 400. Refuse it here, naming the condition, and
        # never reach the API.
        from roomkit.providers.ai.base import ProviderError

        mock_genai = _mock_genai_module()
        with patch.dict("sys.modules", _genai_modules(mock_genai)):
            from roomkit.providers.gemini.ai import GeminiAIProvider

            provider = GeminiAIProvider(_config())
            context = _context(
                messages=[
                    AIMessage(role="user", content="Hi"),
                    AIMessage(role="assistant", content="Hello!"),
                ]
            )

            with pytest.raises(ProviderError) as excinfo:
                [e async for e in provider.generate_structured_stream(context)]

            assert "ends with a model turn" in str(excinfo.value)
            assert "(1 trailing model content(s))" in str(excinfo.value)
            assert excinfo.value.retryable is False
            assert excinfo.value.provider == "gemini"
            provider._client.aio.models.generate_content_stream.assert_not_called()

    @pytest.mark.asyncio
    async def test_history_ending_with_tool_results_is_not_a_model_turn(self) -> None:
        # Function responses are sent as role="user" Contents — the guard must
        # not mistake a tool round mid-flight for a model tail.
        mock_genai = _mock_genai_module()
        with patch.dict("sys.modules", _genai_modules(mock_genai)):
            from roomkit.providers.ai.base import AIToolCallPart, AIToolResultPart
            from roomkit.providers.gemini.ai import GeminiAIProvider

            provider = GeminiAIProvider(_config())
            provider._client.aio.models.generate_content_stream.return_value = _stream_chunks(
                text_parts=["done"],
            )
            context = _context(
                messages=[
                    AIMessage(role="user", content="Hi"),
                    AIMessage(
                        role="assistant",
                        content=[AIToolCallPart(id="c1", name="foo", arguments={})],
                    ),
                    AIMessage(
                        role="tool",
                        content=[AIToolResultPart(tool_call_id="c1", name="foo", result="ok")],
                    ),
                ]
            )

            events = [e async for e in provider.generate_structured_stream(context)]

            assert [e.text for e in events if isinstance(e, StreamTextDelta)] == ["done"]

    @pytest.mark.asyncio
    async def test_generate_stream_yields_text(self) -> None:
        mock_genai = _mock_genai_module()
        with patch.dict("sys.modules", _genai_modules(mock_genai)):
            from roomkit.providers.gemini.ai import GeminiAIProvider

            provider = GeminiAIProvider(_config())
            provider._client.aio.models.generate_content_stream.return_value = _stream_chunks(
                text_parts=["one", "two", "three"],
            )

            parts = []
            async for text in provider.generate_stream(_context()):
                parts.append(text)

            assert parts == ["one", "two", "three"]

    @pytest.mark.asyncio
    async def test_structured_stream_with_tool_calls(self) -> None:
        mock_genai = _mock_genai_module()
        with patch.dict("sys.modules", _genai_modules(mock_genai)):
            from roomkit.providers.gemini.ai import GeminiAIProvider

            provider = GeminiAIProvider(_config())
            provider._client.aio.models.generate_content_stream.return_value = _stream_chunks(
                text_parts=["Let me search"],
                tool_calls=[{"name": "search", "args": {"q": "test"}}],
            )

            events = []
            async for event in provider.generate_structured_stream(_context()):
                events.append(event)

            text_deltas = [e for e in events if isinstance(e, StreamTextDelta)]
            tool_calls = [e for e in events if isinstance(e, StreamToolCall)]
            done_events = [e for e in events if isinstance(e, StreamDone)]

            assert len(text_deltas) == 1
            assert text_deltas[0].text == "Let me search"
            assert len(tool_calls) == 1
            assert tool_calls[0].name == "search"
            assert tool_calls[0].arguments == {"q": "test"}
            assert len(done_events) == 1

    @pytest.mark.asyncio
    async def test_structured_stream_api_error(self) -> None:
        mock_genai = _mock_genai_module()
        with patch.dict("sys.modules", _genai_modules(mock_genai)):
            from roomkit.providers.ai.base import ProviderError
            from roomkit.providers.gemini.ai import GeminiAIProvider

            provider = GeminiAIProvider(_config())
            provider._client.aio.models.generate_content_stream.side_effect = Exception(
                "Stream error"
            )

            with pytest.raises(ProviderError) as exc_info:
                async for _ in provider.generate_structured_stream(_context()):
                    pass

            assert "Stream error" in str(exc_info.value)
            assert exc_info.value.provider == "gemini"


class TestGeminiImageDataURIs:
    """A data: URI is read by the shared reader, and refused before the request."""

    def test_mime_type_falls_back_to_the_part_when_the_header_has_none(self) -> None:
        mock_genai = _mock_genai_module()
        with patch.dict("sys.modules", _genai_modules(mock_genai)):
            from roomkit.providers.gemini.ai import GeminiAIProvider

            provider = GeminiAIProvider(_config())
            format_content(
                provider._types,
                [AIImagePart(url="data:;base64,QUJDMTIz", mime_type="image/png")],
            )
            mock_genai.types.Part.from_bytes.assert_called_once_with(
                data=b"ABC123", mime_type="image/png"
            )

    def test_a_malformed_payload_is_refused_before_the_request(self) -> None:
        mock_genai = _mock_genai_module()
        with patch.dict("sys.modules", _genai_modules(mock_genai)):
            from roomkit.providers.gemini.ai import GeminiAIProvider

            provider = GeminiAIProvider(_config())
            with pytest.raises(ProviderError, match="not valid base64") as excinfo:
                format_content(
                    provider._types, [AIImagePart(url="data:image/png;base64,not*base64")]
                )
            assert excinfo.value.retryable is False
            assert excinfo.value.provider == "gemini"
            mock_genai.types.Part.from_bytes.assert_not_called()


_VERDICT: dict[str, Any] = {
    "type": "object",
    "properties": {"label": {"type": "string", "enum": ["yes", "no"]}},
    "required": ["label"],
    "additionalProperties": False,
}


def _chunks_ending(text: str, finish_reason: str) -> _FakeStreamIterator:
    """One text chunk whose candidate reports ``finish_reason``."""
    return _FakeStreamIterator(
        [
            SimpleNamespace(
                candidates=[
                    SimpleNamespace(
                        content=SimpleNamespace(
                            parts=[SimpleNamespace(text=text, function_call=None)]
                        ),
                        finish_reason=finish_reason,
                    )
                ],
                usage_metadata=None,
            )
        ]
    )


class TestGeminiResponseSchema:
    """RFC §6.7: a constrained answer ending on a safety or length stop raises."""

    @staticmethod
    def _provider(stream: _FakeStreamIterator) -> Any:
        mock_genai = _mock_genai_module()
        with patch.dict("sys.modules", _genai_modules(mock_genai)):
            from roomkit.providers.gemini.ai import GeminiAIProvider

            provider = GeminiAIProvider(_config())
        provider._client.aio.models.generate_content_stream.return_value = stream
        return provider

    async def test_generate_returns_the_document(self) -> None:
        provider = self._provider(_chunks_ending('{"label": "no"}', "STOP"))

        result = await provider.generate(_context(response_schema=_VERDICT))

        assert result.content == '{"label": "no"}'
        assert result.finish_reason == "STOP"

    @pytest.mark.parametrize(
        ("text", "finish_reason", "reason"),
        [
            ("", "SAFETY", "refusal"),
            ("", "PROHIBITED_CONTENT", "refusal"),
            ('{"label": "n', "MAX_TOKENS", "truncated"),
            ("No.", "STOP", "invalid_json"),
        ],
    )
    async def test_an_answer_without_its_document_raises(
        self, text: str, finish_reason: str, reason: str
    ) -> None:
        provider = self._provider(_chunks_ending(text, finish_reason))

        with pytest.raises(ResponseSchemaError) as exc:
            await provider.generate(_context(response_schema=_VERDICT))

        assert exc.value.reason == reason

    async def test_a_blocked_prompt_is_a_refusal_not_bad_json(self) -> None:
        blocked = SimpleNamespace(
            candidates=None,
            usage_metadata=None,
            prompt_feedback=SimpleNamespace(
                block_reason=SimpleNamespace(name="PROHIBITED_CONTENT")
            ),
        )
        provider = self._provider(_FakeStreamIterator([blocked]))

        with pytest.raises(ResponseSchemaError, match="PROHIBITED_CONTENT") as exc:
            await provider.generate(_context(response_schema=_VERDICT))

        assert exc.value.reason == "refusal"

    async def test_a_blocked_prompt_names_its_reason_as_the_refusal(self) -> None:
        blocked = SimpleNamespace(
            candidates=None,
            usage_metadata=None,
            prompt_feedback=SimpleNamespace(
                block_reason=SimpleNamespace(name="PROHIBITED_CONTENT")
            ),
        )
        provider = self._provider(_FakeStreamIterator([blocked]))

        result = await provider.generate(_context())

        assert result.metadata["refusal"] == "PROHIBITED_CONTENT"

    @staticmethod
    async def _drain(stream: Any) -> tuple[list[Any], ResponseSchemaError | None]:
        events: list[Any] = []
        try:
            async for event in stream:
                events.append(event)
        except ResponseSchemaError as exc:
            return events, exc
        return events, None

    async def test_a_streamed_answer_is_checked_before_its_done_event(self) -> None:
        provider = self._provider(_chunks_ending('{"label": "yes"}', "STOP"))

        events, error = await self._drain(
            provider.generate_structured_stream(_context(response_schema=_VERDICT))
        )

        assert error is None
        assert isinstance(events[-1], StreamDone)
        assert (
            "".join(e.text for e in events if isinstance(e, StreamTextDelta)) == '{"label": "yes"}'
        )
        config = provider._client.aio.models.generate_content_stream.call_args.kwargs["config"]
        assert config.response_json_schema == _VERDICT

    async def test_a_streamed_answer_that_is_not_the_document_raises_instead_of_done(
        self,
    ) -> None:
        provider = self._provider(_chunks_ending("Yes.", "STOP"))

        events, error = await self._drain(
            provider.generate_structured_stream(_context(response_schema=_VERDICT))
        )

        assert error is not None and error.reason == "invalid_json"
        assert not any(isinstance(e, StreamDone) for e in events)
