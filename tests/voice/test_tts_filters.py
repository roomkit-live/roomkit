"""Unit tests for TTS text filters."""

from __future__ import annotations

from collections.abc import AsyncIterator

from roomkit.voice.tts.filters import (
    StripBrackets,
    StripInternalTags,
    filtered_stream,
)

# ---------------------------------------------------------------------------
# StripInternalTags — non-streaming (__call__)
# ---------------------------------------------------------------------------


class TestStripInternalTagsCall:
    def test_basic(self):
        f = StripInternalTags()
        assert f("[internal]Respond in French[/internal] Bonjour!") == "Bonjour!"

    def test_multiple_tags(self):
        f = StripInternalTags()
        text = "[internal]A[/internal] Hello [internal]B[/internal] World"
        assert f(text) == "Hello World"

    def test_no_tags(self):
        f = StripInternalTags()
        assert f("Hello, how are you?") == "Hello, how are you?"

    def test_empty_result(self):
        f = StripInternalTags()
        assert f("[internal]everything hidden[/internal]") == ""

    def test_case_insensitive(self):
        f = StripInternalTags()
        assert f("[Internal]hidden[/Internal] visible") == "visible"

    def test_multiline(self):
        f = StripInternalTags()
        text = "[internal]\nthinking\nstuff\n[/internal]\nHello!"
        assert f(text) == "Hello!"

    def test_reusable(self):
        f = StripInternalTags()
        assert f("[internal]a[/internal] first") == "first"
        assert f("[internal]b[/internal] second") == "second"


# ---------------------------------------------------------------------------
# StripInternalTags — streaming (feed/flush)
# ---------------------------------------------------------------------------


class TestStripInternalTagsStreaming:
    def test_tag_in_single_chunk(self):
        f = StripInternalTags()
        f.reset()
        assert f.feed("[internal]hidden[/internal]Hello") == "Hello"
        assert f.flush() == ""

    def test_tag_split_across_chunks(self):
        f = StripInternalTags()
        f.reset()
        r1 = f.feed("[inter")
        r2 = f.feed("nal]hidden[/inter")
        r3 = f.feed("nal] visible")
        r4 = f.flush()
        assert r1 + r2 + r3 + r4 == " visible"

    def test_partial_tag_at_end(self):
        f = StripInternalTags()
        f.reset()
        r1 = f.feed("Hello [intern")
        r2 = f.feed("al]secret[/internal] world")
        r3 = f.flush()
        assert r1 + r2 + r3 == "Hello  world"

    def test_no_tags_streaming(self):
        f = StripInternalTags()
        f.reset()
        r1 = f.feed("Hello ")
        r2 = f.feed("world")
        r3 = f.flush()
        assert r1 + r2 + r3 == "Hello world"

    def test_unclosed_tag_discarded(self):
        f = StripInternalTags()
        f.reset()
        r1 = f.feed("Hello [internal]never closed")
        r2 = f.flush()
        # Text before tag emitted, content inside discarded
        assert r1 + r2 == "Hello "

    def test_text_before_and_after(self):
        f = StripInternalTags()
        f.reset()
        r1 = f.feed("Before ")
        r2 = f.feed("[internal]")
        r3 = f.feed("hidden")
        r4 = f.feed("[/internal]")
        r5 = f.feed(" After")
        r6 = f.flush()
        assert r1 + r2 + r3 + r4 + r5 + r6 == "Before  After"


# ---------------------------------------------------------------------------
# StripBrackets — non-streaming (__call__)
# ---------------------------------------------------------------------------


class TestStripBracketsCall:
    def test_basic(self):
        f = StripBrackets()
        assert f("[Respond in French] Bonjour!") == "Bonjour!"

    def test_multiple_brackets(self):
        f = StripBrackets()
        assert f("[laughs] Ha! [thinking] Sure") == "Ha! Sure"

    def test_no_brackets(self):
        f = StripBrackets()
        assert f("Hello world") == "Hello world"

    def test_empty_result(self):
        f = StripBrackets()
        assert f("[everything]") == ""


# ---------------------------------------------------------------------------
# StripBrackets — streaming (feed/flush)
# ---------------------------------------------------------------------------


class TestStripBracketsStreaming:
    def test_bracket_split_across_chunks(self):
        f = StripBrackets()
        f.reset()
        r1 = f.feed("Hello [lau")
        r2 = f.feed("ghs] world")
        r3 = f.flush()
        assert r1 + r2 + r3 == "Hello  world"

    def test_unclosed_bracket_discarded(self):
        f = StripBrackets()
        f.reset()
        r1 = f.feed("Hello [unclosed")
        r2 = f.flush()
        assert r1 + r2 == "Hello "

    def test_no_brackets_streaming(self):
        f = StripBrackets()
        f.reset()
        r1 = f.feed("Just ")
        r2 = f.feed("text")
        r3 = f.flush()
        assert r1 + r2 + r3 == "Just text"


# ---------------------------------------------------------------------------
# filtered_stream
# ---------------------------------------------------------------------------


async def _async_iter(*items: str) -> AsyncIterator[str]:
    for item in items:
        yield item


class TestFilteredStream:
    async def test_strips_tags(self):
        source = _async_iter(
            "[internal]think[/internal]",
            " Hello ",
            "[internal]more[/internal]",
            "world",
        )
        f = StripInternalTags()
        chunks = [c async for c in filtered_stream(source, f)]
        assert "".join(chunks) == " Hello world"

    async def test_strips_brackets(self):
        source = _async_iter("[laughs]", " Ha!", " [aside]", " Sure")
        f = StripBrackets()
        chunks = [c async for c in filtered_stream(source, f)]
        assert "".join(chunks) == " Ha!  Sure"

    async def test_empty_chunks_skipped(self):
        source = _async_iter("[internal]hidden[/internal]", "", "visible")
        f = StripInternalTags()
        chunks = [c async for c in filtered_stream(source, f)]
        # Empty string chunks should not appear
        assert all(c != "" for c in chunks)
        assert "".join(chunks) == "visible"

    async def test_cross_chunk_tag(self):
        source = _async_iter("Hi [inter", "nal]secret[/inter", "nal] there")
        f = StripInternalTags()
        chunks = [c async for c in filtered_stream(source, f)]
        result = "".join(chunks)
        assert "secret" not in result
        assert "Hi" in result
        assert "there" in result
