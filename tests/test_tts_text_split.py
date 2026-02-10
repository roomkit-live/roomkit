"""Tests for the _split_text helper in the sherpa-onnx TTS provider."""

from roomkit.voice.tts.sherpa_onnx import _split_text


class TestSplitTextSentences:
    def test_splits_on_sentence_boundaries(self):
        text = "Hello world. How are you? I am fine! Great."
        chunks = _split_text(text)
        # Short fragments get merged, but sentence boundaries are respected
        assert len(chunks) >= 1
        joined = " ".join(chunks)
        assert "Hello world." in joined
        assert "How are you?" in joined
        assert "I am fine!" in joined

    def test_preserves_all_content(self):
        text = "First sentence. Second sentence. Third sentence."
        chunks = _split_text(text)
        joined = " ".join(chunks)
        assert "First sentence." in joined
        assert "Second sentence." in joined
        assert "Third sentence." in joined


class TestSplitTextParagraphs:
    def test_splits_on_double_newlines(self):
        text = "First paragraph here.\n\nSecond paragraph here."
        chunks = _split_text(text)
        assert len(chunks) == 2
        assert chunks[0] == "First paragraph here."
        assert chunks[1] == "Second paragraph here."

    def test_multiple_paragraphs(self):
        text = "Para one.\n\nPara two.\n\nPara three."
        chunks = _split_text(text)
        assert len(chunks) == 3


class TestSplitTextLongLines:
    def test_breaks_long_chunk_on_whitespace(self):
        # Create a string that exceeds max_chars
        words = ["word"] * 100  # 100 * 5 = 500 chars + spaces
        text = " ".join(words)
        chunks = _split_text(text, max_chars=50)
        assert all(len(c) <= 50 for c in chunks)
        # Reassembled text should contain all words
        assert " ".join(chunks).count("word") == 100

    def test_breaks_on_newline_before_whitespace(self):
        line1 = "A" * 40
        line2 = "B" * 40
        text = f"{line1}\n{line2}"
        chunks = _split_text(text, max_chars=50)
        assert len(chunks) == 2
        assert chunks[0] == line1
        assert chunks[1] == line2


class TestSplitTextShortText:
    def test_short_text_returns_single_chunk(self):
        text = "Hello world"
        chunks = _split_text(text)
        assert chunks == ["Hello world"]

    def test_empty_text_returns_empty_list(self):
        assert _split_text("") == []
        assert _split_text("   ") == []

    def test_single_sentence(self):
        text = "This is a single sentence."
        chunks = _split_text(text)
        assert chunks == ["This is a single sentence."]


class TestSplitTextPoem:
    def test_poem_with_line_breaks(self):
        poem = "Roses are red,\nViolets are blue.\n\nSugar is sweet,\nAnd so are you."
        chunks = _split_text(poem)
        # Should split on paragraph break
        assert len(chunks) >= 2
        joined = " ".join(chunks)
        assert "Roses are red" in joined
        assert "Violets are blue" in joined
        assert "Sugar is sweet" in joined
        assert "And so are you" in joined

    def test_long_poem_splits_into_manageable_chunks(self):
        # Simulate a long poem with many stanzas
        stanzas = []
        for i in range(10):
            stanzas.append(f"This is stanza {i} line one.\nThis is stanza {i} line two.")
        poem = "\n\n".join(stanzas)
        chunks = _split_text(poem)
        assert len(chunks) >= 5
        assert all(len(c) <= 300 for c in chunks)


class TestSplitTextMerging:
    def test_merges_short_fragments(self):
        # Fragments shorter than 40 chars get merged with previous
        text = "Hi. Ok. Yes. Sure. Fine. Right. Good. Nice."
        chunks = _split_text(text)
        # Should merge into fewer chunks rather than 8 separate ones
        assert len(chunks) < 8
