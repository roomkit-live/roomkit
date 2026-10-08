"""Text as a model reads it: a word, a phrase or a tag's name in any of the
forms a reader takes for it (RFC §6.4).

A model reads past an invisible character and a combining mark, takes a
fullwidth, mathematical or small-capital letter, or a Cyrillic, Greek or
Armenian homoglyph, for the Latin letter, and a copy wrapped in markup for
the words it wraps. The finders of a closing tag (:func:`roomkit._text.fence`),
of a copy of a runtime mark and of a name that reads as the agent's are built
from these patterns, compiled with ``re.IGNORECASE``. No class between two
letters reads a letter: a run of such a character is then scanned once.
"""

from __future__ import annotations

import functools
import re
import unicodedata

INVISIBLE = (
    "\u00ad\u034f\u061c\u115f\u1160\u17b4\u17b5\u180b-\u180f\u200b-\u200f"
    "\u202a-\u202e\u2060-\u206f\u3164\ufe00-\ufe0f\ufeff\uffa0\ufff0-\ufff8"
    "\U0001bca0-\U0001bca3\U0001d173-\U0001d17a\U000e0000-\U000e0fff"
    "\x00-\x08\x0e-\x1f\x7f-\x9f\ud800-\udfff"
)
"""The characters a text holds without showing them, as the body of a regular
expression's class: those Unicode marks as ignorable by default
(Default_Ignorable_Code_Point: soft hyphen, zero-width spaces and joiners,
direction marks, bidirectional embeddings, overrides and isolates, variation
selectors, tag characters and the like), and the control characters other than
whitespace and the lone surrogates a provider may strip before it sends a
text. A model reads past them, and a stripped one leaves its neighbours
side by side."""


OPEN = "<\u02c2\u1438\u2039\u2329\u276e\u27e8\u3008"
SLASH = "/\u2044\u2215\u2571\u29f8"
CLOSE = ">\u02c3\u1433\u203a\u232a\u276f\u27e9\u3009"
"""A tag's angle brackets and slash as a model reads them beyond what NFKC
folds onto them (fullwidth and small forms, see :func:`lookalikes`): their
modifier, syllabics, quotation, angle, ornament, box-drawing and mathematical
look-alikes."""

_IOTA = "(?-i:\u0345)"
"""The combining iota, matched as itself only: under case folding it is the
Greek iota, one of ``i``'s look-alikes. A class that holds it beside a letter
class that reads it too would read a run of it in quadratic time, so the
combining marks' classes leave it to this, and a letter's class refuses it."""

_COMBINING = r"\u0300-\u0344\u0346-\u036f\u20d0-\u20ff\ufe20-\ufe2f"
"""The combining marks a model reads past (underlined, struck letters), U+0345
aside (:data:`_IOTA`)."""

JOINT = rf"(?:[{INVISIBLE}\t\n\x0b\x0c\r{_COMBINING}]|{_IOTA})"
"""One character that may sit between a tag's letters for a model still to
read the name: an invisible, control or line-break character, a combining
mark."""

_NAME_JOIN = f"{JOINT}*"

SPACE = rf"(?:[\s{INVISIBLE}{_COMBINING}\u2800\ufff9-\ufffb]|{_IOTA})"
"""One character of the room between a tag's brackets, slash and name:
spacing, the invisible characters, combining marks, a braille blank,
interlinear annotation marks."""

GAP = f"{SPACE}*"

_MARKUP = r"*_~`\-/|\\"
"""Markup a copy of a phrase may wrap or join its words with (bold, struck,
code, hyphenated, slashed)."""


_CONFUSABLES = {
    "a": "\u0430\u0251\u03b1\u1d00",
    "b": "\u044c\u0412\u0392\u0299",
    "c": "\u0441\u03f2\u1d04",
    "d": "\u0501\u1d05",
    "e": "\u0435\u04bd\u0395\u1d07",
    "f": "\ua730",
    "g": "\u0261\u0581\u0262",
    "h": "\u04bb\u0570\u041d\u0397\u029c",
    "i": "\u0456\u03b9\u0269\u04cf\u0131\u026a",
    "j": "\u0458\u03f3\u1d0a",
    "k": "\u03ba\u043a\u039a\u1d0b",
    "l": "\u04cf\u01c0\u053c\u029f",
    "m": "\u043c\u039c\u1d0d",
    "n": "\u0578\u039d\u0274",
    "o": "\u043e\u03bf\u03c3\u0585\u1d0f",
    "p": "\u0440\u03c1\u1d18",
    "q": "\u051b\u0566\ua7af",
    "r": "\u0433\u0280",
    "s": "\u0455\ua731",
    "t": "\u0442\u03c4\u03a4\u1d1b",
    "u": "\u03c5\u057d\u1d1c",
    "v": "\u03bd\u0475\u1d20",
    "w": "\u051d\u0461\u1d21",
    "x": "\u0445\u03c7",
    "y": "\u0443\u04af\u03b3\u03a5\u028f",
    "z": "\u1d22\u0396",
}
"""Letters of other scripts a model reads as a Latin one beyond what NFKC folds:
the Cyrillic, Greek and Armenian homoglyphs (``о``, ``ο``, ``օ`` for ``o``), the
capitals whose own lowercase reads as another letter (Greek ``Ν`` is a
capital N, its ``ν`` a v), and the small capitals; other capitals match
through case folding."""


@functools.cache
def lookalikes() -> dict[str, str]:
    """Each lowercase ASCII letter, digit, underscore, bracket and slash, with
    every character NFKC folds onto it under case folding (fullwidth,
    mathematical, circled, small forms: ``ｔ``, ``𝐭``, ``ⓣ``, ``＜``) and its
    homoglyphs in other scripts (:data:`_CONFUSABLES`), as the body of a
    regular expression's class. Read once, on first use."""
    found = {char: [re.escape(char)] for char in "abcdefghijklmnopqrstuvwxyz0123456789_<>/"}
    for code in range(0x80, 0x110000):
        if 0xD800 <= code <= 0xDFFF:
            continue
        folded = unicodedata.normalize("NFKC", chr(code)).casefold()
        if folded in found:
            found[folded].append(re.escape(chr(code)))
    for char, homoglyphs in _CONFUSABLES.items():
        found[char] += map(re.escape, homoglyphs)
    return {char: "".join(forms) for char, forms in found.items()}


def char_class(char: str, more: str = "") -> str:
    """A regular expression class of *char* as a model reads it; a character
    the look-alike table does not hold (a custom tag's hyphen or accented
    letter) is matched as it is."""
    forms = lookalikes().get(char.lower()) or re.escape(char)
    return f"(?!{_IOTA})[{forms}{re.escape(more)}]"


_APOSTROPHES = "'\u2019\u02bc"


def word_pattern(word: str) -> str:
    """*word* as a model reads it, a regular expression to compile with
    ``re.IGNORECASE``: each letter in any of its forms (NFKC look-alikes,
    homoglyphs), an apostrophe straight or typographic, an invisible, control
    or combining character between the letters."""
    return _NAME_JOIN.join(
        f"[{_APOSTROPHES}]" if char in _APOSTROPHES else char_class(char) for char in word
    )


def phrase_pattern(phrase: str, *, bracketed: bool = False) -> str:
    """*phrase* as a model reads it, a regular expression to compile with
    ``re.IGNORECASE``: its words in order (:func:`word_pattern`), with any
    spacing, its own punctuation, markup (bold, struck, code, hyphens,
    slashes), combining marks and invisible characters between them (none
    included), its opening bracket, square or fullwidth, optional unless
    *bracketed*, its last word not part of a longer one, and the closing
    punctuation taken with the copy.

    One quantified class between two words, never two in a row: a long run of
    spaces after a partial copy is then scanned once, not once per split."""
    marks = re.escape("".join(sorted(set(re.findall(r"[^\w\s'\[\]]", phrase)))))
    gap = f"{phrase_space(marks)}*"
    body = gap.join(map(word_pattern, re.findall(r"[\w']+", phrase)))
    end = rf"(?:{gap}\]|[{marks}]+)?" if marks else rf"(?:{gap}\])?"
    bracket = rf"[\[\uff3b]{gap}"
    opening = bracket if bracketed else f"(?:{bracket})?"
    return rf"{opening}{body}(?![^\W_]){end}"


def phrase_space(marks: str) -> str:
    """One character that may sit between a phrase's words: spacing, *marks*
    (the phrase's own punctuation, escaped), markup, a combining mark, an
    invisible character. Never a bracket: a run of brackets then starts no
    copy at each of its brackets."""
    return rf"(?:[\s{INVISIBLE}{_COMBINING}{_MARKUP}{marks}]|{_IOTA})"


def reads_as(text: str, word: str) -> bool:
    """Whether *text* reads as *word* to a model: each letter in any of its forms
    (case, NFKC look-alikes, homoglyphs), invisible characters ignored, any
    spacing, punctuation or mark around it (``You.``, ``YOU!``)."""
    return re.fullmatch(rf"[\W_]*{word_pattern(word)}[\W_]*", text, re.IGNORECASE) is not None
