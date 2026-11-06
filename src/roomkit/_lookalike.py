"""Text as a model reads it: a word, a phrase or a tag's name in any of the
forms a reader takes for it (RFC §6.4).

A model reads past an invisible character and a combining mark, takes what
Unicode lists as confusable with a Latin letter (a fullwidth, mathematical or
accented letter, a Cyrillic, Greek, Armenian, Coptic or Cherokee one, ``0``
for ``o``) and a small capital for that letter, a character such as ``ⅵ``
for the letters it spells, a copy wrapped in markup for the words it
wraps, and a tag name's separator as any dash, underscore or space, or none.
The forms come from :mod:`roomkit._lookalike_data`, generated ahead of
time. The finders of a closing tag (:func:`roomkit._text.fence`), of a copy
of a runtime mark and of a name that reads as the agent's are built from
these patterns, compiled with ``re.IGNORECASE``. No class between two letters
reads a letter: a run of such a character is then scanned once.
"""

from __future__ import annotations

import functools
import re
import sys
import unicodedata

from roomkit._lookalike_data import FORMS, SEQUENCES

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


_IOTA = "(?-i:\u0345)"
"""The combining iota, matched as itself only: under case folding it is the
Greek iota, one of ``i``'s look-alikes. A class that holds it beside a letter
class that reads it too would read a run of it in quadratic time, so the
combining marks' classes leave it to this, and a letter's class refuses it."""

_COMBINING = r"\u0300-\u0344\u0346-\u036f\u20d0-\u20ff\ufe20-\ufe2f"
"""The combining marks a model reads past (underlined, struck letters), U+0345
aside (:data:`_IOTA`)."""

JOINT = rf"(?:[{INVISIBLE}\t\n\x0b\x0c\r\u2028\u2029{_COMBINING}]|{_IOTA})"
"""One character that may sit between a tag's letters for a model still to
read the name: an invisible, control or line-break character, a combining
mark."""

_NAME_JOIN = f"{JOINT}*"

SPACE = rf"(?:[\s{INVISIBLE}{_COMBINING}\u2800\ufff9-\ufffb]|{_IOTA})"
"""One character of the room between a tag's brackets, slash and name:
spacing, the invisible characters, combining marks, a braille blank,
interlinear annotation marks."""

GAP = f"{SPACE}*"

_DASHES = (
    "\\-\u058a\u05be\u1400\u1806\u2010-\u2015\u2e17\u2e1a\u2e3a\u2e3b\u2e40\u2e5d"
    "\u301c\u3030\u30a0\ufe32\ufe58\ufe63\uff0d\U00010d6e\U00010ead\u2212"
)
"""Every dash Unicode 16.0 lists (Dash_Punctuation) and the minus sign, as the
body of a class. U+FE31, a vertical em dash, is left out: Unicode's
confusables read it as ``l``, and a name's ``l`` reads it."""

_UNDERSCORES = "_" + re.escape(FORMS["_"])
"""The underscore and the forms Unicode's confusables read as it, as the body
of a class."""

SEPARATOR = rf"(?:[\s{INVISIBLE}{_COMBINING}\u2800\ufff9-\ufffb{_DASHES}{_UNDERSCORES}]|{_IOTA})"
"""One character a model reads as a tag name's separator, or past it: what
:data:`SPACE` takes, a dash, an underscore in any of their forms. A run of
them, or none, sits where a name holds one (RMK-1320)."""

_SEPARATED = f"{SEPARATOR}*"

_NAME_BREAK = re.compile(f"{SEPARATOR}+")
"""A run of separators in a tag's name as written."""

_MARKUP = r"*_~`\-/|\\"
"""Markup a copy of a phrase may wrap or join its words with (bold, struck,
code, hyphenated, slashed)."""

_PHRASE_SPACE = rf"\s{INVISIBLE}{_COMBINING}{_MARKUP}"
"""What may sit between any phrase's words, as the body of a class: spacing,
invisible and combining characters, markup."""


COLON = FORMS[":"]
"""The characters a model reads as a colon (the Devanagari visarga ``ः``
among them), from Unicode's confusables."""

QUOTE = FORMS['"']
"""The characters a model reads as a double quote (``ײ``, ``〃``), from
Unicode's confusables."""

_TURKISH_I = frozenset("iI\u0131\u0130")
"""Letters ``re.IGNORECASE`` takes for one another beyond their simple case
mappings."""


def lookalikes() -> dict[str, str]:
    """Each lowercase ASCII letter, digit, underscore, bracket, slash, colon and
    apostrophe, as a regular expression class of the characters a model reads
    as it in a tag's name (:func:`char_class`)."""
    return _table(phrase=False)


@functools.cache
def _table(*, phrase: bool) -> dict[str, str]:
    """:data:`FORMS` as regular expression classes. A form that may also sit
    between a name's letters, or between a phrase's words when *phrase* (``|``
    is markup there), is left out: a class holding it beside one that reads it
    too would be read in quadratic time. Built once, on first use."""
    spacing = _spacing(phrase)
    return {char: _class(char, set(forms), spacing) for char, forms in FORMS.items()}


@functools.cache
def _sequences(*, phrase: bool) -> dict[str, str]:
    """Each run of two to four letters a single character reads as (``vi`` for
    ``ⅵ``), with a regular expression class of those characters."""
    spacing = _spacing(phrase)
    return {run: _class("", set(chars), spacing) for run, chars in SEQUENCES.items()}


def _spacing(phrase: bool) -> re.Pattern[str]:
    room = f"{JOINT}|{SPACE}|{phrase_space('')}" if phrase else f"{JOINT}|{SPACE}"
    return re.compile(room, re.IGNORECASE)


def _class(char: str, forms: set[str], spacing: re.Pattern[str]) -> str:
    """A class of *char* and those of its *forms* that are not *spacing*. A
    form whose other case reads as something else (``I`` is an ``l``, its
    ``i`` is not; ``ſ`` is an ``f``, its ``s`` is not) is matched in its own
    case only."""
    kept = {form for form in forms if form != char and not spacing.fullmatch(form)}
    members = {char, *kept} - {""}
    cases = members | {m.lower() for m in members} | {m.upper() for m in members}
    exact = sorted(form for form in kept if not _case_safe(form, cases))
    loose = "".join(map(re.escape, sorted(members - set(exact))))
    if not exact:
        return f"[{loose}]"
    return f"(?:(?-i:[{''.join(map(re.escape, exact))}])|[{loose}])"


def _case_safe(form: str, cases: set[str]) -> bool:
    """Whether every character ``re.IGNORECASE`` takes for *form* is in
    *cases*, a class's members in either case."""
    others = {form.lower(), form.upper(), form.casefold(), form.swapcase()}
    if form in _TURKISH_I:
        others |= _TURKISH_I
    return all(other in cases for other in others - {form} if len(other) == 1)


def char_class(char: str, *, phrase: bool = False) -> str:
    """A regular expression of *char* as a model reads it, one character; one
    the look-alike table does not hold (a custom tag's hyphen or accented
    letter) is matched as it is. *phrase* when it sits between a phrase's
    words (:func:`phrase_space`)."""
    forms = _table(phrase=phrase).get(char.lower()) or f"[{re.escape(char)}]"
    return f"(?!{_IOTA}){forms}"


_MAX_SPELLINGS = 64
"""Spellings of one word a pattern holds in full; a word with more takes its
runs from the left, without overlap."""


def word_pattern(word: str, *, phrase: bool = False) -> str:
    """*word* as a model reads it, a regular expression to compile with
    ``re.IGNORECASE``: each letter in any of its forms, any run of its letters
    as the one character that reads as it (``ⅵ`` for ``vi``), an invisible,
    control or combining character between them. Every way to spell the word
    with such runs is held, up to :data:`_MAX_SPELLINGS`; each run's class
    shares no character with its first letter's, so the alternatives never
    read the same text twice. *phrase* when the word sits in a phrase."""
    return _spelled_word(word, phrase, frozenset())


def name_pattern(name: str) -> str:
    """*name*, a tag's, as a model reads it, a regular expression to compile
    with ``re.IGNORECASE``: :func:`word_pattern` of its letters, and where a
    run of separators (a dash, an underscore, a space, in any of their forms)
    sits between two of them, any run of :data:`SEPARATOR` or none
    (``</tool‐result>``, ``</tool result>``, ``</toolresult>`` for
    ``tool_result``). A separator at the name's start or end is read as
    written. One quantified class between two letters, as everywhere."""
    letters, breaks, pos = [], set(), 0
    for run in _NAME_BREAK.finditer(name):
        if run.start() == 0 or run.end() == len(name):
            continue
        letters.append(name[pos : run.start()])
        pos = run.end()
        breaks.add(sum(map(len, letters)))
    letters.append(name[pos:])
    return _spelled_word("".join(letters), False, frozenset(breaks))


def _spelled_word(word: str, phrase: bool, breaks: frozenset[int]) -> str:
    runs = _runs(word, phrase, breaks)
    if _spellings(word, runs) > _MAX_SPELLINGS:
        runs = _leftmost(word, runs)
    return _spelled(word, 0, runs, phrase, breaks)


def _runs(word: str, phrase: bool, breaks: frozenset[int]) -> dict[int, list[int]]:
    """For each position of *word*, the sizes of the runs a single character
    reads as there, none across a name's separator (*breaks*)."""
    sequences = _sequences(phrase=phrase)
    runs = {}
    for at in range(len(word)):
        ends = (at + size for size in (2, 3, 4) if at + size <= len(word))
        whole = (end for end in ends if not any(at < cut < end for cut in breaks))
        sizes = [end - at for end in whole if word[at:end].lower() in sequences]
        if sizes:
            runs[at] = sizes
    return runs


def _spellings(word: str, runs: dict[int, list[int]]) -> int:
    count = [0] * len(word) + [1]
    for at in range(len(word) - 1, -1, -1):
        count[at] = count[at + 1] + sum(count[at + n] for n in runs.get(at, []))
    return count[0]


def _leftmost(word: str, runs: dict[int, list[int]]) -> dict[int, list[int]]:
    kept, at = {}, 0
    while at < len(word):
        if at in runs:
            kept[at] = [max(runs[at])]
            at += kept[at][0]
        else:
            at += 1
    return kept


def _spelled(
    word: str, at: int, runs: dict[int, list[int]], phrase: bool, breaks: frozenset[int]
) -> str:
    """The pattern of *word* from *at*: its letter there, or a run's character;
    a name's separator before each position of *breaks*."""
    if at == len(word):
        return ""
    unit = char_class(word[at], phrase=phrase)
    choices = [_then(unit, word, at + 1, runs, phrase, breaks)]
    for size in runs.get(at, []):
        glyphs = _sequences(phrase=phrase)[word[at : at + size].lower()]
        choices.append(_then(glyphs, word, at + size, runs, phrase, breaks))
    return choices[0] if len(choices) == 1 else f"(?:{'|'.join(choices)})"


def _then(
    unit: str,
    word: str,
    at: int,
    runs: dict[int, list[int]],
    phrase: bool,
    breaks: frozenset[int],
) -> str:
    rest = _spelled(word, at, runs, phrase, breaks)
    join = _SEPARATED if at in breaks else _NAME_JOIN
    return f"{unit}{join}{rest}" if rest else unit


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
    marks = phrase_marks(phrase)
    gap = f"{phrase_space(marks)}*"
    body = gap.join(word_pattern(word, phrase=True) for word in phrase_words(phrase))
    end = rf"(?:{gap}\]|[{marks}]+)?" if marks else rf"(?:{gap}\])?"
    bracket = rf"[\[\uff3b]{gap}"
    opening = bracket if bracketed else f"(?:{bracket})?"
    return rf"{opening}{body}(?![^\W_]){end}"


def phrase_words(phrase: str) -> list[str]:
    """*phrase*'s words, as :func:`phrase_pattern` reads them in order."""
    return re.findall(r"[\w']+", phrase)


def phrase_marks(phrase: str) -> str:
    """*phrase*'s own punctuation, escaped for a class: what
    :func:`phrase_pattern` takes between its words and after its last
    (:func:`phrase_space`). Never a bracket."""
    return re.escape("".join(sorted(set(re.findall(r"[^\w\s'\[\]]", phrase)))))


def phrase_space(marks: str) -> str:
    """One character that may sit between a phrase's words: spacing, *marks*
    (the phrase's own punctuation, escaped), markup, a combining mark, an
    invisible character. Never a bracket: a run of brackets then starts no
    copy at each of its brackets."""
    return rf"(?:[{_PHRASE_SPACE}{marks}]|{_IOTA})"


def phrase_space_chars(marks: str) -> str:
    """The characters :func:`phrase_space` takes, U+0345 among them, as the
    body of a class to read in its own case: ``[^...]`` is then every other
    character, each tried once."""
    return rf"{_PHRASE_SPACE}{marks}\u0345"


@functools.cache
def _reads() -> dict[str, str]:
    """Each form in the look-alike tables, with the letters it reads as."""
    reads = {char: run for run, chars in SEQUENCES.items() for char in chars}
    reads |= {form: target for target, forms in FORMS.items() for form in forms}
    return reads


@functools.cache
def _joint() -> re.Pattern[str]:
    return re.compile(JOINT, re.IGNORECASE)


@functools.lru_cache(maxsize=4096)
def skeletons(text: str) -> frozenset[str]:
    """*text* as a model reads it, to tell whether two names read alike: each
    character as the letters the look-alike tables read it as, in lower case;
    an invisible character, a mark on a Latin letter (``Alice҉``), spacing and
    punctuation left out (``A.lice``, ``Jean Luc`` and ``Jean-Luc``); once as
    written and once in lower case, since a capital may read as another
    letter than its lowercase (``Ian`` reads like ``lan``, ``ALICE`` like
    ``Alice``). Two names read alike when they share one. A mark on another
    script's letter (a Devanagari vowel sign) is kept: it makes another
    name."""
    bare = _joint().sub("", unicodedata.normalize("NFC", text))
    return frozenset({_skeleton(bare), _skeleton(bare.lower())})


def _skeleton(text: str) -> str:
    reads = _reads()
    kept: list[str] = []
    for char in text:
        if unicodedata.category(char).startswith("M"):
            if kept and kept[-1].isascii():
                continue
            kept.append(char)
            continue
        read = reads.get(char) or reads.get(char.lower()) or char.lower()
        kept += [letter for letter in read if letter.isalnum()]
    return "".join(kept)


def reads_as(text: str, word: str) -> bool:
    """Whether *text* reads as *word* to a model: each letter in any of its forms
    (case, Unicode's confusables, a run as the character that spells it),
    invisible characters ignored, any spacing, punctuation or mark around it
    (``You.``, ``YOU!``)."""
    letters = word_pattern(word, phrase=True)
    return re.fullmatch(rf"[\W_]*{letters}[\W_]*", text, re.IGNORECASE) is not None


@functools.cache
def hidden_marks() -> frozenset[str]:
    """The combining marks no reader sees that a phrase's pattern does not read
    past (RFC §6.4): held apart, so that a pattern does not repeat them at each
    letter."""
    read = re.compile(rf"[{_COMBINING}{INVISIBLE}]")
    marks = (chr(code) for code in range(sys.maxunicode + 1))
    return frozenset(
        mark
        for mark in marks
        if unicodedata.category(mark).startswith("M") and not read.match(mark)
    )


def replaced_past_hidden_marks(pattern: re.Pattern[str], replacement: str, text: str) -> str:
    """*text* with each match of *pattern* replaced, then each match of a reading
    of it without its :func:`hidden_marks`, wherever one sits: first as written,
    since a look-alike of a letter may be such a mark. What a match does not
    cover is kept as it is."""
    text = pattern.sub(lambda _match: replacement, text)
    hidden = hidden_marks()
    if hidden.isdisjoint(text):
        return text
    offsets = [at for at, char in enumerate(text) if char not in hidden]
    offsets.append(len(text))
    read = "".join(text[at] for at in offsets[:-1])
    pieces, at = [], 0
    for copy in pattern.finditer(read):
        pieces += [text[at : offsets[copy.start()]], replacement]
        at = offsets[copy.end()]
    return "".join(pieces) + text[at:] if pieces else text
