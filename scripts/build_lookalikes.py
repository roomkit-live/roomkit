"""Build ``src/roomkit/_lookalike_data.py`` from Unicode's confusables and NFKC.

Usage::

    uv run python scripts/build_lookalikes.py [path/to/confusables.txt]

Without a path, ``confusables.txt`` is downloaded from unicode.org (UTS #39,
latest). The output is a Python module holding, for each ASCII letter, digit,
underscore, angle bracket, slash, colon and apostrophe, the characters a model
reads as it, and the single characters that read as several letters (``ⅵ`` for
``vi``). Run it when Unicode publishes a new confusables.txt, and commit the
result with the version it names. The same confusables.txt, Python (its
Unicode data) and ruff reproduce the module byte for byte; it records the
source's SHA-256.
"""

from __future__ import annotations

import hashlib
import re
import subprocess  # nosec B404
import sys
import unicodedata
import urllib.request
from pathlib import Path

URL = "https://www.unicode.org/Public/security/latest/confusables.txt"
OUTPUT = Path(__file__).resolve().parent.parent / "src" / "roomkit" / "_lookalike_data.py"
TARGETS = "abcdefghijklmnopqrstuvwxyz0123456789_<>/:'\""
SEQUENCE = re.compile(r"[a-z0-9]{2,4}")
WIDTH = 72

LICENSE = """\
UNICODE LICENSE V3

COPYRIGHT AND PERMISSION NOTICE

Copyright (c) 1991-2026 Unicode, Inc.

NOTICE TO USER: Carefully read the following legal agreement. BY
DOWNLOADING, INSTALLING, COPYING OR OTHERWISE USING DATA FILES, AND/OR
SOFTWARE, YOU UNEQUIVOCALLY ACCEPT, AND AGREE TO BE BOUND BY, ALL OF THE
TERMS AND CONDITIONS OF THIS AGREEMENT. IF YOU DO NOT AGREE, DO NOT
DOWNLOAD, INSTALL, COPY, DISTRIBUTE OR USE THE DATA FILES OR SOFTWARE.

Permission is hereby granted, free of charge, to any person obtaining a
copy of data files and any associated documentation (the "Data Files") or
software and any associated documentation (the "Software") to deal in the
Data Files or Software without restriction, including without limitation
the rights to use, copy, modify, merge, publish, distribute, and/or sell
copies of the Data Files or Software, and to permit persons to whom the
Data Files or Software are furnished to do so, provided that either (a)
this copyright and permission notice appear with all copies of the Data
Files or Software, or (b) this copyright and permission notice appear in
associated Documentation.

THE DATA FILES AND SOFTWARE ARE PROVIDED "AS IS", WITHOUT WARRANTY OF ANY
KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT OF
THIRD PARTY RIGHTS.

IN NO EVENT SHALL THE COPYRIGHT HOLDER OR HOLDERS INCLUDED IN THIS NOTICE
BE LIABLE FOR ANY CLAIM, OR ANY SPECIAL INDIRECT OR CONSEQUENTIAL DAMAGES,
OR ANY DAMAGES WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS,
WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS ACTION,
ARISING OUT OF OR IN CONNECTION WITH THE USE OR PERFORMANCE OF THE DATA
FILES OR SOFTWARE.

Except as contained in this notice, the name of a copyright holder shall
not be used in advertising or otherwise to promote the sale, use or other
dealings in these Data Files or Software without prior written
authorization of the copyright holder.
"""

Table = dict[str, set[str]]

_QUOTE = '"'

CURATED: dict[str, str] = {
    "a": "\u1d00",
    "b": "\u044c\u0299",
    "d": "\u1d05",
    "e": "\u1d07",
    "f": "\ua730",
    "g": "\u0262",
    "h": "\u029c",
    "i": "\u04cf",
    "j": "\u1d0a",
    "k": "\u03ba\u043a\u1d0b",
    "l": "\u053c\u029f",
    "m": "\u043c\u1d0d",
    "n": "\u0274",
    "p": "\u1d18",
    "q": "\ua7af",
    "r": "\u0280",
    "t": "\u0442\u03c4\u1d1b",
    "x": "\u03c7",
    "<": "\u2329\u27e8\u3008",
    ">": "\u232a\u27e9\u3009",
}
"""What a model reads as a target that UTS #39 does not list: small capitals,
lowercase Cyrillic and Greek letters that read as one (``т``, ``к``, ``τ``),
Armenian ``Լ``, the angle brackets. Each brings along every character Unicode
reads as it (``𝛕`` for ``τ``)."""


def parse(text: str) -> tuple[str, Table, Table, Table]:
    """The version of a confusables.txt and what it lists: the characters that
    read as each target, those that read as a run of letters, and for every
    single character, the characters that read as it."""
    found = re.search(r"^# Version: (\S+)", text, re.MULTILINE)
    if found is None:
        raise ValueError("confusables.txt names no version")
    forms: Table = {target: set() for target in TARGETS}
    sequences: Table = {}
    prototypes: Table = {}
    for line in text.splitlines():
        data = line.split("#", 1)[0].strip()
        if not data:
            continue
        source, target = (field.strip() for field in data.split(";")[:2])
        char, read = _chars(source), _chars(target)
        if len(char) != 1:
            continue
        # UTS #39 spells a double quote's prototype as two apostrophes.
        read = _QUOTE if read == "''" else read
        if len(read) == 1:
            prototypes.setdefault(read, set()).add(char)
        # A skeleton maps an ASCII letter onto letters it is not (m onto rn).
        if not (char.isascii() and len(read) > 1):
            _file(char, read.lower(), forms, sequences)
    return found.group(1), forms, sequences, prototypes


def add_unicode_forms(forms: Table, sequences: Table) -> Table:
    """Every character outside ASCII that NFKC and case folding turn into a
    target or a run of letters (fullwidth, mathematical, circled, ``ⅵ``), or
    whose canonical decomposition is a target letter or digit under accents
    only (``ó``), or that Unicode names a Latin letter with a decoration
    (``ł``, ``ø``); returns what each other character's forms fold onto."""
    folds: Table = {}
    for code in range(0x80, 0x110000):
        if 0xD800 <= code <= 0xDFFF:
            continue
        char = chr(code)
        read = unicodedata.normalize("NFKC", char).casefold()
        _file(char, read, forms, sequences)
        if read != char:
            folds.setdefault(read, set()).add(char)
        base = _accented_base(char) or _decorated_base(char)
        if base is not None:
            forms[base].add(char)
    return folds


def _accented_base(char: str) -> str | None:
    """The target letter or digit *char* decomposes into under accents only."""
    parts = unicodedata.normalize("NFD", char)
    base = parts[0].casefold()
    accents = parts[1:]
    if accents and base.isascii() and base.isalnum():
        return base if all(map(unicodedata.combining, accents)) else None
    return None


_DECORATED = re.compile(r"LATIN (?:SMALL|CAPITAL) LETTER ([A-Z]) WITH ")


def _decorated_base(char: str) -> str | None:
    """The Latin letter *char* is under a stroke, a bar, a hook or another
    decoration Unicode does not decompose (``ł``, ``ø``, ``đ``)."""
    found = _DECORATED.match(unicodedata.name(char, ""))
    return found.group(1).lower() if found else None


def add_curated(forms: Table, prototypes: Table, folds: Table) -> None:
    """:data:`CURATED`, each form with what Unicode reads as it."""
    for target, chars in CURATED.items():
        for char in chars:
            forms[target] |= {char, *prototypes.get(char, ()), *folds.get(char.casefold(), ())}


def _file(char: str, read: str, forms: Table, sequences: Table) -> None:
    if read in forms:
        if char != read:
            forms[read].add(char)
    elif SEQUENCE.fullmatch(read):
        sequences.setdefault(read, set()).add(char)


def _chars(codes: str) -> str:
    return "".join(chr(int(code, 16)) for code in codes.split())


def build(raw: bytes) -> str:
    """The module for a confusables.txt's bytes."""
    version, forms, sequences, prototypes = parse(raw.decode("utf-8-sig"))
    folds = add_unicode_forms(forms, sequences)
    add_curated(forms, prototypes, folds)
    return render(version, hashlib.sha256(raw).hexdigest(), forms, sequences)


def render(version: str, digest: str, forms: Table, sequences: Table) -> str:
    """The module's source: the license, the versions, then both tables."""
    lines = [
        '"""Characters a model reads as an ASCII letter, digit or mark, and those',
        "that read as several letters: generated by ``scripts/build_lookalikes.py``",
        f"from Unicode's confusables.txt (UTS #39, version {version}), the",
        f"compatibility and canonical forms of Unicode {unicodedata.unidata_version}",
        "(NFKC with case folding, accented letters) and the script's own list of",
        "what UTS #39 does not hold. Do not edit; run the script again.",
        "",
        "The data is derived from Unicode data files, under this notice:",
        "",
        *LICENSE.splitlines(),
        '"""',
        "",
        "from __future__ import annotations",
        "",
        f'CONFUSABLES_VERSION = "{version}"',
        f'CONFUSABLES_SHA256 = "{digest}"',
        f'UNICODE_VERSION = "{unicodedata.unidata_version}"',
        "",
        "FORMS: dict[str, str] = {",
        *_entries(forms),
        "}",
        '"""For each target, the characters that read as it, in code point order."""',
        "",
        "SEQUENCES: dict[str, str] = {",
        *_entries(sequences),
        "}",
        '"""For each run of two to four letters, the characters that read as it."""',
        "",
    ]
    return "\n".join(lines)


def _entries(table: Table) -> list[str]:
    lines = []
    for key in sorted(table):
        chunks = _chunks(sorted(table[key]))
        if not chunks:
            continue
        lines.append(f"    {_literal(key)}: (")
        lines += [f'        "{chunk}"' for chunk in chunks]
        lines.append("    ),")
    return lines


def _chunks(chars: list[str]) -> list[str]:
    chunks, current = [], ""
    for char in chars:
        escaped = _escape(char)
        if len(current) + len(escaped) > WIDTH:
            chunks.append(current)
            current = ""
        current += escaped
    return [*chunks, current] if current else chunks


def _escape(char: str) -> str:
    code = ord(char)
    if 0x20 <= code < 0x7F and char not in '"\\':
        return char
    return f"\\u{code:04x}" if code < 0x10000 else f"\\U{code:08x}"


def _literal(key: str) -> str:
    return f'"{"".join(map(_escape, key))}"'


def main(argv: list[str]) -> None:
    if len(argv) > 1:
        raw = Path(argv[1]).read_bytes()
    else:
        with urllib.request.urlopen(URL, timeout=60) as response:  # nosec B310
            raw = response.read()
    OUTPUT.write_text(build(raw), encoding="utf-8")
    # Formatted as the repository formats it, so a run on the same data
    # reproduces the committed file byte for byte.
    # A fixed command: this interpreter's ruff on the file just written.
    command = [sys.executable, "-m", "ruff", "format", "--quiet", str(OUTPUT)]
    subprocess.run(command, check=True)  # nosec B603
    print(f"Wrote {OUTPUT} (Unicode {unicodedata.unidata_version})")


if __name__ == "__main__":
    main(sys.argv)
