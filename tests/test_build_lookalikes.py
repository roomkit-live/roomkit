"""``scripts/build_lookalikes.py``: Unicode's confusables and compatibility forms
turned into the tables ``roomkit._lookalike`` reads (RMK-602)."""

from __future__ import annotations

import importlib.util
import pathlib
from types import ModuleType

from roomkit import _lookalike_data

SAMPLE = """\
# confusables.txt
# Version: 99.0.0
0430 ;\t0061 ;\tMA\t# ( а → a ) CYRILLIC SMALL LETTER A → LATIN SMALL LETTER A
0030 ;\t004F ;\tMA\t# ( 0 → O ) DIGIT ZERO → LATIN CAPITAL LETTER O
0903 ;\t003A ;\tMA\t# ( ः → : ) DEVANAGARI SIGN VISARGA → COLON
00E6 ;\t0061 0065 ;\tMA\t# ( æ → ae ) LATIN SMALL LETTER AE → a, e
0072 006E ;\t006D ;\tMA\t# ( rn → m ) a run read as one letter: not kept
"""


def _script() -> ModuleType:
    path = pathlib.Path(__file__).resolve().parent.parent / "scripts" / "build_lookalikes.py"
    spec = importlib.util.spec_from_file_location("build_lookalikes", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_confusables_are_filed_by_what_they_read_as() -> None:
    version, forms, sequences = _script().parse(SAMPLE)

    assert version == "99.0.0"
    assert forms["a"] == {"\u0430"}
    assert forms["o"] == {"0"}
    assert forms[":"] == {"\u0903"}
    assert sequences == {"ae": {"\u00e6"}}
    assert forms["m"] == set()


def test_the_module_carries_unicode_s_notice_and_versions() -> None:
    script = _script()
    source = script.render(*script.parse(SAMPLE))

    assert "UNICODE LICENSE V3" in source
    assert 'CONFUSABLES_VERSION = "99.0.0"' in source
    assert '"o": (\n        "0"\n    ),' in source
    assert '"ae": (\n        "\\u00e6"\n    ),' in source


def test_the_shipped_tables_name_their_sources() -> None:
    assert _lookalike_data.CONFUSABLES_VERSION == "18.0.0"
    assert "\u2175" in _lookalike_data.SEQUENCES["vi"]
    assert "0" in _lookalike_data.FORMS["o"]
