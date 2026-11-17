"""Served YAML normalization must verify identically across host processes."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from roomkit.skills.errors import SkillVerificationError
from roomkit.skills.mcp.entry import parse_entry
from roomkit.skills.mcp.verify import verify_frontmatter
from roomkit.skills.parser import read_frontmatter_yaml

_CASES = json.loads(Path(__file__).with_name("served_yaml_cases.json").read_text())


@pytest.mark.parametrize("case", _CASES, ids=lambda case: case["expression"])
def test_served_yaml_normalization_matches_the_frozen_corpus(case: dict[str, Any]) -> None:
    frontmatter = case.get("frontmatter", {"name": "guide", "description": "Read the guide"})
    entry = parse_entry(
        {"uri": "skill://guide/SKILL.md", "frontmatter": frontmatter, "resources": "dynamic"}
    )
    text = f"---\n{case['yaml']}---\nBody"
    if case["valid"]:
        assert read_frontmatter_yaml(case["yaml"]) == frontmatter
        assert verify_frontmatter(entry, text) == "Body"
    else:
        with pytest.raises(SkillVerificationError):
            verify_frontmatter(entry, text)


_SCRIPT = r"""
import hashlib
import json
import sys
from roomkit.skills.mcp.entry import parse_entry
from roomkit.skills.mcp.verify import verify_frontmatter
from roomkit.skills.parser import read_frontmatter_yaml
head = "name: guide\ndescription: Read the guide\n"
head += "meta: !!set {alpha: null, beta: null, gamma: null}\n"
text = "---\n" + head + "---\nBody"
normalized = read_frontmatter_yaml(head)
raw = json.loads(sys.argv[1]) if len(sys.argv) > 1 else {
    "uri": "skill://guide/SKILL.md",
    "frontmatter": normalized,
    "resources": [{"uri": "skill://guide/SKILL.md", "size": len(text.encode()),
                   "digest": "sha256:" + hashlib.sha256(text.encode()).hexdigest()}],
}
entry = parse_entry(raw)
try:
    verify_frontmatter(entry, text)
    verified = True
except Exception as error:
    verified = type(error).__name__
print(json.dumps({"raw": raw, "normalized": normalized,
                  "fingerprint": entry.fingerprint, "verified": verified}))
"""


def _process(seed: int, entry: dict[str, Any] | None = None) -> dict[str, Any]:
    arguments = [sys.executable, "-c", _SCRIPT]
    if entry is not None:
        arguments.append(json.dumps(entry))
    result = subprocess.run(
        arguments,
        env={**os.environ, "PYTHONHASHSEED": str(seed)},
        check=True,
        capture_output=True,
        text=True,
        timeout=15,
    )
    return json.loads(result.stdout)


def test_a_served_set_verifies_the_same_bytes_and_entry_in_every_process() -> None:
    """File hashes and approval fingerprints cannot hide unstable normalization."""
    first = _process(1)
    observations = [first, _process(2, first["raw"]), _process(3, first["raw"])]
    assert all(item["fingerprint"] == first["fingerprint"] for item in observations)
    assert all(item["verified"] is True for item in observations), observations
    assert all(item["normalized"] == first["normalized"] for item in observations)


def test_nested_sets_and_python_equal_members_have_canonical_arrays() -> None:
    from roomkit.skills.parser import read_frontmatter_yaml

    assert read_frontmatter_yaml(
        "meta: {empty: !!set {}, numeric: !!set {1: null, true: null, 1.0: null, 2: null}}"
    ) == {"meta": {"empty": [], "numeric": [1, 2]}}
