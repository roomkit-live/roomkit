"""Served-skill entry boundaries shared with the Rust public model tests."""

from __future__ import annotations

import hashlib
import json
import sys

import pytest

from roomkit.skills.errors import SkillValidationError, SkillVerificationError
from roomkit.skills.mcp import (
    MAX_RESOURCES_PER_SKILL,
    MAX_TOTAL_SIZE,
    SkillEntry,
    SkillResource,
    parse_entry,
    verify_skill_resource,
)

URI = "skill://acme/check/SKILL.md"
DIGEST = "sha256:" + "0" * 64


def _entry(resources: object = "dynamic", *, uri: str = URI) -> dict[str, object]:
    return {
        "uri": uri,
        "frontmatter": {"name": "check", "description": "Check"},
        "resources": resources,
    }


@pytest.mark.parametrize("character", ["x", "é", "😀", "\x00"])
def test_the_frontmatter_bound_counts_the_default_json_characters(
    character: str,
) -> None:
    frontmatter = {"name": "check", "description": "Check", "notes": ""}
    available = 65536 - len(json.dumps(frontmatter))
    width = len(json.dumps(character)) - 2
    frontmatter["notes"] = character * (available // width) + "a" * (available % width)
    assert len(json.dumps(frontmatter)) == 65536
    raw = {**_entry(), "frontmatter": frontmatter}
    entry = parse_entry(raw)
    assert entry.frontmatter == frontmatter
    frontmatter["notes"] += "a"
    assert len(json.dumps(frontmatter)) == 65537
    with pytest.raises(SkillValidationError, match="past 65536 characters"):
        parse_entry(raw)
    assert len(json.dumps(entry.frontmatter)) == 65536


def test_the_total_byte_limit_is_inclusive_and_the_file_limit_has_precedence() -> None:
    resources = [{"uri": URI, "digest": DIGEST, "size": MAX_TOTAL_SIZE}]
    entry = parse_entry(_entry(resources))
    assert entry.total_size == MAX_TOTAL_SIZE
    assert entry.limit_exceeded() is None
    resources.append({"uri": "skill://acme/check/extra", "digest": DIGEST, "size": 1})
    assert parse_entry(_entry(resources)).limit_exceeded() == (
        "its files hold 16777217 bytes, past 16777216"
    )
    resources.extend(
        {"uri": f"skill://acme/check/{index}", "digest": DIGEST, "size": 0}
        for index in range(MAX_RESOURCES_PER_SKILL - 1)
    )
    assert len(resources) == 513
    assert parse_entry(_entry(resources)).limit_exceeded() == "it lists 513 files, past 512"


@pytest.mark.parametrize("scheme", ["skill", "Skill", "custom+v1.2"])
@pytest.mark.parametrize("relative", ["%2e%2e/file", "a?query", "a#fragment", "é/😀"])
def test_paths_keep_their_lexical_identity(scheme: str, relative: str) -> None:
    uri = f"{scheme}://acme/check/SKILL.md"
    entry = parse_entry(_entry(uri=uri))
    expected_root = f"{scheme}://acme/check"
    assert entry.root == expected_root
    assert entry.path == ("acme/check" if scheme == "skill" else expected_root)
    assert entry.resolve(relative) == expected_root + "/" + relative
    assert entry.relative(entry.resolve(relative)) == relative


@pytest.mark.parametrize("content", [b"", b"\x00\xff\x80", "é😀".encode()])
def test_resource_verification_uses_the_bytes(content: bytes) -> None:
    digest = "sha256:" + hashlib.sha256(content).hexdigest()
    resource = SkillResource(URI, digest, len(content))
    verify_skill_resource(resource, content)
    with pytest.raises(SkillVerificationError, match="bytes where the entry lists"):
        verify_skill_resource(resource, content + b"x")
    with pytest.raises(SkillVerificationError, match="digest is not the entry's"):
        verify_skill_resource(SkillResource(URI, DIGEST, len(content)), content)


def test_a_direct_duplicate_keeps_both_records_and_looks_up_the_last() -> None:
    first = SkillResource(URI, "sha256:" + "1" * 64, 2)
    last = SkillResource(URI, "sha256:" + "2" * 64, 3)
    entry = SkillEntry(URI, {"name": "check", "description": "Check"}, (first, last))
    assert entry.resources == (first, last)
    assert entry.resource(URI) is last
    assert entry.total_size == 5
    lines = sorted([f"{first.uri} {first.digest}", f"{last.uri} {last.digest}"])
    assert entry.fingerprint == "sha256:" + hashlib.sha256("\n".join(lines).encode()).hexdigest()


def test_direct_uri_helpers_use_character_slices_without_validation() -> None:
    entry = SkillEntry("é😀abcdefghij", {}, "dynamic")
    assert entry.root == "é😀a"
    assert entry.path == "é😀a"
    assert entry.relative("é😀abTAIL") == "TAIL"
    assert SkillEntry("é😀", {}, "dynamic").root == ""


@pytest.mark.parametrize(("left", "right"), [(True, 1), (False, 0.0), (1, 1.0)])
def test_direct_entry_equality_compares_nested_values_as_python(
    left: object, right: object
) -> None:
    resources = (SkillResource(URI, DIGEST, 1),)
    first = SkillEntry(URI, {"name": "check", "extra": [{"value": left}]}, resources)
    second = SkillEntry(URI, {"extra": [{"value": right}], "name": "check"}, resources)
    assert first == second
    assert first != SkillEntry(URI, {"name": "check", "extra": [{"value": "1"}]}, resources)
    assert first != SkillEntry(URI, first.frontmatter, "dynamic")


def test_parsing_retains_the_digest_regexs_final_newline() -> None:
    content = b"data"
    digest = "sha256:" + hashlib.sha256(content).hexdigest() + "\n"
    entry = parse_entry(_entry([{"uri": URI, "digest": digest, "size": len(content)}]))
    resource = entry.resource(URI)
    assert resource is not None
    assert resource.digest == digest
    with pytest.raises(SkillVerificationError, match="digest is not the entry's"):
        verify_skill_resource(resource, content)


@pytest.mark.parametrize("value", [None, True, 0, 1.5, "read", [], ["read", None, 2]])
def test_gating_keys_preserve_scalars_and_flat_lists(value: object) -> None:
    frontmatter = {"name": "check", "description": "Check", "allowed_tools": value}
    assert parse_entry({**_entry(), "frontmatter": frontmatter}).frontmatter == frontmatter


@pytest.mark.parametrize("depth", [128, 129, 256, 900, 2000])
def test_json_nesting_beyond_yaml_depth_can_be_parsed(depth: int) -> None:
    nested: object = None
    for _ in range(depth):
        nested = [nested]
    frontmatter = {"name": "check", "description": "Check", "nested": nested}
    entry = parse_entry({**_entry(), "frontmatter": frontmatter})
    current = entry.frontmatter["nested"]
    for _ in range(depth):
        assert isinstance(current, list) and len(current) == 1
        current = current[0]
    assert current is None


def test_the_json_recursion_boundary_depends_on_the_python_runtime() -> None:
    nested: object = None
    for _ in range(10000):
        nested = [nested]
    frontmatter = {"name": "check", "description": "Check", "nested": nested}
    # Two brackets per level: this JSON would be far below 65,536 characters.
    assert len(json.dumps({**frontmatter, "nested": None})) + 20000 < 65536
    raw = {**_entry(), "frontmatter": frontmatter}
    if sys.version_info[:2] in {(3, 12), (3, 13)}:
        with pytest.raises(SkillValidationError, match="not plain JSON"):
            parse_entry(raw)
    else:
        entry = parse_entry(raw)
        current = entry.frontmatter["nested"]
        for _ in range(10000):
            assert isinstance(current, list) and len(current) == 1
            current = current[0]
        assert current is None
