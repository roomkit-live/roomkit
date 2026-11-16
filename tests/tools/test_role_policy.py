"""Policy resolution and validation boundaries shared with the Rust port."""

from __future__ import annotations

import pytest

from roomkit.tools.policy import RoleOverride, ToolPolicy, alias_admitted, call_admitted


def test_an_empty_role_bypasses_even_its_own_override() -> None:
    policy = ToolPolicy(role_overrides={"": RoleOverride(deny=["*"])})
    assert policy.resolve("") is policy
    assert policy.resolve().is_allowed("anything")


def test_resolving_detaches_lists_and_discards_overrides() -> None:
    override = RoleOverride(allow=["read_*"], deny=["blocked", "extra", "extra"])
    policy = ToolPolicy(
        allow=["*"], deny=["blocked", "blocked"], role_overrides={"member": override}
    )
    resolved = policy.resolve("member")
    assert resolved.deny == ["blocked", "blocked", "extra", "extra"]
    assert resolved.role_overrides == {}
    assert resolved.resolve("member") is resolved
    policy.allow.clear()
    policy.deny.clear()
    override.allow.clear()
    override.deny.clear()
    assert resolved.allow == ["*"]
    assert not resolved.is_allowed("write_file")
    assert resolved.is_allowed("read_file")


def test_serialization_omits_the_private_intersection() -> None:
    policy = ToolPolicy(allow=["*"], role_overrides={"reader": RoleOverride(allow=["read_*"])})
    resolved = policy.resolve("reader")
    assert not resolved.is_allowed("write_file")
    assert resolved.model_dump() == {"allow": ["*"], "deny": [], "role_overrides": {}}
    assert ToolPolicy(**resolved.model_dump()).is_allowed("write_file")


def test_the_filter_reads_later_policy_mutations() -> None:
    policy = ToolPolicy(allow=["read_*"])
    filter_ = policy.as_filter()
    assert filter_("read_file")
    policy.deny.append("read_file")
    assert not filter_("read_file")


def test_resolution_validates_only_the_rules_it_constructs() -> None:
    policy = ToolPolicy(
        role_overrides={"replace": RoleOverride(mode="replace"), "restrict": RoleOverride()}
    )
    policy.deny.append("rm_[a-z]*")
    assert policy.resolve(None) is policy
    assert policy.resolve("replace").is_allowed("rm_file")
    with pytest.raises(ValueError, match="holds a bracket"):
        policy.resolve("restrict")


def test_alias_admission_applies_both_allow_constraints() -> None:
    policy = ToolPolicy(
        allow=["mcp__a__*", "mcp__b__*"],
        role_overrides={"reader": RoleOverride(allow=["mcp__b__read_*"])},
    ).resolve("reader")
    assert not alias_admitted(policy, "mcp__a__read_file")
    assert alias_admitted(policy, "mcp__b__read_file")
    assert not alias_admitted(policy, "mcp__b__write_file")


def test_a_retained_override_is_not_revalidated_by_the_policy_constructor() -> None:
    override = RoleOverride()
    override.allow.append("rm_[a-z]*")
    policy = ToolPolicy(role_overrides={"member": override})
    assert policy.role_overrides["member"] is override
    assert policy.resolve("member").is_allowed("rm_[a-z]file")
    assert not policy.resolve("member").is_allowed("rm_file")


@pytest.mark.parametrize("field", ["allow", "deny"])
def test_a_raw_override_is_validated_on_construction(field: str) -> None:
    with pytest.raises(ValueError, match="holds a bracket"):
        ToolPolicy(role_overrides={"member": {field: ["rm_[a-z]*"]}})


@pytest.mark.parametrize(
    ("mode", "field"), [("replace", "allow"), ("replace", "deny"), ("restrict", "deny")]
)
def test_resolving_revalidates_constructed_lists_from_a_mutated_override(
    mode: str, field: str
) -> None:
    override = RoleOverride(mode=mode)
    getattr(override, field).append("rm_[a-z]*")
    policy = ToolPolicy(role_overrides={"member": override})
    with pytest.raises(ValueError, match="holds a bracket"):
        policy.resolve("member")


def test_clone_and_equality_keep_the_private_intersection() -> None:
    resolved = ToolPolicy(
        allow=["*"], role_overrides={"reader": RoleOverride(allow=["read_*"])}
    ).resolve("reader")
    clone = resolved.model_copy(deep=True)
    plain = ToolPolicy.model_validate(resolved.model_dump())
    assert clone == resolved
    assert plain != resolved
    assert not clone.is_allowed("write_file")
    assert plain.is_allowed("write_file")
    clone.deny.append("read_file")
    assert resolved.is_allowed("read_file")
    assert not clone.is_allowed("read_file")


def test_resolving_a_new_role_replaces_the_previous_private_constraint() -> None:
    resolved = ToolPolicy(
        allow=["*"], role_overrides={"reader": RoleOverride(allow=["read_*"])}
    ).resolve("reader")
    assert resolved.resolve(None) is resolved
    assert resolved.resolve("unknown") is resolved
    resolved.role_overrides["writer"] = RoleOverride(allow=["write_*"])
    writer = resolved.resolve("writer")
    assert writer.is_allowed("write_file")
    assert not writer.is_allowed("read_file")
    assert not resolved.is_allowed("write_file")


def test_a_mutated_base_allow_is_validated_only_when_restrict_constructs_it() -> None:
    policy = ToolPolicy(
        role_overrides={"reader": RoleOverride(), "owner": RoleOverride(mode="replace")}
    )
    policy.allow.append("[invalid]")
    assert policy.resolve(None) is policy
    assert policy.resolve("unknown") is policy
    assert policy.resolve("owner").is_allowed("anything")
    with pytest.raises(ValueError, match="holds a bracket"):
        policy.resolve("reader")


def test_resolve_reads_later_mutations_through_the_role_map() -> None:
    policy = ToolPolicy(role_overrides={"member": RoleOverride()})
    before = policy.resolve("member")
    policy.role_overrides["member"].allow.append("read_*")
    after = policy.resolve("member")
    assert before.is_allowed("write_file")
    assert not after.is_allowed("write_file")
    assert after.is_allowed("read_file")


@pytest.mark.parametrize(
    ("base", "secondary", "deny", "expected"),
    [
        (["mcp__a__*"], [], [], (True, False)),
        ([], ["mcp__a__*"], [], (True, False)),
        (["read_*"], ["search_*"], [], (True, True)),
        (["read_*", "mcp__a__*"], [], [], (True, False)),
        ([], ["read_*", "mcp__a__*"], [], (True, False)),
        (["mcp__*"], ["mcp__a__*"], ["mcp__a__read_*"], (False, False)),
        (["mcp__a__*"], ["mcp__b__*"], [], (False, False)),
    ],
)
def test_alias_admission_keeps_each_server_constraint(
    base: list[str], secondary: list[str], deny: list[str], expected: tuple[bool, bool]
) -> None:
    policy = ToolPolicy(
        allow=base,
        deny=deny,
        role_overrides={"member": RoleOverride(allow=secondary)},
    ).resolve("member")
    actual = tuple(alias_admitted(policy, f"mcp__{server}__read_file") for server in ("a", "b"))
    assert actual == expected


def test_alias_call_must_pass_both_alias_and_bare_names_after_resolution() -> None:
    policy = ToolPolicy(
        allow=["*"],
        role_overrides={"reader": RoleOverride(allow=["mcp__a__*", "read_*"])},
    ).resolve("reader")
    assert call_admitted(policy, "mcp__a__read_file")
    assert not call_admitted(policy, "mcp__a__write_file")
    assert not call_admitted(policy, "mcp__b__read_file")
    assert call_admitted(policy, "read_file")
