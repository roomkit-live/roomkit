"""Unit tests for ToolPolicy."""

from __future__ import annotations

from roomkit.tools.policy import RoleOverride, ToolPolicy


class TestEmptyPolicy:
    def test_empty_allows_all(self) -> None:
        policy = ToolPolicy()
        assert policy.is_allowed("anything") is True
        assert policy.is_allowed("mcp_search") is True

    def test_empty_lists_allows_all(self) -> None:
        policy = ToolPolicy(allow=[], deny=[])
        assert policy.is_allowed("anything") is True


class TestAllowPatterns:
    def test_exact_match(self) -> None:
        policy = ToolPolicy(allow=["search"])
        assert policy.is_allowed("search") is True
        assert policy.is_allowed("delete") is False

    def test_glob_match(self) -> None:
        policy = ToolPolicy(allow=["mcp_*"])
        assert policy.is_allowed("mcp_search") is True
        assert policy.is_allowed("mcp_delete") is True
        assert policy.is_allowed("local_search") is False

    def test_multiple_allow_patterns(self) -> None:
        policy = ToolPolicy(allow=["search", "read_*"])
        assert policy.is_allowed("search") is True
        assert policy.is_allowed("read_file") is True
        assert policy.is_allowed("delete_file") is False


class TestDenyPatterns:
    def test_exact_deny(self) -> None:
        policy = ToolPolicy(deny=["delete"])
        assert policy.is_allowed("delete") is False
        assert policy.is_allowed("search") is True

    def test_glob_deny(self) -> None:
        policy = ToolPolicy(deny=["delete_*"])
        assert policy.is_allowed("delete_file") is False
        assert policy.is_allowed("delete_user") is False
        assert policy.is_allowed("search") is True


class TestDenyOverridesAllow:
    def test_deny_wins(self) -> None:
        policy = ToolPolicy(allow=["*"], deny=["delete"])
        assert policy.is_allowed("search") is True
        assert policy.is_allowed("delete") is False

    def test_deny_glob_wins_over_allow_glob(self) -> None:
        policy = ToolPolicy(allow=["mcp_*"], deny=["mcp_delete*"])
        assert policy.is_allowed("mcp_search") is True
        assert policy.is_allowed("mcp_delete") is False
        assert policy.is_allowed("mcp_delete_user") is False


class TestAsFilter:
    def test_as_filter_callable(self) -> None:
        policy = ToolPolicy(allow=["a", "b"], deny=["b"])
        f = policy.as_filter()
        assert callable(f)
        assert f("a") is True
        assert f("b") is False
        assert f("c") is False

    def test_as_filter_with_builtin_filter(self) -> None:
        policy = ToolPolicy(allow=["search", "read_*"])
        names = ["search", "read_file", "delete", "read_log"]
        result = list(filter(policy.as_filter(), names))
        assert result == ["search", "read_file", "read_log"]


class TestEdgeCases:
    def test_deny_only_blocks_matched(self) -> None:
        """Deny-only policy allows everything except denied patterns."""
        policy = ToolPolicy(deny=["bad_*"])
        assert policy.is_allowed("good_tool") is True
        assert policy.is_allowed("bad_tool") is False

    def test_question_mark_glob(self) -> None:
        policy = ToolPolicy(allow=["tool_?"])
        assert policy.is_allowed("tool_a") is True
        assert policy.is_allowed("tool_ab") is False

    def test_serialization_roundtrip(self) -> None:
        policy = ToolPolicy(allow=["a*"], deny=["ab"])
        data = policy.model_dump()
        restored = ToolPolicy(**data)
        assert restored.allow == ["a*"]
        assert restored.deny == ["ab"]
        assert restored.is_allowed("ac") is True
        assert restored.is_allowed("ab") is False


# ---------------------------------------------------------------------------
# Role overrides
# ---------------------------------------------------------------------------


class TestRoleOverrideResolve:
    def test_no_overrides_returns_self(self) -> None:
        policy = ToolPolicy(allow=["search"])
        resolved = policy.resolve("member")
        assert resolved is policy

    def test_none_role_returns_self(self) -> None:
        policy = ToolPolicy(allow=["search"], role_overrides={"member": RoleOverride(deny=["x"])})
        resolved = policy.resolve(None)
        assert resolved is policy

    def test_unknown_role_returns_self(self) -> None:
        policy = ToolPolicy(allow=["search"], role_overrides={"member": RoleOverride(deny=["x"])})
        resolved = policy.resolve("unknown_role")
        assert resolved is policy


class TestRoleOverrideRestrict:
    def test_deny_union(self) -> None:
        """Restrict mode: deny lists are merged."""
        policy = ToolPolicy(
            deny=["delete"],
            role_overrides={"observer": RoleOverride(deny=["write_*"])},
        )
        resolved = policy.resolve("observer")
        assert resolved.is_allowed("delete") is False
        assert resolved.is_allowed("write_file") is False
        assert resolved.is_allowed("read_file") is True

    def test_allow_intersection(self) -> None:
        """Restrict mode: tool must match both base AND override allow lists."""
        policy = ToolPolicy(
            allow=["mcp_*", "local_*"],
            role_overrides={"member": RoleOverride(allow=["mcp_search", "mcp_read"])},
        )
        resolved = policy.resolve("member")
        # mcp_search matches both base mcp_* and override mcp_search
        assert resolved.is_allowed("mcp_search") is True
        # local_tool matches base but NOT override
        assert resolved.is_allowed("local_tool") is False
        # mcp_delete matches base but NOT override
        assert resolved.is_allowed("mcp_delete") is False

    def test_deny_plus_allow_intersection(self) -> None:
        """Restrict mode with both deny and allow overrides."""
        policy = ToolPolicy(
            allow=["*"],
            deny=["admin_*"],
            role_overrides={
                "observer": RoleOverride(allow=["read_*", "search"], deny=["read_secret"]),
            },
        )
        resolved = policy.resolve("observer")
        assert resolved.is_allowed("read_file") is True
        assert resolved.is_allowed("search") is True
        assert resolved.is_allowed("write_file") is False  # not in override allow
        assert resolved.is_allowed("admin_panel") is False  # base deny
        assert resolved.is_allowed("read_secret") is False  # override deny

    def test_restrict_empty_override_allow_no_restriction(self) -> None:
        """Restrict with empty override allow doesn't add restriction."""
        policy = ToolPolicy(
            allow=["read_*"],
            role_overrides={"member": RoleOverride(deny=["read_secret"])},
        )
        resolved = policy.resolve("member")
        assert resolved.is_allowed("read_file") is True
        assert resolved.is_allowed("read_secret") is False


class TestRoleOverrideReplace:
    def test_replace_mode(self) -> None:
        """Replace mode: override completely replaces the base policy."""
        policy = ToolPolicy(
            allow=["*"],
            deny=["admin_*"],
            role_overrides={
                "bot": RoleOverride(allow=["ping"], deny=[], mode="replace"),
            },
        )
        resolved = policy.resolve("bot")
        assert resolved.is_allowed("ping") is True
        assert resolved.is_allowed("search") is False  # not in replace allow
        assert resolved.is_allowed("admin_panel") is False  # not in replace allow

    def test_replace_mode_no_allow(self) -> None:
        """Replace mode with empty allow+deny permits everything."""
        policy = ToolPolicy(
            deny=["dangerous"],
            role_overrides={"owner": RoleOverride(mode="replace")},
        )
        resolved = policy.resolve("owner")
        assert resolved.is_allowed("dangerous") is True
        assert resolved.is_allowed("anything") is True


class TestRoleOverrideSerialization:
    def test_roundtrip(self) -> None:
        policy = ToolPolicy(
            allow=["search"],
            role_overrides={
                "observer": RoleOverride(allow=["search"], deny=["write_*"]),
                "bot": RoleOverride(allow=["ping"], mode="replace"),
            },
        )
        data = policy.model_dump()
        restored = ToolPolicy(**data)
        assert set(restored.role_overrides.keys()) == {"observer", "bot"}
        assert restored.role_overrides["bot"].mode == "replace"

        # Verify functional behavior survives roundtrip
        resolved = restored.resolve("observer")
        assert resolved.is_allowed("search") is True
        assert resolved.is_allowed("write_file") is False
