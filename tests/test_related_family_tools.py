"""find_tools peripheral vision: same-family sibling names in the payload.

A model that found ``square_get-menu`` has no idea the same source also does
carts, and refuses the next in-domain ask instead of searching (observed with
small models). ``related_family_tools`` lists the matched sources' unmatched
tool names; ``render_find_payload`` surfaces them with an explicit
"not yet invocable — search for it" note.
"""

from __future__ import annotations

import json

from roomkit.channels._tool_search import related_family_tools, render_find_payload

_CATALOGUE = [
    {"name": "square_get-menu", "description": "Menu"},
    {"name": "square_create-cart", "description": "Cart"},
    {"name": "square_update-cart", "description": "Update cart"},
    {"name": "jira_create-issue", "description": "Jira"},
    {"name": "standalone", "description": "No family"},
]


class TestRelatedFamilyTools:
    def test_siblings_of_the_matched_family_only(self) -> None:
        matches = [{"name": "square_get-menu"}]
        assert related_family_tools(_CATALOGUE, matches) == [
            "square_create-cart",
            "square_update-cart",
        ]

    def test_no_family_no_related(self) -> None:
        assert related_family_tools(_CATALOGUE, [{"name": "standalone"}]) == []
        assert related_family_tools(_CATALOGUE, []) == []

    def test_bounded(self) -> None:
        catalogue = [{"name": f"fam_tool{i}"} for i in range(30)]
        related = related_family_tools(catalogue, [{"name": "fam_tool0"}], limit=10)
        assert len(related) == 10
        assert "fam_tool0" not in related


class TestRenderFindPayloadRelated:
    def test_related_rendered_with_steering_note(self) -> None:
        payload = json.loads(
            render_find_payload(
                [{"name": "square_get-menu", "description": "Menu"}],
                related=["square_create-cart"],
            )
        )
        assert payload["related_tools_same_source"] == ["square_create-cart"]
        assert "NOT yet invocable" in payload["_note"]

    def test_no_related_keeps_payload_unchanged(self) -> None:
        payload = json.loads(
            render_find_payload([{"name": "square_get-menu", "description": "Menu"}], related=[])
        )
        assert "related_tools_same_source" not in payload

    def test_miss_keeps_the_miss_note_without_related(self) -> None:
        payload = json.loads(render_find_payload([], related=["square_create-cart"]))
        assert "related_tools_same_source" not in payload
        assert "No tools matched" in payload["_note"]
