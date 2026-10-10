"""The names a discussion reads in a message, and the silence token (RFC §19.7.5)."""

from __future__ import annotations

import time

import pytest

from roomkit.orchestration.strategies.discussion._names import (
    Names,
    SilentToken,
    name_key,
    read_names,
)

AGENTS = ("investigator", "sre", "dev", "comms")


def test_names_are_read_in_order_ignoring_case_and_the_speaker() -> None:
    text = "@SRE please check deploys, then @comms; cc @sre again"
    assert read_names(text, AGENTS, speaker="investigator") == Names(("sre", "comms"))
    assert read_names("@sre over to you", AGENTS, speaker="sre") == Names()


def test_a_name_needs_a_boundary_before_it_and_drops_a_final_dot() -> None:
    assert read_names("mail ops@example.com", AGENTS) == Names()
    assert read_names("thanks @dev.", AGENTS) == Names(("dev",))
    assert read_names("@dev.team is not dev", AGENTS) == Names()
    assert read_names("(@comms)", AGENTS) == Names(("comms",))


def test_all_names_every_agent_and_unknown_names_nobody() -> None:
    assert read_names("@all heads up", AGENTS, speaker="sre") == Names(
        ("investigator", "dev", "comms")
    )
    assert read_names("@nobody here", AGENTS) == Names()


def test_names_in_code_or_quotes_are_not_read() -> None:
    text = "see\n```\n@dev run this\n```\n> @sre said this\nok @comms"
    assert read_names(text, AGENTS) == Names(("comms",))


def test_a_hostile_text_is_read_in_linear_time() -> None:
    """Thousands of unclosed fences and blank lines must not stall the room."""
    hostile = "```\n" * 20_000 + "\n" * 20_000 + "> \n" * 20_000 + "@sre"
    started = time.monotonic()
    read_names(hostile, AGENTS)
    read_names("~~~ x\n" + "@sre " * 50_000, AGENTS)
    assert time.monotonic() - started < 1.0


def test_an_unclosed_fence_hides_what_follows() -> None:
    assert read_names("@dev before\n```\n@sre inside, never closed", AGENTS) == Names(("dev",))


def test_people_are_named_too_and_an_agent_wins_a_shared_name() -> None:
    people = ("ops", "Alice Martin", "sre")
    names = read_names("@ops and @AliceMartin, ask @sre", AGENTS, people=people)
    assert names == Names(agents=("sre",), people=("ops", "Alice Martin"))
    assert name_key("Alice Martin") == "AliceMartin"


@pytest.mark.parametrize("text", ["(silent)", " (Silent). ", "SILENT", "silent.", ""])
def test_the_silent_token_is_read_leniently(text: str) -> None:
    assert SilentToken().is_silent(text)


@pytest.mark.parametrize("text", ["I stay (silent) for now", "silently", "pass"])
def test_text_that_says_something_is_not_silent(text: str) -> None:
    assert not SilentToken().is_silent(text)


def test_the_start_of_a_streamed_answer_is_held_while_it_may_become_the_token() -> None:
    token = SilentToken()
    for start in ("", " ", "(", "(sil", "(silent", "(silent)", "(silent).", "Sil"):
        assert token.may_become(start), start
    for start in ("(silent) but", "Hello", "(s!"):
        assert not token.may_become(start), start


def test_a_custom_token_without_brackets() -> None:
    token = SilentToken("PASS")
    assert token.is_silent("pass.") and token.may_become("pa")
    assert not token.is_silent("(pass)")
