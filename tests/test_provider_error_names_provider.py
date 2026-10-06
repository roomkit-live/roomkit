"""A provider error says which provider answered, and with what status."""

from __future__ import annotations

from roomkit.providers.ai.base import ProviderError


def test_the_message_leads_with_the_provider_and_its_status() -> None:
    error = ProviderError(
        "Error code: 402 - {'message': 'Payment required to access this resource.'}",
        provider="cerebras",
        status_code=402,
    )
    assert str(error) == (
        "cerebras (402): Error code: 402 - {'message': 'Payment required to access "
        "this resource.'}"
    )
    # The provider's own text is untouched for whoever reads it.
    assert error.args[0].startswith("Error code: 402")


def test_without_a_status_the_provider_still_leads() -> None:
    assert str(ProviderError("connection reset", provider="anthropic")) == (
        "anthropic: connection reset"
    )


def test_an_error_without_a_provider_reads_as_given() -> None:
    assert str(ProviderError("boom")) == "boom"


def test_a_rewrapped_error_is_not_prefixed_twice() -> None:
    inner = ProviderError("Payment required", provider="xai", status_code=402)
    outer = ProviderError(str(inner), provider="xai")
    assert str(outer) == "xai (402): Payment required"


def test_a_message_that_names_its_provider_is_left_as_it_is() -> None:
    error = ProviderError("tool 'X': acme accepts tool names matching [a-z]+", provider="acme")
    assert str(error) == "tool 'X': acme accepts tool names matching [a-z]+"
