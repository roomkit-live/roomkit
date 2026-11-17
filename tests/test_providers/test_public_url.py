"""The URL validator used by applications before a transport is chosen."""

from __future__ import annotations

import socket

import pytest

from roomkit.providers import validate_public_url


@pytest.mark.parametrize("port", ["invalid", "65536"])
@pytest.mark.parametrize("resolve", [False, True])
@pytest.mark.parametrize("host", ["service.example", "93.184.216.34"])
def test_an_invalid_port_is_rejected_before_dns(host: str, resolve: bool, port: str) -> None:
    with pytest.raises(ValueError, match="(?i)port"):
        validate_public_url(f"https://{host}:{port}/", resolve_dns=resolve)


def test_public_url_retains_its_spelling_and_checks_all_dns_answers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    addresses = ["93.184.216.34"]
    lookups: list[tuple[str, int]] = []

    def resolve(host: str, port: int, *, type: int) -> list[tuple[object, ...]]:
        assert type == socket.SOCK_STREAM
        lookups.append((host, port))
        return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", (ip, port)) for ip in addresses]

    monkeypatch.setattr(socket, "getaddrinfo", resolve)
    url = "https://service.example:8443/A?q=1#fragment"
    assert validate_public_url(url, resolve_dns=False) == url
    assert lookups == []
    assert validate_public_url(url) == url
    assert lookups == [("service.example", 8443)]
    addresses.append("127.0.0.1")
    with pytest.raises(ValueError, match="loopback"):
        validate_public_url(url)
