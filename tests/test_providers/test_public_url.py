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


@pytest.mark.parametrize(
    "authority",
    [
        r"name\part.example",
        "name\tpart.example",
        "name%2Epart.example",
        "one@two@service.example",
        "[2001:db8::1%25example]",
        "café.example",
        "K.example",
    ],
)
def test_ambiguous_or_non_ascii_authorities_are_rejected_before_dns(
    authority: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    def no_lookup(*args: object, **kwargs: object) -> None:
        pytest.fail("the resolver must not receive an invalid authority")

    monkeypatch.setattr(socket, "getaddrinfo", no_lookup)
    for scheme in ("https", "service"):
        for resolve in (False, True):
            with pytest.raises(ValueError):
                validate_public_url(
                    f"{scheme}://{authority}/", allowed_schemes=(scheme,), resolve_dns=resolve
                )


def test_ascii_dns_spelling_preserves_userinfo_path_and_query(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def resolve(host: str, port: int, *, type: int) -> list[tuple[object, ...]]:
        assert (host, port, type) == ("xn--caf-dma.example", 80, socket.SOCK_STREAM)
        return [(socket.AF_INET, type, 6, "", ("93.184.216.34", port))]

    monkeypatch.setattr(socket, "getaddrinfo", resolve)
    url = "service://user%40mail@XN--CAF-DMA.example/café?q=été"
    assert validate_public_url(url, allowed_schemes=("service",)) == url


def test_mixed_families_check_every_address(monkeypatch: pytest.MonkeyPatch) -> None:
    def resolve(host: str, port: int, *, type: int) -> list[tuple[object, ...]]:
        return [
            (socket.AF_INET, type, 6, "", ("93.184.216.34", port)),
            (socket.AF_INET6, type, 6, "", ("::1", port, 0, 0)),
        ]

    monkeypatch.setattr(socket, "getaddrinfo", resolve)
    with pytest.raises(ValueError, match="loopback"):
        validate_public_url("https://service.example/")


@pytest.mark.parametrize(
    ("literal", "public"),
    [
        ("[2001:db8::1]", False),
        ("[3fff::1]", False),
        ("[ff02::1]", False),
        ("[fe80::1]", False),
        ("[2001:1::1]", True),
        ("[2001:3::1]", True),
        ("[2002::1]", False),
        ("[64:ff9b:1::1]", False),
        ("192.0.0.8", False),
        ("192.0.0.9", True),
        ("[::ffff:192.0.0.9]", True),
        ("[::ffff:127.0.0.1]", False),
    ],
)
def test_literal_address_policy_is_independent_of_the_python_patch(
    literal: str, public: bool
) -> None:
    url = f"https://{literal}/"
    if public:
        assert validate_public_url(url, resolve_dns=False) == url
    else:
        with pytest.raises(ValueError):
            validate_public_url(url, resolve_dns=False)


@pytest.mark.parametrize("prefix", [" ", "\t", "\r", "\n", "\u00a0"])
def test_schemes_share_the_same_leading_whitespace_rule(prefix: str) -> None:
    url = prefix + "service://service.example/"
    if prefix == "\u00a0":
        with pytest.raises(ValueError):
            validate_public_url(url, allowed_schemes=("service",), resolve_dns=False)
    else:
        assert validate_public_url(url, allowed_schemes=("service",), resolve_dns=False) == url
