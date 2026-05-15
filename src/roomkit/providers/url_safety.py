"""Shared URL-safety helpers for SSRF protection.

Used by any provider that takes a URL from configuration or message
content and then dereferences it server-side. Two surfaces today:

- ``HTTPProviderConfig.webhook_url`` (operator-configured outbound webhook)
- ``AudioContent.url`` consumers (e.g. STT providers that fetch the URL)

The validator catches:

- localhost variants including the ``localhost.`` trailing-dot DNS form
- IPv4 literals in every Python-accepted numeric form: dotted-quad,
  shorthand (``127.1``), octal, hex, integer (``2130706433``) — these
  bypass ``ipaddress.ip_address`` which only accepts canonical dotted-quad
- IPv6 literals including loopback, link-local, unique-local, etc.
- hostnames whose A/AAAA records resolve to any non-public address

A note on DNS rebinding: this validator resolves at validation time and
checks every resolved address. It does NOT pin DNS at connect time, so a
TOCTOU attacker who controls both DNS and timing can still rebind between
validation and the actual HTTP request. Pin-on-connect is out of scope
for a config-time helper; callers needing that should wire a custom
``httpx.AsyncHTTPTransport`` with a resolved-IP host header.
"""

from __future__ import annotations

import ipaddress
import socket
from urllib.parse import urlparse

_LOCALHOST_NAMES = frozenset(
    {
        "localhost",
        "ip6-localhost",
        "ip6-loopback",
    }
)


def _is_blocked_address(addr: ipaddress.IPv4Address | ipaddress.IPv6Address) -> str | None:
    """Return a reason string if *addr* is non-public, else None."""
    if addr.is_loopback:
        return "loopback"
    if addr.is_private:
        return "private"
    if addr.is_link_local:
        return "link-local"
    if addr.is_reserved:
        return "reserved"
    if addr.is_multicast:
        return "multicast"
    if addr.is_unspecified:
        return "unspecified"
    return None


def _parse_ipv4_numeric(hostname: str) -> ipaddress.IPv4Address | None:
    """Accept every IPv4 numeric form Python's stdlib parses.

    ``ipaddress.ip_address`` only accepts canonical dotted-quad. The
    shorthand forms (``127.1``), integer form (``2130706433``), octal
    form (``0177.0.0.1``), and hex form (``0x7f000001``) are all
    accepted by ``socket.inet_aton`` and the kernel resolver — and all
    resolve to loopback. Normalize via ``inet_aton`` so the blocklist
    sees the canonical address.
    """
    try:
        packed = socket.inet_aton(hostname)
    except OSError:
        return None
    return ipaddress.IPv4Address(packed)


def _resolve_hostname(
    hostname: str, port: int
) -> list[ipaddress.IPv4Address | ipaddress.IPv6Address]:
    """Resolve *hostname* to all A/AAAA records.

    Returns an empty list when the name doesn't resolve (the caller
    decides whether that's pass or fail — generally fail-closed).
    """
    try:
        infos = socket.getaddrinfo(hostname, port, type=socket.SOCK_STREAM)
    except socket.gaierror:
        return []
    addrs: list[ipaddress.IPv4Address | ipaddress.IPv6Address] = []
    for family, _stype, _proto, _canon, sockaddr in infos:
        if family == socket.AF_INET:
            addrs.append(ipaddress.IPv4Address(str(sockaddr[0])))
        elif family == socket.AF_INET6:
            # sockaddr[0] may include a zone (e.g. fe80::1%en0). Strip
            # the zone before constructing the IPv6Address.
            host = str(sockaddr[0]).split("%", 1)[0]
            addrs.append(ipaddress.IPv6Address(host))
    return addrs


def validate_public_url(
    url: str,
    *,
    allowed_schemes: tuple[str, ...] = ("http", "https"),
    resolve_dns: bool = True,
) -> str:
    """Validate that *url* points to a public host.

    Args:
        url: The URL to validate.
        allowed_schemes: Acceptable URL schemes. Default ``("http",
            "https")``. Pass ``("https",)`` to require TLS.
        resolve_dns: If True (default), resolve the hostname and
            validate every A/AAAA record. Set to False only in tests
            where the host isn't expected to resolve.

    Returns:
        The original *url* string, unchanged, on success.

    Raises:
        ValueError: With a human-readable reason describing what was
            rejected (scheme, IP class, DNS resolution).
    """
    parsed = urlparse(url)
    if not parsed.scheme or not parsed.hostname:
        raise ValueError("URL must have a scheme and host")

    if parsed.scheme not in allowed_schemes:
        raise ValueError(f"URL scheme must be one of {allowed_schemes}, got {parsed.scheme!r}")

    hostname = parsed.hostname
    # Strip the trailing-dot DNS form: ``localhost.`` resolves identically
    # to ``localhost`` but bypasses literal-string blocklists.
    if hostname.endswith("."):
        hostname = hostname.rstrip(".")

    if hostname.lower() in _LOCALHOST_NAMES:
        raise ValueError(f"URL must not point to localhost (host={parsed.hostname!r})")

    # Inline IPv4/IPv6 literal, including non-canonical numeric forms.
    ipv4 = _parse_ipv4_numeric(hostname)
    if ipv4 is not None:
        reason = _is_blocked_address(ipv4)
        if reason is not None:
            raise ValueError(f"URL host {hostname!r} resolves to {reason} address {ipv4}")
        return url

    # Bare IPv6 literals in URLs are enclosed in brackets, which urlparse
    # already strips. Try parsing directly.
    try:
        ipv6 = ipaddress.IPv6Address(hostname)
    except (ipaddress.AddressValueError, ValueError):
        ipv6 = None
    if ipv6 is not None:
        reason = _is_blocked_address(ipv6)
        if reason is not None:
            raise ValueError(f"URL host {hostname!r} resolves to {reason} address {ipv6}")
        return url

    # Not a literal — resolve.
    if not resolve_dns:
        return url

    port = parsed.port or (443 if parsed.scheme == "https" else 80)
    addrs = _resolve_hostname(hostname, port)
    if not addrs:
        raise ValueError(f"URL host {hostname!r} did not resolve to any address")
    for addr in addrs:
        reason = _is_blocked_address(addr)
        if reason is not None:
            raise ValueError(f"URL host {hostname!r} resolves to {reason} address {addr}")
    return url
