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
- IPv6 literals including loopback, link-local, unique-local, site-local, etc.
- shared address space (``100.64.0.0/10``, carrier-grade NAT and overlay
  networks), and any other address ``ipaddress`` does not call global
- hostnames whose A/AAAA records resolve to any non-public address

A note on DNS rebinding: this validator resolves at validation time and
checks every resolved address. It does NOT pin DNS at connect time, so a
TOCTOU attacker who controls both DNS and timing can still rebind between
validation and the actual HTTP request. Pin-on-connect is out of scope
for a config-time helper; a caller that needs it hands its own transport
(one that resolves, judges and dials the address it verified) to
``WebhookHTTPProvider(config, transport=...)``.
"""

from __future__ import annotations

import ipaddress
import re
import socket
from urllib.parse import urlparse

_LOCALHOST_NAMES = frozenset(
    {
        "localhost",
        "ip6-localhost",
        "ip6-loopback",
    }
)


# The special-purpose ranges that changed in supported CPython patch releases.
_V4_PROTOCOLS = ipaddress.IPv4Network("192.0.0.0/24")
_V4_PUBLIC_PROTOCOLS = {ipaddress.IPv4Address("192.0.0.9"), ipaddress.IPv4Address("192.0.0.10")}
_V6_PROTOCOLS = ipaddress.IPv6Network("2001::/23")
_V6_PUBLIC_PROTOCOLS = tuple(
    ipaddress.IPv6Network(prefix)
    for prefix in (
        "2001:1::1/128",
        "2001:1::2/128",
        "2001:3::/32",
        "2001:4:112::/48",
        "2001:20::/27",
    )
)
_V6_NON_PUBLIC = tuple(
    ipaddress.IPv6Network(prefix) for prefix in ("64:ff9b:1::/48", "2002::/16", "3fff::/20")
)
# Shared address space (RFC 6598): carrier-grade NAT and overlay networks.
# ``ipaddress`` calls it neither private nor global.
_V4_SHARED = ipaddress.IPv4Network("100.64.0.0/10")


def _is_private_destination(addr: ipaddress.IPv4Address | ipaddress.IPv6Address) -> bool:
    """Use the same special-purpose table on older Python patch releases."""
    if isinstance(addr, ipaddress.IPv4Address):
        if addr in _V4_PROTOCOLS:
            return addr not in _V4_PUBLIC_PROTOCOLS
    else:
        if addr in _V6_PROTOCOLS:
            return not any(addr in net for net in _V6_PUBLIC_PROTOCOLS)
        if any(addr in net for net in _V6_NON_PUBLIC):
            return True
    return addr.is_private


def _is_off_the_public_internet(addr: ipaddress.IPv4Address | ipaddress.IPv6Address) -> bool:
    """Whether an address the named reasons let through is still not global.

    The ranges whose answer changed between CPython patch releases are
    decided by the tables above, never by ``is_global``.
    """
    if isinstance(addr, ipaddress.IPv4Address):
        if addr in _V4_PROTOCOLS:
            return False
        return addr in _V4_SHARED or not addr.is_global
    if addr in _V6_PROTOCOLS:
        return False
    # Site-local addresses (fec0::/10, RFC 3879) are deprecated, and
    # ``ipaddress`` calls them global; sites still route them inside.
    return addr.is_site_local or not addr.is_global


def _is_blocked_address(addr: ipaddress.IPv4Address | ipaddress.IPv6Address) -> str | None:
    """Return a reason string if *addr* is non-public, else None."""
    if isinstance(addr, ipaddress.IPv6Address) and addr.ipv4_mapped is not None:
        addr = addr.ipv4_mapped
    if addr.is_loopback:
        return "loopback"
    if _is_private_destination(addr):
        return "private"
    if addr.is_link_local:
        return "link-local"
    if addr.is_reserved:
        return "reserved"
    if addr.is_multicast:
        return "multicast"
    if addr.is_unspecified:
        return "unspecified"
    if _is_off_the_public_internet(addr):
        return "non-public"
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


def _is_loopback_host(hostname: str) -> bool:
    """Whether *hostname* names this machine, written as a name or an address."""
    host = hostname.rstrip(".").lower()
    if host in _LOCALHOST_NAMES:
        return True
    ipv4 = _parse_ipv4_numeric(host)
    if ipv4 is not None:
        return ipv4.is_loopback
    try:
        return ipaddress.IPv6Address(host).is_loopback
    except (ipaddress.AddressValueError, ValueError):
        return False


def validate_api_base_url(url: str) -> str:
    """Validate the base URL of a vendor API a provider sends its credentials to.

    A sandbox, a self-hosted server or a local fake under test replaces the
    vendor's host, and the provider's credentials go with every request: the
    URL must be HTTPS, or plain HTTP to this machine only, and carry no
    credentials of its own.

    Returns:
        *url* without its trailing slash.

    Raises:
        ValueError: Naming what was refused.
    """
    parsed = urlparse(url)
    if not parsed.scheme or not parsed.hostname:
        raise ValueError("API base URL must have a scheme and host")
    if parsed.username or parsed.password:
        raise ValueError("API base URL must not carry credentials")
    if parsed.scheme == "https":
        return url.rstrip("/")
    if parsed.scheme == "http" and _is_loopback_host(parsed.hostname):
        return url.rstrip("/")
    raise ValueError(
        f"API base URL must use https (plain http only to this machine), got {parsed.scheme}://"
        f"{parsed.hostname}"
    )


_PUBLIC_AUTHORITY = re.compile(r"[A-Za-z][A-Za-z0-9.+-]*://([^/?#]*)")
_DNS_HOST = re.compile(r"[A-Za-z0-9._-]+")
_IPV6_HOST = re.compile(r"\[([A-Fa-f0-9:.]+)\](?::[0-9]*)?")


def _check_public_authority(url: str) -> None:
    """Check the raw authority before a URL parser normalizes the hostname."""
    match = _PUBLIC_AUTHORITY.match(url.lstrip(" \t\r\n"))
    if not match or not (authority := match[1]):
        raise ValueError("URL must have a scheme and host")
    if "\\" in authority or any(ord(char) <= 32 or ord(char) == 127 for char in authority):
        raise ValueError("Invalid URL authority")
    if authority.count("@") > 1:
        raise ValueError("Invalid URL authority")
    host_port = authority.rpartition("@")[2]
    if host_port.startswith("["):
        literal = _IPV6_HOST.fullmatch(host_port)
        if not literal:
            raise ValueError("Invalid IPv6 host or port in URL authority")
        ipaddress.IPv6Address(literal[1])
        return
    host = host_port.partition(":")[0]
    if not _DNS_HOST.fullmatch(host) or not host.rstrip("."):
        raise ValueError("URL host must be an ASCII DNS name or IP address")


def validate_public_url(
    url: str,
    *,
    allowed_schemes: tuple[str, ...] = ("http", "https"),
    resolve_dns: bool = True,
) -> str:
    """Validate that *url* points to a public host.

    Supply internationalized DNS names as ASCII (Punycode) and IPv6 addresses
    without a zone identifier. Paths and queries can contain Unicode.

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
    _check_public_authority(url)
    parsed = urlparse(url)
    if not parsed.scheme or not parsed.hostname:
        raise ValueError("URL must have a scheme and host")

    if parsed.scheme not in allowed_schemes:
        raise ValueError(f"URL scheme must be one of {allowed_schemes}, got {parsed.scheme!r}")

    # Read the property even for literals and offline checks: urllib validates
    # the port lazily, and callers can reuse the URL without resolving here.
    port = parsed.port or (443 if parsed.scheme == "https" else 80)
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

    addrs = _resolve_hostname(hostname, port)
    if not addrs:
        raise ValueError(f"URL host {hostname!r} did not resolve to any address")
    for addr in addrs:
        reason = _is_blocked_address(addr)
        if reason is not None:
            raise ValueError(f"URL host {hostname!r} resolves to {reason} address {addr}")
    return url
