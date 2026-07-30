# Security Policy

## Reporting a Vulnerability

If you discover a security vulnerability in RoomKit, please report it responsibly by emailing:

**sylvainboilydroid@gmail.com**

Please include:

- A description of the vulnerability
- Steps to reproduce the issue
- Any potential impact

We will acknowledge your report within 48 hours and work to address the issue promptly.

**Please do not open public GitHub issues for security vulnerabilities.**

## Deploying the SIP backend

`SIPVoiceBackend` is written for a **trusted PBX or SBC in front of it**. Its
port must not be reachable from the open internet. Three properties of the
design follow from that assumption, and each one is a hole if the assumption
does not hold:

- **Authentication is off by default.** `auth_users` and `set_auth_resolver()`
  are both optional, and with neither set every INVITE is accepted. Configure
  one whenever anything other than your own PBX can reach the port.
- **The caller chooses its room.** `X-Room-ID` on the INVITE becomes the room
  the call is routed to, and `X-Session-ID` becomes the session id. Both are
  ordinary SIP headers: whoever sends the INVITE writes them. A PBX that sets
  them itself, and strips whatever the far end sent, is what makes them
  trustworthy — RoomKit cannot tell the difference.
- **The offer chooses where media goes.** The RTP destination comes from the
  SDP. RoomKit rejects addresses that cannot be a destination — `0.0.0.0`,
  port 0, loopback, multicast, link-local — but it cannot reject an address
  that is merely someone else's, because a caller behind NAT legitimately
  advertises an address its packets do not come from. On a reachable port that
  is an amplification primitive: the call becomes an RTP stream aimed at a
  third party.

Bound the blast radius with `max_sessions` (concurrent calls, answered `503`
past the cap) and leave `rtp_establishment_timeout` on, so a call that is
answered and then sends nothing releases its RTP port instead of holding it.

Protocol traces (`on_trace`, `ON_PROTOCOL_TRACE`) carry SIP messages close to
verbatim. The digest `response` is masked, but everything else — caller,
callee, X-headers, SDP — reaches whatever consumes traces. Treat that stream
as you would the call metadata itself.
