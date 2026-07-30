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

## Webhook endpoints

A webhook URL is public by construction: the provider has to reach it, so
anyone else can too. Nothing downstream of your handler can tell a forged
payload from a real one — the parsers produce a perfectly well-formed
`InboundMessage` either way, and `kit.process_webhook()` does not
authenticate. Verifying the signature is the endpoint's job, and it has to
happen before anything else (RFC §17.1).

RoomKit ships the check for the providers that support one. All of them
compare in constant time; Telnyx additionally rejects replays outside a
five-minute window.

| Provider | Method | Header |
|---|---|---|
| Twilio SMS / RCS | `verify_signature(payload, sig, url=...)` | `X-Twilio-Signature` |
| Telnyx SMS / RCS | `verify_signature(payload, sig, timestamp)` | `telnyx-signature-ed25519`, `telnyx-timestamp` |
| Sinch SMS | `verify_signature(payload, sig)` | `X-Sinch-Signature` |
| Messenger | `verify_signature(payload, sig)` | `X-Hub-Signature-256` |
| Telegram | `verify_signature(payload, secret_token)` | `X-Telegram-Bot-Api-Secret-Token` |
| Teams | `process_inbound(payload, auth_header, on_turn)` | `Authorization` (JWT) |

Two details that are easy to get wrong. Twilio signs the **URL** along with
the parameters, so `url` must be the public URL your provider called, not the
one your application sees behind a proxy — pass the wrong one and every
request fails to verify. And the signature is computed over the **raw body**:
read the bytes before any framework parses and re-serialises them.

Providers with no `verify_signature` — WhatsApp, Discord, ElasticEmail,
SendGrid, VoiceMeUp, and the generic HTTP provider, which signs its *outbound*
requests but verifies nothing inbound — leave authentication entirely to your
endpoint. Calling the inherited method on one raises `NotImplementedError`
rather than returning `True`, so it fails closed if you try.

## What `participant_id` is, and is not

Without an `identity_resolver`, a channel stamps the sender id it was given
straight onto the event as `participant_id`, and RoomKit leaves it there. This
is required behaviour, not an oversight — the specification says so explicitly
(RFC §11.6: when resolution is skipped the id "MUST be left as the channel set
it") — because only the channel knows whether its sender ids mean anything.

What follows from it is worth stating plainly, because it is not obvious from
the outside: **`participant_id` is whatever your transport put there.** If your
WebSocket handler takes it from a client-supplied field, then it is
client-controlled, and so is everything the framework decides from it. That
includes authorship: the edit/delete rules (RFC §10.3) establish who may
rewrite or remove a message by comparing `participant_id` to the target
event's. An unauthenticated id there means an unauthenticated author check.

So: derive `sender_id` from your own authenticated session — the token you
validated when the socket opened, not a field in the message — or install an
`identity_resolver` and let the framework resolve it. RoomKit cannot tell the
difference between the two, and does not try to.

Conference participants are the exception, and deliberately so: identity there
is resolved only from `asserted_metadata`, which a backend may populate solely
with values the SFU established (RFC §12.10). Widening that to what a client
supplied at join requires setting `identity_trusts_unasserted_metadata=True`,
which exists to be a visible decision.

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

Three settings bound that last one, and it is worth being precise about which
covers what, because none of them covers all of it:

- `symmetric_rtp=True` follows the address the caller's packets actually come
  from (RFC 4961), so an offer pointing elsewhere stops being followed as soon
  as the caller sends anything of its own. It is also the ordinary fix for
  callers behind NAT. **It does not stop a caller that stays silent**:
  latching only fires on an inbound packet, so an INVITE that advertises a
  third party and then sends nothing keeps the stream aimed there. Requires
  `aiosipua>=0.7.1`; off by default, since it changes how media is addressed
  mid-call.
- `rtp_establishment_timeout` is what bounds the silent case — a session that
  never receives a packet releases its port, so a reflector lasts that long
  rather than forever. On by default (60 s).
- `max_sessions` bounds how many can run at once, answering `503` past the cap.

Authentication is what actually prevents it. The three above limit what an
unauthenticated caller can do; they do not make the port safe to expose.

Protocol traces (`on_trace`, `ON_PROTOCOL_TRACE`) carry SIP messages close to
verbatim. The digest `response` is masked, but everything else — caller,
callee, X-headers, SDP — reaches whatever consumes traces. Treat that stream
as you would the call metadata itself.
