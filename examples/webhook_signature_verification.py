"""Verifying a webhook signature before trusting its payload.

A webhook URL is public: the provider has to reach it, so anyone else can too.
Once a payload is parsed, nothing downstream can tell a forged one from a real
one — it becomes a perfectly well-formed ``InboundMessage`` either way, with
whatever sender the forger chose. The signature check is the only place the
difference exists, and it has to run first.

This example needs no credentials and no network: it signs a payload the way
Twilio would, then shows the provider accepting it, rejecting a tampered body,
and rejecting a request whose URL does not match.

Run::

    uv run python examples/webhook_signature_verification.py
"""

from __future__ import annotations

import asyncio
import base64
import hashlib
import hmac
from urllib.parse import quote_plus

from roomkit.providers.twilio import TwilioConfig, TwilioSMSProvider

WEBHOOK_URL = "https://example.com/webhooks/sms"
AUTH_TOKEN = "example-auth-token-not-a-real-one"  # noqa: S105


def _form_body(params: dict[str, str]) -> bytes:
    """Encode parameters the way Twilio posts them."""
    return "&".join(f"{k}={quote_plus(v)}" for k, v in params.items()).encode()


def _twilio_signature(url: str, params: dict[str, str], token: str) -> str:
    """Compute X-Twilio-Signature — what Twilio's servers would send.

    The signed string is the full URL followed by every parameter sorted by
    name and concatenated as key + value. The URL being part of it is why
    verification needs the *public* URL, not the one seen behind a proxy.
    """
    signed = url + "".join(k + params[k] for k in sorted(params))
    digest = hmac.new(token.encode(), signed.encode(), hashlib.sha1).digest()
    return base64.b64encode(digest).decode()


async def main() -> None:
    provider = TwilioSMSProvider(
        TwilioConfig(
            account_sid="ACxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx",
            auth_token=AUTH_TOKEN,
            from_number="+15550000000",
        )
    )

    params = {
        "From": "+15551234567",
        "To": "+15550000000",
        "Body": "Transfer approved",
        "MessageSid": "SM00000000000000000000000000000000",
    }
    body = _form_body(params)
    signature = _twilio_signature(WEBHOOK_URL, params, AUTH_TOKEN)

    print("1. A genuine request")
    ok = provider.verify_signature(body, signature, url=WEBHOOK_URL)
    print(f"   verified: {ok}   -> parse it, then kit.process_webhook(...)\n")

    print("2. The same signature, with the body edited in flight")
    tampered = _form_body({**params, "Body": "Transfer approved to +15559999999"})
    ok = provider.verify_signature(tampered, signature, url=WEBHOOK_URL)
    print(f"   verified: {ok}   -> 403, and nothing reaches the room\n")

    print("3. A replay aimed at a different endpoint")
    ok = provider.verify_signature(body, signature, url="https://example.com/other")
    print(f"   verified: {ok}   -> the URL is part of what was signed\n")

    print("4. The mistake to avoid: no signature at all")
    ok = provider.verify_signature(body, "", url=WEBHOOK_URL)
    print(f"   verified: {ok}   -> a missing header is a failure, not a pass\n")

    await provider.close()

    print("In a real endpoint, read the RAW body before any framework parses")
    print("it, and pass the PUBLIC url — behind a proxy, request.url is not it:")
    print()
    print('    @app.post("/webhooks/sms")')
    print("    async def inbound(request: Request):")
    print("        raw = await request.body()")
    print("        if not provider.verify_signature(")
    print('            raw, request.headers.get("X-Twilio-Signature", ""),')
    print("            url=PUBLIC_WEBHOOK_URL,")
    print("        ):")
    print("            raise HTTPException(status_code=403)")
    print("        meta = extract_sms_meta('twilio', dict(await request.form()))")
    print('        await kit.process_webhook(meta, channel_id="sms-twilio")')


if __name__ == "__main__":
    asyncio.run(main())
