"""Identity verification's calls, made on the Python core (roomkit-specs
``conformance/README.md``, "Calls").

``request_verification`` answers the request and its secret; its ``as``
label names the request's id, and ``<label>.secret`` its secret, which a later
step completes it with. ``sender_verification``, an action of a scripted
hook, reads the verification of the sender of the event the hook was handed,
with the hook's context. The kit's verification policy comes from
``setup.kit.verification_policy``, its lifetimes in milliseconds.
"""

from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime
from typing import TYPE_CHECKING, Any

from roomkit import RoomKit, VerificationPolicy
from tests.conformance._calls import Call, UnreadableError, read
from tests.conformance._script import HOOKED

if TYPE_CHECKING:
    from tests.conformance._harness import Harness

# The framework events of identity verification (§8.2, §11.7).
VERIFICATION_EVENTS = ("verification_requested", "verification_completed", "verification_ended")

# Each call's arguments, beside `call` and `organization_id`.
VERIFICATION_ARGUMENTS: dict[str, set[str]] = {
    "request_verification": {"identity_id", "room_id", "channel_id", "method", "window_start"},
    "complete_verification": {"request_id", "secret", "method"},
    "record_failed_attempt": {"request_id"},
    "revoke_verification": {"identity_id", "reason"},
    "verification_of": {"identity_id"},
    "check_verifications": set(),
    "sender_verification": set(),
}

# The policy's members a scenario names, each in milliseconds when its
# Python field is in seconds.
_POLICY_SECONDS = ("ttl", "request_ttl", "code_request_ttl", "release_max_age")
_POLICY_AS_IS = (
    "max_attempts",
    "code_length",
    "release_on_verify",
    "release_max_messages",
    "hold_unverified",
)


def verification_options(kit: Mapping[str, Any]) -> dict[str, Any]:
    """The kit's ``verification_policy`` keyword a scenario's ``setup.kit`` names."""
    if "verification_policy" not in kit:
        return {}
    spec = kit["verification_policy"]
    fields: dict[str, Any] = {
        name: spec[f"{name}_ms"] / 1000 for name in _POLICY_SECONDS if f"{name}_ms" in spec
    }
    fields |= {name: spec[name] for name in _POLICY_AS_IS if name in spec}
    return {"verification_policy": read(lambda: VerificationPolicy(**fields))}


def opened_labels(spec: Mapping[str, Any], result: Any) -> dict[str, str]:
    """The labels a labelled ``request_verification`` names: the request's id,
    and its secret under ``<label>.secret``."""
    if spec.get("call") != "request_verification" or "as" not in spec:
        return {}
    label = spec["as"]
    return {label: result["request"].id, f"{label}.secret": result["secret"]}


def _org(spec: Mapping[str, Any]) -> dict[str, Any]:
    return {"organization_id": spec["organization_id"]} if "organization_id" in spec else {}


async def _request(kit: RoomKit, harness: Harness, spec: Mapping[str, Any]) -> Any:
    kwargs: dict[str, Any] = {"room_id": spec["room_id"], "channel_id": spec["channel_id"]}
    if "method" in spec:
        kwargs["method"] = spec["method"]
    if "window_start" in spec:
        kwargs["window_start"] = read(lambda: datetime.fromisoformat(spec["window_start"]))
    request, secret = await kit.request_verification(spec["identity_id"], **kwargs, **_org(spec))
    return {"request": request, "secret": secret}


async def _complete(kit: RoomKit, harness: Harness, spec: Mapping[str, Any]) -> Any:
    method = {"method": spec["method"]} if "method" in spec else {}
    return await kit.complete_verification(
        spec["request_id"], spec["secret"], **method, **_org(spec)
    )


async def _failed_attempt(kit: RoomKit, harness: Harness, spec: Mapping[str, Any]) -> Any:
    return await kit.record_failed_attempt(spec["request_id"], **_org(spec))


async def _revoke(kit: RoomKit, harness: Harness, spec: Mapping[str, Any]) -> Any:
    reason = {"reason": spec["reason"]} if "reason" in spec else {}
    return await kit.revoke_verification(spec["identity_id"], **reason, **_org(spec))


async def _verification_of(kit: RoomKit, harness: Harness, spec: Mapping[str, Any]) -> Any:
    return await kit.verification_of(spec["identity_id"], **_org(spec))


async def _check(kit: RoomKit, harness: Harness, spec: Mapping[str, Any]) -> Any:
    return await kit.check_verifications()


async def _sender_verification(kit: RoomKit, harness: Harness, spec: Mapping[str, Any]) -> Any:
    hooked = HOOKED.get()
    if hooked is None:
        raise UnreadableError("sender_verification outside a scripted hook's action")
    event, context = hooked
    return await kit.sender_verification(event, context)


VERIFICATION_CALLS: dict[str, Call] = {
    "request_verification": _request,
    "complete_verification": _complete,
    "record_failed_attempt": _failed_attempt,
    "revoke_verification": _revoke,
    "verification_of": _verification_of,
    "check_verifications": _check,
    "sender_verification": _sender_verification,
}
