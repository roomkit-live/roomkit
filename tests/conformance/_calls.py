"""The kit's calls a scenario names, made on the Python core.

Each call takes the RFC's name and arguments (roomkit-specs
``conformance/README.md``, "Calls"); its result is written in the RFC's
serialized form, and an error it raises is named by its RFC meaning
("Errors"), with the table of roomkit-rs ``docs/rfc-gaps.md`` G-085.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Mapping
from typing import TYPE_CHECKING, Any

from pydantic import TypeAdapter, ValidationError

from roomkit import ACPChannel, HostChannel, RoomKit
from roomkit.channels.base import Channel
from roomkit.core.exceptions import (
    ChangeRefusedError,
    ChannelNotFoundError,
    ChannelNotRegisteredError,
    EventNotFoundError,
    IdempotencyKeyTakenError,
    IdentityNotFoundError,
    KitClosedError,
    ParticipantNotFoundError,
    ProcessTimeoutError,
    ReentrantCallError,
    ReentryLoopCapError,
    RoomClosedError,
    RoomNotFoundError,
    TaskNotFoundError,
    TransitionRefusedError,
    TurnLinkError,
    VerificationRefusedError,
)
from roomkit.models.delivery import DeliveryOutcome, DeliveryStatus, InboundMessage, InboundResult
from roomkit.models.enums import (
    Access,
    AgentResponsePolicy,
    ChannelCategory,
    ChannelType,
    EventDirection,
    EventType,
    HookExecution,
    HookTrigger,
)
from roomkit.models.event import EventContent, EventSource, RoomEvent
from roomkit.models.room import Room, RoomTimers
from roomkit.models.steering import SteeringDirective
from roomkit.providers.sms import WebhookMeta
from roomkit.tasks import DelegatedTask
from tests.conformance._acp import scripted_acp_channel
from tests.conformance._agent import ScriptedAIRouter, scripted_agent
from tests.conformance._room_queries import call as room_query
from tests.conformance._script import (
    ScriptedChannel,
    ScriptedFailureError,
    complete_event,
    complete_participant,
    hook_handler,
)
from tests.conformance._strategies import room_orchestration

if TYPE_CHECKING:
    from tests.conformance._harness import Harness

_CONTENT = TypeAdapter(EventContent)
_DIRECTIVE: TypeAdapter[SteeringDirective] = TypeAdapter(SteeringDirective)


class HandleWaitsForItselfError(Exception):
    """A handle's cancel from where its cleanup would wait for its caller."""


# The Python class of each error the suite names (README, "Errors").
_ERRORS: tuple[tuple[type[Exception], str], ...] = (
    (KitClosedError, "kit_closed"),
    (RoomNotFoundError, "room_not_found"),
    (RoomClosedError, "room_closed"),
    (ChannelNotRegisteredError, "channel_not_registered"),
    (ChannelNotFoundError, "channel_not_bound"),
    (ChangeRefusedError, "change_refused"),
    (TransitionRefusedError, "transition_refused"),
    (ProcessTimeoutError, "process_timeout"),
    (ParticipantNotFoundError, "participant_not_found"),
    (IdentityNotFoundError, "identity_not_found"),
    (ReentrantCallError, "reentrant"),
    (ReentryLoopCapError, "reentry_loop_cap"),
    (EventNotFoundError, "event_not_found"),
    (IdempotencyKeyTakenError, "idempotency_key_taken"),
    (TaskNotFoundError, "task_not_found"),
    (VerificationRefusedError, "verification_refused"),
    (TurnLinkError, "turn_link"),
    (ValueError, "invalid_argument"),
    (HandleWaitsForItselfError, "waits_for_itself"),
)


class UnmappedError(Exception):
    """A call raised something the suite has no name for: a defect to look at."""


class UnreadableError(Exception):
    """The runner could not build a call's arguments from the scenario."""


class RefusedArgumentError(Exception):
    """A roomkit model's own validator refused a call's argument."""


def read[T](build: Callable[[], T]) -> T:
    """Builds a call's argument from the scenario. A roomkit validator's
    refusal (a pydantic ``value_error``) is the call's ``invalid_argument``;
    anything else that fails is the runner's, never mapped to an RFC name."""
    try:
        return build()
    except ValidationError as error:
        if all(item["type"] == "value_error" for item in error.errors()):
            raise RefusedArgumentError(str(error)) from error
        raise UnreadableError(str(error)) from error
    except (ValueError, KeyError, TypeError) as error:
        raise UnreadableError(f"{type(error).__name__}: {error}") from error


class ChannelNotBoundError(Exception):
    """A detach the room does not bind, which the Python API answers ``False``."""


def error_of(error: Exception) -> dict[str, Any]:
    """The suite's name of *error*, with its reason when it has one."""
    if isinstance(error, ConnectionError) or getattr(error, "kind", "").endswith(
        ":ConnectionError"
    ):
        return {"name": "acp_connection_closed"}
    if getattr(error, "kind", "").endswith(":RequestError"):
        return {"name": "acp_request_refused"}
    if (
        isinstance(error, RuntimeError)
        and str(error).startswith("ACP session for room ")
        and "it is not prompted. Set another value with set_config_option" in str(error)
    ):
        return {"name": "acp_setup_refused"}
    if isinstance(error, RuntimeError) and (
        "setup_handoff() already called" in str(error)
        or "setup_delegation() already called" in str(error)
    ):
        return {"name": "already_set_up"}
    if isinstance(error, TypeError) and "not a strategy stored with the room" in str(error):
        return {"name": "kind_not_stored"}
    if isinstance(error, ChannelNotBoundError):
        return {"name": "channel_not_bound"}
    if isinstance(error, RefusedArgumentError):
        return {"name": "invalid_argument"}
    if isinstance(error, ScriptedFailureError):
        return {"name": "channel_failed"}
    for cls, name in _ERRORS:
        if isinstance(error, cls):
            reason = getattr(error, "reason", None)
            return {"name": name, "reason": reason} if reason else {"name": name}
    raise UnmappedError(f"{type(error).__name__}: {error}") from error


def dump(value: Any) -> Any:
    """A result in the RFC's serialized form."""
    if isinstance(value, InboundResult):
        return {
            "event": dump(value.event),
            "blocked": value.blocked,
            "duplicate": value.duplicate,
            "reason": value.reason,
            "error": None if value.error is None else str(value.error),
            "response_metadata": dump(dict(value.response_metadata)),
            "response_events": [dump(event) for event in value.response_events],
            "unavailable_targets": list(value.unavailable_targets or ()),
            "delivery_results": {k: dump(v) for k, v in (value.delivery_results or {}).items()},
            "cancellation_reason": value.cancellation_reason,
        }
    if isinstance(value, DelegatedTask):
        return {
            "id": value.id,
            "child_room_id": value.child_room_id,
            "parent_room_id": value.parent_room_id,
            "agent_id": value.agent_id,
            "task": value.task,
            "brief": value.brief,
            "description": value.description,
            "hand_back": value.hand_back,
            "status": str(value.status),
            "result": dump(value.result),
        }
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    if isinstance(value, list):
        return [dump(item) for item in value]
    if isinstance(value, dict):
        return {key: dump(item) for key, item in value.items()}
    return value


Call = Callable[[RoomKit, "Harness", Mapping[str, Any]], Awaitable[Any]]


def _org(spec: Mapping[str, Any]) -> dict[str, Any]:
    return {"organization_id": spec["organization_id"]} if "organization_id" in spec else {}


async def _process_inbound(kit: RoomKit, harness: Harness, spec: Mapping[str, Any]) -> Any:
    message = read(lambda: InboundMessage.model_validate(spec["message"]))
    if not spec.get("defer_delivery"):
        return await kit.process_inbound(message, room_id=spec.get("room_id"), **_org(spec))
    result = await kit.process_inbound(
        message, room_id=spec.get("room_id"), defer_delivery=True, **_org(spec)
    )
    # Its completion handle (RFC §10.1 step 18), which `delivery.*` calls
    # name by the call's label.
    if "as" in spec and result.delivery is not None:
        harness.handles[spec["as"]] = result.delivery
    return result


async def _delivery_cancel(kit: RoomKit, harness: Harness, spec: Mapping[str, Any]) -> Any:
    handle = harness.handles[spec["of"]]
    options: dict[str, Any] = {"reason": spec["reason"]} if "reason" in spec else {}
    if "timeout_ms" in spec:
        options["timeout"] = spec["timeout_ms"] / 1000
    try:
        return await handle.cancel(**options)
    except RuntimeError as error:
        raise HandleWaitsForItselfError(str(error)) from error


async def _delivery_wait(kit: RoomKit, harness: Harness, spec: Mapping[str, Any]) -> Any:
    return await harness.handles[spec["of"]].wait()


_SEND_ARGS = (
    "participant_id",
    "visibility",
    "response_visibility",
    "addressed_to",
    "parent_event_id",
    "idempotency_key",
    "metadata",
    "chain_depth",
    "standalone",
)


async def _send_event(kit: RoomKit, harness: Harness, spec: Mapping[str, Any]) -> Any:
    kwargs = {key: spec[key] for key in _SEND_ARGS if key in spec}
    if "event_type" in spec:
        kwargs["event_type"] = read(lambda: EventType(spec["event_type"]))
    content = read(lambda: _CONTENT.validate_python(spec["content"]))
    return await kit.send_event(
        spec["room_id"],
        spec["channel_id"],
        content,
        **kwargs,
        **_org(spec),
    )


_POST_ARGS = (
    "participant_id",
    "addressed_to",
    "metadata",
    "channel_id",
    "idempotency_key",
    "chain_depth",
)


async def _post(kit: RoomKit, harness: Harness, spec: Mapping[str, Any]) -> Any:
    kwargs = {key: spec[key] for key in _POST_ARGS if key in spec}
    content = read(lambda: _CONTENT.validate_python(spec["content"]))
    return await kit.post(spec["room_id"], content, **kwargs, **_org(spec))


async def _commit_event(kit: RoomKit, harness: Harness, spec: Mapping[str, Any]) -> Any:
    source = {"channel_id": "system", "channel_type": "system", "direction": "inbound"}
    record = read(lambda: complete_event(spec["event"], spec["room_id"], source))
    return await kit.commit_event(spec["room_id"], record, **_org(spec))


async def _update_event(kit: RoomKit, harness: Harness, spec: Mapping[str, Any]) -> Any:
    kwargs: dict[str, Any] = {}
    if "content" in spec:
        kwargs["content"] = read(lambda: _CONTENT.validate_python(spec["content"]))
    if "source" in spec:
        kwargs["source"] = read(lambda: EventSource.model_validate(spec["source"]))
    if "metadata" in spec:
        kwargs["metadata"] = spec["metadata"]
    return await kit.update_event(spec["room_id"], spec["event_id"], **kwargs, **_org(spec))


async def _delete_event(kit: RoomKit, harness: Harness, spec: Mapping[str, Any]) -> Any:
    kwargs = {"cascade_replies": spec["cascade_replies"]} if "cascade_replies" in spec else {}
    return await kit.delete_event(spec["room_id"], spec["event_id"], **kwargs, **_org(spec))


async def _create_room(kit: RoomKit, harness: Harness, spec: Mapping[str, Any]) -> Any:
    return await create_room(kit, spec["room"], room_orchestration(spec, harness))


async def create_room(
    kit: RoomKit, room: Mapping[str, Any], extra: Mapping[str, Any] | None = None
) -> Room:
    kwargs: dict[str, Any] = {"room_id": room["id"], **(extra or {})}
    if "metadata" in room:
        kwargs["metadata"] = room["metadata"]
    if room.get("organization_id") is not None:
        kwargs["organization_id"] = room["organization_id"]
    if "timers" in room:
        kwargs["timers"] = read(lambda: RoomTimers.model_validate(room["timers"]))
    return await kit.create_room(**kwargs)


async def add_participant(kit: RoomKit, room_id: str, participant: Mapping[str, Any]) -> None:
    """Seeds a participant record (§5.5) of the setup, as the store keeps it."""
    record = read(lambda: complete_participant(participant, room_id))
    await kit.store.add_participant(record)


async def _attach_channel(kit: RoomKit, harness: Harness, spec: Mapping[str, Any]) -> Any:
    return await attach_channel(kit, spec["room_id"], spec, **_org(spec))


async def attach_channel(
    kit: RoomKit, room_id: str, binding: Mapping[str, Any], **org: Any
) -> Any:
    kwargs: dict[str, Any] = dict(org)
    if "category" in binding:
        kwargs["category"] = read(lambda: ChannelCategory(binding["category"]))
    if "access" in binding:
        kwargs["access"] = read(lambda: Access(binding["access"]))
    for key in ("visibility", "muted", "metadata", "participant_id"):
        if key in binding:
            kwargs[key] = binding[key]
    return await kit.attach_channel(room_id, binding["channel_id"], **kwargs)


async def _detach_channel(kit: RoomKit, harness: Harness, spec: Mapping[str, Any]) -> Any:
    if not await kit.detach_channel(spec["room_id"], spec["channel_id"], **_org(spec)):
        raise ChannelNotBoundError()
    return None


async def _mute(kit: RoomKit, harness: Harness, spec: Mapping[str, Any]) -> Any:
    return await kit.mute(spec["room_id"], spec["channel_id"], **_org(spec))


async def _unmute(kit: RoomKit, harness: Harness, spec: Mapping[str, Any]) -> Any:
    return await kit.unmute(spec["room_id"], spec["channel_id"], **_org(spec))


async def _set_access(kit: RoomKit, harness: Harness, spec: Mapping[str, Any]) -> Any:
    access = read(lambda: Access(spec["access"]))
    return await kit.set_access(spec["room_id"], spec["channel_id"], access, **_org(spec))


async def _set_visibility(kit: RoomKit, harness: Harness, spec: Mapping[str, Any]) -> Any:
    return await kit.set_visibility(
        spec["room_id"], spec["channel_id"], spec["visibility"], **_org(spec)
    )


async def _update_binding_metadata(kit: RoomKit, harness: Harness, spec: Mapping[str, Any]) -> Any:
    return await kit.update_binding_metadata(
        spec["room_id"], spec["channel_id"], spec["metadata"], **_org(spec)
    )


async def _close_room(kit: RoomKit, harness: Harness, spec: Mapping[str, Any]) -> Any:
    return await kit.close_room(spec["room_id"], **_org(spec))


async def _pause_room(kit: RoomKit, harness: Harness, spec: Mapping[str, Any]) -> Any:
    return await kit.pause_room(spec["room_id"], **_org(spec))


async def _resume_room(kit: RoomKit, harness: Harness, spec: Mapping[str, Any]) -> Any:
    return await kit.resume_room(spec["room_id"], **_org(spec))


async def _reopen_room(kit: RoomKit, harness: Harness, spec: Mapping[str, Any]) -> Any:
    return await kit.reopen_room(spec["room_id"], **_org(spec))


async def _regenerate_response(kit: RoomKit, harness: Harness, spec: Mapping[str, Any]) -> Any:
    trigger = {"trigger_id": spec["trigger_id"]} if "trigger_id" in spec else {}
    return await kit.regenerate_response(spec["room_id"], **trigger, **_org(spec))


async def _regenerate_target(kit: RoomKit, harness: Harness, spec: Mapping[str, Any]) -> Any:
    return await kit.regenerate_target(spec["room_id"], **_org(spec))


async def _send_greeting(kit: RoomKit, harness: Harness, spec: Mapping[str, Any]) -> Any:
    names = {key: spec[key] for key in ("channel_id", "agent_id", "greeting") if key in spec}
    return await kit.send_greeting(spec["room_id"], **names, **_org(spec))


async def _archive_room(kit: RoomKit, harness: Harness, spec: Mapping[str, Any]) -> Any:
    return await kit.archive_room(spec["room_id"], **_org(spec))


async def _check_timers(kit: RoomKit, harness: Harness, spec: Mapping[str, Any]) -> Any:
    return await kit.check_room_timers(spec["room_id"], **_org(spec))


async def _set_room_timers(kit: RoomKit, harness: Harness, spec: Mapping[str, Any]) -> Any:
    timers = read(lambda: RoomTimers.model_validate(spec["timers"]))
    return await kit.set_room_timers(spec["room_id"], timers, **_org(spec))


async def _set_agent_response_policy(
    kit: RoomKit, harness: Harness, spec: Mapping[str, Any]
) -> Any:
    policy = read(lambda: AgentResponsePolicy(spec["policy"]))
    return await kit.set_agent_response_policy(spec["room_id"], policy, **_org(spec))


async def _register_channel(kit: RoomKit, harness: Harness, spec: Mapping[str, Any]) -> Any:
    register_channel(kit, harness, spec["channel"])


def register_channel(
    kit: RoomKit, harness: Harness, channel: Mapping[str, Any], *, process: bool = False
) -> None:
    """Registers *channel* on *kit*, by its declaration alone when the setup
    runs it on executors of their own (README, ``executors``)."""
    built = scripted_channel(kit, harness, channel, process=process)
    kit.register_channel(built.declaration() if channel.get("declared") else built)


def scripted_channel(
    kit: RoomKit, harness: Harness, channel: Mapping[str, Any], *, process: bool = False
) -> Channel:
    """The channel the scenario describes; in a *process* of the setup or on
    an executor, a scripted agent answers from the model the run's kit's agent
    of its id has."""
    if channel["type"] == ChannelType.HOST:
        return HostChannel(channel["id"])
    if "acp" in channel:
        acp_channel, acp_agent, transport, mcp = read(
            lambda: scripted_acp_channel(channel, harness.gates)
        )
        harness.acp_agents[channel["id"]] = acp_agent
        harness.acp_transports[channel["id"]] = transport
        harness.acp_mcp[channel["id"]] = mcp
        return acp_channel
    if "model" in channel:
        agent, models = read(
            lambda: scripted_agent(
                channel,
                harness.gates,
                kit,
                harness.labelled,
                harness.auditors,
                lambda room_id, channel_id, event: harness.record(
                    "ingested", room_id, channel_id, event
                ),
                harness.models if process else None,
                harness.record_compacted,
            )
        )
        harness.models.update(models)
        if isinstance(agent.router, ScriptedAIRouter):
            harness.ai_routers[channel["id"]] = agent.router
        return agent
    return read(lambda: ScriptedChannel(channel, harness))


async def _unregister_channel(kit: RoomKit, harness: Harness, spec: Mapping[str, Any]) -> Any:
    kit.unregister_channel(spec["channel_id"])


async def _register_hook(kit: RoomKit, harness: Harness, spec: Mapping[str, Any]) -> Any:
    register_hook(kit, harness, spec["hook"])


def register_hook(kit: RoomKit, harness: Harness, hook: Mapping[str, Any]) -> None:
    kwargs: dict[str, Any] = {
        "execution": read(lambda: HookExecution(hook.get("execution", "sync"))),
        "priority": hook.get("priority", 0),
        "name": hook["name"],
        "fail_closed": hook.get("fail_closed", False),
        "needs_lock": hook.get("needs_lock", True),
    }
    if "timeout_ms" in hook:
        kwargs["timeout"] = hook["timeout_ms"] / 1000
    if "channel_types" in hook:
        kwargs["channel_types"] = read(lambda: {ChannelType(t) for t in hook["channel_types"]})
    if "channel_ids" in hook:
        kwargs["channel_ids"] = set(hook["channel_ids"])
    if "directions" in hook:
        kwargs["directions"] = read(lambda: {EventDirection(d) for d in hook["directions"]})
    trigger = read(lambda: HookTrigger(hook["trigger"]))
    kit.hook(trigger, **kwargs)(hook_handler(hook, harness))


async def _add_room_hook(kit: RoomKit, harness: Harness, spec: Mapping[str, Any]) -> Any:
    hook = spec["hook"]
    execution = read(lambda: HookExecution(hook.get("execution", "sync")))
    trigger = read(lambda: HookTrigger(hook["trigger"]))
    await kit.add_room_hook(
        spec["room_id"],
        trigger,
        execution,
        hook_handler(hook, harness),
        priority=hook.get("priority", 0),
        name=hook["name"],
        fail_closed=hook.get("fail_closed", False),
        needs_lock=hook.get("needs_lock", True),
        **_org(spec),
    )


async def _remove_room_hook(kit: RoomKit, harness: Harness, spec: Mapping[str, Any]) -> Any:
    return await kit.remove_room_hook(spec["room_id"], spec["name"], **_org(spec))


async def _mute_output(kit: RoomKit, harness: Harness, spec: Mapping[str, Any]) -> Any:
    return await kit.mute_output(spec["room_id"], spec["channel_id"], **_org(spec))


async def _unmute_output(kit: RoomKit, harness: Harness, spec: Mapping[str, Any]) -> Any:
    return await kit.unmute_output(spec["room_id"], spec["channel_id"], **_org(spec))


async def _process_delivery_status(kit: RoomKit, harness: Harness, spec: Mapping[str, Any]) -> Any:
    status = read(lambda: DeliveryStatus.model_validate(spec["status"]))
    return await kit.process_delivery_status(status, **_org(spec))


async def _process_webhook(kit: RoomKit, harness: Harness, spec: Mapping[str, Any]) -> Any:
    meta = read(lambda: WebhookMeta(**spec["meta"]))
    return await kit.process_webhook(meta, spec["channel_id"], **_org(spec))


_DELIVER_ARGS = (
    "channel_id",
    "strategy",
    "metadata",
    "addressed_to",
    "idempotency_key",
    "session_id",
    "chain_depth",
    "instruction",
)


async def _deliver(kit: RoomKit, harness: Harness, spec: Mapping[str, Any]) -> Any:
    kwargs = {key: spec[key] for key in _DELIVER_ARGS if key in spec}
    return await kit.deliver(spec["room_id"], spec["content"], **kwargs, **_org(spec))


async def _cancel_delivery(kit: RoomKit, harness: Harness, spec: Mapping[str, Any]) -> Any:
    reason = {"reason": spec["reason"]} if "reason" in spec else {}
    return await kit.cancel_delivery(spec["key"], **reason, **_org(spec))


async def _acp_cancel(kit: RoomKit, harness: Harness, spec: Mapping[str, Any]) -> bool:
    channel = kit.get_channel(spec["channel_id"])
    if not isinstance(channel, ACPChannel):
        raise UnreadableError("acp.cancel requires an ACP channel")
    return await channel.cancel(spec["room_id"])


async def _steer(kit: RoomKit, harness: Harness, spec: Mapping[str, Any]) -> Any:
    directive = _DIRECTIVE.validate_python(spec["directive"])
    return await kit.steer(spec["room_id"], spec["channel_id"], directive, **_org(spec))


async def _close(kit: RoomKit, harness: Harness, spec: Mapping[str, Any]) -> Any:
    """Closes the kit the call names: a process that stops (README, ``close``)."""
    harness.closed(kit)
    await kit.close()


CALLS: dict[str, Call] = {
    **dict.fromkeys(
        (
            "get_room",
            "get_binding",
            "list_bindings",
            "list_members",
            "get_timeline",
            "check_all_timers",
        ),
        room_query,
    ),
    "close": _close,
    "process_inbound": _process_inbound,
    "send_event": _send_event,
    "post": _post,
    "commit_event": _commit_event,
    "update_event": _update_event,
    "delete_event": _delete_event,
    "create_room": _create_room,
    "attach_channel": _attach_channel,
    "detach_channel": _detach_channel,
    "mute": _mute,
    "unmute": _unmute,
    "set_access": _set_access,
    "set_visibility": _set_visibility,
    "update_binding_metadata": _update_binding_metadata,
    "close_room": _close_room,
    "pause_room": _pause_room,
    "resume_room": _resume_room,
    "reopen_room": _reopen_room,
    "send_greeting": _send_greeting,
    "regenerate_response": _regenerate_response,
    "regenerate_target": _regenerate_target,
    "archive_room": _archive_room,
    "check_timers": _check_timers,
    "set_room_timers": _set_room_timers,
    "set_agent_response_policy": _set_agent_response_policy,
    "register_channel": _register_channel,
    "unregister_channel": _unregister_channel,
    "register_hook": _register_hook,
    "add_room_hook": _add_room_hook,
    "remove_room_hook": _remove_room_hook,
    "mute_output": _mute_output,
    "unmute_output": _unmute_output,
    "process_delivery_status": _process_delivery_status,
    "process_webhook": _process_webhook,
    "deliver": _deliver,
    "cancel_delivery": _cancel_delivery,
    "delivery.cancel": _delivery_cancel,
    "delivery.wait": _delivery_wait,
    "steer": _steer,
    "acp.cancel": _acp_cancel,
}


def result_event_id(result: Any) -> str | None:
    """The id of the event a call's result names, for ``$label`` references."""
    if isinstance(result, InboundResult):
        return None if result.event is None else result.event.id
    if isinstance(result, RoomEvent):
        return result.id
    if isinstance(result, DeliveryOutcome):
        return result.event_id
    if isinstance(result, DelegatedTask):
        return result.id
    return None
