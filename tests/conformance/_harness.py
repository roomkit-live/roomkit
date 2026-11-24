"""Runs one conformance scenario on the Python core and compares what it did.

roomkit-specs ``conformance/README.md`` is the contract: the setup is made
before anything is recorded, the steps run in order, the kit is left to go
idle, and what the run produced is normalized and matched against the
scenario's expectations.
"""

from __future__ import annotations

import asyncio
import base64
import re
from collections.abc import Awaitable, Callable, Coroutine, Mapping
from typing import Any

from pydantic_core import to_jsonable_python

from roomkit import (
    HumanInputHandler,
    InMemoryStore,
    RoomKit,
    RoomLockManager,
    ThinkingEvent,
    current_turn_host,
)
from roomkit.models.channel import ChannelBinding, RateLimit
from roomkit.models.context import RoomContext
from roomkit.models.delivery import DeliveryHandle, DeliveryStatus
from roomkit.models.event import RoomEvent
from roomkit.models.framework_event import FrameworkEvent
from roomkit.models.pending_input import PendingInputEvent
from roomkit.models.plan_event import PlanUpdatedEvent
from roomkit.models.store_filter import EventFilter, PersistencePolicy
from roomkit.models.task import Observation
from roomkit.models.tool_call import (
    AIGenerationEvent,
    AIResponseEvent,
    ToolCallEvent,
    ToolRoundEvent,
)
from roomkit.models.trace import ProtocolTrace
from roomkit.orchestration import DispatchDecisionEvent, HandoffHandler, SpeakQueueEvent
from roomkit.orchestration.base import Orchestration
from roomkit.orchestration.status_bus import StatusEntry
from roomkit.realtime import EphemeralEvent, EphemeralEventType, InMemoryRealtime
from roomkit.recorder.feed import RoomRecordingFeed
from roomkit.recorder.mock import MockMediaRecorder
from roomkit.speaking import SpeakDecisionEvent, ThoughtEvent
from roomkit.store.base import ConversationStore
from roomkit.store.sqlite import SQLiteStore
from roomkit.tasks import DelegatedTask, DelegateHandler
from roomkit.voice.events import RecordingStartedEvent, RecordingStoppedEvent
from roomkit.voice.realtime.events import RealtimeDelegationEvent
from tests import redis_turn_mode
from tests.conformance._acp import ScriptedACPAgent, ScriptedACPTransport
from tests.conformance._acp_context import ScriptedContextContributor
from tests.conformance._acp_handler import ScriptedExternalHandler
from tests.conformance._acp_mcp import ScriptedMCPServers
from tests.conformance._addressing import ADDRESSING_CALLS
from tests.conformance._agent import ScriptedAIRouter, ScriptedModel, tool_call_record
from tests.conformance._ai_hooks import generation_record, response_record, tool_round_record
from tests.conformance._audits import Auditors
from tests.conformance._below_kit import BELOW_KIT_CALLS
from tests.conformance._calls import (
    CALLS,
    UnmappedError,
    UnreadableError,
    add_participant,
    attach_channel,
    create_room,
    dump,
    error_of,
    register_channel,
    register_hook,
    result_event_id,
)
from tests.conformance._compare import Check, compare
from tests.conformance._conference import CONFERENCE_CALLS, ConferenceFixture
from tests.conformance._delegate_tool import DELEGATE_TOOL_CALLS
from tests.conformance._delivery import (
    backend_busy,
    delivery_options,
    start_backend,
    wait_for_lock,
)
from tests.conformance._discussion import (
    DISCUSSION_CALLS,
    discussions_busy,
    dispatch_decision_record,
    speak_queue_record,
)
from tests.conformance._executors import EXECUTOR_CALLS, Executors
from tests.conformance._handoff import HANDOFF_CALLS
from tests.conformance._host_verbs import HOST_CALLS
from tests.conformance._identity import kit_options, register_identity_hook
from tests.conformance._inputs import INPUT_CALLS, request_record
from tests.conformance._kit_executor import KIT_EXECUTOR_CALLS
from tests.conformance._locks import LOCK_EVENTS, lock_options, set_lock_releases
from tests.conformance._mcp import MCPServers
from tests.conformance._mcp_host import MCP_HOST_CALLS
from tests.conformance._media import media_options
from tests.conformance._members import MEMBER_CALLS, store_identities
from tests.conformance._normalize import Observed, is_lifecycle_record
from tests.conformance._publications import Publications
from tests.conformance._realtime_session import REALTIME_CALLS, ScriptedRealtimeSession
from tests.conformance._recording import RECORDING_CALLS
from tests.conformance._routing import ROUTING_CALLS
from tests.conformance._script import (
    GATE_BOUND_S,
    Gates,
    ScriptedFailureError,
    ScriptedRouter,
    event_handler,
)
from tests.conformance._skill_registry import SKILL_REGISTRY_CALLS
from tests.conformance._speaking import decision_record
from tests.conformance._store import STORE_CALLS
from tests.conformance._store_faults import FaultingStore, StoreFaults
from tests.conformance._strategies import (
    STRATEGY_CALLS,
    kit_orchestration,
    strategies,
)
from tests.conformance._subscriptions import SUBSCRIPTION_CALLS
from tests.conformance._tasks import TASK_CALLS
from tests.conformance._thinking import thought_record
from tests.conformance._verification import (
    VERIFICATION_CALLS,
    VERIFICATION_EVENTS,
    opened_labels,
    verification_options,
)
from tests.conformance._video import VIDEO_CALLS, VideoFixture, video_payload
from tests.conformance._voice import VOICE_CALLS, VoiceFixture, voice_payload

# The framework events of the §26.1 table, and Processing timeout's (§8.2).
LEVEL_0_EVENTS = (
    "room_created",
    "room_paused",
    "room_closed",
    "room_archived",
    "room_refused_event",
    "channel_registered",
    "channel_unregistered",
    "room_channel_attached",
    "room_channel_detached",
    "event_processed",
    "event_blocked",
    "delivery_succeeded",
    "delivery_failed",
    "broadcast_partial_failure",
    "chain_depth_exceeded",
    "hook_error",
    "hook_timeout",
    "session_started",
    "process_timeout",
    # Identity resolution's (Level 1, §8.2).
    "identity_resolved",
    "identity_timeout",
)

# The framework events of the Level 2 features a scenario exercises: the
# status bus's (§19.8) and a feedback's (§15.9).
LEVEL_2_EVENTS = ("status_posted", "feedback")

# The framework event of a human-input request's decision (§8.2, §9.3).
INPUT_EVENTS = ("user_input_required",)

# The framework events of an AI channel's turn (§8.2, Level 1), which
# ``expect.ai_events`` compares apart from ``framework_events``.
AI_EVENTS = ("before_ai_generation", "ai_response")

# A call that has not returned by then fails the scenario instead of hanging
# it; so does a kit that does not go idle.
CALL_BOUND_S = 10.0
_EVENT_ALIAS = re.compile(r"^(.+)#(\d+)$")
_CREATED_ROOM = re.compile(r"^\$r(\d+)$")
_REQUEST = re.compile(r"^\$p(\d+)$")


_THINKING_EVENTS = {
    EphemeralEventType.THINKING_START,
    EphemeralEventType.THINKING_DELTA,
    EphemeralEventType.THINKING_END,
}
_TOOL_EVENTS = {
    EphemeralEventType.TOOL_CALL_START,
    EphemeralEventType.TOOL_CALL_END,
    EphemeralEventType.TOOL_CALL_DELTA,
}
_PRESENCE_EVENTS = {
    EphemeralEventType.PRESENCE_ONLINE,
    EphemeralEventType.PRESENCE_OFFLINE,
}
_EPHEMERALS = ("thinking_events", "tool_events", "presence_events", "custom_events")
"""The expectations a room's subscribers' ephemeral events are recorded for."""


class _TextRecorder(InMemoryRealtime):
    """The answer's text a kit's realtime is handed, a room minted during the
    scenario included, which nothing could subscribe to ahead (README:
    ``text_events``)."""

    def __init__(self, recorded: dict[str, list[Any]]) -> None:
        super().__init__()
        self._recorded = recorded

    async def publish(self, channel: str, event: EphemeralEvent) -> None:
        if event.type == EphemeralEventType.TEXT_DELTA:
            self._recorded.setdefault(event.room_id, []).append(dict(event.data))
        await super().publish(channel, event)


# The calls a call's `kit` may run on another process (README, "Calls").
KIT_CALLS = frozenset(
    {
        "process_inbound",
        "create_room",
        "attach_channel",
        "add_room_hook",
        "subscribe_room",
        "publish_typing",
        "steer",
        "agent_presence",
        "regenerate_response",
        "install_strategy",
        "uninstall_strategy",
        "claim_room_strategy",
        "room_strategy",
        "room_strategy_is_current",
        "speak_queue",
        "listen_only",
        "talk_again",
        "close",
    }
)


# What another process of ``setup.processes`` opens on the run's database:
# a store of its own and a lock manager of its own, both initialized.
OpenProcess = Callable[[], Awaitable[tuple[ConversationStore, RoomLockManager]]]


class Harness:
    """One run of one scenario."""

    def __init__(
        self,
        scenario: Mapping[str, Any],
        store: ConversationStore | None = None,
        *,
        observation_store: ConversationStore | None = None,
        lock_manager: RoomLockManager | None = None,
        open_process: OpenProcess | None = None,
    ) -> None:
        self.scenario = scenario
        # A separate reader can inspect persistent records after the scenario
        # closes its kit and its store. The caller owns this reader's lifetime.
        self._observation_store = observation_store
        # The run's kit's lock manager when the scenario scripts none, and what
        # each other process opens. Without them the kit has its default lock
        # manager, and the processes share it and the run's store.
        self._lock_manager = lock_manager
        self._open_process = open_process
        self.gates = Gates()
        self.conferences: dict[str, ConferenceFixture] = {}
        self.video_fixtures: dict[str, VideoFixture] = {}
        self.media_recorders: dict[str, MockMediaRecorder] = {}
        self.recording_feeds: dict[tuple[str, str], RoomRecordingFeed] = {}
        self.voice_fixtures: dict[str, VoiceFixture] = {}
        self.observed = Observed(literal_rooms=_literal_rooms(scenario))
        self._recording = False
        self._labels: dict[str, str | None] = {}
        # The completion handles of deferred calls, by their call's label.
        self.handles: dict[str, DeliveryHandle] = {}
        self._background: dict[str, asyncio.Task[dict[str, Any]]] = {}
        self._after_turn: list[asyncio.Task[None]] = []
        # The tasks still unwinding or started by reactions (README, Steps and Reactions).
        self._spawned: list[asyncio.Task[Any]] = []
        self.tasks: dict[str, DelegatedTask] = {}
        self._checks: list[Check] = []
        self._step_outcomes: list[dict[str, Any]] = []
        self._problems: list[str] = []
        self.models: dict[str, ScriptedModel] = {}
        self.realtime_sessions: dict[str, ScriptedRealtimeSession] = {}
        self.ai_routers: dict[str, ScriptedAIRouter] = {}
        self.acp_agents: dict[str, ScriptedACPAgent] = {}
        self.acp_handlers: dict[str, ScriptedExternalHandler] = {}
        self.external_handlers: dict[str, ScriptedExternalHandler] = {}
        self.acp_mcp: dict[str, ScriptedMCPServers] = {}
        self.acp_context: dict[str, ScriptedContextContributor] = {}
        self.acp_transports: dict[str, ScriptedACPTransport] = {}
        self.auditors = Auditors()
        self.mcp = MCPServers()
        self.handoff_handlers: dict[str, HandoffHandler] = {}
        self.delegate_handlers: dict[str, DelegateHandler] = {}
        # The human-input handler the runner holds per channel id (README,
        # `human_input.create`), made at the first call naming it.
        self.human_inputs: dict[str, HumanInputHandler] = {}
        setup = scenario.get("setup", {})
        self.discussion_done: dict[str, bool] = {}
        self.strategies = strategies(setup, self)
        self.executors = Executors(self)
        # The faults the steps arm on the kit's store (README, "Steps").
        self.store_faults = StoreFaults(self.gates.hold)
        self.kit = RoomKit(**self._kit_arguments(setup.get("kit", {}), store))
        self._publications = Publications()
        self._publications.watch(self.kit)
        # The other processes of `setup.processes`, by id, and the kits closed.
        self.processes: dict[str, RoomKit] = {}
        self._closed: list[RoomKit] = []

    def _kit_arguments(
        self, kit_setup: Mapping[str, Any], store: ConversationStore | None
    ) -> dict[str, Any]:
        """What the run's kit is built with, read from the setup's ``kit``."""
        kwargs: dict[str, Any] = {"error_classes": (ScriptedFailureError,)}
        if "max_chain_depth" in kit_setup:
            kwargs["max_chain_depth"] = kit_setup["max_chain_depth"]
        if "process_timeout_ms" in kit_setup:
            kwargs["process_timeout"] = kit_setup["process_timeout_ms"] / 1000
        if "close_grace_ms" in kit_setup:
            kwargs["close_grace"] = kit_setup["close_grace_ms"] / 1000
        if "inbound_rate_limit" in kit_setup:
            kwargs["inbound_rate_limit"] = RateLimit(**kit_setup["inbound_rate_limit"])
        if "persistence_policy" in kit_setup:
            kwargs["persistence_policy"] = PersistencePolicy(**kit_setup["persistence_policy"])
        if store is not None:
            kwargs["store"] = store
        if kit_setup.get("store_faults") or "store_owner" in kit_setup:
            self.store_faults.owner = kit_setup.get("store_owner")
            kwargs["store"] = FaultingStore(
                kwargs.get("store") or InMemoryStore(), self.store_faults
            )
        if self._lock_manager is not None:
            kwargs["lock_manager"] = self._lock_manager
        if "router" in kit_setup:
            router = kit_setup["router"]
            kwargs["inbound_router"] = ScriptedRouter(router["room_id"], self, router.get("gate"))
        kwargs |= kit_options(kit_setup, self)
        kwargs |= delivery_options(kit_setup)
        kwargs |= lock_options(kit_setup)
        kwargs |= verification_options(kit_setup)
        kwargs |= media_options(kit_setup, self)
        kwargs |= kit_orchestration(kit_setup, self)
        kwargs |= self.executors.site_options("owner", kit_setup)
        kwargs |= self.executors.kit_options()
        if "text_events" in self.scenario.get("expect", {}):
            kwargs["realtime"] = _TextRecorder(self.observed.text_events)
        return kwargs

    def kit_of(self, spec: Mapping[str, Any]) -> RoomKit:
        """The kit a call's ``kit`` names: a process of the setup, the run's
        own kit when it names none (README, ``processes``)."""
        name = spec.get("kit")
        if name is None:
            return self.kit
        if spec["call"] not in KIT_CALLS:
            raise UnreadableError(f"{spec['call']} takes no kit")
        if name not in self.processes:
            raise UnreadableError(f"no process {name!r} in setup.processes")
        return self.processes[name]

    def closed(self, kit: RoomKit) -> None:
        """*kit* was closed by a ``close`` call: no wait, no close of it again."""
        self._closed.append(kit)

    def _open_kits(self) -> list[RoomKit]:
        kits = [self.kit, *self.processes.values()]
        return [kit for kit in kits if all(kit is not closed for closed in self._closed)]

    def store_of(self, kit: RoomKit) -> ConversationStore:
        """The store a call on *kit* reaches: its own, or once a ``close`` call
        closed the kit, the run's reader of the same database, since the
        kit's close released a store it opened (RFC §12.10.4)."""
        if self.store_faults.owner is not None:
            # The store as its owner closes it (README, ``kit.store_owner``):
            # a call reaches it after the kit's close too.
            return kit.store
        if self._observation_store is not None and any(kit is c for c in self._closed):
            return self._observation_store
        return kit.store

    def strategy(self, strategy_id: str) -> Orchestration:
        """The scripted strategy *strategy_id* names (README, \"Setup\")."""
        if strategy_id not in self.strategies:
            raise UnreadableError(f"no scripted strategy {strategy_id!r}")
        return self.strategies[strategy_id]

    # -- recording --------------------------------------------------------

    def record(self, kind: str, room_id: str, channel_id: str, event: RoomEvent) -> None:
        if not self._recording:
            return
        per_room = getattr(self.observed, kind).setdefault(room_id, {})
        per_room.setdefault(channel_id, []).append(event.model_dump(mode="json"))

    def record_context(
        self, room_id: str, channel_id: str, call: str, context: RoomContext
    ) -> None:
        """The context a scripted channel's ``parse``, ``deliver`` or
        ``on_event`` was handed (README, ``contexts``): its events, the ids
        alone and whole, lifecycle records left out, its room, its bindings
        and its participants."""
        if not self._recording:
            return
        per_room = self.observed.contexts.setdefault(room_id, {})
        per_room.setdefault(channel_id, []).append(
            {
                "call": call,
                "recent_events": _recent_ids(context),
                "events": _context_events(context),
                "room": context.room.model_dump(mode="json"),
                "bindings": [b.model_dump(mode="json") for b in context.bindings],
                "participants": [p.model_dump(mode="json") for p in context.participants],
            }
        )

    def record_streamed(self, room_id: str, channel_id: str, item: dict[str, Any]) -> None:
        """One item a streaming channel's ``deliver_stream`` read, as the
        scenario reads it: ``{"text": ...}``, ``{"row": event}``,
        ``{"failed": true}`` where the stream raised, or ``{"cut": event}``
        for the end of a stream cut from outside."""
        if not self._recording:
            return
        per_room = self.observed.streamed.setdefault(room_id, {})
        per_room.setdefault(channel_id, []).append(item)

    def record_compacted(self, room_id: str, channel_id: str, item: dict[str, Any]) -> None:
        """One compaction between two tool rounds a scripted memory was asked:
        ``{history, usage}``."""
        if not self._recording:
            return
        per_room = self.observed.compacted.setdefault(room_id, {})
        per_room.setdefault(channel_id, []).append(item)

    def record_room_callback(
        self,
        room_id: str,
        channel_id: str,
        call: str,
        binding: ChannelBinding | None,
        **details: Any,
    ) -> None:
        """One room callback a scripted channel was called with (§6.1), also
        named in the hooks' order as ``<channel_id>.<call>``."""
        if not self._recording:
            return
        item: dict[str, Any] = {"call": call, **details}
        if binding is not None:
            item["binding"] = binding.model_dump(mode="json")
        per_room = self.observed.room_callbacks.setdefault(room_id, {})
        per_room.setdefault(channel_id, []).append(item)
        self.observed.hook_order.append(f"{channel_id}.{call}")

    def record_executor(
        self,
        executor_id: str,
        channel_id: str,
        build: int,
        call: str,
        room_id: str | None,
        **details: Any,
    ) -> dict[str, Any] | None:
        """One call an executor made on a channel it built (README,
        ``executors``): its record, None before the steps."""
        if not self._recording:
            return None
        item = {"call": call, "channel_id": channel_id, "build": build, **details}
        if room_id is not None:
            item["room_id"] = room_id
        self.executors.calls.setdefault(executor_id, []).append(item)
        return item

    def executor_calls(self, executor_id: str) -> list[dict[str, Any]]:
        """What an executor did so far (README, ``executors``)."""
        return list(self.executors.calls.get(executor_id, []))

    def record_dispatch_turn(self, strategy_id: str, turn: dict[str, Any]) -> None:
        """A turn the scripted dispatch policy of *strategy_id* was handed."""
        if self._recording:
            self.observed.dispatch_turns.setdefault(strategy_id, []).append(turn)

    def record_hook(
        self, name: str, trigger: str, room_id: str, payload: Any, context: RoomContext
    ) -> None:
        if not self._recording:
            return
        event = payload.model_dump(mode="json") if isinstance(payload, RoomEvent) else None
        call: dict[str, Any] = {
            "trigger": trigger,
            "room_id": room_id,
            "event": event,
            "context_room": context.room.model_dump(mode="json"),
            "recent_events": _recent_ids(context),
            "context_events": _context_events(context),
            "context_bindings": [b.model_dump(mode="json") for b in context.bindings],
            "context_participants": [p.model_dump(mode="json") for p in context.participants],
        }
        if trigger in {"before_tts", "after_tts"} and isinstance(payload, str):
            call["voice"] = {"text": payload}
        elif (voice := voice_payload(payload)) is not None:
            call["voice"] = voice
        if (video := video_payload(payload)) is not None:
            call["video"] = video
        if isinstance(payload, RecordingStartedEvent | RecordingStoppedEvent):
            call["recording"] = {"recording_id": payload.id, "room_id": payload.room_id}
            if isinstance(payload, RecordingStoppedEvent):
                call["recording"]["urls"] = list(payload.urls)
        if isinstance(payload, RealtimeDelegationEvent):
            call["delegation"] = {"delegation_id": payload.delegation_id, "target": payload.target}
        if isinstance(payload, ProtocolTrace):
            raw = payload.raw
            call["trace"] = {
                "channel_id": payload.channel_id,
                "room_id": payload.room_id,
                "session_id": payload.session_id,
                "direction": payload.direction,
                "protocol": payload.protocol,
                "summary": payload.summary,
                "raw": {"base64": base64.b64encode(raw).decode("ascii")}
                if isinstance(raw, bytes)
                else raw,
                "metadata": dict(payload.metadata),
                "timestamp": payload.timestamp.isoformat(),
            }
        if isinstance(payload, ToolCallEvent):
            call["tool_call"] = tool_call_record(payload)
        if isinstance(payload, DeliveryStatus):
            call["delivery_status"] = payload.model_dump(mode="json")
        if isinstance(payload, StatusEntry):
            call["status_entry"] = payload.model_dump(mode="json")
        if isinstance(payload, Observation):
            call["observation"] = payload.model_dump(mode="json")
        if isinstance(payload, SpeakQueueEvent):
            call["speak_queue"] = speak_queue_record(payload)
        if isinstance(payload, DispatchDecisionEvent):
            call["dispatch_decision"] = dispatch_decision_record(payload)
        if isinstance(payload, PendingInputEvent):
            self.saw_request(payload.pending_id)
            call["request"] = request_record(payload)
        if isinstance(payload, ThoughtEvent):
            call["thought"] = thought_record(payload)
        if isinstance(payload, SpeakDecisionEvent):
            call["speak_decision"] = decision_record(payload)
        if isinstance(payload, AIGenerationEvent):
            call["generation"] = generation_record(payload)
        if isinstance(payload, ToolRoundEvent):
            call["tool_round"] = tool_round_record(payload)
        if isinstance(payload, AIResponseEvent):
            call["response"] = response_record(payload)
        if isinstance(payload, ThinkingEvent):
            call["thinking"] = {
                "channel_id": payload.channel_id,
                "thinking": payload.thinking,
                "round_index": payload.round_index,
            }
        if isinstance(payload, PlanUpdatedEvent):
            call["plan"] = {
                "room_id": payload.room_id,
                "channel_id": payload.channel_id,
                "tasks": payload.tasks,
                "timestamp": payload.timestamp.isoformat(),
            }
        self.observed.hooks.setdefault(name, []).append(call)
        self.observed.hook_order.append(name)

    def record_resolver(self, room_id: str, message: Any) -> None:
        """One call of the scripted resolver (§11.1)."""
        if not self._recording:
            return
        call = {
            "room_id": room_id,
            "channel_id": message.channel_id,
            "sender_id": message.sender_id,
        }
        self.observed.resolver.append(call)

    def record_router_call(self, call: dict[str, Any]) -> None:
        """The custom inbound router's inputs, recorded before any gate."""
        if self._recording:
            self.observed.router_calls.append(call)

    def record_media_resolver(self, url: str, room_id: str) -> None:
        """One call of the kit's media resolver (§6.4)."""
        if not self._recording:
            return
        self.observed.media_resolver.append({"url": url, "room_id": room_id})

    def record_identity_hook(
        self, name: str, trigger: str, room_id: str, event: RoomEvent, id_result: Any
    ) -> None:
        """One call of a scripted identity hook (§11.2): its event and the
        IdentityResult it is handed."""
        if not self._recording:
            return
        call = {
            "trigger": trigger,
            "room_id": room_id,
            "event": event.model_dump(mode="json"),
            "identity_result": id_result.model_dump(mode="json"),
        }
        self.observed.hooks.setdefault(name, []).append(call)
        self.observed.hook_order.append(name)

    def saw_request(self, pending_id: str) -> None:
        """A human-input request the run shows: the next ``$p<n>``."""
        if pending_id not in self.observed.requests:
            self.observed.requests.append(pending_id)

    async def _on_framework_event(self, event: FrameworkEvent) -> None:
        if event.type == "room_created" and event.room_id is not None:
            self.observed.created_rooms.append(event.room_id)
        if event.type == "user_input_required":
            self.saw_request(str(event.data.get("pending_id")))
        if not self._recording:
            return
        # The members the RFC's Data column names, flat; `data` kept whole
        # too, for the rows that name `data.x` (§8.2). Read in the serialized
        # form: a status_posted's data is its entry's dump, its status an enum.
        data = to_jsonable_python(event.data, fallback=str)
        flat: dict[str, Any] = {"event": event.type, **data, "data": data}
        for key in ("room_id", "channel_id", "event_id"):
            value = getattr(event, key)
            if value is not None:
                flat[key] = value
        if event.type in AI_EVENTS:
            self.observed.ai_events.append(flat)
        else:
            self.observed.framework_events.append(flat)

    # -- the run ----------------------------------------------------------

    async def _subscribe_thinking(self, room_id: str) -> None:
        """Records the room's thinking and tool call events, for a scenario
        that expects them."""
        expect = self.scenario.get("expect", {})
        if not any(kind in expect for kind in _EPHEMERALS):
            return

        async def record(event: EphemeralEvent) -> None:
            if event.type in _PRESENCE_EVENTS:
                # From the steps on: the setup's attaches publish too.
                if self._recording:
                    presence = {"type": event.type.value, "user_id": event.user_id}
                    self.observed.presence_events.setdefault(room_id, []).append(presence)
                return
            if event.type in _THINKING_EVENTS:
                recorded = self.observed.thinking_events
            elif event.type in _TOOL_EVENTS:
                recorded = self.observed.tool_events
            elif event.type == EphemeralEventType.CUSTOM:
                self.observed.custom_events.setdefault(room_id, []).append(
                    {"room_id": event.room_id, "channel_id": event.channel_id, "data": event.data}
                )
                return
            else:
                return
            recorded.setdefault(room_id, []).append({"type": event.type.value, **event.data})

        await self.kit.realtime.subscribe_to_room(room_id, record)

    async def _thinking_read(self) -> None:
        """Finish owner publications before reading ephemeral observations."""
        if any(kind in self.scenario.get("expect", {}) for kind in (*_EPHEMERALS, "subscribers")):
            await self.idle()

    async def run(self) -> list[str]:
        """Runs the scenario; returns every mismatch with its expectations."""
        try:
            await self._setup()
            self._recording = True
            for position, step in enumerate(self.scenario["steps"]):
                await self._step(step, f"steps[{position}]")
            await self.idle()
            # A fault no call took tested nothing (README, "Steps").
            self._problems.extend(self.store_faults.untaken())
            self._problems.extend(self.executors.faults.untaken())
            # The runner's own reads read what a store the kit closed holds.
            self.store_faults.closed = False
            await self._thinking_read()
            await self._observe()
            # A subscriber records the scenario's events. Closing the fixture can
            # publish additional offline events after the final idle wait.
            self.observed.subscribers = {
                name: list(events) for name, events in self.observed.subscribers.items()
            }
        finally:
            for kit in self._open_kits():
                await kit.close()
            if isinstance(self.kit.store, FaultingStore) and self.kit.store.owned_by_integrator:
                # Its owner closes a store the kit left open.
                await self.kit.store.inner.close()
            await self.executors.close_all()
            for handler in self.human_inputs.values():
                await handler.close()
            await self.mcp.close()
            self.auditors.close()
        return self._problems + compare(
            self.scenario.get("expect", {}), self.observed, self._checks
        )

    async def _set_room_status(self, room: Mapping[str, Any]) -> None:
        transitions = {
            "paused": (self.kit.pause_room,),
            "closed": (self.kit.close_room,),
            "archived": (self.kit.close_room, self.kit.archive_room),
        }
        for transition in transitions.get(room.get("status", "active"), ()):
            await transition(room["id"])

    async def _setup(self) -> None:
        setup = self.scenario.get("setup", {})
        recorded = LEVEL_0_EVENTS + LEVEL_2_EVENTS + INPUT_EVENTS + VERIFICATION_EVENTS + AI_EVENTS
        recorded += LOCK_EVENTS
        for event_type in recorded:
            self.kit.on(event_type)(self._on_framework_event)
        self.auditors.set_up(setup.get("session_auditor"), self.kit)
        await self.mcp.connect(setup.get("channels", []))
        for channel in setup.get("channels", []):
            register_channel(self.kit, self, channel)
        await self.executors.start_all()
        await store_identities(self.kit, setup.get("identities", []))
        for room in setup.get("rooms", []):
            await create_room(self.kit, room)
            if "store_metadata" in room:
                await self.kit.store.patch_room_metadata(room["id"], room["store_metadata"])
            await self._subscribe_thinking(room["id"])
            for binding in room.get("bindings", []):
                await attach_channel(self.kit, room["id"], binding)
                if binding.get("muted"):
                    await self.kit.mute(room["id"], binding["channel_id"])
            for participant in room.get("participants", []):
                await add_participant(self.kit, room["id"], participant)
            await self._set_room_status(room)
        for hook in setup.get("hooks", []):
            register_hook(self.kit, self, hook)
        for hook in setup.get("identity_hooks", []):
            register_identity_hook(self.kit, self, hook)
        for handler in setup.get("event_handlers", []):
            self.kit.on(handler["event"])(event_handler(handler, self))
        for process in setup.get("processes", []):
            self.processes[process["id"]] = await self._process(process["id"])
        await start_backend(self.kit)
        await self.idle()

    async def _process(self, process_id: str) -> RoomKit:
        """Another process serving the rooms: a kit over the run's store and
        lock manager, or what the caller opens for it, holding the setup's
        channels as objects of its own."""
        if self._open_process is not None:
            store, lock_manager = await self._open_process()
        else:
            store, lock_manager = self.kit.store, self.kit.lock_manager
            if isinstance(store, SQLiteStore):
                # Processes share the file, never a connection that one's close
                # would close for all the others (as in the Rust SQLite runner).
                store = SQLiteStore(store._path)
        kit = RoomKit(
            store=store, lock_manager=lock_manager, **self.executors.process_options(process_id)
        )
        self._publications.watch(kit)
        for channel in self.scenario.get("setup", {}).get("channels", []):
            register_channel(kit, self, channel, process=True)
        return kit

    async def _start_background(self, label: str, call: Mapping[str, Any]) -> None:
        self._background[label] = asyncio.create_task(self.invoke(call, {}))
        # The call runs to its first wait before the next step (README, Steps).
        await asyncio.sleep(0)

    async def _step(self, step: Mapping[str, Any], path: str) -> None:
        if "cancel" in step or "close_while_pending" in step:
            label = step["cancel"] if "cancel" in step else step["close_while_pending"]
            caller = self._background.get(label)
            if caller is None or caller.done():
                self._problems.append(f"{path}: no pending background call {label}")
                return
            if "cancel" in step:
                self._spawned.append(self._background.pop(label))
                caller.cancel()
                # Let the caller receive cancellation. A nested call it owns can
                # still be waiting for a gate a later step will open.
                await asyncio.sleep(0)
                return
            await self._start_background(step["as"], {"call": "close"})
            return
        if "close_cut_at" in step:
            await self._close_cut_at(step["close_cut_at"], path)
            return
        if "call" in step:
            if step.get("background"):
                await self._start_background(step["as"], step)
                return
            outcome = await self.invoke(step, {})
            self._step_outcomes.append(outcome)
            self._checks.append((path, step.get("expect"), outcome))
        elif "restart_process" in step:
            name = step["restart_process"]
            kit = self.processes[name]
            await self.executors.close_process_workers(name)
            if kit not in self._closed:
                await kit.close()
                self.closed(kit)
            self.executors.forget_process(name)
            self.processes[name] = await self._process(name)
        elif "await" in step:
            outcome = await self._background.pop(step["await"])
            self._step_outcomes.append(outcome)
            self._checks.append((path, step.get("expect"), outcome))
        elif "concurrent" in step:
            calls = step["concurrent"]
            outcomes = await asyncio.gather(*(self.invoke(call, {}) for call in calls))
            for position, (call, outcome) in enumerate(zip(calls, outcomes, strict=True)):
                self._step_outcomes.append(outcome)
                self._checks.append(
                    (f"{path}.concurrent[{position}]", call.get("expect"), outcome)
                )
        elif "wait_for" in step:
            try:
                gate = step["wait_for"]
                await self.gates.wait_for(gate)
                if "after" in step and not self.gates.reached_after(gate, step["after"]):
                    self._problems.append(f"{path}: {gate} did not follow {step['after']}")
                if "pending" in step:
                    caller = self._background.get(step["pending"])
                    if caller is None or caller.done():
                        self._problems.append(
                            f"{path}: no pending background call {step['pending']}"
                        )
            except TimeoutError:
                self._problems.append(f"{path}: no call reached the gate {step['wait_for']}")
        elif "wait_for_lock" in step:
            room_id = await self.resolve(step["wait_for_lock"], {})
            waiters = step.get("waiters", 1)
            try:
                await wait_for_lock(self.kit, room_id, waiters)
            except TimeoutError:
                self._problems.append(f"{path}: {waiters} calls never waited for {room_id}'s lock")
        elif "lock_releases" in step:
            set_lock_releases(self.kit, step["lock_releases"])
        elif "acp_disconnect" in step:
            await self.acp_transports[step["acp_disconnect"]].close()
        elif "under_lock" in step:
            await self._under_lock(step, path)
        elif "store_fault" in step:
            await self._arm_store_fault(step, path)
        elif "open" in step:
            self.gates.open(step["open"])
        elif "idle" in step:
            await self.idle()

    async def _close_cut_at(self, gate: str, path: str) -> None:
        """Start the kit's close, wait until it reaches *gate*, then cut its
        shutdown as a process's end does, every task but the runner's own
        cancelled (README, "Steps", ``close_cut_at``)."""
        self.closed(self.kit)
        closing = asyncio.create_task(self.kit.close())
        try:
            await self.gates.wait_for(gate)
        except TimeoutError:
            self._problems.append(f"{path}: the close never reached the gate {gate}")
        for task in asyncio.all_tasks() - {asyncio.current_task()}:
            task.cancel()
        await asyncio.gather(closing, return_exceptions=True)

    def record_channel_close(self, channel_id: str) -> None:
        """A scripted channel's close ran (README, ``channel_closes``)."""
        if self._recording:
            self.observed.channel_closes.append(channel_id)

    async def _arm_store_fault(self, step: Mapping[str, Any], path: str) -> None:
        """Arms a fault on the kit's store for the next call it matches
        (README, "Steps", ``store_fault``)."""
        if not isinstance(self.kit.store, FaultingStore):
            self._problems.append(f"{path}: a store_fault step needs setup.kit.store_faults")
            return
        room_id = await self.resolve(step["room_id"], {}) if "room_id" in step else None
        try:
            self.store_faults.arm(step, path, room_id)
        except ValueError as error:
            self._problems.append(str(error))

    async def _under_lock(self, step: Mapping[str, Any], path: str) -> None:
        """Make the step's calls under the room lock the host takes from the
        kit's lock manager, outside any pass (README, "Steps", RFC §13.5)."""
        room_id = await self.resolve(step["under_lock"], {})
        async with self.kit.lock_manager.locked(room_id):
            for position, call in enumerate(step["do"]):
                outcome = await self.invoke(call, {})
                self._step_outcomes.append(outcome)
                self._checks.append((f"{path}.do[{position}]", call.get("expect"), outcome))

    async def run_action(self, action: Mapping[str, Any], scope: Mapping[str, str]) -> None:
        """A call a hook or a channel makes while it runs; its outcome is kept.
        With ``own_task``, made on a task of its own, started in the caller's
        context, which it awaits: a cut of the caller cuts that wait alone."""
        if action.get("own_task"):
            own = asyncio.create_task(self.invoke(action, scope))
            self._spawned.append(own)
            outcome = await asyncio.shield(own)
        else:
            outcome = await self.invoke(action, scope)
        label = action.get("as")
        if label is not None:
            self.observed.actions.setdefault(label, []).append(outcome)
        # An action's outcome is checked by its own expect; with none, it
        # must not fail, unless the scenario's expect.actions checks its label.
        asserted = self.scenario.get("expect", {}).get("actions", {})
        if "expect" in action or label not in asserted:
            self._checks.append(
                (f"action {label or action['call']}", action.get("expect"), outcome)
            )

    def spawn(self, work: Coroutine[Any, Any, None]) -> None:
        """Start *work* in a task of the caller's context, as a hook's body
        or a parse starts one it does not wait for: ``create_task`` copies
        the context (RFC §9.5.1 Reentrance)."""
        self._spawned.append(asyncio.create_task(work))

    def leave_after_turn(self, actions: list[Mapping[str, Any]], scope: Mapping[str, str]) -> None:
        """Start the work a turn leaves for after it: a task in the turn's
        context, which makes *actions* in order once the turn ended."""
        host = current_turn_host()
        if host is None:
            self._problems.append("an after_turn reaction outside a turn: no turn's host")
            return
        self._after_turn.append(asyncio.ensure_future(self._after(host, actions, scope)))

    async def _after(
        self, host: Any, actions: list[Mapping[str, Any]], scope: Mapping[str, str]
    ) -> None:
        try:
            async with asyncio.timeout(GATE_BOUND_S):
                while host.open:
                    await asyncio.sleep(0.001)
                # The owner reads the turn's frames on its own: across a
                # backend, a call could overtake them. Work after the turn
                # is made once the kit's own work for it is over.
                quiet = 0
                while quiet < 3:
                    busy = await self._drain_once(after_turn=False)
                    await asyncio.sleep(0)
                    quiet = 0 if busy else quiet + 1
        except TimeoutError:
            self._problems.append(f"a turn did not end within {GATE_BOUND_S} s for its after_turn")
            return
        for action in actions:
            await self.run_action(action, scope)

    async def invoke(self, spec: Mapping[str, Any], scope: Mapping[str, str]) -> dict[str, Any]:
        """Makes one call; returns its outcome, its error named, never raising."""
        resolved = await self.resolve(dict(spec), scope)
        calls = CALLS | STORE_CALLS | TASK_CALLS | MEMBER_CALLS | INPUT_CALLS | STRATEGY_CALLS
        calls |= VERIFICATION_CALLS | ROUTING_CALLS | HANDOFF_CALLS | DISCUSSION_CALLS
        calls |= CONFERENCE_CALLS | REALTIME_CALLS
        calls |= ADDRESSING_CALLS
        calls |= MCP_HOST_CALLS
        calls |= SKILL_REGISTRY_CALLS
        calls |= HOST_CALLS
        calls |= SUBSCRIPTION_CALLS
        calls |= BELOW_KIT_CALLS | VOICE_CALLS | VIDEO_CALLS | RECORDING_CALLS
        call = (calls | DELEGATE_TOOL_CALLS | EXECUTOR_CALLS | KIT_EXECUTOR_CALLS)[spec["call"]]
        try:
            kit = self.kit_of(spec)
            result = await asyncio.wait_for(call(kit, self, resolved), CALL_BOUND_S)
        except TimeoutError:
            self._problems.append(f"{spec['call']} did not return within {CALL_BOUND_S} s")
            return {"result": None, "error": {"name": "<hung>"}}
        except UnreadableError as error:
            self._problems.append(f"{spec['call']}: the runner cannot read the call: {error}")
            return {"result": None, "error": {"name": "<unreadable>"}}
        except Exception as error:
            try:
                return {"result": None, "error": error_of(error)}
            except UnmappedError as unmapped:
                self._problems.append(f"{spec['call']} raised an unmapped error: {unmapped}")
                return {"result": None, "error": {"name": "<unmapped>"}}
        if "as" in spec:
            # A delegation through the turn's host answers its task's id, and a
            # delegate handler's call its task's.
            named = result if spec["call"] == "host.delegate" else result_event_id(result)
            if spec["call"] == "delegate.handle":
                named = result["task_id"]
            self._labels[spec["as"]] = named
            self._labels |= opened_labels(spec, result)
        return {"result": dump(result), "error": None}

    async def idle(self) -> None:
        """Waits until the kit runs nothing more in the background.

        It reads the kit's internals (its lanes, its observers, its held
        tasks): the Python core offers no public wait for its background work.
        With other processes, until no kit does: each discussion's driver is
        roused first, so that it reads what another process stored, until a
        round changes nothing the store holds of the rooms.
        """
        try:
            async with asyncio.timeout(CALL_BOUND_S):
                if not self.processes:
                    await self._quiet()
                    return
                before = None
                while True:
                    for kit in self._open_kits():
                        for discussion in list(kit._discussions.values()):
                            discussion.wake()
                        await self._quiet(kit)
                    after = await self._rooms_read()
                    if after == before:
                        return
                    before = after
        except TimeoutError:
            self._problems.append(f"the kit did not go idle within {CALL_BOUND_S} s")

    async def _quiet(self, kit: RoomKit | None = None) -> None:
        quiet = 0
        while quiet < 3:
            busy = await self._drain_once(kit=kit)
            await asyncio.sleep(0)
            quiet = 0 if busy else quiet + 1
            # Across Redis, work a turn began reaches its owner after the
            # turn, through the inbox: none may still be on its way.
            if quiet == 3 and await redis_turn_mode.handled_across(kit or self.kit):
                quiet = 0

    async def _rooms_read(self) -> list[tuple[str, int | None, Any, bool]]:
        """What the store holds of each room that the processes move: its
        latest index and its discussion's queue version."""
        read = []
        for room in await self.store_of(self.kit).list_rooms(limit=1000):
            queue = (room.metadata or {}).get("_speak_queue") or {}
            stored = "_discussion" in (room.metadata or {})
            read.append((room.id, room.latest_index, queue.get("version"), stored))
        return read

    async def _drain_once(self, *, after_turn: bool = True, kit: RoomKit | None = None) -> bool:
        if kit is not None and kit is not self.kit:
            return await self._drain_process(kit)
        busy = await self._publications.drain(self.kit)
        for lane in list(self.kit._lanes._lanes.values()):
            if lane._indexed or lane._unindexed or lane._busy:
                busy = True
                await lane.drain()
        if self.kit.hook_engine.running_observers():
            busy = True
            await self.kit.hook_engine.drain_observers()
        tasks = [
            task
            for held in (
                self.kit._pending_hook_tasks,
                self.kit._decided_writes,
                # A strategy's background runs (RFC §19.7.3): a run waits
                # for its task, then hands its outcome back.
                self.kit._background_runs,
            )
            for task in held
            if not task.done()
        ]
        tasks.extend(task for task in self._background.values() if not task.done())
        tasks.extend(self._task_runs())
        for fixture in self.realtime_sessions.values():
            tasks.extend(task for task in fixture.channel._scheduled_tasks if not task.done())
        if after_turn:
            tasks.extend(task for task in self._after_turn if not task.done())
            tasks.extend(task for task in self._spawned if not task.done())
        if tasks:
            busy = True
            await asyncio.wait(tasks, timeout=CALL_BOUND_S)
        if await backend_busy(self.kit):
            busy = True
            await asyncio.sleep(0.005)
        if discussions_busy(self.kit):
            busy = True
            await asyncio.sleep(0.001)
        return busy

    def _task_runs(self) -> list[asyncio.Task[Any]]:
        """Task work and completion callbacks, including tasks an agent's
        work_in_background tool started without a runner-side task handle.
        A task can finish its turn before its result has been handed back.
        """
        runner = self.kit.task_runner
        held = list(runner._tasks.values())  # ty: ignore[attr-defined]
        held.extend(runner._ends.values())  # ty: ignore[attr-defined]
        for handler in self.delegate_handlers.values():
            held.extend(handler._background_tasks)
        return [task for task in held if not task.done()]

    async def _drain_process(self, kit: RoomKit) -> bool:
        """One pass of the idle wait over another process's kit: its lanes,
        its observers and hook tasks, its discussions."""
        busy = await self._publications.drain(kit)
        for lane in list(kit._lanes._lanes.values()):
            if lane._indexed or lane._unindexed or lane._busy:
                busy = True
                await lane.drain()
        if kit.hook_engine.running_observers():
            busy = True
            await kit.hook_engine.drain_observers()
        tasks = [task for task in kit._pending_hook_tasks if not task.done()]
        if tasks:
            busy = True
            await asyncio.wait(tasks, timeout=CALL_BOUND_S)
        if discussions_busy(kit):
            busy = True
            await asyncio.sleep(0.001)
        return busy

    # -- references -------------------------------------------------------

    async def resolve(self, value: Any, scope: Mapping[str, str]) -> Any:
        if isinstance(value, dict):
            return {key: await self.resolve(item, scope) for key, item in value.items()}
        if isinstance(value, list):
            return [await self.resolve(item, scope) for item in value]
        if isinstance(value, str):
            return await self._reference(value, scope)
        return value

    def labelled(self, value: Any) -> Any:
        """*value* with each ``$<label>`` a call's ``as`` named read as what
        it names: a scripted model's call arguments, as the model writes
        the call (README, "Scripted agents")."""
        if isinstance(value, dict):
            return {key: self.labelled(item) for key, item in value.items()}
        if isinstance(value, list):
            return [self.labelled(item) for item in value]
        if isinstance(value, str) and value.startswith("$") and self._labels.get(value[1:]):
            return self._labels[value[1:]]
        return value

    async def _reference(self, value: str, scope: Mapping[str, str]) -> str:
        if value in ("$room", "$event"):
            return scope[value[1:]]
        if value == "$request" and "request" in scope:
            return scope["request"]
        if match := _REQUEST.match(value):
            number = int(match[1])
            requests = self.observed.requests
            return requests[number - 1] if number <= len(requests) else value
        if match := _CREATED_ROOM.match(value):
            return self._created_room(int(match[1]))
        if value.startswith("$") and value[1:] in self._labels:
            return self._labels[value[1:]] or value
        if match := _EVENT_ALIAS.match(value):
            room_id = await self._reference(match[1], scope)
            return await self._event_at(room_id, int(match[2])) or value
        return value

    def _created_room(self, number: int) -> str:
        created = [r for r in self.observed.created_rooms if r not in self.observed.literal_rooms]
        return created[number - 1] if number <= len(created) else f"$r{number}"

    async def _event_at(self, room_id: str, index: int) -> str | None:
        """The id of the room's event at *index*, read without lifecycle records."""
        timeline = await self._timeline(room_id)
        lifecycle = 0
        for event in timeline:
            if is_lifecycle_record(event):
                lifecycle += 1
            elif event["index"] - lifecycle == index:
                return event["id"]
        return None

    async def _timeline(self, room_id: str) -> list[dict[str, Any]]:
        store = self._observation_store or self.kit.store
        events = await store.list_events(
            room_id, limit=100_000, event_filter=EventFilter(include_blocked=True)
        )
        return sorted((e.model_dump(mode="json") for e in events), key=lambda e: e["index"])

    # -- observation and comparison ---------------------------------------

    async def _observe(self) -> None:
        store = self._observation_store or self.kit.store
        for room in await store.list_rooms(limit=100_000):
            self.observed.timelines[room.id] = await self._timeline(room.id)
            self.observed.rooms[room.id] = room.model_dump(mode="json")
            self.observed.bindings[room.id] = {
                binding.channel_id: binding.model_dump(mode="json")
                for binding in await store.list_bindings(room.id)
            }
            self.observed.observations[room.id] = [
                o.model_dump(mode="json") for o in await store.list_observations(room.id)
            ]
            self.observed.tasks[room.id] = [
                t.model_dump(mode="json") for t in await store.list_tasks(room.id)
            ]
            self.observed.participants[room.id] = [
                p.model_dump(mode="json") for p in await store.list_participants(room.id)
            ]
        self.observed.steps = self._step_outcomes
        self.observed.models = {
            channel_id: list(model.requests) for channel_id, model in self.models.items()
        }
        self.observed.ai_routes = {
            channel_id: list(router.requests) for channel_id, router in self.ai_routers.items()
        }
        self.observed.acp_wire = {
            channel_id: list(agent.received) for channel_id, agent in self.acp_agents.items()
        }
        self.observed.acp_prompts = {
            channel_id: list(agent.attempts) for channel_id, agent in self.acp_agents.items()
        }
        self._collect_external_handlers()
        self._collect_acp_permissions()
        self._collect_acp_mcp()
        self._collect_acp_context()
        self.observed.executors = {
            executor_id: list(calls) for executor_id, calls in self.executors.calls.items()
        }
        self.observed.turn_ends = {
            channel_id: list(ends) for channel_id, ends in self.executors.held_ends.ends.items()
        }
        self.observed.killed_owner_frames = sorted(self.executors.owner_kill.frames)
        tools, sessions, problems = self.auditors.observe()
        self.observed.tool_audit, self.observed.session_audit = tools, sessions
        self._problems.extend(problems)

    def _collect_external_handlers(self) -> None:
        self.observed.external_handlers = {
            channel_id: list(handler.calls)
            for channel_id, handler in self.external_handlers.items()
        }
        for handler in self.external_handlers.values():
            self._problems.extend(handler.problems)

    def _collect_acp_permissions(self) -> None:
        self.observed.acp_handlers = {
            channel_id: list(handler.calls) for channel_id, handler in self.acp_handlers.items()
        }
        for handler in self.acp_handlers.values():
            self._problems.extend(handler.problems)
        self.observed.acp_permissions = {
            channel_id: list(agent.permissions.received)
            for channel_id, agent in self.acp_agents.items()
        }

    def _collect_acp_mcp(self) -> None:
        self.observed.acp_mcp_requests = {
            channel_id: list(source.requests) for channel_id, source in self.acp_mcp.items()
        }
        for source in self.acp_mcp.values():
            self._problems.extend(source.problems)

    def _collect_acp_context(self) -> None:
        self.observed.acp_context_requests = {
            channel_id: list(source.requests) for channel_id, source in self.acp_context.items()
        }
        for source in self.acp_context.values():
            self._problems.extend(source.problems)


def _literal_rooms(scenario: Mapping[str, Any]) -> set[str]:
    """The room ids a scenario writes itself; any other room a run creates is aliased."""
    found = {room["id"] for room in scenario.get("setup", {}).get("rooms", [])}

    def walk(value: Any) -> None:
        if isinstance(value, dict):
            for key, item in value.items():
                if key == "room_id" and isinstance(item, str) and not item.startswith("$"):
                    found.add(item)
                if key == "room" and isinstance(item, dict) and isinstance(item.get("id"), str):
                    found.add(item["id"])
                walk(item)
        elif isinstance(value, list):
            for item in value:
                walk(item)

    walk(scenario.get("steps", []))
    walk(scenario.get("setup", {}).get("hooks", []))
    walk(scenario.get("setup", {}).get("channels", []))
    return found


def _context_events(context: RoomContext) -> list[dict[str, Any]]:
    """*context*'s events, lifecycle records left out: what a parse, a
    delivery and a hook read of the room (README, ``contexts``, ``hooks``)."""
    events = [event.model_dump(mode="json") for event in context.recent_events]
    return [event for event in events if not is_lifecycle_record(event)]


def _recent_ids(context: RoomContext) -> list[str]:
    """The ids of *context*'s events, lifecycle records left out."""
    return [event["id"] for event in _context_events(context)]
