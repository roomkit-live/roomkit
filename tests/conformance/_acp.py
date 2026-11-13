"""The scripted ACP agent (roomkit-specs conformance/README.md, "Scripted ACP
agents"): served in this process over a socket pair, written at the wire's
level, JSON-RPC one message per line, so that it shares none of the
channel's ACP code."""

from __future__ import annotations

import asyncio
import contextlib
import json
import socket
from collections.abc import Mapping
from typing import Any

import acp

from roomkit import ACPChannel, ACPSessionInvalidatedError, ACPTransport
from tests.conformance._acp_context import ScriptedContextContributor
from tests.conformance._acp_handler import ScriptedExternalHandler
from tests.conformance._acp_mcp import ScriptedMCPServers
from tests.conformance._acp_permissions import PermissionExchanges
from tests.conformance._script import Gates

AGENT = {"initialize", "prompts", "new_sessions", "set_config_options", "close_sessions"}
INITIALIZE = {"protocol_version", "closes_sessions", "agent_info", "mcp_capabilities"}
PROMPT = {
    "gate",
    "updates",
    "after_updates_gate",
    "permission",
    "stop_reason",
    "usage",
    "error",
    "permission_steps",
}
SESSION_REPLY = {"gate", "result", "error"}
REFUSAL = {"reason", "recovery_authorized", "after_prompt"}
SESSION_METHODS = {
    "session/new": "new_sessions",
    "session/set_config_option": "set_config_options",
    "session/close": "close_sessions",
}


class ScriptedACPAgent:
    """An ACP agent answering as its script says, recording what it receives."""

    def __init__(self, spec: Mapping[str, Any], gates: Gates) -> None:
        self._spec = spec
        self._gates = gates
        self.received: list[dict[str, Any]] = []
        self.permissions = PermissionExchanges()
        self.attempts: list[dict[str, Any]] = []
        self._sessions = 0
        self._prompts = 0
        self._reply_counts: dict[str, int] = {}
        self._answering: set[asyncio.Task[None]] = set()

    async def serve(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        """Read the client's messages to the end of its stream; each request
        is answered by a task of its own, so that a prompt held at its gate
        does not stop the reading."""
        try:
            while line := await reader.readline():
                message = json.loads(line)
                method = message.get("method")
                if method is None:
                    self.permissions.accept(message)
                    continue
                self.received.append({"method": method, "params": message.get("params")})
                if "id" not in message:
                    continue
                prompt = None
                if method == "session/prompt":
                    prompt = self._next_prompt()
                elif method in SESSION_METHODS:
                    kind = SESSION_METHODS[method]
                    index = self._reply_counts.get(kind, 0)
                    self._reply_counts[kind] = index + 1
                    replies = self._spec.get(kind, [])
                    prompt = replies[index] if index < len(replies) else {}
                task = asyncio.create_task(self._answer(message, prompt, writer))
                self._answering.add(task)
                task.add_done_callback(self._answering.discard)
        finally:
            answering = list(self._answering)
            for task in answering:
                task.cancel()
            await asyncio.gather(*answering, return_exceptions=True)
            writer.close()

    def _next_prompt(self) -> Mapping[str, Any]:
        prompts = self._spec.get("prompts", [])
        index = self._prompts
        self._prompts += 1
        return prompts[index] if index < len(prompts) else {}

    async def _answer(
        self,
        message: Mapping[str, Any],
        prompt: Mapping[str, Any] | None,
        writer: asyncio.StreamWriter,
    ) -> None:
        method = message["method"]
        params = message.get("params") or {}
        error: Mapping[str, Any] | None = None
        result: Any = None
        if method == "initialize":
            result = self._initialized(params)
        elif method in SESSION_METHODS:
            result, error = await self._session_reply(method, prompt or {})
        elif method == "session/prompt" and prompt is not None:
            result, error = await self._prompt_reply(message, prompt, writer)
        else:
            error = {"code": -32601, "message": "Method not found", "data": {"method": method}}
        answer: dict[str, Any] = {"jsonrpc": "2.0", "id": message["id"]}
        if error is not None:
            answer["error"] = dict(error)
        else:
            answer["result"] = result
        _send(writer, answer)

    async def _prompt_reply(
        self, message: Mapping[str, Any], prompt: Mapping[str, Any], writer: asyncio.StreamWriter
    ) -> tuple[dict[str, Any] | None, Mapping[str, Any] | None]:
        session = message["params"]["sessionId"]
        if "gate" in prompt:
            await self._gates.hold(prompt["gate"])
        self._updates(session, prompt.get("updates", []), writer)
        await self._after_updates(prompt)
        if "permission" in prompt:
            _send(
                writer,
                {
                    "jsonrpc": "2.0",
                    "id": f"permission-{message['id']}",
                    "method": "session/request_permission",
                    "params": {**prompt["permission"], "sessionId": session},
                },
            )
        for step in prompt.get("permission_steps", []):
            await self.permissions.exchange(session, step["request"], writer)
            self._updates(session, step.get("updates", []), writer)
        if "error" in prompt:
            return None, prompt["error"]
        result = {"stopReason": prompt.get("stop_reason", "end_turn")}
        if "usage" in prompt:
            result["usage"] = prompt["usage"]
        return result, None

    @staticmethod
    def _updates(session: str, updates: list[Any], writer: asyncio.StreamWriter) -> None:
        for update in updates:
            _send(
                writer,
                {
                    "jsonrpc": "2.0",
                    "method": "session/update",
                    "params": {"sessionId": session, "update": update},
                },
            )

    async def _after_updates(self, prompt: Mapping[str, Any]) -> None:
        """Keep the prompt active after its updates have reached the pipe."""
        if "after_updates_gate" in prompt:
            await self._gates.hold(prompt["after_updates_gate"])

    async def _session_reply(
        self, method: str, reply: Mapping[str, Any]
    ) -> tuple[dict[str, Any], Mapping[str, Any] | None]:
        result: dict[str, Any] = {}
        if method == "session/new":
            self._sessions += 1
            result["sessionId"] = f"session-{self._sessions}"
        elif method == "session/set_config_option":
            result["configOptions"] = []
        if "gate" in reply:
            await self._gates.hold(reply["gate"])
        result.update(reply.get("result", {}))
        return result, reply.get("error")

    def _initialized(self, params: Mapping[str, Any]) -> dict[str, Any]:
        spec = self._spec.get("initialize", {})
        capabilities: dict[str, Any] = {}
        if spec.get("closes_sessions", True):
            capabilities["sessionCapabilities"] = {"close": {}}
        if "mcp_capabilities" in spec:
            capabilities["mcpCapabilities"] = spec["mcp_capabilities"]
        result: dict[str, Any] = {
            "protocolVersion": spec.get("protocol_version", params.get("protocolVersion")),
            "agentCapabilities": capabilities,
        }
        if "agent_info" in spec:
            result["agentInfo"] = spec["agent_info"]
        return result


def _send(writer: asyncio.StreamWriter, message: Mapping[str, Any]) -> None:
    writer.write(json.dumps(message).encode() + b"\n")


class ScriptedACPTransport(ACPTransport):
    """The transport that hands the channel a connection to its scripted agent."""

    def __init__(self, agent: ScriptedACPAgent, spec: Mapping[str, Any]) -> None:
        self._agent = agent
        self._refusals = iter(spec.get("refuse_prompt", []))
        self._serving: asyncio.Task[None] | None = None
        self._writer: asyncio.StreamWriter | None = None
        self._connection: Any = None

    @property
    def name(self) -> str:
        return "scripted"

    def is_alive(self) -> bool:
        return (
            self._writer is not None
            and not self._writer.is_closing()
            and self._serving is not None
            and not self._serving.done()
        )

    async def open(self, client: Any, *, queue: Any) -> Any:
        ours, theirs = socket.socketpair()
        reader, writer = await asyncio.open_connection(sock=ours)
        agent_reader, agent_writer = await asyncio.open_connection(sock=theirs)
        self._writer = writer
        self._serving = asyncio.create_task(self._agent.serve(agent_reader, agent_writer))
        kwargs = {} if queue is None else {"queue": queue}
        connection = acp.connect_to_agent(client, writer, reader, **kwargs)
        self._connection = connection
        return _RefusingConnection(connection, self._agent, self._refusals)

    async def close(self) -> None:
        connection, self._connection = self._connection, None
        if connection is not None:
            await connection.close()
        writer, self._writer = self._writer, None
        if writer is not None:
            writer.close()
        serving, self._serving = self._serving, None
        if serving is not None:
            serving.cancel()
            with contextlib.suppress(BaseException):
                await serving


def scripted_acp_channel(
    channel: Mapping[str, Any], gates: Gates
) -> tuple[
    ACPChannel,
    ScriptedACPAgent,
    ScriptedACPTransport,
    ScriptedMCPServers,
    ScriptedExternalHandler | None,
    ScriptedContextContributor | None,
]:
    """The channel a scenario's `acp` member describes, and its agent."""
    spec = channel["acp"]
    handler = (
        ScriptedExternalHandler(spec["external_tool_handler"])
        if "external_tool_handler" in spec
        else None
    )
    context = (
        ScriptedContextContributor(spec["context_contributor"], gates)
        if "context_contributor" in spec
        else None
    )
    mcp = ScriptedMCPServers(spec.get("mcp_servers", []), gates)
    agent = ScriptedACPAgent(spec["agent"], gates)
    transport = ScriptedACPTransport(agent, spec.get("transport", {}))
    return (
        ACPChannel(
            channel["id"],
            transport=transport,
            cwd=spec["cwd"],
            additional_directories=spec.get("additional_directories"),
            mcp_servers=mcp.source(),
            session_config=spec.get("session_config"),
            publish_thinking_text=spec.get("publish_thinking_text", False),
            external_tool_handler=handler,
            room_history=spec.get("room_history", 20),
            context_contributor=context,
            instructions=spec.get("instructions"),
            name=spec.get("name"),
            role=spec.get("role"),
            description=spec.get("description"),
        ),
        agent,
        transport,
        mcp,
        handler,
        context,
    )


class _RefusingConnection:
    """A host refusal is a native exception, never an agent's wire error."""

    def __init__(self, connection: Any, agent: ScriptedACPAgent, refusals: Any) -> None:
        self._connection = connection
        self._agent = agent
        self._refusals = refusals

    def __getattr__(self, name: str) -> Any:
        return getattr(self._connection, name)

    async def prompt(self, session_id: str, prompt: Any, **kwargs: Any) -> Any:
        self._agent.attempts.append(
            {
                "session_id": session_id,
                "event_id": kwargs.get("roomkit.live/eventId"),
                "text": "".join(getattr(block, "text", "") for block in prompt),
            }
        )
        refusal = next(self._refusals, None)
        if refusal is None or refusal.get("after_prompt", False):
            result = await self._connection.prompt(session_id, prompt, **kwargs)
            if refusal is None:
                return result
        raise ACPSessionInvalidatedError(
            refusal["reason"], recovery_authorized=refusal.get("recovery_authorized", False)
        )
