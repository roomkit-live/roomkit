"""MCP as RoomKit reads it on the wire (RFC §21.2).

The MCP SDK's models keep only the fields they model and check the handshake
more strictly than RoomKit does. Where the RFC says a host keeps what the
server sent, RoomKit reads the raw result (:mod:`roomkit.tools._mcp_capture`)
by these rules and gives the SDK only a copy it can parse, as roomkit-rs does
(``roomkit-mcp``: ``raw.rs``, ``protocol.rs``, ``resources.rs``). The SDK
still carries the transports, the request ids, the deadlines and the
cancellations.
"""

from __future__ import annotations

import copy
from collections.abc import Mapping
from typing import Any

from pydantic import AnyUrl, BaseModel, ConfigDict, Field, TypeAdapter, model_validator

RAW_RESULT: TypeAdapter[dict[str, Any]] = TypeAdapter(dict[str, Any])
"""A result read whole, as the server sent it."""

_URI: TypeAdapter[AnyUrl] = TypeAdapter(AnyUrl)


class _WithMeta(BaseModel):
    """What RoomKit checks of a host request's parameters and of its result:
    a ``_meta``, when there is one, is an object (RK-21.2-046). Every other
    member is the caller's, or the server's, own."""

    model_config = ConfigDict(extra="allow")

    meta: dict[str, Any] | None = Field(default=None, alias="_meta")


class _Content(BaseModel):
    """One content of a ``resources/read`` result: a URI, and a text or a blob."""

    model_config = ConfigDict(extra="allow")

    uri: str
    mime_type: str | None = Field(default=None, alias="mimeType")
    meta: dict[str, Any] | None = Field(default=None, alias="_meta")
    text: Any = None
    blob: Any = None

    @model_validator(mode="after")
    def _text_or_blob(self) -> _Content:
        # The model's union: a text or a blob; the other member, when present
        # too, is an ordinary extension of the content.
        if not isinstance(self.text, str) and not isinstance(self.blob, str):
            raise ValueError("a resource content carries a text or a blob")
        return self


class _Resource(BaseModel):
    model_config = ConfigDict(extra="allow")

    contents: list[_Content]
    meta: dict[str, Any] | None = Field(default=None, alias="_meta")


def host_params(params: Mapping[str, Any] | None) -> dict[str, Any]:
    """The parameters of a host request, as the caller wrote them.

    The SDK sends an empty set as no ``params`` member at all, and an empty
    or ``None`` ``_meta`` as none (RK-21.2-046).

    Raises:
        pydantic.ValidationError: Its ``_meta`` is not an object.
    """
    sent = dict(params or {})
    _WithMeta.model_validate(sent)
    return sent


def host_result(result: dict[str, Any]) -> dict[str, Any]:
    """A host request's result as the server sent it, but for its members
    sent as ``null``; nulls nested in a member are kept.

    Raises:
        pydantic.ValidationError: Its ``_meta`` is not an object.
    """
    _WithMeta.model_validate(result)
    return {name: value for name, value in result.items() if value is not None}


def resource_uri(uri: str) -> str:
    """*uri* as RoomKit sends it: parsed and normalized, the host lowercased
    and a space percent-encoded (``https://EXAMPLE.com/a b`` reads
    ``https://example.com/a%20b``).

    Raises:
        pydantic.ValidationError: *uri* is not an absolute URI.
    """
    return str(_URI.validate_python(uri))


def resource_result(result: dict[str, Any]) -> dict[str, Any]:
    """A ``resources/read`` result as the server sent it, extension fields and
    explicit nulls kept, once its shape is checked: contents that each carry
    a URI and a text or a blob. Each content's URI is normalized as
    :func:`resource_uri` does.

    Raises:
        pydantic.ValidationError: The result is not a resource read's.
    """
    parsed = _Resource.model_validate(result)
    for raw, content in zip(result["contents"], parsed.contents, strict=True):
        raw["uri"] = resource_uri(content.uri)
    return result


def compatible_handshake(result: Mapping[str, Any]) -> dict[str, Any]:
    """A copy of an ``initialize`` or ``server/discover`` result that the
    SDK's model accepts where
    RoomKit's reading is laxer: extension settings that are not objects, and
    a ``serverInfo`` ``description`` or icon ``theme`` of any shape, are left
    out of it. The result itself is kept whole."""
    parsable = copy.deepcopy(dict(result))
    capabilities = parsable.get("capabilities")
    if isinstance(capabilities, dict):
        extensions = capabilities.get("extensions")
        if isinstance(extensions, dict):
            capabilities["extensions"] = {
                name: settings
                for name, settings in extensions.items()
                if isinstance(settings, dict)
            }
        else:
            capabilities.pop("extensions", None)
    info = parsable.get("serverInfo")
    if isinstance(info, dict):
        info.pop("description", None)
        icons = info.get("icons")
        if isinstance(icons, list):
            for icon in icons:
                if isinstance(icon, dict):
                    icon.pop("theme", None)
    return parsable


SERVER_INFO_META_KEY = "io.modelcontextprotocol/serverInfo"
"""Where a ``server/discover`` result (MCP 2026-07-28) says what the server
is: its ``_meta``, where an ``initialize`` result has a ``serverInfo``."""


def server_info(handshake: Mapping[str, Any]) -> dict[str, Any]:
    """What the server said of itself in its handshake or its discovery
    result, as it sent it, but for the members it sent as ``null`` (in it and
    in each of its icons)."""
    if "supportedVersions" in handshake:
        # A discovery result: what it says of the server is in its _meta.
        meta = handshake.get("_meta")
        info = meta.get(SERVER_INFO_META_KEY) if isinstance(meta, dict) else None
    else:
        info = handshake.get("serverInfo")
    if not isinstance(info, dict):
        return {}
    kept = {name: copy.deepcopy(value) for name, value in info.items() if value is not None}
    icons = kept.get("icons")
    if isinstance(icons, list):
        kept["icons"] = [
            {name: value for name, value in icon.items() if value is not None}
            if isinstance(icon, dict)
            else icon
            for icon in icons
        ]
    return kept


def server_extensions(handshake: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    """The extensions the server declared in its handshake whose settings are
    an object, by identifier, their settings as it wrote them."""
    capabilities = handshake.get("capabilities")
    declared = capabilities.get("extensions") if isinstance(capabilities, dict) else None
    if not isinstance(declared, dict):
        return {}
    return {
        str(name): copy.deepcopy(settings)
        for name, settings in declared.items()
        if isinstance(settings, dict)
    }
