"""The turn's response metadata: one live mapping, shared by every writer of the turn.

An AI turn produces one or more MESSAGE events (a text segment per tool round
plus the answer), and the host may want the same turn-level facts on each of
them — which documents were cited, which credential was used, what the turn
read. Those facts are not all known when generation starts: a memory provider
learns some while the context is built, a ``BEFORE_AI_GENERATION`` hook adds
others, and a tool handler finds more in the middle of the loop. So the record
has to be *one* object, alive from the first moment of the turn to its last
event, and every extension point has to write into that same object.

A plain ``dict`` cannot play that part. Pydantic copies a dict field on every
model construction — ``ChannelOutput(response_metadata=d).response_metadata is d``
is ``False`` — and again on assignment where ``validate_assignment`` is on, as
it is for ``AIContext``. The copy taken when the streaming output was built
therefore froze the metadata at stream start, and nothing written during the
tool loop ever reached a persisted segment. This type validates by *identity*:
an instance handed to a model is stored as-is, so the mapping RoomKit reads
when it persists an event is the one the host wrote into.

Where it lives: the turn's loop context creates it; ``_build_context`` hands
the same instance to ``AIContext.response_metadata`` (the hooks' face) and the
generation paths hand it on to ``ChannelOutput.response_metadata`` (the core's
face). Code that holds neither reaches it through
:func:`roomkit.tools.current_response_metadata`.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping, MutableMapping
from typing import Any

from pydantic_core import core_schema


class ResponseMetadata(MutableMapping[str, Any]):
    """Turn-level metadata merged into every MESSAGE event the turn produces.

    Behaves as a ``dict`` for every reader and writer (``[...]``, ``.get``,
    ``.update``, ``**``, ``dict(...)``, ``==``); the only thing it adds is that
    Pydantic keeps the instance instead of copying it, so one turn has one
    record. A bare mapping passed where this type is expected is wrapped — the
    caller's dict is then a snapshot, which is what passing a literal means.

    Each MESSAGE event carries the record *as it stands when the event is
    created*: a streamed segment persisted before a tool round shows what was
    known then, the final answer shows everything the turn learned; the
    non-streaming path builds all its events at the end, so they read alike.
    """

    __slots__ = ("_data",)

    def __init__(self, initial: Mapping[str, Any] | None = None) -> None:
        self._data: dict[str, Any] = dict(initial or {})

    def __getitem__(self, key: str) -> Any:
        return self._data[key]

    def __setitem__(self, key: str, value: Any) -> None:
        self._data[key] = value

    def __delitem__(self, key: str) -> None:
        del self._data[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self._data)

    def __len__(self) -> int:
        return len(self._data)

    def __repr__(self) -> str:
        return f"ResponseMetadata({self._data!r})"

    @classmethod
    def coerce(cls, value: Any) -> ResponseMetadata:
        """The instance itself, or a bare mapping wrapped — anything else is refused."""
        if isinstance(value, cls):
            return value
        if isinstance(value, Mapping):
            return cls(value)
        raise TypeError(f"response metadata must be a mapping, got {type(value).__name__}")

    @classmethod
    def __get_pydantic_core_schema__(cls, _source: Any, _handler: Any) -> core_schema.CoreSchema:
        # Python input: validate by identity (``coerce`` returns the instance
        # it was given). JSON input and the JSON schema: a plain string-keyed
        # object, wrapped on the way in. Serialisation: the dict it wraps.
        dict_schema = core_schema.dict_schema(core_schema.str_schema(), core_schema.any_schema())
        return core_schema.json_or_python_schema(
            json_schema=core_schema.no_info_after_validator_function(cls, dict_schema),
            python_schema=core_schema.no_info_plain_validator_function(cls.coerce),
            serialization=core_schema.plain_serializer_function_ser_schema(dict),
        )
