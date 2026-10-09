"""Every text AI provider over its real SDK, answered by a fake HTTP layer.

The wire drivers of this suite stand behind the SDK, as its objects; a
failure, though, is the SDK's own reading of what the server sent (an error
event in a 200 stream, a gateway's HTML page), so the failure scenarios need
the SDK itself in the path. Each :class:`HttpWire` builds one provider whose
client talks to a mock transport of the HTTP client its SDK runs on (``httpx``,
or ``httpx2`` for Anthropic and Mistral), no network, and answers one way: an
error status, an overload written into a 200 stream, or an HTML page.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass
from typing import Any

import anthropic
import httpx
import httpx2
import ollama
import openai
import polargrid
from google import genai
from mistralai.client import Mistral
from polargrid.client import _parse_error_response

from roomkit.providers.ai.base import AIProvider
from roomkit.providers.anthropic.ai import AnthropicAIProvider
from roomkit.providers.anthropic.config import AnthropicConfig
from roomkit.providers.gemini.ai import GeminiAIProvider
from roomkit.providers.gemini.config import GeminiConfig
from roomkit.providers.mistral.ai import MistralAIProvider
from roomkit.providers.mistral.config import MistralConfig
from roomkit.providers.ollama.ai import OllamaAIProvider
from roomkit.providers.ollama.config import OllamaConfig
from roomkit.providers.polargrid.ai import PolarGridAIProvider
from roomkit.providers.polargrid.config import PolarGridConfig
from tests.text_conformance import openai_wire

OVERLOAD = "The server is overloaded, please try again later"
HTML = b"<html><body><h1>502 Bad Gateway</h1></body></html>"


def sse(lines: list[dict[str, Any]]) -> bytes:
    """Server-sent events, one ``data:`` line per object, no ``[DONE]``."""
    return "".join(f"data: {json.dumps(line)}\n\n" for line in lines).encode()


def _status(status: int, body: dict[str, Any]) -> Callable[[Any], httpx.Response]:
    return lambda request: httpx.Response(status, json=body)


def _stream(content: bytes, content_type: str = "text/event-stream") -> Any:
    return lambda request: httpx.Response(
        200, headers={"content-type": content_type}, content=content
    )


def _html(request: Any) -> httpx.Response:
    return httpx.Response(200, headers={"content-type": "text/html"}, content=HTML)


@dataclass(frozen=True)
class HttpWire:
    """One provider over its SDK, and how its server writes a failure."""

    label: str
    build: Callable[[Callable[[Any], Any]], AIProvider]
    """The provider whose client the handler answers."""
    status_body: Callable[[int], dict[str, Any]]
    """The error body the server sends with an error status."""
    stream_error: Callable[[Any], Any]
    """The handler answering 200 with an overload as the stream's first event."""
    overload_status: int = 503
    """The status the server answers an overload with."""
    stream_names_status: bool = True
    """Whether the stream's error event names a status the SDK keeps."""
    html: Callable[[Any], Any] = _html
    """The handler answering 200 with a gateway's HTML page."""
    lost_statuses: frozenset[int] = frozenset()
    """Statuses the SDK raises without, so nothing can read them."""
    stream_drops_status: bool = False
    """The SDK raises an error written into a stream with no status kept."""

    def empty_stream(self) -> AIProvider:
        """Answered 200 with a stream that carries no event."""
        return self.build(_stream(b"", self.stream_content_type))

    def unnamed_stream_error(self) -> AIProvider:
        """Answered 200, the stream's first event an error naming no status."""
        return self.build(_stream(self.unnamed_error_event, self.stream_content_type))

    @property
    def stream_content_type(self) -> str:
        return "application/x-ndjson" if self.label == "ollama" else "text/event-stream"

    @property
    def unnamed_error_event(self) -> bytes:
        error = {"message": "Failed to generate a reply"}
        if self.label == "ollama":
            return json.dumps({"error": error["message"]}).encode()
        if self.label == "anthropic":
            body = {"type": "error", "error": error}
            return _anthropic_events([_MESSAGE_START, ("error", body)])
        return sse([{"error": error}])

    def status(self, status: int) -> AIProvider:
        return self.build(_status(status, self.status_body(status)))


# -- OpenAI wire (OpenAI and its ten derivatives) ----------------------------------


def _openai(build: Callable[[], AIProvider]) -> Callable[[Callable[[Any], Any]], AIProvider]:
    def provider(handler: Callable[[Any], Any]) -> AIProvider:
        built = build()
        http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        built._client = openai.AsyncOpenAI(  # type: ignore[attr-defined]
            api_key="k", base_url="http://wire.test/v1", http_client=http, max_retries=0
        )
        return built

    return provider


def _openai_wires() -> list[HttpWire]:
    # The in-stream form, as OpenRouter and the OpenAI-compatible servers
    # write it: the error object on a data line, the HTTP status 200.
    event = sse([{"error": {"message": OVERLOAD, "type": "server_error", "code": 503}}])
    return [
        HttpWire(
            label=wire.label,
            build=_openai(wire._build),
            status_body=lambda s: {
                "error": {"message": f"status {s}", "type": "server_error", "code": None}
            },
            stream_error=_stream(event),
        )
        for wire in openai_wire.wires()
    ]


# -- Anthropic (httpx2) --------------------------------------------------------------


def _httpx2_transport(handler: Callable[[Any], Any]) -> httpx2.MockTransport:
    """An httpx2 transport answering what the handler's httpx response says, for
    the SDKs that run on httpx2 (Anthropic, Mistral)."""

    def answer(request: Any) -> Any:
        response = handler(request)
        return httpx2.Response(
            response.status_code, headers=dict(response.headers), content=response.content
        )

    return httpx2.MockTransport(answer)


def _anthropic(handler: Callable[[Any], Any]) -> AIProvider:
    provider = AnthropicAIProvider(AnthropicConfig(api_key="k", model="claude-sonnet-5-5"))
    provider._client = anthropic.AsyncAnthropic(
        api_key="k",
        max_retries=0,
        http_client=anthropic.DefaultAsyncHttpxClient(transport=_httpx2_transport(handler)),
    )
    return provider


def _anthropic_events(events: list[tuple[str, dict[str, Any]]]) -> bytes:
    return "".join(
        f"event: {kind}\ndata: {json.dumps(data)}\n\n" for kind, data in events
    ).encode()


_MESSAGE_START = (
    "message_start",
    {
        "type": "message_start",
        "message": {
            "id": "msg_1",
            "type": "message",
            "role": "assistant",
            "model": "served-model",
            "content": [],
            "stop_reason": None,
            "stop_sequence": None,
            "usage": {"input_tokens": 1, "output_tokens": 1},
        },
    },
)
_OVERLOADED = {"type": "error", "error": {"type": "overloaded_error", "message": "Overloaded"}}


# -- Gemini, Mistral, Ollama ------------------------------------------------------------


def _gemini(handler: Callable[[Any], Any]) -> AIProvider:
    provider = GeminiAIProvider(GeminiConfig(api_key="k"))
    http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    provider._client = genai.Client(
        api_key="k", http_options=genai.types.HttpOptions(httpx_async_client=http)
    )
    provider._http = http
    return provider


def _mistral(handler: Callable[[Any], Any]) -> AIProvider:
    """mistralai 3.x runs on httpx2, as Anthropic's SDK does."""
    provider = MistralAIProvider(MistralConfig(api_key="k", model="mistral-large-latest"))
    http = httpx2.AsyncClient(transport=_httpx2_transport(handler))
    provider._client = Mistral(api_key="k", async_client=http)
    return provider


def _ollama(handler: Callable[[Any], Any]) -> AIProvider:
    provider = OllamaAIProvider(OllamaConfig(model="qwen3:8b"))
    provider._client = ollama.AsyncClient(
        host="http://ollama.test", transport=httpx.MockTransport(handler)
    )
    return provider


# -- PolarGrid (the SDK builds an httpx client per call: faked at its two seams,
# raising what the SDK's own code raises for each answer) ---------------------------


def _polargrid(handler: Callable[[Any], Any]) -> AIProvider:
    def read(response: httpx.Response) -> Any:
        if response.status_code >= 400:
            raise _parse_error_response(response.status_code, response.text, "req_x")
        return response

    async def make_request(endpoint: str, method: str = "GET", body: Any = None, **_: Any) -> Any:
        return json.loads(read(handler(None)).content)  # what response.json() does

    async def stream_post(endpoint: str, body: Any) -> AsyncIterator[dict[str, Any]]:
        response = read(handler(None))
        for line in response.content.decode().splitlines():
            if line.startswith("data: "):
                yield json.loads(line[len("data: ") :])

    provider = PolarGridAIProvider(PolarGridConfig(api_key="k", model="qwen-3.8-27b"))
    client = polargrid.PolarGrid(api_key="k", base_url="http://127.0.0.1:1")
    client._make_request = make_request  # type: ignore[method-assign]
    client._stream_post = stream_post  # type: ignore[method-assign]
    provider._client = client
    return provider


def http_wires() -> list[HttpWire]:
    """Every text AI provider's wire, the OpenAI family's eleven included."""
    return [
        *_openai_wires(),
        HttpWire(
            label="anthropic",
            build=_anthropic,
            status_body=lambda s: {
                "type": "error",
                "error": {"type": "api_error", "message": f"status {s}"},
            },
            stream_error=_stream(_anthropic_events([_MESSAGE_START, ("error", _OVERLOADED)])),
            overload_status=529,
        ),
        HttpWire(
            label="gemini",
            build=_gemini,
            status_body=lambda s: {"error": {"code": s, "message": f"status {s}", "status": "X"}},
            stream_error=_stream(
                sse([{"error": {"code": 503, "message": OVERLOAD, "status": "UNAVAILABLE"}}])
            ),
        ),
        HttpWire(
            label="mistral",
            build=_mistral,
            status_body=lambda s: {"message": f"status {s}"},
            stream_error=_stream(
                sse([{"error": {"message": OVERLOAD, "type": "server_error", "code": 503}}])
            ),
        ),
        HttpWire(
            label="ollama",
            build=_ollama,
            status_body=lambda s: {"error": f"status {s}"},
            stream_error=_stream(
                ("\n".join([json.dumps({"error": OVERLOAD})])).encode(), "application/x-ndjson"
            ),
            stream_names_status=False,
        ),
        HttpWire(
            label="polargrid",
            build=_polargrid,
            status_body=lambda s: {"message": f"status {s}", "code": "x"},
            stream_error=_stream(sse([{"error": {"message": OVERLOAD, "code": 503}}])),
            stream_names_status=False,
            # polargrid-sdk raises a bare PolarGridError, no status kept, for
            # any status it does not classify (408 and 409 among them), and an
            # error written into a stream as a NetworkError with no status.
            lost_statuses=frozenset({408, 409}),
            stream_drops_status=True,
        ),
    ]
