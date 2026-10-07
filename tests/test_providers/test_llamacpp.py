"""Tests for the llama.cpp provider: build selection, download checks, server lifecycle."""

from __future__ import annotations

import hashlib
import io
import socket
import stat
import sys
import tarfile
import textwrap
import zipfile
from pathlib import Path

import pytest
from pydantic import ValidationError

from roomkit.providers.ai.base import AIContext, AIMessage, AITool, ProviderError
from roomkit.providers.ai.response_schema import ResponseSchemaError
from roomkit.providers.llamacpp import LlamaCppAIProvider, LlamaCppConfig
from roomkit.providers.llamacpp import binary as binary_mod

pytest.importorskip("openai")
httpx = pytest.importorskip("httpx")

# ---------------------------------------------------------------------------
# A stand-in llama-server: /health, then one chat completion that calls a tool
# ---------------------------------------------------------------------------

_FAKE_SERVER = textwrap.dedent(
    """\
    #!{python}
    import json, os, sys
    from http.server import BaseHTTPRequestHandler, HTTPServer

    args = sys.argv[1:]
    port = int(args[args.index("--port") + 1])
    starts = os.environ.get("FAKE_STARTS_FILE")
    if starts:
        with open(starts, "a") as f:
            f.write(" ".join(args) + "\\n")
    if os.environ.get("FAKE_FAIL"):
        print("error: failed to load model 'nope'", flush=True)
        sys.exit(3)
    if os.environ.get("FAKE_HANG"):
        import time
        time.sleep(3600)

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def _json(self, body):
            data = json.dumps(body).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            self._json({{"status": "ok"}})

        def do_POST(self):
            request = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            if request.get("stream"):
                chunk = {{
                    "id": "c1", "object": "chat.completion.chunk", "created": 0, "model": "m",
                    "choices": [{{"index": 0, "delta": {{"content": "ok"}},
                                 "finish_reason": "stop"}}],
                }}
                data = ("data: " + json.dumps(chunk) + "\\n\\ndata: [DONE]\\n\\n").encode()
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)
                return
            call = {{
                "id": "call_1", "type": "function",
                "function": {{"name": request["tools"][0]["function"]["name"],
                              "arguments": json.dumps({{"city": "Montréal"}})}},
            }}
            self._json({{
                "id": "c1", "object": "chat.completion", "created": 0, "model": "m",
                "choices": [{{"index": 0, "finish_reason": "tool_calls",
                             "message": {{"role": "assistant", "content": None,
                                         "tool_calls": [call]}}}}],
                "usage": {{"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}},
            }})

    print("server is listening", flush=True)
    HTTPServer(("127.0.0.1", port), Handler).serve_forever()
    """
)


@pytest.fixture
def fake_server(tmp_path: Path) -> Path:
    path = tmp_path / "llama-server"
    path.write_text(_FAKE_SERVER.format(python=sys.executable))
    path.chmod(path.stat().st_mode | stat.S_IXUSR)
    return path


_WEATHER = AITool(
    name="get_weather",
    description="Current weather for a city",
    parameters={"type": "object", "properties": {"city": {"type": "string"}}},
)


def _ask() -> AIContext:
    return AIContext(messages=[AIMessage(role="user", content="Weather?")], tools=[_WEATHER])


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


class TestLlamaCppConfig:
    def test_model_is_the_only_required_field(self) -> None:
        config = LlamaCppConfig(model="org/repo-GGUF:Q4_K_M")
        assert config.context_size == 8192
        assert config.gpu_layers is None
        assert config.binary is None

        with pytest.raises(ValidationError):
            LlamaCppConfig()  # type: ignore[call-arg]

    def test_blank_model_is_refused(self) -> None:
        with pytest.raises(ValidationError, match="GGUF"):
            LlamaCppConfig(model="  ")

    def test_unknown_variant_is_refused(self) -> None:
        with pytest.raises(ValidationError, match="unknown variant"):
            LlamaCppConfig(model="m", variant="linux-x64-quantum")


# ---------------------------------------------------------------------------
# Which build runs here
# ---------------------------------------------------------------------------


class TestDetectVariant:
    @pytest.mark.parametrize(
        ("system", "machine", "cuda", "expected"),
        [
            ("Linux", "x86_64", 13, "linux-x64-cuda-13"),
            ("Linux", "x86_64", 12, "linux-x64-cuda-12"),
            ("Linux", "x86_64", 11, "linux-x64-cpu"),
            ("Linux", "x86_64", None, "linux-x64-cpu"),
            ("Linux", "aarch64", 13, "linux-arm64-cuda-13"),
            ("Linux", "aarch64", 12, "linux-arm64-cpu"),  # no CUDA 12 arm64 build
            ("Darwin", "arm64", None, "macos-arm64"),
            ("Windows", "AMD64", 13, "windows-x64-cuda-13"),
            ("Windows", "AMD64", None, "windows-x64-cpu"),
        ],
    )
    def test_picks_the_build_for_the_os_cpu_and_driver(
        self, monkeypatch: pytest.MonkeyPatch, system, machine, cuda, expected
    ) -> None:
        monkeypatch.setattr(binary_mod.platform, "system", lambda: system)
        monkeypatch.setattr(binary_mod.platform, "machine", lambda: machine)
        monkeypatch.setattr(binary_mod, "cuda_driver_major", lambda: cuda)

        assert binary_mod.detect_variant() == expected

    def test_an_unknown_cpu_names_the_way_out(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(binary_mod.platform, "system", lambda: "Linux")
        monkeypatch.setattr(binary_mod.platform, "machine", lambda: "riscv64")

        with pytest.raises(ProviderError, match="binary="):
            binary_mod.detect_variant()

    def test_reads_the_driver_cuda_version(self, monkeypatch: pytest.MonkeyPatch) -> None:
        out = "| NVIDIA-SMI 580.95   Driver Version: 580.95   CUDA Version: 13.0     |"
        monkeypatch.setattr(binary_mod.shutil, "which", lambda name: "/usr/bin/nvidia-smi")
        monkeypatch.setattr(
            binary_mod.subprocess,
            "run",
            lambda *a, **k: binary_mod.subprocess.CompletedProcess(a, 0, stdout=out),
        )
        assert binary_mod.cuda_driver_major() == 13

    def test_no_driver_means_no_cuda(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(binary_mod.shutil, "which", lambda name: None)
        assert binary_mod.cuda_driver_major() is None


# ---------------------------------------------------------------------------
# Resolution and download
# ---------------------------------------------------------------------------


def _tar_with_server(name: str = "llama-b0-bin/llama-server") -> bytes:
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tf:
        for member, data in ((name, b"#!/bin/sh\n"), ("llama-b0-bin/libggml.so", b"")):
            info = tarfile.TarInfo(member)
            info.size = len(data)
            info.mode = 0o755
            tf.addfile(info, io.BytesIO(data))
    return buf.getvalue()


class _FakeStream:
    def __init__(self, payload: bytes) -> None:
        self._payload = payload

    def __enter__(self) -> _FakeStream:
        return self

    def __exit__(self, *exc: object) -> None:
        return None

    def raise_for_status(self) -> None:
        return None

    def iter_bytes(self, size: int):
        yield self._payload


def _serve(monkeypatch: pytest.MonkeyPatch, payloads: dict[str, bytes], urls: list[str]) -> None:
    def stream(method: str, url: str, **kwargs: object) -> _FakeStream:
        urls.append(url)
        return _FakeStream(payloads[url.rsplit("/", 1)[1]])

    monkeypatch.setattr(httpx, "stream", stream)


class TestResolveBinary:
    def test_an_explicit_path_is_used_as_is(self, fake_server: Path) -> None:
        found = binary_mod.resolve_binary(binary=str(fake_server), variant=None, cache_dir=None)
        assert found.executable == fake_server

    def test_an_explicit_name_is_looked_up_on_the_path(
        self, monkeypatch: pytest.MonkeyPatch, fake_server: Path
    ) -> None:
        monkeypatch.setenv("PATH", str(fake_server.parent))
        found = binary_mod.resolve_binary(binary="llama-server", variant=None, cache_dir=None)
        assert found.executable == fake_server

    def test_a_missing_binary_is_an_error(self, tmp_path: Path) -> None:
        with pytest.raises(ProviderError, match="not found"):
            binary_mod.resolve_binary(binary=str(tmp_path / "nope"), variant=None, cache_dir=None)

    def test_a_llama_server_on_the_path_is_not_picked_up_by_itself(
        self, monkeypatch: pytest.MonkeyPatch, fake_server: Path, tmp_path: Path
    ) -> None:
        # A stray llama-server may be old, CPU-only or a broken wrapper.
        monkeypatch.setenv("PATH", str(fake_server.parent))
        payload = _tar_with_server()
        name = "llama-b0-bin-test.tar.gz"
        monkeypatch.setitem(
            binary_mod.ASSETS, "linux-x64-cpu", ((name, hashlib.sha256(payload).hexdigest()),)
        )
        urls: list[str] = []
        _serve(monkeypatch, {name: payload}, urls)

        found = binary_mod.resolve_binary(
            binary=None, variant="linux-x64-cpu", cache_dir=str(tmp_path / "cache")
        )

        assert found.executable != fake_server
        assert found.executable.name == "llama-server"
        assert found.executable.parent in found.library_dirs
        assert len(urls) == 1

    def test_the_download_is_verified_and_cached(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        payload = _tar_with_server()
        name = "llama-b0-bin-test.tar.gz"
        monkeypatch.setitem(
            binary_mod.ASSETS, "linux-x64-cpu", ((name, hashlib.sha256(payload).hexdigest()),)
        )
        urls: list[str] = []
        _serve(monkeypatch, {name: payload}, urls)
        cache = str(tmp_path / "cache")

        first = binary_mod.resolve_binary(binary=None, variant="linux-x64-cpu", cache_dir=cache)
        second = binary_mod.resolve_binary(binary=None, variant="linux-x64-cpu", cache_dir=cache)

        assert first == second
        assert len(urls) == 1  # the second resolution reads the cache
        assert f"/download/{binary_mod.BUILD}/{name}" in urls[0]

    def test_a_tampered_download_is_refused_and_not_installed(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        name = "llama-b0-bin-test.tar.gz"
        monkeypatch.setitem(binary_mod.ASSETS, "linux-x64-cpu", ((name, "0" * 64),))
        _serve(monkeypatch, {name: _tar_with_server()}, [])
        cache = tmp_path / "cache"

        with pytest.raises(ProviderError, match="does not match the pinned"):
            binary_mod.resolve_binary(binary=None, variant="linux-x64-cpu", cache_dir=str(cache))

        assert not (cache / binary_mod.BUILD / "linux-x64-cpu").exists()

    def test_a_build_installed_concurrently_is_used(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        # Two providers starting on a fresh cache: the second finds the first's
        # install in place when it renames its own, and keeps it.
        payload = _tar_with_server()
        name = "llama-b0-bin-test.tar.gz"
        monkeypatch.setitem(
            binary_mod.ASSETS, "linux-x64-cpu", ((name, hashlib.sha256(payload).hexdigest()),)
        )
        _serve(monkeypatch, {name: payload}, [])
        target = tmp_path / "cache" / binary_mod.BUILD / "linux-x64-cpu"
        binary_mod._download_build(target, "linux-x64-cpu")
        first = binary_mod._locate(target)

        binary_mod._download_build(target, "linux-x64-cpu")  # loses the race

        assert binary_mod._locate(target) == first

    def test_a_zip_member_escaping_the_archive_is_refused(self, tmp_path: Path) -> None:
        archive = tmp_path / "evil.zip"
        with zipfile.ZipFile(archive, "w") as zf:
            zf.writestr("../../outside", b"x")
        dest = tmp_path / "dest"
        dest.mkdir()

        with pytest.raises(ProviderError, match="escapes"):
            binary_mod._extract(archive, dest)
        assert not (tmp_path / "outside").exists()


# ---------------------------------------------------------------------------
# The provider owns the server
# ---------------------------------------------------------------------------


class TestLlamaCppAIProvider:
    async def test_first_request_starts_the_server_once_and_reads_the_tool_call(
        self, monkeypatch: pytest.MonkeyPatch, fake_server: Path, tmp_path: Path
    ) -> None:
        starts = tmp_path / "starts"
        monkeypatch.setenv("FAKE_STARTS_FILE", str(starts))
        config = LlamaCppConfig(model="org/repo-GGUF:Q4_K_M", binary=str(fake_server))
        ai = LlamaCppAIProvider(config)
        try:
            first = await ai.generate(_ask())
            await ai.generate(_ask())
        finally:
            await ai.close()

        assert [(c.name, c.arguments) for c in first.tool_calls] == [
            ("get_weather", {"city": "Montréal"})
        ]
        launches = starts.read_text().splitlines()
        assert len(launches) == 1
        assert "-hf org/repo-GGUF:Q4_K_M" in launches[0]
        assert "--jinja" in launches[0]
        assert "-ngl" not in launches[0]  # llama.cpp fits the GPU itself

    async def test_a_local_gguf_is_passed_as_a_file(
        self, monkeypatch: pytest.MonkeyPatch, fake_server: Path, tmp_path: Path
    ) -> None:
        starts = tmp_path / "starts"
        monkeypatch.setenv("FAKE_STARTS_FILE", str(starts))
        gguf = tmp_path / "model.gguf"
        gguf.write_bytes(b"")
        ai = LlamaCppAIProvider(
            LlamaCppConfig(model=str(gguf), binary=str(fake_server), gpu_layers=0)
        )
        try:
            await ai.start()
        finally:
            await ai.close()

        launch = starts.read_text()
        assert f"-m {gguf}" in launch
        assert "-ngl 0" in launch

    async def test_a_streamed_reply_starts_the_server_too(self, fake_server: Path) -> None:
        ai = LlamaCppAIProvider(LlamaCppConfig(model="m:Q4", binary=str(fake_server)))
        try:
            events = [e async for e in ai.generate_structured_stream(_ask())]
        finally:
            await ai.close()

        assert events
        assert not ai._server.running

    async def test_a_server_that_never_answers_times_out_and_is_stopped(
        self, monkeypatch: pytest.MonkeyPatch, fake_server: Path
    ) -> None:
        monkeypatch.setenv("FAKE_HANG", "1")
        ai = LlamaCppAIProvider(
            LlamaCppConfig(model="m:Q4", binary=str(fake_server), startup_timeout=1)
        )
        with pytest.raises(ProviderError, match="not ready after 1s"):
            await ai.start()

        assert not ai._server.running
        await ai.close()

    async def test_a_missing_gguf_file_says_so(self, fake_server: Path, tmp_path: Path) -> None:
        ai = LlamaCppAIProvider(
            LlamaCppConfig(model=str(tmp_path / "absent.gguf"), binary=str(fake_server))
        )
        with pytest.raises(ProviderError, match="model file not found"):
            await ai.start()
        await ai.close()

    async def test_a_configured_port_already_taken_is_refused(self, fake_server: Path) -> None:
        with socket.socket() as busy:
            busy.bind(("127.0.0.1", 0))
            busy.listen()
            port = busy.getsockname()[1]
            ai = LlamaCppAIProvider(
                LlamaCppConfig(model="m:Q4", binary=str(fake_server), port=port)
            )
            with pytest.raises(ProviderError, match="already in use"):
                await ai.start()
            await ai.close()

    async def test_close_stops_the_server(self, fake_server: Path) -> None:
        ai = LlamaCppAIProvider(LlamaCppConfig(model="m:Q4", binary=str(fake_server)))
        await ai.start()
        process = ai._server._process
        assert process is not None and process.returncode is None

        await ai.close()

        assert process.returncode is not None
        assert not ai._server.running

    async def test_a_server_that_dies_at_startup_says_why(
        self, monkeypatch: pytest.MonkeyPatch, fake_server: Path
    ) -> None:
        monkeypatch.setenv("FAKE_FAIL", "1")
        ai = LlamaCppAIProvider(LlamaCppConfig(model="nope", binary=str(fake_server)))
        try:
            with pytest.raises(ProviderError) as info:
                await ai.start()
        finally:
            await ai.close()

        assert "exited with code 3" in str(info.value)
        assert "failed to load model 'nope'" in str(info.value)

    def test_name_and_endpoint(self, fake_server: Path) -> None:
        ai = LlamaCppAIProvider(LlamaCppConfig(model="m", binary=str(fake_server), port=18089))
        assert ai.name == "llamacpp"
        assert ai.base_url == "http://127.0.0.1:18089/v1"
        assert ai.model_name == "m"


class TestLlamaCppResponseSchema:
    async def test_a_schema_it_cannot_carry_is_refused_before_the_server_starts(
        self, monkeypatch: pytest.MonkeyPatch, fake_server: Path, tmp_path: Path
    ) -> None:
        starts = tmp_path / "starts"
        monkeypatch.setenv("FAKE_STARTS_FILE", str(starts))
        ai = LlamaCppAIProvider(LlamaCppConfig(model="m:Q4", binary=str(fake_server)))
        schema = {
            "type": "object",
            "properties": {},
            "required": [],
            "additionalProperties": False,
        }
        try:
            with pytest.raises(ResponseSchemaError) as exc:
                await ai.generate(_ask().model_copy(update={"response_schema": schema}))
        finally:
            await ai.close()

        assert exc.value.reason == "unsupported"  # the turn also carries tools
        assert not starts.exists()
