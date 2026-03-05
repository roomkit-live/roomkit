"""Tests for PyroscopeProfiler."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from roomkit.telemetry.pyroscope import PyroscopeProfiler


class TestPyroscopeProfiler:
    def test_init_defaults(self):
        profiler = PyroscopeProfiler()
        assert profiler._application_name == "roomkit"
        assert profiler._server_address == "http://localhost:4040"
        assert profiler._sample_rate == 100
        assert not profiler._started

    def test_init_custom(self):
        profiler = PyroscopeProfiler(
            application_name="my-app",
            server_address="http://pyroscope:4040",
            sample_rate=50,
            tags={"env": "test"},
        )
        assert profiler._application_name == "my-app"
        assert profiler._server_address == "http://pyroscope:4040"
        assert profiler._sample_rate == 50
        assert profiler._tags == {"env": "test"}

    def test_start_without_pyroscope_raises(self):
        profiler = PyroscopeProfiler()
        with (
            patch.dict("sys.modules", {"pyroscope": None}),
            pytest.raises(ImportError, match="pyroscope-io"),
        ):
            profiler.start()

    @patch("roomkit.telemetry.pyroscope.pyroscope", create=True)
    def test_start_calls_configure(self, mock_pyroscope: MagicMock):
        """start() calls pyroscope.configure with expected args."""
        import sys

        mock_mod = MagicMock()
        sys.modules["pyroscope"] = mock_mod

        try:
            profiler = PyroscopeProfiler(
                application_name="test-app",
                server_address="http://localhost:4040",
                sample_rate=200,
                tags={"env": "test"},
            )
            profiler.start()

            mock_mod.configure.assert_called_once()
            call_kwargs = mock_mod.configure.call_args[1]
            assert call_kwargs["application_name"] == "test-app"
            assert call_kwargs["server_address"] == "http://localhost:4040"
            assert call_kwargs["sample_rate"] == 200
            assert call_kwargs["tags"] == {"env": "test"}
            assert profiler._started
        finally:
            sys.modules.pop("pyroscope", None)

    @patch("roomkit.telemetry.pyroscope.pyroscope", create=True)
    def test_start_idempotent(self, mock_pyroscope: MagicMock):
        """Calling start() twice doesn't reconfigure."""
        import sys

        mock_mod = MagicMock()
        sys.modules["pyroscope"] = mock_mod

        try:
            profiler = PyroscopeProfiler()
            profiler.start()
            profiler.start()  # second call should be no-op
            assert mock_mod.configure.call_count == 1
        finally:
            sys.modules.pop("pyroscope", None)

    def test_stop_without_start(self):
        """stop() without start() is a no-op."""
        profiler = PyroscopeProfiler()
        profiler.stop()  # should not raise

    def test_tag_without_start(self):
        """tag() without start() just yields (no-op)."""
        profiler = PyroscopeProfiler()
        with profiler.tag(room_id="room-1"):
            pass  # should not raise

    def test_tag_session_without_start(self):
        """tag_session() without start() just yields (no-op)."""
        profiler = PyroscopeProfiler()
        with profiler.tag_session(room_id="room-1", session_id="sess-1"):
            pass  # should not raise

    def test_tag_session_builds_tags(self):
        """tag_session passes well-known keys to tag()."""
        import sys

        mock_mod = MagicMock()
        sys.modules["pyroscope"] = mock_mod

        try:
            profiler = PyroscopeProfiler()
            profiler.start()

            with profiler.tag_session(
                room_id="room-1",
                session_id="sess-1",
                backend="sip",
            ):
                mock_mod.tag_wrapper.assert_called_once_with(
                    {"room_id": "room-1", "session_id": "sess-1", "backend": "sip"}
                )
        finally:
            sys.modules.pop("pyroscope", None)

    def test_grafana_cloud_config(self):
        """Auth params are passed to configure for Grafana Cloud."""
        import sys

        mock_mod = MagicMock()
        sys.modules["pyroscope"] = mock_mod

        try:
            profiler = PyroscopeProfiler(
                server_address="https://profiles-prod.grafana.net",
                basic_auth_username="12345",
                basic_auth_password="token-abc",
                tenant_id="my-tenant",
            )
            profiler.start()

            call_kwargs = mock_mod.configure.call_args[1]
            assert call_kwargs["basic_auth_username"] == "12345"
            assert call_kwargs["basic_auth_password"] == "token-abc"
            assert call_kwargs["tenant_id"] == "my-tenant"
        finally:
            sys.modules.pop("pyroscope", None)

    def test_reimport_from_package(self):
        """Can be imported from the telemetry package."""
        from roomkit.telemetry import PyroscopeProfiler as Cls

        assert Cls is PyroscopeProfiler
