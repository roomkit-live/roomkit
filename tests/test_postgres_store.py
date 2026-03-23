"""Tests for PostgresStore (store/postgres.py)."""

from __future__ import annotations

import importlib
import sys
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


def _build_mock_asyncpg() -> MagicMock:
    """Build a mock asyncpg module."""
    asyncpg = MagicMock()
    asyncpg.create_pool = AsyncMock()
    return asyncpg


class TestPostgresStore:
    def test_constructor_with_dsn(self) -> None:
        mock_asyncpg = _build_mock_asyncpg()
        with patch.dict(sys.modules, {"asyncpg": mock_asyncpg}):
            importlib.invalidate_caches()
            mod = importlib.import_module("roomkit.store.postgres")
            importlib.reload(mod)
            store = mod.PostgresStore(dsn="postgres://localhost/test")
            assert store._dsn == "postgres://localhost/test"
            assert store._pool is None
            assert store._owns_pool is True

    def test_constructor_with_pool(self) -> None:
        mock_asyncpg = _build_mock_asyncpg()
        mock_pool = MagicMock()
        with patch.dict(sys.modules, {"asyncpg": mock_asyncpg}):
            importlib.invalidate_caches()
            mod = importlib.import_module("roomkit.store.postgres")
            importlib.reload(mod)
            store = mod.PostgresStore(pool=mock_pool)
            assert store._pool is mock_pool
            assert store._owns_pool is False

    def test_constructor_without_asyncpg_raises(self) -> None:
        # Replace asyncpg in sys.modules with a sentinel that triggers ImportError
        import builtins

        real_import = builtins.__import__

        def _block_asyncpg(name: str, *args: object, **kwargs: object) -> object:
            if name == "asyncpg":
                raise ImportError("No module named 'asyncpg'")
            return real_import(name, *args, **kwargs)

        saved = sys.modules.pop("asyncpg", None)
        try:
            with patch.object(builtins, "__import__", side_effect=_block_asyncpg):
                mod = importlib.import_module("roomkit.store.postgres")
                importlib.reload(mod)
                with pytest.raises(ImportError, match="asyncpg"):
                    mod.PostgresStore(dsn="postgres://localhost/test")
        finally:
            if saved is not None:
                sys.modules["asyncpg"] = saved

    async def test_close_releases_pool(self) -> None:
        mock_asyncpg = _build_mock_asyncpg()
        mock_pool = AsyncMock()
        with patch.dict(sys.modules, {"asyncpg": mock_asyncpg}):
            importlib.invalidate_caches()
            mod = importlib.import_module("roomkit.store.postgres")
            importlib.reload(mod)
            # _owns_pool=True so close() should release the pool
            store = mod.PostgresStore(dsn="postgres://localhost/test")
            store._pool = mock_pool
            store._owns_pool = True
            await store.close()
            mock_pool.close.assert_awaited_once()
            assert store._pool is None

    async def test_close_skips_external_pool(self) -> None:
        mock_asyncpg = _build_mock_asyncpg()
        mock_pool = AsyncMock()
        with patch.dict(sys.modules, {"asyncpg": mock_asyncpg}):
            importlib.invalidate_caches()
            mod = importlib.import_module("roomkit.store.postgres")
            importlib.reload(mod)
            store = mod.PostgresStore(pool=mock_pool)
            await store.close()
            # External pool should NOT be closed
            mock_pool.close.assert_not_awaited()

    def test_ensure_pool_raises_without_init(self) -> None:
        mock_asyncpg = _build_mock_asyncpg()
        with patch.dict(sys.modules, {"asyncpg": mock_asyncpg}):
            importlib.invalidate_caches()
            mod = importlib.import_module("roomkit.store.postgres")
            importlib.reload(mod)
            store = mod.PostgresStore(dsn="postgres://localhost/test")
            with pytest.raises(RuntimeError, match="init"):
                store._ensure_pool()
