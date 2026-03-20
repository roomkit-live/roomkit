"""Tests for the Orchestration ABC."""

from __future__ import annotations

import pytest

from roomkit.orchestration.base import Orchestration


class TestOrchestrationABC:
    def test_cannot_instantiate_abc(self):
        with pytest.raises(TypeError):
            Orchestration()  # type: ignore[abstract]

    def test_subclass_must_implement_agents(self):
        class Incomplete(Orchestration):
            async def install(self, kit, room_id):
                pass

        with pytest.raises(TypeError):
            Incomplete()  # type: ignore[abstract]

    def test_subclass_must_implement_install(self):
        class Incomplete(Orchestration):
            def agents(self):
                return []

        with pytest.raises(TypeError):
            Incomplete()  # type: ignore[abstract]

    def test_concrete_subclass_works(self):
        class Concrete(Orchestration):
            def agents(self):
                return []

            async def install(self, kit, room_id):
                pass

        orch = Concrete()
        assert orch.agents() == []
