---
name: test-writer
description: Generate comprehensive pytest test cases from source code
license: MIT
---

# Test Writer Skill

Generate tests using pytest following these principles:

1. **Structure** -- one test module per source module, mirroring the package layout.
2. **Fixtures** -- use `@pytest.fixture` for shared setup; prefer factory fixtures over complex state.
3. **Coverage** -- test happy path, error cases, and boundary conditions for every public function.
4. **Parametrize** -- use `@pytest.mark.parametrize` when testing multiple inputs for the same logic.
5. **Async** -- use `async def test_...` directly (asyncio_mode = "auto"), no decorator needed.

Refer to the patterns reference for reusable fixture and assertion templates.
