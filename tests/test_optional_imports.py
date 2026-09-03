"""``import roomkit`` needs only the base dependencies.

An optional SDK is imported where it is used, never at module level on a path
the package root reaches. The root imports the video vision providers, which
import the Gemini client helper: one ``import httpx`` at the top of that
module broke ``import roomkit`` for every install without the ``httpx`` extra.
The check runs in a fresh interpreter, because this one has httpx loaded.
"""

from __future__ import annotations

import subprocess
import sys


def test_roomkit_imports_without_httpx() -> None:
    code = "import sys; sys.modules['httpx'] = None; import roomkit; print(roomkit.__version__)"
    proc = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, timeout=120, check=False
    )
    assert proc.returncode == 0, proc.stderr[-1200:]
    assert proc.stdout.strip(), "the interpreter printed no version"
