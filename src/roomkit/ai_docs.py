"""AI documentation helpers for llms.txt and AGENTS.md access."""

from __future__ import annotations

from importlib import resources
from pathlib import Path


def _find_file(filename: str) -> str:
    """Find and read a file from package or repository root."""
    # Try 1: importlib.resources (installed package)
    try:
        files = resources.files("roomkit")
        return (files / filename).read_text(encoding="utf-8")
    except (FileNotFoundError, TypeError):
        pass

    # Try 2: package directory (editable install with files copied)
    pkg_dir = Path(__file__).parent
    pkg_path = pkg_dir / filename
    if pkg_path.exists():
        return pkg_path.read_text(encoding="utf-8")

    # Try 3: repository root (development mode)
    # Go up from src/roomkit to repository root
    repo_root = pkg_dir.parent.parent
    repo_path = repo_root / filename
    if repo_path.exists():
        return repo_path.read_text(encoding="utf-8")

    raise FileNotFoundError(f"{filename} not found in roomkit package or repository")


def get_llms_txt() -> str:
    """Get the contents of llms.txt for LLM consumption.

    Returns:
        The llms.txt content as a string.

    Example:
        >>> from roomkit.ai_docs import get_llms_txt
        >>> content = get_llms_txt()
        >>> print(content[:50])
        # RoomKit
        ...
    """
    return _find_file("llms.txt")


def get_agents_md() -> str:
    """Get the contents of AGENTS.md for AI coding assistants.

    Returns:
        The AGENTS.md content as a string.

    Example:
        >>> from roomkit.ai_docs import get_agents_md
        >>> content = get_agents_md()
        >>> print(content[:50])
        # RoomKit
        ...
    """
    return _find_file("AGENTS.md")


def get_ai_context() -> str:
    """Get combined AI context (AGENTS.md + llms.txt summary).

    Useful for providing complete context to AI assistants.

    Returns:
        Combined content optimized for AI consumption.

    Example:
        >>> from roomkit.ai_docs import get_ai_context
        >>> context = get_ai_context()
        >>> # Pass to AI assistant as system context
    """
    agents = get_agents_md()
    llms = get_llms_txt()

    return f"""# RoomKit AI Context

## Project Guidelines (AGENTS.md)

{agents}

---

## Documentation Index (llms.txt)

{llms}
"""
