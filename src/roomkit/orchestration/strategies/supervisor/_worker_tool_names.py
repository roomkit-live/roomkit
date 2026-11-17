"""The name of each worker's ``delegate_to_<id>`` tool (RFC §19.7.3).

A worker's tool is named after its channel id, which need not be a tool name
a provider accepts (``agent:<uuid>``). Every built-in provider accepts
``[A-Za-z0-9_-]`` up to 128 characters, so a worker whose
``delegate_to_<id>`` is such a name keeps it, and any other's id has each
other character replaced by ``_``. A replaced name that another worker's
name already is, or one longer than 128 characters, ends with ``_`` and the
first 8 hex digits of the id's SHA-256 instead, cut to fit the bound.
"""

from __future__ import annotations

import hashlib
import re
from collections import Counter
from collections.abc import Sequence

_PREFIX = "delegate_to_"
_MAX_NAME = 128
_DIGEST_HEX = 8
_ANY_PROVIDER_NAME = re.compile(rf"[A-Za-z0-9_-]{{1,{_MAX_NAME}}}")
_REPLACED = re.compile(r"[^A-Za-z0-9_-]")


def worker_tool_names(worker_ids: Sequence[str]) -> dict[str, str]:
    """Each worker's tool name, keyed by its worker id, in the workers' order."""
    kept = {
        worker_id: _PREFIX + worker_id
        for worker_id in worker_ids
        if _ANY_PROVIDER_NAME.fullmatch(_PREFIX + worker_id)
    }
    replaced = {
        worker_id: _PREFIX + _REPLACED.sub("_", worker_id)
        for worker_id in worker_ids
        if worker_id not in kept
    }
    uses = Counter(replaced.values())
    taken = set(kept.values())
    for worker_id, name in replaced.items():
        if name in taken or uses[name] > 1 or len(name) > _MAX_NAME:
            replaced[worker_id] = _with_digest(worker_id, name)
    return {worker_id: kept.get(worker_id) or replaced[worker_id] for worker_id in worker_ids}


def _with_digest(worker_id: str, name: str) -> str:
    """*name* cut to fit, ending with ``_`` and the start of *worker_id*'s digest."""
    digest = hashlib.sha256(worker_id.encode("utf-8")).hexdigest()[:_DIGEST_HEX]
    return f"{name[: _MAX_NAME - _DIGEST_HEX - 1]}_{digest}"
