"""Supervisor orchestration strategy.

A supervisor agent delegates tasks to worker agents via ``kit.delegate()``.
Workers are registered on the kit but NOT attached to the parent room.
"""

from __future__ import annotations

from roomkit.orchestration.strategies.supervisor._common import (
    _DEFAULT_MAX_REVISIONS as _DEFAULT_MAX_REVISIONS,
)
from roomkit.orchestration.strategies.supervisor._common import (
    _DEFAULT_TASK_TIMEOUT_SECONDS as _DEFAULT_TASK_TIMEOUT_SECONDS,
)
from roomkit.orchestration.strategies.supervisor._common import (
    WorkerStrategy as WorkerStrategy,
)
from roomkit.orchestration.strategies.supervisor.core import Supervisor as Supervisor
from roomkit.orchestration.strategies.supervisor.delegate import (
    _async_run_and_deliver as _async_run_and_deliver,
)
from roomkit.orchestration.strategies.supervisor.delegate import (
    _build_pass1_instruction as _build_pass1_instruction,
)
from roomkit.orchestration.strategies.supervisor.delegate import (
    _one_pass_delegate as _one_pass_delegate,
)
from roomkit.orchestration.strategies.supervisor.delegate import (
    _run_workers as _run_workers,
)
from roomkit.orchestration.strategies.supervisor.delegate import (
    _two_pass_delegate as _two_pass_delegate,
)
from roomkit.orchestration.strategies.supervisor.execution import (
    _compose_sequential_input as _compose_sequential_input,
)
from roomkit.orchestration.strategies.supervisor.execution import (
    _run_parallel as _run_parallel,
)
from roomkit.orchestration.strategies.supervisor.execution import (
    _run_sequential as _run_sequential,
)
from roomkit.orchestration.strategies.supervisor.prompts import (
    _format_supervised_digest as _format_supervised_digest,
)
from roomkit.orchestration.strategies.supervisor.prompts import (
    _parse_verdict as _parse_verdict,
)
from roomkit.orchestration.strategies.supervisor.results import (
    _extract_output_text as _extract_output_text,
)
from roomkit.orchestration.strategies.supervisor.results import (
    _format_supervisor_review as _format_supervisor_review,
)
from roomkit.orchestration.strategies.supervisor.results import (
    _format_worker_results as _format_worker_results,
)
from roomkit.orchestration.strategies.supervisor.results import (
    _present_worker_results as _present_worker_results,
)
from roomkit.orchestration.strategies.supervisor.results import (
    _render_result as _render_result,
)
from roomkit.orchestration.strategies.supervisor.results import (
    _worker_label as _worker_label,
)
from roomkit.orchestration.strategies.supervisor.supervised import (
    _run_supervised_sequential as _run_supervised_sequential,
)

__all__ = ["Supervisor", "WorkerStrategy"]
