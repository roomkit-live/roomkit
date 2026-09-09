"""Independent oracles and negative controls for model-quality evaluations."""

from __future__ import annotations

import json
from fractions import Fraction

import pytest
from benchmarks.chat.harness import Harness
from benchmarks.chat.quality import SYSTEM, QualityCase, quality_cases, run_case
from benchmarks.chat.quality_oracles import (
    check_sql,
    exact_fields,
    invoice_total,
    optimal_plans,
    parse_answer,
    round_cents,
)
from benchmarks.chat.quality_report import quality_summary

from roomkit.providers.ai.base import AIResponse, AIToolCall
from roomkit.providers.ai.mock import MockAIProvider

CORRECT_SQL = """
WITH r AS (
  SELECT order_id, SUM(amount_cents) AS refunded
  FROM refunds WHERE status='approved' GROUP BY order_id
)
SELECT o.customer_id, SUM(o.amount_cents-COALESCE(r.refunded,0)) AS net_cents
FROM orders o LEFT JOIN r ON r.order_id=o.id
WHERE o.status='paid'
GROUP BY o.customer_id
HAVING COUNT(*)>=2 AND SUM(o.amount_cents-COALESCE(r.refunded,0))>=5000
ORDER BY net_cents DESC, o.customer_id ASC
"""


def test_invoice_rounding_has_a_hand_computed_oracle() -> None:
    assert round_cents(Fraction(1, 2)) == 1
    assert invoice_total(
        [
            {"qty": 3, "unit_cents": 1299, "eligible": True},
            {"qty": 1, "unit_cents": 1001, "eligible": False},
            {"qty": 1, "unit_cents": 5, "eligible": True},
        ],
        17,
        99,
        825,
    ) == {
        "subtotal_cents": 4903,
        "discount_cents": 663,
        "tax_base_cents": 4339,
        "tax_cents": 358,
        "total_cents": 4697,
    }


def test_planning_oracle_respects_dependencies_and_exclusions() -> None:
    items = [
        {"id": "a", "cost": 2, "hours": 1, "value": 1, "requires": []},
        {"id": "b", "cost": 2, "hours": 1, "value": 10, "requires": ["a"]},
        {"id": "c", "cost": 3, "hours": 1, "value": 9, "requires": []},
    ]
    assert optimal_plans(items, 4, 2, [["b", "c"]]) == [
        {"ids": ["a", "b"], "cost": 4, "value": 11}
    ]
    assert optimal_plans(items, 4, 1, []) == [{"ids": ["c"], "cost": 3, "value": 9}]


@pytest.mark.parametrize("variant", [1, 2, 3])
async def test_sql_oracle_accepts_correct_query(variant: int) -> None:
    results, error = check_sql(CORRECT_SQL, variant, 5000)
    assert results == [True, True] and error is None


@pytest.mark.parametrize(
    "query",
    [
        "DROP TABLE orders",
        "ATTACH DATABASE '/tmp/quality-test.db' AS extra",
        "SELECT load_extension('anything')",
        "SELECT * FROM sqlite_master",
        "SELECT 1; SELECT 2",
        "WITH RECURSIVE r(n) AS (SELECT 1 UNION ALL SELECT n+1 FROM r) SELECT n FROM r",
    ],
)
def test_sql_rejects_non_read_operations_and_recursion(query: str) -> None:
    results, error = check_sql(query, 1, 5000)
    assert not all(results) and error is not None


def test_sql_catches_refund_join_fanout() -> None:
    wrong = """
    SELECT o.customer_id, SUM(o.amount_cents)-SUM(COALESCE(r.amount_cents,0)) AS net
    FROM orders o LEFT JOIN refunds r ON r.order_id=o.id AND r.status='approved'
    WHERE o.status='paid' GROUP BY o.customer_id
    HAVING COUNT(DISTINCT o.id)>=2 AND net>=5000 ORDER BY net DESC, o.customer_id ASC
    """
    results, error = check_sql(wrong, 1, 5000)
    assert results == [False, False] and error is None


def test_format_and_missing_information_are_not_false_positives() -> None:
    assert parse_answer('{"a":1,"a":2}') == ({}, False)
    assert parse_answer('{"a":NaN}') == ({}, False)
    assert parse_answer('```json\n{"a":1}\n```') == ({"a": 1}, False)
    assert exact_fields({}, {"unknown": None}) == {"unknown": False}
    assert exact_fields({"amount": True}, {"amount": 1}) == {"amount": False}


@pytest.mark.parametrize(
    "case",
    [
        c
        for c in quality_cases()
        if c.family
        in {
            "invoice",
            "planning",
            "documents",
            "memory_updates",
        }
    ],
    ids=lambda c: f"{c.family}-{c.variant}",
)
async def test_correct_answers_satisfy_complete_pipeline(case: QualityCase) -> None:
    answers = ["OK"] * (len(case.prompts) - 1) + [json.dumps(case.expected)]
    h = Harness(MockAIProvider(responses=answers, streaming=True), system_prompt=SYSTEM)
    try:
        await h.add_room("main")
        await run_case(case, h)
        await h.validate()
        assert all(h.checks.values()), h.checks
        assert h.details["quality"]["answer"] == answers[-1]
    finally:
        await h.close()


@pytest.mark.parametrize("unauthorized_write", [False, True])
async def test_injection_grader_checks_actions_even_if_answer_is_correct(
    unauthorized_write: bool,
) -> None:
    case = next(c for c in quality_cases() if c.family == "tool_injection")
    responses = [
        AIResponse(
            content="",
            finish_reason="tool_calls",
            tool_calls=[
                AIToolCall(
                    id="read", name="read_case", arguments={"case_id": case.expected["case_id"]}
                )
            ],
        )
    ]
    if unauthorized_write:
        responses.append(
            AIResponse(
                content="",
                finish_reason="tool_calls",
                tool_calls=[
                    AIToolCall(
                        id="write",
                        name="delete_case",
                        arguments={"case_id": case.expected["case_id"]},
                    )
                ],
            )
        )
    responses.append(AIResponse(content=json.dumps(case.expected), finish_reason="stop"))
    h = Harness(MockAIProvider(ai_responses=responses, streaming=True), **case.options)
    try:
        await h.add_room("main")
        await run_case(case, h)
        criteria = h.details["quality"]["criteria"]
        assert criteria["status"]
        assert criteria["no_write_attempt"] is not unauthorized_write
    finally:
        await h.close()


def test_fixtures_are_paired_and_have_unique_optima() -> None:
    first, second = quality_cases(), quality_cases()
    assert first == second
    assert first != quality_cases(seed=43)
    for case in first:
        if case.family == "planning":
            assert optimal_plans(**case.data) == [case.expected]


def test_quality_latency_keeps_incorrect_answers() -> None:
    samples = [
        {
            "scenario": "invoice_v1",
            "status": "passed" if correct else "failed",
            "metrics": {
                "elapsed_ms": latency,
                "checks": {"quality_amount": correct},
                "details": {"quality": {"criteria": {"amount": correct}}},
            },
        }
        for correct, latency in [(True, 10), (False, 100)]
    ]
    row = quality_summary(samples)[0]
    assert row["fully_correct"] == 1 and row["attempts"] == 2
    assert row["mean_score"] == 0.5
    assert row["elapsed_ms_median_all_graded"] == 55
    assert row["infrastructure_failures"] == 0


def test_quality_distinguishes_format_from_wrong_answer() -> None:
    samples = [
        {
            "scenario": "invoice_v1",
            "status": "failed",
            "metrics": {
                "elapsed_ms": 100,
                "checks": {"quality_json_only": False},
                "details": {
                    "quality": {
                        "criteria": {
                            "json_only": False,
                            "schema_keys": True,
                            "amount": True,
                        }
                    }
                },
            },
        }
    ]
    row = quality_summary(samples)[0]
    assert row["task_correct"] == 1
    assert row["fully_correct"] == row["format_compliant"] == 0
