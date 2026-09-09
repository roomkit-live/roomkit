"""Deterministic model graders; no LLM judge and no arbitrary code execution."""

from __future__ import annotations

import itertools
import json
import sqlite3
from fractions import Fraction
from typing import Any


def parse_answer(text: str) -> tuple[dict[str, Any], bool]:
    """Fences lose the format point but can still receive semantic credit."""

    def unique(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("Duplicate JSON key")
            result[key] = value
        return result

    def reject_constant(value: str) -> None:
        raise ValueError("Non-finite JSON number")

    raw = text.strip()
    candidate = raw
    if raw.startswith("```json\n") and raw.endswith("```"):
        candidate = raw[8:-3].strip()
    try:
        value = json.loads(candidate, object_pairs_hook=unique, parse_constant=reject_constant)
    except (ValueError, TypeError):
        return {}, False
    if not isinstance(value, dict):
        return {}, False
    return value, candidate == raw


def exact_fields(actual: dict[str, Any], expected: dict[str, Any]) -> dict[str, bool]:
    """Missing nulls and bool-as-int coercion do not accidentally earn points."""
    return {
        key: key in actual and type(actual[key]) is type(value) and actual[key] == value
        for key, value in expected.items()
    }


def round_cents(value: Fraction) -> int:
    """Half-up for non-negative synthetic invoice amounts, using exact rationals."""
    if value < 0:
        raise ValueError("Only non-negative invoice amounts are supported")
    return (2 * value.numerator + value.denominator) // (2 * value.denominator)


def invoice_total(lines: list[dict[str, Any]], discount: int, shipping: int, tax: int) -> dict:
    subtotal = sum(line["qty"] * line["unit_cents"] for line in lines)
    rebate = sum(
        round_cents(Fraction(line["qty"] * line["unit_cents"] * discount, 100))
        for line in lines
        if line["eligible"]
    )
    base = subtotal - rebate + shipping
    vat = round_cents(Fraction(base * tax, 10_000))
    return {
        "subtotal_cents": subtotal,
        "discount_cents": rebate,
        "tax_base_cents": base,
        "tax_cents": vat,
        "total_cents": base + vat,
    }


def feasible(
    selection: set[str], items: list[dict], budget: int, capacity: int, excludes: list[list[str]]
) -> bool:
    by_id = {item["id"]: item for item in items}
    return (
        selection <= by_id.keys()
        and sum(by_id[k]["cost"] for k in selection) <= budget
        and sum(by_id[k]["hours"] for k in selection) <= capacity
        and all(set(by_id[k]["requires"]) <= selection for k in selection)
        and not any(set(pair) <= selection for pair in excludes)
    )


def optimal_plans(
    items: list[dict], budget: int, capacity: int, excludes: list[list[str]]
) -> list[dict]:
    """Exhaustively enumerate the small problem instead of trusting a heuristic."""
    plans: list[dict] = []
    best = -1
    for flags in itertools.product([False, True], repeat=len(items)):
        selected = [item for item, flag in zip(items, flags, strict=True) if flag]
        ids = {item["id"] for item in selected}
        if not feasible(ids, items, budget, capacity, excludes):
            continue
        score = sum(item["value"] for item in selected)
        if score < best:
            continue
        if score > best:
            plans.clear()
            best = score
        plans.append(
            {"ids": sorted(ids), "value": score, "cost": sum(item["cost"] for item in selected)}
        )
    return plans


def sql_fixture(variant: int, hidden: int) -> tuple[list[tuple], list[tuple], list[tuple]]:
    """Different data, duplicate order amounts and multiple refunds per order."""
    offset = variant * 10 + hidden * 100
    customers = [(i + offset, f"Customer-{i}") for i in range(1, 6)]
    orders = []
    refunds = []
    for i in range(1, 6):
        for j in range(i):
            order_id = offset * 100 + i * 10 + j
            status = "cancelled" if i == 4 or (hidden and j == 0) else "paid"
            orders.append((order_id, i + offset, status, 2500 + i * 150 + hidden * 70))
            if j % 2 == 0:
                refunds.extend(
                    [
                        (order_id, "approved", 120),
                        (order_id, "approved", 80),
                        (order_id, "pending", 900),
                    ]
                )
    return customers, orders, refunds


def sql_expected(orders: list[tuple], refunds: list[tuple], threshold: int) -> list[tuple]:
    totals: dict[int, int] = {}
    counts: dict[int, int] = {}
    for oid, cid, status, amount in orders:
        if status != "paid":
            continue
        totals[cid] = (
            totals.get(cid, 0)
            + amount
            - sum(value for rid, state, value in refunds if rid == oid and state == "approved")
        )
        counts[cid] = counts.get(cid, 0) + 1
    return sorted(
        [(cid, value) for cid, value in totals.items() if counts[cid] >= 2 and value >= threshold],
        key=lambda row: (-row[1], row[0]),
    )


def check_sql(query: str, variant: int, threshold: int) -> tuple[list[bool], str | None]:
    """Execute only bounded read-only SQLite, without extensions or filesystem access."""
    if not query or len(query) > 12_000:
        return [False, False], "Missing or oversized query"
    checks = []
    for hidden in range(2):
        customers, orders, refunds = sql_fixture(variant, hidden)
        db = sqlite3.connect(":memory:")
        try:
            db.executescript(
                "CREATE TABLE customers(id INTEGER PRIMARY KEY, name TEXT);"
                "CREATE TABLE orders(id INTEGER PRIMARY KEY, customer_id INTEGER, "
                "status TEXT, amount_cents INTEGER);"
                "CREATE TABLE refunds(order_id INTEGER, status TEXT, amount_cents INTEGER);"
            )
            db.executemany("INSERT INTO customers VALUES (?,?)", customers)
            db.executemany("INSERT INTO orders VALUES (?,?,?,?)", orders)
            db.executemany("INSERT INTO refunds VALUES (?,?,?)", refunds)

            def authorize(
                action: int,
                arg1: str | None,
                arg2: str | None,
                database: str | None,
                source: str | None,
            ) -> int:
                if action == sqlite3.SQLITE_SELECT:
                    return sqlite3.SQLITE_OK
                if action == sqlite3.SQLITE_READ and arg1 in {"customers", "orders", "refunds"}:
                    return sqlite3.SQLITE_OK
                if action == sqlite3.SQLITE_FUNCTION and arg2 in {
                    "sum",
                    "count",
                    "coalesce",
                    "ifnull",
                    "min",
                    "max",
                    "abs",
                    "round",
                }:
                    return sqlite3.SQLITE_OK
                return sqlite3.SQLITE_DENY

            steps = 0

            def progress() -> int:
                nonlocal steps
                steps += 1
                return int(steps > 100)  # At most ~100,000 VM instructions.

            db.set_authorizer(authorize)
            db.set_progress_handler(progress, 1000)
            rows = db.execute(query).fetchmany(101)
            checks.append(rows == sql_expected(orders, refunds, threshold))
        except sqlite3.Error as exc:
            return [*checks, *([False] * (2 - len(checks)))], str(exc)
        finally:
            db.close()
    return checks, None
