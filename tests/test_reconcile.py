"""Reconciliation pipeline tests. Uses the real sample data (deterministic)."""

import json
from pathlib import Path

import pandas as pd
import pytest

from src.reconcile import STATUSES, _vendor_sim, explain, export_json, run, to_records


@pytest.fixture(scope="module")
def result() -> pd.DataFrame:
    return run()


def _row(result, **kw):
    """Single row by (col=value); fails unless exactly one matches."""
    ((col, val),) = kw.items()
    rows = result[result[col] == val]
    assert len(rows) == 1, f"expected 1 row with {col}={val}, found {len(rows)}"
    return rows.iloc[0]


EXPECTED = {
    "INV-1001": "Matched",
    "INV-1002": "Matched",
    "INV-1003": "Partial Match",
    "INV-1004": "Needs Review",
    "INV-1005": "Matched",
    "INV-1006": "Matched",
    "INV-1007": "Suspicious",
    "INV-1008": "Needs Review",
}


@pytest.mark.parametrize("invoice_id,flag", EXPECTED.items())
def test_known_classifications(result, invoice_id, flag):
    got = result.loc[result["invoice_id"] == invoice_id, "flag"].unique().tolist()
    assert got == [flag], f"{invoice_id}: expected {flag}, got {got}"


def test_every_record_is_classified(result):
    # mandatory classification: no row without a flag, all within the domain
    assert result["flag"].notna().all()
    assert set(result["flag"]) <= STATUSES


def test_orphan_payment_is_unmatched(result):
    # PAY-9010 is claimed by no invoice -> must show up as Unmatched
    orphan = _row(result, payment_id="PAY-9010")
    assert orphan["flag"] == "Unmatched"
    assert pd.isna(orphan["invoice_id"])


def test_remaining_balance(result):
    assert _row(result, invoice_id="INV-1003")["remaining_balance"] == 1300.0  # partial
    assert (
        _row(result, invoice_id="INV-1001")["remaining_balance"] == 0.0
    )  # exact match
    assert (
        _row(result, invoice_id="INV-1005")["remaining_balance"] == 0.0
    )  # discount: gap is authorized, nothing outstanding
    assert pd.isna(_row(result, payment_id="PAY-9010")["remaining_balance"])  # orphan


def test_confidence_score(result):
    assert _row(result, invoice_id="INV-1001")["confidence"] > 0.8  # clean match, typo
    assert _row(result, payment_id="PAY-9010")["confidence"] == 0.2  # unmatched
    assert result["confidence"].between(0, 1).all()


# --- Edge cases called out in the brief --------------------------------------


def test_edge_partial_payment(result):
    row = _row(result, invoice_id="INV-1003")
    assert row["flag"] == "Partial Match"
    assert row["remaining_balance"] == 1300.0
    assert "outstanding balance 1300.00" in explain(row)


def test_edge_suspicious_duplicate(result):
    rows = result[result["invoice_id"] == "INV-1007"]
    assert len(rows) == 2  # two payments to the same invoice
    assert (rows["flag"] == "Suspicious").all()


def test_edge_vendor_typo(result):
    row = _row(result, invoice_id="INV-1001")  # payer "ACME Logistcs"
    assert row["flag"] == "Matched"
    assert _vendor_sim(row["vendor"], row["payer_name"]) > 0.9


def test_edge_discount_is_matched(result):
    row = _row(result, invoice_id="INV-1005")  # paid 1490 of 1500, note: discount
    assert row["flag"] == "Matched"
    assert "discount" in explain(row)


def test_vendor_note_fallback(result):
    # Nova's note has no INV/PO: linked by vendor, cited before the currency detail
    row = _row(result, invoice_id="INV-1008")
    assert "manually reviewed" in (row["text"] or "")
    assert explain(row).startswith("Flagged for review by the note policy")


def test_vendor_sim_distinguishes_typo_from_stranger():
    assert _vendor_sim("ACME Logistics", "ACME Logistcs") > 0.8
    assert _vendor_sim("Nova Packaging", "Random Supplier") < 0.5


# --- Output format ------------------------------------------------------------

REQUIRED_FIELDS = {
    "invoice_id",
    "matched_payment_ids",
    "status",
    "confidence",
    "remaining_balance",
    "suggested_action",
    "explanation",
}


def test_records_group_payments_per_invoice(result):
    records = to_records(result)
    by_invoice = {r["invoice_id"]: r for r in records if r["invoice_id"]}

    assert set(records[0]) == REQUIRED_FIELDS
    # duplicate payments collapse into one invoice record with a list of ids
    assert by_invoice["INV-1007"]["matched_payment_ids"] == ["PAY-9007", "PAY-9008"]
    # the orphan payment is its own record, no invoice
    orphan = [r for r in records if r["invoice_id"] is None][0]
    assert orphan["matched_payment_ids"] == ["PAY-9010"]
    assert orphan["status"] == "Unmatched"
    assert orphan["remaining_balance"] is None


def test_export_json_is_clean(result, tmp_path):
    path = export_json(result, tmp_path / "out.json")
    text = Path(path).read_text()
    data = json.loads(text)
    assert "NaN" not in text  # valid JSON, no NaN tokens
    assert all(set(r) == REQUIRED_FIELDS for r in data)
