"""Tests del pipeline de conciliación. Usa los datos reales de data/ (deterministas)."""

import math

import pandas as pd
import pytest

from src.reconcile import STATUSES, _vendor_sim, run


@pytest.fixture(scope="module")
def result() -> pd.DataFrame:
    return run()


def _row(result, **kw):
    """Una sola fila por (col=valor); falla si no hay exactamente una."""
    ((col, val),) = kw.items()
    rows = result[result[col] == val]
    assert len(rows) == 1, f"esperaba 1 fila con {col}={val}, hay {len(rows)}"
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
    assert got == [flag], f"{invoice_id}: esperado {flag}, got {got}"


def test_every_record_is_classified(result):
    # clasificación OBLIGATORIA: ninguna fila sin flag y todas dentro del dominio
    assert result["flag"].notna().all()
    assert set(result["flag"]) <= STATUSES


def test_orphan_payment_is_unmatched(result):
    # PAY-9010 no lo reclama ninguna factura -> debe aparecer como Unmatched
    orphan = _row(result, payment_id="PAY-9010")
    assert orphan["flag"] == "Unmatched"
    assert pd.isna(orphan["invoice_id"])
    assert "Random Supplier" in orphan["explanation"]


def test_remaining_balance(result):
    # solo Partial Match tiene saldo; INV-1003 = 4300 - 3000
    assert _row(result, invoice_id="INV-1003")["remaining_balance"] == 1300.0
    non_partial = result.loc[result["flag"] != "Partial Match", "remaining_balance"]
    assert non_partial.isna().all()


def test_partial_explanation_cites_balance(result):
    assert (
        "saldo pendiente 1300.00" in _row(result, invoice_id="INV-1003")["explanation"]
    )


def test_vendor_note_fallback_general_before_particular(result):
    # la nota de Nova no traía INV/PO: se liga por vendor y se cita ANTES que la moneda
    row = _row(result, invoice_id="INV-1008")
    assert "manually reviewed" in (row["text"] or "")  # nota ligada por vendor
    assert row["explanation"].startswith("Revisión por política de la nota")
    assert row["explanation"].index("nota") < row["explanation"].index("EUR")


def test_discount_difference_is_matched(result):
    # INV-1005: pagó 1490 de 1500, la nota explica el descuento -> Matched (no Partial)
    row = _row(result, invoice_id="INV-1005")
    assert row["flag"] == "Matched"
    assert "descuento" in row["explanation"]


def test_all_explanations_present(result):
    assert result["explanation"].str.len().gt(0).all()


def test_vendor_sim_distinguishes_typo_from_stranger():
    assert _vendor_sim("ACME Logistics", "ACME Logistcs") > 0.8  # typo
    assert _vendor_sim("Nova Packaging", "Random Supplier") < 0.5  # otro
    assert math.isclose(_vendor_sim("Same Co", "Same Co"), 1.0)
