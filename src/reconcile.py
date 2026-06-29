"""Invoice reconciliation pipeline.

Financial decisions are rule-based and deterministic here. The LLM layer
(`ai_explain.py`) only rewrites explanations; it never decides a status.
"""

import json
import re
from difflib import SequenceMatcher
from pathlib import Path

import pandas as pd

# Layer 1: text signals from the payment reference + operational note.
# Order matters: the first matching rule wins (general policy before data).
RULES = [
    ("Suspicious", r"twice|duplicate|same payment|accidentally|double"),
    (
        "Needs Review",
        r"verify|manually review|review before|EUR.*payment|any.*EUR|mismatch",
    ),
    (
        "Partial Match",
        r"partial payment|partial|remaining balance|balance will be paid|installment",
    ),
    (
        "Matched",
        r"confirmed payment|confirmed|applied.*discount|payment applied|invoice.*paid",
    ),
]

STATUSES = {"Matched", "Partial Match", "Needs Review", "Unmatched", "Suspicious"}


def _extract(text: str) -> tuple[str | None, str | None]:
    """Find an invoice id and/or a PO number inside free text (reference or note)."""
    text = str(text)
    m = re.search(r"(?i)\bINV-?(\d{4})\b", text) or re.search(
        r"(?i)\binvoice\s+(\d{4})\b", text
    )
    po = re.search(r"PO-\d{4}", text)
    return (f"INV-{m.group(1)}" if m else None), (po.group(0) if po else None)


def _vendor_sim(vendor: str, payer: str) -> float:
    """0..1 name similarity; tells a typo (~0.95) from a stranger payer (~0.2)."""
    return SequenceMatcher(None, str(vendor).lower(), str(payer).lower()).ratio()


def classify_text(text: str) -> str:
    for flag, pattern in RULES:
        if re.search(pattern, text, re.IGNORECASE):
            return flag
    return "Unmatched"


def _classify_data(row: pd.Series) -> str:
    """Layer 2: data signals, only for what the text could not classify."""
    if pd.isna(row["invoice_id"]) or pd.isna(row["payment_id"]):
        return "Unmatched"  # orphan payment, or an invoice nobody paid
    if row["flag"] != "Unmatched":
        return row["flag"]
    if row["IS_DUE_PAYMENT"]:
        return "Needs Review"
    if row["currency_invoice_tbl"] != row["currency_payments"]:
        return "Needs Review"
    if row["payer_name"] == "Unknown Vendor":
        return "Needs Review"
    if row["amount_invoice_tbl"] != row["amount_payments"]:
        return (
            "Partial Match"
            if row["amount_payments"] < row["amount_invoice_tbl"]
            else "Needs Review"
        )
    if pd.notna(row["payment_id"]):
        return "Matched"
    return "Unmatched"


def confidence(row: pd.Series) -> float:
    """Confidence (0..1) that the payment belongs to this invoice, from match
    signals (vendor name, currency, amount closeness). Independent from whether
    the case still needs review: a Suspicious case can be a confident match."""
    if row["flag"] == "Unmatched":
        return 0.2
    signals = []
    payer, vendor = row.get("payer_name"), row.get("vendor")
    if pd.notna(payer) and pd.notna(vendor):
        signals.append(_vendor_sim(vendor, payer))
    signals.append(
        1.0 if row.get("currency_invoice_tbl") == row.get("currency_payments") else 0.0
    )
    amt_i, amt_p = row.get("amount_invoice_tbl"), row.get("amount_payments")
    if pd.notna(amt_i) and pd.notna(amt_p) and max(amt_i, amt_p):
        signals.append(min(amt_i, amt_p) / max(amt_i, amt_p))
    return round(sum(signals) / len(signals), 2)


def explain(row: pd.Series) -> str:
    """Plain-language reason. Re-evaluates the same signals as _classify_data, in
    the same order, so the 'why' matches the real cause of the flag."""
    flag = row["flag"]
    cur_p, cur_i = row.get("currency_payments"), row.get("currency_invoice_tbl")
    amt_p, amt_i = row.get("amount_payments"), row.get("amount_invoice_tbl")
    payer, vendor = row.get("payer_name"), row.get("vendor")
    note = row.get("text")
    note = note if pd.notna(note) else ""
    tail = f" Note: '{note}'." if note else ""

    if flag == "Unmatched":
        if pd.notna(payer):  # orphan payment
            return (
                f"Payment {row.get('payment_id')} from '{payer}' ({amt_p} {cur_p}) "
                f"has no matching invoice (ref: '{row.get('reference')}')."
            )
        return "No payment could be matched to this invoice."

    if flag == "Suspicious":
        return f"Possible duplicate payment; do not close until verified.{tail}"

    if flag == "Needs Review":  # general -> particular: note policy before raw data
        if note and re.search(
            r"(?i)verify|manually review|review before|\bEUR\b", note
        ):
            extra = f" — paid in {cur_p} vs {cur_i} invoiced" if cur_p != cur_i else ""
            return f"Flagged for review by the note policy: '{note}'{extra}."
        if bool(row.get("IS_DUE_PAYMENT")):
            return f"Payment is past the invoice due date; review.{tail}"
        if pd.notna(amt_i) and cur_p != cur_i:
            return f"Currency {cur_p} does not match invoiced {cur_i}.{tail}"
        if payer == "Unknown Vendor":
            return "Unknown payer ('Unknown Vendor'); origin cannot be confirmed."
        if pd.notna(amt_i) and pd.notna(amt_p) and amt_p > amt_i:
            return f"Overpayment: {amt_p} {cur_p} against {amt_i} invoiced.{tail}"
        return f"Ambiguous signal; needs human review.{tail}"

    if flag == "Partial Match":
        rb = row.get("remaining_balance")
        base = f"Partial payment: {amt_p} of {amt_i} {cur_p}"
        if pd.notna(rb):
            base += f"; outstanding balance {rb:.2f}"
        return base + (tail or ".")

    # Matched
    sim = _vendor_sim(vendor, payer) if pd.notna(vendor) and pd.notna(payer) else 1.0
    parts = []
    if pd.notna(amt_i) and pd.notna(amt_p) and abs(amt_p - amt_i) >= 0.005:
        parts.append(
            f"{amt_i - amt_p:.2f} difference explained by a discount/adjustment"
        )
    else:
        parts.append(f"amount {amt_p} {cur_p} matches")
    if sim < 1.0:
        parts.append(f"payer ~{sim:.0%} similar to vendor (likely typo)")
    if note:
        parts.append(f"note: '{note}'")
    return "Matched: " + "; ".join(parts) + "."


def load(data_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, list[dict]]:
    invoices = pd.read_csv(
        data_dir / "invoices.csv", parse_dates=["invoice_date", "due_date"]
    ).drop_duplicates()
    payments = pd.read_csv(
        data_dir / "payments.csv", parse_dates=["payment_date"]
    ).drop_duplicates()
    notes = json.loads((data_dir / "notes.json").read_text())
    return invoices, payments, notes


def _note_lookup(notes: list[dict], invoices: pd.DataFrame):
    """Return a function invoice -> (note_text, source). A note is keyed by the
    invoice id / PO in its text; a note with neither is a vendor-level policy
    (e.g. Nova's EUR rule), matched by vendor name."""
    by_inv, by_po, loose = {}, {}, []
    for n in notes:
        inv_id, po = _extract(n["text"])
        if inv_id:
            by_inv[inv_id] = (n["text"], n["source"])
        elif po:
            by_po[po] = (n["text"], n["source"])
        else:
            loose.append((n["text"], n["source"]))

    def lookup(invoice_id, po, vendor):
        if invoice_id in by_inv:
            return by_inv[invoice_id]
        if po in by_po:
            return by_po[po]
        token = str(vendor).split()[0].lower()  # general fallback: by vendor name
        return next(((t, s) for t, s in loose if token in t.lower()), (None, None))

    return lookup


def _build(
    invoices: pd.DataFrame, payments: pd.DataFrame, notes: list[dict]
) -> pd.DataFrame:
    inv_by_id = invoices.set_index("invoice_id")
    po_to_id = dict(zip(invoices["po_number"], invoices["invoice_id"]))
    note_for = _note_lookup(notes, invoices)

    def make_row(invoice_id, payment) -> dict:
        inv = inv_by_id.loc[invoice_id] if invoice_id in inv_by_id.index else None
        text, source = (
            note_for(invoice_id, inv["po_number"], inv["vendor"])
            if inv is not None
            else (None, None)
        )
        ref = payment.reference if payment is not None else ""
        return {
            "invoice_id": invoice_id if inv is not None else None,
            "vendor": inv["vendor"] if inv is not None else None,
            "po_number": inv["po_number"] if inv is not None else None,
            "due_date": inv["due_date"] if inv is not None else pd.NaT,
            "currency_invoice_tbl": inv["currency"] if inv is not None else None,
            "amount_invoice_tbl": inv["amount"] if inv is not None else None,
            "payment_id": payment.payment_id if payment is not None else None,
            "payment_date": payment.payment_date if payment is not None else pd.NaT,
            "payer_name": payment.payer_name if payment is not None else None,
            "reference": ref,
            "currency_payments": payment.currency if payment is not None else None,
            "amount_payments": payment.amount if payment is not None else None,
            "text": text,
            "source": source,
            "ALL_REFERENCES": ref + (text or ""),
        }

    rows, paid = [], set()
    for p in payments.itertuples(index=False):
        inv_id, po = _extract(p.reference)
        if inv_id not in inv_by_id.index:
            inv_id = po_to_id.get(po)  # fall back to the PO number
        matched = inv_id if inv_id in inv_by_id.index else None
        rows.append(make_row(matched, p))
        if matched:
            paid.add(matched)
    for inv_id in inv_by_id.index.difference(paid):  # invoices nobody paid
        rows.append(make_row(inv_id, None))

    df = pd.DataFrame(rows)
    df["IS_DUE_PAYMENT"] = df["payment_date"] > df["due_date"]
    return df


def run(data_dir: Path = Path("data")) -> pd.DataFrame:
    invoices, payments, notes = load(data_dir)
    df = _build(invoices, payments, notes)

    df["flag"] = df["ALL_REFERENCES"].apply(classify_text)
    df["flag"] = df.apply(_classify_data, axis=1)

    df["remaining_balance"] = (df["amount_invoice_tbl"] - df["amount_payments"]).round(
        2
    )
    df["confidence"] = df.apply(confidence, axis=1)
    df["explanation"] = df.apply(explain, axis=1)
    return df


def to_records(df: pd.DataFrame) -> list[dict]:
    """Output rows, one per invoice (duplicate payments grouped into a list), plus
    one record per orphan payment. Matches the required output fields."""

    def explanation_of(row: pd.Series) -> str:
        return row.get("ai_explanation", row["explanation"])

    records = []
    invoiced = df[df["invoice_id"].notna()]
    for invoice_id, group in invoiced.groupby("invoice_id", sort=True):
        first = group.iloc[0]
        bal = first.get("remaining_balance")
        records.append(
            {
                "invoice_id": invoice_id,
                "matched_payment_ids": group["payment_id"].dropna().tolist(),
                "status": first["flag"],
                "confidence": round(float(group["confidence"].mean()), 2),
                "remaining_balance": None if pd.isna(bal) else round(float(bal), 2),
                "suggested_action": first.get("suggested_action"),
                "explanation": explanation_of(first),
            }
        )
    for _, row in df[df["invoice_id"].isna()].iterrows():
        records.append(
            {
                "invoice_id": None,
                "matched_payment_ids": [row["payment_id"]],
                "status": row["flag"],
                "confidence": round(float(row["confidence"]), 2),
                "remaining_balance": None,
                "suggested_action": row.get("suggested_action"),
                "explanation": explanation_of(row),
            }
        )
    return records


def export_json(df: pd.DataFrame, path: str | Path = "reconciliation.json") -> str:
    path = Path(path)
    path.write_text(json.dumps(to_records(df), indent=2, ensure_ascii=False))
    return str(path)


if __name__ == "__main__":
    from src.ai_explain import enrich

    result = enrich(run())  # AI layer; without OPENAI_API_KEY it falls back cleanly
    out = export_json(result)

    view = pd.DataFrame(to_records(result))
    cols = [
        "invoice_id",
        "matched_payment_ids",
        "status",
        "confidence",
        "remaining_balance",
        "suggested_action",
    ]
    print(view[cols].to_string(index=False))
    print(f"\nWrote {len(view)} records to {out}  (full explanations inside)")
    # Assertions live in tests/ (uv run pytest).
