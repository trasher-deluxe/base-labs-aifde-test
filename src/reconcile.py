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

_INV_DIRECT = r"(?i)\bINV-?(\d{4})(?!\d)"
_INV_WORD = r"(?i)\binvoice\s+(\d{4})\b"
_PO = r"(PO-\d{4})"

STATUSES = {"Matched", "Partial Match", "Needs Review", "Unmatched", "Suspicious"}


def _extract_ids(ser: pd.Series) -> pd.DataFrame:
    inv = ser.str.extract(_INV_DIRECT)[0].fillna(ser.str.extract(_INV_WORD)[0])
    return pd.DataFrame(
        {
            "EXTRACTED_invoice_id": "INV-" + inv,
            "EXTRACTED_po_number": ser.str.extract(_PO)[0],
        }
    )


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
    if pd.isna(row["invoice_id"]):  # orphan payment: no invoice claims it
        return "Unmatched"
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


def load(data_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    invoices = pd.read_csv(
        data_dir / "invoices.csv", parse_dates=["invoice_date", "due_date"]
    ).drop_duplicates()
    payments = pd.read_csv(
        data_dir / "payments.csv", parse_dates=["payment_date"]
    ).drop_duplicates()
    notes = pd.read_json(data_dir / "notes.json")
    return invoices, payments, notes


def _build(
    invoices: pd.DataFrame, payments: pd.DataFrame, notes: pd.DataFrame
) -> pd.DataFrame:
    pay = pd.concat([payments, _extract_ids(payments["reference"])], axis=1)
    nts = pd.concat([notes, _extract_ids(notes["text"])], axis=1)

    DROP = ["EXTRACTED_invoice_id", "EXTRACTED_po_number"]

    df = (
        invoices.merge(
            pay,
            how="left",
            left_on="invoice_id",
            right_on="EXTRACTED_invoice_id",
            suffixes=("_invoice_tbl", "_payment_tbl"),
        )
        .drop(columns=DROP)
        .merge(
            pay,
            how="left",
            left_on="po_number",
            right_on="EXTRACTED_po_number",
            suffixes=("_first_merge", "_second_payment_tbl"),
        )
        .drop(columns=DROP)
    )

    for col in ["payment_id", "payment_date", "payer_name", "reference"]:
        df[col] = df[f"{col}_first_merge"].fillna(df[f"{col}_second_payment_tbl"])
        df = df.drop(columns=[f"{col}_first_merge", f"{col}_second_payment_tbl"])

    df["currency_payments"] = df["currency_payment_tbl"].fillna(df["currency"])
    df["amount_payments"] = df["amount_payment_tbl"].fillna(df["amount"])
    df = df.drop(
        columns=["currency_payment_tbl", "currency", "amount_payment_tbl", "amount"]
    )

    df["IS_DUE_PAYMENT"] = df["payment_date"] > df["due_date"]

    df = (
        df.merge(
            nts[["EXTRACTED_invoice_id", "source", "text"]],
            how="left",
            left_on="invoice_id",
            right_on="EXTRACTED_invoice_id",
        )
        .merge(
            nts[["EXTRACTED_po_number", "source", "text"]],
            how="left",
            left_on="po_number",
            right_on="EXTRACTED_po_number",
        )
        .drop(columns=DROP)
    )
    df["source"] = df["source_x"].fillna(df["source_y"])
    df["text"] = df["text_x"].fillna(df["text_y"])
    df = df.drop(columns=["source_x", "source_y", "text_x", "text_y"])

    # General -> particular fallback: a note without an INV/PO id is a vendor-level
    # policy (e.g. "Nova Packaging ... review EUR payments"). Link it by vendor name
    # when the invoice has no note yet, before falling back to raw data signals.
    # ponytail: match on the vendor's first token; enough for these policy notes.
    loose = nts[nts["EXTRACTED_invoice_id"].isna() & nts["EXTRACTED_po_number"].isna()]
    for i in df.index[df["text"].isna()]:
        token = str(df.at[i, "vendor"]).split()[0].lower()
        hit = loose[loose["text"].str.lower().str.contains(token, na=False)]
        if not hit.empty:
            df.at[i, "text"] = hit.iloc[0]["text"]
            df.at[i, "source"] = hit.iloc[0]["source"]

    df["ALL_REFERENCES"] = df["reference"].fillna("") + df["text"].fillna("")
    return df


def run(data_dir: Path = Path("data")) -> pd.DataFrame:
    invoices, payments, notes = load(data_dir)
    df = _build(invoices, payments, notes)

    # Mandatory classification: append payments no invoice claimed (-> Unmatched).
    matched = set(df["payment_id"].dropna())
    orphans = payments[~payments["payment_id"].isin(matched)].rename(
        columns={"currency": "currency_payments", "amount": "amount_payments"}
    )
    df = pd.concat([df, orphans], ignore_index=True)
    df["ALL_REFERENCES"] = df["ALL_REFERENCES"].fillna(df["reference"]).fillna("")

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
