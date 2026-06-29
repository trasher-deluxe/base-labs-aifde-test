"""Invoice reconciliation pipeline."""

import re
from pathlib import Path

import pandas as pd

RULES = [
    ("Suspicious",    r"twice|duplicate|same payment|accidentally|doble|double"),
    ("Needs Review",  r"verify|manually review|review before|EUR.*payment|any.*EUR|mismatch"),
    ("Partial Match", r"partial payment|partial|remaining balance|balance will be paid|installment"),
    ("Matched",       r"confirmed payment|confirmed|applied.*discount|payment applied|invoice.*paid"),
]

_INV_DIRECT = r"(?i)\bINV-?(\d{4})(?!\d)"
_INV_WORD   = r"(?i)\binvoice\s+(\d{4})\b"
_PO         = r"(PO-\d{4})"


def _extract_ids(ser: pd.Series) -> pd.DataFrame:
    inv = ser.str.extract(_INV_DIRECT)[0].fillna(ser.str.extract(_INV_WORD)[0])
    return pd.DataFrame({
        "EXTRACTED_invoice_id": "INV-" + inv,
        "EXTRACTED_po_number":  ser.str.extract(_PO)[0],
    })


def classify_text(text: str) -> str:
    for flag, pattern in RULES:
        if re.search(pattern, text, re.IGNORECASE):
            return flag
    return "Unmatched"


def _classify_data(row: pd.Series) -> str:
    if row["flag"] != "Unmatched":
        return row["flag"]
    if row["IS_DUE_PAYMENT"]:
        return "Needs Review"
    if row["currency_invoice_tbl"] != row["currency_payments"]:
        return "Needs Review"
    if row["payer_name"] == "Unknown Vendor":
        return "Needs Review"
    if row["amount_invoice_tbl"] != row["amount_payments"]:
        return "Partial Match" if row["amount_payments"] < row["amount_invoice_tbl"] else "Needs Review"
    if pd.notna(row["payment_id"]):
        return "Matched"
    return "Unmatched"


def load(data_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    invoices = pd.read_csv(data_dir / "invoices.csv", parse_dates=["invoice_date", "due_date"]).drop_duplicates()
    payments = pd.read_csv(data_dir / "payments.csv", parse_dates=["payment_date"]).drop_duplicates()
    notes    = pd.read_json(data_dir / "notes.json")
    return invoices, payments, notes


def _build(invoices: pd.DataFrame, payments: pd.DataFrame, notes: pd.DataFrame) -> pd.DataFrame:
    pay = pd.concat([payments, _extract_ids(payments["reference"])], axis=1)
    nts = pd.concat([notes,    _extract_ids(notes["text"])],         axis=1)

    DROP = ["EXTRACTED_invoice_id", "EXTRACTED_po_number"]

    df = (
        invoices
        .merge(pay, how="left", left_on="invoice_id",  right_on="EXTRACTED_invoice_id",
               suffixes=("_invoice_tbl", "_payment_tbl"))
        .drop(columns=DROP)
        .merge(pay, how="left", left_on="po_number", right_on="EXTRACTED_po_number",
               suffixes=("_first_merge", "_second_payment_tbl"))
        .drop(columns=DROP)
    )

    for col in ["payment_id", "payment_date", "payer_name", "reference"]:
        df[col] = df[f"{col}_first_merge"].fillna(df[f"{col}_second_payment_tbl"])
        df = df.drop(columns=[f"{col}_first_merge", f"{col}_second_payment_tbl"])

    df["currency_payments"] = df["currency_payment_tbl"].fillna(df["currency"])
    df["amount_payments"]   = df["amount_payment_tbl"].fillna(df["amount"])
    df = df.drop(columns=["currency_payment_tbl", "currency", "amount_payment_tbl", "amount"])

    df["IS_DUE_PAYMENT"] = df["payment_date"] > df["due_date"]

    df = (
        df
        .merge(nts[["EXTRACTED_invoice_id", "source", "text"]], how="left",
               left_on="invoice_id", right_on="EXTRACTED_invoice_id")
        .merge(nts[["EXTRACTED_po_number",  "source", "text"]], how="left",
               left_on="po_number",  right_on="EXTRACTED_po_number")
        .drop(columns=DROP)
    )
    df["source"] = df["source_x"].fillna(df["source_y"])
    df["text"]   = df["text_x"].fillna(df["text_y"])
    df = df.drop(columns=["source_x", "source_y", "text_x", "text_y"])

    df["ALL_REFERENCES"] = df["reference"].fillna("") + df["text"].fillna("")
    return df


def run(data_dir: Path = Path("data")) -> pd.DataFrame:
    invoices, payments, notes = load(data_dir)
    df = _build(invoices, payments, notes)
    df["flag"] = df["ALL_REFERENCES"].apply(classify_text)
    df["flag"] = df.apply(_classify_data, axis=1)
    return df


if __name__ == "__main__":
    result = run()
    print(result[["invoice_id", "flag"]].to_string(index=False))

    # ponytail: self-check; promote to test_reconcile.py when adding more cases
    expected = {
        "INV-1001": "Matched",       "INV-1002": "Matched",
        "INV-1003": "Partial Match", "INV-1004": "Needs Review",
        "INV-1005": "Matched",       "INV-1006": "Matched",
        "INV-1007": "Suspicious",    "INV-1008": "Needs Review",
    }
    for inv_id, flag in expected.items():
        rows = result.loc[result["invoice_id"] == inv_id, "flag"].values
        assert all(f == flag for f in rows), f"{inv_id}: expected {flag}, got {rows}"
    print("all assertions passed")
