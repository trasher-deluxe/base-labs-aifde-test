"""Invoice reconciliation pipeline."""

import re
from difflib import SequenceMatcher
from pathlib import Path

import pandas as pd

RULES = [
    ("Suspicious", r"twice|duplicate|same payment|accidentally|doble|double"),
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


def _extract_ids(ser: pd.Series) -> pd.DataFrame:
    inv = ser.str.extract(_INV_DIRECT)[0].fillna(ser.str.extract(_INV_WORD)[0])
    return pd.DataFrame(
        {
            "EXTRACTED_invoice_id": "INV-" + inv,
            "EXTRACTED_po_number": ser.str.extract(_PO)[0],
        }
    )


def _vendor_sim(vendor: str, payer: str) -> float:
    """0..1 similitud de nombres; distingue typo (~0.95) de payer ajeno (~0.2)."""
    return SequenceMatcher(None, str(vendor).lower(), str(payer).lower()).ratio()


def classify_text(text: str) -> str:
    for flag, pattern in RULES:
        if re.search(pattern, text, re.IGNORECASE):
            return flag
    return "Unmatched"


def _classify_data(row: pd.Series) -> str:
    if pd.isna(row["invoice_id"]):  # pago huérfano: ninguna factura lo reclama
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


def explain(row: pd.Series) -> str:
    """Motivo en lenguaje plano. Re-evalúa las mismas señales que _classify_data,
    en el mismo orden, para que el 'por qué' coincida con la causa real del flag."""
    flag = row["flag"]
    cur_p, cur_i = row.get("currency_payments"), row.get("currency_invoice_tbl")
    amt_p, amt_i = row.get("amount_payments"), row.get("amount_invoice_tbl")
    payer, vendor = row.get("payer_name"), row.get("vendor")
    note = row.get("text")
    note = note if pd.notna(note) else ""
    tail = f" Nota: «{note}»." if note else ""

    if flag == "Unmatched":
        if pd.notna(payer):  # pago huérfano
            return (
                f"Pago {row.get('payment_id')} de '{payer}' ({amt_p} {cur_p}) "
                f"sin factura vinculable (ref: '{row.get('reference')}')."
            )
        return "Ninguna factura encontró un pago que la concilie."

    if flag == "Suspicious":
        return f"Posible pago duplicado; no cerrar hasta verificar.{tail}"

    if flag == "Needs Review":  # general -> particular: política (nota) antes que datos
        if note and re.search(
            r"(?i)verify|manually review|review before|\bEUR\b", note
        ):
            extra = f" — pago {cur_p} vs {cur_i} facturado" if cur_p != cur_i else ""
            return f"Revisión por política de la nota: «{note}»{extra}."
        if bool(row.get("IS_DUE_PAYMENT")):
            return f"Pago posterior a la fecha de vencimiento; revisar.{tail}"
        if pd.notna(amt_i) and cur_p != cur_i:
            return f"Moneda {cur_p} no coincide con {cur_i} facturada.{tail}"
        if payer == "Unknown Vendor":
            return "Pagador desconocido ('Unknown Vendor'); no se confirma el origen."
        if pd.notna(amt_i) and pd.notna(amt_p) and amt_p > amt_i:
            return f"Sobrepago: {amt_p} {cur_p} contra {amt_i} facturado.{tail}"
        return f"Señal ambigua; requiere revisión humana.{tail}"

    if flag == "Partial Match":
        rb = row.get("remaining_balance")
        base = f"Pago parcial: {amt_p} de {amt_i} {cur_p}"
        if pd.notna(rb):
            base += f"; saldo pendiente {rb:.2f}"
        return base + (tail or ".")

    # Matched
    sim = _vendor_sim(vendor, payer) if pd.notna(vendor) and pd.notna(payer) else 1.0
    parts = []
    if pd.notna(amt_i) and pd.notna(amt_p) and abs(amt_p - amt_i) >= 0.005:
        parts.append(
            f"diferencia de {amt_i - amt_p:.2f} explicada por descuento/ajuste"
        )
    else:
        parts.append(f"monto {amt_p} {cur_p} coincide")
    if sim < 1.0:
        parts.append(f"payer ~{sim:.0%} similar al vendor (typo probable)")
    if note:
        parts.append(f"nota: «{note}»")
    return "Conciliado: " + "; ".join(parts) + "."


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

    # Fallback general -> particular: una nota sin INV/PO es una política a nivel
    # vendor (p.ej. "Nova Packaging ... revisar pagos EUR"). Se liga por nombre de
    # vendor cuando la factura aún no tiene nota, antes de caer en señales de datos.
    # ponytail: match por primer token del vendor; suficiente para notas de política.
    loose = nts[nts["EXTRACTED_invoice_id"].isna() & nts["EXTRACTED_po_number"].isna()]
    for i in df.index[df["text"].isna()]:
        token = str(df.at[i, "vendor"]).split()[0].lower()
        hit = loose[loose["text"].str.lower().str.contains(token, na=False)]
        if not hit.empty:
            df.at[i, "text"] = hit.iloc[0]["text"]
            df.at[i, "source"] = hit.iloc[0]["source"]

    df["ALL_REFERENCES"] = df["reference"].fillna("") + df["text"].fillna("")
    return df


STATUSES = {"Matched", "Partial Match", "Needs Review", "Unmatched", "Suspicious"}


def run(data_dir: Path = Path("data")) -> pd.DataFrame:
    invoices, payments, notes = load(data_dir)
    df = _build(invoices, payments, notes)

    # Clasificación obligatoria: anexar pagos que ninguna factura reclamó.
    matched = set(df["payment_id"].dropna())
    orphans = payments[~payments["payment_id"].isin(matched)].rename(
        columns={"currency": "currency_payments", "amount": "amount_payments"}
    )
    df = pd.concat([df, orphans], ignore_index=True)
    df["ALL_REFERENCES"] = df["ALL_REFERENCES"].fillna(df["reference"]).fillna("")

    df["flag"] = df["ALL_REFERENCES"].apply(classify_text)
    df["flag"] = df.apply(_classify_data, axis=1)

    df["remaining_balance"] = (df["amount_invoice_tbl"] - df["amount_payments"]).where(
        df["flag"] == "Partial Match"
    )
    df["explanation"] = df.apply(explain, axis=1)
    return df


if __name__ == "__main__":
    from src.ai_explain import enrich

    result = enrich(run())  # capa de IA; sin OPENAI_API_KEY cae a fallback determinista
    cols = ["invoice_id", "payment_id", "flag", "suggested_action", "ai_explanation"]
    print(result[cols].to_string(index=False))
    # Las aserciones viven en tests/ (uv run pytest).
