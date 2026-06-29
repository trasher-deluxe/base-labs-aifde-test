"""Capa de IA: redacta `ai_explanation` en inglés simple + `suggested_action`.

Guardrail: el LLM NO decide el status. Recibe las señales YA calculadas (el motivo
determinista de `explain()` + el texto de la nota) y solo redacta la explicación y
elige una acción de un enum cerrado. Sin `OPENAI_API_KEY` o ante un error de la
API, cae a un fallback determinista, así el pipeline corre offline y los tests no
tocan la red.
"""

import json
import os
from pathlib import Path
from typing import Literal

import pandas as pd
from pydantic import BaseModel

from src.reconcile import explain

# Mapa determinista status -> acción sugerida (fuente de verdad y fallback).
ACTIONS = {
    "Matched": "Auto-match",
    "Partial Match": "Request remaining balance",
    "Needs Review": "Manual review required",
    "Suspicious": "Hold and investigate possible duplicate",
    "Unmatched": "Route to AP for manual investigation",
}

# Enum cerrado para la salida estructurada (acota lo que el LLM puede elegir).
Action = Literal[
    "Auto-match",
    "Request remaining balance",
    "Manual review required",
    "Hold and investigate possible duplicate",
    "Route to AP for manual investigation",
]

SYSTEM = (
    "You write explanations for an invoice reconciliation tool used by a finance "
    "back-office team. You do NOT decide the status; it is already computed. For "
    "each case, given its status, the rule-based reasoning and the optional "
    "operational note, write a one or two sentence plain-English explanation, and "
    "pick the suggested_action that matches the status. Keep each case's key "
    "unchanged so results can be matched back."
)


class CaseExplanation(BaseModel):
    key: str
    explanation: str
    suggested_action: Action


class Batch(BaseModel):
    items: list[CaseExplanation]


def _key(row: pd.Series) -> str:
    """payment_id es único por fila (incl. huérfanos); invoice_id como respaldo."""
    return row["payment_id"] if pd.notna(row.get("payment_id")) else row["invoice_id"]


def _load_dotenv() -> None:
    # ponytail: mini-loader; evita dep de python-dotenv. Solo si la var no está ya.
    if os.environ.get("OPENAI_API_KEY"):
        return
    env = Path(".env")
    if env.exists():
        for line in env.read_text().splitlines():
            if line.startswith("OPENAI_API_KEY="):
                os.environ.setdefault("OPENAI_API_KEY", line.split("=", 1)[1].strip())


def _fallback(df: pd.DataFrame) -> pd.DataFrame:
    df["ai_explanation"] = df.apply(explain, axis=1)
    df["suggested_action"] = df["flag"].map(ACTIONS)
    return df


def enrich(df: pd.DataFrame, model: str | None = None) -> pd.DataFrame:
    """Agrega `ai_explanation` + `suggested_action`. Offline-safe."""
    df = df.copy()
    _load_dotenv()
    if not os.environ.get("OPENAI_API_KEY"):
        return _fallback(df)

    rule_reason = df.apply(explain, axis=1)
    cases = [
        {
            "key": _key(row),
            "flag": row["flag"],
            "rule_reason": rule_reason[i],
            "note_text": row["text"] if pd.notna(row.get("text")) else "",
        }
        for i, row in df.iterrows()
    ]

    try:
        from openai import OpenAI

        resp = OpenAI().chat.completions.parse(
            model=model or os.environ.get("OPENAI_MODEL", "gpt-4o-mini"),
            temperature=0,
            timeout=30,
            response_format=Batch,
            messages=[
                {"role": "system", "content": SYSTEM},
                {"role": "user", "content": json.dumps({"cases": cases})},
            ],
        )
        by_key = {it.key: it for it in resp.choices[0].message.parsed.items}
    except Exception as e:  # red caída, error de API, rate limit, etc.
        print(f"[ai_explain] LLM no disponible ({e}); usando fallback determinista.")
        return _fallback(df)

    keys = df.apply(_key, axis=1)
    df["ai_explanation"] = [
        by_key[k].explanation if k in by_key else explain(df.loc[i])
        for i, k in keys.items()
    ]
    # el LLM elige la acción (acotada al enum); si faltara una clave, mapa determinista
    df["suggested_action"] = [
        by_key[k].suggested_action if k in by_key else ACTIONS[df.at[i, "flag"]]
        for i, k in keys.items()
    ]
    return df
