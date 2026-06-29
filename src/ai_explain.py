"""AI layer: rewrites `ai_explanation` in plain English.

Guardrail: the LLM makes NO financial decision. The status and the
`suggested_action` are both rule-based (`reconcile.py` + the ACTIONS map). The LLM
only receives the signals already computed (the deterministic `explain()` reason +
the note text) and rewrites them into a friendlier explanation. Without
`OPENAI_API_KEY` or on any API error it falls back to the deterministic reason, so
the pipeline runs offline and tests never hit the network.
"""

import json
import os
from pathlib import Path

import pandas as pd
from pydantic import BaseModel

from src.reconcile import explain

# Deterministic status -> suggested action (rule-based, not an LLM decision).
ACTIONS = {
    "Matched": "Auto-match",
    "Partial Match": "Keep invoice open",
    "Needs Review": "Manual review required",
    "Suspicious": "Manual review required",
    "Unmatched": "Investigate unmatched payment",
}

SYSTEM = (
    "You write explanations for an invoice reconciliation tool used by a finance "
    "back-office team. You do NOT decide the status or the action; they are already "
    "computed. For each case, given its status, the rule-based reasoning and the "
    "optional operational note, write a one or two sentence plain-English "
    "explanation for the operator. Keep each case's key unchanged so results can be "
    "matched back."
)


class CaseExplanation(BaseModel):
    key: str
    explanation: str


class Batch(BaseModel):
    items: list[CaseExplanation]


def _key(row: pd.Series) -> str:
    """payment_id is unique per row (incl. orphans); invoice_id as a fallback."""
    return row["payment_id"] if pd.notna(row.get("payment_id")) else row["invoice_id"]


def _load_dotenv() -> None:
    # ponytail: tiny loader, avoids a python-dotenv dep. Only if the var is unset.
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
    """Add `ai_explanation` + `suggested_action`. Offline-safe."""
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
    except Exception as e:  # network down, API error, rate limit, etc.
        print(f"[ai_explain] LLM unavailable ({e}); using deterministic fallback.")
        return _fallback(df)

    keys = df.apply(_key, axis=1)
    df["ai_explanation"] = [
        by_key[k].explanation if k in by_key else explain(df.loc[i])
        for i, k in keys.items()
    ]
    df["suggested_action"] = df["flag"].map(ACTIONS)  # rule-based, never the LLM
    return df
