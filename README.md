# Invoice ↔ Payment Reconciliation

An AI-assisted reconciliation assistant for a finance back-office team. It matches
payments (`payments.csv`) to invoices (`invoices.csv`), using operational notes in
natural language (`notes.json`) as extra signal. Every record is classified into
one of five statuses, with a confidence score, a suggested next action, and a
plain-English explanation.

**Financial decisions are rule-based and deterministic.** An LLM is used only to
rewrite the explanation into friendlier English — it never decides a status or an
action (and the pipeline runs fully without it).

## How to run

```bash
uv sync
uv run python -m src.reconcile
```

This prints a CLI table and writes `reconciliation.json`. Step-by-step exploration
is in `notebooks/eda.ipynb`; tests are in `tests/` (`uv run pytest`).

## Output

One record per invoice (duplicate payments are grouped into `matched_payment_ids`),
plus one record per orphan payment. Fields:

| field | meaning |
|---|---|
| `invoice_id` | invoice id (`null` for an orphan payment) |
| `matched_payment_ids` | list of payment ids matched to it |
| `status` | one of the five statuses below |
| `confidence` | 0..1, how strongly the payment matches the invoice |
| `remaining_balance` | `invoice − paid` (only meaningful for Partial Match; `null` for orphans) |
| `suggested_action` | next step for the operator (rule-based) |
| `explanation` | short plain-English reason |

## Statuses

| Status | Meaning | Action |
|---|---|---|
| **Matched** | Payment reconciles (amount matches, or the gap is a discount/adjustment explained by a note). | Auto-match |
| **Partial Match** | Payment covers only part of the invoice; `remaining_balance` reported. | Keep invoice open |
| **Needs Review** | Uncertain: currency mismatch, unknown payer, overpayment, or a note asking for review. | Manual review required |
| **Suspicious** | Possible duplicate/anomaly (e.g. two payments to one invoice). | Manual review required |
| **Unmatched** | A payment no invoice claims. | Investigate unmatched payment |

## Matching strategy

The pipeline (`src/reconcile.py`) is **invoice-centric** and combines signals:

1. **ID extraction** from the free-text `reference` (payments) and `text` (notes)
   via regex (`_extract_ids`): normalizes `INV-1001`, `INV1002`, `invoice 1001` →
   `INV-####`, and captures `PO-####`.
2. **Invoice ↔ payment**: merge on the extracted `invoice_id`, with a `po_number`
   fallback when the reference only carries the PO.
3. **Invoice ↔ note**: first by the `invoice_id`/`po_number` extracted from the
   note; then, as a **general fallback, by vendor name** when the note carries no
   id (it is a vendor-level policy, e.g. Nova's EUR rule). Reference + note are
   concatenated into `ALL_REFERENCES`.
4. **Two-layer classification, general → particular** (first matching rule wins):
   - *Layer 1 — text/policy (general)* (`classify_text`): what a human wrote in the
     note/reference wins. Priority regex over `ALL_REFERENCES` → Suspicious →
     Needs Review → Partial Match → Matched.
   - *Layer 2 — data signals (particular)* (`_classify_data`): only for what the
     text did not resolve — due date, currency, `Unknown Vendor`, partial/over
     amount, presence of a payment.
   - Example: INV-1008 is `Needs Review` because of Nova's note **policy** ("review
     EUR payments", general); the **EUR≠USD** currency is the particular detail
     that confirms it, so the note is cited first.
5. **Orphan payments**: payments no invoice claimed are appended and become
   `Unmatched`, so every record is classified.
6. **`confidence`**: a 0..1 blend of vendor-name similarity, currency match, and
   amount closeness — independent of the status (a Suspicious case can still be a
   confident match).

## Where AI is used — `src/ai_explain.py`

Two levels, separated on purpose:

1. **Deterministic reason** (`reconcile.explain`): from the decided flag, it
   re-evaluates the same signals in the same order as `_classify_data`, so the
   "why" matches the real cause. Uses `difflib.SequenceMatcher` for payer
   similarity (typo ~95% vs stranger ~20%).
2. **LLM phrasing** (`ai_explain.enrich`): takes that deterministic reason + the
   note text and, in **one batched OpenAI call with structured output** (pydantic),
   returns a friendlier English `explanation` per case.

### Setup (do not commit secrets)

```bash
export OPENAI_API_KEY=sk-...        # or put it in .env (already gitignored)
export OPENAI_MODEL=gpt-4o-mini     # optional (default gpt-4o-mini)
uv run python -m src.reconcile
```

### Guardrails

- **The LLM makes no financial decision.** Status and `suggested_action` are both
  rule-based; the LLM only rewrites the explanation.
- **Structured output** (pydantic `Batch`/`CaseExplanation`): no free-text parsing.
- **Re-alignment by `key`** (`payment_id`), not by response order.
- **Offline fallback:** without `OPENAI_API_KEY`, or on any API error, it falls
  back to the deterministic reason. The pipeline always produces every field and
  **tests never hit the network** (the LLM is mocked).

## Edge cases handled

- Payer-name typos (fuzzy similarity, not exact equality) — e.g. `ACME Logistcs`.
- Mixed reference formats (`INV-1001`, `INV1002`, `invoice 1001`, PO-only).
- Partial payments (remaining balance) and discounts/adjustments (gap justified by a note).
- Duplicate payments to the same invoice → Suspicious, payments grouped.
- Currency mismatch between invoice and payment.
- Orphan payments → not dropped, reported as Unmatched.

## What I would improve

- **Orphan payments → best candidate**: today only marked Unmatched; could suggest
  the most likely invoice by vendor similarity + amount.
- **More robust note↔vendor matching**: today uses the vendor's first token; use
  fuzzy similarity for ambiguous vendors or vendors with several invoices.
- **Confidence calibration**: weights are a simple average; could be tuned against
  labelled outcomes.
- **Persistence + review state** (approve/reject/resolved) and an audit trail.
