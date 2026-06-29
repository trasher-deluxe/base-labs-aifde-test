# Invoice ↔ Payment Reconciliation

Concilia facturas (`invoices.csv`) contra pagos (`payments.csv`) usando además
notas operativas en lenguaje natural (`notes.json`). Cada caso queda clasificado
**obligatoriamente** en uno de 5 estados y acompañado de un motivo en lenguaje
plano.

## Cómo correrlo

```bash
uv sync                       # instala dependencias (pandas, etc.)
uv run python -m src.reconcile
```

Imprime una tabla CLI con `invoice_id, payment_id, flag, remaining_balance,
explanation` y al final corre un self-check con `assert` (`all assertions
passed`). La exploración paso a paso está en `notebooks/eda.ipynb`.

## Estados

| Estado | Significado |
|---|---|
| **Matched** | El pago concilia con la factura (monto coincide, o la diferencia está explicada por un descuento/ajuste en una nota). |
| **Partial Match** | El pago cubre solo parte de la factura. Se reporta `remaining_balance`. |
| **Needs Review** | Match incierto: moneda distinta, pagador desconocido, sobrepago o nota que pide revisión. |
| **Suspicious** | Posible duplicado o anomalía (p. ej. dos pagos a la misma factura). No cerrar. |
| **Unmatched** | Registro sin contraparte: un pago que ninguna factura reclama. |

## Estrategia de matching

El pipeline (`src/reconcile.py`) es **invoice-céntrico** y combina señales:

1. **Extracción de IDs** desde el campo libre `reference` (pagos) y `text`
   (notas) con regex (`_extract_ids`): normaliza `INV-1001`, `INV1002` e
   `invoice 1001` → `INV-####`, y captura `PO-####`.
2. **Vínculo factura↔pago**: merge por `invoice_id` extraído, con *fallback* por
   `po_number` cuando la referencia solo trae el PO.
3. **Vínculo factura↔nota**: primero por `invoice_id`/`po_number` extraído de la
   nota; y como **fallback general, por nombre de vendor** cuando la nota no trae
   ningún ID (es una política a nivel proveedor, p. ej. la de Nova/EUR). La
   referencia y la nota se concatenan en `ALL_REFERENCES`.
4. **Clasificación en 2 capas, de general a particular** (gana la primera regla
   que aplica):
   - *Capa 1 — texto/política (general)* (`classify_text`): lo que un humano
     escribió en nota/referencia manda. Regex de prioridad sobre
     `ALL_REFERENCES` → Suspicious → Needs Review → Partial Match → Matched.
   - *Capa 2 — señales de datos (particular)* (`_classify_data`): solo para lo
     que el texto no resolvió — vencimiento, moneda, `Unknown Vendor`, monto
     parcial/excedente, presencia de pago.
   - Ejemplo: INV-1008 es `Needs Review` por la **política** de la nota de Nova
     ("revisar pagos EUR", general); la **moneda EUR≠USD** es el detalle
     particular que la confirma, por eso la nota se cita primero.
5. **Pagos huérfanos**: los pagos que ninguna factura reclamó se anexan y caen en
   `Unmatched` (garantiza que *todo* registro queda clasificado).
6. **`remaining_balance`**: `amount_factura − amount_pago`, solo en Partial Match.

## Dónde se usa "AI"

La explicación por caso (`explain`) es **lógica AI-assisted / mock**, no una
llamada a LLM todavía: a partir del flag ya decidido, re-evalúa las mismas
señales en el mismo orden que `_classify_data` y arma el motivo, de modo que el
"por qué" coincide con la causa real y no es texto inventado. Usa
`difflib.SequenceMatcher` para expresar la similitud del nombre del pagador
(distingue un typo ~95% de un pagador ajeno ~20%).

**Si se cambiara a un LLM real**: el prompt recibiría las señales ya calculadas
(montos, moneda, similitud de vendor, nota) y el flag, pidiendo *solo* redactar
el motivo — nunca decidir el estado (guardrail: la clasificación queda en
código determinista y auditable; el LLM no inventa cifras ni cambia el flag). La
API key iría por variable de entorno, nunca commiteada.

## Casos observados en los datos

- **INV-1001** — payer `ACME Logistcs` (typo, ~96% del vendor) + nota confirma → Matched.
- **INV-1002** — `Grupo Norte` vs `Grupo Norte SA`; ref `INV1002` sin guion → Matched.
- **INV-1003** — pagó 3000 de 4300; nota dice pago parcial → Partial (saldo 1300).
- **INV-1004** — `Unknown Vendor`, solo casa por `PO-8894` → Needs Review.
- **INV-1005** — 1490 vs 1500; nota explica descuento de 10 USD → Matched.
- **INV-1006** — `Northwind Food` vs `Northwind Foods`, moneda MXN → Matched.
- **INV-1007** — dos pagos (PAY-9007/9008); nota marca duplicado → Suspicious (×2).
- **INV-1008** — pago en EUR vs USD facturado; nota pide revisar EUR → Needs Review.
- **PAY-9010** — `Random Supplier`, "No invoice reference" → Unmatched.

## Edge cases manejados

- Typos en nombres de pagador (similitud difusa, no igualdad exacta).
- Referencias en formatos mixtos (`INV-1001`, `INV1002`, `invoice 1001`, solo PO).
- Pagos parciales (saldo pendiente) y descuentos/ajustes (diferencia justificada por nota).
- Pagos duplicados a la misma factura.
- Moneda distinta entre factura y pago.
- Pagos sin contraparte (huérfanos) → no se descartan, se reportan como Unmatched.

## Qué mejoraría

- **Pagos huérfanos → mejor candidato**: hoy solo se marcan Unmatched; podría
  sugerir la factura más probable por similitud de vendor + monto.
- **Matching de notas por vendor más robusto**: hoy usa el primer token del
  nombre; usar similitud difusa para vendors ambiguos o con varias facturas.
- **LLM real** para `explain` (ver guardrails arriba) y para extraer intención de
  notas más libres.
- **Tests** en `test_reconcile.py` en vez del self-check embebido en `__main__`.
- **Tolerancias configurables** (umbral de similitud, ventana de fechas, moneda).
