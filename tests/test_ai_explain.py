"""Tests de la capa de IA. Nunca tocan la red: fallback offline + LLM mockeado."""

import json
from types import SimpleNamespace

import pytest

from src.ai_explain import ACTIONS, Batch, CaseExplanation, enrich
from src.reconcile import STATUSES, explain, run


@pytest.fixture(scope="module")
def classified():
    return run()


def test_action_map_covers_every_status():
    assert set(ACTIONS) == STATUSES


def test_fallback_when_no_api_key(classified, monkeypatch):
    monkeypatch.setattr("src.ai_explain._load_dotenv", lambda: None)  # ignora .env
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    out = enrich(classified)
    assert out["ai_explanation"].str.len().gt(0).all()
    # sin LLM, la explicacion es la determinista y la accion sale del mapa
    assert (out["ai_explanation"] == out.apply(explain, axis=1)).all()
    assert (out["suggested_action"] == out["flag"].map(ACTIONS)).all()


def test_llm_path_mocked(classified, monkeypatch):
    """Con cliente OpenAI stubbeado: re-alinea por key, no cambia el status."""

    def fake_parse(**kwargs):
        cases = json.loads(kwargs["messages"][1]["content"])["cases"]
        items = [
            CaseExplanation(
                key=c["key"],
                explanation=f"LLM says: {c['flag']}",
                suggested_action=ACTIONS[c["flag"]],
            )
            for c in cases
        ]
        parsed = Batch(items=items)
        msg = SimpleNamespace(parsed=parsed)
        return SimpleNamespace(choices=[SimpleNamespace(message=msg)])

    fake_client = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(parse=fake_parse))
    )
    monkeypatch.setattr("src.ai_explain._load_dotenv", lambda: None)
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setattr("openai.OpenAI", lambda *a, **k: fake_client)

    before = classified["flag"].tolist()
    out = enrich(classified)

    # (a) salio del path LLM, no del fallback
    assert out["ai_explanation"].str.startswith("LLM says:").all()
    # (b) la clasificacion NO cambio
    assert out["flag"].tolist() == before
    # (c) accion dentro del enum y consistente con el status
    assert set(out["suggested_action"]) <= set(ACTIONS.values())
    assert (out["suggested_action"] == out["flag"].map(ACTIONS)).all()
    # (d) re-alineacion correcta por fila
    row = out[out["invoice_id"] == "INV-1003"].iloc[0]
    assert row["ai_explanation"] == "LLM says: Partial Match"


def test_llm_error_falls_back(classified, monkeypatch):
    def boom(**kwargs):
        raise RuntimeError("api down")

    fake_client = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(parse=boom))
    )
    monkeypatch.setattr("src.ai_explain._load_dotenv", lambda: None)
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setattr("openai.OpenAI", lambda *a, **k: fake_client)

    out = enrich(classified)
    assert (out["ai_explanation"] == out.apply(explain, axis=1)).all()
    assert (out["suggested_action"] == out["flag"].map(ACTIONS)).all()
