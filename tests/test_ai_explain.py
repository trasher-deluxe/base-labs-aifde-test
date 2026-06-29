"""AI layer tests. Never hit the network: offline fallback + mocked LLM."""

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
    monkeypatch.setattr("src.ai_explain._load_dotenv", lambda: None)  # ignore .env
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    out = enrich(classified)
    assert out["ai_explanation"].str.len().gt(0).all()
    # without an LLM, the explanation is the deterministic one and the action the map
    assert (out["ai_explanation"] == out.apply(explain, axis=1)).all()
    assert (out["suggested_action"] == out["flag"].map(ACTIONS)).all()


def test_llm_path_mocked(classified, monkeypatch):
    """With a stubbed OpenAI client: re-aligns by key, does not change the status."""

    def fake_parse(**kwargs):
        cases = json.loads(kwargs["messages"][1]["content"])["cases"]
        items = [
            CaseExplanation(key=c["key"], explanation=f"LLM says: {c['flag']}")
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

    # (a) it went through the LLM path, not the fallback
    assert out["ai_explanation"].str.startswith("LLM says:").all()
    # (b) the classification did NOT change
    assert out["flag"].tolist() == before
    # (c) action stays within the enum and consistent with the status
    assert set(out["suggested_action"]) <= set(ACTIONS.values())
    assert (out["suggested_action"] == out["flag"].map(ACTIONS)).all()
    # (d) correct per-row re-alignment
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
