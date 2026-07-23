"""Judge tests — the LLM client is mocked, so no network/key is needed."""

import app.llm.judge as judge_mod
from app.schemas import ExtractedProfile

PROFILE = ExtractedProfile(years_experience=3, skills=["python"])


def test_valid_verdict_parses(monkeypatch):
    monkeypatch.setattr(
        judge_mod, "chat",
        lambda *a, **k: '{"score": 85, "reasoning": "strong fit", '
                        '"citations": ["built RAG pipelines"]}',
    )
    verdict = judge_mod.judge_candidate("job", PROFILE, ["built RAG pipelines"])
    assert verdict.score == 85
    assert verdict.citations == ["built RAG pipelines"]


def test_out_of_range_score_is_clamped(monkeypatch):
    monkeypatch.setattr(
        judge_mod, "chat",
        lambda *a, **k: '{"score": 150, "reasoning": "x", "citations": []}',
    )
    verdict = judge_mod.judge_candidate("job", PROFILE, ["evidence"])
    assert verdict.score == 100.0    # clamped, not rejected


def test_invalid_output_falls_back(monkeypatch):
    monkeypatch.setattr(judge_mod, "chat", lambda *a, **k: "not json")
    verdict = judge_mod.judge_candidate("job", PROFILE, [], max_retries=1)
    assert verdict.score == 0.0
    assert "failed" in verdict.reasoning.lower()
