"""Extraction tests — the LLM client is mocked, so no network/key is needed."""

import app.llm.extract as extract_mod


def test_valid_json_parses(monkeypatch):
    monkeypatch.setattr(
        extract_mod, "chat",
        lambda *a, **k: '{"years_experience": 5, "skills": ["python", "ml"], '
                        '"education": "BSc CS", "seniority": "senior"}',
    )
    profile = extract_mod.extract_profile("some resume")
    assert profile.years_experience == 5
    assert "python" in profile.skills
    assert profile.seniority == "senior"


def test_retries_then_succeeds(monkeypatch):
    calls = {"n": 0}

    def fake_chat(*a, **k):
        calls["n"] += 1
        if calls["n"] == 1:
            return "not json at all"
        return '{"years_experience": 2, "skills": [], "education": "", "seniority": "junior"}'

    monkeypatch.setattr(extract_mod, "chat", fake_chat)
    profile = extract_mod.extract_profile("resume")
    assert calls["n"] == 2            # retried once after the invalid response
    assert profile.years_experience == 2


def test_all_invalid_falls_back(monkeypatch):
    monkeypatch.setattr(extract_mod, "chat", lambda *a, **k: "garbage")
    profile = extract_mod.extract_profile("resume", max_retries=1)
    assert profile.years_experience == 0   # graceful fallback, no exception
