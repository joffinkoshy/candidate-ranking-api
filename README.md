# RAG-Powered Candidate Ranking API

Ranks job candidates for a single opening by combining **structured ML scoring**
with a **Retrieval-Augmented Generation (RAG)** pipeline: it reads raw resumes,
retrieves the passages most relevant to the job, and uses an LLM to score and
**explain** each candidate's fit with citations back to the resume.

Everything runs on **free / open-source** components — local embeddings + a free
hosted open-model API (Groq).

---

## What it does

`POST /rank_candidates` with a job description and a list of candidates (each with
a few structured features + raw `resume_text`) returns the candidates ranked, each
with a score **breakdown** and a grounded justification.

```
resume → chunk → embed ─┬─► Chroma (dense / semantic)  ─┐
                        └─► BM25   (sparse / keyword)   ─┴─► RRF fusion → top-k evidence
                                                                            │
resume → LLM extract → structured profile ─────────────────────► LLM judge (score + citations)
                                                                            │
structured features → sklearn regression score ───────────► blend → deterministic ranking
```

If no LLM key is set, it degrades gracefully to **dense-retrieval-only** mode
(semantic similarity blended with the structured score).

---

## Concepts demonstrated

**Retrieval / RAG**
- Document **chunking**, text **embeddings**, a **vector database** (Chroma) for
  dense semantic search with metadata filtering
- **BM25** keyword (sparse) retrieval
- **Hybrid search** via **Reciprocal Rank Fusion (RRF)**
- **Grounding & citations** — the judge may only cite retrieved evidence

**LLM usage**
- **Structured output** — resume → validated Pydantic profile, with a
  **retry-on-validation** guardrail
- **LLM-as-a-judge** — rubric scoring with explanations
- **Model tiering** — cheap model (`llama-3.1-8b-instant`) extracts, strong model
  (`llama-3.3-70b-versatile`) judges
- Provider-agnostic client (OpenAI-compatible API → Groq / OpenRouter)

**ML & engineering**
- Per-request **min–max normalization**, sklearn **regression** scoring, weighted
  **score blending**, deterministic tie-breaking
- **Offline evaluation** with a golden set — **Kendall's tau** + top-3 overlap
- Env-based config, layered architecture, Pydantic validation

---

## Architecture

```
app/
  main.py               FastAPI routes
  config.py             env-based settings (models, weights, paths)
  schemas.py            Pydantic request/response + ExtractedProfile
  nlp/
    chunking.py         resume → passages
    embeddings.py       sentence-transformers (all-MiniLM-L6-v2)
  retrieval/
    vector_store.py     Chroma dense index
    keyword_index.py    BM25 sparse index
    hybrid.py           Reciprocal Rank Fusion
    index.py            bundles dense + sparse
  llm/
    client.py           OpenAI-compatible client (Groq)
    extract.py          resume → structured profile        (cheap model)
    judge.py            evidence → score + citations        (strong model)
  services/
    ingestion.py        chunk → embed → index
    ranking.py          orchestrates the full pipeline
  ml/                   training scripts + saved .pkl models
eval/
  golden_set.json       hand-labeled candidates
  evaluate.py           Kendall's tau + top-3 overlap
test/                   pytest suite
```

---

## Setup

> Use **Python 3.11 or 3.12** (chromadb / torch don't ship wheels for 3.14 yet).

```bash
cd candidate-ranking-api
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env        # add a free Groq key from https://console.groq.com
uvicorn app.main:app --reload
```

### Example request

```bash
curl -X POST localhost:8000/rank_candidates -H 'content-type: application/json' -d '{
  "job_id": "j1",
  "job_description": "Machine learning engineer with Python and RAG experience",
  "candidates": [
    {"candidate_id":"A","years_experience":1,"skill_match_score":0.2,"interview_score":0.3,"salary_expectation":10,"resume_text":"retail sales associate, cashier"},
    {"candidate_id":"B","years_experience":5,"skill_match_score":0.9,"interview_score":0.8,"salary_expectation":6,"resume_text":"ML engineer, built RAG pipelines in Python with FastAPI"}
  ]
}'
```

Response includes `final_score`, `structured_score`, `judge_score`, `reasoning`,
and `citations` per candidate.

---

## Evaluation

```bash
python -m eval.evaluate      # prints Kendall's tau + top-3 overlap vs. human labels
```

## Tests

```bash
pytest                       # LLM tests are mocked; no key needed
```

Set `ENABLE_LLM=false` to run the API without any LLM calls (dense-only mode).

---

## Design notes & future work

- **Per-request index**: ranking is within one job, so the candidate pool is
  indexed in-memory per request (stateless, no stale data).
- **Graceful degradation**: missing key / failed extraction never crashes a
  request — it falls back to dense mode / neutral defaults.
- **Future**: cross-encoder reranking, PII/bias redaction for fair hiring, prompt
  caching, streaming justifications, persistent vector store, model versioning.

---

## Interview guide

### 60-second pitch

> It's a resume-ranking API. Instead of just comparing text with cosine
> similarity, I built a RAG pipeline: I chunk and embed resumes into a vector
> store, then use **hybrid search** — keyword *and* semantic — to pull the
> passages relevant to a job. A small LLM extracts structured fields from the raw
> resume, and a stronger LLM acts as a **judge**, scoring each candidate against a
> rubric and **citing the exact resume lines** behind its reasoning, so the output
> is explainable. I used a cheaper model for extraction and a stronger one for the
> final judgment to control cost, and I validated the rankings against a
> hand-labeled set (Kendall's τ = 1.0 on my golden set).

### Questions you'll likely get

**"Why hybrid search instead of just embeddings?"**
Embeddings capture meaning but blur exact tokens — a job needing "CUDA" or "BM25"
should reward a resume that literally says it. BM25 (keyword) covers that failure
mode. I fuse the two rankings with Reciprocal Rank Fusion, which merges by rank so
I don't have to reconcile cosine scores with BM25 scores.

**"How do you stop the LLM from hallucinating?"**
Grounding. The judge is only given the retrieved evidence passages and is
instructed to cite exact phrases from them. It scores against that evidence, not
its own priors — so its reasoning is traceable back to the resume.

**"How do you know the ranking is any good?"**
An offline eval harness: I hand-labeled a golden set and compare the system's
order to the human order using Kendall's τ (rank correlation) and top-3 overlap.

**"Why two different models?"**
Model tiering for cost/latency: extraction is high-volume and easy, so it runs on
a small fast model; the final judgment is the decision that matters, so it runs on
a stronger one.

**"What would you add for production?"**
Cross-encoder reranking for precision, a PII/bias-redaction layer (important for
hiring), prompt caching, a persistent vector store, and request tracing.
