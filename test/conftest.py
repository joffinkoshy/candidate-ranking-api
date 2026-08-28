"""Test configuration.

Force dense-only mode for tests so they never make real LLM (Groq) calls, even
though a key may be present in .env. LLM parsing logic is tested separately with
a mocked client. This must run before app.config is imported.

Reranking is likewise off by default so most tests stay fast and don't need
the cross-encoder loaded; test_reranker.py and test_hybrid.py exercise it
directly with ENABLE_RERANK monkeypatched on.
"""

import os

os.environ.setdefault("ENABLE_LLM", "false")
os.environ.setdefault("ENABLE_RERANK", "false")
