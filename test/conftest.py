"""Test configuration.

Force dense-only mode for tests so they never make real LLM (Groq) calls, even
though a key may be present in .env. LLM parsing logic is tested separately with
a mocked client. This must run before app.config is imported.
"""

import os

os.environ.setdefault("ENABLE_LLM", "false")
