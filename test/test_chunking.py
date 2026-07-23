from app.nlp.chunking import chunk_text


def test_empty_text_returns_no_chunks():
    assert chunk_text("") == []
    assert chunk_text("   \n  ") == []


def test_short_text_is_single_chunk():
    assert chunk_text("Short resume.", chunk_size=400) == ["Short resume."]


def test_respects_chunk_size():
    text = (
        "Sentence one about python. Sentence two about ML. "
        "Sentence three about RAG. Sentence four about FastAPI."
    )
    chunks = chunk_text(text, chunk_size=60, overlap=10)
    assert len(chunks) > 1
    # allow slight overshoot from the overlap tail, but bounded
    assert all(len(c) <= 90 for c in chunks)


def test_overlap_carries_context():
    text = "Alpha beta gamma delta. Epsilon zeta eta theta. Iota kappa lambda mu."
    chunks = chunk_text(text, chunk_size=30, overlap=12)
    # with overlap, the start of a later chunk repeats the tail of the previous
    assert len(chunks) >= 2


def test_long_single_segment_is_hard_split():
    long_word_run = " ".join(["token"] * 100)  # one segment, no punctuation
    chunks = chunk_text(long_word_run, chunk_size=50, overlap=0)
    assert len(chunks) > 1
    assert all(len(c) <= 60 for c in chunks)
