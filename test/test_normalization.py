from app.services.ranking import min_max_normalize

def test_min_max_normalize_basic():
    values = [1, 2, 3]
    normalized = min_max_normalize(values)

    assert normalized[0] == 0.0
    assert normalized[-1] == 1.0

def test_min_max_normalize_all_same():
    values = [5, 5, 5]
    normalized = min_max_normalize(values)

    assert all(v == 0.5 for v in normalized)