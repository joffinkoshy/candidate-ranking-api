from app.retrieval.hybrid import rrf


def test_rrf_empty():
    assert rrf([[], []]) == []


def test_rrf_prefers_items_ranked_well_by_both():
    dense = ["d1", "d2", "d3"]
    sparse = ["d3", "d1", "d4"]
    fused = rrf([dense, sparse])
    assert fused[0] == "d1"                 # high in both lists
    assert set(fused) == {"d1", "d2", "d3", "d4"}  # union of inputs


def test_rrf_single_list_preserves_order():
    assert rrf([["a", "b", "c"]]) == ["a", "b", "c"]


def test_rrf_k_dampens_rank_gap():
    # larger k => rank differences matter less
    dense = ["a", "b"]
    sparse = ["b", "a"]
    fused = rrf([dense, sparse], k=60)
    assert set(fused) == {"a", "b"}
