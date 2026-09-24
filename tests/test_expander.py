from src.retrieval.expansion.query_expander import QueryExpander

def test_query_expander():
    expander = QueryExpander()
    
    # 1. Test Dense, Lexical Synonyms, and Heuristic Action Stemming
    query = "How is the data normalized?"
    result = expander.expand(query)
    
    assert result["dense_query"] == query
    
    bm25 = result["bm25_query"]
    assert "normalize" in bm25
    assert "preprocessing" in bm25
    assert "data" in bm25
    assert "how" not in bm25
    assert "the" not in bm25
    
    struct = result["structural_query"]
    assert struct.get("action") == "normalize"

    # 2. Test Secondary Action Verification
    query2 = "Where do we calculate loss?"
    result2 = expander.expand(query2)
    
    bm25_2 = result2["bm25_query"]
    assert "loss" in bm25_2
    assert "cost" in bm25_2
    assert "error" in bm25_2
    
    struct2 = result2["structural_query"]
    assert struct2.get("action") == "calculate"
