from src.retrieval.fusion.rrf import ReciprocalRankFusion
from src.retrieval.fusion.cross_encoder import CrossEncoderReRanker

def test_rrf():
    # 1. Setup Mock Results across three modalities
    dense = [
        {"chunk_id": "chunk-1", "rank": 1},
        {"chunk_id": "chunk-2", "rank": 2},
    ]
    bm25 = [
        {"chunk_id": "chunk-2", "rank": 1},
        {"chunk_id": "chunk-3", "rank": 2},
    ]
    structural = [
        {"chunk_id": "chunk-1", "rank": 1},
    ]
    
    # 2. Execute RRF Fusion Pipeline
    rrf = ReciprocalRankFusion(k=60)
    results = rrf.fuse(dense, bm25, structural, top_k=30)
    
    assert len(results) == 3
    
    # chunk-1 is heavily ranked in dense and structural:
    # 1/61 + 1/61 = 2/61 (0.0327)
    
    # chunk-2 is heavily ranked in dense and bm25:
    # 1/62 + 1/61 = 0.0161 + 0.0163 = 0.0325
    
    # chunk-3 is moderately ranked in bm25:
    # 1/62 = 0.0161
    
    # Assert structural math checks out: chunk-1 edges out chunk-2 perfectly
    assert results[0]["chunk_id"] == "chunk-1"
    assert results[0]["rank"] == 1
    assert "rrf_score" in results[0]
    
    assert results[1]["chunk_id"] == "chunk-2"
    assert results[1]["rank"] == 2
    
    assert results[2]["chunk_id"] == "chunk-3"
    assert results[2]["rank"] == 3
    
def test_cross_encoder():
    # 1. Setup Neural Re-Ranker
    reranker = CrossEncoderReRanker()
    
    query = "How to normalize data"
    
    # Feed Mock RRF results containing a false positive
    rrf_results = [
        {"chunk_id": "chunk-1", "rank": 1},
        {"chunk_id": "chunk-2", "rank": 2}
    ]
    
    chunk_contents = {
        "chunk-1": "def normalize(data): return data.lower()",
        "chunk-2": "def unrelated(): pass"
    }
    
    # 2. Re-Rank
    results = reranker.rerank(query, rrf_results, chunk_contents, top_k=1)
    
    # 3. Assert neural alignment drops the false positive perfectly
    assert len(results) == 1
    assert results[0]["chunk_id"] == "chunk-1"
    assert results[0]["rank"] == 1
    assert "cross_encoder_score" in results[0]

def test_rrf_top_30_truncation():
    # Simulate a dense retriever returning 40 results
    dense = [{"chunk_id": f"chunk-{i}", "rank": i} for i in range(1, 41)]
    rrf = ReciprocalRankFusion(k=60)
    
    # Test that default truncation precisely cuts at Top-30 per architectural requirement
    results = rrf.fuse(dense, [], [])
    
    assert len(results) == 30
    assert results[-1]["chunk_id"] == "chunk-30"
