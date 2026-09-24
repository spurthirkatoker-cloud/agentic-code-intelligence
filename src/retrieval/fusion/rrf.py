from typing import List, Dict, Any
from collections import defaultdict

class ReciprocalRankFusion:
    def __init__(self, k: int = 60):
        self.k = k
        
    def fuse(self, dense_results: List[Dict[str, Any]], bm25_results: List[Dict[str, Any]], structural_results: List[Dict[str, Any]], top_k: int = 30) -> List[Dict[str, Any]]:
        """Combine results from multiple retrievers using Reciprocal Rank Fusion (RRF)."""
        chunk_scores: Dict[str, float] = defaultdict(float)
        
        # Helper to apply the RRF scoring formula: 1 / (k + rank)
        def apply_rrf(results: List[Dict[str, Any]]):
            for result in results:
                chunk_id = result["chunk_id"]
                rank = result["rank"]
                chunk_scores[chunk_id] += 1.0 / (self.k + rank)
                
        # Blend the three modalities
        apply_rrf(dense_results)
        apply_rrf(bm25_results)
        apply_rrf(structural_results)
        
        # Sort strictly by the aggregated RRF score descending
        sorted_chunks = sorted(chunk_scores.items(), key=lambda x: x[1], reverse=True)
        
        fused_results = []
        for new_rank, (chunk_id, score) in enumerate(sorted_chunks[:top_k], start=1):
            fused_results.append({
                "chunk_id": chunk_id,
                "rank": new_rank,
                "rrf_score": score
            })
            
        return fused_results
