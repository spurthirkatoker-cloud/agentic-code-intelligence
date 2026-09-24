from typing import List, Dict, Any
from sentence_transformers import CrossEncoder as STCrossEncoder

class CrossEncoderReRanker:
    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        self.model_name = model_name
        # Run on CPU explicitly as required by architectural constraints
        self.model = STCrossEncoder(model_name, device="cpu")
        
    def rerank(self, query: str, rrf_results: List[Dict[str, Any]], chunk_contents: Dict[str, str], top_k: int = 10) -> List[Dict[str, Any]]:
        """
        Re-rank the top chunks based on deep cross-encoder semantics.
        chunk_contents is a dictionary mapping chunk_id -> raw textual code representation string.
        """
        if not rrf_results:
            return []
            
        # Build evaluation pairs: (query, text_representation)
        pairs = []
        valid_chunks = []
        
        for result in rrf_results:
            chunk_id = result["chunk_id"]
            if chunk_id in chunk_contents:
                pairs.append((query, chunk_contents[chunk_id]))
                valid_chunks.append(chunk_id)
                
        if not pairs:
            return []
            
        # Perform deep neural cross-encoding and predict similarity
        scores = self.model.predict(pairs)
        
        # Bind the resulting float scores back to chunk_ids
        scored_chunks = list(zip(valid_chunks, scores))
        scored_chunks.sort(key=lambda x: x[1], reverse=True)
        
        final_results = []
        for rank, (chunk_id, score) in enumerate(scored_chunks[:top_k], start=1):
            final_results.append({
                "chunk_id": chunk_id,
                "rank": rank,
                "cross_encoder_score": float(score)
            })
            
        return final_results
