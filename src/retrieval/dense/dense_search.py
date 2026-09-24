import faiss
from typing import List, Dict, Any
from .embedder import Embedder
from .faiss_index import DenseIndex

class DenseRetriever:
    def __init__(self, embedder: Embedder, index: DenseIndex):
        self.embedder = embedder
        self.index = index
        
    def search(self, query: str, top_k: int = 50) -> List[Dict[str, Any]]:
        if self.index.index.ntotal == 0:
            return []
            
        # Encode the natural language query
        query_embedding = self.embedder.encode([query])
        
        # Normalize the query to properly align with the normalized index vectors (cosine similarity)
        faiss.normalize_L2(query_embedding)
        
        actual_k = min(top_k, self.index.index.ntotal)
        
        scores, faiss_ids = self.index.index.search(query_embedding, actual_k)
        
        results = []
        for rank, (score, faiss_id) in enumerate(zip(scores[0], faiss_ids[0]), start=1):
            if faiss_id == -1:
                continue
                
            chunk_id = self.index.faiss_id_to_chunk_id.get(faiss_id)
            if chunk_id:
                results.append({
                    "chunk_id": chunk_id,
                    "rank": rank,
                    "score": float(score)
                })
                
        return results
