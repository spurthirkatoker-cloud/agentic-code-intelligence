import numpy as np
from typing import List, Dict, Any
from .bm25_index import BM25Index

class BM25Retriever:
    def __init__(self, index: BM25Index):
        self.index = index
        
    def search(self, query: str, top_k: int = 50) -> List[Dict[str, Any]]:
        if not self.index.bm25 or not self.index.chunk_ids:
            return []
            
        tokenized_query = self.index._tokenize(query)
        
        # Get raw BM25 scores
        scores = self.index.bm25.get_scores(tokenized_query)
        
        # Sort indices by score in descending order
        top_indices = np.argsort(scores)[::-1][:top_k]
        
        results = []
        rank = 1
        for idx in top_indices:
            score = float(scores[idx])
            if score > 0: # Only return results with non-zero BM25 score
                results.append({
                    "chunk_id": self.index.chunk_ids[idx],
                    "rank": rank,
                    "score": score
                })
                rank += 1
                
        return results
