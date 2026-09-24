import math
from typing import Dict, List

class RetrievalEvaluator:
    @staticmethod
    def compute_mrr(qrels: Dict[str, List[str]], predictions: Dict[str, List[str]]) -> float:
        """
        Computes the Mean Reciprocal Rank (MRR).
        Evaluates how far down the ranked list the FIRST relevant document appears.
        """
        if not qrels:
            return 0.0
            
        mrr_sum = 0.0
        query_count = 0
        
        for qid, rel_docs in qrels.items():
            if not rel_docs:
                continue
                
            query_count += 1
            pred_docs = predictions.get(qid, [])
            
            # Find the rank of the first explicitly relevant document
            rank = 0
            for i, doc_id in enumerate(pred_docs):
                if doc_id in rel_docs:
                    rank = i + 1
                    break
                    
            if rank > 0:
                mrr_sum += 1.0 / rank
                
        return mrr_sum / query_count if query_count > 0 else 0.0

    @staticmethod
    def compute_ndcg_at_k(qrels: Dict[str, List[str]], predictions: Dict[str, List[str]], k: int = 10) -> float:
        """
        Computes the Normalized Discounted Cumulative Gain at K (NDCG@K).
        Assumes binary relevance bounds (score = 1 for relevant, 0 for non-relevant).
        """
        if not qrels:
            return 0.0
            
        ndcg_sum = 0.0
        query_count = 0
        
        for qid, rel_docs in qrels.items():
            if not rel_docs:
                continue
                
            query_count += 1
            pred_docs = predictions.get(qid, [])[:k]
            
            # Compute actual DCG for system predictions
            dcg = 0.0
            for i, doc_id in enumerate(pred_docs):
                if doc_id in rel_docs:
                    dcg += 1.0 / math.log2(i + 2)
                    
            # Compute Ideal DCG (IDCG) - what if all relevant docs were packed perfectly at the top?
            idcg = 0.0
            ideal_hits = min(len(rel_docs), k)
            for i in range(ideal_hits):
                idcg += 1.0 / math.log2(i + 2)
                
            if idcg > 0:
                ndcg_sum += dcg / idcg
                
        return ndcg_sum / query_count if query_count > 0 else 0.0

    @classmethod
    def evaluate(cls, qrels: Dict[str, List[str]], predictions: Dict[str, List[str]]) -> Dict[str, float]:
        """
        Executes the full evaluation suite yielding strictly standard IR metrics.
        """
        return {
            "MRR": cls.compute_mrr(qrels, predictions),
            "NDCG@10": cls.compute_ndcg_at_k(qrels, predictions, k=10)
        }
