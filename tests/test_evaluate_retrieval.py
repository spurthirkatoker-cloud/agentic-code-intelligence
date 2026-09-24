import pytest
from evaluation.evaluate_retrieval import RetrievalEvaluator

def test_evaluate_mrr():
    qrels = {
        "q1": ["docA", "docB"],
        "q2": ["docC"],
        "q3": ["docD"]
    }
    
    predictions = {
        "q1": ["docX", "docA"], # rank 2 -> 1/2
        "q2": ["docC"],         # rank 1 -> 1/1
        "q3": ["docY", "docZ"]  # rank 0 -> 0
    }
    
    # Expected MRR = (0.5 + 1.0 + 0) / 3 = 1.5 / 3 = 0.5
    mrr = RetrievalEvaluator.compute_mrr(qrels, predictions)
    assert abs(mrr - 0.5) < 1e-6

def test_evaluate_ndcg_at_10():
    qrels = {
        "q1": ["docA", "docB", "docC"]
    }
    
    predictions = {
        "q1": ["docA", "docX", "docB", "docY", "docZ"]
    }
    
    # DCG calculation:
    # rank 1 (docA) -> 1/log2(2) = 1.0
    # rank 3 (docB) -> 1/log2(4) = 0.5
    # Total DCG = 1.5
    
    # IDCG calculation (best case rank 1, 2, 3):
    # rank 1 -> 1.0
    # rank 2 -> 1/log2(3) ≈ 0.63093
    # rank 3 -> 0.5
    # Total IDCG ≈ 2.13093
    
    # Expected NDCG = 1.5 / 2.13093 ≈ 0.703918
    ndcg = RetrievalEvaluator.compute_ndcg_at_k(qrels, predictions, k=10)
    assert abs(ndcg - 0.703918) < 1e-5

def test_empty_and_missing_edge_cases():
    qrels = {"q1": [], "q2": ["docA"]}
    predictions = {"q1": ["docX"]} # missing q2 prediction entirely
    
    results = RetrievalEvaluator.evaluate(qrels, predictions)
    
    # q1 has no relevant docs so skipped perfectly.
    # q2 has relevant docs but no predictions submitted by the pipeline -> 0.
    assert results["MRR"] == 0.0
    assert results["NDCG@10"] == 0.0
