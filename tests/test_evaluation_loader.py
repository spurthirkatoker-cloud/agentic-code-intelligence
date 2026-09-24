import pytest
from unittest.mock import patch
from evaluation.dataset_loader import EvaluationDatasetLoader

class MockSearchService:
    def search(self, query: str):
        # Mock predictable behavior returning fake schema-compliant data
        if "normalize" in query.lower():
            return [{"chunk_id": "doc1", "rank": 1}, {"chunk_id": "doc2", "rank": 2}]
        return [{"chunk_id": "doc3", "rank": 1}]

@patch("evaluation.dataset_loader.load_dataset")
def test_dataset_loader(mock_load_dataset):
    # Safely mock the Hugging Face API to prevent downloading multi-GB datasets during unit testing
    def mock_loader(path, name, split):
        if name == "queries":
            return [
                {"_id": "q1", "text": "How to normalize data?"},
                {"_id": "q2", "text": "How to save data?"}
            ]
        elif name == "qrels":
            return [
                {"query-id": "q1", "corpus-id": "doc1", "score": 1.0},
                {"query-id": "q1", "corpus-id": "doc2", "score": 1.0},
                {"query-id": "q2", "corpus-id": "doc3", "score": 1.0},
                {"query-id": "q2", "corpus-id": "doc4", "score": 0.0} # Should be correctly filtered out
            ]
            
    mock_load_dataset.side_effect = mock_loader
    
    # 1. Test Dataset Loading Logic
    loader = EvaluationDatasetLoader("mock/path")
    queries, qrels = loader.load_data(split="test")
    
    assert len(queries) == 2
    assert queries[0]["query_id"] == "q1"
    assert queries[0]["text"] == "How to normalize data?"
    
    assert len(qrels) == 2
    assert "q1" in qrels
    assert "doc1" in qrels["q1"]
    assert "doc2" in qrels["q1"]
    
    assert "q2" in qrels
    assert "doc3" in qrels["q2"]
    assert "doc4" not in qrels["q2"] # Proves irrelevant data is ignored
    
    # 2. Test Execution Engine against API Wrapper
    service = MockSearchService()
    predictions = loader.execute_queries_against_service(queries, service)
    
    assert "q1" in predictions
    assert "q2" in predictions
    assert predictions["q1"] == ["doc1", "doc2"]
    assert predictions["q2"] == ["doc3"]
