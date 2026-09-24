import os
import tempfile
from src.retrieval.lexical.bm25_index import BM25Index
from src.retrieval.lexical.bm25_search import BM25Retriever

def test_bm25_retrieval_pipeline():
    # 1. Setup Data
    chunks = [
        {
            "chunk_id": "chunk-1",
            "name": "normalize",
            "code": "def normalize(text): return text.strip().lower()",
            "file_path": "src/preprocessing.py"
        },
        {
            "chunk_id": "chunk-2",
            "name": "validate_token",
            "code": "def validate_token(token): return True",
            "file_path": "src/auth.py",
            "imports": ["import jwt"]
        },
        {
            "chunk_id": "chunk-3",
            "name": "calculate_loss",
            "code": "def calculate_loss(pred, target): pass",
            "file_path": "src/math.py",
            "calls": ["mean", "pow"]
        }
    ]
    
    # 2. Build BM25 Index
    with tempfile.TemporaryDirectory() as temp_dir:
        index_path = os.path.join(temp_dir, "bm25.pkl")
        
        bm25_index = BM25Index()
        bm25_index.add_chunks(chunks)
        
        assert len(bm25_index.chunk_ids) == 3
        
        bm25_index.save(index_path)
        
        loaded_index = BM25Index()
        loaded_index.load(index_path)
        
        assert len(loaded_index.chunk_ids) == 3
        
        # 3. BM25 Search Execution
        retriever = BM25Retriever(loaded_index)
        
        # Query matching exact identifier
        results = retriever.search("Where is validate_token implemented?", top_k=2)
        
        assert len(results) > 0
        assert results[0]["chunk_id"] == "chunk-2"
        assert results[0]["rank"] == 1
        assert "score" in results[0]
        
        # Query matching call
        results2 = retriever.search("What calls mean?", top_k=1)
        assert len(results2) > 0
        assert results2[0]["chunk_id"] == "chunk-3"
