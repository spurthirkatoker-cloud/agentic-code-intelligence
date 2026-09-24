import os
import tempfile
from src.retrieval.structural.structural_index import StructuralIndex
from src.retrieval.structural.structural_search import StructuralRetriever

def test_ast_retrieval_pipeline():
    # 1. Setup Data
    chunks = [
        {
            "chunk_id": "chunk-1",
            "name": "MyClass",
            "type": "class",
            "file_path": "src/models.py"
        },
        {
            "chunk_id": "chunk-2",
            "name": "forward",
            "type": "method",
            "parent_class": "MyClass",
            "calls": ["linear", "relu"],
            "imports": ["import torch.nn as nn"],
            "file_path": "src/models.py"
        },
        {
            "chunk_id": "chunk-3",
            "name": "train_model",
            "type": "function",
            "calls": ["forward", "backward", "step"],
            "file_path": "src/train.py"
        }
    ]
    
    # 2. Build Structural Index
    with tempfile.TemporaryDirectory() as temp_dir:
        index_path = os.path.join(temp_dir, "ast.pkl")
        
        ast_index = StructuralIndex()
        ast_index.add_chunks(chunks)
        
        assert len(ast_index.chunk_ids) == 3
        assert "chunk-2" in ast_index.calls_to_chunks["relu"]
        assert "chunk-3" in ast_index.calls_to_chunks["forward"]
        assert "chunk-2" in ast_index.class_to_methods["MyClass"]
        
        ast_index.save(index_path)
        
        loaded_index = StructuralIndex()
        loaded_index.load(index_path)
        
        assert len(loaded_index.chunk_ids) == 3
        
        # 3. AST Search Execution
        retriever = StructuralRetriever(loaded_index)
        
        # A query explicitly utilizing structural function call hints
        results1 = retriever.search("Which method calls relu?", top_k=2)
        assert len(results1) > 0
        assert results1[0]["chunk_id"] == "chunk-2"
        
        # A query utilizing class relationship hints
        results2 = retriever.search("What are the methods in class MyClass?", top_k=2)
        assert len(results2) > 0
        top_cids = [r["chunk_id"] for r in results2]
        assert "chunk-2" in top_cids
        
        # A query utilizing function namespace hints
        results3 = retriever.search("Where is function train_model?", top_k=1)
        assert len(results3) > 0
        assert results3[0]["chunk_id"] == "chunk-3"
