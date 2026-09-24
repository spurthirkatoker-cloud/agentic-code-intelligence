import os
import tempfile
from src.storage.metadata_store import MetadataStore

def test_metadata_store():
    with tempfile.TemporaryDirectory() as temp_dir:
        db_path = os.path.join(temp_dir, "test.db")
        store = MetadataStore(db_path)
        
        metadata = {
            "chunk_id": "chunk-123",
            "repository_id": "repo-1",
            "file_path": "src/utils.py",
            "language": "python",
            "chunk_type": "function",
            "class_name": None,
            "function_name": "normalize",
            "start_line": 10,
            "end_line": 20,
            "imports": ["import os"],
            "function_calls": ["strip"],
            "code": "def normalize(text): return text.strip()",
            "git_commit": "abc1234"
        }
        
        representation = "Function: normalize\nFile: src/utils.py\nCode: ..."
        
        # 1. Test insert
        store.insert_chunk(metadata, representation)
        
        # 2. Test get by ID
        result = store.get_by_chunk_id("chunk-123")
        assert result is not None
        assert result["chunk_id"] == "chunk-123"
        assert result["function_name"] == "normalize"
        assert result["function_calls"] == ["strip"]
        assert result["imports"] == ["import os"]
        assert result["git_commit"] == "abc1234"
        assert result["representation"] == representation
        
        # 3. Test search by function
        results = store.search_by_function("normalize")
        assert len(results) == 1
        assert results[0]["chunk_id"] == "chunk-123"
        
        # 4. Test search by file
        file_results = store.search_by_file("src/utils.py")
        assert len(file_results) == 1
        
        # 5. Test missing chunk
        assert store.get_by_chunk_id("non-existent") is None
