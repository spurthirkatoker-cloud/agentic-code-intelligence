import os
import tempfile
from src.versioning.git_manager import GitManager
from src.versioning.versioned_index import VersionedIndexManager

def test_git_manager():
    # Test against our own scratch repository to see if it reads git native state
    repo_path = os.getcwd()
    manager = GitManager(repo_path)
    
    info = manager.get_current_version_info()
    assert "repository" in info
    assert "branch" in info
    assert "commit_hash" in info
    assert "commit_date" in info
    assert info["repository"] == "code-intelligence"

def test_versioned_index_manager():
    with tempfile.TemporaryDirectory() as temp_dir:
        manager = VersionedIndexManager(base_indexes_dir=temp_dir)
        
        # Request paths for a specific dummy commit
        paths = manager.get_paths("abc123hash")
        
        assert "abc123hash" in paths["faiss_index_path"]
        assert "abc123hash" in paths["bm25_index_path"]
        assert "abc123hash" in paths["metadata_db_path"]
        assert "abc123hash" in paths["structural_index_path"]
        
        # Verify the isolated directory was successfully created
        assert os.path.exists(os.path.dirname(paths["faiss_index_path"]))
