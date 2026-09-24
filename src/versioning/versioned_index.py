import os
from typing import Dict

class VersionedIndexManager:
    def __init__(self, base_indexes_dir: str = "indexes"):
        self.base_indexes_dir = base_indexes_dir
        os.makedirs(self.base_indexes_dir, exist_ok=True)
        
    def get_index_path_for_commit(self, commit_hash: str) -> str:
        """Returns the isolated index directory for a specific commit."""
        commit_dir = os.path.join(self.base_indexes_dir, f"commit_{commit_hash}")
        os.makedirs(commit_dir, exist_ok=True)
        return commit_dir
        
    def get_paths(self, commit_hash: str) -> Dict[str, str]:
        """Returns the specific file paths for FAISS, BM25, and SQLite for a given commit."""
        base = self.get_index_path_for_commit(commit_hash)
        return {
            "faiss_index_path": os.path.join(base, "faiss.index"),
            "bm25_index_path": os.path.join(base, "bm25.pkl"),
            "structural_index_path": os.path.join(base, "structural.pkl"),
            "metadata_db_path": os.path.join(base, "metadata.db")
        }
