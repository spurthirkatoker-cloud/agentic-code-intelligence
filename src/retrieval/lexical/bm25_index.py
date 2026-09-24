from rank_bm25 import BM25Okapi
import pickle
import os
import re
from typing import List, Dict, Any

class BM25Index:
    def __init__(self):
        self.bm25: BM25Okapi = None
        self.chunk_ids: List[str] = []
        
    def _tokenize(self, text: str) -> List[str]:
        """Simple tokenization splitting on non-word characters and lowercasing."""
        tokens = re.split(r'\W+', text.lower())
        return [t for t in tokens if t]
        
    def add_chunks(self, chunks: List[Dict[str, Any]]):
        """Build the BM25 index from code chunks."""
        corpus = []
        self.chunk_ids = []
        
        for chunk in chunks:
            # Combine all useful lexical information into a single text representation
            lexical_text = []
            if chunk.get("name"): lexical_text.append(chunk["name"])
            if chunk.get("parent_class"): lexical_text.append(chunk["parent_class"])
            if chunk.get("file_path"): lexical_text.append(chunk["file_path"])
            if chunk.get("imports"): lexical_text.extend(chunk["imports"])
            if chunk.get("calls"): lexical_text.extend(chunk["calls"])
            if chunk.get("code"): lexical_text.append(chunk["code"])
            if chunk.get("representation"): lexical_text.append(chunk["representation"])
            
            combined_text = " ".join(lexical_text)
            corpus.append(self._tokenize(combined_text))
            self.chunk_ids.append(chunk["chunk_id"])
            
        if corpus:
            self.bm25 = BM25Okapi(corpus)
            
    def save(self, index_path: str):
        if not self.bm25:
            return
        with open(index_path, 'wb') as f:
            pickle.dump({
                "bm25": self.bm25,
                "chunk_ids": self.chunk_ids
            }, f)
            
    def load(self, index_path: str):
        if os.path.exists(index_path):
            with open(index_path, 'rb') as f:
                data = pickle.load(f)
                self.bm25 = data["bm25"]
                self.chunk_ids = data["chunk_ids"]
