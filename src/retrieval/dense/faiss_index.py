import faiss
import numpy as np
import pickle
import os
from typing import List, Dict

class DenseIndex:
    def __init__(self, embedding_dimension: int = 384):
        self.dimension = embedding_dimension
        # Inner Product correlates exactly to Cosine Similarity when vectors are L2-normalized
        self.index = faiss.IndexFlatIP(self.dimension)
        self.faiss_id_to_chunk_id: Dict[int, str] = {}
        
    def add_embeddings(self, embeddings: np.ndarray, chunk_ids: List[str]):
        if len(embeddings) != len(chunk_ids):
            raise ValueError("Number of embeddings must match number of chunk IDs.")
            
        if len(embeddings) == 0:
            return
            
        # Normalize vectors for accurate cosine similarity inside IndexFlatIP
        faiss.normalize_L2(embeddings)
        
        start_id = self.index.ntotal
        self.index.add(embeddings)
        
        for i, chunk_id in enumerate(chunk_ids):
            self.faiss_id_to_chunk_id[start_id + i] = chunk_id
            
    def save(self, index_path: str, mapping_path: str):
        faiss.write_index(self.index, index_path)
        with open(mapping_path, 'wb') as f:
            pickle.dump(self.faiss_id_to_chunk_id, f)
            
    def load(self, index_path: str, mapping_path: str):
        if os.path.exists(index_path) and os.path.exists(mapping_path):
            self.index = faiss.read_index(index_path)
            with open(mapping_path, 'rb') as f:
                self.faiss_id_to_chunk_id = pickle.load(f)
