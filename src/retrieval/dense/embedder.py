from sentence_transformers import SentenceTransformer
from typing import List
import numpy as np

class Embedder:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        # Run on CPU explicitly as required by the architecture specification
        self.model = SentenceTransformer(model_name, device="cpu")
        
    def encode(self, texts: List[str]) -> np.ndarray:
        """Encode a list of text strings into a numpy array of dense vectors."""
        embeddings = self.model.encode(texts, convert_to_numpy=True)
        return embeddings
