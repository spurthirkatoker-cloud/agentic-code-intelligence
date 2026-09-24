import os
import tempfile
from src.retrieval.dense.embedder import Embedder
from src.retrieval.dense.faiss_index import DenseIndex
from src.retrieval.dense.dense_search import DenseRetriever

def test_dense_retrieval_pipeline():
    # 1. Setup Embedder
    embedder = Embedder()
    
    docs = [
        "def normalize(text): return text.strip().lower()",
        "def calculate_loss(predictions, targets): return np.mean((predictions - targets)**2)",
        "class Model:\n    def forward(self, x): return x"
    ]
    chunk_ids = ["chunk-1", "chunk-2", "chunk-3"]
    
    embeddings = embedder.encode(docs)
    assert embeddings.shape == (3, 384)
    
    # 2. Setup FAISS Index
    with tempfile.TemporaryDirectory() as temp_dir:
        index_path = os.path.join(temp_dir, "faiss.index")
        mapping_path = os.path.join(temp_dir, "mapping.pkl")
        
        dense_index = DenseIndex(embedding_dimension=384)
        dense_index.add_embeddings(embeddings, chunk_ids)
        
        assert dense_index.index.ntotal == 3
        dense_index.save(index_path, mapping_path)
        
        loaded_index = DenseIndex(embedding_dimension=384)
        loaded_index.load(index_path, mapping_path)
        
        assert loaded_index.index.ntotal == 3
        
        # 3. Dense Search Execution
        retriever = DenseRetriever(embedder, loaded_index)
        
        # This query cleanly aligns semantically with chunk-2
        results = retriever.search("How is the loss computed?", top_k=2)
        
        assert len(results) == 2
        # Semantic search should confidently push chunk-2 to Rank 1
        assert results[0]["chunk_id"] == "chunk-2"
        assert results[0]["rank"] == 1
        assert "score" in results[0]
