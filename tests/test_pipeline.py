import os
import tempfile
from src.retrieval.expansion.query_expander import QueryExpander
from src.retrieval.expansion.orchestrator import RetrievalOrchestrator
from src.retrieval.dense.embedder import Embedder
from src.retrieval.dense.faiss_index import DenseIndex
from src.retrieval.dense.dense_search import DenseRetriever
from src.retrieval.lexical.bm25_index import BM25Index
from src.retrieval.lexical.bm25_search import BM25Retriever
from src.retrieval.structural.structural_index import StructuralIndex
from src.retrieval.structural.structural_search import StructuralRetriever
from src.retrieval.fusion.rrf import ReciprocalRankFusion
from src.retrieval.fusion.cross_encoder import CrossEncoderReRanker
from src.storage.metadata_store import MetadataStore
from src.retrieval.pipeline import RetrievalPipeline

def test_full_pipeline_reranking():
    with tempfile.TemporaryDirectory() as temp_dir:
        db_path = os.path.join(temp_dir, "test.db")
        store = MetadataStore(db_path)
        
        # Generate 40 dummy chunks to prove Top-50 -> Top-30 -> Top-10 funnel
        chunks = []
        representations = []
        for i in range(1, 41):
            chunk = {
                "chunk_id": f"chunk-{i}",
                "name": f"func_{i}",
                "code": f"def func_{i}(): pass",
                "file_path": f"src/file_{i}.py",
                "type": "function",
                "calls": []
            }
            # Inject a heavily relevant chunk to test end-to-end Neural promotion
            if i == 13:
                chunk["code"] = "def normalize(data): return data.lower()"
                chunk["name"] = "normalize"
                rep = "Function: normalize\nFile: src/file_13.py\nCode:\ndef normalize(data): return data.lower()"
            else:
                rep = f"Function: func_{i}\nFile: src/file_{i}.py\nCode:\ndef func_{i}(): pass"
                
            chunks.append(chunk)
            representations.append(rep)
            
        store.insert_chunks(chunks, representations)
        
        # 1. Setup Dense Pipeline
        embedder = Embedder()
        dense_idx = DenseIndex(384)
        embeddings = embedder.encode([c["code"] for c in chunks])
        dense_idx.add_embeddings(embeddings, [c["chunk_id"] for c in chunks])
        dense_retriever = DenseRetriever(embedder, dense_idx)
        
        # 2. Setup BM25 Pipeline
        bm25_idx = BM25Index()
        bm25_idx.add_chunks(chunks)
        bm25_retriever = BM25Retriever(bm25_idx)
        
        # 3. Setup Structural Pipeline
        struct_idx = StructuralIndex()
        struct_idx.add_chunks(chunks)
        struct_retriever = StructuralRetriever(struct_idx)
        
        # 4. Orchestrator
        expander = QueryExpander()
        orchestrator = RetrievalOrchestrator(
            expander, dense_retriever, bm25_retriever, struct_retriever
        )
        
        # 5. Fusion & Reranking Models
        rrf = ReciprocalRankFusion(k=60)
        cross_encoder = CrossEncoderReRanker()
        
        # 6. Instantiate Master Pipeline
        pipeline = RetrievalPipeline(
            orchestrator, rrf, cross_encoder, store
        )
        
        # Execute query
        results = pipeline.retrieve("How to normalize data")
        
        # Verify Top-10 output (Funnel: 40 -> RRF 30 -> Cross-Encoder 10)
        assert len(results) == 10
        
        # Verify chunk-13 was successfully promoted by semantic neural networks to Rank 1
        assert results[0]["chunk_id"] == "chunk-13"
        assert "cross_encoder_score" in results[0]
