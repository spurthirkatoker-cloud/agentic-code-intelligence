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
from src.search.search_service import SearchService

def test_search_service():
    with tempfile.TemporaryDirectory() as temp_dir:
        db_path = os.path.join(temp_dir, "test_search.db")
        store = MetadataStore(db_path)
        
        # 1. Insert Dummy Chunks
        chunks = [
            {
                "chunk_id": "chunk-1",
                "name": "normalize",
                "code": "def normalize(data): return data.lower()",
                "file_path": "src/utils.py",
                "type": "function",
                "calls": ["lower"]
            },
            {
                "chunk_id": "chunk-2",
                "name": "unrelated",
                "code": "def unrelated(): pass",
                "file_path": "src/unrelated.py",
                "type": "function",
                "calls": []
            }
        ]
        reps = [
            "Function: normalize\nFile: src/utils.py\nCode:\ndef normalize(data): return data.lower()",
            "Function: unrelated\nFile: src/unrelated.py\nCode:\ndef unrelated(): pass"
        ]
        store.insert_chunks(chunks, reps)
        
        # 2. Build Pipeline
        embedder = Embedder()
        dense_idx = DenseIndex(384)
        embeddings = embedder.encode([c["code"] for c in chunks])
        dense_idx.add_embeddings(embeddings, [c["chunk_id"] for c in chunks])
        dense_retriever = DenseRetriever(embedder, dense_idx)
        
        bm25_idx = BM25Index()
        bm25_idx.add_chunks(chunks)
        bm25_retriever = BM25Retriever(bm25_idx)
        
        struct_idx = StructuralIndex()
        struct_idx.add_chunks(chunks)
        struct_retriever = StructuralRetriever(struct_idx)
        
        expander = QueryExpander()
        orchestrator = RetrievalOrchestrator(expander, dense_retriever, bm25_retriever, struct_retriever)
        rrf = ReciprocalRankFusion(k=60)
        cross_encoder = CrossEncoderReRanker()
        
        pipeline = RetrievalPipeline(orchestrator, rrf, cross_encoder, store)
        
        # 3. Instantiate Service
        search_service = SearchService(pipeline, store)
        
        # --- Test 1: Valid Execution and Enrichment ---
        results = search_service.search("How to normalize data")
        assert len(results) > 0
        
        # Assert schema enrichment successfully bonded SQLite data back to the reranked objects
        top_result = results[0]
        assert top_result["chunk_id"] == "chunk-1"
        assert top_result["file_path"] == "src/utils.py"
        assert "score" in top_result
        assert "code" in top_result
        assert "representation" in top_result
        
        # --- Test 2: Input Validation ---
        empty_results = search_service.search("   ")
        assert empty_results == []
        
        none_results = search_service.search(None)
        assert none_results == []
