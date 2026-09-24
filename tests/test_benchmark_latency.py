import tempfile
import os
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
from evaluation.benchmark_latency import LatencyBenchmark

def test_latency_benchmark():
    with tempfile.TemporaryDirectory() as temp_dir:
        db_path = os.path.join(temp_dir, "test.db")
        store = MetadataStore(db_path)
        
        chunks = [{"chunk_id": "c1", "code": "def normalize(): pass", "type": "function", "name": "normalize"}]
        store.insert_chunks(chunks, ["rep"])
        
        embedder = Embedder()
        dense_idx = DenseIndex(384)
        dense_idx.add_embeddings(embedder.encode(["def normalize(): pass"]), ["c1"])
        dense_retriever = DenseRetriever(embedder, dense_idx)
        
        bm25_idx = BM25Index()
        bm25_idx.add_chunks(chunks)
        bm25_retriever = BM25Retriever(bm25_idx)
        
        struct_idx = StructuralIndex()
        struct_idx.add_chunks(chunks)
        struct_retriever = StructuralRetriever(struct_idx)
        
        expander = QueryExpander()
        orchestrator = RetrievalOrchestrator(expander, dense_retriever, bm25_retriever, struct_retriever)
        rrf = ReciprocalRankFusion()
        cross_encoder = CrossEncoderReRanker()
        
        pipeline = RetrievalPipeline(orchestrator, rrf, cross_encoder, store)
        
        # 1. Execute Benchmark Suite
        benchmark = LatencyBenchmark(pipeline)
        benchmark.run_benchmark(["How to normalize data?", "Where is the API?"])
        
        report = benchmark.get_report()
        
        # 2. Assert component-level measurement tracking
        assert "query_preprocessing" in report
        assert "dense_retrieval" in report
        assert "bm25_retrieval" in report
        assert "ast_retrieval" in report
        assert "rrf_fusion" in report
        assert "cross_encoder" in report
        assert "total_latency" in report
        
        # 3. Assert mathematical aggregation
        assert "average_ms" in report["total_latency"]
        assert "p95_ms" in report["total_latency"]
        
        # Ensure measurements actually captured time
        assert report["total_latency"]["average_ms"] > 0.0
