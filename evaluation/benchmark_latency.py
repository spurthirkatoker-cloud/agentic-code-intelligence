import time
import numpy as np
from typing import List, Dict, Any
from src.retrieval.pipeline import RetrievalPipeline

class LatencyBenchmark:
    def __init__(self, pipeline: RetrievalPipeline):
        self.pipeline = pipeline
        self.metrics = {
            "query_preprocessing": [],
            "dense_retrieval": [],
            "bm25_retrieval": [],
            "ast_retrieval": [],
            "rrf_fusion": [],
            "cross_encoder": [],
            "total_latency": []
        }

    def run_benchmark(self, queries: List[str]):
        """Executes a payload of queries while intercepting execution to measure micro-latencies."""
        # Monkey patch the internal modules to perfectly intercept layer-by-layer execution times
        original_expand = self.pipeline.orchestrator.expander.expand
        original_dense = self.pipeline.orchestrator.dense_retriever.search
        original_bm25 = self.pipeline.orchestrator.bm25_retriever.search
        original_ast = self.pipeline.orchestrator.structural_retriever.search
        original_rrf = self.pipeline.rrf.fuse
        original_cross_encoder = self.pipeline.cross_encoder.rerank
        
        def timed_expand(q):
            start = time.perf_counter()
            res = original_expand(q)
            self.metrics["query_preprocessing"].append((time.perf_counter() - start) * 1000)
            return res
            
        def timed_dense(q, top_k):
            start = time.perf_counter()
            res = original_dense(q, top_k=top_k)
            self.metrics["dense_retrieval"].append((time.perf_counter() - start) * 1000)
            return res
            
        def timed_bm25(q, top_k):
            start = time.perf_counter()
            res = original_bm25(q, top_k=top_k)
            self.metrics["bm25_retrieval"].append((time.perf_counter() - start) * 1000)
            return res
            
        def timed_ast(q, top_k):
            start = time.perf_counter()
            res = original_ast(q, top_k=top_k)
            self.metrics["ast_retrieval"].append((time.perf_counter() - start) * 1000)
            return res
            
        def timed_rrf(dense_results, bm25_results, structural_results, top_k):
            start = time.perf_counter()
            res = original_rrf(dense_results, bm25_results, structural_results, top_k=top_k)
            self.metrics["rrf_fusion"].append((time.perf_counter() - start) * 1000)
            return res
            
        def timed_cross_encoder(query, rrf_results, chunk_contents, top_k):
            start = time.perf_counter()
            res = original_cross_encoder(query, rrf_results, chunk_contents, top_k=top_k)
            self.metrics["cross_encoder"].append((time.perf_counter() - start) * 1000)
            return res

        # Apply measurement wrappers
        self.pipeline.orchestrator.expander.expand = timed_expand
        self.pipeline.orchestrator.dense_retriever.search = timed_dense
        self.pipeline.orchestrator.bm25_retriever.search = timed_bm25
        self.pipeline.orchestrator.structural_retriever.search = timed_ast
        self.pipeline.rrf.fuse = timed_rrf
        self.pipeline.cross_encoder.rerank = timed_cross_encoder

        try:
            for q in queries:
                t0 = time.perf_counter()
                self.pipeline.retrieve(q)
                self.metrics["total_latency"].append((time.perf_counter() - t0) * 1000)
        finally:
            # Safely restore original pipeline methods
            self.pipeline.orchestrator.expander.expand = original_expand
            self.pipeline.orchestrator.dense_retriever.search = original_dense
            self.pipeline.orchestrator.bm25_retriever.search = original_bm25
            self.pipeline.orchestrator.structural_retriever.search = original_ast
            self.pipeline.rrf.fuse = original_rrf
            self.pipeline.cross_encoder.rerank = original_cross_encoder

    def get_report(self) -> Dict[str, Dict[str, float]]:
        """Calculates Average and P95 latency stats across the entire run."""
        report = {}
        for metric, times in self.metrics.items():
            if not times:
                continue
            arr = np.array(times)
            report[metric] = {
                "average_ms": round(float(np.mean(arr)), 2),
                "p95_ms": round(float(np.percentile(arr, 95)), 2)
            }
        return report
