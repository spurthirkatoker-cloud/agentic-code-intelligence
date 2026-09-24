from typing import List, Dict, Any
from src.retrieval.expansion.orchestrator import RetrievalOrchestrator
from src.retrieval.fusion.rrf import ReciprocalRankFusion
from src.retrieval.fusion.cross_encoder import CrossEncoderReRanker
from src.storage.metadata_store import MetadataStore

class RetrievalPipeline:
    def __init__(
        self,
        orchestrator: RetrievalOrchestrator,
        rrf: ReciprocalRankFusion,
        cross_encoder: CrossEncoderReRanker,
        metadata_store: MetadataStore
    ):
        self.orchestrator = orchestrator
        self.rrf = rrf
        self.cross_encoder = cross_encoder
        self.metadata_store = metadata_store
        
    def retrieve(self, query: str) -> List[Dict[str, Any]]:
        """
        Master method executing the full retrieval architecture:
        1. Query Expansion (Dense, BM25, Structural)
        2. Parallel Retrieval (Top-50 each)
        3. Reciprocal Rank Fusion (Top-30)
        4. Cross-Encoder Re-ranking (Top-10)
        """
        
        # 1 & 2. Expand query and execute Retrievers in parallel (Top-50 each)
        raw_results = self.orchestrator.retrieve_parallel(query, top_k=50)
        
        # 3. Reciprocal Rank Fusion (Combines into Top-30)
        fused_results = self.rrf.fuse(
            dense_results=raw_results["dense"],
            bm25_results=raw_results["bm25"],
            structural_results=raw_results["structural"],
            top_k=30
        )
        
        if not fused_results:
            return []
            
        # 4. Fetch rich text representations from SQLite for the Top-30 Cross-Encoder candidates
        chunk_contents = {}
        for res in fused_results:
            chunk_id = res["chunk_id"]
            metadata = self.metadata_store.get_by_chunk_id(chunk_id)
            if metadata and metadata.get("representation"):
                chunk_contents[chunk_id] = metadata["representation"]
                
        # 5. Cross-Encoder Deep Re-ranking (Refines to Top-10)
        final_results = self.cross_encoder.rerank(
            query=query,
            rrf_results=fused_results,
            chunk_contents=chunk_contents,
            top_k=10
        )
        
        return final_results
