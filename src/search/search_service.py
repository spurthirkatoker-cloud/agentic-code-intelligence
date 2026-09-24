from typing import List, Dict, Any
import logging
from src.retrieval.pipeline import RetrievalPipeline
from src.storage.metadata_store import MetadataStore

class SearchService:
    def __init__(self, pipeline: RetrievalPipeline, metadata_store: MetadataStore):
        self.pipeline = pipeline
        self.metadata_store = metadata_store
        
    def search(self, query: str) -> List[Dict[str, Any]]:
        """
        Public API interface for code retrieval.
        Accepts a natural language query and orchestrates the full parallel multi-agent search pipeline.
        Returns a stable schema with the Top-10 results enriched with code contents.
        """
        # Validate Input
        if not query or not isinstance(query, str) or not query.strip():
            logging.warning("SearchService received an empty or invalid query.")
            return []
            
        try:
            # 1. Execute the master retrieval pipeline (Expansion -> Parallel Search -> RRF -> Cross-Encoder)
            pipeline_results = self.pipeline.retrieve(query.strip())
            
            # 2. Enrich the Top-10 results with full metadata strings for frontend/downstream usage
            enriched_results = []
            for res in pipeline_results:
                chunk_id = res["chunk_id"]
                metadata = self.metadata_store.get_by_chunk_id(chunk_id) or {}
                
                enriched_results.append({
                    "chunk_id": chunk_id,
                    "rank": res["rank"],
                    "score": res.get("cross_encoder_score", 0.0),
                    "file_path": metadata.get("file_path"),
                    "name": metadata.get("name"),
                    "type": metadata.get("type"),
                    "code": metadata.get("code"),
                    "representation": metadata.get("representation")
                })
                
            return enriched_results
            
        except Exception as e:
            logging.error(f"SearchService encountered a fatal error during retrieval: {str(e)}")
            return []
