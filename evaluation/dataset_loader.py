from typing import List, Dict, Tuple
from datasets import load_dataset
import logging
from src.search.search_service import SearchService

class EvaluationDatasetLoader:
    def __init__(self, dataset_path: str = "mteb/apps"):
        self.dataset_path = dataset_path

    def load_data(self, split: str = "test") -> Tuple[List[Dict[str, str]], Dict[str, List[str]]]:
        """
        Loads the evaluation queries and standard relevance judgments (qrels) from a huggingface IR dataset.
        Returns:
            queries: List of dicts, e.g., [{"query_id": "q1", "text": "how to load data"}]
            qrels: Dict mapping query_id to a list of strictly relevant corpus document IDs.
        """
        try:
            # 1. Load IR queries split
            queries_ds = load_dataset(self.dataset_path, "queries", split=split)
            queries = [{"query_id": str(item["_id"]), "text": item["text"]} for item in queries_ds]
            
            # 2. Load relevance judgments (qrels)
            qrels_ds = load_dataset(self.dataset_path, "qrels", split=split)
            qrels: Dict[str, List[str]] = {}
            for item in qrels_ds:
                qid = str(item["query-id"])
                docid = str(item["corpus-id"])
                score = float(item["score"])
                
                # We only want to map ground-truth correct documents (Score > 0)
                if score > 0:
                    if qid not in qrels:
                        qrels[qid] = []
                    qrels[qid].append(docid)
                    
            return queries, qrels
            
        except Exception as e:
            logging.error(f"Fatal error loading CoIR evaluation dataset from {self.dataset_path}: {e}")
            return [], {}

    def execute_queries_against_service(
        self, 
        queries: List[Dict[str, str]], 
        search_service: SearchService
    ) -> Dict[str, List[str]]:
        """
        Executes an array of loaded evaluation queries through the fully integrated SearchService.
        Returns a dictionary mapping query_id to a list of retrieved chunk_ids in exact ranked order.
        """
        system_predictions = {}
        
        for q in queries:
            qid = q["query_id"]
            text = q["text"]
            
            # Execute through the secure SearchService wrapper API
            results = search_service.search(text)
            
            # Strip metadata and extract strictly the ranked chunk IDs
            retrieved_chunk_ids = [res["chunk_id"] for res in results]
            system_predictions[qid] = retrieved_chunk_ids
            
        return system_predictions
