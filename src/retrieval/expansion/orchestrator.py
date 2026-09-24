from typing import List, Dict, Any
from .query_expander import QueryExpander
from src.retrieval.dense.dense_search import DenseRetriever
from src.retrieval.lexical.bm25_search import BM25Retriever
from src.retrieval.structural.structural_search import StructuralRetriever

class RetrievalOrchestrator:
    def __init__(
        self, 
        expander: QueryExpander, 
        dense_retriever: DenseRetriever, 
        bm25_retriever: BM25Retriever,
        structural_retriever: StructuralRetriever
    ):
        self.expander = expander
        self.dense_retriever = dense_retriever
        self.bm25_retriever = bm25_retriever
        self.structural_retriever = structural_retriever
        
    def retrieve_parallel(self, query: str, top_k: int = 50) -> Dict[str, List[Dict[str, Any]]]:
        """Expands the query and executes all retrievers independently."""
        expanded_queries = self.expander.expand(query)
        
        # Execute Dense Retrieval Top-K
        dense_results = self.dense_retriever.search(
            expanded_queries["dense_query"], 
            top_k=top_k
        )
        
        # Execute Lexical BM25 Retrieval Top-K
        bm25_results = self.bm25_retriever.search(
            expanded_queries["bm25_query"], 
            top_k=top_k
        )
        
        # Execute Structural/AST Retrieval Top-K
        struct_query_dict = expanded_queries.get("structural_query", {})
        action_str = struct_query_dict.get("action", "")
        struct_query = action_str if action_str else query
        
        structural_results = self.structural_retriever.search(
            struct_query, 
            top_k=top_k
        )
        
        return {
            "dense": dense_results,
            "bm25": bm25_results,
            "structural": structural_results
        }
