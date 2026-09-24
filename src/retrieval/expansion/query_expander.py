from typing import Dict, Any
import re

class QueryExpander:
    def __init__(self):
        # Lightweight domain-specific dictionary to simulate LLM expansion rules natively
        self.synonyms = {
            "normalize": ["normalize", "preprocessing", "scale", "standardize", "clean"],
            "loss": ["loss", "cost", "criterion", "objective", "error"],
            "train": ["train", "fit", "optimize", "learn"],
            "database": ["database", "db", "sql", "storage", "store"],
            "api": ["api", "endpoint", "route", "rest", "http"],
            "test": ["test", "assert", "mock", "fixture", "spec"],
            "auth": ["auth", "login", "jwt", "token", "credential", "security"]
        }
        
        self.stop_words = {
            "how", "is", "the", "a", "an", "what", "where", "when", "why", "who", "which", 
            "are", "do", "does", "did", "can", "could", "would", "should", "to", "in", 
            "for", "of", "on", "with", "by", "at", "from", "about", "as", "into", "like", 
            "through", "after", "over", "between", "out", "against", "during", "without", 
            "before", "under", "around", "among", "we", "we'll", "they"
        }

    def expand(self, query: str) -> Dict[str, Any]:
        """Expands a single natural language query into 3 distinct specialized queries."""
        return {
            "dense_query": self._build_dense_query(query),
            "bm25_query": self._build_bm25_query(query),
            "structural_query": self._build_structural_query(query)
        }

    def _build_dense_query(self, query: str) -> str:
        # Dense retrieval benefits heavily from retaining the full natural context
        return query

    def _build_bm25_query(self, query: str) -> str:
        # BM25 is strictly lexical: strip stopwords, pull raw tokens, inject known domain synonyms
        words = re.split(r'\W+', query.lower())
        keywords = set()
        
        for w in words:
            if not w or w in self.stop_words:
                continue
            keywords.add(w)
            
            # Map synonyms using prefix matching for crude stemming
            for key, syns in self.synonyms.items():
                if w == key or w in syns or (len(w) >= 5 and w.startswith(key[:5])):
                    for s in syns:
                        keywords.add(s)
                        
        return " ".join(list(keywords))

    def _build_structural_query(self, query: str) -> Dict[str, str]:
        # Extract explicit actions for AST querying using a rule-based heuristic
        words = re.split(r'\W+', query.lower())
        
        known_actions = ["normalize", "train", "validate", "calculate", "compute", "fetch", "save", "load", "test"]
        
        action = None
        for w in words:
            if not w or w in self.stop_words:
                continue
                
            # Exact Match
            if w in known_actions:
                action = w
                break
                
            # Sub-string stemming heuristic (normalized -> normaliz -> normalizes mapping to 'normalize')
            for ka in known_actions:
                if len(w) > 4 and w.startswith(ka[:5]):
                    action = ka
                    break
                    
            if action: 
                break
                
        return {"action": action} if action else {}
