from typing import List, Dict, Any
from .structural_index import StructuralIndex
import re

class StructuralRetriever:
    def __init__(self, index: StructuralIndex):
        self.index = index
        
    def extract_hints(self, query: str) -> Dict[str, List[str]]:
        """A lightweight rule-based structural signal extractor."""
        hints = {
            "calls": [],
            "classes": [],
            "names": [],
            "imports": []
        }
        
        lower_query = query.lower()
        words = [w for w in re.split(r'\W+', lower_query) if w]
        
        for i, word in enumerate(words):
            if word in ("calls", "calling", "uses", "using", "invokes"):
                if i + 1 < len(words):
                    hints["calls"].append(words[i+1])
                    
            if word in ("class", "object"):
                if i + 1 < len(words):
                    hints["classes"].append(words[i+1])
                    
            if word in ("imports", "importing", "requires"):
                if i + 1 < len(words):
                    hints["imports"].append(words[i+1])
                    
            if word in ("function", "method", "def", "implement"):
                if i > 0:
                    hints["names"].append(words[i-1])
                if i + 1 < len(words):
                    hints["names"].append(words[i+1])
                    
        # Allow isolated action keywords (often passed by QueryExpander) to match structurally
        if len(words) == 1 and len(words[0]) > 3:
            hints["names"].append(words[0])
            hints["calls"].append(words[0])
            hints["classes"].append(words[0])
            hints["imports"].append(words[0])
            
        return hints
        
    def search(self, query: str, top_k: int = 50) -> List[Dict[str, Any]]:
        if not self.index.chunk_ids:
            return []
            
        hints = self.extract_hints(query)
        scores: Dict[str, float] = {cid: 0.0 for cid in self.index.chunk_ids}
        
        # Aggregate structural scores across the edges
        for call_hint in hints["calls"]:
            for call_key, cids in self.index.calls_to_chunks.items():
                if call_hint in call_key.lower():
                    for cid in cids: scores[cid] += 2.0
                        
        for name_hint in hints["names"]:
            for name_key, cids in self.index.name_to_chunk.items():
                if name_hint == name_key.lower():
                    for cid in cids: scores[cid] += 3.0
                        
        for class_hint in hints["classes"]:
            for class_key, cids in self.index.class_to_methods.items():
                if class_hint == class_key.lower():
                    for cid in cids: scores[cid] += 2.0
            for name_key, cids in self.index.name_to_chunk.items():
                if class_hint == name_key.lower(): 
                    for cid in cids: scores[cid] += 3.0
                        
        for imp_hint in hints["imports"]:
            for imp_key, cids in self.index.imports_to_chunks.items():
                if imp_hint == imp_key.lower():
                    for cid in cids: scores[cid] += 1.0
                        
        # Filter zero-score chunks and sort
        scored_chunks = [(cid, score) for cid, score in scores.items() if score > 0]
        scored_chunks.sort(key=lambda x: x[1], reverse=True)
        
        results = []
        for rank, (cid, score) in enumerate(scored_chunks[:top_k], start=1):
            results.append({
                "chunk_id": cid,
                "rank": rank,
                "score": score
            })
            
        return results
