import pickle
import os
from typing import List, Dict, Any, Set

class StructuralIndex:
    def __init__(self):
        self.chunk_ids: List[str] = []
        
        # Pure Python dictionaries used natively to emulate a fast Graph/AST layout
        self.calls_to_chunks: Dict[str, Set[str]] = {} # call name -> chunk_ids
        self.class_to_methods: Dict[str, Set[str]] = {} # class name -> method chunk_ids
        self.method_to_class: Dict[str, str] = {} # method chunk_id -> class name
        self.imports_to_chunks: Dict[str, Set[str]] = {} # import keyword -> chunk_ids
        self.name_to_chunk: Dict[str, Set[str]] = {} # function/class name -> chunk_ids

    def add_chunks(self, chunks: List[Dict[str, Any]]):
        for chunk in chunks:
            cid = chunk["chunk_id"]
            self.chunk_ids.append(cid)
            
            name = chunk.get("name")
            if name:
                if name not in self.name_to_chunk:
                    self.name_to_chunk[name] = set()
                self.name_to_chunk[name].add(cid)
                
            for call in chunk.get("calls", []):
                if call not in self.calls_to_chunks:
                    self.calls_to_chunks[call] = set()
                self.calls_to_chunks[call].add(cid)
                
            parent = chunk.get("parent_class")
            if parent:
                if parent not in self.class_to_methods:
                    self.class_to_methods[parent] = set()
                self.class_to_methods[parent].add(cid)
                self.method_to_class[cid] = parent
                
            for imp in chunk.get("imports", []):
                words = imp.replace(".", " ").split()
                for w in words:
                    if w not in ("import", "from", "as"):
                        if w not in self.imports_to_chunks:
                            self.imports_to_chunks[w] = set()
                        self.imports_to_chunks[w].add(cid)

    def save(self, index_path: str):
        with open(index_path, 'wb') as f:
            pickle.dump({
                "chunk_ids": self.chunk_ids,
                "calls_to_chunks": self.calls_to_chunks,
                "class_to_methods": self.class_to_methods,
                "method_to_class": self.method_to_class,
                "imports_to_chunks": self.imports_to_chunks,
                "name_to_chunk": self.name_to_chunk
            }, f)

    def load(self, index_path: str):
        if os.path.exists(index_path):
            with open(index_path, 'rb') as f:
                data = pickle.load(f)
                self.chunk_ids = data.get("chunk_ids", [])
                self.calls_to_chunks = data.get("calls_to_chunks", {})
                self.class_to_methods = data.get("class_to_methods", {})
                self.method_to_class = data.get("method_to_class", {})
                self.imports_to_chunks = data.get("imports_to_chunks", {})
                self.name_to_chunk = data.get("name_to_chunk", {})
