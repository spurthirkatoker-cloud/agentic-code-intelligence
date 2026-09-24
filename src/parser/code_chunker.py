import hashlib
from typing import List, Dict, Any

def generate_chunk_id(file_path: str, entity_type: str, parent_class: str, entity_name: str) -> str:
    """Generate a stable, unique ID for a code chunk based on its unique structural properties."""
    parent = parent_class if parent_class else ""
    unique_string = f"{file_path}::{entity_type}::{parent}::{entity_name}"
    return hashlib.md5(unique_string.encode('utf-8')).hexdigest()

def chunk_code(entities: List[Dict[str, Any]], file_path: str) -> List[Dict[str, Any]]:
    """Convert parsed AST entities into standardized code chunks."""
    chunks = []
    
    for entity in entities:
        # The primary retrieval units are structurally meaningful blocks
        if entity["type"] not in ("class", "function", "method"):
            continue
            
        chunk_id = generate_chunk_id(
            file_path, 
            entity["type"], 
            entity.get("parent_class"), 
            entity["name"]
        )
        
        chunk = {
            "chunk_id": chunk_id,
            "file_path": file_path,
            "chunk_type": entity["type"],
            "name": entity["name"],
            "start_line": entity["start_line"],
            "end_line": entity["end_line"],
            "code": entity["code"],
            # Preserve additional structural metadata
            "imports": entity.get("imports", []),
            "calls": entity.get("calls", []),
            "parent_class": entity.get("parent_class")
        }
        
        chunks.append(chunk)
        
    return chunks
