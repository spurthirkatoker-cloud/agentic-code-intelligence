from typing import Dict, Any

def extract_metadata(chunk: Dict[str, Any], repository_id: str, git_commit: str = "unknown", git_branch: str = "unknown", language: str = "unknown") -> Dict[str, Any]:
    """Enhance a code chunk with additional metadata fields."""
    return {
        "chunk_id": chunk["chunk_id"],
        "repository_id": repository_id,
        "file_path": chunk["file_path"],
        "language": language,
        "chunk_type": chunk["chunk_type"],
        "class_name": chunk.get("parent_class") if chunk["chunk_type"] == "method" else (chunk["name"] if chunk["chunk_type"] == "class" else None),
        "function_name": chunk["name"] if chunk["chunk_type"] in ("function", "method") else None,
        "name": chunk.get("name"),
        "calls": chunk.get("calls", []),
        "start_line": chunk["start_line"],
        "end_line": chunk["end_line"],
        "imports": chunk.get("imports", []),
        "function_calls": chunk.get("calls", []),
        "parent_class": chunk.get("parent_class"),
        "code": chunk["code"],
        "git_commit": git_commit,
        "git_branch": git_branch
    }
