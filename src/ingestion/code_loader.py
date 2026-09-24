import logging
from typing import Dict, Any, Optional

def load_file(file_path: str) -> Optional[Dict[str, Any]]:
    """Read a source file and return its content along with metadata."""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
            
        language = "python" if file_path.endswith(".py") else "unknown"
        
        return {
            "file_path": file_path,
            "language": language,
            "content": content
        }
    except Exception as e:
        logging.error(f"Error reading file {file_path}: {e}")
        return None
