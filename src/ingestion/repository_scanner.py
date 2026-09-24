import os
from typing import List
from .file_filter import should_process_file, IGNORED_DIRS

def scan_repository(repo_path: str) -> List[str]:
    """Recursively scan a repository and return a list of supported source files."""
    source_files = []
    
    for root, dirs, files in os.walk(repo_path):
        # Modify dirs in-place to avoid traversing ignored directories
        dirs[:] = [d for d in dirs if not (d.startswith('.') or d in IGNORED_DIRS)]
        
        for file in files:
            file_path = os.path.join(root, file)
            if should_process_file(file_path):
                source_files.append(file_path)
                
    return source_files
