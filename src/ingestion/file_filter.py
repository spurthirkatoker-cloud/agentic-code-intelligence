import os

SUPPORTED_EXTENSIONS = {".py"}
IGNORED_DIRS = {".git", "node_modules", "venv", ".venv", "__pycache__", "build", "env", "dist"}

def should_process_file(file_path: str) -> bool:
    """Determine if a file should be processed based on extension and path."""
    _, ext = os.path.splitext(file_path)
    if ext not in SUPPORTED_EXTENSIONS:
        return False
        
    # Check if any parent directory is in IGNORED_DIRS or is hidden
    normalized_path = file_path.replace("\\", "/")
    parts = normalized_path.split("/")
    
    for part in parts[:-1]: # exclude the file name itself
        if part in IGNORED_DIRS or part.startswith("."):
            return False
            
    return True
