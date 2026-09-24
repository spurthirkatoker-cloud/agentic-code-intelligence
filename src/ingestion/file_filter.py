import os

SUPPORTED_EXTENSIONS = {".py"}
IGNORED_DIRS = {".git", "node_modules", "venv", ".venv", "__pycache__", "build", "env", "dist", "tests"}

def should_process_file(file_path: str) -> bool:
    """Determine if a file should be processed based on extension."""
    _, ext = os.path.splitext(file_path)
    if ext not in SUPPORTED_EXTENSIONS:
        return False
        
    # The repository scanner already prunes ignored and hidden directories via os.walk dirs injection.
    # We only need to guarantee the specific file itself isn't hidden.
    file_name = os.path.basename(file_path)
    if file_name.startswith("."):
        return False
        
    return True
