def detect_language(file_path: str) -> str:
    """Detect the programming language based on the file extension."""
    if file_path.endswith('.py'):
        return 'python'
    elif file_path.endswith(('.js', '.jsx')):
        return 'javascript'
    elif file_path.endswith(('.ts', '.tsx')):
        return 'typescript'
    elif file_path.endswith('.java'):
        return 'java'
    elif file_path.endswith(('.cpp', '.cc', '.cxx', '.hpp', '.h')):
        return 'cpp'
    elif file_path.endswith('.c'):
        return 'c'
    return 'unknown'
