from typing import Dict, Any

def build_representation(metadata: Dict[str, Any]) -> str:
    """Build a rich, text-based searchable representation from chunk metadata."""
    
    chunk_type_display = str(metadata.get('chunk_type', 'unknown')).capitalize()
    name = metadata.get('function_name') or metadata.get('class_name') or 'unknown'
    
    lines = []
    lines.append(f"{chunk_type_display}: {name}")
    
    if metadata.get('parent_class'):
        lines.append(f"Class: {metadata['parent_class']}")
        
    lines.append(f"File: {metadata.get('file_path', 'unknown')}")
    lines.append(f"Language: {str(metadata.get('language', 'unknown')).capitalize()}")
    
    calls = metadata.get('function_calls', [])
    if calls:
        lines.append(f"Calls: {', '.join(calls)}")
        
    imports = metadata.get('imports', [])
    if imports:
        lines.append(f"Imports: {', '.join(imports)}")
        
    lines.append("\nCode:")
    lines.append(metadata.get('code', ''))
    
    return "\n".join(lines)
