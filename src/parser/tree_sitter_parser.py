import tree_sitter_python as tspython
from tree_sitter import Language, Parser, Node
from typing import List, Dict, Any, Optional

PY_LANGUAGE = Language(tspython.language())

def get_parser(language_name: str) -> Optional[Parser]:
    if language_name == "python":
        return Parser(PY_LANGUAGE)
    return None

def extract_code(node: Node, source_bytes: bytes) -> str:
    return source_bytes[node.start_byte:node.end_byte].decode('utf-8')

def parse_source_code(content: str, language_name: str) -> List[Dict[str, Any]]:
    parser = get_parser(language_name)
    if not parser:
        return []

    source_bytes = content.encode('utf-8')
    tree = parser.parse(source_bytes)
    root_node = tree.root_node

    entities = []
    imports = []
    
    def traverse(node: Node, parent_class: Optional[str] = None):
        if node.type in ('import_statement', 'import_from_statement'):
            imports.append(extract_code(node, source_bytes))
            
        elif node.type == 'class_definition':
            name_node = node.child_by_field_name('name')
            class_name = extract_code(name_node, source_bytes) if name_node else "unknown"
            
            entities.append({
                "type": "class",
                "name": class_name,
                "start_line": node.start_point[0] + 1,
                "end_line": node.end_point[0] + 1,
                "code": extract_code(node, source_bytes),
                "parent_class": parent_class,
                "calls": []
            })
            
            body = node.child_by_field_name('body')
            if body:
                for child in body.children:
                    traverse(child, parent_class=class_name)
                    
        elif node.type == 'function_definition':
            name_node = node.child_by_field_name('name')
            func_name = extract_code(name_node, source_bytes) if name_node else "unknown"
            
            calls = []
            def find_calls(n: Node):
                if n.type == 'call':
                    func = n.child_by_field_name('function')
                    if func:
                        call_name = extract_code(func, source_bytes)
                        calls.append(call_name)
                for c in n.children:
                    find_calls(c)
            find_calls(node)
            
            entities.append({
                "type": "method" if parent_class else "function",
                "name": func_name,
                "start_line": node.start_point[0] + 1,
                "end_line": node.end_point[0] + 1,
                "code": extract_code(node, source_bytes),
                "parent_class": parent_class,
                "calls": list(set(calls))
            })
        else:
            for child in node.children:
                traverse(child, parent_class)
                
    traverse(root_node)
    
    for entity in entities:
        entity["imports"] = imports
        
    return entities
