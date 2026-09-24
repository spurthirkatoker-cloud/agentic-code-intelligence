from src.parser.language_detector import detect_language
from src.parser.tree_sitter_parser import parse_source_code

def test_language_detector():
    assert detect_language("main.py") == "python"
    assert detect_language("script.js") == "javascript"
    assert detect_language("data.txt") == "unknown"

def test_parse_source_code():
    code = """
import os
from sys import path

class MyClass:
    def my_method(self, arg):
        os.path.join(arg, 'test')
        print(arg)

def my_function():
    MyClass().my_method('dir')
"""
    entities = parse_source_code(code, "python")
    
    classes = [e for e in entities if e["type"] == "class"]
    assert len(classes) == 1
    assert classes[0]["name"] == "MyClass"
    assert "import os" in classes[0]["imports"]
    
    methods = [e for e in entities if e["type"] == "method"]
    assert len(methods) == 1
    assert methods[0]["name"] == "my_method"
    assert methods[0]["parent_class"] == "MyClass"
    assert "print" in methods[0]["calls"]
    
    funcs = [e for e in entities if e["type"] == "function"]
    assert len(funcs) == 1
    assert funcs[0]["name"] == "my_function"
    assert "MyClass().my_method" in funcs[0]["calls"] or any("my_method" in c for c in funcs[0]["calls"])
