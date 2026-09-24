from src.parser.code_chunker import generate_chunk_id, chunk_code

def test_generate_chunk_id():
    id1 = generate_chunk_id("src/main.py", "function", None, "my_func")
    id2 = generate_chunk_id("src/main.py", "function", None, "my_func")
    id3 = generate_chunk_id("src/other.py", "function", None, "my_func")
    id4 = generate_chunk_id("src/main.py", "method", "MyClass", "my_func")
    
    # ID should be deterministic and stable
    assert id1 == id2
    # Differing by file path should yield different IDs
    assert id1 != id3
    # Differing by type/parent should yield different IDs
    assert id1 != id4

def test_chunk_code():
    entities = [
        {
            "type": "class",
            "name": "MyClass",
            "start_line": 10,
            "end_line": 20,
            "code": "class MyClass:\n    pass",
            "imports": ["import os"],
            "calls": [],
            "parent_class": None
        },
        {
            "type": "method",
            "name": "my_method",
            "start_line": 15,
            "end_line": 18,
            "code": "def my_method(self):\n    pass",
            "imports": ["import os"],
            "calls": ["print"],
            "parent_class": "MyClass"
        },
        {
            "type": "variable", # This type should be filtered out by the chunker
            "name": "x",
            "start_line": 1,
            "end_line": 1,
            "code": "x = 10"
        }
    ]
    
    chunks = chunk_code(entities, "src/test.py")
    
    # Assert variable was filtered out, leaving class and method
    assert len(chunks) == 2
    
    class_chunk = next(c for c in chunks if c["chunk_type"] == "class")
    assert class_chunk["file_path"] == "src/test.py"
    assert class_chunk["name"] == "MyClass"
    assert "chunk_id" in class_chunk
    assert class_chunk["code"] == "class MyClass:\n    pass"
    
    method_chunk = next(c for c in chunks if c["chunk_type"] == "method")
    assert method_chunk["parent_class"] == "MyClass"
    assert method_chunk["calls"] == ["print"]
    assert "chunk_id" in method_chunk
