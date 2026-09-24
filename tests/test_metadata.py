from src.parser.metadata_extractor import extract_metadata
from src.parser.code_representation import build_representation

def test_extract_metadata():
    chunk = {
        "chunk_id": "123",
        "file_path": "src/preprocessing.py",
        "chunk_type": "function",
        "name": "normalize",
        "start_line": 20,
        "end_line": 40,
        "code": "def normalize(text):\n    pass",
        "imports": ["import re"],
        "calls": ["strip", "lower"]
    }
    
    metadata = extract_metadata(chunk, repository_id="repo-1", git_commit="abc1234", git_branch="main", language="python")
    
    assert metadata["chunk_id"] == "123"
    assert metadata["repository_id"] == "repo-1"
    assert metadata["git_commit"] == "abc1234"
    assert metadata["function_name"] == "normalize"
    assert metadata["class_name"] is None
    assert metadata["function_calls"] == ["strip", "lower"]

def test_build_representation():
    metadata = {
        "chunk_type": "function",
        "function_name": "normalize",
        "file_path": "preprocessing.py",
        "language": "python",
        "function_calls": ["strip", "lower"],
        "imports": ["import re"],
        "code": "def normalize(text):\n    pass"
    }
    
    rep = build_representation(metadata)
    
    assert "Function: normalize" in rep
    assert "File: preprocessing.py" in rep
    assert "Language: Python" in rep
    assert "Calls: strip, lower" in rep
    assert "Imports: import re" in rep
    assert "Code:\ndef normalize(text):" in rep
