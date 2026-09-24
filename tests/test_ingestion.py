import os
import tempfile
from src.ingestion.repository_scanner import scan_repository
from src.ingestion.file_filter import should_process_file
from src.ingestion.code_loader import load_file

def test_should_process_file():
    assert should_process_file("src/main.py") == True
    assert should_process_file("tests/test_main.py") == True
    assert should_process_file("README.md") == False
    assert should_process_file(".git/config") == False
    assert should_process_file("src/__pycache__/main.cpython-310.pyc") == False
    assert should_process_file(os.path.join("src", "utils", "helper.py")) == True

def test_scan_repository():
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create some files
        os.makedirs(os.path.join(temp_dir, "src"))
        os.makedirs(os.path.join(temp_dir, "venv"))
        
        with open(os.path.join(temp_dir, "src", "main.py"), "w", encoding="utf-8") as f:
            f.write("print('hello')")
            
        with open(os.path.join(temp_dir, "README.md"), "w", encoding="utf-8") as f:
            f.write("readme")
            
        with open(os.path.join(temp_dir, "venv", "test.py"), "w", encoding="utf-8") as f:
            f.write("print('ignore me')")
            
        files = scan_repository(temp_dir)
        
        assert len(files) == 1
        assert files[0].endswith("main.py")

def test_code_loader():
    with tempfile.TemporaryDirectory() as temp_dir:
        file_path = os.path.join(temp_dir, "main.py")
        with open(file_path, "w", encoding="utf-8") as f:
            f.write("print('hello')")
            
        result = load_file(file_path)
        
        assert result is not None
        assert result["file_path"] == file_path
        assert result["language"] == "python"
        assert result["content"] == "print('hello')"
