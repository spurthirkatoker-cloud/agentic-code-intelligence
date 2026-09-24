import pytest
import sys
from unittest.mock import patch, MagicMock
from io import StringIO
import src.cli as cli

@patch("src.cli.init_service")
def test_cli_single_query(mock_init):
    """Validates the CLI can execute single queries natively without instantiating heavy Neural models."""
    mock_service = MagicMock()
    mock_service.search.return_value = [
        {
            "rank": 1,
            "score": 0.95,
            "file_path": "src/main.py",
            "name": "main",
            "code": "def main():\n    print('hello')\n    return 0"
        }
    ]
    mock_init.return_value = mock_service
    
    test_args = ["cli.py", "How to run main?"]
    with patch.object(sys, 'argv', test_args):
        # Intercept output
        with patch('sys.stdout', new_callable=StringIO) as mock_stdout:
            cli.main()
            output = mock_stdout.getvalue()
            
            # Assertions ensuring display requirements are met natively
            assert "Search completed in" in output
            assert "File: src/main.py" in output
            assert "Target: main" in output
            assert "Neural Score: 0.9500" in output
            assert "def main():" in output

@patch("src.cli.init_service")
def test_cli_interactive_mode(mock_init):
    """Validates the interactive REPL while preventing infinite loops."""
    mock_service = MagicMock()
    mock_service.search.return_value = []
    mock_init.return_value = mock_service
    
    test_args = ["cli.py", "--interactive"]
    
    # Pass 'Where is normalize?' then force quit to avoid REPL hang
    inputs = ["Where is normalize?", "exit"]
    def mock_input(prompt):
        return inputs.pop(0)
        
    with patch.object(sys, 'argv', test_args):
        with patch('sys.stdout', new_callable=StringIO) as mock_stdout:
            with patch('builtins.input', mock_input):
                cli.main()
                output = mock_stdout.getvalue()
                
                assert "Interactive Terminal Search" in output
                assert "No relevant code snippets found" in output
