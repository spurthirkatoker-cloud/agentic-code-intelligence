import pytest
import sys
import os
from streamlit.testing.v1 import AppTest

def test_streamlit_app_loads():
    """
    Smoke test to strictly verify the Streamlit frontend securely bounds itself 
    to the underlying SearchService without throwing fatal Python loading errors.
    """
    # Path to the actual Streamlit script
    app_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "app", "app.py")
    
    # Instantiate the virtual headless browser mimicking a user
    at = AppTest.from_file(app_path)
    
    # Execute the file loading process (extended timeout for Neural model loading)
    at.run(timeout=30)
    
    # 1. Assert no fatal compiler/loading crashes occurred
    assert not at.exception
    
    # 2. Assert structural UI text loaded
    assert len(at.title) > 0
    assert "CodeLens AI" in at.title[0].value
    
    # 3. Assert our graceful index-missing crash protection works (or the app boots completely if indices exist)
    if not os.path.exists(os.path.join(os.path.dirname(os.path.dirname(__file__)), "indexes")):
        assert len(at.error) > 0
        assert "Retrieval indexes" in at.error[0].value
