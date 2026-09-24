import sys
import importlib

print(f"Python version: {sys.version}\n")

core_modules = [
    "tree_sitter",
    "sentence_transformers",
    "faiss",
    "rank_bm25",
    "numpy",
    "pandas",
    "sklearn",
    "mteb",
    "datasets",
    "streamlit"
]

all_passed = True
for module in core_modules:
    try:
        importlib.import_module(module)
        print(f"[OK] {module} imported successfully.")
    except ImportError as e:
        print(f"[ERROR] Failed to import {module}: {e}")
        all_passed = False

print("\n--------------------------------")
if all_passed:
    print("Project setup successful")
    print("All core dependencies available")
else:
    print("Project setup failed: missing dependencies.")
    sys.exit(1)
