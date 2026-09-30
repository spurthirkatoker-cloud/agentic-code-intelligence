# CodeLens AI - Agentic Code Intelligence

## 1. Project Title
**CodeLens AI - Agentic Code Intelligence**

## 2. Problem Statement / Objective
To build a highly accurate, CPU-friendly Code Retrieval System capable of taking a massive codebase and a natural-language query to return the Top-10 most relevant code snippets. This enables semantic search across repositories and serves as a foundational component for retrieval-augmented generation (RAG) agents.

## 3. Architecture
The system employs a multi-modal retrieval pipeline and precision reranking:

```text
Query
  → Query preprocessing (Synonym expansion, stopword removal, intent parsing)
  → Dense retrieval (Semantic search)
  → BM25 retrieval (Lexical keyword matching)
  → Structural/AST retrieval (Code structure and intent matching)
  → RRF (Reciprocal Rank Fusion blending Top-50 candidates into Top-30)
  → Cross-Encoder reranking (Precision ranking via Cross-Encoder)
  → Top-10
```

## 4. Technologies Used
- **Sentence Transformers**
- **all-MiniLM-L6-v2** (Lightweight Dense Embedding Model)
- **FAISS** (High-speed Vector Indexing)
- **BM25** (Lexical/Keyword Indexing)
- **Tree-sitter** (Structural/AST Parsing)
- **RRF** (Reciprocal Rank Fusion)
- **Cross-Encoder** (Reranking)
- **version-aware indexes** (Via SQLite Metadata Store)

## 5. Installation Requirements
- Python 3.10+
- `pip install -r requirements.txt` (including dependencies like `torch`, `sentence-transformers`, `faiss-cpu`, `rank_bm25`, `tree-sitter`, `streamlit`, `mteb`, etc.)
- Git

## 6. Repository Setup
1. Clone the repository.
2. Create and activate a Python virtual environment:
   - Windows: `python -m venv .venv` and `.\.venv\Scripts\activate`
   - Linux/Mac: `python -m venv .venv` and `source .venv/bin/activate`
3. Install dependencies: `pip install -r requirements.txt`
4. Ensure required `tree-sitter` language bindings (e.g., Python, JS, Go) are available.

## 7. How to Build the Code Index
To index a local codebase repository:
```bash
python -m src.indexing.builder /path/to/your/repo
```
This generates the FAISS index, BM25 pickle, Structural pickle, and the SQLite metadata store in version-isolated persistence paths. 

## 8. How to Run CLI Retrieval
To execute a query against your built index from the command line:
```bash
python -m src.cli "how to calculate gradient loss"
```

## 9. How to Run Streamlit Demo
Launch the interactive web UI to explore retrieval results visually:
```bash
streamlit run app/app.py
```

## 10. How to Run Official MTEB AppsRetrieval Evaluation
To run the standardized MTEB benchmark on the `CoIR-Retrieval/apps` dataset:
```bash
python evaluation/mteb_eval.py
```

## 11. Where appsretrieval_results.json is generated
Upon completing the MTEB evaluation script, the official benchmark outputs are generated and saved at the root directory:
`./appsretrieval_results.json`

## 12. P0 Evaluation Results
The official `AppsRetrieval` evaluation results for the current P0 architecture are:
- **NDCG@10 = 0.05499**
- **MRR@10 = 0.04259**

*(Note: These figures represent the official AppsRetrieval evaluation results for the baseline architecture).*

## 13. P1 Version-Aware Retrieval Instructions
The P1 pipeline supports version-aware indexes. The underlying `MetadataStore` schema natively tracks versions (commit hashes/tags). 
To query a specific version of your code, simply checkout that commit/branch in Git, and the CLI will automatically load the correct version-isolated index:
```bash
git checkout v1.2.0
python -m src.cli "user auth endpoint"
```

## 14. Example Queries and Example Retrieved Results
**Query:** "function to normalize a dataset array"

**Retrieved Results:**
1. `utils/preprocessing.py: normalize_array(arr)` - Rank 1
2. `data/transforms.py: scale_and_normalize(dataset)` - Rank 2
3. `tests/test_preprocessing.py: test_normalize_array()` - Rank 3

## 15. CPU/Runtime Information
- **Hardware:** This architecture is specifically designed to be CPU-friendly. No GPU is required.
- **Dense Model:** `all-MiniLM-L6-v2` executes efficiently on standard multi-core CPUs.
- **Reranker:** The Cross-Encoder is computationally intensive but strictly limited to the RRF Top-30 candidates to maintain real-time sub-second latency on CPU.
- **Index Build Time:** ~5 minutes for an 8,000+ document corpus on a standard desktop CPU.

## 16. Project Directory Structure
```text
code-intelligence/
├── src/
│   ├── retrieval/
│   │   ├── dense/         (FAISS, Embedder)
│   │   ├── lexical/       (BM25)
│   │   ├── structural/    (Tree-sitter AST)
│   │   ├── expansion/     (Query preprocessing, Orchestrator)
│   │   └── fusion/        (RRF, Cross-Encoder)
│   ├── storage/           (SQLite MetadataStore)
│   └── ui/                (Streamlit App)
├── evaluation/            (MTEB benchmarking scripts)
├── scripts/               (Index building & CLI tools)
├── indexes/               (Persisted indices - untracked)
├── requirements.txt
└── README.md
```

## 17. Reproducibility Instructions
To completely reproduce the official MTEB results:
1. Ensure the environment is identical using `requirements.txt`.
2. Do NOT change any architecture code in `src/` or configuration parameters in `evaluation/mteb_eval.py`.
3. Run `python evaluation/mteb_eval.py`.
4. The output will identicaly match the provided official `appsretrieval_results.json` file.
