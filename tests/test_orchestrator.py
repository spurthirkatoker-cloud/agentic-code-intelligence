from src.retrieval.expansion.query_expander import QueryExpander
from src.retrieval.expansion.orchestrator import RetrievalOrchestrator
from src.retrieval.dense.embedder import Embedder
from src.retrieval.dense.faiss_index import DenseIndex
from src.retrieval.dense.dense_search import DenseRetriever
from src.retrieval.lexical.bm25_index import BM25Index
from src.retrieval.lexical.bm25_search import BM25Retriever
from src.retrieval.structural.structural_index import StructuralIndex
from src.retrieval.structural.structural_search import StructuralRetriever

def test_retrieval_orchestrator():
    chunks = [
        {
            "chunk_id": "chunk-1",
            "name": "normalize",
            "code": "def normalize(text): return text.strip().lower()",
            "file_path": "src/preprocessing.py",
            "type": "function",
            "calls": ["strip", "lower"]
        },
        {
            "chunk_id": "chunk-2",
            "name": "unrelated",
            "code": "def unrelated(): pass",
            "file_path": "src/unrelated.py",
            "type": "function",
            "calls": []
        },
        {
            "chunk_id": "chunk-3",
            "name": "foo",
            "code": "def foo(): pass",
            "file_path": "src/foo.py",
            "type": "function",
            "calls": []
        },
        {
            "chunk_id": "chunk-4",
            "name": "bar",
            "code": "def bar(): pass",
            "file_path": "src/bar.py",
            "type": "function",
            "calls": []
        }
    ]
    
    # 1. Setup Dense Pipeline
    embedder = Embedder()
    dense_idx = DenseIndex(384)
    embeddings = embedder.encode([
        "def normalize(text): return text.strip().lower()",
        "def unrelated(): pass",
        "def foo(): pass",
        "def bar(): pass"
    ])
    dense_idx.add_embeddings(embeddings, ["chunk-1", "chunk-2", "chunk-3", "chunk-4"])
    dense_retriever = DenseRetriever(embedder, dense_idx)
    
    # 2. Setup BM25 Pipeline
    bm25_idx = BM25Index()
    bm25_idx.add_chunks(chunks)
    bm25_retriever = BM25Retriever(bm25_idx)
    
    # 3. Setup Structural Pipeline
    struct_idx = StructuralIndex()
    struct_idx.add_chunks(chunks)
    struct_retriever = StructuralRetriever(struct_idx)
    
    # 4. Orchestrator
    expander = QueryExpander()
    orchestrator = RetrievalOrchestrator(
        expander, dense_retriever, bm25_retriever, struct_retriever
    )
    
    # 5. Execute unified retrieval
    results = orchestrator.retrieve_parallel("How is data normalized?", top_k=10)
    
    assert "dense" in results
    assert "bm25" in results
    assert "structural" in results
    
    # Assert Dense retrieval hit
    assert len(results["dense"]) > 0
    assert results["dense"][0]["chunk_id"] == "chunk-1"
    
    # Assert Lexical retrieval hit
    assert len(results["bm25"]) > 0
    assert results["bm25"][0]["chunk_id"] == "chunk-1"
    
    # Assert Structural retrieval hit
    assert len(results["structural"]) > 0
    assert results["structural"][0]["chunk_id"] == "chunk-1"
