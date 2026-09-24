import sys
import os

repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
venv_path = os.path.join(repo_root, ".venv", "Lib", "site-packages")
if os.path.exists(venv_path) and venv_path not in sys.path:
    sys.path.insert(0, venv_path)

from src.versioning.git_manager import GitManager
from src.versioning.versioned_index import VersionedIndexManager
from src.retrieval.dense.embedder import Embedder
from src.retrieval.dense.faiss_index import DenseIndex
from src.retrieval.dense.dense_search import DenseRetriever
from src.retrieval.lexical.bm25_index import BM25Index
from src.retrieval.lexical.bm25_search import BM25Retriever
from src.retrieval.structural.structural_index import StructuralIndex
from src.retrieval.structural.structural_search import StructuralRetriever
from src.retrieval.expansion.query_expander import QueryExpander
from src.retrieval.expansion.orchestrator import RetrievalOrchestrator
from src.retrieval.fusion.rrf import ReciprocalRankFusion
from src.retrieval.fusion.cross_encoder import CrossEncoderReRanker
from src.storage.metadata_store import MetadataStore
from src.retrieval.pipeline import RetrievalPipeline
from src.search.search_service import SearchService

def debug_query(query: str):
    repo_path = os.getcwd()
    git_manager = GitManager(repo_path)
    commit_hash = git_manager.get_current_version_info().get("commit_hash", "unknown")
    
    index_manager = VersionedIndexManager()
    paths = index_manager.get_paths(commit_hash)
    
    store = MetadataStore(paths["metadata_db_path"])
    embedder = Embedder()
    dense_idx = DenseIndex(384)
    dense_idx.load(paths["faiss_index_path"], paths["faiss_index_path"] + ".mapping")
    dense_retriever = DenseRetriever(embedder, dense_idx)
    
    bm25_idx = BM25Index()
    bm25_idx.load(paths["bm25_index_path"])
    bm25_retriever = BM25Retriever(bm25_idx)
    
    struct_idx = StructuralIndex()
    struct_idx.load(paths["structural_index_path"])
    struct_retriever = StructuralRetriever(struct_idx)
    
    expander = QueryExpander()
    rrf = ReciprocalRankFusion(k=60)
    cross_encoder = CrossEncoderReRanker()

    pipeline = RetrievalPipeline(
        RetrievalOrchestrator(expander, dense_retriever, bm25_retriever, struct_retriever),
        rrf,
        cross_encoder,
        store
    )
    
    print(f"QUERY: {query}\n")
    
    raw = pipeline.orchestrator.retrieve_parallel(query, top_k=50)
    
    print("--- DENSE TOP 50 ---")
    for r in raw["dense"]:
        meta = store.get_by_chunk_id(r["chunk_id"])
        print(f"[{r['rank']}] Score: {r['score']:.4f} | {meta['file_path']} -> {meta.get('function_name') or meta.get('class_name') or meta.get('name')}")

    print("\n--- BM25 TOP 50 ---")
    for r in raw["bm25"]:
        meta = store.get_by_chunk_id(r["chunk_id"])
        print(f"[{r['rank']}] Score: {r['score']:.4f} | {meta['file_path']} -> {meta.get('function_name') or meta.get('class_name') or meta.get('name')}")

    print("\n--- STRUCTURAL TOP 50 ---")
    for r in raw["structural"]:
        meta = store.get_by_chunk_id(r["chunk_id"])
        print(f"[{r['rank']}] Score: {r['score']:.4f} | {meta['file_path']} -> {meta.get('function_name') or meta.get('class_name') or meta.get('name')}")

    fused = rrf.fuse(raw["dense"], raw["bm25"], raw["structural"], top_k=30)
    print("\n--- RRF TOP 30 ---")
    for r in fused:
        meta = store.get_by_chunk_id(r["chunk_id"])
        print(f"[{r['rank']}] Score: {r['rrf_score']:.4f} | {meta['file_path']} -> {meta.get('function_name') or meta.get('class_name') or meta.get('name')}")

    chunk_contents = {}
    for res in fused:
        meta = store.get_by_chunk_id(res["chunk_id"])
        if meta and meta.get("representation"):
            chunk_contents[res["chunk_id"]] = meta["representation"]
            
    final = cross_encoder.rerank(query, fused, chunk_contents, top_k=10)
    
    print("\n--- CROSS ENCODER TOP 10 ---")
    for r in final[:10]:
        meta = store.get_by_chunk_id(r["chunk_id"])
        print(f"[{r['rank']}] Score: {r.get('cross_encoder_score', 0.0):.4f} | {meta['file_path']} -> {meta.get('function_name') or meta.get('class_name') or meta.get('name')}")
        
if __name__ == '__main__':
    debug_query("How is the input preprocessed before going to the main function?")
