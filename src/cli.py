import argparse
import sys
import time
import os

from src.retrieval.expansion.query_expander import QueryExpander
from src.retrieval.expansion.orchestrator import RetrievalOrchestrator
from src.retrieval.dense.embedder import Embedder
from src.retrieval.dense.faiss_index import DenseIndex
from src.retrieval.dense.dense_search import DenseRetriever
from src.retrieval.lexical.bm25_index import BM25Index
from src.retrieval.lexical.bm25_search import BM25Retriever
from src.retrieval.structural.structural_index import StructuralIndex
from src.retrieval.structural.structural_search import StructuralRetriever
from src.retrieval.fusion.rrf import ReciprocalRankFusion
from src.retrieval.fusion.cross_encoder import CrossEncoderReRanker
from src.storage.metadata_store import MetadataStore
from src.retrieval.pipeline import RetrievalPipeline
from src.search.search_service import SearchService

def init_service() -> SearchService:
    """Safely instantiates the CPU-bound Neural Engine for local terminal usage."""
    from src.versioning.git_manager import GitManager
    from src.versioning.versioned_index import VersionedIndexManager
    
    repo_path = os.getcwd()
    git_manager = GitManager(repo_path)
    version_info = git_manager.get_current_version_info()
    commit_hash = version_info.get("commit_hash", "unknown")
    
    index_manager = VersionedIndexManager()
    paths = index_manager.get_paths(commit_hash)
    
    if not os.path.exists(paths["faiss_index_path"]):
        print(f"⚠️ Retrieval indexes for the current repository commit ({commit_hash}) were not found.")
        print("Please initialize your local intelligence cache by running: python -m src.indexing.builder")
        sys.exit(1)
        
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
    orchestrator = RetrievalOrchestrator(expander, dense_retriever, bm25_retriever, struct_retriever)
    rrf = ReciprocalRankFusion(k=60)
    cross_encoder = CrossEncoderReRanker()
    
    pipeline = RetrievalPipeline(orchestrator, rrf, cross_encoder, store)
    return SearchService(pipeline, store)

def print_results(results: list, latency_ms: float):
    """Cleanly prints the standard JSON schema to standard output."""
    print(f"\n✅ Search completed in {latency_ms:.2f} ms")
    print("-" * 60)
    
    if not results:
        print("⚠️ No relevant code snippets found in the active index.")
        return
        
    for res in results:
        rank = res.get("rank", "?")
        score = res.get("score", 0.0)
        file_path = res.get("file_path", "Unknown File")
        name = res.get("name") or "N/A"
        
        print(f"[{rank}] File: {file_path} | Target: {name} | Neural Score: {score:.4f}")
        
        # Display a fast readable 3-line chunk code preview for the terminal
        code = res.get("code") or res.get("representation") or ""
        preview_lines = code.split("\n")[:3]
        preview = "\n".join(preview_lines)
        if len(code.split("\n")) > 3:
            preview += "\n  ..."
            
        print(f"Preview:\n  {preview}")
        print("-" * 60)

def main():
    parser = argparse.ArgumentParser(description="CodeLens AI - Command Line Code Intelligence")
    parser.add_argument("query", nargs="?", type=str, help="Natural-language query to search the repository")
    parser.add_argument("--interactive", "-i", action="store_true", help="Launch persistent interactive search mode")
    
    args = parser.parse_args()
    
    if not args.query and not args.interactive:
        parser.print_help()
        sys.exit(1)
        
    print("Initializing CodeLens AI Engine... (Loading Local CPU Weights)")
    service = init_service()
    
    if args.interactive:
        print("\n=== Interactive Terminal Search ===")
        print("Type your query and press Enter. Type 'exit' or 'quit' to close.")
        while True:
            try:
                q = input("\n🔍 Query: ").strip()
                if q.lower() in ['exit', 'quit']:
                    break
                if not q:
                    continue
                    
                start = time.perf_counter()
                results = service.search(q)
                latency = (time.perf_counter() - start) * 1000
                print_results(results, latency)
                
            except KeyboardInterrupt:
                break
    else:
        # Standard Single-Shot Terminal Command
        start = time.perf_counter()
        results = service.search(args.query)
        latency = (time.perf_counter() - start) * 1000
        print_results(results, latency)

if __name__ == "__main__":
    main()
