import streamlit as st
import time
import os
import sys
from typing import Optional

# Ensure the root 'src' directory is inside the Python path so Streamlit can locate our modules natively
repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(repo_root)

# Force-inject the local virtual environment packages just in case the user's IDE terminal is routing to a global Python
venv_path = os.path.join(repo_root, ".venv", "Lib", "site-packages")
if os.path.exists(venv_path) and venv_path not in sys.path:
    sys.path.insert(0, venv_path)

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
from src.versioning.git_manager import GitManager
from src.versioning.versioned_index import VersionedIndexManager

# ---------------------------------------------------------
# CPU-Bound Singleton Initialization of Core Architecture
# ---------------------------------------------------------
@st.cache_resource
def init_search_service() -> Optional[SearchService]:
    """Safely initializes the Neural Pipeline by dynamically binding to Version-Isolated persistence paths."""
    repo_path = os.getcwd()
    git_manager = GitManager(repo_path)
    version_info = git_manager.get_current_version_info()
    commit_hash = version_info.get("commit_hash", "unknown")
    
    index_manager = VersionedIndexManager()
    paths = index_manager.get_paths(commit_hash)
    
    # Fail cleanly if the isolated FAISS database doesn't exist for this commit
    if not os.path.exists(paths["faiss_index_path"]):
        return None
        
    store = MetadataStore(paths["metadata_db_path"])
    
    embedder = Embedder()
    dense_idx = DenseIndex(384)
    # Correctly load both FAISS and its mathematical chunk_id mapping
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
    
    # 60 acts as the RRF K-constant mathematically smoothing out rank variances
    rrf = ReciprocalRankFusion(k=60)
    
    # Lightweight CPU-friendly re-ranker
    cross_encoder = CrossEncoderReRanker()
    
    # Secure API Wrapping
    pipeline = RetrievalPipeline(orchestrator, rrf, cross_encoder, store)
    return SearchService(pipeline, store)

search_service = init_search_service()

# ---------------------------------------------------------
# Streamlit Interface
# ---------------------------------------------------------
st.set_page_config(page_title="CodeLens AI", layout="wide")

st.title("CodeLens AI")
st.subheader("Agentic Code Intelligence: Find the right code, faster.")

if not search_service:
    st.error("⚠️ **Retrieval indexes for the current repository commit were not found.**\n\nPlease initialize your local intelligence cache by running the indexing script in your terminal:\n\n`python -m src.indexing.builder`")
    st.stop()

st.info("**Retrieval Flow Architecture:**\n\n`Dense Top-50 + BM25 Top-50 + Structural Top-50 → RRF Top-30 → Cross-Encoder Top-10`")

query = st.text_input("Enter your natural-language query:")

if st.button("Search Codebase") and query:
    start_time = time.perf_counter()
    
    with st.spinner("Executing Parallel Neural Search Engine..."):
        # Strictly enforces the single entry point (no duplicated code)
        results = search_service.search(query)
        
    latency = (time.perf_counter() - start_time) * 1000
    st.success(f"Search successfully completed in {latency:.2f} ms")
    
    if not results:
        st.warning("No relevant code snippets were found in the active index.")
    else:
        st.markdown("### Top-10 Extracted Results")
        for idx, res in enumerate(results):
            rank = res.get("rank", idx + 1)
            score = res.get("score", 0.0)
            file_path = res.get("file_path", "Unknown File")
            
            # The top rank expands automatically for speed reading
            with st.expander(f"Rank {rank} | {file_path} (Neural Score: {score:.4f})", expanded=(idx == 0)):
                st.markdown(f"**Target Signature:** `{res.get('name', 'N/A')}`")
                
                # Display structural representation if present, fallback to raw source code
                content = res.get('representation') or res.get('code') or "No code text available."
                st.code(content, language='python')
