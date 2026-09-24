import streamlit as st
import time
import os
import sys

# Ensure the root 'src' directory is inside the Python path so Streamlit can locate our modules natively
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

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

# ---------------------------------------------------------
# CPU-Bound Singleton Initialization of Core Architecture
# ---------------------------------------------------------
@st.cache_resource
def init_search_service() -> SearchService:
    # Safely point to an isolated DB; for MVP demo, defaults to memory/scratch
    db_path = os.environ.get("DATABASE_PATH", "demo_scratch.db")
    store = MetadataStore(db_path)
    
    embedder = Embedder()
    dense_idx = DenseIndex(384)
    dense_retriever = DenseRetriever(embedder, dense_idx)
    
    bm25_idx = BM25Index()
    bm25_retriever = BM25Retriever(bm25_idx)
    
    struct_idx = StructuralIndex()
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
