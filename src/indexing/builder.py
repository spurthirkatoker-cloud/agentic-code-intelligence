import os
import sys
import logging
from src.versioning.git_manager import GitManager
from src.versioning.versioned_index import VersionedIndexManager
from src.ingestion.repository_scanner import scan_repository
from src.ingestion.code_loader import load_file
from src.parser.language_detector import detect_language
from src.parser.tree_sitter_parser import parse_source_code
from src.parser.code_chunker import chunk_code
from src.parser.metadata_extractor import extract_metadata
from src.parser.code_representation import build_representation
from src.retrieval.dense.embedder import Embedder
from src.retrieval.dense.faiss_index import DenseIndex
from src.retrieval.lexical.bm25_index import BM25Index
from src.retrieval.structural.structural_index import StructuralIndex
from src.storage.metadata_store import MetadataStore

def build_index(repo_path: str):
    print(f"Scanning repository: {repo_path}")
    
    # 1. Obtain strict repository commit metadata securely
    git_manager = GitManager(repo_path)
    version_info = git_manager.get_current_version_info()
    commit_hash = version_info.get("commit_hash", "unknown")
    branch = version_info.get("branch", "unknown")
    repository_id = version_info.get("repository", "unknown")
    
    # 2. Get Version-Isolated Paths to prevent collision across git commits
    index_manager = VersionedIndexManager()
    paths = index_manager.get_paths(commit_hash)
    
    # 3. Setup core components and persistence bridges
    store = MetadataStore(paths["metadata_db_path"])
    embedder = Embedder()
    dense_idx = DenseIndex(384)
    bm25_idx = BM25Index()
    struct_idx = StructuralIndex()
    
    # 4. Unified Scanning & Parsing Pipeline
    source_files = scan_repository(repo_path)
    print(f"Found {len(source_files)} supported source files.")
    
    all_chunks = []
    all_representations = []
    all_embeddings_text = []
    
    for file_path in source_files:
        lang = detect_language(file_path)
        # Ensure we strictly conform to the locked Tree-Sitter support matrix (Python only for MVP)
        if lang != "python":
            continue
            
        loaded = load_file(file_path)
        if not loaded:
            continue
            
        ast_entities = parse_source_code(loaded["content"], lang)
        chunks = chunk_code(ast_entities, file_path)
        
        for c in chunks:
            metadata = extract_metadata(c, repository_id, commit_hash, branch, lang)
            rep = build_representation(metadata)
            
            all_chunks.append(metadata)
            all_representations.append(rep)
            all_embeddings_text.append(c["code"]) # TRD specs use raw code for embeddings
            
    if not all_chunks:
        logging.warning("No Python chunks were generated. Check repository path or language support.")
        return
        
    # 5. Build Indexes
    print(f"Generating semantic embeddings for {len(all_chunks)} unique code blocks...")
    embeddings = embedder.encode(all_embeddings_text)
    
    chunk_ids = [c["chunk_id"] for c in all_chunks]
    
    # Load them natively into the Indexers
    dense_idx.add_embeddings(embeddings, chunk_ids)
    bm25_idx.add_chunks(all_chunks)
    struct_idx.add_chunks(all_chunks)
    
    # 6. Save Indexes directly to their Version-Isolated Paths
    print("Persisting parallel indexes...")
    dense_idx.save(paths["faiss_index_path"], paths["faiss_index_path"] + ".mapping")
    bm25_idx.save(paths["bm25_index_path"])
    struct_idx.save(paths["structural_index_path"])
    
    # Complete the SQLite transaction for the final structural binding
    store.insert_chunks(all_chunks, all_representations)
    
    print(f"\n✅ Successfully built indexes for commit {commit_hash}")
    print(f"Isolated persistence paths located at: {index_manager.get_index_path_for_commit(commit_hash)}\n")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        build_index(sys.argv[1])
    else:
        build_index(os.getcwd())
