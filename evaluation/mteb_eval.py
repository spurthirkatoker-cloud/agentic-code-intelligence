import json
import logging
from typing import Dict, Any, List
import os

from mteb.models.abs_encoder import AbsEncoder
from mteb.models.model_meta import ModelMeta

import mteb

from src.retrieval.pipeline import RetrievalPipeline
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
from src.search.search_service import SearchService
from src.storage.metadata_store import MetadataStore
from src.parser.code_representation import build_representation

class PrePostPipelineEncoder(AbsEncoder):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.mteb_model_meta = ModelMeta(
            name="myorg/prepost",
            revision="1",
            release_date="2023-01-01",
            languages=["eng-Latn"],
            framework=["PyTorch"],
            n_parameters=1,
            memory_usage_mb=1,
            max_tokens=100,
            embed_dim=384,
            license="mit",
            open_weights=True,
            public_training_data=True,
            public_training_code="unknown",
            similarity_fn_name="cosine",
            use_instructions=False,
            training_datasets=set(),
            loader=None
        )
        self.search_service = None

    def encode(self, sentences, **kwargs):
        pass

    def index(
        self,
        corpus: Any,
        **kwargs
    ) -> None:
        print("DEBUG: Entering index() method...")
        all_chunks = []
        all_representations = []
        all_embeddings_text = []

        # Robust iteration over corpus (dict or HF dataset)
        if isinstance(corpus, dict):
            iterator = corpus.items()
        else:
            iterator = [(row.get("_id", row.get("id")), row) for row in corpus]

        for doc_id, doc in iterator:
            text = doc.get("text", "") if isinstance(doc, dict) else ""
            chunk = {
                "chunk_id": doc_id,
                "code": text,
                "type": "function",
                "file_path": f"{doc_id}.py",
                "name": doc_id,
                "repository": "mteb_eval",
                "commit_hash": "mteb_eval",
                "branch": "main",
                "language": "python"
            }
            all_chunks.append(chunk)
            rep = build_representation(chunk)
            all_representations.append(rep)
            all_embeddings_text.append(text)
            
        db_path = "mteb_eval_metadata.db"
        if os.path.exists(db_path):
            os.remove(db_path)
        self.store = MetadataStore(db_path)
        
        self.embedder = Embedder()
        self.dense_idx = DenseIndex(384)
        self.bm25_idx = BM25Index()
        self.struct_idx = StructuralIndex()
        
        embeddings = self.embedder.encode(all_embeddings_text)
        chunk_ids = [c["chunk_id"] for c in all_chunks]
        
        self.dense_idx.add_embeddings(embeddings, chunk_ids)
        self.bm25_idx.add_chunks(all_chunks)
        self.struct_idx.add_chunks(all_chunks)
        
        print(f"DEBUG: Processed {len(all_chunks)} chunks from corpus.")
        
        self.store.insert_chunks(all_chunks, all_representations)
        print("DEBUG: Instantiating retrievers...")
        
        dense_retriever = DenseRetriever(self.embedder, self.dense_idx)
        bm25_retriever = BM25Retriever(self.bm25_idx)
        struct_retriever = StructuralRetriever(self.struct_idx)
        
        expander = QueryExpander()
        orchestrator = RetrievalOrchestrator(expander, dense_retriever, bm25_retriever, struct_retriever)
        rrf = ReciprocalRankFusion(k=60)
        cross_encoder = CrossEncoderReRanker()
        
        pipeline = RetrievalPipeline(orchestrator, rrf, cross_encoder, self.store)
        self.search_service = SearchService(pipeline, self.store)
        print("DEBUG: Exiting index() method...")

    def search(
        self,
        queries: Any,
        **kwargs
    ) -> Dict[str, Dict[str, float]]:
        print("DEBUG: Entering search() method...")
        results = {}
        
        if isinstance(queries, dict):
            iterator = queries.items()
        else:
            iterator = [(row.get("_id", row.get("id")), row.get("text", "")) for row in queries]

        print(f"DEBUG: Found {len(list(iterator)) if isinstance(iterator, list) else 'unknown'} queries.")
        for i, (qid, query_text) in enumerate(iterator):
            print(f"DEBUG: Processing query {i} (qid: {qid})")
            pipeline_results = self.search_service.search(query_text)
            qid_results = {}
            for res in pipeline_results:
                qid_results[res["chunk_id"]] = res["score"]
            results[qid] = qid_results
        return results

def main() -> None:
    model = PrePostPipelineEncoder()
    task = mteb.get_task("AppsRetrieval")
    result = mteb.evaluate(
        model,
        [task],
        encode_kwargs={"batch_size": 64},
    )
    task_result = list(result.task_results)[0]
    with open("appsretrieval_results.json", "w") as f:
        json.dump(task_result.to_dict(), f, indent=2, default=str)

if __name__ == "__main__":
    main()
