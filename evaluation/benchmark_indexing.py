import time
import os
from typing import Dict, Any

class IndexingBenchmark:
    def __init__(self):
        self.metrics = {
            "repository_scanning_time": 0.0,
            "ast_parsing_time": 0.0,
            "chunking_time": 0.0,
            "embedding_time": 0.0,
            "faiss_build_time": 0.0,
            "bm25_build_time": 0.0,
            "total_indexing_time": 0.0,
            "number_of_files": 0,
            "number_of_chunks": 0,
            "embedding_dimensions": 384,
            "index_size_bytes": 0
        }
        self._start_times = {}

    def start_timer(self, metric: str):
        """Starts a high-resolution timer for a specific ingestion phase."""
        self._start_times[metric] = time.perf_counter()

    def stop_timer(self, metric: str):
        """Stops the timer and aggregates the elapsed milliseconds."""
        if metric in self._start_times:
            elapsed = (time.perf_counter() - self._start_times[metric]) * 1000
            if metric in self.metrics:
                self.metrics[metric] += elapsed
            del self._start_times[metric]
            
    def record_stat(self, stat: str, value: Any):
        """Records flat statistical numbers (e.g. file counts, chunk counts)."""
        if stat in self.metrics:
            self.metrics[stat] = value
            
    def calculate_directory_size(self, directory: str) -> int:
        """Walks the isolated index directory to sum the total byte footprint."""
        total_size = 0
        if not os.path.exists(directory):
            return 0
        for dirpath, _, filenames in os.walk(directory):
            for f in filenames:
                fp = os.path.join(dirpath, f)
                if not os.path.islink(fp):
                    total_size += os.path.getsize(fp)
        self.metrics["index_size_bytes"] = total_size
        return total_size

    def get_report(self) -> Dict[str, Any]:
        return self.metrics
