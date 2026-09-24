import tempfile
import os
import time
from evaluation.benchmark_indexing import IndexingBenchmark

def test_indexing_benchmark():
    benchmark = IndexingBenchmark()
    
    # 1. Test Timer Tracking
    benchmark.start_timer("total_indexing_time")
    benchmark.start_timer("ast_parsing_time")
    
    # Simulate work
    time.sleep(0.01)
    
    benchmark.stop_timer("ast_parsing_time")
    benchmark.stop_timer("total_indexing_time")
    
    report = benchmark.get_report()
    
    assert report["ast_parsing_time"] > 0.0
    assert report["total_indexing_time"] > 0.0
    # AST parsing is a sub-component, so total time should be at least equal or greater
    assert report["total_indexing_time"] >= report["ast_parsing_time"]
    
    # 2. Test Statistical Data Tracking
    benchmark.record_stat("number_of_files", 150)
    benchmark.record_stat("number_of_chunks", 3500)
    
    assert report["number_of_files"] == 150
    assert report["number_of_chunks"] == 3500
    assert report["embedding_dimensions"] == 384
    
    # 3. Test File Footprint Calculation
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create a dummy payload file simulating an index
        dummy_file = os.path.join(temp_dir, "faiss.index")
        with open(dummy_file, "wb") as f:
            f.write(b"0" * 1024) # 1 KB
            
        size = benchmark.calculate_directory_size(temp_dir)
        
        assert size == 1024
        assert report["index_size_bytes"] == 1024
