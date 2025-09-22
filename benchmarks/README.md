# CyborgDB Blueprint Performance Benchmarking Suite

This benchmarking suite provides comprehensive performance comparisons between the CyborgDB Blueprint and the Original (Milvus) Blueprint across multiple metrics.

## Metrics Collected

### Vector Database Performance
- **QPS vs Recall**: Queries per second at different recall levels
- **Query Latency**: P50, P95, P99 latencies in milliseconds
- **Index Build Time**: Time to build vector index (seconds)
- **Upsert Performance**: Embeddings per second ingestion rate

### End-to-End Performance
- **Accuracy**: Response quality metrics
- **Latency (TTFT)**: Time to first token in milliseconds
- **Throughput**: Tokens per second generation rate

### GPU vs CPU Comparison
- **Index Build Time**: Speedup factor
- **Query Performance**: Latency comparison
- **Upsert Performance**: Throughput comparison

## Quick Start

### Prerequisites

```bash
# Install Python dependencies
pip install -r benchmarks/requirements.txt

# Set environment variables
export CYBORGDB_API_KEY="your_api_key"
export CYBORGDB_INDEX_KEY="your_index_key"
export CYBORGDB_URI="http://cyborgdb:8000"
export MILVUS_URI="http://milvus:19530"
```

### Running Benchmarks

#### Option 1: Using the Shell Script

```bash
# Run full benchmark suite
./benchmarks/run_benchmark.sh full

# Run only vector DB benchmarks
./benchmarks/run_benchmark.sh vdb-only

# Run only end-to-end benchmarks
./benchmarks/run_benchmark.sh e2e-only

# Run quick benchmark (smaller dataset)
./benchmarks/run_benchmark.sh quick

# Custom parameters
./benchmarks/run_benchmark.sh full 100000 1000 ./my_results
# Arguments: [type] [num_documents] [num_queries] [output_dir]
```

#### Option 2: Using Python Directly

```bash
# Run CyborgDB benchmarks on GPU
python benchmarks/cyborgdb_benchmark.py \
  --vdb-type cyborgdb \
  --device gpu \
  --num-documents 10000 \
  --num-queries 100 \
  --output-dir ./results

# Run comparison between CyborgDB and Milvus
python benchmarks/cyborgdb_benchmark.py \
  --vdb-type both \
  --device both \
  --run-e2e

# Run with custom configuration
python benchmarks/cyborgdb_benchmark.py \
  --vdb-type cyborgdb \
  --device gpu \
  --num-documents 100000 \
  --num-queries 1000 \
  --output-dir ./benchmark_results
```

#### Option 3: Using Docker Compose

```bash
# Start benchmark environment
docker-compose -f benchmarks/docker-compose.benchmark.yml up -d

# Run benchmarks in container
docker-compose -f benchmarks/docker-compose.benchmark.yml run benchmark-runner

# View results
ls benchmark_results/
```

## Output Files

The benchmark suite generates the following outputs:

- `raw_results.json`: Complete benchmark data in JSON format
- `results.csv`: Tabular results for analysis
- `summary.json`: Statistical summary of all metrics
- `comparison.csv`: Side-by-side comparison of CyborgDB vs Milvus
- `benchmark_report.md`: Markdown report ready for blog publication
- `consolidated_results.csv`: Combined results from all runs

## Benchmark Configuration

### BenchmarkConfig Parameters

```python
@dataclass
class BenchmarkConfig:
    vdb_type: str           # "cyborgdb" or "milvus"
    device: str             # "gpu" or "cpu"
    collection_name: str    # Name of test collection
    embedding_dim: int      # Dimension of embeddings (default: 1536)
    num_documents: int      # Number of test documents
    num_queries: int        # Number of test queries
    batch_sizes: List[int]  # Batch sizes to test
    top_k_values: List[int] # Top-k values to test
    num_threads: int        # Parallel threads for testing
    warmup_runs: int        # Number of warmup iterations
    test_runs: int          # Number of test iterations
```

### Environment Variables

```bash
# CyborgDB Configuration
CYBORGDB_URI=http://cyborgdb:8000
CYBORGDB_API_KEY=your_api_key
CYBORGDB_INDEX_KEY=your_index_key

# Milvus Configuration
MILVUS_URI=http://milvus:19530

# RAG Server (for E2E benchmarks)
RAG_SERVER_URL=http://localhost:8081

# GPU Configuration
CUDA_VISIBLE_DEVICES=0
CYBORGDB_GPU_ENABLED=true
```

## Interpreting Results

### Key Metrics to Watch

1. **Query Latency**: Lower is better
   - P50 < 10ms (excellent)
   - P95 < 50ms (good)
   - P99 < 100ms (acceptable)

2. **QPS (Queries Per Second)**: Higher is better
   - > 1000 QPS (excellent)
   - > 500 QPS (good)
   - > 100 QPS (acceptable)

3. **Recall@K**: Higher is better
   - > 0.95 (excellent)
   - > 0.90 (good)
   - > 0.85 (acceptable)

4. **Index Build Time**: Lower is better
   - Measured in seconds per 100K documents

5. **Upsert Performance**: Higher is better
   - Measured in embeddings/second

### Sample Output

```markdown
# CyborgDB Blueprint Performance Benchmark Results

## Executive Summary

### Key Performance Improvements (CyborgDB vs Original Blueprint)

| Metric | CyborgDB | Original (Milvus) | Improvement |
|--------|----------|-------------------|-------------|
| Query Latency P50 (ms) | 5.2 | 12.3 | +57.7% |
| Query Latency P95 (ms) | 15.1 | 45.6 | +66.9% |
| QPS (batch_10) | 1823 | 892 | +104.4% |
| Recall@10 | 0.96 | 0.94 | +2.1% |
| Index Build Time (s) | 45.2 | 78.3 | +42.3% |
| Upsert Rate (emb/s) | 5420 | 3210 | +68.8% |
```

## Advanced Usage

### Custom Benchmark Implementation

```python
from benchmarks.cyborgdb_benchmark import VectorDBBenchmark, BenchmarkConfig

class MyCustomBenchmark(VectorDBBenchmark):
    def setup(self):
        # Initialize your vector database
        pass
    
    def measure_custom_metric(self):
        # Implement custom measurement
        pass

# Run custom benchmark
config = BenchmarkConfig(
    vdb_type="custom",
    device="gpu",
    num_documents=10000
)
benchmark = MyCustomBenchmark(config)
results = benchmark.run_benchmark_suite()
```

### Batch Processing

For large-scale benchmarks, use batch processing:

```bash
# Run benchmarks in parallel
for size in 1000 10000 100000; do
    ./benchmarks/run_benchmark.sh full $size 100 "./results_${size}" &
done
wait

# Consolidate results
python benchmarks/consolidate_results.py ./results_*
```

## Troubleshooting

### Common Issues

1. **GPU Not Available**
   ```bash
   # Check GPU availability
   nvidia-smi
   
   # Set CUDA device
   export CUDA_VISIBLE_DEVICES=0
   ```

2. **Connection Errors**
   ```bash
   # Verify services are running
   docker ps
   
   # Check network connectivity
   curl http://cyborgdb:8000/health
   curl http://milvus:19530/health
   ```

3. **Memory Issues**
   ```bash
   # Reduce dataset size
   ./benchmarks/run_benchmark.sh quick
   
   # Or adjust in Python
   --num-documents 1000 --num-queries 10
   ```

## Contributing

To add new benchmarks:

1. Extend the `VectorDBBenchmark` base class
2. Implement required methods
3. Add metric collection
4. Update the reporter for new metrics

## License

SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0