#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Script to run CyborgDB vs Original Blueprint benchmarks

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}CyborgDB Blueprint Performance Benchmark${NC}"
echo -e "${GREEN}========================================${NC}"

# Parse command line arguments
BENCHMARK_TYPE=${1:-"full"}  # full, vdb-only, e2e-only
NUM_DOCS=${2:-10000}
NUM_QUERIES=${3:-100}
OUTPUT_DIR=${4:-"./benchmark_results_$(date +%Y%m%d_%H%M%S)"}

echo -e "\n${YELLOW}Configuration:${NC}"
echo "  Benchmark Type: $BENCHMARK_TYPE"
echo "  Number of Documents: $NUM_DOCS"
echo "  Number of Queries: $NUM_QUERIES"
echo "  Output Directory: $OUTPUT_DIR"

# Check if running in Docker or bare metal
if [ -f /.dockerenv ]; then
    echo -e "\n${YELLOW}Running in Docker container${NC}"
    DOCKER_ENV=true
else
    echo -e "\n${YELLOW}Running on bare metal${NC}"
    DOCKER_ENV=false
fi

# Install dependencies if needed
echo -e "\n${YELLOW}Installing benchmark dependencies...${NC}"
pip install -q -r benchmarks/requirements.txt

# Set environment variables if not already set
export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"

# Function to run benchmark with error handling
run_benchmark() {
    local vdb_type=$1
    local device=$2
    local extra_args=$3
    
    echo -e "\n${GREEN}Running $vdb_type benchmark on $device...${NC}"
    
    if python benchmarks/cyborgdb_benchmark.py \
        --vdb-type "$vdb_type" \
        --device "$device" \
        --num-documents "$NUM_DOCS" \
        --num-queries "$NUM_QUERIES" \
        --output-dir "$OUTPUT_DIR" \
        $extra_args; then
        echo -e "${GREEN}✓ $vdb_type on $device completed successfully${NC}"
    else
        echo -e "${RED}✗ $vdb_type on $device failed${NC}"
        return 1
    fi
}

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Run benchmarks based on type
case $BENCHMARK_TYPE in
    "full")
        echo -e "\n${YELLOW}Running full benchmark suite...${NC}"
        
        # Vector DB benchmarks
        echo -e "\n${GREEN}=== Vector Database Benchmarks ===${NC}"
        
        # CyborgDB GPU
        run_benchmark "cyborgdb" "gpu" ""
        
        # CyborgDB CPU
        run_benchmark "cyborgdb" "cpu" ""
        
        # Milvus GPU
        run_benchmark "milvus" "gpu" ""
        
        # Milvus CPU
        run_benchmark "milvus" "cpu" ""
        
        # End-to-end benchmarks
        echo -e "\n${GREEN}=== End-to-End Benchmarks ===${NC}"
        run_benchmark "cyborgdb" "gpu" "--run-e2e"
        run_benchmark "milvus" "gpu" "--run-e2e"
        ;;
        
    "vdb-only")
        echo -e "\n${YELLOW}Running vector database benchmarks only...${NC}"
        run_benchmark "cyborgdb" "gpu" ""
        run_benchmark "cyborgdb" "cpu" ""
        run_benchmark "milvus" "gpu" ""
        run_benchmark "milvus" "cpu" ""
        ;;
        
    "e2e-only")
        echo -e "\n${YELLOW}Running end-to-end benchmarks only...${NC}"
        run_benchmark "cyborgdb" "gpu" "--run-e2e"
        run_benchmark "milvus" "gpu" "--run-e2e"
        ;;
        
    "quick")
        echo -e "\n${YELLOW}Running quick benchmark (reduced dataset)...${NC}"
        NUM_DOCS=1000
        NUM_QUERIES=10
        run_benchmark "both" "gpu" ""
        ;;
        
    *)
        echo -e "${RED}Unknown benchmark type: $BENCHMARK_TYPE${NC}"
        echo "Usage: $0 [full|vdb-only|e2e-only|quick] [num_docs] [num_queries] [output_dir]"
        exit 1
        ;;
esac

# Generate consolidated report
echo -e "\n${YELLOW}Generating consolidated report...${NC}"
python -c "
import json
import pandas as pd
from pathlib import Path

output_dir = Path('$OUTPUT_DIR')
all_results = []

# Load all result files
for result_file in output_dir.glob('*/raw_results.json'):
    with open(result_file) as f:
        all_results.extend(json.load(f))

if all_results:
    # Create summary DataFrame
    df = pd.DataFrame(all_results)
    
    # Save consolidated results
    df.to_csv(output_dir / 'consolidated_results.csv', index=False)
    
    # Print summary statistics
    print('\n=== Benchmark Summary ===')
    print(df.groupby(['vdb_type', 'device', 'metric_type'])['value'].agg(['mean', 'std', 'min', 'max']))
    
    # Calculate improvements
    cyborgdb_metrics = df[df['vdb_type'] == 'cyborgdb'].groupby('metric_name')['value'].mean()
    milvus_metrics = df[df['vdb_type'] == 'milvus'].groupby('metric_name')['value'].mean()
    
    print('\n=== CyborgDB vs Milvus Improvements ===')
    for metric in cyborgdb_metrics.index:
        if metric in milvus_metrics.index:
            cyborgdb_val = cyborgdb_metrics[metric]
            milvus_val = milvus_metrics[metric]
            if milvus_val != 0:
                improvement = ((cyborgdb_val - milvus_val) / milvus_val) * 100
                print(f'{metric}: {improvement:+.1f}%')
"

echo -e "\n${GREEN}========================================${NC}"
echo -e "${GREEN}Benchmark completed successfully!${NC}"
echo -e "${GREEN}Results saved to: $OUTPUT_DIR${NC}"
echo -e "${GREEN}========================================${NC}"

# Display location of key files
echo -e "\n${YELLOW}Key output files:${NC}"
echo "  - Summary Report: $OUTPUT_DIR/benchmark_report.md"
echo "  - Comparison Data: $OUTPUT_DIR/comparison.csv"
echo "  - Raw Results: $OUTPUT_DIR/raw_results.json"
echo "  - Consolidated Results: $OUTPUT_DIR/consolidated_results.csv"