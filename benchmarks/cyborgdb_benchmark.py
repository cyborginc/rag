#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
CyborgDB Blueprint Performance Benchmark Suite

This module provides comprehensive benchmarking for CyborgDB vs Original Blueprint
across various metrics including:
- Vector DB performance (QPS vs Recall)
- Vector DB latency (query milliseconds)
- End-to-end accuracy
- End-to-end latency (time to first token)
- Throughput (tokens/second)
- GPU vs CPU comparison
- Index build time
- Upsert performance (embeddings/second)
"""

import argparse
import json
import logging
import os
import time
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
import psutil
import GPUtil

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class BenchmarkConfig:
    """Configuration for benchmark runs"""
    vdb_type: str  # "cyborgdb" or "milvus"
    device: str  # "gpu" or "cpu"
    collection_name: str = "benchmark_collection"
    embedding_dim: int = 1536
    num_documents: int = 100000
    num_queries: int = 1000
    batch_sizes: List[int] = None
    top_k_values: List[int] = None
    num_threads: int = 1
    warmup_runs: int = 3
    test_runs: int = 10
    
    def __post_init__(self):
        if self.batch_sizes is None:
            self.batch_sizes = [1, 10, 50, 100, 500, 1000]
        if self.top_k_values is None:
            self.top_k_values = [1, 5, 10, 20, 50, 100]


@dataclass
class BenchmarkResult:
    """Container for benchmark results"""
    timestamp: str
    vdb_type: str
    device: str
    metric_type: str
    metric_name: str
    value: float
    metadata: Dict[str, Any] = None
    
    def to_dict(self):
        return asdict(self)


class VectorDBBenchmark:
    """Base class for vector database benchmarking"""
    
    def __init__(self, config: BenchmarkConfig):
        self.config = config
        self.results: List[BenchmarkResult] = []
        self.vdb_client = None
        
    def setup(self):
        """Initialize vector database connection"""
        raise NotImplementedError
        
    def teardown(self):
        """Cleanup vector database connection"""
        raise NotImplementedError
        
    def generate_synthetic_data(self) -> Tuple[np.ndarray, List[Dict]]:
        """Generate synthetic embeddings and metadata"""
        logger.info(f"Generating {self.config.num_documents} synthetic documents...")
        
        # Generate random embeddings
        embeddings = np.random.randn(
            self.config.num_documents, 
            self.config.embedding_dim
        ).astype(np.float32)
        
        # Normalize embeddings
        embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
        
        # Generate metadata
        metadata = []
        for i in range(self.config.num_documents):
            metadata.append({
                "doc_id": f"doc_{i}",
                "source": f"source_{i % 100}",
                "category": f"category_{i % 10}",
                "timestamp": datetime.now().isoformat(),
                "content": f"This is document {i} content for benchmarking purposes."
            })
        
        return embeddings, metadata
    
    def generate_query_embeddings(self, num_queries: int) -> np.ndarray:
        """Generate query embeddings"""
        queries = np.random.randn(
            num_queries,
            self.config.embedding_dim
        ).astype(np.float32)
        
        # Normalize
        queries = queries / np.linalg.norm(queries, axis=1, keepdims=True)
        return queries
    
    def measure_index_build_time(self, embeddings: np.ndarray, metadata: List[Dict]) -> float:
        """Measure time to build index"""
        raise NotImplementedError
    
    def measure_upsert_performance(self, embeddings: np.ndarray, metadata: List[Dict]) -> Dict:
        """Measure upsert performance metrics"""
        raise NotImplementedError
    
    def measure_query_latency(self, queries: np.ndarray, top_k: int) -> Dict:
        """Measure query latency statistics"""
        raise NotImplementedError
    
    def measure_qps_vs_recall(self, queries: np.ndarray, ground_truth: np.ndarray) -> Dict:
        """Measure QPS vs Recall trade-off"""
        raise NotImplementedError
    
    def calculate_recall(self, retrieved: np.ndarray, ground_truth: np.ndarray) -> float:
        """Calculate recall@k metric"""
        correct = 0
        for ret, gt in zip(retrieved, ground_truth):
            correct += len(set(ret) & set(gt))
        return correct / (len(retrieved) * len(ground_truth[0]))
    
    def get_system_metrics(self) -> Dict:
        """Get current system resource usage"""
        metrics = {
            "cpu_percent": psutil.cpu_percent(interval=1),
            "memory_percent": psutil.virtual_memory().percent,
            "memory_used_gb": psutil.virtual_memory().used / (1024**3),
        }
        
        # GPU metrics if available
        try:
            gpus = GPUtil.getGPUs()
            if gpus:
                gpu = gpus[0]  # Use first GPU
                metrics.update({
                    "gpu_utilization": gpu.load * 100,
                    "gpu_memory_percent": gpu.memoryUtil * 100,
                    "gpu_memory_used_mb": gpu.memoryUsed,
                    "gpu_temperature": gpu.temperature
                })
        except Exception as e:
            logger.debug(f"Could not get GPU metrics: {e}")
        
        return metrics
    
    def run_benchmark_suite(self):
        """Run complete benchmark suite"""
        logger.info(f"Starting benchmark suite for {self.config.vdb_type} on {self.config.device}")
        
        # Setup
        self.setup()
        
        try:
            # Generate test data
            embeddings, metadata = self.generate_synthetic_data()
            queries = self.generate_query_embeddings(self.config.num_queries)
            
            # 1. Index Build Time
            logger.info("Measuring index build time...")
            build_time = self.measure_index_build_time(embeddings, metadata)
            self.results.append(BenchmarkResult(
                timestamp=datetime.now().isoformat(),
                vdb_type=self.config.vdb_type,
                device=self.config.device,
                metric_type="index_build",
                metric_name="build_time_seconds",
                value=build_time,
                metadata={"num_documents": self.config.num_documents}
            ))
            
            # 2. Upsert Performance
            logger.info("Measuring upsert performance...")
            upsert_metrics = self.measure_upsert_performance(embeddings[:10000], metadata[:10000])
            for metric_name, value in upsert_metrics.items():
                self.results.append(BenchmarkResult(
                    timestamp=datetime.now().isoformat(),
                    vdb_type=self.config.vdb_type,
                    device=self.config.device,
                    metric_type="upsert",
                    metric_name=metric_name,
                    value=value,
                    metadata={"batch_size": 1000}
                ))
            
            # 3. Query Latency
            for top_k in self.config.top_k_values:
                logger.info(f"Measuring query latency for top_k={top_k}...")
                latency_metrics = self.measure_query_latency(queries[:100], top_k)
                for metric_name, value in latency_metrics.items():
                    self.results.append(BenchmarkResult(
                        timestamp=datetime.now().isoformat(),
                        vdb_type=self.config.vdb_type,
                        device=self.config.device,
                        metric_type="query_latency",
                        metric_name=f"{metric_name}_top{top_k}",
                        value=value,
                        metadata={"top_k": top_k}
                    ))
            
            # 4. QPS vs Recall
            logger.info("Measuring QPS vs Recall...")
            # Generate ground truth (for synthetic data, we'll use brute force search)
            ground_truth = self.compute_ground_truth(queries[:100], embeddings, k=10)
            qps_recall_metrics = self.measure_qps_vs_recall(queries[:100], ground_truth)
            for metric_name, value in qps_recall_metrics.items():
                self.results.append(BenchmarkResult(
                    timestamp=datetime.now().isoformat(),
                    vdb_type=self.config.vdb_type,
                    device=self.config.device,
                    metric_type="qps_recall",
                    metric_name=metric_name,
                    value=value,
                    metadata={}
                ))
            
            # 5. System Resource Usage
            sys_metrics = self.get_system_metrics()
            for metric_name, value in sys_metrics.items():
                self.results.append(BenchmarkResult(
                    timestamp=datetime.now().isoformat(),
                    vdb_type=self.config.vdb_type,
                    device=self.config.device,
                    metric_type="system",
                    metric_name=metric_name,
                    value=value,
                    metadata={}
                ))
                
        finally:
            self.teardown()
        
        return self.results
    
    def compute_ground_truth(self, queries: np.ndarray, embeddings: np.ndarray, k: int) -> np.ndarray:
        """Compute ground truth using brute force search"""
        logger.info("Computing ground truth with brute force search...")
        ground_truth = []
        
        for query in queries:
            # Compute cosine similarity
            similarities = np.dot(embeddings, query)
            # Get top-k indices
            top_k_indices = np.argsort(similarities)[-k:][::-1]
            ground_truth.append(top_k_indices)
        
        return np.array(ground_truth)


class CyborgDBBenchmark(VectorDBBenchmark):
    """CyborgDB-specific benchmark implementation"""
    
    def setup(self):
        """Initialize CyborgDB connection"""
        from nvidia_rag.utils.vdb.cyborgdb.cyborgdb_vdb import CyborgDBVDB
        
        # Get configuration from environment or use defaults
        self.vdb_client = CyborgDBVDB(
            collection_name=self.config.collection_name,
            cyborgdb_uri=os.getenv("CYBORGDB_URI", "http://cyborgdb:8000"),
            api_key=os.getenv("CYBORGDB_API_KEY", ""),
            index_key=os.getenv("CYBORGDB_INDEX_KEY", "").encode(),
        )
        
        # Create collection if needed
        if not self.vdb_client.check_collection_exists(self.config.collection_name):
            self.vdb_client.create_collection(
                self.config.collection_name,
                dimension=self.config.embedding_dim
            )
    
    def teardown(self):
        """Cleanup CyborgDB connection"""
        # Optionally delete test collection
        pass
    
    def measure_index_build_time(self, embeddings: np.ndarray, metadata: List[Dict]) -> float:
        """Measure time to build CyborgDB index"""
        # Prepare records
        records = []
        for i, (emb, meta) in enumerate(zip(embeddings, metadata)):
            records.append({
                "id": f"vec_{i}",
                "vector": emb.tolist(),
                "metadata": meta
            })
        
        # Measure build time
        start_time = time.time()
        
        # Write in batches
        batch_size = 1000
        for i in range(0, len(records), batch_size):
            batch = records[i:i+batch_size]
            self.vdb_client.write_to_index(batch, collection_name=self.config.collection_name)
        
        build_time = time.time() - start_time
        return build_time
    
    def measure_upsert_performance(self, embeddings: np.ndarray, metadata: List[Dict]) -> Dict:
        """Measure CyborgDB upsert performance"""
        records = []
        for i, (emb, meta) in enumerate(zip(embeddings, metadata)):
            records.append({
                "id": f"upsert_{i}",
                "vector": emb.tolist(),
                "metadata": meta
            })
        
        # Warmup
        for _ in range(self.config.warmup_runs):
            self.vdb_client.write_to_index(records[:100], collection_name=self.config.collection_name)
        
        # Actual measurement
        latencies = []
        for _ in range(self.config.test_runs):
            start = time.time()
            self.vdb_client.write_to_index(records[:1000], collection_name=self.config.collection_name)
            latencies.append(time.time() - start)
        
        embeddings_per_second = 1000 / np.mean(latencies)
        
        return {
            "mean_latency_ms": np.mean(latencies) * 1000,
            "p50_latency_ms": np.percentile(latencies, 50) * 1000,
            "p95_latency_ms": np.percentile(latencies, 95) * 1000,
            "p99_latency_ms": np.percentile(latencies, 99) * 1000,
            "embeddings_per_second": embeddings_per_second
        }
    
    def measure_query_latency(self, queries: np.ndarray, top_k: int) -> Dict:
        """Measure CyborgDB query latency"""
        vectorstore = self.vdb_client.get_langchain_vectorstore(self.config.collection_name)
        
        # Warmup
        for _ in range(self.config.warmup_runs):
            vectorstore.similarity_search_by_vector(queries[0].tolist(), k=top_k)
        
        # Actual measurement
        latencies = []
        for query in queries:
            start = time.time()
            vectorstore.similarity_search_by_vector(query.tolist(), k=top_k)
            latencies.append(time.time() - start)
        
        return {
            "mean_latency_ms": np.mean(latencies) * 1000,
            "p50_latency_ms": np.percentile(latencies, 50) * 1000,
            "p95_latency_ms": np.percentile(latencies, 95) * 1000,
            "p99_latency_ms": np.percentile(latencies, 99) * 1000,
            "min_latency_ms": np.min(latencies) * 1000,
            "max_latency_ms": np.max(latencies) * 1000
        }
    
    def measure_qps_vs_recall(self, queries: np.ndarray, ground_truth: np.ndarray) -> Dict:
        """Measure CyborgDB QPS vs Recall"""
        vectorstore = self.vdb_client.get_langchain_vectorstore(self.config.collection_name)
        
        # Test different batch sizes for QPS
        results = {}
        for batch_size in [1, 10, 50, 100]:
            start = time.time()
            retrieved = []
            
            for i in range(0, len(queries), batch_size):
                batch = queries[i:i+batch_size]
                for query in batch:
                    docs = vectorstore.similarity_search_by_vector(query.tolist(), k=10)
                    # Extract IDs from results
                    doc_ids = [int(doc.metadata.get("doc_id", "doc_0").split("_")[1]) for doc in docs]
                    retrieved.append(doc_ids)
            
            elapsed = time.time() - start
            qps = len(queries) / elapsed
            
            # Calculate recall
            recall = self.calculate_recall(np.array(retrieved), ground_truth)
            
            results[f"qps_batch_{batch_size}"] = qps
            results[f"recall_batch_{batch_size}"] = recall
        
        return results


class ElasticsearchBenchmark(VectorDBBenchmark):
    """Elasticsearch-specific benchmark implementation"""
    
    def setup(self):
        """Initialize Elasticsearch connection"""
        from nvidia_rag.utils.vdb.elasticsearch.elastic_vdb import ElasticVDB
        
        self.vdb_client = ElasticVDB(
            index_name=self.config.collection_name,
            es_url=os.getenv("ELASTICSEARCH_URL", "http://elasticsearch:9200"),
            embedding_model=None
        )
        
        # Create index if needed
        if not self.vdb_client.check_collection_exists(self.config.collection_name):
            self.vdb_client.create_collection(
                self.config.collection_name,
                dimension=self.config.embedding_dim
            )
    
    def teardown(self):
        """Cleanup Elasticsearch connection"""
        pass
    
    def measure_index_build_time(self, embeddings: np.ndarray, metadata: List[Dict]) -> float:
        """Measure time to build Elasticsearch index"""
        records = []
        for i, (emb, meta) in enumerate(zip(embeddings, metadata)):
            records.append({
                "id": f"vec_{i}",
                "vector": emb.tolist(),
                "metadata": meta
            })
        
        start_time = time.time()
        
        # Write in batches
        batch_size = 1000
        for i in range(0, len(records), batch_size):
            batch = records[i:i+batch_size]
            self.vdb_client.write_to_index(batch, collection_name=self.config.collection_name)
        
        build_time = time.time() - start_time
        return build_time
    
    def measure_upsert_performance(self, embeddings: np.ndarray, metadata: List[Dict]) -> Dict:
        """Measure Elasticsearch upsert performance"""
        records = []
        for i, (emb, meta) in enumerate(zip(embeddings, metadata)):
            records.append({
                "id": f"upsert_{i}",
                "vector": emb.tolist(),
                "metadata": meta
            })
        
        # Warmup
        for _ in range(self.config.warmup_runs):
            self.vdb_client.write_to_index(records[:100], collection_name=self.config.collection_name)
        
        # Actual measurement
        latencies = []
        for _ in range(self.config.test_runs):
            start = time.time()
            self.vdb_client.write_to_index(records[:1000], collection_name=self.config.collection_name)
            latencies.append(time.time() - start)
        
        embeddings_per_second = 1000 / np.mean(latencies)
        
        return {
            "mean_latency_ms": np.mean(latencies) * 1000,
            "p50_latency_ms": np.percentile(latencies, 50) * 1000,
            "p95_latency_ms": np.percentile(latencies, 95) * 1000,
            "p99_latency_ms": np.percentile(latencies, 99) * 1000,
            "embeddings_per_second": embeddings_per_second
        }
    
    def measure_query_latency(self, queries: np.ndarray, top_k: int) -> Dict:
        """Measure Elasticsearch query latency"""
        vectorstore = self.vdb_client.get_langchain_vectorstore(self.config.collection_name)
        
        # Warmup
        for _ in range(self.config.warmup_runs):
            vectorstore.similarity_search_by_vector(queries[0].tolist(), k=top_k)
        
        # Actual measurement
        latencies = []
        for query in queries:
            start = time.time()
            vectorstore.similarity_search_by_vector(query.tolist(), k=top_k)
            latencies.append(time.time() - start)
        
        return {
            "mean_latency_ms": np.mean(latencies) * 1000,
            "p50_latency_ms": np.percentile(latencies, 50) * 1000,
            "p95_latency_ms": np.percentile(latencies, 95) * 1000,
            "p99_latency_ms": np.percentile(latencies, 99) * 1000,
            "min_latency_ms": np.min(latencies) * 1000,
            "max_latency_ms": np.max(latencies) * 1000
        }
    
    def measure_qps_vs_recall(self, queries: np.ndarray, ground_truth: np.ndarray) -> Dict:
        """Measure Elasticsearch QPS vs Recall"""
        vectorstore = self.vdb_client.get_langchain_vectorstore(self.config.collection_name)
        
        # Test different batch sizes for QPS
        results = {}
        for batch_size in [1, 10, 50, 100]:
            start = time.time()
            retrieved = []
            
            for i in range(0, len(queries), batch_size):
                batch = queries[i:i+batch_size]
                for query in batch:
                    docs = vectorstore.similarity_search_by_vector(query.tolist(), k=10)
                    # Extract IDs from results
                    doc_ids = [int(doc.metadata.get("doc_id", "doc_0").split("_")[1]) for doc in docs]
                    retrieved.append(doc_ids)
            
            elapsed = time.time() - start
            qps = len(queries) / elapsed
            
            # Calculate recall
            recall = self.calculate_recall(np.array(retrieved), ground_truth)
            
            results[f"qps_batch_{batch_size}"] = qps
            results[f"recall_batch_{batch_size}"] = recall
        
        return results


class MilvusBenchmark(VectorDBBenchmark):
    """Milvus-specific benchmark implementation"""
    
    def setup(self):
        """Initialize Milvus connection"""
        from nvidia_rag.utils.vdb.milvus.milvus_vdb import MilvusVDB
        
        self.vdb_client = MilvusVDB(
            collection_name=self.config.collection_name,
            milvus_uri=os.getenv("MILVUS_URI", "http://milvus:19530"),
            embedding_model=None
        )
        
        # Create collection if needed
        if not self.vdb_client.check_collection_exists(self.config.collection_name):
            self.vdb_client.create_collection(
                self.config.collection_name,
                dimension=self.config.embedding_dim
            )
    
    def teardown(self):
        """Cleanup Milvus connection"""
        pass
    
    def measure_index_build_time(self, embeddings: np.ndarray, metadata: List[Dict]) -> float:
        """Measure time to build Milvus index"""
        # Similar implementation to CyborgDB
        records = []
        for i, (emb, meta) in enumerate(zip(embeddings, metadata)):
            records.append({
                "id": f"vec_{i}",
                "vector": emb.tolist(),
                "metadata": meta
            })
        
        start_time = time.time()
        
        # Write in batches
        batch_size = 1000
        for i in range(0, len(records), batch_size):
            batch = records[i:i+batch_size]
            self.vdb_client.write_to_index(batch)
        
        build_time = time.time() - start_time
        return build_time
    
    def measure_upsert_performance(self, embeddings: np.ndarray, metadata: List[Dict]) -> Dict:
        """Measure Milvus upsert performance"""
        # Similar to CyborgDB implementation
        records = []
        for i, (emb, meta) in enumerate(zip(embeddings, metadata)):
            records.append({
                "id": f"upsert_{i}",
                "vector": emb.tolist(),
                "metadata": meta
            })
        
        # Warmup
        for _ in range(self.config.warmup_runs):
            self.vdb_client.write_to_index(records[:100])
        
        # Actual measurement
        latencies = []
        for _ in range(self.config.test_runs):
            start = time.time()
            self.vdb_client.write_to_index(records[:1000])
            latencies.append(time.time() - start)
        
        embeddings_per_second = 1000 / np.mean(latencies)
        
        return {
            "mean_latency_ms": np.mean(latencies) * 1000,
            "p50_latency_ms": np.percentile(latencies, 50) * 1000,
            "p95_latency_ms": np.percentile(latencies, 95) * 1000,
            "p99_latency_ms": np.percentile(latencies, 99) * 1000,
            "embeddings_per_second": embeddings_per_second
        }
    
    def measure_query_latency(self, queries: np.ndarray, top_k: int) -> Dict:
        """Measure Milvus query latency"""
        # Get vectorstore
        vectorstore = self.vdb_client.get_langchain_vectorstore(self.config.collection_name)
        
        # Warmup
        for _ in range(self.config.warmup_runs):
            vectorstore.similarity_search_by_vector(queries[0].tolist(), k=top_k)
        
        # Actual measurement
        latencies = []
        for query in queries:
            start = time.time()
            vectorstore.similarity_search_by_vector(query.tolist(), k=top_k)
            latencies.append(time.time() - start)
        
        return {
            "mean_latency_ms": np.mean(latencies) * 1000,
            "p50_latency_ms": np.percentile(latencies, 50) * 1000,
            "p95_latency_ms": np.percentile(latencies, 95) * 1000,
            "p99_latency_ms": np.percentile(latencies, 99) * 1000,
            "min_latency_ms": np.min(latencies) * 1000,
            "max_latency_ms": np.max(latencies) * 1000
        }
    
    def measure_qps_vs_recall(self, queries: np.ndarray, ground_truth: np.ndarray) -> Dict:
        """Measure Milvus QPS vs Recall"""
        vectorstore = self.vdb_client.get_langchain_vectorstore(self.config.collection_name)
        
        # Test different batch sizes for QPS
        results = {}
        for batch_size in [1, 10, 50, 100]:
            start = time.time()
            retrieved = []
            
            for i in range(0, len(queries), batch_size):
                batch = queries[i:i+batch_size]
                for query in batch:
                    docs = vectorstore.similarity_search_by_vector(query.tolist(), k=10)
                    # Extract IDs from results
                    doc_ids = [int(doc.metadata.get("doc_id", "doc_0").split("_")[1]) for doc in docs]
                    retrieved.append(doc_ids)
            
            elapsed = time.time() - start
            qps = len(queries) / elapsed
            
            # Calculate recall
            recall = self.calculate_recall(np.array(retrieved), ground_truth)
            
            results[f"qps_batch_{batch_size}"] = qps
            results[f"recall_batch_{batch_size}"] = recall
        
        return results


class EndToEndBenchmark:
    """End-to-end RAG pipeline benchmarking"""
    
    def __init__(self, config: BenchmarkConfig):
        self.config = config
        self.results: List[BenchmarkResult] = []
    
    def measure_rag_latency(self, query: str) -> Dict:
        """Measure end-to-end RAG latency (time to first token)"""
        import requests
        
        # RAG server endpoint
        rag_url = os.getenv("RAG_SERVER_URL", "http://localhost:8081")
        
        # Prepare request
        payload = {
            "messages": [{"role": "user", "content": query}],
            "use_knowledge_base": True,
            "temperature": 0.1
        }
        
        # Measure latency
        start = time.time()
        response = requests.post(
            f"{rag_url}/generate",
            json=payload,
            stream=True
        )
        
        # Time to first token
        first_token_time = None
        tokens = []
        
        for line in response.iter_lines():
            if first_token_time is None:
                first_token_time = time.time() - start
            
            if line:
                tokens.append(line.decode('utf-8'))
        
        total_time = time.time() - start
        
        return {
            "time_to_first_token_ms": first_token_time * 1000 if first_token_time else 0,
            "total_latency_ms": total_time * 1000,
            "num_tokens": len(tokens),
            "tokens_per_second": len(tokens) / total_time if total_time > 0 else 0
        }
    
    def measure_rag_accuracy(self, test_questions: List[Dict]) -> Dict:
        """Measure end-to-end RAG accuracy using test questions"""
        import requests
        from difflib import SequenceMatcher
        
        rag_url = os.getenv("RAG_SERVER_URL", "http://localhost:8081")
        
        correct = 0
        total = len(test_questions)
        similarities = []
        
        for qa in test_questions:
            question = qa["question"]
            expected_answer = qa["answer"]
            
            payload = {
                "messages": [{"role": "user", "content": question}],
                "use_knowledge_base": True,
                "temperature": 0.1
            }
            
            response = requests.post(f"{rag_url}/generate", json=payload)
            actual_answer = response.json().get("choices", [{}])[0].get("message", {}).get("content", "")
            
            # Calculate similarity
            similarity = SequenceMatcher(None, expected_answer.lower(), actual_answer.lower()).ratio()
            similarities.append(similarity)
            
            # Consider correct if similarity > 0.8
            if similarity > 0.8:
                correct += 1
        
        return {
            "accuracy": correct / total,
            "mean_similarity": np.mean(similarities),
            "median_similarity": np.median(similarities)
        }
    
    def run_benchmark(self, test_questions: List[Dict] = None):
        """Run end-to-end benchmarks"""
        logger.info("Starting end-to-end benchmark suite")
        
        # Default test questions if none provided
        if test_questions is None:
            test_questions = [
                {"question": "What is RAG?", "answer": "Retrieval Augmented Generation"},
                {"question": "What vector database does this use?", "answer": "CyborgDB or Milvus"},
            ]
        
        # 1. Latency measurements
        logger.info("Measuring end-to-end latency...")
        latencies = []
        for _ in range(10):
            metrics = self.measure_rag_latency("What is retrieval augmented generation?")
            latencies.append(metrics)
        
        # Aggregate latency results
        avg_ttft = np.mean([m["time_to_first_token_ms"] for m in latencies])
        avg_total = np.mean([m["total_latency_ms"] for m in latencies])
        avg_tps = np.mean([m["tokens_per_second"] for m in latencies])
        
        self.results.append(BenchmarkResult(
            timestamp=datetime.now().isoformat(),
            vdb_type=self.config.vdb_type,
            device=self.config.device,
            metric_type="e2e_latency",
            metric_name="time_to_first_token_ms",
            value=avg_ttft
        ))
        
        self.results.append(BenchmarkResult(
            timestamp=datetime.now().isoformat(),
            vdb_type=self.config.vdb_type,
            device=self.config.device,
            metric_type="e2e_latency",
            metric_name="total_latency_ms",
            value=avg_total
        ))
        
        self.results.append(BenchmarkResult(
            timestamp=datetime.now().isoformat(),
            vdb_type=self.config.vdb_type,
            device=self.config.device,
            metric_type="e2e_throughput",
            metric_name="tokens_per_second",
            value=avg_tps
        ))
        
        # 2. Accuracy measurements
        logger.info("Measuring end-to-end accuracy...")
        accuracy_metrics = self.measure_rag_accuracy(test_questions)
        
        for metric_name, value in accuracy_metrics.items():
            self.results.append(BenchmarkResult(
                timestamp=datetime.now().isoformat(),
                vdb_type=self.config.vdb_type,
                device=self.config.device,
                metric_type="e2e_accuracy",
                metric_name=metric_name,
                value=value
            ))
        
        return self.results


class BenchmarkReporter:
    """Generate benchmark reports and visualizations"""
    
    def __init__(self, results: List[BenchmarkResult]):
        self.results = results
        self.df = pd.DataFrame([r.to_dict() for r in results])
    
    def generate_summary_report(self) -> Dict:
        """Generate summary statistics"""
        summary = {}
        
        # Group by metric type
        for metric_type in self.df['metric_type'].unique():
            type_df = self.df[self.df['metric_type'] == metric_type]
            summary[metric_type] = {}
            
            for metric_name in type_df['metric_name'].unique():
                metric_df = type_df[type_df['metric_name'] == metric_name]
                summary[metric_type][metric_name] = {
                    "mean": metric_df['value'].mean(),
                    "std": metric_df['value'].std(),
                    "min": metric_df['value'].min(),
                    "max": metric_df['value'].max()
                }
        
        return summary
    
    def save_results(self, output_dir: str):
        """Save results to files"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save raw results as JSON
        with open(output_path / "raw_results.json", "w") as f:
            json.dump([r.to_dict() for r in self.results], f, indent=2)
        
        # Save as CSV
        self.df.to_csv(output_path / "results.csv", index=False)
        
        # Save summary
        summary = self.generate_summary_report()
        with open(output_path / "summary.json", "w") as f:
            json.dump(summary, f, indent=2)
        
        # Generate comparison report
        self.generate_comparison_report(output_path)
    
    def generate_comparison_report(self, output_path: Path):
        """Generate comparison report between all VDB types"""
        comparison = []
        
        # Get unique VDB types
        vdb_types = self.df['vdb_type'].unique()
        
        # Compare metrics between VDB types
        for metric_type in self.df['metric_type'].unique():
            for metric_name in self.df[self.df['metric_type'] == metric_type]['metric_name'].unique():
                row = {
                    "metric_type": metric_type,
                    "metric_name": metric_name
                }
                
                # Get values for each VDB type
                for vdb_type in vdb_types:
                    vdb_df = self.df[
                        (self.df['vdb_type'] == vdb_type) &
                        (self.df['metric_type'] == metric_type) &
                        (self.df['metric_name'] == metric_name)
                    ]
                    
                    if not vdb_df.empty:
                        row[vdb_type] = vdb_df['value'].mean()
                    else:
                        row[vdb_type] = None
                
                # Calculate improvements if we have cyborgdb and other VDBs
                if 'cyborgdb' in row and row['cyborgdb'] is not None:
                    if 'milvus' in row and row['milvus'] is not None and row['milvus'] != 0:
                        row['cyborgdb_vs_milvus'] = ((row['cyborgdb'] - row['milvus']) / row['milvus']) * 100
                    
                    if 'elasticsearch' in row and row['elasticsearch'] is not None and row['elasticsearch'] != 0:
                        row['cyborgdb_vs_elasticsearch'] = ((row['cyborgdb'] - row['elasticsearch']) / row['elasticsearch']) * 100
                
                comparison.append(row)
        
        comparison_df = pd.DataFrame(comparison)
        comparison_df.to_csv(output_path / "comparison.csv", index=False)
        
        # Generate markdown report
        self.generate_markdown_report(comparison_df, output_path)
    
    def generate_markdown_report(self, comparison_df: pd.DataFrame, output_path: Path):
        """Generate markdown report for blog post"""
        with open(output_path / "benchmark_report.md", "w") as f:
            f.write("# Vector Database Performance Benchmark Results\n\n")
            f.write(f"Generated: {datetime.now().isoformat()}\n\n")
            
            f.write("## Executive Summary\n\n")
            
            # Key metrics - CyborgDB vs Milvus
            if 'cyborgdb' in comparison_df.columns and 'milvus' in comparison_df.columns:
                f.write("### CyborgDB vs Milvus Performance\n\n")
                f.write("| Metric | CyborgDB | Milvus | Improvement |\n")
                f.write("|--------|----------|--------|-------------|\n")
                
                milvus_comparison = comparison_df[comparison_df['milvus'].notna() & comparison_df['cyborgdb'].notna()].head(10)
                for _, row in milvus_comparison.iterrows():
                    if 'cyborgdb_vs_milvus' in row:
                        improvement_str = f"+{row['cyborgdb_vs_milvus']:.1f}%" if row['cyborgdb_vs_milvus'] > 0 else f"{row['cyborgdb_vs_milvus']:.1f}%"
                    else:
                        improvement_str = "N/A"
                    f.write(f"| {row['metric_name']} | {row['cyborgdb']:.2f} | {row['milvus']:.2f} | {improvement_str} |\n")
            
            # Key metrics - CyborgDB vs Elasticsearch
            if 'cyborgdb' in comparison_df.columns and 'elasticsearch' in comparison_df.columns:
                f.write("\n### CyborgDB vs Elasticsearch Performance\n\n")
                f.write("| Metric | CyborgDB | Elasticsearch | Improvement |\n")
                f.write("|--------|----------|---------------|-------------|\n")
                
                es_comparison = comparison_df[comparison_df['elasticsearch'].notna() & comparison_df['cyborgdb'].notna()].head(10)
                for _, row in es_comparison.iterrows():
                    if 'cyborgdb_vs_elasticsearch' in row:
                        improvement_str = f"+{row['cyborgdb_vs_elasticsearch']:.1f}%" if row['cyborgdb_vs_elasticsearch'] > 0 else f"{row['cyborgdb_vs_elasticsearch']:.1f}%"
                    else:
                        improvement_str = "N/A"
                    f.write(f"| {row['metric_name']} | {row['cyborgdb']:.2f} | {row['elasticsearch']:.2f} | {improvement_str} |\n")
            
            f.write("\n## Detailed Results\n\n")
            
            # Group by metric type
            for metric_type in comparison_df['metric_type'].unique():
                f.write(f"### {metric_type.replace('_', ' ').title()}\n\n")
                type_df = comparison_df[comparison_df['metric_type'] == metric_type]
                
                # Determine which columns exist
                vdb_cols = [col for col in ['cyborgdb', 'milvus', 'elasticsearch'] if col in type_df.columns]
                
                if len(vdb_cols) > 0:
                    # Create header
                    header = "| Metric |"
                    separator = "|--------|"
                    for vdb in vdb_cols:
                        header += f" {vdb.title()} |"
                        separator += "----------|"
                    f.write(header + "\n")
                    f.write(separator + "\n")
                    
                    # Write data rows
                    for _, row in type_df.iterrows():
                        row_str = f"| {row['metric_name']} |"
                        for vdb in vdb_cols:
                            if vdb in row and row[vdb] is not None:
                                row_str += f" {row[vdb]:.2f} |"
                            else:
                                row_str += " N/A |"
                        f.write(row_str + "\n")
                    
                    f.write("\n")
            
            # GPU vs CPU comparison
            f.write("## GPU vs CPU Performance (CyborgDB)\n\n")
            
            gpu_df = self.df[self.df['device'] == 'gpu']
            cpu_df = self.df[self.df['device'] == 'cpu']
            
            if not gpu_df.empty and not cpu_df.empty:
                f.write("| Metric | GPU | CPU | GPU Speedup |\n")
                f.write("|--------|-----|-----|-------------|\n")
                
                for metric_name in gpu_df['metric_name'].unique():
                    gpu_val = gpu_df[gpu_df['metric_name'] == metric_name]['value'].mean()
                    cpu_val = cpu_df[cpu_df['metric_name'] == metric_name]['value'].mean()
                    
                    if cpu_val != 0:
                        speedup = gpu_val / cpu_val
                        f.write(f"| {metric_name} | {gpu_val:.2f} | {cpu_val:.2f} | {speedup:.2f}x |\n")


def main():
    """Main benchmark execution"""
    parser = argparse.ArgumentParser(description="CyborgDB Blueprint Performance Benchmark")
    parser.add_argument("--vdb-type", choices=["cyborgdb", "milvus", "elasticsearch", "all"], default="all",
                       help="Vector database to benchmark")
    parser.add_argument("--device", choices=["gpu", "cpu", "both"], default="both",
                       help="Device to use for benchmarking")
    parser.add_argument("--num-documents", type=int, default=10000,
                       help="Number of documents for testing")
    parser.add_argument("--num-queries", type=int, default=100,
                       help="Number of queries for testing")
    parser.add_argument("--output-dir", default="./benchmark_results",
                       help="Directory to save results")
    parser.add_argument("--run-e2e", action="store_true",
                       help="Run end-to-end benchmarks")
    
    args = parser.parse_args()
    
    all_results = []
    
    # Determine configurations to test
    if args.vdb_type == "all":
        vdb_types = ["cyborgdb", "milvus", "elasticsearch"]
    else:
        vdb_types = [args.vdb_type]
    devices = ["gpu", "cpu"] if args.device == "both" else [args.device]
    
    # Run vector DB benchmarks
    for vdb_type in vdb_types:
        for device in devices:
            logger.info(f"Running benchmark for {vdb_type} on {device}")
            
            config = BenchmarkConfig(
                vdb_type=vdb_type,
                device=device,
                num_documents=args.num_documents,
                num_queries=args.num_queries
            )
            
            # Select appropriate benchmark class
            if vdb_type == "cyborgdb":
                benchmark = CyborgDBBenchmark(config)
            elif vdb_type == "milvus":
                benchmark = MilvusBenchmark(config)
            elif vdb_type == "elasticsearch":
                benchmark = ElasticsearchBenchmark(config)
            else:
                raise ValueError(f"Unknown VDB type: {vdb_type}")
            
            try:
                results = benchmark.run_benchmark_suite()
                all_results.extend(results)
            except Exception as e:
                logger.error(f"Failed to run benchmark for {vdb_type} on {device}: {e}")
    
    # Run end-to-end benchmarks if requested
    if args.run_e2e:
        for vdb_type in vdb_types:
            config = BenchmarkConfig(
                vdb_type=vdb_type,
                device="gpu"  # E2E typically uses GPU
            )
            
            e2e_benchmark = EndToEndBenchmark(config)
            results = e2e_benchmark.run_benchmark()
            all_results.extend(results)
    
    # Generate reports
    reporter = BenchmarkReporter(all_results)
    reporter.save_results(args.output_dir)
    
    logger.info(f"Benchmark complete! Results saved to {args.output_dir}")
    
    # Print summary
    summary = reporter.generate_summary_report()
    print("\n=== Benchmark Summary ===")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()