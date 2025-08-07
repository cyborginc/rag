# cyborg.py
import logging
import json
import numpy as np
from typing import Dict, List, Optional, Union, Any
from pathlib import Path
import binascii
from nv_ingest_client.util.vdb.adt_vdb import VDB
from cyborgdb import (
    Client,
    EncryptedIndex,
    IndexConfig,
    IndexIVF,
    IndexIVFPQ,
    IndexIVFFlat,
    generate_key
)

logger = logging.getLogger(__name__)


class Cyborg(VDB):
    """
    CyborgDB implementation of the VDB interface.
    
    This class provides a unified interface for interacting with CyborgDB,
    similar to the Milvus implementation.
    """
    
    def __init__(
        self,
        index_name: str = "cyborg_index",
        api_url: str = "http://localhost:8000",
        api_key: Optional[str] = None,
        index_key: Optional[bytes] = None,
        verify_ssl: Optional[bool] = None,
        index_type: str = "IVFFlat",
        dimension: int = 1536,
        n_lists: int = 128,
        metric:str = "euclidean",
        pq_bits: Optional[int] = 8,
        pq_dim: Optional[int] = 8,
        embedding_model: Optional[str] = None,
        recreate: bool = True,
        max_cache_size: int = 0,
        **kwargs
    ):
        """
        Initialize the CyborgDB operator.
        
        Args:
            index_name: Name of the index
            api_url: URL of the CyborgDB server
            api_key: API key for authentication
            index_key: 32-byte encryption key (generated if not provided)
            verify_ssl: SSL verification setting
            index_type: Type of index ("IVFFlat", "IVF", "IVFPQ")
            dimension: Dimension of vectors
            n_lists: Number of inverted lists
            metric: type of metric for index
            pq_bits: Number of bits per subquantizer (for IVFPQ)
            pq_dim: number of bits per PQ code (for IVFPQ)
            embedding_model: Optional embedding model name
            recreate: Whether to recreate index if it exists
            max_cache_size: Maximum cache size
            **kwargs: Additional parameters
        """
        # Store all parameters
        kwargs.update(locals())
        kwargs.pop("self", None)
        super().__init__(**kwargs)
        
        # Generate index key if not provided
        if self.index_key is None:
            self.index_key = generate_key()
            logger.info(f"Generated new index key for {index_name}")
        
        # Initialize client
        self._client = None
        self._index = None
        
    @property
    def client(self) -> Client:
        """Lazy initialization of client."""
        if self._client is None:
            self._client = Client(
                api_url=self.api_url,
                api_key=self.api_key,
                verify_ssl=self.verify_ssl
            )
        return self._client
    
    def _create_index_config(self) -> Union[IndexIVF, IndexIVFPQ, IndexIVFFlat]:
        """Create appropriate index configuration based on index_type."""
        if self.index_type == "IVFFlat":
            return IndexIVFFlat(
                dimension=self.dimension,
                n_lists=self.n_lists,
                metric=self.metric
            )
        elif self.index_type == "IVF":
            return IndexIVF(
                dimension=self.dimension,
                n_lists=self.n_lists,
                metric=self.metric
            )
        elif self.index_type == "IVFPQ":
            return IndexIVFPQ(
                dimension=self.dimension,
                n_lists=self.n_lists,
                pq_dim=self.pq_dim,
                pq_bits=self.pq_bits,
                metric=self.metric
            )
        else:
            raise ValueError(f"Unsupported index type: {self.index_type}")
    
    def create_index(self, **kwargs) -> EncryptedIndex:
        """
        Create a new CyborgDB index.
        
        Returns:
            EncryptedIndex: The created index
        """
        # Override with any provided kwargs
        params = self.__dict__.copy()
        params.update(kwargs)
        
        index_name = params.get("index_name", self.index_name)
        
        # Check if index exists and handle recreation
        try:
            existing_indexes = self.client.list_indexes()
            if index_name in existing_indexes:
                if params.get("recreate", self.recreate):
                    # Load existing index to delete it
                    existing_index = EncryptedIndex(
                        index_name=index_name,
                        index_key=self.index_key,
                        api=self.client.api,
                        api_client=self.client.api_client,
                        max_cache_size=self.max_cache_size
                    )
                    existing_index.delete_index()
                    logger.info(f"Deleted existing index: {index_name}")
                else:
                    # Return existing index
                    self._index = EncryptedIndex(
                        index_name=index_name,
                        index_key=self.index_key,
                        api=self.client.api,
                        api_client=self.client.api_client,
                        max_cache_size=self.max_cache_size
                    )
                    return self._index
        except Exception as e:
            logger.debug(f"Error checking existing indexes: {e}")
        
        # Create index configuration
        index_config = self._create_index_config()
        
        # Create new index
        self._index = self.client.create_index(
            index_name=index_name,
            index_key=self.index_key,
            index_config=index_config,
            embedding_model=params.get("embedding_model", self.embedding_model),
            max_cache_size=params.get("max_cache_size", self.max_cache_size)
        )
        
        logger.info(f"Created new index: {index_name}")
        return self._index
    
    def write_to_index(self, records: List[Dict[str, Any]], **kwargs):
        """
        Write records to the CyborgDB index.
        
        Args:
            records: List of records to insert
            **kwargs: Additional parameters
        """
        if self._index is None:
            raise ValueError("Index not created. Call create_index() first.")
        
        # Convert records to CyborgDB format
        items = []
        for record in records:
            item = {
                "id": str(record.get("id", record.get("pk", record.get("_id")))),
            }
            
            # Handle vector field
            if "vector" in record:
                item["vector"] = record["vector"]
            elif "embedding" in record:
                item["vector"] = record["embedding"]
            elif "metadata" in record and "embedding" in record["metadata"]:
                item["vector"] = record["metadata"]["embedding"]
            
            # Handle content/text field
            if "text" in record:
                item["contents"] = record["text"]
            elif "content" in record:
                item["contents"] = record["content"]
            elif "metadata" in record and "content" in record["metadata"]:
                item["contents"] = record["metadata"]["content"]
            
            # Handle metadata
            metadata = {}
            if "metadata" in record:
                metadata.update(record["metadata"])
            if "source" in record:
                metadata["source"] = record["source"]
            if "content_metadata" in record:
                metadata["content_metadata"] = record["content_metadata"]
            
            if metadata:
                item["metadata"] = metadata
            
            items.append(item)
        
        # Upsert items
        self._index.upsert(items)
        logger.info(f"Inserted {len(items)} records into index")
    
    def retrieval(
        self,
        queries: List[Union[str, List[float], np.ndarray]],
        top_k: int = 10,
        n_probes: int = 1,
        filters: Optional[Dict[str, Any]] = None,
        include: List[str] = ["distance", "metadata"],
        greedy: bool = False,
        **kwargs
    ) -> List[List[Dict[str, Any]]]:
        """
        Retrieve nearest neighbors for queries.
        
        Args:
            queries: List of query vectors or texts
            top_k: Number of results per query
            n_probes: Number of lists to probe
            filters: Optional metadata filters
            include: Fields to include in results
            greedy: Whether to use greedy search
            **kwargs: Additional parameters
            
        Returns:
            List of results for each query
        """
        if self._index is None:
            raise ValueError("Index not created. Call create_index() first.")
        
        all_results = []
        
        for query in queries:
            if isinstance(query, str):
                # Text query - use query_contents
                results = self._index.query(
                    query_contents=query,
                    top_k=top_k,
                    n_probes=n_probes,
                    filters=filters,
                    include=include,
                    greedy=greedy
                )
            else:
                # Vector query
                if isinstance(query, np.ndarray):
                    query = query.tolist()
                results = self._index.query(
                    query_vector=query,
                    top_k=top_k,
                    n_probes=n_probes,
                    filters=filters,
                    include=include,
                    greedy=greedy
                )
            
            # Ensure results is always a list of lists
            if results and not isinstance(results[0], list):
                results = [results]
            
            all_results.extend(results)
        
        return all_results
    
    def reindex(self, **kwargs):
        """
        Reindex data with new configuration.
        
        This involves:
        1. Retrieving all data from current index
        2. Creating a new index with updated configuration
        3. Re-inserting all data
        """
        if self._index is None:
            raise ValueError("Index not created. Call create_index() first.")
        
        current_index_name = kwargs.pop("current_index_name", self.index_name)
        new_index_name = kwargs.pop("new_index_name", f"{current_index_name}_reindexed")
        batch_size = kwargs.pop("batch_size", 1000)
        
        logger.info(f"Starting reindex from {current_index_name} to {new_index_name}")
        
        # Get all IDs (this is a limitation - CyborgDB doesn't have a direct "get all" method)
        # In practice, you'd need to track IDs separately or implement a scan method
        raise NotImplementedError(
            "Reindexing requires tracking document IDs separately. "
            "CyborgDB doesn't provide a method to retrieve all document IDs."
        )
    
    def train(self, **kwargs):
        """
        Train the index for better search performance.
        
        Args:
            **kwargs: Training parameters (batch_size, max_iters, tolerance, max_memory)
        """
        if self._index is None:
            raise ValueError("Index not created. Call create_index() first.")
        
        # Use provided kwargs or defaults
        batch_size = kwargs.get("batch_size", 2048)
        max_iters = kwargs.get("max_iters", 100)
        tolerance = kwargs.get("tolerance", 1e-6)
        max_memory = kwargs.get("max_memory", 0)
        
        self._index.train(
            batch_size=batch_size,
            max_iters=max_iters,
            tolerance=tolerance,
            max_memory=max_memory
        )
        logger.info("Index training completed")
    
    def delete(self, ids: List[str]):
        """
        Delete records by IDs.
        
        Args:
            ids: List of IDs to delete
        """
        if self._index is None:
            raise ValueError("Index not created. Call create_index() first.")
        
        self._index.delete(ids)
        logger.info(f"Deleted {len(ids)} records from index")
    
    def run(self, records: List[Dict[str, Any]]):
        """
        Main entry point for processing records.
        
        This method orchestrates the creation of index and insertion of records.
        
        Args:
            records: List of records to process
        """
        # Create or get index
        self.create_index()
        
        # Insert records
        if records:
            self.write_to_index(records)
            
            # Train index if we have enough data
            if len(records) >= 2 * self.n_lists:
                try:
                    self.train()
                except Exception as e:
                    logger.warning(f"Failed to train index: {e}")
        
        # Update recreate flag for subsequent runs
        self.recreate = False
    
    def get_connection_params(self) -> tuple:
        """Get connection parameters for the index."""
        conn_dict = {
            "api_url": self.api_url,
            "api_key": self.api_key,
            "verify_ssl": self.verify_ssl,
            "index_type": self.index_type,
            "dimension": self.dimension,
            "n_lists": self.n_lists,
            "recreate": self.recreate,
        }
        if self.index_type == "IVFPQ":
            conn_dict["m"] = self.m
            conn_dict["n_bits"] = self.n_bits
        
        return (self.index_name, conn_dict)
    
    def get_write_params(self) -> tuple:
        """Get parameters for writing to the index."""
        write_params = {
            "embedding_model": self.embedding_model,
            "max_cache_size": self.max_cache_size,
        }
        return (self.index_name, write_params)


# Additional helper functions
def cleanup_records(
    records: List[Dict[str, Any]],
    enable_text: bool = True,
    enable_embeddings: bool = True,
    **kwargs
) -> List[Dict[str, Any]]:
    """
    Clean and validate records before insertion.
    
    Args:
        records: Raw records
        enable_text: Whether to include text content
        enable_embeddings: Whether to include embeddings
        **kwargs: Additional parameters
        
    Returns:
        Cleaned records ready for insertion
    """
    cleaned = []
    
    for record in records:
        # Skip records without required fields
        if not enable_embeddings and "embedding" not in record.get("metadata", {}):
            continue
        if not enable_text and "content" not in record.get("metadata", {}):
            continue
        
        cleaned.append(record)
    
    return cleaned