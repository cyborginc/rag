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
import random
import uuid
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
        verify_ssl: Optional[bool] = False,
        index_type: str = "IVFFlat",
        dimension: int = 1536,
        n_lists: int = 128,
        metric:str = "euclidean",
        pq_bits: Optional[int] = 8,
        pq_dim: Optional[int] = 8,
        embedding_model: Optional[str] = None,
        recreate: bool = True,
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
            **kwargs: Additional parameters
        """
        logger.info("=== Initializing CyborgDB ===")
        logger.debug(f"Parameters: index_name={index_name}, api_url={api_url}, "
                    f"index_type={index_type}, dimension={dimension}, n_lists={n_lists}, "
                    f"metric={metric}, recreate={recreate}")
        logger.debug(f"Optional params: api_key={'***' if api_key else 'None'}, "
                    f"index_key={'provided' if index_key else 'will generate'}, "
                    f"verify_ssl={verify_ssl}, embedding_model={embedding_model}")
        
        if index_type == "IVFPQ":
            logger.debug(f"IVFPQ specific params: pq_bits={pq_bits}, pq_dim={pq_dim}")
        
        logger.debug(f"Additional kwargs: {kwargs}")
        
        # Store all parameters
        kwargs.update(locals())
        kwargs.pop("self", None)
        super().__init__(**kwargs)
        
        # Generate index key if not provided
        if self.index_key is None:

            logger.info(f"index key not provided for {index_name}")
            raise ValueError("index_key must be provided for CyborgDB initialization")
        else:
            logger.info(f"Using provided index key for {index_name}")
            logger.debug(f"Provided key length: {len(self.index_key)} bytes")
        
        # Initialize client
        self._client = None
        self._index = None
        self.total_record = 0
        logger.debug("Client and index initialized to None (lazy loading)")
        
    @property
    def client(self) -> Client:
        """Lazy initialization of client."""
        if self._client is None:
            logger.info("Initializing CyborgDB client (lazy load)")
            logger.debug(f"Client params: api_url={self.api_url}, "
                        f"api_key={'***' if self.api_key else 'None'}, "
                        f"verify_ssl={self.verify_ssl}")
            try:
                self._client = Client(
                    api_url=self.api_url,
                    api_key=self.api_key,
                    verify_ssl=self.verify_ssl
                )
                logger.info("CyborgDB client initialized successfully")
            except Exception as e:
                logger.error(f"Failed to initialize CyborgDB client: {e}", exc_info=True)
                raise
        else:
            logger.debug("Returning existing CyborgDB client")
        return self._client
    
    def _create_index_config(self) -> Union[IndexIVF, IndexIVFPQ, IndexIVFFlat]:
        """Create appropriate index configuration based on index_type."""
        logger.info(f"Creating index configuration for type: {self.index_type}")
        
        try:
            if self.index_type == "IVFFlat":
                logger.debug(f"Creating IndexIVFFlat config: dimension={self.dimension}, "
                           f"n_lists={self.n_lists}, metric={self.metric}")
                config = IndexIVFFlat(
                    dimension=self.dimension,
                    n_lists=self.n_lists,
                    metric=self.metric
                )
                logger.info("IndexIVFFlat configuration created successfully")
                return config
                
            elif self.index_type == "IVF":
                logger.debug(f"Creating IndexIVF config: dimension={self.dimension}, "
                           f"n_lists={self.n_lists}, metric={self.metric}")
                config = IndexIVF(
                    dimension=self.dimension,
                    n_lists=self.n_lists,
                    metric=self.metric
                )
                logger.info("IndexIVF configuration created successfully")
                return config
                
            elif self.index_type == "IVFPQ":
                logger.debug(f"Creating IndexIVFPQ config: dimension={self.dimension}, "
                           f"n_lists={self.n_lists}, pq_dim={self.pq_dim}, "
                           f"pq_bits={self.pq_bits}, metric={self.metric}")
                config = IndexIVFPQ(
                    dimension=self.dimension,
                    n_lists=self.n_lists,
                    pq_dim=self.pq_dim,
                    pq_bits=self.pq_bits,
                    metric=self.metric
                )
                logger.info("IndexIVFPQ configuration created successfully")
                return config
                
            else:
                logger.error(f"Unsupported index type: {self.index_type}")
                raise ValueError(f"Unsupported index type: {self.index_type}")
                
        except Exception as e:
            logger.error(f"Failed to create index configuration: {e}", exc_info=True)
            raise
    
    def create_index(self, **kwargs) -> EncryptedIndex:
        """
        Create a new CyborgDB index.
        
        Returns:
            EncryptedIndex: The created index
        """
        logger.info("=== Creating CyborgDB Index ===")
        logger.debug(f"create_index called with kwargs: {kwargs}")
        
        # Override with any provided kwargs
        params = self.__dict__.copy()
        params.update(kwargs)
        logger.debug(f"Calling get() on params of type {type(params)}")
        logger.debug(f"Merged parameters: recreate={params.get('recreate')}, "
                    f"index_name={params.get('index_name')}")
        
        index_name = params.get("index_name", self.index_name)
        logger.info(f"Target index name: {index_name}")
        
        # Check if index exists and handle recreation
        try:
            logger.info("Checking for existing indexes...")
            existing_indexes = self.client.list_indexes()
            logger.debug(f"Found {len(existing_indexes)} existing indexes: {existing_indexes}")
            
            if index_name in existing_indexes:
                logger.warning(f"Index '{index_name}' already exists")
                
                if params.get("recreate", self.recreate):
                    logger.info(f"Recreate flag is True, deleting existing index: {index_name}")
                    try:
                        # Load existing index to delete it
                        existing_index = EncryptedIndex(
                            index_name=index_name,
                            index_key=self.index_key,
                            api=self.client.api,
                            api_client=self.client.api_client
                        )
                        logger.debug("Existing index loaded for deletion")
                        
                        existing_index.delete_index()
                        logger.info(f"Successfully deleted existing index: {index_name}")
                    except Exception as e:
                        logger.error(f"Failed to delete existing index: {e}", exc_info=True)
                        raise
                else:
                    logger.info(f"Recreate flag is False, returning existing index: {index_name}")
                    # Return existing index
                    self._index = EncryptedIndex(
                        index_name=index_name,
                        index_key=self.index_key,
                        api=self.client.api,
                        api_client=self.client.api_client
                    )
                    logger.debug("Loaded existing index successfully")
                    return self._index
            else:
                logger.info(f"Index '{index_name}' does not exist, will create new")
                
        except Exception as e:
            logger.warning(f"Error checking existing indexes: {e}")
            logger.debug("Proceeding with index creation despite error", exc_info=True)
        
        # Create index configuration
        logger.info("Creating index configuration...")
        index_config = self._create_index_config()
        logger.debug(f"Index config created: {type(index_config).__name__}")
        
        # Create new index
        try:
            logger.info(f"Creating new index '{index_name}' via client...")
            logger.debug(f"Create params: embedding_model={params.get('embedding_model', self.embedding_model)}")
            
            self._index = self.client.create_index(
                index_name=index_name,
                index_key=self.index_key,
                index_config=index_config,
                embedding_model=params.get("embedding_model", self.embedding_model)
            )
            
            logger.info(f"Successfully created new index: {index_name}")
            logger.debug(f"Index object type: {type(self._index).__name__}")
            
        except Exception as e:
            logger.error(f"Failed to create index: {e}", exc_info=True)
            raise
            
        return self._index
    
    def write_to_index(self, records: List[Dict[str, Any]], **kwargs):
        """
        Write records to the CyborgDB index.
        
        Args:
            records: List of records to insert
            **kwargs: Additional parameters
        """
        logger.info("=== Writing Records to Index ===")
        logger.debug(f"write_to_index called with {len(records)} records")
        logger.debug(f"Additional kwargs: {kwargs}")
        
        if self._index is None:
            logger.error("Index not created. Call create_index() first.")
            raise ValueError("Index not created. Call create_index() first.")
        
        logger.info(f"Processing {len(records)} records for insertion")
        
        # Convert records to CyborgDB format
        items = []
        
        for idx, record in enumerate(records):
            logger.debug(f"Processing record {idx + 1}/{len(records)}")
            logger.debug(f"Type of record: {type(record).__name__}")

            if not isinstance(record, dict):
                logger.error(f"Record {idx} is not a dict.")
                raise TypeError(f"Expected dict, got {type(record).__name__}")
            logger.debug(f"Record keys: {list(record.keys())}")

            logger.debug(f"Calling get() on record of type {type(record)}")

            for k, v in record.items():
                logger.debug(f"  attr='{k}' type={type(v).__name__} value={repr(v)[:20]}")

            # Extract ID
            if "id" in record and record["id"] is not None:
                id_value = str(record["id"])
                logger.debug(f"Using provided ID: {id_value}")
            elif "_id" in record and record["_id"] is not None:
                id_value = str(record["_id"])
                logger.debug(f"Using provided _id: {id_value}")
            else:
                id_value = str(uuid.uuid4())
                logger.debug(f"Generated new UUID for record ID: {id_value}")

            item = {"id": id_value}
            
            # Handle vector field
            vector_found = False
            if "vector" in record:
                item["vector"] = record["vector"]
                vector_found = True
                logger.debug(f"Found vector field, length: {len(record['vector'])}")
            elif "embedding" in record:
                item["vector"] = record["embedding"]
                vector_found = True
                logger.debug(f"Found embedding field, length: {len(record['embedding'])}")
            elif "metadata" in record and "embedding" in record["metadata"]:
                item["vector"] = record["metadata"]["embedding"]
                vector_found = True
                if item["vector"]:
                    logger.debug(f"Found embedding in metadata, length: {len(item["vector"])}")

            
            if not vector_found:
                logger.warning(f"No vector/embedding found for record {id_value}")
            
            # Handle metadata
            metadata = {}
            if "metadata" in record:
                logger.debug(f"Before metadata fields change: {list(record['metadata'].keys())}")
                metadata.update(record["metadata"])
                # Rename 'content' -> '_content' if it exists and truncate if needed
                if "content" in metadata:
                    content = metadata.pop("content")
                    # Truncate content to match Milvus's 65535 char limit
                    if content and len(content) > 65535:
                        logger.warning(f"Truncating content from {len(content)} to 65535 chars for record {id_value}")
                        content = content[:65535]
                    metadata["_content"] = content
                if "source_metadata" in metadata:
                    metadata["source"] = metadata.pop("source_metadata")    
                logger.debug(f"Added metadata fields: {list(metadata.keys())}")
            if "source" in record:
                metadata["source"] = record["source"]
                logger.debug(f"Added source: {record['source']}")
            if "content_metadata" in record:
                metadata["content_metadata"] = record["content_metadata"]
                logger.debug(f"Added content_metadata")
            
            if metadata:
                item["metadata"] = metadata
                logger.debug(f"Total metadata fields: {len(metadata)}")
            else:
                logger.debug("No metadata for this record")
            
            items.append(item)
        
        logger.info(f"Prepared {len(items)} items for upsertion")
        
        # Upsert items
        try:
            logger.info("Starting upsert operation...")
            self._index.upsert(items)
            logger.info(f"Successfully inserted {len(items)} records into index")
        except Exception as e:
            logger.error(f"Failed to upsert records: {e}", exc_info=True)
            # logger.debug(f"Failed items sample (first item): {items[0] if items else 'No items'}")
            raise
    
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
        logger.info("=== Performing Retrieval ===")
        logger.debug(f"Retrieval params: num_queries={len(queries)}, top_k={top_k}, "
                    f"n_probes={n_probes}, greedy={greedy}")
        logger.debug(f"Filters: {filters}")
        logger.debug(f"Include fields: {include}")
        logger.debug(f"Additional kwargs: {kwargs}")
        
        if self._index is None:
            logger.error("Index not created. Call create_index() first.")
            raise ValueError("Index not created. Call create_index() first.")
        
        all_results = []
        
        for query_idx, query in enumerate(queries):
            logger.debug(f"Processing query {query_idx + 1}/{len(queries)}")
            
            try:
                if isinstance(query, str):
                    logger.info(f"Text query detected: '{query[:50]}...'")
                    # Text query - use query_contents
                    results = self._index.query(
                        query_contents=query,
                        top_k=top_k,
                        n_probes=n_probes,
                        filters=filters,
                        include=include,
                        greedy=greedy
                    )
                    logger.debug(f"Text query completed, got {len(results) if results else 0} results")
                else:
                    # Vector query
                    if isinstance(query, np.ndarray):
                        logger.debug(f"Converting numpy array to list, shape: {query.shape}")
                        query = query.tolist()
                    
                    logger.info(f"Vector query detected, dimension: {len(query)}")
                    results = self._index.query(
                        query_vector=query,
                        top_k=top_k,
                        n_probes=n_probes,
                        filters=filters,
                        include=include,
                        greedy=greedy
                    )
                    logger.debug(f"Vector query completed, got {len(results) if results else 0} results")
                
                # Ensure results is always a list of lists
                if results and not isinstance(results[0], list):
                    logger.debug("Converting single result list to list of lists")
                    results = [results]
                
                logger.debug(f"Query {query_idx + 1} returned {len(results)} result sets")
                if results and results[0]:
                    logger.debug(f"Calling get on results[0][0] of type {type(results[0][0])}")
                    logger.debug(f"First result sample: id={results[0][0].get('id')}, "
                               f"distance={results[0][0].get('distance')}")
                
                all_results.extend(results)
                
            except Exception as e:
                logger.error(f"Failed to process query {query_idx + 1}: {e}", exc_info=True)
                raise
        
        logger.info(f"Retrieval completed. Total result sets: {len(all_results)}")
        return all_results
    
    def reindex(self, **kwargs):
        """
        Reindex data with new configuration.
        
        This involves:
        1. Retrieving all data from current index
        2. Creating a new index with updated configuration
        3. Re-inserting all data
        """
        logger.info("=== Starting Reindex Operation ===")
        logger.debug(f"Reindex kwargs: {kwargs}")
        
        if self._index is None:
            logger.error("Index not created. Call create_index() first.")
            raise ValueError("Index not created. Call create_index() first.")
        
        current_index_name = kwargs.pop("current_index_name", self.index_name)
        new_index_name = kwargs.pop("new_index_name", f"{current_index_name}_reindexed")
        batch_size = kwargs.pop("batch_size", 1000)
        
        logger.info(f"Reindex plan: {current_index_name} -> {new_index_name}")
        logger.debug(f"Batch size: {batch_size}")
        
        # Get all IDs (this is a limitation - CyborgDB doesn't have a direct "get all" method)
        # CyborgDB will expose this method in an upcoming release
        logger.error("Reindexing not implemented - CyborgDB lacks document scan capability")
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
        logger.info("=== Training Index ===")
        logger.debug(f"Training kwargs: {kwargs}")
        
        if self._index is None:
            logger.error("Index not created. Call create_index() first.")
            raise ValueError("Index not created. Call create_index() first.")

        logger.debug(f"Calling get() on kwargs of type {type(kwargs)}")
        # Use provided kwargs or defaults
        batch_size = kwargs.get("batch_size", 2048)
        max_iters = kwargs.get("max_iters", 100)
        tolerance = kwargs.get("tolerance", 1e-6)
        max_memory = kwargs.get("max_memory", 0)
        
        logger.info(f"Training parameters: batch_size={batch_size}, max_iters={max_iters}, "
                   f"tolerance={tolerance}, max_memory={max_memory}")
        
        try:
            logger.info("Starting index training...")
            self._index.train(
                batch_size=batch_size,
                max_iters=max_iters,
                tolerance=tolerance,
            )
            logger.info("Index training completed successfully")
        except Exception as e:
            logger.error(f"Index training failed: {e}", exc_info=True)
            raise
    
    def delete(self, ids: List[str]):
        """
        Delete records by IDs.
        
        Args:
            ids: List of IDs to delete
        """
        logger.info("=== Deleting Records ===")
        logger.debug(f"Deleting {len(ids)} records")
        logger.debug(f"First few IDs: {ids[:5] if len(ids) > 5 else ids}")
        
        if self._index is None:
            logger.error("Index not created. Call create_index() first.")
            raise ValueError("Index not created. Call create_index() first.")
        
        try:
            logger.info(f"Deleting {len(ids)} records from index...")
            self._index.delete(ids)
            logger.info(f"Successfully deleted {len(ids)} records from index")
        except Exception as e:
            logger.error(f"Failed to delete records: {e}", exc_info=True)
            raise
    
    def run(self, records: List[Dict[str, Any]]):
        """
        Main entry point for processing records.
        
        This method orchestrates the creation of index and insertion of records.
        
        Args:
            records: List of records to process
        """
        logger.info("=== Running CyborgDB Pipeline ===")
        logger.debug(f"Input type: {type(records)}")
        
        # Create or get index
        logger.info("Step 1: Creating/getting index")
        self.create_index()
        
        # Insert records
        if records:
            logger.debug(f"Input records type: {infer_type_structure(records)}")

            flat_records = []
            try:
                flat_records = normalize_records(records)
            except Exception as e:
                logger.error(f"Failed to normalize records: {e}", exc_info=True)
                raise

            logger.debug(f"Normalized to List[Dict]; count={len(flat_records)}")
            if not flat_records:
                logger.warning("No records to write after normalization")
                return

            logger.info(f"Step 2: Writing {len(flat_records)} records to index")
            self.write_to_index(flat_records)
            self.total_record += len(flat_records)
            logger.info(f"Step 2.1: total record {self.total_record}")
            # Train index if we have enough data
            min_records_for_training = self.n_lists
            logger.debug(f"Flatten Records: {len(flat_records)}, Min for training: {min_records_for_training}")

            
            if self.total_record >= min_records_for_training:
                logger.info(f"Step 3: Training index (have {self.total_record} records, "
                           f"need {min_records_for_training})")
                try:
                    self.train()
                except Exception as e:
                    logger.warning(f"Failed to train index: {e}")
                    logger.debug("Training failure details:", exc_info=True)
            else:
                logger.info(f"Skipping training - insufficient records "
                           f"({self.total_record} < {min_records_for_training})")
        else:
            logger.warning("No records provided to process")
        
        # Update recreate flag for subsequent runs
        logger.debug(f"Setting recreate flag from {self.recreate} to False")
        self.recreate = False
        logger.info("Pipeline completed successfully")
    
    def get_connection_params(self) -> tuple:
        """Get connection parameters for the index."""
        logger.debug("Getting connection parameters")
        
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
            conn_dict["n_bits"] = self.n_bits
            conn_dict["pq_dim"] = self.pq_dim
            logger.debug("Added IVFPQ parameters to connection dict")
        
        logger.debug(f"Connection params: index_name={self.index_name}, "
                    f"params_keys={list(conn_dict.keys())}")
        
        return (self.index_name, conn_dict)
    
    def get_write_params(self) -> tuple:
        """Get parameters for writing to the index."""
        logger.debug("Getting write parameters")
        
        write_params = {
            "embedding_model": self.embedding_model
        }
        
        logger.debug(f"Write params: index_name={self.index_name}, "
                    f"embedding_model={self.embedding_model}")
        
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
    logger.info("=== Cleaning Records ===")
    logger.debug(f"Cleanup params: enable_text={enable_text}, "
                f"enable_embeddings={enable_embeddings}")
    logger.debug(f"Input records: {len(records)}")
    logger.debug(f"Additional kwargs: {kwargs}")
    
    cleaned = []
    skipped_no_embedding = 0
    skipped_no_content = 0
    skipped_too_long = 0
    
    for idx, record in enumerate(records):
        logger.debug(f"Checking record {idx + 1}/{len(records)}")

        logger.debug(f"Calling get() on record in cleanup_records of type {type(record)}")

        # Skip records without required fields
        if not enable_embeddings and "embedding" not in record.get("metadata", {}):
            logger.debug(f"Skipping record {idx + 1} - no embedding")
            skipped_no_embedding += 1
            continue
            
        if not enable_text and "content" not in record.get("metadata", {}):
            logger.debug(f"Skipping record {idx + 1} - no content")
            skipped_no_content += 1
            continue
        
        # Check and truncate content length to match Milvus limit
        if "metadata" in record and "content" in record["metadata"]:
            content = record["metadata"]["content"]
            if content and len(content) > 65535:
                logger.warning(f"Content too long ({len(content)} chars) in record {idx + 1}, truncating to 65535")
                record["metadata"]["content"] = content[:65535]
                skipped_too_long += 1
        
        cleaned.append(record)
    
    logger.info(f"Record cleaning complete: {len(cleaned)} cleaned, "
               f"{skipped_no_embedding} skipped (no embedding), "
               f"{skipped_no_content} skipped (no content), "
               f"{skipped_too_long} truncated (too long)")
    
    return cleaned

def infer_type_structure(obj):
    """Simple type structure inference"""
    if isinstance(obj, list):
        if obj:
            # Check first element
            inner = infer_type_structure(obj[0])
            return f"List[{inner}]"
        return "List[Unknown]"
    elif isinstance(obj, dict):
        if obj:
            # Sample first key-value pair
            key = next(iter(obj))
            val = obj[key]
            return f"Dict[{type(key).__name__}, {infer_type_structure(val)}]"
        return "Dict[Unknown, Unknown]"
    else:
        return type(obj).__name__
    
def get_records_dict(obj):
    if isinstance(obj, dict):
        return obj
    elif isinstance(obj, list) and obj:
        return get_records_dict(obj[0])
    elif isinstance(obj, tuple) and len(obj) > 0:
        return get_records_dict(obj[0])
    return {}

def normalize_records(records) -> List[Dict[str, Any]]:
    """
    Flattens arbitrarily nested lists/tuples of dicts into List[Dict].
    Raises if it encounters a non-dict leaf.
    """
    flat: List[Dict[str, Any]] = []
    stack = [records]
    max_items_warned = False

    while stack:
        cur = stack.pop()
        if isinstance(cur, dict):
            flat.append(cur)
        elif isinstance(cur, (list, tuple)):
            # push children; keep order by reversing
            stack.extend(reversed(cur))
        elif cur is None:
            # ignore stray None
            continue
        else:
            raise TypeError(
                f"normalize_records expected dict/list/tuple leaves, got {type(cur).__name__}"
            )

        # (optional) safeguard on giant nesting explosions
        if not max_items_warned and len(flat) > 1_000_000:
            max_items_warned = True
            logger.warning("normalize_records: more than 1,000,000 dicts collected")

    return flat