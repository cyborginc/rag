# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
This module contains the implementation of the CyborgDBVDB class,
which provides CyborgDB vector database operations for RAG applications.
Extends VDBRag for RAG-specific functionality.

Collection Management:
1. create_collection: Create a new collection (index) with specified dimensions and type
2. check_collection_exists: Check if the specified collection (index) exists
3. get_collection: Retrieve all collections with their metadata schemas
4. delete_collections: Delete multiple collections and their associated metadata

Document Management:
5. get_documents: Retrieve all unique documents from the specified collection
6. delete_documents: Remove documents matching the specified source values

Metadata Schema Management:
7. create_metadata_schema_collection: Initialize the metadata schema storage (not applicable for CyborgDB)
8. add_metadata_schema: Store metadata schema configuration (not applicable for CyborgDB)
9. get_metadata_schema: Retrieve the metadata schema (not applicable for CyborgDB)

Retrieval Operations:
10. retrieval_langchain: Perform semantic search and return top-k relevant documents
11. get_langchain_vectorstore: Get the vectorstore for a collection
"""

import json
import logging
import os
import time
from typing import Any, Dict, List, Optional, Union
from uuid import uuid4

from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStore
from langchain_core.runnables import RunnableAssign, RunnableLambda
from langchain_core.embeddings import DeterministicFakeEmbedding
from opentelemetry import context as otel_context

from nvidia_rag.utils.common import get_config
from nvidia_rag.utils.vdb.vdb_base import VDBRag

logger = logging.getLogger(__name__)

try:
    from cyborgdb import Client, EncryptedIndex, IndexIVFFlat
    from cyborgdb.integrations.langchain import CyborgVectorStore
    CYBORGDB_AVAILABLE = True
except ImportError:
    logger.error("CyborgDB not installed. Please install cyborgdb-py[langchain]")
    CYBORGDB_AVAILABLE = False
    raise ImportError("CyborgDB not installed. Please install cyborgdb-py[langchain]")

try:
    from nv_ingest_client.util.vdb.adt_vdb import VDB
except ImportError:
    logger.warning("Optional nv_ingest_client module not installed.")
    VDB = object  # Fallback to object if not available

CONFIG = get_config()


class CyborgDBVDB(VDBRag):
    def __init__(
        self,
        collection_name: str,
        cyborgdb_uri: str,
        api_key: str,
        index_key: bytes,
        meta_source_field: str = None,
        meta_fields: list[str] = None,
        embedding_model: str = None,
        csv_file_path: str = None,
    ):
        self.embedding_model = embedding_model
        if self.embedding_model is None:
            self.embedding_model = DeterministicFakeEmbedding(size=1536)
        
        # CyborgDB specific parameters
        self.vdb_endpoint = cyborgdb_uri
        self._collection_name = collection_name
        self.api_key = api_key
        self.index_key = index_key

        # Validate required parameters
        if not self.api_key:
            raise ValueError("CyborgDB API key is required. Set it in config or CYBORGDB_API_KEY env var")
        
        if not self.index_key:
            raise ValueError("CyborgDB index key is required. Set it in config or provide a valid index key")
        
        # Create CyborgDB client
        try:
            self.client = Client(
                base_url=self.vdb_endpoint,
                api_key=self.api_key,
                verify_ssl=False # Set to True in production with valid SSL certificates
            )
            self._connected = True
            logger.debug(f"Connected to CyborgDB at {self.vdb_endpoint}")
        except Exception as e:
            logger.error(f"Failed to connect to CyborgDB at {self.vdb_endpoint}: {e}")
            raise
        
        # Store vectorstore instances per collection
        self._vectorstores = {}
        
        # Store direct index instances for upsert operations
        self._indexes = {}
        
        # Connection alias for compatibility
        self.connection_alias = f"cyborgdb_{str(uuid4())[:8]}"

        kwargs = locals().copy()
        kwargs.pop("self", None)
        super().__init__(**kwargs)

    @property
    def collection_name(self) -> str:
        """Get the collection name."""
        return self._collection_name
    
    @collection_name.setter
    def collection_name(self, collection_name: str) -> None:
        """Set the collection name."""
        self._collection_name = collection_name

    def _get_or_create_index(self, collection_name: str, dimension: int = 1536) -> EncryptedIndex:
        """
        Get existing index or create a new one for direct upsert operations.
        """
        # Check cache first
        if collection_name in self._indexes:
            return self._indexes[collection_name]
        
        # Check if index exists
        try:
            existing_indexes = self.client.list_indexes()
            if collection_name in existing_indexes:
                # Load existing index
                index = EncryptedIndex(
                    index_name=collection_name,
                    index_key=self.index_key,
                    api=self.client.api,
                    api_client=self.client.api_client
                )
                self._indexes[collection_name] = index
                logger.info(f"Loaded existing index: {collection_name}")
                return index
        except Exception as e:
            logger.warning(f"Error checking existing indexes: {e}")
        
        # Create new index
        try:
            # Create index configuration
            index_config = IndexIVFFlat(
                dimension=dimension,
                n_lists=128,  # Default number of inverted lists
                metric="euclidean"
            )
            
            # Create index via client
            index = self.client.create_index(
                index_name=collection_name,
                index_key=self.index_key,
                index_config=index_config,
                embedding_model=None  # We'll provide embeddings directly
            )
            
            self._indexes[collection_name] = index
            logger.info(f"Created new index: {collection_name}")
            return index
            
        except Exception as e:
            logger.error(f"Failed to create index: {e}")
            raise
    
    def _get_or_create_vectorstore(self, collection_name: str, dimension: Optional[int] = None, get_only: bool = False) -> CyborgVectorStore | None:
        """
        Get existing vectorstore or create a new one for the collection.
        """
        # Check cache first
        if collection_name in self._vectorstores:
            return self._vectorstores[collection_name]

        # If only getting existing, check if collection exists
        if get_only and not self.check_collection_exists(collection_name):
            return None
        
        # Create/load vectorstore instance
        vectorstore = CyborgVectorStore(
            index_name=collection_name,
            index_key=self.index_key,
            api_key=self.api_key,
            base_url=self.vdb_endpoint,
            embedding=self.embedding_model,
            dimension=dimension,
        )
        
        # Add collection_name attribute for compatibility with the retrieval chain
        vectorstore.collection_name = collection_name
        
        # Cache it
        self._vectorstores[collection_name] = vectorstore
        return vectorstore

    # ----------------------------------------------------------------------------------------------
    # Implementations of the abstract methods of the NV-Ingest Client VDB class
    def _check_index_exists(self, index_name: str) -> bool:
        """
        Check if an index exists in CyborgDB.
        """
        return index_name in self.client.list_indexes()
    
    def create_index(self):
        """
        Create an index (collection) in CyborgDB if it doesn't exist.
        """
        logger.info(f"Creating CyborgDB index if not exists: {self._collection_name}")
        self._get_or_create_vectorstore(
            collection_name=self._collection_name
        )

    def write_to_index(self, records: List[Dict[str, Any]], **kwargs):
        """
        Write records to the CyborgDB index using direct upsert.
        Based on implementation from cyborg.py.
        
        Args:
            records: List of records to insert
            **kwargs: Additional parameters
        """
        logger.info(f"Writing {len(records)} records to CyborgDB index")
        
        collection_name = kwargs.get("collection_name", self._collection_name)
        
        # Get or create the direct index for upsert
        index = self._get_or_create_index(collection_name)
        
        if not index:
            raise ValueError(f"Failed to get index for collection {collection_name}")
        
        # Convert records to CyborgDB format for direct upsert
        items = []
        
        for idx, record in enumerate(records):
            logger.debug(f"Processing record {idx + 1}/{len(records)}")
            
            if not isinstance(record, dict):
                logger.error(f"Record {idx} is not a dict.")
                raise TypeError(f"Expected dict, got {type(record).__name__}")
            
            # Extract ID
            if "id" in record and record["id"] is not None:
                id_value = str(record["id"])
            elif "_id" in record and record["_id"] is not None:
                id_value = str(record["_id"])
            else:
                id_value = str(uuid4())
                logger.debug(f"Generated new UUID for record: {id_value}")
            
            item = {"id": id_value}
            
            # Handle vector field - look for embeddings in various locations
            vector_found = False
            if "vector" in record:
                item["vector"] = record["vector"]
                vector_found = True
                logger.debug(f"Found vector field")
            elif "embedding" in record:
                item["vector"] = record["embedding"]
                vector_found = True
                logger.debug(f"Found embedding field")
            elif "metadata" in record and "embedding" in record["metadata"]:
                item["vector"] = record["metadata"]["embedding"]
                vector_found = True
                logger.debug(f"Found embedding in metadata")
            
            if not vector_found:
                logger.warning(f"No vector/embedding found for record {id_value}")
            
            # Handle metadata
            metadata = {}
            
            # Copy metadata from record
            if "metadata" in record:
                metadata.update(record["metadata"])
                
                # Rename 'content' -> '_content' and truncate if needed
                if "content" in metadata:
                    content = metadata.pop("content")
                    max_length = 65535  # CyborgDB limit
                    if content and len(content) > max_length:
                        logger.warning(f"Truncating content from {len(content)} to {max_length} chars")
                        content = content[:max_length] + "...[truncated]"
                    metadata["_content"] = content
                
                # Handle source_metadata
                if "source_metadata" in metadata:
                    metadata["source"] = metadata.pop("source_metadata")
                
                # Remove embedding from metadata if it was there (already in vector field)
                metadata.pop("embedding", None)
            
            # Add source field if present
            if "source" in record:
                metadata["source"] = record["source"]
            
            # Add content_metadata if present
            if "content_metadata" in record:
                metadata["content_metadata"] = record["content_metadata"]
            
            # Apply preprocessing to metadata
            processed_metadata = self._preprocess_metadata(metadata)
            
            if processed_metadata:
                item["metadata"] = processed_metadata
                logger.debug(f"Added {len(processed_metadata)} metadata fields")
            
            items.append(item)
        
        logger.info(f"Prepared {len(items)} items for upsertion")
        
        # Upsert items directly to index
        try:
            logger.info("Starting upsert operation...")
            index.upsert(items)
            logger.info(f"Successfully inserted {len(items)} records into index")
        except Exception as e:
            logger.error(f"Failed to upsert records: {e}", exc_info=True)
            raise
        
        logger.info(f"Successfully wrote {len(records)} records to CyborgDB")

    def retrieval(self, queries: list, **kwargs) -> list[dict[str, Any]]:
        """
        Retrieve documents from CyborgDB based on queries.
        """
        # Placeholder for future implementation if needed
        raise NotImplementedError("Direct retrieval method not implemented. Use retrieval_langchain instead.")
    
    def reindex(self, records: list, **kwargs) -> None:
        """
        Reindex documents in CyborgDB.
        """
        # Placeholder: implement actual reindex logic
        raise NotImplementedError("reindex must be implemented for CyborgDB")

    def run(self, records: List[Dict[str, Any]]) -> None:
        """
        Run the process of ingestion of records to the CyborgDB index.
        
        Args:
            records: List of records to process (can be nested lists/tuples)
        """
        logger.info("Running CyborgDB ingestion pipeline")
        logger.debug(f"Input type: {type(records)}")
        
        # Step 1: Create index if it doesn't exist
        self.create_index()
        
        # Step 2: Flatten and write records to index
        if records:
            # Flatten nested structure to ensure we have List[Dict]
            flat_records = []
            try:
                flat_records = self._normalize_records(records)
                logger.debug(f"Normalized {len(flat_records)} records from input")
            except Exception as e:
                logger.error(f"Failed to normalize records: {e}", exc_info=True)
                raise
            
            if flat_records:
                logger.info(f"Writing {len(flat_records)} records to index")
                self.write_to_index(flat_records)
            else:
                logger.warning("No records to write after normalization")
        else:
            logger.warning("No records provided to process")
        
        logger.info("CyborgDB ingestion pipeline completed")
    
    def _normalize_records(self, records) -> List[Dict[str, Any]]:
        """
        Flattens arbitrarily nested lists/tuples of dicts into List[Dict].
        Handles cases where records might be nested or wrapped.
        
        Args:
            records: Input records (can be dict, list, tuple, or nested combinations)
            
        Returns:
            List of dictionaries
        """
        flat: List[Dict[str, Any]] = []
        stack = [records]
        
        while stack:
            cur = stack.pop()
            if isinstance(cur, dict):
                flat.append(cur)
            elif isinstance(cur, (list, tuple)):
                # Push children in reverse order to maintain order
                stack.extend(reversed(cur))
            elif cur is None:
                # Ignore None values
                continue
            else:
                raise TypeError(
                    f"normalize_records expected dict/list/tuple, got {type(cur).__name__}"
                )
        
        return flat


    # ----------------------------------------------------------------------------------------------
    # Implementations of the abstract methods specific to VDBRag class for ingestion\
    async def check_health(self) -> dict[str, Any]:
        """
        Check CyborgDB health
        """
        status = {
            "service": "CyborgDB",
            "url": self.vdb_endpoint,
            "status": "unknown",
            "error": None
        }

        if not self.vdb_endpoint:
            status["status"] = "skipped"
            status["error"] = "No URL provided"
            return status
        
        try:
            start_time = time.time()
            health = self.client.get_health()

            if health.get("status") == "healthy":
                status["status"] = "healthy"
            else:
                status["status"] = "unhealthy"
                status["error"] = f"Health check returned: {health}"

            status["latency_ms"] = round((time.time() - start_time) * 1000, 2)

        except ImportError:
            status["status"] = "error"
            status["error"] = "CyborgDB client library not installed"

        except Exception as e:
            status["status"] = "error"
            status["error"] = str(e)

        return status

    def create_collection(
        self,
        collection_name: str,
        dimension: int = 2048,
        collection_type: str = "text",
    ) -> None:
        """
        Create a new collection (index) in CyborgDB.
        """
        self._get_or_create_vectorstore(
            collection_name=collection_name, 
            dimension=dimension,
            get_only=False
        )

    def check_collection_exists(self, collection_name: str) -> bool:
        """
        Check if a collection (index) exists in CyborgDB.
        """
        return self._check_index_exists(collection_name)

    def get_collection(self) -> List[Dict[str, Any]]:
        """
        Get the list of collections (indexes) in CyborgDB.
        Note: CyborgDB doesn't support metadata schemas like Milvus.
        """
        try:
            indexes = self.client.list_indexes()
            collection_info = []
            
            for index_name in indexes:
                # We can't easily get the number of entities without the index key
                # and loading each index would be expensive
                collection_info.append({
                    "collection_name": index_name,
                    "num_entities": -1,  # Unknown without index key
                    "metadata_schema": []  # CyborgDB doesn't have separate metadata schema
                })
            
            return collection_info
        except Exception as e:
            logger.error(f"Failed to list CyborgDB indexes: {str(e)}")
            return []

    def delete_collections(
        self,
        collection_names: List[str],
    ) -> Dict[str, Any]:
        """
        Delete collections (indexes) from CyborgDB.
        """
        deleted_collections = []
        failed_collections = []
        
        for collection_name in collection_names:
            try:
                # Get the vectorstore for this collection
                vectorstore = self._get_or_create_vectorstore(collection_name)

                if not vectorstore:
                    failed_collections.append({
                        "collection_name": collection_name,
                        "error_message": "Collection not found"
                    })
                    continue

                if vectorstore.delete(delete_index=True):
                    deleted_collections.append(collection_name)
                    logger.info(f"Deleted CyborgDB collection: {collection_name}")
                    # Remove from cache
                    if collection_name in self._vectorstores:
                        del self._vectorstores[collection_name]
                else:
                    failed_collections.append({
                        "collection_name": collection_name,
                        "error_message": "Failed to delete index"
                    })
            except Exception as e:
                failed_collections.append({
                    "collection_name": collection_name,
                    "error_message": str(e)
                })
                logger.error(f"Failed to delete collection {collection_name}: {str(e)}")
        
        return {
            "message": "Collection deletion process completed.",
            "successful": deleted_collections,
            "failed": failed_collections,
            "total_success": len(deleted_collections),
            "total_failed": len(failed_collections),
        }

    def get_documents(self, collection_name: str) -> List[Dict[str, Any]]:
        """
        Get the list of documents in a collection.
        """
        try:
            vectorstore = self._get_or_create_vectorstore(collection_name, get_only=True)

            if not vectorstore:
                logger.warning(f"Collection {collection_name} not found")
                return []
            
            # Use list_ids to get all document IDs
            all_ids = vectorstore.list_ids()
            logger.info(f"Retrieved {len(all_ids)} document IDs from CyborgDB collection {collection_name}")
            
            # Process documents in batches
            batch_size = 1000
            documents_list = []
            filepaths_added = set()
            
            for i in range(0, len(all_ids), batch_size):
                batch_ids = all_ids[i:i + batch_size]
                batch_docs = vectorstore.get(ids=batch_ids)

                # Process each document
                for doc in batch_docs:
                    try:
                        metadata = doc.metadata if doc.metadata else {}

                        # Extract filename from source
                        filename = self._extract_filename(metadata)
                        
                        if filename and filename not in filepaths_added:
                            # Build metadata dict with proper serialization
                            metadata_dict = {}
                            for key, value in metadata.items():
                                if key not in ['vector', 'embedding', '_content']:
                                    # Ensure value is JSON-serializable
                                    if isinstance(value, (dict, list)):
                                        # For nested structures, convert to string representation
                                        import json
                                        try:
                                            # Try to serialize to JSON string
                                            metadata_dict[key] = json.dumps(value)
                                        except:
                                            # If serialization fails, convert to string
                                            metadata_dict[key] = str(value)
                                    elif isinstance(value, (str, int, float, bool, type(None))):
                                        metadata_dict[key] = value
                                    else:
                                        # Convert other types to string
                                        metadata_dict[key] = str(value)
                            
                            documents_list.append({
                                "document_name": filename,
                                "metadata": metadata_dict
                            })
                            filepaths_added.add(filename)
                        elif not filename:
                            raise ValueError("Filename could not be extracted from metadata")
                    except Exception as e:
                        logger.debug(f"Error processing document: {e}")
                        continue
            
            return documents_list
            
        except Exception as e:
            logger.error(f"Error retrieving documents from CyborgDB collection {collection_name}: {e}")
            return []

    def delete_documents(
        self,
        collection_name: str,
        source_values: List[str],
    ) -> bool:
        """
        Delete documents from a collection by source values.
        Note: CyborgDB doesn't support deletion by metadata query,
        so we need to find document IDs first.
        """
        try:
            vectorstore = self._get_or_create_vectorstore(collection_name, get_only=True)

            if not vectorstore:
                logger.warning(f"Collection {collection_name} not found")
                return False
            
            print(f"Deleting documents with source values: {source_values}")

            # Find document IDs matching the source values
            all_ids = vectorstore.list_ids()
            matches = [doc_id for doc_id in all_ids if (doc_id in source_values)]

            print(f"{len(matches)}/{len(source_values)} documents matched for deletion")

            return vectorstore.delete(ids=source_values)
            
        except Exception as e:
            logger.error(f"Error deleting documents from CyborgDB: {e}")
            return False

    def create_metadata_schema_collection(self) -> None:
        """
        Create metadata schema collection.
        Note: CyborgDB doesn't have a separate metadata schema concept like Milvus.
        """
        # Not applicable for CyborgDB
        logger.debug("CyborgDB doesn't require separate metadata schema collections")
        pass

    def add_metadata_schema(
        self,
        collection_name: str,
        metadata_schema: List[Dict[str, Any]],
    ) -> None:
        """
        Add metadata schema to collection.
        Note: CyborgDB doesn't have a separate metadata schema concept.
        """
        # Not applicable for CyborgDB
        logger.debug(f"CyborgDB doesn't support separate metadata schemas for collection {collection_name}")
        pass

    def get_metadata_schema(
        self,
        collection_name: str,
    ) -> List[Dict[str, Any]]:
        """
        Get metadata schema for collection.
        Note: CyborgDB doesn't have a separate metadata schema concept.
        """
        # Return empty list as CyborgDB doesn't have metadata schemas
        return []

    # ----------------------------------------------------------------------------------------------
    # Implementations of the abstract methods specific to VDBRag class for retrieval
    def get_langchain_vectorstore(
        self,
        collection_name: str,
    ) -> VectorStore:
        """
        Get the vectorstore for a collection.
        """
        return self._get_or_create_vectorstore(collection_name, get_only=True)

    def retrieval_langchain(
        self,
        query: str,
        collection_name: str,
        vectorstore: CyborgVectorStore = None,
        top_k: int = 10,
        filter_expr: Union[str, List[Dict[str, Any]]] = "",
        otel_ctx: otel_context = None,
    ) -> List[Dict[str, Any]]:
        """
        Retrieve documents from a collection using langchain
        """
        if vectorstore is None:
            vectorstore = self.get_langchain_vectorstore(collection_name)

        start_time = time.time()

        retriever = vectorstore.as_retriever(search_kwargs={"k": top_k})

        retriever_lambda = RunnableLambda(
            lambda x: retriever.invoke(
                x,
                expr=filter_expr,
            )
        )
        retriever_chain = {"context": retriever_lambda} | RunnableAssign(
            {"context": lambda input: input["context"]}
        )
        retriever_docs = retriever_chain.invoke(query, config={"run_name": "retriever"})
        docs = retriever_docs.get("context", [])
        # collection_name is already provided as a parameter

        end_time = time.time()
        latency = end_time - start_time
        logger.info(f" CyborgDB Retrieval latency: {latency:.4f} seconds")

        return self._add_collection_name_to_retrieved_docs(docs, collection_name)

    # ----------------------------------------------------------------------------------------------
    # Helper methods
    @staticmethod
    def _extract_filename(metadata: Dict[str, Any]) -> Optional[str]:
        """
        Extract filename from metadata.
        """
        if 'source' in metadata:
            if isinstance(metadata['source'], str):
                return os.path.basename(metadata['source'])
            elif isinstance(metadata['source'], dict) and 'source_name' in metadata['source']:
                return os.path.basename(metadata['source']['source_name'])
        return None
    
    @staticmethod
    def _preprocess_metadata(metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        Preprocess metadata before insertion, following best practices from cyborg.py.
        
        This includes:
        - Renaming 'content' to '_content' to avoid conflicts
        - Truncating content to prevent issues
        - Handling source_metadata field
        """
        processed = metadata.copy()
        
        # Rename 'content' -> '_content' if it exists and truncate if needed
        if "content" in processed:
            content = processed.pop("content")
            # Get max content length from env or use default (2000 chars to prevent reranker issues)
            max_content_length = int(os.getenv('MAX_DOCUMENT_CONTENT_LENGTH', '2000'))
            # Use the smaller of env setting or CyborgDB limit
            max_length = min(max_content_length, 65535)
            if content and len(content) > max_length:
                logger.warning(f"Truncating content from {len(content)} to {max_length} chars")
                content = content[:max_length] + "...[truncated]"
            processed["_content"] = content
        
        # Handle source_metadata field
        if "source_metadata" in processed:
            processed["source"] = processed.pop("source_metadata")
        
        return processed
    
    @staticmethod
    def _add_collection_name_to_retrieved_docs(
        docs: List[Document],
        collection_name: str
    ) -> list[Document]:
        """
        Add the collection name to the retrieved documents.
        This is done to ensure the collection name is available in the
        metadata of the documents for preparing citations in case of multi-collection retrieval.
        """
        for doc in docs:
            doc.metadata["collection_name"] = collection_name
        return docs