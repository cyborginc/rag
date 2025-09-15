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

from nvidia_rag.utils.common import get_config
from nvidia_rag.utils.vdb.vdb_base import VDBRag

logger = logging.getLogger(__name__)

try:
    from cyborgdb import Client, generate_key
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

    # ----------------------------------------------------------------------------------------------
    # Implementations of the abstract methods specific to VDBRag class for ingestion
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
        try:
            existing_indexes = self.client.list_indexes()
            return collection_name in existing_indexes
        except Exception as e:
            logger.error(f"Error checking if collection exists: {e}")
            return False

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
                            # Build metadata dict
                            # Should we be using metadata schema here?
                            metadata_dict = {}
                            for key, value in metadata.items():
                                if key not in ['vector', 'embedding', '_content']:
                                    metadata_dict[key] = value
                            
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
            
            # CyborgDB doesn't support deletion by metadata query
            # We would need to track document IDs separately or iterate through all documents
            logger.warning(
                "CyborgDB doesn't support deletion by source values directly. "
                "Document IDs need to be tracked separately for deletion."
            )
            return False
            
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
        collection_name = retriever.vectorstore.collection_name

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

    def write_to_index(self, records: List[Dict[str, Any]], **kwargs):
        """
        Write records to the CyborgDB index.
        Based on implementation from cyborg.py.
        
        Args:
            records: List of records to insert
            **kwargs: Additional parameters
        """
        logger.info(f"Writing {len(records)} records to CyborgDB index")
        
        collection_name = kwargs.get("collection_name", self._collection_name)
        vectorstore = self._get_or_create_vectorstore(collection_name)
        
        if not vectorstore:
            raise ValueError(f"Failed to get vectorstore for collection {collection_name}")
        
        # Convert records to Document format for LangChain
        documents = []
        for idx, record in enumerate(records):
            # Extract text content
            text = record.get("text", "")
            if not text and "metadata" in record:
                text = record["metadata"].get("content", "")
                if not text:
                    text = record["metadata"].get("_content", "")
            
            # Prepare metadata
            metadata = {}
            
            # Handle metadata field
            if "metadata" in record:
                metadata.update(record["metadata"])
            
            # Handle source field
            if "source" in record:
                metadata["source"] = record["source"]
            elif "source_metadata" in record:
                metadata["source"] = record["source_metadata"]
            
            # Handle content_metadata
            if "content_metadata" in record:
                metadata["content_metadata"] = record["content_metadata"]
            
            # Handle vector/embedding (will be used by vectorstore if provided)
            if "vector" in record:
                metadata["embedding"] = record["vector"]
            elif "embedding" in record:
                metadata["embedding"] = record["embedding"]
            
            # Apply preprocessing to metadata
            processed_metadata = self._preprocess_metadata(metadata)
            
            # Create Document
            doc = Document(
                page_content=text,
                metadata=processed_metadata
            )
            documents.append(doc)
        
        # Batch insert documents
        batch_size = kwargs.get("batch_size", 100)
        total_docs = len(documents)
        
        logger.info(f"Inserting {total_docs} documents in batches of {batch_size}")
        
        for i in range(0, total_docs, batch_size):
            batch = documents[i:i + batch_size]
            try:
                # Use the vectorstore to add documents
                # If embeddings are in metadata, the vectorstore should use them
                ids = vectorstore.add_documents(batch)
                logger.debug(f"Inserted batch {i//batch_size + 1}, {len(ids)} documents")
            except Exception as e:
                logger.error(f"Failed to insert batch starting at index {i}: {e}")
                raise
        
        logger.info(f"Successfully inserted {total_docs} documents into CyborgDB")
    
    def create_index(self):
        """
        Create an index (collection) in CyborgDB if it doesn't exist.
        Similar to Elasticsearch's create_index method.
        """
        logger.info(f"Creating CyborgDB index if not exists: {self._collection_name}")
        
        if not self.check_collection_exists(self._collection_name):
            # Use default dimension from config or embedding model
            dimension = self.dense_dim or getattr(CONFIG.embeddings, 'dimensions', 1536)
            logger.info(f"Creating new collection: {self._collection_name} with dimension {dimension}")
            self.create_collection(
                self._collection_name,
                dimension=dimension,
                collection_type="text"
            )
        else:
            logger.info(f"Collection {self._collection_name} already exists")
    
    def run(self, records: List[Dict[str, Any]]) -> None:
        """
        Run the process of ingestion of records to the CyborgDB index.
        Follows the same pattern as Elasticsearch implementation.
        
        Args:
            records: List of records to process
        """
        logger.info("Running CyborgDB ingestion pipeline")
        
        # Step 1: Create index if it doesn't exist
        self.create_index()
        
        # Step 2: Write records to index
        if records:
            self.write_to_index(records)
        else:
            logger.warning("No records provided to process")
        
        logger.info("CyborgDB ingestion pipeline completed")
    
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
        
        # Cache it
        self._vectorstores[collection_name] = vectorstore
        return vectorstore
    
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