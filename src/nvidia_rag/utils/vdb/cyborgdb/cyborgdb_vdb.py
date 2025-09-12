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
from typing import Any, Dict, List, Optional, Union
from uuid import uuid4

from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStore

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


class CyborgDBVDB(VDB, VDBRag):
    def __init__(self, **kwargs):
        self.embedding_model = kwargs.pop("embedding_model")  # Needed for retrieval
        if VDB != object:
            super().__init__(**kwargs)
        
        # CyborgDB specific parameters
        self.vdb_endpoint = kwargs.get("cyborgdb_uri", kwargs.get("api_url", kwargs.get("base_url")))
        self._collection_name = kwargs.get("collection_name", kwargs.get("index_name"))
        self.api_key = kwargs.get("api_key") or os.getenv('CYBORGDB_API_KEY')
        self.index_key = kwargs.get("index_key", None)
        
        # Validate required parameters
        if not self.api_key:
            raise ValueError("CyborgDB API key is required. Set it in config or CYBORGDB_API_KEY env var")
        
        if not self.index_key:
            # Generate a secure key if not provided
            self.index_key = generate_key()
            logger.warning("No index_key provided, generated a new one. Store this key securely for future access.")
        
        # Create CyborgDB client
        try:
            self.client = Client(
                base_url=self.vdb_endpoint,
                api_key=self.api_key
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

    def close(self):
        """Close the CyborgDB connection."""
        if self._connected:
            try:
                # CyborgDB client doesn't have explicit disconnect
                self._connected = False
                logger.debug(f"Closed CyborgDB client for {self.vdb_endpoint}")
            except Exception as e:
                logger.warning(f"Error closing CyborgDB client: {e}")

    def __enter__(self):
        """Enter the runtime context (for use as context manager)."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit the runtime context."""
        self.close()

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
        try:
            # Check if index already exists
            existing_indexes = self.client.list_indexes()
            if collection_name in existing_indexes:
                logger.info(f"Collection '{collection_name}' already exists in CyborgDB")
                return
            
            # Prepare index config params
            index_config_params = {}
            if hasattr(CONFIG.vector_store, 'pq_dim'):
                index_config_params['pq_dim'] = CONFIG.vector_store.pq_dim
            if hasattr(CONFIG.vector_store, 'pq_bits'):
                index_config_params['pq_bits'] = CONFIG.vector_store.pq_bits
            
            # Create the vectorstore which will create the index
            vectorstore = CyborgVectorStore(
                index_name=collection_name,
                index_key=self.index_key,
                api_key=self.api_key,
                base_url=self.vdb_endpoint,
                embedding=self.embedding_model,
                index_type=CONFIG.vector_store.index_type.lower() if hasattr(CONFIG.vector_store, 'index_type') else "ivfflat",
                index_config_params=index_config_params if index_config_params else None,
                dimension=dimension,
                metric=CONFIG.vector_store.metric if hasattr(CONFIG.vector_store, 'metric') else "cosine"
            )
            
            # Store the vectorstore for later use
            self._vectorstores[collection_name] = vectorstore
            
            logger.info(f"Created CyborgDB collection '{collection_name}' with dimension {dimension}")
        except Exception as e:
            logger.error(f"Failed to create CyborgDB collection {collection_name}: {str(e)}")
            raise

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
                # Get or create vectorstore for this collection
                vectorstore = self._get_or_create_vectorstore(collection_name)
                
                if vectorstore:
                    # Delete the entire index
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
                else:
                    failed_collections.append({
                        "collection_name": collection_name,
                        "error_message": "Collection not found or unable to access"
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
            vectorstore = self._get_or_create_vectorstore(collection_name)
            
            if not vectorstore:
                logger.warning(f"Collection {collection_name} not found")
                return []
            
            # Use list_ids to get all document IDs
            if hasattr(vectorstore, 'list_ids'):
                all_ids = vectorstore.list_ids()
                logger.info(f"Retrieved {len(all_ids)} document IDs from CyborgDB collection {collection_name}")
                
                if not all_ids:
                    return []
                
                # Process documents in batches
                batch_size = 1000
                documents_list = []
                filepaths_added = set()
                
                for i in range(0, len(all_ids), batch_size):
                    batch_ids = all_ids[i:i + batch_size]
                    
                    # Get documents using the get method
                    if hasattr(vectorstore, 'get'):
                        batch_docs = vectorstore.get(batch_ids)
                    elif hasattr(vectorstore, 'index') and hasattr(vectorstore.index, 'get'):
                        # Fallback to using the underlying index
                        batch_items = vectorstore.index.get(batch_ids, include=["metadata"])
                        batch_docs = []
                        for item in batch_items:
                            if item and item.get('metadata'):
                                metadata = item['metadata']
                                if isinstance(metadata, str):
                                    try:
                                        metadata = json.loads(metadata)
                                    except json.JSONDecodeError:
                                        metadata = {"raw": metadata}
                                # Extract content from metadata
                                metadata_copy = metadata.copy() if isinstance(metadata, dict) else {}
                                content = metadata_copy.pop("_content", "")
                                batch_docs.append(Document(page_content=content, metadata=metadata_copy))
                    else:
                        batch_docs = []
                    
                    # Process each document
                    for doc in batch_docs:
                        if doc:
                            try:
                                # Extract metadata
                                if hasattr(doc, 'metadata'):
                                    metadata = doc.metadata
                                elif isinstance(doc, dict) and 'metadata' in doc:
                                    metadata = doc['metadata']
                                else:
                                    continue
                                
                                # Extract filename from source
                                filename = self._extract_filename(metadata)
                                
                                if filename and filename not in filepaths_added:
                                    # Build metadata dict
                                    metadata_dict = {}
                                    for key, value in metadata.items():
                                        if key not in ['vector', 'embedding', '_content']:
                                            metadata_dict[key] = value
                                    
                                    documents_list.append({
                                        "document_name": filename,
                                        "metadata": metadata_dict
                                    })
                                    filepaths_added.add(filename)
                            except Exception as e:
                                logger.debug(f"Error processing document: {e}")
                                continue
                
                return documents_list
            else:
                logger.warning("CyborgDB vectorstore doesn't support list_ids method")
                return []
                
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
            vectorstore = self._get_or_create_vectorstore(collection_name)
            
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
        return self._get_or_create_vectorstore(collection_name)

    def retrieval_langchain(
        self,
        query: str,
        collection_name: str,
        top_k: int = 10,
        filter_expr: Union[str, List[Dict[str, Any]]] = "",
    ) -> List[Dict[str, Any]]:
        """
        Perform semantic search and return top-k relevant documents.
        """
        try:
            vectorstore = self._get_or_create_vectorstore(collection_name)
            
            if not vectorstore:
                logger.warning(f"Collection {collection_name} not found")
                return []
            
            # Convert filter expression to CyborgDB format if needed
            filter_dict = None
            if filter_expr and isinstance(filter_expr, str):
                # CyborgDB uses dict filters, not string expressions
                # This would need custom parsing logic based on your filter format
                logger.warning("String filter expressions not yet implemented for CyborgDB")
            elif filter_expr and isinstance(filter_expr, list):
                # Convert list of dicts to single dict
                filter_dict = {}
                for f in filter_expr:
                    filter_dict.update(f)
            
            # Perform similarity search
            docs = vectorstore.similarity_search(
                query=query,
                k=top_k,
                filter=filter_dict
            )
            
            # Convert documents to expected format
            results = []
            for doc in docs:
                result = {
                    "page_content": doc.page_content,
                    "metadata": doc.metadata.copy() if doc.metadata else {}
                }
                # Add collection name to metadata
                result["metadata"]["collection_name"] = collection_name
                results.append(result)
            
            return results
            
        except Exception as e:
            logger.error(f"Error during CyborgDB retrieval: {e}")
            return []

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

    def _get_or_create_vectorstore(self, collection_name: str) -> Optional[CyborgVectorStore]:
        """
        Get existing vectorstore or create a new one for the collection.
        """
        # Check cache first
        if collection_name in self._vectorstores:
            return self._vectorstores[collection_name]
        
        try:
            # Check if index exists
            existing_indexes = self.client.list_indexes()
            if collection_name not in existing_indexes:
                logger.warning(f"Collection {collection_name} does not exist")
                return None
            
            # Prepare index config params
            index_config_params = {}
            if hasattr(CONFIG.vector_store, 'nlist'):
                index_config_params['n_lists'] = CONFIG.vector_store.nlist
            if hasattr(CONFIG.vector_store, 'pq_dim'):
                index_config_params['pq_dim'] = CONFIG.vector_store.pq_dim
            if hasattr(CONFIG.vector_store, 'pq_bits'):
                index_config_params['pq_bits'] = CONFIG.vector_store.pq_bits
            
            # Get dimension from config if available
            dimension = None
            if hasattr(CONFIG.embeddings, 'dimensions'):
                dimension = CONFIG.embeddings.dimensions
            
            # Create vectorstore instance
            vectorstore = CyborgVectorStore(
                index_name=collection_name,
                index_key=self.index_key,
                api_key=self.api_key,
                base_url=self.vdb_endpoint,
                embedding=self.embedding_model,
                index_type=CONFIG.vector_store.index_type.lower() if hasattr(CONFIG.vector_store, 'index_type') else "ivfflat",
                index_config_params=index_config_params if index_config_params else None,
                dimension=dimension,
                metric=CONFIG.vector_store.metric if hasattr(CONFIG.vector_store, 'metric') else "cosine"
            )
            
            # Cache it
            self._vectorstores[collection_name] = vectorstore
            return vectorstore
            
        except Exception as e:
            logger.error(f"Failed to create vectorstore for collection {collection_name}: {e}")
            return None