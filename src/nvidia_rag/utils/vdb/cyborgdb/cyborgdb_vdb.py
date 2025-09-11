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
This module contains the implementation of the ElasticVDB class,
which provides Elasticsearch vector database operations for RAG applications.
Extends both VDB class for nv-ingest operations and VDBRag for RAG-specific functionality.

NV-Ingest Client VDB Operations:
1. _check_index_exists: Check if the index exists in Elasticsearch
2. create_index: Create an index in Elasticsearch
3. write_to_index: Write records to the Elasticsearch index
4. retrieval: Retrieve documents from Elasticsearch based on queries
5. reindex: Reindex documents in Elasticsearch
6. run: Run the process of ingestion of records to the Elasticsearch index

Collection Management:
7. create_collection: Create a new collection with specified dimensions and type
8. check_collection_exists: Check if the specified collection exists
9. get_collection: Retrieve all collections with their metadata schemas
10. delete_collections: Delete multiple collections and their associated metadata

Document Management:
11. get_documents: Retrieve all unique documents from the specified collection
12. delete_documents: Remove documents matching the specified source values

Metadata Schema Management:
13. create_metadata_schema_collection: Initialize the metadata schema storage collection
14. add_metadata_schema: Store metadata schema configuration for the collection
15. get_metadata_schema: Retrieve the metadata schema for the specified collection

Retrieval Operations:
16. retrieval_langchain: Perform semantic search and return top-k relevant documents
17. _get_langchain_vectorstore: Get the vectorstore for a collection
18. _add_collection_name_to_retrieved_docs: Add the collection name to the retrieved documents
"""

import logging
import os
import time
from typing import Dict, List, Optional, Union, Any

import numpy as np
from cyborgdb import CyborgDB
# probably import some other cyborgdb types here

from nvidia_rag.utils.common import get_config
from nvidia_rag.utils.embedding import get_embedding_model
from nvidia_rag.utils.vdb import DEFAULT_METADATA_SCHEMA_COLLECTION
from nvidia_rag.utils.vdb.elasticsearch.es_queries import (
    create_metadata_collection_mapping,
    get_delete_docs_query,
    get_delete_metadata_schema_query,
    get_metadata_schema_query,
    get_unique_sources_query,
)
from nvidia_rag.utils.vdb.vdb_base import VDBRag

logger = logging.getLogger(__name__)
CONFIG = get_config()


class CyborgDBVDB(VDBRag):
    """
    CyborgDBVDB is a subclass of the VDBRag class which is a subclass of the VDB class
    in the nv_ingest_client.util.vdb module.
    It is used to store and retrieve documents from a CyborgDB vector database.
    """

    def __init__(
        self,
        api_url: str,
        api_key: Optional[str],
        index_name: str,
        index_key: Optional[bytes],
        verify_ssl: Optional[bool],
        index_type: Optional[str],
        dimension: Optional[int],
        metric: Optional[str],
        pq_bits: Optional[int],
        pq_dim: Optional[int],
        embedding_model: Optional[str],
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