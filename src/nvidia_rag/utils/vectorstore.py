# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The wrapper for interacting with milvus/cyborgdb vectorstore and associated functions.
Extended to support CyborgDB as an additional vector store option.
"""

import os
import time
import logging
from typing import List, Dict, Any, Optional, Union
from pathlib import Path
from urllib.parse import urlparse
from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStore
from pymilvus import connections, utility, Collection, MilvusClient, DataType, MilvusException
from pymilvus.orm.types import CONSISTENCY_STRONG
from langchain_milvus import Milvus
from langchain_core.runnables import RunnableAssign, RunnableLambda
from opentelemetry import context as otel_context

from nvidia_rag.utils.common import get_config

logger = logging.getLogger(__name__)

CONFIG = get_config()

DEFAULT_METADATA_SCHEMA_COLLECTION = "metadata_schema"

try:
    from nv_ingest_client.util.milvus import create_nvingest_collection
except Exception:
    logger.debug("Optional nv_ingest_client module not installed.")

try:
    from langchain_milvus import BM25BuiltInFunction
except ImportError:
    logger.error("langchain-milvus is not installed.")
    raise Exception("langchain-milvus is not installed.")

# Import CyborgDB if available
try:
    from cyborgdb import Client, EncryptedIndex, IndexIVFFlat, generate_key
    from cyborgdb.integrations.langchain import CyborgVectorStore
    CYBORGDB_AVAILABLE = True
except ImportError:
    logger.info("CyborgDB not installed. Only Milvus will be available.")
    CYBORGDB_AVAILABLE = False

DOCUMENT_EMBEDDER = None

def create_vectorstore_langchain(document_embedder, collection_name: str = "", vdb_endpoint: str = "") -> VectorStore:
    """Create the vector db index for langchain (supports both Milvus and CyborgDB)."""
    
    config = get_config()
    
    if vdb_endpoint == "":
        vdb_endpoint = config.vector_store.url
    
    if config.vector_store.name == "milvus":
        return _create_milvus_vectorstore(document_embedder, collection_name, vdb_endpoint, config)
    elif config.vector_store.name == "cyborgdb":
        if not CYBORGDB_AVAILABLE:
            raise ValueError("CyborgDB is not installed. Please install cyborgdb-py[langchain]")
        return _create_cyborgdb_vectorstore(document_embedder, collection_name, vdb_endpoint, config)
    else:
        raise ValueError(f"{config.vector_store.name} vector database is not supported. Use 'milvus' or 'cyborgdb'")


def _create_milvus_vectorstore(document_embedder, collection_name: str, vdb_endpoint: str, config) -> VectorStore:
    """Create Milvus vector store."""
    logger.debug("Trying to connect to milvus collection: %s", collection_name)
    if not collection_name:
        collection_name = os.getenv('COLLECTION_NAME', "vector_db")
    
    # Connect to Milvus to check for collection availability
    url = urlparse(vdb_endpoint)
    connection_alias = f"milvus_{url.hostname}_{url.port}"
    connections.connect(connection_alias, host=url.hostname, port=url.port)
    
    # Check if the collection exists
    if not utility.has_collection(collection_name, using=connection_alias):
        logger.debug(f"Collection '{collection_name}' does not exist in Milvus. Aborting vectorstore creation.")
        connections.disconnect(connection_alias)
        return None
    
    logger.debug(f"Collection '{collection_name}' exists. Proceeding with vector store creation.")
    
    if config.vector_store.search_type == "hybrid":
        logger.info("Creating Langchain Milvus object for Hybrid search")
        vectorstore = Milvus(
            document_embedder,
            connection_args={
                "uri": config.vector_store.url
            },
            builtin_function=BM25BuiltInFunction(
                output_field_names="sparse",
                enable_match=True
            ),
            collection_name=collection_name,
            vector_field=["vector", "sparse"]
        )
    elif config.vector_store.search_type == "dense":
        logger.debug("Index type for milvus: %s", config.vector_store.index_type)
        vectorstore = Milvus(
            document_embedder,
            connection_args={
                "uri": vdb_endpoint
            },
            collection_name=collection_name,
            index_params={
                "index_type": config.vector_store.index_type,
                "metric_type": "L2",
                "nlist": config.vector_store.nlist
            },
            search_params={"nprobe": config.vector_store.nprobe},
            auto_id=True
        )
    else:
        logger.error("Invalid search_type: %s. Please select from ['hybrid', 'dense']",
                    config.vector_store.search_type)
        raise ValueError(
            f"{config.vector_store.search_type} search type is not supported" + \
            "Please select from ['hybrid', 'dense']"
        )
    
    logger.debug("Vector store created and saved.")
    return vectorstore


def _create_cyborgdb_vectorstore(document_embedder, collection_name: str, vdb_endpoint: str, config) -> VectorStore:
    """Create CyborgDB vector store."""
    logger.debug("Trying to connect to CyborgDB collection: %s", collection_name)
    if not collection_name:
        collection_name = os.getenv('COLLECTION_NAME', "vector_db")

    # Write document embedder to global variable
    global DOCUMENT_EMBEDDER
    DOCUMENT_EMBEDDER = document_embedder

    # Get CyborgDB specific config
    api_key = config.vector_store.api_key or os.getenv('CYBORGDB_API_KEY')

    if not api_key or api_key == "":
        raise ValueError("CyborgDB API key is required. Set it in config or CYBORGDB_API_KEY env var")
    
    # Create CyborgDB vector store
    vectorstore = CyborgVectorStore(
        index_name=collection_name,
        index_key=config.vector_store.index_key,
        api_key=api_key,
        api_url=vdb_endpoint,
        embedding=document_embedder,
        index_type=config.vector_store.index_type.lower(),
        index_config_params={"n_lists": config.vector_store.nlist},
        metric=config.vector_store.metric
    )
    
    logger.debug("CyborgDB vector store created and saved.")
    return vectorstore


def get_vectorstore(
        document_embedder: "Embeddings",
        collection_name: str = "",
        vdb_endpoint: str = "") -> VectorStore:
    """
    Send a vectorstore object.
    If a Vectorstore object already exists, the function returns that object.
    Otherwise, it creates a new Vectorstore object and returns it.
    """
    return create_vectorstore_langchain(document_embedder, collection_name, vdb_endpoint)


def create_collection(collection_name: str, vdb_endpoint: str, dimension: int = 2048, collection_type: str = "text",document_embedder) -> None:
    """
    Create a new collection in the vector database (Milvus or CyborgDB).
    
    Args:
        collection_name (str): The name of the collection to be created.
        vdb_endpoint (str): The database endpoint.
        dimension (int): The dimension of the embedding vectors.
        collection_type (str): The type of collection to be created.
    
    Raises:
        Exception: If the collection was not created successfully.
    """
    config = get_config()
    
    if config.vector_store.name == "milvus":
        _create_milvus_collection(collection_name, vdb_endpoint, dimension, collection_type, config)
    elif config.vector_store.name == "cyborgdb":
        if not CYBORGDB_AVAILABLE:
            raise ValueError("CyborgDB is not installed. Please install cyborgdb-py[langchain]")
        _create_cyborgdb_collection(collection_name, vdb_endpoint, dimension, collection_type, config,document_embedder)
    else:
        raise ValueError(f"{config.vector_store.name} vector database is not supported")


def _create_milvus_collection(collection_name: str, vdb_endpoint: str, dimension: int, collection_type: str, config) -> None:
    """Create Milvus collection."""
    try:
        url = urlparse(vdb_endpoint)
        connection_alias = f"milvus_{url.hostname}_{url.port}"
        connections.connect(connection_alias, host=url.hostname, port=url.port)
        
        create_nvingest_collection(
            collection_name=collection_name,
            milvus_uri=vdb_endpoint,
            sparse=(config.vector_store.search_type == "hybrid"),
            recreate=False,
            gpu_index=config.vector_store.enable_gpu_index,
            gpu_search=config.vector_store.enable_gpu_search,
            dense_dim=dimension
        )
        connections.disconnect(connection_alias)
    except Exception as e:
        logger.error(f"Failed to create Milvus collection {collection_name}: {str(e)}")
        raise Exception(f"Failed to create Milvus collection {collection_name}: {str(e)}")


def _create_cyborgdb_collection(collection_name: str, vdb_endpoint: str, dimension: int, collection_type: str, config,document_embedder) -> None:
    """Create CyborgDB collection (index)."""
    try:
        # Get CyborgDB specific config
        api_key = config.vector_store.api_key or os.getenv('CYBORGDB_API_KEY')
    
        if not api_key or api_key == "":
            raise ValueError("CyborgDB API key is required. Set it in config or CYBORGDB_API_KEY env var")
        
        if DOCUMENT_EMBEDDER is None:
            raise ValueError("Document embedder is not set. Please provide a valid embedding model.")
        
        # Create CyborgDB vector store which will create the index
        vectorstore = CyborgVectorStore(
            index_name=collection_name,
            index_key=config.vector_store.index_key,
            api_key=config.vector_store.api_key,
            api_url=vdb_endpoint,
            embedding=document_embedder,
            index_type=config.vector_store.index_type.lower(),
            index_config_params={'n_lists': config.vector_store.nlist},
            dimension=dimension,
            metric=config.vector_store.metric        
        )
        
        logger.info(f"CyborgDB collection '{collection_name}' created successfully")
    except Exception as e:
        logger.error(f"Failed to create CyborgDB collection {collection_name}: {str(e)}")
        raise Exception(f"Failed to create CyborgDB collection {collection_name}: {str(e)}")


def create_collections(collection_names: List[str], vdb_endpoint: str, dimension: int = 2048, collection_type: str = "text") -> Dict[str, any]:
    """
    Create multiple collections in the vector database.
    
    Args:
        vdb_endpoint (str): The database endpoint.
        collection_names (List[str]): List of collection names to be created.
        dimension (int): The dimension of the embedding vectors.
        collection_type (str): The type of collection to be created.
    
    Returns:
        dict: Response with creation status.
    """
    try:
        if not len(collection_names):
            return {
                "message": "No collections to create. Please provide a list of collection names.",
                "successful": [],
                "failed": [],
                "total_success": 0,
                "total_failed": 0
            }
        
        created_collections = []
        failed_collections = []
        
        for collection_name in collection_names:
            try:
                create_collection(
                    collection_name=collection_name,
                    vdb_endpoint=vdb_endpoint,
                    dimension=dimension,
                    collection_type=collection_type
                )
                created_collections.append(collection_name)
                logger.info(f"Collection '{collection_name}' created successfully in {vdb_endpoint}.")
            except Exception as e:
                failed_collections.append({
                    "collection_name": collection_name,
                    "error_message": str(e)
                })
                logger.error(f"Failed to create collection {collection_name}: {str(e)}")
        
        return {
            "message": "Collection creation process completed.",
            "successful": created_collections,
            "failed": failed_collections,
            "total_success": len(created_collections),
            "total_failed": len(failed_collections)
        }
    except Exception as e:
        logger.error(f"Failed to create collections due to error: {str(e)}")
        return {
            "message": f"Failed to create collections due to error: {str(e)}",
            "successful": [],
            "failed": collection_names,
            "total_success": 0,
            "total_failed": len(collection_names)
        }


def get_collection(vdb_endpoint: str = "") -> Dict[str, Any]:
    """Get list of all collections in vectorstore along with the number of rows in each collection."""
    
    config = get_config()

    print(f"\n\n\t\tVDB ENDPOINT:\t{vdb_endpoint}\n")
    print(f"\n\n\t\tVECTOR STORE NAME:\t{config.vector_store.name}\n")
    
    if config.vector_store.name == "milvus":
        return _get_milvus_collections(vdb_endpoint)
    elif config.vector_store.name == "cyborgdb":
        if not CYBORGDB_AVAILABLE:
            raise ValueError("CyborgDB is not installed. Please install cyborgdb-py[langchain]")
        return _get_cyborgdb_collections(vdb_endpoint, config)
    else:
        raise ValueError(f"{config.vector_store.name} vector database does not support collection listing")


def _get_milvus_collections(vdb_endpoint: str) -> List[Dict[str, Any]]:
    """Get Milvus collections."""
    url = urlparse(vdb_endpoint)
    connection_alias = f"milvus_{url.hostname}_{url.port}"
    connections.connect(connection_alias, host=url.hostname, port=url.port)
    
    # Get list of collections
    collections = utility.list_collections(using=connection_alias)
    
    # Get document count for each collection
    collection_info = []
    for collection in collections:
        collection_obj = Collection(collection, using=connection_alias)
        num_entities = collection_obj.num_entities
        collection_info.append({"collection_name": collection, "num_entities": num_entities})
    
    # Disconnect from Milvus
    connections.disconnect(connection_alias)
    
    # Get metadata schema for each collection
    entities = get_milvus_entities(DEFAULT_METADATA_SCHEMA_COLLECTION, vdb_endpoint, filter="")
    collection_metadata_schema_map = {}
    for entity in entities:
        collection_metadata_schema_map[entity["collection_name"]] = entity["metadata_schema"]
    for collection_info_item in collection_info:
        collection_name = collection_info_item["collection_name"]
        collection_info_item.update({
            "metadata_schema": collection_metadata_schema_map.get(collection_name, [])
        })
    
    return collection_info


def _get_cyborgdb_collections(vdb_endpoint: str, config) -> List[Dict[str, Any]]:
    """Get CyborgDB collections (indexes)."""
    api_key = config.vector_store.api_key or os.getenv('CYBORGDB_API_KEY')
    
    if not api_key:
        raise ValueError("CyborgDB API key is required")
    
    # Create CyborgDB client
    client = Client(api_url=vdb_endpoint, api_key=api_key)
    
    # Get list of indexes
    try:
        indexes = client.list_indexes()
        collection_info = []
        
        for index_name in indexes:
            # For CyborgDB, we can't easily get the number of entities without loading the index
            # This would require the index key, which we might not have for all indexes
            collection_info.append({
                "collection_name": index_name,
                "num_entities": -1,  # Unknown without index key
                "metadata_schema": []  # CyborgDB doesn't have a separate metadata schema
            })
        
        return collection_info
    except Exception as e:
        logger.error(f"Failed to list CyborgDB indexes: {str(e)}")
        return []


def delete_collections(vdb_endpoint: str, collection_names: List[str]) -> dict:
    """
    Delete a list of collections from the vector database.
    
    Args:
        vdb_endpoint (str): The database endpoint.
        collection_names (List[str]): List of collection names to be deleted.
    
    Returns:
        dict: Response with deletion status.
    """
    config = get_config()
    
    if config.vector_store.name == "milvus":
        return _delete_milvus_collections(vdb_endpoint, collection_names)
    elif config.vector_store.name == "cyborgdb":
        if not CYBORGDB_AVAILABLE:
            raise ValueError("CyborgDB is not installed. Please install cyborgdb-py[langchain]")
        return _delete_cyborgdb_collections(vdb_endpoint, collection_names, config)
    else:
        raise ValueError(f"{config.vector_store.name} vector database is not supported")


def _delete_milvus_collections(vdb_endpoint: str, collection_names: List[str]) -> dict:
    """Delete Milvus collections."""
    try:
        if not len(collection_names):
            return {
                "message": "No collections to delete. Please provide a list of collection names.",
                "successful": [],
                "failed": [],
                "total_success": 0,
                "total_failed": 0
            }
        
        # Parse endpoint and connect
        url = urlparse(vdb_endpoint)
        connection_alias = f"milvus_{url.hostname}_{url.port}"
        connections.connect(connection_alias, host=url.hostname, port=url.port)
        
        deleted_collections = []
        failed_collections = []
        
        for collection in collection_names:
            try:
                if utility.has_collection(collection, using=connection_alias):
                    utility.drop_collection(collection, using=connection_alias)
                    deleted_collections.append(collection)
                    logger.info(f"Deleted collection: {collection}")
                else:
                    failed_collections.append(collection)
                    logger.warning(f"Collection {collection} not found.")
            except Exception as e:
                failed_collections.append(collection)
                logger.error(f"Failed to delete collection {collection}: {str(e)}")
        
        # Disconnect from Milvus
        connections.disconnect(connection_alias)
        
        # Delete the metadata schema from the collection
        for collection_name in deleted_collections:
            delete_entities(DEFAULT_METADATA_SCHEMA_COLLECTION, vdb_endpoint, f"collection_name == '{collection_name}'")
        
        return {
            "message": "Collection deletion process completed.",
            "successful": deleted_collections,
            "failed": failed_collections,
            "total_success": len(deleted_collections),
            "total_failed": len(failed_collections)
        }
    except Exception as e:
        logger.error(f"Failed to delete collections due to error: {str(e)}")
        return {
            "message": f"Failed to delete collections due to error: {str(e)}",
            "successful": [],
            "failed": collection_names,
            "total_success": 0,
            "total_failed": len(collection_names)
        }


def _delete_cyborgdb_collections(
    vdb_endpoint: str,
    collection_names: List[str],
    config: Any  # Should match your config object structure
) -> dict:
    """
    Delete CyborgDB indexes corresponding to given collection names.
    
    Args:
        vdb_endpoint (str): The base API URL of CyborgDB.
        collection_names (List[str]): List of index (collection) names.
        config (Any): Config object containing api_key, encryption key, and embedding.

    Returns:
        dict: Mapping of collection name to deletion success (True/False).
    """

    if not DOCUMENT_EMBEDDER:
        raise ValueError("Document embedder is not set. Please provide a valid embedding model.")
    
    successful = []
    failed = []
    for name in collection_names:
        try:
            store = CyborgVectorStore(
                index_name=name,
                index_key=config.vector_store.index_key,
                api_key=config.vector_store.api_key,
                api_url=vdb_endpoint,
                embedding=DOCUMENT_EMBEDDER,
            )
            if store.delete(delete_index=True):
                successful.append(name)
            else:
                failed.append(name)
        except Exception as e:
            failed.append(name)
            logger.error(f"Failed to delete CyborgDB collection '{name}': {e}")
    return {
        "message": "CyborgDB collection deletion process completed.",
        "successful": successful,
        "failed": failed,
        "total_success": len(successful),
        "total_failed": len(failed)
    }


def get_docs_vectorstore_langchain(
        vectorstore: VectorStore,
        collection_name: str,
        vdb_endpoint: str
    ) -> List[Dict[str, Any]]:
    """Retrieves filenames stored in the vector store implemented in LangChain."""
    
    settings = get_config()
    
    if settings.vector_store.name == "milvus":
        return _get_docs_milvus(vectorstore, collection_name, vdb_endpoint, settings)
    elif settings.vector_store.name == "cyborgdb":
        return _get_docs_cyborgdb(vectorstore, collection_name, vdb_endpoint, settings)
    else:
        logger.error(f"Unsupported vector store: {settings.vector_store.name}")
        return []


def _get_docs_milvus(vectorstore: VectorStore, collection_name: str, vdb_endpoint: str, settings) -> List[Dict[str, Any]]:
    """Get documents from Milvus."""
    try:
        extract_filename = lambda metadata: os.path.basename(metadata['source'] if type(metadata['source']) == str else metadata.get('source').get('source_name'))
        
        metadata_schema = get_metadata_schema(collection_name, vdb_endpoint)
        
        if vectorstore.col:
            milvus_data = vectorstore.col.query(expr="pk >= 0", output_fields=["pk", "source", "content_metadata"])
            filepaths_added = set()
            documents_list = list()
            
            for item in milvus_data:
                if extract_filename(item) not in filepaths_added:
                    metadata_dict = {}
                    for metadata_item in metadata_schema:
                        metadata_name = metadata_item.get("name")
                        metadata_value = item.get("content_metadata", {}).get(metadata_name, None)
                        metadata_dict[metadata_name] = metadata_value
                    documents_list.append({
                        "document_name": extract_filename(item),
                        "metadata": metadata_dict
                    })
                filepaths_added.add(extract_filename(item))
            
            return documents_list
    except Exception as e:
        logger.error("Error occurred while retrieving documents from Milvus: %s", e)
    return []


def _get_docs_cyborgdb(vectorstore: VectorStore, collection_name: str, vdb_endpoint: str, settings) -> List[Dict[str, Any]]:
    """Get documents from CyborgDB."""
    # CyborgDB doesn't have a direct way to list all documents
    # This is a limitation of the current implementation
    logger.warning("Listing all documents is not directly supported in CyborgDB")
    return []


def del_docs_vectorstore_langchain(vectorstore: VectorStore, filenames: List[str], collection_name: str="", include_upload_path: bool = False) -> bool:
    """Delete documents from the vector index implemented in LangChain."""
    
    settings = get_config()
    
    if settings.vector_store.name == "milvus":
        return _del_docs_milvus(vectorstore, filenames, collection_name, include_upload_path, settings)
    elif settings.vector_store.name == "cyborgdb":
        return _del_docs_cyborgdb(vectorstore, filenames, collection_name, include_upload_path, settings)
    else:
        logger.error(f"Unsupported vector store: {settings.vector_store.name}")
        return False


def _del_docs_milvus(vectorstore: VectorStore, filenames: List[str], collection_name: str, include_upload_path: bool, settings) -> bool:
    """Delete documents from Milvus."""
    if include_upload_path:
        upload_folder = str(Path(os.path.join(settings.temp_dir, f"uploaded_files/{collection_name}")))
    else:
        upload_folder = ""
    
    deleted = False
    try:
        for filename in filenames:
            source_value = os.path.join(upload_folder, filename)
            logger.info(f"Deleting document {source_value} from collection {collection_name} at {settings.vector_store.url}")
            try:
                resp = vectorstore.col.delete(f"source['source_name'] == '{source_value}'")
            except MilvusException as e:
                logger.debug(f"Failed to delete document {source_value}, source name might be available in the source field")
                resp = vectorstore.col.delete(f"source == '{source_value}'")
            deleted = True
            if resp.delete_count == 0:
                logger.info("File does not exist in the vectorstore")
                return False
        if deleted:
            vectorstore.col.flush()
        return True
    except Exception as e:
        logger.error("Error occurred while deleting documents: %s", e)
        return False


def _del_docs_cyborgdb(vectorstore: VectorStore, filenames: List[str], collection_name: str, include_upload_path: bool, settings) -> bool:
    """Delete documents from CyborgDB."""
    # CyborgDB doesn't support deletion by filename metadata
    # You would need to track document IDs separately
    logger.warning("Document deletion by filename is not directly supported in CyborgDB. You need to track document IDs.")
    return False


# Keep existing helper functions for Milvus metadata management
def create_metadata_collection_schema():
    """Create metadata collection for the collection."""
    schema = MilvusClient.create_schema(auto_id=True, enable_dynamic_field=True)
    schema.add_field(field_name="pk", datatype=DataType.INT64, is_primary=True, auto_id=True)
    schema.add_field(field_name="collection_name", datatype=DataType.VARCHAR, max_length=65535)
    schema.add_field(field_name="vector", datatype=DataType.FLOAT_VECTOR, dim=2)
    schema.add_field(field_name="metadata_schema", datatype=DataType.JSON)
    return schema


def create_metadata_schema_collection(vdb_endpoint: str) -> None:
    """Create metadata schema collection for the collection."""
    config = get_config()
    if config.vector_store.name != "milvus":
        # Only Milvus supports metadata schema collections
        return
    
    try:
        url = urlparse(vdb_endpoint)
        connection_alias = f"milvus_{url.hostname}_{url.port}"
        connections.connect(connection_alias, host=url.hostname, port=url.port)
        
        client = MilvusClient(vdb_endpoint)
        
        # Check if the metadata schema collection exists
        if not client.has_collection(DEFAULT_METADATA_SCHEMA_COLLECTION):
            # Create the metadata schema collection
            schema = create_metadata_collection_schema()
            index_params = MilvusClient.prepare_index_params()
            index_params.add_index(
                field_name="vector",
                index_name="dense_index",
                index_type="FLAT",
                metric_type="L2",
            )
            client.create_collection(
                collection_name=DEFAULT_METADATA_SCHEMA_COLLECTION,
                schema=schema,
                index_params=index_params,
                consistency_level=CONSISTENCY_STRONG
            )
            logger.info(f"Metadata schema collection created at {vdb_endpoint}")
        
        connections.disconnect(connection_alias)
    except Exception as e:
        logger.error(f"Failed to create metadata schema collection: {str(e)}")
        raise Exception(f"Failed to create metadata schema collection: {str(e)}")


# Keep remaining functions as they are (they're Milvus-specific)
def add_metadata_schema(collection_name: str, vdb_endpoint: str, metadata_schema: List[Dict[str, str]]) -> None:
    """Add metadata schema to the collection (Milvus only)."""
    config = get_config()
    if config.vector_store.name != "milvus":
        return
    
    try:
        url = urlparse(vdb_endpoint)
        connection_alias = f"milvus_{url.hostname}_{url.port}"
        connections.connect(connection_alias, host=url.hostname, port=url.port)
        
        client = MilvusClient(vdb_endpoint)
        
        # Delete the metadata schema from the collection
        client.delete(collection_name=DEFAULT_METADATA_SCHEMA_COLLECTION, filter=f"collection_name == '{collection_name}'")
        
        # Add the metadata schema to the collection
        data = {
            "collection_name": collection_name,
            "vector": [0.0] * 2,
            "metadata_schema": metadata_schema
        }
        client.insert(collection_name=DEFAULT_METADATA_SCHEMA_COLLECTION, data=data)
        logger.info(f"Metadata schema added to the collection {collection_name}. Metadata schema: {metadata_schema}")
        
        connections.disconnect(connection_alias)
    except Exception as e:
        logger.error(f"Failed to add metadata schema to the collection {collection_name}: {str(e)}")
        raise Exception(f"Failed to add metadata schema to the collection {collection_name}: {str(e)}")


def get_milvus_entities(collection_name: str, vdb_endpoint: str, filter: str) -> List[Dict[str, Any]]:
    """Get a milvus entity from the collection."""
    config = get_config()
    if config.vector_store.name != "milvus":
        return []
    
    try:
        logger.info(f"Getting milvus entity from the collection {DEFAULT_METADATA_SCHEMA_COLLECTION} at {vdb_endpoint} with filter {filter}")
        url = urlparse(vdb_endpoint)
        connection_alias = f"milvus_{url.hostname}_{url.port}"
        connections.connect(connection_alias, host=url.hostname, port=url.port)
        
        client = MilvusClient(vdb_endpoint)
        entities = client.query(collection_name=collection_name, filter=filter, limit=1000)
        
        if len(entities) == 0:
            raise Exception(f"No metadata schema found for filter {filter}")
        
        connections.disconnect(connection_alias)
        
        return entities
    except Exception as e:
        logging_message = f"Unable to get milvus entity from the collection {collection_name} for filter {filter}. Error: {str(e)}"
        if collection_name == DEFAULT_METADATA_SCHEMA_COLLECTION:
            logger.debug(f"Checking if {DEFAULT_METADATA_SCHEMA_COLLECTION} collection exists at {vdb_endpoint} and creating it if it does not exist")
            create_metadata_schema_collection(vdb_endpoint)
        logger.debug(logging_message)
        return []


def get_metadata_schema(collection_name: str, vdb_endpoint: str) -> List[Dict[str, Any]]:
    """Get the metadata schema from the collection (Milvus only)."""
    config = get_config()
    if config.vector_store.name != "milvus":
        return []
    
    try:
        filter = f"collection_name == '{collection_name}'"
        entities = get_milvus_entities(DEFAULT_METADATA_SCHEMA_COLLECTION, vdb_endpoint, filter)
        if len(entities) > 0:
            return entities[0]["metadata_schema"]
        else:
            logging_message = f"No metadata schema found for the collection: {collection_name}. Possible reason: The collection is not created with metadata schema."
            logger.info(logging_message)
            return []
    except Exception as e:
        logging_message = f"Unable to get metadata schema for the collection: {collection_name}. Error: {str(e)}"
        logger.error(logging_message)
        return []


def delete_entities(collection_name: str, vdb_endpoint: str, filter: str) -> None:
    """Delete an entity from the collection (Milvus only)."""
    config = get_config()
    if config.vector_store.name != "milvus":
        return
    
    try:
        logger.info(f"Deleting entity from the collection {collection_name} at {vdb_endpoint} with filter {filter}")
        url = urlparse(vdb_endpoint)
        connection_alias = f"milvus_{url.hostname}_{url.port}"
        connections.connect(connection_alias, host=url.hostname, port=url.port)
        
        client = MilvusClient(vdb_endpoint)
        if client.has_collection(collection_name):
            client.delete(collection_name=collection_name, filter=filter)
        else:
            logger.warning(f"Collection {collection_name} does not exist. Skipping deletion for filter {filter}")
        
        connections.disconnect(connection_alias)
    except Exception as e:
        logger.exception(f"Failed to delete entity from the collection {collection_name} with filter {filter}: {str(e)}")
        raise Exception(f"Failed to delete entity from the collection {collection_name} with filter {filter}: {str(e)}")


def retreive_docs_from_retriever(retriever, retriever_query: str, expr: str, otel_ctx: otel_context) -> List[Document]:
    """Retreive documents from the retriever."""
    token = otel_context.attach(otel_ctx)
    start_time = time.time()
    retriever_docs = []
    docs = []
    
    # Handle different retriever configurations based on vector store type
    config = get_config()
    
    if config.vector_store.name == "milvus":
        retriever_lambda = RunnableLambda(lambda x: retriever.invoke(x, expr=expr, consistency_level=CONFIG.vector_store.consistency_level))
    else:
        # CyborgDB doesn't use expr or consistency_level
        retriever_lambda = RunnableLambda(lambda x: retriever.invoke(x))
    
    retriever_chain = {"context": retriever_lambda} | RunnableAssign({"context": lambda input: input["context"]})
    retriever_docs = retriever_chain.invoke(retriever_query, config={'run_name':'retriever'})
    docs = retriever_docs.get("context", [])
    collection_name = retriever.vectorstore.collection_name if hasattr(retriever.vectorstore, 'collection_name') else retriever.vectorstore.index_name
    end_time = time.time()
    latency = end_time - start_time
    logger.info(f"Retriever latency: {latency:.4f} seconds")
    otel_context.detach(token)
    return add_collection_name_to_retreived_docs(docs, collection_name)


def add_collection_name_to_retreived_docs(docs: List[Document], collection_name: str) -> List[Document]:
    """Add the collection name to the retreived documents."""
    for doc in docs:
        doc.metadata["collection_name"] = collection_name
    return docs