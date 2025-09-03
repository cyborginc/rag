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

"""CyborgDB metadata operator Module to store metadata - replacement for MinIO.
1. CyborgMetadataOperator: Class to store metadata using CyborgDB.
2. get_cyborg_metadata_operator: Get the CyborgMetadataOperator object.
3. Helper functions for thumbnail ID generation (same as MinIO).

Configuration:
Add to your config file (variables.env or config.yaml):
    CYBORG_API_URL=http://localhost:8000
    CYBORG_API_KEY=your_api_key_here
    CYBORG_METADATA_INDEX_KEY=your_32_byte_hex_key_here

Or in config object:
    cyborg_db:
        api_url: http://localhost:8000
        api_key: your_api_key_here
        metadata_index_key: your_32_byte_hex_key_here
"""

import os
import json
import logging
from typing import Dict, List

try:
    from cyborgdb import Client, EncryptedIndex, IndexIVFFlat, generate_key
except ImportError:
    logging.error("CyborgDB not installed. Please install with: pip install cyborgdb")
    raise

from nvidia_rag.utils.common import get_config

logger = logging.getLogger(__name__)
CONFIG = get_config()

# Global instance to reuse across calls
_CYBORG_METADATA_CLIENT = None
_CYBORG_METADATA_INDEX = None

class CyborgMetadataOperator:
    """CyborgDB operator Class to store metadata - MinIO replacement"""

    def __init__(
        self,
        api_url: str = None,
        api_key: str = None,
        index_key: bytes = None,
        index_name: str = "metadata_store"
    ):
        """Initialize CyborgDB metadata operator.
        
        Args:
            api_url: CyborgDB API URL (defaults to config)
            api_key: CyborgDB API key (defaults to config)
            index_key: 32-byte encryption key (generates if not provided)
            index_name: Name of the index to use for metadata storage
        """
        global _CYBORG_METADATA_CLIENT, _CYBORG_METADATA_INDEX
        
        # Use existing instances if available
        if _CYBORG_METADATA_CLIENT and _CYBORG_METADATA_INDEX:
            self.client = _CYBORG_METADATA_CLIENT
            self.index = _CYBORG_METADATA_INDEX
            self.index_name = index_name
            logger.info(f"Reusing existing CyborgDB metadata client and index: {index_name}")
            return
            
        # Get configuration
        api_url = api_url or getattr(CONFIG.cyborg_db, 'api_url', 'http://localhost:8000')
        api_key = api_key or getattr(CONFIG.cyborg_db, 'api_key', None)
        
        # Generate or use provided index key
        if index_key is None:
            # Try to get from config or generate new one
            index_key_str = getattr(CONFIG.cyborg_db, 'index_key', None)
            if index_key_str:
                # Convert hex string to bytes if needed
                try:
                    index_key = bytes.fromhex(index_key_str)
                except:
                    index_key = index_key_str.encode() if isinstance(index_key_str, str) else index_key_str
            else:
                index_key = generate_key()
                logger.warning(f"Generated new index key for metadata store. Save this for future use: {index_key.hex()}")
        
        # Ensure key is 32 bytes
        if len(index_key) != 32:
            # Pad or truncate to 32 bytes
            if len(index_key) < 32:
                index_key = index_key + b'\0' * (32 - len(index_key))
            else:
                index_key = index_key[:32]
        
        self.index_name = index_name
        self.index_key = index_key
        
        try:
            # Initialize CyborgDB client
            self.client = Client(
                api_url=api_url,
                api_key=api_key,
                verify_ssl=False
            )
            
            # Check if index exists
            existing_indexes = self.client.list_indexes()
            
            if index_name in existing_indexes:
                # Load existing index
                logger.info(f"Loading existing CyborgDB metadata index: {index_name}")
                self.index = EncryptedIndex(
                    index_name=index_name,
                    index_key=self.index_key,
                    api=self.client.api,
                    api_client=self.client.api_client
                )
            else:
                # Create minimal index for metadata storage
                # Using dimension=1 with a dummy vector since we only need metadata storage
                logger.info(f"Creating new CyborgDB metadata index: {index_name}")
                self.index = self.client.create_index(
                    index_name=index_name,
                    index_key=self.index_key,
                    index_config=IndexIVFFlat(
                        dimension=1,  # Minimal dimension for dummy vectors
                        n_lists=1,    # Single list since we don't need vector search
                        metric="euclidean"
                    )
                )
            
            # Cache the instances
            _CYBORG_METADATA_CLIENT = self.client
            _CYBORG_METADATA_INDEX = self.index
            
            logger.info(f"CyborgDB metadata operator initialized with index: {index_name}")
            
        except Exception as e:
            logger.error(f"Failed to initialize CyborgDB metadata operator: {e}")
            raise

    def put_payload(
        self,
        payload: dict,
        object_name: str
    ):
        """Put dictionary to CyborgDB storage (MinIO replacement)"""
        try:
            # Create a dummy vector (single dimension) - using list instead of numpy
            dummy_vector = [[1.0]]
            
            # Convert payload to JSON string for metadata
            json_data = json.dumps(payload)
            
            # Upsert to CyborgDB with metadata
            self.index.upsert(
                ids=[object_name],
                vectors=dummy_vector,
                json_metadata_array=[json_data]
            )
            
            logger.debug(f"Successfully stored payload for: {object_name}")
            
        except Exception as e:
            logger.error(f"Failed to store payload in CyborgDB: {e}")
            raise

    def put_payloads_bulk(
        self,
        payloads: List[dict],
        object_names: List[str]
    ):
        """Put list of dictionaries to CyborgDB storage (bulk operation)"""
        try:
            # Create dummy vectors for all items - using list comprehension instead of numpy
            dummy_vectors = [[1.0] for _ in range(len(payloads))]
            
            # Convert all payloads to JSON strings
            json_metadata = [json.dumps(payload) for payload in payloads]
            
            # Bulk upsert to CyborgDB
            self.index.upsert(
                ids=object_names,
                vectors=dummy_vectors,
                json_metadata_array=json_metadata
            )
            
            logger.debug(f"Successfully stored {len(payloads)} payloads in bulk")
            
        except Exception as e:
            logger.error(f"Failed to store bulk payloads in CyborgDB: {e}")
            raise

    def get_payload(
        self,
        object_name: str
    ) -> Dict:
        """Get dictionary from CyborgDB storage"""
        try:
            # Use the Get method to retrieve by ID
            results = self.index.get(
                ids=[object_name],
                include=["metadata"]
            )
            
            if results and len(results) > 0:
                # Parse the JSON metadata
                metadata_str = results[0].metadata
                if metadata_str:
                    return json.loads(metadata_str)
            
            logger.warning(f"No payload found for object: {object_name}")
            return {}
            
        except Exception as e:
            logger.warning(f"Error while getting object from CyborgDB! Object name: {object_name}")
            logger.debug(f"Error details: {e}")
            return {}

    def list_payloads(
        self,
        prefix: str = ""
    ) -> List[str]:
        """List payloads from CyborgDB storage with optional prefix filter"""
        try:
            # Get all IDs from the index
            all_ids = self.index.list_ids()
            
            if prefix:
                # Filter IDs by prefix
                filtered_ids = [id for id in all_ids if id.startswith(prefix)]
                return filtered_ids
            else:
                return all_ids
                
        except Exception as e:
            logger.error(f"Failed to list payloads from CyborgDB: {e}")
            return []

    def delete_payloads(
        self,
        object_names: List[str]
    ) -> None:
        """Delete payloads from CyborgDB storage"""
        try:
            if object_names:
                self.index.delete(ids=object_names)
                logger.debug(f"Successfully deleted {len(object_names)} payloads")
        except Exception as e:
            logger.error(f"Failed to delete payloads from CyborgDB: {e}")
            raise

def get_cyborg_metadata_operator():
    """
    Prepares and return CyborgMetadataOperator object (MinIO replacement)
    
    Returns:
        - metadata_operator: CyborgMetadataOperator
    """
    # Try to get CyborgDB config, fall back to MinIO config for compatibility
    if hasattr(CONFIG, 'cyborg_db'):
        api_url = getattr(CONFIG.cyborg_db, 'api_url', 'http://localhost:8000')
        api_key = getattr(CONFIG.cyborg_db, 'api_key', None)
        index_key_str = getattr(CONFIG.cyborg_db, 'metadata_index_key', None)
    else:
        # Fallback: use default CyborgDB settings
        api_url = os.getenv('CYBORG_API_URL', 'http://localhost:8000')
        api_key = os.getenv('CYBORG_API_KEY', None)
        index_key_str = os.getenv('CYBORG_METADATA_INDEX_KEY', None)
    
    # Convert index key from hex string if provided
    index_key = None
    if index_key_str:
        try:
            index_key = bytes.fromhex(index_key_str)
        except:
            logger.warning("Invalid index key format, generating new one")
    
    metadata_operator = CyborgMetadataOperator(
        api_url=api_url,
        api_key=api_key,
        index_key=index_key,
        index_name="rag_metadata_store"
    )
    return metadata_operator

# Keep the same helper functions for compatibility
def get_unique_thumbnail_id_collection_prefix(
        collection_name: str,
    ) -> str:
    """
    Prepares unique thumbnail id prefix based on input collection name
    Returns:
        - unique_thumbnail_id_prefix: str
    """
    prefix = f"{collection_name}_::"
    return prefix

def get_unique_thumbnail_id_file_name_prefix(
        collection_name: str,
        file_name: str,
    ) -> str:
    """
    Prepares unique thumbnail id prefix based on input collection name and file name
    Returns:
        - unique_thumbnail_id_prefix: str
    """
    collection_prefix = get_unique_thumbnail_id_collection_prefix(collection_name)
    prefix = f"{collection_prefix}_{file_name}_::"
    return prefix

def get_unique_thumbnail_id(
        collection_name: str,
        file_name: str,
        page_number: int,
        location: List[float] # Bbox information
    ) -> str:
    """
    Prepares unique thumbnail id based on input arguments
    Returns:
        - unique_thumbnail_id: str
    """
    # Round bbox values to reduce precision
    rounded_bbox = [round(coord, 4) for coord in location]
    prefix = get_unique_thumbnail_id_file_name_prefix(collection_name, file_name)
    # Create a string representation
    unique_thumbnail_id = f"{prefix}_{page_number}_" + \
                          "_".join(map(str, rounded_bbox))
    return unique_thumbnail_id

# Alias for backwards compatibility
MinioOperator = CyborgMetadataOperator
get_minio_operator = get_cyborg_metadata_operator