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

"""Unit tests for CyborgDB VDB functionality."""

import unittest
from unittest.mock import MagicMock, Mock, patch, call
import pandas as pd
import pytest
from langchain_core.documents import Document

from nvidia_rag.utils.vdb.cyborgdb.cyborgdb_vdb import CyborgDBVDB

class TestCyborgDBVDB(unittest.TestCase):
    """Test cases for CyborgDBVDB class."""

    def setUp(self):
        """Set up test fixtures."""
        self.collection_name = "test_collection"
        self.cyborgdb_uri = "http://cyborgdb:8000"
        self.api_key = "test_api_key"
        self.index_key = b"test_index_key_bytes_32_chars_ok"
        self.meta_source_field = "source"
        self.meta_fields = ["field1"]
        self.embedding_model = "test_embedding_model"
        self.csv_file_path = "/path/to/test.csv"

    @patch('nvidia_rag.utils.vdb.cyborgdb.cyborgdb_vdb.get_config')
    @patch('nvidia_rag.utils.vdb.cyborgdb.cyborgdb_vdb.Client')
    def test_init(self, mock_client, mock_get_config):
        """Test CyborgDBVDB initialization."""
        # Mock config
        mock_config = Mock()
        mock_config.embeddings.dimensions = 768
        mock_get_config.return_value = mock_config
        
        # Mock CyborgDB Client
        mock_client_instance = Mock()
        mock_client.return_value = mock_client_instance
        
        # Create CyborgDBVDB instance
        cyborgdb_vdb = CyborgDBVDB(
            collection_name=self.collection_name,
            cyborgdb_uri=self.cyborgdb_uri,
            api_key=self.api_key,
            index_key=self.index_key,
            meta_source_field=self.meta_source_field,
            meta_fields=self.meta_fields,
            embedding_model=self.embedding_model,
            csv_file_path=self.csv_file_path
        )
        
        # Assertions
        self.assertEqual(cyborgdb_vdb.collection_name, self.collection_name)
        self.assertEqual(cyborgdb_vdb.vdb_endpoint, self.cyborgdb_uri)
        self.assertEqual(cyborgdb_vdb.api_key, self.api_key)
        self.assertEqual(cyborgdb_vdb.index_key, self.index_key)
        self.assertEqual(cyborgdb_vdb.embedding_model, self.embedding_model)
        self.assertEqual(cyborgdb_vdb.meta_source_field, self.meta_source_field)
        self.assertEqual(cyborgdb_vdb.meta_fields, self.meta_fields)
        self.assertEqual(cyborgdb_vdb.csv_file_path, self.csv_file_path)
        self.assertTrue(cyborgdb_vdb._connected)
        
        mock_client.assert_called_once_with(
            base_url=self.cyborgdb_uri,
            api_key=self.api_key,
            verify_ssl=False
        )

    @patch('nvidia_rag.utils.vdb.cyborgdb.cyborgdb_vdb.get_config')
    @patch('nvidia_rag.utils.vdb.cyborgdb.cyborgdb_vdb.Client')
    def test_check_index_exists(self, mock_client, mock_get_config):
        """Test _check_index_exists method."""
        # Setup mocks
        mock_config = Mock()
        mock_config.embeddings.dimensions = 768
        mock_get_config.return_value = mock_config
        
        mock_client_instance = Mock()
        mock_client.return_value = mock_client_instance
        mock_client_instance.list_indexes.return_value = ["test_index", "another_index"]
        
        # Create instance and test
        cyborgdb_vdb = CyborgDBVDB(
            collection_name=self.collection_name,
            cyborgdb_uri=self.cyborgdb_uri,
            api_key=self.api_key,
            index_key=self.index_key
        )
        result = cyborgdb_vdb._check_index_exists("test_index")
        
        self.assertTrue(result)
        mock_client_instance.list_indexes.assert_called_once()
        
        # Test non-existent index
        result = cyborgdb_vdb._check_index_exists("non_existent")
        self.assertFalse(result)

    @patch('nvidia_rag.utils.vdb.cyborgdb.cyborgdb_vdb.get_config')
    @patch('nvidia_rag.utils.vdb.cyborgdb.cyborgdb_vdb.Client')
    @patch('nvidia_rag.utils.vdb.cyborgdb.cyborgdb_vdb.CyborgVectorStore')
    @patch('nvidia_rag.utils.vdb.cyborgdb.cyborgdb_vdb.logger')
    def test_create_index(self, mock_logger, mock_vectorstore, mock_client, mock_get_config):
        """Test create_index method."""
        # Setup mocks
        mock_config = Mock()
        mock_config.embeddings.dimensions = 768
        mock_get_config.return_value = mock_config
        
        mock_client_instance = Mock()
        mock_client.return_value = mock_client_instance
        
        mock_vs_instance = Mock()
        mock_vectorstore.return_value = mock_vs_instance
        
        # Create instance and test
        cyborgdb_vdb = CyborgDBVDB(
            collection_name=self.collection_name,
            cyborgdb_uri=self.cyborgdb_uri,
            api_key=self.api_key,
            index_key=self.index_key
        )
        cyborgdb_vdb.create_index()
        
        mock_logger.info.assert_called_with(f"Creating CyborgDB index if not exists: {self.collection_name}")
        mock_vectorstore.assert_called_once()

    @patch('nvidia_rag.utils.vdb.cyborgdb.cyborgdb_vdb.get_config')
    @patch('nvidia_rag.utils.vdb.cyborgdb.cyborgdb_vdb.Client')
    @patch('nvidia_rag.utils.vdb.cyborgdb.cyborgdb_vdb.CyborgVectorStore')
    @patch('nvidia_rag.utils.vdb.cyborgdb.cyborgdb_vdb.logger')
    def test_write_to_index(self, mock_logger, mock_vectorstore, mock_client, mock_get_config):
        """Test write_to_index method."""
        # Setup mocks
        mock_config = Mock()
        mock_config.embeddings.dimensions = 768
        mock_get_config.return_value = mock_config
        
        mock_client_instance = Mock()
        mock_client.return_value = mock_client_instance
        
        mock_vs_instance = Mock()
        mock_vectorstore.return_value = mock_vs_instance
        mock_vs_instance.add_documents.return_value = ["id1", "id2"]
        
        # Test data
        records = [
            {
                "text": "test text 1",
                "vector": [0.1, 0.2, 0.3],
                "source": "doc1.pdf",
                "content_metadata": {"title": "Test Doc 1"}
            },
            {
                "text": "test text 2", 
                "vector": [0.4, 0.5, 0.6],
                "source": "doc2.pdf",
                "content_metadata": {"title": "Test Doc 2"}
            }
        ]
        
        # Create instance and test
        cyborgdb_vdb = CyborgDBVDB(
            collection_name=self.collection_name,
            cyborgdb_uri=self.cyborgdb_uri,
            api_key=self.api_key,
            index_key=self.index_key
        )
        cyborgdb_vdb.write_to_index(records)
        
        # Verify vectorstore was created/retrieved
        mock_vectorstore.assert_called()
        
        # Verify documents were added
        mock_vs_instance.add_documents.assert_called_once()
        added_docs = mock_vs_instance.add_documents.call_args[0][0]
        
        # Check the documents were properly converted
        self.assertEqual(len(added_docs), 2)
        self.assertEqual(added_docs[0].page_content, "test text 1")
        self.assertEqual(added_docs[1].page_content, "test text 2")
        
        # Check metadata was properly set
        self.assertEqual(added_docs[0].metadata["source"], "doc1.pdf")
        self.assertEqual(added_docs[0].metadata["content_metadata"], {"title": "Test Doc 1"})
        
        mock_logger.info.assert_any_call("Writing 2 records to CyborgDB index")
        mock_logger.info.assert_any_call("Successfully inserted 2 documents into CyborgDB")

    @patch('nvidia_rag.utils.vdb.cyborgdb.cyborgdb_vdb.get_config')
    @patch('nvidia_rag.utils.vdb.cyborgdb.cyborgdb_vdb.Client')
    def test_retrieval_not_implemented(self, mock_client, mock_get_config):
        """Test retrieval method raises NotImplementedError."""
        # Setup mocks
        mock_config = Mock()
        mock_config.embeddings.dimensions = 768
        mock_get_config.return_value = mock_config
        
        mock_client_instance = Mock()
        mock_client.return_value = mock_client_instance
        
        # Create instance and test
        cyborgdb_vdb = CyborgDBVDB(
            collection_name=self.collection_name,
            cyborgdb_uri=self.cyborgdb_uri,
            api_key=self.api_key,
            index_key=self.index_key
        )
        
        with self.assertRaises(NotImplementedError) as context:
            cyborgdb_vdb.retrieval(["query1", "query2"])
        
        self.assertIn("Direct retrieval method not implemented", str(context.exception))

    @patch('nvidia_rag.utils.vdb.cyborgdb.cyborgdb_vdb.get_config')
    @patch('nvidia_rag.utils.vdb.cyborgdb.cyborgdb_vdb.Client')
    def test_reindex_not_implemented(self, mock_client, mock_get_config):
        """Test reindex method raises NotImplementedError."""
        # Setup mocks
        mock_config = Mock()
        mock_config.embeddings.dimensions = 768
        mock_get_config.return_value = mock_config
        
        mock_client_instance = Mock()
        mock_client.return_value = mock_client_instance
        
        # Create instance and test
        cyborgdb_vdb = CyborgDBVDB(
            collection_name=self.collection_name,
            cyborgdb_uri=self.cyborgdb_uri,
            api_key=self.api_key,
            index_key=self.index_key
        )
        
        with self.assertRaises(NotImplementedError) as context:
            cyborgdb_vdb.reindex([{"record": "data"}])
        
        self.assertEqual(str(context.exception), "reindex must be implemented for CyborgDB")

    @patch('nvidia_rag.utils.vdb.cyborgdb.cyborgdb_vdb.get_config')
    @patch('nvidia_rag.utils.vdb.cyborgdb.cyborgdb_vdb.Client')
    def test_run(self, mock_client, mock_get_config):
        """Test run method."""
        # Setup mocks
        mock_config = Mock()
        mock_config.embeddings.dimensions = 768
        mock_get_config.return_value = mock_config
        
        mock_client_instance = Mock()
        mock_client.return_value = mock_client_instance
        
        # Create instance and test
        cyborgdb_vdb = CyborgDBVDB(
            collection_name=self.collection_name,
            cyborgdb_uri=self.cyborgdb_uri,
            api_key=self.api_key,
            index_key=self.index_key
        )
        
        # Mock the methods that run() calls
        cyborgdb_vdb.create_index = Mock()
        cyborgdb_vdb.write_to_index = Mock()
        
        records = [{"test": "data"}]
        cyborgdb_vdb.run(records)
        
        cyborgdb_vdb.create_index.assert_called_once()
        cyborgdb_vdb.write_to_index.assert_called_once_with(records)

    @patch('nvidia_rag.utils.vdb.cyborgdb.cyborgdb_vdb.get_config')
    @patch('nvidia_rag.utils.vdb.cyborgdb.cyborgdb_vdb.Client')
    @patch('nvidia_rag.utils.vdb.cyborgdb.cyborgdb_vdb.CyborgVectorStore')
    def test_create_collection(self, mock_vectorstore, mock_client, mock_get_config):
        """Test create_collection method."""
        # Setup mocks
        mock_config = Mock()
        mock_config.embeddings.dimensions = 768
        mock_get_config.return_value = mock_config
        
        mock_client_instance = Mock()
        mock_client.return_value = mock_client_instance
        
        mock_vs_instance = Mock()
        mock_vectorstore.return_value = mock_vs_instance
        
        # Create instance and test
        cyborgdb_vdb = CyborgDBVDB(
            collection_name=self.collection_name,
            cyborgdb_uri=self.cyborgdb_uri,
            api_key=self.api_key,
            index_key=self.index_key
        )
        
        cyborgdb_vdb.create_collection("test_collection", dimension=1024, collection_type="text")
        
        # Verify vectorstore was created with correct parameters
        mock_vectorstore.assert_called_with(
            index_name="test_collection",
            index_key=self.index_key,
            api_key=self.api_key,
            base_url=self.cyborgdb_uri,
            embedding=cyborgdb_vdb.embedding_model,
            dimension=1024
        )

    @patch('nvidia_rag.utils.vdb.cyborgdb.cyborgdb_vdb.get_config')
    @patch('nvidia_rag.utils.vdb.cyborgdb.cyborgdb_vdb.Client')
    def test_check_collection_exists(self, mock_client, mock_get_config):
        """Test check_collection_exists method."""
        # Setup mocks
        mock_config = Mock()
        mock_config.embeddings.dimensions = 768
        mock_get_config.return_value = mock_config
        
        mock_client_instance = Mock()
        mock_client.return_value = mock_client_instance
        mock_client_instance.list_indexes.return_value = ["test_collection", "other_collection"]
        
        # Create instance and test
        cyborgdb_vdb = CyborgDBVDB(
            collection_name=self.collection_name,
            cyborgdb_uri=self.cyborgdb_uri,
            api_key=self.api_key,
            index_key=self.index_key
        )
        result = cyborgdb_vdb.check_collection_exists("test_collection")
        
        self.assertTrue(result)
        mock_client_instance.list_indexes.assert_called()

    @patch('nvidia_rag.utils.vdb.cyborgdb.cyborgdb_vdb.get_config')
    @patch('nvidia_rag.utils.vdb.cyborgdb.cyborgdb_vdb.Client')
    def test_get_collection(self, mock_client, mock_get_config):
        """Test get_collection method."""
        # Setup mocks
        mock_config = Mock()
        mock_config.embeddings.dimensions = 768
        mock_get_config.return_value = mock_config
        
        mock_client_instance = Mock()
        mock_client.return_value = mock_client_instance
        mock_client_instance.list_indexes.return_value = ["test_index_1", "test_index_2"]
        
        # Create instance and test
        cyborgdb_vdb = CyborgDBVDB(
            collection_name=self.collection_name,
            cyborgdb_uri=self.cyborgdb_uri,
            api_key=self.api_key,
            index_key=self.index_key
        )
        
        result = cyborgdb_vdb.get_collection()
        
        expected_result = [
            {
                "collection_name": "test_index_1",
                "num_entities": -1,  # CyborgDB doesn't provide count
                "metadata_schema": []  # CyborgDB doesn't have metadata schemas
            },
            {
                "collection_name": "test_index_2", 
                "num_entities": -1,
                "metadata_schema": []
            }
        ]
        
        self.assertEqual(result, expected_result)
        mock_client_instance.list_indexes.assert_called_once()

    @patch('nvidia_rag.utils.vdb.cyborgdb.cyborgdb_vdb.get_config')
    @patch('nvidia_rag.utils.vdb.cyborgdb.cyborgdb_vdb.Client')
    @patch('nvidia_rag.utils.vdb.cyborgdb.cyborgdb_vdb.EncryptedIndex')
    @patch('nvidia_rag.utils.vdb.cyborgdb.cyborgdb_vdb.logger')
    def test_delete_collections(self, mock_logger, mock_encrypted_index, mock_client, mock_get_config):
        """Test delete_collections method with logging."""
        # Setup mocks
        mock_config = Mock()
        mock_config.embeddings.dimensions = 768
        mock_get_config.return_value = mock_config
        
        mock_client_instance = Mock()
        mock_client.return_value = mock_client_instance
        mock_client_instance.list_indexes.return_value = ["collection1", "collection2"]
        
        # Mock EncryptedIndex instances
        mock_index1 = Mock()
        mock_index1.index_name = "collection1"
        mock_index1.delete_index.return_value = True
        
        mock_index2 = Mock()
        mock_index2.index_name = "collection2"
        mock_index2.delete_index.return_value = True
        
        mock_encrypted_index.side_effect = [mock_index1, mock_index2]
        
        # Create instance and test
        cyborgdb_vdb = CyborgDBVDB(
            collection_name=self.collection_name,
            cyborgdb_uri=self.cyborgdb_uri,
            api_key=self.api_key,
            index_key=self.index_key
        )
        collection_names = ["collection1", "collection2"]
        
        result = cyborgdb_vdb.delete_collections(collection_names)
        
        expected_result = {
            "message": "Collection deletion process completed.",
            "successful": collection_names,
            "failed": [],
            "total_success": 2,
            "total_failed": 0
        }
        
        self.assertEqual(result, expected_result)
        
        # Verify logging was called
        mock_logger.info.assert_any_call("=" * 80)
        mock_logger.info.assert_any_call(f"DELETE_COLLECTIONS called with: {collection_names}")
        mock_logger.info.assert_any_call(f"Number of collections to delete: {len(collection_names)}")
        
        # Verify delete_index was called on each index
        mock_index1.delete_index.assert_called_once()
        mock_index2.delete_index.assert_called_once()

    @patch('nvidia_rag.utils.vdb.cyborgdb.cyborgdb_vdb.get_config')
    @patch('nvidia_rag.utils.vdb.cyborgdb.cyborgdb_vdb.Client')
    @patch('nvidia_rag.utils.vdb.cyborgdb.cyborgdb_vdb.CyborgVectorStore')
    def test_get_documents(self, mock_vectorstore, mock_client, mock_get_config):
        """Test get_documents method."""
        # Setup mocks
        mock_config = Mock()
        mock_config.embeddings.dimensions = 768
        mock_get_config.return_value = mock_config
        
        mock_client_instance = Mock()
        mock_client.return_value = mock_client_instance
        
        # Mock vectorstore
        mock_vs_instance = Mock()
        mock_vectorstore.return_value = mock_vs_instance
        
        # Mock document retrieval
        mock_doc1 = Mock()
        mock_doc1.metadata = {
            "source": "/path/to/doc1.pdf",
            "title": "Document 1",
            "author": "Author 1"
        }
        
        mock_doc2 = Mock()
        mock_doc2.metadata = {
            "source": "/path/to/doc2.pdf",
            "title": "Document 2"
        }
        
        mock_vs_instance.list_ids.return_value = ["id1", "id2"]
        mock_vs_instance.get.return_value = [mock_doc1, mock_doc2]
        
        # Create instance and test
        cyborgdb_vdb = CyborgDBVDB(
            collection_name=self.collection_name,
            cyborgdb_uri=self.cyborgdb_uri,
            api_key=self.api_key,
            index_key=self.index_key
        )
        
        result = cyborgdb_vdb.get_documents("test_collection")
        
        expected_result = [
            {
                "document_name": "doc1.pdf",
                "metadata": {
                    "source": "/path/to/doc1.pdf",
                    "title": "Document 1",
                    "author": "Author 1"
                }
            },
            {
                "document_name": "doc2.pdf", 
                "metadata": {
                    "source": "/path/to/doc2.pdf",
                    "title": "Document 2"
                }
            }
        ]
        
        self.assertEqual(result, expected_result)
        mock_vs_instance.list_ids.assert_called_once()
        mock_vs_instance.get.assert_called_once_with(ids=["id1", "id2"])

    @patch('nvidia_rag.utils.vdb.cyborgdb.cyborgdb_vdb.get_config')
    @patch('nvidia_rag.utils.vdb.cyborgdb.cyborgdb_vdb.Client')
    @patch('nvidia_rag.utils.vdb.cyborgdb.cyborgdb_vdb.CyborgVectorStore')
    @patch('nvidia_rag.utils.vdb.cyborgdb.cyborgdb_vdb.logger')
    def test_delete_documents(self, mock_logger, mock_vectorstore, mock_client, mock_get_config):
        """Test delete_documents method."""
        # Setup mocks
        mock_config = Mock()
        mock_config.embeddings.dimensions = 768
        mock_get_config.return_value = mock_config
        
        mock_client_instance = Mock()
        mock_client.return_value = mock_client_instance
        
        mock_vs_instance = Mock()
        mock_vectorstore.return_value = mock_vs_instance
        
        # Create instance and test
        cyborgdb_vdb = CyborgDBVDB(
            collection_name=self.collection_name,
            cyborgdb_uri=self.cyborgdb_uri,
            api_key=self.api_key,
            index_key=self.index_key
        )
        source_values = ["doc1.pdf", "doc2.pdf"]
        
        # CyborgDB doesn't support deletion by source values
        result = cyborgdb_vdb.delete_documents("test_collection", source_values)
        
        self.assertFalse(result)
        mock_logger.warning.assert_called_once()

    @patch('nvidia_rag.utils.vdb.cyborgdb.cyborgdb_vdb.get_config')
    @patch('nvidia_rag.utils.vdb.cyborgdb.cyborgdb_vdb.Client')
    @patch('nvidia_rag.utils.vdb.cyborgdb.cyborgdb_vdb.EncryptedIndex')
    @patch('nvidia_rag.utils.vdb.cyborgdb.cyborgdb_vdb.logger')
    def test_delete_collections_with_non_existent(self, mock_logger, mock_encrypted_index, mock_client, mock_get_config):
        """Test delete_collections with non-existent collections."""
        # Setup mocks
        mock_config = Mock()
        mock_config.embeddings.dimensions = 768
        mock_get_config.return_value = mock_config
        
        mock_client_instance = Mock()
        mock_client.return_value = mock_client_instance
        # Only collection1 exists
        mock_client_instance.list_indexes.return_value = ["collection1"]
        
        # Mock EncryptedIndex for existing collection
        mock_index1 = Mock()
        mock_index1.index_name = "collection1"
        mock_index1.delete_index.return_value = True
        
        mock_encrypted_index.return_value = mock_index1
        
        # Create instance and test
        cyborgdb_vdb = CyborgDBVDB(
            collection_name=self.collection_name,
            cyborgdb_uri=self.cyborgdb_uri,
            api_key=self.api_key,
            index_key=self.index_key
        )
        collection_names = ["collection1", "non_existent"]
        
        result = cyborgdb_vdb.delete_collections(collection_names)
        
        # Should only delete collection1
        self.assertEqual(result["successful"], ["collection1"])
        self.assertEqual(len(result["failed"]), 1)
        self.assertEqual(result["failed"][0]["collection_name"], "non_existent")
        self.assertEqual(result["failed"][0]["error_message"], "Collection not found")
        self.assertEqual(result["total_success"], 1)
        self.assertEqual(result["total_failed"], 1)
        
        # Verify logging for non-existent collection
        mock_logger.warning.assert_any_call("Collection 'non_existent' NOT FOUND, skipping deletion")
        
        # Verify delete_index was only called once
        mock_index1.delete_index.assert_called_once()

    @patch('nvidia_rag.utils.vdb.cyborgdb.cyborgdb_vdb.get_config')
    @patch('nvidia_rag.utils.vdb.cyborgdb.cyborgdb_vdb.Client')
    @patch('nvidia_rag.utils.vdb.cyborgdb.cyborgdb_vdb.EncryptedIndex')
    @patch('nvidia_rag.utils.vdb.cyborgdb.cyborgdb_vdb.logger')
    def test_delete_collections_with_cached_indexes(self, mock_logger, mock_encrypted_index, mock_client, mock_get_config):
        """Test delete_collections with cached indexes."""
        # Setup mocks
        mock_config = Mock()
        mock_config.embeddings.dimensions = 768
        mock_get_config.return_value = mock_config
        
        mock_client_instance = Mock()
        mock_client.return_value = mock_client_instance
        mock_client_instance.list_indexes.return_value = ["cached_collection"]
        
        # Create instance
        cyborgdb_vdb = CyborgDBVDB(
            collection_name=self.collection_name,
            cyborgdb_uri=self.cyborgdb_uri,
            api_key=self.api_key,
            index_key=self.index_key
        )
        
        # Pre-populate cache with an index
        cached_index = Mock()
        cached_index.index_name = "cached_collection"
        cached_index.delete_index.return_value = True
        cyborgdb_vdb._indexes["cached_collection"] = cached_index
        
        # Also cache a vectorstore
        cached_vectorstore = Mock()
        cyborgdb_vdb._vectorstores["cached_collection"] = cached_vectorstore
        
        result = cyborgdb_vdb.delete_collections(["cached_collection"])
        
        # Verify successful deletion
        self.assertEqual(result["successful"], ["cached_collection"])
        self.assertEqual(result["failed"], [])
        self.assertEqual(result["total_success"], 1)
        self.assertEqual(result["total_failed"], 0)
        
        # Verify cache was cleared
        self.assertNotIn("cached_collection", cyborgdb_vdb._indexes)
        self.assertNotIn("cached_collection", cyborgdb_vdb._vectorstores)
        
        # Verify logging for cached index
        mock_logger.info.assert_any_call("Using CACHED index for 'cached_collection'")
        mock_logger.info.assert_any_call("Removed 'cached_collection' from vectorstore cache")
        mock_logger.info.assert_any_call("Removed 'cached_collection' from index cache")
        
        # Verify delete was called on cached index
        cached_index.delete_index.assert_called_once()
        
        # EncryptedIndex constructor should not be called for cached index
        mock_encrypted_index.assert_not_called()

    @patch('nvidia_rag.utils.vdb.cyborgdb.cyborgdb_vdb.get_config')
    @patch('nvidia_rag.utils.vdb.cyborgdb.cyborgdb_vdb.Client')
    @patch('nvidia_rag.utils.vdb.cyborgdb.cyborgdb_vdb.logger')
    def test_create_metadata_schema_collection(self, mock_logger, mock_client, mock_get_config):
        """Test create_metadata_schema_collection method (no-op for CyborgDB)."""
        # Setup mocks
        mock_config = Mock()
        mock_config.embeddings.dimensions = 768
        mock_get_config.return_value = mock_config
        
        mock_client_instance = Mock()
        mock_client.return_value = mock_client_instance
        
        # Create instance and test
        cyborgdb_vdb = CyborgDBVDB(
            collection_name=self.collection_name,
            cyborgdb_uri=self.cyborgdb_uri,
            api_key=self.api_key,
            index_key=self.index_key
        )
        
        # This should be a no-op for CyborgDB
        cyborgdb_vdb.create_metadata_schema_collection()
        
        # Should just log debug message
        mock_logger.debug.assert_called_once_with("CyborgDB doesn't require separate metadata schema collections")

    @patch('nvidia_rag.utils.vdb.cyborgdb.cyborgdb_vdb.get_config')
    @patch('nvidia_rag.utils.vdb.cyborgdb.cyborgdb_vdb.Client')
    def test_add_metadata_schema(self, mock_client, mock_get_config):
        """Test add_metadata_schema method (no-op for CyborgDB)."""
        # Setup mocks
        mock_config = Mock()
        mock_config.embeddings.dimensions = 768
        mock_get_config.return_value = mock_config
        
        mock_client_instance = Mock()
        mock_client.return_value = mock_client_instance
        
        # Create instance and test
        cyborgdb_vdb = CyborgDBVDB(
            collection_name=self.collection_name,
            cyborgdb_uri=self.cyborgdb_uri,
            api_key=self.api_key,
            index_key=self.index_key
        )
        
        metadata_schema = [{"name": "title", "type": "string"}]
        
        # This should be a no-op for CyborgDB
        cyborgdb_vdb.add_metadata_schema("test_collection", metadata_schema)
        
        # Nothing should be called since it's a no-op

    @patch('nvidia_rag.utils.vdb.cyborgdb.cyborgdb_vdb.get_config')
    @patch('nvidia_rag.utils.vdb.cyborgdb.cyborgdb_vdb.Client')
    def test_get_metadata_schema(self, mock_client, mock_get_config):
        """Test get_metadata_schema method (returns empty for CyborgDB)."""
        # Setup mocks
        mock_config = Mock()
        mock_config.embeddings.dimensions = 768
        mock_get_config.return_value = mock_config
        
        mock_client_instance = Mock()
        mock_client.return_value = mock_client_instance
        
        # Create instance and test
        cyborgdb_vdb = CyborgDBVDB(
            collection_name=self.collection_name,
            cyborgdb_uri=self.cyborgdb_uri,
            api_key=self.api_key,
            index_key=self.index_key
        )
        
        # Should always return empty list for CyborgDB
        result = cyborgdb_vdb.get_metadata_schema("test_collection")
        
        self.assertEqual(result, [])


    @patch('nvidia_rag.utils.vdb.cyborgdb.cyborgdb_vdb.get_config')
    @patch('nvidia_rag.utils.vdb.cyborgdb.cyborgdb_vdb.Client')
    @patch('nvidia_rag.utils.vdb.cyborgdb.cyborgdb_vdb.CyborgVectorStore')
    @patch('nvidia_rag.utils.vdb.cyborgdb.cyborgdb_vdb.time')
    @patch('nvidia_rag.utils.vdb.cyborgdb.cyborgdb_vdb.logger')
    def test_retrieval_langchain(self, mock_logger, mock_time, mock_vectorstore, mock_client, mock_get_config):
        """Test retrieval_langchain method."""
        # Setup mocks
        mock_config = Mock()
        mock_config.embeddings.dimensions = 768
        mock_get_config.return_value = mock_config
        
        mock_client_instance = Mock()
        mock_client.return_value = mock_client_instance
        
        # Mock time
        mock_time.time.side_effect = [1000.0, 1002.5]  # 2.5 second latency
        
        # Mock vectorstore and retriever
        mock_vs_instance = Mock()
        mock_vectorstore.return_value = mock_vs_instance
        mock_vs_instance.collection_name = "test_collection"
        
        mock_retriever = Mock()
        mock_retriever.vectorstore = mock_vs_instance
        mock_vs_instance.as_retriever.return_value = mock_retriever
        
        # Mock documents returned by retriever
        mock_docs = [
            Document(page_content="doc1", metadata={"source": "file1.pdf"}),
            Document(page_content="doc2", metadata={"source": "file2.pdf"})
        ]
        mock_retriever.invoke.return_value = mock_docs
        
        # Create instance and test
        cyborgdb_vdb = CyborgDBVDB(
            collection_name=self.collection_name,
            cyborgdb_uri=self.cyborgdb_uri,
            api_key=self.api_key,
            index_key=self.index_key,
            embedding_model="test_model"
        )
        
        result = cyborgdb_vdb.retrieval_langchain(
            query="test query",
            collection_name="test_collection",
            top_k=5,
            filter_expr={"field": "value"}
        )
        
        # Verify results have collection_name added
        for doc in result:
            self.assertEqual(doc.metadata["collection_name"], "test_collection")
        
        # Update assertion to match new logging format with document count
        calls = [str(call) for call in mock_logger.info.call_args_list]
        found_latency_log = False
        for call in calls:
            if "CyborgDB Retrieval latency" in call and "2.5000 seconds" in call:
                found_latency_log = True
                break
        self.assertTrue(found_latency_log, f"Expected latency log not found. Calls: {calls}")
        mock_vs_instance.as_retriever.assert_called_once_with(search_kwargs={"k": 5})



if __name__ == '__main__':
    unittest.main()

