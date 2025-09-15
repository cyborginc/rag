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
    @patch('nvidia_rag.utils.vdb.cyborgdb.cyborgdb_vdb.CyborgVectorStore')
    @patch('nvidia_rag.utils.vdb.cyborgdb.cyborgdb_vdb.logger')
    def test_delete_collections(self, mock_logger, mock_vectorstore, mock_client, mock_get_config):
        """Test delete_collections method."""
        # Setup mocks
        mock_config = Mock()
        mock_config.embeddings.dimensions = 768
        mock_get_config.return_value = mock_config
        
        mock_client_instance = Mock()
        mock_client.return_value = mock_client_instance
        
        mock_vs_instance = Mock()
        mock_vectorstore.return_value = mock_vs_instance
        mock_vs_instance.delete.return_value = True
        
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
        # Verify delete was called with delete_index=True
        self.assertEqual(mock_vs_instance.delete.call_count, 2)
        mock_vs_instance.delete.assert_called_with(delete_index=True)

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
        
        mock_logger.info.assert_called_with(" CyborgDB Retrieval latency: 2.5000 seconds")
        mock_vs_instance.as_retriever.assert_called_once_with(search_kwargs={"k": 5})

    @patch('nvidia_rag.utils.vdb.elasticsearch.elastic_vdb.CONFIG')
    @patch('nvidia_rag.utils.vdb.elasticsearch.elastic_vdb.get_config')
    @patch('nvidia_rag.utils.vdb.elasticsearch.elastic_vdb.Elasticsearch')
    @patch('nvidia_rag.utils.vdb.elasticsearch.elastic_vdb.VectorStore')
    @patch('nvidia_rag.utils.vdb.elasticsearch.elastic_vdb.ElasticsearchStore')
    @patch('nvidia_rag.utils.vdb.elasticsearch.elastic_vdb.DenseVectorStrategy')
    def test_get_langchain_vectorstore(self, mock_dense_strategy, mock_es_store_class, mock_vector_store, mock_elasticsearch, mock_get_config, mock_config):
        """Test get_langchain_vectorstore method."""
        # Setup mocks
        mock_config_obj = Mock()
        mock_config_obj.embeddings.dimensions = 768
        mock_config_obj.vector_store.search_type = "hybrid"
        mock_get_config.return_value = mock_config_obj
        
        # Mock the global CONFIG object used in get_langchain_vectorstore
        mock_config.vector_store.search_type = "hybrid"
        
        mock_es_connection = Mock()
        mock_elasticsearch.return_value = mock_es_connection
        
        mock_vectorstore = Mock()
        mock_es_store_class.return_value = mock_vectorstore
        
        mock_strategy = Mock()
        mock_dense_strategy.return_value = mock_strategy
        
        # Create instance and test
        elastic_vdb = ElasticVDB(self.index_name, self.es_url, embedding_model="test_model")
        
        # Reset mock to only track calls from the method being tested
        mock_dense_strategy.reset_mock()
        
        result = elastic_vdb.get_langchain_vectorstore("test_collection")
        
        self.assertEqual(result, mock_vectorstore)
        
        mock_es_store_class.assert_called_once_with(
            index_name="test_collection",
            es_url=self.es_url,
            embedding="test_model",
            strategy=mock_strategy
        )
        
        # Now it should be called once with hybrid=True
        mock_dense_strategy.assert_called_once_with(hybrid=True)

    def test_add_collection_name_to_retreived_docs(self):
        """Test _add_collection_name_to_retreived_docs static method."""
        # Create test documents
        docs = [
            Document(page_content="doc1", metadata={"source": "file1.pdf"}),
            Document(page_content="doc2", metadata={"source": "file2.pdf"})
        ]
        
        # Test the static method
        result = ElasticVDB._add_collection_name_to_retreived_docs(docs, "test_collection")
        
        # Verify collection_name is added to metadata
        for doc in result:
            self.assertEqual(doc.metadata["collection_name"], "test_collection")
        
        # Verify original metadata is preserved
        self.assertEqual(result[0].metadata["source"], "file1.pdf")
        self.assertEqual(result[1].metadata["source"], "file2.pdf")


class TestEsQueries(unittest.TestCase):
    """Test cases for es_queries module functions."""

    def test_get_unique_sources_query(self):
        """Test get_unique_sources_query function returns correct aggregation query."""
        result = es_queries.get_unique_sources_query()
        
        # Verify the basic structure
        self.assertIn("size", result)
        self.assertEqual(result["size"], 0)
        self.assertIn("aggs", result)
        
        # Verify aggregation structure
        unique_sources = result["aggs"]["unique_sources"]
        self.assertIn("composite", unique_sources)
        self.assertIn("aggs", unique_sources)
        
        # Verify composite aggregation
        composite = unique_sources["composite"]
        self.assertEqual(composite["size"], 1000)
        self.assertIn("sources", composite)
        
        # Verify source field configuration
        sources = composite["sources"][0]
        self.assertIn("source_name", sources)
        terms = sources["source_name"]["terms"]
        self.assertEqual(terms["field"], "metadata.source.source_name.keyword")
        
        # Verify top_hits aggregation
        top_hit = unique_sources["aggs"]["top_hit"]
        self.assertIn("top_hits", top_hit)
        self.assertEqual(top_hit["top_hits"]["size"], 1)

    def test_get_delete_metadata_schema_query(self):
        """Test get_delete_metadata_schema_query function with collection name."""
        collection_name = "test_collection"
        result = es_queries.get_delete_metadata_schema_query(collection_name)
        
        # Verify query structure
        self.assertIn("query", result)
        self.assertIn("term", result["query"])
        
        # Verify term query
        term_query = result["query"]["term"]
        self.assertIn("collection_name.keyword", term_query)
        self.assertEqual(term_query["collection_name.keyword"], collection_name)

    def test_get_metadata_schema_query(self):
        """Test get_metadata_schema_query function with collection name."""
        collection_name = "test_collection"
        result = es_queries.get_metadata_schema_query(collection_name)
        
        # Verify query structure
        self.assertIn("query", result)
        self.assertIn("term", result["query"])
        
        # Verify term query
        term_query = result["query"]["term"]
        self.assertIn("collection_name", term_query)
        self.assertEqual(term_query["collection_name"], collection_name)

    def test_get_delete_docs_query(self):
        """Test get_delete_docs_query function with source value."""
        source_value = "test_document.pdf"
        result = es_queries.get_delete_docs_query(source_value)
        
        # Verify query structure
        self.assertIn("query", result)
        self.assertIn("term", result["query"])
        
        # Verify term query
        term_query = result["query"]["term"]
        self.assertIn("metadata.source.source_name.keyword", term_query)
        self.assertEqual(term_query["metadata.source.source_name.keyword"], source_value)

    def test_create_metadata_collection_mapping(self):
        """Test create_metadata_collection_mapping function returns correct mapping."""
        result = es_queries.create_metadata_collection_mapping()
        
        # Verify top-level structure
        self.assertIn("mappings", result)
        self.assertIn("properties", result["mappings"])
        
        # Verify properties structure
        properties = result["mappings"]["properties"]
        self.assertIn("collection_name", properties)
        self.assertIn("metadata_schema", properties)
        
        # Verify collection_name field
        collection_name_field = properties["collection_name"]
        self.assertEqual(collection_name_field["type"], "keyword")
        
        # Verify metadata_schema field
        metadata_schema_field = properties["metadata_schema"]
        self.assertEqual(metadata_schema_field["type"], "object")
        self.assertTrue(metadata_schema_field["enabled"])

    def test_get_delete_metadata_schema_query_empty_collection(self):
        """Test get_delete_metadata_schema_query with empty collection name."""
        collection_name = ""
        result = es_queries.get_delete_metadata_schema_query(collection_name)
        
        # Should still return valid structure with empty string
        self.assertIn("query", result)
        term_query = result["query"]["term"]
        self.assertEqual(term_query["collection_name.keyword"], "")

    def test_get_metadata_schema_query_special_characters(self):
        """Test get_metadata_schema_query with special characters in collection name."""
        collection_name = "test-collection_with.special@chars"
        result = es_queries.get_metadata_schema_query(collection_name)
        
        # Should handle special characters properly
        self.assertIn("query", result)
        term_query = result["query"]["term"]
        self.assertEqual(term_query["collection_name"], collection_name)

    def test_get_delete_docs_query_with_spaces(self):
        """Test get_delete_docs_query with source value containing spaces."""
        source_value = "document with spaces.pdf"
        result = es_queries.get_delete_docs_query(source_value)
        
        # Should handle spaces in source value
        self.assertIn("query", result)
        term_query = result["query"]["term"]
        self.assertEqual(term_query["metadata.source.source_name.keyword"], source_value)


if __name__ == '__main__':
    unittest.main()

