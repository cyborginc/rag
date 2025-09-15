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
Pytest configuration for NVIDIA RAG tests
"""

import sys
import types
from unittest.mock import MagicMock, patch

import pytest

# --------------------------------------------------------------------------------------------------
# Prepare import for minio operator to avoid actual Minio connection
# during testing
fake_minio_operator = types.ModuleType("minio_operator")


def mock_minio_operator_methods():
    return MagicMock()


fake_minio_operator.get_minio_operator = mock_minio_operator_methods
fake_minio_operator.get_unique_thumbnail_id_collection_prefix = (
    mock_minio_operator_methods
)
fake_minio_operator.get_unique_thumbnail_id_file_name_prefix = (
    mock_minio_operator_methods
)
fake_minio_operator.get_unique_thumbnail_id = mock_minio_operator_methods
fake_minio_operator.MinioOperator = MagicMock()
# Temporarily inject the fake module into sys.modules
sys.modules["nvidia_rag.utils.minio_operator"] = fake_minio_operator
