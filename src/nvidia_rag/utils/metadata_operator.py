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

"""Conditional metadata operator selector.
Automatically selects between MinIO and CyborgDB based on vector store configuration.
- If vector_store.name == "milvus": uses MinIO
- Otherwise: uses CyborgDB metadata operator
"""

import logging
from nvidia_rag.utils.common import get_config

logger = logging.getLogger(__name__)
CONFIG = get_config()

# Determine which operator to use based on vector store
if CONFIG.vector_store.name == "milvus":
    logger.info("Using MinIO operator for metadata storage (Milvus vector store detected)")
    from nvidia_rag.utils.minio_operator import (
        MinioOperator as MetadataOperator,
        get_minio_operator as get_metadata_operator,
        get_unique_thumbnail_id_collection_prefix,
        get_unique_thumbnail_id_file_name_prefix,
        get_unique_thumbnail_id
    )
else:
    logger.info(f"Using CyborgDB operator for metadata storage (vector store: {CONFIG.vector_store.name})")
    from nvidia_rag.utils.cyborg_metadata_operator import (
        CyborgMetadataOperator as MetadataOperator,
        get_cyborg_metadata_operator as get_metadata_operator,
        get_unique_thumbnail_id_collection_prefix,
        get_unique_thumbnail_id_file_name_prefix,
        get_unique_thumbnail_id
    )

# Export the selected implementations
__all__ = [
    'MetadataOperator',
    'get_metadata_operator',
    'get_unique_thumbnail_id_collection_prefix', 
    'get_unique_thumbnail_id_file_name_prefix',
    'get_unique_thumbnail_id'
]

# For backward compatibility, also export as the old names
MinioOperator = MetadataOperator
get_minio_operator = get_metadata_operator