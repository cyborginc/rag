# CyborgDB RAG Setup Guide

## Overview

CyborgDB is an encrypted vector database proxy that provides end-to-end encryption for vector embeddings while maintaining high-performance similarity search capabilities. This guide explains how to set up and use CyborgDB with the Confidential Enterprise RAG, and highlights its unique features compared to traditional vector databases.

## Table of Contents

1. [What Makes CyborgDB Different](#what-makes-cyborgdb-different)
2. [Architecture Overview](#architecture-overview)
3. [Setup Requirements](#setup-requirements)
4. [Configuration Steps](#configuration-steps)
5. [GPU to CPU Mode Switch](#gpu-to-cpu-mode-switch)
6. [Security Features](#security-features)
7. [Troubleshooting](#troubleshooting)
8. [Migration Guide](#migration-guide)

## What Makes CyborgDB Different

### CyborgDB vs Milvus vs Elasticsearch

| Feature | CyborgDB | Milvus | Elasticsearch |
|---------|----------|---------|---------------|
| **Primary Focus** | Encrypted vector search | High-performance vector search | Full-text + vector search |
| **Encryption** | End-to-end client-side encryption | TLS/SSL only | TLS/SSL + optional at-rest |
| **Architecture** | Proxy + backing store (Redis) | Standalone vector DB | Search engine with vector support |
| **Zero-Trust** | ✅ Server never sees plaintext | ❌ Server processes plaintext | ❌ Server processes plaintext |
| **Storage Backend** | Redis (or PostgreSQL) | Custom storage engine | Lucene-based indices |
| **GPU Acceleration** | ✅ Via NVIDIA cuVS | ✅ Native support | Limited |
| **Encrypted Search** | ✅ Search on encrypted vectors | ❌ Must decrypt first | ❌ Must decrypt first |
| **Key Management** | Client-side index keys | N/A | Server-side key management |
| **Use Case** | Confidential AI/ML | General vector search | Hybrid search (text + vectors) |

### Key Differentiators

1. **Client-Side Encryption**: Unlike Milvus and Elasticsearch which only encrypt data in transit and optionally at rest, CyborgDB encrypts vectors on the client before they ever leave your application.

2. **Zero-Trust Architecture**: The CyborgDB server/proxy never has access to unencrypted vectors. Even during similarity search, computations are performed on encrypted data.

3. **Database Proxy Model**: CyborgDB transforms existing databases (Redis, PostgreSQL) into encrypted vector stores, rather than being a standalone database.

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                   Client Application                        │
│  ┌─────────────────────────────────────────────────────┐    │
│  │  1. Generate embeddings from documents              │    │
│  │  2. Encrypt vectors with 32-byte index key          │    │
│  │  3. Send encrypted vectors to CyborgDB              │    │
│  └─────────────────────────────────────────────────────┘    │
└─────────────────────────────┬───────────────────────────────┘
                              │ Encrypted vectors
                              │ + metadata
                              ▼
         ┌──────────────────────────────────────────┐
         │       CyborgDB Proxy Service             │
         │  ┌────────────────────────────────┐      │
         │  │ • Never sees plaintext vectors │      │
         │  │ • Performs encrypted search    │      │
         │  │ • GPU acceleration via cuVS    │      │
         │  └────────────────────────────────┘      │
         └────────────────────┬─────────────────────┘
                              │ Encrypted storage
                              ▼
         ┌──────────────────────────────────────────┐
         │         Redis Backing Store              │
         │     (Encrypted vectors at rest)          │
         └──────────────────────────────────────────┘
```

## Setup Requirements

### Prerequisites

1. **Docker & Docker Compose**: Version 20.10 or higher
2. **NVIDIA GPU**: For optimal performance (optional, but recommended)
3. **System Resources**: 
   - Minimum 16GB RAM
   - 50GB free disk space
   - NVIDIA GPU with 8GB+ VRAM (for GPU acceleration)

## Configuration Steps

### Get an API Key

For information on how to get your CyborgDB API Key, follow [this guide](./api-key.md#cyborgdb-api-key)

Store your api key as an environment variable.

```bash
export CYBORGDB_API_KEY="cyborg_..."
```

### Generate and Store your Index Key

> [!WARNING]
> This guide covers storing keys locally for development purposes. For production purposes, you should always use secure options such as Hardware Security Modules (HSMs) or a Key Management Service (KMS). See [Using Key Management Services (for Production)](https://docs.cyborg.co/versions/v0.12.x/service/guides/advanced/managing-keys#using-key-management-services-for-production)

Encrypted Indexes are the main organizational unit of CyborgDB. One encrypted index is secured with one index key, which provides useful segmentation:

* **Cryptographic isolation**: each index is isolated via encryption keys, making it impossible to query/view the contents of an index without proper access/authorization.
* **Multi-tenancy**: this separation makes it easy to separate data scopes (e.g., tenants) in a robust and secure manner.

One client can manage an arbitary number of indexes, and an index can contain an arbitrary amount of items/vectors. All contents of the index are end-to-end encrypted, meaning that they remain encrypted throughout their lifecycle (at-rest and in-use).

```bash
# first generate your index key and store it
openssl rand -base64 32 > index_key.txt

# then write it to your environment for use with the ingestion server
export APP_VECTORSTORE_INDEXKEY=$(cat index_key.txt)
```

### Deploy CyborgDB Services

```bash
# Deploy CyborgDB with Redis backend
docker-compose -f deploy/compose/vectordb.yaml --profile cyborgdb up -d

# Verify services are running
docker ps | grep -E "cyborgdb|redis"
```

### Deploy RAG Components

```bash
# Deploy ingestion service with CyborgDB support
docker-compose -f deploy/compose/docker-compose-ingestor-server.yaml up -d

# Deploy RAG server with CyborgDB support
docker-compose -f deploy/compose/docker-compose-rag-server.yaml up -d
```

## GPU to CPU Mode Switch

By default, CyborgDB uses GPU acceleration for upsert, train, and query operations. To switch to CPU mode, follow the steps below.

### 1. Update Docker Compose Configuration (vectordb.yaml)

First, you need to modify the `deploy/compose/vectordb.yaml` file to disable GPU usage:

#### Step 1: Comment Out GPU Reservations
Comment out the entire deploy section that reserves GPU resources:
```yaml
# deploy:
#   resources:
#     reservations:
#       devices:
#         - driver: nvidia
#           capabilities: ["gpu"]
#           device_ids: ['${VECTORSTORE_GPU_DEVICE_ID:-0}']
```

#### Step 2: Change the CyborgDB Docker Image
```yaml
# Change this line:
image: cyborginc/cyborgdb-service:v0.13.0-gpu

# To this:
image: cyborginc/cyborgdb-service:v0.13.0
```

### 2. Restart Services

```bash
# 1. Stop existing vectordb services
docker compose -f deploy/compose/vectordb.yaml down

# 2. Start CyborgDB and dependencies
docker compose -f deploy/compose/vectordb.yaml up -d
```

## Security Features

### 1. Client-Side Encryption

```python
# Example: How vectors are encrypted before storage
from cyborgdb import Client, EncryptedIndex

# Initialize with encryption key
index = EncryptedIndex(
    index_name="my_collection",
    index_key=index_key,  # 32-byte key
    api=client.api
)

# Vectors are encrypted client-side before upsert
index.upsert([{
    "id": "doc_1",
    "vector": embedding,  # Encrypted before transmission
    "metadata": {"source": "document.pdf"}
}])
```

### 2. Encrypted Search

```python
# Queries are also encrypted before search
query_vector = embed_model.encode("What is RAG?")
# Query vector is encrypted client-side, then sent to CyborgDB
results = index.query(query_vectors=query_vector, top_k=5)
# Results are returned with IDs and distances
```

### 3. Key Management Best Practices

- **Never commit keys to version control**: Use environment variables or secret management systems
- **Rotate keys periodically**: Plan for key rotation procedures
- **Separate keys per environment**: Use different keys for dev, staging, and production
- **Backup keys securely**: Store encrypted backups of your index keys

## Troubleshooting

### Common Issues and Solutions

#### 1. Connection Errors

```bash
# Check if CyborgDB is running
docker logs cyborgdb

# Verify Redis backend is accessible
docker exec cyborgdb redis-cli ping
```

#### 2. Encryption Key Issues

```bash
# Validate key format (must be 32 bytes)
echo $APP_VECTORSTORE_INDEXKEY | base64 -d | wc -c
# Should output: 32
```

#### 3. Performance Issues

```bash
# Check Redis memory usage
docker exec cyborgdb-redis redis-cli INFO memory

# Monitor CyborgDB metrics
docker stats cyborgdb
```

## Migration Guide

### Migrating from Milvus

1. Export vectors from Milvus
2. Generate encryption keys
3. Re-index with CyborgDB using encryption
4. Update connection strings in your application

### Migrating from Elasticsearch

1. Extract dense vectors from Elasticsearch
2. Set up CyborgDB with Redis backend
3. Batch upload vectors with encryption
4. Update search queries to use CyborgDB client

## Best Practices

1. **Security First**: Always use strong, randomly generated keys
2. **Monitor Performance**: Track query latencies and throughput
3. **Plan for Scale**: Redis memory should be 2-3x your vector dataset size
4. **Test Recovery**: Practice key rotation and backup restoration
5. **Audit Access**: Log all vector database operations

## Additional Resources

- [CyborgDB Documentation](https://docs.cyborg.co)
- [NVIDIA cuVS Library](https://github.com/rapidsai/cuvs)
- [Redis Best Practices](https://redis.io/docs/manual/patterns/)
- [Confidential Enterprise RAG Docs](../README.md)

## Support

For issues specific to CyborgDB integration:
1. Check the [troubleshooting section](#troubleshooting)
2. Review container logs: `docker logs cyborgdb`
3. Consult the [CyborgDB documentation](https://docs.cyborg.co)
4. Open an issue in the Confidential Enterprise RAG repository

---

**Security Note**: This setup provides end-to-end encryption for your vector embeddings. However, ensure you follow all security best practices for key management and access control in your production environment.

## Related Topics

- [NVIDIA RAG Blueprint Documentation](readme.md)
- [Best Practices for Common Settings](accuracy_perf.md).
- [RAG Pipeline Debugging Guide](debugging.md)
- [Troubleshoot](troubleshooting.md)
- [Notebooks](notebooks.md)