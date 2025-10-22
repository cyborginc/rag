# CyborgDB RAG Setup Guide

## Overview

CyborgDB is an encrypted vector database proxy that provides end-to-end encryption for vector embeddings while maintaining high-performance similarity search capabilities. This guide explains how to set up and use CyborgDB with the Confidential Enterprise RAG, and highlights its unique features compared to traditional vector databases.

## Table of Contents

1. [What Makes CyborgDB Different](#what-makes-cyborgdb-different)
2. [Architecture Overview](#architecture-overview)
3. [Setup Requirements](#setup-requirements)
4. [Configuration Steps](#configuration-steps)
5. [Security Features](#security-features)
6. [Performance Considerations](#performance-considerations)
7. [Troubleshooting](#troubleshooting)

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

### Required Environment Variables

```bash
# CyborgDB Configuration
export CYBORGDB_API_KEY="your-api-key-here"
export APP_VECTORSTORE_INDEX_KEY="your-32-byte-key-in-base64"
export APP_VECTORSTORE_URL="http://cyborgdb:8000"
export APP_VECTORSTORE_NAME="cyborgdb"
```

## Configuration Steps

### Step 1: Generate Security Keys

```python
import os
import base64
from cyborgdb import Client

# Generate a secure API key for CyborgDB authentication
# Get your CyborgDB API key from: https://docs.cyborg.co/versions/v0.12.x/intro/get-api-key
# Or refer to api-key.md for detailed instructions
cyborgdb_api_key = ""  # Paste your CyborgDB API key here
os.environ["CYBORGDB_API_KEY"] = cyborgdb_api_key

# Generate a 32-byte encryption key for the index using Client.generate_key()
index_key = Client.generate_key()
os.environ["APP_VECTORSTORE_INDEX_KEY"] = index_key

# Save these keys securely - you'll need them for both ingestion and retrieval
with open('.env', 'a') as f:
    f.write(f"\nCYBORGDB_API_KEY={cyborgdb_api_key}\n")
    f.write(f"APP_VECTORSTORE_INDEX_KEY={os.environ['APP_VECTORSTORE_INDEX_KEY']}\n")

print("Keys generated and saved to .env file")
```

### Step 2: Deploy CyborgDB Services

```bash
# Deploy CyborgDB with Redis backend
docker-compose -f deploy/compose/vectordb.yaml --profile cyborgdb up -d

# Verify services are running
docker ps | grep -E "cyborgdb|redis"
```

### Step 3: Configure RAG Services

Update your deployment configuration:

```yaml
# deploy/compose/docker-compose-ingestor-server.yaml
environment:
  APP_VECTORSTORE_NAME: "cyborgdb"
  APP_VECTORSTORE_URL: "http://cyborgdb:8000"
  APP_VECTORSTORE_APIKEY: ${CYBORGDB_API_KEY}
  APP_VECTORSTORE_INDEX_KEY: ${APP_VECTORSTORE_INDEX_KEY}
```

### Step 4: Deploy RAG Components

```bash
# Deploy ingestion service with CyborgDB support
docker-compose -f deploy/compose/docker-compose-ingestor-server.yaml up -d

# Deploy RAG server with CyborgDB support
docker-compose -f deploy/compose/docker-compose-rag-server.yaml up -d
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
# Query vector is encrypted client-side
results = index.search(query_vector, top_k=5)
# Results are decrypted client-side
```

### 3. Key Management Best Practices

- **Never commit keys to version control**: Use environment variables or secret management systems
- **Rotate keys periodically**: Plan for key rotation procedures
- **Separate keys per environment**: Use different keys for dev, staging, and production
- **Backup keys securely**: Store encrypted backups of your index keys

## Performance Considerations

### GPU Acceleration

CyborgDB supports GPU acceleration through NVIDIA cuVS:

```yaml
# Enable GPU support in docker-compose
services:
  cyborgdb:
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
```

### Optimization Tips

1. **Batch Operations**: Use batch upsert for better throughput
   ```python
   # Good: Batch multiple documents
   index.upsert(documents_batch)
   
   # Avoid: Individual upserts in a loop
   for doc in documents:
       index.upsert([doc])
   ```

2. **Index Configuration**: Optimize for your use case
   ```python
   from cyborgdb import IndexIVFFlat
   
   index_config = IndexIVFFlat(
       dimension=1536,
       n_lists=128,  # Adjust based on dataset size
       metric="euclidean"
   )
   ```

3. **Connection Pooling**: Reuse client connections
   ```python
   # Create once and reuse
   client = Client(base_url=url, api_key=api_key)
   ```

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
echo $APP_VECTORSTORE_INDEX_KEY | base64 -d | wc -c
# Should output: 32
```

#### 3. Performance Issues

```bash
# Check Redis memory usage
docker exec cyborgdb-redis redis-cli INFO memory

# Monitor CyborgDB metrics
docker stats cyborgdb
```

#### 4. Index Not Found

```python
# List all available indexes
client = Client(base_url=url, api_key=api_key)
indexes = client.list_indexes()
print(f"Available indexes: {indexes}")
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