# Ingestor Server Volume Mounting

## Overview

Mount a host directory to access NV-Ingest extraction results directly from the filesystem. Designed for advanced developers who need programmatic access to raw extraction results for custom processing pipelines or external vector database integration.

## Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `INGESTOR_SERVER_EXTERNAL_VOLUME_MOUNT` | `./volumes/ingestor-server` | Host filesystem path |
| `INGESTOR_SERVER_DATA_DIR` | `/data/` | Container internal path |
| `APP_NVINGEST_SAVETODISK` | `False` | Enable disk persistence |

### Setup

1. **Export environment variables:**
   ```bash
   # Enable disk persistence
   export APP_NVINGEST_SAVETODISK=True
   
   # Set host directory path (optional - customize as needed)
   export INGESTOR_SERVER_EXTERNAL_VOLUME_MOUNT=./volumes/ingestor-server
   
   # Set container internal path (optional - customize as needed)
   export INGESTOR_SERVER_DATA_DIR=/data/
   ```

## Troubleshooting

**Optional: Fix permissions issues**
If you encounter permission errors when accessing the volume:
```bash
sudo chown -R 1000:1000 ${INGESTOR_SERVER_EXTERNAL_VOLUME_MOUNT}
sudo chmod -R 755 ${INGESTOR_SERVER_EXTERNAL_VOLUME_MOUNT}
```

## Result Structure

Results are saved as `.jsonl` files with naming convention: `{original_filename}.results.jsonl`

```
${INGESTOR_SERVER_EXTERNAL_VOLUME_MOUNT}/
└── nv-ingest-results/
    ├── collection_name1/
    │   ├── document1.pdf.results.jsonl
    │   ├── presentation.pptx.results.jsonl
    │   └── spreadsheet.xlsx.results.jsonl
    └── collection_name2/
        ├── report.pdf.results.jsonl
        ├── analysis.docx.results.jsonl
        └── data.xlsx.results.jsonl
```

Each `.jsonl` file contains structured extraction metadata including text segments, document structure, images, tables, and chunk boundaries.

**Advanced Usage**: These `.jsonl` files can be used for storing data in vector databases or performing custom processing workflows as desired. This functionality is intended for advanced developers who need direct access to the structured extraction results.

---

**Note**: This is an advanced feature for custom processing workflows. Standard RAG functionality stores results directly in the vector database.