<!--
  SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
  SPDX-License-Identifier: Apache-2.0
-->

# Enable NeMo Retriever OCR

For enhanced optical character recognition capabilities, you can use the NeMo Retriever OCR service instead of the default Paddle OCR. This service provides improved text extraction from images within documents.

## Using Docker Compose

### Using On-Prem Models

1. Follow steps outlined in the [quickstart guide](quickstart.md#start-using-on-prem-models) till step 4. Deploy all the deployed NIMs for ingestion.

2. **For NeMo Retriever OCR** (recommended for enhanced text extraction):
   
   Export the environment variables:
   ```bash
   export OCR_GRPC_ENDPOINT=nemoretriever-ocr:8001
   export OCR_HTTP_ENDPOINT=http://nemoretriever-ocr:8000/v1/infer
   export OCR_INFER_PROTOCOL=grpc
   export OCR_MODEL_NAME=scene_text_ensemble
   ```

   > [!Important]
   > **Health Check Requirement**: Even when using gRPC protocol (`OCR_INFER_PROTOCOL=grpc`), you must also export the `OCR_HTTP_ENDPOINT` because the health check from nv-ingest uses HTTP.

   Start the NeMo Retriever OCR service:
   ```bash
   USERID=$(id -u) docker compose -f deploy/compose/nims.yaml --profile nemoretriever-ocr up -d
   ```

   Verify the service is running:
   ```bash
   watch -n 2 'docker ps --format "table {{.Names}}\t{{.Status}}"'
   ```

   Restart the ingestor-server to pick up the new OCR configuration:
   ```bash
   docker compose -f deploy/compose/docker-compose-ingestor-server.yaml up -d
   ```

3. **For Paddle OCR** (default OCR service):
   
   Paddle OCR runs automatically with the default `docker compose` command. If you want to explicitly start it:
   ```bash
   USERID=$(id -u) docker compose -f deploy/compose/nims.yaml up -d
   ```

   Verify the service is running:
   ```bash
   watch -n 2 'docker ps --format "table {{.Names}}\t{{.Status}}"'
   ```

4. You can now ingest documents using the [ingestion API usage notebook](../notebooks/ingestion_api_usage.ipynb).

> [!Note]
> **Default Behavior**: Paddle OCR is the default OCR service and runs automatically when you start the NIMs. To use NeMo Retriever OCR instead, you must explicitly start it with the `--profile nemoretriever-ocr` flag.

### Using NVIDIA Hosted API Endpoints

1. Follow steps outlined in the [quickstart guide](quickstart.md#start-using-nvidia-hosted-models) till step 2. Export the following variables to use NeMo Retriever OCR API endpoints:

   ```bash
   export OCR_HTTP_ENDPOINT=https://ai.api.nvidia.com/v1/cv/nvidia/nemoretriever-ocr
   export OCR_INFER_PROTOCOL=http
   export OCR_MODEL_NAME=scene_text_ensemble
   ```

2. Deploy the ingestion-server and rag-server containers following the remaining steps in the quickstart guide.

3. You can now ingest documents using the [ingestion API usage notebook](../notebooks/ingestion_api_usage.ipynb).

> [!Note]
> When using NVIDIA hosted endpoints, you may encounter rate limiting with larger file ingestions (>10 files).

## Using Helm

To enable NeMo Retriever OCR using Helm, you need to enable the NeMo Retriever OCR service and disable Paddle OCR to save GPU resources:

```bash
# Apply following cmd to a fresh deployment, recommended to uninstall all deployments 1st if already present - `helm uninstall rag -n rag`
helm upgrade --install rag -n rag https://helm.ngc.nvidia.com/nvstaging/blueprint/charts/nvidia-blueprint-rag-v2.3.0-rc2.tgz \
  --username '$oauthtoken' \
  --password "${NGC_API_KEY}" \
  --set nv-ingest.paddleocr-nim.deployed=false \
  --set nv-ingest.nemoretriever-ocr.deployed=true \
  --set nv-ingest.envVars.OCR_MODEL_NAME="scene_text_ensemble" \
  --set imagePullSecret.password=$NGC_API_KEY \
  --set ngcApiSecret.password=$NGC_API_KEY
```

## Limitations and Requirements

When using NeMo Retriever OCR, please note the following:

- The service requires GPU resources. Make sure you have sufficient GPU resources available before enabling this feature.
- The extraction quality may vary depending on the image quality and text complexity.
- NeMo Retriever OCR is designed for document text extraction and may perform better than Paddle OCR for certain use cases.

For detailed information about hardware requirements and supported GPUs, refer to the [NeMo Retriever OCR Support Matrix](https://docs.nvidia.com/nim/ingestion/image-ocr/latest/support-matrix.html).

## Available OCR Models

The OCR service can be configured to use either:

- **NeMo Retriever OCR**: Enhanced text extraction optimized for document processing
- **Paddle OCR**: Default OCR service for general text extraction

> [!Important]
> **Resource Optimization**: When using NeMo Retriever OCR, it's recommended to disable Paddle OCR to save GPU resources. This saves 1 GPU that would otherwise be used by the unused Paddle OCR service. 