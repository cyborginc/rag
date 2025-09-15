<!--
  SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
  SPDX-License-Identifier: Apache-2.0
-->

# Troubleshoot NVIDIA RAG Blueprint

The following issues might arise when you work with the NVIDIA RAG Blueprint.


## Known issues

- The Blueprint responses can have significant latency when using [NVIDIA API Catalog cloud hosted models](quickstart.md#deploy-with-docker-compose).
- The accuracy of the pipeline is optimized for certain file types like `.pdf`, `.txt`, `.docx`. The accuracy may be poor for other file types supported by NvIngest, since image captioning is disabled by default.
- The NeMo LLM microservice may take upto 5-6 mins to start for every deployment.
- In case of failure while uploading files, error messages may not be shown in the user interface of rag-frontend. Developers are encouraged to check the `ingestor-server` logs for details.
- B200 GPUs are not supported for the following advanced features:
  - Image captioning support for ingested documents
  - NeMo Guardrails for guardrails at input/output
  - VLM based inferencing in RAG
  - PDF extraction with Nemoretriever Parse
  For these features, please use H100 or A100 GPUs instead.
- Sometimes when HTTP cloud NIM endpoints are used from `deploy/compose/.env`, the `nv-ingest-ms-runtime` still logs gRPC environment variables. Following log entries can be ignored.
- Poor retrieval accuracy is observed with Milvus GPU indexing and search in B200 and A100. Switch to cpu based search and indexing.
- If one of the file in a bulk ingestion job is of type svg, which is a unsupported format, the full bulk ingestion job fails.
- Complicated filter expressions with custom metadata while sending a query, are not supported from the sample user interface.
- Due to a known issue with MIG support, currently the ingestion profile has been scaled down while deploying the chart with MIG slicing This affects the ingestion performance during bulk ingestion, specifically large bulk ingestion jobs might fail.
- Individual file uploads are limited to a maximum size of 400 MB during ingestion. Files exceeding this limit will be rejected and must be split into smaller segments before ingesting.
- Being a reasoning driven model, nemotron `llama-3.3-nemotron-super-49b-v1.5` model may provide verbose responses, which may not reside within ingested documents. It can also hallucinate for some complicated queries which involves reasoning. Developers are strongly advised to tune [the prompt](prompt-customization.md) for their usecases to avoid these scenarios.
- Ensure to select a model profile explicitly before deploying the nim-llm. Currently, selection of the most optimized profile for `llama-3.3-nemotron-super-49b-v1.5` model on any given GPU does not work, follow steps outlined in [quickstart guide](quickstart.md) to select an optimized profile using `NIM_MODEL_PROFILE` before deploying.
- Slow VDB upload is observed in Helm deployments for Elasticsearch (ES).

- **Confidence threshold filtering issues**
  - If no documents are returned when using confidence threshold filtering, the threshold may be set too high. Try lowering the `confidence_threshold` value or ensure the reranker is enabled to provide relevance scores.
  - Confidence threshold filtering works best when reranker is enabled. Without reranker, documents may not have meaningful relevance scores.
  - For optimal results, use confidence threshold values between 0.3-0.7. Values above 0.7 may be too restrictive.

## Error details: [###] Unknown Error
{'object': 'error', 'message': 'The model `nvidia/llama-3-3-nemotron-super-49b-v1-5` does not exist.', 'type': 'NotFoundError', 'param': None, 'code': 404}

This error happens when incorrect model name is passed to the hosted llm endpoint on https://build.nvidia.com/nvidia/llama-3_3-nemotron-super-49b-v1_5. Ensure for on-prem deployed version of the model `nvidia/llama-3-3-nemotron-super-49b-v1-5` is passed and for cloud hosted version `nvidia/llama-3.3-nemotron-super-49b-v1.5` is passed.

## Out of memory issues while deploying nim-llm service
If you run into `torch.OutOfMemoryError: CUDA out of memory.` while deploying the model, this is most likely due to wrong model profile being auto selected during deployment. Refer to steps in [quickstart guide](quickstart.md) and set the correct profile using `NIM_MODEL_PROFILE` variable.

## Ingestion failures

In case a PDF or PPTx file is not ingested properly, check if that PDF/PPTx only contains images. If the images contain text that you want to extract, try enabling `APP_NVINGEST_EXTRACTINFOGRAPHICS` from [`deploy/compose/docker-compose-ingestor-server.yaml`](../deploy/compose/docker-compose-ingestor-server.yaml).

You may also enable image captioning to better extract content from images. For more details on enabling image captioning, refer to [image_captioning.md](image_captioning.md).

## ERROR: pip's dependency resolver during container building
```
ERROR: pip's dependency resolver does not currently take into account all the packages that are installed. This behaviour is the source of the following dependency conflicts.
```

If the above error related to dependency conflicts are seen while building containers, clear stale docker images using `docker system prune -af` and then execute the build command using `--no-cache` flag.


## pymilvus error: not allowed to retrieve raw data of field sparse
```
pymilvus.exceptions.MilvusException: <MilvusException: (code=65535, message=not allowed to retrieve raw data of field sparse)>
```
This happens when a collection created with vector search type `hybrid` is accessed using vector search type `dense` on retrieval side. Make sure both the search types are same in ingestor-server-compose and rag-server-compose file using `APP_VECTORSTORE_SEARCHTYPE` environment variable.

## DNS resolution failed for <service_name:port>
This category of errors in either `rag-server` or `ingestor-server` container logs indicates:
The server is trying to reach a on-prem deployed NIM at `service_name:port` but it is unreachable. You can ensure that the service is up using `docker ps`.

For example, the below logs in ingestor server container indicates `page-elements` service is unreachable at port `8001`:

```output
Original error: Error during NimClient inference [yolox-page-elements, grpc]: [StatusCode.UNAVAILABLE] DNS resolution failed for page-elements:8001: C-ares status is not ARES_SUCCESS qtype=AAAA name=page-elements is_balancer=0: Could not contact DNS servers
```

In case you were expecting to use cloud hosted model for this NIM, then ensure the corresponding environment variables were set in the same terminal from where you did docker compose up. Following the above example the environment variables which are expected to be set are:

```output
   export YOLOX_HTTP_ENDPOINT="https://ai.api.nvidia.com/v1/cv/nvidia/nemoretriever-page-elements-v2"
   export YOLOX_INFER_PROTOCOL="http"
```

## Elasticsearch connection timeout

If you encounter Elasticsearch connection timeout errors during ingestion, you can adjust the `ES_REQUEST_TIMEOUT` environment variable to increase the timeout duration. This is particularly useful when dealing with large documents or slow Elasticsearch clusters.

### For Helm deployment

Add the `ES_REQUEST_TIMEOUT` environment variable to the `envVars` section in your `values.yaml` file:

```yaml
envVars:
  # ... existing environment variables ...
  ES_REQUEST_TIMEOUT: "1200"  # Timeout in seconds (default is typically 600)
```

### For Docker Compose deployment

Add the `ES_REQUEST_TIMEOUT` environment variable to the `environment` section in your `docker-compose-ingestor-server.yaml` file:

```yaml
environment:
  # ... existing environment variables ...
  ES_REQUEST_TIMEOUT: "1200"  # Timeout in seconds (default is typically 600)
```

After updating the configuration, restart the ingestor server and try the ingestion again. You can increase the timeout value if you continue to experience connection issues, but be aware that very high timeout values may indicate underlying performance issues with your Elasticsearch cluster.

## Device error

You might encounter an `unknown device` error during the [container build process for self-hosted NIMs](quickstart.md#start-the-containers-using-on-prem-models).
This error typically indicates that the container is attempting to access GPUs that are unavailable or non-existent on the host.
To resolve this issue, verify the GPU count specified in the [nims.yaml](../deploy/compose/nims.yaml) configuration file.

```bash
nvidia-container-cli: device error: {n}: unknown device: unknown
```

## Deploy.Resources.Reservations.devices error

You might encounter an error resembling the following during the [container build process for self-hosted NIMs](quickstart.md#start-the-containers-using-on-prem-models) process.
This is likely caused by an [outdated Docker Compose version](https://github.com/docker/compose/issues/11097).
To resolve this issue, upgrade Docker Compose to version `v2.29.0` or later.

```
1 error(s) decoding:

* error decoding 'Deploy.Resources.Reservations.devices[0]': invalid string value for 'count' (the only value allowed is 'all')
```


## Resetting the entire cache

To reset the entire cache, you can run the following command.
This deletes all the volumes associated with the containers, including the cache.

```bash
docker compose down -v
```


## External Vector databases

We've integrated VDB and embedding creation directly into the pipeline with caching included for expediency.
However, in a production environment, it's better to use a separately managed VDB service.

NVIDIA offers optimized models and tools like NIMs ([build.nvidia.com/explore/retrieval](https://build.nvidia.com/explore/retrieval))
and cuVS ([github.com/rapidsai/cuvs](https://github.com/rapidsai/cuvs)).


## Running out of credits

If you run out of credits for the NVIDIA API Catalog,
you will need to obtain more credits to continue using the API.
Please contact your NVIDIA representative to get more credits.


## Password Issue Fix

If you encounter any `password authentication failed` issues with the structured retriever container,
consider removing the volumes directory located at `deploy/compose/volumes`.
In this case, you may need to reprocess the data ingestion.

## Node exporter pod crash with prometheus stack enabled in helm deployment

If you experience issues with the `prometheus-node-exporter` pod crashing after enabling the `kube-prometheus-stack`, and you encounter an error message like:

```sh
msg="listen tcp 0.0.0.0:9100: bind: address already in use"
```

This error indicates that the port `9100` is already in use. To resolve this, you can update the port for `prometheus-node-exporter` in the `values.yaml` file.

Update the following in `values.yaml`:

```yaml
kube-prometheus-stack:
   # ... existing code ...
  prometheus-node-exporter:
    service:
      port: 9101 # Changed from 9100 to 9101
      targetPort: 9101  # Changed from 9100 to 9101
```

## Handling Hallucination and Out-of-Context Responses

The current prompt configuration does not strictly enforce response generation from the retrieved context. This can result in the following scenarios:

1. **Out-of-context responses**: The LLM generates responses that are not grounded in the provided context
2. **Irrelevant context usage**: The model provides information from the retrieved context that doesn't directly answer the user's query

### Solution

These issues can be addressed by adding the following instruction to the `rag_chain` user prompt in [prompt.yaml](../src/nvidia_rag/rag_server/prompt.yaml):

```yaml
Handling Missing Information: If the context does not contain the answer, you must state directly that you do not have information on the specific subject of the user's query. For example, if the query is about the "capital of France", your response should be "I did not find information about capital of France." Do not add any other words, apologies, or explanations.
```

**Note**: Adding this information may impact response accuracy, especially when partial information is available instead of complete information in the retrieved context. The system may become more conservative in providing answers, potentially refusing to respond even when some relevant information exists in the context.
