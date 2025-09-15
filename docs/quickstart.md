<!--
  SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
  SPDX-License-Identifier: Apache-2.0
-->

# Get Started With NVIDIA RAG Blueprint

Use the following documentation to get started with the NVIDIA RAG Blueprint.

- [Obtain an API Key](#obtain-an-api-key)
- [Interact using native python APIs](#interact-using-native-python-apis)
- [Deploy With Docker Compose](#deploy-with-docker-compose)
- [Deploy With Helm Chart](#deploy-with-helm-chart)
- [Data Ingestion](#data-ingestion)


## Obtain an API Key

You need to generate an API key
to access NIM services, to access models hosted in the NVIDIA API Catalog, and to download models on-premises.
For more information, refer to [NGC API Keys](https://docs.nvidia.com/ngc/gpu-cloud/ngc-private-registry-user-guide/index.html#ngc-api-keys).

To generate an API key, use the following procedure.

1. Go to https://org.ngc.nvidia.com/setup/api-keys.
2. Click **+ Generate Personal Key**.
3. Enter a **Key Name**.
4. For **Services Included**, select **NGC Catalog** and **Public API Endpoints**.
5. Click **Generate Personal Key**.

After you generate your key, export your key as an environment variable by using the following code.

```bash
export NGC_API_KEY="<your-ngc-api-key>"
```


## Interact using native python APIs

You can interact with and deploy the NVIDIA RAG Blueprint directly from Python using the provided Jupyter notebook. This approach is ideal for users who prefer a programmatic interface for setup, ingestion, and querying, or for those who want to automate workflows and integrate with other Python tools.

- **Notebook:** [rag_library_usage.ipynb](../notebooks/rag_library_usage.ipynb)

The notebook demonstrates environment setup, document ingestion, collection management, and querying using the NVIDIA RAG Python client. Follow the cells in the notebook for an end-to-end example of using the RAG system natively from Python.


## Deploy With Docker Compose

Use these procedures to deploy with Docker Compose for a single node deployment. Alternatively, you can [Deploy With Helm Chart](#deploy-with-helm-chart) to deploy on a Kubernetes cluster.

Developers need to deploy ingestion services and rag services using seperate dedicated docker compose files.
For both retrieval and ingestion services, by default all the models are deployed on-prem. Follow relevant section below as per your requirement and hardware availability.

- Start the Microservices
  - [Using on-prem models](#start-using-on-prem-models)
  - [Using NVIDIA hosted models](#start-using-nvidia-hosted-models)

### Prerequisites

1. Install Docker Engine. For more information, see [Ubuntu](https://docs.docker.com/engine/install/ubuntu/).

2. Install Docker Compose. For more information, see [install the Compose plugin](https://docs.docker.com/compose/install/linux/).

   a. Ensure the Docker Compose plugin version is 2.29.1 or later.

   b. After you get the Docker Compose plugin installed, run `docker compose version` to confirm.

3. To pull images required by the blueprint from NGC, you must first authenticate Docker with nvcr.io. Use the NGC API Key you created in [Obtain an API Key](#obtain-an-api-key).

   ```bash
   export NGC_API_KEY="nvapi-..."
   echo "${NGC_API_KEY}" | docker login nvcr.io -u '$oauthtoken' --password-stdin
   ```

4. Some containers with are enabled with GPU acceleration, such as Milvus and NVIDIA NIMS deployed on-prem. To configure Docker for GPU-accelerated containers, [install](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html), the NVIDIA Container Toolkit

5. Ensure you meet [the hardware requirements if you are deploying models on-prem](./support-matrix.md).


### Start using on-prem models

Use the following procedure to start all containers needed for this blueprint. This launches the ingestion services followed by the rag services and all of its dependent NIMs on-prem.

1. Fulfill the [prerequisites](#prerequisites). Ensure you meet [the hardware requirements](./support-matrix.md).

2. Create a directory to cache the models and export the path to the cache as an environment variable.

   ```bash
   mkdir -p ~/.cache/model-cache
   export MODEL_DIRECTORY=~/.cache/model-cache
   ```

3. Export all the required environment variables to use on-prem models. Ensure the section `Endpoints for using cloud NIMs` is commented in this file.

   ```bash
   source deploy/compose/.env
   ```

   [Optional]:
   Check out the guidance here for some [best practices] for choosing configurations(./accuracy_perf.md).
   To turn on recommended configs for accuracy optimized profile set additional configs:
   ```bash
   source deploy/compose/accuracy_profile.env
   ```

   To turn on recommended configs for perf optimized profile set additional configs:
   ```bash
   source deploy/compose/perf_profile.env
   ```

4. List available profiles for your system for nim llm container. More details about profiles can be found [here](https://docs.nvidia.com/nim/large-language-models/latest/profiles.html).

   ```bash
   USERID=$(id -u) docker compose -f deploy/compose/nims.yaml run nim-llm list-model-profiles
   ```

   Example output for H100-NVL when 1 GPU is allocated:

   ```output
   MODEL PROFILES
   - Compatible with system and runnable:
   - d491076c9c5fbbc0f5ec92916ee84050c3b51a1c93055a64de8d0f31a22ed209 (vllm-bf16-tp1-pp1-32c3b968468aefcfb3ea1db5a16e3dc9d64395f02ef68a06175e8bbdb0038601)
   - e2f00b2cbfb168f907c8d6d4d40406f7261111fbab8b3417a485dcd19d10cc98 (vllm)
   - e759b32c9a5c4d16aced45cd7797ec030feb7d9164a7fe74278a76c795baf8cb (tensorrt_llm-h100_nvl-fp8-tp1-pp1-throughput-2321:10de-6343e21ba5cccf783d18951c6627c207b81803c3c45f1e8b59eee062ed350143-1)
   - 668b575f1701fa70a97cfeeae998b5d70b048a9b917682291bb82b67f308f80c (tensorrt_llm)
   - 50e138f94d85b97117e484660d13b6b54234e60c20584b1de6ed55d109ca4f21 (sglang)
   ```

5. Set the required profile. It is preferrable to select `tensorrt_llm-*` profiles for best performance. By default, automatic selection of profile is enabled but due to a known issue, vllm based profiles are selected, so it is recommended to manually select a tensorrt_llm profile before starting the `nim-llm` service.

   Example of most common profiles for different supported systems are given below:

   1xH100 NVL based
   ```bash
   export NIM_MODEL_PROFILE=tensorrt_llm-h100_nvl-fp8-tp1-pp1-throughput-2321:10de-6343e21ba5cccf783d18951c6627c207b81803c3c45f1e8b59eee062ed350143-1
   ```

   1xH100 SXM based
   ```bash
   export NIM_MODEL_PROFILE=tensorrt_llm-h100-fp8-tp1-pp1-throughput-2330:10de-ed15592b3e4d174a719e8188493420073c39448d9b7ed742cfe614b96fecbdd9-1
   ```

   2xA100

    The default configuration allocates one GPU (GPU ID 1) to `nim-llm-ms` which defaults to minimum GPUs needed for H100 or B200 or RTX6000 profile. If you are deploying the solution on A100, please allocate 2 available GPUs by exporting below env variable before launching or listing profiles:
     ```bash
     export LLM_MS_GPU_ID=1,2
     ```
   ```bash
   export NIM_MODEL_PROFILE=tensorrt_llm-a100-bf16-tp2-pp1-throughput-20b2:10de-4c44baded168675f277ffcbd4f9bce56cac0239944062c44d81d359f9c718f06-2
   ```
   1xRTX6000
   ```bash
   export NIM_MODEL_PROFILE=tensorrt_llm-rtx6000_blackwell_sv-fp8-tp1-pp1-throughput-2bb5:10de-b5e4ef5f657564a90ff8b1fc8e8e6d59f24845f183209d563aa93d91fc6629ac-1
   ```

   1xB200
   ```bash
   export NIM_MODEL_PROFILE=tensorrt_llm-gb200-fp8-tp1-pp1-throughput-2941:10de-a642befc43aa112e87399e2fea6d3ef2994cadb5ccb2715c9bc7e89c35e92383-1
   ```

6. Start all required NIMs.

   Before running the command please ensure the GPU allocation is done appropriately in the deploy/compose/.env. You might need to override them
   for the hardware you are deploying this blueprint on. The default assumes you are deploying this on a 2XH100 environment.

   ```bash
   USERID=$(id -u) docker compose -f deploy/compose/nims.yaml up -d
   ```

   - Wait till the `nemoretriever-ranking-ms`, `nemoretriever-embedding-ms` and `nim-llm-ms`  NIMs are in healthy state before proceeding further.

   - The nemo LLM service may take upto 30 mins to start for the first time as the model is downloaded and cached. The models are downloaded and cached in the path specified by `MODEL_DIRECTORY`. Subsequent deployments will take 2-5 mins to startup based on the GPU profile. For other common deployment issues you can [checkout troubleshooting guide](./troubleshooting.md).

   - To start just the NIMs specific to rag or ingestion add the `--profile rag` or `--profile ingest` flag to the command.

   - To use NeMo Retriever OCR instead of Paddle OCR, add the `--profile nemoretriever-ocr` flag and update your OCR environment variables accordingly. See [NeMo Retriever OCR](nemoretriever-ocr.md) for detailed instructions.

   - Ensure all the below are running before proceeding further

     ```bash
     watch -n 2 'docker ps --format "table {{.Names}}\t{{.Status}}"'
     ```

     ```output
        NAMES                                   STATUS

        nemoretriever-ranking-ms                Up 14 minutes (healthy)
        compose-page-elements-1                 Up 14 minutes
        compose-paddle-1                        Up 14 minutes
        compose-graphic-elements-1              Up 14 minutes
        compose-table-structure-1               Up 14 minutes
        nemoretriever-embedding-ms              Up 14 minutes (healthy)
        nim-llm-ms                              Up 14 minutes (healthy)
     ```


7. Start the vector db containers from the repo root.
   ```bash
   docker compose -f deploy/compose/vectordb.yaml up -d
   ```

   [!TIP]
   By default GPU accelerated Milvus DB is deployed. You can choose the GPU ID to be allocated using below env variable.
   ```bash
   VECTORSTORE_GPU_DEVICE_ID=0
   ```

   For B200 and A100 GPUs, use Milvus CPU indexing due to known retrieval accuracy issues with Milvus GPU indexing and search. Export following environment variables to disable Milvus GPU ingexing and search.
   ```bash
   export APP_VECTORSTORE_ENABLEGPUSEARCH=False
   export APP_VECTORSTORE_ENABLEGPUINDEX=False
   ```

8. Start the ingestion containers from the repo root. This pulls the prebuilt containers from NGC and deploys it on your system.

   ```bash
   docker compose -f deploy/compose/docker-compose-ingestor-server.yaml up -d
   ```

   [!TIP]
   You can add a `--build` argument in case you have made some code changes or have any requirement of re-building ingestion containers from source code:

   ```bash
   docker compose -f deploy/compose/docker-compose-ingestor-server.yaml up -d --build
   ```

   [!TIP]
   For advanced users who need direct filesystem access to extraction results, see [Ingestor Server Volume Mounting](mount-ingestor-volume.md) to configure volume mounting for custom processing pipelines.

   You can check the status of the ingestor-server and its dependencies by issuing this curl command
   ```bash
   curl -X 'GET' 'http://workstation_ip:8082/v1/health?check_dependencies=true' -H 'accept: application/json'
   ```

9. Start the rag containers from the repo root. This pulls the prebuilt containers from NGC and deploys it on your system.

   ```bash
   docker compose -f deploy/compose/docker-compose-rag-server.yaml up -d
   ```

   [!TIPS]
   You can add a `--build` argument in case you have made some code changes or have any requirement of re-building containers from source code:

   ```bash
   docker compose -f deploy/compose/docker-compose-rag-server.yaml up -d --build
   ```

   You can check the status of the rag-server and its dependencies by issuing this curl command
   ```bash
   curl -X 'GET' 'http://workstation_ip:8081/v1/health?check_dependencies=true' -H 'accept: application/json'
   ```

10. Confirm all the below mentioned containers are running.

   ```bash
   docker ps --format "table {{.ID}}\t{{.Names}}\t{{.Status}}"
   ```

   *Example Output*

   ```output
   NAMES                                   STATUS
   compose-nv-ingest-ms-runtime-1          Up 5 minutes (healthy)
   ingestor-server                         Up 5 minutes
   compose-redis-1                         Up 5 minutes
   rag-frontend                            Up 9 minutes
   rag-server                              Up 9 minutes
   milvus-standalone                       Up 36 minutes
   milvus-minio                            Up 35 minutes (healthy)
   milvus-etcd                             Up 35 minutes (healthy)
   nemoretriever-ranking-ms                Up 38 minutes (healthy)
   compose-page-elements-1                 Up 38 minutes
   compose-paddle-1                        Up 38 minutes
   compose-graphic-elements-1              Up 38 minutes
   compose-table-structure-1               Up 38 minutes
   nemoretriever-embedding-ms              Up 38 minutes (healthy)
   nim-llm-ms                              Up 38 minutes (healthy)
   ```

11.  Open a web browser and access `http://localhost:8090` to use the RAG Playground. You can use the upload tab to ingest files into the server or follow [the notebooks](../notebooks/) to understand the API usage.

12. To stop all running services, after making some [customizations](#next-steps)
    ```bash
    docker compose -f deploy/compose/docker-compose-ingestor-server.yaml down
    docker compose -f deploy/compose/nims.yaml down
    docker compose -f deploy/compose/docker-compose-rag-server.yaml down
    docker compose -f deploy/compose/vectordb.yaml down
    ```

**📝 Notes:**

1. A single NVIDIA A100-80GB or H100-80GB, B200 GPU can be used to start non-LLM NIMs (nemoretriever-embedding-ms, nemoretriever-ranking-ms, and ingestion services like page-elements, ocr, graphic-elements, and table-structure) for ingestion and RAG workflows. You can control which GPU is used for each service by setting these environment variables in `deploy/compose/.env` file before launching:
   ```bash
   EMBEDDING_MS_GPU_ID=0
   RANKING_MS_GPU_ID=0
   YOLOX_MS_GPU_ID=0
   YOLOX_GRAPHICS_MS_GPU_ID=0
   YOLOX_TABLE_MS_GPU_ID=0
   OCR_MS_GPU_ID=0
   ```

2. If the NIMs are deployed in a different workstation or outside the nvidia-rag docker network on the same system, replace the host address of the below URLs with workstation IPs.

   ```bash
   APP_EMBEDDINGS_SERVERURL="workstation_ip:8000"
   APP_LLM_SERVERURL="workstation_ip:8000"
   APP_RANKING_SERVERURL="workstation_ip:8000"
   OCR_GRPC_ENDPOINT="workstation_ip:8001"
   YOLOX_GRPC_ENDPOINT="workstation_ip:8001"
   YOLOX_GRAPHIC_ELEMENTS_GRPC_ENDPOINT="workstation_ip:8001"
   YOLOX_TABLE_STRUCTURE_GRPC_ENDPOINT="workstation_ip:8001"
   ```

### Start using nvidia hosted models

1. Verify that you meet the [prerequisites](#prerequisites).

2. Open `deploy/compose/.env` and uncomment the section `Endpoints for using cloud NIMs`.
   Then set the environment variables by executing below command.
   ```bash
   source deploy/compose/.env
   ```

   [Optional]:
   Check out the guidance here for some [best practices] for choosing configurations(./accuracy_perf.md).
   To turn on recommended configs for accuracy optimized profile set additional configs:
   ```bash
   source deploy/compose/accuracy_profile.env
   ```

   To turn on recommended configs for perf optimized profile set additional configs:
   ```bash
   source deploy/compose/perf_profile.env
   ```

   **📝 Note:**
   - When using NVIDIA hosted endpoints, you may encounter rate limiting with larger file ingestions (>10 files).

    [!IMPORTANT]
    The LLM model name is different for cloud deployment!
    Ensure to set APP_LLM_MODELNAME=nvidia/llama-3.3-nemotron-super-49b-v1.5 instead of export APP_LLM_MODELNAME=nvidia/llama-3-3-nemotron-super-49b-v1-5 when switching to cloud from on-prem! This is preset in .env file.

3. Start the vector db containers from the repo root.
   ```bash
   docker compose -f deploy/compose/vectordb.yaml up -d
   ```
   [!NOTE]
   If you don't have a GPU available, you can switch to CPU-only Milvus by following the instructions in [milvus-configuration.md](./milvus-configuration.md).

   [!TIP]
   For B200 and A100 GPUs, use Milvus CPU indexing due to known retrieval accuracy issues with Milvus GPU indexing and search. Export following environment variables to disable Milvus GPU ingexing and search.
   ```bash
   export APP_VECTORSTORE_ENABLEGPUSEARCH=False
   export APP_VECTORSTORE_ENABLEGPUINDEX=False
   ```

4. Start the ingestion containers from the repo root. This pulls the prebuilt containers from NGC and deploys it on your system.

   ```bash
   docker compose -f deploy/compose/docker-compose-ingestor-server.yaml up -d
   ```

   [!TIP]
   You can add a `--build` argument in case you have made some code changes or have any requirement of re-building ingestion containers from source code:

   ```bash
   docker compose -f deploy/compose/docker-compose-ingestor-server.yaml up -d --build
   ```

   You can check the status of the ingestor-server and its dependencies by issuing this curl command
   ```bash
   curl -X 'GET' 'http://workstation_ip:8081/v1/health?check_dependencies=true' -H 'accept: application/json'
   ```

   In case you have a requirement of building Nvidia Ingest runtime container from source, you can do it by following instructions from [this link](https://github.com/NVIDIA/nv-ingest), using docker compose.


5. Start the rag containers from the repo root. This pulls the prebuilt containers from NGC and deploys it on your system.

   ```bash
   docker compose -f deploy/compose/docker-compose-rag-server.yaml up -d
   ```

   [!TIP]
   You can add a `--build` argument in case you have made some code changes or have any requirement of re-building containers from source code:

   ```bash
   docker compose -f deploy/compose/docker-compose-rag-server.yaml up -d --build
   ```

   You can check the status of the rag-server and its dependencies by issuing this curl command
   ```bash
   curl -X 'GET' 'http://workstation_ip:8081/v1/health?check_dependencies=true' -H 'accept: application/json'
   ```

6. Confirm all the below mentioned containers are running.

   ```bash
   docker ps --format "table {{.ID}}\t{{.Names}}\t{{.Status}}"
   ```

   *Example Output*

   ```output
   NAMES                                   STATUS
   compose-nv-ingest-ms-runtime-1          Up 5 minutes (healthy)
   ingestor-server                         Up 5 minutes
   compose-redis-1                         Up 5 minutes
   rag-frontend                                  Up 9 minutes
   rag-server                              Up 9 minutes
   milvus-standalone                       Up 36 minutes
   milvus-minio                            Up 35 minutes (healthy)
   milvus-etcd                             Up 35 minutes (healthy)
   ```

7. Open a web browser and access `http://localhost:8090` to use the RAG Playground. You can use the upload tab to ingest files into the server or follow [the notebooks](../notebooks/) to understand the API usage.

8. To stop all running services, after making some [customizations](#next-steps)
    ```bash
    docker compose -f deploy/compose/docker-compose-ingestor-server.yaml down
    docker compose -f deploy/compose/docker-compose-rag-server.yaml down
    docker compose -f deploy/compose/vectordb.yaml down
    ```


## Deploy With Helm Chart

Use these procedures to deploy with Helm Chart to deploy on a Kubernetes cluster. Alternatively, you can [Deploy With Docker Compose](#deploy-with-docker-compose) for a single node deployment.


### Prerequisites

- Verify that you meet the [prerequisites](#prerequisites).

- Verify that you meet the hardware requirements.

  - The total GPU requirement for deploying this chart is as follows:
    - 8xH100-80GB
    - 9xA100-80GB
    - 8xB200

- Verify that you have the NGC CLI available on your client machine. You can download the CLI from <https://ngc.nvidia.com/setup/installers/cli>.

- Verify that you have Kubernetes installed and running Ubuntu 22.04. For more information, see [Kubernetes documentation](https://kubernetes.io/docs/setup/) and [NVIDIA Cloud Native Stack repository](https://github.com/NVIDIA/cloud-native-stack/).

- Verify that you have a default storage class available in the cluster for PVC provisioning.
  One option is the local path provisioner by Rancher.
  Refer to the [installation](https://github.com/rancher/local-path-provisioner?tab=readme-ov-file#installation)
  section of the README in the GitHub repository.

  ```console
  kubectl apply -f https://raw.githubusercontent.com/rancher/local-path-provisioner/v0.0.26/deploy/local-path-storage.yaml
  kubectl get pods -n local-path-storage
  kubectl get storageclass
  ```

- If the local path storage class is not set as default, it can be made default by using the following command.

  ```
  kubectl patch storageclass local-path -p '{"metadata": {"annotations":{"storageclass.kubernetes.io/is-default-class":"true"}}}'
  ```

- Verify that you have installed the NVIDIA GPU Operator following steps [here](https://docs.nvidia.com/datacenter/cloud-native/gpu-operator/latest/getting-started.html).

- Optionally you can also enable time slicing for sharing GPUs between pods. Refer to [this](https://docs.nvidia.com/datacenter/cloud-native/gpu-operator/latest/gpu-sharing.html) on GPU operator user guide for more details.

### Helm deployment

#### Core Services
- RAG server
- Ingestor server
- NV-Ingest

#### Setup

**Export NGC API Key**
   Refer to [this](#obtaining-the-ngc-api-key) for obtaining the API Key.

   ```sh
   export NGC_API_KEY="nvapi-*"
   ```

**Change directory to deploy/helm/**
   ```sh
   cd deploy/helm/
   ```

#### Deploying End to End RAG Server + Ingestor Server (NV-Ingest)

Create a namespace for the deployment:

```sh
kubectl create namespace rag
```

Run the following command to install the RAG server with the Ingestor Server and Frontend enabled:

```sh
helm upgrade --install rag -n rag https://helm.ngc.nvidia.com/nvstaging/blueprint/charts/nvidia-blueprint-rag-v2.3.0-rc2.tgz \
--username '$oauthtoken' \
--password "${NGC_API_KEY}" \
--set imagePullSecret.password=$NGC_API_KEY \
--set ngcApiSecret.password=$NGC_API_KEY
```

[!NOTE]
For B200 based deployment, you need to add an environment variable to the `nvidia-nim-llama-32-nv-embedqa-1b-v2` deployment.
Due to a known issue in the chart, we currently need to manually edit the deployment and add this env variable.

1. Edit the embedding deployment with the command below.

```bash
kubectl edit deployment 'rag-nvidia-nim-llama-32-nv-embedqa-1b-v2'  -n rag
```

2. Add the env variable `NIM_TRT_ENGINE_HOST_CODE_ALLOWED` and set its value to `1`.

```yaml
spec:
      containers:
      - env:
        - name: NIM_CACHE_PATH
          value: /opt/nim/.cache
        - name: NGC_API_KEY
          valueFrom:
            secretKeyRef:
              key: NGC_API_KEY
              name: ngc-api
        - name: NIM_TRT_ENGINE_HOST_CODE_ALLOWED
          value: "1"
```

3. Delete the NIM Embedding pod for the changes in the deployment to reflect.

```bash
kubectl delete pod <embedqa-pod-name>  -n rag
```


[!TIP]
For B200 and A100 GPUs, it is recommended to use CPU indexing and search for better response. You can set this by either:

1. Using helm set command:
```sh
helm upgrade --install rag -n rag https://helm.ngc.nvidia.com/nvstaging/blueprint/charts/nvidia-blueprint-rag-v2.3.0-rc2.tgz \
--username '$oauthtoken' \
--password "${NGC_API_KEY}" \
--set imagePullSecret.password=$NGC_API_KEY \
--set ngcApiSecret.password=$NGC_API_KEY \
--set ingestor-server.envVars.APP_VECTORSTORE_ENABLEGPUINDEX=False \
--set ingestor-server.envVars.APP_VECTORSTORE_ENABLEGPUSEARCH=False
```

2. Or by modifying values.yaml:
```yaml
ingestor-server:
  envVars:
    APP_VECTORSTORE_ENABLEGPUINDEX: "False"
    APP_VECTORSTORE_ENABLEGPUSEARCH: "False"
```

> **Note:** If you're using the source Helm chart, make these changes in `deploy/helm/nvidia-blueprint-rag/values.yaml`.

> [!NOTE]
> To deploy the Helm chart within 4xH100 using MIG slicing, see [RAG Deployment with MIG Support](./mig-deployment.md).


#### Deploying E2E From the Source (Optional)

Follow this section only if you are working directly with the source Helm chart and want to customize components individually.

**Change directory to deploy/helm/**
   ```sh
   cd deploy/helm/
   ```


**Create a namespace for the deployment:**

```sh
kubectl create namespace rag
```

##### Helm Repo Additions

```sh
helm repo add nvidia-nim https://helm.ngc.nvidia.com/nim/nvidia/ --username='$oauthtoken' --password=$NGC_API_KEY
helm repo add nim https://helm.ngc.nvidia.com/nim/ --username='$oauthtoken' --password=$NGC_API_KEY
helm repo add nemo-microservices https://helm.ngc.nvidia.com/nvidia/nemo-microservices --username='$oauthtoken' --password=$NGC_API_KEY
helm repo add baidu-nim https://helm.ngc.nvidia.com/nim/baidu --username='$oauthtoken' --password=$NGC_API_KEY
# TODO: remove after moving to GA NV-ingest helm chart
helm repo add nvstaging-nim  https://helm.ngc.nvidia.com/nvstaging/nim --username='$oauthtoken' --password=$NGC_API_KEY
helm repo add bitnami https://charts.bitnami.com/bitnami
helm repo add otel https://open-telemetry.github.io/opentelemetry-helm-charts
helm repo add zipkin https://zipkin.io/zipkin-helm
helm repo add prometheus https://prometheus-community.github.io/helm-charts
```

##### Updating Helm Chart Dependencies

```sh
helm dependency update nvidia-blueprint-rag
```

##### Install the chart

```sh
helm upgrade --install rag -n rag nvidia-blueprint-rag/ \
--set imagePullSecret.password=$NGC_API_KEY \
--set ngcApiSecret.password=$NGC_API_KEY
```

#### Changing NIM LLM Model

If you wish to switch the LLM for example to Llama-3.1-8b-instruct in this case, refer to the steps below.

Update the following in values.yaml and patch the deployment following the steps from [here](#patching-the-deployment).

```yaml
env:
  # ... existing code ...
  # LLM Model Config
  APP_LLM_MODELNAME: "meta/llama-3.1-8b-instruct"
nim-llm:
  # ... existing code ...
  image:
      repository: nvcr.io/nim/meta/llama-3.1-8b-instruct
      pullPolicy: IfNotPresent
      tag: "1.3.3"
  resources:
    limits:
      nvidia.com/gpu: 1
    requests:
      nvidia.com/gpu: 1
  # Configure NIM Model Profile for optimal performance
  env:
    - name: NIM_MODEL_PROFILE
      value: ""  # Empty for automatic selection, or specify tensorrt_llm profile
  model:
    ngcApiKey: ""
    modelName: "meta/llama-3.1-8b-instruct"
```

**List available profiles for your system:**

You can list available profiles by running the NIM container directly:

```bash
USERID=$(id -u) docker run --rm --gpus all \
  -v ~/.cache/model-cache:/opt/nim/.cache \
  nvcr.io/nim/nvidia/llama-3.3-nemotron-super-49b-v1.5:1.12.0 \
  list-model-profiles
```

Example output for H100-NVL when 1 GPU is allocated:

```output
MODEL PROFILES
- Compatible with system and runnable:
- d491076c9c5fbbc0f5ec92916ee84050c3b51a1c93055a64de8d0f31a22ed209 (vllm-bf16-tp1-pp1-32c3b968468aefcfb3ea1db5a16e3dc9d64395f02ef68a06175e8bbdb0038601)
- e2f00b2cbfb168f907c8d6d4d40406f7261111fbab8b3417a485dcd19d10cc98 (vllm)
- e759b32c9a5c4d16aced45cd7797ec030feb7d9164a7fe74278a76c795baf8cb (tensorrt_llm-h100_nvl-fp8-tp1-pp1-throughput-2321:10de-6343e21ba5cccf783d18951c6627c207b81803c3c45f1e8b59eee062ed350143-1)
- 668b575f1701fa70a97cfeeae998b5d70b048a9b917682291bb82b67f308f80c (tensorrt_llm)
- 50e138f94d85b97117e484660d13b6b54234e60c20584b1de6ed55d109ca4f21 (sglang)
```

#### NIM Model Profile Configuration

Set the required profile. It is preferable to select `tensorrt_llm-*` profiles for best performance. By default, automatic selection of profile is enabled but due to a known issue, vllm based profiles are selected, so it is recommended to manually select a tensorrt_llm profile.

**Example profiles for different hardware configurations:**

```yaml
# H100 NVL based
nim-llm:
  env:
    - name: NIM_MODEL_PROFILE
      value: "tensorrt_llm-h100_nvl-fp8-tp1-pp1-throughput-2321:10de-6343e21ba5cccf783d18951c6627c207b81803c3c45f1e8b59eee062ed350143-1"

# H100 SXM based
nim-llm:
  env:
    - name: NIM_MODEL_PROFILE
      value: "tensorrt_llm-h100-fp8-tp1-pp1-throughput-2330:10de-ed15592b3e4d174a719e8188493420073c39448d9b7ed742cfe614b96fecbdd9-1"

# 2xA100
nim-llm:
  env:
    - name: NIM_MODEL_PROFILE
      value: "tensorrt_llm-a100-bf16-tp2-pp1-throughput-20b2:10de-4c44baded168675f277ffcbd4f9bce56cac0239944062c44d81d359f9c718f06-2"

# RTX6000
nim-llm:
  env:
    - name: NIM_MODEL_PROFILE
      value: "tensorrt_llm-rtx6000_blackwell_sv-fp8-tp1-pp1-throughput-2bb5:10de-b5e4ef5f657564a90ff8b1fc8e8e6d59f24845f183209d563aa93d91fc6629ac-1"

# B200
nim-llm:
  env:
    - name: NIM_MODEL_PROFILE
      value: "tensorrt_llm-gb200-fp8-tp1-pp1-throughput-2941:10de-a642befc43aa112e87399e2fea6d3ef2994cadb5ccb2715c9bc7e89c35e92383-1"
```

#### Verifying Deployment

##### List Pods
```sh
kubectl get pods -n rag
```

##### Expected Output
```sh
NAME                                                        READY   STATUS    RESTARTS      AGE
ingestor-server-7bcff75fbb-s655f                            1/1     Running   0             23m
nv-ingest-paddle-0                                          1/1     Running   0             23m
rag-etcd-0                                                  1/1     Running   0             23m
rag-frontend-5d6c6dc4bd-5xpcw                               1/1     Running   0             23m
rag-milvus-standalone-5f5699dfb6-dzlhr                      1/1     Running   3 (23m ago)   23m
rag-minio-f88fb7fd4-29fxk                                   1/1     Running   0             23m
rag-nemoretriever-graphic-elements-v1-b6d465575-rl66q       1/1     Running   0             23m
rag-nemoretriever-page-elements-v2-596679ff54-z2kkf         1/1     Running   0             23m
rag-nemoretriever-table-structure-v1-748df88f86-z7mwb       1/1     Running   0             23m
rag-nim-llm-0                                               1/1     Running   0             23m
rag-nv-ingest-75cdb75c48-kbr7r                              1/1     Running   0             23m
rag-nvidia-nim-llama-32-nv-embedqa-1b-v2-5b6dc664d8-8flpd   1/1     Running   0             23m
rag-opentelemetry-collector-558b89885-c7c8j                 1/1     Running   0             23m
rag-redis-master-0                                          1/1     Running   0             23m
rag-redis-replicas-0                                        1/1     Running   0             23m
rag-server-7758bbf9bd-rw2wh                                 1/1     Running   0             23m
rag-text-reranking-nim-74c5f499cd-clcdg                     1/1     Running   0             23m
rag-zipkin-5dc8d6d977-nqvvc                                 1/1     Running   0             23m
```

> **Note:** It takes around 5 minutes for all pods to come up. Check K8s events using
   ```sh
   kubectl get events -n rag
   ```


##### List Services
```sh
kubectl get svc -n rag
```

##### Expected Output
```sh
NAME                                TYPE            EXTERNAL-IP   PORT(S)                                                   AGE
ingestor-server                     ClusterIP      <none>        8082/TCP                                                  26m
kubernetes                          ClusterIP      <none>        443/TCP                                                   4d20h
nemoretriever-embedding-ms                   ClusterIP      <none>        8000/TCP                                                  26m
nemoretriever-ranking-ms                     ClusterIP      <none>        8000/TCP                                                  26m
nemoretriever-graphic-elements-v1   ClusterIP      <none>        8000/TCP,8001/TCP                                         26m
nemoretriever-page-elements-v2      ClusterIP      <none>        8000/TCP,8001/TCP                                         26m
nemoretriever-table-structure-v1    ClusterIP      <none>        8000/TCP,8001/TCP                                         26m
nim-llm                             ClusterIP      <none>        8000/TCP                                                  26m
nim-llm-sts                         ClusterIP      <none>        8000/TCP                                                  26m
nv-ingest-paddle                    ClusterIP      <none>        8000/TCP,8001/TCP                                         26m
nv-ingest-paddle-sts                ClusterIP      <none>        8000/TCP,8001/TCP                                         26m
rag-etcd                            ClusterIP      <none>        2379/TCP,2380/TCP                                         26m
rag-etcd-headless                   ClusterIP      <none>        2379/TCP,2380/TCP                                         26m
rag-frontend                        NodePort       <none>        3000:31645/TCP                                            26m
rag-milvus                          ClusterIP      <none>        19530/TCP,9091/TCP                                        26m
rag-minio                           ClusterIP      <none>        9000/TCP                                                  26m
rag-nv-ingest                       ClusterIP      <none>        7670/TCP                                                  26m
rag-opentelemetry-collector         ClusterIP      <none>        6831/UDP,14250/TCP,14268/TCP,4317/TCP,4318/TCP,9411/TCP   26m
rag-redis-headless                  ClusterIP      <none>        6379/TCP                                                  26m
rag-redis-master                    ClusterIP      <none>        6379/TCP                                                  26m
rag-redis-replicas                  ClusterIP      <none>        6379/TCP                                                  26m
rag-server                          ClusterIP      <none>        8081/TCP                                                  26m
rag-zipkin                          ClusterIP      <none>        9411/TCP                                                  26m
```

#### Patching the deployment
For patching an existing deployment, modify `values.yaml` with required changes and run
```sh
helm upgrade --install rag -n rag https://helm.ngc.nvidia.com/nvstaging/blueprint/charts/nvidia-blueprint-rag-v2.3.0-rc2.tgz \
--username '$oauthtoken' \
--password "${NGC_API_KEY}" \
--set imagePullSecret.password=$NGC_API_KEY \
--set ngcApiSecret.password=$NGC_API_KEY \
-f nvidia-blueprint-rag/values.yaml
```

#### Enable NIM Operator with the chart

- Prerequisites
   - Please [Install NIM Operator 3.0](https://docs.nvidia.com/nim-operator/latest/install.html).
   - Ensure you meet [the hardware requirements](./support-matrix.md).

1. Create ImagePullSecrets with below commands

   ```sh
   kubectl create secret -n rag docker-registry ngc-secret \
   --docker-server=nvcr.io \
   --docker-username='$oauthtoken' \
   --docker-password=$NGC_API_KEY

   kubectl create secret -n rag generic ngc-api-secret \
   --from-literal=NGC_API_KEY=$NGC_API_KEY
   ```

2. Create a NIM Cache with available storage class on the cluster.

  ```sh
  kubectl apply -f deploy/helm/nim-operator/rag-nimcache.yaml -n rag
  ```

3. Now create a NIM Services

   - **No GPU Sharing**

      ```sh
      kubectl apply -f deploy/helm/nim-operator/rag-nimservice.yaml -n rag
      ```

   - **GPU Sharing with DRA**

      > [!TIP] With DRA Setup, All NIM Service can run on 3 GPUs with atleast 80GB should be enough, it could be A100 or H100 or B200

      - Prerequisite: [NVIDIA DRA Driver](https://docs.nvidia.com/datacenter/cloud-native/gpu-operator/25.3.2/dra-intro-install.html)
   
      ```sh
      kubectl apply -f deploy/helm/nim-operator/rag-nimservice-dra.yaml -n rag
      ```

   - **(Optional) List available profiles for your system for NIM LLM container**

      More details about profiles can be found [here](https://docs.nvidia.com/nim/large-language-models/latest/profiles.html).

      You can list available profiles by running the NIM container directly:

      ```bash
      USERID=$(id -u) docker run --rm --gpus all \
      -v ~/.cache/model-cache:/opt/nim/.cache \
      nvcr.io/nim/nvidia/llama-3.3-nemotron-super-49b-v1.5:1.12.0 \
      list-model-profiles
      ```

   - **(Optional) Configure NIM Model Profile**

      For optimal performance, configure the NIM model profile. See the [NIM Model Profile Configuration](#nim-model-profile-configuration) section in the "Changing NIM LLM Model" part for detailed instructions and hardware-specific examples.

      Configure the `NIM_MODEL_PROFILE` in `deploy/helm/nim-operator/rag-nimservice.yaml`:

      ```yaml
      storage:
      nimCache:
         name: nemotron-llama3-49b-super
         profile: ''
      env:
      - name: NIM_MODEL_PROFILE
        value: "tensorrt_llm-h100_nvl-fp8-tp1-pp1-throughput-2321:10de-6343e21ba5cccf783d18951c6627c207b81803c3c45f1e8b59eee062ed350143-1"  # Example for H100 NVL
         # H100 tp 1 profile non-NVL based
         # - name: NIM_MODEL_PROFILE
         #   value: "tensorrt_llm-h100-fp8-tp1-pp1-throughput-2330:10de-ed15592b3e4d174a719e8188493420073c39448d9b7ed742cfe614b96fecbdd9-1"
         # NVL based H100 tp 1 profile
         # - name: NIM_MODEL_PROFILE
         #   value: "tensorrt_llm-h100_nvl-fp8-tp1-pp1-throughput-2321:10de-6343e21ba5cccf783d18951c6627c207b81803c3c45f1e8b59eee062ed350143-1"
         # A100 tp 2 profile
         # - name: NIM_MODEL_PROFILE
         #   value: "tensorrt_llm-a100-bf16-tp2-pp1-throughput-20b2:10de-4c44baded168675f277ffcbd4f9bce56cac0239944062c44d81d359f9c718f06-2"
      ```

      After modifying the profile, reapply the NIM service:

      ```sh
      kubectl apply -f deploy/helm/nim-operator/rag-nimservice.yaml -n rag
      ```

4. Wait a few minutes and ensure that the NIMService status is `Ready` before proceeding to the next steps.

   ```sh
   kubectl get nimservice -n rag
   NAME                                STATUS     AGE
   nemoretriever-embedding-ms          Ready      20m
   nemoretriever-reranking-ms          Ready      20m
   nemoretriever-graphic-elements-v1   Ready      20m
   nemoretriever-page-elements-v2      Ready      20m
   nemoretriever-table-structure-v1    Ready      20m
   nim-llm                             Ready      20m

7. Delete the existing secret as it's conflict with RAG installation

  ```sh
  kubectl delete secret ngc-secret -n rag
  ```

8. Use the following `values-nim-operator.yaml` to deploy the RAG with NIM Operator NIM Services

  ```sh
  helm upgrade --install rag -n rag https://helm.ngc.nvidia.com/nvstaging/blueprint/charts/nvidia-blueprint-rag-v2.3.0-rc2.tgz \
  --username '$oauthtoken' \
  --password "${NGC_API_KEY}" \
  --set imagePullSecret.password=$NGC_API_KEY \
  --set ngcApiSecret.password=$NGC_API_KEY \
  -f deploy/helm/nim-operator/values-nim-operator.yaml
  ```


#### Port-Forwarding to Access UIs

- Frontend:

  Run the following command to port-forward the Forntend service to your local machine:

  ```sh
  kubectl port-forward -n rag service/rag-frontend 3000:3000 --address 0.0.0.0
  ```

  Access the Frontend UI at `http://localhost:3000`.

- Zipkin UI:

  Run the following command to port-forward the Zipkin service to your local machine:

  ```sh
  kubectl port-forward -n rag service/rag-zipkin 9411:9411 --address 0.0.0.0
  ```

  Access the Zipkin UI at `http://localhost:9411`.

- Grafana UI:

  Run the following command to port-forward the Grafana service:

  ```sh
  kubectl port-forward -n rag service/rag-grafana 3001:80 --address 0.0.0.0
  ```

  Access the Grafana UI at `http://localhost:3001` using the default credentials (`admin`/`admin`).

   #### Creating a Dashboard in Grafana

   1. Upload JSON to Grafana:

      - Navigate to the Grafana UI at `http://localhost:3000`.
      - Log in with the default credentials (`admin`/`admin`).
      - Go to the "Dashboards" section and click on "Import".
      - Upload the JSON file located in the `deploy/config` directory.

   2. Configure the Dashboard:

      - After uploading, select the data source that the dashboard will use. Ensure that the data source is correctly configured to pull metrics from your Prometheus instance.

   3. Save and View:

      - Once the dashboard is configured, save it and start viewing your metrics and traces.

#### Cleanup

To uninstall the deployment, run:
```sh
helm uninstall rag -n rag
```

#### (Optional) Deploying Standalone RAG Server

Run the following command to install the RAG Server:

```sh
helm upgrade --install rag https://helm.ngc.nvidia.com/nvstaging/blueprint/charts/nvidia-blueprint-rag-v2.3.0-rc2.tgz -n rag \
  --username '$oauthtoken' \
  --password "${NGC_API_KEY}" \
  --set imagePullSecret.password=$NGC_API_KEY \
  --set nvidia-nim-llama-32-nv-embedqa-1b-v2.nim.ngcAPIKey=$NGC_API_KEY \
  --set text-reranking-nim.nim.ngcAPIKey=$NGC_API_KEY \
  --set nim-llm.model.ngcAPIKey=$NGC_API_KEY \
  --set ingestor-server.enabled=false
```


#### (Optional) Enabling persistence for NIMs

- For enabling persistence for NIM LLM refer to the [NIM LLM](https://docs.nvidia.com/nim/large-language-models/latest/deploy-helm.html#storage) documentation.
   Update the required fields in `values.yaml` file for `nim-llm` section.

- For enabling persistence for Nemo Retriever embedding [Nemo Retriever Text Embedding](https://docs.nvidia.com/nim/nemo-retriever/text-embedding/latest/deploying.html#storage) documentation.
   Update the required fields in `values.yaml` file for `nvidia-nim-llama-32-nv-embedqa-1b-v2` section.

- For enabling persistence for Nemo Retriever reranking [Nemo Retriever Text Reranking](https://docs.nvidia.com/nim/nemo-retriever/text-reranking/latest/deploying.html#storage) documentation.
   Update the required fields in `values.yaml` file for `text-reranking-nim` section.

#### (Optional) Customizing Milvus Endpoint

To use a custom Milvus endpoint, you need to update the `APP_VECTORSTORE_URL` environment variable in the `values.yaml` file for both the RAG server and the Ingestor server. Follow these steps:

1. **Edit `values.yaml`:**

   Open the `deploy/helm/nvidia-blueprint-rag/values.yaml` file and update the `APP_VECTORSTORE_URL` and `MINIO_ENDPOINT` for both the RAG server and the Ingestor server sections:

   ```yaml
   env:
     # ... existing code ...
     APP_VECTORSTORE_URL: "http://your-custom-milvus-endpoint:19530"
     MINIO_ENDPOINT: "http://your-custom-minio-endpoint:9000"
     # ... existing code ...

   ingestor-server:
     env:
       # ... existing code ...
       APP_VECTORSTORE_URL: "http://your-custom-milvus-endpoint:19530"
       MINIO_ENDPOINT: "http://your-custom-minio-endpoint:9000"
       # ... existing code ...

   nv-ingest:
     envVars:
       # ... existing code ...
       MINIO_INTERNAL_ADDRESS: "http://your-custom-minio-endpoint:9000"
       # ... existing code ...
   ```

2. **Disable Milvus Deployment:**

   Set `milvusDeployed: false` in the `ingestor-server.nv-ingest` section to prevent deploying the default Milvus instance:

   ```yaml
   ingestor-server:
     nv-ingest:
       # ... existing code ...
       milvusDeployed: false
       # ... existing code ...
   ```

3. **Deploy the Changes:**

   Redeploy the Helm chart to apply these changes:

   ```sh
   helm upgrade rag https://helm.ngc.nvidia.com/nvstaging/blueprint/charts/nvidia-blueprint-rag-v2.3.0-rc2.tgz -f nvidia-blueprint-rag/values.yaml -n rag
   ```

#### (Optional) Customizing the RAG Server UI

Currently, Frontend doesn't support dynamic variables.
The default variables and values are the following:

   ```
    - name: VITE_API_CHAT_URL
      value: "http://rag-server:8081/v1"
    - name: VITE_API_VDB_URL
      value: "http://ingestor-server:8082/v1"
    - name: VITE_MILVUS_URL
      value: "http://milvus:19530"
   ```

  If you have a plan to customize the RAG server deployment like LLM Model Change then please follow the steps to deploy the Frontend

  - Build the new docker image with updated model name from docker compose

    ```
    cd ../deploy/compose
    ```

    Modify the `image` and `args` accordingly in `docker-compose-rag-server.yaml` for `rag-frontend` service

    Example:

    ```
    # Sample UI container which interacts with APIs exposed by rag-server container
    rag-frontend:
      container_name: rag-frontend
      image: <image-registry-with-tag>
      build:
        # Set context to repo's root directory
        context: ../../frontend
        dockerfile: ./Dockerfile
        args:
          VITE_API_CHAT_URL: "http://rag-server:8081/v1"
          # URL for ingestor container
          VITE_API_VDB_URL: "http://ingestor-server:8082/v1"
      ports:
        - "8090:3000"
      expose:
        - "3000"
      depends_on:
        - rag-server
    ```

    Run the below command to create a docker image

    ```
    docker compose -f docker-compose-rag-server.yaml build --no-cache
    ```

    Once docker image has been build to push the image to a docker a registry

   - Run the following command to install the RAG server with the Ingestor Server and New Frontend with updated `<new-image-repository>` and `<new-image-tag>`:

      ```sh
      helm install rag -n rag https://helm.ngc.nvidia.com/nvstaging/blueprint/charts/nvidia-blueprint-rag-v2.3.0-rc2.tgz \
      --username '$oauthtoken' \
      --password "${NGC_API_KEY}" \
      --set imagePullSecret.password=$NGC_API_KEY \
      --set ngcApiSecret.password=$NGC_API_KEY
      --set frontend.image.repository='<new-image-repository>' \
      --set frontend.image.tag="<new-image-tag>" \
      --set frontend.imagePullSecret.password="$NGC_API_KEY"
      ```

### Troubleshooting Helm Issues
For troubleshooting issues with Helm deployment, checkout the troubleshooting section [here](./troubleshooting.md#node-exporter-pod-crash-with-prometheus-stack-enabled-in-helm-deployment).

## Data Ingestion

[!IMPORTANT]
Before you can use this procedure, you must deploy the blueprint by using [Deploy With Docker Compose](#deploy-with-docker-compose) or [Deploy With Helm Chart](#deploy-with-helm-chart).


1. Download and install Git LFS by following the [installation instructions](https://git-lfs.com/).

2. Initialize Git LFS in your environment.

   ```bash
   git lfs install
   ```

3. Pull the dataset into the current repo.

   ```bash
   git-lfs pull
   ```

4. Install jupyterlab.

   ```bash
   pip install jupyterlab
   ```

5. Use this command to run Jupyter Lab so that you can execute this IPython notebook.

   ```bash
   jupyter lab --allow-root --ip=0.0.0.0 --NotebookApp.token='' --port=8889
   ```

6. Run the [ingestion_api_usage](../notebooks/ingestion_api_usage.ipynb) notebook.

Follow the cells in the notebook to ingest the PDF files from the data/dataset folder into the vector store.

> [!TIP]
> **Important Configuration Tip**
>
> Check out the [best practices guide](accuracy_perf.md) especially the **Vector Store Retriever Consistency Level** section to configure the required settings before starting the ingestion/retrieval process based on your use case.



## Next Steps

- [Change the Inference or Embedding Model](change-model.md)
- [Change the vector database](docs/change-vectordb.md)
- [Customize Prompts](prompt-customization.md)
- [Customize LLM Parameters at Runtime](llm-params.md)
- [Support Multi-Turn Conversations](multiturn.md)
- [Enable NeMo Guardrails for Content Safety](nemo-guardrails.md)
- [Enable Query decomposition support](docs/query_decomposition.md)
- [Query Across Multiple Collections](multi-collection-retrieval.md)
- [Troubleshoot NVIDIA RAG Blueprint](troubleshooting.md)
- [Understand latency breakdowns and debug errors using observability services](observability.md)
- [Enable Self-Reflection to improve accuracy](self-reflection.md)
- [Enable Query rewriting to Improve accuracy of Multi-Turn Conversations](query_rewriter.md)
- [Enable Image captioning support for ingested documents](image_captioning.md)
- [Enable VLM embedding model](vlm-embed.md)
- [Enable hybrid search for milvus](hybrid_search.md)
- [Add custom metadata while uploaded documents](custom-metadata.md)
- [Enable low latency, low compute text only pipeline](text_only_ingest.md)
- [Enable VLM based inferencing in RAG](vlm.md)
- [Multimodal Input Guide RAG](multimodal_input_guide.md)
- [Enable PDF extraction with Nemoretriever Parse](nemoretriever-parse-extraction.md)
- [Enable NeMo Retriever OCR for enhanced text extraction](docs/nemoretriever-ocr.md)
- [Enable standalone NV-Ingest for direct document ingestion without ingestor server](nv-ingest-standalone.md)
- [Mount ingestor server volume for direct filesystem access to extraction results](mount-ingestor-volume.md)
- [Filter documents by confidence threshold](python-client.md#search-with-confidence-threshold-filtering) in search and generation
- Explore [best practices for enhancing accuracy or latency](accuracy_perf.md)
- Explore [migration guide](migration_guide.md) if you are migrating from rag v1.0.0 to this version.
- [Change the Vector Database Backend](change-vectordb.md)



> **⚠️ Important B200 Limitation Notice:**
>
> B200 GPUs are **not supported** for the following advanced features:
> - Image captioning support for ingested documents
> - NeMo Guardrails for guardrails at input/output
> - VLM based inferencing in RAG
> - PDF extraction with Nemoretriever Parse
> - Poor retrieval accuracy is observed with Milvus GPU indexing and search.
>
> For these features, please use H100 or A100 GPUs instead.
