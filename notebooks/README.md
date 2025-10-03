# Build a Confidential Enterprise RAG - API Interaction and Deployment Notebooks

## Overview
This repository contains Jupyter notebooks demonstrating the usage of Confidential Enterprise RAG APIs and advanced development features.

### Notebooks:

#### **Core API Usage Notebooks**
1. **`ingestion_api_usage.ipynb`**: Demonstrates how to interact with the Confidential Enterprise RAG ingestion service, showcasing how to upload and process documents for retrieval-augmented generation (RAG).

2. **`retriever_api_usage.ipynb`**: Illustrates the use of the Confidential Enterprise RAG retriever service, highlighting different querying techniques and retrieval strategies.

3. **`rag_library_usage.ipynb`**: Demonstrates native usage of the Confidential Enterprise RAG Python client, including environment setup, document ingestion, collection management, and querying. This notebook provides end-to-end API usage examples for interacting directly with the RAG system from Python, covering both ingestion and retrieval workflows.

#### **Advanced Features Notebooks**
4. **`nb_metadata.ipynb`**: **Comprehensive metadata functionality demonstration** - This notebook showcases advanced metadata features including metadata ingestion, filtering, and extraction. It provides step-by-step examples of how to use metadata for enhanced document retrieval and Q&A capabilities. Perfect for users wanting to implement sophisticated metadata-based filtering in their RAG applications.

5. **`building_rag_vdb_operator.ipynb`**: **Advanced Developer Guide** - Learn how to create and integrate custom vector database (VDB) operators with the Confidential Enterprise RAG. This notebook demonstrates building a complete OpenSearch VDB operator from scratch, understanding the VDBRag base class architecture, and implementing custom vector database integrations. Essential for developers wanting to extend the Confidential Enterprise RAG with their own vector database implementations.

#### **Deployment Notebooks**
6. **`launchable.ipynb`**: A deployment-ready notebook intended for execution within the brev.dev environment.

## Setting Up the Environment
To run these notebooks in a Python virtual environment, follow the steps below:

### 1. Create and Activate a Virtual Environment
```bash
python3 -m virtualenv venv
source venv/bin/activate
```

### 2. Install Dependencies
Ensure you have JupyterLab and required dependencies installed:
```bash
pip3 install jupyterlab
```

### 3. Start JupyterLab
Run the following command to start JupyterLab, allowing access from any IP:
```bash
jupyter lab --allow-root --ip=0.0.0.0 --NotebookApp.token='' --port=8889 --no-browser
```

Once running, you can access JupyterLab by navigating to `http://<your-server-ip>:8889` in your browser.

## Running the Notebooks
- Open JupyterLab in your browser.
- Navigate to the desired notebook and run the cells sequentially.

## Notebook Categories

### **For Beginners**
Start with `ingestion_api_usage.ipynb` and `retriever_api_usage.ipynb` to understand basic API interactions.

### **For Intermediate Users**
Use `rag_library_usage.ipynb` for comprehensive Python client usage and `nb_metadata.ipynb` for advanced metadata features.

### **For Advanced Developers**
Use `building_rag_vdb_operator.ipynb` to learn how to extend the system with custom vector database implementations.

### **For Deployment**
Use `launchable.ipynb` for cloud deployment scenarios.

## Deployment (Brev.dev)
For deploying `launchable.ipynb` in [brev.dev](https://console.brev.dev/environment/new), follow the platform's instructions for executing Jupyter notebooks within a cloud-based environment selected based on the hardware requirements specified in the launchable.

## Notes
- Ensure API keys and credentials are correctly set up before making API requests.
- Modify endpoints or request parameters as necessary to align with your specific use case.
- For the custom VDB operator notebook, ensure Docker is available for running OpenSearch services.

