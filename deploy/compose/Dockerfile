# Use miniconda3 with Python 3.12
FROM continuumio/miniconda3:latest

# Accept build argument for version
ARG GIT_VERSION=0.11.1-dev

# Set working directory
WORKDIR /app

# Set environment variables
ENV PYTHONPATH=/app
ENV KMP_DUPLICATE_LIB_OK=TRUE
ENV PYTHONUNBUFFERED=1

# Update system packages and install necessary tools
RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    g++ \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy environment file first for better caching
COPY environment.yml .

# Create conda environment (use Python 3.12 to match environment.yml)
RUN conda env create -n cyborgdb-service -f environment.yml && \
    conda clean -afy

# Make RUN commands use the new environment
SHELL ["conda", "run", "-n", "cyborgdb-service", "/bin/bash", "-c"]

# Debug: Check Python version and environment
RUN echo "=== Python Version Check ===" && \
    python --version && \
    echo "=== Environment Info ===" && \
    conda info --envs && \
    echo "=== PyTorch Info ===" && \
    python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CPU available: {torch.backends.mkldnn.is_available() or True}')"

# Copy the entire project
COPY . .

# Copy and set up entrypoint script
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

# Set version for setuptools-scm using the build arg
ENV SETUPTOOLS_SCM_PRETEND_VERSION_FOR_CYBORGDB_SERVICE=${GIT_VERSION}

# Debug: Show the version being used
RUN echo "Building with version: ${GIT_VERSION}"

# Create a non-root user for security
RUN useradd -m -u 1000 cyborguser && chown -R cyborguser:cyborguser /app
USER cyborguser

# Install the application package
RUN pip install . && \
    echo "Checking installed scripts:" && \
    ls -la /opt/conda/envs/cyborgdb-service/bin/ | grep cyborg || echo "No cyborgdb scripts found"

# Expose the port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/v1/health || exit 1

# Set default environment variables (will require override at runtime)
ENV CYBORGDB_API_KEY=""
ENV CYBORGDB_DB_TYPE=""
ENV CYBORGDB_CONNECTION_STRING=""
ENV CYBORGDB_PORT="8000"

# Add labels for documentation
LABEL org.opencontainers.image.title="CyborgDB Service"
LABEL org.opencontainers.image.description="CyborgDB service with sentence transformers"
LABEL org.opencontainers.image.documentation="Set CYBORGDB_API_KEY environment variable to run"

# Run the application with helpful entrypoint
ENTRYPOINT ["/entrypoint.sh"]