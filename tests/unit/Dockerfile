# Use Python 3.12 base image
FROM python:3.12-slim

# Set working directory
WORKDIR /workdir

# Install system dependencies if needed
RUN apt-get update && apt-get install -y \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements files first for better caching
COPY ./tests/requirements-test.txt requirements-test.txt

# Copy project files
COPY . /workdir/

# Install the package with all optional dependencies first
RUN pip install -e .[all]

# Install test-specific dependencies last
RUN pip install --no-cache-dir -r requirements-test.txt

# Set default command for running tests
CMD ["python", "-m", "pytest", "-v", "-s", "--cov=src", "--cov-report=term-missing", "tests/"]
