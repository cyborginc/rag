# CI of Unit Tests in Gitlab Using Jenkins

## Table of Contents
- [Getting Started](#getting-started)
- [Local Development Setup](#local-development-setup)
- [Getting Jenkins Instance](#getting-jenkins-instance)
- [Setting up Unit Tests in Jenkins Pipeline](#setting-up-unit-tests-in-jenkins-pipeline)
  - [Setting up the pipeline in Jenkins](#setting-up-the-pipeline-in-jenkins)
  - [Creating Docker Image and Jenkinsfile](#creating-a-docker-image-and-jenkinsfile-in-the-source-code-to-run-the-tests)
  - [Running the Pipeline](#running-the-pipeline)
  - [Setting up Gitlab CI](#setting-up-gitlab-ci-for-the-repository)
- [Adding New Testcases](#adding-new-testcases)
- [References](#references)

## Getting Started

Blossom is the provider for the Jenkins instance. Blossom is now offering CloudBees Jenkins as a service, an enterprise-grade CI/CD solution. This Jenkins environment is hosted within NVIDIA's infrastructure and offers seamless access to Blossom worker nodes and supporting infrastructure.

## Local Development Setup

### Prerequisites
- Python 3.12+
- uv (recommended) or pip

### Quick Setup for Local Development

**Option 1: Simple Setup (Recommended for new users)**
```bash
# Create and activate virtual environment
uv venv
source .venv/bin/activate  # On Linux/Mac
# or
.venv\Scripts\activate     # On Windows

# Install everything at once
uv pip install -r tests/unit/requirements-dev.txt
# or
pip install -r tests/unit/requirements-dev.txt

# Run tests
pytest tests/unit/ -v
```

**Option 2: Step-by-step Setup**
1. **Create and activate a virtual environment:**
   ```bash
   # Using uv (recommended)
   uv venv
   source .venv/bin/activate  # On Linux/Mac
   # or
   .venv\Scripts\activate     # On Windows

   # Or using venv
   python3 -m venv .venv
   source .venv/bin/activate  # On Linux/Mac
   # or
   .venv\Scripts\activate     # On Windows
   ```

2. **Install the package with all optional dependencies:**
   ```bash
   # Using uv
   uv pip install -e .[all]

   # Or using pip
   pip install -e .[all]
   ```

3. **Install test-specific dependencies:**
   ```bash
   # Using uv
   uv pip install -r tests/unit/requirements-test.txt

   # Or using pip
   pip install -r tests/unit/requirements-test.txt
   ```

4. **Run the tests:**
   ```bash
   pytest tests/unit -v
   ```

### Important Notes for Local Development

- **Always install the package with `[all]` optional dependencies first** before installing test requirements
- The test requirements file (`requirements-test.txt`) only contains testing-specific dependencies
- Main package dependencies are managed through `pyproject.toml` and should be installed via `pip install -e .[all]`
- If you encounter "ModuleNotFoundError: No module named 'nvidia_rag'", make sure you've installed the package in editable mode first

### Troubleshooting Local Development

1. **ModuleNotFoundError: No module named 'nvidia_rag'**
   - Solution: Install the package first with `pip install -e .[all]`

2. **Import errors for langchain or opentelemetry packages**
   - Solution: These are included in the `[all]` optional dependencies, make sure to install them

3. **Test failures due to missing dependencies**
   - Solution: Ensure both the package and test requirements are installed in the correct order

4. **MinIO connection errors in tests**
   - Solution: The tests use mocks for external services like MinIO, so these errors should not occur in properly configured tests

### Running Tests with Coverage

```bash
pytest tests/unit -v --cov=src --cov-report=term-missing --cov-report=html:coverage_report
```

### Running Specific Test Files

```bash
# Run specific test file
pytest tests/unit/test_server.py -v

# Run specific test class
pytest tests/unit/test_ingestor_server/test_ingestor_server.py::TestHealthEndpoint -v

# Run specific test method
pytest tests/unit/test_ingestor_server/test_ingestor_server.py::TestHealthEndpoint::test_health_check -v
```

## Getting Jenkins Instance

1. Request Access:
   - Join blossom-cust-group-admins DL through [DL Request Portal](https://dlrequest/GroupID/Search/QuickSearch?searchValue=blossom-cust-group-admins&searchType=Contact,%20Group,%20User)

2. File NV Bug to request a new CloudBees Jenkins instance: ([NV Bug Templates](https://confluence.nvidia.com/display/BLOS/How+to+get+support))
   - Module: `IPP Blossom - Support`
   - Category: `CloudBees`
   - Required Information:
     * Team PiC contact information
     * Team DL
     * Tenancy name (format: <org>-<team>-<sub-team>), for example: ipp-blossom-dev
     * Org3 First and last name
     * Windows jobs requirement (Yes/No)
     * Startfleet Admin Portal ID (SSA ID). (if not available, create a new ID for your project using [SSA Admin Portal Login](https://admin.login.nvidia.com/login))

3. Wait for Confirmation:
   - Blossom team will create your Jenkins controller
   - You'll receive Jenkins URL and access details

## Setting up Unit Tests in Jenkins Pipeline

### Setting up the pipeline in Jenkins

- Open the Jenkins instance URL shared with you.
- Setup gitlab connection in Jenkins.
    - Go to `Manage Jenkins` -> `Manage Credentials` -> `Global Credentials` -> `Add Credentials`.
    - Add a new credential with the following details:
        - Kind: `API Token`
        - Token: `your-gitlab-api-token` ([How to get a Gitlab API token](https://confluence.nvidia.com/display/BLOS/Git+access+in+pipeline+jobs))
        - ID: `gitlab`
        - Description: `Gitlab API token`
    - Add another credential with the following details:
        - Kind: `Username with password`
        - Username: `your-gitlab-username`
        - Password: same `API Token`
- Create a new pipeline job.
    - In the Jenkins instance, go to `New Item` -> `Pipeline` -> OK
    - Enter the job name, for example: `CI-Unit-Tests`
    - Goto Configuration
        - In the Gitlab Connection section, select the `gitlab` credential with API Token you created in the previous step.
        - In the Pipeline section, select `Pipeline script from SCM`
            - Select `Pipeline script from SCM`
            - Select `Git` for the SCM.
            - Enter the repository URL.
            - Select the credentials with `username and password` you created in the previous step.
            - Select `Branch`
            - Enter the `Jenkinsfile` path.
            - If the repository has LFS files, select `Additional Behaviours` -> `Git LFS pull after checkout`.
        - Save


### Creating a docker image and Jenkinsfile in the source code to run the tests

- Create tests/unit folder with all the unit test files, Dockerfile with other dependencies and a Jenkinsfile in the following structure:

    ```
    your-project/
    ├── src/
    ├── tests/unit
    │   ├── Dockerfile
    │   ├── requirements-test.txt
    │   └── test_*.py
    ├── Jenkinsfile
    └── requirements.txt
    ```
- Create a `Dockerfile` to build a docker image that will run the tests as shown in the example below.

    Example Dockerfile:

    ```dockerfile
    FROM python:3.12-slim

    # Set working directory
    WORKDIR /workdir

    # Install system dependencies if needed
    RUN apt-get update && apt-get install -y \
        gcc \
        && rm -rf /var/lib/apt/lists/*

    # Copy requirements files
    COPY requirements.txt requirements.txt
    COPY ./tests/unit/requirements-test.txt requirements-test.txt

    # Copy project files first
    COPY . /workdir/

    # Install the package with all optional dependencies first
    RUN pip install --no-cache-dir -r requirements.txt
    RUN pip install -e .[all]

    # Install test-specific dependencies last
    RUN pip install --no-cache-dir -r requirements-test.txt
    ```
- Make sure the tests are executable by building the docker image and running the tests.

- Create a `Jenkinsfile` to define the pipeline as shown in the example below.

    Example Jenkinsfile:

    ```groovy
    pipeline {
        agent {
            kubernetes {
                yaml '''
    apiVersion: v1
    kind: Pod
    spec:
    containers:
    - name: docker
        image: docker:dind
        securityContext:
        privileged: true
        volumeMounts:
        - name: dind-storage
        mountPath: /var/lib/docker
    - name: git
        image: bitnami/git:latest
        command:
        - cat
        tty: true
    volumes:
    - name: dind-storage
        emptyDir: {}
                '''
                defaultContainer 'git'
            }
        }

        triggers {
            gitlab(
                triggerOnPush: true,
                triggerOnMergeRequest: true,
                branchFilterType: 'NameBasedFilter',
                includeBranchesSpec: 'main'
            )
        }
        options {
            gitLabConnection('gitlab')
            gitlabBuilds(builds: ['build'])
        }
        environment {
            DOCKER_IMAGE_NAME = 'python-test-image'
            TEST_RESULTS_DIR = 'test-results'
        }
        stages {

            stage('Build Test Image') {
                steps {
                    container('docker') {
                        script {
                            sh "docker build -t \${DOCKER_IMAGE_NAME} -f tests/unit/Dockerfile ."
                        }
                    }
                }
            }

            stage('Run Tests') {
                steps {
                    container('docker') {
                        script {
                            try {
                                sh '''
                                    mkdir -p ${WORKSPACE}/test-results
                                    docker run -v ${WORKSPACE}:/app/workspace \${DOCKER_IMAGE_NAME} python -m pytest -v -s \
                                    --cov=src \
                                    --junitxml=/app/workspace/test-results/test-results.xml \
                                    tests/unit/
                                '''
                            } finally {
                                // Use relative paths from workspace
                                junit allowEmptyResults: true, testResults: "**/${TEST_RESULTS_DIR}/test-results.xml"
                            }
                        }
                    }
                }
            }
        }

        post {
            success {
                updateGitlabCommitStatus name: 'build', state: 'success'
            }
            failure {
                updateGitlabCommitStatus name: 'build', state: 'failed'
            }
            always {
                script {
                    container('docker') {
                        sh '''
                            # Remove any containers using the image
                            docker ps -a | grep \${DOCKER_IMAGE_NAME} | awk '{print $1}' | xargs -r docker rm -f
                            # Force remove the image
                            docker rmi -f \${DOCKER_IMAGE_NAME} || true
                        '''
                    }
                }
            }
        }
    }
    ```


### Running the pipeline

- Push a change to the repository.
- The pipeline will still not trigger automatically as gitlab is not configured to trigger the pipeline.
- Go to the Jenkins instance and manually trigger the pipeline.
- The pipeline will build the docker image and run the tests.
- The results will be shown in the Jenkins instance.

### Setting up Gitlab CI for the repository

- Navigate to your GitLab project
- Go to Settings → Webhooks (not Integration)
- Add a new webhook with the following details:
   - URL: `<Jenkins-URL>/project/<Project-Name>`
   - Secret Token: Generate a secure token
   - Select triggers: Push events, Merge request events
   - SSL verification: Enable
- Test the webhook to ensure it's properly configured
- The pipeline will now trigger automatically on push and merge requests


## Adding New Testcases

1. Create a new test file in the tests/unit/ folder:
   - Name the file `test_<file_name>.py` to match your source file
   - Example: `test_utils.py` for testing `utils.py`

2. In Cursor, gather the necessary context:
   - Add your source file (e.g., `utils.py`) using the `+` button
   - Add `tests/unit/test_server.py` as a reference for test structure

3. Give Cursor these specific instructions:
   ```
   Create a pytest unit test file for [your source file] following these requirements:
   1. Use the structure and patterns from test_server.py
   2. Mock these specific dependencies:
      - List each external dependency that needs mocking
      - Example: requests, boto3, database connections
   3. Include tests for the main functions/methods
   4. Add appropriate fixtures and setup/teardown if needed
   ```

4. Validate and iterate:
   - Review the generated test file
   - Run the tests locally:
     ```bash
     pytest tests/unit/test_<file_name>.py -v -s
     ```
   - If tests fail, provide Cursor with:
     - The exact error message
     - Relevant parts of your source file
     - Ask for specific fixes

5. Add more test coverage:
   - Identify untested scenarios
   - Ask Cursor to add specific test cases:
     ```
     Add test cases for these scenarios in test_<file_name>.py:
     1. [Describe specific scenario]
     2. [Describe edge case]
     3. [Describe error condition]
     ```

6. Finalize:
   - Commit and push your changes
   - The Jenkins pipeline will automatically run all tests
   - Review the test results in Jenkins

Note: Always ensure your test file follows the project's existing testing patterns and includes appropriate mocking of external dependencies.

## References

- [Blossom User Documentation](https://confluence.nvidia.com/pages/viewpage.action?spaceKey=BLOS&title=Blossom+User+Documentation#tab-Tutorials)
- [Customer SOP Cloudbees](https://confluence.nvidia.com/pages/viewpage.action?spaceKey=BLOS&title=Customer+SOP+-+Cloudbees)
- [Using Blossom for CI](https://confluence.nvidia.com/pages/viewpage.action?spaceKey=BLOS&title=Using+Blossom+for+CI)