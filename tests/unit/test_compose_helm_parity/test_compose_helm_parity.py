import re
from pathlib import Path

import yaml


def load_yaml(path: Path) -> dict[str, object]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def parse_image(image_str: str) -> tuple[str, str | None]:
    """Split an image string into repository and tag.
    Supports forms like 'repo:tag' and 'repo:${TAG:-default}'.
    Returns (repository, tag_or_default) where tag_or_default may be None if not parseable.
    """
    if ":" not in image_str:
        return image_str, None
    repo, tag_part = image_str.split(":", 1)
    # Handle ${VAR:-default}
    m = re.match(r"^\$\{[^:}]+:-([^}]+)\}$", tag_part)
    if m:
        return repo, m.group(1)
    # Handle ${VAR}
    if tag_part.startswith("${") and tag_part.endswith("}"):
        return repo, None
    return repo, tag_part


def get_dict_path(d: dict[str, object], path: list[str]) -> object:
    ref: object = d
    for key in path:
        if not isinstance(ref, dict) or key not in ref:
            return None
        ref = ref[key]
    return ref


def ensure_frontend_env_names(values_env_list: object) -> set[str]:
    names = set()
    if isinstance(values_env_list, list):
        for item in values_env_list:
            if isinstance(item, dict) and "name" in item:
                names.add(str(item["name"]))
    return names


def extract_compose_env_names(env_obj: object) -> set[str]:
    """Return environment variable names from docker-compose service env.
    Supports both dict mapping (key: value) and list form ["KEY=val", ...].
    """
    names: set[str] = set()
    if isinstance(env_obj, dict):
        names = {str(k) for k in env_obj.keys()}
    elif isinstance(env_obj, list):
        for entry in env_obj:
            if isinstance(entry, str) and entry:
                name = entry.split("=", 1)[0]
                if name:
                    names.add(name)
    return names


def extract_values_env_keys(values_env_obj: object) -> set[str]:
    """Return env keys defined in Helm values. For dict returns keys, for list of {name,value} returns names."""
    if isinstance(values_env_obj, dict):
        return {str(k) for k in values_env_obj.keys()}
    return ensure_frontend_env_names(values_env_obj)


def test_compose_helm_image_and_env_parity():
    repo_root = Path(__file__).resolve().parents[3]

    # Paths
    values_path = repo_root / "deploy/helm/nvidia-blueprint-rag/values.yaml"
    compose_rag_path = repo_root / "deploy/compose/docker-compose-rag-server.yaml"
    compose_ingestor_path = repo_root / "deploy/compose/docker-compose-ingestor-server.yaml"
    compose_nims_path = repo_root / "deploy/compose/nims.yaml"

    # Load YAMLs
    values = load_yaml(values_path)
    compose_rag = load_yaml(compose_rag_path)
    compose_ingestor = load_yaml(compose_ingestor_path)
    compose_nims = load_yaml(compose_nims_path)

    # Mapping between docker-compose services and Helm values.yaml
    mapping = {
        str(compose_rag_path): {
            "rag-server": {
                "values_image_repo_path": ["image", "repository"],
                "values_image_tag_path": ["image", "tag"],
                "values_env_path": ["envVars"],
                # dynamic env parity from compose
                "require_all_env_from_compose": True,
                # API keys are represented via Helm secrets, not envVars
                "ignore_env_keys": {"NGC_API_KEY", "NVIDIA_API_KEY"},
            },
            "rag-frontend": {
                "values_image_repo_path": ["frontend", "image", "repository"],
                "values_image_tag_path": ["frontend", "image", "tag"],
                "values_env_path": ["frontend", "envVars"],
                # enforce presence of these runtime names
                "required_env_names": [
                    "VITE_API_CHAT_URL",
                    "VITE_API_VDB_URL",
                ],
            },
        },
        str(compose_ingestor_path): {
            "ingestor-server": {
                "values_image_repo_path": ["ingestor-server", "image", "repository"],
                "values_image_tag_path": ["ingestor-server", "image", "tag"],
                "values_env_path": ["ingestor-server", "envVars"],
                # dynamic env parity from compose, with known exceptions
                "require_all_env_from_compose": True,
                "ignore_env_keys": {"NGC_API_KEY", "NVIDIA_API_KEY"},
            }
        },
        str(compose_nims_path): {
            # Image parity checks only; env parity enforced for API key presence
            "nim-llm": {
                "values_image_repo_path": ["nim-llm", "image", "repository"],
                "values_image_tag_path": ["nim-llm", "image", "tag"],
                "requires_ngc_api_key_path": ["nim-llm", "model", "ngcAPIKey"],
            },
            "nemoretriever-embedding-ms": {
                "values_image_repo_path": ["nvidia-nim-llama-32-nv-embedqa-1b-v2", "image", "repository"],
                "values_image_tag_path": ["nvidia-nim-llama-32-nv-embedqa-1b-v2", "image", "tag"],
                "requires_ngc_api_key_path": ["nvidia-nim-llama-32-nv-embedqa-1b-v2", "nim", "ngcAPIKey"],
            },
            "nemoretriever-ranking-ms": {
                "values_image_repo_path": ["nvidia-nim-llama-32-nv-rerankqa-1b-v2", "image", "repository"],
                "values_image_tag_path": ["nvidia-nim-llama-32-nv-rerankqa-1b-v2", "image", "tag"],
                "requires_ngc_api_key_path": ["nvidia-nim-llama-32-nv-rerankqa-1b-v2", "nim", "ngcAPIKey"],
            },
            "vlm-ms": {
                "values_image_repo_path": ["nim-vlm", "image", "repository"],
                "values_image_tag_path": ["nim-vlm", "image", "tag"],
                "requires_ngc_api_key_path": ["nim-vlm", "nim", "ngcAPIKey"],
            },
        },
    }

    def assert_image_parity(compose_service: dict[str, object], values_repo_path: list[str], values_tag_path: list[str]):
        compose_image = compose_service.get("image")
        assert compose_image, "compose image must be set"
        repo, tag = parse_image(str(compose_image))
        values_repo = get_dict_path(values, values_repo_path)
        values_tag = get_dict_path(values, values_tag_path)
        assert values_repo == repo, f"Repository mismatch: {values_repo_path}='{values_repo}' != '{repo}'"
        if tag is not None:
            assert values_tag == tag, f"Tag mismatch: {values_tag_path}='{values_tag}' != '{tag}'"

    def assert_env_presence(
        compose_service: dict[str, object],
        values_env_path: list[str],
        required_keys: list[str] | None = None,
        required_names: list[str] | None = None,
        require_all_from_compose: bool = False,
        ignore_env_keys: set[str] | None = None,
    ):
        values_env = get_dict_path(values, values_env_path)
        # values env for rag-server/ingestor-server is a dict; for frontend it is a list of {name,value}
        if required_keys:
            assert isinstance(values_env, dict), f"Expected dict at values path {values_env_path}"
            for key in required_keys:
                assert key in values_env, f"Missing env key '{key}' in values at {values_env_path}"
        if required_names:
            names = ensure_frontend_env_names(values_env)
            for name in required_names:
                assert name in names, f"Missing env name '{name}' in values at {values_env_path}"
        if require_all_from_compose:
            compose_env = compose_service.get("environment", {})
            compose_names = extract_compose_env_names(compose_env)
            if ignore_env_keys:
                compose_names -= set(ignore_env_keys)
            values_keys = extract_values_env_keys(values_env)
            missing = compose_names - values_keys
            assert not missing, (
                f"Missing env keys from compose in Helm at {values_env_path}: {sorted(missing)}"
            )

    # Execute checks
    for compose_file, services_spec in mapping.items():
        compose_data = {
            str(compose_rag_path): compose_rag,
            str(compose_ingestor_path): compose_ingestor,
            str(compose_nims_path): compose_nims,
        }[compose_file]
        services = compose_data.get("services", {})
        for svc_name, rules in services_spec.items():
            assert svc_name in services, f"Service '{svc_name}' not found in {compose_file}"
            svc = services[svc_name]
            assert_image_parity(
                svc,
                rules["values_image_repo_path"],
                rules["values_image_tag_path"],
            )
            if "values_env_path" in rules:
                assert_env_presence(
                    svc,
                    rules["values_env_path"],
                    rules.get("required_env_keys"),
                    rules.get("required_env_names"),
                    rules.get("require_all_env_from_compose", False),
                    rules.get("ignore_env_keys"),
                )

            # For NIM services ensure API key is configurable in Helm when present in compose
            if "requires_ngc_api_key_path" in rules:
                compose_env = svc.get("environment")
                # Some NIM services use list form; check for presence of NGC_API_KEY in any form
                compose_env_names = extract_compose_env_names(compose_env)
                if "NGC_API_KEY" in compose_env_names:
                    ngc_key_path = rules["requires_ngc_api_key_path"]
                    ngc_key_value = get_dict_path(values, ngc_key_path)
                    assert ngc_key_value is not None, (
                        f"Expected Helm values key for NGC API under {ngc_key_path}, but it was missing"
                    )


def test_prompt_yaml_parity():
    repo_root = Path(__file__).resolve().parents[3]
    src_prompt = repo_root / "src/nvidia_rag/rag_server/prompt.yaml"
    helm_prompt = repo_root / "deploy/helm/nvidia-blueprint-rag/files/prompt.yaml"

    assert src_prompt.exists(), f"Missing source prompt file: {src_prompt}"
    assert helm_prompt.exists(), f"Missing Helm prompt file: {helm_prompt}"

    src_content = src_prompt.read_text(encoding="utf-8").strip()
    helm_content = helm_prompt.read_text(encoding="utf-8").strip()

    assert src_content == helm_content, (
        "prompt.yaml mismatch: src and Helm copies must be identical. "
        "Update deploy/helm/nvidia-blueprint-rag/files/prompt.yaml to reflect changes in src/nvidia_rag/rag_server/prompt.yaml"
    )