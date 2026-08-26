import json
import os
from pathlib import Path


MODEL_ALIASES = {
    "bert-base-uncased": {
        "local_names": ["bert-base-uncased", "AI-ModelScope/bert-base-uncased"],
        "modelscope_id": "AI-ModelScope/bert-base-uncased",
        "hf_id": "bert-base-uncased",
    },
    "AI-ModelScope/bert-base-uncased": {
        "local_names": ["bert-base-uncased", "AI-ModelScope/bert-base-uncased"],
        "modelscope_id": "AI-ModelScope/bert-base-uncased",
        "hf_id": "bert-base-uncased",
    },
    "bert-base-cased": {
        "local_names": ["bert-base-cased", "AI-ModelScope/bert-base-cased"],
        "modelscope_id": "AI-ModelScope/bert-base-cased",
        "hf_id": "bert-base-cased",
    },
    "AI-ModelScope/bert-base-cased": {
        "local_names": ["bert-base-cased", "AI-ModelScope/bert-base-cased"],
        "modelscope_id": "AI-ModelScope/bert-base-cased",
        "hf_id": "bert-base-cased",
    },
    "facebook/bart-base": {
        "local_names": ["bart-base", "facebook/bart-base"],
        "modelscope_id": None,
        "hf_id": "facebook/bart-base",
    },
    "bart-base": {
        "local_names": ["bart-base", "facebook/bart-base"],
        "modelscope_id": None,
        "hf_id": "facebook/bart-base",
    },
}


def _repo_root():
    return Path(__file__).resolve().parents[1]


def _candidate_roots():
    roots = []
    env_root = os.environ.get("DICRA_MODEL_ROOT") or os.environ.get(
        "RECAP_MODEL_ROOT"
    )
    if env_root:
        roots.append(Path(env_root).expanduser())

    roots.extend(
        [
            _repo_root() / "models",
            Path.home() / "models",
            Path("/home/share/models"),
        ]
    )
    return roots


def _load_config_map():
    config_path = os.environ.get("DICRA_MODEL_CONFIG") or os.environ.get(
        "RECAP_MODEL_CONFIG"
    )
    candidates = []
    if config_path:
        candidates.append(Path(config_path).expanduser())
    candidates.append(_repo_root() / "configs" / "model_paths.json")

    for path in candidates:
        if not path.exists():
            continue
        with path.open("r", encoding="utf-8") as f:
            payload = json.load(f)
        if "models" in payload:
            payload = payload["models"]
        return {str(k): str(v) for k, v in payload.items()}
    return {}


def _valid_model_dir(path):
    path = Path(path).expanduser()
    if not path.exists():
        return False
    return (path / "config.json").exists()


def _try_configured_path(model_id):
    mapping = _load_config_map()
    for key in (model_id, model_id.split("/")[-1]):
        if key in mapping and _valid_model_dir(mapping[key]):
            return str(Path(mapping[key]).expanduser())
    return None


def _try_local_roots(model_id):
    alias = MODEL_ALIASES.get(model_id, {})
    names = list(alias.get("local_names", []))
    names.extend([model_id, model_id.split("/")[-1]])

    seen = set()
    unique_names = []
    for name in names:
        if name not in seen:
            seen.add(name)
            unique_names.append(name)

    for root in _candidate_roots():
        for name in unique_names:
            path = root / name
            if _valid_model_dir(path):
                return str(path)
    return None


def _try_modelscope(model_id):
    alias = MODEL_ALIASES.get(model_id, {})
    search_id = alias.get("modelscope_id")
    if search_id is None and "bert" in model_id:
        search_id = model_id if "/" in model_id else f"AI-ModelScope/{model_id}"
    if search_id is None:
        return None

    try:
        from modelscope.hub.snapshot_download import snapshot_download
    except Exception as exc:
        print(f"[ModelPath] ModelScope unavailable: {exc}")
        return None

    cache_dir = os.path.expanduser("~/.cache/modelscope/hub/models")
    cached = Path(cache_dir) / search_id
    if _valid_model_dir(cached):
        return str(cached)
    try:
        return snapshot_download(search_id, cache_dir=cache_dir)
    except Exception as exc:
        print(f"[ModelPath] ModelScope download failed for {search_id}: {exc}")
        return None


def resolve_model_path(model_id):











    model_id = str(model_id)

    explicit = Path(model_id).expanduser()
    if _valid_model_dir(explicit):
        return str(explicit)

    configured = _try_configured_path(model_id)
    if configured:
        return configured

    local = _try_local_roots(model_id)
    if local:
        return local

    modelscope_path = _try_modelscope(model_id)
    if modelscope_path:
        return modelscope_path

    alias = MODEL_ALIASES.get(model_id, {})
    return alias.get("hf_id", model_id)


def get_bert_path(model_id="bert-base-uncased"):
    return resolve_model_path(model_id)
