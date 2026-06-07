"""Runtime planning for model-bound pipeline steps."""

from __future__ import annotations

import json
import os
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from .mode import PipelineMode, requires_model_runtime_plan
from .paths import resolve_under_project


_TP_CACHE_DIR = Path.home() / ".cache" / "flashinfer_trace"
_TP_CACHE_FILE = _TP_CACHE_DIR / "sgl_cookbook_tp.json"
_TP_CACHE_TTL_DAYS = 7
_RUNTIME_PROFILES_FILE = Path(__file__).with_name("runtime_profiles.yaml")
_DEFAULT_RUNTIME_PROFILES: dict[str, Any] = {
    "tp_rules": {
        "model_type": {
            "deepseek_v4": {
                "default_tp": 8,
                "leaf_token_overrides": {"pro": 16},
            },
        },
        "config_thresholds": [
            {"name": "large_moe", "routed_experts_min": 256, "hidden_size_min": 4096, "tp": 8},
            {"name": "large_hidden", "hidden_size_min": 8192, "tp": 4},
        ],
        "model_name_tokens": [
            {"tokens": ["405b"], "tp": 8},
            {"tokens": ["72b", "70b", "mixtral"], "tp": 4},
            {"tokens": ["35b", "32b", "27b"], "tp": 2},
        ],
        "default_tp": 1,
    },
    "gpu_rules": {
        "config_thresholds": [
            {"name": "deepseek_v4", "model_type": "deepseek_v4", "gpu": "A100-80GB"},
            {
                "name": "large_moe",
                "routed_experts_min": 256,
                "hidden_size_min": 4096,
                "gpu": "A100-80GB",
            },
        ],
        "model_name_tokens": [
            {
                "tokens": ["a3b", "moe", "mixtral", "36b", "35b", "32b", "27b"],
                "gpu": "A100-80GB",
            },
            {
                "tokens": ["9b", "8b", "7b", "4b", "2b", "1.7b", "1.5b", "0.8b", "0.6b"],
                "gpu": "L40S",
            },
        ],
    },
}


@dataclass
class RuntimePlan:
    tp: int
    hf_config: Path | None
    modal_gpu: str | None
    modal_gpu_count: str | None


def _load_tp_cache() -> dict:
    try:
        if _TP_CACHE_FILE.exists():
            return json.loads(_TP_CACHE_FILE.read_text())
    except Exception:
        pass
    return {}


def _save_tp_cache(cache: dict, log: Callable[[str], None]) -> None:
    try:
        _TP_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        _TP_CACHE_FILE.write_text(json.dumps(cache, indent=2, ensure_ascii=False))
    except Exception as exc:
        log(f"[sgl-cookbook] cache write failed: {exc}")


def _read_hf_config_json(hf_config: Path | None, project_root: Path) -> dict | None:
    """Load a HuggingFace config JSON if one is available."""
    if hf_config is None:
        return None
    try:
        path = resolve_under_project(hf_config, project_root)
        if not path.exists():
            return None
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else None
    except (OSError, json.JSONDecodeError):
        return None


def _config_int(config: dict, *keys: str) -> int | None:
    for key in keys:
        value = config.get(key)
        if isinstance(value, bool):
            continue
        if isinstance(value, int):
            return value
        if isinstance(value, str) and value.isdigit():
            return int(value)
    return None


def _load_runtime_profiles() -> dict[str, Any]:
    """Load runtime planning heuristics from YAML."""
    try:
        import yaml

        data = yaml.safe_load(_RUNTIME_PROFILES_FILE.read_text(encoding="utf-8"))
    except Exception:
        return _DEFAULT_RUNTIME_PROFILES
    return data if isinstance(data, dict) else _DEFAULT_RUNTIME_PROFILES


def _model_leaf_tokens(model_name: str) -> set[str]:
    leaf = model_name.lower().rsplit("/", 1)[-1]
    return set(leaf.replace("_", "-").replace(".", "-").split("-"))


def _model_type_matches(rule_model_type: str | None, model_type: str, architectures: str) -> bool:
    if not rule_model_type:
        return True
    normalized_rule = rule_model_type.replace("_", "").lower()
    return model_type == rule_model_type.lower() or normalized_rule in architectures.replace("_", "")


def _config_rule_matches(
    rule: dict[str, Any],
    *,
    model_type: str,
    architectures: str,
    hidden_size: int | None,
    routed_experts: int | None,
) -> bool:
    if not _model_type_matches(rule.get("model_type"), model_type, architectures):
        return False
    hidden_min = rule.get("hidden_size_min")
    if hidden_min is not None and (hidden_size is None or hidden_size < int(hidden_min)):
        return False
    experts_min = rule.get("routed_experts_min")
    if experts_min is not None and (routed_experts is None or routed_experts < int(experts_min)):
        return False
    return True


def _lookup_tp_from_sgl_cookbook(model_name: str, log: Callable[[str], None]) -> int | None:
    """Look up the recommended TP value from sgl-cookbook YAML files."""
    import urllib.request

    cache = _load_tp_cache()
    entry = cache.get(model_name)
    if entry is not None:
        cached_at = entry.get("cached_at", "")
        try:
            age = datetime.now() - datetime.fromisoformat(cached_at)
            if age.days < _TP_CACHE_TTL_DAYS:
                tp_val = entry.get("tp")
                if tp_val is not None:
                    log(f"[sgl-cookbook] TP={tp_val} (cached {age.days}d ago)")
                    return int(tp_val)
                log("[sgl-cookbook] cached as not-found, skipping HTTP")
                return None
        except (ValueError, TypeError):
            pass

    try:
        import yaml
    except ImportError:
        log("[sgl-cookbook] pyyaml not installed — run `pip install pyyaml` to enable sgl-cookbook lookup")
        return None

    base_raw = "https://raw.githubusercontent.com/sgl-project/sgl-cookbook/main/data/models/generated"
    base_api = "https://api.github.com/repos/sgl-project/sgl-cookbook/contents/data/models/generated"

    def _get_tp_from_yaml_data(data: dict) -> int | None:
        for family in data.get("families", []):
            for model in family.get("models", []):
                if model.get("model_path") != model_name:
                    continue
                hardware = model.get("hardware", {})
                seen: set[str] = set()
                for hw_name in ["H100", "H200", "A100", *hardware.keys()]:
                    if hw_name in seen:
                        continue
                    seen.add(hw_name)
                    cfgs = hardware.get(hw_name, {}).get("configurations", [])
                    if cfgs:
                        tp = cfgs[0].get("engine", {}).get("tp")
                        if tp:
                            return int(tp)
        return None

    org = model_name.split("/")[0].lower() if "/" in model_name else model_name.lower()
    vendor_hints: dict[str, list[str]] = {
        "qwen": ["qwen.yaml", "qwen35.yaml", "qwen3next.yaml", "qwen3vl.yaml", "qwen-image.yaml", "qwen3codernext.yaml"],
        "meta-llama": ["llama31.yaml", "llama4scout.yaml"],
        "deepseek-ai": ["deepseek.yaml", "deepseek-r1.yaml", "deepseek-math-v2.yaml"],
        "mistralai": ["mistral.yaml", "mistral-small-4.yaml"],
        "thudm": ["glm46.yaml", "glm46v.yaml", "glm5.yaml", "glm51.yaml"],
        "nvidia": ["nemotron.yaml", "nemotron-super.yaml"],
        "internlm": ["intern-s1.yaml"],
        "moonshotai": ["kimi-k2.yaml", "kimi-k25.yaml"],
        "google": ["gemma4.yaml"],
        "stepfun-ai": ["step35.yaml"],
    }
    hints = set(vendor_hints.get(org, []))

    try:
        with urllib.request.urlopen(base_api, timeout=8) as resp:  # noqa: S310
            versions = sorted(
                [item["name"] for item in json.loads(resp.read()) if item["type"] == "dir"],
                reverse=True,
            )

        for version in versions:
            try:
                version_url = f"{base_api}/{version}"
                with urllib.request.urlopen(version_url, timeout=8) as resp:  # noqa: S310
                    all_yaml = [
                        item["name"]
                        for item in json.loads(resp.read())
                        if item["name"].endswith(".yaml")
                    ]
            except Exception:
                continue

            prioritized = [name for name in all_yaml if name in hints]
            remaining = [name for name in all_yaml if name not in hints]
            for fname in prioritized + remaining:
                try:
                    raw_url = f"{base_raw}/{version}/{fname}"
                    with urllib.request.urlopen(raw_url, timeout=8) as resp:  # noqa: S310
                        data = yaml.safe_load(resp.read())
                    tp = _get_tp_from_yaml_data(data)
                    if tp is not None:
                        cache[model_name] = {
                            "tp": tp,
                            "cached_at": datetime.now().isoformat(),
                        }
                        _save_tp_cache(cache, log)
                        return tp
                except Exception:
                    continue
    except Exception as exc:
        log(f"[sgl-cookbook] lookup error: {exc}")

    cache[model_name] = {
        "tp": None,
        "cached_at": datetime.now().isoformat(),
    }
    _save_tp_cache(cache, log)
    return None


def _infer_tp_from_hf_config(
    model_name: str,
    hf_config: Path | None,
    *,
    project_root: Path,
    log: Callable[[str], None],
) -> int | None:
    """Infer TP from model config when cookbook does not cover a new model."""
    config = _read_hf_config_json(hf_config, project_root)
    if not config:
        return None

    model_type = str(config.get("model_type", "")).lower()
    architectures = " ".join(
        item for item in config.get("architectures", []) if isinstance(item, str)
    ).lower()
    hidden_size = _config_int(config, "hidden_size")
    layers = _config_int(config, "num_hidden_layers", "n_layers")
    routed_experts = _config_int(config, "n_routed_experts", "num_experts", "num_local_experts")
    profiles = _load_runtime_profiles()
    tp_rules = profiles.get("tp_rules", {}) if isinstance(profiles.get("tp_rules"), dict) else {}

    model_type_rules = tp_rules.get("model_type", {})
    if isinstance(model_type_rules, dict):
        leaf_tokens = _model_leaf_tokens(model_name)
        for rule_model_type, rule in model_type_rules.items():
            if not isinstance(rule, dict):
                continue
            if not _model_type_matches(str(rule_model_type), model_type, architectures):
                continue
            tp = int(rule.get("default_tp", 1))
            overrides = rule.get("leaf_token_overrides", {})
            if isinstance(overrides, dict):
                for token, override_tp in overrides.items():
                    if str(token).lower() in leaf_tokens:
                        tp = int(override_tp)
                        break
            log(
                "TP="
                f"{tp} (from HF config profile={rule_model_type}, "
                f"model_type={model_type or '-'}, hidden_size={hidden_size or '-'}, "
                f"layers={layers or '-'}, routed_experts={routed_experts or '-'})"
            )
            return tp

    for rule in tp_rules.get("config_thresholds", []):
        if not isinstance(rule, dict):
            continue
        if not _config_rule_matches(
            rule,
            model_type=model_type,
            architectures=architectures,
            hidden_size=hidden_size,
            routed_experts=routed_experts,
        ):
            continue
        tp = int(rule["tp"])
        log(
            "TP="
            f"{tp} (from HF config profile={rule.get('name', 'unnamed')}: "
            f"hidden_size={hidden_size or '-'}, routed_experts={routed_experts or '-'})"
        )
        return tp

    return None


def _infer_tp_size(
    model_name: str,
    hf_config: Path | None,
    *,
    project_root: Path,
    log: Callable[[str], None],
) -> int:
    """Infer Tensor Parallelism from sgl-cookbook, HF config, then name heuristics."""
    tp = _lookup_tp_from_sgl_cookbook(model_name, log)
    if tp is not None:
        log(f"TP={tp} (from sgl-cookbook)")
        return tp

    tp = _infer_tp_from_hf_config(
        model_name,
        hf_config,
        project_root=project_root,
        log=log,
    )
    if tp is not None:
        return tp

    lower = model_name.lower()
    profiles = _load_runtime_profiles()
    tp_rules = profiles.get("tp_rules", {}) if isinstance(profiles.get("tp_rules"), dict) else {}
    for rule in tp_rules.get("model_name_tokens", []):
        if not isinstance(rule, dict):
            continue
        tokens = rule.get("tokens") or []
        if any(str(token).lower() in lower for token in tokens):
            tp = int(rule["tp"])
            break
    else:
        tp = int(tp_rules.get("default_tp", 1)) if isinstance(tp_rules, dict) else 1
    log(f"TP={tp} (heuristic fallback, sgl-cookbook lookup failed)")
    return tp


def _valid_tp_divisors(head_count: int) -> list[int]:
    if head_count <= 0:
        return []
    return [tp for tp in range(1, head_count + 1) if head_count % tp == 0]


def _normalize_tp_for_hf_config(
    *,
    model_name: str,
    hf_config: Path | None,
    tp: int,
    explicit: bool,
    project_root: Path,
    log: Callable[[str], None],
) -> int:
    """Validate TP against HF config attention heads when available."""
    if tp <= 0:
        return tp
    config = _read_hf_config_json(hf_config, project_root)
    if not config:
        return tp

    num_heads = _config_int(config, "num_attention_heads", "n_head", "num_heads")
    if not num_heads or num_heads <= 0 or num_heads % tp == 0:
        return tp

    valid = _valid_tp_divisors(num_heads)
    valid_text = ", ".join(str(item) for item in valid) if valid else "none"
    model_type = str(config.get("model_type", "") or "-")
    message = (
        f"TP={tp} is incompatible with {model_name}: "
        f"num_attention_heads={num_heads} is not divisible by TP. "
        f"model_type={model_type}; valid TP values: {valid_text}"
    )
    if explicit:
        raise SystemExit(f"❌ {message}")

    fallback = max((item for item in valid if item <= tp), default=1)
    log(f"⚠️  {message}; using TP={fallback} instead")
    return fallback


def _validate_hf_config_for_probe(
    *,
    model_name: str,
    hf_config: Path | None,
    tp: int,
    project_root: Path,
    log: Callable[[str], None],
) -> None:
    """Run conservative CPU-side HF config sanity checks before Modal GPU launch."""
    config = _read_hf_config_json(hf_config, project_root)
    if not config:
        return

    model_type = str(config.get("model_type", "") or "-")
    q_heads = _config_int(config, "num_attention_heads", "n_head", "num_heads")
    kv_heads = _config_int(
        config,
        "num_key_value_heads",
        "n_kv_heads",
        "num_kv_heads",
        "multi_query_group_num",
    )
    hidden_size = _config_int(config, "hidden_size", "n_embd")
    head_dim = _config_int(config, "head_dim", "attention_head_dim")

    if q_heads and kv_heads:
        if kv_heads > q_heads:
            raise SystemExit(
                "❌ HF config is inconsistent for "
                f"{model_name}: num_key_value_heads={kv_heads} is larger than "
                f"num_attention_heads={q_heads}. model_type={model_type}"
            )
        if q_heads % kv_heads != 0:
            raise SystemExit(
                "❌ HF config is inconsistent for "
                f"{model_name}: num_attention_heads={q_heads} must be divisible "
                f"by num_key_value_heads={kv_heads}. model_type={model_type}"
            )
        if tp > 0 and kv_heads % tp != 0:
            log(
                "⚠️  HF config warning: "
                f"num_key_value_heads={kv_heads} is not divisible by TP={tp}. "
                "Some runtimes replicate KV heads across TP ranks, so this is "
                "not treated as fatal."
            )

    if hidden_size and q_heads:
        if hidden_size % q_heads != 0:
            log(
                "⚠️  HF config warning: "
                f"hidden_size={hidden_size} is not divisible by "
                f"num_attention_heads={q_heads}. model_type={model_type}"
            )
        else:
            inferred_head_dim = hidden_size // q_heads
            if head_dim and head_dim != inferred_head_dim:
                log(
                    "⚠️  HF config warning: "
                    f"head_dim={head_dim} differs from hidden_size/heads="
                    f"{inferred_head_dim}. model_type={model_type}"
                )


def _select_modal_gpu(model_name: str, hf_config: Path | None, project_root: Path) -> str | None:
    """Choose a conservative default Modal GPU for common model sizes."""
    lower = model_name.lower()
    config = _read_hf_config_json(hf_config, project_root)
    gpu_rules = _load_runtime_profiles().get("gpu_rules", {})
    if config:
        model_type = str(config.get("model_type", "")).lower()
        architectures = " ".join(
            item for item in config.get("architectures", []) if isinstance(item, str)
        ).lower()
        hidden_size = _config_int(config, "hidden_size") or 0
        routed_experts = _config_int(config, "n_routed_experts", "num_experts", "num_local_experts") or 0
        if isinstance(gpu_rules, dict):
            for rule in gpu_rules.get("config_thresholds", []):
                if not isinstance(rule, dict):
                    continue
                if _config_rule_matches(
                    rule,
                    model_type=model_type,
                    architectures=architectures,
                    hidden_size=hidden_size,
                    routed_experts=routed_experts,
                ):
                    return str(rule["gpu"])

    if isinstance(gpu_rules, dict):
        for rule in gpu_rules.get("model_name_tokens", []):
            if not isinstance(rule, dict):
                continue
            tokens = rule.get("tokens") or []
            if any(str(token).lower() in lower for token in tokens):
                return str(rule["gpu"])
    return None


def _modal_work_requested(
    *,
    mode: PipelineMode,
    args,
    probe_output: Path | None,
    inventory_path: Path | None,
) -> bool:
    """Return whether this pipeline invocation is expected to launch Modal."""
    needs_probe = (
        mode.runs_step("probe")
        and mode.runtime_probe_requested
        and not args.skip_probe
        and probe_output is None
        and inventory_path is None
    )
    needs_collect = mode.runs_step("collect") and not args.skip_collect
    return needs_probe or needs_collect


def _apply_modal_gpu_defaults(
    *,
    model_name: str,
    tp: int,
    mode: PipelineMode,
    args,
    probe_output: Path | None,
    inventory_path: Path | None,
    project_root: Path,
    log: Callable[[str], None],
) -> None:
    """Set Modal GPU env defaults only when the user did not set them."""
    if not _modal_work_requested(
        mode=mode,
        args=args,
        probe_output=probe_output,
        inventory_path=inventory_path,
    ):
        return

    if os.environ.get("MODAL_GPU"):
        log(f"Using MODAL_GPU={os.environ['MODAL_GPU']} from environment")
    else:
        modal_gpu = _select_modal_gpu(model_name, args.hf_config, project_root)
        if modal_gpu:
            os.environ["MODAL_GPU"] = modal_gpu
            log(f"Auto-selected MODAL_GPU={modal_gpu} for {model_name}")
        else:
            log("MODAL_GPU is not set; using Modal script default GPU")

    if tp > 0 and not os.environ.get("MODAL_GPU_COUNT"):
        os.environ["MODAL_GPU_COUNT"] = str(tp)
        log(f"Auto-selected MODAL_GPU_COUNT={tp} from TP={tp}")


def auto_download_hf_config(
    model_name: str,
    *,
    log: Callable[[str], None],
) -> Path | None:
    """Download a model config.json from HuggingFace Hub when available."""
    try:
        from huggingface_hub import hf_hub_download

        log(f"[HF] Downloading config.json for {model_name} ...")
        path = hf_hub_download(repo_id=model_name, filename="config.json")
        log(f"[HF] Downloaded: {path}")
        return Path(path)
    except ImportError:
        log("[HF] huggingface_hub not installed (pip install huggingface-hub), skipping auto-download")
        return None
    except Exception as exc:
        log(f"[HF] Failed to download config.json: {exc}")
        return None


def prefetch_hf_config_for_pipeline(
    args,
    mode: PipelineMode,
    *,
    log: Callable[[str], None],
) -> None:
    """Download HF config once, before TP inference and downstream steps."""
    if args.hf_config is not None:
        return
    if args.step == "validate":
        return

    needs_config = (
        args.tp == 0
        or mode.runs_runtime_parse
        or mode.runs_step("static")
        or mode.static_sidecar_requested
    )
    if not needs_config:
        return

    if args.dry_run:
        log("[HF] Dry-run metadata prefetch: downloading config.json for TP/config planning")
    hf_config = auto_download_hf_config(args.model_name, log=log)
    if hf_config is not None:
        args.hf_config = hf_config
        log(f"[HF] Using config for pipeline planning: {hf_config}")


def prepare_runtime_plan(
    *,
    args,
    mode: PipelineMode,
    probe_output: Path | None,
    inventory_path: Path | None,
    project_root: Path,
    log: Callable[[str], None],
) -> RuntimePlan:
    """Prepare HF config, TP, sanity checks, and Modal GPU defaults."""
    if mode.step == "definitions" and inventory_path is not None:
        log("v2 Pipeline — model runtime planning skipped (definitions use supplied inventory)")
        return RuntimePlan(
            tp=args.tp if args.tp > 0 else 0,
            hf_config=args.hf_config,
            modal_gpu=os.environ.get("MODAL_GPU"),
            modal_gpu_count=os.environ.get("MODAL_GPU_COUNT"),
        )

    if not requires_model_runtime_plan(mode):
        log("v2 Pipeline — model runtime planning skipped (no model-bound steps)")
        return RuntimePlan(
            tp=0,
            hf_config=args.hf_config,
            modal_gpu=os.environ.get("MODAL_GPU"),
            modal_gpu_count=os.environ.get("MODAL_GPU_COUNT"),
        )

    prefetch_hf_config_for_pipeline(args, mode, log=log)

    explicit_tp = args.tp > 0
    if explicit_tp:
        tp = args.tp
        tp = _normalize_tp_for_hf_config(
            model_name=args.model_name,
            hf_config=args.hf_config,
            tp=tp,
            explicit=True,
            project_root=project_root,
            log=log,
        )
    else:
        tp = _infer_tp_size(
            args.model_name,
            args.hf_config,
            project_root=project_root,
            log=log,
        )
        tp = _normalize_tp_for_hf_config(
            model_name=args.model_name,
            hf_config=args.hf_config,
            tp=tp,
            explicit=False,
            project_root=project_root,
            log=log,
        )
        log(f"🤖 Auto-detected TP={tp} for {args.model_name}")

    _validate_hf_config_for_probe(
        model_name=args.model_name,
        hf_config=args.hf_config,
        tp=tp,
        project_root=project_root,
        log=log,
    )

    tp_label = str(tp) if tp else "n/a"
    log(f"v2 Pipeline — Model: {args.model_name}, TP: {tp_label}")

    _apply_modal_gpu_defaults(
        model_name=args.model_name,
        tp=tp,
        mode=mode,
        args=args,
        probe_output=probe_output,
        inventory_path=inventory_path,
        project_root=project_root,
        log=log,
    )

    return RuntimePlan(
        tp=tp,
        hf_config=args.hf_config,
        modal_gpu=os.environ.get("MODAL_GPU"),
        modal_gpu_count=os.environ.get("MODAL_GPU_COUNT"),
    )
