"""Runtime hook helpers for M2 probe capture.

The capture path records two artifacts per call:

* ``events.jsonl``: lightweight metadata for parse/filtering.
* ``captures/*.pt``: selective payload snapshots for later workload review.

Large floating tensors are summarized instead of dumped. Integer/index tensors
are preserved because their values usually encode structure.
"""

from __future__ import annotations

import importlib
import inspect
import json
import os
import sys
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from flashinfer_trace.core.schemas import (
    CaptureSpec,
    DispatchRule,
    DispatchSpec,
    ProbePlan,
    ProbeTarget,
    max_captures_from_plan_payload,
)


CAPTURE_SCHEMA_VERSION = 1
COMPANION_CAPTURE_ATTR = "_fitrace_companion_capture"
CAPTURE_SCOPE_FILENAME = "capture_scope.json"
HF_CONFIG_OVERRIDE_ENV = "FLASHINFER_TRACE_HF_CONFIG_OVERRIDE_JSON"

_HF_CONFIG_OVERRIDE: dict[str, Any] | None = None
_HF_CONFIG_PATCHED = False


@dataclass(frozen=True)
class InstalledHook:
    """One installed monkey patch that can be restored."""

    module: Any
    attr: str
    original: Callable[..., Any]


def install_hf_config_override(data: dict[str, Any] | None) -> None:
    """Install a config override before Transformers builds remote configs."""
    if data is None:
        return
    global _HF_CONFIG_OVERRIDE, _HF_CONFIG_PATCHED
    _HF_CONFIG_OVERRIDE = data
    if _HF_CONFIG_PATCHED:
        return

    from transformers import PretrainedConfig

    original = PretrainedConfig.from_dict.__func__

    @classmethod  # type: ignore[misc]
    def patched(cls, config_dict: dict[str, Any], **kwargs: Any) -> Any:
        override = _HF_CONFIG_OVERRIDE
        if isinstance(override, dict):
            config_dict = dict(override)
        return original(cls, config_dict, **kwargs)

    PretrainedConfig.from_dict = patched
    _HF_CONFIG_PATCHED = True


def install_hf_config_override_from_env() -> None:
    raw = os.environ.get(HF_CONFIG_OVERRIDE_ENV)
    if not raw:
        return
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"{HF_CONFIG_OVERRIDE_ENV} is not valid JSON: {exc}") from exc
    if not isinstance(data, dict):
        raise RuntimeError(f"{HF_CONFIG_OVERRIDE_ENV} must decode to a JSON object")
    install_hf_config_override(data)



def _shape_list(value: Any) -> list[int] | None:
    shape = getattr(value, "shape", None)
    if shape is None:
        return None
    try:
        return [int(dim) for dim in shape]
    except (TypeError, ValueError):
        return None


def summarize_value(value: Any) -> dict[str, Any]:
    """Return lightweight metadata for an argument value."""
    summary: dict[str, Any] = {"type": type(value).__name__}

    shape = _shape_list(value)
    if shape is not None:
        summary["shape"] = shape

    dtype = getattr(value, "dtype", None)
    if dtype is not None:
        summary["dtype"] = str(dtype)

    device = getattr(value, "device", None)
    if device is not None:
        summary["device"] = str(device)

    if isinstance(value, (str, int, float, bool)) or value is None:
        summary["value"] = value

    # Recurse into short tuples/lists to capture element shapes (e.g. paged_kv_cache tuple).
    if isinstance(value, (tuple, list)) and 1 <= len(value) <= 8:
        summary["elements"] = [
            {k: v for k, v in summarize_value(item).items() if k in ("type", "shape", "dtype")}
            for item in value
        ]

    return summary


def infer_dispatch_value(dispatch: DispatchSpec, args: tuple[Any, ...]) -> int | None:
    """Evaluate reviewed dispatch rules against one runtime call."""
    for rule in dispatch.rules:
        value = _dispatch_rule_value(rule, args)
        if value is not None:
            return value
    return None


def _dispatch_rule_value(rule: DispatchRule, args: tuple[Any, ...]) -> int | None:
    if rule.arg_index >= len(args):
        return None
    value = args[rule.arg_index]
    if rule.kind == "arg_attr":
        for attr_name in rule.attrs:
            attr_value = getattr(value, attr_name, None)
            if type(attr_value) is int and attr_value > 0:
                return attr_value
        return None
    if rule.tuple_index is not None:
        if not isinstance(value, (tuple, list)) or rule.tuple_index >= len(value):
            return None
        value = value[rule.tuple_index]
    shape = _shape_list(value)
    if shape is None:
        return None
    if rule.rank is not None and len(shape) != rule.rank:
        return None
    if rule.min_rank is not None and len(shape) < rule.min_rank:
        return None
    if rule.equals_index is not None:
        if rule.equals_index >= len(shape) or shape[rule.equals_index] != rule.equals:
            return None
    if rule.value is not None:
        return rule.value
    if rule.shape_index is None or rule.shape_index >= len(shape):
        return None
    detected = shape[rule.shape_index]
    return detected if detected > 0 else None


def _is_tensor(value: Any) -> bool:
    return hasattr(value, "shape") and hasattr(value, "dtype") and hasattr(value, "detach")


def _is_integer_like_tensor(value: Any) -> bool:
    dtype = str(getattr(value, "dtype", "")).lower()
    return any(token in dtype for token in ("int", "uint", "bool", "long"))


def _require_capture_spec(target: ProbeTarget) -> CaptureSpec:
    if target.capture is None:
        raise RuntimeError(f"probe target has no reviewed capture spec: {target.name}")
    return target.capture


def _capture_spec_from_fitrace_definition(definition: dict[str, Any] | None) -> CaptureSpec:
    if not isinstance(definition, dict):
        return CaptureSpec()
    inputs = definition.get("inputs")
    if not isinstance(inputs, dict):
        return CaptureSpec()
    input_names = [name for name in inputs if isinstance(name, str) and name]
    return CaptureSpec(structural_attr_tokens=input_names)


def should_dump_value(value: Any, *, capture: CaptureSpec, force: bool = False) -> bool:
    """Return whether a value should be copied into a capture payload."""
    if force and _is_tensor(value):
        return True
    if not _is_tensor(value):
        return isinstance(value, (str, int, float, bool)) or value is None
    if _is_integer_like_tensor(value):
        return True
    return False


def _to_cpu_tensor(value: Any) -> Any:
    tensor = value.detach()
    cpu = getattr(tensor, "cpu", None)
    return cpu() if callable(cpu) else tensor


def capture_value(value: Any, *, capture: CaptureSpec, force: bool = False) -> dict[str, Any]:
    """Return a selective payload record for one argument value."""
    summary = summarize_value(value)
    if not should_dump_value(value, capture=capture, force=force):
        attrs = capture_structural_attrs(value, capture=capture)
        record: dict[str, Any] = {"saved": False, "summary": summary}
        if attrs:
            record["attrs"] = attrs
        return record
    if _is_tensor(value):
        return {"saved": True, "summary": summary, "value": _to_cpu_tensor(value)}
    return {"saved": True, "summary": summary, "value": value}


def capture_structural_attrs(value: Any, *, capture: CaptureSpec) -> dict[str, Any]:
    """Capture selected structural tensor attrs from wrapper-like objects."""
    raw_attrs = getattr(value, "__dict__", None)
    captured: dict[str, Any] = {}
    tokens = list(dict.fromkeys(capture.structural_attr_tokens))
    if isinstance(raw_attrs, dict):
        for attr_name, attr_value in sorted(raw_attrs.items()):
            lowered = attr_name.lower()
            if not any(token in lowered for token in tokens):
                continue
            captured[attr_name] = capture_value(attr_value, capture=capture)
    for token in tokens:
        if token in captured or not token.isidentifier():
            continue
        try:
            attr_value = getattr(value, token)
        except Exception:
            continue
        captured[token] = capture_value(attr_value, capture=capture)
    return captured


def _resolve_attr_parent(root: Any, attr_path: str) -> tuple[Any, str]:
    parts = attr_path.split(".")
    if not parts or any(not part for part in parts):
        raise ValueError(f"invalid attribute path: {attr_path}")
    parent = root
    for part in parts[:-1]:
        parent = getattr(parent, part)
    return parent, parts[-1]


class CaptureSession:
    """Install probe hooks and append lightweight events to JSONL."""

    def __init__(
        self,
        *,
        probe_plan: ProbePlan,
        output_dir: Path,
        reset_output: bool = True,
        max_captures_per_target: int,
    ) -> None:
        self.probe_plan = probe_plan
        self.output_dir = output_dir
        self.events_path = output_dir / "events.jsonl"
        self.captures_dir = output_dir / "captures"
        self.reset_output = reset_output
        if type(max_captures_per_target) is not int or max_captures_per_target < 1:
            raise ValueError("max_captures_per_target must be a positive integer")
        self.max_captures_per_target = max_captures_per_target
        self._hooks: list[InstalledHook] = []
        self._warmup_depth = 0
        self._event_index = 0
        self._capture_counts: dict[tuple[str, str], int] = defaultdict(int)
        self.install_status: list[dict[str, Any]] = []

    @property
    def is_warmup(self) -> bool:
        """Whether a reviewed warmup window is currently on the call stack."""
        return self._warmup_depth > 0

    def install(self) -> None:
        """Install wrappers for every target in the probe plan.

        Targets that share the same Python callable are grouped so that a
        single hook dispatches to each matching target using reviewed dispatch
        rules from the probe plan.
        """
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.captures_dir.mkdir(parents=True, exist_ok=True)
        if self.reset_output:
            self.events_path.write_text("", encoding="utf-8")
        else:
            self.events_path.touch(exist_ok=True)
        self.install_status = []

        # Group targets by (module_name, attr_path) so shared callables get one hook.
        by_callable: dict[tuple[str, str], list[ProbeTarget]] = defaultdict(list)
        for target in self.probe_plan.targets:
            by_callable[(target.module, target.attr)].append(target)

        for (module_name, attr_path), targets in by_callable.items():
            statuses = [
                {"name": t.name, "target": t.target, "module": module_name, "attr": attr_path, "installed": False}
                for t in targets
            ]
            try:
                module = importlib.import_module(module_name)
                parent, attr = _resolve_attr_parent(module, attr_path)
                original = getattr(parent, attr)
                if not callable(original):
                    raise TypeError(f"target is not callable: {module_name}.{attr_path}")
                if len(targets) == 1:
                    wrapper = self._make_wrapper(targets[0], original)
                else:
                    wrapper = self._make_multi_dispatch_wrapper(targets, original)
                setattr(parent, attr, wrapper)
                self._hooks.append(InstalledHook(module=parent, attr=attr, original=original))
                for s in statuses:
                    s["installed"] = True
            except Exception as exc:
                for s in statuses:
                    s["error_type"] = type(exc).__name__
                    s["error"] = str(exc)
                self.install_status.extend(statuses)
                raise
            self.install_status.extend(statuses)

        self._install_companion_hooks()
        self._install_warmup_hooks()

    def _install_companion_hooks(self) -> None:
        """Wrap reviewed companion methods (e.g. plan/begin_forward/forward).

        FlashInfer wrappers receive some definition inputs on a sibling call,
        not on the hooked ``run`` call. ``sm_scale`` is passed to the deprecated
        ``forward`` (which stashes it and calls ``run``), and structural index
        tensors are passed to ``plan``/``begin_forward``. Each companion call's
        named arguments are stashed on the wrapper instance and merged into the
        next ``run`` capture by name. This is generic: core records whatever the
        companion received, with no per-kernel parameter knowledge. Which methods
        to watch comes from the reviewed ``companion_attrs`` on the target.

        A reviewed companion must be hookable. Missing or invalid companions
        fail the probe instead of silently producing incomplete captures.
        """
        by_companion: dict[tuple[str, str], list[ProbeTarget]] = defaultdict(list)
        for target in self.probe_plan.targets:
            if "." not in target.attr:
                for companion in target.companion_attrs:
                    raise ValueError(
                        f"companion_attrs invalid for module-level target {target.name}: "
                        f"{target.attr}.{companion}"
                    )
                continue
            class_path = target.attr.rsplit(".", 1)[0]
            for companion in target.companion_attrs:
                companion_attr = f"{class_path}.{companion}"
                by_companion[(target.module, companion_attr)].append(target)

        for (module_name, companion_attr), targets in by_companion.items():
            capture = CaptureSpec.merge([_require_capture_spec(target) for target in targets])
            first_target = targets[0]
            status: dict[str, Any] = {
                "kind": "companion",
                "name": first_target.name,
                "module": module_name,
                "attr": companion_attr,
                "installed": False,
            }
            try:
                module = importlib.import_module(module_name)
                parent, attr = _resolve_attr_parent(module, companion_attr)
                original = getattr(parent, attr)
                if not callable(original):
                    raise TypeError(f"companion is not callable: {module_name}.{companion_attr}")
                wrapper = self._make_companion_wrapper(original, capture)
                setattr(parent, attr, wrapper)
                self._hooks.append(InstalledHook(module=parent, attr=attr, original=original))
                status["installed"] = True
            except Exception as exc:
                status["error_type"] = type(exc).__name__
                status["error"] = str(exc)
                self.install_status.append(status)
                raise
            self.install_status.append(status)

    def _install_warmup_hooks(self) -> None:
        """Wrap reviewed warmup callables so nested events are tagged is_warmup.

        A reviewed warmup callable must be hookable. Otherwise the run could
        silently label warmup events as real traffic, so installation failures
        abort the probe.
        """
        for hook in self.probe_plan.warmup_hooks:
            status: dict[str, Any] = {
                "kind": "warmup",
                "name": hook.name,
                "module": hook.module,
                "attr": hook.attr,
                "installed": False,
            }
            try:
                module = importlib.import_module(hook.module)
                parent, attr = _resolve_attr_parent(module, hook.attr)
                original = getattr(parent, attr)
                if not callable(original):
                    raise TypeError(f"warmup target is not callable: {hook.module}.{hook.attr}")
                wrapper = self._make_warmup_wrapper(original)
                setattr(parent, attr, wrapper)
                self._hooks.append(InstalledHook(module=parent, attr=attr, original=original))
                status["installed"] = True
            except Exception as exc:
                status["error_type"] = type(exc).__name__
                status["error"] = str(exc)
                self.install_status.append(status)
                raise
            self.install_status.append(status)

    def uninstall(self) -> None:
        """Restore all installed wrappers in reverse order."""
        while self._hooks:
            hook = self._hooks.pop()
            setattr(hook.module, hook.attr, hook.original)

    def _make_wrapper(self, target: ProbeTarget, original: Callable[..., Any]) -> Callable[..., Any]:
        def wrapped(*args: Any, **kwargs: Any) -> Any:
            if self._target_matches_dispatch(target, args):
                traced_definition = self._trace_definition(original, target, args, kwargs)
                self._write_event(target, args, kwargs, traced_definition=traced_definition)
            return original(*args, **kwargs)

        wrapped.__name__ = getattr(original, "__name__", target.attr)
        wrapped.__doc__ = getattr(original, "__doc__", None)
        return wrapped

    @staticmethod
    def _trace_definition(
        original: Callable[..., Any],
        target: ProbeTarget,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> dict[str, Any] | None:
        """Return the fitrace definition for this exact call when required."""
        if not target.collect or target.backend != "flashinfer":
            return None
        trace_fn = getattr(original, "fi_trace", None)
        if not callable(trace_fn):
            raise RuntimeError(f"collect target has no callable fi_trace: {target.name}")
        try:
            signature = inspect.signature(original)
            bound = signature.bind(*args, **kwargs)
            bound.apply_defaults()
        except (TypeError, ValueError) as exc:
            raise RuntimeError(f"failed to bind call for fi_trace target {target.name}: {exc}") from exc
        definition = trace_fn(**dict(bound.arguments))
        if not isinstance(definition, dict) or not isinstance(definition.get("name"), str) or not definition["name"]:
            raise RuntimeError(f"fi_trace returned no definition name for target {target.name}")
        definition = dict(definition)
        definition["_capture_arg_names"] = list(bound.arguments)[:len(args)]
        return definition

    def _make_warmup_wrapper(self, original: Callable[..., Any]) -> Callable[..., Any]:
        """Mark the warmup window for the duration of the wrapped call.

        Uses a depth counter so nested or repeated warmup calls toggle the flag
        correctly, and always clears in finally even if warmup raises.
        """
        def wrapped(*args: Any, **kwargs: Any) -> Any:
            self._warmup_depth += 1
            try:
                return original(*args, **kwargs)
            finally:
                self._warmup_depth = max(0, self._warmup_depth - 1)

        wrapped.__name__ = getattr(original, "__name__", "warmup")
        wrapped.__doc__ = getattr(original, "__doc__", None)
        return wrapped

    def _make_companion_wrapper(self, original: Callable[..., Any], capture: CaptureSpec) -> Callable[..., Any]:
        """Stash a companion call's named arguments on the wrapper instance.

        Does not write an event; only records arguments so the following ``run``
        capture can merge them by name.
        """
        try:
            signature: inspect.Signature | None = inspect.signature(original)
        except (TypeError, ValueError):
            signature = None

        def wrapped(*args: Any, **kwargs: Any) -> Any:
            if args is not None and len(args) >= 1:
                self._stash_companion_args(args[0], signature, args, kwargs, capture)
            return original(*args, **kwargs)

        wrapped.__name__ = getattr(original, "__name__", "companion")
        wrapped.__doc__ = getattr(original, "__doc__", None)
        return wrapped

    @staticmethod
    def _stash_companion_args(
        instance: Any,
        signature: inspect.Signature | None,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
        capture: CaptureSpec,
    ) -> None:
        """Record only the arguments actually passed, keyed by parameter name."""
        recorded: dict[str, Any] = {}
        if signature is not None:
            bound = signature.bind(*args, **kwargs)
            for name, value in bound.arguments.items():
                if name == "self":
                    continue
                parameter = signature.parameters.get(name)
                kind = parameter.kind if parameter is not None else None
                if kind == inspect.Parameter.VAR_POSITIONAL:
                    continue
                if kind == inspect.Parameter.VAR_KEYWORD and isinstance(value, dict):
                    for var_name, var_value in value.items():
                        recorded[var_name] = capture_value(
                            var_value,
                            capture=capture,
                            force=var_name in capture.full_kwargs,
                        )
                    continue
                recorded[name] = capture_value(value, capture=capture, force=name in capture.full_kwargs)
        else:
            for name, value in kwargs.items():
                recorded[name] = capture_value(value, capture=capture, force=name in capture.full_kwargs)
        if not recorded:
            return
        existing = getattr(instance, COMPANION_CAPTURE_ATTR, None)
        merged = dict(existing) if isinstance(existing, dict) else {}
        merged.update(recorded)
        setattr(instance, COMPANION_CAPTURE_ATTR, merged)

    def _make_multi_dispatch_wrapper(
        self, targets: list[ProbeTarget], original: Callable[..., Any]
    ) -> Callable[..., Any]:
        """One hook dispatches to all targets sharing the same callable.

        Each target is written only when its reviewed dispatch rules match the
        actual runtime call arguments.
        """
        def wrapped(*args: Any, **kwargs: Any) -> Any:
            scope = self._read_capture_scope()
            for target in targets:
                if not self._target_allowed_by_scope(target, scope):
                    continue
                if not self._target_matches_dispatch(target, args):
                    continue
                traced_definition = self._trace_definition(original, target, args, kwargs)
                self._write_event(target, args, kwargs, scope, traced_definition=traced_definition)
            return original(*args, **kwargs)

        wrapped.__name__ = getattr(original, "__name__", "")
        wrapped.__doc__ = getattr(original, "__doc__", None)
        return wrapped

    def _read_capture_scope(self) -> dict[str, Any]:
        """Read the current capture scope shared with worker processes."""
        scope_path = self.output_dir / CAPTURE_SCOPE_FILENAME
        if not scope_path.exists():
            return {}
        try:
            data = json.loads(scope_path.read_text(encoding="utf-8"))
        except Exception:
            return {}
        return data if isinstance(data, dict) else {}

    @staticmethod
    def _target_allowed_by_scope(target: ProbeTarget, scope: dict[str, Any]) -> bool:
        allowed = scope.get("allowed_op_types")
        if not isinstance(allowed, list) or not allowed:
            return True
        allowed_op_types = {item for item in allowed if isinstance(item, str)}
        return bool(target.op_type and target.op_type in allowed_op_types)

    @staticmethod
    def _target_matches_dispatch(target: ProbeTarget, args: tuple[Any, ...]) -> bool:
        if target.dispatch is None:
            return True
        expected = target.dispatch_value if target.dispatch_value is not None else getattr(target, target.dispatch.field, None)
        if expected is None:
            return False
        return infer_dispatch_value(target.dispatch, args) == expected

    def _write_event(
        self,
        target: ProbeTarget,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
        scope: dict[str, Any] | None = None,
        traced_definition: dict[str, Any] | None = None,
    ) -> None:
        if scope is None:
            scope = self._read_capture_scope()
        if not self._target_allowed_by_scope(target, scope):
            return
        scope_name = scope.get("name") if isinstance(scope.get("name"), str) else "default"
        capture_key = (target.name, scope_name)
        if self.max_captures_per_target > 0 and self._capture_counts[capture_key] >= self.max_captures_per_target:
            return
        self._capture_counts[capture_key] += 1
        self._event_index += 1
        capture_path = self.captures_dir / f"{self._event_index:06d}_{target.name}.pt"
        definition_name = (
            traced_definition.get("name")
            if isinstance(traced_definition, dict) and isinstance(traced_definition.get("name"), str)
            else target.definition_name
        )
        actual_capture_path = self._write_capture_file(
            capture_path,
            target,
            args,
            kwargs,
            scope,
            definition_name=definition_name,
            traced_definition=traced_definition,
        )
        event = {
            "schema_version": CAPTURE_SCHEMA_VERSION,
            "name": target.name,
            "definition_name": definition_name,
            "target": target.target,
            "op_type": target.op_type,
            "variant": target.variant,
            "probe_mode": target.probe_mode,
            "active_probe_mode": os.environ.get("FLASHINFER_TRACE_ACTIVE_PROBE_MODE"),
            "capture_scope": scope_name,
            "is_warmup": self.is_warmup,
            "capture_path": str(actual_capture_path),
            "capture_format": "torch.pt" if actual_capture_path.suffix == ".pt" else "json",
        }
        if target.dispatch is not None:
            detected = infer_dispatch_value(target.dispatch, args)
            if detected is not None:
                event[target.dispatch.field] = detected
        with self.events_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(event, ensure_ascii=False, default=repr) + "\n")

    @staticmethod
    def _capture_call_with_companions(
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
        target: ProbeTarget,
        traced_definition: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Build the run capture payload and merge stashed companion arguments.

        Companion arguments (from plan/begin_forward/forward) are merged into
        ``kwargs`` by name; the run call's own arguments take precedence.
        """
        capture = CaptureSpec.merge([
            _require_capture_spec(target),
            _capture_spec_from_fitrace_definition(traced_definition),
        ])
        full_positions = set(capture.full_args)
        full_kwargs = set(capture.full_kwargs)
        call_payload = {
            "args": [
                capture_value(arg, capture=capture, force=index in full_positions)
                for index, arg in enumerate(args)
            ],
            "kwargs": {
                key: capture_value(value, capture=capture, force=key in full_kwargs)
                for key, value in sorted(kwargs.items())
            },
        }
        if not target.companion_attrs or not args:
            return call_payload
        companion = getattr(args[0], COMPANION_CAPTURE_ATTR, None)
        if not isinstance(companion, dict):
            return call_payload
        merged_kwargs = call_payload.get("kwargs")
        if not isinstance(merged_kwargs, dict):
            merged_kwargs = {}
            call_payload["kwargs"] = merged_kwargs
        for name, value in companion.items():
            merged_kwargs.setdefault(name, value)
        return call_payload

    def _write_capture_file(
        self,
        capture_path: Path,
        target: ProbeTarget,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
        scope: dict[str, Any] | None = None,
        definition_name: str | None = None,
        traced_definition: dict[str, Any] | None = None,
    ) -> Path:
        scope = scope or {}
        payload = {
            "schema_version": CAPTURE_SCHEMA_VERSION,
            "name": target.name,
            "definition_name": definition_name,
            "target": target.target,
            "op_type": target.op_type,
            "variant": target.variant,
            "probe_mode": target.probe_mode,
            "active_probe_mode": os.environ.get("FLASHINFER_TRACE_ACTIVE_PROBE_MODE"),
            "capture_scope": scope.get("name") if isinstance(scope.get("name"), str) else "default",
            "is_warmup": self.is_warmup,
            "payload": self._capture_call_with_companions(args, kwargs, target, traced_definition),
        }
        import torch

        torch.save(payload, capture_path)
        return capture_path

    def __enter__(self) -> CaptureSession:
        self.install()
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        self.uninstall()


# ---------------------------------------------------------------------------
# Worker-process bootstrap
# ---------------------------------------------------------------------------
#
# This entry point is imported from an injected ``sitecustomize.py``. That
# matters because SGLang may run model execution in child Python worker
# processes; hooks installed only in the Modal parent process do not affect
# those workers.

PLAN_ENV = "FLASHINFER_TRACE_PROBE_PLAN"
OUTPUT_ENV = "FLASHINFER_TRACE_OUTPUT_DIR"

_SESSION: CaptureSession | None = None


def _write_bootstrap_status(output_dir: Path, status: dict[str, Any]) -> None:
    status_dir = output_dir / "hook_status"
    status_dir.mkdir(parents=True, exist_ok=True)
    path = status_dir / f"{os.getpid()}.json"
    path.write_text(json.dumps(status, indent=2, ensure_ascii=False), encoding="utf-8")


def install_from_env() -> None:
    """Install approved probe hooks in the current worker process."""
    global _SESSION
    if _SESSION is not None:
        return

    plan_path = os.environ.get(PLAN_ENV)
    output_dir_raw = os.environ.get(OUTPUT_ENV)
    if not plan_path or not output_dir_raw:
        return

    output_dir = Path(output_dir_raw)
    status: dict[str, Any] = {
        "pid": os.getpid(),
        "plan_path": plan_path,
        "output_dir": str(output_dir),
        "installed": False,
        "hooks_installed": 0,
    }
    try:
        install_hf_config_override_from_env()
        payload = json.loads(Path(plan_path).read_text(encoding="utf-8"))
        probe_plan = ProbePlan.from_jsonable(payload)
        max_captures_per_target = max_captures_from_plan_payload(payload)
        session = CaptureSession(
            probe_plan=probe_plan,
            output_dir=output_dir,
            reset_output=False,
            max_captures_per_target=max_captures_per_target,
        )
        session.install()
        _SESSION = session
        status["installed"] = True
        status["hooks_installed"] = sum(1 for item in session.install_status if item.get("installed"))
        status["hook_status"] = session.install_status
    except Exception as exc:  # noqa: BLE001 - startup diagnostics must not crash the worker.
        status["error_type"] = type(exc).__name__
        status["error"] = str(exc)
        print(f"[flashinfer_trace worker bootstrap] failed: {exc}", file=sys.stderr, flush=True)
    finally:
        try:
            _write_bootstrap_status(output_dir, status)
        except Exception as exc:  # noqa: BLE001
            print(
                f"[flashinfer_trace worker bootstrap] failed to write status: {exc}",
                file=sys.stderr,
                flush=True,
            )
