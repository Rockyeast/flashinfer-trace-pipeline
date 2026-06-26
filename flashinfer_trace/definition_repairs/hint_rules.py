"""Shared sanitize hint rule handlers."""

from __future__ import annotations

from typing import Any, Callable


AxisHintHandler = Callable[[Any, dict[str, Any], dict[str, int]], int | None]


def _last_int(value: Any) -> int | None:
    try:
        if value is not None and value.numel() > 0:
            return int(value.reshape(-1)[-1].item())
    except Exception:
        return None
    return None


def _max_plus_one(value: Any, limit: int | None = None) -> int | None:
    try:
        if value is None:
            return None
        flat = value.reshape(-1)
        if limit is not None:
            flat = flat[:limit]
        if flat.numel() > 0:
            return int(flat.max().item()) + 1
    except Exception:
        return None
    return None


def _axis_tensor_last(value: Any, rule: dict[str, Any], axes: dict[str, int]) -> int | None:
    return _last_int(value)


def _axis_tensor_max_plus_one(value: Any, rule: dict[str, Any], axes: dict[str, int]) -> int | None:
    limit_axis = rule.get("limit_axis")
    limit = axes.get(limit_axis) if isinstance(limit_axis, str) else None
    return _max_plus_one(value, limit)


def _axis_tensor_numel_minus_one(value: Any, rule: dict[str, Any], axes: dict[str, int]) -> int | None:
    try:
        if value is not None and value.numel() > 0:
            return max(0, int(value.numel()) - 1)
    except Exception:
        return None
    return None


AXIS_HINT_RULES: dict[str, AxisHintHandler] = {
    "tensor_last": _axis_tensor_last,
    "tensor_max_plus_one": _axis_tensor_max_plus_one,
    "tensor_numel_minus_one": _axis_tensor_numel_minus_one,
}


def apply_axis_hint_rule(rule: dict[str, Any], value: Any, axes: dict[str, int]) -> int | None:
    source = rule.get("source")
    handler = AXIS_HINT_RULES.get(source) if isinstance(source, str) else None
    return handler(value, rule, axes) if handler is not None else None
