"""Small schemas for the v3 reviewed-target core.

This module intentionally uses stdlib dataclasses instead of Pydantic. The v3
core should validate only the fields it needs to make deterministic decisions.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any, Literal, get_args


TargetRole = Literal["target", "warmup"]


def _jsonable_scalar(value: Any) -> Any:
    """Make a single dataclass field value JSON-serializable."""
    if isinstance(value, Path):
        return str(value)
    return value


def target_to_jsonable(target: Any) -> dict[str, Any]:
    """Serialize a target dataclass from its fields (single source of truth).

    Using ``dataclasses.asdict`` means new schema fields are persisted
    automatically; there is no hand-maintained field whitelist to drift out of
    sync and silently drop data.
    """
    return {key: _jsonable_scalar(value) for key, value in asdict(target).items()}


def max_captures_from_plan_payload(payload: dict[str, Any]) -> int:
    capture_limits = payload.get("capture_limits")
    if not isinstance(capture_limits, dict):
        raise ValueError("modal probe plan missing capture_limits")
    value = capture_limits.get("max_captures_per_target")
    if type(value) is not int or value < 1:
        raise ValueError("capture_limits.max_captures_per_target must be a positive integer")
    return value


Backend = Literal["flashinfer", "sglang_kernel", "torch", "unknown"]
DefinitionSource = Literal["fitrace", "agent", "manual", "unknown"]
ProbeMode = Literal["default", "paged", "both"]
DispatchRuleKind = Literal["arg_attr", "arg_shape"]

# Runtime-checkable sets derived from the Literal types above, so the allowed
# values live in exactly one place and cannot drift.
PROBE_MODES: set[str] = set(get_args(ProbeMode))
BACKENDS: set[str] = set(get_args(Backend))
DEFINITION_SOURCES: set[str] = set(get_args(DefinitionSource))
DISPATCH_RULE_KINDS: set[str] = set(get_args(DispatchRuleKind))


@dataclass(frozen=True)
class CaptureSpec:
    """Reviewed capture policy for one runtime target."""

    full_args: list[int] = field(default_factory=list)
    full_kwargs: list[str] = field(default_factory=list)
    structural_attr_tokens: list[str] = field(default_factory=list)

    @classmethod
    def merge(cls, specs: list["CaptureSpec"]) -> "CaptureSpec":
        """Merge reviewed capture policies without duplicating field names."""
        if not specs:
            raise ValueError("cannot merge empty capture spec list")
        merged: dict[str, set[Any]] = {field_info.name: set() for field_info in fields(cls)}
        for spec in specs:
            for field_info in fields(cls):
                value = getattr(spec, field_info.name)
                if not isinstance(value, list):
                    raise TypeError(f"capture field {field_info.name} must be a list")
                merged[field_info.name].update(value)
        return cls(**{name: sorted(values) for name, values in merged.items()})


@dataclass(frozen=True)
class DispatchRule:
    """One reviewed rule for extracting a dispatch value from call arguments."""

    kind: DispatchRuleKind
    arg_index: int
    attrs: list[str] = field(default_factory=list)
    tuple_index: int | None = None
    min_rank: int | None = None
    rank: int | None = None
    shape_index: int | None = None
    equals_index: int | None = None
    equals: int | None = None
    value: int | None = None


@dataclass(frozen=True)
class DispatchSpec:
    """Reviewed dispatch policy for shared hook callables."""

    field: str
    rules: list[DispatchRule] = field(default_factory=list)


def dispatch_spec_from_jsonable(raw: Any, *, context: str) -> DispatchSpec | None:
    """Parse reviewed dispatch rules from JSON data."""
    if raw is None:
        return None
    if not isinstance(raw, dict):
        raise ValueError(f"{context} dispatch must be an object")
    unexpected = sorted(set(raw) - {"field", "rules"})
    if unexpected:
        raise ValueError(f"{context} dispatch has unexpected fields: {unexpected}")
    field_name = raw.get("field")
    if not isinstance(field_name, str) or not field_name:
        raise ValueError(f"{context} dispatch.field must be a non-empty string")
    raw_rules = raw.get("rules")
    if not isinstance(raw_rules, list) or not raw_rules:
        raise ValueError(f"{context} dispatch.rules must be a non-empty list")
    rules = [
        _dispatch_rule_from_jsonable(item, context=f"{context} dispatch.rules[{index}]")
        for index, item in enumerate(raw_rules)
    ]
    return DispatchSpec(field=field_name, rules=rules)


def _dispatch_rule_from_jsonable(raw: Any, *, context: str) -> DispatchRule:
    if not isinstance(raw, dict):
        raise ValueError(f"{context} must be an object")
    unexpected = sorted(set(raw) - {
        "kind",
        "arg_index",
        "attrs",
        "tuple_index",
        "min_rank",
        "rank",
        "shape_index",
        "equals_index",
        "equals",
        "value",
    })
    if unexpected:
        raise ValueError(f"{context} has unexpected fields: {unexpected}")
    kind = raw.get("kind")
    if kind not in DISPATCH_RULE_KINDS:
        raise ValueError(f"{context}.kind must be one of {sorted(DISPATCH_RULE_KINDS)}")
    arg_index = raw.get("arg_index")
    if type(arg_index) is not int or arg_index < 0:
        raise ValueError(f"{context}.arg_index must be a non-negative integer")
    attrs = raw.get("attrs", [])
    if not isinstance(attrs, list) or not all(isinstance(item, str) and item for item in attrs):
        raise ValueError(f"{context}.attrs must be a list of non-empty strings")
    ints: dict[str, int | None] = {}
    for key in ("tuple_index", "min_rank", "rank", "shape_index", "equals_index", "equals", "value"):
        value = raw.get(key)
        if value is not None and type(value) is not int:
            raise ValueError(f"{context}.{key} must be an integer")
        if key in {"tuple_index", "min_rank", "rank", "shape_index", "equals_index"} and isinstance(value, int) and value < 0:
            raise ValueError(f"{context}.{key} must be non-negative")
        ints[key] = value
    if kind == "arg_attr" and not attrs:
        raise ValueError(f"{context}.attrs is required for arg_attr")
    if kind == "arg_shape" and ints["value"] is None and ints["shape_index"] is None:
        raise ValueError(f"{context} must declare value or shape_index for arg_shape")
    return DispatchRule(
        kind=kind,
        arg_index=arg_index,
        attrs=sorted(set(attrs)),
        tuple_index=ints["tuple_index"],
        min_rank=ints["min_rank"],
        rank=ints["rank"],
        shape_index=ints["shape_index"],
        equals_index=ints["equals_index"],
        equals=ints["equals"],
        value=ints["value"],
    )


def capture_spec_from_jsonable(raw: Any, *, context: str) -> CaptureSpec:
    """Parse a reviewed capture policy from JSON data."""
    if not isinstance(raw, dict):
        raise ValueError(f"{context} capture must be an object")
    expected_fields = {field_info.name for field_info in fields(CaptureSpec)}
    unexpected = sorted(set(raw) - expected_fields)
    if unexpected:
        raise ValueError(f"{context} capture has unexpected fields: {unexpected}")
    full_args = raw.get("full_args", [])
    if (
        not isinstance(full_args, list)
        or not all(type(item) is int and item >= 0 for item in full_args)
    ):
        raise ValueError(f"{context} capture.full_args must be a list of non-negative integers")
    full_kwargs = raw.get("full_kwargs", [])
    if not isinstance(full_kwargs, list) or not all(isinstance(item, str) and item for item in full_kwargs):
        raise ValueError(f"{context} capture.full_kwargs must be a list of non-empty strings")
    tokens = raw.get("structural_attr_tokens", [])
    if not isinstance(tokens, list) or not all(isinstance(item, str) and item for item in tokens):
        raise ValueError(f"{context} capture.structural_attr_tokens must be a list of non-empty strings")
    return CaptureSpec(
        full_args=sorted(set(full_args)),
        full_kwargs=sorted(set(full_kwargs)),
        structural_attr_tokens=sorted(set(tokens)),
    )


@dataclass(frozen=True)
class DefinitionRef:
    """Minimal metadata loaded from one definition JSON."""

    name: str
    op_type: str
    path: Path
    tags: list[str] = field(default_factory=list)
    axes: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ApprovedTarget:
    """Human-reviewed target entry.

    Entries loaded from config are already reviewed; proposal lifecycle status
    belongs in proposal/candidates, not in the runtime config schema.
    """

    name: str
    role: TargetRole = "target"
    target: str | None = None
    module: str | None = None
    attr: str | None = None
    definition_source: DefinitionSource = "unknown"
    backend: Backend = "unknown"
    collect: bool = False
    definition_name: str | None = None
    op_type: str | None = None
    variant: str | None = None
    probe_mode: ProbeMode = "default"
    page_size: int | None = None
    dispatch_value: int | None = None
    companion_attrs: list[str] = field(default_factory=list)
    capture: CaptureSpec | None = None
    dispatch: DispatchSpec | None = None


@dataclass(frozen=True)
class ProbeTarget:
    """A reviewed target eligible for runtime probing."""

    name: str
    target: str
    module: str
    attr: str
    definition_name: str | None = None
    op_type: str | None = None
    variant: str | None = None
    backend: Backend = "unknown"
    collect: bool = False
    probe_mode: ProbeMode = "default"
    page_size: int | None = None
    dispatch_value: int | None = None
    companion_attrs: list[str] = field(default_factory=list)
    capture: CaptureSpec | None = None
    dispatch: DispatchSpec | None = None


@dataclass(frozen=True)
class WarmupHook:
    """A reviewed callable whose execution window marks events as warmup.

    The capture session wraps this callable; every event recorded while it is
    on the stack is tagged ``is_warmup=True``. This replaces count-based warmup
    guessing with an explicit SGLang warmup-window signal.
    """

    name: str
    module: str
    attr: str


@dataclass(frozen=True)
class ProbePlan:
    """Deterministic probe plan produced from approved targets."""

    targets: list[ProbeTarget]
    skipped: list[dict[str, str]]
    warmup_hooks: list[WarmupHook] = field(default_factory=list)

    def to_jsonable(self) -> dict[str, Any]:
        """Return a JSON-serializable probe plan."""
        return {
            "targets": [target_to_jsonable(target) for target in self.targets],
            "warmup_hooks": [target_to_jsonable(hook) for hook in self.warmup_hooks],
            "capture": {
                "format": "pt",
            },
            "skipped": self.skipped,
            "summary": {
                "targets": len(self.targets),
                "skipped": len(self.skipped),
                "warmup_hooks": len(self.warmup_hooks),
            },
        }

    @classmethod
    def from_jsonable(cls, payload: dict[str, Any]) -> "ProbePlan":
        """Rebuild a ProbePlan from a serialized plan (inverse of to_jsonable).

        Accepts either a bare probe-plan dict or a wrapper dict that nests the
        plan under ``"probe_plan"`` (as produced for Modal payloads).
        """
        probe_payload = payload.get("probe_plan", payload)
        raw_targets = probe_payload.get("targets", [])
        if not isinstance(raw_targets, list):
            raise ValueError("probe_plan.targets must be a list")

        targets: list[ProbeTarget] = []
        raw_skipped = probe_payload.get("skipped", [])
        skipped = list(raw_skipped) if isinstance(raw_skipped, list) else []
        for index, raw in enumerate(raw_targets):
            if not isinstance(raw, dict):
                raise ValueError(f"probe target #{index} must be an object")
            name = raw.get("name")
            target = raw.get("target")
            module = raw.get("module")
            attr = raw.get("attr")
            if not all(isinstance(value, str) and value for value in (name, target, module, attr)):
                raise ValueError(f"probe target {name or f'#{index}'} missing name/target/module/attr")
            probe_mode = raw.get("probe_mode")
            if probe_mode not in PROBE_MODES:
                raise ValueError(f"probe target {name} has invalid probe_mode: {probe_mode}")
            backend = raw.get("backend", "unknown")
            if backend not in BACKENDS:
                raise ValueError(f"probe target {name} has invalid backend: {backend}")
            collect = raw.get("collect", False)
            if not isinstance(collect, bool):
                raise ValueError(f"probe target {name} has invalid collect: {collect}")
            capture = capture_spec_from_jsonable(raw.get("capture"), context=f"probe target {name}")
            dispatch = dispatch_spec_from_jsonable(raw.get("dispatch"), context=f"probe target {name}")
            targets.append(
                ProbeTarget(
                    name=name,
                    target=target,
                    module=module,
                    attr=attr,
                    definition_name=raw.get("definition_name") if isinstance(raw.get("definition_name"), str) else None,
                    op_type=raw.get("op_type") if isinstance(raw.get("op_type"), str) else None,
                    variant=raw.get("variant") if isinstance(raw.get("variant"), str) else None,
                    backend=backend,
                    collect=collect,
                    probe_mode=probe_mode,
                    page_size=int(raw["page_size"]) if isinstance(raw.get("page_size"), int) else None,
                    dispatch_value=int(raw["dispatch_value"]) if isinstance(raw.get("dispatch_value"), int) else None,
                    companion_attrs=[
                        c for c in raw.get("companion_attrs", []) if isinstance(c, str) and c
                    ]
                    if isinstance(raw.get("companion_attrs"), list)
                    else [],
                    capture=capture,
                    dispatch=dispatch,
                )
            )

        warmup_hooks: list[WarmupHook] = []
        raw_warmup = probe_payload.get("warmup_hooks", [])
        if isinstance(raw_warmup, list):
            for index, raw in enumerate(raw_warmup):
                if not isinstance(raw, dict):
                    raise ValueError(f"warmup hook #{index} must be an object")
                name = raw.get("name")
                module = raw.get("module")
                attr = raw.get("attr")
                if not all(isinstance(value, str) and value for value in (name, module, attr)):
                    raise ValueError(f"warmup hook {name or f'#{index}'} missing name/module/attr")
                warmup_hooks.append(WarmupHook(name=name, module=module, attr=attr))

        return cls(targets=targets, skipped=skipped, warmup_hooks=warmup_hooks)


@dataclass(frozen=True)
class CollectTarget:
    """A reviewed target that is eligible for workload collection."""

    name: str
    definition_name: str
    op_type: str
    target: str
    backend: Backend
    collect: bool
    definition_path: Path
    page_size: int | None = None


@dataclass(frozen=True)
class CollectPlan:
    """Deterministic collect plan produced from definitions and approved targets."""

    targets: list[CollectTarget]
    skipped: list[dict[str, str]]

    def to_jsonable(self) -> dict[str, Any]:
        """Return a JSON-serializable plan."""
        return {
            "targets": [target_to_jsonable(target) for target in self.targets],
            "skipped": self.skipped,
            "summary": {
                "targets": len(self.targets),
                "skipped": len(self.skipped),
            },
        }

    @classmethod
    def from_jsonable(cls, payload: dict[str, Any]) -> "CollectPlan":
        """Rebuild a CollectPlan from a serialized collect plan."""
        raw_targets = payload.get("targets")
        if not isinstance(raw_targets, list):
            raise ValueError("collect_plan.targets must be a list")
        raw_skipped = payload.get("skipped", [])
        if not isinstance(raw_skipped, list):
            raise ValueError("collect_plan.skipped must be a list")
        skipped: list[dict[str, str]] = []
        for index, item in enumerate(raw_skipped):
            if not isinstance(item, dict) or not all(isinstance(k, str) and isinstance(v, str) for k, v in item.items()):
                raise ValueError(f"collect_plan.skipped[{index}] must be a string map")
            skipped.append(dict(item))
        return cls(
            targets=[
                _collect_target_from_jsonable(item, index=index)
                for index, item in enumerate(raw_targets)
            ],
            skipped=skipped,
        )


def _collect_target_from_jsonable(raw: Any, *, index: int) -> CollectTarget:
    if not isinstance(raw, dict):
        raise ValueError(f"collect target #{index} must be an object")
    expected = {field_info.name for field_info in fields(CollectTarget)}
    unexpected = sorted(set(raw) - expected)
    if unexpected:
        raise ValueError(f"collect target #{index} has unexpected fields: {unexpected}")
    name = raw.get("name")
    definition_name = raw.get("definition_name")
    op_type = raw.get("op_type")
    target = raw.get("target")
    definition_path = raw.get("definition_path")
    if not all(isinstance(value, str) and value for value in (name, definition_name, op_type, target, definition_path)):
        raise ValueError(f"collect target {name or f'#{index}'} missing name/definition_name/op_type/target/definition_path")
    backend = raw.get("backend")
    if backend not in BACKENDS:
        raise ValueError(f"collect target {name} has invalid backend: {backend}")
    collect = raw.get("collect")
    if type(collect) is not bool:
        raise ValueError(f"collect target {name} has invalid collect: {collect}")
    page_size = raw.get("page_size")
    if page_size is not None and (type(page_size) is not int or page_size < 1):
        raise ValueError(f"collect target {name} has invalid page_size: {page_size}")
    return CollectTarget(
        name=name,
        definition_name=definition_name,
        op_type=op_type,
        target=target,
        backend=backend,
        collect=collect,
        definition_path=Path(definition_path),
        page_size=page_size,
    )
