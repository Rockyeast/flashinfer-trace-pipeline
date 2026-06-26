"""Definition repair rule interface."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable


IssueFn = Callable[[dict[str, Any]], str | None]
RepairFn = Callable[[dict[str, Any], list[dict[str, Any]]], tuple[dict[str, Any] | None, str]]
HintsFn = Callable[[dict[str, Any], list[dict[str, Any]]], dict[str, Any] | None]


@dataclass(frozen=True)
class DefinitionRepairer:
    name: str
    issue: IssueFn
    repair: RepairFn
    hints: HintsFn | None = None
