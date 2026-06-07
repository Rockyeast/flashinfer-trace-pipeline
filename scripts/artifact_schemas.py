"""Strict schemas for core FlashInfer Trace JSON artifacts.

The models validate pipeline boundaries only. Runtime code should continue to
pass plain dicts around internally.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class StrictModel(BaseModel):
    """Base model that rejects unknown fields by default."""

    model_config = ConfigDict(extra="forbid")


class AxisSpec(StrictModel):
    type: Literal["var", "const"]
    value: int | None = None
    description: str = ""

    @model_validator(mode="after")
    def validate_axis_value(self) -> "AxisSpec":
        if self.type == "const" and self.value is None:
            raise ValueError("const axis requires value")
        if self.type == "var" and self.value is not None:
            raise ValueError("var axis must not define value")
        return self


class TensorSpec(StrictModel):
    shape: list[str | int] | None = None
    dtype: str = "bfloat16"
    description: str = ""
    optional: bool = False


class Definition(StrictModel):
    """Schema for definitions/{op_type}/{name}.json."""

    name: str
    op_type: str
    description: str
    tags: list[str]
    axes: dict[str, AxisSpec]
    inputs: dict[str, TensorSpec]
    outputs: dict[str, TensorSpec]
    reference: str
    constraints: list[str] = Field(default_factory=list)


class WorkloadInput(StrictModel):
    type: Literal["scalar", "random", "safetensors", "pending_safetensors"]
    value: Any | None = None
    path: str | None = None
    tensor_key: str | None = None

    @model_validator(mode="after")
    def validate_payload(self) -> "WorkloadInput":
        if self.type == "scalar" and self.value is None:
            raise ValueError("scalar workload input requires value")
        if self.type in ("safetensors", "pending_safetensors"):
            if not self.path:
                raise ValueError(f"{self.type} workload input requires path")
            if not self.tensor_key:
                raise ValueError(f"{self.type} workload input requires tensor_key")
        if self.type == "random":
            extras = {
                "value": self.value,
                "path": self.path,
                "tensor_key": self.tensor_key,
            }
            present = [name for name, value in extras.items() if value is not None]
            if present:
                raise ValueError(f"random workload input must not define {present}")
        return self


class Workload(StrictModel):
    uuid: str
    axes: dict[str, int]
    inputs: dict[str, WorkloadInput]


class WorkloadEntry(StrictModel):
    """Schema for one JSONL row in workloads/{op_type}/{name}.jsonl."""

    definition: str
    workload: Workload
    solution: Any | None
    evaluation: Any | None


class SolutionSource(StrictModel):
    path: str
    content: str


class SolutionSpec(StrictModel):
    language: str
    target_hardware: list[str]
    entry_point: str
    dependencies: list[str]
    validation_policy: str | None = None
    destination_passing_style: bool
    binding: Any | None


class Solution(StrictModel):
    """Schema for solutions/{author}/{op_type}/{definition}/{name}.json."""

    name: str
    definition: str
    author: str
    description: str
    spec: SolutionSpec
    sources: list[SolutionSource]


def validate_definition(data: dict[str, Any]) -> dict[str, Any]:
    """Validate a definition dict and return the original dict."""

    Definition.model_validate(data)
    return data


def validate_workload_entry(data: dict[str, Any]) -> dict[str, Any]:
    """Validate a workload JSONL entry dict and return the original dict."""

    WorkloadEntry.model_validate(data)
    return data


def validate_solution(data: dict[str, Any]) -> dict[str, Any]:
    """Validate a solution dict and return the original dict."""

    Solution.model_validate(data)
    return data
