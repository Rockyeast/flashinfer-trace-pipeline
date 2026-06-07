"""Baseline solution adapter for simple GEMM definitions."""

from __future__ import annotations

import textwrap

from ._solution_utils import has_inputs, solution_payload

OP_TYPE = "gemm"


def pr_reference_source() -> str:
    return "PyTorch GEMM reference"


def pr_kernel_description(_def_name: str, model_label: str) -> str:
    return f"{model_label} GEMM"


def pr_baseline_description(_def_name: str) -> str:
    return "torch.nn.functional.linear"


def skip_submit_definition(def_name: str, op_type: str) -> bool:
    """GEMM definitions are local/reference-only for now, not dataset submit scope."""
    return op_type in {"gemm", "gemm_bf16"} or def_name.startswith(
        ("gemm_", "gemm_bf16_")
    )


def definition_name(_variant: str, params: dict) -> str:
    """Return the canonical GEMM definition name."""
    return f"gemm_n{params['N']}_k{params['K']}"


def classify_trace_id(_trace_id: str) -> tuple[str, str] | None:
    return None


def build_kernels(_matched_kernels: dict[str, dict]) -> list[dict]:
    return []


def generate_definition(_kernel: dict, _model_tag: str, _tp: int) -> dict:
    raise NotImplementedError("GEMM definitions are generated outside probe adapters.")


def generate_baseline_solution(definition: dict) -> dict | None:
    """Generate a correctness-first PyTorch GEMM baseline."""
    if definition.get("op_type") != OP_TYPE or not has_inputs(definition, {"A", "B"}):
        return None
    source = textwrap.dedent("""\
        import torch
        import torch.nn.functional as F


        def run(A: torch.Tensor, B: torch.Tensor):
            return F.linear(A, B)
        """) + "\n"
    return solution_payload(
        definition,
        source,
        name_prefix="torch_matmul",
        author="baseline",
        dependencies=[],
        description="Baseline GEMM solution using torch.nn.functional.linear.",
        target_hardware=["NVIDIA_H100", "NVIDIA_A100", "CPU"],
    )
