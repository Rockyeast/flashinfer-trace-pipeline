"""Registered definition repair rules."""

from __future__ import annotations

from flashinfer_trace.definition_repairs.base import DefinitionRepairer
from flashinfer_trace.definition_repairs.cascade_merge import CASCADE_MERGE_REPAIRER
from flashinfer_trace.definition_repairs.gdn import GDN_REPAIRER
from flashinfer_trace.definition_repairs.gqa_paged import GQA_PAGED_REPAIRER
from flashinfer_trace.definition_repairs.gqa_ragged import GQA_RAGGED_REPAIRER
from flashinfer_trace.definition_repairs.rmsnorm import RMSNORM_REPAIRER
from flashinfer_trace.definition_repairs.sampling import SAMPLING_REPAIRER
from flashinfer_trace.definition_repairs.silu_and_mul import SILU_AND_MUL_REPAIRER


DEFINITION_REPAIRERS: tuple[DefinitionRepairer, ...] = (
    GQA_PAGED_REPAIRER,
    GQA_RAGGED_REPAIRER,
    SAMPLING_REPAIRER,
    GDN_REPAIRER,
    RMSNORM_REPAIRER,
    SILU_AND_MUL_REPAIRER,
    CASCADE_MERGE_REPAIRER,
)

__all__ = ["DEFINITION_REPAIRERS", "DefinitionRepairer"]
