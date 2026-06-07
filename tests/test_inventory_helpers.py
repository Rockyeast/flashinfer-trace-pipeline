from parse.inventory_helpers import (
    dedup_kernels_by_definition_name,
    merge_matched_kernel,
    observed_fi_api_from_trace_id,
    split_api_fields,
)


def test_observed_fi_api_from_trace_id_removes_wrapper_method_suffix() -> None:
    assert (
        observed_fi_api_from_trace_id("flashinfer.norm.rmsnorm")
        == "flashinfer.norm.rmsnorm"
    )
    assert (
        observed_fi_api_from_trace_id(
            "flashinfer.decode.BatchDecodeWithPagedKVCacheWrapper.run"
        )
        == "flashinfer.decode.BatchDecodeWithPagedKVCacheWrapper"
    )
    assert observed_fi_api_from_trace_id("sgl_kernel.elementwise.rmsnorm") is None


def test_split_api_fields_keeps_non_flashinfer_as_target_api() -> None:
    assert split_api_fields("flashinfer.norm.rmsnorm") == (
        "flashinfer.norm.rmsnorm",
        None,
    )
    assert split_api_fields("sglang.srt.layers.layernorm.RMSNorm.forward_cuda") == (
        None,
        "sglang.srt.layers.layernorm.RMSNorm.forward_cuda",
    )


def test_merge_matched_kernel_tracks_fi_and_target_api_boundaries() -> None:
    matched: dict[str, dict] = {}

    merge_matched_kernel(
        matched,
        key="rmsnorm:rmsnorm",
        op_type="rmsnorm",
        variant="rmsnorm",
        fi_api="sgl_kernel.elementwise.rmsnorm",
        source="unit",
        trace_id="sgl_kernel.elementwise.rmsnorm",
        count=2,
        signatures=[],
    )
    assert matched["rmsnorm:rmsnorm"]["fi_api"] is None
    assert matched["rmsnorm:rmsnorm"]["target_api"] == "sgl_kernel.elementwise.rmsnorm"

    merge_matched_kernel(
        matched,
        key="rmsnorm:rmsnorm",
        op_type="rmsnorm",
        variant="rmsnorm",
        fi_api="flashinfer.norm.rmsnorm",
        source="unit",
        trace_id="flashinfer.norm.rmsnorm",
        count=3,
        signatures=[],
    )
    assert matched["rmsnorm:rmsnorm"]["fi_api"] == "flashinfer.norm.rmsnorm"
    assert matched["rmsnorm:rmsnorm"]["target_api"] == "sgl_kernel.elementwise.rmsnorm"
    assert matched["rmsnorm:rmsnorm"]["total_count"] == 5


def test_dedup_kernels_merges_observation_metadata() -> None:
    kernels = [
        {
            "definition_name": "rmsnorm_h4096",
            "call_count": 1,
            "observed_kwargs": ["eps"],
            "trace_ids": ["flashinfer.norm.rmsnorm"],
            "runtime_evidence": {"function_call": 1},
            "fi_api": None,
        },
        {
            "definition_name": "rmsnorm_h4096",
            "call_count": 3,
            "observed_kwargs": ["eps"],
            "trace_ids": ["sgl_kernel.elementwise.rmsnorm"],
            "runtime_evidence": {"function_call": 2},
            "fi_api": "flashinfer.norm.rmsnorm",
        },
    ]

    [merged] = dedup_kernels_by_definition_name(kernels)

    assert merged["call_count"] == 3
    assert merged["observed_kwargs"] == ["eps"]
    assert merged["trace_ids"] == [
        "flashinfer.norm.rmsnorm",
        "sgl_kernel.elementwise.rmsnorm",
    ]
    assert merged["runtime_evidence"] == {"function_call": 3}
    assert merged["fi_api"] == "flashinfer.norm.rmsnorm"
