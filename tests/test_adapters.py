from adapters import ADAPTERS, build_kernels, classify_trace_id
from adapters._param_utils import analysis_allows_family


def _bucket(
    *,
    op_type: str,
    variant: str,
    fi_api: str | None,
    signatures: list[dict],
) -> dict:
    return {
        "op_type": op_type,
        "variant": variant,
        "fi_api": fi_api,
        "target_api": None,
        "trace_ids": [fi_api or f"{op_type}.{variant}"],
        "trace_counts": {fi_api or f"{op_type}.{variant}": 1},
        "total_count": 1,
        "signatures": signatures,
        "runtime_evidence": {"function_call": 1},
        "source": "unit",
        "classification_sources": ["unit"],
    }


def test_all_discovered_adapters_expose_required_contract() -> None:
    assert ADAPTERS
    for adapter in ADAPTERS:
        assert isinstance(adapter.OP_TYPE, str)
        assert adapter.classify_trace_id("flashinfer.unknown.noop") is None
        assert isinstance(adapter.build_kernels({}), list)
        assert callable(adapter.generate_definition)


def test_classify_trace_id_routes_common_flashinfer_apis() -> None:
    assert classify_trace_id("flashinfer.norm.rmsnorm") == ("rmsnorm", "rmsnorm")
    assert classify_trace_id("flashinfer.norm.fused_add_rmsnorm") == (
        "rmsnorm",
        "fused_add_rmsnorm",
    )
    assert classify_trace_id("flashinfer.sampling.top_k_sampling_from_probs") == (
        "sampling",
        "top_k",
    )
    assert classify_trace_id("flashinfer.sampling.top_k_top_p_sampling_from_probs") == (
        "sampling",
        "top_k_top_p",
    )


def test_static_analysis_filter_only_excludes_after_successful_scan() -> None:
    assert analysis_allows_family({"available": False}, "has_moe")
    assert analysis_allows_family({}, "has_moe")
    assert analysis_allows_family({"available": True, "has_moe": True}, "has_moe")
    assert not analysis_allows_family({"available": True, "has_moe": False}, "has_moe")
    assert not analysis_allows_family({"available": True}, "has_moe")


def test_rmsnorm_adapter_builds_kernel_from_tensor_shape() -> None:
    signatures = [
        {
            "trace_id": "flashinfer.norm.rmsnorm",
            "signature": {
                "args": [
                    {"type": "tensor", "shape": [2, 4096], "dtype": "bfloat16"},
                    {"type": "tensor", "shape": [4096], "dtype": "bfloat16"},
                ],
                "kwargs": {},
            },
            "param_names": ["input", "weight"],
        }
    ]
    matched = {
        "rmsnorm:rmsnorm": _bucket(
            op_type="rmsnorm",
            variant="rmsnorm",
            fi_api="flashinfer.norm.rmsnorm",
            signatures=signatures,
        )
    }

    kernels = build_kernels(matched)

    assert len(kernels) == 1
    assert kernels[0]["definition_name"] == "rmsnorm_h4096"
    assert kernels[0]["params"] == {"hidden_size": 4096}
    assert kernels[0]["fi_api"] == "flashinfer.norm.rmsnorm"


def test_sampling_adapter_builds_kernel_from_probability_shape() -> None:
    signatures = [
        {
            "trace_id": "flashinfer.sampling.top_k_sampling_from_probs",
            "signature": {
                "args": [
                    {"type": "tensor", "shape": [1, 151936], "dtype": "float32"},
                    {"type": "tensor", "shape": [1], "dtype": "int32"},
                ],
                "kwargs": {},
            },
            "param_names": ["probs", "top_k"],
        }
    ]
    matched = {
        "sampling:top_k": _bucket(
            op_type="sampling",
            variant="top_k",
            fi_api="flashinfer.sampling.top_k_sampling_from_probs",
            signatures=signatures,
        )
    }

    kernels = build_kernels(matched)

    assert len(kernels) == 1
    assert kernels[0]["definition_name"] == "top_k_sampling_v151936"
    assert kernels[0]["params"] == {"vocab_size": 151936}
    assert kernels[0]["fi_api"] == "flashinfer.sampling.top_k_sampling_from_probs"


def test_matched_buckets_without_extractable_params_emit_needs_config() -> None:
    matched = {
        "rmsnorm:rmsnorm": _bucket(
            op_type="rmsnorm",
            variant="rmsnorm",
            fi_api="flashinfer.norm.rmsnorm",
            signatures=[],
        ),
        "sampling:top_k": _bucket(
            op_type="sampling",
            variant="top_k",
            fi_api="flashinfer.sampling.top_k_sampling_from_probs",
            signatures=[],
        ),
        "cascade_merge:merge_state": _bucket(
            op_type="cascade_merge",
            variant="merge_state",
            fi_api="flashinfer.cascade.merge_state",
            signatures=[],
        ),
        "comm:decode_cp_a2a_alltoall": _bucket(
            op_type="comm",
            variant="decode_cp_a2a_alltoall",
            fi_api="flashinfer.comm.dcp_alltoall.decode_cp_a2a_alltoall",
            signatures=[],
        ),
        "quantization:fp4": _bucket(
            op_type="quantization",
            variant="fp4",
            fi_api="flashinfer.quantization.fp4_quantization.fp4_quantize",
            signatures=[],
        ),
        "moe:bf16": _bucket(
            op_type="moe",
            variant="bf16",
            fi_api="flashinfer.fused_moe.trtllm_bf16_moe",
            signatures=[],
        ),
    }

    kernels = build_kernels(matched)
    by_op = {kernel["op_type"]: kernel for kernel in kernels}

    assert set(by_op) >= {
        "rmsnorm",
        "sampling",
        "cascade_merge",
        "comm",
        "quantization",
        "moe",
    }
    for op_type in (
        "rmsnorm",
        "sampling",
        "cascade_merge",
        "comm",
        "quantization",
        "moe",
    ):
        assert "NEEDS_CONFIG" in by_op[op_type]["definition_name"]
        assert by_op[op_type]["note"]
