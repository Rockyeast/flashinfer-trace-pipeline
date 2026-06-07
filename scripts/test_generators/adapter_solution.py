"""Adapter-backed strict reference test generator.

This generator reuses scripts/adapters/*.py baseline solution builders as the
ground-truth implementation under test. It compares definition.reference
against an op-specific FlashInfer/PyTorch solution instead of only checking
that reference run() executes.
"""

from __future__ import annotations

import json
import textwrap

from adapters import eval_validation_policy, generate_baseline_solution

from .common import input_arg_exprs, render_test_file

# Tolerance standards aligned with flashinfer-trace-official tests/references.
_TOLERANCE_BY_OP_TYPE: dict[str, tuple[float, float]] = {
    # (atol, rtol)
    "rmsnorm": (8e-3, 1e-2),
    "gdn": (5e-3, 5e-3),
    "gqa_paged": (1e-2, 5e-2),
    "gqa_ragged": (1e-2, 5e-2),
    "mla_paged": (1e-2, 5e-2),
    "dsa_paged": (1e-2, 5e-2),
    "moe": (1e-1, 2e-1),
}
_DEFAULT_TOLERANCE = (1e-2, 1e-2)


def _tolerance_for(definition: dict) -> tuple[float, float]:
    """Return (atol, rtol) for a definition based on its op_type."""
    return _TOLERANCE_BY_OP_TYPE.get(definition.get("op_type", ""), _DEFAULT_TOLERANCE)


def _solution_main_source(solution: dict) -> str:
    for source in solution.get("sources", []):
        if isinstance(source, dict) and source.get("path") == "main.py":
            content = source.get("content")
            if isinstance(content, str) and content.strip():
                return content
    raise NotImplementedError("adapter baseline solution has no main.py source")


def _extra_imports(solution: dict) -> list[str]:
    spec = solution.get("spec", {}) if isinstance(solution, dict) else {}
    deps = spec.get("dependencies", []) if isinstance(spec, dict) else []
    imports = ["import copy"]
    if isinstance(deps, list) and "flashinfer" in deps:
        imports.append("import flashinfer")
    return imports


def _sampling_validation_body(
    *,
    name: str,
    solution_name: str,
    solution_source: str,
    solution_args: str,
    atol: float,
    rtol: float,
) -> str:
    return textwrap.dedent(f"""\
        _SOLUTION_SOURCE = {json.dumps(solution_source)}
        _SOLUTION_RUN = None


        def _clone_value(value):
            if torch.is_tensor(value):
                return value.detach().clone()
            return copy.deepcopy(value)


        def _load_solution_run():
            global _SOLUTION_RUN
            if _SOLUTION_RUN is None:
                namespace = {{}}
                exec(compile(_SOLUTION_SOURCE, "<solution:{name}>", "exec"), namespace)
                _SOLUTION_RUN = namespace["run"]
            return _SOLUTION_RUN


        def _row_param(inputs, name, row_idx):
            if name not in inputs:
                return None
            value = inputs[name]
            if torch.is_tensor(value):
                return value[row_idx] if value.ndim > 0 else value
            return value


        def _filtered_sampling_probs(row, *, top_k=None, top_p=None):
            row = row.to(torch.float32)
            vocab_size = row.numel()

            if top_k is not None:
                k = int(top_k.item()) if torch.is_tensor(top_k) else int(top_k)
                if 0 < k < vocab_size:
                    keep_idx = torch.argsort(row, descending=True)[:k]
                    filtered = torch.zeros_like(row)
                    filtered[keep_idx] = row[keep_idx]
                    row = filtered / filtered.sum()

            if top_p is not None:
                p = float(top_p.item()) if torch.is_tensor(top_p) else float(top_p)
                if p <= 0.0:
                    filtered = torch.zeros_like(row)
                    filtered[torch.argmax(row)] = 1.0
                    row = filtered
                elif p < 1.0:
                    vals, idx = torch.sort(row, descending=True)
                    cdf = torch.cumsum(vals, dim=0)
                    to_remove = cdf > p
                    if vocab_size > 1:
                        to_remove[1:] = to_remove[:-1].clone()
                        to_remove[0] = False
                    keep_idx = idx[~to_remove]
                    filtered = torch.zeros_like(row)
                    filtered[keep_idx] = row[keep_idx]
                    row = filtered / filtered.sum()

            return row


        def _assert_sampling_output_valid(samples, inputs):
            assert torch.is_tensor(samples)
            probs = inputs["probs"]
            batch_size, vocab_size = probs.shape
            assert tuple(samples.shape) == (batch_size,)
            assert samples.dtype in (torch.int32, torch.int64)

            for row_idx in range(batch_size):
                token = int(samples[row_idx].item())
                assert 0 <= token < vocab_size
                filtered = _filtered_sampling_probs(
                    probs[row_idx],
                    top_k=_row_param(inputs, "top_k", row_idx),
                    top_p=_row_param(inputs, "top_p", row_idx),
                )
                assert filtered[token] > 0, (
                    f"sampled token {{token}} is outside filtered sampling support "
                    f"for row {{row_idx}}"
                )


        def test_correctness(batch_size=4, atol={atol}, rtol={rtol}):
            print(f"\\n{{'='*60}}")
            print(f"Testing {name} sampling support against adapter solution {solution_name}")
            print(f"{{'='*60}}")

            device = "cuda" if torch.cuda.is_available() else "cpu"
            if device == "cpu":
                pytest.skip("CUDA not available")

            original_inputs = generate_random_inputs(batch_size, device)
            solution_run = _load_solution_run()
            torch.manual_seed(1000003)
            solution_inputs = {{
                name: _clone_value(value) for name, value in original_inputs.items()
            }}
            inputs = solution_inputs
            solution_output = solution_run({solution_args})

            _assert_sampling_output_valid(solution_output, solution_inputs)
            print("Adapter solution output is valid for the filtered sampling support")
        """)


def generate(definition: dict) -> str:
    solution = generate_baseline_solution(definition)
    if solution is None:
        raise NotImplementedError(
            f"no adapter baseline solution for op_type={definition.get('op_type')!r}"
        )

    name = definition["name"]
    solution_name = solution.get("name", "adapter_solution")
    solution_source = _solution_main_source(solution)
    ref_args = input_arg_exprs(definition, include_optional=True)
    solution_args = input_arg_exprs(definition, include_optional=True)

    tol_atol, tol_rtol = _tolerance_for(definition)

    if eval_validation_policy(definition) == "sampling_support":
        body = _sampling_validation_body(
            name=name,
            solution_name=solution_name,
            solution_source=solution_source,
            solution_args=solution_args,
            atol=tol_atol,
            rtol=tol_rtol,
        )
        return render_test_file(definition, body, extra_imports=_extra_imports(solution))

    body = textwrap.dedent(f"""\
        _SOLUTION_SOURCE = {json.dumps(solution_source)}
        _SOLUTION_RUN = None


        def _clone_value(value):
            if torch.is_tensor(value):
                return value.detach().clone()
            return copy.deepcopy(value)


        def _load_solution_run():
            global _SOLUTION_RUN
            if _SOLUTION_RUN is None:
                namespace = {{}}
                exec(compile(_SOLUTION_SOURCE, "<solution:{name}>", "exec"), namespace)
                _SOLUTION_RUN = namespace["run"]
            return _SOLUTION_RUN


        def _flatten_output(value, prefix="output"):
            if torch.is_tensor(value):
                return [(prefix, value)]
            if isinstance(value, dict):
                items = []
                for key in sorted(value):
                    items.extend(_flatten_output(value[key], f"{{prefix}}.{{key}}"))
                return items
            if isinstance(value, (tuple, list)):
                items = []
                for idx, item in enumerate(value):
                    items.extend(_flatten_output(item, f"{{prefix}}[{{idx}}]"))
                return items
            return [(prefix, value)]


        def _assert_outputs_close(actual, expected, *, atol, rtol):
            actual_items = _flatten_output(actual)
            expected_items = _flatten_output(expected)
            assert len(actual_items) == len(expected_items)
            for (actual_path, actual_value), (expected_path, expected_value) in zip(
                actual_items, expected_items
            ):
                assert actual_path == expected_path
                if torch.is_tensor(actual_value) or torch.is_tensor(expected_value):
                    assert torch.is_tensor(actual_value)
                    assert torch.is_tensor(expected_value)
                    torch.testing.assert_close(
                        actual_value.to(torch.float32),
                        expected_value.to(torch.float32),
                        atol=atol,
                        rtol=rtol,
                    )
                else:
                    assert actual_value == expected_value


        def test_correctness(batch_size=4, atol={tol_atol}, rtol={tol_rtol}):
            print(f"\\n{{'='*60}}")
            print(f"Testing {name} against adapter solution {solution_name}")
            print(f"{{'='*60}}")

            device = "cuda" if torch.cuda.is_available() else "cpu"
            if device == "cpu":
                pytest.skip("CUDA not available")

            original_inputs = generate_random_inputs(batch_size, device)
            torch.manual_seed(1000003)
            ref_inputs = {{name: _clone_value(value) for name, value in original_inputs.items()}}
            inputs = ref_inputs
            ref_output = run({ref_args})

            solution_run = _load_solution_run()
            torch.manual_seed(1000003)
            solution_inputs = {{
                name: _clone_value(value) for name, value in original_inputs.items()
            }}
            inputs = solution_inputs
            solution_output = solution_run({solution_args})

            _assert_outputs_close(solution_output, ref_output, atol=atol, rtol=rtol)
            print("Adapter solution matches reference")
        """)

    return render_test_file(definition, body, extra_imports=_extra_imports(solution))
