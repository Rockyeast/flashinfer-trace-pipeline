"""Common helpers for generated reference tests."""

from __future__ import annotations

import re
import textwrap


def extract_fi_api(definition: dict) -> str | None:
    """Return a definition's explicit FlashInfer fi_api tag."""
    for tag in definition.get("tags", []):
        if tag.startswith("fi_api:"):
            return tag.split(":", 1)[1]
    return None


def imports_for(definition: dict, extra: list[str] | None = None) -> list[str]:
    """Build import lines needed by a generated test."""
    imports = ["import pytest", "import torch"]
    if "math." in definition.get("reference", ""):
        imports.append("import math")
    fi_api = extract_fi_api(definition)
    if fi_api and fi_api.startswith("flashinfer."):
        imports.insert(0, "import flashinfer")
    for line in extra or []:
        if line not in imports:
            imports.insert(0, line)
    return imports


def _build_random_input_line(
    name: str,
    inp: dict,
    axes: dict,
    var_param: str,
) -> str:
    """Build code that creates one random input tensor."""
    shape = inp.get("shape")
    dtype_str = inp.get("dtype", "bfloat16")
    if dtype_str == "unknown":
        if any(part in name for part in ("indptr", "indices", "cu_seqlens", "cu_seq")):
            dtype_str = "int32"
        else:
            dtype_str = "float32"
    dtype_map = {
        "bfloat16": "torch.bfloat16",
        "float16": "torch.float16",
        "float32": "torch.float32",
        "int32": "torch.int32",
        "int64": "torch.int64",
        "float8_e4m3fn": "torch.float8_e4m3fn",
    }
    torch_dtype = dtype_map.get(dtype_str, f"torch.{dtype_str}")
    is_fp8 = "float8" in dtype_str

    if shape is None:
        if dtype_str == "float32":
            return f"    {name} = 1.0 / (128 ** 0.5)  # default scalar"
        return f"    {name} = torch.tensor(0.0, dtype={torch_dtype}, device=device)"

    shape_parts: list[str] = []
    primary_var = next((k for k, v in axes.items() if v.get("type") == "var"), var_param)
    for elem in shape:
        if isinstance(elem, int):
            shape_parts.append(str(elem))
        elif isinstance(elem, str):
            axis = axes.get(elem, {})
            if axis.get("type") == "const":
                shape_parts.append(str(axis["value"]))
            elif elem == primary_var:
                shape_parts.append(var_param)
            else:
                shape_parts.append(elem)
        else:
            shape_parts.append(str(elem))
    shape_str = ", ".join(shape_parts)

    if name in ("top_k", "top_ks"):
        # Sampling APIs expect a positive top-k. A generic randint(0, 10)
        # can produce k=0, whose semantics differ across implementations and
        # makes validity tests flaky.
        return f"    {name} = torch.randint(1, 64, ({shape_str},), dtype={torch_dtype}, device=device)"

    if dtype_str in ("int32", "int64"):
        if "indptr" in name or "cu_seqlens" in name:
            total_var: str | None = inp.get("_total_var")
            if total_var is None:
                if "qo" in name:
                    total_var = "total_q"
                elif "kv" in name:
                    total_var = "num_kv_indices"
                elif "cu_seqlens" in name or "cu_seq" in name:
                    total_var = "total_seq_len"
                else:
                    total_var = "40"
            return (
                f"    {name} = torch.zeros({shape_str}, dtype={torch_dtype}, device=device)\n"
                f"    if {shape_str} > 1:\n"
                f"        {name}[1:] = torch.cumsum(\n"
                f"            torch.full(({shape_str} - 1,), max(1, {total_var} // max(1, {shape_str} - 1)), dtype=torch.int32, device=device),\n"
                f"            dim=0,\n"
                f"        ).to({torch_dtype})"
            )
        return f"    {name} = torch.randint(0, 10, ({shape_str},), dtype={torch_dtype}, device=device)"

    if name == "probs":
        return (
            f"    {name} = torch.rand({shape_str}, dtype={torch_dtype}, device=device)\n"
            f"    {name} = {name} / {name}.sum(dim=-1, keepdim=True)"
        )
    if name in ("top_p", "top_ps", "min_p", "min_ps"):
        return f"    {name} = torch.rand({shape_str}, dtype={torch_dtype}, device=device)"
    if name in ("temperatures",):
        return f"    {name} = torch.rand({shape_str}, dtype={torch_dtype}, device=device) + 0.5"
    if is_fp8:
        return (
            f"    {name} = torch.randn({shape_str}, dtype=torch.float32, device=device)"
            f".to({torch_dtype})"
        )
    return f"    {name} = torch.randn({shape_str}, dtype={torch_dtype}, device=device)"


def build_generate_inputs(definition: dict) -> str:
    """Build generate_random_inputs() for a definition."""
    axes = definition.get("axes", {})
    inputs = definition.get("inputs", {})
    constraints = definition.get("constraints", [])

    indptr_total_map: dict[str, str] = {}
    for constraint in constraints:
        match = re.match(r"^\s*(\w+)\s*==\s*(\w+)\[-1\]\.item\(\)\s*$", constraint)
        if match:
            indptr_total_map[match.group(2)] = match.group(1)
        reverse = re.match(r"^\s*(\w+)\[-1\]\.item\(\)\s*==\s*(\w+)\s*$", constraint)
        if reverse:
            indptr_total_map[reverse.group(1)] = reverse.group(2)

    var_axes = [k for k, v in axes.items() if v.get("type") == "var"]
    var_param = var_axes[0] if var_axes else "batch_size"
    lines = [f'def generate_random_inputs({var_param}, device="cuda"):']

    for ax_name, ax_spec in axes.items():
        if ax_spec.get("type") == "const":
            lines.append(f"    {ax_name} = {ax_spec['value']}")

    for ax_name in var_axes:
        if ax_name == var_param:
            continue
        if "indptr" in ax_name or "cu_seqlens" in ax_name:
            default_expr = f"{var_param} + 1"
        elif "indices" in ax_name or "pages" in ax_name:
            default_expr = f"{var_param} * 10"
        elif ax_name in ("total_q", "total_kv", "total_seq_len"):
            default_expr = f"({var_param} - 1) * 4"
        elif ax_name == "seq_len":
            default_expr = "4"
        elif ax_name == "pool_size":
            default_expr = f"{var_param} * 2"
        else:
            default_expr = f"{var_param} * 4"
        lines.append(f"    {ax_name} = {default_expr}")

    for inp_name, inp_spec in inputs.items():
        augmented = dict(inp_spec)
        if inp_name in indptr_total_map:
            augmented["_total_var"] = indptr_total_map[inp_name]
        lines.append(_build_random_input_line(inp_name, augmented, axes, var_param))

    input_names = list(inputs)
    dict_items = ", ".join(f'"{name}": {name}' for name in input_names)
    lines.append(f"    return {{{dict_items}}}")
    return "\n".join(lines)


def build_reference(definition: dict) -> str:
    """Return definition reference code."""
    reference = definition.get("reference", "")
    if reference and not reference.startswith("#"):
        reference = reference.rstrip()
        if re.search(r"^\s*def\s+run\s*\(", reference, flags=re.MULTILINE):
            return reference
        match = re.search(
            r"^\s*def\s+([A-Za-z_][A-Za-z0-9_]*)\s*\(",
            reference,
            flags=re.MULTILINE,
        )
        if match:
            return f"{reference}\n\nrun = {match.group(1)}"
        return reference
    return textwrap.dedent("""\
        @torch.no_grad()
        def run(*args, **kwargs):
            raise NotImplementedError("Reference not implemented")
        """).rstrip()


def input_arg_exprs(
    definition: dict,
    *,
    clone: bool = False,
    include_optional: bool = False,
) -> str:
    """Return run() argument expression list for non-optional inputs."""
    names = [
        n
        for n, spec in definition.get("inputs", {}).items()
        if include_optional or not spec.get("optional")
    ]
    if clone:
        return ", ".join(f'inputs["{name}"].clone()' for name in names)
    return ", ".join(f'inputs["{name}"]' for name in names)


def build_main(definition: dict) -> str:
    """Build a simple standalone main()."""
    name = definition["name"]
    var_axes = [k for k, v in definition.get("axes", {}).items() if v.get("type") == "var"]
    var_param = var_axes[0] if var_axes else "batch_size"
    if var_param == "len_indptr":
        configs = "[2, 4, 8, 16, 32]"
    else:
        configs = "[1, 4, 8, 16, 32]"

    return textwrap.dedent(f"""\
        def main():
            print("Testing {name} Reference Implementation")
            test_configs = {configs}
            passed = 0
            total = len(test_configs)
            for val in test_configs:
                try:
                    test_correctness(val)
                    passed += 1
                except Exception as exc:
                    print(f"Test failed with exception: {{exc}}")
                    import traceback
                    traceback.print_exc()
            print(f"\\n{{'='*60}}")
            print(f"Summary: {{passed}}/{{total}} tests passed")
            print(f"{{'='*60}}")
            if passed != total:
                raise SystemExit(1)


        if __name__ == "__main__":
            main()
        """)


def render_test_file(
    definition: dict,
    test_body: str,
    *,
    extra_imports: list[str] | None = None,
) -> str:
    """Assemble a full pytest file."""
    parts = [
        "\n".join(imports_for(definition, extra_imports)),
        "",
        "",
        build_reference(definition),
        "",
        "",
        build_generate_inputs(definition),
        "",
        "",
        test_body.rstrip(),
        "",
        "",
        build_main(definition).rstrip(),
        "",
    ]
    return "\n".join(parts)
