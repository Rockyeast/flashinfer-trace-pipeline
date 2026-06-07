import pytest
from pydantic import ValidationError

from artifact_schemas import validate_definition, validate_workload_entry


def _minimal_definition() -> dict:
    return {
        "name": "rmsnorm_h4096",
        "op_type": "rmsnorm",
        "description": "RMSNorm test definition.",
        "tags": ["unit"],
        "axes": {
            "batch_size": {"type": "var", "description": "Rows."},
            "hidden_size": {"type": "const", "value": 4096},
        },
        "inputs": {
            "input": {"shape": ["batch_size", "hidden_size"], "dtype": "bfloat16"},
            "weight": {"shape": ["hidden_size"], "dtype": "bfloat16"},
        },
        "outputs": {
            "output": {"shape": ["batch_size", "hidden_size"], "dtype": "bfloat16"}
        },
        "reference": "def run(input, weight):\n    return input\n",
    }


def test_validate_definition_accepts_minimal_valid_definition() -> None:
    definition = _minimal_definition()

    assert validate_definition(definition) is definition


def test_validate_definition_rejects_unknown_fields() -> None:
    definition = _minimal_definition()
    definition["unexpected"] = True

    with pytest.raises(ValidationError):
        validate_definition(definition)


def test_validate_definition_rejects_invalid_axis_payloads() -> None:
    definition = _minimal_definition()
    definition["axes"]["hidden_size"] = {"type": "const"}

    with pytest.raises(ValidationError):
        validate_definition(definition)

    definition = _minimal_definition()
    definition["axes"]["batch_size"] = {"type": "var", "value": 4}

    with pytest.raises(ValidationError):
        validate_definition(definition)


def test_validate_workload_entry_rejects_random_payload_fields() -> None:
    entry = {
        "definition": "rmsnorm_h4096",
        "workload": {
            "uuid": "unit",
            "axes": {"batch_size": 1, "hidden_size": 4096},
            "inputs": {"input": {"type": "random", "value": 1}},
        },
        "solution": None,
        "evaluation": None,
    }

    with pytest.raises(ValidationError):
        validate_workload_entry(entry)
