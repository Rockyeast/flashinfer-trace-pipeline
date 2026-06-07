import pytest

from collect_workloads.batching import (
    merge_batch_diagnostics,
    parse_batch_sizes,
    rounds_for_batch_size,
)


def test_parse_batch_sizes_accepts_strings_and_deduplicates() -> None:
    assert parse_batch_sizes("1, 2 4,2") == [1, 2, 4]
    assert parse_batch_sizes([8, 4, 8]) == [8, 4]
    assert parse_batch_sizes(None) == []


def test_parse_batch_sizes_rejects_non_positive_values() -> None:
    with pytest.raises(ValueError, match="positive"):
        parse_batch_sizes("1 0")


def test_rounds_for_batch_size_uses_existing_rounds_or_fallback() -> None:
    rounds = [(1, 300, 8), (4, 50, 16), (4, 200, 4)]

    assert rounds_for_batch_size(rounds, 4) == [(4, 50, 16), (4, 200, 4)]
    assert rounds_for_batch_size(rounds, 2) == [(2, 300, 8)]
    assert rounds_for_batch_size(rounds, 8) == [(8, 50, 8)]


def test_merge_batch_diagnostics_accumulates_counts_and_reasons() -> None:
    aggregate: dict = {}

    merge_batch_diagnostics(
        aggregate,
        4,
        {
            "rmsnorm_h4096": {
                "candidates": 3,
                "selected": 2,
                "discarded": 1,
                "discard_reasons": {"duplicate_axes": 1},
            },
            "_capture_summary": {"ignored": True},
        },
    )
    merge_batch_diagnostics(
        aggregate,
        8,
        {
            "rmsnorm_h4096": {
                "candidates": 2,
                "selected": 1,
                "discarded": 1,
                "discard_reasons": {"duplicate_axes": 1},
            },
        },
    )

    assert aggregate["_batches"]["4"]["_capture_summary"] == {"ignored": True}
    assert aggregate["rmsnorm_h4096"]["candidates"] == 5
    assert aggregate["rmsnorm_h4096"]["selected"] == 3
    assert aggregate["rmsnorm_h4096"]["discarded"] == 2
    assert aggregate["rmsnorm_h4096"]["discard_reasons"] == {"duplicate_axes": 2}
    assert aggregate["rmsnorm_h4096"]["batches"]["8"]["selected"] == 1
