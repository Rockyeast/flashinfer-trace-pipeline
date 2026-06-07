"""Batch sizing and diagnostics helpers for streaming workload collect."""

from __future__ import annotations


def parse_batch_sizes(value: str | list[int] | tuple[int, ...] | None) -> list[int]:
    """Parse comma/space separated batch sizes, preserving order and removing duplicates."""
    if value is None:
        raw: list[str | int] = []
    elif isinstance(value, str):
        raw = [part for part in value.replace(",", " ").split() if part]
    else:
        raw = list(value)

    out: list[int] = []
    seen: set[int] = set()
    for item in raw:
        batch_size = int(item)
        if batch_size <= 0:
            raise ValueError(f"batch size must be positive, got {batch_size}")
        if batch_size not in seen:
            out.append(batch_size)
            seen.add(batch_size)
    return out


def rounds_for_batch_size(
    rounds: list[tuple[int, int, int]],
    batch_size: int,
) -> list[tuple[int, int, int]]:
    """Return existing rounds for a batch size, or a small fallback round."""
    matched = [round_spec for round_spec in rounds if int(round_spec[0]) == batch_size]
    if matched:
        return matched
    prompt_tokens = 300 if batch_size <= 4 else 50
    return [(batch_size, prompt_tokens, 8)]


def merge_batch_diagnostics(
    aggregate: dict,
    batch_size: int,
    batch_diagnostics: dict,
) -> None:
    """Merge one sanitize pass's diagnostics into a streaming summary."""
    batch_key = str(batch_size)
    aggregate.setdefault("_batches", {})[batch_key] = batch_diagnostics
    for def_name, diag in batch_diagnostics.items():
        if def_name.startswith("_"):
            continue
        target = aggregate.setdefault(
            def_name,
            {
                "candidates": 0,
                "selected": 0,
                "discarded": 0,
                "discard_reasons": {},
                "batches": {},
            },
        )
        for key in ("candidates", "selected", "discarded"):
            target[key] += int(diag.get(key, 0))
        reasons = target.setdefault("discard_reasons", {})
        for reason, count in diag.get("discard_reasons", {}).items():
            reasons[reason] = reasons.get(reason, 0) + int(count)
        target["batches"][batch_key] = {
            "candidates": diag.get("candidates", 0),
            "selected": diag.get("selected", 0),
            "discarded": diag.get("discarded", 0),
            "discard_reasons": diag.get("discard_reasons", {}),
        }
