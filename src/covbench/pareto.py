"""Multi-objective Pareto frontier extraction within matched comparison groups."""
from __future__ import annotations

from collections.abc import Mapping
import json
import math
import numbers


def _path(record, path):
    """Resolve a dot-separated field path inside a nested mapping."""
    value = record
    for key in path.split("."):
        if not isinstance(value, Mapping) or key not in value:
            raise KeyError(path)
        value = value[key]
    return value


def _names(values, name):
    """Validate that ``values`` is a non-empty sequence of distinct dot-separated paths."""
    values = tuple(values)
    if not values or any(not isinstance(value, str) or not value or
                         any(not part for part in value.split(".")) for value in values):
        raise ValueError(f"{name} must contain nonempty field paths")
    if len(set(values)) != len(values):
        raise ValueError(f"{name} must contain distinct paths")
    return values


def frontiers(records, objectives, comparison_keys):
    """Partition records by ``comparison_keys`` and compute nondominated Pareto sets.

    All objectives are minimized. A record is eligible if ``status`` is ``"ok"``
    or ``"partial"``, every path in ``comparison_keys`` resolves to a non-null
    value, and every path in ``objectives`` resolves to a finite real number.
    Ineligible records are returned in ``excluded`` with their original index
    and reason code. Within each comparison group, exact ties are retained in
    ``frontier``.
    """
    objectives = _names(objectives, "objectives")
    comparison_keys = _names(comparison_keys, "comparison_keys")
    groups, excluded = {}, []
    for index, record in enumerate(records):
        if not isinstance(record, Mapping):
            excluded.append({"index": index, "record": record, "reason": "invalid_record"})
            continue
        if record.get("status") not in {"ok", "partial"}:
            excluded.append({"index": index, "record": record, "reason": "invalid_status",
                             "status": record.get("status")})
            continue
        try:
            comparison = {key: _path(record, key) for key in comparison_keys}
            if any(value is None for value in comparison.values()):
                raise ValueError("comparison values cannot be null")
            group_id = json.dumps(comparison, sort_keys=True, separators=(",", ":"), allow_nan=False)
        except (KeyError, ValueError, TypeError) as exc:
            excluded.append({"index": index, "record": record, "reason": "invalid_comparison_metadata", "detail": str(exc)})
            continue
        try:
            values = tuple(_path(record, objective) for objective in objectives)
            if any(isinstance(value, bool) or not isinstance(value, numbers.Real)
                   or not math.isfinite(float(value)) for value in values):
                raise ValueError("selected objectives must be finite real numbers")
            values = tuple(float(value) for value in values)
        except (KeyError, ValueError, TypeError, OverflowError) as exc:
            excluded.append({"index": index, "record": record, "reason": "invalid_objective", "detail": str(exc)})
            continue
        group = groups.setdefault(group_id, {"comparison": comparison, "points": []})
        group["points"].append((index, record, values))
    output = []
    for group in groups.values():
        points = group.pop("points")
        frontier, dominated = [], []
        for index, record, values in points:
            is_dominated = any(
                all(other <= value for other, value in zip(other_values, values))
                and any(other < value for other, value in zip(other_values, values))
                for _, _, other_values in points
            )
            destination = dominated if is_dominated else frontier
            destination.append({"index": index, "record": record})
        output.append({**group, "frontier": frontier, "dominated": dominated})
    return {"objectives": list(objectives), "comparison_keys": list(comparison_keys),
            "groups": output, "excluded": excluded}
