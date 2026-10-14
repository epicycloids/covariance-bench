"""Pareto frontiers within explicitly matching benchmark regimes."""
from __future__ import annotations

from collections.abc import Mapping
import json
import math
import numbers


def _path(record, path):
    value = record
    for key in path.split("."):
        if not isinstance(value, Mapping) or key not in value:
            raise KeyError(path)
        value = value[key]
    return value


def _names(values, name):
    values = tuple(values)
    if not values or any(not isinstance(value, str) or not value or
                         any(not part for part in value.split(".")) for value in values):
        raise ValueError(f"{name} must contain nonempty field paths")
    if len(set(values)) != len(values):
        raise ValueError(f"{name} must contain distinct paths")
    return values


def frontiers(records, objectives, comparison_keys):
    """Return nondominated records and explicit failures for each regime.

    Every objective is minimized. A record must have ``status='ok'`` or
    ``'partial'``, finite selected objectives, and every comparison field.
    ``partial`` can therefore compete on available losses without pretending an
    undefined SPD score exists. Timeout/invalid/missing records remain in
    ``excluded``. Exact ties are retained. Field paths can select nested values,
    for example ``metrics.covariance_nmse`` and ``hardware.id``.

    Comparison keys are required; this function does not invent a pooled
    cross-scenario ranking or combine objectives into a scalar score.
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
