from __future__ import annotations

import math
from typing import Any

from traj_analyzer.features.table import FeatureRow
from traj_analyzer.operators.base import FeatureSpec
from traj_analyzer.project import ConfigError


def agreement(
    reference: dict[str, dict[str, Any]], specs: dict[str, FeatureSpec], rows: dict[tuple[str, str], FeatureRow]
) -> dict[str, Any]:
    """Compare reference answers with the feature table, per feature.

    `reference` maps a trajectory key to feature values, each given bare or as {"value": ...}. Sets are also
    compared by Jaccard similarity; numbers are equal within 1e-6.
    """
    report: dict[str, dict[str, Any]] = {}
    jaccard: dict[str, list[float]] = {}
    for key, answers in reference.items():
        for feature, answer in answers.items():
            if feature not in specs:
                raise ConfigError(f"{key}: {feature} is not an enabled feature")
            expected = answer["value"] if isinstance(answer, dict) and "value" in answer else answer
            entry = report.setdefault(feature, {"compared": 0, "equal": 0, "missing": [], "not_ok": [],
                                                "mismatches": []})
            row = rows.get((key, feature))
            if row is None:
                entry["missing"].append(key)
                continue
            if row.status != "ok":
                entry["not_ok"].append({"key": key, "status": row.status, "detail": row.detail})
                continue
            entry["compared"] += 1
            same = _same(specs[feature], expected, row.value)
            entry["equal"] += same
            if specs[feature].type == "set":
                jaccard.setdefault(feature, []).append(_jaccard(expected, row.value))
            if not same:
                entry["mismatches"].append({"key": key, "reference": expected, "extracted": row.value,
                                            "evidence": row.evidence})
    for feature, entry in report.items():
        entry["agreement"] = round(entry["equal"] / entry["compared"], 3) if entry["compared"] else None
        if feature in jaccard:
            entry["mean_jaccard"] = round(sum(jaccard[feature]) / len(jaccard[feature]), 3)
    return report


def _same(spec: FeatureSpec, expected: Any, extracted: Any) -> bool:
    if expected is None or extracted is None:
        return expected is None and extracted is None
    match spec.type:
        case "set":
            return set(expected) == set(extracted)
        case "map":
            return set(expected) == set(extracted) and all(
                math.isclose(float(expected[k]), float(extracted[k]), abs_tol=1e-6) for k in expected)
        case "vector":
            return len(expected) == len(extracted) and all(
                math.isclose(float(x), float(y), abs_tol=1e-6) for x, y in zip(expected, extracted, strict=True))
        case "scalar":
            return math.isclose(float(expected), float(extracted), abs_tol=1e-6)
        case _:
            return expected == extracted


def _jaccard(expected: Any, extracted: Any) -> float:
    left, right = set(expected or []), set(extracted or [])
    return 1.0 if not left and not right else len(left & right) / len(left | right)
