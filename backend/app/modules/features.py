"""Feature intelligence: convert geometric + slicing info → semantic chair.

Heuristic rule-based classifier. Designed to be deterministic so the QR
encoding is stable across runs. A learned classifier could swap in here
later (the input is already a feature vector).
"""
from __future__ import annotations

from typing import Any, Dict, List

import numpy as np


CHAIR_TYPES = [
    "office", "dining", "lounge", "hospital", "wheelchair_compatible",
    "gaming", "stool", "outdoor",
]


def build_feature_vector(geometry: Dict[str, Any], attrs: Dict[str, Any],
                         density: List[float]) -> Dict[str, Any]:
    """Combine geometry descriptors + slicing attrs + density signature."""
    dims = geometry.get("dimensions", {})
    return {
        "dim_x": dims.get("x", 0.0),
        "dim_y": dims.get("y", 0.0),
        "dim_z": dims.get("z", 0.0),
        "surface_area": geometry.get("surface_area", 0.0),
        "volume": geometry.get("volume", 0.0),
        "triangle_count": geometry.get("triangle_count", 0),
        "curvature_mean": geometry.get("curvature", {}).get("mean", 0.0),
        "curvature_std": geometry.get("curvature", {}).get("std", 0.0),
        "seat_height": attrs.get("seat_height", 0.0),
        "leg_count": attrs.get("leg_count", 0),
        "base_type": attrs.get("base_type", "unknown"),
        "armrests": attrs.get("armrests", False),
        "backrest_angle_deg": attrs.get("backrest_angle_deg"),
        "stability": attrs.get("stability", 0.0),
        "ergonomic_score": attrs.get("ergonomic_score", 0.0),
        "density_signature": _compress_signature(density, 16),
    }


def classify_chair(features: Dict[str, Any]) -> Dict[str, Any]:
    """Heuristic chair-type classifier with per-class scores."""
    scores = {t: 0.0 for t in CHAIR_TYPES}

    base = features.get("base_type", "")
    seat_h = float(features.get("seat_height") or 0.0)
    arm = bool(features.get("armrests"))
    leg = int(features.get("leg_count") or 0)
    stab = float(features.get("stability") or 0.0)
    ergo = float(features.get("ergonomic_score") or 0.0)
    backrest = features.get("backrest_angle_deg")

    # Office: pedestal / 5-wheel + armrests + moderate seat.
    if base in ("5-wheel", "pedestal"):
        scores["office"] += 0.6
    if arm:
        scores["office"] += 0.2
        scores["gaming"] += 0.2
    if 40 <= seat_h <= 60:
        scores["office"] += 0.2

    # Dining: 4 legs, no armrests, upright back.
    if leg == 4 and not arm:
        scores["dining"] += 0.6
    if backrest is not None and abs(backrest) < 8:
        scores["dining"] += 0.2
    if 42 <= seat_h <= 50:
        scores["dining"] += 0.2

    # Lounge: low seat, reclined back, wide.
    if seat_h < 42:
        scores["lounge"] += 0.4
    if backrest is not None and abs(backrest) > 15:
        scores["lounge"] += 0.3
    if features.get("dim_x", 0) > 60 and features.get("dim_y", 0) > 60:
        scores["lounge"] += 0.3

    # Hospital: high stability, armrests with grab bars (proxy: high stability + armrests).
    if stab > 1.2 and arm:
        scores["hospital"] += 0.5
    if 45 <= seat_h <= 55 and arm:
        scores["hospital"] += 0.3

    # Wheelchair-compatible: wheels (5-wheel-ish but with side support), wide footprint.
    if base in ("5-wheel",) and features.get("dim_x", 0) > 55:
        scores["wheelchair_compatible"] += 0.4
    if arm and stab > 1.4:
        scores["wheelchair_compatible"] += 0.3

    # Gaming: very high backrest angle range + armrests + 5-wheel.
    if base == "5-wheel" and arm and ergo > 0.7:
        scores["gaming"] += 0.5

    # Stool: no backrest detected (None), no armrests, smallish.
    if backrest is None and not arm:
        scores["stool"] += 0.6
    if seat_h > 60 and not arm:
        scores["stool"] += 0.3

    # Outdoor: 4-leg + low ergonomic + simple.
    if leg == 4 and ergo < 0.6:
        scores["outdoor"] += 0.3

    # Normalize.
    total = sum(scores.values()) or 1.0
    norm = {k: round(v / total, 4) for k, v in scores.items()}
    label = max(norm, key=norm.get)
    confidence = norm[label]
    return {"label": label, "confidence": confidence, "scores": norm}


def _compress_signature(values: List[float], buckets: int) -> List[float]:
    """Resample a 1-D signal down to `buckets` floats with 2-decimal precision."""
    if not values:
        return [0.0] * buckets
    arr = np.array(values, dtype=np.float64)
    if len(arr) == buckets:
        return [round(float(v), 2) for v in arr]
    idx = np.linspace(0, len(arr) - 1, buckets)
    sampled = np.interp(idx, np.arange(len(arr)), arr)
    return [round(float(v), 2) for v in sampled]


def assemble_cad_intelligence(source_name: str, geometry: Dict[str, Any],
                              attrs: Dict[str, Any], features: Dict[str, Any],
                              classification: Dict[str, Any]) -> Dict[str, Any]:
    """The canonical JSON object embedded in QR + returned to clients."""
    return {
        "version": 1,
        "source": source_name,
        "geometry": {
            "dimensions": geometry.get("dimensions", {}),
            "surface_area": round(float(geometry.get("surface_area", 0.0)), 3),
            "volume": round(float(geometry.get("volume", 0.0)), 3),
            "triangle_count": int(geometry.get("triangle_count", 0)),
        },
        "slicing": {
            "seat_height": round(float(attrs.get("seat_height", 0.0)), 3),
            "leg_count": attrs.get("leg_count", 0),
            "base_type": attrs.get("base_type", "unknown"),
            "armrests": attrs.get("armrests", False),
            "backrest_angle_deg": (round(float(attrs["backrest_angle_deg"]), 2)
                                   if attrs.get("backrest_angle_deg") is not None else None),
            "stability": round(float(attrs.get("stability", 0.0)), 3),
            "ergonomic_score": round(float(attrs.get("ergonomic_score", 0.0)), 3),
            "slice_layers": int(attrs.get("slice_layers", 0)),
        },
        "features": {
            "density_signature": features.get("density_signature", []),
            "curvature_mean": round(float(features.get("curvature_mean", 0.0)), 4),
            "curvature_std": round(float(features.get("curvature_std", 0.0)), 4),
        },
        "classification": classification,
    }
