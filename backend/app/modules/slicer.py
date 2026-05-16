"""CAD slicing engine.

Walks the mesh from z_min to z_max in N layers, intersects with planes,
and derives a layer-by-layer profile. Slice features (area, centroid,
component count, occupancy footprint) feed the chair attribute
extractor: seat height, backrest angle, armrests, leg config, etc.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import numpy as np
import trimesh
from shapely.geometry import MultiPolygon, Polygon


@dataclass
class LayerInfo:
    z: float
    area: float
    perimeter: float
    component_count: int
    centroid: Tuple[float, float]
    bbox: Tuple[float, float, float, float]  # minx, miny, maxx, maxy
    occupancy: float  # filled fraction of layer-bounding rectangle

    def to_dict(self) -> Dict[str, Any]:
        return {
            "z": self.z,
            "area": self.area,
            "perimeter": self.perimeter,
            "component_count": self.component_count,
            "centroid": list(self.centroid),
            "bbox": list(self.bbox),
            "occupancy": self.occupancy,
        }


def slice_mesh(mesh: trimesh.Trimesh, num_layers: int = 80) -> List[LayerInfo]:
    """Slice along Z and produce per-layer geometric stats."""
    z_min, z_max = float(mesh.bounds[0, 2]), float(mesh.bounds[1, 2])
    if z_max - z_min < 1e-6:
        return []
    # Skip the very edges to avoid degenerate zero-area slices at z_min / z_max.
    margin = (z_max - z_min) * 0.01
    zs = np.linspace(z_min + margin, z_max - margin, num_layers)

    layers: List[LayerInfo] = []
    for z in zs:
        layers.append(_slice_at(mesh, float(z)))
    return layers


def _slice_at(mesh: trimesh.Trimesh, z: float) -> LayerInfo:
    section = mesh.section(plane_origin=[0, 0, z], plane_normal=[0, 0, 1])
    if section is None:
        return LayerInfo(z=z, area=0.0, perimeter=0.0, component_count=0,
                         centroid=(0.0, 0.0), bbox=(0, 0, 0, 0), occupancy=0.0)
    try:
        planar, _ = section.to_planar()
    except Exception:
        return LayerInfo(z=z, area=0.0, perimeter=0.0, component_count=0,
                         centroid=(0.0, 0.0), bbox=(0, 0, 0, 0), occupancy=0.0)

    polys: List[Polygon] = []
    for poly in planar.polygons_full:
        # poly is already a shapely polygon in trimesh.path
        if poly.is_valid and poly.area > 1e-9:
            polys.append(poly)
    if not polys:
        return LayerInfo(z=z, area=0.0, perimeter=0.0, component_count=0,
                         centroid=(0.0, 0.0), bbox=(0, 0, 0, 0), occupancy=0.0)

    mpoly = MultiPolygon(polys) if len(polys) > 1 else polys[0]
    area = float(mpoly.area)
    perimeter = float(mpoly.length)
    minx, miny, maxx, maxy = mpoly.bounds
    bbox_area = max((maxx - minx) * (maxy - miny), 1e-9)
    occupancy = float(area / bbox_area)
    cx, cy = mpoly.centroid.x, mpoly.centroid.y
    return LayerInfo(
        z=z,
        area=area,
        perimeter=perimeter,
        component_count=len(polys),
        centroid=(float(cx), float(cy)),
        bbox=(float(minx), float(miny), float(maxx), float(maxy)),
        occupancy=occupancy,
    )


def density_map(layers: List[LayerInfo]) -> List[float]:
    """Normalized area profile across Z — a 1D density signal."""
    if not layers:
        return []
    areas = np.array([l.area for l in layers], dtype=np.float64)
    if areas.max() <= 0:
        return [0.0] * len(layers)
    return (areas / areas.max()).tolist()


def chair_attributes(mesh: trimesh.Trimesh, layers: List[LayerInfo]) -> Dict[str, Any]:
    """Derive chair-specific attributes from slicing profile."""
    if not layers:
        return _empty_attrs()
    zs = np.array([l.z for l in layers])
    areas = np.array([l.area for l in layers])
    comps = np.array([l.component_count for l in layers])
    occ = np.array([l.occupancy for l in layers])

    z_total = float(zs.max() - zs.min())

    # 1) Seat plate: the lowest local-maximum of area where the cross-section
    #    is broad (high occupancy). Walk from bottom up looking for the first
    #    layer with area >> mean of layers below.
    seat_idx = _find_seat_index(areas, occ)
    seat_height = float(zs[seat_idx] - zs.min()) if seat_idx >= 0 else 0.0

    # 2) Leg configuration: count connected components in the lower 30% of Z
    #    (below seat plate). Mode of component count = number of legs / posts.
    leg_zone = comps[: max(seat_idx, 1)]
    if len(leg_zone) > 0:
        # Use a robust mode-ish estimate.
        unique, counts = np.unique(leg_zone, return_counts=True)
        leg_count = int(unique[np.argmax(counts)])
    else:
        leg_count = 0
    # If lower zone is dominated by a single broad component, classify as pedestal.
    base_type = _classify_base(leg_count, leg_zone, occ[: max(seat_idx, 1)])

    # 3) Armrests: layers slightly above seat (10–25% of remaining height) that
    #    contain extra side components beyond the back.
    armrest_present = _detect_armrests(areas, comps, seat_idx, len(layers))

    # 4) Backrest angle: above the seat, fit a line through the centroid X (or
    #    Y) of the back-component to estimate tilt.
    backrest_angle = _estimate_backrest_angle(layers, seat_idx)

    # 5) Stability metric: ratio of base footprint area to seat area.
    base_area = float(areas[: max(seat_idx, 1)].max()) if seat_idx > 0 else float(areas[0])
    seat_area = float(areas[seat_idx]) if seat_idx >= 0 else float(areas.mean())
    stability = float(base_area / max(seat_area, 1e-6))

    # 6) Ergonomic score: simple blend that rewards seat presence, backrest, and stability.
    ergonomic = float(np.clip(
        0.4 * (seat_area > 0.5 * seat_area)  # always 0.4 if seat found
        + 0.3 * (backrest_angle is not None)
        + 0.2 * min(stability, 1.0)
        + 0.1 * (1.0 if armrest_present else 0.0),
        0.0, 1.0,
    ))

    return {
        "z_total": z_total,
        "seat_height": seat_height,
        "seat_index": int(seat_idx),
        "seat_area": seat_area,
        "leg_count": leg_count,
        "base_type": base_type,
        "armrests": bool(armrest_present),
        "backrest_angle_deg": backrest_angle,
        "stability": stability,
        "ergonomic_score": ergonomic,
        "slice_layers": int(len(layers)),
    }


def _empty_attrs() -> Dict[str, Any]:
    return {
        "z_total": 0.0, "seat_height": 0.0, "seat_index": -1, "seat_area": 0.0,
        "leg_count": 0, "base_type": "unknown", "armrests": False,
        "backrest_angle_deg": None, "stability": 0.0, "ergonomic_score": 0.0,
        "slice_layers": 0,
    }


def _find_seat_index(areas: np.ndarray, occ: np.ndarray) -> int:
    """Pick the first 'broad' layer with high occupancy from the bottom."""
    if len(areas) == 0:
        return -1
    mean_area = float(areas.mean())
    for i in range(len(areas)):
        if areas[i] > 1.5 * mean_area and occ[i] > 0.5:
            return i
    # Otherwise pick the global max in the lower 60%.
    cutoff = int(len(areas) * 0.6)
    if cutoff <= 0:
        return int(np.argmax(areas))
    return int(np.argmax(areas[:cutoff]))


def _classify_base(leg_count: int, leg_zone: np.ndarray, occ_zone: np.ndarray) -> str:
    if len(leg_zone) == 0:
        return "unknown"
    if leg_count >= 4 and occ_zone.mean() < 0.6:
        if leg_count == 5:
            return "5-wheel"
        return f"{leg_count}-leg"
    if leg_count <= 1 and occ_zone.mean() > 0.5:
        return "pedestal"
    if leg_count in (2, 3):
        return f"{leg_count}-leg"
    return "sled"


def _detect_armrests(areas: np.ndarray, comps: np.ndarray, seat_idx: int, n_layers: int) -> bool:
    if seat_idx < 0 or seat_idx >= n_layers - 1:
        return False
    upper_start = seat_idx + 1
    upper_end = min(seat_idx + int(0.25 * (n_layers - seat_idx)) + 1, n_layers)
    zone_comps = comps[upper_start:upper_end]
    if len(zone_comps) == 0:
        return False
    # Armrest signature: more than 1 component above seat (back + 2 arms => 3).
    return bool(np.any(zone_comps >= 3))


def _estimate_backrest_angle(layers: List["LayerInfo"], seat_idx: int) -> float | None:
    if seat_idx < 0 or seat_idx >= len(layers) - 3:
        return None
    upper = layers[seat_idx + 1 :]
    # Use Y-centroid drift as a proxy for forward/back tilt.
    zs = np.array([l.z for l in upper])
    cys = np.array([l.centroid[1] for l in upper])
    if len(zs) < 2 or np.allclose(zs, zs[0]):
        return None
    slope, _ = np.polyfit(zs, cys, 1)  # dy/dz
    angle = float(np.degrees(np.arctan(slope)))
    return angle


def slice_summary(layers: List[LayerInfo], max_layers: int = 24) -> List[Dict[str, Any]]:
    """A compact subsample of layers for client display / QR encoding."""
    if not layers:
        return []
    if len(layers) <= max_layers:
        sample = layers
    else:
        step = len(layers) / max_layers
        sample = [layers[int(i * step)] for i in range(max_layers)]
    return [l.to_dict() for l in sample]
