"""Synthetic CAD generation.

Procedurally constructs a chair STL from:
  - the original chair's intelligence (size, seat height, base type)
  - the transformation spec produced by the NL engine

We build the chair as a union of primitive boxes/cylinders, then export
ASCII STL. The goal is a *plausible* synthetic counterpart that visibly
reflects the requested changes — not a CAD-accurate reproduction.
"""
from __future__ import annotations

import io
from typing import Any, Dict, Optional, Tuple

import numpy as np
import trimesh


# All dims internally in cm.
DEFAULT_SEAT_W = 45.0  # x
DEFAULT_SEAT_D = 45.0  # y
DEFAULT_SEAT_T = 5.0   # seat plate thickness
DEFAULT_SEAT_H = 45.0  # top-of-seat height from floor
DEFAULT_BACK_H = 50.0  # backrest height above seat
DEFAULT_LEG_T = 4.0    # leg thickness


def generate_chair(intelligence: Dict[str, Any], spec: Dict[str, Any]) -> trimesh.Trimesh:
    """Build the synthetic chair as a single Trimesh."""
    dims = intelligence.get("geometry", {}).get("dimensions", {}) or {}
    slicing = intelligence.get("slicing", {}) or {}

    seat_w = _safe(dims.get("x"), DEFAULT_SEAT_W) * spec["scale"]["x"]
    seat_d = _safe(dims.get("y"), DEFAULT_SEAT_D) * spec["scale"]["y"]
    seat_h = _safe(slicing.get("seat_height"), DEFAULT_SEAT_H)
    if seat_h < 10:
        seat_h = DEFAULT_SEAT_H

    # Apply seat tweaks.
    seat_w *= 1.0 + float(spec["seat"].get("widen", 0.0))
    seat_h += float(spec["seat"].get("height_delta_cm", 0.0))
    seat_h *= spec["scale"]["z"]

    # Pick base.
    base_type = spec["base"].get("type") or "keep"
    if base_type == "keep":
        base_type = slicing.get("base_type") or "4-leg"
    add_wheels = bool(spec["base"].get("add_wheels"))

    parts: list[trimesh.Trimesh] = []

    # 1) Seat plate.
    seat = _box(seat_w, seat_d, DEFAULT_SEAT_T)
    seat.apply_translation([0, 0, seat_h - DEFAULT_SEAT_T / 2])
    parts.append(seat)

    if spec["seat"].get("cushion"):
        cushion = _box(seat_w * 0.95, seat_d * 0.95, 2.0)
        cushion.apply_translation([0, 0, seat_h + 1.0])
        parts.append(cushion)

    # 2) Base.
    parts.extend(_build_base(base_type, seat_w, seat_d, seat_h, add_wheels))

    # 3) Backrest.
    back_h = DEFAULT_BACK_H + float(spec["backrest"].get("raise_cm", 0.0))
    recline = float(spec["backrest"].get("recline_deg", 0.0))
    parts.append(_build_backrest(seat_w, seat_d, seat_h, back_h, recline))
    if spec["backrest"].get("headrest"):
        head = _box(seat_w * 0.6, 4.0, 12.0)
        head.apply_translation([0, -seat_d / 2 + 2.0, seat_h + back_h + 6.0])
        parts.append(head)

    # 4) Armrests.
    if spec["armrests"].get("add") and not spec["armrests"].get("remove"):
        padded = bool(spec["armrests"].get("padded"))
        parts.extend(_build_armrests(seat_w, seat_d, seat_h, padded))

    # 5) Safety.
    if spec["safety"].get("grab_handles"):
        parts.extend(_build_grab_handles(seat_w, seat_d, seat_h))
    if spec["safety"].get("side_rails"):
        parts.extend(_build_side_rails(seat_w, seat_d, seat_h))
    if spec["safety"].get("anti_slip"):
        # Visualize as small floor pads.
        for sx in (-1, 1):
            for sy in (-1, 1):
                pad = _cyl(2.0, 0.3)
                pad.apply_translation([sx * (seat_w / 2 - 4), sy * (seat_d / 2 - 4), 0.15])
                parts.append(pad)

    mesh = trimesh.util.concatenate(parts)
    mesh.merge_vertices()
    mesh.remove_duplicate_faces() if hasattr(mesh, "remove_duplicate_faces") else None
    mesh.remove_unreferenced_vertices()
    return mesh


# --------------------------- Builders -------------------------------

def _build_base(base_type: str, seat_w: float, seat_d: float, seat_h: float,
                add_wheels: bool) -> list:
    parts = []
    floor_to_seat = seat_h - DEFAULT_SEAT_T
    if base_type == "5-wheel" or (base_type == "pedestal" and add_wheels):
        # Central column + 5 radial spokes + caster cylinders.
        col = _cyl(3.0, floor_to_seat * 0.7)
        col.apply_translation([0, 0, floor_to_seat * 0.7 / 2 + 3.0])
        parts.append(col)
        spoke_len = max(seat_w, seat_d) * 0.55
        for k in range(5):
            theta = (2 * np.pi * k) / 5
            spoke = _box(spoke_len, 3.0, 2.0)
            spoke.apply_translation([spoke_len / 2 - 1.0, 0, 1.0])
            T = trimesh.transformations.rotation_matrix(theta, [0, 0, 1])
            spoke.apply_transform(T)
            parts.append(spoke)
            # Caster.
            cx = np.cos(theta) * spoke_len
            cy = np.sin(theta) * spoke_len
            caster = _cyl(2.0, 2.0)
            caster.apply_translation([cx, cy, 1.0])
            parts.append(caster)
        return parts

    if base_type == "pedestal":
        col = _cyl(4.0, floor_to_seat * 0.9)
        col.apply_translation([0, 0, floor_to_seat * 0.9 / 2])
        parts.append(col)
        disk = _cyl(seat_w * 0.45, 1.5)
        disk.apply_translation([0, 0, 0.75])
        parts.append(disk)
        return parts

    if base_type == "sled":
        for sy in (-1, 1):
            rail = _box(seat_w * 0.95, 3.0, 2.0)
            rail.apply_translation([0, sy * (seat_d / 2 - 3), 1.0])
            parts.append(rail)
            riser = _box(3.0, 3.0, floor_to_seat)
            riser.apply_translation([seat_w / 2 - 3, sy * (seat_d / 2 - 3), floor_to_seat / 2])
            parts.append(riser)
            riser2 = _box(3.0, 3.0, floor_to_seat)
            riser2.apply_translation([-seat_w / 2 + 3, sy * (seat_d / 2 - 3), floor_to_seat / 2])
            parts.append(riser2)
        return parts

    # Default: N-leg (4-leg, 3-leg, 2-leg).
    n_legs = 4
    if base_type.endswith("-leg"):
        try:
            n_legs = int(base_type.split("-")[0])
        except Exception:
            n_legs = 4
    n_legs = max(2, min(n_legs, 6))
    if n_legs == 4:
        coords = [(-1, -1), (1, -1), (-1, 1), (1, 1)]
    else:
        coords = [
            (np.cos(2 * np.pi * k / n_legs), np.sin(2 * np.pi * k / n_legs))
            for k in range(n_legs)
        ]
    for sx, sy in coords:
        leg = _box(DEFAULT_LEG_T, DEFAULT_LEG_T, floor_to_seat)
        leg.apply_translation([
            sx * (seat_w / 2 - DEFAULT_LEG_T / 2),
            sy * (seat_d / 2 - DEFAULT_LEG_T / 2),
            floor_to_seat / 2,
        ])
        parts.append(leg)
    return parts


def _build_backrest(seat_w: float, seat_d: float, seat_h: float,
                    back_h: float, recline_deg: float) -> trimesh.Trimesh:
    back = _box(seat_w, 3.0, back_h)
    # Pivot at the back edge of the seat top.
    pivot_y = -seat_d / 2 + 1.5
    pivot_z = seat_h
    back.apply_translation([0, pivot_y, pivot_z + back_h / 2])
    if abs(recline_deg) > 1e-6:
        # Rotate around the X axis at the pivot.
        T1 = trimesh.transformations.translation_matrix([0, -pivot_y, -pivot_z])
        R = trimesh.transformations.rotation_matrix(np.radians(-recline_deg), [1, 0, 0])
        T2 = trimesh.transformations.translation_matrix([0, pivot_y, pivot_z])
        M = T2 @ R @ T1
        back.apply_transform(M)
    return back


def _build_armrests(seat_w: float, seat_d: float, seat_h: float,
                    padded: bool) -> list:
    parts = []
    arm_h = 22.0
    arm_l = seat_d * 0.85
    arm_t = 4.0 if padded else 2.5
    arm_top = seat_h + arm_h
    for sx in (-1, 1):
        # Vertical post.
        post = _box(arm_t, arm_t, arm_h)
        post.apply_translation([sx * (seat_w / 2 + arm_t / 2),
                                 -seat_d / 2 + arm_l / 2, seat_h + arm_h / 2])
        parts.append(post)
        # Horizontal pad.
        pad = _box(arm_t * 1.2, arm_l, arm_t)
        pad.apply_translation([sx * (seat_w / 2 + arm_t / 2),
                                -seat_d / 2 + arm_l / 2, arm_top])
        parts.append(pad)
    return parts


def _build_grab_handles(seat_w: float, seat_d: float, seat_h: float) -> list:
    parts = []
    for sx in (-1, 1):
        handle = _cyl(1.2, seat_d * 0.6)
        # Lay the cylinder along Y by rotating 90deg around X.
        R = trimesh.transformations.rotation_matrix(np.pi / 2, [1, 0, 0])
        handle.apply_transform(R)
        handle.apply_translation([sx * (seat_w / 2 + 4), 0, seat_h + 30])
        parts.append(handle)
    return parts


def _build_side_rails(seat_w: float, seat_d: float, seat_h: float) -> list:
    parts = []
    for sx in (-1, 1):
        rail = _box(2.0, seat_d * 0.9, 18.0)
        rail.apply_translation([sx * (seat_w / 2 + 2), 0, seat_h + 9])
        parts.append(rail)
    return parts


# --------------------------- Primitives -----------------------------

def _box(x: float, y: float, z: float) -> trimesh.Trimesh:
    return trimesh.creation.box(extents=[max(x, 0.1), max(y, 0.1), max(z, 0.1)])


def _cyl(radius: float, height: float, sections: int = 24) -> trimesh.Trimesh:
    return trimesh.creation.cylinder(radius=max(radius, 0.1),
                                      height=max(height, 0.1),
                                      sections=sections)


def _safe(v, default):
    try:
        f = float(v) if v is not None else default
        if not np.isfinite(f) or f <= 0:
            return default
        return f
    except Exception:
        return default


# --------------------------- Export helpers --------------------------

def mesh_to_ascii_stl(mesh: trimesh.Trimesh) -> str:
    """Export to ASCII STL string."""
    buf = io.BytesIO()
    mesh.export(file_obj=buf, file_type="stl_ascii")
    return buf.getvalue().decode("utf-8")


def mesh_to_binary_stl(mesh: trimesh.Trimesh) -> bytes:
    buf = io.BytesIO()
    mesh.export(file_obj=buf, file_type="stl")
    return buf.getvalue()
