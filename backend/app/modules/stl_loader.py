"""STL loader and base geometry analysis.

Supports binary + ASCII STL via trimesh (which falls back to numpy-stl
parsing internally). Extracts a dictionary of geometric descriptors
used downstream by the slicing engine and feature intelligence layer.
"""
from __future__ import annotations

import io
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np
import trimesh


@dataclass
class MeshBundle:
    """Container for a loaded mesh + cached descriptors."""

    mesh: trimesh.Trimesh
    source_name: str

    def as_dict(self) -> Dict[str, Any]:
        return {
            "source_name": self.source_name,
            "geometry": geometry_descriptors(self.mesh),
        }


def load_stl(path_or_bytes, source_name: Optional[str] = None) -> MeshBundle:
    """Load an STL file from disk path or raw bytes.

    Trimesh autodetects ASCII vs binary, so this works for both. We also
    force the mesh to be a single Trimesh (concatenating scenes if the
    file contained multiple bodies).
    """
    if isinstance(path_or_bytes, (bytes, bytearray)):
        mesh = trimesh.load(io.BytesIO(bytes(path_or_bytes)), file_type="stl")
        name = source_name or "uploaded.stl"
    else:
        mesh = trimesh.load(path_or_bytes, file_type="stl")
        name = source_name or os.path.basename(str(path_or_bytes))

    if isinstance(mesh, trimesh.Scene):
        geoms = [g for g in mesh.geometry.values() if isinstance(g, trimesh.Trimesh)]
        if not geoms:
            raise ValueError("STL contained no triangle meshes")
        mesh = trimesh.util.concatenate(geoms)

    if not isinstance(mesh, trimesh.Trimesh):
        raise ValueError("Loaded geometry is not a Trimesh")

    # Repair common issues so volume/area are sane.
    mesh.merge_vertices()
    mesh.remove_unreferenced_vertices()
    return MeshBundle(mesh=mesh, source_name=name)


def load_many(paths: List[str]) -> List[MeshBundle]:
    return [load_stl(p) for p in paths]


def geometry_descriptors(mesh: trimesh.Trimesh) -> Dict[str, Any]:
    """Bounding box, dims, surface area, volume, triangle count, curvature."""
    bounds = mesh.bounds  # (2, 3)
    extents = mesh.extents  # (3,)
    triangle_count = int(len(mesh.faces))

    # Volume may be negative / unreliable on non-watertight meshes.
    try:
        volume = float(abs(mesh.volume))
    except Exception:
        volume = float(_voxel_volume_estimate(mesh))

    surface_area = float(mesh.area)

    # Curvature stats: approximate using face-normal dispersion at each vertex.
    curvature = _vertex_curvature_stats(mesh)

    # Structural geometry summary.
    structural = {
        "center_of_mass": mesh.center_mass.tolist() if mesh.is_watertight else mesh.centroid.tolist(),
        "is_watertight": bool(mesh.is_watertight),
        "euler_number": int(mesh.euler_number) if hasattr(mesh, "euler_number") else None,
        "convex_hull_volume": float(abs(mesh.convex_hull.volume)),
        "compactness": float(surface_area ** 3 / max(volume ** 2, 1e-9)) if volume > 0 else None,
    }

    return {
        "bounding_box": {
            "min": bounds[0].tolist(),
            "max": bounds[1].tolist(),
        },
        "dimensions": {
            "x": float(extents[0]),
            "y": float(extents[1]),
            "z": float(extents[2]),
        },
        "surface_area": surface_area,
        "volume": volume,
        "triangle_count": triangle_count,
        "vertex_count": int(len(mesh.vertices)),
        "curvature": curvature,
        "structural": structural,
    }


def _vertex_curvature_stats(mesh: trimesh.Trimesh) -> Dict[str, float]:
    """Cheap proxy for curvature: dispersion of incident face normals.

    For each vertex we look at the angle between adjacent face normals.
    High dispersion => high local curvature.
    """
    try:
        face_normals = mesh.face_normals  # (F, 3)
        # vertex_faces is (V, max_incident) padded with -1.
        vf = mesh.vertex_faces
        if vf is None or len(vf) == 0:
            return {"mean": 0.0, "std": 0.0, "max": 0.0}
        v_curvatures = np.zeros(len(mesh.vertices), dtype=np.float64)
        for i in range(len(mesh.vertices)):
            idxs = vf[i]
            idxs = idxs[idxs >= 0]
            if len(idxs) < 2:
                continue
            normals = face_normals[idxs]
            ref = normals.mean(axis=0)
            ref /= max(np.linalg.norm(ref), 1e-9)
            cos = np.clip(normals @ ref, -1.0, 1.0)
            angles = np.arccos(cos)
            v_curvatures[i] = float(angles.mean())
        return {
            "mean": float(v_curvatures.mean()),
            "std": float(v_curvatures.std()),
            "max": float(v_curvatures.max()),
        }
    except Exception:
        return {"mean": 0.0, "std": 0.0, "max": 0.0}


def _voxel_volume_estimate(mesh: trimesh.Trimesh, pitch_div: int = 64) -> float:
    """Fallback volume when the mesh is not watertight."""
    longest = float(mesh.extents.max())
    pitch = max(longest / pitch_div, 1e-3)
    try:
        vox = mesh.voxelized(pitch=pitch)
        return float(vox.filled_count * (pitch ** 3))
    except Exception:
        return float(mesh.convex_hull.volume) * 0.5
