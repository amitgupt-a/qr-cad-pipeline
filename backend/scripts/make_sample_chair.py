"""Generate a sample chair STL so the pipeline can be exercised offline."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.modules import synthesis


def build_sample(kind: str = "office") -> "trimesh.Trimesh":
    intelligence = {
        "geometry": {"dimensions": {"x": 50.0, "y": 50.0, "z": 100.0}},
        "slicing": {"seat_height": 45.0, "base_type": "4-leg"},
    }
    if kind == "office":
        intelligence["slicing"]["base_type"] = "5-wheel"
        spec = _spec(armrests=True, base="5-wheel")
    elif kind == "stool":
        spec = _spec(base="4-leg")
        spec["backrest"]["raise_cm"] = -50  # essentially hides it
    else:  # dining
        spec = _spec(base="4-leg")
    return synthesis.generate_chair(intelligence, spec)


def _spec(armrests=False, base="4-leg") -> dict:
    return {
        "target_environment": "sample",
        "scale": {"x": 1.0, "y": 1.0, "z": 1.0},
        "seat": {"widen": 0.0, "height_delta_cm": 0.0, "cushion": False},
        "backrest": {"raise_cm": 0.0, "recline_deg": 0.0, "headrest": False},
        "armrests": {"add": armrests, "remove": False, "padded": False},
        "base": {"type": base, "add_wheels": False, "lock_wheels": False},
        "safety": {"grab_handles": False, "anti_slip": False, "side_rails": False},
        "material_hint": "",
        "notes": "",
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/sample_chairs/office_chair.stl")
    ap.add_argument("--kind", default="office", choices=["office", "dining", "stool"])
    ap.add_argument("--binary", action="store_true")
    args = ap.parse_args()

    mesh = build_sample(args.kind)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    if args.binary:
        out.write_bytes(synthesis.mesh_to_binary_stl(mesh))
    else:
        out.write_text(synthesis.mesh_to_ascii_stl(mesh))
    print(f"wrote {out}  triangles={len(mesh.faces)}  bytes={out.stat().st_size}")


if __name__ == "__main__":
    main()
