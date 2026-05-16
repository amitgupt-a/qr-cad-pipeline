"""End-to-end smoke test for the pipeline.

Builds a sample chair via the synthesis module, then drives:
  load -> slice -> features -> classify -> intelligence -> QR -> decode
  -> NL transform (rule-based) -> generate synthetic STL.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.modules import features as feat_mod
from app.modules import nl_context, qr_codec, slicer, stl_loader, synthesis


def build_sample_bytes(kind: str = "office") -> bytes:
    from scripts.make_sample_chair import build_sample
    mesh = build_sample(kind)
    return synthesis.mesh_to_binary_stl(mesh)


def run():
    print("→ generating sample office chair STL …")
    raw = build_sample_bytes("office")
    print(f"  STL bytes: {len(raw)}")

    bundle = stl_loader.load_stl(raw, source_name="sample_office.stl")
    geom = stl_loader.geometry_descriptors(bundle.mesh)
    print(f"  triangles={geom['triangle_count']}  vol={geom['volume']:.2f}  area={geom['surface_area']:.2f}")

    layers = slicer.slice_mesh(bundle.mesh, num_layers=60)
    print(f"  slice layers: {len(layers)}")

    density = slicer.density_map(layers)
    attrs = slicer.chair_attributes(bundle.mesh, layers)
    print(f"  seat_height={attrs['seat_height']:.1f}  base_type={attrs['base_type']}  "
          f"armrests={attrs['armrests']}  ergo={attrs['ergonomic_score']:.2f}")

    features = feat_mod.build_feature_vector(geom, attrs, density)
    cls = feat_mod.classify_chair(features)
    print(f"  classified as: {cls['label']} (conf={cls['confidence']})")

    intelligence = feat_mod.assemble_cad_intelligence(
        "sample_office.stl", geom, attrs, features, cls
    )

    qr_pkg = qr_codec.encode_intelligence(intelligence)
    print(f"  QR: {qr_pkg['n_chunks']} chunk(s)  raw={qr_pkg['raw_bytes']}B  "
          f"gz={qr_pkg['compressed_bytes']}B")

    decoded = qr_codec.decode_intelligence_from_b64gz(qr_pkg["payload_b64gz"])
    assert decoded == intelligence, "QR round-trip mismatch!"
    print("  QR round-trip OK ✓")

    prompt = "How would this chair look in a hospital? Make it safer for patient transfers."
    spec = nl_context.llm_transform(prompt, intelligence)
    print(f"  NL spec source: {spec.get('_source')}  target={spec['target_environment']}")

    synth = synthesis.generate_chair(intelligence, spec)
    out = Path(__file__).resolve().parent.parent / "data" / "outputs" / "smoke_synth.stl"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(synthesis.mesh_to_ascii_stl(synth))
    print(f"  synth STL: {out}  triangles={len(synth.faces)}")
    print("\nALL OK ✓")


if __name__ == "__main__":
    run()
