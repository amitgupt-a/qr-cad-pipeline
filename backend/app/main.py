"""FastAPI entrypoint for the QR-CAD pipeline."""
from __future__ import annotations

import io
import json
import os
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from app.modules import features as feat_mod
from app.modules import nl_context, qr_codec, slicer
from app.modules import stl_loader, synthesis

load_dotenv()

ROOT = Path(__file__).resolve().parent.parent
UPLOADS = ROOT / "data" / "uploads"
OUTPUTS = ROOT / "data" / "outputs"
QR_DIR = ROOT / "data" / "qr"
for d in (UPLOADS, OUTPUTS, QR_DIR):
    d.mkdir(parents=True, exist_ok=True)

# In-memory session store: { session_id: {...} }
SESSIONS: Dict[str, Dict[str, Any]] = {}


app = FastAPI(title="QR-Guided Synthetic CAD Pipeline", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve generated STLs at /files/...
app.mount("/files", StaticFiles(directory=str(OUTPUTS)), name="files")
app.mount("/uploads", StaticFiles(directory=str(UPLOADS)), name="uploads")


# ----------------------------- Models -------------------------------

class TransformRequest(BaseModel):
    session_id: str
    prompt: str


class GenerateRequest(BaseModel):
    session_id: str
    spec: Optional[Dict[str, Any]] = None  # if absent, uses last transform


# --------------------------- Endpoints ------------------------------

@app.get("/api/health")
def health():
    return {
        "ok": True,
        "llm_configured": bool(os.environ.get("ANTHROPIC_API_KEY")),
        "sessions": len(SESSIONS),
    }


@app.post("/api/analyze")
async def analyze(file: UploadFile = File(...)):
    """Upload an STL, run the full geometric + slicing + classification pass.

    Returns a session_id used by later endpoints.
    """
    if not file.filename.lower().endswith(".stl"):
        raise HTTPException(400, "Only .stl files supported")

    raw = await file.read()
    if len(raw) == 0:
        raise HTTPException(400, "Empty upload")

    session_id = uuid.uuid4().hex[:12]
    upload_path = UPLOADS / f"{session_id}.stl"
    upload_path.write_bytes(raw)

    try:
        bundle = stl_loader.load_stl(raw, source_name=file.filename)
    except Exception as e:
        raise HTTPException(400, f"Failed to parse STL: {e}")

    t0 = time.time()
    geometry = stl_loader.geometry_descriptors(bundle.mesh)
    layers = slicer.slice_mesh(bundle.mesh, num_layers=80)
    density = slicer.density_map(layers)
    attrs = slicer.chair_attributes(bundle.mesh, layers)
    features = feat_mod.build_feature_vector(geometry, attrs, density)
    classification = feat_mod.classify_chair(features)
    intelligence = feat_mod.assemble_cad_intelligence(
        bundle.source_name, geometry, attrs, features, classification
    )
    elapsed = round(time.time() - t0, 3)

    qr_pkg = qr_codec.encode_intelligence(intelligence)
    # Persist the first QR for download.
    qr_paths = []
    for i, b64 in enumerate(qr_pkg["qr_images_b64"]):
        p = QR_DIR / f"{session_id}_qr_{i+1}.png"
        import base64
        p.write_bytes(base64.b64decode(b64))
        qr_paths.append(f"/qr/{p.name}")

    SESSIONS[session_id] = {
        "source_name": bundle.source_name,
        "upload_path": str(upload_path),
        "intelligence": intelligence,
        "layers": [l.to_dict() for l in layers],
        "density": density,
        "qr": {
            "n_chunks": qr_pkg["n_chunks"],
            "raw_bytes": qr_pkg["raw_bytes"],
            "compressed_bytes": qr_pkg["compressed_bytes"],
            "payload_b64gz": qr_pkg["payload_b64gz"],
            "paths": qr_paths,
        },
    }

    return {
        "session_id": session_id,
        "source": bundle.source_name,
        "elapsed_sec": elapsed,
        "intelligence": intelligence,
        "slice_preview": slicer.slice_summary(layers, max_layers=32),
        "density": density,
        "qr": {
            "n_chunks": qr_pkg["n_chunks"],
            "raw_bytes": qr_pkg["raw_bytes"],
            "compressed_bytes": qr_pkg["compressed_bytes"],
            "images_b64": qr_pkg["qr_images_b64"],
            "paths": qr_paths,
        },
        "original_stl_url": f"/uploads/{upload_path.name}",
    }


@app.get("/qr/{name}")
def get_qr(name: str):
    p = QR_DIR / name
    if not p.exists():
        raise HTTPException(404, "QR not found")
    return FileResponse(str(p), media_type="image/png")


@app.get("/api/sessions/{session_id}")
def get_session(session_id: str):
    s = SESSIONS.get(session_id)
    if not s:
        raise HTTPException(404, "Unknown session")
    # Strip layers/density from default view; expose via dedicated route.
    out = {k: v for k, v in s.items() if k not in ("layers", "density")}
    out["layer_count"] = len(s["layers"])
    return out


@app.get("/api/sessions/{session_id}/slices")
def get_slices(session_id: str, max_layers: int = 32):
    s = SESSIONS.get(session_id)
    if not s:
        raise HTTPException(404, "Unknown session")
    layers = s["layers"]
    if len(layers) > max_layers:
        step = len(layers) / max_layers
        sample = [layers[int(i * step)] for i in range(max_layers)]
    else:
        sample = layers
    return {"layers": sample, "density": s["density"]}


@app.post("/api/transform")
def transform(req: TransformRequest):
    s = SESSIONS.get(req.session_id)
    if not s:
        raise HTTPException(404, "Unknown session")
    spec = nl_context.llm_transform(req.prompt, s["intelligence"])
    s["last_spec"] = spec
    s["last_prompt"] = req.prompt
    return {"spec": spec, "prompt": req.prompt}


@app.post("/api/generate")
def generate(req: GenerateRequest):
    s = SESSIONS.get(req.session_id)
    if not s:
        raise HTTPException(404, "Unknown session")
    spec = req.spec or s.get("last_spec")
    if not spec:
        raise HTTPException(400, "No transformation spec — call /api/transform first or pass `spec`.")

    mesh = synthesis.generate_chair(s["intelligence"], spec)

    ascii_stl = synthesis.mesh_to_ascii_stl(mesh)
    binary_stl = synthesis.mesh_to_binary_stl(mesh)

    base = f"{req.session_id}_synth_{int(time.time())}"
    ascii_path = OUTPUTS / f"{base}.stl"  # ascii by default for portability
    ascii_path.write_text(ascii_stl)
    bin_path = OUTPUTS / f"{base}_bin.stl"
    bin_path.write_bytes(binary_stl)

    s["last_synth_path"] = str(ascii_path)
    s["last_synth_bin"] = str(bin_path)
    s["last_synth_spec"] = spec

    return {
        "spec": spec,
        "synthetic_stl_url": f"/files/{ascii_path.name}",
        "synthetic_stl_binary_url": f"/files/{bin_path.name}",
        "bbox": {"min": mesh.bounds[0].tolist(), "max": mesh.bounds[1].tolist()},
        "triangle_count": int(len(mesh.faces)),
    }


@app.get("/api/sessions/{session_id}/compare")
def compare(session_id: str):
    s = SESSIONS.get(session_id)
    if not s:
        raise HTTPException(404, "Unknown session")
    synth = s.get("last_synth_path")
    if not synth:
        raise HTTPException(400, "No synthetic STL yet — call /api/generate first.")
    orig_name = Path(s["upload_path"]).name
    synth_name = Path(synth).name
    return {
        "original_stl_url": f"/uploads/{orig_name}",
        "synthetic_stl_url": f"/files/{synth_name}",
        "original_intelligence": s["intelligence"],
        "transformation_spec": s.get("last_synth_spec"),
        "prompt": s.get("last_prompt"),
    }


@app.get("/api/sessions/{session_id}/qr_decode")
def qr_decode(session_id: str):
    """Round-trip: decode our saved b64gz payload back to JSON to prove it works."""
    s = SESSIONS.get(session_id)
    if not s:
        raise HTTPException(404, "Unknown session")
    payload = s["qr"]["payload_b64gz"]
    return qr_codec.decode_intelligence_from_b64gz(payload)


@app.get("/", response_class=PlainTextResponse)
def root():
    return (
        "QR-Guided Synthetic CAD Pipeline\n\n"
        "POST /api/analyze         (multipart STL)\n"
        "POST /api/transform       {session_id, prompt}\n"
        "POST /api/generate        {session_id, spec?}\n"
        "GET  /api/sessions/{id}\n"
        "GET  /api/sessions/{id}/slices\n"
        "GET  /api/sessions/{id}/compare\n"
        "GET  /api/sessions/{id}/qr_decode\n"
        "GET  /api/health\n"
    )
