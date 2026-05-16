# QR-Guided Synthetic CAD Generation from STL Slices using Natural Language Context

An end-to-end research pipeline that takes a chair STL, slices and
analyzes it, encodes the resulting CAD intelligence into a QR code, then
generates a new synthetic STL CAD model adapted to a user-described
environment (hospital, lounge, gaming, elderly support, …) using a
natural-language → transformation-spec engine.

```
chair.stl ──► loader ──► slicer ──► feature intelligence ──► QR memory
                                              │
        ┌─────────────────────────────────────┘
        ▼
  "How would this chair look in a hospital?"
        │
        ▼
   NL context engine (Claude API + rule-based fallback)
        │
        ▼
  transformation spec  ──►  procedural synthesis  ──►  synth.stl
```

## Project layout

```
qr-cad-pipeline/
├── backend/
│   ├── app/
│   │   ├── main.py                 FastAPI app + endpoints
│   │   └── modules/
│   │       ├── stl_loader.py       STL parsing + descriptors
│   │       ├── slicer.py           Z-axis slicing → chair attributes
│   │       ├── features.py         Feature vector + classifier
│   │       ├── qr_codec.py         JSON ⇄ gzip ⇄ base64 ⇄ QR PNG(s)
│   │       ├── nl_context.py       Claude API + rule-based fallback
│   │       └── synthesis.py        Procedural chair STL synthesis
│   ├── scripts/make_sample_chair.py
│   ├── tests/test_pipeline.py      End-to-end smoke test
│   ├── data/{uploads,outputs,qr,sample_chairs}/
│   ├── requirements.txt
│   └── .env.example
└── frontend/
    ├── src/
    │   ├── App.jsx                 Layout + flow
    │   ├── components/
    │   │   ├── StlViewer.jsx       three.js viewer
    │   │   ├── FeaturePanel.jsx    Extracted feature table
    │   │   └── DensityChart.jsx    Slice density signal
    │   └── styles.css
    ├── index.html
    ├── package.json
    └── vite.config.js              Proxies /api, /files, /uploads, /qr → :8000
```

## Quick start

### 1. Backend (Python 3.10+)

```bash
cd backend
pip install --user -r requirements.txt          # or use a venv
cp .env.example .env                            # optional: add ANTHROPIC_API_KEY
uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
```

Smoke test (no API needed — uses rule-based fallback):

```bash
python3 tests/test_pipeline.py
```

Generate a sample chair if you don't have one:

```bash
python3 scripts/make_sample_chair.py --out data/sample_chairs/office.stl --kind office --binary
```

### 2. Frontend (Node 18+)

```bash
cd frontend
npm install
npm run dev
```

Open `http://localhost:5173` (or whichever port Vite assigns).

The Vite dev server proxies `/api`, `/files`, `/uploads`, `/qr` to
`http://127.0.0.1:8000`, so the React UI talks to FastAPI without CORS
headaches.

### 3. Use it

1. Drop a chair STL into the dropzone.
2. The right panel shows extracted features + classification.
3. A QR is generated automatically (gzip + base64; chunked if needed).
4. Type a prompt (or pick an example) and hit **Plan + Generate STL**.
5. Compare original vs synthetic in the two viewers; download either.

## API

| Method | Path | Body | Description |
| --- | --- | --- | --- |
| `POST` | `/api/analyze` | multipart `file` (.stl) | Load + slice + classify + QR-encode. Returns `session_id`, intelligence JSON, slice preview, density profile, QR images (base64). |
| `POST` | `/api/transform` | `{session_id, prompt}` | LLM-driven transformation spec (rule-based fallback if no key). |
| `POST` | `/api/generate` | `{session_id, spec?}` | Procedurally build synthetic STL from spec; returns URL. |
| `GET` | `/api/sessions/{id}` | — | Cached intelligence + QR metadata. |
| `GET` | `/api/sessions/{id}/slices` | `?max_layers=32` | Down-sampled slice info + density profile. |
| `GET` | `/api/sessions/{id}/compare` | — | URLs for original + synthetic, plus spec/prompt. |
| `GET` | `/api/sessions/{id}/qr_decode` | — | Round-trip decode of the QR payload to prove it's lossless. |
| `GET` | `/api/health` | — | Liveness + LLM-configured flag. |

## How the pieces work

### STL loader (`stl_loader.py`)
- `trimesh.load` handles both ASCII and binary; multi-body STLs are concatenated.
- Extracts bounding box, dimensions, surface area, volume (voxel fallback if non-watertight), triangle/vertex count, and a face-normal-dispersion curvature proxy.

### Slicer (`slicer.py`)
- Cuts the mesh with `mesh.section(plane_origin=[0,0,z], plane_normal=[0,0,1])` for N layers.
- Per layer: area, perimeter, polygon-count, centroid, bbox, occupancy (filled / bbox).
- Derives chair attributes: seat height (lowest broad, high-occupancy layer), leg count (mode of components below seat), base type (`5-wheel`, `pedestal`, `sled`, `N-leg`), armrest presence (≥3 components above seat), backrest angle (centroid-Y drift above seat), stability (base/seat area ratio), ergonomic score.

### Feature intelligence (`features.py`)
- Resamples the density profile down to 16 floats — a compact shape signature.
- Heuristic classifier with per-class scores across `office / dining / lounge / hospital / wheelchair_compatible / gaming / stool / outdoor`.
- Assembles the canonical intelligence JSON returned to clients and embedded into QR.

### QR codec (`qr_codec.py`)
- Pipeline: JSON → gzip → base64 → QR (auto version, ECC=M).
- Chunked transparently if the payload exceeds a single QR (≈ 2.2 KB).
- `decode_intelligence_from_b64gz` is the round-trip; `decode_intelligence_from_qrs` requires `pyzbar` (optional native dep).

### NL context engine (`nl_context.py`)
- Uses Anthropic Claude (`claude-opus-4-7` by default) when `ANTHROPIC_API_KEY` is set.
- Tight system prompt that pins the output schema; numeric deltas clamped to sane ranges.
- Falls back to a deterministic keyword classifier when no key is configured — so the demo works offline.

### Synthesis (`synthesis.py`)
- Builds the chair as a union of `trimesh.creation.box` and `cylinder` primitives.
- Honors transformation spec: scale, seat width / height delta / cushion, backrest raise / recline / headrest, armrests (padded), base type (`keep / 5-wheel / 4-leg / pedestal / sled`), wheels, grab handles, side rails, anti-slip pads.
- Exports both ASCII (default; portable) and binary STL.

## Configuration

Create `backend/.env` to enable real LLM calls:

```
ANTHROPIC_API_KEY=sk-ant-...
LLM_MODEL=claude-opus-4-7
```

Without it the transform step uses the deterministic rule-based fallback,
which is enough to exercise the pipeline end-to-end.

## Sample run (offline, rule-based)

```text
→ generating sample office chair STL …
  STL bytes: 35484
  triangles=708  vol=22644.40  area=15296.72
  slice layers: 60
  seat_height=39.4  base_type=pedestal  armrests=True  ergo=0.84
  classified as: office (conf=0.5714)
  QR: 1 chunk(s)  raw=705B  gz=425B
  QR round-trip OK ✓
  NL spec source: rule_based:no_api_key  target=hospital
  synth STL: data/outputs/smoke_synth.stl  triangles=852
ALL OK ✓
```

## Research extensions baked in

- **QR-based compressed CAD memory** — gzip + base64 + auto-chunking; the QR is a portable lossless intelligence packet.
- **Geometric tokenization** — `features.density_signature` is a 16-bucket resampling of the Z-density profile, a compact shape descriptor.
- **Procedural geometry synthesis** — primitive-grammar chair generator parameterized by the transformation spec.
- **Hooks for ML upgrades**:
  - `classify_chair` returns a feature vector — swap the heuristic for a trained classifier without API changes.
  - `synthesis.generate_chair` could be replaced with a diffusion or implicit-shape decoder; the spec schema stays the same.
  - `density_signature` is a candidate for contrastive embedding training (CAD retrieval, RAG over an STL corpus).

## Notes

- The slicer's "seat height" is robust on well-formed chairs but degrades on heavily decorative meshes. Tune `num_layers` if your STLs are very tall/thin.
- `pyzbar` (QR decode from PNG) requires the native zbar library. The pipeline doesn't depend on it — encoding works without it.
- The procedural synthesis is intentionally schematic: the goal is a visibly-adapted CAD model, not a CAD-quality reproduction. To plug in a real generative model, replace `generate_chair` while keeping the spec contract.
