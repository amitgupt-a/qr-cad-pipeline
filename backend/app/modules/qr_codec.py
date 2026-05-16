"""QR encode/decode of CAD intelligence JSON.

We gzip the JSON, base64-encode, and stuff into a QR code (max version
~40 ≈ 2.9 KB binary). For larger payloads we fall back to chunked QRs:
multiple PNGs with a small header `{n_chunks, idx, payload}`.
"""
from __future__ import annotations

import base64
import gzip
import io
import json
from typing import Any, Dict, List, Optional

import qrcode
from PIL import Image

try:
    from pyzbar.pyzbar import decode as zbar_decode  # type: ignore
    _HAS_ZBAR = True
except Exception:  # pragma: no cover - optional native dep
    _HAS_ZBAR = False


# Conservative single-QR payload (QR v40 binary ~ 2953 bytes; we keep margin).
SINGLE_QR_BUDGET = 2200


def encode_intelligence(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Encode an intelligence JSON to one or more QR PNGs (base64 strings)."""
    raw = json.dumps(payload, separators=(",", ":")).encode("utf-8")
    compressed = gzip.compress(raw, compresslevel=9)
    b64 = base64.b64encode(compressed).decode("ascii")

    chunks = _split_chunks(b64, SINGLE_QR_BUDGET)
    qr_images_b64: List[str] = []
    for i, chunk in enumerate(chunks):
        wrapper = f"QCAD:{i+1}/{len(chunks)}:{chunk}"
        qr_images_b64.append(_make_qr_png_b64(wrapper))

    return {
        "n_chunks": len(chunks),
        "raw_bytes": len(raw),
        "compressed_bytes": len(compressed),
        "encoded_bytes": len(b64),
        "qr_images_b64": qr_images_b64,
        # Echo back the canonical compressed string so a client can rebuild
        # the JSON even without a QR reader.
        "payload_b64gz": b64,
    }


def decode_intelligence_from_b64gz(b64gz: str) -> Dict[str, Any]:
    """Inverse of the b64+gzip+json packing."""
    raw = gzip.decompress(base64.b64decode(b64gz))
    return json.loads(raw.decode("utf-8"))


def decode_intelligence_from_qrs(qr_image_paths: List[str]) -> Dict[str, Any]:
    """Decode a list of QR PNGs back into the original intelligence dict.

    Requires the pyzbar native lib. Falls back with an informative error
    if not available on this host.
    """
    if not _HAS_ZBAR:
        raise RuntimeError("pyzbar/zbar is not installed; cannot decode QR images")
    pieces: Dict[int, str] = {}
    total = None
    for p in qr_image_paths:
        img = Image.open(p)
        decoded = zbar_decode(img)
        if not decoded:
            raise ValueError(f"Could not decode QR: {p}")
        text = decoded[0].data.decode("utf-8")
        if not text.startswith("QCAD:"):
            raise ValueError(f"Unexpected payload header: {text[:16]}")
        header, payload = text[5:].split(":", 1)
        idx_str, total_str = header.split("/")
        idx = int(idx_str)
        total = int(total_str)
        pieces[idx] = payload
    if total is None or len(pieces) != total:
        raise ValueError(f"Missing QR chunks: have {len(pieces)} of {total}")
    full_b64 = "".join(pieces[i + 1] for i in range(total))
    return decode_intelligence_from_b64gz(full_b64)


def _make_qr_png_b64(text: str) -> str:
    qr = qrcode.QRCode(
        version=None,  # auto-size
        error_correction=qrcode.constants.ERROR_CORRECT_M,
        box_size=6,
        border=2,
    )
    qr.add_data(text)
    qr.make(fit=True)
    img = qr.make_image(fill_color="black", back_color="white")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _split_chunks(s: str, size: int) -> List[str]:
    return [s[i : i + size] for i in range(0, len(s), size)] or [""]
