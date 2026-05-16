"""Natural language context engine.

Given a user prompt + the current chair intelligence, produce a
transformation spec the synthesis engine understands. Uses Anthropic
Claude when ANTHROPIC_API_KEY is set, otherwise a rule-based fallback.

Transformation schema (kept stable; the synthesis engine reads it):
{
  "target_environment": "hospital" | "office" | "lounge" | ...,
  "scale": {"x": 1.0, "y": 1.0, "z": 1.0},
  "seat": {"widen": 0.0, "height_delta_cm": 0.0, "cushion": false},
  "backrest": {"raise_cm": 0.0, "recline_deg": 0.0, "headrest": false},
  "armrests": {"add": false, "remove": false, "padded": false},
  "base": {"type": "keep"|"5-wheel"|"4-leg"|"pedestal"|"sled",
            "add_wheels": false, "lock_wheels": false},
  "safety": {"grab_handles": false, "anti_slip": false, "side_rails": false},
  "material_hint": "medical-grade plastic",
  "notes": "..."
}
"""
from __future__ import annotations

import json
import os
from typing import Any, Dict, Optional


SYSTEM_PROMPT = """You are a CAD design transformation planner.
Given a chair intelligence JSON and a user request, output ONLY a single JSON
object matching this schema (use defaults shown):

{
  "target_environment": "<short tag, e.g. hospital, office, lounge, gaming, elderly_care, classroom, outdoor>",
  "scale": {"x": 1.0, "y": 1.0, "z": 1.0},
  "seat": {"widen": 0.0, "height_delta_cm": 0.0, "cushion": false},
  "backrest": {"raise_cm": 0.0, "recline_deg": 0.0, "headrest": false},
  "armrests": {"add": false, "remove": false, "padded": false},
  "base": {"type": "keep", "add_wheels": false, "lock_wheels": false},
  "safety": {"grab_handles": false, "anti_slip": false, "side_rails": false},
  "material_hint": "",
  "notes": ""
}

Rules:
- Output JSON only, no markdown fences, no commentary.
- Keep numeric deltas modest (-15..15 cm, scales 0.7..1.4).
- Use base.type ∈ {"keep","5-wheel","4-leg","pedestal","sled"}.
- For hospitals: prefer grab_handles, anti_slip, padded armrests, medical-grade plastic.
- For elderly: add_wheels rarely, add grab_handles, raise seat 2-5cm, padded armrests.
- For gaming: high backrest, headrest, recline 5-15deg, 5-wheel base.
- For lounge: recline 10-25deg, widen 0.1-0.3, no wheels.
"""


def llm_transform(prompt: str, intelligence: Dict[str, Any],
                  api_key: Optional[str] = None,
                  model: Optional[str] = None) -> Dict[str, Any]:
    """Call Claude to produce a transformation spec; fall back if unavailable."""
    api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        return _rule_based(prompt, intelligence, reason="no_api_key")

    try:
        import anthropic  # type: ignore
    except Exception:
        return _rule_based(prompt, intelligence, reason="anthropic_sdk_missing")

    model = model or os.environ.get("LLM_MODEL") or "claude-opus-4-7"
    user_msg = (
        f"Chair intelligence:\n```json\n{json.dumps(intelligence, indent=2)}\n```\n\n"
        f"User request: {prompt}\n\n"
        "Return the transformation JSON now."
    )
    try:
        client = anthropic.Anthropic(api_key=api_key)
        resp = client.messages.create(
            model=model,
            max_tokens=1024,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_msg}],
        )
        text = "".join(b.text for b in resp.content if getattr(b, "type", "") == "text")
        spec = _extract_json(text)
        spec.setdefault("notes", "")
        spec["_source"] = "anthropic"
        return spec
    except Exception as e:  # network, parsing, rate limit, etc.
        spec = _rule_based(prompt, intelligence, reason=f"llm_error:{type(e).__name__}")
        return spec


def _extract_json(text: str) -> Dict[str, Any]:
    """Strip code fences if the model included them, then parse."""
    t = text.strip()
    if t.startswith("```"):
        # Remove leading fence
        first_nl = t.find("\n")
        t = t[first_nl + 1 :] if first_nl >= 0 else t
        if t.endswith("```"):
            t = t[: -3]
    # Try whole-string parse first.
    try:
        return json.loads(t)
    except json.JSONDecodeError:
        pass
    # Find first '{' to last '}'.
    a = t.find("{")
    b = t.rfind("}")
    if a >= 0 and b > a:
        return json.loads(t[a : b + 1])
    raise ValueError("Model output did not contain JSON")


# ----------------------------- Fallback --------------------------------

def _rule_based(prompt: str, intelligence: Dict[str, Any], reason: str = "") -> Dict[str, Any]:
    p = prompt.lower()
    spec = _default_spec()
    spec["_source"] = f"rule_based:{reason}" if reason else "rule_based"

    if any(w in p for w in ("hospital", "clinic", "patient", "medical")):
        spec["target_environment"] = "hospital"
        spec["safety"].update(grab_handles=True, anti_slip=True, side_rails=False)
        spec["armrests"].update(add=True, padded=True)
        spec["seat"].update(widen=0.1, cushion=True)
        spec["material_hint"] = "medical-grade plastic"
        spec["notes"] = "Hardened for patient transfers; non-porous, wipeable surfaces."
    elif any(w in p for w in ("elder", "elderly", "senior", "aging", "ageing")):
        spec["target_environment"] = "elderly_care"
        spec["safety"].update(grab_handles=True, anti_slip=True)
        spec["armrests"].update(add=True, padded=True)
        spec["seat"].update(height_delta_cm=4.0, cushion=True)
        spec["material_hint"] = "soft-touch polymer"
        spec["notes"] = "Raised seat for sit-to-stand assist; grab handles for support."
    elif any(w in p for w in ("gaming", "esports", "gamer")):
        spec["target_environment"] = "gaming"
        spec["backrest"].update(raise_cm=10.0, recline_deg=8.0, headrest=True)
        spec["armrests"].update(add=True, padded=True)
        spec["base"].update(type="5-wheel", add_wheels=True)
        spec["material_hint"] = "carbon-fiber-reinforced polymer"
        spec["notes"] = "High back + headrest for long sessions; 5-wheel mobile base."
    elif any(w in p for w in ("lounge", "relax", "living room", "cafe")):
        spec["target_environment"] = "lounge"
        spec["backrest"].update(recline_deg=18.0)
        spec["seat"].update(widen=0.2, cushion=True)
        spec["base"].update(type="sled")
        spec["material_hint"] = "upholstered foam over molded shell"
        spec["notes"] = "Reclined back and widened seat for casual seating."
    elif any(w in p for w in ("office", "desk", "work")):
        spec["target_environment"] = "office"
        spec["armrests"].update(add=True)
        spec["base"].update(type="5-wheel", add_wheels=True)
        spec["seat"].update(height_delta_cm=2.0)
        spec["material_hint"] = "mesh-back ergonomic polymer"
        spec["notes"] = "Mobile office configuration with armrests."
    elif any(w in p for w in ("outdoor", "garden", "patio")):
        spec["target_environment"] = "outdoor"
        spec["base"].update(type="4-leg")
        spec["safety"].update(anti_slip=True)
        spec["material_hint"] = "UV-resistant polypropylene"
        spec["notes"] = "Weatherproofed for outdoor use."
    elif any(w in p for w in ("classroom", "school", "student")):
        spec["target_environment"] = "classroom"
        spec["base"].update(type="4-leg")
        spec["armrests"].update(remove=True)
        spec["material_hint"] = "impact-resistant polymer"
        spec["notes"] = "Lightweight, stackable, no armrests."
    elif any(w in p for w in ("wheelchair", "accessible", "mobility")):
        spec["target_environment"] = "wheelchair_compatible"
        spec["armrests"].update(add=True, padded=True)
        spec["safety"].update(grab_handles=True)
        spec["base"].update(type="4-leg")
        spec["scale"]["x"] = 1.15
        spec["material_hint"] = "anti-microbial polymer"
        spec["notes"] = "Widened seat and side support for transfer."
    else:
        spec["notes"] = "No specific environment matched — kept original geometry."

    return spec


def _default_spec() -> Dict[str, Any]:
    return {
        "target_environment": "generic",
        "scale": {"x": 1.0, "y": 1.0, "z": 1.0},
        "seat": {"widen": 0.0, "height_delta_cm": 0.0, "cushion": False},
        "backrest": {"raise_cm": 0.0, "recline_deg": 0.0, "headrest": False},
        "armrests": {"add": False, "remove": False, "padded": False},
        "base": {"type": "keep", "add_wheels": False, "lock_wheels": False},
        "safety": {"grab_handles": False, "anti_slip": False, "side_rails": False},
        "material_hint": "",
        "notes": "",
    }
