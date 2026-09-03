#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = ["google-genai>=1.0"]
# ///
"""One-shot: EN pack -> data/es_overlay.json via GEMINI_API_KEY_FREE."""

from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path

from google import genai
from google.genai import types

ROOT = Path(__file__).resolve().parents[1]


def load_env() -> None:
    env = Path.home() / "Projects" / ".env.local"
    if not env.exists():
        return
    for line in env.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def main() -> None:
    load_env()
    key = os.environ.get("GEMINI_API_KEY_FREE")
    if not key:
        raise SystemExit("GEMINI_API_KEY_FREE missing")

    reqs = json.loads((ROOT / "data/requests.json").read_text())
    rec = json.loads((ROOT / "data/judgments.json").read_text())
    judgments = rec["judgments"]
    items = []
    for r in reqs:
        j = judgments[r["id"]]
        items.append(
            {
                "id": r["id"],
                "text": r["text"],
                "quote": j.get("quote") or "",
                "reason": j.get("reason") or "",
                "decision": j.get("decision") or "",
            }
        )

    prompt = (
        "You translate a lead-qualification demo pack from English to Spanish "
        "(LatAm, neutral).\n"
        "Return ONLY a JSON array. For each item keep the same id.\n"
        "Rules:\n"
        "- text: natural Spanish translation of the customer message\n"
        "- quote: a contiguous substring COPIED from your Spanish text "
        "(must appear verbatim inside text). Prefer the meaning of the original "
        "quote; if it cannot sit inside text, pick the best justifying sentence "
        "from the Spanish text.\n"
        "- reason: Spanish translation of the reason\n"
        "Do not add fields. Do not wrap in markdown.\n"
        "Input:\n"
        + json.dumps(items, ensure_ascii=False)
    )

    client = genai.Client(api_key=key)
    cfg = types.GenerateContentConfig(
        temperature=0.2, response_mime_type="application/json"
    )
    t0 = time.time()
    resp = client.models.generate_content(
        model="gemini-2.5-flash", contents=prompt, config=cfg
    )
    print("seconds", round(time.time() - t0, 1), "chars", len(resp.text or ""))
    data = json.loads(resp.text)
    if not isinstance(data, list) or len(data) != 40:
        raise SystemExit(f"bad response shape: {type(data)} len={getattr(data, '__len__', None)}")

    bad = [d["id"] for d in data if d.get("quote") and d["quote"] not in d["text"]]
    print("quote misses", bad)

    out = {
        "requests": {d["id"]: {"text": d["text"]} for d in data},
        "judgments": {
            d["id"]: {
                "quote": d.get("quote") or "",
                "reason": d.get("reason") or "",
            }
            for d in data
        },
        "rule_labels": {
            "spam_or_ads": "Spam o publicidad",
            "no_clear_request": "Sin pedido claro",
            "duplicate_same_day": "Duplicado de la misma persona hoy",
            "outside_service_area": "Fuera de nuestra zona",
            "not_our_service": "No es un servicio que ofrecemos",
            "below_minimum": "Pedido por debajo del mínimo",
            "after_hours_no_surcharge": "Visita fuera de horario sin aceptar el recargo",
        },
        "mechanical_reasons": {
            "spam_or_ads": "Parece publicidad o spam de leads",
            "no_clear_request": "El mensaje no tiene un pedido accionable",
            "duplicate_same_day": "Misma persona ya escribió hoy",
            "outside_service_area": "Está fuera de la zona de servicio",
            "not_our_service": "No es un servicio que ofrecemos",
            "below_minimum": "Por debajo del mínimo de pedido",
            "after_hours_no_surcharge": "Pide fuera de horario sin aceptar el recargo",
        },
    }
    for rid in bad:
        text = out["requests"][rid]["text"]
        sent = re.split(r"(?<=[.!?])\s+", text.strip())[0][:180]
        out["judgments"][rid]["quote"] = sent
        assert sent in text, rid

    path = ROOT / "data/es_overlay.json"
    path.write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n")
    print("wrote", path)


if __name__ == "__main__":
    main()
