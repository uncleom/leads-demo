#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = ["fastapi>=0.115", "uvicorn>=0.32", "google-genai>=1.0"]
# ///
"""Demo: inbound requests for a field-service company, rules then the model.

Cheap rules close the obvious junk. The model only sees who is left
and sorts them into outcomes: a manager, a calendar link, a newsletter, a decline.
Disputed ones (no confidence, or a quote that is not in the text) go to a person.

    ./app.py                 # locally on :8080
    ./app.py --selftest      # mechanics and assembly, no model calls
    ./app.py --record        # write model opinions to data/judgments.json once
"""
from __future__ import annotations

import base64
import datetime
import json
import logging
import os
import pathlib
import re
import threading
import time

from fastapi import FastAPI, Request
from fastapi.exception_handlers import (
    http_exception_handler,
    request_validation_exception_handler,
)
from fastapi.exceptions import RequestValidationError
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel, Field
from starlette.exceptions import HTTPException as StarletteHTTPException

ROOT = pathlib.Path(__file__).resolve().parent
RECORDING_PATH = ROOT / "data/judgments.json"
# Where recording credentials come from. Both are optional: the page serves the
# recorded run and never calls a model, so a deployment needs neither.
ACCOUNTS = pathlib.Path(os.environ.get("VERTEX_ACCOUNTS_FILE", "")) if os.environ.get("VERTEX_ACCOUNTS_FILE") else pathlib.Path.home() / ".config/lead-demo/accounts.json"
ENV_FILE = pathlib.Path(os.environ.get("ENV_FILE", "")) if os.environ.get("ENV_FILE") else pathlib.Path.home() / ".config/lead-demo/.env"
CHAT_MODEL = os.environ.get("CHAT_MODEL", "gemini-2.5-flash")
# Gemini 2.5 Flash list price, USD per token. Source: ai.google.dev/gemini-api/docs/pricing,
# checked 2026-08-23 ($0.30 per 1M input, $2.50 per 1M output).
PRICE_IN, PRICE_OUT = 0.30 / 1e6, 2.50 / 1e6

BUSINESS = json.loads((ROOT / "data/business.json").read_text())
REQUESTS = json.loads((ROOT / "data/requests.json").read_text())
REQ_BY_ID = {r["id"]: r for r in REQUESTS}
DEFAULT_CRITERIA = dict(BUSINESS["criteria"])
ES_OVERLAY_PATH = ROOT / "data/es_overlay.json"
ES_OVERLAY = (
    json.loads(ES_OVERLAY_PATH.read_text(encoding="utf-8"))
    if ES_OVERLAY_PATH.exists()
    else {}
)
GATE_REASONS_ES = {
    "Model cited a phrase that is not in the request": (
        "El modelo citó una frase que no está en el mensaje"
    ),
    "Model was not sure": "El modelo no estaba seguro",
    "Needs a person": "Necesita a una persona",
    "Survived the rules, waiting for the model.": (
        "Pasó las reglas, esperando al modelo."
    ),
    "No sentence was stored for this drop.": (
        "No se guardó ninguna frase para este descarte."
    ),
}

app = FastAPI()
_client_cache: dict[str, object] = {}
_client_lock = threading.Lock()

MSG_BROKEN = "Something broke on our side. This page should never show a raw error."
MSG_NO_RECORDING = (
    "This demo shows a recorded screening, and that recording is missing right now."
)

MODEL_DECISIONS = ("manager", "calendar", "newsletter", "decline", "uncertain")
BUCKETS = ("manager", "calendar", "newsletter", "decline", "disputed")


# --- Rules. Order = junk first, then the substance of the request ----------------
RULE_LABELS = {
    "spam_or_ads": "Spam or advertising",
    "no_clear_request": "No clear request",
    "duplicate_same_day": "Duplicate from the same person today",
    "outside_service_area": "Outside our service area",
    "not_our_service": "Not a service we offer",
    "below_minimum": "Order below our minimum",
    "after_hours_no_surcharge": "After-hours visit without accepting the surcharge",
}

SPAM_RE = re.compile(
    r"""(?ix)
    SEO\s+package
    | buy\s+(?:our\s+)?(?:SEO|followers|likes)
    | earn\s+from\s+home
    | crypto\s+signal
    | click\s+here\s+for\s+a\s+demo
    | free\s+Instagram\s+growth
    | VIP\s+list
    | marketing\s+agency
    | exclusive\s+HVAC\s+leads
    | media\s+kit
    | buy\s+followers\s+and\s+likes
    """,
)

FOREIGN_SERVICE_RE = re.compile(
    r"""(?ix)
    \broof\b | shingles
    | wedding\s+catering | \bcatering\b
    | company\s+website | Google\s+Ads
    | dog\s+walking
    | kitchen\s+remodel | cabinets,\s*counters
    | build\s+a\s+new\s+company\s+website
    """,
)

INTENT_RE = re.compile(
    r"""(?ix)
    \bAC\b | air\s*cond | mini-?split | thermostat | duct
    | cool(?:ing)? | heat(?:ing)? | install | repair | service | tune-?up
    | leak | outlet | plumb | electric | washer | hookup | filter
    | quote | book | visit | tech | unit | broken | not\s+cooling
    | dripping | mold | maintenance | waitlist | newsletter
    """,
)

AMOUNT_RE = re.compile(
    r"""(?ix)
    \$\s*(\d{1,4})(?:\s*max)?
    | (?:under|below|for)\s+\$\s*(\d{1,4})
    | cash\s+\$\s*(\d{1,4})
    | (?:do\s+it\s+for|only\s+want.*?)\s+\$\s*(\d{1,4})
    """,
)

TIME_CLOCK_RE = re.compile(
    r"""(?ix)
    \b(?:at\s+)?(\d{1,2})\s*(:(\d{2}))?\s*(am|pm)\b
    | \b(\d{1,2})\s*(am|pm)\b
    """,
)

WEEKEND_RE = re.compile(r"\b(Saturday|Sunday)\b", re.I)
TONIGHT_RE = re.compile(r"\btonight\b", re.I)
SURCHARGE_OK_RE = re.compile(
    r"""(?ix)
    after-?hours\s+fee
    | surcharge
    | extra\s+(?:fee|charge)
    | fee\s+is\s+fine
    | ok\s+with\s+the\s+(?:fee|extra)
    | fine\s+with\s+(?:the\s+)?extra
    """,
)

GREETING_ONLY_RE = re.compile(
    r"""(?ix)
    ^[\s?!.]*
    (?:hi|hello|hey|hola|hello\s+are\s+you\s+there)
    [\s?!.]*$
    """,
)


def first_sentence(text: str) -> str:
    part = re.split(r"(?<=[.!?])\s+", text.strip(), maxsplit=1)[0]
    words = part.split()
    return " ".join(words[:18])


def sentence_with(text: str, needle: str) -> str:
    if not needle:
        return first_sentence(text)
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    low = needle.lower()
    for p in parts:
        if low in p.lower():
            words = p.strip().split()
            return " ".join(words[:28])
    # Phrase with no period - cut a window around the match.
    idx = text.lower().find(low)
    if idx >= 0:
        left = max(0, idx - 40)
        right = min(len(text), idx + len(needle) + 40)
        chunk = text[left:right].strip()
        return " ".join(chunk.split()[:28])
    return first_sentence(text)


def word_count(text: str) -> int:
    return len(text.split())


def headline_for(req: dict) -> str:
    area = (req.get("client") or {}).get("area") or "no area"
    when = str(req.get("time") or "")[11:16] or "??:??"
    return f"{req.get('channel', '?')} · {when} · {area}"


def person_key(req: dict) -> tuple[str, str] | None:
    c = req.get("client") or {}
    phone = (c.get("phone") or "").strip()
    if phone:
        return ("phone", phone)
    email = (c.get("email") or "").strip().lower()
    if email:
        return ("email", email)
    handle = (c.get("handle") or "").strip().lower()
    if handle:
        return ("handle", handle)
    name = (req.get("name") or "").strip().lower()
    if name and name not in {"unknown", "anon"}:
        return ("name", name)
    return None


def parse_amount(text: str) -> tuple[int | None, str]:
    m = AMOUNT_RE.search(text)
    if not m:
        return None, ""
    raw = next(g for g in m.groups() if g)
    return int(raw), m.group(0).strip()


def parse_clock_hour(text: str) -> tuple[int | None, str]:
    m = TIME_CLOCK_RE.search(text)
    if not m:
        return None, ""
    if m.group(1) is not None:
        hour = int(m.group(1))
        ampm = (m.group(4) or "").lower()
    else:
        hour = int(m.group(5))
        ampm = (m.group(6) or "").lower()
    if ampm == "pm" and hour != 12:
        hour += 12
    if ampm == "am" and hour == 12:
        hour = 0
    return hour, m.group(0).strip()


def after_hours_hit(text: str, start: int, end: int) -> tuple[bool, str]:
    """Ask to come outside the shift or on a weekend. Quote is taken from the text."""
    weekend = WEEKEND_RE.search(text)
    hour, hour_src = parse_clock_hour(text)
    tonight = TONIGHT_RE.search(text)

    if hour is not None and (hour < start or hour >= end):
        return True, sentence_with(text, hour_src or (tonight.group(0) if tonight else ""))
    if weekend and hour is not None:
        # A weekend at any named hour is outside the usual shift.
        return True, sentence_with(text, weekend.group(0))
    if weekend and re.search(r"\b(come|visit|book|only\s+(?:time|window))\b", text, re.I):
        return True, sentence_with(text, weekend.group(0))
    if tonight and hour is not None and (hour < start or hour >= end):
        return True, sentence_with(text, tonight.group(0))
    if tonight and hour is None:
        # "tonight" with no hour still counts as outside the day shift for a visit.
        if re.search(r"\b(come|tech|visit|please\s+come)\b", text, re.I):
            return True, sentence_with(text, tonight.group(0))
    return False, ""


def _reject(req: dict, rule: str, quote: str, reason: str) -> dict:
    return {
        "id": req["id"],
        "name": req["name"],
        "channel": req["channel"],
        "time": req["time"],
        "headline": headline_for(req),
        "body": req["text"],
        "client": req.get("client") or {},
        "decision": "decline",
        "layer": "mechanical",
        "rule": rule,
        "rule_label": RULE_LABELS[rule],
        "quote": quote,
        "reason": reason,
        "confidence": 1.0,
    }


def _survive(req: dict) -> dict:
    return {
        "id": req["id"],
        "name": req["name"],
        "channel": req["channel"],
        "time": req["time"],
        "headline": headline_for(req),
        "body": req["text"],
        "client": req.get("client") or {},
        "decision": "pending",
        "layer": "model",
        "rule": None,
        "rule_label": None,
        "quote": "",
        "reason": "Survived the rules, waiting for the model.",
        "confidence": None,
    }


def screen_one(req: dict, c: dict, seen: dict[tuple[str, str], str]) -> dict:
    """The first matching rule closes the request. No model."""
    text = req["text"]
    client = req.get("client") or {}

    m = SPAM_RE.search(text)
    if m:
        return _reject(
            req, "spam_or_ads", sentence_with(text, m.group(0)),
            "Looks like advertising or lead spam",
        )

    stripped = text.strip()
    if (
        not stripped
        or stripped in {"?", "??", "???"}
        or GREETING_ONLY_RE.match(stripped)
        or (
            not INTENT_RE.search(text)
            and word_count(text) < 16
            and re.search(r"\b(um|uh|nevermind|never mind)\b", text, re.I)
        )
    ):
        return _reject(
            req, "no_clear_request", first_sentence(text) or stripped or "?",
            "Message has no actionable request",
        )

    key = person_key(req)
    if key is not None:
        if key in seen:
            return _reject(
                req, "duplicate_same_day", first_sentence(text),
                f"Same person as {seen[key]} earlier today",
            )
        seen[key] = req["id"]

    area = (client.get("area") or "").strip()
    service_area = [str(a).strip() for a in (c.get("service_area") or []) if str(a).strip()]
    if area and service_area and area not in service_area:
        needle = area if area.lower() in text.lower() else ""
        return _reject(
            req, "outside_service_area",
            sentence_with(text, needle) if needle else first_sentence(text),
            f"{area} is outside the service area",
        )

    m = FOREIGN_SERVICE_RE.search(text)
    if m:
        return _reject(
            req, "not_our_service", sentence_with(text, m.group(0)),
            "Asked for work we do not offer",
        )

    amount, amount_src = parse_amount(text)
    min_order = int(c["min_order_usd"])
    if amount is not None and amount < min_order:
        return _reject(
            req, "below_minimum", sentence_with(text, amount_src or ""),
            f"${amount} is below the ${min_order} minimum",
        )

    start, end = int(c["work_hours_start"]), int(c["work_hours_end"])
    hit, quote = after_hours_hit(text, start, end)
    if hit and not SURCHARGE_OK_RE.search(text):
        return _reject(
            req, "after_hours_no_surcharge", quote or first_sentence(text),
            f"Outside {start:02d}:00-{end:02d}:00 without accepting the surcharge",
        )

    return _survive(req)


def screen_all(reqs: list[dict], criteria: dict) -> list[dict]:
    ordered = sorted(reqs, key=lambda r: r["time"])
    seen: dict[tuple[str, str], str] = {}
    by_id = {r["id"]: screen_one(r, criteria, seen) for r in ordered}
    return [by_id[r["id"]] for r in reqs]


# --- Recording provider: Vertex first, then a plain Gemini API key ---
def _env(name: str) -> str | None:
    if os.environ.get(name):
        return os.environ[name]
    if not ENV_FILE.exists():
        return None
    for line in ENV_FILE.read_text().splitlines():
        line = line.strip()
        if line.startswith(f"{name}="):
            return line.split("=", 1)[1].strip().strip('"').strip("'")
    return None


def _vertex_accounts() -> list[dict]:
    if os.environ.get("GOOGLE_ADC_B64"):
        return [{
            "name": "env",
            "project": os.environ["VERTEX_PROJECT"],
            "location": os.environ.get("VERTEX_LOCATION", "us-central1"),
            "adc_b64": os.environ["GOOGLE_ADC_B64"],
        }]
    if not ACCOUNTS.exists():
        return []
    today = datetime.date.today().isoformat()
    live = [a for a in json.loads(ACCOUNTS.read_text())
            if not (a.get("expires") and a["expires"] < today)]
    return sorted(live, key=lambda a: a.get("expires") or "9999")


def model_client():
    """Model client. In the container, creds come from env; locally, accounts.json, then the free key."""
    today = datetime.date.today().isoformat()
    if today in _client_cache:
        return _client_cache[today]
    # Under a lock, with a second check: parallel calls without it each built
    # their own client and overwrote the other through clear().
    with _client_lock:
        if today in _client_cache:
            return _client_cache[today]
        return _build_client(today)


def _build_client(today: str):
    from google import genai

    last_err: Exception | None = None
    for acc in _vertex_accounts():
        try:
            if acc.get("adc_b64"):
                adc = pathlib.Path("/tmp/adc.json")
                adc.write_bytes(base64.b64decode(acc["adc_b64"]))
                os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = str(adc)
            elif acc.get("adc"):
                os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = str(
                    pathlib.Path(acc["adc"]).expanduser())
            c = genai.Client(
                vertexai=True,
                project=acc["project"],
                location=acc.get("location", "us-central1"),
            )
            _client_cache.clear()
            _client_cache[today] = c
            return c
        except Exception as e:
            last_err = e

    key = _env("GEMINI_API_KEY_FREE")
    if key:
        c = genai.Client(api_key=key)
        _client_cache.clear()
        _client_cache[today] = c
        return c

    raise RuntimeError(
        "no live Vertex account and no GEMINI_API_KEY_FREE"
        + (f" ({type(last_err).__name__}: {last_err})" if last_err else "")
    )


JUDGE_PROMPT = """You triage one inbound service request for a field-service company.

COMPANY
{company} — {title}
{summary}

SERVICE AREA: {service_area}
SERVICES WE OFFER: {services}
MINIMUM ORDER: ${min_order} USD
WORKING HOURS: {hours_start}:00-{hours_end}:00 local, surcharges apply outside that window

REQUEST
Channel: {channel}
When: {time}
Name: {name}
Area on file: {area}
Text:
{body}

Choose exactly one decision:
- manager: valuable or complex job that needs a person (multi-unit, commercial, unclear scope with real money, smell/mold/safety).
- calendar: straightforward job the customer can self-book (single clear repair, service, install quote, hookup).
- newsletter: warm but not now (traveling, next month, waitlist, "keep me posted").
- decline: polite no — outside what we should take, even if rules did not catch it.
- uncertain: you cannot tell from the text.

quote: a short phrase COPIED VERBATIM from the request text (at most 25 words). Do not invent a quote.
reason: one sentence, in English.
confidence: number from 0 to 1 for how sure you are.

Return JSON only with keys decision, quote, reason, confidence.
"""

JUDGE_SCHEMA = {
    "type": "object",
    "properties": {
        "decision": {
            "type": "string",
            "enum": ["manager", "calendar", "newsletter", "decline", "uncertain"],
        },
        "quote": {"type": "string"},
        "reason": {"type": "string"},
        "confidence": {"type": "number"},
    },
    "required": ["decision", "quote", "reason", "confidence"],
}


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip().lower()


def quote_in_body(quote: str, body: str) -> bool:
    q = _norm(quote)
    return bool(q) and q in _norm(body)


def _is_exhausted(e: Exception) -> bool:
    """429 / RESOURCE_EXHAUSTED / quota - this rung is spent, not a real breakage."""
    t = f"{type(e).__name__}: {e}".lower()
    return "429" in t or "resource_exhausted" in t or "quota" in t


_free_lock = threading.Lock()


def free_client():
    """Fallback rung: a plain Gemini API key when Vertex is out of quota."""
    if "free" in _client_cache:
        return _client_cache["free"]
    with _free_lock:
        if "free" in _client_cache:
            return _client_cache["free"]
        from google import genai
        key = _env("GEMINI_API_KEY_FREE")
        if not key:
            raise RuntimeError("Vertex is exhausted and GEMINI_API_KEY_FREE is missing")
        _client_cache["free"] = genai.Client(api_key=key)
        return _client_cache["free"]


def judge(req: dict, criteria: dict) -> dict:
    """One model call for one request."""
    from google.genai import types

    client = req.get("client") or {}
    prompt = JUDGE_PROMPT.format(
        company=BUSINESS["company"],
        title=BUSINESS["title"],
        summary=BUSINESS["summary"],
        service_area=", ".join(criteria.get("service_area") or []),
        services=", ".join(criteria.get("services") or []),
        min_order=int(criteria["min_order_usd"]),
        hours_start=int(criteria["work_hours_start"]),
        hours_end=int(criteria["work_hours_end"]),
        channel=req["channel"],
        time=req["time"],
        name=req["name"],
        area=client.get("area") or "(not given)",
        body=req["text"],
    )
    t0 = time.time()
    cfg = types.GenerateContentConfig(
        temperature=0,
        response_mime_type="application/json",
        response_json_schema=JUDGE_SCHEMA,
    )
    try:
        r = model_client().models.generate_content(
            model=CHAT_MODEL, contents=prompt, config=cfg)
    except Exception as e:
        # This rung's quota ran out - step down, do not crash.
        if not _is_exhausted(e):
            raise
        r = free_client().models.generate_content(
            model=CHAT_MODEL, contents=prompt, config=cfg)
    u = getattr(r, "usage_metadata", None)
    tin = getattr(u, "prompt_token_count", 0) or 0
    tout = getattr(u, "candidates_token_count", 0) or 0
    try:
        parsed = json.loads((r.text or "").strip())
    except json.JSONDecodeError:
        parsed = {
            "decision": "uncertain",
            "quote": "",
            "reason": "Model did not return JSON.",
            "confidence": 0.0,
        }
    decision = parsed.get("decision") if parsed.get("decision") in MODEL_DECISIONS else "uncertain"
    quote = str(parsed.get("quote") or "").strip()
    reason = str(parsed.get("reason") or "").strip()
    try:
        confidence = float(parsed.get("confidence"))
    except (TypeError, ValueError):
        confidence = 0.0
    confidence = max(0.0, min(1.0, confidence))
    if quote and not quote_in_body(quote, req["text"]):
        decision = "uncertain"
        reason = "Model cited a phrase that is not in the request"
        confidence = min(confidence, 0.4)
    return {
        "decision": decision,
        "quote": quote,
        "reason": reason,
        "confidence": confidence,
        "model": CHAT_MODEL,
        "tokens_in": tin,
        "tokens_out": tout,
        "cost": tin * PRICE_IN + tout * PRICE_OUT,
        "seconds": round(time.time() - t0, 1),
    }


class Criteria(BaseModel):
    service_area: list[str] = Field(
        default_factory=lambda: list(DEFAULT_CRITERIA["service_area"]))
    services: list[str] = Field(
        default_factory=lambda: list(DEFAULT_CRITERIA["services"]))
    min_order_usd: int = Field(default=DEFAULT_CRITERIA["min_order_usd"], ge=0, le=10_000)
    work_hours_start: int = Field(
        default=DEFAULT_CRITERIA["work_hours_start"], ge=0, le=23)
    work_hours_end: int = Field(
        default=DEFAULT_CRITERIA["work_hours_end"], ge=1, le=24)
    confidence_threshold: float = Field(
        default=DEFAULT_CRITERIA["confidence_threshold"], ge=0.0, le=1.0)

    def as_dict(self) -> dict:
        d = self.model_dump()
        d["service_area"] = [s.strip() for s in d["service_area"] if s and s.strip()][:20]
        d["services"] = [s.strip() for s in d["services"] if s and s.strip()][:20]
        for lst in (d["service_area"], d["services"]):
            for i, s in enumerate(lst):
                lst[i] = s[:60]
        if d["work_hours_start"] >= d["work_hours_end"]:
            d["work_hours_start"], d["work_hours_end"] = 8, 18
        return d


class RunBody(BaseModel):
    criteria: Criteria | None = None
    with_model: bool = False  # ignored: /run never calls the model
    lang: str = "en"


def _translate_gate_reason(reason: str) -> str:
    if reason in GATE_REASONS_ES:
        return GATE_REASONS_ES[reason]
    # Confidence 0.40 is below the 0.70 threshold
    m = re.match(
        r"Confidence ([0-9.]+) is below the ([0-9.]+) threshold", reason or ""
    )
    if m:
        return (
            f"La confianza {m.group(1)} está por debajo del umbral {m.group(2)}"
        )
    return reason


def apply_es_overlay(out: dict) -> dict:
    """Mechanics stay on the English pack; only display strings switch to Spanish."""
    if not ES_OVERLAY:
        return out
    reqs = ES_OVERLAY.get("requests") or {}
    jud = ES_OVERLAY.get("judgments") or {}
    labels = ES_OVERLAY.get("rule_labels") or {}
    mech_reasons = ES_OVERLAY.get("mechanical_reasons") or {}

    def paint(row: dict) -> dict:
        rid = row.get("id")
        es_text = (reqs.get(rid) or {}).get("text")
        es_j = jud.get(rid) or {}
        row = dict(row)
        if es_text:
            row["body"] = es_text
        if row.get("rule"):
            if row["rule"] in labels:
                row["rule_label"] = labels[row["rule"]]
            if row["rule"] in mech_reasons:
                row["reason"] = mech_reasons[row["rule"]]
            q = es_j.get("quote") or ""
            if q and es_text and q in es_text:
                row["quote"] = q
            elif es_text and row.get("quote"):
                # Fall back: keep area token if it still appears in the ES body.
                token = str(row["quote"])
                row["quote"] = token if token in es_text else (es_j.get("quote") or token)
        else:
            raw_reason = str(row.get("reason") or "")
            if raw_reason.startswith(("Model cited", "Model was", "Needs a", "Confidence ")):
                row["reason"] = _translate_gate_reason(raw_reason)
            elif es_j.get("reason"):
                row["reason"] = es_j["reason"]
            q = es_j.get("quote") or ""
            if q and es_text and q in es_text:
                row["quote"] = q
        return row

    out = dict(out)
    out["structured"] = [paint(r) for r in out.get("structured") or []]
    out["naive"] = [paint(r) for r in out.get("naive") or []]
    buckets = {}
    for key, rows in (out.get("buckets") or {}).items():
        buckets[key] = [paint(r) for r in rows]
    out["buckets"] = buckets
    src = dict(out.get("source") or {})
    when = _pretty_when(str(src.get("recorded_at") or ""))
    src["message"] = (
        f"Corrida grabada del {when} con solicitudes sintéticas. Bayline y Port "
        "Meridian son ficticios. El botón no llama al modelo. Cambiar umbrales "
        "vuelve a correr las reglas; las opiniones del modelo siguen siendo las "
        "grabadas."
    )
    out["source"] = src
    return out


def _usage(parts: list[dict]) -> dict:
    tin = sum(p.get("tokens_in", 0) for p in parts)
    tout = sum(p.get("tokens_out", 0) for p in parts)
    cost = sum(p.get("cost", 0) for p in parts)
    secs = sum(p.get("seconds", 0) for p in parts)
    return {
        "calls": len(parts),
        "tokens_in": tin,
        "tokens_out": tout,
        "cost": cost,
        "cost_per_1000": cost * 1000,
        "seconds": round(secs, 1),
        "model": CHAT_MODEL,
    }


def load_recording() -> dict | None:
    if not RECORDING_PATH.exists():
        return None
    try:
        rec = json.loads(RECORDING_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    js = rec.get("judgments")
    if not isinstance(js, dict):
        return None
    if any(r["id"] not in js for r in REQUESTS):
        return None
    return rec


def _pretty_when(iso: str) -> str:
    try:
        dt = datetime.datetime.fromisoformat(iso.replace("Z", "+00:00"))
        return f"{dt.day} {dt.strftime('%B %Y')}"
    except (ValueError, TypeError):
        return (iso or "")[:10] or "an earlier run"


def _source(rec: dict) -> dict:
    when = _pretty_when(str(rec.get("recorded_at") or ""))
    return {
        "kind": "recorded",
        "recorded_at": rec.get("recorded_at") or "",
        "model": rec.get("model") or CHAT_MODEL,
        "message": (
            f"Recorded run from {when} on synthetic requests. Bayline and Port "
            "Meridian are fictional. Pressing the button does not call the model. "
            "Changing thresholds re-runs the rules; the model answers stay the "
            "recorded ones."
        ),
    }


def finalize_model_row(cheap_row: dict, req_row: dict, j: dict, criteria: dict) -> dict:
    """Model opinion plus the quote and confidence gates -> an outcome or disputed."""
    decision = j.get("decision") if j.get("decision") in MODEL_DECISIONS else "uncertain"
    quote = str(j.get("quote") or "").strip()
    reason = str(j.get("reason") or "").strip()
    try:
        confidence = float(j.get("confidence"))
    except (TypeError, ValueError):
        confidence = 0.0
    threshold = float(criteria["confidence_threshold"])

    disputed_reason = None
    if not quote or not quote_in_body(quote, req_row["text"]):
        decision = "uncertain"
        disputed_reason = "Model cited a phrase that is not in the request"
        quote = quote if quote_in_body(quote, req_row["text"]) else ""
    elif decision == "uncertain":
        disputed_reason = reason or "Model was not sure"
    elif confidence < threshold:
        disputed_reason = (
            f"Confidence {confidence:.2f} is below the {threshold:.2f} threshold"
        )
        decision = "uncertain"

    if decision == "uncertain":
        bucket = "disputed"
        final = "disputed"
        reason = disputed_reason or reason or "Needs a person"
    else:
        bucket = decision
        final = decision

    return {
        **cheap_row,
        "decision": final,
        "bucket": bucket,
        "layer": "model",
        "rule": None,
        "rule_label": None,
        "quote": quote,
        "reason": reason,
        "confidence": confidence,
    }


def _empty_buckets() -> dict[str, list]:
    return {b: [] for b in BUCKETS}


def assemble(criteria: dict) -> dict | None:
    """Live mechanics plus recorded model opinions. The model is not called."""
    rec = load_recording()
    if rec is None:
        return None
    cheap = screen_all(REQUESTS, criteria)
    dropped = [r for r in cheap if r["rule"]]
    survivors = [r for r in cheap if not r["rule"]]
    funnel = {
        "in": len(REQUESTS),
        "dropped": len(dropped),
        "to_model": len(survivors),
        "saved": len(dropped),
        "by_rule": {},
        "buckets": {b: 0 for b in BUCKETS},
    }
    for r in dropped:
        funnel["by_rule"][r["rule"]] = funnel["by_rule"].get(r["rule"], 0) + 1

    buckets = _empty_buckets()
    naive_rows: list[dict] = []
    structured_rows: list[dict] = []
    naive_parts: list[dict] = []
    struct_parts: list[dict] = []
    js = rec["judgments"]

    for req_row, cheap_row in zip(REQUESTS, cheap):
        j = js[req_row["id"]]
        naive_row = finalize_model_row(cheap_row, req_row, j, criteria)
        naive_rows.append(naive_row)
        naive_parts.append(j)

        if cheap_row["rule"]:
            row = {
                **cheap_row,
                "bucket": "decline",
                "decision": "decline",
            }
            structured_rows.append(row)
            buckets["decline"].append(row)
        else:
            row = finalize_model_row(cheap_row, req_row, j, criteria)
            structured_rows.append(row)
            buckets[row["bucket"]].append(row)
            struct_parts.append(j)

    for b in BUCKETS:
        funnel["buckets"][b] = len(buckets[b])

    model_name = rec.get("model") or CHAT_MODEL
    usage_naive = _usage(naive_parts)
    usage_struct = _usage(struct_parts)
    usage_naive["model"] = model_name
    usage_struct["model"] = model_name
    return {
        "funnel": funnel,
        "criteria": criteria,
        "buckets": buckets,
        "structured": structured_rows,
        "naive": naive_rows,
        "usage": {"naive": usage_naive, "structured": usage_struct},
        "source": _source(rec),
    }


def _judgment_payload(j: dict) -> dict:
    return {
        "decision": j["decision"],
        "quote": j["quote"],
        "reason": j["reason"],
        "confidence": j.get("confidence", 0.0),
        "tokens_in": j.get("tokens_in", 0),
        "tokens_out": j.get("tokens_out", 0),
        "cost": j.get("cost", 0),
        "seconds": j.get("seconds", 0),
    }


def record_judgments(*, force: bool = False) -> None:
    """Sequential write. Parallel would blow the free 5/min limit."""
    rec = {
        "recorded_at": datetime.datetime.now(datetime.timezone.utc).strftime(
            "%Y-%m-%dT%H:%M:%SZ"),
        "model": CHAT_MODEL,
        "judgments": {},
    }
    if RECORDING_PATH.exists() and not force:
        try:
            old = json.loads(RECORDING_PATH.read_text(encoding="utf-8"))
            rec["judgments"] = old.get("judgments") or {}
            rec["model"] = old.get("model") or CHAT_MODEL
        except (OSError, json.JSONDecodeError):
            pass
    if force:
        rec["judgments"] = {}

    criteria = dict(DEFAULT_CRITERIA)
    pending = [r for r in REQUESTS if r["id"] not in rec["judgments"]]
    for i, req_row in enumerate(pending):
        if i:
            time.sleep(13)
        last_err: Exception | None = None
        for attempt in range(4):
            try:
                j = judge(req_row, criteria)
                rec["judgments"][req_row["id"]] = _judgment_payload(j)
                last_err = None
                break
            except Exception as e:
                last_err = e
                if _is_exhausted(e) and attempt < 3:
                    print(f"{req_row['id']}: quota, wait", flush=True)
                    time.sleep(20)
                    continue
                raise
        if last_err:
            raise last_err
        print(req_row["id"], rec["judgments"][req_row["id"]]["decision"], flush=True)
        RECORDING_PATH.write_text(
            json.dumps(rec, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    missing = [r["id"] for r in REQUESTS if r["id"] not in rec["judgments"]]
    if missing:
        raise SystemExit(f"recording incomplete: {missing}")
    rec["recorded_at"] = datetime.datetime.now(datetime.timezone.utc).strftime(
        "%Y-%m-%dT%H:%M:%SZ")
    RECORDING_PATH.write_text(
        json.dumps(rec, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print("wrote", RECORDING_PATH)


@app.get("/boot")
def boot(lang: str = "en"):
    lang = "es" if (lang or "").lower().startswith("es") else "en"
    requests = REQUESTS
    rules = RULE_LABELS
    if lang == "es" and ES_OVERLAY:
        rules = ES_OVERLAY.get("rule_labels") or RULE_LABELS
        overlay_reqs = ES_OVERLAY.get("requests") or {}
        requests = [
            {**r, "text": (overlay_reqs.get(r["id"]) or {}).get("text", r["text"])}
            for r in REQUESTS
        ]
    return {
        "business": {
            k: BUSINESS[k] for k in ("company", "title", "summary", "city")
            if k in BUSINESS
        },
        "criteria": DEFAULT_CRITERIA,
        "requests": requests,
        "rules": rules,
        "buckets": list(BUCKETS),
        "lang": lang,
    }


@app.exception_handler(Exception)
async def hide_traces(request: Request, exc: Exception):
    if isinstance(exc, StarletteHTTPException):
        return await http_exception_handler(request, exc)
    if isinstance(exc, RequestValidationError):
        return await request_validation_exception_handler(request, exc)
    logging.exception("unhandled error")
    return JSONResponse(
        status_code=500,
        content={"error": "unavailable", "message": MSG_BROKEN},
    )


@app.post("/run")
def run(body: RunBody):
    criteria = (body.criteria or Criteria()).as_dict()
    out = assemble(criteria)
    if out is None:
        return JSONResponse(
            status_code=503,
            content={"error": "no_recording", "message": MSG_NO_RECORDING},
        )
    lang = "es" if (body.lang or "").lower().startswith("es") else "en"
    if lang == "es":
        out = apply_es_overlay(out)
    out["lang"] = lang
    return out


def _index_html() -> str:
    return (ROOT / "static/index.html").read_text(encoding="utf-8")


@app.get("/")
def index():
    return HTMLResponse(_index_html())


@app.get("/es")
@app.get("/es/")
def index_es():
    return HTMLResponse(_index_html())


# --- Selftest: mechanics, not a single model call --------------------------------
EXPECTED_RULE = {
    "r01": "spam_or_ads",
    "r02": "no_clear_request",
    "r03": "outside_service_area",
    "r04": "not_our_service",
    "r05": None,
    "r06": "below_minimum",
    "r07": "after_hours_no_surcharge",
    "r08": "spam_or_ads",
    "r09": "outside_service_area",
    "r10": None,
    "r11": "no_clear_request",
    "r12": None,
    "r13": "not_our_service",
    "r14": None,
    "r15": "duplicate_same_day",
    "r16": "outside_service_area",
    "r17": "below_minimum",
    "r18": "after_hours_no_surcharge",
    "r19": "spam_or_ads",
    "r20": None,
    "r21": "not_our_service",
    "r22": "no_clear_request",
    "r23": None,
    "r24": "outside_service_area",
    "r25": "below_minimum",
    "r26": "after_hours_no_surcharge",
    "r27": None,
    "r28": "duplicate_same_day",
    "r29": "not_our_service",
    "r30": None,
    "r31": "outside_service_area",
    "r32": "after_hours_no_surcharge",
    "r33": "spam_or_ads",
    "r34": "below_minimum",
    "r35": "no_clear_request",
    "r36": "not_our_service",
    "r37": None,
    "r38": "duplicate_same_day",
    "r39": None,
    "r40": None,
}


def selftest() -> None:
    c = dict(DEFAULT_CRITERIA)
    assert Criteria().as_dict()["service_area"] == DEFAULT_CRITERIA["service_area"]
    assert len(REQUESTS) == 40, f"expected 40 requests, got {len(REQUESTS)}"
    assert set(EXPECTED_RULE) == {r["id"] for r in REQUESTS}

    results = {r["id"]: r for r in screen_all(REQUESTS, c)}
    for rid, rule in EXPECTED_RULE.items():
        got = results[rid]["rule"]
        assert got == rule, f"{rid}: expected {rule!r}, got {got!r}"
        if rule:
            quote = results[rid]["quote"]
            assert quote, f"{rid}: empty quote"
            assert quote_in_body(quote, REQ_BY_ID[rid]["text"]), (
                f"{rid}: mechanical quote is not in the text: {quote!r}")
            assert results[rid]["rule_label"] == RULE_LABELS[rule]

    survivors = [rid for rid, rule in EXPECTED_RULE.items() if rule is None]
    dropped = [rid for rid, rule in EXPECTED_RULE.items() if rule]
    assert len(survivors) == 11, f"{len(survivors)} survivors, expected 11"
    assert len(dropped) == 29, f"{len(dropped)} closed, expected 29"

    assert parse_amount("cash $40 max")[0] == 40
    assert parse_amount("Can you do $35?")[0] == 35
    assert parse_amount("under $60 cash")[0] == 60
    assert parse_amount("Budget is fine")[0] is None
    assert parse_clock_hour("tonight at 10pm")[0] == 22
    assert parse_clock_hour("Sunday at 7am")[0] == 7
    assert parse_clock_hour("Sunday at 10am")[0] == 10

    # Thresholds on the page must change the result.
    wide_area = dict(c, service_area=list(c["service_area"]) + ["Cedar Falls"])
    assert screen_all([REQ_BY_ID["r03"]], wide_area)[0]["rule"] is None, (
        "adding Cedar Falls to the area - Nora should survive")

    cheap_min = dict(c, min_order_usd=30)
    assert screen_all([REQ_BY_ID["r06"]], cheap_min)[0]["rule"] is None, (
        "dropping the minimum to 30 - Sam should survive")

    long_hours = dict(c, work_hours_start=0, work_hours_end=24)
    # 10pm falls inside 0-24, but "tonight" plus a visit could still look after-hours;
    # with a round-the-clock shift, after_hours_hit will not fire on the hour.
    r07 = screen_all([REQ_BY_ID["r07"]], long_hours)[0]
    assert r07["rule"] is None, f"round-the-clock - Jules should survive, got {r07['rule']}"

    # Sunday 10am without the surcharge - r39 survives on the default area/hours;
    # without accepting the fee it must close.
    assert results["r39"]["rule"] is None
    no_fee = dict(REQ_BY_ID["r39"])
    no_fee["text"] = (
        "Lakeside condo AC not cooling. Sunday at 10am is the only time I am home - just book me."
    )
    no_fee["id"] = "r39x"
    assert screen_all([no_fee], c)[0]["rule"] == "after_hours_no_surcharge"

    # Channels and synthetic data.
    channels = {r["channel"] for r in REQUESTS}
    assert channels == {"form", "whatsapp", "instagram"}
    for r in REQUESTS:
        assert "@" not in (r.get("name") or "") or r["channel"] == "instagram"
        email = (r.get("client") or {}).get("email")
        if email:
            assert email.endswith(".example.com") or email.endswith("@example.com") or ".example" in email

    # The quote gate: a model phrase that is not a substring of the request
    # becomes disputed and never reaches the client as a decision.
    fake_req = dict(REQ_BY_ID["r05"])
    fake_j = {
        "decision": "manager",
        "quote": "this phrase is not in the request at all",
        "reason": "Looks like a job",
        "confidence": 0.95,
    }
    gated = finalize_model_row(_survive(fake_req), fake_req, fake_j, dict(c))
    assert gated["bucket"] == "disputed", gated
    assert gated["decision"] == "disputed"
    assert gated["quote"] == ""

    rec = load_recording()
    if rec is None:
        print("selftest ok (mechanics only; no data/judgments.json yet - run ./app.py --record)")
        return

    js = rec["judgments"]
    assert set(js) == {r["id"] for r in REQUESTS}
    for rid, j in js.items():
        assert j.get("decision") in MODEL_DECISIONS, rid

    out = assemble(dict(DEFAULT_CRITERIA))
    assert out is not None
    assert out["funnel"]["in"] == 40
    assert out["funnel"]["dropped"] == 29
    assert out["funnel"]["to_model"] == 11
    assert len(out["naive"]) == 40
    assert len(out["structured"]) == 40
    assert set(out["buckets"]) == set(BUCKETS)
    assert sum(len(out["buckets"][b]) for b in BUCKETS) == 40
    assert out["source"]["kind"] == "recorded"
    assert "does not call the model" in out["source"]["message"]

    # Recorded quotes are allowed to be inexact: the model sometimes stitches
    # two parts of a sentence. What we assert is that an unverified quote
    # lands in disputed and does not reach the client as a decision.
    by_id = {row["id"]: row for row in out["structured"]}
    client_facing = ("manager", "calendar", "newsletter", "decline")
    for rid, j in js.items():
        quote = str(j.get("quote") or "").strip()
        if quote and quote_in_body(quote, REQ_BY_ID[rid]["text"]):
            continue
        row = by_id[rid]
        if row.get("layer") != "model":
            continue
        assert row["bucket"] == "disputed", (
            f"{rid}: unverified quote must land in disputed, got {row['bucket']!r}")
        assert row["decision"] == "disputed"
        for b in client_facing:
            assert rid not in [x["id"] for x in out["buckets"][b]], (
                f"{rid}: unverified quote leaked to the client via {b}")
        assert rid in [x["id"] for x in out["buckets"]["disputed"]], (
            f"{rid}: unverified quote missing from disputed")

    # Closed by a rule sit in decline with the rule name and a quote from the text.
    for row in out["buckets"]["decline"]:
        if row.get("layer") == "mechanical":
            assert row["rule"] in RULE_LABELS
            assert row["quote"] and quote_in_body(row["quote"], row["body"])

    wide = assemble(dict(DEFAULT_CRITERIA, service_area=list(c["service_area"]) + ["Cedar Falls"]))
    assert wide["funnel"]["dropped"] == out["funnel"]["dropped"] - 1
    assert wide["funnel"]["to_model"] == out["funnel"]["to_model"] + 1

    from fastapi.testclient import TestClient

    def boom_judge(*_a, **_k):
        raise AssertionError("judge must not run on /run")

    saved_judge = run.__globals__["judge"]
    run.__globals__["judge"] = boom_judge
    try:
        with TestClient(app, raise_server_exceptions=False) as tc:
            boot_r = tc.get("/boot")
            assert boot_r.status_code == 200
            boot_d = boot_r.json()
            assert boot_d["business"]["company"] == BUSINESS["company"]
            assert len(boot_d["requests"]) == 40

            for _ in range(5):
                r = tc.post("/run", json={
                    "criteria": DEFAULT_CRITERIA, "with_model": True,
                })
                assert r.status_code == 200, r.text
                blob = r.text
                assert "Traceback" not in blob
                assert "ClientError" not in blob
                assert "RESOURCE_EXHAUSTED" not in blob
                d = r.json()
                assert d["buckets"] and d["structured"] and d["naive"]
                assert d["source"]["message"]

            r = tc.post("/run", json={
                "criteria": {
                    **DEFAULT_CRITERIA,
                    "service_area": list(c["service_area"]) + ["Cedar Falls"],
                },
            })
            assert r.status_code == 200
            assert r.json()["funnel"]["to_model"] == 12

            saved_assemble = run.__globals__["assemble"]

            def boom_assemble(*_a, **_k):
                raise RuntimeError("secret internals boom")

            run.__globals__["assemble"] = boom_assemble
            try:
                r = tc.post("/run", json={})
                assert r.status_code == 500
                assert "secret internals" not in r.text
                assert "Traceback" not in r.text
                assert "ClientError" not in r.text
                assert r.json().get("message")
            finally:
                run.__globals__["assemble"] = saved_assemble
    finally:
        run.__globals__["judge"] = saved_judge

    print("selftest ok")


if __name__ == "__main__":
    import sys

    if "--selftest" in sys.argv:
        selftest()
        raise SystemExit(0)
    if "--record" in sys.argv:
        record_judgments(force="--force" in sys.argv)
        raise SystemExit(0)

    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))
