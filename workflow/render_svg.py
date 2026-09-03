#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///
"""Рисует схему воркфлоу из его же экспорта - чтобы картинка на витрине не расходилась с тем,
что реально работает. Запуск: ./render_svg.py lead-qualification.json diagram.svg"""
import json
import sys
from pathlib import Path

NODE_W, NODE_H = 190, 56
PAD = 90

PALETTE = {  # витрина Ledger: сталь - поток, янтарь - модель, зелёный - выход клиенту
    "trigger": ("#1f3a5f", "#eaf0f7"),
    "model": ("#b4762b", "#fbf1e2"),
    "exit": ("#2f6b45", "#e9f2ec"),
    "plain": ("#c9cfd8", "#ffffff"),
}

MODEL_NODES = {"AI: intent and urgency", "Quote must be in the text"}
EXIT_NODES = {"Telegram", "WhatsApp", "Slack", "Email digest",
              "Create hot lead in CRM", "Needs a human", "Polite reply"}


def kind(node):
    if node["type"].endswith("webhook"):
        return "trigger"
    if node["name"] in MODEL_NODES:
        return "model"
    if node["name"] in EXIT_NODES:
        return "exit"
    return "plain"


def subtitle(node):
    """Короткая подпись: что нода делает, а не как называется её тип."""
    t = node["type"].rsplit(".", 1)[-1]
    return {"webhook": "incoming request", "set": "fields", "removeDuplicates": "seen before?",
            "switch": "branches", "httpRequest": "HTTP", "code": "code", "if": "condition",
            "merge": "collect", "noOp": "handoff"}.get(t, t)


def output_labels(node):
    """Имена веток switch/if - они и есть человеческие названия правил."""
    if node["type"].endswith(".switch"):
        vals = node["parameters"].get("rules", {}).get("values", [])
        labels = [v.get("outputKey", "") for v in vals]
        fb = node["parameters"].get("options", {}).get("renameFallbackOutput")
        if fb:
            labels.append(fb)
        return labels
    if node["type"].endswith(".if"):
        return ["yes", "no"]
    return []


def esc(s):
    return (s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;"))


def render(wf):
    nodes = {n["name"]: n for n in wf["nodes"]}
    xs = [n["position"][0] for n in wf["nodes"]]
    ys = [n["position"][1] for n in wf["nodes"]]
    minx, miny = min(xs) - PAD, min(ys) - PAD
    width = max(xs) - minx + NODE_W + PAD
    height = max(ys) - miny + NODE_H + PAD

    def box(n):
        x = n["position"][0] - minx
        y = n["position"][1] - miny
        return x, y

    out = [f'<svg viewBox="0 0 {width} {height}" xmlns="http://www.w3.org/2000/svg" '
           f'font-family="Inter,system-ui,sans-serif" role="img" aria-label="Lead qualification workflow">',
           '<defs><marker id="arrow" markerWidth="9" markerHeight="9" refX="8" refY="4.5" '
           'orient="auto"><path d="M0,0 L9,4.5 L0,9 z" fill="#8a93a0"/></marker></defs>',
           f'<rect width="{width}" height="{height}" fill="#f7f5f0"/>']

    # Связи рисуются первыми, чтобы ноды лежали поверх линий.
    for src, conn in wf["connections"].items():
        if src not in nodes:
            continue
        sx, sy = box(nodes[src])
        labels = output_labels(nodes[src])
        for i, branch in enumerate(conn.get("main", [])):
            for link in branch or []:
                if link["node"] not in nodes:
                    continue
                tx, ty = box(nodes[link["node"]])
                x1, y1 = sx + NODE_W, sy + NODE_H / 2
                x2, y2 = tx, ty + NODE_H / 2
                mid = (x1 + x2) / 2
                out.append(f'<path d="M{x1},{y1} C{mid},{y1} {mid},{y2} {x2},{y2}" fill="none" '
                           f'stroke="#8a93a0" stroke-width="1.5" marker-end="url(#arrow)"/>')
                if i < len(labels) and labels[i]:
                    # Ветки выходят из одной точки: разносим подписи по длине, иначе слипаются.
                    frac = 0.22 + min(i, 5) * 0.13
                    lx = x1 + (x2 - x1) * frac
                    ly = y1 + (y2 - y1) * frac - 8
                    label = labels[i]
                    out.append(f'<rect x="{lx - len(label) * 3.4 - 5}" y="{ly - 12}" '
                               f'width="{len(label) * 6.8 + 10}" height="17" rx="4" fill="#f7f5f0"/>')
                    out.append(f'<text x="{lx}" y="{ly}" text-anchor="middle" '
                               f'font-size="12.5" fill="#5b6570">{esc(label)}</text>')

    for n in wf["nodes"]:
        x, y = box(n)
        stroke, fill = PALETTE[kind(n)]
        w = 2 if kind(n) in ("trigger", "model") else 1.2
        out.append(f'<rect x="{x}" y="{y}" width="{NODE_W}" height="{NODE_H}" rx="8" '
                   f'fill="{fill}" stroke="{stroke}" stroke-width="{w}"/>')
        out.append(f'<text x="{x + 14}" y="{y + 24}" font-size="13.5" font-weight="600" '
                   f'fill="#1b1f23">{esc(n["name"][:26])}</text>')
        out.append(f'<text x="{x + 14}" y="{y + 42}" font-size="12" fill="#5b6570">'
                   f'{esc(subtitle(n))}</text>')

    out.append("</svg>")
    return "\n".join(out)


if __name__ == "__main__":
    src = Path(sys.argv[1] if len(sys.argv) > 1 else "lead-qualification.json")
    dst = Path(sys.argv[2] if len(sys.argv) > 2 else "diagram.svg")
    wf = json.loads(src.read_text())
    svg = render(wf)
    dst.write_text(svg)
    print(f"{dst} - {len(wf['nodes'])} nodes, {len(svg)} bytes")
