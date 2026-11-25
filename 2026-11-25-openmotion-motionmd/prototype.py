#!/usr/bin/env python3
"""
Minimal MOTION.md interpreter prototype for OpenMotion-style motion specs.
Parses a structured timing document and generates timed HTML animation.
Compares structured MOTION.md approach vs naive fixed-delay animation.
"""

import json
import re
from pathlib import Path

SAMPLE_MOTION_MD = """
# Portal launch — MOTION_DESIGN.md
Version: 0.1.0

## Keep
Light card → expanding portal → dimensional phrase → white exit.
Keep the camera path, reveal timing, deterministic seed and easing.

## Replace
Title: OPENMOTION
Dimensional phrase: TELL YOUR STORY
Subtitle: YOUR STORY. IN MOTION.

## Timing
0–0.45 s: card entrance.
0.45–2.5 s: portal reveal.
2.5–6 s: dimensional phrase.
6–8 s: push through and white exit.

## Review
Compare the card, portal handoff and phrase hold at matching times.
"""

def parse_motion_md(md_text: str) -> dict:
    """Very lightweight parser for timing and replaceable fields."""
    result = {"title": "DEMO", "phrase": "IN MOTION", "subtitle": "STRUCTURED", "beats": []}
    
    title_m = re.search(r"Title:\s*(.+)", md_text)
    if title_m:
        result["title"] = title_m.group(1).strip()
    phrase_m = re.search(r"Dimensional phrase:\s*(.+)", md_text)
    if phrase_m:
        result["phrase"] = phrase_m.group(1).strip()
    sub_m = re.search(r"Subtitle:\s*(.+)", md_text)
    if sub_m:
        result["subtitle"] = sub_m.group(1).strip()
    
    timing_section = re.search(r"## Timing\n(.*?)(?=\n##|\Z)", md_text, re.DOTALL)
    if timing_section:
        for line in timing_section.group(1).strip().split("\n"):
            m = re.match(r"([\d.]+)[–-]([\d.]+)\s*s:\s*(.+)", line)
            if m:
                start, end, label = float(m.group(1)), float(m.group(2)), m.group(3).strip()
                result["beats"].append({"start": start, "end": end, "label": label, "duration": end - start})
    return result

def generate_html(spec: dict, mode: str = "structured") -> str:
    title = spec["title"]
    phrase = spec["phrase"]
    subtitle = spec["subtitle"]
    
    if mode == "structured":
        beats = spec["beats"]
        css = f"""
        .card {{ animation: cardIn {beats[0]['duration']}s ease-out forwards; }}
        .portal {{ animation: portalReveal {beats[1]['duration']}s {beats[1]['start']}s ease-in-out forwards; opacity: 0; }}
        .phrase {{ animation: phraseIn {beats[2]['duration']}s {beats[2]['start']}s cubic-bezier(0.22, 1, 0.36, 1) forwards; opacity: 0; }}
        .exit {{ animation: exitOut {beats[3]['duration']}s {beats[3]['start']}s ease-in forwards; }}
        """
    else:
        css = """
        .card { animation: cardIn 1.5s ease-out forwards; }
        .portal { animation: portalReveal 2s 1.5s ease-in-out forwards; opacity: 0; }
        .phrase { animation: phraseIn 3s 3.5s ease forwards; opacity: 0; }
        .exit { animation: exitOut 1s 6.5s ease-in forwards; }
        """
    
    html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>MOTION.md Prototype — {mode}</title>
<style>
body {{ margin:0; background:#0a0a0a; color:#fff; font-family: system-ui, sans-serif; overflow:hidden; }}
.stage {{ width:1280px; height:720px; position:relative; margin:20px auto; background:linear-gradient(135deg,#1a1a2e,#16213e); border-radius:12px; overflow:hidden; }}
.card {{ position:absolute; left:50%; top:40%; transform:translate(-50%,-50%) scale(0.6); background:#fff; color:#111; padding:40px 80px; border-radius:16px; font-size:48px; font-weight:700; box-shadow:0 20px 60px rgba(0,0,0,.4); }}
.portal {{ position:absolute; inset:0; background:radial-gradient(circle at center, transparent 20%, #000 70%); mix-blend-mode:multiply; }}
.phrase {{ position:absolute; left:50%; top:55%; transform:translate(-50%,-50%); font-size:72px; font-weight:800; letter-spacing:-1px; text-align:center; text-shadow:0 4px 20px rgba(0,0,0,.6); }}
.subtitle {{ position:absolute; left:50%; top:70%; transform:translateX(-50%); font-size:24px; opacity:0.7; }}
.exit {{ position:absolute; inset:0; background:#fff; }}
@keyframes cardIn {{ from {{ transform:translate(-50%,-50%) scale(0.4); opacity:0; }} to {{ transform:translate(-50%,-50%) scale(1); opacity:1; }} }}
@keyframes portalReveal {{ from {{ opacity:0; transform:scale(0.8); }} to {{ opacity:1; transform:scale(1); }} }}
@keyframes phraseIn {{ from {{ opacity:0; transform:translate(-50%,-50%) translateY(40px); }} to {{ opacity:1; transform:translate(-50%,-50%) translateY(0); }} }}
@keyframes exitOut {{ from {{ opacity:0; }} to {{ opacity:1; }} }}
{css}
</style>
</head>
<body>
<div class="stage">
  <div class="card">{title}</div>
  <div class="portal"></div>
  <div class="phrase">{phrase}<div class="subtitle">{subtitle}</div></div>
  <div class="exit"></div>
</div>
<script>
window.getTime = () => performance.now() / 1000;
console.log("MOTION prototype ready:", "{mode}");
</script>
</body>
</html>"""
    return html

def main():
    spec = parse_motion_md(SAMPLE_MOTION_MD)
    print("Parsed spec:", json.dumps(spec, indent=2))
    
    out_dir = Path("/workspace/artifacts/2026-11-25-openmotion-motionmd")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    for mode in ["structured", "naive"]:
        html = generate_html(spec, mode)
        (out_dir / f"prototype_{mode}.html").write_text(html)
        print(f"Wrote prototype_{mode}.html")
    
    (out_dir / "sample.MOTION.md").write_text(SAMPLE_MOTION_MD)
    print("Wrote sample.MOTION.md")

if __name__ == "__main__":
    main()
