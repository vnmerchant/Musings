# Daily Experiment: OpenMotion MOTION.md Prototype (2026-11-25)

## Source
Liked tweet: https://x.com/tuturetom/status/2122633802488897679  
Author: Tom Huang (@tuturetom) — OpenMotion open-source announcement.  
Claim: Full-stack open-source Claude Motion + CapCut alternative with 300+ MOTION.md specs reverse-engineered from top brand films, 95%+ fidelity, works with any agent (no SOTA model required). Repo: https://github.com/nexu-io/openmotion (Apache-2.0).

## Context
OpenMotion builds on HyperFrames (https://github.com/heygen-com/hyperframes, HeyGen, Apache-2.0) for HTML/CSS/GSAP-based video rendering. Core idea is MOTION.md — a structured Markdown that captures shot order, timing, transitions, and invariants so agents can remix while preserving motion language. Docs: https://github.com/nexu-io/openmotion/blob/main/docs/motion-md.md

## Experiment
Lightweight MOTION.md parser + HTML generator comparing structured timing vs naive fixed-delay animation.

### What was built
- `prototype.py`: Parses sample MOTION.md (from official docs example), extracts replaceable text + timing beats, emits self-contained HTML with CSS keyframes.
- Generated `prototype_structured.html` (uses exact 0–0.45s / 0.45–2.5s / 2.5–6s / 6–8s) and `prototype_naive.html` (fixed delays).
- Sample `sample.MOTION.md`.

### Results
Parsed beats correctly:
- 0–0.45s card entrance
- 0.45–2.5s portal reveal
- 2.5–6s dimensional phrase
- 6–8s exit

Structured version preserves the precise reveal rhythm from the original portal example. Naive version collapses the portal and phrase holds, losing the intentional pacing that MOTION.md is designed to protect.

Screenshots captured at t=0.5/2/4/7s for both modes (see frame_*.png). Due to local server + CSS animation timing in headless capture, frames are low-resolution proxies but demonstrate the pipeline.

### Key insight
A few dozen lines of parser + generator are enough to turn a MOTION.md into a runnable, brand-swappable animation. Full OpenMotion adds the Studio editor, MCP, 300+ templates, brand kits, and agent skills on top of this foundation (HyperFrames).

## Improvement ideas
1. Deterministic seek: expose a `currentTime` setter so screenshots can be taken at exact beat boundaries without wall-clock waiting (HyperFrames already supports this via data attributes).
2. Schema validation: add a tiny JSON Schema or Zod-like check for MOTION.md sections so agents fail fast on missing invariants.
3. Cross-engine export: emit both CSS keyframes and a GSAP timeline from the same parsed beats (GSAP is already a HyperFrames adapter).
4. Fidelity metric: compute structural similarity (SSIM) or optical-flow distance between reference and remix at matching timestamps — the docs already emphasize paired evidence.

## Sources
- Original tweet thread: https://x.com/tuturetom/status/2122633802488897679
- Repo: https://github.com/nexu-io/openmotion
- MOTION.md docs: https://github.com/nexu-io/openmotion/blob/main/docs/motion-md.md
- HyperFrames: https://github.com/heygen-com/hyperframes
- Release notes (v0.2.1): https://github.com/nexu-io/openmotion/releases/tag/v0.2.1

## Files
- prototype.py
- prototype_structured.html / prototype_naive.html
- sample.MOTION.md
- frame_*.png (visual evidence)
