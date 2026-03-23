# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This repository contains a scientific paper and supporting computational scripts on **q-thermodynamics of high-entropy alloys (HEAs)**. The paper proposes that the four "core effects" of HEAs are projections of a single mathematical object: the curvature of the q-exponential family (CES partition function) on the composition simplex. It proposes novel HEA compositions for spaceflight applications and high-entropy ceramics for thermal protection systems.

**Author**: Jon Smirl (independent researcher)

## Repository Structure

- `HEA_CES_Predictions.tex` — Main paper (LaTeX, ~2800 lines). Contains the full mathematical framework, predictions, and proposed alloy/ceramic compositions.
- `HEA_CES_Predictions.pdf` — Compiled paper.
- `scripts/` — Python scripts for quantitative analysis:
  - `heo6_deep_dive.py` — Deep-dive evaluation of HEO-6 pyrochlore foam tile; verifies claims and explores physics.
  - `heo6_alternatives.py` — Models alternative heat shield architectures (nanofiber, aerogel, dual-sublattice designs).

## Build Commands

Compile the paper:
```bash
pdflatex HEA_CES_Predictions.tex
```

Run analysis scripts:
```bash
python3 scripts/heo6_deep_dive.py
python3 scripts/heo6_alternatives.py
```

## Key Domain Concepts

- **CES partition function**: `Z_q = Σ a_j x_j^q` — the central generating function. The parameter `q` (complementarity) controls deviation from rule-of-mixtures (`q=1`).
- **Escort distribution**: `P_j = a_j x_j^q / Z_q` — effective property share of element j, distinct from atomic fraction.
- **Curvature**: `K = (1-q)(1-H)` where `H = Σ a_j²` is the Herfindahl index. Unifies phase stability, lattice distortion, diffusion modification, and the cocktail effect.
- **Pyrochlore/fluorite structure**: A₂B₂O₇ ceramics with rare-earth A-site and Zr/Hf B-site elements.
- Scripts use Shannon ionic radii, Klemens phonon scattering model, and radiative transport through porous ceramics (κ_rad ∝ T³·d_pore).

## Python Dependencies

Scripts use only `numpy` and Python standard library (`dataclasses`, `json`, `typing`). No virtual environment or package manager is configured.
