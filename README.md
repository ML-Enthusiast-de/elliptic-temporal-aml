# elliptic-temporal-aml

# Elliptic Temporal AML — Illicit Transaction Detection on a Crypto Transaction Graph

Research-grade pipeline for detecting illicit cryptocurrency transactions using the Elliptic dataset.
The project progresses from strong tabular baselines to temporal graph models and (optional) self-supervised pretraining.

> **Disclaimer:** For research/education only. Not financial advice. Not a compliance product.

---

## Why this project
Financial crime is rarely i.i.d.: behavior unfolds over time and across networks.
Crypto transaction graphs contain strong relational/temporal signals (bursts, neighborhood effects, flows, motifs).
This repo demonstrates:
- rigorous time-aware validation (no leakage)
- graph + temporal modeling
- practical decisioning (calibration, thresholds, cost curves)
- reproducible, reviewable ML engineering

---

## Dataset
We use the **Elliptic Bitcoin Dataset** (public research dataset):
- nodes = transactions
- edges = money flow between transactions
- labels = `licit` / `illicit` / `unknown`
- features = anonymized transaction features + aggregated neighborhood features
- time steps = discrete “time slices” for temporal evaluation

**You must download the dataset yourself** (license/terms).
Place raw files under: `data/raw/elliptic/`
