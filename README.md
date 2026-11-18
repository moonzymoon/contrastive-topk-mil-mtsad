# Bag-Level Contrastive Learning for Weakly-Supervised MIL Anomaly Detection in Multivariate Time Series

Official code for **"Bag-Level Contrastive Learning for Weakly-Supervised Multiple-Instance Anomaly Detection in Multivariate Time Series"**.

## Overview

A weakly-supervised MIL framework for multivariate time-series anomaly detection (MTSAD) that introduces **bag-level contrastive learning**: normal bags serve as mutual positives, anomaly bags as negatives — to our knowledge the first such use of MIL bag labels in the MIL-MTSAD setting.

Key components:
1. **Bag-level contrastive learning** — under all-normal training (the standard MTSAD case), a mean-cosine dispersion regulariser (not Wang & Isola uniformity; it constrains only the representation mean) prevents collapse. Under the track-2 weak-annotation protocol, a joint InfoNCE objective activates.
2. **Quantile radius** — hypersphere radii are data statistics (EMA of per-sphere 90% distance quantile), stored as buffers with no gradient channel, eliminating the unbounded-radius degeneracy of learned radii.
3. **TopK ranking loss** — handles anomaly sparsity; the equal-budget track-2 experiment quantifies its combined contribution with the contrastive objective (+0.124 AUPRC over bag-CE alone on PSM).
4. **Gradient attribution** — under a corrected coordinate alignment, Hit@3 = 0.612 vs. unweighted hypergeometric chance 0.567 (5 of 6 entities above their own chance; single seed).

## Dual-Track Protocol (Plan C)

All experiments use a **dual-track protocol** ensuring no training contact with official-test data (track 1) or equal-budget weak annotation (track 2):

- **Track 1** (main comparison): For datasets with official train/test splits (SMD, PSM, MSL, SMAP, WADI), training/validation/calibration happen entirely **within the official training segment** (70/15/15 split); the held-out test is the official test segment in full. For TSB-AD-M datasets (Exathlon-1/2, Genesis, Daphnet, CATSv2), a single-stream 35/10/10/45 split is used (no official split exists). **SWaT is excluded** — the locally available test file contains only attack-labelled rows, making AUPRC degenerate.
- **Track 2** (equal-budget weak supervision): The first 15% of the official test segment serves as a weak-annotation block, consumed at equal label budget by our method (InfoNCE branch), a bag-CE-only ablation, and TAIL-MIL (official model).

All metrics are computed at **point level** (instance scores max-aggregated onto the timestamp axis). Every method receives the same splits.

## Requirements

- Python >= 3.9
- PyTorch (CPU sufficient)
- numpy, scipy, scikit-learn, pyyaml

## Quick Start

```bash
export CMH_DATA_ROOT=/path/to/datasets

# Main experiment (track 1, all datasets, all baselines, 5 seeds)
cd src/
PYTHONPATH=. python run_full_v3_c.py

# Or single dataset / method
PYTHONPATH=. python run_v3.py --dataset MSL --seeds 7,42,123,202,999 --split_mode plan-c

# Track 2 (equal-budget weak supervision)
PYTHONPATH=. python run_v3.py --dataset SMD --split_mode plan-c-annot
PYTHONPATH=. python run_tailmil_c.py SMD
```

## Repository Structure

```
├── configs/default.yaml       # all hyperparameters
├── src/
│   ├── data/
│   │   ├── protocol.py        # dual-track protocol (official_split, weak_annotation_split,
│   │   │                      #   make_bags_v2 n=8, point_scores, Scaler)
│   │   ├── loaders.py         # returns (X, Y, n_official_train) — raw values
│   │   └── tsb_ad_loader.py   # TSB-AD-M loader (single-stream)
│   ├── models/
│   │   ├── encoder.py         # TCN + descriptors (var_attn removed)
│   │   ├── hyperspheres.py    # quantile radius (EMA, buffer, no gradient)
│   │   └── cmh_mil.py         # max pooling, any-point semantics
│   ├── losses/
│   │   └── cmh_loss_v3.py     # InfoNCE (joint), dispersion (mean-cos), TopK
│   ├── evaluation/
│   │   ├── metrics.py         # Event-F1 (maximum-cardinality matching), AUPRC, FAR
│   │   └── conformal.py
│   ├── baselines_v3.py        # PCA, IF, AE, DeepSVDD(bias-free), USAD(faithful), MS-SVDD-style
│   ├── baseline_at_faithful.py # Anomaly Transformer (official minimax objective)
│   ├── run_tailmil_c.py       # TAIL-MIL (official model, equal-budget track-2)
│   ├── run_full_v3_c.py       # main experiment runner (dual-track, resumable)
│   ├── run_grad_attribution_c.py # gradient attribution (corrected alignment)
│   └── make_tables_c.py       # programmatic table generation (main/track2/q/ef1)
├── results/                   # per-seed JSON logs (all experiments)
└── paper/                     # LaTeX source + programmatically generated tables
```

## Key Results (under the unified dual-track protocol)

| Dataset | Ours | Best baseline |
|---------|------|---------------|
| SMAP | **0.147** | AT 0.117 |
| WADI | **0.158** | AE 0.154 |
| Exathlon-1 | 0.993 | IF 0.997 |
| SMD | 0.490 | PCA 0.637 |
| PSM | 0.437 | AT 0.511 |

- vs. USAD (faithful re-implementation): **9/10 wins**
- vs. Anomaly Transformer (faithful): **8/10 wins**
- Track-2 equal-budget: **5/5 wins over TAIL-MIL**; +0.124 AUPRC over bag-CE on PSM
- Attribution: Hit@3 = 0.612 vs. chance 0.567 (5/6 entities above own chance)
- SMD-28 macro: 0.426 (entity SD 0.264, seed SD 0.010)

## Honest Limitations

- Reconstruction methods (PCA, AE) remain stronger on SMD, PSM, Daphnet, CATSv2
- Event-F1 on SMAP trails AT (0.081 vs. 0.307) despite better AUPRC
- Quantile threshold is an empirical device; coverage violated on drifting datasets
- Silhouette-based applicability scores are exploratory, not validated
- SWaT excluded (degenerate local test data)

## Reproducibility

- Fixed seeds {7, 42, 123, 202, 999} for all methods
- All results generated programmatically from archived per-seed JSONs
- All tables in the paper generated by `make_tables_c.py` (no hand-typed numbers)
- Gradient attribution uses single seed (7); multi-seed replication is future work

## License

MIT License (code). Datasets follow their respective original licenses.
