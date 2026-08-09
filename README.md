# Contrastive TopK-MIL for Weakly-Supervised MTSAD

Official code for **"Contrastive TopK-MIL with Gradient Attribution for Weakly-Supervised Multivariate Time-Series Anomaly Detection"** (under review).

## Overview

This repository implements a weakly-supervised multiple-instance learning (MIL) framework for multivariate time-series anomaly detection (MTSAD). The method addresses the **weak-representation problem**: existing MIL-MTSAD methods train their encoders only through indirect one-class regularisation, yielding representations that are often outperformed by classical unsupervised baselines such as PCA and autoencoders under honest evaluation protocols.

The framework introduces three components:

1. **Bag-level contrastive learning** (Contribution 1) — uses the natural structure of MIL: normal bags serve as mutual positives while anomaly bags act as negatives, learning discriminative representations without anomaly injection. Under the all-normal training block (the standard case in MTSAD), a uniformity regulariser spreads normal bag representations across the feature space, preventing collapse.
2. **TopK ranking loss** (Contribution 2) — replaces standard bag-level cross-entropy with a ranking objective that focuses on the highest-scoring instances, addressing the anomaly sparsity problem that dilutes standard pooling.
3. **Gradient-based attribution** (exploratory) — variable-level fault localisation via input gradients, validated against SMD's true root-cause labels (`interpretation_label`). We report openly that this exploratory analysis does not surpass random on SMD's multi-variable events — an honest negative result discussed as a limitation rather than a contribution.

## Requirements

- Python >= 3.9
- PyTorch (CPU version is sufficient)
- numpy, scipy, scikit-learn, pyyaml

```bash
pip install torch numpy scipy scikit-learn pyyaml
```

## Datasets

All datasets are publicly available. Download and place under `datasets/`:

| Dataset | Source | Directory |
|---|---|---|
| SMD (28 entities) | [OmniAnomaly](https://github.com/NetManAIOps/OmniAnomaly/tree/master/ServerMachineDataset) | `datasets/SMD/` |
| PSM | [eBay](https://github.com/eBay/tsAdBenchmark) | `datasets/PSM/` |
| MSL / SMAP | [Telemanom](https://github.com/khundman/telemanom) | `datasets/NASA/MSL/`, `datasets/NASA/SMAP/` |
| SWaT | [iTrust SUTD](https://itrust.sutd.edu.sg/itrust-labs_datasets/) | `datasets/SWaT/` |
| WADI | [iTrust SUTD](https://itrust.sutd.edu.sg/itrust-labs_datasets/) | `datasets/WADI/` |
| Exathlon | [Exathlon](https://github.com/exathlon-benchmark/exathlon) | `datasets/Exathlon/` |

SMD also provides `interpretation_label/` (domain-expert root-cause variable labels) from the OmniAnomaly repository, used for the attribution evaluation.

## Repository Structure

```
├── configs/
│   └── default.yaml          # all hyperparameters
├── src/
│   ├── data/                 # dataset loaders + chronological split
│   │   ├── loaders.py
│   │   └── split.py
│   ├── models/               # encoder, multi-hypersphere prototype layer, CMH-MIL
│   │   ├── encoder.py
│   │   ├── hyperspheres.py
│   │   └── cmh_mil.py
│   ├── losses/               # v1 (compact/sep/rank) and v2 (+ contrastive + TopK)
│   │   ├── cmh_loss.py
│   │   └── cmh_loss_v2.py
│   ├── evaluation/           # conformal threshold + event-level metrics
│   │   ├── conformal.py
│   │   └── metrics.py
│   ├── utils/
│   │   └── repro.py          # seeds, hashing, logging
│   ├── baselines_ext/        # TAIL-MIL official baseline
│   ├── run_v2_experiment.py  # main experiment (v2, 4 benchmarks, 5 seeds)
│   ├── baselines.py          # classical baselines (PCA, IF, AE, DeepSVDD, DMSVDD)
│   ├── baselines_sota.py     # USAD baseline
│   └── run_grad_attribution.py  # gradient attribution evaluation
└── results/                  # per-seed JSON logs (5-seed mean reported)
```

## Quick Start

### Run the main experiment (v2, 4 benchmarks, 5 seeds)
```bash
cd src/
PYTHONPATH=. python run_v2_experiment.py
```

### Run classical baselines
```bash
PYTHONPATH=. python baselines.py
PYTHONPATH=. python baselines_sota.py      # USAD
PYTHONPATH=. python baselines_ext/run_tailmil.py
```

### Run gradient attribution evaluation (SMD)
```bash
PYTHONPATH=. python run_grad_attribution.py
```

## Configuration

All hyperparameters are in `configs/default.yaml`. Key settings:

- **Seeds**: {7, 42, 123, 202, 999}
- **Temporal split**: weak-train 35% / validation 10% / calibration 10% / test 45% (four disjoint blocks)
- **Bag**: L=128, instance l=16, stride=64
- **Loss weights** (v2): contrast=0.5, topk=1.0, compact=0.10, sep=0.05, balance=0.10
- **Conformal**: alpha=0.05, threshold computed on calibration block
- **CPU-only** (no GPU required)

## Results

Results are saved to `results/` as JSON files with per-seed logs. The paper reports 5-seed means across 8 benchmarks (SMD, PSM, MSL, SMAP, SWaT, WADI, Exathlon-1, Exathlon-2).

Key findings (reported honestly, including negative results):
- The method **surpasses classical detectors on MSL, SMAP, and Exathlon-1** (AUPRC up to 0.999 on Exathlon-1).
- On **SMD/PSM/SWaT/WADI**, classical reconstruction-based detectors remain stronger on AUPRC.
- **Single-entity SMD results overstate performance**: the 28-entity macro-average AUPRC drops to 0.233 (vs. 0.511 for machine-1-1).
- **Gradient attribution does not surpass random** on SMD's multi-variable events (Hit@3 = 0.438 vs. random 0.566), reported as a limitation.
- **Conformal coverage** holds on MSL/Exathlon but is violated on SMD/PSM/SMAP due to time-series non-exchangeability.

See `results/v2_experiment.json` for the full per-seed numbers.

## Reproducibility

- Fixed random seeds {7, 42, 123, 202, 999}; means reported.
- SHA-256 checksums of data/config/code archived per run (`utils/repro.py`).
- All negative results retained (sphere collapse, non-significant comparisons, attribution failure).
- The four-block temporal split strictly separates model selection, threshold calibration, and final evaluation — addressing the leakage concerns raised by reliable-benchmark studies.

## Citation

```bibtex
@article{zhang2026contrastive,
  title={Contrastive TopK-MIL with Gradient Attribution for Weakly-Supervised Multivariate Time-Series Anomaly Detection},
  author={Zhang, Yao and Li, Baocai and Wang, Yichen},
  journal={International Journal of Machine Learning and Cybernetics},
  year={2026},
  note={Under review}
}
```

## License

MIT License (code). Datasets follow their respective original licenses.
