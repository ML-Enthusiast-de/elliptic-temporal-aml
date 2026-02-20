# Elliptic Temporal AML

Illicit transaction detection on the Elliptic Bitcoin graph using time-aware tabular and graph ML pipelines.

> Research/education only. Not financial advice. Not a compliance product.

## What this repo does
- Builds leakage-aware train/validation/test splits over time.
- Trains strong tabular baselines.
- Trains graph models with temporal validation and threshold tuning.
- Tracks stability by `time_step` (PR-AUC drift plots/CSVs).

## Dataset
This project uses the Elliptic Bitcoin dataset.

Expected raw files in `data/raw/elliptic/`:
- `elliptic_txs_features.csv`
- `elliptic_txs_edgelist.csv`
- `elliptic_txs_classes.csv`

Processed artifacts used by training scripts:
- `data/processed/nodes.parquet`
- `data/processed/edges.parquet`

Current processed schema (observed):
- `nodes.parquet`: ~203k rows, columns include `txId`, `time_step`, numeric features, `y`, `split`
- `edges.parquet`: columns `src`, `dst`

## Project structure
```text
elliptic-temporal-aml/
  data/
    raw/elliptic/
    processed/
  models/
    pretrain_mfm_pyg.py
    train_baseline.py
    train_graphsage_pyg.py
    train_graphsage_pyg_temporal_causal.py
    train_tgat_pyg.py
  reports/
    metrics/
    figures/
  checkpoints/
  README.md
```

## Model scripts
`models/train_baseline.py`
- Tabular baselines and ablation outputs.

`models/pretrain_mfm_pyg.py`
- Self-supervised masked feature modeling (MFM) pretraining with GraphSAGE encoder.
- Saves encoder checkpoint under `checkpoints/`.

`models/train_graphsage_pyg.py`
- Supervised GraphSAGE with temporal validation windows.
- Supports threshold modes: `target_precision`, `topk`, `alert_rate`.

`models/train_graphsage_pyg_temporal_causal.py`
- GraphSAGE variant with optional causal-edge filtering and optional `time_step` feature.
- Writes isolated temporal-causal metric/figure files.

`models/train_tgat_pyg.py`
- Transformer-based graph model (`TransformerConv`) with temporal edge features:
  - `dt_norm`, `log_dt_norm`, `same_step`, `src_t_norm`, `dst_t_norm`
- Uses the same split/eval protocol so comparisons are direct.

## Quick start
Run with the environment that has PyG deps (`torch-sparse`/`pyg-lib`) installed.

GraphSAGE (temporal-aware split):
```bash
python models/train_graphsage_pyg.py \
  --feature_set local \
  --val_steps 3 --val_windows 3 \
  --train_recent_steps 12 \
  --threshold_mode target_precision --target_precision 0.8
```

GraphSAGE temporal-causal:
```bash
python models/train_graphsage_pyg_temporal_causal.py \
  --feature_set local \
  --val_steps 3 --val_windows 3 \
  --train_recent_steps 12 \
  --causal_edges --add_time_feature
```

TGAT:
```bash
python models/train_tgat_pyg.py \
  --feature_set local \
  --val_steps 3 --val_windows 3 \
  --train_recent_steps 12 \
  --threshold_mode target_precision --target_precision 0.8
```

## Outputs
Main outputs are written to:
- `reports/metrics/*.json` (run summary metrics)
- `reports/metrics/*by_time_step*.csv` (per-time-step PR-AUC)
- `reports/figures/*.png` (PR curves and drift plots)

Examples:
- `reports/metrics/baseline_graphsage_local.json`
- `reports/metrics/baseline_graphsage_temporal_causal_local.json`
- `reports/metrics/baseline_tgat_local.json`

## Evaluation philosophy
- Validate on recent temporal windows, not random splits.
- Tune thresholds on validation, then evaluate fixed threshold on test.
- Report both aggregate metrics (`PR-AUC`, `ROC-AUC`) and time-sliced stability.
