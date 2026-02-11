#!/usr/bin/env python3
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


# -----------------------------
# Utilities
# -----------------------------
def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def detect_feature_cols(df: pd.DataFrame) -> List[str]:
    """Pick numeric feature columns, excluding identifiers/targets/meta."""
    exclude = {"txId", "time_step", "y", "split"}
    # keep only numeric
    num_cols = [c for c in df.columns if c not in exclude and pd.api.types.is_numeric_dtype(df[c])]
    if not num_cols:
        raise ValueError("No numeric feature columns detected. Check nodes.parquet schema.")
    return num_cols


def to_binary_y(y: np.ndarray) -> np.ndarray:
    """
    Map Elliptic y to binary:
      y=1 -> illicit (positive)
      y=0 -> licit  (negative)
    """
    y = np.asarray(y, dtype=int)
    if not np.isin(y, [0, 1]).all():
        raise ValueError("Expected y in {0,1} for baseline. Filter unknowns first.")
    return y


def pick_threshold_by_best_f1(y_true: np.ndarray, y_prob: np.ndarray) -> Dict[str, float]:
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_prob)
    # thresholds has length n-1; precisions/recalls length n
    f1s = (2 * precisions * recalls) / (precisions + recalls + 1e-12)
    best_idx = int(np.nanargmax(f1s))
    # If best_idx points to last element, threshold isn't defined; clamp
    thr = float(thresholds[min(best_idx, len(thresholds) - 1)]) if len(thresholds) > 0 else 0.5
    return {"threshold": thr, "precision": float(precisions[best_idx]), "recall": float(recalls[best_idx]), "f1": float(f1s[best_idx])}


def pick_threshold_for_target_precision(
    y_true: np.ndarray, y_prob: np.ndarray, target_precision: float = 0.80
) -> Dict[str, float]:
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_prob)
    # We want the highest recall among points with precision >= target
    mask = precisions >= target_precision
    if not mask.any():
        # Can't reach target precision; return max precision point
        best_idx = int(np.nanargmax(precisions))
        thr = float(thresholds[min(best_idx, len(thresholds) - 1)]) if len(thresholds) > 0 else 0.5
        return {
            "threshold": thr,
            "precision": float(precisions[best_idx]),
            "recall": float(recalls[best_idx]),
            "note": f"Target precision {target_precision:.2f} not reached; using max precision point.",
        }

    # Among valid points, pick max recall
    candidates = np.where(mask)[0]
    best_idx = int(candidates[np.nanargmax(recalls[candidates])])
    thr = float(thresholds[min(best_idx, len(thresholds) - 1)]) if len(thresholds) > 0 else 0.5
    return {"threshold": thr, "precision": float(precisions[best_idx]), "recall": float(recalls[best_idx])}


def evaluate_predictions(y_true: np.ndarray, y_prob: np.ndarray, threshold: float) -> Dict[str, float]:
    y_hat = (y_prob >= threshold).astype(int)
    p = precision_score(y_true, y_hat, zero_division=0)
    r = recall_score(y_true, y_hat, zero_division=0)
    f1 = f1_score(y_true, y_hat, zero_division=0)
    tn, fp, fn, tp = confusion_matrix(y_true, y_hat).ravel()
    return {
        "threshold": float(threshold),
        "precision": float(p),
        "recall": float(r),
        "f1": float(f1),
        "tp": int(tp),
        "fp": int(fp),
        "tn": int(tn),
        "fn": int(fn),
    }


def plot_pr_curve(y_true: np.ndarray, y_prob: np.ndarray, out_path: Path, title: str) -> None:
    ap = average_precision_score(y_true, y_prob)
    precisions, recalls, _ = precision_recall_curve(y_true, y_prob)
    plt.figure()
    plt.plot(recalls, precisions)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"{title} | PR-AUC={ap:.4f}")
    plt.grid(True, alpha=0.3)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


@dataclass
class RunResult:
    model_name: str
    n_train: int
    n_test: int
    n_features: int
    pr_auc: float
    roc_auc: float
    best_f1: Dict[str, float]
    target_precision_080: Dict[str, float]
    eval_at_best_f1: Dict[str, float]
    eval_at_p80: Dict[str, float]


# -----------------------------
# Main
# -----------------------------
def main(target_precision: float = 0.80) -> None:
    root = repo_root()
    nodes_path = root / "data" / "processed" / "nodes.parquet"

    reports_metrics = root / "reports" / "metrics"
    reports_figures = root / "reports" / "figures"
    ensure_dir(reports_metrics)
    ensure_dir(reports_figures)

    df = pd.read_parquet(nodes_path)

    # Use labeled only for supervised baseline
    df = df[df["split"].isin(["train", "test"])].copy()
    df = df[df["y"].isin([0, 1])].copy()

    feat_cols = detect_feature_cols(df)

    train_df = df[df["split"] == "train"]
    test_df = df[df["split"] == "test"]

    X_train = train_df[feat_cols].to_numpy()
    y_train = to_binary_y(train_df["y"].to_numpy())

    X_test = test_df[feat_cols].to_numpy()
    y_test = to_binary_y(test_df["y"].to_numpy())

    print(f"Loaded nodes: {len(df)} labeled (train+test)")
    print(f"Train: {len(train_df)} | Test: {len(test_df)} | Features: {len(feat_cols)}")
    print(f"Positive rate train (illicit): {y_train.mean():.4f} | test: {y_test.mean():.4f}")

    # -----------------------------
    # Model 1: Logistic Regression (strong simple baseline)
    # -----------------------------
    logreg = Pipeline(
        steps=[
            ("scaler", StandardScaler(with_mean=True, with_std=True)),
            ("clf", LogisticRegression(max_iter=2000, class_weight="balanced", n_jobs=-1)),
        ]
    )
    logreg.fit(X_train, y_train)
    y_prob_lr = logreg.predict_proba(X_test)[:, 1]

    pr_auc_lr = float(average_precision_score(y_test, y_prob_lr))
    roc_auc_lr = float(roc_auc_score(y_test, y_prob_lr))

    best_f1_lr = pick_threshold_by_best_f1(y_test, y_prob_lr)
    p80_lr = pick_threshold_for_target_precision(y_test, y_prob_lr, target_precision=target_precision)

    eval_best_lr = evaluate_predictions(y_test, y_prob_lr, best_f1_lr["threshold"])
    eval_p80_lr = evaluate_predictions(y_test, y_prob_lr, p80_lr["threshold"])

    res_lr = RunResult(
        model_name="logreg_balanced",
        n_train=int(len(train_df)),
        n_test=int(len(test_df)),
        n_features=int(len(feat_cols)),
        pr_auc=pr_auc_lr,
        roc_auc=roc_auc_lr,
        best_f1=best_f1_lr,
        target_precision_080=p80_lr,
        eval_at_best_f1=eval_best_lr,
        eval_at_p80=eval_p80_lr,
    )

    plot_pr_curve(
        y_test,
        y_prob_lr,
        reports_figures / "pr_curve_logreg.png",
        title="LogReg (balanced)",
    )

    with open(reports_metrics / "baseline_logreg.json", "w", encoding="utf-8") as f:
        json.dump(asdict(res_lr), f, indent=2)

    print(f"[LogReg] PR-AUC={pr_auc_lr:.4f} ROC-AUC={roc_auc_lr:.4f}")
    print(f"[LogReg] Best-F1 @ thr={best_f1_lr['threshold']:.4f} -> P={best_f1_lr['precision']:.3f} R={best_f1_lr['recall']:.3f} F1={best_f1_lr['f1']:.3f}")
    if "note" in p80_lr:
        print(f"[LogReg] P@target note: {p80_lr['note']}")
    print(f"[LogReg] P@{target_precision:.2f}thr={p80_lr['threshold']:.4f} -> P={p80_lr['precision']:.3f} R={p80_lr['recall']:.3f}")

    # -----------------------------
    # Model 2: HistGradientBoosting (handles nonlinearity well)
    # -----------------------------
    # Use class_weight-like behavior by adjusting sample weights
    # (HGB doesn't support class_weight directly in older versions)
    pos = (y_train == 1).sum()
    neg = (y_train == 0).sum()
    w_pos = neg / max(pos, 1)
    sample_weight = np.where(y_train == 1, w_pos, 1.0)

    hgb = HistGradientBoostingClassifier(
        max_depth=6,
        learning_rate=0.05,
        max_iter=400,
        l2_regularization=1.0,
        random_state=42,
    )
    hgb.fit(X_train, y_train, sample_weight=sample_weight)
    y_prob_hgb = hgb.predict_proba(X_test)[:, 1]

    pr_auc_hgb = float(average_precision_score(y_test, y_prob_hgb))
    roc_auc_hgb = float(roc_auc_score(y_test, y_prob_hgb))

    best_f1_hgb = pick_threshold_by_best_f1(y_test, y_prob_hgb)
    p80_hgb = pick_threshold_for_target_precision(y_test, y_prob_hgb, target_precision=target_precision)

    eval_best_hgb = evaluate_predictions(y_test, y_prob_hgb, best_f1_hgb["threshold"])
    eval_p80_hgb = evaluate_predictions(y_test, y_prob_hgb, p80_hgb["threshold"])

    res_hgb = RunResult(
        model_name="histgb_weighted",
        n_train=int(len(train_df)),
        n_test=int(len(test_df)),
        n_features=int(len(feat_cols)),
        pr_auc=pr_auc_hgb,
        roc_auc=roc_auc_hgb,
        best_f1=best_f1_hgb,
        target_precision_080=p80_hgb,
        eval_at_best_f1=eval_best_hgb,
        eval_at_p80=eval_p80_hgb,
    )

    plot_pr_curve(
        y_test,
        y_prob_hgb,
        reports_figures / "pr_curve_histgb.png",
        title="HistGB (weighted)",
    )

    with open(reports_metrics / "baseline_histgb.json", "w", encoding="utf-8") as f:
        json.dump(asdict(res_hgb), f, indent=2)

    print(f"[HistGB] PR-AUC={pr_auc_hgb:.4f} ROC-AUC={roc_auc_hgb:.4f}")
    print(f"[HistGB] Best-F1 @ thr={best_f1_hgb['threshold']:.4f} -> P={best_f1_hgb['precision']:.3f} R={best_f1_hgb['recall']:.3f} F1={best_f1_hgb['f1']:.3f}")
    if "note" in p80_hgb:
        print(f"[HistGB] P@target note: {p80_hgb['note']}")
    print(f"[HistGB] P@{target_precision:.2f}thr={p80_hgb['threshold']:.4f} -> P={p80_hgb['precision']:.3f} R={p80_hgb['recall']:.3f}")

    print("\n✅ Saved:")
    print(" - reports/metrics/baseline_logreg.json")
    print(" - reports/metrics/baseline_histgb.json")
    print(" - reports/figures/pr_curve_logreg.png")
    print(" - reports/figures/pr_curve_histgb.png")


if __name__ == "__main__":
    main()
