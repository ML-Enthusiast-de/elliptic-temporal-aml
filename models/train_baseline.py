#!/usr/bin/env python3
from __future__ import annotations

import argparse
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
from xgboost import XGBClassifier



# -----------------------------
# Repo root detection (robust)
# -----------------------------
def find_repo_root(start: Path) -> Path:
    for p in [start] + list(start.parents):
        if (p / ".git").exists():
            return p
    # fallback: assume file is under repo root somewhere
    return start.parents[1]


# -----------------------------
# Feature selection
# -----------------------------
def detect_feature_cols(df: pd.DataFrame) -> List[str]:
    """Pick numeric feature columns, excluding identifiers/targets/meta."""
    exclude = {"txId", "time_step", "y", "split"}
    cols = []
    for c in df.columns:
        if c in exclude:
            continue
        if pd.api.types.is_numeric_dtype(df[c]):
            cols.append(c)
    if not cols:
        raise ValueError("No numeric feature columns detected. Check nodes.parquet schema.")
    return cols


def ordered_feature_cols(df: pd.DataFrame, feat_cols: List[str]) -> List[str]:
    """
    Ensure stable feature ordering.
    If column names are ints (or numeric strings), sort by numeric value.
    Otherwise keep DataFrame column order.
    """
    # Preserve df order by default
    df_order = [c for c in df.columns if c in feat_cols]

    # If all are int-like, sort numerically for safety
    def to_int(x):
        if isinstance(x, (int, np.integer)):
            return int(x)
        if isinstance(x, str) and x.isdigit():
            return int(x)
        raise ValueError

    try:
        _ = [to_int(c) for c in df_order]
        return sorted(df_order, key=lambda c: to_int(c))
    except Exception:
        return df_order


def select_feature_set(df: pd.DataFrame, feature_set: str, local_n: int = 94) -> List[str]:
    feat_cols = detect_feature_cols(df)
    feat_cols = ordered_feature_cols(df, feat_cols)

    if feature_set == "all":
        return feat_cols

    if feature_set == "local":
        if len(feat_cols) < local_n:
            raise ValueError(f"Requested local_n={local_n}, but only {len(feat_cols)} features found.")
        return feat_cols[:local_n]

    raise ValueError(f"Unknown feature_set: {feature_set}. Use 'local' or 'all'.")


# -----------------------------
# Thresholding + evaluation
# -----------------------------
def pick_threshold_by_best_f1(y_true: np.ndarray, y_prob: np.ndarray) -> Dict[str, float]:
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_prob)
    f1s = (2 * precisions * recalls) / (precisions + recalls + 1e-12)
    best_idx = int(np.nanargmax(f1s))
    thr = float(thresholds[min(best_idx, len(thresholds) - 1)]) if len(thresholds) > 0 else 0.5
    return {"threshold": thr, "precision": float(precisions[best_idx]), "recall": float(recalls[best_idx]), "f1": float(f1s[best_idx])}


def pick_threshold_for_target_precision(
    y_true: np.ndarray, y_prob: np.ndarray, target_precision: float = 0.80
) -> Dict[str, float]:
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_prob)
    mask = precisions >= target_precision
    if not mask.any():
        best_idx = int(np.nanargmax(precisions))
        thr = float(thresholds[min(best_idx, len(thresholds) - 1)]) if len(thresholds) > 0 else 0.5
        return {
            "threshold": thr,
            "precision": float(precisions[best_idx]),
            "recall": float(recalls[best_idx]),
            "note": f"Target precision {target_precision:.2f} not reached; using max precision point.",
        }

    candidates = np.where(mask)[0]
    best_idx = int(candidates[np.nanargmax(recalls[candidates])])
    thr = float(thresholds[min(best_idx, len(thresholds) - 1)]) if len(thresholds) > 0 else 0.5
    return {"threshold": thr, "precision": float(precisions[best_idx]), "recall": float(recalls[best_idx])}


def evaluate_predictions(y_true: np.ndarray, y_prob: np.ndarray, threshold: float) -> Dict[str, float | int]:
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


def pr_auc_by_time_step(test_df: pd.DataFrame, y_prob: np.ndarray) -> pd.DataFrame:
    """
    Compute PR-AUC per time_step on test split.
    Skip steps with no positive labels.
    """
    tmp = test_df[["time_step", "y"]].copy()
    tmp["y_prob"] = y_prob
    rows = []
    for t, g in tmp.groupby("time_step"):
        y = g["y"].to_numpy().astype(int)
        if (y == 1).sum() == 0:
            continue
        ap = average_precision_score(y, g["y_prob"].to_numpy())
        rows.append({"time_step": int(t), "pr_auc": float(ap), "n": int(len(g)), "pos": int((y == 1).sum())})
    out = pd.DataFrame(rows).sort_values("time_step").reset_index(drop=True)
    return out


def plot_pr_auc_by_time_step(df_ts: pd.DataFrame, out_path: Path, title: str) -> None:
    if df_ts.empty:
        return
    plt.figure()
    plt.plot(df_ts["time_step"], df_ts["pr_auc"])
    plt.xlabel("time_step")
    plt.ylabel("PR-AUC")
    plt.title(title)
    plt.grid(True, alpha=0.3)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


@dataclass
class RunResult:
    model_name: str
    feature_set: str
    n_train: int
    n_test: int
    n_features: int
    train_pos_rate: float
    test_pos_rate: float
    pr_auc: float
    roc_auc: float
    best_f1: Dict[str, float]
    target_precision: float
    threshold_at_target_precision: Dict[str, float]
    eval_at_best_f1: Dict[str, float | int]
    eval_at_target_precision: Dict[str, float | int]
    pr_auc_by_time_step_path: str


# -----------------------------
# Training runners
# -----------------------------
def run_logreg(X_train, y_train, X_test) -> np.ndarray:
    model = Pipeline(
        steps=[
            ("scaler", StandardScaler(with_mean=True, with_std=True)),
            ("clf", LogisticRegression(max_iter=2000, class_weight="balanced", n_jobs=-1)),
        ]
    )
    model.fit(X_train, y_train)
    return model.predict_proba(X_test)[:, 1]


def run_histgb(X_train, y_train, X_test) -> np.ndarray:
    # Emulate class weights via sample weights
    pos = int((y_train == 1).sum())
    neg = int((y_train == 0).sum())
    w_pos = neg / max(pos, 1)
    sample_weight = np.where(y_train == 1, w_pos, 1.0)

    model = HistGradientBoostingClassifier(
        max_depth=6,
        learning_rate=0.05,
        max_iter=400,
        l2_regularization=1.0,
        random_state=42,
    )
    model.fit(X_train, y_train, sample_weight=sample_weight)
    return model.predict_proba(X_test)[:, 1]

def run_xgboost(X_train, y_train, X_test) -> np.ndarray:
    pos = int((y_train == 1).sum())
    neg = int((y_train == 0).sum())
    scale_pos_weight = neg / max(pos, 1)

    model = XGBClassifier(
        n_estimators=800,
        learning_rate=0.05,
        max_depth=4,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        min_child_weight=1.0,
        objective="binary:logistic",
        eval_metric="aucpr",
        tree_method="hist",
        n_jobs=-1,
        random_state=42,
        scale_pos_weight=scale_pos_weight,  # ✅ put it here
    )

    model.fit(X_train, y_train)            # ✅ no scale_pos_weight here
    return model.predict_proba(X_test)[:, 1]

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target_precision", type=float, default=0.80)
    ap.add_argument("--local_n", type=int, default=94)
    ap.add_argument("--feature_sets", nargs="+", default=["local", "all"], choices=["local", "all"])
    args = ap.parse_args()

    here = Path(__file__).resolve()
    root = find_repo_root(here)

    nodes_path = root / "data" / "processed" / "nodes.parquet"
    metrics_dir = root / "reports" / "metrics"
    figs_dir = root / "reports" / "figures"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    figs_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(nodes_path)

    # labeled supervised baseline
    df = df[df["split"].isin(["train", "test"])].copy()
    df = df[df["y"].isin([0, 1])].copy()

    train_df = df[df["split"] == "train"].copy()
    test_df = df[df["split"] == "test"].copy()

    summary_rows = []

    for feature_set in args.feature_sets:
        feat_cols = select_feature_set(df, feature_set=feature_set, local_n=args.local_n)

        X_train = train_df[feat_cols].to_numpy()
        y_train = train_df["y"].to_numpy().astype(int)

        X_test = test_df[feat_cols].to_numpy()
        y_test = test_df["y"].to_numpy().astype(int)

        train_pos = float(y_train.mean())
        test_pos = float(y_test.mean())

        print(f"\n=== Feature set: {feature_set} | n_features={len(feat_cols)} ===")
        print(f"Train: {len(train_df)} | Test: {len(test_df)}")
        print(f"Positive rate train={train_pos:.4f} | test={test_pos:.4f}")

        for model_name, runner in [
            ("logreg_balanced", run_logreg),
            ("histgb_weighted", run_histgb),
            ("xgboost", run_xgboost),
        ]:

            y_prob = runner(X_train, y_train, X_test)

            pr_auc = float(average_precision_score(y_test, y_prob))
            roc_auc = float(roc_auc_score(y_test, y_prob))

            best_f1 = pick_threshold_by_best_f1(y_test, y_prob)
            thr_p = pick_threshold_for_target_precision(y_test, y_prob, target_precision=args.target_precision)

            eval_best = evaluate_predictions(y_test, y_prob, best_f1["threshold"])
            eval_p = evaluate_predictions(y_test, y_prob, thr_p["threshold"])

            # PR curve
            pr_path = figs_dir / f"pr_curve_{model_name}_{feature_set}.png"
            plot_pr_curve(y_test, y_prob, pr_path, title=f"{model_name} ({feature_set})")

            # Drift by time_step
            ts_df = pr_auc_by_time_step(test_df, y_prob)
            ts_path = metrics_dir / f"pr_auc_by_time_step_{model_name}_{feature_set}.csv"
            ts_df.to_csv(ts_path, index=False)

            ts_fig = figs_dir / f"pr_auc_by_time_step_{model_name}_{feature_set}.png"
            plot_pr_auc_by_time_step(ts_df, ts_fig, title=f"PR-AUC by time_step — {model_name} ({feature_set})")

            res = RunResult(
                model_name=model_name,
                feature_set=feature_set,
                n_train=int(len(train_df)),
                n_test=int(len(test_df)),
                n_features=int(len(feat_cols)),
                train_pos_rate=train_pos,
                test_pos_rate=test_pos,
                pr_auc=pr_auc,
                roc_auc=roc_auc,
                best_f1=best_f1,
                target_precision=float(args.target_precision),
                threshold_at_target_precision=thr_p,
                eval_at_best_f1=eval_best,
                eval_at_target_precision=eval_p,
                pr_auc_by_time_step_path=str(ts_path.relative_to(root)),
            )

            out_json = metrics_dir / f"baseline_{model_name}_{feature_set}.json"
            with open(out_json, "w", encoding="utf-8") as f:
                json.dump(asdict(res), f, indent=2)

            print(f"[{model_name} | {feature_set}] PR-AUC={pr_auc:.4f} ROC-AUC={roc_auc:.4f}")
            print(f"  Best-F1 thr={best_f1['threshold']:.4f} -> P={best_f1['precision']:.3f} R={best_f1['recall']:.3f} F1={best_f1['f1']:.3f}")
            note = thr_p.get("note", "")
            if note:
                print(f"  Target-P note: {note}")
            print(f"  P@{args.target_precision:.2f} thr={thr_p['threshold']:.4f} -> P={thr_p['precision']:.3f} R={thr_p['recall']:.3f}")

            summary_rows.append(
                {
                    "model": model_name,
                    "feature_set": feature_set,
                    "n_features": int(len(feat_cols)),
                    "pr_auc": pr_auc,
                    "roc_auc": roc_auc,
                    "p_at_target": float(res.eval_at_target_precision["precision"]),
                    "r_at_target": float(res.eval_at_target_precision["recall"]),
                }
            )

    # Write markdown summary table
    summary = pd.DataFrame(summary_rows).sort_values(["feature_set", "model"]).reset_index(drop=True)
    md_lines = []
    md_lines.append("## Baseline results (tabular)\n")
    md_lines.append("| Model | Feature set | #features | PR-AUC | ROC-AUC | Precision@target | Recall@target |")
    md_lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for _, r in summary.iterrows():
        md_lines.append(
            f"| {r['model']} | {r['feature_set']} | {int(r['n_features'])} | {r['pr_auc']:.4f} | {r['roc_auc']:.4f} | {r['p_at_target']:.3f} | {r['r_at_target']:.3f} |"
        )
    md_out = metrics_dir / "baseline_ablation_table.md"
    md_out.write_text("\n".join(md_lines) + "\n", encoding="utf-8")

    print("\n✅ Wrote summary table:", md_out)
    print("✅ Figures in:", figs_dir)
    print("✅ Metrics in:", metrics_dir)


if __name__ == "__main__":
    main()
