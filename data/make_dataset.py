#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

import pandas as pd

from pathlib import Path

def find_repo_root(start: Path) -> Path:
    for p in [start] + list(start.parents):
        if (p / ".git").exists():
            return p
    raise FileNotFoundError("Could not find repo root (no .git folder found).")


def repo_root() -> Path:
    return find_repo_root(Path(__file__).resolve())


def main() -> None:
    root = repo_root()
    raw_dir = root / "data" / "raw" / "elliptic"
    out_dir = root / "data" / "processed"
    out_dir.mkdir(parents=True, exist_ok=True)

    features_path = raw_dir / "elliptic_txs_features.csv"
    edges_path = raw_dir / "elliptic_txs_edgelist.csv"
    classes_path = raw_dir / "elliptic_txs_classes.csv"

    # Features: header=None; col0=txId, col1=time_step, rest=features
    feat = pd.read_csv(features_path, header=None)
    feat = feat.rename(columns={0: "txId", 1: "time_step"})
    feat["txId"] = feat["txId"].astype(str)
    feat["time_step"] = feat["time_step"].astype(int)

    # Labels
    cls = pd.read_csv(classes_path)
    cls["txId"] = cls["txId"].astype(str)

    # Map to numeric: licit=0, illicit=1, unknown=2 (same mapping PyG uses)
    mapping = {"2": 0, "1": 1, "unknown": 2}
    cls["y"] = cls["class"].map(mapping).astype("int64")

    # Merge
    nodes = feat.merge(cls[["txId", "y"]], on="txId", how="left")
    if nodes["y"].isna().any():
        raise ValueError("Some txIds in features have no label row in classes.")

    # Edges
    edges = pd.read_csv(edges_path)
    edges["txId1"] = edges["txId1"].astype(str)
    edges["txId2"] = edges["txId2"].astype(str)

    # Map txId -> integer index (stable order from features file)
    id2idx = pd.Series(range(len(nodes)), index=nodes["txId"]).to_dict()
    edges["src"] = edges["txId1"].map(id2idx)
    edges["dst"] = edges["txId2"].map(id2idx)
    edges = edges.dropna(subset=["src", "dst"]).astype({"src": "int64", "dst": "int64"})
    edges = edges[["src", "dst"]]

    # Time-based split like PyG (train: <35, test: >=35), labeled only (y != 2)
    labeled = nodes["y"] != 2
    nodes["split"] = "ignore"
    nodes.loc[(nodes["time_step"] < 35) & labeled, "split"] = "train"
    nodes.loc[(nodes["time_step"] >= 35) & labeled, "split"] = "test"

    # Save
    nodes_out = out_dir / "nodes.parquet"
    edges_out = out_dir / "edges.parquet"
    nodes.to_parquet(nodes_out, index=False)
    edges.to_parquet(edges_out, index=False)

    print("✅ Wrote:", nodes_out)
    print("✅ Wrote:", edges_out)
    print("Nodes:", len(nodes), "| Edges:", len(edges))
    print(nodes["split"].value_counts())


if __name__ == "__main__":
    main()
