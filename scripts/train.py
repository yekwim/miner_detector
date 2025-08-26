#!/usr/bin/env python3
import argparse, json, re
from pathlib import Path
from typing import List, Dict
import numpy as np
import pandas as pd
from urllib.parse import urlparse

from sklearn.model_selection import GroupKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, precision_recall_fscore_support, roc_curve
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import joblib

def read_parquet_shards(in_path: Path) -> pd.DataFrame:
    if in_path.is_dir():
        files = sorted(in_path.glob("*.parquet"))
        df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    else:
        df = pd.read_parquet(in_path)
    return df

def extract_domain(url: str) -> str:
    try:
        host = urlparse(url).hostname or ""
        return re.sub(r"^www\.", "", host)
    except:
        return ""

def agg_embeddings(df: pd.DataFrame, level: str = "sha256") -> pd.DataFrame:
    key = level if level in df.columns else "path"
    X = np.stack(df["embedding"].apply(lambda x: np.array(x, dtype=np.float32)))
    df = df.assign(_emb_obj=list(X))
    groups = df.groupby(df[key].fillna(""))

    rows = []
    for k, g in groups:
        arr = np.stack(g["_emb_obj"].to_list())
        mean = arr.mean(axis=0)
        mx = arr.max(axis=0)
        vec = np.concatenate([mean, mx], axis=0).astype(np.float32)
        rows.append({
            key: k,
            "type_major": g["type"].mode().iat[0] if not g["type"].empty else "",
            "domain": extract_domain(g["url"].iloc[0]) if "url" in g.columns and len(g["url"]) else "",
            "n_chunks": len(g),
            "embedding_agg": vec
        })
    return pd.DataFrame(rows)

def load_labels(labels_csv: Path) -> pd.DataFrame:
    lab = pd.read_csv(labels_csv)
    lab.columns = [c.strip().lower() for c in lab.columns]
    if "label" not in lab.columns:
        raise ValueError("labels.csv deve conter coluna 'label'")
    return lab

def align_features_labels(agg_df: pd.DataFrame, lab: pd.DataFrame) -> pd.DataFrame:
    for key in ["sha256","url","path"]:
        if key in agg_df.columns and key in lab.columns:
            return agg_df.merge(lab[[key,"label"]], on=key, how="inner")
    raise ValueError("labels.csv precisa ter uma chave em comum: sha256, url ou path")

def choose_threshold_at_fpr(y_true: np.ndarray, y_prob: np.ndarray, target_fpr: float = 0.01) -> float:
    fpr, tpr, thr = roc_curve(y_true, y_prob)
    mask = fpr <= target_fpr
    if not np.any(mask):
        return 0.5
    idx = np.argmax(thr[mask])
    return float(thr[mask][idx])

def main():
    ap = argparse.ArgumentParser(description="Treino e calibração")
    ap.add_argument("--embeddings", required=True, help="arquivo ou diretório parquet de embeddings")
    ap.add_argument("--labels", required=True, help="CSV com labels (sha256,label ou url,label)")
    ap.add_argument("--level", default="sha256", choices=["sha256","url"])
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--cv_splits", type=int, default=5)
    ap.add_argument("--target_fpr", type=float, default=0.01)
    args = ap.parse_args()

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    df = read_parquet_shards(Path(args.embeddings))
    agg = agg_embeddings(df, level=args.level)
    lab = load_labels(Path(args.labels))
    data = align_features_labels(agg, lab)

    X = np.stack(data["embedding_agg"].to_list())
    y = data["label"].astype(int).to_numpy()
    groups = data["domain"].fillna("").astype(str).to_numpy()

    base = Pipeline([
        ("scaler", StandardScaler(with_mean=True, with_std=True)),
        ("clf", LogisticRegression(max_iter=1000, class_weight="balanced"))
    ])

    gkf = GroupKFold(n_splits=args.cv_splits)
    probs_cv, y_cv = [], []
    for tr, te in gkf.split(X, y, groups):
        base_cv = Pipeline([
            ("scaler", StandardScaler(with_mean=True, with_std=True)),
            ("clf", LogisticRegression(max_iter=1000, class_weight="balanced"))
        ])
        calib = CalibratedClassifierCV(base_estimator=base_cv, method="isotonic", cv="prefit")
        base_cv.fit(X[tr], y[tr])
        calib.fit(X[tr], y[tr])
        p = calib.predict_proba(X[te])[:,1]
        probs_cv.append(p); y_cv.append(y[te])

    y_cv = np.concatenate(y_cv); p_cv = np.concatenate(probs_cv)

    auc = roc_auc_score(y_cv, p_cv)
    thr = choose_threshold_at_fpr(y_cv, p_cv, target_fpr=args.target_fpr)
    y_hat = (p_cv >= thr).astype(int)
    prec, rec, f1, _ = precision_recall_fscore_support(y_cv, y_hat, average="binary")

    metrics = {
        "roc_auc": float(auc),
        "precision": float(prec),
        "recall": float(rec),
        "f1": float(f1),
        "threshold_at_target_fpr": float(thr),
        "target_fpr": args.target_fpr,
        "cv_splits": args.cv_splits,
        "n_samples": int(len(y_cv))
    }
    with open(out_dir / "cv_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    base.fit(X, y)
    calib_full = CalibratedClassifierCV(base_estimator=base, method="isotonic", cv="prefit")
    calib_full.fit(X, y)

    import joblib
    joblib.dump(calib_full, out_dir / "model.joblib")
    with open(out_dir / "model_card.json", "w", encoding="utf-8") as f:
        json.dump({
            "agg_level": args.level,
            "feature": "concat(mean,max) of chunk embeddings",
            "model": "LogisticRegression + isotonic calibration",
            "cv_metrics": metrics
        }, f, indent=2)

    print("Treino concluído.\nArtefatos em:", str(out_dir))
    print("Métricas:", metrics)

if __name__ == "__main__":
    main()
