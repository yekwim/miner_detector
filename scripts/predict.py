#!/usr/bin/env python3
# predict.py — pontua embeddings já agregados (use a mesma função de agregação do train se precisar)
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd
import joblib

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--embeddings_parquet", required=True, help="Parquet agregado com coluna embedding_agg")
    ap.add_argument("--model_dir", required=True, help="pasta com model.joblib")
    ap.add_argument("--key", default="sha256", help="coluna chave (sha256 ou path)")
    ap.add_argument("--out_json", required=True)
    args = ap.parse_args()

    df = pd.read_parquet(Path(args.embeddings_parquet))
    model = joblib.load(Path(args.model_dir) / "model.joblib")

    X = np.stack(df["embedding_agg"].apply(lambda v: np.array(v, dtype=np.float32)))
    proba = model.predict_proba(X)[:,1].tolist()

    out = []
    for k, p in zip(df[args.key], proba):
        out.append({ args.key: k, "score": float(p) })

    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

    print("Scores salvos em:", args.out_json)

if __name__ == "__main__":
    main()
