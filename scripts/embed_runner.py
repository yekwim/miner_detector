#!/usr/bin/env python3
# embed_runner.py — chunking (300–800) + embeddings Qwen3-Embedding-0.6B
# Uso:
#   python scripts/embed_runner.py \
#     --jsonl data/parser_output/embedding_texts.jsonl \
#     --out_dir data/embeddings/ \
#     --batch_size 16 --min 300 --max 800 --overlap 80

import os, json, argparse, socket
from pathlib import Path
from typing import List, Dict, Any
import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
from transformers import AutoTokenizer, AutoModel

def load_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            yield json.loads(line)

def chunk_by_tokens(text: str, tokenizer, minT=300, maxT=800, overlap=80):
    ids = tokenizer.encode(text, add_special_tokens=False)
    chunks = []
    start = 0
    L = len(ids)
    if L == 0:
        return [{"n_tokens":0, "text": ""}]
    while start < L:
        end = min(start + maxT, L)
        if end - start < minT and end < L:
            end = min(start + minT, L)
        slice_ids = ids[start:end]
        chunks.append({
            "n_tokens": len(slice_ids),
            "text": tokenizer.decode(slice_ids)
        })
        if end == L:
            break
        start = end - min(overlap, end)
    return chunks

def masked_mean_pool(last_hidden_state: torch.Tensor, attn_mask: torch.Tensor) -> torch.Tensor:
    mask = attn_mask.unsqueeze(-1).float()
    summed = (last_hidden_state * mask).sum(dim=1)
    counts = mask.sum(dim=1).clamp(min=1e-6)
    out = summed / counts
    return torch.nn.functional.normalize(out, p=2, dim=1)

def embed_texts(texts: List[str], tokenizer, model, device: str) -> np.ndarray:
    batch = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
    batch = {k: v.to(device) for k, v in batch.items()}
    with torch.no_grad():
        out = model(**batch)
        if hasattr(out, "last_hidden_state"):
            vec = masked_mean_pool(out.last_hidden_state, batch["attention_mask"])
        elif hasattr(out, "pooler_output"):
            vec = torch.nn.functional.normalize(out.pooler_output, p=2, dim=1)
        else:
            raise RuntimeError("Modelo não retornou last_hidden_state nem pooler_output")
    return vec.detach().cpu().numpy().astype(np.float32)

def write_parquet_shard(rows: List[Dict[str, Any]], out_dir: Path, shard_idx: int):
    if not rows:
        return None
    df = pd.DataFrame(rows)
    shard_path = out_dir / f"embeddings_shard_{shard_idx:04d}.parquet"
    df.to_parquet(shard_path, index=False)
    return shard_path

def main():
    ap = argparse.ArgumentParser(description="Embed runner Qwen3-Embedding-0.6B")
    ap.add_argument("--jsonl", required=True, help="parser_output/embedding_texts.jsonl")
    ap.add_argument("--out_dir", required=True, help="diretório de saída para shards parquet")
    ap.add_argument("--model", default="Qwen/Qwen3-Embedding-0.6B")
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--rows_per_shard", type=int, default=5000)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--min", type=int, default=300)
    ap.add_argument("--max", type=int, default=800)
    ap.add_argument("--overlap", type=int, default=80)
    args = ap.parse_args()

    jsonl_path = Path(args.jsonl)
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModel.from_pretrained(args.model, trust_remote_code=True).to(args.device).eval()

    buffer_rows = []
    shard_idx = 0
    texts_batch, meta_batch = [], []

    for rec in tqdm(load_jsonl(jsonl_path), desc="Chunking + Embedding"):
        # cria chunks a partir do texto normalizado
        chunks = chunk_by_tokens(rec["text"], tokenizer, minT=args.min, maxT=args.max, overlap=args.overlap)
        for i, c in enumerate(chunks):
            meta_batch.append({
                "id": f"{rec.get('id','')}::{i}",
                "parent_id": rec.get("id"),
                "sha256": rec.get("sha256"),
                "type": rec.get("type"),
                "url": rec.get("url"),
                "path": rec.get("path"),
                "chunk_index": i,
                "n_tokens": c["n_tokens"],
                "text": c["text"]
            })
            texts_batch.append(c["text"])

        # processa em lotes
        if len(texts_batch) >= args.batch_size:
            emb = embed_texts(texts_batch, tokenizer, model, args.device)
            for m, v in zip(meta_batch, emb):
                m["embedding"] = v.tolist()
                buffer_rows.append(m)
            texts_batch, meta_batch = [], []
            if len(buffer_rows) >= args.rows_per_shard:
                write_parquet_shard(buffer_rows, out_dir, shard_idx)
                shard_idx += 1
                buffer_rows = []

    # flush final
    if texts_batch:
        emb = embed_texts(texts_batch, tokenizer, model, args.device)
        for m, v in zip(meta_batch, emb):
            m["embedding"] = v.tolist()
            buffer_rows.append(m)

    if buffer_rows:
        write_parquet_shard(buffer_rows, out_dir, shard_idx)

    meta = {
        "model": args.model,
        "device": args.device,
        "batch_size": args.batch_size,
        "rows_per_shard": args.rows_per_shard,
        "tokenizer": "Qwen official (HF)",
        "min_tokens": args.min, "max_tokens": args.max, "overlap": args.overlap,
        "host": socket.gethostname(),
    }
    with open(out_dir / "embeddings_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print("Concluído. Shards em:", str(out_dir))

if __name__ == "__main__":
    main()
