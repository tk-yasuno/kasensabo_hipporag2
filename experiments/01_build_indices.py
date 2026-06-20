"""
experiments/01_build_indices.py
────────────────────────────────────────────────────────────
Phase 1: インデックス構築

既存の data/rag/chunks.jsonl を使い、以下を生成する:
  experiments/indices/embeddings.npy        — チャンク埋め込みベクトル
  experiments/indices/faiss.index           — FAISS IndexFlatIP
  experiments/indices/bm25.pkl              — BM25 インデックス (rank_bm25)
  experiments/indices/chunks.jsonl          — チャンクデータ（元データのコピー）
  experiments/indices/hierarchy.json        — 階層メタデータ (Volume→Chapter→Chunk)

Embedding: hotchpotch/static-embedding-japanese (1024d)
チャンク  : 500字 / 100字オーバーラップ（元データを再利用）

Usage:
    python experiments/01_build_indices.py
    python experiments/01_build_indices.py --rebuild   # 既存インデックスを強制再構築
    python experiments/01_build_indices.py --dry-run   # チャンク数確認のみ
"""

from __future__ import annotations

import argparse
import json
import pickle
import re
import shutil
import sys
from pathlib import Path

ROOT       = Path(__file__).parent.parent
CHUNKS_SRC = ROOT / "data" / "rag" / "chunks.jsonl"
OUT_DIR    = Path(__file__).parent / "indices"
OUT_DIR.mkdir(parents=True, exist_ok=True)

EMBED_MODEL_NAME = "hotchpotch/static-embedding-japanese"
EMBED_DIM        = 1024
BATCH_SIZE       = 64


# ─────────────────────────────────────────────────────────
# ボリューム / 章 の正規化マッピング
# ─────────────────────────────────────────────────────────

# doc_id → Volume ラベル
VOLUME_MAP: dict[str, str] = {
    "概要編":          "概要",
    "調査編":          "調査",
    "計画編_基本":     "計画",
    "計画編_施設":     "計画",
    "設計編":          "設計",
    "維持管理編_河川": "維持管理",
    "維持管理編_ダム": "維持管理",
    "維持管理編_砂防": "維持管理",
}

# Volume ラベル → キーワードセット（HippoRAG2 粗検索に使用）
VOLUME_KEYWORDS: dict[str, list[str]] = {
    "調査":    ["調査", "測量", "流量観測", "地質", "土質", "水位"],
    "計画":    ["計画", "方針", "整備計画", "治水", "洪水", "目標流量", "環境"],
    "設計":    ["設計", "構造", "許容応力", "安定", "水理", "断面", "施設"],
    "維持管理": ["維持管理", "点検", "診断", "補修", "モニタリング", "劣化", "管理"],
    "概要":    ["概要", "基準", "体系", "四編"],
}


def extract_chapter(heading: str) -> str:
    """見出しから章番号（第X章）を抽出する。見つからない場合は空文字。"""
    m = re.search(r"第\s*\d+\s*章", heading or "")
    return m.group(0).replace(" ", "") if m else ""


# ─────────────────────────────────────────────────────────
# チャンクロード
# ─────────────────────────────────────────────────────────

def load_chunks() -> list[dict]:
    chunks = []
    with open(CHUNKS_SRC, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                c = json.loads(line)
                c["volume"] = VOLUME_MAP.get(c.get("doc_id", ""), "その他")
                c["chapter"] = extract_chapter(c.get("heading", ""))
                chunks.append(c)
    return chunks


# ─────────────────────────────────────────────────────────
# 埋め込み計算
# ─────────────────────────────────────────────────────────

def compute_embeddings(chunks: list[dict]) -> "np.ndarray":
    import numpy as np
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(EMBED_MODEL_NAME)
    texts = [c["text"] for c in chunks]
    print(f"  埋め込み計算: {len(texts)} チャンク (バッチサイズ={BATCH_SIZE})...", flush=True)

    embeddings = model.encode(
        texts,
        batch_size=BATCH_SIZE,
        normalize_embeddings=True,  # コサイン類似度 → 内積で代替
        show_progress_bar=True,
        convert_to_numpy=True,
    )
    return embeddings.astype(np.float32)


# ─────────────────────────────────────────────────────────
# FAISS インデックス構築
# ─────────────────────────────────────────────────────────

def build_faiss(embeddings: "np.ndarray") -> "faiss.IndexFlatIP":
    import faiss
    dim   = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    print(f"  FAISS IndexFlatIP: {index.ntotal} ベクトル (dim={dim})")
    return index


# ─────────────────────────────────────────────────────────
# BM25 インデックス構築
# ─────────────────────────────────────────────────────────

def build_bm25(chunks: list[dict]) -> object:
    from rank_bm25 import BM25Okapi

    # 日本語シンプルトークナイズ（1文字ずつ分解 + 2-gram）
    def tokenize(text: str) -> list[str]:
        chars  = list(text)
        bigrams = [text[i:i+2] for i in range(len(text) - 1)]
        return chars + bigrams

    corpus = [tokenize(c["text"]) for c in chunks]
    bm25   = BM25Okapi(corpus)
    print(f"  BM25 インデックス: {len(corpus)} 文書")
    return bm25


# ─────────────────────────────────────────────────────────
# 階層メタデータ構築
# ─────────────────────────────────────────────────────────

def build_hierarchy(chunks: list[dict]) -> dict:
    """
    {
      "volumes": {
        "調査": {
          "doc_ids": ["調査編"],
          "chunk_ids": [0, 1, ...],
          "chapters": {
            "第1章": { "chunk_ids": [0, 1] },
            ...
          }
        },
        ...
      }
    }
    """
    hierarchy: dict[str, dict] = {}

    for i, c in enumerate(chunks):
        vol = c["volume"]
        doc_id  = c.get("doc_id", "")
        chapter = c.get("chapter", "")

        if vol not in hierarchy:
            hierarchy[vol] = {"doc_ids": [], "chunk_ids": [], "chapters": {}}

        vol_entry = hierarchy[vol]
        if doc_id and doc_id not in vol_entry["doc_ids"]:
            vol_entry["doc_ids"].append(doc_id)
        vol_entry["chunk_ids"].append(i)

        if chapter:
            if chapter not in vol_entry["chapters"]:
                vol_entry["chapters"][chapter] = {"chunk_ids": []}
            vol_entry["chapters"][chapter]["chunk_ids"].append(i)
        else:
            # 章不明チャンクは "その他" に分類
            if "__uncategorized__" not in vol_entry["chapters"]:
                vol_entry["chapters"]["__uncategorized__"] = {"chunk_ids": []}
            vol_entry["chapters"]["__uncategorized__"]["chunk_ids"].append(i)

    print(f"  階層メタデータ: {len(hierarchy)} ボリューム")
    for vol, entry in hierarchy.items():
        print(f"    [{vol}]  {len(entry['doc_ids'])} doc  {len(entry['chunk_ids'])} chunk"
              f"  {len(entry['chapters'])} chapter")

    return {"volumes": hierarchy, "keywords": VOLUME_KEYWORDS}


# ─────────────────────────────────────────────────────────
# main
# ─────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rebuild",  action="store_true", help="既存インデックスを強制再構築")
    parser.add_argument("--dry-run",  action="store_true", help="チャンク数確認のみ")
    args = parser.parse_args()

    emb_path     = OUT_DIR / "embeddings.npy"
    faiss_path   = OUT_DIR / "faiss.index"
    bm25_path    = OUT_DIR / "bm25.pkl"
    hier_path    = OUT_DIR / "hierarchy.json"
    chunks_path  = OUT_DIR / "chunks.jsonl"

    already_built = (
        emb_path.exists()   and
        faiss_path.exists() and
        bm25_path.exists()  and
        hier_path.exists()  and
        chunks_path.exists()
    )

    if already_built and not args.rebuild:
        print(f"インデックス既存 (--rebuild で強制再構築)")
        import numpy as np
        emb = np.load(emb_path)
        print(f"  embeddings.npy : {emb.shape}")
        return

    # ── チャンクロード ──
    print(f"チャンクロード: {CHUNKS_SRC}")
    chunks = load_chunks()
    print(f"  ロード完了: {len(chunks)} チャンク")

    if args.dry_run:
        print("\n[dry-run] チャンク先頭3件:")
        for c in chunks[:3]:
            print(json.dumps(c, ensure_ascii=False, indent=2))
        return

    # ── 埋め込み計算 ──
    print(f"\n埋め込み計算 ({EMBED_MODEL_NAME})...")
    import numpy as np
    embeddings = compute_embeddings(chunks)
    np.save(emb_path, embeddings)
    print(f"  保存: {emb_path}  shape={embeddings.shape}")

    # ── FAISS ──
    print("\nFAISS インデックス構築...")
    import faiss
    index = build_faiss(embeddings)
    faiss.write_index(index, str(faiss_path))
    print(f"  保存: {faiss_path}")

    # ── BM25 ──
    print("\nBM25 インデックス構築...")
    bm25 = build_bm25(chunks)
    with open(bm25_path, "wb") as f:
        pickle.dump(bm25, f)
    print(f"  保存: {bm25_path}")

    # ── 階層メタデータ ──
    print("\n階層メタデータ構築...")
    hierarchy = build_hierarchy(chunks)
    with open(hier_path, "w", encoding="utf-8") as f:
        json.dump(hierarchy, f, ensure_ascii=False, indent=2)
    print(f"  保存: {hier_path}")

    # ── チャンクデータコピー ──
    shutil.copy(CHUNKS_SRC, chunks_path)
    print(f"  コピー: {chunks_path}")

    print(f"\n✓ インデックス構築完了 → {OUT_DIR}")


if __name__ == "__main__":
    main()
