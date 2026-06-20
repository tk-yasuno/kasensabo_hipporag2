"""
experiments/01b_build_hipporag2_index.py
────────────────────────────────────────────────────────────
Phase 1b: HippoRAG2 用 階層ベクトルインデックス構築

01_build_indices.py 実行後に実行すること。
（チャンク埋め込み embeddings.npy が必要）

出力:
  experiments/indices/hipporag2_volumes.json   — ボリューム代表ベクトル + メタ
  experiments/indices/hipporag2_chapters.json  — 章代表ベクトル + メタ

代表ベクトルの計算方法:
  各ボリューム / 章に属するチャンクの埋め込みベクトルを平均して L2 正規化する。

Usage:
    python experiments/01b_build_hipporag2_index.py
"""

from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

ROOT    = Path(__file__).parent.parent
IDX_DIR = Path(__file__).parent / "indices"


def load_base_indices() -> tuple[list[dict], "np.ndarray", dict]:
    import numpy as np

    chunks_path = IDX_DIR / "chunks.jsonl"
    emb_path    = IDX_DIR / "embeddings.npy"
    hier_path   = IDX_DIR / "hierarchy.json"

    if not emb_path.exists():
        print("ERROR: embeddings.npy が見つかりません。先に 01_build_indices.py を実行してください。")
        sys.exit(1)

    chunks: list[dict] = []
    with open(chunks_path, encoding="utf-8") as f:
        for line in f:
            l = line.strip()
            if l:
                chunks.append(json.loads(l))

    embeddings  = np.load(emb_path)
    with open(hier_path, encoding="utf-8") as f:
        hierarchy = json.load(f)

    print(f"ロード完了: {len(chunks)} チャンク  embeddings={embeddings.shape}")
    return chunks, embeddings, hierarchy


def build_volume_index(
    embeddings: "np.ndarray",
    hierarchy:  dict,
) -> list[dict]:
    """
    各ボリュームの代表ベクトルと chunk_ids を返す。
    [{"volume": "調査", "chunk_ids": [...], "vector": [...]}, ...]
    """
    import numpy as np

    volumes_info = []
    for vol_name, vol_data in hierarchy["volumes"].items():
        chunk_ids = vol_data["chunk_ids"]
        if not chunk_ids:
            continue
        vecs = embeddings[chunk_ids]          # (n, dim)
        rep  = vecs.mean(axis=0)              # 平均ベクトル
        rep  = rep / (np.linalg.norm(rep) + 1e-10)   # L2 正規化
        volumes_info.append({
            "volume":    vol_name,
            "doc_ids":   vol_data.get("doc_ids", []),
            "chunk_ids": chunk_ids,
            "vector":    rep.tolist(),
        })
    print(f"ボリューム代表ベクトル: {len(volumes_info)} 個")
    return volumes_info


def build_chapter_index(
    embeddings: "np.ndarray",
    hierarchy:  dict,
) -> list[dict]:
    """
    各章の代表ベクトルと chunk_ids を返す。
    [{"volume": "設計", "chapter": "第1章", "chunk_ids": [...], "vector": [...]}, ...]
    """
    import numpy as np

    chapters_info = []
    for vol_name, vol_data in hierarchy["volumes"].items():
        for chap_name, chap_data in vol_data.get("chapters", {}).items():
            if chap_name == "__uncategorized__":
                # 章未分類チャンクはボリューム全体にフォールバックするため除外
                continue
            chunk_ids = chap_data["chunk_ids"]
            if not chunk_ids:
                continue
            vecs = embeddings[chunk_ids]
            rep  = vecs.mean(axis=0)
            rep  = rep / (np.linalg.norm(rep) + 1e-10)
            chapters_info.append({
                "volume":    vol_name,
                "chapter":   chap_name,
                "chunk_ids": chunk_ids,
                "vector":    rep.tolist(),
            })
    print(f"章代表ベクトル: {len(chapters_info)} 個")
    return chapters_info


def main():
    vol_path  = IDX_DIR / "hipporag2_volumes.json"
    chap_path = IDX_DIR / "hipporag2_chapters.json"

    if vol_path.exists() and chap_path.exists():
        print("HippoRAG2 インデックス既存 (削除して再実行すれば再構築)")

    print("\nベースインデックスロード中...")
    chunks, embeddings, hierarchy = load_base_indices()

    print("\nボリューム代表ベクトル構築...")
    volumes_info = build_volume_index(embeddings, hierarchy)
    with open(vol_path, "w", encoding="utf-8") as f:
        json.dump(volumes_info, f, ensure_ascii=False, indent=2)
    print(f"  保存: {vol_path}")

    print("\n章代表ベクトル構築...")
    chapters_info = build_chapter_index(embeddings, hierarchy)
    with open(chap_path, "w", encoding="utf-8") as f:
        json.dump(chapters_info, f, ensure_ascii=False, indent=2)
    print(f"  保存: {chap_path}")

    # ── サマリー出力 ──
    print("\n[HippoRAG2 インデックスサマリー]")
    for vi in volumes_info:
        chaps = [c for c in chapters_info if c["volume"] == vi["volume"]]
        print(f"  {vi['volume']:<12s}  {len(vi['chunk_ids'])} chunk  {len(chaps)} chapter")

    print(f"\n✓ HippoRAG2 インデックス構築完了 → {IDX_DIR}")


if __name__ == "__main__":
    main()
