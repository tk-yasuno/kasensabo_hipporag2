"""
experiments/03_rag_retrievers.py
────────────────────────────────────────────────────────────
Phase 3: 3方式 RAG 検索モジュール

このファイルは単体でも import しても使える。

クラス:
  BaseRetriever       — 共通インターフェース
  NaiveRetriever      — embedding 類似度 → top-k
  LightRetriever      — BM25 + embedding スコア融合 → top-k
  HippoRAG2Retriever  — 巻→章→節チャンクの 3段階 coarse-to-fine

Usage (単体テスト):
    python experiments/03_rag_retrievers.py --test
    python experiments/03_rag_retrievers.py --test --query "ダム設計における許容応力度の設定方法"
"""

from __future__ import annotations

import argparse
import json
import pickle
import time
from pathlib import Path
from typing import NamedTuple

IDX_DIR = Path(__file__).parent / "indices"

EMBED_MODEL_NAME = "hotchpotch/static-embedding-japanese"


# ─────────────────────────────────────────────────────────
# データ型
# ─────────────────────────────────────────────────────────

class RetrievedChunk(NamedTuple):
    chunk_id:   int
    score:      float
    text:       str
    doc_id:     str
    heading:    str
    volume:     str
    chapter:    str


# ─────────────────────────────────────────────────────────
# 共通インデックスロード（シングルトン）
# ─────────────────────────────────────────────────────────

_INDEX_CACHE: dict = {}

def _load_indices() -> dict:
    global _INDEX_CACHE
    if _INDEX_CACHE:
        return _INDEX_CACHE

    import faiss
    import numpy as np

    chunks: list[dict] = []
    with open(IDX_DIR / "chunks.jsonl", encoding="utf-8") as f:
        for line in f:
            l = line.strip()
            if l:
                chunks.append(json.loads(l))

    embeddings = np.load(IDX_DIR / "embeddings.npy")
    faiss_idx  = faiss.read_index(str(IDX_DIR / "faiss.index"))

    with open(IDX_DIR / "bm25.pkl", "rb") as f:
        bm25 = pickle.load(f)

    with open(IDX_DIR / "hierarchy.json", encoding="utf-8") as f:
        hierarchy = json.load(f)

    vol_path  = IDX_DIR / "hipporag2_volumes.json"
    chap_path = IDX_DIR / "hipporag2_chapters.json"
    volumes_info  = json.loads(vol_path.read_text(encoding="utf-8"))  if vol_path.exists()  else []
    chapters_info = json.loads(chap_path.read_text(encoding="utf-8")) if chap_path.exists() else []

    _INDEX_CACHE = {
        "chunks":        chunks,
        "embeddings":    embeddings,
        "faiss":         faiss_idx,
        "bm25":          bm25,
        "hierarchy":     hierarchy,
        "volumes_info":  volumes_info,
        "chapters_info": chapters_info,
    }
    return _INDEX_CACHE


# ─────────────────────────────────────────────────────────
# Embedding モデル（シングルトン）
# ─────────────────────────────────────────────────────────

_EMBED_MODEL = None

def _get_embed_model():
    global _EMBED_MODEL
    if _EMBED_MODEL is None:
        from sentence_transformers import SentenceTransformer
        _EMBED_MODEL = SentenceTransformer(EMBED_MODEL_NAME)
    return _EMBED_MODEL


def _encode(query: str) -> "np.ndarray":
    import numpy as np
    model = _get_embed_model()
    vec   = model.encode([query], normalize_embeddings=True, convert_to_numpy=True)
    return vec.astype("float32")   # shape: (1, dim)


# ─────────────────────────────────────────────────────────
# BM25 クエリトークナイズ（build 側と同一）
# ─────────────────────────────────────────────────────────

def _bm25_tokenize(text: str) -> list[str]:
    chars   = list(text)
    bigrams = [text[i:i+2] for i in range(len(text) - 1)]
    return chars + bigrams


# ─────────────────────────────────────────────────────────
# BaseRetriever
# ─────────────────────────────────────────────────────────

class BaseRetriever:
    def __init__(self, top_k: int = 5):
        self.top_k = top_k
        self._indices: dict | None = None

    def _init(self):
        if self._indices is None:
            self._indices = _load_indices()

    def _make_chunk(self, idx: int, score: float) -> RetrievedChunk:
        c = self._indices["chunks"][idx]
        return RetrievedChunk(
            chunk_id = idx,
            score    = float(score),
            text     = c.get("text", ""),
            doc_id   = c.get("doc_id", ""),
            heading  = c.get("heading", ""),
            volume   = c.get("volume", ""),
            chapter  = c.get("chapter", ""),
        )

    def retrieve(self, query: str) -> list[RetrievedChunk]:
        raise NotImplementedError


# ─────────────────────────────────────────────────────────
# NaiveRetriever — 全チャンク embedding 類似度 → top-k
# ─────────────────────────────────────────────────────────

class NaiveRetriever(BaseRetriever):
    """全チャンク空間に対して embedding 内積類似度で top-k を返す。"""

    def retrieve(self, query: str) -> list[RetrievedChunk]:
        self._init()
        q_vec  = _encode(query)           # (1, dim)
        D, I   = self._indices["faiss"].search(q_vec, self.top_k)
        return [self._make_chunk(int(I[0][i]), float(D[0][i])) for i in range(len(I[0]))]


# ─────────────────────────────────────────────────────────
# LightRetriever — BM25 + embedding スコア融合
# ─────────────────────────────────────────────────────────

class LightRetriever(BaseRetriever):
    """
    BM25 スコア（正規化済み）と embedding スコア（内積）を alpha:1-alpha で融合。
    alpha=0.5 がデフォルト。
    """

    def __init__(self, top_k: int = 5, alpha: float = 0.5, pre_k: int = 50):
        super().__init__(top_k)
        self.alpha = alpha       # embedding 重み
        self.pre_k = pre_k       # BM25 候補取得数（融合前）

    def retrieve(self, query: str) -> list[RetrievedChunk]:
        import numpy as np
        self._init()
        n_chunks = len(self._indices["chunks"])

        # ── BM25 スコア ──
        tokens    = _bm25_tokenize(query)
        bm25_raw  = self._indices["bm25"].get_scores(tokens)   # shape: (n_chunks,)
        bm25_max  = bm25_raw.max()
        bm25_norm = bm25_raw / (bm25_max + 1e-10)              # 0〜1 正規化

        # ── embedding スコア ──
        q_vec       = _encode(query)                            # (1, dim)
        emb_scores  = (self._indices["embeddings"] @ q_vec.T).flatten()   # (n_chunks,)
        # 内積は L2 正規化済みベクトルに対してコサイン類似度 ≒ 0〜1
        emb_min = emb_scores.min()
        emb_max = emb_scores.max()
        emb_norm = (emb_scores - emb_min) / (emb_max - emb_min + 1e-10)

        # ── 融合 ──
        fused = self.alpha * emb_norm + (1 - self.alpha) * bm25_norm

        # ── top-k 取得 ──
        top_ids = np.argsort(-fused)[:self.top_k]
        return [self._make_chunk(int(i), float(fused[i])) for i in top_ids]


# ─────────────────────────────────────────────────────────
# HippoRAG2Retriever — 3段階 coarse-to-fine 検索 (v0.2.1: キーワード辞書対応)
# ─────────────────────────────────────────────────────────

class HippoRAG2Retriever(BaseRetriever):
    """
    Level 1 (coarse): クエリ → ボリューム代表ベクトル + キーワード → 上位 n_vol ボリューム選択
    Level 2 (mid):    選択ボリューム内の章代表ベクトル → 上位 n_chap 章選択
    Level 3 (fine):   選択章のチャンクのみで embedding 検索 → top-k

    v0.2.1で改善: ボリューム選択にキーワード辞書を組み込み（embedding + keyword融合）
    """

    def __init__(self, top_k: int = 5, n_vol: int = 2, n_chap: int = 3, use_keywords: bool = True):
        super().__init__(top_k)
        self.n_vol  = n_vol
        self.n_chap = n_chap
        self.use_keywords = use_keywords
        self._vol_vecs: "np.ndarray | None"  = None
        self._chap_vecs: "np.ndarray | None" = None
        self._keywords_dict: dict | None = None

    def _load_keywords(self):
        """キーワード辞書をロード"""
        if self._keywords_dict is not None:
            return
        
        kw_file = IDX_DIR.parent / "volume_keywords.json"
        if not kw_file.exists():
            self._keywords_dict = {}
            return
        
        try:
            with open(kw_file, encoding="utf-8") as f:
                self._keywords_dict = json.load(f)
        except Exception:
            self._keywords_dict = {}

    def _compute_keyword_scores(self, query: str) -> dict[str, float]:
        """クエリのキーワード解析でボリュームスコアを計算"""
        if not self.use_keywords or not self._keywords_dict:
            return {}
        
        query_lower = query.lower()
        volume_scores = {}
        
        for vol_name, vol_info in self._keywords_dict.get("volumes", {}).items():
            score = 0.0
            
            # 主キーワードマッチ
            for kw in vol_info.get("keywords", []):
                if kw.lower() in query_lower:
                    score += 1.0
            
            # 除外キーワード（負のスコア）
            for ex_kw in vol_info.get("exclusion_keywords", []):
                if ex_kw.lower() in query_lower:
                    score -= 0.5
            
            volume_scores[vol_name] = max(0.0, score)
        
        # 正規化
        max_score = max(volume_scores.values()) if volume_scores else 1.0
        if max_score > 0:
            volume_scores = {k: v / max_score for k, v in volume_scores.items()}
        
        return volume_scores

    def _init(self):
        import numpy as np
        super()._init()

        if self._vol_vecs is None:
            vols  = self._indices["volumes_info"]
            chaps = self._indices["chapters_info"]

            if not vols:
                raise RuntimeError("HippoRAG2: volumes_info が空です。01b_build_hipporag2_index.py を実行してください。")

            self._vol_vecs  = np.array([v["vector"] for v in vols],  dtype="float32")
            self._chap_vecs = np.array([c["vector"] for c in chaps], dtype="float32") if chaps else np.zeros((0, 1024), dtype="float32")
        
        # キーワード辞書のロード
        if self.use_keywords and self._keywords_dict is None:
            self._load_keywords()

    def retrieve(self, query: str) -> list[RetrievedChunk]:
        import numpy as np
        self._init()

        q_vec  = _encode(query)   # (1, dim)

        # ── Level 1: ボリューム選択 (embedding + キーワード融合) ──
        vols   = self._indices["volumes_info"]
        v_sim  = (self._vol_vecs @ q_vec.T).flatten()     # (n_vol_total,)
        
        # キーワードスコアの取得と融合
        if self.use_keywords and self._keywords_dict:
            kw_scores = self._compute_keyword_scores(query)
            fusion_cfg = self._keywords_dict.get("fusion_strategy", {})
            emb_weight = fusion_cfg.get("embedding_weight", 0.6)
            kw_weight = fusion_cfg.get("keyword_weight", 0.4)
            
            # ボリューム名とスコアの融合
            combined_scores = []
            for i, vol_info in enumerate(vols):
                vol_name = vol_info["volume"]
                emb_score = (v_sim[i] + 1.0) / 2.0  # [-1, 1] → [0, 1] に正規化
                kw_score = kw_scores.get(vol_name, 0.0)
                combined = emb_weight * emb_score + kw_weight * kw_score
                combined_scores.append(combined)
            
            v_scores = np.array(combined_scores)
        else:
            v_scores = v_sim
        
        top_v  = np.argsort(-v_scores)[:self.n_vol]
        selected_volumes = [vols[i]["volume"] for i in top_v]

        # ── Level 2: 章選択 ──
        chaps  = self._indices["chapters_info"]
        # 選択ボリュームに属する章のみ
        cand_chap_indices = [
            i for i, c in enumerate(chaps)
            if c["volume"] in selected_volumes
        ]

        if cand_chap_indices and self._chap_vecs.shape[0] > 0:
            cand_vecs = self._chap_vecs[cand_chap_indices]    # (n_cand, dim)
            c_sim     = (cand_vecs @ q_vec.T).flatten()
            top_c_rel = np.argsort(-c_sim)[:self.n_chap]
            top_c     = [cand_chap_indices[j] for j in top_c_rel]
            selected_chapters = [chaps[i] for i in top_c]
        else:
            selected_chapters = []

        # ── Level 3: チャンク検索 ──
        # 選択章の chunk_ids を収集
        candidate_ids: set[int] = set()
        for chap in selected_chapters:
            candidate_ids.update(chap["chunk_ids"])

        # 章候補が少ない場合はボリューム全体の chunk_ids でフォールバック
        if len(candidate_ids) < self.top_k:
            for vol_name in selected_volumes:
                vol_entry = self._indices["hierarchy"]["volumes"].get(vol_name, {})
                candidate_ids.update(vol_entry.get("chunk_ids", []))

        candidate_ids_list = sorted(candidate_ids)
        if not candidate_ids_list:
            # フォールバック: 全チャンク検索（Naive と同じ）
            D, I = self._indices["faiss"].search(q_vec, self.top_k)
            return [self._make_chunk(int(I[0][i]), float(D[0][i])) for i in range(len(I[0]))]

        # 候補チャンク内で embedding スコアを計算
        cand_emb = self._indices["embeddings"][candidate_ids_list]   # (n_cand, dim)
        scores   = (cand_emb @ q_vec.T).flatten()
        top_rel  = sorted(range(len(scores)), key=lambda i: -scores[i])[:self.top_k]

        return [
            self._make_chunk(candidate_ids_list[i], float(scores[i]))
            for i in top_rel
        ]

    @property
    def retrieval_info(self) -> dict:
        """デバッグ用: 最後の検索でどのボリューム/章が選ばれたか"""
        return getattr(self, "_last_info", {})


# ─────────────────────────────────────────────────────────
# ファクトリ
# ─────────────────────────────────────────────────────────

def make_retriever(rag_type: str, top_k: int = 5) -> BaseRetriever:
    rag_type = rag_type.lower()
    if rag_type == "naive":
        return NaiveRetriever(top_k=top_k)
    elif rag_type == "light":
        return LightRetriever(top_k=top_k)
    elif rag_type in ("hipporag2", "hippo"):
        return HippoRAG2Retriever(top_k=top_k)
    else:
        raise ValueError(f"不明な RAG タイプ: {rag_type}  (naive / light / hipporag2)")


# ─────────────────────────────────────────────────────────
# context フォーマッタ
# ─────────────────────────────────────────────────────────

def format_context(chunks: list[RetrievedChunk], max_chars: int = 2000) -> str:
    """取得チャンクをプロンプト用テキストにフォーマットする。"""
    parts = []
    total = 0
    for i, c in enumerate(chunks, 1):
        header = f"[{i}] {c.doc_id} / {c.heading}"
        body   = c.text
        chunk_text = f"{header}\n{body}"
        if total + len(chunk_text) > max_chars:
            remaining = max_chars - total
            if remaining > 100:
                chunk_text = chunk_text[:remaining] + "..."
                parts.append(chunk_text)
            break
        parts.append(chunk_text)
        total += len(chunk_text)
    return "\n\n".join(parts)


# ─────────────────────────────────────────────────────────
# 単体テスト
# ─────────────────────────────────────────────────────────

def run_test(query: str, top_k: int = 5):
    print(f"\nクエリ: 「{query}」\n")

    for rag_name in ("naive", "light", "hipporag2"):
        print(f"{'─'*60}")
        print(f"  [{rag_name.upper()}]")
        t0  = time.time()
        ret = make_retriever(rag_name, top_k=top_k)
        res = ret.retrieve(query)
        elapsed = time.time() - t0
        print(f"  取得 {len(res)} チャンク  ({elapsed:.2f}s)")
        for j, c in enumerate(res[:3], 1):
            print(f"  [{j}] {c.doc_id} / {c.heading}  score={c.score:.4f}")
            print(f"       {c.text[:80]}...")
        ctx = format_context(res)
        print(f"  コンテキスト文字数: {len(ctx)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test",  action="store_true", help="単体テスト実行")
    parser.add_argument("--query", type=str,
                        default="ダム設計における許容応力度の設定方法を説明してください",
                        help="テストクエリ")
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()

    run_test(args.query, top_k=args.top_k)
