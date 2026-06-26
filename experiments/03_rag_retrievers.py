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

    def __init__(self, top_k: int = 5, use_calibration: bool = False, calibration_dir: str | None = None):
        super().__init__(top_k)
        self.use_calibration = use_calibration
        self.calibration_dir = calibration_dir
        self._chunk_model = None
        self._last_retrieval_info = {}

    def _init(self):
        super()._init()
        if self.use_calibration and self._chunk_model is None:
            self._load_calibration_model()

    def _load_calibration_model(self):
        """Load Naive RAG chunk calibration model (LightGBM or LinearRegression)"""
        import pickle
        from pathlib import Path
        
        if self.calibration_dir is None:
            model_dir = Path("experiments/calibration_models")
        else:
            model_dir = Path(self.calibration_dir)
        
        # Try loading models in priority: ranker (v0.6) > lgbm (v0.6) > linear (v0.5)
        chunk_model_path = model_dir / "ranker_naive_chunk_model.pkl"
        if not chunk_model_path.exists():
            chunk_model_path = model_dir / "lgbm_naive_chunk_model.pkl"
        if not chunk_model_path.exists():
            chunk_model_path = model_dir / "naive_chunk_model.pkl"
        
        if chunk_model_path.exists():
            with open(chunk_model_path, "rb") as f:
                self._chunk_model = pickle.load(f)
            if "ranker_" in str(chunk_model_path):
                model_type = "LambdaMART"
            elif "lgbm_" in str(chunk_model_path):
                model_type = "LightGBM"
            else:
                model_type = "LinearRegression"
            print(f"[NaiveRetriever] Loaded {model_type} calibration model from {chunk_model_path}")
        else:
            print(f"[NaiveRetriever] Warning: Calibration model not found at {chunk_model_path}")

    def _predict_chunk_score(self, embedding_sim: float, chunk_length: int) -> float:
        """Predict Judge score from chunk features"""
        import numpy as np
        if self._chunk_model is None:
            return embedding_sim
        
        features = np.array([[embedding_sim, chunk_length]])
        pred = self._chunk_model.predict(features)[0]
        return float(np.clip(pred, 0, 3))

    def retrieve(self, query: str) -> list[RetrievedChunk]:
        self._init()
        q_vec  = _encode(query)           # (1, dim)
        D, I   = self._indices["faiss"].search(q_vec, self.top_k)
        
        # v0.5: Feature logging for calibration
        chunk_info = []
        for i in range(len(I[0])):
            chunk_idx = int(I[0][i])
            chunk_data = self._indices["chunks"][chunk_idx]
            chunk_info.append({
                "chunk_id": chunk_data.get("chunk_id"),
                "embedding_sim": float(D[0][i]),
                "chunk_length": len(chunk_data.get("text", "")),
            })
        
        self._last_retrieval_info = {
            "chunks": chunk_info,
            "fallback": False,
        }
        
        # v0.5: Calibration model for chunk reranking
        if self.use_calibration and self._chunk_model:
            calibrated_scores = []
            for i in range(len(I[0])):
                chunk_idx = int(I[0][i])
                chunk_data = self._indices["chunks"][chunk_idx]
                embedding_sim = float(D[0][i])
                chunk_length = len(chunk_data.get("text", ""))
                pred_score = self._predict_chunk_score(embedding_sim, chunk_length)
                calibrated_scores.append((chunk_idx, pred_score))
            
            # Rerank by predicted Judge score
            calibrated_scores.sort(key=lambda x: -x[1])
            return [self._make_chunk(idx, score) for idx, score in calibrated_scores]
        
        return [self._make_chunk(int(I[0][i]), float(D[0][i])) for i in range(len(I[0]))]

    @property
    def retrieval_info(self) -> dict:
        """v0.5 calibration用: 最後の検索でのChunk情報"""
        return getattr(self, "_last_retrieval_info", {})


# ─────────────────────────────────────────────────────────
# LightRetriever — BM25 + embedding スコア融合
# ─────────────────────────────────────────────────────────

class LightRetriever(BaseRetriever):
    """
    BM25 スコア（正規化済み）と embedding スコア（内積）を alpha:1-alpha で融合。
    alpha=0.5 がデフォルト。
    """

    def __init__(self, top_k: int = 5, alpha: float = 0.5, pre_k: int = 50, use_calibration: bool = False, calibration_dir: str | None = None):
        super().__init__(top_k)
        self.alpha = alpha       # embedding 重み
        self.pre_k = pre_k       # BM25 候補取得数（融合前）
        self.use_calibration = use_calibration
        self.calibration_dir = calibration_dir
        self._chunk_model = None
        self._last_retrieval_info = {}

    def _init(self):
        super()._init()
        if self.use_calibration and self._chunk_model is None:
            self._load_calibration_model()

    def _load_calibration_model(self):
        """Load Light RAG chunk calibration model (LightGBM or LinearRegression)"""
        import pickle
        from pathlib import Path
        
        if self.calibration_dir is None:
            model_dir = Path("experiments/calibration_models")
        else:
            model_dir = Path(self.calibration_dir)
        
        # Try loading models in priority: ranker (v0.6) > lgbm (v0.6) > linear (v0.5)
        chunk_model_path = model_dir / "ranker_light_chunk_model.pkl"
        if not chunk_model_path.exists():
            chunk_model_path = model_dir / "lgbm_light_chunk_model.pkl"
        if not chunk_model_path.exists():
            chunk_model_path = model_dir / "light_chunk_model.pkl"
        
        if chunk_model_path.exists():
            with open(chunk_model_path, "rb") as f:
                self._chunk_model = pickle.load(f)
            if "ranker_" in str(chunk_model_path):
                model_type = "LambdaMART"
            elif "lgbm_" in str(chunk_model_path):
                model_type = "LightGBM"
            else:
                model_type = "LinearRegression"
            print(f"[LightRetriever] Loaded {model_type} calibration model from {chunk_model_path}")
        else:
            print(f"[LightRetriever] Warning: Calibration model not found at {chunk_model_path}")

    def _predict_chunk_score(self, embedding_sim: float, bm25_score: float, fused_score: float, chunk_length: int) -> float:
        """Predict Judge score from chunk features"""
        import numpy as np
        if self._chunk_model is None:
            return fused_score
        
        features = np.array([[embedding_sim, bm25_score, fused_score, chunk_length]])
        pred = self._chunk_model.predict(features)[0]
        return float(np.clip(pred, 0, 3))

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
        
        # v0.5: Feature logging for calibration
        chunk_info = []
        for i in top_ids:
            chunk_data = self._indices["chunks"][int(i)]
            chunk_info.append({
                "chunk_id": chunk_data.get("chunk_id"),
                "embedding_sim": float(emb_norm[i]),
                "bm25_score": float(bm25_norm[i]),
                "fused_score": float(fused[i]),
                "chunk_length": len(chunk_data.get("text", "")),
            })
        
        self._last_retrieval_info = {
            "chunks": chunk_info,
            "fallback": False,
        }
        
        # v0.5: Calibration model for chunk reranking
        if self.use_calibration and self._chunk_model:
            calibrated_scores = []
            for i in top_ids:
                chunk_data = self._indices["chunks"][int(i)]
                embedding_sim = float(emb_norm[i])
                bm25_score = float(bm25_norm[i])
                fused_score = float(fused[i])
                chunk_length = len(chunk_data.get("text", ""))
                pred_score = self._predict_chunk_score(embedding_sim, bm25_score, fused_score, chunk_length)
                calibrated_scores.append((int(i), pred_score))
            
            # Rerank by predicted Judge score
            calibrated_scores.sort(key=lambda x: -x[1])
            return [self._make_chunk(idx, score) for idx, score in calibrated_scores]
        
        return [self._make_chunk(int(i), float(fused[i])) for i in top_ids]

    @property
    def retrieval_info(self) -> dict:
        """v0.5 calibration用: 最後の検索でのChunk情報"""
        return getattr(self, "_last_retrieval_info", {})


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

    def __init__(self, top_k: int = 5, n_vol: int = 2, n_chap: int = 3, use_keywords: bool = True, keywords_file: str | None = None, n_triples: int = 20, use_calibration: bool = False, calibration_dir: str | None = None):
        super().__init__(top_k)
        self.n_vol  = n_vol
        self.n_chap = n_chap
        self.use_keywords = use_keywords
        self.keywords_file = keywords_file
        self.n_triples = n_triples  # v0.3: triple filtering の取得数
        self.use_calibration = use_calibration  # v0.5: calibration model 使用フラグ
        self.calibration_dir = calibration_dir  # v0.5: calibration model ディレクトリ
        self._vol_vecs: "np.ndarray | None"  = None
        self._chap_vecs: "np.ndarray | None" = None
        self._keywords_dict: dict | None = None
        self._filtered_triples: list | None = None  # v0.3: triple filtering 対応（クエリごとに更新）
        self._chunk_id_to_idx: dict | None = None  # v0.3.1: chunk_id → index マッピング
        self._triple_index: "faiss.Index | None" = None  # v0.3: triple FAISS index
        self._triple_data: list | None = None  # v0.3: triple metadata
        self._volume_model = None  # v0.5: Volume calibration model
        self._chapter_model = None  # v0.5: Chapter calibration model
        self._chunk_model = None  # v0.5: Chunk calibration model

    def _load_keywords(self):
        """キーワード辞書をロード"""
        if self._keywords_dict is not None:
            return
        
        # カスタムキーワードファイルを優先、なければデフォルト
        if self.keywords_file:
            kw_file = Path(self.keywords_file)
        else:
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
        
        return volume_scores
    
    def _load_calibration_models(self):
        """v0.5/v0.6: Load calibration models from pickle files (LightGBM or LinearRegression)"""
        import pickle
        
        if self.calibration_dir:
            model_dir = Path(self.calibration_dir)
        else:
            model_dir = Path(__file__).parent / "calibration_models"
        
        try:
            # Try loading models in priority: ranker (v0.6) > lgbm (v0.6) > linear (v0.5)
            vol_model_path = model_dir / "ranker_volume_model.pkl"
            if not vol_model_path.exists():
                vol_model_path = model_dir / "lgbm_volume_model.pkl"
            if not vol_model_path.exists():
                vol_model_path = model_dir / "volume_model.pkl"
            if vol_model_path.exists():
                with open(vol_model_path, 'rb') as f:
                    self._volume_model = pickle.load(f)
            
            chap_model_path = model_dir / "ranker_chapter_model.pkl"
            if not chap_model_path.exists():
                chap_model_path = model_dir / "lgbm_chapter_model.pkl"
            if not chap_model_path.exists():
                chap_model_path = model_dir / "chapter_model.pkl"
            if chap_model_path.exists():
                with open(chap_model_path, 'rb') as f:
                    self._chapter_model = pickle.load(f)
            
            chunk_model_path = model_dir / "ranker_chunk_model.pkl"
            if not chunk_model_path.exists():
                chunk_model_path = model_dir / "lgbm_chunk_model.pkl"
            if not chunk_model_path.exists():
                chunk_model_path = model_dir / "chunk_model.pkl"
            if chunk_model_path.exists():
                with open(chunk_model_path, 'rb') as f:
                    self._chunk_model = pickle.load(f)
            
            import sys
            if self._volume_model or self._chapter_model or self._chunk_model:
                if "ranker_" in str(vol_model_path):
                    model_type = "LambdaMART"
                elif "lgbm_" in str(vol_model_path):
                    model_type = "LightGBM"
                else:
                    model_type = "LinearRegression"
                print(f"[INFO] Calibration models ({model_type}) loaded from {model_dir}", file=sys.stderr)
        except Exception as e:
            import sys
            print(f"[WARN] Failed to load calibration models: {e}", file=sys.stderr)
    
    def _predict_volume_score(self, emb_score: float, kw_score: float, triple_score: float, fused_score: float, selected: bool) -> float:
        """v0.5: Predict Judge score for a volume candidate using calibration model"""
        if not self._volume_model:
            return fused_score
        
        import numpy as np
        X = np.array([[emb_score, kw_score, triple_score, fused_score, float(selected)]])
        pred = self._volume_model.predict(X)[0]
        return float(np.clip(pred, 0, 3))
    
    def _predict_chapter_score(self, emb_score: float, triple_score: float, fused_score: float, selected: bool) -> float:
        """v0.5: Predict Judge score for a chapter candidate using calibration model"""
        if not self._chapter_model:
            return fused_score
        
        import numpy as np
        X = np.array([[emb_score, triple_score, fused_score, float(selected)]])
        pred = self._chapter_model.predict(X)[0]
        return float(np.clip(pred, 0, 3))
    
    def _predict_chunk_score(self, embedding_sim: float, chunk_length: int) -> float:
        """v0.5: Predict Judge score for a chunk candidate using calibration model"""
        if not self._chunk_model:
            return embedding_sim
        
        import numpy as np
        X = np.array([[embedding_sim, chunk_length]])
        pred = self._chunk_model.predict(X)[0]
        return float(np.clip(pred, 0, 3))
        
        # 正規化
        max_score = max(volume_scores.values()) if volume_scores else 1.0
        if max_score > 0:
            volume_scores = {k: v / max_score for k, v in volume_scores.items()}
        
        return volume_scores

    def _aggregate_triple_scores(self, filtered_triples: list) -> tuple[dict, dict]:
        """
        v0.3: filtered_triples の volume/chapter スコアを集約
        
        Args:
            filtered_triples: [(triple_dict, sim_score), ...] の リスト
        
        Returns:
            (vol_scores, chap_scores): 
              vol_scores[vol_id] = 合計スコア
              chap_scores[(vol_id, chap_id)] = 合計スコア
        """
        from collections import defaultdict
        
        vol_scores = defaultdict(float)
        chap_scores = defaultdict(float)
        
        for t, sim in filtered_triples:
            vol_id = t.get("volume_id", "UNKNOWN")
            chap_id = t.get("chapter_id", "UNKNOWN")
            
            vol_scores[vol_id] += sim
            chap_scores[(vol_id, chap_id)] += sim
        
        return dict(vol_scores), dict(chap_scores)

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
        
        # v0.3.1: chunk_id → index マッピングの構築
        if self._chunk_id_to_idx is None:
            self._chunk_id_to_idx = {}
            for idx, chunk in enumerate(self._indices["chunks"]):
                chunk_id = chunk.get("chunk_id")
                if chunk_id is not None:
                    self._chunk_id_to_idx[chunk_id] = idx
        
        # v0.3: triple index のロード
        if self._triple_index is None and self._triple_data is None:
            try:
                import faiss
                triple_json_path = IDX_DIR / "triples.json"
                triple_emb_path = IDX_DIR / "triple_embs.npy"
                
                if triple_json_path.exists() and triple_emb_path.exists():
                    # triple metadata
                    with open(triple_json_path, encoding="utf-8") as f:
                        self._triple_data = json.load(f)
                    
                    # triple embeddings & FAISS index
                    triple_embs = np.load(triple_emb_path)
                    self._triple_index = faiss.IndexFlatIP(triple_embs.shape[1])
                    self._triple_index.add(triple_embs)
                else:
                    # triple index が存在しない場合は None のまま
                    self._triple_data = None
                    self._triple_index = None
            except Exception as e:
                # エラー時は triple filtering なしで動作
                import sys
                print(f"[WARN] Triple index ロード失敗: {e}", file=sys.stderr)
                self._triple_data = None
                self._triple_index = None
        
        # v0.5: Load calibration models
        if self.use_calibration:
            self._load_calibration_models()

    def retrieve(self, query: str) -> list[RetrievedChunk]:
        import numpy as np
        self._init()

        q_vec  = _encode(query)   # (1, dim)

        # ── v0.3: Triple Filtering（クエリごとに実行）──
        # 04_eval_rag.py で既にセットされている場合（None以外）はスキップ（LLM判定済みのため）
        if self._filtered_triples is None:
            if self._triple_index is not None and self._triple_data is not None:
                # クエリに関連する top-N triples を取得（FAISS検索のみ、LLM判定なし）
                n_retrieve = min(self.n_triples, len(self._triple_data))
                D, I = self._triple_index.search(q_vec, n_retrieve)
                
                # filtered_triples = [(triple_dict, similarity_score), ...]
                self._filtered_triples = []
                for idx, score in zip(I[0], D[0]):
                    if 0 <= idx < len(self._triple_data):
                        self._filtered_triples.append((self._triple_data[idx], float(score)))
            else:
                # triple index がない場合は空リスト
                self._filtered_triples = []

        # ── Level 1: ボリューム選択 (embedding + キーワード融合) ──
        vols   = self._indices["volumes_info"]
        v_sim  = (self._vol_vecs @ q_vec.T).flatten()     # (n_vol_total,)
        
        # キーワード + triple スコアの取得と融合
        if self.use_keywords and self._keywords_dict:
            kw_scores = self._compute_keyword_scores(query)
            fusion_cfg = self._keywords_dict.get("fusion_strategy", {})
            emb_weight = fusion_cfg.get("embedding_weight", 0.6)
            kw_weight = fusion_cfg.get("keyword_weight", 0.4)
        else:
            kw_scores = {}
            emb_weight = 1.0
            kw_weight = 0.0
        
        # v0.3: triple スコアを取得
        if self._filtered_triples:
            triple_vol_scores, _ = self._aggregate_triple_scores(self._filtered_triples)
        else:
            triple_vol_scores = {}
        
        # ボリューム名とスコアの融合
        combined_scores = []
        for i, vol_info in enumerate(vols):
            vol_name = vol_info["volume"]
            emb_score = (v_sim[i] + 1.0) / 2.0  # [-1, 1] → [0, 1] に正規化
            kw_score = kw_scores.get(vol_name, 0.0)
            
            # v0.3: triple スコアを統合
            triple_score = triple_vol_scores.get(vol_name, 0.0)
            gamma = 0.2  # triple の重み
            
            combined = emb_weight * emb_score + kw_weight * kw_score + gamma * triple_score
            combined_scores.append(combined)
        
        v_scores = np.array(combined_scores)
        
        # v0.5: Calibration model for Volume selection
        if self.use_calibration and self._volume_model:
            calibrated_v_scores = []
            for i, vol_info in enumerate(vols):
                vol_name = vol_info["volume"]
                emb_score = (v_sim[i] + 1.0) / 2.0
                kw_score = kw_scores.get(vol_name, 0.0)
                triple_score = triple_vol_scores.get(vol_name, 0.0)
                fused_score = combined_scores[i]
                # Predict with temporary selected=True for ranking (actual selection determined after)
                pred_score = self._predict_volume_score(emb_score, kw_score, triple_score, fused_score, selected=False)
                calibrated_v_scores.append(pred_score)
            v_scores = np.array(calibrated_v_scores)
        
        top_v  = np.argsort(-v_scores)[:self.n_vol]
        selected_volumes = [vols[i]["volume"] for i in top_v]

        # ── Level 2: 章選択 ──
        chaps  = self._indices["chapters_info"]
        # 選択ボリュームに属する章のみ
        cand_chap_indices = [
            i for i, c in enumerate(chaps)
            if c["volume"] in selected_volumes
        ]

        # v0.5: chapter情報記録用の変数を初期化
        c_sim_original = np.array([])
        triple_chap_scores_for_log = {}

        if cand_chap_indices and self._chap_vecs.shape[0] > 0:
            cand_vecs = self._chap_vecs[cand_chap_indices]    # (n_cand, dim)
            c_sim     = (cand_vecs @ q_vec.T).flatten()
            c_sim_original = c_sim.copy()  # v0.5: ログ記録用に元のembedding scoreを保存
            
            # v0.3: triple スコアを Chapter 選択に統合
            if self._filtered_triples:
                _, triple_chap_scores = self._aggregate_triple_scores(self._filtered_triples)
                triple_chap_scores_for_log = triple_chap_scores  # v0.5: ログ用に保存
            else:
                triple_chap_scores = {}
            
            gamma_chap = 0.3  # Chapter の triple 重み（Volume より強め）
            
            # triple スコアを Chapter embedding スコアに加算
            c_sim_with_triple = []
            for idx, chap_idx in enumerate(cand_chap_indices):
                chap = chaps[chap_idx]
                key = (chap["volume"], chap["chapter"])
                tscore = triple_chap_scores.get(key, 0.0)
                c_sim_with_triple.append(c_sim[idx] + gamma_chap * tscore)
            
            c_sim = np.array(c_sim_with_triple)
            
            # v0.5: Calibration model for Chapter selection
            if self.use_calibration and self._chapter_model:
                calibrated_c_scores = []
                for idx, chap_idx in enumerate(cand_chap_indices):
                    chap = chaps[chap_idx]
                    key = (chap["volume"], chap["chapter"])
                    emb_score = c_sim_original[idx]
                    triple_score = triple_chap_scores.get(key, 0.0)
                    fused_score = c_sim[idx]
                    pred_score = self._predict_chapter_score(emb_score, triple_score, fused_score, selected=False)
                    calibrated_c_scores.append(pred_score)
                c_sim = np.array(calibrated_c_scores)
            
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

        # v0.3: filtered_triples から chunk_id を直接追加（triple filtering 結果を反映）
        if self._filtered_triples and self._chunk_id_to_idx:
            for t_dict, _ in self._filtered_triples:
                # v0.3.1: chunk_ids (配列) をサポート
                if "chunk_ids" in t_dict:
                    # 配列の場合は全てを追加
                    for cid in t_dict["chunk_ids"]:
                        # chunk_id → index に変換
                        idx = self._chunk_id_to_idx.get(cid)
                        if idx is not None:
                            candidate_ids.add(idx)
                elif "chunk_id" in t_dict:
                    # 単数の場合（後方互換性）
                    chunk_id = t_dict["chunk_id"]
                    idx = self._chunk_id_to_idx.get(chunk_id)
                    if idx is not None:
                        candidate_ids.add(idx)

        # 章候補が少ない場合はボリューム全体の chunk_ids でフォールバック
        if len(candidate_ids) < self.top_k:
            for vol_name in selected_volumes:
                vol_entry = self._indices["hierarchy"]["volumes"].get(vol_name, {})
                candidate_ids.update(vol_entry.get("chunk_ids", []))

        candidate_ids_list = sorted(candidate_ids)
        if not candidate_ids_list:
            # フォールバック: 全チャンク検索（Naive と同じ）
            D, I = self._indices["faiss"].search(q_vec, self.top_k)
            chunks = [self._make_chunk(int(I[0][i]), float(D[0][i])) for i in range(len(I[0]))]
            # v0.5: retrieval_info を記録（fallback mode）
            self._last_retrieval_info = {
                "volumes": [],
                "chapters": [],
                "chunks": [],
                "n_filtered_triples": len(self._filtered_triples) if self._filtered_triples else 0,
                "fallback": True,
            }
            return chunks

        # 候補チャンク内で embedding スコアを計算
        cand_emb = self._indices["embeddings"][candidate_ids_list]   # (n_cand, dim)
        scores   = (cand_emb @ q_vec.T).flatten()
        
        # v0.5: Calibration model for Chunk selection
        if self.use_calibration and self._chunk_model:
            calibrated_chunk_scores = []
            for i, chunk_idx in enumerate(candidate_ids_list):
                chunk_data = self._indices["chunks"][chunk_idx]
                embedding_sim = scores[i]
                chunk_length = len(chunk_data.get("text", ""))
                pred_score = self._predict_chunk_score(embedding_sim, chunk_length)
                calibrated_chunk_scores.append(pred_score)
            scores = np.array(calibrated_chunk_scores)
        
        top_rel  = sorted(range(len(scores)), key=lambda i: -scores[i])[:self.top_k]

        chunks = [
            self._make_chunk(candidate_ids_list[i], float(scores[i]))
            for i in top_rel
        ]
        
        # v0.5: retrieval_info を記録（calibration用）
        # Volume情報
        volume_info = []
        for i, vol in enumerate(vols):
            vol_name = vol["volume"]
            emb_score = (v_sim[i] + 1.0) / 2.0
            kw_score = kw_scores.get(vol_name, 0.0)
            triple_score = triple_vol_scores.get(vol_name, 0.0)
            fused_score = combined_scores[i]
            volume_info.append({
                "volume_id": vol_name,
                "emb_score": float(emb_score),
                "kw_score": float(kw_score),
                "triple_score": float(triple_score),
                "fused_score": float(fused_score),
                "selected": vol_name in selected_volumes,
            })
        
        # Chapter情報
        chapter_info = []
        if cand_chap_indices:
            gamma_chap = 0.3  # Chapter の triple 重み
            for chap_idx in cand_chap_indices:
                chap = chaps[chap_idx]
                vol_name = chap["volume"]
                chap_name = chap["chapter"]
                # embedding score (元の値を使用)
                idx_in_cand = cand_chap_indices.index(chap_idx)
                emb_score = float(c_sim_original[idx_in_cand]) if idx_in_cand < len(c_sim_original) else 0.0
                # triple score
                key = (vol_name, chap_name)
                triple_score = triple_chap_scores_for_log.get(key, 0.0)
                # fused score (embedding + gamma_chap * triple)
                fused_score = emb_score + gamma_chap * triple_score
                chapter_info.append({
                    "volume_id": vol_name,
                    "chapter_id": chap_name,
                    "emb_score": emb_score,
                    "triple_score": float(triple_score),
                    "fused_score": float(fused_score),
                    "selected": chap in selected_chapters,
                })
        
        # Chunk情報（top-kのみ）
        chunk_info = []
        for i in top_rel:
            chunk_idx = candidate_ids_list[i]
            chunk_data = self._indices["chunks"][chunk_idx]
            chunk_info.append({
                "chunk_id": chunk_data.get("chunk_id"),
                "volume_id": chunk_data.get("doc_id", "").split("/")[0] if "/" in chunk_data.get("doc_id", "") else "",
                "chapter_id": chunk_data.get("heading", ""),
                "embedding_sim": float(scores[i]),
                "chunk_length": len(chunk_data.get("text", "")),
            })
        
        # 計算に用いたtriple統計
        avg_triple_sim = 0.0
        if self._filtered_triples:
            triple_sims = [sim for _, sim in self._filtered_triples]
            avg_triple_sim = sum(triple_sims) / len(triple_sims) if triple_sims else 0.0
        
        self._last_retrieval_info = {
            "volumes": volume_info,
            "chapters": chapter_info,
            "chunks": chunk_info,
            "n_filtered_triples": len(self._filtered_triples) if self._filtered_triples else 0,
            "avg_triple_sim": float(avg_triple_sim),
            "fallback": False,
        }
        
        return chunks

    @property
    def retrieval_info(self) -> dict:
        """v0.5 calibration用: 最後の検索でのVolume/Chapter/Chunk情報"""
        return getattr(self, "_last_retrieval_info", {})


# ─────────────────────────────────────────────────────────
# ファクトリ
# ─────────────────────────────────────────────────────────

def make_retriever(rag_type: str, top_k: int = 5, keywords_file: str | None = None, n_triples: int = 20, use_calibration: bool = False, calibration_dir: str | None = None) -> BaseRetriever:
    rag_type = rag_type.lower()
    if rag_type == "naive":
        return NaiveRetriever(top_k=top_k, use_calibration=use_calibration, calibration_dir=calibration_dir)
    elif rag_type == "light":
        return LightRetriever(top_k=top_k, use_calibration=use_calibration, calibration_dir=calibration_dir)
    elif rag_type in ("hipporag2", "hippo"):
        return HippoRAG2Retriever(top_k=top_k, keywords_file=keywords_file, n_triples=n_triples, use_calibration=use_calibration, calibration_dir=calibration_dir)
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
