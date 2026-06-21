"""
experiments/04e_similarity_only.py
──────────────────────────────────────────────────────────
Phase 4e: 類似度ベース評価（Judge不要）

推論済み結果に対して、生成テキストと正解答のコサイン類似度を計算し、
RAG精度を評価します。

特徴:
- Ollamaの呼び出しが不要（Judge採点より高速）
- embedding モデルを使用したセマンティック評価
- 6条件全体の比較が容易

Usage:
    python experiments/04e_similarity_only.py                    # 全条件を評価
    python experiments/04e_similarity_only.py --condition swallow_naive  # 特定条件のみ
    python experiments/04e_similarity_only.py --batch-size 16    # バッチサイズ指定
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import NamedTuple

import numpy as np

# Windows UTF-8 設定
os.environ.setdefault("PYTHONUTF8", "1")
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).parent.parent
EXP_DIR = Path(__file__).parent
RESULTS_DIR = EXP_DIR / "results"
EVALS_DIR = EXP_DIR / "evals"
TEST_FILE = EXP_DIR / "testset_200.jsonl"

EVALS_DIR.mkdir(parents=True, exist_ok=True)

EMBED_MODEL_NAME = "hotchpotch/static-embedding-japanese"


# ─────────────────────────────────────────────────────────
# Embedding とコサイン類似度
# ─────────────────────────────────────────────────────────

class SimilarityResult(NamedTuple):
    idx: int
    similarity: float
    prediction: str
    gt_answer: str


def _load_embedding_model():
    """sentence-transformers を使用して embedding モデルをロード"""
    from sentence_transformers import SentenceTransformer
    print(f"  Embedding モデルロード中: {EMBED_MODEL_NAME}")
    model = SentenceTransformer(EMBED_MODEL_NAME)
    print(f"  ✓ Embedding モデルロード完了  (次元数: {model.get_sentence_embedding_dimension()})")
    return model


def _cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """コサイン類似度を計算"""
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    if norm1 == 0.0 or norm2 == 0.0:
        return 0.0
    return float(np.dot(vec1, vec2) / (norm1 * norm2))


def _batch_embed(model, texts: list[str], batch_size: int = 32) -> np.ndarray:
    """バッチでテキストを embedding"""
    embeddings = model.encode(texts, batch_size=batch_size, show_progress_bar=False)
    return np.array(embeddings, dtype="float32")


# ─────────────────────────────────────────────────────────
# テストセット / 結果ロード
# ─────────────────────────────────────────────────────────

def load_testset(path: Path) -> list[dict]:
    """テストセットをロード"""
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            l = line.strip()
            if l:
                records.append(json.loads(l))
    return records


def load_results(path: Path) -> list[dict]:
    """結果ファイル（*_results.jsonl）をロード"""
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            l = line.strip()
            if l:
                records.append(json.loads(l))
    return records


# ─────────────────────────────────────────────────────────
# 類似度計算メイン処理
# ─────────────────────────────────────────────────────────

def evaluate_similarity(result_file: Path, model, testset: list[dict], batch_size: int) -> dict:
    """1つの結果ファイルに対して類似度評価を実行"""
    
    condition = result_file.stem.replace("_results", "")
    model_name, rag_name = condition.split("_", 1)
    
    # 結果をロード
    results = load_results(result_file)
    
    print(f"  結果ファイルロード: {len(results)} 件")
    
    if not results:
        print(f"  ⚠ 結果が空です")
        return None
    
    # テストセット（質問と正解答）を準備
    questions = [rec["question"] for rec in testset]
    gt_answers = [rec.get("answer", "") for rec in testset]
    predictions = [rec.get("answer", "") for rec in results]
    
    if len(predictions) != len(questions):
        print(f"  ⚠ 質問数({len(questions)}) と 結果数({len(predictions)}) が不一致")
        predictions = predictions[:len(questions)]
    
    print(f"  Embedding 計算中 ({len(questions) + len(gt_answers) + len(predictions)} テキスト)...")
    
    # 全テキストを embedding
    all_texts = questions + gt_answers + predictions
    all_embeds = _batch_embed(model, all_texts, batch_size=batch_size)
    
    q_embeds = all_embeds[:len(questions)]
    gt_embeds = all_embeds[len(questions):len(questions)+len(gt_answers)]
    pred_embeds = all_embeds[len(questions)+len(gt_answers):]
    
    print(f"  類似度計算中...")
    
    # 類似度を計算
    similarities = []
    for i in range(len(predictions)):
        sim = _cosine_similarity(pred_embeds[i], gt_embeds[i])
        similarities.append(SimilarityResult(
            idx=i,
            similarity=sim,
            prediction=predictions[i][:200],  # 最初の200文字のみ保存
            gt_answer=gt_answers[i][:200]
        ))
    
    # 統計情報を計算
    sim_values = [s.similarity for s in similarities]
    sim_array = np.array(sim_values)
    
    stats = {
        "n_questions": len(similarities),
        "avg_similarity": float(np.mean(sim_array)),
        "std_similarity": float(np.std(sim_array)),
        "min_similarity": float(np.min(sim_array)),
        "max_similarity": float(np.max(sim_array)),
        "median_similarity": float(np.median(sim_array)),
        "percentiles": {
            "25": float(np.percentile(sim_array, 25)),
            "50": float(np.percentile(sim_array, 50)),
            "75": float(np.percentile(sim_array, 75)),
            "90": float(np.percentile(sim_array, 90)),
            "95": float(np.percentile(sim_array, 95)),
        },
        "similarity_dist": {
            "0.0-0.2": sum(1 for s in sim_values if 0.0 <= s < 0.2),
            "0.2-0.4": sum(1 for s in sim_values if 0.2 <= s < 0.4),
            "0.4-0.6": sum(1 for s in sim_values if 0.4 <= s < 0.6),
            "0.6-0.8": sum(1 for s in sim_values if 0.6 <= s < 0.8),
            "0.8-1.0": sum(1 for s in sim_values if 0.8 <= s <= 1.0),
        }
    }
    
    # 結果ファイルに保存
    eval_file = EVALS_DIR / f"{condition}_similarity.json"
    eval_data = {
        "condition": condition,
        "model": model_name,
        "rag": rag_name,
        "embedding_model": EMBED_MODEL_NAME,
        "statistics": stats,
        "similarities": [
            {
                "idx": s.idx,
                "similarity": s.similarity,
                "prediction_preview": s.prediction,
                "gt_answer_preview": s.gt_answer,
            }
            for s in similarities
        ],
        "timestamp": datetime.now().isoformat(),
    }
    
    with open(eval_file, "w", encoding="utf-8") as f:
        json.dump(eval_data, f, ensure_ascii=False, indent=2)
    
    print(f"  ✓ 評価結果保存: {eval_file.name}")
    print(f"  📊 平均類似度: {stats['avg_similarity']:.3f}  "
          f"(σ={stats['std_similarity']:.3f}, min={stats['min_similarity']:.3f}, max={stats['max_similarity']:.3f})")
    
    return eval_data


# ─────────────────────────────────────────────────────────
# メイン
# ─────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="類似度ベース評価")
    parser.add_argument("--condition", default=None, help="特定条件のみ評価 (例: swallow_naive)")
    parser.add_argument("--batch-size", type=int, default=32, help="Embedding バッチサイズ")
    args = parser.parse_args()
    
    # テストセットをロード
    if not TEST_FILE.exists():
        print(f"❌ テストセットが見つかりません: {TEST_FILE}")
        sys.exit(1)
    
    testset = load_testset(TEST_FILE)
    print(f"✓ テストセットロード: {len(testset)} 件")
    
    # embedding モデルをロード
    print()
    model = _load_embedding_model()
    print()
    
    # 対象ファイルを検索
    if args.condition:
        target_files = [RESULTS_DIR / f"{args.condition}_results.jsonl"]
        target_files = [f for f in target_files if f.exists()]
    else:
        target_files = sorted(RESULTS_DIR.glob("*_results.jsonl"))
    
    if not target_files:
        print("❌ 評価対象の結果ファイルが見つかりません")
        sys.exit(1)
    
    print("=" * 60)
    print(f"  類似度評価: {len(target_files)} 条件")
    print(f"  Embedding: {EMBED_MODEL_NAME}")
    print("=" * 60)
    
    start_time = time.time()
    
    all_results = []
    for i, result_file in enumerate(target_files, 1):
        print()
        print("-" * 60)
        print(f"  [{i}/{len(target_files)}] {result_file.stem}")
        print("-" * 60)
        try:
            eval_data = evaluate_similarity(result_file, model, testset, args.batch_size)
            if eval_data:
                all_results.append(eval_data)
        except Exception as e:
            print(f"  ❌ エラー: {e}")
            import traceback
            traceback.print_exc()
    
    elapsed = round((time.time() - start_time) / 60, 1)
    
    # サマリーを出力
    print()
    print("=" * 60)
    print(f"  ✓ 評価完了: {len(all_results)} 条件  ({elapsed} 分)")
    print("=" * 60)
    
    if all_results:
        print("\n📊 全条件の平均類似度ランキング:")
        sorted_results = sorted(all_results, key=lambda x: x["statistics"]["avg_similarity"], reverse=True)
        for i, res in enumerate(sorted_results, 1):
            cond = res["condition"]
            avg_sim = res["statistics"]["avg_similarity"]
            print(f"  [{i}] {cond:25s} : {avg_sim:.4f}")
    
    print("=" * 60)


if __name__ == "__main__":
    main()
