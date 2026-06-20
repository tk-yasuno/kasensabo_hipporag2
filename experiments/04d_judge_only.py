#!/usr/bin/env python
"""
experiments/04d_judge_only.py
────────────────────────────────────────────────────────────
Phase 4d: Judge 採点のみ実行（推論済み結果に対して）

推論済みの results/{model}_{rag}_results.jsonl ファイルに対して
Judge採点を実行し、結果を上書き更新します。

Usage:
    python experiments/04d_judge_only.py                    # 全条件の未採点結果を採点
    python experiments/04d_judge_only.py --condition swallow_naive  # 特定条件のみ
    python experiments/04d_judge_only.py --judge-workers 4  # 並列数指定
    python experiments/04d_judge_only.py --force            # 既採点も再採点
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import httpx

# Windows UTF-8 設定
os.environ.setdefault("PYTHONUTF8", "1")
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).parent.parent
EXP_DIR = Path(__file__).parent
RESULTS_DIR = EXP_DIR / "results"
ENV_CONFIG = EXP_DIR / "env_config.json"

OLLAMA_URL = "http://localhost:11434"

# Judge モデル候補
JUDGE_CANDIDATES = ["qwen2.5:14b", "qwen2.5:7b-instruct-q4_k_m", "qwen2.5:7b",
                    "qwen2.5:7b-instruct", "qwen2.5"]


# ─────────────────────────────────────────────────────────
# Judge プロンプト
# ─────────────────────────────────────────────────────────

JUDGE_SYSTEM = """あなたは河川砂防技術の専門家として、AI が生成した回答を採点します。
以下の採点基準に従い、必ず次の 2 行形式のみで返答してください。
他の文章は一切不要です。

採点基準:
  3点: 技術的に正確で具体的、根拠となる基準名・章番号・技術概念が含まれる
  2点: 概ね正確だが、根拠・具体性がやや不足
  1点: 部分的に正しいが、重要な誤り・不足がある
  0点: 回答なし、または技術的に大きく誤っている

必ず次の 2 行形式のみで返答:
SCORE: <0〜3の整数>
REASON: <50字以内の日本語での採点理由>"""

JUDGE_USER_TEMPLATE = """【質問】
{question}

【回答】
{answer}

SCORE: と REASON: の 2 行形式で採点してください。"""


# ─────────────────────────────────────────────────────────
# Ollama 呼び出し
# ─────────────────────────────────────────────────────────

def _ollama_chat(
    model: str,
    system: str,
    user: str,
    timeout: float = 300.0,
    max_tokens: int = 512,
    temperature: float = 0.3,
    keep_alive: str = "5m",
) -> str:
    payload = {
        "model":      model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user",   "content": user},
        ],
        "stream":     False,
        "keep_alive": keep_alive,
        "options": {
            "temperature":    temperature,
            "num_predict":    max_tokens,
            "num_ctx":        4096,
            "repeat_penalty": 1.1,
            "num_gpu":        99,
        },
    }
    timeout_cfg = httpx.Timeout(connect=15.0, read=timeout, write=15.0, pool=5.0)
    resp = httpx.post(f"{OLLAMA_URL}/api/chat", json=payload, timeout=timeout_cfg)
    resp.raise_for_status()
    return resp.json().get("message", {}).get("content", "")


def _parse_judge(text: str) -> tuple[int, str]:
    score_m  = re.search(r"SCORE:\s*([0-3])", text)
    reason_m = re.search(r"REASON:\s*(.+)", text)
    score    = int(score_m.group(1)) if score_m else -1
    reason   = reason_m.group(1).strip()[:80] if reason_m else text[:80]
    return score, reason


def _judge_one(idx: int, question: str, answer: str, model: str, is_last: bool) -> tuple[int, int, str, str]:
    """1問のJudge採点を実行（並列処理用）"""
    judge_prompt = JUDGE_USER_TEMPLATE.format(
        question=question,
        answer=answer or "(空回答)",
    )
    try:
        judge_text = _ollama_chat(
            model       = model,
            system      = JUDGE_SYSTEM,
            user        = judge_prompt,
            timeout     = 180.0,
            max_tokens  = 128,
            temperature = 0.0,
            keep_alive  = "0" if is_last else "30m",
        )
        score, reason = _parse_judge(judge_text)
    except Exception as e:
        score, reason = -1, str(e)[:50]
    return idx, score, reason, answer


def _pick_judge_model() -> str:
    """利用可能なJudgeモデルを選択"""
    env_cfg = {}
    if ENV_CONFIG.exists():
        env_cfg = json.loads(ENV_CONFIG.read_text(encoding="utf-8"))
    
    if "model_judge" in env_cfg and env_cfg["model_judge"]:
        return env_cfg["model_judge"]
    
    try:
        r = httpx.get(f"{OLLAMA_URL}/api/tags", timeout=10.0)
        available = [m["name"] for m in r.json().get("models", [])]
        low_avail = [m.lower() for m in available]
        for c in JUDGE_CANDIDATES:
            if c in available:
                return c
            c_base = c.split(":")[0].lower()
            for orig, low in zip(available, low_avail):
                if low.startswith(c_base):
                    return orig
    except Exception:
        pass
    return JUDGE_CANDIDATES[0]


# ─────────────────────────────────────────────────────────
# Judge 採点メイン処理
# ─────────────────────────────────────────────────────────

def judge_one_file(result_file: Path, judge_model: str, workers: int, force: bool):
    """1つの結果ファイルに対してJudge採点を実行"""
    
    # 結果読み込み
    results = []
    with open(result_file, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                results.append(json.loads(line))
    
    # 未採点の結果を抽出
    if force:
        targets = results
        print(f"  全 {len(targets)} 問を再採点")
    else:
        targets = [r for r in results if r.get("judge_score", -1) == -1]
        if not targets:
            print(f"  ✓ 全問採点済み（スキップ）")
            return
        print(f"  未採点 {len(targets)}/{len(results)} 問を採点")
    
    # Ollamaウォームアップ
    print(f"  Ollamaモデル {judge_model} をGPUにロード中...")
    try:
        warmup_text = _ollama_chat(
            model=judge_model,
            system="test",
            user="こんにちは",
            timeout=60.0,
            max_tokens=10,
            temperature=0.0,
            keep_alive="30m",
        )
        print(f"  ✓ Ollamaモデルロード完了")
    except Exception as e:
        print(f"  ⚠ Ollamaモデルロード警告: {e}")
    
    # Judge採点（並列実行）
    completed = 0
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {}
        for i, res in enumerate(targets):
            is_last = (i == len(targets) - 1)
            future = executor.submit(
                _judge_one,
                res["idx"],
                res["question"],
                res["answer"],
                judge_model,
                is_last
            )
            futures[future] = res
        
        for future in as_completed(futures):
            res = futures[future]
            try:
                idx, score, reason, answer = future.result()
                res["judge_score"]  = score
                res["judge_reason"] = reason
                completed += 1
                print(f"  [{completed}/{len(targets)}] idx={idx}  score={score}  {reason[:40]}")
            except Exception as e:
                print(f"  [ERROR] Judge失敗 idx={res['idx']}: {e}")
                res["judge_score"]  = -1
                res["judge_reason"] = str(e)[:50]
                completed += 1
    
    # 結果を上書き保存
    with open(result_file, "w", encoding="utf-8") as f:
        for res in results:
            f.write(json.dumps(res, ensure_ascii=False) + "\n")
    print(f"  ✓ 結果を保存: {result_file.name}")
    
    # サマリー更新
    condition = result_file.stem.replace("_results", "")
    model_name, rag_name = condition.split("_", 1)
    
    valid = [r for r in results if r.get("judge_score", -1) >= 0]
    avg_score  = sum(r["judge_score"] for r in valid) / len(valid) if valid else 0
    perfect    = sum(1 for r in valid if r["judge_score"] == 3)
    avg_ret    = sum(r.get("ret_time", 0)  for r in results) / len(results) if results else 0
    avg_gen    = sum(r.get("gen_time", 0)  for r in results) / len(results) if results else 0
    score_dist = {k: sum(1 for r in valid if r["judge_score"] == k) for k in range(4)}
    
    summary = {
        "condition":      condition,
        "model":          model_name,
        "rag":            rag_name,
        "judge_model":    judge_model,
        "n_questions":    len(results),
        "n_judged":       len(valid),
        "avg_score":      round(avg_score, 3),
        "perfect_count":  perfect,
        "perfect_rate":   round(perfect / len(valid) * 100, 1) if valid else 0,
        "score_dist":     score_dist,
        "avg_ret_time":   round(avg_ret, 3),
        "avg_gen_time":   round(avg_gen, 3),
        "avg_total_time": round(avg_ret + avg_gen, 3),
    }
    
    summary_file = RESULTS_DIR / f"{condition}_summary.json"
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"  ✓ サマリー保存: {summary_file.name}")
    print(f"  📊 平均スコア: {avg_score:.2f}/3  完答率: {summary['perfect_rate']}%")


# ─────────────────────────────────────────────────────────
# メイン
# ─────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Judge採点のみ実行")
    parser.add_argument("--condition", default=None, help="特定条件のみ採点 (例: swallow_naive)")
    parser.add_argument("--judge-workers", type=int, default=4, help="Judge並列数")
    parser.add_argument("--force", action="store_true", help="既採点も再採点")
    args = parser.parse_args()
    
    judge_model = _pick_judge_model()
    
    # 対象ファイルを検索
    if args.condition:
        target_files = [RESULTS_DIR / f"{args.condition}_results.jsonl"]
        target_files = [f for f in target_files if f.exists()]
    else:
        target_files = sorted(RESULTS_DIR.glob("*_results.jsonl"))
    
    if not target_files:
        print("❌ 採点対象の結果ファイルが見つかりません")
        sys.exit(1)
    
    print("=" * 60)
    print(f"  Judge採点: {len(target_files)} 条件")
    print(f"  Judge model: {judge_model}")
    print(f"  Workers: {args.judge_workers}")
    print(f"  再採点: {args.force}")
    print("=" * 60)
    
    start_time = time.time()
    
    for i, result_file in enumerate(target_files, 1):
        print()
        print("-" * 60)
        print(f"  [{i}/{len(target_files)}] {result_file.stem}")
        print("-" * 60)
        try:
            judge_one_file(result_file, judge_model, args.judge_workers, args.force)
        except Exception as e:
            print(f"  ❌ エラー: {e}")
            import traceback
            traceback.print_exc()
    
    elapsed = round((time.time() - start_time) / 60, 1)
    print()
    print("=" * 60)
    print(f"  ✓ Judge採点完了: {len(target_files)} 条件  ({elapsed} 分)")
    print("=" * 60)


if __name__ == "__main__":
    main()
