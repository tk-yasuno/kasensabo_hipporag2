"""
experiments/04_eval_rag.py
────────────────────────────────────────────────────────────
Phase 4: 評価パイプライン — 1条件（model × rag方式）を実行

戦略（16GB VRAM を守る）:
  Phase 1: Unsloth (FastLanguageModel) + バッチ推論 (batch_size=8)
           → GPU 4-5GB, ~10s/問  最後にモデルアンロードして VRAM 開放
  Phase 2: Ollama (qwen2.5:14b) で全問 Judge 採点

出力:
  experiments/results/{model}_{rag}_results.jsonl
  experiments/results/{model}_{rag}_summary.json

Usage:
    python experiments/04_eval_rag.py --model swallow --rag naive
    python experiments/04_eval_rag.py --model elyza   --rag light
    python experiments/04_eval_rag.py --model swallow --rag hipporag2
    python experiments/04_eval_rag.py --model swallow --rag naive --dry-run   # 10件のみ
    python experiments/04_eval_rag.py --model swallow --rag naive --no-judge  # 推論のみ
    python experiments/04_eval_rag.py --model swallow --rag naive --batch-size 8

必須環境変数（自動設定済み）:
    UNSLOTH_COMPILE_DISABLE=1  (Windows での torch.compile 無効化)
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

# Windows で torch.compile (triton JIT) を無効化（必須）
os.environ.setdefault("UNSLOTH_COMPILE_DISABLE", "1")

import torch

ROOT = Path(__file__).parent.parent

# 実験設定読み込み（00_check_env.py が生成した env_config.json を優先使用）
EXP_DIR    = Path(__file__).parent
ENV_CONFIG = EXP_DIR / "env_config.json"
RESULTS_DIR = EXP_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
TEST_FILE  = EXP_DIR / "testset_200.jsonl"

os.environ.setdefault("PYTHONUTF8", "1")
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import httpx

# ─────────────────────────────────────────────────────────
# モデル設定（定数）
# ─────────────────────────────────────────────────────────

# Unsloth LoRA アダプタパス
MODEL_PATHS = {
    "swallow": ROOT / "models" / "swallow8b_merged_n4000_r32_d05",
    "elyza":   ROOT / "models" / "elyza8b_merged_n4000",
}

# Unsloth 推論設定
MAX_SEQ_LEN    = 1024  # 速度最適化
MAX_NEW_TOKENS = 384   # バランス型（品質と速度の中間）


# ─────────────────────────────────────────────────────────
# Unsloth バッチ推論
# ─────────────────────────────────────────────────────────

def _format_llama3_prompt(system: str, user: str) -> str:
    """Llama-3 Instruct チャットテンプレートでプロンプトを生成"""
    return (
        "<|begin_of_text|>"
        f"<|start_header_id|>system<|end_header_id|>\n\n{system}<|eot_id|>"
        f"<|start_header_id|>user<|end_header_id|>\n\n{user}<|eot_id|>"
        "<|start_header_id|>assistant<|end_header_id|>\n\n"
    )


def _unsloth_batch_infer(
    model,
    tokenizer,
    prompts: list[str],
    max_new_tokens: int = MAX_NEW_TOKENS,
) -> list[str]:
    """Unsloth FastLanguageModel でバッチ推論。生成部分のみを返す。"""
    inputs = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=MAX_SEQ_LEN,
    ).to("cuda")
    input_len = inputs["input_ids"].shape[1]
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    results = []
    for out in outputs:
        generated = tokenizer.decode(out[input_len:], skip_special_tokens=True)
        results.append(generated.strip())
    return results


# ─────────────────────────────────────────────────────────
# モデル設定（Ollama Judge 用）
# ─────────────────────────────────────────────────────────

# Ollama モデル名（Judge 採点用）
MODEL_CANDIDATES = {
    "swallow": ["swallow8b-lora-n4000-v09-q4", "swallow8b-lora-n4000-v09"],
    "elyza":   ["elyza8b-lora-n4000-q4",       "elyza8b-lora-n4000"],
}
JUDGE_CANDIDATES = ["qwen2.5:14b", "qwen2.5:7b-instruct-q4_k_m", "qwen2.5:7b",
                    "qwen2.5:7b-instruct", "qwen2.5"]

OLLAMA_URL = "http://localhost:11434"


def _load_env_config() -> dict:
    if ENV_CONFIG.exists():
        return json.loads(ENV_CONFIG.read_text(encoding="utf-8"))
    return {}


def _pick_model(role: str, candidates: list[str], env_cfg: dict) -> str:
    # env_config.json に設定済みなら優先
    cfg_key = f"model_{role}"
    if cfg_key in env_cfg and env_cfg[cfg_key]:
        return env_cfg[cfg_key]

    # Ollama に問い合わせて存在確認
    try:
        r = httpx.get(f"{OLLAMA_URL}/api/tags", timeout=10.0)
        available = [m["name"] for m in r.json().get("models", [])]
        low_avail = [m.lower() for m in available]
        for c in candidates:
            if c in available:
                return c
            c_base = c.split(":")[0].lower()
            for orig, low in zip(available, low_avail):
                if low.startswith(c_base):
                    return orig
    except Exception:
        pass
    return candidates[0]  # フォールバック（存在しなくてもエラーは呼び出し時）


# ─────────────────────────────────────────────────────────
# プロンプト定義
# ─────────────────────────────────────────────────────────

SYSTEM_PROMPT_RAG = (
    "あなたは河川砂防技術基準（調査・計画・設計・維持管理）を熟知した専門家です。"
    "以下の「参照文書」に記載がある場合はそれを優先して引用し、"
    "記載がない場合は河川砂防技術基準の知識に基づいて回答してください。"
)

CONTEXT_TEMPLATE = """\
「参照文書」
{rag_context}

上記の参照文書を踏まえて、以下の質問に正確かつ実務的に答えてください。
「質問」
{question}"""

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
            "num_gpu":        99,  # 全レイヤーをGPUにロード
        },
    }
    timeout_cfg = httpx.Timeout(connect=15.0, read=timeout, write=15.0, pool=5.0)
    resp = httpx.post(f"{OLLAMA_URL}/api/chat", json=payload, timeout=timeout_cfg)
    resp.raise_for_status()
    return resp.json().get("message", {}).get("content", "")


# ─────────────────────────────────────────────────────────
# Judge スコアパース
# ─────────────────────────────────────────────────────────

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
            timeout     = 180.0,  # 3分に延長（並列処理でタイムアウト対策）
            max_tokens  = 128,
            temperature = 0.0,
            keep_alive  = "0" if is_last else "30m",  # 処理中はモデルをメモリに保持
        )
        score, reason = _parse_judge(judge_text)
    except Exception as e:
        score, reason = -1, str(e)[:50]
    return idx, score, reason, answer


# ─────────────────────────────────────────────────────────
# テストセットロード
# ─────────────────────────────────────────────────────────

def load_testset(path: Path) -> list[dict]:
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            l = line.strip()
            if l:
                records.append(json.loads(l))
    return records


# ─────────────────────────────────────────────────────────
# main
# ─────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",     required=True, choices=["swallow", "elyza"],
                        help="生成モデル")
    parser.add_argument("--rag",       required=True,
                        choices=["naive", "light", "hipporag2"],
                        help="RAG 方式")
    parser.add_argument("--top-k",     type=int,   default=5,    help="取得チャンク数")
    parser.add_argument("--max-chars", type=int,   default=2000, help="コンテキスト文字数上限")
    parser.add_argument("--timeout",    type=float, default=300.0)
    parser.add_argument("--batch-size", type=int,   default=8,   help="Unsloth バッチサイズ")
    parser.add_argument("--judge-workers", type=int, default=4, help="Judge採点の並列数")
    parser.add_argument("--no-judge",   action="store_true",    help="Judge 採点をスキップ")
    parser.add_argument("--dry-run",    action="store_true",    help="10件のみ実行（動作確認）")
    args = parser.parse_args()

    print(f"=" * 60)
    print(f"  RAG 評価: model={args.model}  rag={args.rag}  top_k={args.top_k}  batch={args.batch_size}")
    print(f"=" * 60)

    # ── モデルパス解決 ──
    env_cfg     = _load_env_config()
    model_path  = MODEL_PATHS[args.model]
    judge_model = _pick_model("judge", JUDGE_CANDIDATES, env_cfg)
    print(f"  推論モデル: {model_path}")
    print(f"  Judge モデル: {judge_model}")

    if not model_path.exists():
        print(f"ERROR: モデルが見つかりません: {model_path}")
        sys.exit(1)

    # ── テストセットロード ──
    if not TEST_FILE.exists():
        print(f"ERROR: テストセットが見つかりません: {TEST_FILE}")
        print("  先に 02_prepare_testset.py を実行してください。")
        sys.exit(1)

    records = load_testset(TEST_FILE)
    if args.dry_run:
        records = records[:10]
        print(f"  [dry-run] 10件のみ実行")
    print(f"  テスト件数: {len(records)} 問")

    # ── RAG Retriever 初期化 ──
    sys.path.insert(0, str(EXP_DIR))
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "rag_retrievers", EXP_DIR / "03_rag_retrievers.py"
    )
    rag_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(rag_mod)
    make_retriever = rag_mod.make_retriever
    format_context = rag_mod.format_context
    print(f"\nRAG インデックスロード中 ({args.rag})...")
    retriever = make_retriever(args.rag, top_k=args.top_k)

    # ── Phase 1: Unsloth バッチ推論 ──
    print(f"\n[Phase 1] Unsloth バッチ推論 (batch_size={args.batch_size})...")
    print(f"  モデルロード中: {model_path}")
    from unsloth import FastLanguageModel
    unsloth_model, unsloth_tokenizer = FastLanguageModel.from_pretrained(
        model_name=str(model_path),
        max_seq_length=MAX_SEQ_LEN,
        load_in_4bit=True,
        dtype=None,
    )
    FastLanguageModel.for_inference(unsloth_model)
    unsloth_tokenizer.padding_side = "left"
    print(f"  モデルロード完了  GPU MB: {round(torch.cuda.memory_allocated()/1024**2)}")

    n_total = len(records)
    infer_results: list[dict] = []

    # 全問の Retrieval とプロンプト生成
    all_ret_data = []
    print(f"  Retrieval 実行中...")
    for rec in records:
        t0 = time.time()
        chunks = retriever.retrieve(rec["question"])
        t_ret  = time.time() - t0
        context  = format_context(chunks, max_chars=args.max_chars)
        user_msg = CONTEXT_TEMPLATE.format(rag_context=context, question=rec["question"])
        prompt   = _format_llama3_prompt(SYSTEM_PROMPT_RAG, user_msg)
        all_ret_data.append({
            "rec": rec, "chunks": chunks, "prompt": prompt, "t_ret": t_ret
        })

    # バッチ推論
    print(f"  バッチ推論中 ({n_total}問 / batch={args.batch_size})...")
    t_gen_total_start = time.time()
    all_answers: list[str] = []

    for batch_start in range(0, n_total, args.batch_size):
        batch_data = all_ret_data[batch_start: batch_start + args.batch_size]
        prompts    = [d["prompt"] for d in batch_data]
        t0 = time.time()
        try:
            answers = _unsloth_batch_infer(unsloth_model, unsloth_tokenizer, prompts)
        except Exception as e:
            print(f"  [ERROR] バッチ推論失敗 batch={batch_start}: {e}")
            answers = ["" for _ in prompts]
        t_batch = time.time() - t0
        per_q   = t_batch / len(prompts)
        end_idx = min(batch_start + args.batch_size, n_total)
        print(f"  [{batch_start+1}-{end_idx}/{n_total}] {t_batch:.1f}s total  {per_q:.2f}s/問")
        all_answers.extend(answers)

    t_gen_avg = (time.time() - t_gen_total_start) / n_total

    # 結果マージ
    for d, answer in zip(all_ret_data, all_answers):
        rec = d["rec"]
        infer_results.append({
            "idx":          rec["idx"],
            "question":     rec["question"],
            "gt_answer":    rec.get("answer", ""),
            "source":       rec.get("source", ""),
            "answer":       answer,
            "answer_len":   len(answer),
            "ret_time":     round(d["t_ret"], 3),
            "gen_time":     round(t_gen_avg, 3),
            "total_time":   round(d["t_ret"] + t_gen_avg, 3),
            "retrieved_chunks": [
                {"chunk_id": c.chunk_id, "score": round(c.score, 4),
                 "doc_id": c.doc_id, "heading": c.heading}
                for c in d["chunks"]
            ],
            "judge_score":  -1,
            "judge_reason": "",
        })

    # Unsloth モデルアンロード（Phase 2 の前に VRAM 解放）
    print(f"\n  モデルアンロード中...")
    print(f"  アンロード前 GPU MB: {round(torch.cuda.memory_allocated()/1024**2)}")
    
    # モデル削除
    del unsloth_model, unsloth_tokenizer
    
    # RAG retrieverもGPUメモリを使用している可能性があるためアンロード
    del retriever
    
    # 完全なメモリ解放
    gc.collect()
    torch.cuda.synchronize()  # GPU操作の完了を待機
    torch.cuda.empty_cache()
    gc.collect()  # 2回目のGC
    
    # 待機時間を追加（OSレベルのメモリ解放を待つ）
    time.sleep(2)
    
    print(f"  VRAM 解放完了  残 GPU MB: {round(torch.cuda.memory_allocated()/1024**2)}")
    print(f"  平均 {t_gen_avg:.2f}s/問")

    # ── Phase 2: Judge 採点（並列処理）──
    if not args.no_judge:
        print(f"\n[Phase 2] Judge 採点 ({judge_model}) - {args.judge_workers}並列...")
        
        # Ollamaモデルのウォームアップ（GPU確認）
        print(f"  Ollamaモデル {judge_model} をGPUにロード中...")
        try:
            warmup_text = _ollama_chat(
                model=judge_model,
                system="test",
                user="こんにちは",
                timeout=60.0,
                max_tokens=10,
                temperature=0.0,
                keep_alive="30m",  # Judge処理中はモデルをメモリに保持
            )
            print(f"  ✓ Ollamaモデルロード完了（ウォームアップ成功）")
        except Exception as e:
            print(f"  ⚠ Ollamaモデルロード警告: {e}")
        
        total = len(infer_results)
        completed = 0
        
        with ThreadPoolExecutor(max_workers=args.judge_workers) as executor:
            # 全タスクをサブミット
            futures = {}
            for i, res in enumerate(infer_results):
                is_last = (i == total - 1)
                future = executor.submit(
                    _judge_one,
                    res["idx"],
                    res["question"],
                    res["answer"],
                    judge_model,
                    is_last
                )
                futures[future] = i
            
            # 完了順に結果を取得
            for future in as_completed(futures):
                i = futures[future]
                try:
                    idx, score, reason, answer = future.result()
                    infer_results[i]["judge_score"]  = score
                    infer_results[i]["judge_reason"] = reason
                    completed += 1
                    print(f"  [{completed}/{total}] idx={idx}  score={score}  {reason[:40]}")
                except Exception as e:
                    print(f"  [ERROR] Judge 失敗 idx={infer_results[i]['idx']}: {e}")
                    infer_results[i]["judge_score"]  = -1
                    infer_results[i]["judge_reason"] = str(e)[:50]
                    completed += 1

    # ── 結果保存 ──
    condition = f"{args.model}_{args.rag}"
    out_path  = RESULTS_DIR / f"{condition}_results.jsonl"
    with open(out_path, "w", encoding="utf-8") as f:
        for res in infer_results:
            f.write(json.dumps(res, ensure_ascii=False) + "\n")
    print(f"\n結果保存: {out_path}")

    # ── サマリー ──
    valid = [r for r in infer_results if r["judge_score"] >= 0]
    avg_score  = sum(r["judge_score"] for r in valid) / len(valid) if valid else 0
    perfect    = sum(1 for r in valid if r["judge_score"] == 3)
    avg_ret    = sum(r["ret_time"]  for r in infer_results) / len(infer_results)
    avg_gen    = sum(r["gen_time"]  for r in infer_results) / len(infer_results)
    score_dist = {k: sum(1 for r in valid if r["judge_score"] == k) for k in range(4)}

    summary = {
        "condition":      condition,
        "model":          args.model,
        "rag":            args.rag,
        "infer_model":    str(model_path.name),
        "judge_model":    judge_model,
        "n_questions":    len(infer_results),
        "n_valid_judge":  len(valid),
        "avg_score":      round(avg_score, 3),
        "perfect_rate":   round(perfect / len(valid) if valid else 0, 3),
        "avg_ret_time":   round(avg_ret, 3),
        "avg_gen_time":   round(avg_gen, 3),
        "score_dist":     score_dist,
        "timestamp":      datetime.now().isoformat(),
    }
    sum_path = RESULTS_DIR / f"{condition}_summary.json"
    with open(sum_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"サマリー保存: {sum_path}")

    print(f"\n{'='*60}")
    print(f"  {condition}")
    print(f"  Judge 平均スコア: {avg_score:.3f} / 3.0  (有効 {len(valid)}/{len(infer_results)} 問)")
    print(f"  Perfect-Score率 (3点率): {perfect}/{len(valid)}  ({(perfect/len(valid)*100 if valid else 0):.1f}%)")
    print(f"  avg Retrieval: {avg_ret:.2f}s  avg Generation: {avg_gen:.1f}s")
    print(f"  Score dist: " + "  ".join(f"{k}点:{v}" for k, v in score_dist.items()))
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
