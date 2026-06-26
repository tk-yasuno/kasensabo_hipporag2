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
MAX_NEW_TOKENS = 512   # [v0.2.2] GT-answer品質向上（時間優先度は低い）


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


def _ollama_batch_infer(
    model: str,
    system: str,
    user_messages: list[str],
    max_new_tokens: int = MAX_NEW_TOKENS,
    temperature: float = 0.3,
) -> list[str]:
    """
    Ollama で複数プロンプトを順次推論（1問ずつ）。
    Ollamaはバッチ推論APIがないため、ループで呼び出す。
    
    Args:
        model: Ollamaモデル名（例: qwen2.5:7b-instruct-q4_k_m）
        system: システムプロンプト（全問共通）
        user_messages: ユーザープロンプトリスト
        max_new_tokens: 最大生成トークン数
        temperature: 温度パラメータ
    
    Returns:
        生成結果のリスト
    """
    results = []
    for user_msg in user_messages:
        try:
            answer = _ollama_chat(
                model=model,
                system=system,
                user=user_msg,
                timeout=300.0,
                max_tokens=max_new_tokens,
                temperature=temperature,
                keep_alive="5m",
            )
            results.append(answer.strip())
        except Exception as e:
            print(f"  [ERROR] Ollama推論失敗: {e}")
            results.append("")
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

# v0.3: Triple filtering 用プロンプト
TRIPLE_FILTER_PROMPT = """\
質問:
{query}

以下の関係（triple）が質問に関連しているか判定してください。

Subject: {subj}
Relation: {rel}
Object: {obj}

判定基準:
- 質問の主題・対象・条件・数値に直接関係する → YES
- 関係が弱い、一般的すぎる、別の話題 → NO

回答（JSONのみ）:
{{"relevant": "YES"}} または {{"relevant": "NO"}}"""

# v0.3 高速化: バッチ処理用プロンプト
TRIPLE_FILTER_BATCH_PROMPT = """\
あなたは高精度の関連性判定モデルです。

与えられた質問に対して、複数の triple が関連しているかどうかを判定します。

絶対条件:
- 出力は JSON 配列のみ
- 各要素は {{"idx": n, "relevant": "YES" or "NO"}}
- JSON 以外の文字を出力しない

質問:
{query}

triple一覧:
{triple_list}

出力形式（厳守）:
[
  {{"idx": 0, "relevant": "YES"}},
  {{"idx": 1, "relevant": "NO"}},
  ...
]
"""

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
# v0.3: Triple Filtering（LLM による Recognition Memory）
# ─────────────────────────────────────────────────────────

def filter_triples_with_llm(
    model: str,
    query: str,
    triples: list,
    top_k: int = 10,
) -> list:
    """
    LLM を使用して triple の relevance を判定し、フィルタリングする。
    
    Args:
        model: Ollama モデル名
        query: クエリ文
        triples: [(triple_dict, sim_score), ...] のリスト
        top_k: 判定対象の triple 数
    
    Returns:
        フィルタリング済み triple リスト [(triple_dict, sim_score), ...]
    """
    filtered = []
    
    for t_dict, score in triples[:top_k]:
        triple = t_dict.get("triple", {})
        subj = triple.get("subject", "")
        rel = triple.get("relation", "")
        obj = triple.get("object", "")
        
        prompt = TRIPLE_FILTER_PROMPT.format(
            query=query,
            subj=subj,
            rel=rel,
            obj=obj,
        )
        
        try:
            out = _ollama_chat(
                model=model,
                system="You are a triple relevance judge. Output JSON only.",
                user=prompt,
                timeout=60.0,
                max_tokens=32,
                temperature=0.0,
                keep_alive="30m",
            )
            result = json.loads(out)
            if result.get("relevant", "").upper() == "YES":
                filtered.append((t_dict, score))
        except Exception as e:
            # パース失敗時はスキップ
            pass
    
    return filtered


def filter_triples_batch(
    llm_fn,
    query: str,
    triples: list,
    batch_size: int = 10,
    debug: bool = False,
) -> list:
    """
    v0.3 高速化: バッチ処理で triple の relevance を判定（10倍高速化）
    
    従来: 質問1 → triple10個 → LLM10回（1個ずつ）
    改善: 質問1 → triple10個 → LLM1回（まとめて）
    
    Args:
        llm_fn: LLM 呼び出し関数 lambda x: _ollama_chat(...)
        query: クエリ文
        triples: [(triple_dict, sim_score), ...] のリスト
        batch_size: バッチサイズ（デフォルト10、最大20推奨）
        debug: デバッグ出力有効フラグ
    
    Returns:
        フィルタリング済み triple リスト
    """
    import json
    import re
    import sys
    
    # ★ 関数呼び出しの追跡用ログ（常に表示）
    if len(triples) > 0:
        print(f"[FILTER] Called with {len(triples)} triples", file=sys.stderr)
    
    results = []
    
    if debug:
        print(f"    [DEBUG] Triple filtering 開始: {len(triples)} triples, batch_size={batch_size}")
    
    # バッチ処理ループ
    for batch_idx, start_idx in enumerate(range(0, len(triples), batch_size)):
        batch = triples[start_idx:start_idx + batch_size]
        
        # Triple 一覧テキスト生成
        triple_list_text = "\n".join([
            f"{i}: (subject={t_dict['triple']['subject']}, "
            f"relation={t_dict['triple']['relation']}, "
            f"object={t_dict['triple']['object']})"
            for i, (t_dict, _) in enumerate(batch)
        ])
        
        prompt = TRIPLE_FILTER_BATCH_PROMPT.format(
            query=query,
            triple_list=triple_list_text,
        )
        
        try:
            # LLM 呼び出し（バッチ）
            out = llm_fn(prompt)
            
            if debug:
                print(f"    [DEBUG] Batch {batch_idx}: LLM response = {out[:100]}...")
            
            # JSON 抽出（ゆるい正規表現）
            json_match = re.search(r"\[.*\]", out, flags=re.DOTALL)
            if not json_match:
                if debug:
                    print(f"    [DEBUG] Batch {batch_idx}: JSON 抽出失敗")
                continue
            
            json_text = json_match.group(0)
            parsed = json.loads(json_text)
            
            if debug:
                print(f"    [DEBUG] Batch {batch_idx}: JSON parse OK, len={len(parsed)}")
            
            # relevance 判定結果をマージ
            if isinstance(parsed, list):
                batch_yes_count = 0
                for item, (t_dict, score) in zip(parsed, batch):
                    if isinstance(item, dict) and item.get("relevant", "").upper() == "YES":
                        results.append((t_dict, score))
                        batch_yes_count += 1
                
                if debug:
                    print(f"    [DEBUG] Batch {batch_idx}: {batch_yes_count}/{len(batch)} triples passed")
        
        except Exception as e:
            # バッチ処理失敗時はスキップ（個別判定せず）
            if debug:
                print(f"    [DEBUG] Batch {batch_idx}: Exception = {str(e)[:100]}")
            # ★ エラーログ（常に表示）
            print(f"[FILTER] Batch {batch_idx} error: {str(e)[:200]}", file=sys.stderr)
            pass
    
    if debug:
        print(f"    [DEBUG] Triple filtering 完了: {len(results)} triples filtered")
    
    # ★ 終了ログ（常に表示）
    if len(triples) > 0:
        print(f"[FILTER] Returned {len(results)}/{len(triples)} triples", file=sys.stderr)
    
    return results


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
    parser.add_argument("--model",     default=None, choices=["swallow", "elyza"],
                        help="生成モデル（Unsloth用）。--ollama-model指定時は不要")
    parser.add_argument("--rag",       required=True,
                        choices=["naive", "light", "hipporag2"],
                        help="RAG 方式")
    parser.add_argument("--top-k",     type=int,   default=5,    help="取得チャンク数")
    parser.add_argument("--max-chars", type=int,   default=2000, help="コンテキスト文字数上限")
    parser.add_argument("--timeout",    type=float, default=300.0)
    parser.add_argument("--batch-size", type=int,   default=8,   help="Unsloth バッチサイズ")
    parser.add_argument("--judge-workers", type=int, default=4, help="Judge採点の並列数")
    parser.add_argument("--keywords-file", type=str, default=None, help="カスタムキーワード辞書ファイル（HippoRAG2用）")
    parser.add_argument("--n-triples", type=int, default=20, help="Triple filtering取得数（HippoRAG2用、デフォルト20）")
    parser.add_argument("--judge-model", type=str, default=None, help="Judge採点モデル明示指定（例: qwen2.5:14b）")
    parser.add_argument("--ollama-model", type=str, default=None, help="Ollama推論モデル（例: qwen2.5:7b-instruct-q4_k_m）指定時はUnslothの代わりにOllamaを使用")
    parser.add_argument("--no-judge",   action="store_true",    help="Judge 採点をスキップ")
    parser.add_argument("--dry-run",    action="store_true",    help="10件のみ実行（動作確認）")
    parser.add_argument("--log-features", action="store_true",  help="[v0.5] Volume/Chapter/Chunk featuresをログ出力（calibration用）")
    parser.add_argument("--use-calibration", action="store_true", help="[v0.5] キャリブレーションモデルを使用してリランキング")
    parser.add_argument("--calibration-dir", type=str, default=None, help="[v0.5] キャリブレーションモデルディレクトリ（デフォルト: experiments/calibration_models）")
    args = parser.parse_args()
    
    # --ollama-model指定時は--modelを不要にする
    if args.ollama_model is None and args.model is None:
        parser.error("--model または --ollama-model のいずれかを指定してください")
    
    use_ollama = bool(args.ollama_model)

    print(f"=" * 60)
    if use_ollama:
        print(f"  RAG 評価: ollama_model={args.ollama_model}  rag={args.rag}  top_k={args.top_k}")
    else:
        print(f"  RAG 評価: model={args.model}  rag={args.rag}  top_k={args.top_k}  batch={args.batch_size}")
    print(f"=" * 60)

    # ── モデルパス解決 ──
    env_cfg = _load_env_config()
    
    if use_ollama:
        model_path = None  # Ollamaはパス不要
        if args.judge_model:
            judge_model = args.judge_model
            print(f"  推論モデル: {args.ollama_model} (Ollama)")
            print(f"  Judge モデル: {judge_model} (指定)")
        else:
            judge_model = _pick_model("judge", JUDGE_CANDIDATES, env_cfg)
            print(f"  推論モデル: {args.ollama_model} (Ollama)")
            print(f"  Judge モデル: {judge_model} (自動選択)")
    else:
        model_path = MODEL_PATHS[args.model]
        if args.judge_model:
            judge_model = args.judge_model
            print(f"  推論モデル: {model_path}")
            print(f"  Judge モデル: {judge_model} (指定)")
        else:
            judge_model = _pick_model("judge", JUDGE_CANDIDATES, env_cfg)
            print(f"  推論モデル: {model_path}")
            print(f"  Judge モデル: {judge_model} (自動選択)")

    if not use_ollama and not model_path.exists():
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
    # v0.3: n_triples パラメータを渡す（HippoRAG2のみ有効）
    # v0.5: use_calibration パラメータを渡す
    retriever = make_retriever(
        args.rag, 
        top_k=args.top_k, 
        keywords_file=args.keywords_file, 
        n_triples=args.n_triples,
        use_calibration=args.use_calibration,
        calibration_dir=args.calibration_dir
    )

    # v0.3: Triple index のロード（HippoRAG2のみ）
    triple_index = None
    triple_data = None
    
    if args.rag in ("hipporag2", "hippo"):
        print(f"  Triple index ロード中...")
        TRIPLE_EMB_PATH = EXP_DIR / "indices" / "triple_embs.npy"
        TRIPLE_JSON_PATH = EXP_DIR / "indices" / "triples.json"
        TRIPLE_INDEX_PATH = EXP_DIR / "indices" / "triple.index"
        
        if TRIPLE_EMB_PATH.exists() and TRIPLE_JSON_PATH.exists() and TRIPLE_INDEX_PATH.exists():
            try:
                import faiss
                import numpy as np
                triple_embs = np.load(TRIPLE_EMB_PATH)
                triple_index = faiss.IndexFlatIP(triple_embs.shape[1])
                triple_index.add(triple_embs)
                with open(TRIPLE_JSON_PATH, encoding="utf-8") as f:
                    triple_data = json.load(f)
                print(f"    Triple index: {len(triple_data)} triples")
            except Exception as e:
                print(f"    [WARN] Triple index ロード失敗: {e}")
                triple_index = None
                triple_data = None
        else:
            print(f"    [WARN] Triple index ファイルが見つかりません。")
            print(f"    先に 01_build_triple_index.py を実行してください。")
            print(f"    Triple filtering なしで実行します。")
    else:
        print(f"  Triple filtering: OFF (Naive/LightRAG はtriple filteringを使用しません)")

    # ── Phase 1: Unsloth/Ollama バッチ推論 ──
    use_ollama = bool(args.ollama_model)
    
    if use_ollama:
        print(f"\n[Phase 1] Ollama 推論 (model={args.ollama_model})...")
        print(f"  Ollamaモデル {args.ollama_model} を使用")
        # Ollamaはモデルロード不要（サーバー側で管理）
        unsloth_model = None
        unsloth_tokenizer = None
    else:
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
    for rec_idx, rec in enumerate(records):
        question = rec["question"]
        
        # v0.3.1.5: Retrieval時間計測開始（triple filtering含む全体を計測）
        t0 = time.time()
        
        # v0.3: triple filtering の状態をリセット（毎回新しく実行するため）
        if hasattr(retriever, "_filtered_triples"):
            retriever._filtered_triples = None
        
        # v0.3: triple filtering を使う場合のみ
        if triple_index is not None and triple_data is not None:
            # 1. query embedding
            q_vec = rag_mod._encode(question)
            
            # 2. triple retrieval
            D, I = triple_index.search(q_vec, 20)
            top_triples = [(triple_data[int(i)], float(D[0][j])) for j, i in enumerate(I[0])]
            
            # 3. LLM filtering（バッチ処理で高速化）
            filter_model = "qwen2.5:7b-instruct-q4_k_m"  # v0.3.1.5: Triple filtering専用モデル
            debug_mode = rec_idx < 3  # 最初の3問のみデバッグ出力
            
            if debug_mode:
                print(f"    [DEBUG] Triple filtering model: {filter_model}")
            
            filtered_triples = filter_triples_batch(
                llm_fn=lambda x: _ollama_chat(
                    model=filter_model,
                    system="You are a triple relevance judge. Output valid JSON only. No extra text.",
                    user=x,
                    timeout=120.0,
                    max_tokens=512,
                    temperature=0.0,
                    keep_alive="5m",
                ),
                query=question,
                triples=top_triples,
                batch_size=10,  # 10個ずつまとめて判定
                debug=debug_mode,
            )
            
            # 4. retriever に渡す
            if hasattr(retriever, "_filtered_triples"):
                retriever._filtered_triples = filtered_triples
                if debug_mode and filtered_triples:
                    print(f"    [DEBUG] {len(filtered_triples)} triples → Retriever に渡す")
        
        # ── 既存の Retrieval ──
        chunks = retriever.retrieve(question)
        t_ret  = time.time() - t0  # triple filtering含む全体の時間
        context  = format_context(chunks, max_chars=args.max_chars)
        user_msg = CONTEXT_TEMPLATE.format(rag_context=context, question=rec["question"])
        
        # プロンプト生成（Unsloth用とOllama用で分岐）
        if use_ollama:
            # Ollamaの場合: system/user分離
            prompt = None
            system_prompt = SYSTEM_PROMPT_RAG
            user_prompt = user_msg
        else:
            # Unslothの場合: Llama-3テンプレート
            prompt = _format_llama3_prompt(SYSTEM_PROMPT_RAG, user_msg)
            system_prompt = None
            user_prompt = None
        
        all_ret_data.append({
            "rec": rec, 
            "chunks": chunks, 
            "prompt": prompt,
            "system_prompt": system_prompt,
            "user_prompt": user_prompt,
            "t_ret": t_ret,
            "retrieval_info": retriever.retrieval_info if hasattr(retriever, "retrieval_info") else {},  # v0.5: calibration用
            "rec_idx": rec_idx,  # v0.5: feature logging用インデックス
        })

    # バッチ推論
    if use_ollama:
        print(f"  Ollama推論中 ({n_total}問)...")
    else:
        print(f"  バッチ推論中 ({n_total}問 / batch={args.batch_size})...")
    
    t_gen_total_start = time.time()
    all_answers: list[str] = []

    for batch_start in range(0, n_total, args.batch_size):
        batch_data = all_ret_data[batch_start: batch_start + args.batch_size]
        t0 = time.time()
        
        try:
            if use_ollama:
                # Ollama推論（1問ずつ順次実行）
                user_messages = [d["user_prompt"] for d in batch_data]
                system = batch_data[0]["system_prompt"]  # 全問同じシステムプロンプト
                answers = _ollama_batch_infer(
                    model=args.ollama_model,
                    system=system,
                    user_messages=user_messages,
                    max_new_tokens=MAX_NEW_TOKENS,
                    temperature=0.3,
                )
            else:
                # Unsloth推論（バッチ処理）
                prompts = [d["prompt"] for d in batch_data]
                answers = _unsloth_batch_infer(unsloth_model, unsloth_tokenizer, prompts)
        except Exception as e:
            print(f"  [ERROR] 推論失敗 batch={batch_start}: {e}")
            answers = ["" for _ in batch_data]
        
        t_batch = time.time() - t0
        per_q   = t_batch / len(batch_data)
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
    if not use_ollama:
        print(f"\n  モデルアンロード中...")
        print(f"  アンロード前 GPU MB: {round(torch.cuda.memory_allocated()/1024**2)}")
        
        # モデル削除
        del unsloth_model, unsloth_tokenizer
    else:
        print(f"\n  Ollamaモデルはサーバー側で管理（アンロード不要）")
    
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
    if use_ollama:
        # Ollamaモデル名から簡略名を生成（例: qwen2.5:7b-instruct-q4_k_m → qwen2.5-7b）
        ollama_short = args.ollama_model.split(':')[0].replace('.', '')  # qwen25
        if ':' in args.ollama_model:
            version_part = args.ollama_model.split(':')[1].split('-')[0]  # 7b
            ollama_short = f"{ollama_short}{version_part}"  # qwen257b
        condition = f"{ollama_short}_{args.rag}"
    else:
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
        "model":          args.model if not use_ollama else None,
        "rag":            args.rag,
        "infer_model":    args.ollama_model if use_ollama else str(model_path.name),
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

    # ── v0.5: Feature Logging（calibration用）──
    if args.log_features:
        print(f"\n[v0.5] Feature logging...")
        
        # Volume features
        volume_features = []
        for idx, (res, ret_data) in enumerate(zip(infer_results, all_ret_data)):
            ret_info = ret_data.get("retrieval_info", {})
            if not ret_info or ret_info.get("fallback", False):
                continue
            
            # question_idはtestsetのidxを優先、なければループインデックスを使用
            question_id = res.get("idx", ret_data.get("rec_idx", idx))
            judge_score = res.get("judge_score", -1)
            
            for vol_info in ret_info.get("volumes", []):
                volume_features.append({
                    "question_id": question_id,
                    "volume_id": vol_info["volume_id"],
                    "emb_score": vol_info["emb_score"],
                    "kw_score": vol_info["kw_score"],
                    "triple_score": vol_info["triple_score"],
                    "fused_score": vol_info["fused_score"],
                    "n_filtered_triples": ret_info.get("n_filtered_triples", 0),
                    "avg_triple_sim": ret_info.get("avg_triple_sim", 0.0),
                    "retrieval_time": ret_data["t_ret"],
                    "selected": vol_info["selected"],
                    "judge_score": judge_score,
                })
        
        vol_path = RESULTS_DIR / f"{condition}_volume_features.jsonl"
        with open(vol_path, "w", encoding="utf-8") as f:
            for feat in volume_features:
                f.write(json.dumps(feat, ensure_ascii=False) + "\n")
        print(f"  Volume features saved: {vol_path} ({len(volume_features)} entries)")
        
        # Chapter features
        chapter_features = []
        for idx, (res, ret_data) in enumerate(zip(infer_results, all_ret_data)):
            ret_info = ret_data.get("retrieval_info", {})
            if not ret_info or ret_info.get("fallback", False):
                continue
            
            question_id = res.get("idx", ret_data.get("rec_idx", idx))
            judge_score = res.get("judge_score", -1)
            
            for chap_info in ret_info.get("chapters", []):
                chapter_features.append({
                    "question_id": question_id,
                    "volume_id": chap_info["volume_id"],
                    "chapter_id": chap_info["chapter_id"],
                    "emb_score": chap_info["emb_score"],
                    "triple_score": chap_info["triple_score"],
                    "fused_score": chap_info["fused_score"],
                    "n_filtered_triples": ret_info.get("n_filtered_triples", 0),
                    "avg_triple_sim": ret_info.get("avg_triple_sim", 0.0),
                    "retrieval_time": ret_data["t_ret"],
                    "selected": chap_info["selected"],
                    "judge_score": judge_score,
                })
        
        chap_path = RESULTS_DIR / f"{condition}_chapter_features.jsonl"
        with open(chap_path, "w", encoding="utf-8") as f:
            for feat in chapter_features:
                f.write(json.dumps(feat, ensure_ascii=False) + "\n")
        print(f"  Chapter features saved: {chap_path} ({len(chapter_features)} entries)")
        
        # Chunk features
        chunk_features = []
        for idx, (res, ret_data) in enumerate(zip(infer_results, all_ret_data)):
            ret_info = ret_data.get("retrieval_info", {})
            if not ret_info or ret_info.get("fallback", False):
                continue
            
            question_id = res.get("idx", ret_data.get("rec_idx", idx))
            judge_score = res.get("judge_score", -1)
            
            for chunk_info in ret_info.get("chunks", []):
                chunk_features.append({
                    "question_id": question_id,
                    "volume_id": chunk_info.get("volume_id", ""),
                    "chapter_id": chunk_info.get("chapter_id", ""),
                    "chunk_id": chunk_info["chunk_id"],
                    "embedding_sim": chunk_info["embedding_sim"],
                    "chunk_length": chunk_info["chunk_length"],
                    "judge_score": judge_score,
                })
        
        chunk_path = RESULTS_DIR / f"{condition}_chunk_features.jsonl"
        with open(chunk_path, "w", encoding="utf-8") as f:
            for feat in chunk_features:
                f.write(json.dumps(feat, ensure_ascii=False) + "\n")
        print(f"  Chunk features saved: {chunk_path} ({len(chunk_features)} entries)")

    # 中間指標の確認（早期エラー検出）
    import warnings
    if avg_ret < 0.05 and args.rag == "hipporag2":
        warnings.warn(
            f"\n⚠️  Retrieval 時間が異常に短い ({avg_ret:.3f}s)。\n"
            f"    Triple filtering が動作していない可能性があります。\n"
            f"    → indices/triples.json の存在を確認してください。\n"
            f"    → 04_eval_rag.py のデバッグ出力を確認してください。",
            UserWarning
        )

    print(f"\n{'='*60}")
    print(f"  {condition}")
    print(f"  Judge 平均スコア: {avg_score:.3f} / 3.0  (有効 {len(valid)}/{len(infer_results)} 問)")
    print(f"  Perfect-Score率 (3点率): {perfect}/{len(valid)}  ({(perfect/len(valid)*100 if valid else 0):.1f}%)")
    print(f"  avg Retrieval: {avg_ret:.2f}s  avg Generation: {avg_gen:.1f}s")
    print(f"  Score dist: " + "  ".join(f"{k}点:{v}" for k, v in score_dist.items()))
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
