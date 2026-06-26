"""
experiments/01_build_triple_index.py
──────────────────────────────────────────────────────────────
河川砂防技術標準の全チャンクから OpenIE triple を抽出し、
triple embedding index（FAISS）を構築する。

出力:
  experiments/indices/triples.json
  experiments/indices/triple_embs.npy
  experiments/indices/triple.index

Usage:
    python experiments/01_build_triple_index.py
"""

from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import time
import httpx

IDX_DIR = Path(__file__).parent / "indices"

CHUNKS_PATH = IDX_DIR / "chunks.jsonl"
TRIPLE_JSON = IDX_DIR / "triples.json"
TRIPLE_EMB = IDX_DIR / "triple_embs.npy"
TRIPLE_INDEX = IDX_DIR / "triple.index"

EMBED_MODEL_NAME = "hotchpotch/static-embedding-japanese"

# OpenIE（Ollama）で triple 抽出用プロンプト（Qwen2.5 最適化版）
OPENIE_PROMPT = """
あなたは高精度の Open Information Extraction (OpenIE) モデルです。

与えられた文章から、(subject, relation, object) の三つ組を抽出します。

絶対条件:
- 出力は JSON 配列のみ
- JSON の前後に文章・説明・改行を付けない
- JSON のキーは "subject", "relation", "object" のみ
- 値はすべて文字列
- JSON 以外の文字を一切出力しない
- 抽出件数は 5～15 個（文章が長い場合は増やしてよい）

抽出方針:
- 主語(subject)は名詞句
- relation は動詞または述語（名詞のみは避ける）
- object は名詞句
- 技術文書の場合、定義・条件・因果関係・構造関係を優先して抽出
- 文脈が複数チャンクにまたがる場合も抽出してよい

出力形式（厳守）:
[
  {{"subject": "...", "relation": "...", "object": "..."}},
  ...
]

文章:
{passage}

上記の JSON 配列のみを返してください。
"""

# ──────────────────────────────────────────────────────────
# OpenIE（Ollama）で triple 抽出
# ──────────────────────────────────────────────────────────

def extract_triples_ollama(passage: str, model: str = "qwen2.5:7b-instruct-q4_k_m") -> list[dict]:
    """Ollama を使用して OpenIE triple を抽出（Qwen2.5 最適化版）"""
    if not passage or len(passage.strip()) < 20:
        return []
    
    payload = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": "You are a high-precision OpenIE extractor. Output ONLY valid JSON arrays."
            },
            {"role": "user", "content": OPENIE_PROMPT.format(passage=passage)},
        ],
        "stream": False,
        "keep_alive": "5m",
        "options": {
            "temperature": 0.0,
            "num_predict": 512,
            "num_ctx": 4096,
            "top_p": 0.9,
            "repeat_penalty": 1.05,
        },
    }
    
    try:
        import re
        resp = httpx.post("http://localhost:11434/api/chat", json=payload, timeout=120)
        resp.raise_for_status()
        out = resp.json().get("message", {}).get("content", "")
        
        # デバッグ: 最初の3回のみ LLM 出力を表示
        if not hasattr(extract_triples_ollama, '_debug_count'):
            extract_triples_ollama._debug_count = 0
        
        if extract_triples_ollama._debug_count < 3:
            print(f"\n  [DEBUG] LLM 出力 ({extract_triples_ollama._debug_count + 1}/3):")
            preview = out[:200] + "..." if len(out) > 200 else out
            print(f"    {preview}")
            extract_triples_ollama._debug_count += 1
        
        if not out or len(out) < 10:
            return []
        
        # JSON パース（堅牢な抽出 + 修復処理）
        json_text = None
        try:
            # シンプルな [...] 抽出（貪欲マッチ）
            match = re.search(r"\[.*\]", out, flags=re.DOTALL)
            if not match:
                return []
            json_text = match.group(0)
            
            # 一般的な JSON エラーの修復
            # 1. 末尾カンマの除去: {...,} → {...}
            json_text = re.sub(r',\s*([}\]])', r'\1', json_text)
            # 2. 連続カンマの除去: ,, → ,
            json_text = re.sub(r',\s*,', ',', json_text)
            # 3. カンマ不足の修復: }{ → },{
            json_text = re.sub(r'\}\s*\{', '},{', json_text)
            # 4. 配列要素間のカンマ不足: ][ → ],[
            json_text = re.sub(r'\]\s*\[', '],[', json_text)
            
        except Exception as e:
            if extract_triples_ollama._debug_count < 5:
                print(f"  [DEBUG] JSON 抽出失敗: {e}")
            return []
        
        try:
            triples = json.loads(json_text)
        except json.JSONDecodeError as e:
            # デバッグ用: エラー箇所を表示
            error_pos = e.pos
            context_start = max(0, error_pos - 50)
            context_end = min(len(json_text), error_pos + 50)
            context = json_text[context_start:context_end]
            print(f"  [WARN] JSONパース失敗: {str(e)[:60]}")
            print(f"        エラー付近: ...{context}...")
            return []
        except Exception as e:
            print(f"  [WARN] JSONパース失敗: {e}")
            return []
        
        # バリデーション
        if isinstance(triples, list):
            valid = []
            for t in triples:
                if isinstance(t, dict) and "subject" in t and "relation" in t and "object" in t:
                    valid.append({
                        "subject": str(t["subject"]).strip(),
                        "relation": str(t["relation"]).strip(),
                        "object": str(t["object"]).strip(),
                    })
            return valid
        return []
    except Exception as e:
        print(f"  [WARN] triple 抽出失敗: {e}")
        return []


# ──────────────────────────────────────────────────────────
# Embedding
# ──────────────────────────────────────────────────────────

_embed_model = None

def get_embed_model():
    global _embed_model
    if _embed_model is None:
        from sentence_transformers import SentenceTransformer
        print(f"  Embedding モデルをロード中: {EMBED_MODEL_NAME}")
        _embed_model = SentenceTransformer(EMBED_MODEL_NAME)
    return _embed_model


def encode_triple(triple: dict) -> np.ndarray:
    """triple を embedding に変換"""
    text = f"{triple['subject']} {triple['relation']} {triple['object']}"
    model = get_embed_model()
    vec = model.encode([text], normalize_embeddings=True, convert_to_numpy=True)
    return vec.astype("float32")[0]


# ──────────────────────────────────────────────────────────
# main
# ──────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  Triple Index ビルド")
    print("=" * 60)

    # チャンクロード
    print(f"\n[Step 1] チャンク読み込み...")
    if not CHUNKS_PATH.exists():
        print(f"ERROR: {CHUNKS_PATH} が見つかりません。")
        print("先に 01_build_indices.py を実行してください。")
        return
    
    chunks = []
    with open(CHUNKS_PATH, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                chunks.append(json.loads(line))
    
    print(f"  読み込み: {len(chunks)} チャンク")

    # Triple 抽出（chunk ごとに個別処理で精度最大化）
    print(f"\n[Step 2] OpenIE triple 抽出（chunk ごとに個別処理）...")
    print(f"  （Ollama が起動していることを確認してください）")
    
    GROUP_SIZE = 1  # chunk ごとに個別処理（triple の粒度を最大化）
    triples_out = []
    triple_embs = []
    failed_count = 0

    for start in range(0, len(chunks), GROUP_SIZE):
        group = chunks[start:start + GROUP_SIZE]
        
        # まとめたテキストを作成
        passages = []
        passage_ids = []
        for idx, c in enumerate(group):
            chunk_id = c.get("chunk_id", start + idx)
            vol = c.get("doc_id", "UNKNOWN")
            chap = c.get("section_id", "UNKNOWN")
            passages.append(f"[CHUNK {start+idx}] {c.get('text', '')}")
            passage_ids.append((start + idx, chunk_id, vol, chap))
        
        merged_text = "\n\n---\n\n".join(passages)
        
        # OpenIE 抽出
        ts = extract_triples_ollama(merged_text)
        
        # triple 保存 + embedding
        for t in ts:
            # 最初のチャンク情報を使用（まとめた場合のデフォルト）
            p_id, chunk_id, vol, chap = passage_ids[0]
            triples_out.append({
                "chunk_ids": [pid[1] for pid in passage_ids],  # グループ内の全chunk_id
                "passage_ids": [pid[0] for pid in passage_ids],  # グループ内の全passage_id
                "volume_id": vol,
                "chapter_id": chap,
                "triple": t,
            })
            try:
                emb = encode_triple(t)
                triple_embs.append(emb)
            except Exception as e:
                failed_count += 1
        
        # 進捗表示
        if (start + GROUP_SIZE) % 50 == 0:
            print(f"  [{start+GROUP_SIZE:4d}/{len(chunks)}] {len(triples_out):5d} triples 抽出")
        
        # メモリ管理
        if (start + GROUP_SIZE) % 100 == 0:
            import gc
            gc.collect()

    print(f"  完了: {len(triples_out)} triple, 失敗: {failed_count}")

    # 品質チェック: chunk_ids の粒度確認
    if triples_out:
        import warnings
        chunk_ids_lengths = [len(t['chunk_ids']) for t in triples_out]
        unique_lengths = set(chunk_ids_lengths)
        
        if len(unique_lengths) == 1 and GROUP_SIZE > 1:
            common_length = list(unique_lengths)[0]
            if common_length == GROUP_SIZE:
                warnings.warn(
                    f"\n⚠️  全ての triple が同じ chunk_ids 長さ ({common_length})。\n"
                    f"    GROUP_SIZE={GROUP_SIZE} が大きすぎる可能性があります。\n"
                    f"    Triple filtering の効果が制限されます。\n"
                    f"    → 推奨: GROUP_SIZE=1 で再実行してください。",
                    UserWarning
                )

    # numpy に変換
    if not triple_embs:
        print("ERROR: embedding された triple が 0 件です。")
        return
    
    triple_embs = np.vstack(triple_embs).astype("float32")
    
    # Embedding 保存
    print(f"\n[Step 3] Triple embedding 保存...")
    np.save(TRIPLE_EMB, triple_embs)
    print(f"  {TRIPLE_EMB}")
    print(f"  形状: {triple_embs.shape}")

    # JSON 保存
    print(f"\n[Step 4] Triple メタ情報保存...")
    with open(TRIPLE_JSON, "w", encoding="utf-8") as f:
        json.dump(triples_out, f, ensure_ascii=False, indent=2)
    print(f"  {TRIPLE_JSON}")
    print(f"  件数: {len(triples_out)}")

    # FAISS index 構築 + 保存
    print(f"\n[Step 5] FAISS Index 構築...")
    import faiss
    
    index = faiss.IndexFlatIP(triple_embs.shape[1])
    index.add(triple_embs)
    faiss.write_index(index, str(TRIPLE_INDEX))
    print(f"  {TRIPLE_INDEX}")

    print(f"\n" + "=" * 60)
    print(f"  完了! v0.3 で 04_eval_rag.py を実行してください")
    print(f"=" * 60)


if __name__ == "__main__":
    main()
