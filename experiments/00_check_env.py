"""
experiments/00_check_env.py
────────────────────────────────────────────────────────────
Phase 0: 実験前環境確認スクリプト

確認項目:
  1. Ollama 疎通 + 利用可能モデル一覧
  2. GPU メモリ (CUDA)
  3. 依存ライブラリ (rank_bm25, faiss, sentence_transformers 等)
  4. データファイル存在確認

Usage:
    python experiments/00_check_env.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent

# ─────────────────────────────────────────────────────────
# 1. Ollama 疎通 + モデル一覧
# ─────────────────────────────────────────────────────────

def check_ollama() -> tuple[bool, list[str]]:
    try:
        import httpx
        r = httpx.get("http://localhost:11434/api/tags", timeout=10.0)
        r.raise_for_status()
        models = [m["name"] for m in r.json().get("models", [])]
        return True, models
    except Exception as e:
        return False, [str(e)]


# ─────────────────────────────────────────────────────────
# 2. GPU メモリ
# ─────────────────────────────────────────────────────────

def check_gpu() -> dict:
    try:
        import torch
        if not torch.cuda.is_available():
            return {"available": False, "reason": "CUDA not available"}
        total_gb  = torch.cuda.get_device_properties(0).total_memory / 1e9
        alloc_gb  = torch.cuda.memory_allocated(0) / 1e9
        return {
            "available": True,
            "device":    torch.cuda.get_device_name(0),
            "total_gb":  round(total_gb, 1),
            "allocated_gb": round(alloc_gb, 1),
            "free_gb":   round(total_gb - alloc_gb, 1),
        }
    except Exception as e:
        return {"available": False, "reason": str(e)}


# ─────────────────────────────────────────────────────────
# 3. 依存ライブラリ
# ─────────────────────────────────────────────────────────

REQUIRED_LIBS = [
    ("httpx",                 "httpx"),
    ("faiss",                 "faiss"),
    ("numpy",                 "numpy"),
    ("rank_bm25",             "rank_bm25"),
    ("sentence_transformers", "sentence_transformers"),
    ("torch",                 "torch"),
    ("sklearn",               "scikit-learn"),
    ("tqdm",                  "tqdm"),
    ("matplotlib",            "matplotlib"),
    ("pandas",                "pandas"),
]

def check_libraries() -> list[dict]:
    results = []
    for import_name, pkg_name in REQUIRED_LIBS:
        try:
            mod = __import__(import_name)
            ver = getattr(mod, "__version__", "?")
            results.append({"lib": pkg_name, "ok": True, "version": ver})
        except ImportError:
            results.append({"lib": pkg_name, "ok": False, "version": None})
    return results


# ─────────────────────────────────────────────────────────
# 4. データファイル確認
# ─────────────────────────────────────────────────────────

REQUIRED_FILES = [
    ("data/rag/chunks.jsonl",                      "チャンクファイル"),
    ("data/generated_QA/subset_merged_4000.jsonl", "QAデータ（テストセット元）"),
    ("data/kasen-dam-sabo_Train_set",              "訓練セット"),
]

def check_files() -> list[dict]:
    results = []
    for rel_path, desc in REQUIRED_FILES:
        p = ROOT / rel_path
        exists = p.exists()
        size   = ""
        if exists:
            if p.is_file():
                lines = sum(1 for _ in open(p, encoding="utf-8", errors="replace"))
                size  = f"{lines} 行" if p.suffix == ".jsonl" else f"{p.stat().st_size // 1024} KB"
            else:
                size = f"{sum(1 for _ in p.iterdir())} ファイル"
        results.append({"path": rel_path, "desc": desc, "ok": exists, "size": size})
    return results


# ─────────────────────────────────────────────────────────
# 5. Ollama モデル確認（必要なモデルが揃っているか）
# ─────────────────────────────────────────────────────────

EXPECTED_MODELS = {
    "swallow": ["swallow8b-lora-n4000-v09-q4", "swallow8b-lora-n4000-v09"],
    "elyza":   ["elyza8b-lora-n4000-q4",       "elyza8b-lora-n4000"],
    "judge":   ["qwen2.5:7b", "qwen2.5:14b",    "qwen2.5:7b-instruct-q4_k_m"],
}

def match_model(available: list[str], candidates: list[str]) -> str | None:
    """利用可能モデルの中から候補を先頭から順に探す。"""
    low_avail = [m.lower() for m in available]
    for c in candidates:
        # exact match
        if c in available:
            return c
        # prefix match (ignore tag variants)
        c_base = c.split(":")[0].lower()
        for orig, low in zip(available, low_avail):
            if low.startswith(c_base):
                return orig
    return None


# ─────────────────────────────────────────────────────────
# main
# ─────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  RAG 比較実験 — 環境確認")
    print("=" * 60)

    all_ok = True

    # ── 1. Ollama ──
    print("\n[1] Ollama 疎通確認")
    ollama_ok, models = check_ollama()
    if ollama_ok:
        print(f"  ✓ Ollama 接続OK  利用可能モデル数: {len(models)}")
        for m in sorted(models):
            print(f"    - {m}")
    else:
        print(f"  ✗ Ollama 接続NG: {models[0]}")
        all_ok = False

    # ── モデルマッチング ──
    print("\n  [モデルマッチング]")
    config: dict[str, str | None] = {}
    for role, candidates in EXPECTED_MODELS.items():
        found = match_model(models, candidates) if ollama_ok else None
        status = f"✓ {found}" if found else f"✗ 見つからない (候補: {candidates})"
        print(f"  {role:8s}: {status}")
        config[role] = found
        if not found:
            all_ok = False

    # ── 2. GPU ──
    print("\n[2] GPU 確認")
    gpu = check_gpu()
    if gpu["available"]:
        print(f"  ✓ {gpu['device']}  VRAM: {gpu['total_gb']} GB  空き: {gpu['free_gb']} GB")
        if gpu["total_gb"] < 14:
            print(f"  ⚠ VRAM が 14GB 未満 ({gpu['total_gb']} GB) — 一部モデルが動作しない可能性")
    else:
        print(f"  ⚠ GPU 利用不可: {gpu.get('reason', '不明')}  (CPU推論にフォールバック)")

    # ── 3. ライブラリ ──
    print("\n[3] 依存ライブラリ確認")
    libs = check_libraries()
    missing = []
    for l in libs:
        if l["ok"]:
            print(f"  ✓ {l['lib']:<30s}  {l['version']}")
        else:
            print(f"  ✗ {l['lib']:<30s}  未インストール")
            missing.append(l["lib"])
            all_ok = False
    if missing:
        print(f"\n  インストールコマンド例:")
        print(f"  pip install {' '.join(missing)}")

    # ── 4. ファイル ──
    print("\n[4] データファイル確認")
    files = check_files()
    for f in files:
        if f["ok"]:
            print(f"  ✓ {f['path']:<50s}  {f['size']}")
        else:
            print(f"  ✗ {f['path']}")
            all_ok = False

    # ── サマリー ──
    print("\n" + "=" * 60)
    if all_ok:
        print("  ✅ 全項目 OK — 実験を開始できます")
    else:
        print("  ⚠  未解決の問題があります。上記を確認してください。")
    print("=" * 60)

    # ── 推奨設定を JSON で保存 ──
    if ollama_ok:
        cfg_path = Path(__file__).parent / "env_config.json"
        with open(cfg_path, "w", encoding="utf-8") as f:
            json.dump({
                "ollama_url":    "http://localhost:11434",
                "model_swallow": config.get("swallow"),
                "model_elyza":   config.get("elyza"),
                "model_judge":   config.get("judge"),
                "gpu":           gpu,
            }, f, ensure_ascii=False, indent=2)
        print(f"\n  設定を保存しました: {cfg_path}")

    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
