"""
experiments/02_prepare_testset.py
────────────────────────────────────────────────────────────
Phase 2: テストセット準備

data/generated_QA/subset_merged_4000.jsonl から 200問を
ランダムサンプリングして experiments/testset_200.jsonl に保存する。

seed=42 で固定（再現性確保）

出力フォーマット (JSONL 各行):
{
  "idx":      0,
  "question": "...",
  "answer":   "...",
  "source":   "..."   // metadata.src があれば
}

Usage:
    python experiments/02_prepare_testset.py
    python experiments/02_prepare_testset.py --n 100
    python experiments/02_prepare_testset.py --seed 123
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

ROOT    = Path(__file__).parent.parent
QA_SRC  = ROOT / "data" / "generated_QA" / "subset_merged_4000.jsonl"
OUT_DIR = Path(__file__).parent
OUT_PATH = OUT_DIR / "testset_200.jsonl"


def load_qa(path: Path) -> list[dict]:
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            l = line.strip()
            if l:
                records.append(json.loads(l))
    return records


def sample_qa(records: list[dict], n: int, seed: int) -> list[dict]:
    rng     = random.Random(seed)
    sampled = rng.sample(records, min(n, len(records)))
    return sampled


def convert(raw: dict, idx: int) -> dict:
    meta   = raw.get("metadata", {})
    source = meta.get("src", "") or meta.get("source", "")
    return {
        "idx":      idx,
        "question": raw.get("instruction", raw.get("question", "")),
        "answer":   raw.get("output",      raw.get("answer",   "")),
        "source":   source,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n",    type=int, default=200,  help="サンプリング件数")
    parser.add_argument("--seed", type=int, default=42,   help="乱数シード")
    parser.add_argument("--force", action="store_true",   help="既存ファイルを上書き")
    args = parser.parse_args()

    if OUT_PATH.exists() and not args.force:
        print(f"テストセット既存: {OUT_PATH}  (--force で上書き)")
        with open(OUT_PATH, encoding="utf-8") as f:
            n_lines = sum(1 for l in f if l.strip())
        print(f"  {n_lines} 件")
        return

    if not QA_SRC.exists():
        print(f"ERROR: QA ファイルが見つかりません: {QA_SRC}")
        sys.exit(1)

    print(f"QAデータロード: {QA_SRC}", flush=True)
    records = load_qa(QA_SRC)
    print(f"  {len(records)} 件ロード")

    sampled = sample_qa(records, args.n, args.seed)
    print(f"  サンプリング: {len(sampled)} 件 (seed={args.seed})")

    converted = [convert(r, i) for i, r in enumerate(sampled)]

    with open(OUT_PATH, "w", encoding="utf-8") as f:
        for rec in converted:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"\n✓ テストセット保存: {OUT_PATH}  ({len(converted)} 件)")

    # 先頭3件プレビュー
    print("\n[先頭3件プレビュー]")
    for rec in converted[:3]:
        q = rec["question"][:60]
        a = rec["answer"][:60]
        print(f"  [{rec['idx']}] Q: {q}")
        print(f"       A: {a}")
        print()


if __name__ == "__main__":
    main()
