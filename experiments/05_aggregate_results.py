"""
experiments/05_aggregate_results.py
────────────────────────────────────────────────────────────
Phase 5: 結果集計

experiments/results/ 内の *_summary.json を読み込み、
6条件の比較表を生成する。

出力:
  experiments/results/summary.csv      — 集計表
  experiments/results/summary.json     — JSON 形式集計

Usage:
    python experiments/05_aggregate_results.py
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

EXP_DIR     = Path(__file__).parent
RESULTS_DIR = EXP_DIR / "results"


def load_summaries() -> list[dict]:
    summaries = []
    for p in sorted(RESULTS_DIR.glob("*_summary.json")):
        if p.name == "summary.json":
            continue
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
            summaries.append(data)
        except Exception as e:
            print(f"  [WARN] 読み込み失敗: {p.name} — {e}")
    return summaries


def main():
    if not RESULTS_DIR.exists():
        print(f"ERROR: results/ ディレクトリが見つかりません: {RESULTS_DIR}")
        print("  先に 04_eval_rag.py を実行してください。")
        sys.exit(1)

    summaries = load_summaries()
    if not summaries:
        print("サマリーファイルが見つかりません。04_eval_rag.py を実行してください。")
        return

    print(f"\n{'='*70}")
    print(f"  RAG 比較実験 — 結果集計  ({len(summaries)} 条件)")
    print(f"{'='*70}\n")

    # ── テーブル表示 ──
    headers = [
        "condition", "model", "rag",
        "avg_score", "perfect_rate",
        "avg_ret_time", "avg_gen_time",
        "n_questions", "n_valid_judge",
        "score_0", "score_1", "score_2", "score_3",
    ]

    rows = []
    for s in summaries:
        dist = s.get("score_dist", {})
        row = {
            "condition":     s.get("condition", ""),
            "model":         s.get("model", ""),
            "rag":           s.get("rag", ""),
            "avg_score":     s.get("avg_score", ""),
            "perfect_rate":  s.get("perfect_rate", ""),
            "avg_ret_time":  s.get("avg_ret_time", ""),
            "avg_gen_time":  s.get("avg_gen_time", ""),
            "n_questions":   s.get("n_questions", ""),
            "n_valid_judge": s.get("n_valid_judge", ""),
            "score_0":       dist.get("0", dist.get(0, "")),
            "score_1":       dist.get("1", dist.get(1, "")),
            "score_2":       dist.get("2", dist.get(2, "")),
            "score_3":       dist.get("3", dist.get(3, "")),
        }
        rows.append(row)

    # ── ターミナル表示 ──
    col_widths = {h: max(len(h), max(len(str(r.get(h, ""))) for r in rows)) for h in headers}
    header_line = "  ".join(h.ljust(col_widths[h]) for h in headers)
    print(header_line)
    print("-" * len(header_line))
    for row in sorted(rows, key=lambda r: (r["rag"], r["model"])):
        print("  ".join(str(row.get(h, "")).ljust(col_widths[h]) for h in headers))

    # ── ランキング ──
    valid_rows = [r for r in rows if r["avg_score"] != ""]
    if valid_rows:
        ranked = sorted(valid_rows, key=lambda r: float(r["avg_score"]), reverse=True)
        print(f"\n[Judge スコアランキング]")
        for i, r in enumerate(ranked, 1):
            perfect_pct = float(r.get("perfect_rate", 0)) * 100
            print(f"  {i}. {r['condition']:<25s}  avg={r['avg_score']}  "
                  f"perfect={perfect_pct:.1f}%  "
                  f"ret={r['avg_ret_time']}s  gen={r['avg_gen_time']}s")

    # ── CSV 保存 ──
    csv_path = RESULTS_DIR / "summary.csv"
    with open(csv_path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\n✓ CSV 保存: {csv_path}")

    # ── JSON 保存 ──
    json_path = RESULTS_DIR / "summary.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({"conditions": summaries, "table": rows}, f, ensure_ascii=False, indent=2)
    print(f"✓ JSON 保存: {json_path}")

    # ── モデル別 / RAG別 平均 ──
    if valid_rows:
        print("\n[RAG 方式別 平均スコア]")
        rag_groups: dict[str, list[float]] = {}
        for r in valid_rows:
            rag_groups.setdefault(r["rag"], []).append(float(r["avg_score"]))
        for rag, scores in sorted(rag_groups.items()):
            print(f"  {rag:<12s}: {sum(scores)/len(scores):.3f}  (n={len(scores)})")

        print("\n[モデル別 平均スコア]")
        model_groups: dict[str, list[float]] = {}
        for r in valid_rows:
            model_groups.setdefault(r["model"], []).append(float(r["avg_score"]))
        for model, scores in sorted(model_groups.items()):
            print(f"  {model:<12s}: {sum(scores)/len(scores):.3f}  (n={len(scores)})")


if __name__ == "__main__":
    main()
