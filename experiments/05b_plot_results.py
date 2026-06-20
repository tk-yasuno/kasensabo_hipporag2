"""
experiments/05b_plot_results.py
────────────────────────────────────────────────────────────
Phase 5b: 結果可視化

experiments/results/summary.json から 3種のグラフを生成する:

  1. bar_judge_score.png   — RAG方式 × モデル の Judge 平均スコア（バーチャート）
  2. bar_perfect_rate.png  — RAG方式 × モデル の 3点率（バーチャート）
  3. latency_comparison.png — Retrieval / Generation latency 比較
  4. score_distribution.png — スコア分布（0/1/2/3点の積み上げ棒グラフ）

Usage:
    python experiments/05b_plot_results.py
    python experiments/05b_plot_results.py --no-show   # ファイル保存のみ
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

EXP_DIR     = Path(__file__).parent
RESULTS_DIR = EXP_DIR / "results"
FIG_DIR     = RESULTS_DIR / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)


# ─────────────────────────────────────────────────────────
# データロード
# ─────────────────────────────────────────────────────────

def load_summary() -> list[dict]:
    path = RESULTS_DIR / "summary.json"
    if not path.exists():
        print("ERROR: summary.json が見つかりません。先に 05_aggregate_results.py を実行してください。")
        sys.exit(1)
    data = json.loads(path.read_text(encoding="utf-8"))
    return data.get("table", [])


# ─────────────────────────────────────────────────────────
# 1. Judge 平均スコア バーチャート
# ─────────────────────────────────────────────────────────

def plot_judge_score(rows: list[dict], show: bool):
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.rcParams["font.family"] = ["Meiryo", "MS Gothic", "DejaVu Sans"]

    models  = sorted(set(r["model"] for r in rows if r["avg_score"] != ""))
    rag_types = ["naive", "light", "hipporag2"]
    rag_labels = {"naive": "Naive RAG", "light": "Light RAG", "hipporag2": "HippoRAG2"}

    x      = range(len(rag_types))
    width  = 0.35
    colors = {"swallow": "#2196F3", "elyza": "#F44336"}

    fig, ax = plt.subplots(figsize=(10, 6))
    for i, model in enumerate(models):
        scores = []
        for rag in rag_types:
            matched = [r for r in rows if r["model"] == model and r["rag"] == rag and r["avg_score"] != ""]
            scores.append(float(matched[0]["avg_score"]) if matched else 0)
        offset = (i - len(models) / 2 + 0.5) * width
        bars = ax.bar([xi + offset for xi in x], scores, width,
                      label=model.capitalize(), color=colors.get(model, "#999"),
                      alpha=0.85, edgecolor="white")
        for bar, val in zip(bars, scores):
            if val > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                        f"{val:.2f}", ha="center", va="bottom", fontsize=9)

    ax.set_xlabel("RAG 方式", fontsize=12)
    ax.set_ylabel("Judge 平均スコア (0-3)", fontsize=12)
    ax.set_title("RAG 方式 × モデル — Judge 平均スコア比較", fontsize=14)
    ax.set_xticks(list(x))
    ax.set_xticklabels([rag_labels.get(r, r) for r in rag_types], fontsize=11)
    ax.set_ylim(0, 3.3)
    ax.axhline(y=3, color="gray", linestyle="--", alpha=0.4, label="満点 (3.0)")
    ax.legend(fontsize=10)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()

    out = FIG_DIR / "bar_judge_score.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"  保存: {out}")
    if show:
        plt.show()
    plt.close()


# ─────────────────────────────────────────────────────────
# 2. 3点率 バーチャート
# ─────────────────────────────────────────────────────────

def plot_perfect_rate(rows: list[dict], show: bool):
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.rcParams["font.family"] = ["Meiryo", "MS Gothic", "DejaVu Sans"]

    models    = sorted(set(r["model"] for r in rows if r["perfect_rate"] != ""))
    rag_types = ["naive", "light", "hipporag2"]
    rag_labels = {"naive": "Naive RAG", "light": "Light RAG", "hipporag2": "HippoRAG2"}

    x     = range(len(rag_types))
    width = 0.35
    colors = {"swallow": "#2196F3", "elyza": "#F44336"}

    fig, ax = plt.subplots(figsize=(10, 6))
    for i, model in enumerate(models):
        rates = []
        for rag in rag_types:
            matched = [r for r in rows if r["model"] == model and r["rag"] == rag and r["perfect_rate"] != ""]
            rates.append(float(matched[0]["perfect_rate"]) * 100 if matched else 0)
        offset = (i - len(models) / 2 + 0.5) * width
        bars = ax.bar([xi + offset for xi in x], rates, width,
                      label=model.capitalize(), color=colors.get(model, "#999"),
                      alpha=0.85, edgecolor="white")
        for bar, val in zip(bars, rates):
            if val > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                        f"{val:.1f}%", ha="center", va="bottom", fontsize=9)

    ax.set_xlabel("RAG 方式", fontsize=12)
    ax.set_ylabel("3点率 (Perfect-Score Rate) [%]", fontsize=12)
    ax.set_title("RAG 方式 × モデル — 3点率比較", fontsize=14)
    ax.set_xticks(list(x))
    ax.set_xticklabels([rag_labels.get(r, r) for r in rag_types], fontsize=11)
    ax.set_ylim(0, 105)
    ax.legend(fontsize=10)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()

    out = FIG_DIR / "bar_perfect_rate.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"  保存: {out}")
    if show:
        plt.show()
    plt.close()


# ─────────────────────────────────────────────────────────
# 3. Latency 比較
# ─────────────────────────────────────────────────────────

def plot_latency(rows: list[dict], show: bool):
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.rcParams["font.family"] = ["Meiryo", "MS Gothic", "DejaVu Sans"]
    import numpy as np

    valid = [r for r in rows if r["avg_ret_time"] != ""]
    if not valid:
        return

    conditions = [r["condition"] for r in valid]
    ret_times  = [float(r["avg_ret_time"])  for r in valid]
    gen_times  = [float(r["avg_gen_time"]) for r in valid]

    x     = np.arange(len(conditions))
    width = 0.6

    fig, ax = plt.subplots(figsize=(12, 6))
    bars1 = ax.bar(x, ret_times, width, label="Retrieval (秒)", color="#4CAF50", alpha=0.8)
    bars2 = ax.bar(x, gen_times, width, bottom=ret_times,
                   label="Generation (秒)", color="#FF9800", alpha=0.8)

    for bar, ret, gen in zip(bars2, ret_times, gen_times):
        total = ret + gen
        ax.text(bar.get_x() + bar.get_width() / 2,
                total + 0.3,
                f"{total:.1f}s", ha="center", va="bottom", fontsize=8)

    ax.set_xlabel("条件 (model_rag)", fontsize=12)
    ax.set_ylabel("平均処理時間 (秒/問)", fontsize=12)
    ax.set_title("条件別 レイテンシ比較", fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(conditions, rotation=30, ha="right", fontsize=9)
    ax.legend(fontsize=10)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()

    out = FIG_DIR / "latency_comparison.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"  保存: {out}")
    if show:
        plt.show()
    plt.close()


# ─────────────────────────────────────────────────────────
# 4. スコア分布 積み上げ棒グラフ
# ─────────────────────────────────────────────────────────

def plot_score_distribution(rows: list[dict], show: bool):
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.rcParams["font.family"] = ["Meiryo", "MS Gothic", "DejaVu Sans"]
    import numpy as np

    valid = [r for r in rows if r["n_valid_judge"] != ""]
    if not valid:
        return

    conditions  = [r["condition"] for r in valid]
    n_valid     = [int(r["n_valid_judge"]) for r in valid]

    score_keys  = ["score_0", "score_1", "score_2", "score_3"]
    score_labels = ["0点", "1点", "2点", "3点"]
    colors       = ["#F44336", "#FF9800", "#2196F3", "#4CAF50"]

    data = {
        k: [int(r.get(k, 0)) / max(int(r.get("n_valid_judge", 1)), 1) * 100
            for r in valid]
        for k in score_keys
    }

    x     = np.arange(len(conditions))
    width = 0.6
    bottom = np.zeros(len(conditions))

    fig, ax = plt.subplots(figsize=(12, 6))
    for k, label, color in zip(score_keys, score_labels, colors):
        vals = data[k]
        ax.bar(x, vals, width, bottom=bottom, label=label, color=color, alpha=0.85)
        bottom = bottom + np.array(vals)

    ax.set_xlabel("条件 (model_rag)", fontsize=12)
    ax.set_ylabel("割合 [%]", fontsize=12)
    ax.set_title("条件別 スコア分布 (0–3点)", fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(conditions, rotation=30, ha="right", fontsize=9)
    ax.set_ylim(0, 105)
    ax.legend(fontsize=10, loc="lower right")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()

    out = FIG_DIR / "score_distribution.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"  保存: {out}")
    if show:
        plt.show()
    plt.close()


# ─────────────────────────────────────────────────────────
# main
# ─────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-show", action="store_true", help="グラフを表示しない（ファイル保存のみ）")
    args = parser.parse_args()
    show = not args.no_show

    rows = load_summary()
    if not rows:
        print("データが空です。")
        return

    print(f"グラフ生成: {len(rows)} 条件")

    print("\n[1] Judge 平均スコア バーチャート...")
    plot_judge_score(rows, show)

    print("[2] 3点率 バーチャート...")
    plot_perfect_rate(rows, show)

    print("[3] Latency 比較...")
    plot_latency(rows, show)

    print("[4] スコア分布...")
    plot_score_distribution(rows, show)

    print(f"\n✓ 全グラフ保存完了 → {FIG_DIR}")


if __name__ == "__main__":
    main()
