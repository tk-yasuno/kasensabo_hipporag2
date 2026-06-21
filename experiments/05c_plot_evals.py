"""
experiments/05c_plot_evals.py
──────────────────────────────────────────────────────────
Phase 5c: 類似度評価結果の可視化

experiments/evals/ 内の類似度評価ファイルを読み込み、
6条件（swallow/elyza × naive/light/hipporag2）を比較する図を生成します。

生成される図:
  1. 類似度スコア：6条件比較（棒グラフ + エラーバー）
  2. 類似度スコア分布：Violin plot
  3. 類似度分布内訳：スタックバー（0.0-0.2, ..., 0.8-1.0）

Usage:
    python experiments/05c_plot_evals.py                 # すべての図を生成
    python experiments/05c_plot_evals.py --no-show       # 画面表示なし
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import NamedTuple

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import japanize_matplotlib

EXP_DIR = Path(__file__).parent
EVALS_DIR = EXP_DIR / "evals"
FIGURES_DIR = EVALS_DIR / "figures"

FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# 図のスタイル設定
plt.rcParams["font.family"] = "DejaVu Sans"
japanize_matplotlib.japanize()
plt.rcParams["figure.figsize"] = (12, 6)
plt.rcParams["font.size"] = 10


# ─────────────────────────────────────────────────────────
# データロード
# ─────────────────────────────────────────────────────────

class EvalData(NamedTuple):
    condition: str
    avg_similarity: float
    std_similarity: float
    min_similarity: float
    max_similarity: float
    similarities: list[float]
    similarity_dist: dict[str, int]


def load_eval_results() -> dict[str, EvalData]:
    """evals ディレクトリから全条件の結果をロード"""
    results = {}
    
    if not EVALS_DIR.exists():
        print(f"❌ {EVALS_DIR} が見つかりません")
        return results
    
    for eval_file in sorted(EVALS_DIR.glob("*_similarity.json")):
        try:
            with open(eval_file, encoding="utf-8") as f:
                data = json.load(f)
            
            condition = data["condition"]
            stats = data["statistics"]
            similarities = [s["similarity"] for s in data["similarities"]]
            
            results[condition] = EvalData(
                condition=condition,
                avg_similarity=stats["avg_similarity"],
                std_similarity=stats["std_similarity"],
                min_similarity=stats["min_similarity"],
                max_similarity=stats["max_similarity"],
                similarities=similarities,
                similarity_dist=stats["similarity_dist"]
            )
            print(f"✓ ロード: {eval_file.name}")
        except Exception as e:
            print(f"⚠ エラー ({eval_file.name}): {e}")
    
    return results


# ─────────────────────────────────────────────────────────
# 可視化関数
# ─────────────────────────────────────────────────────────

def plot_similarity_comparison(results: dict[str, EvalData]):
    """類似度スコア：6条件比較（棒グラフ + エラーバー）"""
    
    if not results:
        print("❌ 結果データがありません")
        return
    
    # 条件を並び替え（swallow → elyza、naive → light → hipporag2）
    order = []
    for model in ["swallow", "elyza"]:
        for rag in ["naive", "light", "hipporag2"]:
            key = f"{model}_{rag}"
            if key in results:
                order.append(key)
    
    conditions = order
    avg_scores = [results[c].avg_similarity for c in conditions]
    std_scores = [results[c].std_similarity for c in conditions]
    
    # 短い条件名を表示
    short_labels = [c.replace("_", "\n") for c in conditions]
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    colors = []
    for cond in conditions:
        if "swallow" in cond:
            if "naive" in cond:
                colors.append("#FF6B6B")
            elif "light" in cond:
                colors.append("#FF8E72")
            else:  # hipporag2
                colors.append("#FF3838")
        else:  # elyza
            if "naive" in cond:
                colors.append("#4ECDC4")
            elif "light" in cond:
                colors.append("#45B7AA")
            else:  # hipporag2
                colors.append("#1A7B74")
    
    x_pos = np.arange(len(conditions))
    bars = ax.bar(x_pos, avg_scores, yerr=std_scores, capsize=5, color=colors, alpha=0.8, edgecolor="black", linewidth=1.5)
    
    ax.set_xlabel("条件（モデル × RAG方式）", fontsize=12, fontweight="bold")
    ax.set_ylabel("平均コサイン類似度", fontsize=12, fontweight="bold")
    ax.set_title("類似度スコア：6条件比較\n（エラーバー = 標準偏差）", fontsize=14, fontweight="bold")
    ax.set_xticks(x_pos)
    ax.set_xticklabels(short_labels, fontsize=10)
    ax.set_ylim(0, 1.0)
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    
    # 値をバーの上に表示
    for i, (bar, score, std) in enumerate(zip(bars, avg_scores, std_scores)):
        ax.text(bar.get_x() + bar.get_width() / 2, score + std + 0.02, f"{score:.3f}", 
                ha="center", va="bottom", fontsize=9, fontweight="bold")
    
    # 凡例
    swallow_patch = mpatches.Patch(color="#FF6B6B", label="Swallow-8B")
    elyza_patch = mpatches.Patch(color="#4ECDC4", label="ELYZA-JP-8B")
    ax.legend(handles=[swallow_patch, elyza_patch], loc="upper right", fontsize=10)
    
    plt.tight_layout()
    fig_path = FIGURES_DIR / "01_similarity_comparison.png"
    plt.savefig(fig_path, dpi=150, bbox_inches="tight")
    print(f"✓ 保存: {fig_path.name}")
    plt.close()


def plot_similarity_violin(results: dict[str, EvalData]):
    """類似度スコア分布：Violin plot"""
    
    if not results:
        return
    
    # 条件を並び替え
    order = []
    for model in ["swallow", "elyza"]:
        for rag in ["naive", "light", "hipporag2"]:
            key = f"{model}_{rag}"
            if key in results:
                order.append(key)
    
    conditions = order
    similarities_list = [results[c].similarities for c in conditions]
    short_labels = [c.replace("_", "\n") for c in conditions]
    
    fig, ax = plt.subplots(figsize=(12, 7))
    
    parts = ax.violinplot(similarities_list, positions=range(len(conditions)), widths=0.7, showmeans=True, showmedians=True)
    
    # Violin の色を調整
    for i, pc in enumerate(parts["bodies"]):
        if "swallow" in conditions[i]:
            pc.set_facecolor("#FF6B6B")
        else:
            pc.set_facecolor("#4ECDC4")
        pc.set_alpha(0.7)
    
    ax.set_xlabel("条件（モデル × RAG方式）", fontsize=12, fontweight="bold")
    ax.set_ylabel("コサイン類似度", fontsize=12, fontweight="bold")
    ax.set_title("類似度スコア分布：Violin Plot\n（中央線 = 中央値、●= 平均値）", fontsize=14, fontweight="bold")
    ax.set_xticks(range(len(conditions)))
    ax.set_xticklabels(short_labels, fontsize=10)
    ax.set_ylim(-0.1, 1.1)
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    
    plt.tight_layout()
    fig_path = FIGURES_DIR / "02_similarity_violin.png"
    plt.savefig(fig_path, dpi=150, bbox_inches="tight")
    print(f"✓ 保存: {fig_path.name}")
    plt.close()


def plot_similarity_distribution(results: dict[str, EvalData]):
    """類似度分布内訳：スタックバー（0.0-0.2, ..., 0.8-1.0）"""
    
    if not results:
        return
    
    # 条件を並び替え
    order = []
    for model in ["swallow", "elyza"]:
        for rag in ["naive", "light", "hipporag2"]:
            key = f"{model}_{rag}"
            if key in results:
                order.append(key)
    
    conditions = order
    short_labels = [c.replace("_", "\n") for c in conditions]
    
    # 分布データを取得
    dist_keys = ["0.0-0.2", "0.2-0.4", "0.4-0.6", "0.6-0.8", "0.8-1.0"]
    dist_data = {key: [] for key in dist_keys}
    
    for cond in conditions:
        dist = results[cond].similarity_dist
        for key in dist_keys:
            dist_data[key].append(dist.get(key, 0))
    
    # パーセンテージに変換
    total_counts = np.array(dist_data["0.0-0.2"]) + np.array(dist_data["0.2-0.4"]) + \
                   np.array(dist_data["0.4-0.6"]) + np.array(dist_data["0.6-0.8"]) + \
                   np.array(dist_data["0.8-1.0"])
    
    percentages = {key: (np.array(dist_data[key]) / total_counts * 100).tolist() for key in dist_keys}
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    x_pos = np.arange(len(conditions))
    bottom = np.zeros(len(conditions))
    
    colors = ["#FF6B6B", "#FFB3B3", "#FFD700", "#90EE90", "#32CD32"]
    
    for i, key in enumerate(dist_keys):
        ax.bar(x_pos, percentages[key], bottom=bottom, label=f"{key}", color=colors[i], edgecolor="black", linewidth=0.5)
        bottom += np.array(percentages[key])
    
    ax.set_xlabel("条件（モデル × RAG方式）", fontsize=12, fontweight="bold")
    ax.set_ylabel("比率 (%)", fontsize=12, fontweight="bold")
    ax.set_title("類似度分布内訳：スタックバー\n（各条件の類似度スコア分布）", fontsize=14, fontweight="bold")
    ax.set_xticks(x_pos)
    ax.set_xticklabels(short_labels, fontsize=10)
    ax.set_ylim(0, 100)
    ax.legend(loc="upper right", fontsize=10, title="類似度範囲")
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    
    plt.tight_layout()
    fig_path = FIGURES_DIR / "03_similarity_distribution.png"
    plt.savefig(fig_path, dpi=150, bbox_inches="tight")
    print(f"✓ 保存: {fig_path.name}")
    plt.close()


def print_summary(results: dict[str, EvalData]):
    """結果サマリーを表示"""
    
    print("\n" + "="*60)
    print("  類似度評価 サマリー")
    print("="*60)
    
    # 条件を並び替え
    order = []
    for model in ["swallow", "elyza"]:
        for rag in ["naive", "light", "hipporag2"]:
            key = f"{model}_{rag}"
            if key in results:
                order.append(key)
    
    print("\n条件別スコア:")
    for cond in order:
        data = results[cond]
        print(f"\n  {cond:25s}")
        print(f"    平均: {data.avg_similarity:.4f}  (σ={data.std_similarity:.4f})")
        print(f"    範囲: [{data.min_similarity:.4f}, {data.max_similarity:.4f}]")
        print(f"    分布: ", end="")
        for key in ["0.0-0.2", "0.2-0.4", "0.4-0.6", "0.6-0.8", "0.8-1.0"]:
            count = data.similarity_dist.get(key, 0)
            pct = count / len(data.similarities) * 100
            print(f"{key}={count}({pct:.1f}%) ", end="")
        print()
    
    # モデル別平均
    print("\n" + "-"*60)
    print("モデル別平均:")
    for model in ["swallow", "elyza"]:
        model_scores = [results[f"{model}_{rag}"].avg_similarity 
                       for rag in ["naive", "light", "hipporag2"] 
                       if f"{model}_{rag}" in results]
        if model_scores:
            avg = np.mean(model_scores)
            print(f"  {model:10s} : {avg:.4f}")
    
    # RAG方式別平均
    print("\nRAG方式別平均:")
    for rag in ["naive", "light", "hipporag2"]:
        rag_scores = [results[f"{model}_{rag}"].avg_similarity 
                     for model in ["swallow", "elyza"] 
                     if f"{model}_{rag}" in results]
        if rag_scores:
            avg = np.mean(rag_scores)
            print(f"  {rag:10s} : {avg:.4f}")
    
    print("\n" + "="*60)


# ─────────────────────────────────────────────────────────
# メイン
# ─────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="類似度評価結果の可視化")
    parser.add_argument("--no-show", action="store_true", help="画面表示なし")
    args = parser.parse_args()
    
    print("="*60)
    print("  類似度評価結果 可視化")
    print("="*60)
    print()
    
    # 結果をロード
    results = load_eval_results()
    
    if not results:
        print("❌ 評価結果が見つかりません")
        return
    
    print(f"\n✓ {len(results)} 条件をロード")
    
    # サマリーを表示
    print_summary(results)
    
    # 図を生成
    print("\n図を生成中...")
    plot_similarity_comparison(results)
    plot_similarity_violin(results)
    plot_similarity_distribution(results)
    
    print(f"\n✓ すべての図を保存しました: {FIGURES_DIR}/")
    print(f"  - 01_similarity_comparison.png")
    print(f"  - 02_similarity_violin.png")
    print(f"  - 03_similarity_distribution.png")
    
    if not args.no_show:
        print("\n画面表示中... (Ctrl+C で終了)")
        plt.show()


if __name__ == "__main__":
    main()
