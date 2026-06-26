"""
v0.2.2 Trial vs v0.2.1 比較分析スクリプト
────────────────────────────────────────
embedding 30% + keyword 70% (v0.2.2)
vs
embedding 60% + keyword 40% (v0.2.1)

を Judge スコア・Cosine類似度で比較
"""

import json
from pathlib import Path

# 結果ファイルパス
RESULTS_DIR = Path(__file__).parent / "results"

def load_summary(condition_name: str) -> dict:
    """サマリーファイルを読み込み"""
    file_path = RESULTS_DIR / f"{condition_name}_summary.json"
    if not file_path.exists():
        return None
    with open(file_path, encoding="utf-8") as f:
        return json.load(f)

def compare_conditions(v021_name: str, v022_name: str):
    """v0.2.1 vs v0.2.2を比較"""
    v021_data = load_summary(v021_name)
    v022_data = load_summary(v022_name)
    
    if not v021_data or not v022_data:
        print(f"ERROR: データが見つかりません")
        return
    
    print(f"\n{'='*70}")
    print(f"  v0.2.1 vs v0.2.2 比較: {v021_name} → {v022_name}")
    print(f"{'='*70}")
    
    # v0.2.1
    v021_score = v021_data.get("avg_score", 0.0)
    v021_perfect_rate = v021_data.get("perfect_rate", 0.0)
    v021_dist = v021_data.get("score_dist", {})
    
    # v0.2.2
    v022_score = v022_data.get("avg_score", 0.0)
    v022_perfect_rate = v022_data.get("perfect_rate", 0.0)
    v022_dist = v022_data.get("score_dist", {})
    
    # 差分計算
    score_diff = v022_score - v021_score
    score_improvement = (score_diff / v021_score * 100) if v021_score > 0 else 0
    perfect_diff = v022_perfect_rate - v021_perfect_rate
    
    # 出力
    print(f"\nJudge 平均スコア:")
    print(f"  v0.2.1 (60:40)  : {v021_score:.3f} / 3.0")
    print(f"  v0.2.2 (30:70)  : {v022_score:.3f} / 3.0")
    print(f"  改善量          : {score_diff:+.3f} ({score_improvement:+.1f}%)")
    
    print(f"\nPerfect-Score率 (3点率):")
    print(f"  v0.2.1 (60:40)  : {v021_perfect_rate:.1f}%")
    print(f"  v0.2.2 (30:70)  : {v022_perfect_rate:.1f}%")
    print(f"  改善量          : {perfect_diff:+.1f}%ポイント")
    
    print(f"\nスコア分布:")
    print(f"{'スコア':<8} {'v0.2.1':<15} {'v0.2.2':<15} {'差分':<10}")
    print(f"{'-'*50}")
    for score in ['0', '1', '2', '3']:
        v021_count = v021_dist.get(score, 0)
        v022_count = v022_dist.get(score, 0)
        diff = v022_count - v021_count
        print(f"{score}点    {v021_count:<15} {v022_count:<15} {diff:+d}")
    
    print(f"\n{'='*70}")
    if score_improvement > 0:
        print(f"✅ キーワード強化 (30:70) が有効！ {score_improvement:.1f}% 改善")
    else:
        print(f"⚠️  キーワード強化による明確な改善なし")
    print(f"{'='*70}\n")

if __name__ == "__main__":
    print("\n🔍 v0.2.2 Trial 比較分析\n")
    
    # elyza HippoRAG2 の比較
    print("[1] elyza HippoRAG2")
    compare_conditions("elyza_hipporag2", "elyza_hipporag2_v022")
    
    # swallow HippoRAG2 の比較（可能であれば）
    print("[2] swallow HippoRAG2")
    compare_conditions("swallow_hipporag2", "swallow_hipporag2_v022")
    
    print("\n📝 次のステップ:")
    print("1. Cosine 類似度評価を実行")
    print("   python experiments/04e_similarity_only.py")
    print("2. 結果を可視化")
    print("   python experiments/05d_v022_comparison_plot.py")
