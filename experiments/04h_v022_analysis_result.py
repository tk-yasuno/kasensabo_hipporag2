"""
v0.2.2 Trial 比較分析 — elyza HippoRAG2
============================================================

v0.2.1: embedding 60% + keyword 40%
v0.2.2: embedding 30% + keyword 70%
"""

print("\n" + "="*70)
print("  v0.2.1 vs v0.2.2 比較: elyza HippoRAG2")
print("="*70)

# v0.2.1 (旧結果)
v021_data = {
    "avg_score": 0.765,
    "perfect_rate": 2.0,
    "perfect_count": 4,
    "score_dist": {
        "0": 70,
        "1": 111,
        "2": 15,
        "3": 4
    }
}

# v0.2.2 (新結果) — ユーザーから報告
v022_data = {
    "avg_score": 0.910,
    "perfect_rate": 10.5,
    "perfect_count": 21,
    "score_dist": {
        "0": 60,
        "1": 119,
        "2": 0,
        "3": 21
    }
}

# 計算
score_diff = v022_data["avg_score"] - v021_data["avg_score"]
score_improvement = (score_diff / v021_data["avg_score"] * 100)
perfect_diff = v022_data["perfect_rate"] - v021_data["perfect_rate"]

print(f"\n📊 Judge 平均スコア:")
print(f"  v0.2.1 (embedding 60% + keyword 40%) : {v021_data['avg_score']:.3f} / 3.0")
print(f"  v0.2.2 (embedding 30% + keyword 70%) : {v022_data['avg_score']:.3f} / 3.0")
print(f"  改善量                                : {score_diff:+.3f}  ({score_improvement:+.1f}%)")

print(f"\n🏆 Perfect-Score率 (3点率):")
print(f"  v0.2.1 (embedding 60% + keyword 40%) : {v021_data['perfect_rate']:.1f}% ({v021_data['perfect_count']}問)")
print(f"  v0.2.2 (embedding 30% + keyword 70%) : {v022_data['perfect_rate']:.1f}% ({v022_data['perfect_count']}問)")
print(f"  改善量                                : {perfect_diff:+.1f}%ポイント ({v022_data['perfect_count'] - v021_data['perfect_count']:+d}問)")

print(f"\n📈 スコア分布の比較:")
print(f"{'スコア':<8} {'v0.2.1':<15} {'v0.2.2':<15} {'差分':<10} {'評価':<15}")
print(f"{'-'*65}")
for score in ['0', '1', '2', '3']:
    v021_count = v021_data["score_dist"].get(score, 0)
    v022_count = v022_data["score_dist"].get(score, 0)
    diff = v022_count - v021_count
    if score == '3':
        eval_text = "✅ 大幅改善"
    elif score == '0':
        eval_text = "✅ 削減"
    else:
        eval_text = ""
    print(f"{score}点    {v021_count:<15} {v022_count:<15} {diff:+d}     {eval_text:<15}")

print(f"\n{'='*70}")
print(f"✅ キーワード強化 (embedding 30% + keyword 70%) が有効！")
print(f"   → Judge スコア: {score_improvement:.1f}% 改善")
print(f"   → Perfect率: {perfect_diff:.1f}%ポイント 改善（{v022_data['perfect_count'] - v021_data['perfect_count']}問増加）")
print(f"\n📌 重要な観察:")
print(f"   ・0点の削減: {abs(v021_data['score_dist']['0'] - v022_data['score_dist']['0'])} 問削減")
print(f"   ・2点がほぼ消滅: v0.2.1では15問→v0.2.2では0問 (極化傾向)")
print(f"   ・Perfect率5倍以上の改善: 2.0% → 10.5%")
print(f"{'='*70}\n")

# 次のステップ
print("📝 次のステップ:")
print("1. swallow HippoRAG2 (v0.2.2) 結果の確認・比較")
print("2. Cosine類似度評価で両バージョンを比較")
print("3. v0.2.2完成版として commit & push")
