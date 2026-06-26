"""
Triple filtering の効果を定量的に分析するスクリプト

filtered triples が chunk selection にどれだけ貢献しているかを確認
"""
import json
from pathlib import Path
from collections import Counter

# Triple データを読み込み
with open('indices/triples.json', encoding='utf-8') as f:
    triples = json.load(f)

print("=== Triple Filtering の効果分析 ===\n")

# シミュレーション: 5個の triple が filtering を通過した場合
print("【シミュレーション】Filtered triples = 5個の場合")
print("-" * 50)

# ケース1: 現状（GROUP_SIZE=4、全て同じバッチから）
case1_triples = triples[0:5]  # 同じバッチの triple
case1_chunks = set()
for t in case1_triples:
    case1_chunks.update(t['chunk_ids'])

print(f"\n現状（GROUP_SIZE=4）:")
print(f"  Filtered triples: 5個")
print(f"  → 追加される chunk 数: {len(case1_chunks)}個")
print(f"  → 効果: {len(case1_chunks)/5*100:.1f}% のみ（重複が多い）")

# ケース2: GROUP_SIZE=1 の場合（異なる chunk）
case2_triples = [triples[i] for i in range(0, len(triples), 200)][:5]  # 異なる位置から抽出
case2_chunks = set()
for t in case2_triples:
    case2_chunks.update(t['chunk_ids'])

print(f"\nGROUP_SIZE=1 の場合:")
print(f"  Filtered triples: 5個")
print(f"  → 追加される chunk 数: {len(case2_chunks)}個")
print(f"  → 効果: {len(case2_chunks)/5*100:.1f}%")

# 実データでの分析
print("\n\n【実データ分析】")
print("-" * 50)

# 連続する N 個の triple からユニークな chunk 数を計算
def analyze_consecutive_triples(triples_list, n=5):
    """連続する n 個の triple から得られるユニークな chunk 数"""
    results = []
    for i in range(0, len(triples_list) - n + 1, 10):  # 10個おきにサンプリング
        batch = triples_list[i:i+n]
        unique_chunks = set()
        for t in batch:
            unique_chunks.update(t['chunk_ids'])
        results.append(len(unique_chunks))
    return results

# 5個、10個、20個の triple での分析
for n_triples in [5, 10, 20]:
    unique_counts = analyze_consecutive_triples(triples, n=n_triples)
    avg_unique = sum(unique_counts) / len(unique_counts)
    max_unique = max(unique_counts)
    min_unique = min(unique_counts)
    
    print(f"\nFiltered triples = {n_triples}個の場合:")
    print(f"  平均追加 chunk 数: {avg_unique:.1f}個")
    print(f"  最大: {max_unique}個  最小: {min_unique}個")
    print(f"  効果率: {avg_unique/n_triples*100:.1f}%")

# GROUP_SIZE=4 の影響を定量化
print("\n\n【GROUP_SIZE=4 の影響】")
print("-" * 50)

# 同じバッチ内の triple 数をカウント
batch_triple_counts = Counter()
for i in range(0, len(triples), 10):  # サンプリング
    batch_id = tuple(triples[i]['chunk_ids'])
    batch_triple_counts[batch_id] += 1

print(f"同じ chunk_ids を持つ triple 数の分布:")
batch_sizes = list(batch_triple_counts.values())
print(f"  平均: {sum(batch_sizes)/len(batch_sizes):.1f}個/バッチ")
print(f"  最大: {max(batch_sizes)}個/バッチ")
print(f"  最小: {min(batch_sizes)}個/バッチ")

print("\n\n【結論】")
print("=" * 50)
print("問題: 4-chunk バッチ処理により、filtered triples の")
print("      chunk selection への効果が大幅に減少している。")
print("")
print("推奨: GROUP_SIZE を 4 → 1 に変更")
print("  メリット: Triple の粒度が細かくなり、検索精度向上")
print("  デメリット: 処理時間が 2.5-3h → 10-12h に増加")
