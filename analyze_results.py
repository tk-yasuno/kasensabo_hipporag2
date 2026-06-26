import json

# 結果ファイルを読み込み
records = [json.loads(l) for l in open('experiments/results/swallow_hipporag2_results.jsonl', encoding='utf-8')]

print("=" * 60)
print("swallow_hipporag2 (v0.3.1.5 with triple filtering)")
print("=" * 60)

# 基本統計
print(f"\nTotal questions: {len(records)}")
print(f"avg Retrieval time: {sum(r.get('ret_time', 0) for r in records) / len(records):.2f}s")
print(f"avg Generation time: {sum(r.get('gen_time', 0) for r in records) / len(records):.2f}s")
print(f"avg Total time: {sum(r.get('total_time', 0) for r in records) / len(records):.2f}s")

# Judge スコア
judge_scores = [r.get('judge_score', 0) for r in records]
print(f"\navg Judge score: {sum(judge_scores) / len(judge_scores):.3f} / 3.0")
print(f"Score distribution:")
for score in range(4):
    count = judge_scores.count(score)
    print(f"  {score}点: {count} ({count/len(records)*100:.1f}%)")
print(f"Perfect-Score (3点): {judge_scores.count(3)}/{len(records)} ({judge_scores.count(3)/len(records)*100:.1f}%)")

# Retrieval time の分布
ret_times = [r.get('ret_time', 0) for r in records]
print(f"\nRetrieval time distribution:")
print(f"  Min: {min(ret_times):.2f}s")
print(f"  Max: {max(ret_times):.2f}s")
print(f"  Median: {sorted(ret_times)[len(ret_times)//2]:.2f}s")
print(f"  Q1 (25%): {sorted(ret_times)[len(ret_times)//4]:.2f}s")
print(f"  Q3 (75%): {sorted(ret_times)[len(ret_times)*3//4]:.2f}s")

# Triple filtering の効果（retrieved_chunks のサイズで推定）
chunk_counts = [len(r.get('retrieved_chunks', [])) for r in records]
if chunk_counts:
    print(f"\nRetrieved chunks per query:")
    print(f"  avg: {sum(chunk_counts) / len(chunk_counts):.1f}")
    print(f"  Min: {min(chunk_counts)}")
    print(f"  Max: {max(chunk_counts)}")
