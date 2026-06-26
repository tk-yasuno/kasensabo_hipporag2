import json

records = [json.loads(l) for l in open('experiments/results/swallow_hipporag2_results.jsonl', encoding='utf-8')]

# 最初の3問のretrieved chunksを確認
for i in range(3):
    print(f'\n=== Question {i} ===')
    print(f'Question: {records[i]["question"][:80]}...')
    print(f'Retrieved chunks: {len(records[i].get("retrieved_chunks", []))}')
    for j, chunk in enumerate(records[i].get('retrieved_chunks', [])[:3]):
        print(f'  Chunk {j}: doc_id={chunk.get("doc_id")}, score={chunk.get("score", 0):.3f}')
