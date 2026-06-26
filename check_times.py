import json

records = [json.loads(l) for l in open('experiments/results/swallow_hipporag2_results.jsonl', encoding='utf-8')]

print('All retrieval times:')
for r in records:
    print(f'Q{r["idx"]}: {r.get("ret_time", 0):.3f}s')

print(f'\nAverage: {sum(r.get("ret_time", 0) for r in records) / len(records):.3f}s')
print(f'Total questions: {len(records)}')
