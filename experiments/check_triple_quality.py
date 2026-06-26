import json
from pathlib import Path
from collections import Counter

# Triple データを読み込み
with open('indices/triples.json', encoding='utf-8') as f:
    triples = json.load(f)

# Chunks データを読み込み
chunks = []
with open('../data/rag/chunks.jsonl', encoding='utf-8') as f:
    for line in f:
        chunks.append(json.loads(line))

# 最初の5個の triple を詳細表示
print('=== Triple サンプル (最初の5個) ===')
for i, t in enumerate(triples[:5]):
    print(f'\n[Triple {i}]')
    print(f'  chunk_ids: {t["chunk_ids"]}')
    print(f'  volume_id: {t["volume_id"]}')
    print(f'  chapter_id: {t["chapter_id"]}')
    print(f'  triple: {t["triple"]["subject"]} --[{t["triple"]["relation"]}]--> {t["triple"]["object"]}')
    
    # 対応する chunks の内容を表示
    print(f'  対応chunk内容:')
    for cid in t['chunk_ids'][:2]:  # 最初の2個だけ
        if cid < len(chunks):
            chunk = chunks[cid]
            text_preview = chunk['text'][:100].replace('\n', ' ')
            print(f'    chunk_{cid}: {text_preview}...')

# ランダムに中間の triple も確認
print('\n\n=== Triple サンプル (中間から5個) ===')
start = len(triples) // 2
for i, t in enumerate(triples[start:start+5]):
    print(f'\n[Triple {start+i}]')
    print(f'  chunk_ids: {t["chunk_ids"]}')
    print(f'  volume_id: {t["volume_id"]}')
    print(f'  chapter_id: {t["chapter_id"]}')
    print(f'  triple: {t["triple"]["subject"]} --[{t["triple"]["relation"]}]--> {t["triple"]["object"]}')

# 統計情報
print(f'\n\n=== 統計情報 ===')
print(f'Total triples: {len(triples)}')
print(f'Total chunks: {len(chunks)}')

# chunk_ids の長さ分布
chunk_ids_lens = Counter([len(t['chunk_ids']) for t in triples])
print(f'chunk_ids 長さ分布: {dict(chunk_ids_lens)}')

# volume_id 分布
volume_dist = Counter([t['volume_id'] for t in triples])
print(f'volume_id 分布 (上位10):')
for vol, count in volume_dist.most_common(10):
    print(f'  {vol}: {count}')

# Triple の関係性の種類
relation_dist = Counter([t['triple']['relation'] for t in triples])
print(f'\nRelation 分布 (上位10):')
for rel, count in relation_dist.most_common(10):
    print(f'  {rel}: {count}')
