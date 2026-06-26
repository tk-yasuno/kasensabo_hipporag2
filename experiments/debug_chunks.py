import json
from pathlib import Path

chunks = []
with open("indices/chunks.jsonl", encoding="utf-8") as f:
    for line in f:
        if line.strip():
            chunks.append(json.loads(line))

print(f"Total chunks: {len(chunks)}")
print(f"\nFirst 5 chunks info:")
for i, c in enumerate(chunks[:5]):
    print(f"{i}: doc_id={c.get('doc_id')}, section_id={c.get('section_id')}, heading={c.get('heading')[:50]}")
