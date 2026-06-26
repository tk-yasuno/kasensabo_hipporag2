#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Test retrieval_info structure for debugging"""

import sys
import importlib.util
from pathlib import Path

# experiments/03_rag_retrievers.pyをインポート
EXP_DIR = Path(__file__).parent
spec = importlib.util.spec_from_file_location("rag_retrievers", EXP_DIR / "03_rag_retrievers.py")
rag_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rag_mod)
make_retriever = rag_mod.make_retriever

def main():
    print("Creating retriever...")
    retriever = make_retriever("hipporag2", top_k=5, n_triples=20)
    
    print("Retrieving with test query...")
    query = "ダム設計における許容応力度の設定方法を説明してください"
    chunks = retriever.retrieve(query)
    
    print(f"\nRetrieved {len(chunks)} chunks")
    
    info = retriever.retrieval_info
    print(f"\nRetrieval Info Structure:")
    print(f"  Volumes: {len(info.get('volumes', []))} entries")
    print(f"  Chapters: {len(info.get('chapters', []))} entries")
    print(f"  Chunks: {len(info.get('chunks', []))} entries")
    print(f"  Fallback: {info.get('fallback', False)}")
    print(f"  n_filtered_triples: {info.get('n_filtered_triples', 0)}")
    
    if info.get('chapters'):
        print(f"\nSample Chapter Entry:")
        print(f"  {info['chapters'][0]}")
    else:
        print(f"\nNo chapters found!")
        print(f"  Checking volumes...")
        if info.get('volumes'):
            print(f"  Sample Volume: {info['volumes'][0]}")

if __name__ == "__main__":
    main()
