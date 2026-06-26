# 🚀 v0.3実行方法（3ステップ）

## Step 1: Triple Index 構築（30-60分）

cd experiments
python 01_build_triple_index.py

OpenIE で triple 抽出
hotchpotch/static-embedding-japanese で embedding
FAISS index 構築

## Step 2: v0.3 RAG 評価実行（60分）

python 04_eval_rag.py --model swallow --rag hipporag2 --judge-model qwen2.5:14b

Triple filtering 付き HippoRAG2 で検索
Swallow 8B で回答生成
Qwen2.5 で採点 qwen2.5:15b
Step 3: 結果分析

## Step 3: 結果分析

python 05_aggregate_results.py
python 05b_plot_results.py

# 💡 アーキテクチャのポイント

## Triple Filtering Pipeline

Query
  ↓ embedding (hotchpotch)
Triple Retrieval (FAISS, top-20)
  ↓ LLM filter (Qwen2.5)
Relevant Triples (top-10)
  ↓ aggregate by volume/chapter
Enhanced Volume/Chapter Scores
  ↓ HippoRAG2Retriever
Final Chunks + Embedding Search

## スコア融合式

Volume: 0.6·embedding + 0.4·keyword + 0.2·triple
Chapter: embedding + 0.3·triple

## 期待される効果

Volume 誤選択：triple による補正で更に減少
Chapter 精度：embedding 単独より安定
スコア向上：v0.2.1 (avg 2.45) → v0.3 (avg 2.55+)

## 📋 前提条件

✅ Ollama 起動: ollama serve
✅ 推論モデル: swallow8b-lora-n4000-v09-q4
✅ Judge/Filter モデル: qwen2.5:14b
✅ 既存インデックス: chunks.jsonl, embeddings.npy, hierarchy.json


### 📚 参考情報

詳細ドキュメント: README_v03.md
v0.2.1 との比較: readme 内の「🔍 技術的分析」セクション
トラブルシューティング: readme 内の「🐛」セクション
実装完了日: 2026-06-22
