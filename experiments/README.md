# RAG 比較実験 — 河川砂防技術標準

Naive RAG / Light RAG / HippoRAG2（階層 coarse-to-fine） の 3方式 ×
Swallow-8B-Q4 / ELYZA-JP-8B-Q4 の 2モデル = **6条件** で性能を比較する。

## 実行順序

```
cd i:\ACT2025.5.26-2030\MVP\kasensabo_hipporag2

# 0. 環境確認（Ollama モデル / GPU / ライブラリ）
python experiments/00_check_env.py

# 1. インデックス構築
python experiments/01_build_indices.py        # FAISS + BM25 + 階層メタデータ
python experiments/01b_build_hipporag2_index.py   # HippoRAG2 用ボリューム・章ベクトル

# 2. テストセット準備
python experiments/02_prepare_testset.py      # 200問サンプリング (seed=42)

# 3. 検索モジュール動作確認（任意）
python experiments/03_rag_retrievers.py --test

# 4. 評価（全6条件）
pwsh experiments/04b_run_all.ps1              # 全条件一括
# または個別実行:
python experiments/04_eval_rag.py --model swallow --rag naive
python experiments/04_eval_rag.py --model swallow --rag light
python experiments/04_eval_rag.py --model swallow --rag hipporag2
python experiments/04_eval_rag.py --model elyza   --rag naive
python experiments/04_eval_rag.py --model elyza   --rag light
python experiments/04_eval_rag.py --model elyza   --rag hipporag2

# 5. 集計・可視化
python experiments/05_aggregate_results.py    # summary.csv / summary.json
python experiments/05b_plot_results.py        # グラフ生成 → results/figures/
```

### dry-run（動作確認用 10問）

```
python experiments/04_eval_rag.py --model swallow --rag naive --dry-run
```

### Judge をスキップして推論のみ

```
python experiments/04_eval_rag.py --model swallow --rag naive --no-judge
```

## ファイル構成

```
experiments/
├── 00_check_env.py           # 環境確認
├── 01_build_indices.py       # FAISS + BM25 + 階層メタデータ構築
├── 01b_build_hipporag2_index.py  # HippoRAG2 階層ベクトル構築
├── 02_prepare_testset.py     # テストセット準備（200問）
├── 03_rag_retrievers.py      # 3方式 RAG 検索クラス
├── 04_eval_rag.py            # 1条件評価パイプライン
├── 04b_run_all.ps1           # 全6条件一括実行
├── 05_aggregate_results.py   # 結果集計
├── 05b_plot_results.py       # 可視化
├── env_config.json           # 00_check_env.py が生成（Ollama モデル名）
├── testset_200.jsonl         # 02_prepare_testset.py が生成
├── indices/
│   ├── chunks.jsonl          # チャンクデータ
│   ├── embeddings.npy        # チャンク埋め込みベクトル
│   ├── faiss.index           # FAISS IndexFlatIP
│   ├── bm25.pkl              # BM25 インデックス
│   ├── hierarchy.json        # 階層メタデータ (Volume→Chapter→Chunk)
│   ├── hipporag2_volumes.json   # ボリューム代表ベクトル
│   └── hipporag2_chapters.json  # 章代表ベクトル
└── results/
    ├── swallow_naive_results.jsonl
    ├── swallow_naive_summary.json
    ├── ...                   # 各条件の結果
    ├── summary.csv           # 集計表
    ├── summary.json
    └── figures/
        ├── bar_judge_score.png
        ├── bar_perfect_rate.png
        ├── latency_comparison.png
        └── score_distribution.png
```

## RAG 方式の設計

### Naive RAG
全チャンクを 1つのベクトル空間で埋め込み、クエリとの内積類似度で top-k を返す。
ベースライン。

### Light RAG
BM25（キーワードマッチ）スコアと embedding スコアを α=0.5 で融合。
技術用語が明確な文書と相性が良い。

### HippoRAG2（本命）
河川砂防技術基準の「巻→章→節・項」階層を踏んだ coarse-to-fine 検索:

1. **Level 1 (coarse)**: クエリ → ボリューム代表ベクトルとの類似度 → 上位 2 巻を選択
2. **Level 2 (mid)**: 選択巻内の章代表ベクトルとの類似度 → 上位 3 章を選択
3. **Level 3 (fine)**: 選択章のチャンク内で embedding 検索 → top-k チャンク

KG/グラフ構造は使わず、「階層メタデータ＋ベクトル検索」のみで実装。
16GB GPU 制約内でも動作する。

## 評価指標

| 指標 | 説明 |
|---|---|
| Judge 平均スコア | 0–3点ルーブリック（技術的正確性＋標準の引用）の平均 |
| 3点率 (Perfect-Score Rate) | Judge スコア 3点の割合 |
| Retrieval Time | チャンク取得の平均時間 (秒/問) |
| Generation Time | Ollama 生成の平均時間 (秒/問) |
| Score 分布 | 0/1/2/3点の件数分布 |

### Judge ルーブリック (Qwen2.5-7B-Instruct)

- **3点**: 技術的に正確で具体的、根拠となる基準名・章番号・技術概念が含まれる
- **2点**: 概ね正確だが、根拠・具体性がやや不足
- **1点**: 部分的に正しいが、重要な誤り・不足がある
- **0点**: 回答なし、または技術的に大きく誤っている

## 前提条件

- Python 3.10+
- Ollama (http://localhost:11434) に以下のモデルがダウンロード済み:
  - `swallow8b-lora-n4000-v09-q4` または `swallow8b-lora-n4000-v09`
  - `elyza8b-lora-n4000-q4` または `elyza8b-lora-n4000`
  - `qwen2.5:7b` または `qwen2.5:14b`（Judge 用）
- GPU 16GB VRAM 推奨（CPU でも動作するが生成が遅い）
- 依存ライブラリ: `pip install rank-bm25 faiss-cpu sentence-transformers httpx tqdm matplotlib pandas`
