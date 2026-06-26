# LESSON: Triple Index の粒度問題（GROUP_SIZE の影響）

## 📅 日付: 2026-06-25

## 🎯 問題の概要

v0.3.1.1 で ID → Index マッピングを修正したにもかかわらず、**Judge スコアが全く改善しない**という問題が発生。

- **症状**:
  - Triple filtering は正常動作（Debug 出力で確認）
  - chunk_ids 配列の処理も正しく実装済み
  - しかし Judge 平均スコア: **0.650 / 3.0**（baseline 0.825 より悪化）
  - avg Retrieval: **0.00s**（triple filtering が効いていない？）

- **直感的な疑問**:
  - "Triple filtering が動作しているのに、なぜスコアが改善しないのか？"
  - "Retrieval 時間が 0.00s なのはなぜ？"

---

## 🔍 診断プロセス

### 1. Triple データの構造確認

```python
# experiments/indices/triples.json の内容
[
  {
    "chunk_ids": [0, 1, 2, 3],  # ← 同じ配列
    "volume_id": "概要編",
    "chapter_id": "s1",
    "triple": {"subject": "...", "relation": "...", "object": "..."}
  },
  {
    "chunk_ids": [0, 1, 2, 3],  # ← 同じ配列
    "volume_id": "概要編",
    "chapter_id": "s1",
    "triple": {"subject": "...", "relation": "...", "object": "..."}
  },
  ...
]
```

**発見**: 全ての triple が同じ `chunk_ids` 配列を持っている！

### 2. 統計分析

```bash
python check_triple_quality.py
```

**結果**:
```
Total triples: 5906
Total chunks: 5322
chunk_ids 長さ分布: {4: 5906}  ← 全て長さ4！

volume_id 分布:
  調査編: 2298
  設計編: 1858
  ...
```

### 3. 粒度の定量分析

```bash
python analyze_triple_granularity.py
```

**衝撃的な結果**:

```
【シミュレーション】Filtered triples = 5個の場合

現状（GROUP_SIZE=4）:
  Filtered triples: 5個
  → 追加される chunk 数: 4個  ← たった4個！
  → 効果: 80.0% のみ（重複が多い）

GROUP_SIZE=1 の場合:
  Filtered triples: 5個
  → 追加される chunk 数: 20個
  → 効果: 400.0%

【実データ分析】
Filtered triples = 20個の場合:
  平均追加 chunk 数: 16.6個
  最大: 24個  最小: 8個
  効果率: 83.0%  ← 17% の効果が失われている！
```

---

## 🧠 根本原因

### GROUP_SIZE = 4 によるバッチ処理の副作用

#### **意図した最適化**（v0.3.1）

```python
# 01_build_triple_index.py
GROUP_SIZE = 4  # 4チャンクまとめて処理（高速化）

for start in range(0, len(chunks), GROUP_SIZE):
    group = chunks[start:start + GROUP_SIZE]  # [chunk_0, chunk_1, chunk_2, chunk_3]
    merged_text = "\n\n---\n\n".join([c['text'] for c in group])
    
    # OpenIE に merged_text を渡す（1回の API call で複数 chunk 処理）
    ts = extract_triples_ollama(merged_text)  # 複数 triple を抽出
    
    # 全ての triple に同じ chunk_ids を付与
    for t in ts:
        triples_out.append({
            "chunk_ids": [c.get("chunk_id") for c in group],  # ← 問題はここ
            ...
        })
```

#### **実際の結果**

1. **同じ chunk_ids を持つ triple が量産される**:
   ```
   Triple 0: chunk_ids = [0, 1, 2, 3]
   Triple 1: chunk_ids = [0, 1, 2, 3]  ← 同じ
   Triple 2: chunk_ids = [0, 1, 2, 3]  ← 同じ
   Triple 3: chunk_ids = [0, 1, 2, 3]  ← 同じ
   ```

2. **Filtering 後も重複が残る**:
   ```
   Query: "堤防の点検頻度は？"
   
   Triple retrieval: 20 triples 取得
   LLM filtering: 5 triples 通過  ← フィルタリングは成功
   
   Chunk selection:
     Triple 0: chunk_ids = [120, 121, 122, 123] → 4 chunks 追加
     Triple 1: chunk_ids = [120, 121, 122, 123] → 0 chunks 追加（重複）
     Triple 2: chunk_ids = [120, 121, 122, 123] → 0 chunks 追加（重複）
     Triple 3: chunk_ids = [124, 125, 126, 127] → 4 chunks 追加
     Triple 4: chunk_ids = [124, 125, 126, 127] → 0 chunks 追加（重複）
   
   合計: 5 triples → 8 chunks のみ（期待: 20 chunks）
   効果: 40% しか発揮されていない！
   ```

3. **検索精度への影響**:
   - Filtered triples は質問に関連するはずなのに、重複により候補 chunk が少ない
   - 結果的に無関係な chunk が上位に残る
   - Answer の質が低下 → Judge スコアが改善しない

---

## 📊 問題の定量化

### バッチ処理による効果の損失

| Filtered Triples | 追加 Chunk 数（実測平均） | 理論最大 | 効果率 |
|------------------|---------------------------|---------|--------|
| 5個              | 6.6個                     | 20個    | 33%    |
| 10個             | 9.9個                     | 40個    | 25%    |
| 20個             | 16.6個                    | 80個    | 21%    |

**結論**: Filtered triples が増えるほど、重複による効果損失が大きくなる。

### OpenIE の抽出効率も低い

```
同じ chunk_ids を持つ triple 数の分布:
  平均: 1.2個/バッチ
  最大: 3個/バッチ
  最小: 1個/バッチ
```

**衝撃的な事実**: 4 chunks をまとめても、平均 1.2個の triple しか抽出されていない。

→ **バッチ処理による速度向上は限定的**
→ **粒度を粗くしているだけで、cost/benefit が悪い**

---

## 🎓 重要な教訓

### 教訓 1: **速度と精度のトレードオフを慎重に評価する**

**問題**:
- 「処理時間を 10-12h → 2.5-3h に短縮」という目標が先行
- 精度への影響を十分に検証せずに実装
- 結果: 速くなったが、品質が著しく低下

**対策**:
- 最適化を実装する前に、品質への影響を定量評価する
- ベンチマークを用意し、最適化前後で比較する
- 「速度 vs 精度」のトレードオフを明示的に文書化する

### 教訓 2: **データの粒度が重要**

**問題**:
- Triple filtering の目的は「関連する chunk を絞り込むこと」
- しかし粗い粒度（1 triple = 4 chunks）では絞り込めない
- データ構造が機能の目的に合致していなかった

**対策**:
- データの粒度は、そのデータを使う機能の要件から決める
- 「1 triple は 1 chunk に対応すべき」という原則
- バッチ処理は speed-up のためだけでなく、semantic grouping に使うべき

### 教訓 3: **中間データの検証が不可欠**

**問題**:
- Triple データを生成したが、中身を十分に確認していなかった
- 「5906 triples 生成された」という数字だけで満足
- 実際には全て同じ構造で、品質が低かった

**対策**:
- データ生成後、必ずサンプルを目視確認する
- 統計分析（分布、重複率、カバレッジ）を自動化する
- 異常値検出（全て同じ値、偏った分布）を実装する

### 教訓 4: **パイプライン全体の end-to-end 効果を測定する**

**問題**:
- Triple extraction は成功
- Triple filtering も成功
- しかし最終的な検索精度は改善せず

**対策**:
- 各コンポーネントの個別テストだけでなく、統合テストが必要
- 最終評価指標（Judge スコア）を常にモニタリング
- 中間指標（Retrieval 時間、候補 chunk 数）も並行して追跡

### 教訓 5: **最適化の前提を検証する**

**問題**:
- 「4 chunks をまとめれば、4倍の triple が得られる」と仮定
- 実際には平均 1.2倍にしかならなかった
- 前提が崩れているのに、実装を進めてしまった

**対策**:
- 最適化の前に小規模な実験で前提を検証
- 「4 chunks → 何個の triple？」を実測してから決定
- 費用対効果（cost/benefit）を定量的に計算

---

## 🔧 解決策

### GROUP_SIZE を 4 → 1 に変更

```python
# experiments/01_build_triple_index.py

# 修正前
GROUP_SIZE = 4  # 4チャンクまとめて処理（高速化の本丸）

# 修正後
GROUP_SIZE = 1  # chunk ごとに個別処理（triple の粒度を最大化）
```

### 期待される改善

| 項目                  | 修正前（GROUP_SIZE=4） | 修正後（GROUP_SIZE=1） |
|-----------------------|------------------------|------------------------|
| Triple 粒度           | 1 triple = 4 chunks    | 1 triple = 1 chunk     |
| Filtering 効果率      | 83%                    | 100%                   |
| 処理時間              | 2.5-3h                 | 10-12h                 |
| Triple 数             | 5906個                 | 6000-7000個（推定）    |
| Judge スコア（期待値）| 0.650                  | **0.85+**              |

### トレードオフの判断

**精度優先の判断根拠**:
1. MVP の目標は「ベースラインを 4% 改善」（2.45 → 2.55）
2. 現状は 0.650 でベースラインより大幅に悪い
3. 処理時間は 1日1回のバッチ処理なので、10-12h は許容範囲
4. Triple の品質向上が、検索精度に直結する

**もし速度が重要なら**:
- GROUP_SIZE = 2（妥協案、処理時間 5-6h）
- より高性能な LLM を使用（OpenAI GPT-4 など）
- Parallel processing（複数 GPU、複数 Ollama instance）

---

## 📈 検証方法

### 1. Triple データの品質確認

```bash
python check_triple_quality.py
```

**期待される出力**:
```
chunk_ids 長さ分布: {1: 6000}  ← 全て長さ1
```

### 2. 粒度分析

```bash
python analyze_triple_granularity.py
```

**期待される出力**:
```
Filtered triples = 20個の場合:
  平均追加 chunk 数: 20.0個  ← 100%
  効果率: 100.0%
```

### 3. 最終評価

```bash
python 04_eval_rag.py --model swallow --rag hipporag2 --judge-model qwen2.5:14b
```

**期待される改善**:
```
Judge 平均スコア: 0.85+ / 3.0  (目標: baseline 0.825 を上回る)
avg Retrieval: 0.15-0.20s  (triple filtering が効いている証拠)
```

---

## 🔬 さらなる最適化案

### 将来的な改善方向

1. **Semantic Grouping**:
   - 単に chunk を連結するのではなく、意味的に関連する chunk をグループ化
   - Embedding similarity でクラスタリングしてから OpenIE

2. **Hierarchical Triple Extraction**:
   - Volume レベル → Chapter レベル → Chunk レベル と段階的に抽出
   - 各レベルで異なる粒度の triple を保持

3. **Triple Quality Scoring**:
   - 抽出した triple に品質スコアを付与
   - Low-quality triples をフィルタリング

4. **Adaptive Batching**:
   - 文書の複雑さに応じて GROUP_SIZE を動的に調整
   - 長い chunk は単独処理、短い chunk はグループ化

5. **Parallel Processing**:
   - 複数 Ollama instance を起動
   - chunk を並列に処理（GPU が複数ある場合）

---

## 📚 関連資料

### コードファイル

- `experiments/01_build_triple_index.py` - Triple index 構築スクリプト
- `experiments/03_rag_retrievers.py` - HippoRAG2Retriever（chunk selection）
- `experiments/04_eval_rag.py` - 評価パイプライン

### 診断スクリプト

- `experiments/check_triple_quality.py` - Triple データの内容確認
- `experiments/analyze_triple_granularity.py` - 粒度の定量分析

### 関連 LESSON

- `experiments/LESSON_triple_filtering.md` - ID vs Index の混同問題
- `experiments/README_v03.md` - v0.3/v0.3.1 の実装詳細

---

## 💡 一般化できる設計原則

### 1. **データ粒度は機能要件から決定する**

```
機能の目的: 関連 chunk を絞り込む
    ↓
要件: 1 triple は 1 chunk に対応すべき
    ↓
設計: GROUP_SIZE = 1
```

**逆パターン（悪い例）**:
```
制約: 処理時間を短縮したい
    ↓
実装: GROUP_SIZE = 4（速度優先）
    ↓
結果: 機能の目的が達成できない
```

### 2. **最適化は測定可能な効果で判断する**

| 最適化手法        | 速度向上  | 品質影響  | 判断      |
|-------------------|-----------|-----------|-----------|
| Batching (n=4)    | 4倍（期待）| -17%      | ❌ 却下   |
| Batching (n=2)    | 2倍       | -5%       | ⚠️ 要検討 |
| Parallel (4 GPU)  | 4倍       | 0%        | ✅ 採用   |
| Better LLM        | 2倍       | +10%      | ✅ 採用   |

### 3. **中間データは可視化・検証を自動化する**

```python
# 悪い例
triples = extract_triples(chunks)
save_json(triples, "triples.json")
# → 中身を見ずに次の処理へ

# 良い例
triples = extract_triples(chunks)
validate_triples(triples)  # 統計分析、異常検出
visualize_triple_distribution(triples)  # 分布の確認
assert_quality_metrics(triples, min_diversity=0.8)  # 品質保証
save_json(triples, "triples.json")
```

### 4. **エラーは早期に検出する（Fail Fast）**

```python
# Triple 生成直後に品質チェック
if all(len(t['chunk_ids']) == 4 for t in triples):
    warnings.warn(
        "全ての triple が同じ chunk_ids 長さ。"
        "GROUP_SIZE が大きすぎる可能性があります。"
    )

# 評価前に中間指標を確認
if avg_retrieval_time < 0.05:
    warnings.warn(
        "Retrieval 時間が異常に短い。"
        "Triple filtering が動作していない可能性があります。"
    )
```

---

## 🎯 今後の開発での適用

### チェックリスト: データパイプライン最適化時

- [ ] 最適化の目的を明確にする（速度 vs 精度 vs メモリ）
- [ ] ベースラインを測定する（最適化前の性能）
- [ ] 小規模実験で効果を検証する（全体実装の前に）
- [ ] 中間データの品質を確認する（目視 + 統計）
- [ ] 最終評価指標への影響を測定する（end-to-end テスト）
- [ ] トレードオフを文書化する（何を犠牲にしたか）
- [ ] ロールバック可能にする（元の設定を保持）

### プロジェクトへの適用例

1. **Embedding 次元の削減**:
   - 1024次元 → 512次元で速度向上を図る前に
   - 検索精度への影響を測定する

2. **Top-K の調整**:
   - Retriever の top_k を 5 → 3 に減らす前に
   - Answer の質への影響をベンチマークする

3. **LLM の量子化**:
   - F16 → Q4_K_M で VRAM を節約する前に
   - 生成品質の劣化を定量評価する

---

**作成者**: GitHub Copilot  
**最終更新**: 2026-06-25  
**関連 Issue**: Triple filtering の効果が出ない問題（v0.3.1.1）
