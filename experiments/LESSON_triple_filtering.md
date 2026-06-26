# LESSON: Triple Filtering 実装の教訓（v0.3.1.1）

## 📅 日付: 2026-06-24

## 🎯 問題の概要

Triple filtering 機能を実装したが、**フィルタリングは実行されているのにスコアが改善しない**という問題が発生。

- **症状**: 
  - Triple filtering の実行時間は正常（0.19s）
  - Debug 出力で "X triples → Retriever に渡す" を確認
  - しかし評価スコアは 0.820（baseline 0.825 より悪化）

- **根本原因**: 
  - `chunk_id` と `chunk index` を混同していた
  - Filtered triples の `chunk_ids` を候補セットに追加していたが、実際には配列のインデックスが必要だった

---

## 🔍 詳細な原因分析

### 1. データ構造の理解不足

#### **Triple データの構造（01_build_triple_index.py）**
```python
triples_out.append({
    "chunk_ids": [0, 1, 2, 3],  # ← これは chunk の ID（metadata）
    "passage_ids": [0, 1, 2, 3],
    "volume_id": "VOLUME_1",
    "chapter_id": "CHAPTER_1_2",
    "triple": {"subject": "...", "relation": "...", "object": "..."}
})
```

#### **Chunks データの構造（chunks.jsonl）**
```json
{
  "chunk_id": 0,           // ← メタデータの ID
  "text": "...",
  "doc_id": "VOLUME_1",
  "section_id": "CHAPTER_1_2"
}
```

#### **Retriever の内部構造（03_rag_retrievers.py）**
```python
self._indices["chunks"] = [
    {"chunk_id": 0, "text": "...", ...},  # index 0
    {"chunk_id": 1, "text": "...", ...},  # index 1
    {"chunk_id": 2, "text": "...", ...},  # index 2
    ...
]
```

### 2. 誤った実装（v0.3.1）

```python
# ❌ 間違い: chunk_id を直接 candidate_ids に追加
if self._filtered_triples:
    for t_dict, _ in self._filtered_triples:
        if "chunk_ids" in t_dict:
            for cid in t_dict["chunk_ids"]:
                candidate_ids.add(cid)  # ← chunk_id をそのまま使用
```

**問題点**:
- `candidate_ids` は `self._indices["chunks"]` の**配列インデックス**を格納するセット
- しかし `chunk_ids` は chunk の **ID（メタデータ）** であり、配列インデックスとは異なる
- たまたま ID と index が一致する場合は動作するが、データの並び順が変わると破綻する

### 3. 正しい実装（v0.3.1.1）

```python
# ✅ 正解: chunk_id → index のマッピングを構築
def _init(self):
    # ... 既存のコード ...
    
    # chunk_id → index マッピングの構築
    if self._chunk_id_to_idx is None:
        self._chunk_id_to_idx = {}
        for idx, chunk in enumerate(self._indices["chunks"]):
            chunk_id = chunk.get("chunk_id")
            if chunk_id is not None:
                self._chunk_id_to_idx[chunk_id] = idx

# Level 3: Chunk selection
if self._filtered_triples and self._chunk_id_to_idx:
    for t_dict, _ in self._filtered_triples:
        if "chunk_ids" in t_dict:
            for cid in t_dict["chunk_ids"]:
                idx = self._chunk_id_to_idx.get(cid)  # ← ID を index に変換
                if idx is not None:
                    candidate_ids.add(idx)  # ← 正しい index を追加
```

---

## 🧠 重要な教訓

### 教訓 1: **ID と Index は別物**

**問題**:
- プログラミングでは「ID（識別子）」と「Index（配列の位置）」は異なる概念
- Python のリストアクセス `chunks[idx]` は **index** を使う
- データベースや JSON の `chunk_id` は **ID**（メタデータ）

**対策**:
- データ構造を設計する際は、ID と Index を明確に区別する
- 必要に応じて ID → Index のマッピング辞書を構築する
- 変数名を明確にする（`chunk_id` vs `chunk_idx`）

### 教訓 2: **「動いている」≠「正しく動いている」**

**問題**:
- Triple filtering は実行されていた（0.19s の処理時間）
- Debug 出力も正常に表示されていた
- しかし**結果が実際の検索に反映されていなかった**

**対策**:
- 機能が「実行される」ことと「効果がある」ことは別
- 中間結果を可視化する（例: 選択された chunk の数を出力）
- ベースラインとの比較で効果を定量的に測定する

### 教訓 3: **バッチ処理の副作用を追跡する**

**問題**:
- v0.3.1 で 4-chunk バッチ処理を導入
- `chunk_id` → `chunk_ids` (配列) に変更
- しかし retriever 側の対応が不完全だった

**対策**:
- データフォーマット変更時は、依存する全コードを追跡する
- 単数形 → 複数形の変更は特に注意（`chunk_id` vs `chunk_ids`）
- 後方互換性を維持する（両方のフォーマットをサポート）

### 教訓 4: **デバッグは段階的に**

**成功した診断プロセス**:
1. Triple filtering が実行されているか確認（処理時間で判定）
2. LLM の出力を確認（batch filtering の結果）
3. Retriever に渡されるデータを確認（debug 出力）
4. Chunk selection の実装を精査（ここで問題発見）
5. 根本原因を特定（ID vs Index の混同）

**対策**:
- 各段階でログを出力し、データの流れを追跡する
- 「どこまで正しいか」を特定してから次に進む
- 複数の箇所を同時に疑わない

---

## 📊 修正による改善

### 修正前（v0.3.1）
```
avg Retrieval: 0.19s  ← Triple filtering は実行されている
Judge 平均スコア: 0.820 / 3.0  ← でもスコアは悪化
Score dist: 0点:65  1点:112  2点:17  3点:6
```

### 修正後（v0.3.1.1）- 期待値
```
avg Retrieval: 0.19s  ← 同じ
Judge 平均スコア: 2.55 / 3.0  ← 目標値（chunk selection が正しく動作）
```

---

## 🔧 技術的な詳細

### マッピング辞書の構築コスト

```python
# O(N) の前処理で、O(1) のルックアップを実現
self._chunk_id_to_idx = {}
for idx, chunk in enumerate(self._indices["chunks"]):
    chunk_id = chunk.get("chunk_id")
    if chunk_id is not None:
        self._chunk_id_to_idx[chunk_id] = idx
```

- **時間計算量**: O(N) where N = chunks 数（5322）
- **空間計算量**: O(N) for 辞書
- **ルックアップ**: O(1) per chunk_id
- **実行時間**: 初期化時のみ、数ミリ秒

### 後方互換性の維持

```python
# chunk_ids (配列) と chunk_id (単数) の両方をサポート
if "chunk_ids" in t_dict:
    # v0.3.1 以降のバッチ処理形式
    for cid in t_dict["chunk_ids"]:
        idx = self._chunk_id_to_idx.get(cid)
        if idx is not None:
            candidate_ids.add(idx)
elif "chunk_id" in t_dict:
    # v0.3 以前の単数形式
    chunk_id = t_dict["chunk_id"]
    idx = self._chunk_id_to_idx.get(chunk_id)
    if idx is not None:
        candidate_ids.add(idx)
```

---

## 🎓 一般化できる原則

### 1. **データの流れを可視化する**

```
Triple Index → Triple Retrieval → LLM Filtering → Chunk Selection → Final Ranking
    ↓              ↓                  ↓                ↓                ↓
 chunk_ids    (chunk_id, score)  filtered list   candidate_ids    top_k chunks
   (配列)         (ID)              (ID)          (index)          (index)
```

### 2. **型の変換を明示的に**

```python
# ❌ 暗黙的な変換（危険）
candidate_ids.add(chunk_id)  # ID? Index? 不明

# ✅ 明示的な変換（安全）
chunk_idx = self._chunk_id_to_idx.get(chunk_id)
if chunk_idx is not None:
    candidate_ids.add(chunk_idx)  # Index であることが明確
```

### 3. **境界条件をテストする**

- Empty list: `chunk_ids = []`
- Missing key: `chunk_id not in mapping`
- Null values: `chunk_id = None`
- Type mismatch: `chunk_id = "0"` vs `0`

---

## 📝 コードレビューのチェックリスト

今後の実装で確認すべき項目:

- [ ] ID と Index を混同していないか？
- [ ] データ構造の変更が依存コード全体に反映されているか？
- [ ] 単数形 ↔ 複数形の変換が正しいか？
- [ ] マッピング辞書が必要な場合、構築されているか？
- [ ] Debug 出力で中間結果を確認できるか？
- [ ] ベースラインとの比較で効果を測定できるか？
- [ ] 後方互換性が必要な場合、維持されているか？
- [ ] None や空配列などの境界条件を処理しているか？

---

## 🚀 今後の改善案

### 1. 型ヒントの強化

```python
from typing import NewType

ChunkID = NewType('ChunkID', int)
ChunkIndex = NewType('ChunkIndex', int)

def get_chunk_index(chunk_id: ChunkID) -> ChunkIndex | None:
    """chunk_id を chunk_index に変換"""
    return self._chunk_id_to_idx.get(chunk_id)
```

### 2. アサーションの追加

```python
# 開発時のみ有効化
if __debug__:
    for idx in candidate_ids:
        assert 0 <= idx < len(self._indices["chunks"]), \
            f"Invalid chunk index: {idx}"
```

### 3. 詳細なロギング

```python
import logging

logger.debug(f"Filtered triples: {len(self._filtered_triples)}")
logger.debug(f"Candidate chunks before filtering: {len(candidate_ids)}")
# ... chunk selection ...
logger.debug(f"Candidate chunks after filtering: {len(candidate_ids)}")
```

---

## 📚 参考資料

- **関連コード**:
  - `experiments/01_build_triple_index.py` - Triple データの生成
  - `experiments/03_rag_retrievers.py` - HippoRAG2Retriever の実装
  - `experiments/04_eval_rag.py` - 評価パイプライン

- **関連ドキュメント**:
  - `experiments/README_v03.md` - v0.3/v0.3.1 の詳細

---

# LESSON: Triple Filtering 実装の教訓（v0.3.1.5）

## 📅 日付: 2026-06-26

## 🎯 問題の概要

Triple filtering の実装は完了していたが、**実際には全く動作していなかった**という問題が発覚。

- **症状**: 
  - `swallow_hipporag2` のスコアが 0.665（期待値 2.5 より大幅に低い）
  - `retrieval_time` が 0.01s（triple filtering なしと同じ）
  - コードレビューで `_filtered_triples` が常に `None` のまま

- **根本原因**: 
  - `_filtered_triples` を設定するコードが存在しなかった
  - Triple index はロードされていたが、クエリごとの検索処理が実装されていなかった
  - 静的な初期化処理と動的なクエリ処理を混同していた

---

## 🔍 詳細な原因分析

### 1. 設計上の誤解

#### **誤った想定（v0.3〜v0.3.1.1）**
```python
class HippoRAG2Retriever:
    def __init__(self, ...):
        self._filtered_triples: list | None = None  # v0.3: triple filtering 対応
        # ↑「初期化時に一度だけ設定される」と想定していた
```

**問題点**:
- Triple filtering は**クエリに依存する**動的な処理
- しかし実装では「初期化時の静的データ」として扱っていた
- 「対応」という曖昧な表現が誤解を招いた

#### **実際の動作フロー**
```python
def retrieve(self, query: str):
    # Level 1: ボリューム選択
    if self._filtered_triples:  # ← 常に None なので False
        triple_vol_scores, _ = self._aggregate_triple_scores(self._filtered_triples)
    else:
        triple_vol_scores = {}  # ← 常にこちらが実行される
```

### 2. 実装の欠落

#### **欠けていた処理: Triple Indexからの検索**

```python
# ❌ v0.3.1.1 まで: Triple index はロードされるが使われない
def _init(self):
    # Triple index の存在は確認していた（04_eval_rag.py）
    # しかし HippoRAG2Retriever 内では使用していなかった
    pass

def retrieve(self, query: str):
    # _filtered_triples を設定する処理が無い！
    # → 常に None のまま
    pass
```

#### **正しい実装: クエリごとの Triple 検索**

```python
# ✅ v0.3.1.5: クエリごとに triple index から検索
def _init(self):
    # Triple index & metadata のロード
    if self._triple_index is None and self._triple_data is None:
        try:
            import faiss
            triple_json_path = IDX_DIR / "triples.json"
            triple_emb_path = IDX_DIR / "triple_embs.npy"
            
            if triple_json_path.exists() and triple_emb_path.exists():
                # Metadata
                with open(triple_json_path, encoding="utf-8") as f:
                    self._triple_data = json.load(f)
                
                # FAISS index
                triple_embs = np.load(triple_emb_path)
                self._triple_index = faiss.IndexFlatIP(triple_embs.shape[1])
                self._triple_index.add(triple_embs)
        except Exception:
            self._triple_data = None
            self._triple_index = None

def retrieve(self, query: str):
    # ── v0.3.1.5: Triple Filtering（クエリごとに実行）──
    if self._triple_index is not None and self._triple_data is not None:
        # クエリに関連する top-N triples を取得
        n_retrieve = min(self.n_triples, len(self._triple_data))
        D, I = self._triple_index.search(q_vec, n_retrieve)
        
        # filtered_triples = [(triple_dict, similarity_score), ...]
        self._filtered_triples = []
        for idx, score in zip(I[0], D[0]):
            if 0 <= idx < len(self._triple_data):
                self._filtered_triples.append((self._triple_data[idx], float(score)))
    else:
        # Triple index がない場合は空リスト
        self._filtered_triples = []
    
    # 以降、既存の _aggregate_triple_scores が正しく動作する
```

---

## 🧠 重要な教訓

### 教訓 1: **静的データと動的処理を区別する**

**問題**:
- Triple index（静的データ）はロードできていた
- しかし「クエリごとの検索」（動的処理）が実装されていなかった
- 「対応済み」という表現が、実装の有無を曖昧にした

**対策**:
```python
# ❌ 曖昧な命名
self._filtered_triples: list | None = None  # v0.3: triple filtering 対応

# ✅ 明確な命名と説明
self._filtered_triples: list | None = None  # v0.3: triple filtering 対応（クエリごとに更新）
self._triple_index: faiss.Index | None = None  # v0.3: triple FAISS index（初期化時にロード）
self._triple_data: list | None = None  # v0.3: triple metadata（初期化時にロード）
```

**設計原則**:
1. **初期化時にロードするデータ**: `_triple_index`, `_triple_data`
2. **クエリごとに更新するデータ**: `_filtered_triples`
3. 両者を明確に区別する

### 教訓 2: **フィールドの命名は処理フローを反映する**

**問題**:
- `_filtered_triples` という名前から「一度フィルタリングされた静的リスト」を連想
- 実際には「クエリごとに更新される動的リスト」

**対策**:
```python
# より明確な命名案
self._query_triples: list | None = None  # クエリに関連する triples
self._current_triples: list | None = None  # 現在のクエリの triples
```

ただし、既存コード（v0.3.1.1）との互換性のため `_filtered_triples` を維持。

### 教訓 3: **「動いていない」ことを検出する仕組み**

**問題の発見方法**:
1. スコアがベースラインより大幅に低い（0.665 vs 期待値 2.5）
2. `retrieval_time` が変化していない（0.01s）
3. コードレビューで `_filtered_triples` の設定処理が無いことを発見

**対策**:
```python
# デバッグ用の統計出力
def retrieve(self, query: str):
    # ... triple filtering ...
    
    if __debug__:
        n_triples = len(self._filtered_triples) if self._filtered_triples else 0
        print(f"[DEBUG] Query: {query[:50]}... | Triples: {n_triples}")
```

### 教訓 4: **依存関係の明示化**

**問題**:
- `_aggregate_triple_scores()` は `_filtered_triples` に依存
- しかし `_filtered_triples` を設定する処理が無かった
- 依存関係が暗黙的だったため、見落とされた

**対策**:
```python
def _aggregate_triple_scores(self, filtered_triples: list) -> tuple[dict, dict]:
    """
    v0.3: filtered_triples の volume/chapter スコアを集約
    
    Args:
        filtered_triples: [(triple_dict, sim_score), ...] の リスト
                         ※ retrieve() で self._filtered_triples として設定
    
    Returns:
        (vol_scores, chap_scores)
    """
    # filtered_triples は引数として渡されるが、
    # 実際には self._filtered_triples を期待している
```

**改善案**:
- 引数として渡すか、self のフィールドとして参照するか、統一する
- ドキュメントで前提条件を明記する

---

## 📊 修正による改善（予測）

### 修正前（v0.3.1.1）
```
swallow_hipporag2:
  avg_score: 0.665 / 3.0
  retrieval_time: 0.01s
  triple filtering: 実装されているが動作していない
```

### 修正後（v0.3.1.5）- 期待値
```
swallow_hipporag2:
  avg_score: 1.8-2.5 / 3.0  ← Triple filtering が効果を発揮
  retrieval_time: 0.3-0.5s  ← Triple 検索のオーバーヘッド
  triple filtering: 正しく動作（クエリごとに top-20 triples 取得）
```

---

## 🔧 実装の詳細

### パラメータの追加

```python
def __init__(self, ..., n_triples: int = 20):
    # n_triples: クエリごとに取得する triple 数
    self.n_triples = n_triples
```

```python
# 04_eval_rag.py
parser.add_argument("--n-triples", type=int, default=20, 
                    help="Triple filtering取得数（HippoRAG2用、デフォルト20）")
```

### Triple Index のロード（初期化時）

```python
def _init(self):
    # ... 既存の処理 ...
    
    # v0.3.1.5: triple index のロード
    if self._triple_index is None and self._triple_data is None:
        try:
            import faiss
            triple_json_path = IDX_DIR / "triples.json"
            triple_emb_path = IDX_DIR / "triple_embs.npy"
            
            if triple_json_path.exists() and triple_emb_path.exists():
                with open(triple_json_path, encoding="utf-8") as f:
                    self._triple_data = json.load(f)
                
                triple_embs = np.load(triple_emb_path)
                self._triple_index = faiss.IndexFlatIP(triple_embs.shape[1])
                self._triple_index.add(triple_embs)
        except Exception:
            # エラー時は triple filtering なしで動作
            self._triple_data = None
            self._triple_index = None
```

### Triple Filtering（クエリごと）

```python
def retrieve(self, query: str):
    self._init()
    q_vec = _encode(query)
    
    # ── v0.3.1.5: Triple Filtering（クエリごとに実行）──
    if self._triple_index is not None and self._triple_data is not None:
        n_retrieve = min(self.n_triples, len(self._triple_data))
        D, I = self._triple_index.search(q_vec, n_retrieve)
        
        self._filtered_triples = []
        for idx, score in zip(I[0], D[0]):
            if 0 <= idx < len(self._triple_data):
                self._filtered_triples.append((self._triple_data[idx], float(score)))
    else:
        self._filtered_triples = []
    
    # ── Level 1: ボリューム選択（既存のコードが正しく動作）──
    if self._filtered_triples:
        triple_vol_scores, _ = self._aggregate_triple_scores(self._filtered_triples)
    else:
        triple_vol_scores = {}
    # ...
```

---

## 🎓 一般化できる原則

### 1. **機能の実装段階を明確にする**

```
実装段階:
  [ ] Phase 0: 設計・仕様定義
  [ ] Phase 1: データ構造の定義（Triple index の schema）
  [ ] Phase 2: データの生成（01_build_triple_index.py）
  [ ] Phase 3: データのロード（_init() で triple_index/data）
  [ ] Phase 4: データの使用（retrieve() で triple filtering）  ← ここが欠けていた
  [ ] Phase 5: 効果の検証（評価スコアの改善）
```

### 2. **「対応」という曖昧な表現を避ける**

```python
# ❌ 曖昧な表現
self._filtered_triples: list | None = None  # v0.3: triple filtering 対応

# ✅ 具体的な説明
self._filtered_triples: list | None = None
"""
v0.3.1.5: クエリごとに更新される triple filtering 結果
- retrieve() メソッド内で self._triple_index.search() により設定
- _aggregate_triple_scores() で volume/chapter スコア計算に使用
- 初期値は None（triple index が無い場合は空リスト）
"""
```

### 3. **処理フローの可視化**

```
クエリ処理フロー（v0.3.1.5）:
  query
    ↓
  _encode(query) → q_vec
    ↓
  ┌──────────────────────────────┐
  │ Triple Filtering             │
  │  self._triple_index.search() │ ← v0.3.1.5 で追加
  │  → self._filtered_triples    │
  └──────────────────────────────┘
    ↓
  Level 1: Volume selection
    ├─ embedding score
    ├─ keyword score
    └─ triple score  ← _filtered_triples から計算
    ↓
  Level 2: Chapter selection
    └─ triple score  ← _filtered_triples から計算
    ↓
  Level 3: Chunk selection
    └─ filtered_triples の chunk_ids を追加  ← v0.3.1.1 で修正済み
    ↓
  Final ranking
```

---

## 📝 コードレビューのチェックリスト（更新版）

今後の実装で確認すべき項目:

- [ ] データ構造は定義されているか？
- [ ] データは生成されているか？（スクリプト実行済み）
- [ ] データはロードされているか？（初期化処理）
- [ ] **データは使用されているか？**（処理ロジック） ← v0.3.1.5 で追加
- [ ] 静的データと動的処理を区別しているか？
- [ ] フィールド名は処理フローを正しく反映しているか？
- [ ] 依存関係は明示的か？
- [ ] デバッグ用の統計出力はあるか？
- [ ] ベースラインとの比較で効果を測定できるか？

---

## 🚀 今後の改善案

### 1. 処理ステータスの可視化

```python
class HippoRAG2Retriever:
    @property
    def status(self) -> dict:
        """デバッグ用: Retriever の状態を返す"""
        return {
            "triple_index_loaded": self._triple_index is not None,
            "triple_data_count": len(self._triple_data) if self._triple_data else 0,
            "last_query_triples": len(self._filtered_triples) if self._filtered_triples else 0,
        }
```

### 2. 単体テストの追加

```python
def test_triple_filtering():
    retriever = HippoRAG2Retriever(n_triples=10)
    chunks = retriever.retrieve("ダム設計における許容応力度")
    
    # Triple filtering が実行されたか確認
    assert retriever._filtered_triples is not None
    assert len(retriever._filtered_triples) > 0
    assert len(retriever._filtered_triples) <= 10
```

### 3. パフォーマンスモニタリング

```python
import time

def retrieve(self, query: str):
    t0 = time.time()
    
    # Triple filtering
    t1 = time.time()
    # ... triple filtering ...
    t2 = time.time()
    
    # Level 1-3
    # ...
    
    t_end = time.time()
    
    self._perf_stats = {
        "triple_filtering": t2 - t1,
        "level1_volume": ...,
        "level2_chapter": ...,
        "level3_chunk": ...,
        "total": t_end - t0,
    }
```

---

## 📚 参考資料

- **関連コード**:
  - `experiments/03_rag_retrievers.py` - HippoRAG2Retriever（v0.3.1.5 で修正）
  - `experiments/04_eval_rag.py` - `--n-triples` オプション追加
  - `experiments/01_build_triple_index.py` - Triple データの生成

- **関連ドキュメント**:
  - `experiments/LESSON_triple_filtering.md` - v0.3.1.1 の教訓
  - `experiments/README_v03.md` - v0.3 シリーズの詳細
  - `experiments/LESSON_Unsloth_Install.md` - 環境構築の教訓

- **Issue Timeline**:
  - 2026-06-24 09:00 - Triple filtering 実装完了
  - 2026-06-24 10:30 - スコア改善せず（0.820）
  - 2026-06-24 11:00 - 原因特定（ID vs Index）
  - 2026-06-24 11:15 - v0.3.1.1 修正完了

---

**作成者**: GitHub Copilot  
**最終更新**: 2026-06-24

---

# LESSON: Triple Filtering 二重実装バグ（v0.3.1.5 修正版）

## 📅 日付: 2026-06-26

## 🎯 問題の概要

v0.3.1.5 実装後、dry-run（10問）では正常動作を確認したが、**200問全体の実行で最初の3問以降、triple filteringが動作しなくなる**という問題が発生。

- **症状**: 
  - 最初の3問: `[DEBUG] Triple filtering 開始/完了` ログ表示
  - 残り197問: ログ出力なし
  - `avg Retrieval: 0.02s`（期待値 0.3-0.5s より大幅に短い）
  - `avg Judge score: 0.745`（期待値 1.8-2.5 より大幅に低い）

- **誤った診断**: 
  - 当初、`debug_mode = rec_idx < 3` が原因と思われた（デバッグ出力が3問で止まるため）
  - しかし実際には**retrieval time が全問で 0.02s** = triple filtering 自体が動作していない

- **根本原因**: 
  - **二重実装**: `04_eval_rag.py` と `03_rag_retrievers.py` の両方で triple filtering を実行
  - **状態管理の不備**: `retriever._filtered_triples` が一度セットされると、次のクエリで上書きされる
  - **計測位置のミス**: triple filtering の実行時間が retrieval time に含まれていなかった

---

## 🔍 詳細な原因分析

### 1. 二重実装の問題

#### **実装の重複箇所**

**場所1: 04_eval_rag.py（評価スクリプト）**
```python
# LLM-based triple filtering（バッチ処理で高速化）
for rec_idx, rec in enumerate(records):
    question = rec["question"]
    
    # 1. FAISS search
    D, I = triple_index.search(q_vec, 20)
    top_triples = [(triple_data[int(i)], float(D[0][j])) for j, i in enumerate(I[0])]
    
    # 2. LLM filtering
    filtered_triples = filter_triples_batch(
        llm_fn=lambda x: _ollama_chat(...),
        query=question,
        triples=top_triples,
        batch_size=10,
    )
    
    # 3. retriever に渡す
    retriever._filtered_triples = filtered_triples  # ← ここでセット
    
    # 4. Retrieval 実行
    chunks = retriever.retrieve(question)
```

**場所2: 03_rag_retrievers.py（Retriever クラス）**
```python
def retrieve(self, query: str):
    q_vec = _encode(query)
    
    # v0.3.1.5: Triple Filtering（FAISS検索のみ、LLM判定なし）
    if self._triple_index is not None and self._triple_data is not None:
        n_retrieve = min(self.n_triples, len(self._triple_data))
        D, I = self._triple_index.search(q_vec, n_retrieve)
        
        self._filtered_triples = []  # ← 04_eval_rag.py でセットした値を上書き！
        for idx, score in zip(I[0], D[0]):
            self._filtered_triples.append((self._triple_data[idx], float(score)))
```

#### **問題点**

1. **04_eval_rag.py** が LLM filtering を実行して `retriever._filtered_triples` にセット
2. **retriever.retrieve()** 内で FAISS search のみの結果で上書き
3. **LLM filtering の結果が失われる**

### 2. 状態管理の不備

#### **問題のあったフロー**

```
Query 1:
  04_eval_rag.py: LLM filtering → retriever._filtered_triples = [7 triples]
  retrieve():     FAISS search → retriever._filtered_triples = [20 triples] (上書き)
  結果: LLM filtering 無効化

Query 2:
  04_eval_rag.py: retriever._filtered_triples = None をリセットしていない
  04_eval_rag.py: LLM filtering → retriever._filtered_triples = [2 triples]
  retrieve():     _filtered_triples != None なので FAISS search スキップ
                  → Query 1 の 20 triples がそのまま使われる！
  結果: Query 2 の filtering 結果が反映されない
```

#### **`debug_mode = rec_idx < 3` の誤解**

```python
debug_mode = rec_idx < 3  # 最初の3問のみデバッグ出力

filtered_triples = filter_triples_batch(
    ...,
    debug=debug_mode,  # rec_idx >= 3 では debug=False
)
```

- **現象**: 最初の3問のみ `[DEBUG]` ログが表示される
- **誤った解釈**: 「triple filtering が 3 問で止まる」
- **実際**: デバッグ出力が止まっただけで、**filtering 自体は全く動作していない**

### 3. 計測位置のミス

#### **問題のあったコード**

```python
# Triple filtering 実行（LLM 呼び出し含む）← 時間計測外
filtered_triples = filter_triples_batch(...)  # 10-15s かかる
retriever._filtered_triples = filtered_triples

# Retrieval 時間計測開始 ← ここから
t0 = time.time()
chunks = retriever.retrieve(question)  # 既に filtered されているのでスキップ
t_ret = time.time() - t0  # ← 0.002-0.006s になる
```

#### **結果**

- `retrieval_time = 0.02s`（triple filtering の時間が含まれない）
- 実際には LLM filtering が動作していないため、**偶然にも正確な値**
- しかし問題の発見を遅らせる原因となった

---

## 🧠 重要な教訓

### 教訓 1: **処理の責任範囲を明確にする**

**問題**:
- `04_eval_rag.py` が LLM filtering を実行
- `03_rag_retrievers.py` も FAISS filtering を実行
- 両者の責任範囲が不明確

**対策**:

**Option A: Retriever に全て任せる**
```python
# 04_eval_rag.py: retriever に全て任せる
chunks = retriever.retrieve(question)  # retriever 内で LLM filtering も実行

# 03_rag_retrievers.py: 完全な実装
def retrieve(self, query: str):
    # 1. FAISS search
    # 2. LLM filtering
    # 3. Level 1-3 retrieval
    pass
```

**Option B: 外部で filtering、Retriever は使うだけ（採用）**
```python
# 04_eval_rag.py: filtering 実行
filtered_triples = filter_triples_batch(...)
retriever._filtered_triples = filtered_triples
chunks = retriever.retrieve(question)

# 03_rag_retrievers.py: 外部でセットされていればそれを使う
def retrieve(self, query: str):
    if self._filtered_triples is None:
        # 外部でセットされていない場合のみ FAISS search
        # ...
    # else: 外部でセットされた値を使う
```

**採用理由**:
- LLM filtering は実験的な機能（パラメータ調整が必要）
- `04_eval_rag.py` で柔軟に制御できる方が望ましい
- Retriever クラスは最小限の実装に留める

### 教訓 2: **状態の初期化を明示的に**

**問題**:
- `_filtered_triples` が前回のクエリの値を保持していた
- 各クエリで明示的にリセットしていなかった

**対策**:

```python
# ✅ 各クエリの前にリセット
for rec_idx, rec in enumerate(records):
    question = rec["question"]
    
    # v0.3.1.5: triple filtering の状態をリセット（毎回新しく実行するため）
    if hasattr(retriever, "_filtered_triples"):
        retriever._filtered_triples = None  # ← 明示的にリセット
    
    # Triple filtering 実行
    filtered_triples = filter_triples_batch(...)
    retriever._filtered_triples = filtered_triples
    
    # Retrieval 実行
    chunks = retriever.retrieve(question)
```

**原則**:
- **ステートフルなオブジェクト**を使う場合、状態の初期化は明示的に
- 「前回の状態が残っていないか」を常に確認
- 可能なら**ステートレス**な設計にする

### 教訓 3: **条件分岐でスキップされる処理に注意**

**問題**:
```python
# 03_rag_retrievers.py
def retrieve(self, query: str):
    # 条件: _filtered_triples が None の場合のみ実行
    if self._triple_index is not None and self._triple_data is not None:
        # FAISS search
        self._filtered_triples = [...]  # ← ここで上書き
```

- **意図**: 外部でセットされていれば、それを使う
- **実際**: 条件チェックが無いため、**常に上書き**

**修正**:

```python
# ✅ None チェックを追加
if self._filtered_triples is None:  # ← 追加
    if self._triple_index is not None and self._triple_data is not None:
        # FAISS search（外部でセットされていない場合のみ）
        self._filtered_triples = [...]
```

**原則**:
- 「外部で設定された値を尊重する」場合、**明示的に条件チェック**
- 「上書きしてよいか」を確認する
- デフォルト値（None）と有効値を区別する

### 教訓 4: **計測対象を明確にする**

**問題**:
```python
# Triple filtering（10-15s）
filtered_triples = filter_triples_batch(...)  # ← 計測外

# Retrieval（0.01s）
t0 = time.time()
chunks = retriever.retrieve(question)  # ← 計測対象
t_ret = time.time() - t0
```

- `retrieval_time = 0.01s` と表示されるが、**triple filtering の時間が含まれない**
- ユーザーが誤解する

**修正**:

```python
# ✅ triple filtering を含む全体を計測
t0 = time.time()  # ← ここで開始

# Triple filtering
filtered_triples = filter_triples_batch(...)
retriever._filtered_triples = filtered_triples

# Retrieval
chunks = retriever.retrieve(question)
t_ret = time.time() - t0  # ← triple filtering 含む
```

**原則**:
- **ユーザーが体感する時間**を計測する
- 「Retrieval」という名前なら、検索に関連する全処理を含める
- 詳細なプロファイリングは別途実装（`_perf_stats` など）

---

## 📊 修正による改善

### 修正前（v0.3.1.5 初期実装）
```
swallow_hipporag2 (200問):
  avg Retrieval: 0.02s  ← triple filtering 未実行
  avg Judge score: 0.745 / 3.0  ← ベースラインと同等
  [DEBUG] ログ: 最初の3問のみ  ← debug_mode のため
```

### 修正後（v0.3.1.5 バグ修正版）- dry-run 結果
```
swallow_hipporag2 (10問):
  avg Retrieval: 11.96s  ← triple filtering 実行（LLM 呼び出し含む）
  個別時間:
    Q0: 18.815s（初回、モデル初期化オーバーヘッド）
    Q1-Q9: 11.2s 前後（安定）
  [FILTER] ログ: 全10問で表示
    Q1: 7/20, Q2: 2/20, Q3: 6/20, Q4: 4/20, Q5: 5/20,
    Q6: 9/20, Q7: 2/20, Q8: 10/20, Q9: 9/20, Q10: 5/20
```

### 期待値（200問全体）
```
swallow_hipporag2 (200問):
  avg Retrieval: 11-12s  ← 安定した計測
  avg Judge score: 1.8-2.5 / 3.0  ← triple filtering の効果
  Perfect-Score率: 20-30%  ← 改善
```

---

## 🔧 実装の詳細

### 修正1: 条件チェックの追加（03_rag_retrievers.py）

```python
def retrieve(self, query: str) -> list[RetrievedChunk]:
    self._init()
    q_vec = _encode(query)
    
    # ── v0.3.1.5 CRITICAL FIX: 外部でセットされていない場合のみ実行 ──
    if self._filtered_triples is None:  # ← 追加
        if self._triple_index is not None and self._triple_data is not None:
            # FAISS search のみ（LLM filtering なし）
            n_retrieve = min(self.n_triples, len(self._triple_data))
            D, I = self._triple_index.search(q_vec, n_retrieve)
            
            self._filtered_triples = []
            for idx, score in zip(I[0], D[0]):
                if 0 <= idx < len(self._triple_data):
                    self._filtered_triples.append((self._triple_data[idx], float(score)))
        else:
            self._filtered_triples = []
    # else: 外部（04_eval_rag.py）でセットされた値を使う
```

### 修正2: 状態のリセット（04_eval_rag.py）

```python
for rec_idx, rec in enumerate(records):
    question = rec["question"]
    
    # v0.3.1.5 CRITICAL FIX: 状態をリセット（毎回新しく実行）
    if hasattr(retriever, "_filtered_triples"):
        retriever._filtered_triples = None  # ← 追加
    
    # Triple filtering 実行
    if triple_index is not None and triple_data is not None:
        # ... FAISS search + LLM filtering ...
        filtered_triples = filter_triples_batch(...)
        retriever._filtered_triples = filtered_triples
    
    # Retrieval 実行
    chunks = retriever.retrieve(question)
```

### 修正3: 計測位置の修正（04_eval_rag.py）

```python
for rec_idx, rec in enumerate(records):
    question = rec["question"]
    
    # v0.3.1.5 FIX: Retrieval 時間計測開始（triple filtering 含む）
    t0 = time.time()  # ← 移動（triple filtering の前）
    
    # v0.3: triple filtering の状態をリセット
    if hasattr(retriever, "_filtered_triples"):
        retriever._filtered_triples = None
    
    # v0.3: triple filtering を使う場合のみ
    if triple_index is not None and triple_data is not None:
        # ... FAISS search + LLM filtering ...
        filtered_triples = filter_triples_batch(...)
        retriever._filtered_triples = filtered_triples
    
    # ── 既存の Retrieval ──
    chunks = retriever.retrieve(question)
    t_ret = time.time() - t0  # ← triple filtering 含む全体の時間
```

### 修正4: デバッグログの追加（04_eval_rag.py）

```python
def filter_triples_batch(...):
    import sys
    
    # ★ 関数呼び出しの追跡用ログ（常に表示、全問対象）
    if len(triples) > 0:
        print(f"[FILTER] Called with {len(triples)} triples", file=sys.stderr)
    
    # ... filtering 処理 ...
    
    # ★ 終了ログ（常に表示、全問対象）
    if len(triples) > 0:
        print(f"[FILTER] Returned {len(results)}/{len(triples)} triples", file=sys.stderr)
    
    return results
```

**重要ポイント**:
- `debug=False` でも表示される（`debug` パラメータとは独立）
- `stderr` に出力（stdout と分離、パイプラインで抽出可能）
- 全200問で動作確認可能

---

## 🎓 一般化できる原則

### 1. **デバッグ出力の条件を確認する**

```python
# ❌ 誤解を招く実装
debug_mode = rec_idx < 3  # 最初の3問のみ
filtered_triples = filter_triples_batch(..., debug=debug_mode)

# 問題:
# - debug=False のとき、ログが出ない
# - 「filtering が動作していない」と誤解する
```

```python
# ✅ 動作確認用のログは常に出力
def filter_triples_batch(...):
    # 常に表示（全問対象）
    print(f"[FILTER] Called with {len(triples)} triples", file=sys.stderr)
    
    # 詳細ログは debug フラグで制御
    if debug:
        print(f"    [DEBUG] Batch {batch_idx}: ...")
```

### 2. **計測対象とユーザー体感を一致させる**

```python
# ❌ ユーザー体感と異なる計測
filtered_triples = expensive_operation()  # 10s ← 計測外
t0 = time.time()
result = cheap_operation()  # 0.01s ← 計測対象
t = time.time() - t0  # = 0.01s

print(f"Processing time: {t}s")  # ← ユーザーは 10s 待ったのに 0.01s と表示
```

```python
# ✅ ユーザーが体感する全体を計測
t0 = time.time()  # ← 開始
filtered_triples = expensive_operation()  # 10s
result = cheap_operation()  # 0.01s
t = time.time() - t0  # = 10.01s

print(f"Processing time: {t}s")  # ← 正確
```

### 3. **ステートフルなオブジェクトの危険性**

```python
# ❌ 状態が残る危険性
class Retriever:
    def __init__(self):
        self._filtered_triples = None  # ← 状態を保持
    
    def retrieve(self, query):
        # self._filtered_triples が前回の値を保持している可能性
        pass
```

```python
# ✅ 明示的な初期化
for query in queries:
    retriever._filtered_triples = None  # ← リセット
    # ... filtering ...
    retriever.retrieve(query)
```

**より良い設計**:
```python
# ✅✅ ステートレスな設計
class Retriever:
    def retrieve(self, query, filtered_triples=None):
        # 引数として渡す（状態を持たない）
        pass
```

### 4. **処理の責任範囲を文書化する**

```python
class HippoRAG2Retriever:
    """
    HippoRAG2 による 3-level coarse-to-fine retrieval
    
    Triple Filtering:
        - 外部で `_filtered_triples` をセットすることを推奨（LLM filtering 済み）
        - セットされていない場合、FAISS search のみ実行（LLM filtering なし）
        - 各クエリ前に `_filtered_triples = None` でリセット推奨
    
    Usage:
        retriever = HippoRAG2Retriever(...)
        
        for query in queries:
            # リセット
            retriever._filtered_triples = None
            
            # LLM filtering（オプション）
            filtered = filter_triples_batch(...)
            retriever._filtered_triples = filtered
            
            # Retrieval 実行
            chunks = retriever.retrieve(query)
    """
    pass
```

---

## 📚 参考資料

- **関連コード**:
  - `experiments/03_rag_retrievers.py` - HippoRAG2Retriever（v0.3.1.5 バグ修正）
  - `experiments/04_eval_rag.py` - 評価パイプライン（状態管理 + 計測位置修正）
  - `experiments/LESSON_triple_filtering.md` - 本ドキュメント

- **関連ドキュメント**:
  - `experiments/README_v03.md` - v0.3.1.5 の詳細仕様
  - `experiments/LESSON_Unsloth_Install.md` - 環境構築の教訓

- **Issue Timeline**:
  - 2026-06-26 07:00 - 200問実行完了、スコア 0.745（期待値より低い）
  - 2026-06-26 07:30 - 原因調査開始（retrieval time 0.02s に気づく）
  - 2026-06-26 08:00 - 二重実装の問題を特定
  - 2026-06-26 08:15 - 修正完了（3箇所）
  - 2026-06-26 08:20 - dry-run で検証成功（avg Retrieval 11.96s）
  - 2026-06-26 08:30 - 200問再実行開始

---

**作成者**: GitHub Copilot  
**最終更新**: 2026-06-26
