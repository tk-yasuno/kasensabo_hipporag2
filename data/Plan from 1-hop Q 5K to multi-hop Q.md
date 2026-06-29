1-hop Question 5000をもとに、multi-hop Question 200を作成する。
目的：v0.6.2まで、1-hop Question 200Qを対象にCoLRAG with triple filteringを評価したが、Naieve RAG, CoLRAGとperformanceは同等であった。提案手法が複数の知識をまたいだ質問に対して有益か評価するため、2～3-hop Questionを作成する。

方針：
**「5000Q から A/B/C を自動抽出して、multi-hop Q を 200 問生成する」**ための、実装レベルでそのまま使える形でまとめます。

### 全体方針

- **入力:** 5000Q（instruction形式の Q 群）
- **出力:** 200 個の multi-hop 質問（A→B→C の 2〜3-hop）
- **戦略:**

1. 各 Q から「中心概念 A」を抽出
2. メタ情報やテキストから「関連概念 B」「上位概念 C」を推定
3. A/B/C を組み合わせて multi-hop 質問テンプレートに流し込む
4. 5000Q の中からバランスよく 200Q をサンプリング

### 1. A/B/C の自動抽出ロジック

#### 1-1. A（中心概念）の抽出

**ルールベースで十分いけるところから始めるのが安定です。**

- **パターン抽出:**
- 「◯◯はなぜ重要か？」→ A = ◯◯
- 「◯◯の目的は何か？」→ A = ◯◯
- 「◯◯の点検はどのように行うか？」→ A = ◯◯
- 「◯◯の維持管理は？」→ A = ◯◯
- **実装イメージ（疑似コード）:**

python

```
def extract_A(instruction: str) -> str:
    patterns = ["はなぜ重要", "の目的は", "の点検は", "の維持管理は"]
    for p in patterns:
        if p in instruction:
            return instruction.split(p)[0].strip("「」『』 ")
    # fallback: 名詞句を抽出（形態素解析 or LLM）
    return heuristic_noun_phrase(instruction)
```

#### 1-2. B（関連概念）の抽出

**B は「A と強い関係を持つ別の章・別の概念」。**

- もし metadata があるなら：
- `USED_IN`, `REQUIRES`, `SUBJECT_TO`, `MITIGATES` などから
- A に紐づく別ノードを B として選ぶ
- metadata がない場合は、**テキストベースで近接概念を拾う**：
- 「上流部で除去」「掘抜く方法」「堆砂対策」「ダム操作」など
- instruction 全体から A 以外の重要名詞を抽出し、そのうち
- 別章・別節に対応するものを B にする
- 疑似コード:

python

```
def extract_B(instruction: str, A: str) -> str:
    # instruction から名詞句を抽出して、A 以外を候補に
    nouns = extract_nouns(instruction)
    candidates = [n for n in nouns if n != A]
    # ルール例：堆砂対策関連なら「掘抜く方法」「上流部で除去」などを優先
    return select_best_related(candidates, A)
```

#### 1-3. C（上位概念）の抽出

**C は「A と B を包含する上位の目的・枠組み」。**

- 河川砂防技術基準なら、典型的な C 候補は：
- 「河川計画の基本構造」
- 「流域治水」
- 「治水安全度の確保」
- 「維持管理計画」
- 「堆砂対策の基本方針」
- 「洪水被害軽減」
- ルール：
- A/B に「堆砂」「除去」「掘削」が含まれる → C = 「堆砂対策の基本方針」
- A/B に「計画高水流量」「洪水流解析」→ C = 「河川計画の基本構造」
- A/B に「維持管理」「点検」「劣化診断」→ C = 「維持管理計画」
- 疑似コード:

python

```
def extract_C(A: str, B: str) -> str:
    text = A + " " + B
    if "堆砂" in text or "除去" in text or "掘抜" in text:
        return "堆砂対策の基本方針"
    if "計画高水流量" in text or "洪水流解析" in text:
        return "河川計画の基本構造"
    if "維持管理" in text or "点検" in text or "劣化診断" in text:
        return "維持管理計画"
    # fallback
    return "流域治水"
```

### 2. multi-hop 質問テンプレート

200Q すべてを同じ形にすると単調なので、**3〜4 種類のテンプレートを回す**のが良いです。

#### テンプレート T1：A→B→C の因果連鎖

text

```
A が B に与える影響を整理し、C の観点から最終的な判断を示せ。
```

#### テンプレート T2：A と B の統合による C

text

```
A と B の両方を踏まえて、C を達成するための総合的な対策を示せ。
```

#### テンプレート T3：A（第X章）と B（第Y章）の比較

text

```
A と B の要件を比較し、C の観点からどのように調整すべきか論じよ。
```

#### テンプレート T4：手順・プロセス型

text

```
A の要件が B にどのように反映されるかを整理し、最終的に C を満たすための手順を示せ。
```

### 3. 200Q 生成の流れ（Pythonイメージ）

python

```
import random

def make_multihop_Q(A: str, B: str, C: str, template_id: int) -> str:
    if template_id == 1:
        return f"{A} が {B} に与える影響を整理し、{C} の観点から最終的な判断を示せ。"
    if template_id == 2:
        return f"{A} と {B} の両方を踏まえて、{C} を達成するための総合的な対策を示せ。"
    if template_id == 3:
        return f"{A} と {B} の要件を比較し、{C} の観点からどのように調整すべきか論じよ。"
    if template_id == 4:
        return f"{A} の要件が {B} にどのように反映されるかを整理し、最終的に {C} を満たすための手順を示せ。"
    return f"{A} と {B} の関係を踏まえ、{C} の観点から説明せよ。"

def generate_200_multihop(questions_5000):
    multihop_Qs = []
    for q in questions_5000:
        A = extract_A(q["instruction"])
        B = extract_B(q["instruction"], A)
        C = extract_C(A, B)
        template_id = random.randint(1, 4)
        mh_q = make_multihop_Q(A, B, C, template_id)
        multihop_Qs.append(mh_q)
        if len(multihop_Qs) >= 200:
            break
    return multihop_Qs
```

### 4. 堆砂対策系の具体例（あなたの添付に寄せた形）

添付のような Q：

> 「堆砂対策として掘抜く方法と比べて、上流部で『除去』（取り除き）を行う方法は、どのような利点と課題があるか。」
> から抽出すると：

- A = 上流部で除去を行う堆砂対策
- B = 掘抜く方法による堆砂対策
- C = 堆砂対策の基本方針（または流域治水）
  multi-hop Q 例：

1. 「上流部で除去を行う堆砂対策と掘抜く方法による堆砂対策の関係を整理し、堆砂対策の基本方針の観点から最適な組み合わせを論じよ。」
2. 「上流部での堆砂除去が掘抜く方法に与える影響を踏まえ、流域治水の観点から堆砂対策をどのように構成すべきか説明せよ。」

作成済みの1-hop Question subset : data/generated_QA/subset_merged_5000.jsonl

詳細な調査結果を受け取りました。既存のコードベースとパイプラインを確認し、multi-hop質問生成の実装計画を作成します。

[]()

十分なコンテキストが集まりました。multi-hop質問200Q生成の詳細計画を作成します。

[]()

計画を作成しました。ユーザーの提案方針を基に、実装可能な詳細設計にまとめました。追加で、5000Qのデータ分布を簡単に分析して、計画の妥当性を確認します。

回答を分析しています...

Q: 上位概念C（例：「堆砂対策の基本方針」「流域治水」）の推定方法をどうしますか？

A: 元のdocumentsの章構造 kasen-dam-sabo-Train_setを用いて、Option Cを実行する

Q: 生成したmulti-hop質問が実際に答えられるかの検証をどこまで厳密にしますか？

A: Option B: LLMによる全200Q判定 - 「この質問は答えられるか？」を全件LLM判定。精度高いが時間増（+30分程度）

Q: 200Qのサンプリングバランスは何を優先しますか？

A: Option A: rel_type優先 - 元の5000Qの rel_type分布を維持（HAS_CHAPTER 30%、DESCRIBED_IN 25%など）

[]()

ユーザーの選択を確認しました。計画を更新し、元のドキュメントから章構造を完全パースする手順を追加します。

## 📋 Plan: Multi-hop Question 200問生成

### **概要**

既存の1-hop Question 5000Q（[subset_merged_5000.jsonl](vscode-file://vscode-app/c:/Users/yasun/AppData/Local/Programs/Microsoft%20VS%20Code/7e7950df89/resources/app/out/vs/code/electron-browser/workbench/workbench.html)）から、**2～3-hop**のmulti-hop Question **200問**を生成します。

 **目的** : CoLRAG with Triple Filteringが複数知識をまたぐ質問に対して有効かを評価

 **推奨アプローチ** :

* Graph metadata（rel_type, src, tgt）を活用した**半自動パイプライン**
* **章構造の完全パース** （ユーザー要望Option C） → 階層辞書から上位概念を決定
* **LLM検証** （Qwen2.5:14b）で全候補を検証（ユーザー要望Option B）
* **rel_type分布優先**のサンプリング（ユーザー要望Option A）

### **Steps**

#### **Phase 1: データ分析とメタデータ整理** ( *parallel* )

1. 5000Qのrel_type分布を分析（HAS_CHAPTER, DESCRIBED_IN, REQUIRES等の出現頻度）
2. 章・節構造の階層を抽出（第X章 → 第Y節 → 具体概念）
3. 概念ペア（src-tgt）の関係マッピングを作成

#### **Phase 2: 章構造の完全パース** ( *depends on 1* )

4. **章構造パーサー実装**
   * 8つのMarkdownファイル（調査編、計画編、施設配置編、設計編、維持管理編3冊）を走査
   * 正規表現で「第X章」「第Y節」「第Z項」を抽出
   * 階層辞書を作成：`{編: {章: {節: {項: [概念リスト]}}}}`
5. **上位概念マッピング辞書生成**
   * 具体概念（例：「砂防堰堤」「掘削除去」）→ 節
   * 節 → 章 → 編
   * 3層の上位概念候補を辞書化 → [chapter_hierarchy.json](vscode-file://vscode-app/c:/Users/yasun/AppData/Local/Programs/Microsoft%20VS%20Code/7e7950df89/resources/app/out/vs/code/electron-browser/workbench/workbench.html)

#### **Phase 3: A/B/C抽出パイプライン構築** ( *depends on 5* )

6. **Aモジュール** : instructionからルールベース抽出

* パターンマッチ："◯◯はなぜ" → A=◯◯
* fallback: metadata.src/tgt から選択

6. **Bモジュール** : グラフ探索で関連概念を検索

* 同じsrcを持つ別QAレコード（異なるrel_type）を探索
* 異なる章に属する概念をB候補として優先

6. **Cモジュール** : 階層辞書から上位概念を選択

* A/Bが共通して属する最小の上位階層を選択
* 例：A=「砂防堰堤」(第4章第2節) + B=「定期点検」(第2節点検) → C=「第4章 砂防関係施設の状態把握」

#### **Phase 4: Multi-hop質問テンプレート生成**

9. 4種類テンプレート実装（ユーザー提示のT1-T4）
   * T1: A→B→Cの因果連鎖
   * T2: A+Bの統合によるC
   * T3: A vs B の比較 → C
   * T4: A→B手順 → C
10. テンプレート選択ロジック（ローテーション + rel_type適合度）

#### **Phase 5: 候補質問の生成** ( *depends on 8, 9* )

11. 5000Qをイテレート、各Qに対して：
    * A抽出 → Bグラフ探索（候補3-5個）
    * 各A-Bペアに対してC推定（階層辞書）
    * テンプレート適用 → 候補質問生成
    * **初期フィルタリング** ：A≠B≠C、同一章でない
12. 候補質問プール作成（500-1000問程度を目標）

#### **Phase 6: LLMによる質問検証** ( *depends on 11* )

13. Qwen2.5:14bで各候補質問を検証
    * プロンプト：「この質問は複数の章・節をまたいだ知識で答えられる適切なmulti-hop質問か？YES/NO + 理由」
    * JSON応答：`{"valid": "YES/NO", "reason": "..."}`
    * バッチ処理（10件ずつ）で高速化
14. YES判定の質問のみ残す（目標：300-400問）

#### **Phase 7: バランスサンプリング200Q** ( *depends on 13, 14* )

15. 層化サンプリング実装（検証済み候補から）
    * **rel_type分布優先** （元の5000Qと類似分布を維持）
    * 章カバレッジ確認（8編 × 主要章）
    * hop数分布：2-hop優先だが3-hopも含める（自然発生比率）
16. seed=42で固定、再現性確保
17. 最終200Qを選択

#### **Phase 8: 出力とドキュメント生成** ( *depends on 15, 16, 17* )

18. `testset_multihop_200.jsonl` 生成
    * 既存形式互換：`{"idx", "question", "answer", "source"}`
    * 拡張メタデータ：`{"concept_A", "concept_B", "concept_C", "rel_types", "template_id", "hop_count", "llm_validation_reason"}`
19. サマリーレポート作成
    * rel_type分布グラフ
    * 章カバレッジ表
    * hop数分布
    * 10件サンプル表示
20. 検証済み候補全体も保存（`multihop_candidates_validated.jsonl`）

### **Relevant files**

 **入力データ** :

* [subset_merged_5000.jsonl](vscode-file://vscode-app/c:/Users/yasun/AppData/Local/Programs/Microsoft%20VS%20Code/7e7950df89/resources/app/out/vs/code/electron-browser/workbench/workbench.html) — 1-hop質問5000件（graph_relation metadata付き）
* data/kasen-dam-sabo_Train_set/*.md — 河川砂防技術基準8冊（Markdown形式）

 **再利用コード** :

* [02_prepare_testset.py](vscode-file://vscode-app/c:/Users/yasun/AppData/Local/Programs/Microsoft%20VS%20Code/7e7950df89/resources/app/out/vs/code/electron-browser/workbench/workbench.html) — サンプリングロジック、convert関数
* [01_build_triple_index.py**:60**](vscode-file://vscode-app/c:/Users/yasun/AppData/Local/Programs/Microsoft%20VS%20Code/7e7950df89/resources/app/out/vs/code/electron-browser/workbench/workbench.html) — Qwen2.5 JSON修復処理（_fix_json関数）
* [04_eval_rag.py**:387**](vscode-file://vscode-app/c:/Users/yasun/AppData/Local/Programs/Microsoft%20VS%20Code/7e7950df89/resources/app/out/vs/code/electron-browser/workbench/workbench.html) — Ollama API呼び出し（_ollama_chat関数）

 **新規作成ファイル** :

* `experiments/02b_prepare_multihop_testset.py` — メインスクリプト
* [experiments/chapter_hierarchy.json](vscode-file://vscode-app/c:/Users/yasun/AppData/Local/Programs/Microsoft%20VS%20Code/7e7950df89/resources/app/out/vs/code/electron-browser/workbench/workbench.html) — 章構造階層辞書
* `experiments/testset_multihop_200.jsonl` — 最終出力（200問）
* `experiments/multihop_candidates_validated.jsonl` — 検証済み候補全体

---

### **Verification**

1. **データ品質検証**
   * 全200Qについて：A≠B≠C が成立
   * hop_count分布が2-3の範囲内
   * rel_type分布が元の5000Qと類似（±10%以内）
2. **RAG取得テスト** （dry-run）

* 10件サンプルでhipporag2 retrieverを実行
* 関連チャンクが取得できるか確認

1. **LLM回答生成テスト**
   * 3件についてSwallow 8Bで回答生成
   * 回答が空でないことを確認
2. **Judge採点テスト** （optional）

* 既存Judge基準で採点可能か確認
* スコア分布が0-3の範囲か

---

### **Decisions**

#### **設計判断（ユーザー確認済み）**

* **A/B抽出** : ルールベース（metadata.src/tgt活用） + グラフ探索
* **C推定** : **章構造の完全パース** → 階層辞書から上位概念を決定的に選択
* 理由: ユーザー要望（Option C）。最も正確で一貫性が高い
* データソース: `kasen-dam-sabo_Train_set/*.md`（8ファイル）
* **質問検証** :  **LLMによる全候補検証** （Qwen2.5:14b）
* 理由: ユーザー要望（Option B）。精度を重視、不適切な質問を排除
* **サンプリング戦略** : **rel_type分布優先**の層化サンプリング
* 理由: ユーザー要望（Option A）。既存テストセットとの一貫性確保

#### **対象範囲**

 **含む** :

* 2-hop（A→B）と3-hop（A→B→C）質問
* 既存metadataに関係情報があるQA対
* 河川砂防技術基準8冊全体からバランスよく

 **除外** :

* 4-hop以上（複雑度過多）
* metadataが不完全なQA（rel_type, src, tgt欠損）
* 同一概念の繰り返し（A=BやB=Cのケース）

---

### **Estimated Timeline**

* Phase 1: データ分析（rel_type分布、章構造確認） → **30分**
* Phase 2: 章構造完全パース →  **1-2時間** （8ファイル処理）
* Phase 3: A/B/C抽出パイプライン実装 → **2-3時間**
* Phase 4: テンプレート実装 → **1時間**
* Phase 5: 候補質問生成（500-1000問） → **30分**
* Phase 6: LLM検証（Qwen2.5） →  **30-60分** （バッチ処理）
* Phase 7: サンプリング200Q → **30分**
* Phase 8: 出力・ドキュメント生成 → **30分**
* 検証・調整 → **1-2時間**

 **Total** : 7-11時間（うちLLM処理待ち時間 30-60分）

実装の核心ポイント

1. 章構造パーサー（Phase 2）

def parse_chapter_structure(md_files: list[Path]) -> dict:
    """
    8つのMarkdownから階層辞書を構築
    Returns: {編名: {章ID: {節ID: {項ID: [概念リスト]}}}}
    """
    # 正規表現: ^第(\d+)章\s+(.+)$
    # 正規表現: ^第(\d+)節\s+(.+)$
    # metadata.src/tgtと章・節名をマッチング

2. グラフ探索によるB抽出（Phase 3）

def find_related_concept_B(A: str, qa_records: list) -> list[str]:
    """
    Aと関連するが異なる概念Bを探索
    - 同じsrc/tgtを持つ別QAレコード
    - 異なるrel_typeを優先（HAS_CHAPTER vs REQUIRES）
    - 異なる章に属する概念を優先
    """

3. LLM検証プロンプト（Phase 6）

VALIDATION_SYSTEM = """あなたは河川砂防技術基準の専門家です。
以下の質問が「複数の章・節をまたいだ知識を必要とする適切なmulti-hop質問か」を判定してください。"""

VALIDATION_USER = """質問: {question}

判定基準:

- 2つ以上の異なる章・節の知識が必要 → YES
- 1つの章で完結する → NO
- 質問が不明瞭・答えられない → NO

回答形式（JSONのみ）:
{"valid": "YES", "reason": "第4章と第7章の知識を統合する必要がある"}
{"valid": "NO", "reason": "第4章のみで完結"}
"""

### **次のステップ**

**即座に開始可能**

1. ✅  **Phase 1実行** : 5000Qのrel_type分布を分析（Python簡易スクリプト）
2. ✅  **Phase 2開始** : 章構造パーサー実装
   * 入力：`kasen-dam-sabo_Train_set/*.md`（8ファイル）
   * 出力：[chapter_hierarchy.json](vscode-file://vscode-app/c:/Users/yasun/AppData/Local/Programs/Microsoft%20VS%20Code/7e7950df89/resources/app/out/vs/code/electron-browser/workbench/workbench.html)
3. Phase 3以降：A/B/C抽出パイプライン実装

#### **進捗確認ポイント**

* Phase 2完了時：階層辞書のサンプル確認（上位概念が適切か）
* Phase 5完了時：候補質問500-1000問の品質サンプリング（10件）
* Phase 6完了時：LLM検証の精度確認（YES/NO比率、理由の妥当性）
* Phase 7完了時：最終200Qの分布確認

---

計画の確認と承認をお願いします。問題なければ、Phase 1から実装を開始できます。


**オプションA（推奨）:** 候補を事前フィルタしてから検証

* 12,712候補をスコアでソート（b_scoreが高い上位3000候補）
* 3000候補を検証（推定時間: 20-40分）
* 有効な候補から200Q選択



**実行内容：**

* 全5000Q → 12,712候補生成
* b_scoreで上位3000候補に絞り込み
* LLM検証（並列度5、qwen2.5-14b-gpu）
* 有効候補から200Q選択

**推定時間：** 20-40分

進捗は定期的に表示されます（50候補ごと）。完了後、以下が生成されます：

1. `multihop_candidates_validated.jsonl` - 検証済み候補
2. `testset_multihop_200.jsonl` - 最終的な200問テストセット
