"""
experiments/02b_build_volume_keywords.py
──────────────────────────────────────────────────────────
Phase 2b: ボリューム分類用キーワード辞書の自動生成（オプション）

volume_keywords.json から初期辞書をロードし、
テストセットの質問文から自動抽出したキーワードで拡張します。

Usage:
    python experiments/02b_build_volume_keywords.py           # volume_keywords.jsonを検証・表示
    python experiments/02b_build_volume_keywords.py --analyze  # テストセットから統計情報を表示
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path

EXP_DIR = Path(__file__).parent
KEYWORDS_FILE = EXP_DIR / "volume_keywords.json"
TESTSET_FILE = EXP_DIR / "testset_200.jsonl"


def load_keywords() -> dict:
    """volume_keywords.jsonをロード"""
    if not KEYWORDS_FILE.exists():
        print(f"❌ {KEYWORDS_FILE} が見つかりません")
        return {}
    
    with open(KEYWORDS_FILE, encoding="utf-8") as f:
        return json.load(f)


def extract_keywords_from_text(text: str, keywords_dict: dict) -> tuple[str, float]:
    """
    テキストからキーワード辞書に基づいてボリューム分類を行う。
    
    Returns:
        (predicted_volume, confidence_score)
    """
    text_lower = text.lower()
    volume_scores = {}
    
    for vol_name, vol_info in keywords_dict.get("volumes", {}).items():
        score = 0.0
        
        # キーワードマッチ
        for kw in vol_info.get("keywords", []):
            if kw.lower() in text_lower:
                score += vol_info.get("confidence", 0.8)
        
        # 除外キーワードのチェック
        for ex_kw in vol_info.get("exclusion_keywords", []):
            if ex_kw.lower() in text_lower:
                score -= 0.2
        
        volume_scores[vol_name] = max(0.0, score)
    
    if not volume_scores:
        return "unknown", 0.0
    
    best_vol = max(volume_scores.items(), key=lambda x: x[1])
    return best_vol[0], best_vol[1]


def analyze_testset(keywords_dict: dict):
    """テストセットの質問を分析してボリューム分類を表示"""
    if not TESTSET_FILE.exists():
        print(f"❌ {TESTSET_FILE} が見つかりません")
        return
    
    print("\n" + "="*60)
    print("  テストセット ボリューム分類分析")
    print("="*60)
    
    volume_counts = Counter()
    predictions = []
    
    with open(TESTSET_FILE, encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            try:
                record = json.loads(line)
                question = record.get("question", "")
                vol, conf = extract_keywords_from_text(question, keywords_dict)
                volume_counts[vol] += 1
                if i <= 10 or i % 20 == 0:
                    print(f"\n[{i}] {question[:50]}...")
                    print(f"    -> {vol} (confidence: {conf:.2f})")
            except Exception as e:
                print(f"❌ 行{i}でエラー: {e}")
    
    print("\n" + "-"*60)
    print("ボリューム分布:")
    for vol, count in volume_counts.most_common():
        pct = count / (i or 1) * 100
        print(f"  {vol}: {count}/{i} ({pct:.1f}%)")


def validate_keywords(keywords_dict: dict):
    """キーワード辞書の内容を検証・表示"""
    print("\n" + "="*60)
    print("  キーワード辞書 検証")
    print("="*60)
    
    if not keywords_dict:
        print("❌ キーワード辞書が空です")
        return
    
    print(f"\nバージョン: {keywords_dict.get('version', 'N/A')}")
    print(f"作成日: {keywords_dict.get('created_date', 'N/A')}")
    print(f"説明: {keywords_dict.get('description', 'N/A')}")
    
    print("\n" + "-"*60)
    print("ボリューム情報:")
    
    for vol_name, vol_info in keywords_dict.get("volumes", {}).items():
        n_kw = len(vol_info.get("keywords", []))
        n_ex = len(vol_info.get("exclusion_keywords", []))
        conf = vol_info.get("confidence", "N/A")
        print(f"\n  {vol_name}:")
        print(f"    主キーワード数: {n_kw}")
        print(f"    除外キーワード数: {n_ex}")
        print(f"    信頼度: {conf}")
        print(f"    説明: {vol_info.get('description', 'N/A')}")
        print(f"    キーワード例: {', '.join(vol_info.get('keywords', [])[:5])}...")
    
    print("\n" + "-"*60)
    print("融合戦略:")
    fusion = keywords_dict.get("fusion_strategy", {})
    print(f"  Embedding重み: {fusion.get('embedding_weight', 'N/A')}")
    print(f"  キーワード重み: {fusion.get('keyword_weight', 'N/A')}")
    print(f"  手法: {fusion.get('method', 'N/A')}")


def main():
    parser = argparse.ArgumentParser(description="キーワード辞書管理")
    parser.add_argument("--analyze", action="store_true", help="テストセット分析")
    args = parser.parse_args()
    
    keywords_dict = load_keywords()
    
    if args.analyze:
        analyze_testset(keywords_dict)
    else:
        validate_keywords(keywords_dict)
    
    print("\n" + "="*60)
    print("✓ 完了")
    print("="*60)


if __name__ == "__main__":
    main()
