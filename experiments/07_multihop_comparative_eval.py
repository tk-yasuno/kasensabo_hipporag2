#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Multi-hop RAG Comparative Evaluation
────────────────────────────────────────────────────────────
Multi-hop reasoning専用のComparative Judge評価

評価軸:
- multi-hop integration
- cross-section reasoning  
- causal reasoning
- global coherence

比較パターン:
1. Naive RAG vs Light RAG (CoLRAG)
2. Naive RAG vs HippoRAG2 (CoLRAG + Triple Filtering)

Judge LLM: qwen2.5:14b (Ollama)
"""

import json
import argparse
import time
from pathlib import Path
from typing import Dict, List, Tuple
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

# 定数
OLLAMA_BASE_URL = "http://localhost:11434"
JUDGE_MODEL = "qwen2.5:14b"

# Multi-hop Judge用のSystem Prompt
SYSTEM_PROMPT = """You are an expert evaluator for multi-hop Retrieval-Augmented Generation (RAG).
Your task is to judge which answer demonstrates **stronger multi-hop reasoning**, defined as:

- Combining information from **multiple distinct sources**
- Connecting **multiple concepts or entities**
- Using **explicit reasoning steps**
- Showing **cross-chunk or cross-section synthesis**
- Providing **causal or structural explanations**, not isolated facts
- Avoiding hallucinations and maintaining factual consistency

You must evaluate **Answer A** and **Answer B** strictly on multi-hop reasoning quality.

Do NOT judge writing style, fluency, politeness, or length.
Do NOT judge based on correctness alone.
Focus ONLY on multi-hop reasoning."""

# User Prompt Template
USER_PROMPT_TEMPLATE = """### **Question**
{question}

### **Answer A**
{answer_a}

### **Answer B**
{answer_b}

### **Evaluation Criteria**
Score each answer on the following 4 dimensions:

1. **Multi-hop Integration**
   - Does the answer combine information from multiple distinct concepts, sections, or chunks?
   - Does it explicitly connect A→B→C のような推論をしているか？

2. **Cross-Section Reasoning**
   - Does the answer reference multiple chapters, sections, or viewpoints?
   - Does it synthesize them rather than listing them?

3. **Causal / Structural Explanation**
   - Does the answer explain *why* or *how* the concepts relate?
   - Does it show causal chains or structural relationships?

4. **Global Coherence**
   - Does the answer maintain consistency across multiple hops?
   - Does it avoid contradictions or isolated statements?

### **Output Format**
Provide your evaluation in the following JSON format:

```json
{{
  "multi_hop_integration_winner": "A or B",
  "cross_section_reasoning_winner": "A or B",
  "causal_structural_reasoning_winner": "A or B",
  "global_coherence_winner": "A or B",
  "overall_winner": "A or B",
  "explanation": "Short explanation (3–5 sentences)"
}}
```

Respond ONLY with valid JSON. Do not include any other text."""


def load_results(file_path: Path) -> List[Dict]:
    """結果ファイルを読み込み、idxでソート"""
    results = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                results.append(json.loads(line))
    # idxでソート
    results.sort(key=lambda x: x.get('idx', 0))
    return results


def call_ollama_chat(system: str, user: str, model: str = JUDGE_MODEL) -> str:
    """Ollama Chat APIを呼び出し"""
    url = f"{OLLAMA_BASE_URL}/api/chat"
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user}
        ],
        "stream": False,
        "options": {
            "temperature": 0.0,
            "num_predict": 512
        }
    }
    
    try:
        response = requests.post(url, json=payload, timeout=120)
        response.raise_for_status()
        data = response.json()
        return data['message']['content'].strip()
    except Exception as e:
        print(f"  ⚠ Ollama API error: {e}")
        return ""


def parse_judge_response(response: str) -> Dict:
    """Judge responseをパース"""
    # JSONブロックを抽出
    if "```json" in response:
        start = response.find("```json") + 7
        end = response.find("```", start)
        json_str = response[start:end].strip()
    elif "```" in response:
        start = response.find("```") + 3
        end = response.find("```", start)
        json_str = response[start:end].strip()
    else:
        json_str = response.strip()
    
    try:
        result = json.loads(json_str)
        # 必須フィールドの検証
        required_fields = [
            "multi_hop_integration_winner",
            "cross_section_reasoning_winner",
            "causal_structural_reasoning_winner",
            "global_coherence_winner",
            "overall_winner"
        ]
        for field in required_fields:
            if field not in result:
                return None
        return result
    except json.JSONDecodeError:
        return None


def evaluate_pair(question: str, answer_a: str, answer_b: str, comparison_name: str, idx: int) -> Dict:
    """1ペアの比較評価を実施"""
    user_prompt = USER_PROMPT_TEMPLATE.format(
        question=question,
        answer_a=answer_a,
        answer_b=answer_b
    )
    
    response = call_ollama_chat(SYSTEM_PROMPT, user_prompt)
    
    if not response:
        return {
            'idx': idx,
            'comparison': comparison_name,
            'success': False,
            'error': 'Ollama API call failed'
        }
    
    parsed = parse_judge_response(response)
    
    if parsed is None:
        return {
            'idx': idx,
            'comparison': comparison_name,
            'success': False,
            'error': 'Failed to parse JSON response',
            'raw_response': response
        }
    
    return {
        'idx': idx,
        'comparison': comparison_name,
        'success': True,
        'multi_hop_integration_winner': parsed['multi_hop_integration_winner'],
        'cross_section_reasoning_winner': parsed['cross_section_reasoning_winner'],
        'causal_structural_reasoning_winner': parsed['causal_structural_reasoning_winner'],
        'global_coherence_winner': parsed['global_coherence_winner'],
        'overall_winner': parsed['overall_winner'],
        'explanation': parsed.get('explanation', ''),
        'question': question,
        'answer_a': answer_a,
        'answer_b': answer_b
    }


def run_comparative_evaluation(naive_results: List[Dict],
                               light_results: List[Dict],
                               hippo_results: List[Dict],
                               output_dir: Path,
                               workers: int = 3,
                               max_questions: int = None) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """Comparative評価を実行"""
    
    # データ長の検証
    n = len(naive_results)
    if len(light_results) != n or len(hippo_results) != n:
        raise ValueError(f"結果ファイルの長さが一致しません: Naive={n}, Light={len(light_results)}, Hippo={len(hippo_results)}")
    
    if max_questions:
        n = min(n, max_questions)
    
    print(f"\n{'='*80}")
    print(f"Multi-hop Comparative Evaluation開始")
    print(f"  対象問題数: {n}")
    print(f"  Judge LLM: {JUDGE_MODEL}")
    print(f"  並列度: {workers}")
    print(f"{'='*80}\n")
    
    # 評価タスクを作成
    tasks = []
    
    # Comparison 1: Naive vs CoLRAG
    for i in range(n):
        tasks.append({
            'idx': i,
            'comparison': 'Naive_vs_CoLRAG',
            'question': naive_results[i]['question'],
            'answer_a': naive_results[i]['answer'],
            'answer_b': light_results[i]['answer']
        })
    
    # Comparison 2: Naive vs CoLRAG-Triple Filtering
    for i in range(n):
        tasks.append({
            'idx': i,
            'comparison': 'Naive_vs_CoLRAG_TF',
            'question': naive_results[i]['question'],
            'answer_a': naive_results[i]['answer'],
            'answer_b': hippo_results[i]['answer']
        })
    
    # Comparison 3: CoLRAG vs CoLRAG-Triple Filtering
    for i in range(n):
        tasks.append({
            'idx': i,
            'comparison': 'CoLRAG_vs_CoLRAG_TF',
            'question': light_results[i]['question'],
            'answer_a': light_results[i]['answer'],
            'answer_b': hippo_results[i]['answer']
        })
    
    print(f"総評価タスク数: {len(tasks)} (各比較 {n}問 × 3比較)")
    
    # 並列実行
    results = []
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(
                evaluate_pair,
                task['question'],
                task['answer_a'],
                task['answer_b'],
                task['comparison'],
                task['idx']
            ): task for task in tasks
        }
        
        with tqdm(total=len(tasks), desc="Comparative Evaluation") as pbar:
            for future in as_completed(futures):
                result = future.result()
                results.append(result)
                pbar.update(1)
                
                # 成功/失敗の簡易表示
                if result['success']:
                    pbar.set_postfix({'status': 'OK', 'winner': result['overall_winner']})
                else:
                    pbar.set_postfix({'status': 'FAIL'})
    
    # 比較別に分割
    naive_vs_colrag = [r for r in results if r['comparison'] == 'Naive_vs_CoLRAG']
    naive_vs_colrag_tf = [r for r in results if r['comparison'] == 'Naive_vs_CoLRAG_TF']
    colrag_vs_colrag_tf = [r for r in results if r['comparison'] == 'CoLRAG_vs_CoLRAG_TF']
    
    # idxでソート
    naive_vs_colrag.sort(key=lambda x: x['idx'])
    naive_vs_colrag_tf.sort(key=lambda x: x['idx'])
    colrag_vs_colrag_tf.sort(key=lambda x: x['idx'])
    
    # 保存
    output_dir.mkdir(parents=True, exist_ok=True)
    
    with open(output_dir / 'naive_vs_colrag.jsonl', 'w', encoding='utf-8') as f:
        for r in naive_vs_colrag:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')
    
    with open(output_dir / 'naive_vs_colrag_tf.jsonl', 'w', encoding='utf-8') as f:
        for r in naive_vs_colrag_tf:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')
    
    with open(output_dir / 'colrag_vs_colrag_tf.jsonl', 'w', encoding='utf-8') as f:
        for r in colrag_vs_colrag_tf:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')
    
    print(f"\n✓ 評価結果保存:")
    print(f"  - {output_dir / 'naive_vs_colrag.jsonl'}")
    print(f"  - {output_dir / 'naive_vs_colrag_tf.jsonl'}")
    print(f"  - {output_dir / 'colrag_vs_colrag_tf.jsonl'}")
    
    return naive_vs_colrag, naive_vs_colrag_tf, colrag_vs_colrag_tf


def aggregate_results(naive_vs_colrag: List[Dict], naive_vs_colrag_tf: List[Dict], colrag_vs_colrag_tf: List[Dict], output_dir: Path):
    """評価結果を集計"""
    
    def count_winners(results: List[Dict]) -> Dict:
        """各軸の勝者を集計"""
        successful = [r for r in results if r['success']]
        total = len(successful)
        
        if total == 0:
            return {'error': 'No successful evaluations'}
        
        axes = [
            'multi_hop_integration',
            'cross_section_reasoning',
            'causal_structural_reasoning',
            'global_coherence',
            'overall'
        ]
        
        stats = {}
        for axis in axes:
            winner_key = f'{axis}_winner'
            a_wins = sum(1 for r in successful if r.get(winner_key) == 'A')
            b_wins = sum(1 for r in successful if r.get(winner_key) == 'B')
            
            stats[axis] = {
                'A_wins': a_wins,
                'B_wins': b_wins,
                'A_win_rate': a_wins / total,
                'B_win_rate': b_wins / total,
                'total': total
            }
        
        return stats
    
    print(f"\n{'='*80}")
    print("集計結果")
    print(f"{'='*80}\n")
    
    # Naive vs CoLRAG
    print("【Comparison 1: Naive RAG (A) vs CoLRAG (B)】")
    stats_nc = count_winners(naive_vs_colrag)
    if 'error' not in stats_nc:
        print(f"\n{'Axis':<30} {'Naive Win':<15} {'CoLRAG Win':<15} {'CoLRAG Win rate':<20}")
        print("-" * 80)
        for axis in ['multi_hop_integration', 'cross_section_reasoning', 
                     'causal_structural_reasoning', 'global_coherence', 'overall']:
            s = stats_nc[axis]
            print(f"{axis:<30} {s['A_wins']:<15} {s['B_wins']:<15} {s['B_win_rate']*100:<19.1f}%")
    
    # Naive vs CoLRAG-Triple Filtering
    print(f"\n{'='*80}")
    print("【Comparison 2: Naive RAG (A) vs CoLRAG-Triple Filtering (B)】")
    stats_nct = count_winners(naive_vs_colrag_tf)
    if 'error' not in stats_nct:
        print(f"\n{'Axis':<30} {'Naive Win':<15} {'CoLRAG-TF Win':<20} {'CoLRAG-TF Win rate':<20}")
        print("-" * 85)
        for axis in ['multi_hop_integration', 'cross_section_reasoning', 
                     'causal_structural_reasoning', 'global_coherence', 'overall']:
            s = stats_nct[axis]
            print(f"{axis:<30} {s['A_wins']:<15} {s['B_wins']:<20} {s['B_win_rate']*100:<19.1f}%")
    
    # CoLRAG vs CoLRAG-Triple Filtering
    print(f"\n{'='*80}")
    print("【Comparison 3: CoLRAG (A) vs CoLRAG-Triple Filtering (B)】")
    stats_cct = count_winners(colrag_vs_colrag_tf)
    if 'error' not in stats_cct:
        print(f"\n{'Axis':<30} {'CoLRAG Win':<15} {'CoLRAG-TF Win':<20} {'CoLRAG-TF Win rate':<20}")
        print("-" * 85)
        for axis in ['multi_hop_integration', 'cross_section_reasoning', 
                     'causal_structural_reasoning', 'global_coherence', 'overall']:
            s = stats_cct[axis]
            print(f"{axis:<30} {s['A_wins']:<15} {s['B_wins']:<20} {s['B_win_rate']*100:<19.1f}%")
    
    # サマリー保存
    summary = {
        'naive_vs_colrag': stats_nc,
        'naive_vs_colrag_tf': stats_nct,
        'colrag_vs_colrag_tf': stats_cct,
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
    }
    
    summary_path = output_dir / 'comparative_eval_summary.json'
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    
    print(f"\n✓ サマリー保存: {summary_path}")
    print(f"{'='*80}\n")


def main():
    parser = argparse.ArgumentParser(description='Multi-hop RAG Comparative Evaluation')
    parser.add_argument('--results-dir', type=str, default='experiments/results',
                       help='結果ファイルのディレクトリ')
    parser.add_argument('--output-dir', type=str, default='experiments/results/comparative_eval',
                       help='評価結果の出力ディレクトリ')
    parser.add_argument('--prefix', type=str, default='qwen257b',
                       help='結果ファイルのプレフィックス')
    parser.add_argument('--workers', type=int, default=3,
                       help='並列評価ワーカー数')
    parser.add_argument('--max-questions', type=int, default=None,
                       help='評価する最大問題数（デバッグ用）')
    parser.add_argument('--judge-model', type=str, default='qwen2.5:14b',
                       help='Judge LLMモデル名')
    
    args = parser.parse_args()
    
    global JUDGE_MODEL
    JUDGE_MODEL = args.judge_model
    
    results_dir = Path(args.results_dir)
    output_dir = Path(args.output_dir)
    
    # 結果ファイル読み込み
    print("結果ファイル読み込み中...")
    naive_results = load_results(results_dir / f'{args.prefix}_naive_results.jsonl')
    light_results = load_results(results_dir / f'{args.prefix}_light_results.jsonl')
    hippo_results = load_results(results_dir / f'{args.prefix}_hipporag2_results.jsonl')
    
    print(f"  ✓ Naive RAG: {len(naive_results)} questions")
    print(f"  ✓ Light RAG: {len(light_results)} questions")
    print(f"  ✓ HippoRAG2: {len(hippo_results)} questions")
    
    # Comparative評価実行
    naive_vs_colrag, naive_vs_colrag_tf, colrag_vs_colrag_tf = run_comparative_evaluation(
        naive_results, light_results, hippo_results,
        output_dir, args.workers, args.max_questions
    )
    
    # 集計
    aggregate_results(naive_vs_colrag, naive_vs_colrag_tf, colrag_vs_colrag_tf, output_dir)
    
    print("✅ Comparative Evaluation完了！")


if __name__ == '__main__':
    main()
