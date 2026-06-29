#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Multi-hop Question評価結果の統計分析と可視化
3つのRAG手法（Naive, Light, HippoRAG2）の比較分析
"""

import json
import argparse
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns

# 日本語フォント設定
matplotlib.rcParams['font.family'] = 'MS Gothic'
matplotlib.rcParams['axes.unicode_minus'] = False

# テンプレート名マッピング
TEMPLATE_NAMES = {
    "T1": "因果連鎖",
    "T2": "統合",
    "T3": "比較",
    "T4": "手順"
}

def load_results(file_path: Path) -> List[Dict]:
    """結果ファイルを読み込み"""
    results = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                results.append(json.loads(line))
    return results

def extract_scores_by_template(results: List[Dict]) -> Dict[str, List[float]]:
    """Template別にスコアを抽出"""
    scores = {f"T{i}": [] for i in range(1, 5)}
    for r in results:
        # template_idがない場合はスキップ（結果ファイルにはない可能性）
        template_id = r.get('template_id')
        if template_id is None:
            continue
        judge_score = r.get('judge_score', 0)
        if template_id in scores:
            scores[template_id].append(judge_score)
    return scores

def extract_all_scores(results: List[Dict]) -> np.ndarray:
    """全スコアを抽出"""
    return np.array([r.get('judge_score', 0) for r in results])

def calculate_score_distribution(scores: np.ndarray) -> Dict[int, int]:
    """スコア分布を計算"""
    unique, counts = np.unique(scores, return_counts=True)
    return {int(s): int(c) for s, c in zip(unique, counts)}

def perform_statistical_tests(naive_scores: np.ndarray, 
                              light_scores: np.ndarray, 
                              hippo_scores: np.ndarray) -> Dict:
    """統計的有意性検定を実施"""
    results = {}
    
    # Wilcoxon符号順位検定（対応あり）
    # Naive vs Light
    stat_nl, p_nl = stats.wilcoxon(naive_scores, light_scores, alternative='two-sided')
    results['wilcoxon_naive_vs_light'] = {
        'statistic': float(stat_nl),
        'p_value': float(p_nl),
        'significant': bool(p_nl < 0.05)
    }
    
    # Naive vs HippoRAG2
    stat_nh, p_nh = stats.wilcoxon(naive_scores, hippo_scores, alternative='two-sided')
    results['wilcoxon_naive_vs_hippo'] = {
        'statistic': float(stat_nh),
        'p_value': float(p_nh),
        'significant': bool(p_nh < 0.05)
    }
    
    # Light vs HippoRAG2
    stat_lh, p_lh = stats.wilcoxon(light_scores, hippo_scores, alternative='two-sided')
    results['wilcoxon_light_vs_hippo'] = {
        'statistic': float(stat_lh),
        'p_value': float(p_lh),
        'significant': bool(p_lh < 0.05)
    }
    
    # 効果量 (Cohen's d)
    def cohens_d(x, y):
        nx, ny = len(x), len(y)
        dof = nx + ny - 2
        return (np.mean(x) - np.mean(y)) / np.sqrt(((nx-1)*np.std(x, ddof=1)**2 + (ny-1)*np.std(y, ddof=1)**2) / dof)
    
    results['effect_size'] = {
        'naive_vs_light': float(cohens_d(naive_scores, light_scores)),
        'naive_vs_hippo': float(cohens_d(naive_scores, hippo_scores)),
        'light_vs_hippo': float(cohens_d(light_scores, hippo_scores))
    }
    
    return results

def plot_score_distributions(naive_scores: np.ndarray,
                             light_scores: np.ndarray,
                             hippo_scores: np.ndarray,
                             output_dir: Path):
    """スコア分布の可視化"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    methods = ['Naive RAG', 'Light RAG', 'HippoRAG2']
    scores_list = [naive_scores, light_scores, hippo_scores]
    colors = ['#3498db', '#e74c3c', '#2ecc71']
    
    for ax, method, scores, color in zip(axes, methods, scores_list, colors):
        dist = calculate_score_distribution(scores)
        x = [0, 1, 2, 3]
        y = [dist.get(i, 0) for i in x]
        
        ax.bar(x, y, color=color, alpha=0.7, edgecolor='black')
        ax.set_xlabel('Judge Score', fontsize=12)
        ax.set_ylabel('Count', fontsize=12)
        ax.set_title(f'{method}\nMean: {scores.mean():.3f}', fontsize=13, fontweight='bold')
        ax.set_xticks(x)
        ax.set_ylim(0, max(y) * 1.1)
        ax.grid(axis='y', alpha=0.3)
        
        # 各バーに数値を表示
        for i, v in enumerate(y):
            if v > 0:
                ax.text(i, v + max(y)*0.02, str(v), ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'score_distributions.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Score分布グラフ保存: {output_dir / 'score_distributions.png'}")

def plot_template_comparison(naive_by_template: Dict,
                             light_by_template: Dict,
                             hippo_by_template: Dict,
                             output_dir: Path):
    """Template別スコア比較"""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # 空でないテンプレートのみを使用
    templates = ['T1', 'T2', 'T3', 'T4']
    valid_templates = [t for t in templates if len(naive_by_template[t]) > 0]
    
    if len(valid_templates) == 0:
        print("  ⚠ Template別比較: 有効なテンプレートがありません")
        return
    
    x = np.arange(len(valid_templates))
    width = 0.25
    
    naive_means = [np.mean(naive_by_template[t]) for t in valid_templates]
    light_means = [np.mean(light_by_template[t]) for t in valid_templates]
    hippo_means = [np.mean(hippo_by_template[t]) for t in valid_templates]
    
    naive_stds = [np.std(naive_by_template[t]) for t in valid_templates]
    light_stds = [np.std(light_by_template[t]) for t in valid_templates]
    hippo_stds = [np.std(hippo_by_template[t]) for t in valid_templates]
    
    ax.bar(x - width, naive_means, width, label='Naive RAG', 
           color='#3498db', alpha=0.8, yerr=naive_stds, capsize=5)
    ax.bar(x, light_means, width, label='Light RAG', 
           color='#e74c3c', alpha=0.8, yerr=light_stds, capsize=5)
    ax.bar(x + width, hippo_means, width, label='HippoRAG2', 
           color='#2ecc71', alpha=0.8, yerr=hippo_stds, capsize=5)
    
    ax.set_xlabel('Question Template', fontsize=13, fontweight='bold')
    ax.set_ylabel('Mean Judge Score', fontsize=13, fontweight='bold')
    ax.set_title('Template別スコア比較（平均±標準偏差）', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([f'{t}\n{TEMPLATE_NAMES[t]}' for t in valid_templates])
    ax.legend(fontsize=11)
    ax.set_ylim(0, 3.2)
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'template_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Template別比較グラフ保存: {output_dir / 'template_comparison.png'}")

def plot_boxplot_comparison(naive_scores: np.ndarray,
                           light_scores: np.ndarray,
                           hippo_scores: np.ndarray,
                           output_dir: Path):
    """箱ひげ図による比較"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    data = [naive_scores, light_scores, hippo_scores]
    labels = ['Naive RAG', 'Light RAG', 'HippoRAG2']
    colors = ['#3498db', '#e74c3c', '#2ecc71']
    
    bp = ax.boxplot(data, patch_artist=True, notch=True, showmeans=True)
    
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
    
    # 平均値を表示
    for i, (scores, label) in enumerate(zip(data, labels), 1):
        mean_val = scores.mean()
        ax.text(i, mean_val, f'{mean_val:.3f}', 
               ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    ax.set_xticklabels(labels)
    ax.set_ylabel('Judge Score', fontsize=13, fontweight='bold')
    ax.set_title('3手法のスコア分布比較（箱ひげ図）', fontsize=14, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim(-0.2, 3.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'boxplot_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ 箱ひげ図保存: {output_dir / 'boxplot_comparison.png'}")

def plot_heatmap_template_scores(naive_by_template: Dict,
                                 light_by_template: Dict,
                                 hippo_by_template: Dict,
                                 output_dir: Path):
    """Template別スコアのヒートマップ"""
    templates = ['T1', 'T2', 'T3', 'T4']
    valid_templates = [t for t in templates if len(naive_by_template[t]) > 0]
    
    if len(valid_templates) == 0:
        print("  ⚠ ヒートマップ: 有効なテンプレートがありません")
        return
    
    methods = ['Naive RAG', 'Light RAG', 'HippoRAG2']
    
    # 平均スコアの行列を作成
    data = []
    for method_scores in [naive_by_template, light_by_template, hippo_by_template]:
        row = [np.mean(method_scores[t]) for t in valid_templates]
        data.append(row)
    
    data = np.array(data)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    im = ax.imshow(data, cmap='RdYlGn', aspect='auto', vmin=2.0, vmax=2.6)
    
    # カラーバー
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Mean Judge Score', fontsize=12, fontweight='bold')
    
    # 軸設定
    ax.set_xticks(np.arange(len(valid_templates)))
    ax.set_yticks(np.arange(len(methods)))
    ax.set_xticklabels([f'{t}\n{TEMPLATE_NAMES[t]}' for t in valid_templates])
    ax.set_yticklabels(methods)
    
    # 各セルに数値を表示
    for i in range(len(methods)):
        for j in range(len(valid_templates)):
            text = ax.text(j, i, f'{data[i, j]:.3f}',
                          ha='center', va='center', color='black', fontsize=11, fontweight='bold')
    
    ax.set_title('Template別平均スコア（ヒートマップ）', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / 'heatmap_template_scores.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ ヒートマップ保存: {output_dir / 'heatmap_template_scores.png'}")

def generate_analysis_report(naive_results: List[Dict],
                            light_results: List[Dict],
                            hippo_results: List[Dict],
                            statistical_tests: Dict,
                            output_dir: Path):
    """統計分析レポートを生成"""
    naive_scores = extract_all_scores(naive_results)
    light_scores = extract_all_scores(light_results)
    hippo_scores = extract_all_scores(hippo_results)
    
    naive_by_template = extract_scores_by_template(naive_results)
    light_by_template = extract_scores_by_template(light_results)
    hippo_by_template = extract_scores_by_template(hippo_results)
    
    report = {
        'summary': {
            'total_questions': len(naive_results),
            'methods': ['Naive RAG', 'Light RAG', 'HippoRAG2']
        },
        'overall_scores': {
            'naive_rag': {
                'mean': float(naive_scores.mean()),
                'std': float(naive_scores.std()),
                'median': float(np.median(naive_scores)),
                'perfect_rate': float((naive_scores == 3).sum() / len(naive_scores)),
                'distribution': calculate_score_distribution(naive_scores)
            },
            'light_rag': {
                'mean': float(light_scores.mean()),
                'std': float(light_scores.std()),
                'median': float(np.median(light_scores)),
                'perfect_rate': float((light_scores == 3).sum() / len(light_scores)),
                'distribution': calculate_score_distribution(light_scores)
            },
            'hipporag2': {
                'mean': float(hippo_scores.mean()),
                'std': float(hippo_scores.std()),
                'median': float(np.median(hippo_scores)),
                'perfect_rate': float((hippo_scores == 3).sum() / len(hippo_scores)),
                'distribution': calculate_score_distribution(hippo_scores)
            }
        },
        'template_scores': {},
        'statistical_tests': statistical_tests,
        'key_findings': []
    }
    
    # Template別統計
    for template in ['T1', 'T2', 'T3', 'T4']:
        # 空のテンプレートをスキップ
        if (len(naive_by_template[template]) == 0 or 
            len(light_by_template[template]) == 0 or 
            len(hippo_by_template[template]) == 0):
            continue
            
        report['template_scores'][template] = {
            'name': TEMPLATE_NAMES[template],
            'naive_mean': float(np.mean(naive_by_template[template])),
            'light_mean': float(np.mean(light_by_template[template])),
            'hippo_mean': float(np.mean(hippo_by_template[template])),
            'sample_size': len(naive_by_template[template])
        }
    
    # 主要な発見を自動生成
    best_method = max([('Naive RAG', naive_scores.mean()), 
                       ('Light RAG', light_scores.mean()), 
                       ('HippoRAG2', hippo_scores.mean())], key=lambda x: x[1])
    report['key_findings'].append(f"最高平均スコア: {best_method[0]} ({best_method[1]:.3f})")
    
    if statistical_tests['wilcoxon_naive_vs_hippo']['significant']:
        report['key_findings'].append("Naive RAG vs HippoRAG2: 統計的有意差あり (p<0.05)")
    else:
        report['key_findings'].append("Naive RAG vs HippoRAG2: 統計的有意差なし")
    
    if statistical_tests['wilcoxon_light_vs_hippo']['significant']:
        report['key_findings'].append("Light RAG vs HippoRAG2: 統計的有意差あり (p<0.05)")
    else:
        report['key_findings'].append("Light RAG vs HippoRAG2: 統計的有意差なし")
    
    # レポート保存
    report_path = output_dir / 'statistical_analysis_report.json'
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    
    print(f"✓ 統計分析レポート保存: {report_path}")
    return report

def print_summary_table(report: Dict):
    """サマリーテーブルを出力"""
    print("\n" + "="*80)
    print("Multi-hop Question評価 統計分析サマリー")
    print("="*80)
    
    print("\n【全体スコア】")
    print(f"{'手法':<15} {'平均':<10} {'標準偏差':<10} {'中央値':<10} {'Perfect率':<12}")
    print("-" * 80)
    
    for method, key in [('Naive RAG', 'naive_rag'), 
                        ('Light RAG', 'light_rag'), 
                        ('HippoRAG2', 'hipporag2')]:
        data = report['overall_scores'][key]
        print(f"{method:<15} {data['mean']:<10.3f} {data['std']:<10.3f} "
              f"{data['median']:<10.1f} {data['perfect_rate']*100:<11.1f}%")
    
    print("\n【統計的有意性検定（Wilcoxon符号順位検定）】")
    print(f"{'比較':<30} {'p値':<15} {'有意差':<10} {'効果量(d)':<10}")
    print("-" * 80)
    
    tests = report['statistical_tests']
    comparisons = [
        ('Naive vs Light', 'wilcoxon_naive_vs_light', 'naive_vs_light'),
        ('Naive vs HippoRAG2', 'wilcoxon_naive_vs_hippo', 'naive_vs_hippo'),
        ('Light vs HippoRAG2', 'wilcoxon_light_vs_hippo', 'light_vs_hippo')
    ]
    
    for comp_name, test_key, effect_key in comparisons:
        p_val = tests[test_key]['p_value']
        sig = '有意' if tests[test_key]['significant'] else '無し'
        effect = tests['effect_size'][effect_key]
        print(f"{comp_name:<30} {p_val:<15.6f} {sig:<10} {effect:<10.3f}")
    
    print("\n【Template別平均スコア】")
    if len(report['template_scores']) == 0:
        print("  (Template情報が結果ファイルに含まれていません)")
    else:
        print(f"{'Template':<15} {'Naive':<10} {'Light':<10} {'HippoRAG2':<12} {'サンプル数':<10}")
        print("-" * 80)
        
        for template in ['T1', 'T2', 'T3', 'T4']:
            if template not in report['template_scores']:
                continue
            data = report['template_scores'][template]
            print(f"{template} {data['name']:<10} {data['naive_mean']:<10.3f} "
                  f"{data['light_mean']:<10.3f} {data['hippo_mean']:<12.3f} {data['sample_size']:<10}")
    
    print("\n【主要な発見】")
    for i, finding in enumerate(report['key_findings'], 1):
        print(f"  {i}. {finding}")
    
    print("\n" + "="*80)

def main():
    parser = argparse.ArgumentParser(description='Multi-hop Question評価結果の統計分析')
    parser.add_argument('--results-dir', type=str, default='experiments/results',
                       help='結果ファイルのディレクトリ')
    parser.add_argument('--output-dir', type=str, default='experiments/results/analysis',
                       help='分析結果の出力ディレクトリ')
    parser.add_argument('--prefix', type=str, default='qwen257b',
                       help='結果ファイルのプレフィックス')
    
    args = parser.parse_args()
    
    results_dir = Path(args.results_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*80)
    print("Multi-hop Question評価 統計分析開始")
    print("="*80)
    
    # 結果ファイルを読み込み
    print("\n[1/6] 結果ファイル読み込み中...")
    naive_file = results_dir / f'{args.prefix}_naive_results.jsonl'
    light_file = results_dir / f'{args.prefix}_light_results.jsonl'
    hippo_file = results_dir / f'{args.prefix}_hipporag2_results.jsonl'
    
    naive_results = load_results(naive_file)
    light_results = load_results(light_file)
    hippo_results = load_results(hippo_file)
    
    print(f"  ✓ Naive RAG: {len(naive_results)} questions")
    print(f"  ✓ Light RAG: {len(light_results)} questions")
    print(f"  ✓ HippoRAG2: {len(hippo_results)} questions")
    
    # スコア抽出
    print("\n[2/6] スコア抽出中...")
    naive_scores = extract_all_scores(naive_results)
    light_scores = extract_all_scores(light_results)
    hippo_scores = extract_all_scores(hippo_results)
    
    naive_by_template = extract_scores_by_template(naive_results)
    light_by_template = extract_scores_by_template(light_results)
    hippo_by_template = extract_scores_by_template(hippo_results)
    
    # 統計的検定
    print("\n[3/6] 統計的有意性検定実施中...")
    statistical_tests = perform_statistical_tests(naive_scores, light_scores, hippo_scores)
    
    # 分析レポート生成
    print("\n[4/6] 統計分析レポート生成中...")
    report = generate_analysis_report(naive_results, light_results, hippo_results,
                                     statistical_tests, output_dir)
    
    # 可視化
    print("\n[5/6] グラフ生成中...")
    plot_score_distributions(naive_scores, light_scores, hippo_scores, output_dir)
    plot_boxplot_comparison(naive_scores, light_scores, hippo_scores, output_dir)
    plot_template_comparison(naive_by_template, light_by_template, hippo_by_template, output_dir)
    plot_heatmap_template_scores(naive_by_template, light_by_template, hippo_by_template, output_dir)
    
    # サマリー出力
    print("\n[6/6] サマリー出力...")
    print_summary_table(report)
    
    print(f"\n✅ 分析完了！出力ディレクトリ: {output_dir}")
    print(f"   - 統計レポート: statistical_analysis_report.json")
    print(f"   - グラフ: score_distributions.png, boxplot_comparison.png, etc.")

if __name__ == '__main__':
    main()
