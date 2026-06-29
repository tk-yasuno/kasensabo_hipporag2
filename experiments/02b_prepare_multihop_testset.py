"""
experiments/02b_prepare_multihop_testset.py
────────────────────────────────────────────────────────────
Multi-hop Question Generator

5000Q から multi-hop question 200問を生成する。

Phases:
  1. A/B/C 抽出
  2. テンプレート適用
  3. LLM検証
  4. サンプリング

Usage:
    python experiments/02b_prepare_multihop_testset.py
    python experiments/02b_prepare_multihop_testset.py --dry-run
    python experiments/02b_prepare_multihop_testset.py --skip-validation
"""

from __future__ import annotations

import argparse
import json
import random
import re
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Optional

import httpx

ROOT = Path(__file__).parent.parent
DATA_FILE = ROOT / "data" / "generated_QA" / "subset_merged_5000.jsonl"
HIERARCHY_FILE = Path(__file__).parent / "chapter_hierarchy.json"
OUTPUT_FILE = Path(__file__).parent / "testset_multihop_200.jsonl"
CANDIDATES_FILE = Path(__file__).parent / "multihop_candidates_validated.jsonl"

# Ollama settings
OLLAMA_URL = "http://localhost:11434"
OLLAMA_MODEL = "qwen2.5-14b-gpu:latest"


# ─────────────────────────────────────────────────────────
# Templates
# ─────────────────────────────────────────────────────────

TEMPLATES = [
    # T1: A→B→Cの因果連鎖
    "{A} が {B} に与える影響を整理し、{C} の観点から最終的な判断を示せ。",
    
    # T2: A+Bの統合によるC
    "{A} と {B} の両方を踏まえて、{C} を達成するための総合的な対策を示せ。",
    
    # T3: A vs B の比較 → C
    "{A} と {B} の要件を比較し、{C} の観点からどのように調整すべきか論じよ。",
    
    # T4: A→B手順 → C
    "{A} の要件が {B} にどのように反映されるかを整理し、最終的に {C} を満たすための手順を示せ。",
]


# ─────────────────────────────────────────────────────────
# Data Loading
# ─────────────────────────────────────────────────────────

def load_qa_records(path: Path) -> list[dict]:
    """Load all QA records"""
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def load_hierarchy(path: Path) -> dict:
    """Load chapter hierarchy"""
    with open(path, encoding="utf-8") as f:
        data = json.loads(f.read())
    return data


# ─────────────────────────────────────────────────────────
# Phase 3: A/B/C Extraction
# ─────────────────────────────────────────────────────────

def extract_concept_A(record: dict) -> str:
    """Extract central concept A from instruction"""
    instruction = record.get("instruction", "")
    meta = record.get("metadata", {})
    
    # Pattern matching
    patterns = [
        r"『(.+?)』",
        r"「(.+?)」",
        r"^(.+?)はなぜ",
        r"^(.+?)の目的",
        r"^(.+?)の点検",
        r"^(.+?)の維持管理",
        r"^(.+?)を",
    ]
    
    for pattern in patterns:
        match = re.search(pattern, instruction)
        if match:
            concept = match.group(1).strip()
            if concept and len(concept) < 100:  # reasonable length
                return concept
    
    # Fallback: use src or tgt from metadata
    src = meta.get("src", "")
    tgt = meta.get("tgt", "")
    
    if src and len(src) < 100:
        return src
    if tgt and len(tgt) < 100:
        return tgt
    
    # Last fallback: first noun phrase (simplified)
    words = instruction.split()
    if words:
        return words[0][:30]
    
    return "概念A"


def find_related_concepts_B(
    concept_A: str,
    current_record: dict,
    all_records: list[dict],
    hierarchy_data: dict,
    max_candidates: int = 5
) -> list[tuple[str, str, float]]:
    """
    Find related but different concepts B via graph traversal
    
    Returns: [(concept_B, rel_type, score), ...]
    """
    candidates = []
    current_meta = current_record.get("metadata", {})
    current_src = current_meta.get("src", "")
    current_tgt = current_meta.get("tgt", "")
    current_rel = current_meta.get("rel_type", "")
    
    concept_map = hierarchy_data.get("concept_map", {})
    current_hier = concept_map.get(concept_A, concept_map.get(current_src, concept_map.get(current_tgt, {})))
    current_chapter = current_hier.get("chapter", "")
    
    for rec in all_records:
        meta = rec.get("metadata", {})
        src = meta.get("src", "")
        tgt = meta.get("tgt", "")
        rel_type = meta.get("rel_type", "")
        
        # Skip same record
        if rec is current_record:
            continue
        
        # Check if A is related to this record
        if concept_A in [src, tgt] or current_src in [src, tgt] or current_tgt in [src, tgt]:
            # Find the "other" concept
            if src == concept_A or src == current_src or src == current_tgt:
                other = tgt
            else:
                other = src
            
            if not other or other == concept_A:
                continue
            
            # Score this candidate
            score = 0.0
            
            # Prefer different rel_type
            if rel_type != current_rel:
                score += 2.0
            
            # Prefer different chapter
            other_hier = concept_map.get(other, {})
            other_chapter = other_hier.get("chapter", "")
            if other_chapter and other_chapter != current_chapter:
                score += 3.0
            elif other_chapter:
                score += 1.0
            
            # Prefer certain rel_types for multi-hop
            if rel_type in ["REQUIRES", "SUBJECT_TO", "AFFECTS", "MITIGATES", "USED_IN"]:
                score += 1.0
            
            candidates.append((other, rel_type, score))
    
    # Sort by score and return top candidates
    candidates.sort(key=lambda x: x[2], reverse=True)
    return candidates[:max_candidates]


def find_common_upper_concept(
    concept_A: str,
    concept_B: str,
    hierarchy_data: dict
) -> Optional[str]:
    """
    Find the smallest common upper-level concept (chapter or volume) for A and B
    """
    concept_map = hierarchy_data.get("concept_map", {})
    
    hier_A = concept_map.get(concept_A, {})
    hier_B = concept_map.get(concept_B, {})
    
    vol_A = hier_A.get("volume", "")
    vol_B = hier_B.get("volume", "")
    ch_A = hier_A.get("chapter", "")
    ch_B = hier_B.get("chapter", "")
    sec_A = hier_A.get("section", "")
    sec_B = hier_B.get("section", "")
    
    # Same volume, different chapters -> use volume as C
    if vol_A and vol_A == vol_B and ch_A != ch_B:
        return f"{vol_A}"
    
    # Same chapter, different sections -> use chapter as C
    if ch_A and ch_A == ch_B and sec_A != sec_B:
        ch_title = hier_A.get("chapter_title", "")
        return f"{ch_A} {ch_title}" if ch_title else ch_A
    
    # Different volumes -> use generic upper concept
    if vol_A and vol_B and vol_A != vol_B:
        # Try to find a common theme
        if "維持管理" in vol_A and "維持管理" in vol_B:
            return "河川砂防施設の維持管理"
        if "計画" in vol_A or "計画" in vol_B:
            return "河川砂防計画"
        if "設計" in vol_A or "設計" in vol_B:
            return "河川砂防施設の設計"
        return "河川砂防技術基準"
    
    # Fallback: use chapter if available
    if ch_A:
        ch_title = hier_A.get("chapter_title", "")
        return f"{ch_A} {ch_title}" if ch_title else ch_A
    if ch_B:
        ch_title = hier_B.get("chapter_title", "")
        return f"{ch_B} {ch_title}" if ch_title else ch_B
    
    # Last fallback
    return "河川砂防技術基準"


# ─────────────────────────────────────────────────────────
# Phase 4: Template Application
# ─────────────────────────────────────────────────────────

def apply_template(A: str, B: str, C: str, template_id: int) -> str:
    """Apply a question template"""
    template = TEMPLATES[template_id % len(TEMPLATES)]
    return template.format(A=A, B=B, C=C)


def calculate_hop_count(
    concept_A: str,
    concept_B: str,
    concept_C: str,
    hierarchy_data: dict
) -> int:
    """Calculate the hop count (2 or 3) based on hierarchy"""
    concept_map = hierarchy_data.get("concept_map", {})
    
    hier_A = concept_map.get(concept_A, {})
    hier_B = concept_map.get(concept_B, {})
    hier_C = concept_map.get(concept_C, {})
    
    ch_A = hier_A.get("chapter", "")
    ch_B = hier_B.get("chapter", "")
    ch_C = hier_C.get("chapter", "")
    
    unique_chapters = len(set([ch for ch in [ch_A, ch_B, ch_C] if ch]))
    
    # If 3 different chapters -> 3-hop
    if unique_chapters >= 3:
        return 3
    # If 2 different chapters -> 2-hop
    elif unique_chapters == 2:
        return 2
    # Default to 2-hop
    else:
        return 2


# ─────────────────────────────────────────────────────────
# Phase 5: Candidate Generation
# ─────────────────────────────────────────────────────────

def generate_candidates(
    records: list[dict],
    hierarchy_data: dict,
    max_candidates_per_record: int = 3,
    dry_run: bool = False
) -> list[dict]:
    """Generate multi-hop question candidates"""
    
    candidates = []
    total = len(records) if not dry_run else min(100, len(records))
    
    print(f"\nGenerating candidates from {total} records...")
    
    for idx, record in enumerate(records[:total]):
        if (idx + 1) % 500 == 0:
            print(f"  Progress: {idx+1}/{total}")
        
        # Extract A
        concept_A = extract_concept_A(record)
        
        # Find B candidates
        B_candidates = find_related_concepts_B(
            concept_A, record, records, hierarchy_data, max_candidates=5
        )
        
        if not B_candidates:
            continue
        
        # Generate questions for top B candidates
        for concept_B, rel_type_B, score in B_candidates[:max_candidates_per_record]:
            # Find C
            concept_C = find_common_upper_concept(concept_A, concept_B, hierarchy_data)
            
            if not concept_C:
                continue
            
            # Initial filter: A ≠ B ≠ C
            if concept_A == concept_B or concept_B == concept_C or concept_A == concept_C:
                continue
            
            # Calculate hop count
            hop_count = calculate_hop_count(concept_A, concept_B, concept_C, hierarchy_data)
            
            # Apply template (rotate through templates)
            template_id = idx % len(TEMPLATES)
            question = apply_template(concept_A, concept_B, concept_C, template_id)
            
            # Collect metadata
            meta = record.get("metadata", {})
            candidate = {
                "source_idx": idx,
                "question": question,
                "concept_A": concept_A,
                "concept_B": concept_B,
                "concept_C": concept_C,
                "rel_type_A": meta.get("rel_type", ""),
                "rel_type_B": rel_type_B,
                "template_id": template_id,
                "hop_count": hop_count,
                "b_score": score,
                "source_instruction": record.get("instruction", ""),
            }
            
            candidates.append(candidate)
    
    print(f"  Generated {len(candidates)} candidates")
    return candidates


# ─────────────────────────────────────────────────────────
# Phase 6: LLM Validation
# ─────────────────────────────────────────────────────────

def _ollama_chat(
    model: str,
    system: str,
    user: str,
    timeout: float = 60.0,
    temperature: float = 0.3,
) -> str:
    """Call Ollama API"""
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "stream": False,
        "keep_alive": "5m",
        "options": {
            "temperature": temperature,
            "num_predict": 512,
            "num_ctx": 4096,
            "num_gpu": 99,  # 全レイヤーをGPUにロード
        },
    }
    timeout_cfg = httpx.Timeout(connect=15.0, read=timeout, write=15.0, pool=5.0)
    try:
        resp = httpx.post(f"{OLLAMA_URL}/api/chat", json=payload, timeout=timeout_cfg)
        resp.raise_for_status()
        return resp.json().get("message", {}).get("content", "")
    except Exception as e:
        print(f"  ⚠ Ollama error: {e}")
        return ""


def validate_candidate_with_llm(candidate: dict, model: str = OLLAMA_MODEL) -> dict:
    """Validate a single candidate with LLM"""
    
    system_prompt = """あなたは河川砂防技術基準の専門家です。
以下の質問が「複数の章・節をまたいだ知識を必要とする適切なmulti-hop質問か」を判定してください。

判定基準：
1. 質問が2つ以上の異なる概念（A, B）を含んでいる
2. これらの概念が異なる章や節にまたがっている
3. 最終的な判断や統合的な対策（C）を求めている
4. 質問として文法的に正しく、意味が通る

以下のJSON形式で回答してください：
{"valid": "YES", "reason": "理由を簡潔に"}
または
{"valid": "NO", "reason": "理由を簡潔に"}
"""
    
    user_prompt = f"""質問：{candidate['question']}

概念A：{candidate['concept_A']}
概念B：{candidate['concept_B']}
概念C：{candidate['concept_C']}
hop数：{candidate['hop_count']}

この質問は適切なmulti-hop質問ですか？"""
    
    response = _ollama_chat(model, system_prompt, user_prompt, timeout=60.0, temperature=0.1)
    
    # Parse JSON response
    try:
        # Extract JSON from markdown code block if present
        if "```json" in response:
            json_str = response.split("```json")[1].split("```")[0].strip()
        elif "```" in response:
            json_str = response.split("```")[1].split("```")[0].strip()
        else:
            json_str = response.strip()
        
        result = json.loads(json_str)
        valid = result.get("valid", "NO").upper() == "YES"
        reason = result.get("reason", "")
    except:
        # Fallback: simple YES/NO detection
        valid = "YES" in response.upper() and "NO" not in response[:50].upper()
        reason = "Parse error - fallback heuristic"
    
    candidate["llm_valid"] = valid
    candidate["llm_reason"] = reason
    
    return candidate


def validate_candidates_batch(
    candidates: list[dict],
    model: str = OLLAMA_MODEL,
    max_workers: int = 5,
    max_validate: int = None
) -> list[dict]:
    """Validate candidates with LLM in parallel"""
    
    to_validate = candidates if max_validate is None else candidates[:max_validate]
    
    print(f"\nValidating {len(to_validate)} candidates with {model}...")
    print(f"  (parallel workers: {max_workers})")
    
    validated = []
    total = len(to_validate)
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(validate_candidate_with_llm, cand, model): cand for cand in to_validate}
        
        for i, future in enumerate(as_completed(futures)):
            try:
                result = future.result()
                validated.append(result)
                
                if (i + 1) % 50 == 0 or (i + 1) == total:
                    yes_count = sum(1 for v in validated if v.get("llm_valid", False))
                    print(f"  Progress: {i+1}/{total} (YES: {yes_count}/{i+1} = {yes_count/(i+1)*100:.1f}%)")
            except Exception as e:
                print(f"  ⚠ Validation error: {e}")
                validated.append(futures[future])  # Keep original
    
    # Filter to keep only valid candidates
    valid_candidates = [c for c in validated if c.get("llm_valid", False)]
    
    print(f"\n✓ Validation complete:")
    print(f"  Total validated: {len(validated)}")
    print(f"  Valid (YES):     {len(valid_candidates)} ({len(valid_candidates)/len(validated)*100:.1f}%)")
    print(f"  Invalid (NO):    {len(validated) - len(valid_candidates)}")
    
    return valid_candidates


# ─────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true", help="Process only 100 records")
    parser.add_argument("--skip-validation", action="store_true", help="Skip LLM validation (Phase 6)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--max-validate", type=int, default=None, help="Max candidates to validate (for testing)")
    parser.add_argument("--validation-workers", type=int, default=5, help="Parallel workers for validation")
    parser.add_argument("--filter-top-n", type=int, default=None, help="Filter top N candidates by b_score before validation")
    args = parser.parse_args()
    
    random.seed(args.seed)
    
    print("=" * 60)
    print("  Multi-hop Question Generator")
    print("=" * 60)
    
    # Load data
    print("\n[Loading Data]")
    if not DATA_FILE.exists():
        print(f"ERROR: Data file not found: {DATA_FILE}")
        return
    if not HIERARCHY_FILE.exists():
        print(f"ERROR: Hierarchy file not found: {HIERARCHY_FILE}")
        print("  Run 02a_parse_chapter_structure.py first")
        return
    
    records = load_qa_records(DATA_FILE)
    print(f"  5000Q records: {len(records)}")
    
    hierarchy_data = load_hierarchy(HIERARCHY_FILE)
    print(f"  Hierarchy concepts: {len(hierarchy_data.get('concept_map', {}))}")
    
    # Phase 3-5: Generate candidates
    print("\n" + "=" * 60)
    print("  Phase 3-5: Candidate Generation")
    print("=" * 60)
    
    candidates = generate_candidates(
        records,
        hierarchy_data,
        max_candidates_per_record=3,
        dry_run=args.dry_run
    )
    
    if not candidates:
        print("ERROR: No candidates generated")
        return
    
    print(f"\n✓ Generated {len(candidates)} candidates")
    
    # Show sample
    print("\n[Sample: First 5 candidates]")
    for i, cand in enumerate(candidates[:5]):
        print(f"\n  {i+1}. Question:")
        print(f"     {cand['question']}")
        print(f"     A={cand['concept_A'][:30]}, B={cand['concept_B'][:30]}, C={cand['concept_C'][:30]}")
        print(f"     hop={cand['hop_count']}, template={cand['template_id']}, b_score={cand['b_score']:.2f}")
    
    # Filter candidates by b_score if requested
    if args.filter_top_n and len(candidates) > args.filter_top_n:
        print(f"\n[Pre-filtering by b_score]")
        print(f"  Before filter: {len(candidates)} candidates")
        
        # Sort by b_score (descending)
        candidates_sorted = sorted(candidates, key=lambda x: x['b_score'], reverse=True)
        candidates = candidates_sorted[:args.filter_top_n]
        
        print(f"  After filter:  {len(candidates)} candidates (top {args.filter_top_n})")
        print(f"  b_score range: {candidates[0]['b_score']:.2f} ~ {candidates[-1]['b_score']:.2f}")
    
    # Phase 6: LLM Validation
    if not args.skip_validation:
        print("\n" + "=" * 60)
        print("  Phase 6: LLM Validation")
        print("=" * 60)
        
        # Check Ollama
        try:
            resp = httpx.get(f"{OLLAMA_URL}/api/tags", timeout=5.0)
            resp.raise_for_status()
            print(f"  ✓ Ollama is running at {OLLAMA_URL}")
        except Exception as e:
            print(f"  ⚠ Ollama not available: {e}")
            print(f"  Please start Ollama and ensure {OLLAMA_MODEL} is available")
            print(f"  Continuing without validation...")
            validated = candidates
        else:
            validated = validate_candidates_batch(
                candidates,
                model=OLLAMA_MODEL,
                max_workers=args.validation_workers,
                max_validate=args.max_validate
            )
    else:
        print("\n  Skipping LLM validation (--skip-validation)")
        validated = candidates
    
    # Save candidates
    with open(CANDIDATES_FILE, "w", encoding="utf-8") as f:
        for cand in validated:
            f.write(json.dumps(cand, ensure_ascii=False) + "\n")
    print(f"\n✓ Saved candidates to: {CANDIDATES_FILE}")
    
    # Phase 7: Sampling
    print("\n" + "=" * 60)
    print("  Phase 7: Sampling 200Q")
    print("=" * 60)
    
    if len(validated) == 0:
        print("ERROR: No valid candidates after validation")
        return
    
    # For now, random sample 200 (will implement stratified sampling later)
    n_sample = min(200, len(validated))
    sampled = random.sample(validated, n_sample)
    
    print(f"  Sampled {len(sampled)} questions (seed={args.seed})")
    
    # Convert to testset format
    testset = []
    for idx, cand in enumerate(sampled):
        testset.append({
            "idx": idx,
            "question": cand["question"],
            "answer": "",  # Will be generated by RAG
            "source": "multihop_generated",
            "concept_A": cand["concept_A"],
            "concept_B": cand["concept_B"],
            "concept_C": cand["concept_C"],
            "hop_count": cand["hop_count"],
            "template_id": cand["template_id"],
            "rel_types": [cand["rel_type_A"], cand["rel_type_B"]],
        })
    
    # Save
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for rec in testset:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    
    print(f"\n✓ Saved testset to: {OUTPUT_FILE}")
    
    # Summary
    print("\n" + "=" * 60)
    print("  Summary")
    print("=" * 60)
    print(f"  Total candidates:  {len(candidates)}")
    print(f"  Validated:         {len(validated)}")
    print(f"  Final testset:     {len(testset)}")
    
    # Distribution
    hop_counts = Counter(q["hop_count"] for q in testset)
    print(f"\n  Hop count distribution:")
    for hop, count in sorted(hop_counts.items()):
        print(f"    {hop}-hop: {count} ({count/len(testset)*100:.1f}%)")
    
    template_counts = Counter(q["template_id"] for q in testset)
    print(f"\n  Template distribution:")
    for tid, count in sorted(template_counts.items()):
        print(f"    T{tid+1}: {count} ({count/len(testset)*100:.1f}%)")
    
    print("\n" + "=" * 60)
    print("  ✓ Multi-hop testset generation complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
