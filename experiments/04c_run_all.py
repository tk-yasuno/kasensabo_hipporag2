#!/usr/bin/env python
"""
experiments/04c_run_all.py
──────────────────────────────────────────────────────────
Phase 4c: 全6条件（3 RAG方式 × 2モデル）を順次実行

Usage:
  python experiments/04c_run_all.py --batch-size 8
  python experiments/04c_run_all.py --dry-run
"""
import os
import sys
import time
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT)

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--no-judge", action="store_true", help="Judge採点をスキップ（推論のみ）")
    parser.add_argument("--models", default="swallow,elyza")
    parser.add_argument("--rags", default="naive,light,hipporag2")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--judge-workers", type=int, default=4)
    args = parser.parse_args()

    model_list = args.models.split(",")
    rag_list = args.rags.split(",")
    
    total = len(model_list) * len(rag_list)
    current = 0
    failed = []
    start_time = time.time()

    print("=" * 60)
    print(f"  RAG comparison - Run all conditions")
    print(f"  Conditions: {total}  DryRun: {args.dry_run}  NoJudge: {args.no_judge}")
    print("=" * 60)

    for model in model_list:
        for rag in rag_list:
            current += 1
            cond = f"{model}_{rag}"
            print()
            print("-" * 60)
            print(f"  [{current}/{total}] {cond}")
            print("-" * 60)

            cmd = [
                sys.executable,
                "experiments/04_eval_rag.py",
                "--model", model,
                "--rag", rag,
                "--batch-size", str(args.batch_size),
                "--judge-workers", str(args.judge_workers)
            ]
            if args.dry_run:
                cmd.append("--dry-run")
            if args.no_judge:
                cmd.append("--no-judge")

            t0 = time.time()
            result = subprocess.run(cmd)
            elapsed = int(time.time() - t0)

            if result.returncode != 0:
                print(f"  [FAILED] {cond}  (exit={result.returncode})")
                failed.append(cond)
            else:
                print(f"  [OK] {cond}  ({elapsed} s)")

    total_elapsed = round((time.time() - start_time) / 60, 1)
    print()
    print("=" * 60)
    print(f"  Done: {total} conditions  Failed: {len(failed)}  Time: {total_elapsed} min")
    if failed:
        print(f"  Failed: {', '.join(failed)}")
    print("=" * 60)

    sys.exit(1 if failed else 0)

if __name__ == "__main__":
    main()
