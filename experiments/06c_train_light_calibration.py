#!/usr/bin/env python3
"""
Train Light RAG calibration model (v0.5 Phase 3)

Usage:
    python experiments/06c_train_light_calibration.py

Input files:
    experiments/results/qwen257b_light_chunk_features.jsonl

Output files:
    experiments/calibration_models/light_chunk_model.pkl
    experiments/calibration_models/light_training_report.json
"""

import json
import argparse
import numpy as np
from pathlib import Path
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy.stats import pearsonr
import pickle

try:
    from lightgbm import LGBMRegressor, LGBMRanker
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False
    LGBMRanker = None
    print("[WARN] LightGBM not available. Install with: pip install lightgbm")


def load_chunk_features(chunk_features_file: Path):
    """Load chunk features from JSONL file"""
    chunks = []
    with open(chunk_features_file, "r", encoding="utf-8") as f:
        for line in f:
            chunks.append(json.loads(line))
    return chunks


def prepare_chunk_data(chunks):
    """Prepare training data for Light RAG: X = [embedding_sim, bm25_score, fused_score, chunk_length], y = judge_score, qids = question_id"""
    X = []
    y = []
    qids = []
    
    for chunk in chunks:
        embedding_sim = chunk.get("embedding_sim", 0.0)
        bm25_score = chunk.get("bm25_score", 0.0)
        fused_score = chunk.get("fused_score", 0.0)
        chunk_length = chunk.get("chunk_length", 0)
        judge_score = chunk.get("judge_score", 0)
        question_id = chunk.get("question_id", 0)
        
        X.append([embedding_sim, bm25_score, fused_score, chunk_length])
        y.append(judge_score)
        qids.append(question_id)
    
    return np.array(X), np.array(y), np.array(qids)


def compute_ndcg_at_k(y_true: np.ndarray, y_pred: np.ndarray, k: int = 5) -> float:
    """Compute NDCG@k for a single query"""
    k = min(k, len(y_true))
    if k == 0:
        return 0.0
    order = np.argsort(y_pred)[::-1][:k]
    y_true_sorted = y_true[order]
    dcg = np.sum((2 ** y_true_sorted - 1) / np.log2(np.arange(2, k + 2)))
    ideal_order = np.argsort(y_true)[::-1][:k]
    y_true_ideal = y_true[ideal_order]
    idcg = np.sum((2 ** y_true_ideal - 1) / np.log2(np.arange(2, k + 2)))
    if idcg == 0:
        return 0.0
    return dcg / idcg


def train_and_evaluate_ranker(X, y, qids, model_name: str):
    """Train LGBMRanker (LambdaMART) and evaluate"""
    if not LIGHTGBM_AVAILABLE or LGBMRanker is None:
        raise ImportError("LightGBM with Ranker not available. Install with: pip install lightgbm")
    
    # Split by question ID
    unique_qids = np.unique(qids)
    train_qids, test_qids = train_test_split(unique_qids, test_size=0.2, random_state=42)
    
    train_mask = np.isin(qids, train_qids)
    test_mask = np.isin(qids, test_qids)
    
    X_train, y_train, qids_train = X[train_mask], y[train_mask], qids[train_mask]
    X_test, y_test, qids_test = X[test_mask], y[test_mask], qids[test_mask]
    
    train_groups = [np.sum(qids_train == qid) for qid in np.unique(qids_train)]
    test_groups = [np.sum(qids_test == qid) for qid in np.unique(qids_test)]
    
    model = LGBMRanker(
        objective='lambdarank',
        metric='ndcg',
        n_estimators=100,
        learning_rate=0.05,
        max_depth=5,
        num_leaves=31,
        min_child_samples=5,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        verbose=-1
    )
    
    model.fit(X_train, y_train, group=train_groups)
    
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    y_pred_train = np.clip(y_pred_train, 0, 3)
    y_pred_test = np.clip(y_pred_test, 0, 3)
    
    mae_train = mean_absolute_error(y_train, y_pred_train)
    mae_test = mean_absolute_error(y_test, y_pred_test)
    rmse_train = np.sqrt(mean_squared_error(y_train, y_pred_train))
    rmse_test = np.sqrt(mean_squared_error(y_test, y_pred_test))
    r2_train = r2_score(y_train, y_pred_train)
    r2_test = r2_score(y_test, y_pred_test)
    pearson_train, _ = pearsonr(y_train, y_pred_train)
    pearson_test, _ = pearsonr(y_test, y_pred_test)
    
    ndcg5_train = np.mean([compute_ndcg_at_k(y_train[qids_train == qid], y_pred_train[qids_train == qid], 5) for qid in np.unique(qids_train)])
    ndcg10_train = np.mean([compute_ndcg_at_k(y_train[qids_train == qid], y_pred_train[qids_train == qid], 10) for qid in np.unique(qids_train)])
    ndcg5_test = np.mean([compute_ndcg_at_k(y_test[qids_test == qid], y_pred_test[qids_test == qid], 5) for qid in np.unique(qids_test)])
    ndcg10_test = np.mean([compute_ndcg_at_k(y_test[qids_test == qid], y_pred_test[qids_test == qid], 10) for qid in np.unique(qids_test)])
    
    metrics = {
        "model": model_name,
        "model_type": "ranker",
        "n_train": len(X_train),
        "n_test": len(X_test),
        "n_train_queries": len(np.unique(qids_train)),
        "n_test_queries": len(np.unique(qids_test)),
        "mae_train": float(mae_train),
        "mae_test": float(mae_test),
        "rmse_train": float(rmse_train),
        "rmse_test": float(rmse_test),
        "r2_train": float(r2_train),
        "r2_test": float(r2_test),
        "pearson_train": float(pearson_train),
        "pearson_test": float(pearson_test),
        "ndcg5_train": float(ndcg5_train),
        "ndcg10_train": float(ndcg10_train),
        "ndcg5_test": float(ndcg5_test),
        "ndcg10_test": float(ndcg10_test),
        "feature_importance": model.feature_importances_.tolist(),
        "n_estimators": model.n_estimators,
    }
    
    return model, metrics


def train_and_evaluate(X, y, model_name: str, model_type: str = "lgbm"):
    """Train calibration model (LightGBM or LinearRegression) and evaluate"""
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    if model_type == "lgbm":
        if not LIGHTGBM_AVAILABLE:
            raise ImportError("LightGBM not available. Install with: pip install lightgbm")
        model = LGBMRegressor(
            n_estimators=100,
            learning_rate=0.05,
            max_depth=5,
            num_leaves=31,
            min_child_samples=20,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            verbose=-1
        )
    else:
        model = LinearRegression()
    
    model.fit(X_train, y_train)
    
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    
    mae_train = mean_absolute_error(y_train, y_pred_train)
    mae_test = mean_absolute_error(y_test, y_pred_test)
    rmse_train = np.sqrt(mean_squared_error(y_train, y_pred_train))
    rmse_test = np.sqrt(mean_squared_error(y_test, y_pred_test))
    r2_train = r2_score(y_train, y_pred_train)
    r2_test = r2_score(y_test, y_pred_test)
    
    # Pearson correlation
    pearson_train, _ = pearsonr(y_train, y_pred_train)
    pearson_test, _ = pearsonr(y_test, y_pred_test)
    
    metrics = {
        "model": model_name,
        "model_type": model_type,
        "n_train": len(X_train),
        "n_test": len(X_test),
        "mae_train": float(mae_train),
        "mae_test": float(mae_test),
        "rmse_train": float(rmse_train),
        "rmse_test": float(rmse_test),
        "r2_train": float(r2_train),
        "r2_test": float(r2_test),
        "pearson_train": float(pearson_train),
        "pearson_test": float(pearson_test),
    }
    
    if model_type == "linear":
        metrics["coefficients"] = model.coef_.tolist()
        metrics["intercept"] = float(model.intercept_)
    elif model_type == "lgbm":
        metrics["feature_importance"] = model.feature_importances_.tolist()
        metrics["n_estimators"] = model.n_estimators
    
    return model, metrics


def print_metrics(metrics: dict):
    """Pretty-print training metrics"""
    model_type = metrics.get('model_type', 'linear').upper()
    info_str = f"\n{metrics['model'].upper()} ({model_type}): n_train={metrics['n_train']}, n_test={metrics['n_test']}"
    if 'n_train_queries' in metrics:
        info_str += f", queries={metrics['n_test_queries']}"
    info_str += f", MAE={metrics['mae_test']:.2f}, RMSE={metrics['rmse_test']:.2f}, R²={metrics['r2_test']:.2f}, Pearson={metrics['pearson_test']:.2f}"
    print(info_str)
    
    if metrics['model_type'] == 'ranker':
        print(f"  NDCG@5={metrics['ndcg5_test']:.4f}, NDCG@10={metrics['ndcg10_test']:.4f}")
        print(f"  Feature importance: {[f'{f:.2f}' for f in metrics['feature_importance']]}")
    elif metrics['model_type'] == 'linear':
        print(f"  Coefficients: {[f'{c:.2f}' for c in metrics['coefficients']]}, Intercept: {metrics['intercept']:.2f}")
    elif metrics['model_type'] == 'lgbm':
        print(f"  Feature importance: {[f'{f:.2f}' for f in metrics['feature_importance']]}")


def main():
    parser = argparse.ArgumentParser(description="Train Light RAG calibration model")
    parser.add_argument("--chunk-features", type=str,
                        default="experiments/results/qwen257b_light_chunk_features.jsonl",
                        help="Path to chunk features JSONL file")
    parser.add_argument("--output-dir", type=str,
                        default="experiments/calibration_models",
                        help="Directory to save trained models")
    parser.add_argument("--model-type", type=str,
                        default="lgbm",
                        choices=["linear", "lgbm", "ranker"],
                        help="Model type: linear (v0.5), lgbm (v0.6, LightGBM Regressor), or ranker (v0.6, LambdaMART)")
    args = parser.parse_args()
    
    chunk_features_file = Path(args.chunk_features)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print(f"Light RAG Calibration Model Training (v0.6, {args.model_type.upper()})")
    print("=" * 60)
    
    # Load features
    print(f"\nLoading chunk features from {chunk_features_file}...")
    chunks = load_chunk_features(chunk_features_file)
    print(f"  Loaded {len(chunks)} chunk features")
    
    # Prepare data
    print("\n[1/1] Training Chunk model...")
    X_chunk, y_chunk, qids_chunk = prepare_chunk_data(chunks)
    if args.model_type == "ranker":
        chunk_model, chunk_metrics = train_and_evaluate_ranker(X_chunk, y_chunk, qids_chunk, "Chunk")
    else:
        chunk_model, chunk_metrics = train_and_evaluate(X_chunk, y_chunk, "Chunk", model_type=args.model_type)
    print_metrics(chunk_metrics)
    
    # Save models
    if args.model_type == "lgbm":
        prefix = "lgbm_"
    elif args.model_type == "ranker":
        prefix = "ranker_"
    else:
        prefix = ""
    chunk_model_path = output_dir / f"{prefix}light_chunk_model.pkl"
    with open(chunk_model_path, "wb") as f:
        pickle.dump(chunk_model, f)
    print(f"\n  Saved: {chunk_model_path}")
    
    # Save training report
    report = {
        "chunk": chunk_metrics,
        "config": {
            "model_type": args.model_type
        }
    }
    report_path = output_dir / f"{prefix}light_training_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"  Saved: {report_path}")
    
    print("\n" + "=" * 60)
    print("Training complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
