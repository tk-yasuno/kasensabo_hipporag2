#!/usr/bin/env python3
"""
06_train_calibration.py

Train hierarchical calibration models for v0.5 RAG optimization.

This script trains three independent regression models (Volume, Chapter, Chunk)
to predict Judge scores from retrieval features. The models enable data-driven
optimization of retrieval scoring through supervised learning.

Usage:
    python experiments/06_train_calibration.py \\
        --volume-features experiments/results/qwen257b_hipporag2_volume_features.jsonl \\
        --chapter-features experiments/results/qwen257b_hipporag2_chapter_features.jsonl \\
        --chunk-features experiments/results/qwen257b_hipporag2_chunk_features.jsonl \\
        --output-dir experiments/calibration_models \\
        --test-split 0.2

Output:
    - experiments/calibration_models/volume_model.pkl
    - experiments/calibration_models/chapter_model.pkl
    - experiments/calibration_models/chunk_model.pkl
    - experiments/calibration_models/training_report.json
"""

import argparse
import json
import pickle
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy.stats import pearsonr

try:
    from lightgbm import LGBMRegressor, LGBMRanker
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False
    LGBMRanker = None
    print("[WARN] LightGBM not available. Install with: pip install lightgbm")


def load_features(jsonl_path: str) -> List[Dict]:
    """Load feature JSONL file."""
    features = []
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                features.append(json.loads(line))
    return features


def prepare_volume_data(features: List[Dict]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Extract Volume feature matrix and target Judge scores.
    
    Features: [emb_score, kw_score, triple_score, fused_score, selected]
    Target: judge_score
    Returns: (X, y, question_ids)
    """
    X = []
    y = []
    qids = []
    
    for feat in features:
        x_row = [
            feat.get('emb_score', 0.0),
            feat.get('kw_score', 0.0),
            feat.get('triple_score', 0.0),
            feat.get('fused_score', 0.0),
            float(feat.get('selected', False))  # Boolean to 0/1
        ]
        X.append(x_row)
        y.append(feat.get('judge_score', 0))
        qids.append(feat.get('question_id', 0))
    
    return np.array(X), np.array(y), np.array(qids)


def prepare_chapter_data(features: List[Dict]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Extract Chapter feature matrix and target Judge scores.
    
    Features: [emb_score, triple_score, fused_score, selected]
    Target: judge_score
    Returns: (X, y, question_ids)
    """
    X = []
    y = []
    qids = []
    
    for feat in features:
        x_row = [
            feat.get('emb_score', 0.0),
            feat.get('triple_score', 0.0),
            feat.get('fused_score', 0.0),
            float(feat.get('selected', False))
        ]
        X.append(x_row)
        y.append(feat.get('judge_score', 0))
        qids.append(feat.get('question_id', 0))
    
    return np.array(X), np.array(y), np.array(qids)


def prepare_chunk_data(features: List[Dict]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Extract Chunk feature matrix and target Judge scores.
    
    Features: [embedding_sim, chunk_length]
    Target: judge_score
    Returns: (X, y, question_ids)
    """
    X = []
    y = []
    qids = []
    
    for feat in features:
        x_row = [
            feat.get('embedding_sim', 0.0),
            feat.get('chunk_length', 0)
        ]
        X.append(x_row)
        y.append(feat.get('judge_score', 0))
        qids.append(feat.get('question_id', 0))
    
    return np.array(X), np.array(y), np.array(qids)


def train_and_evaluate(
    X: np.ndarray,
    y: np.ndarray,
    test_size: float,
    random_state: int = 42
) -> Tuple[LinearRegression, Dict]:
    """
    Train LinearRegression model and evaluate on test set.
    
    Returns:
        model: Trained LinearRegression model
        metrics: Dict with MAE, RMSE, R², Pearson correlation
    """
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    # Train model
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # Predictions
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    
    # Clip predictions to [0, 3] range
    y_pred_train = np.clip(y_pred_train, 0, 3)
    y_pred_test = np.clip(y_pred_test, 0, 3)
    
    # Metrics
    mae_train = mean_absolute_error(y_train, y_pred_train)
    mae_test = mean_absolute_error(y_test, y_pred_test)
    
    rmse_train = np.sqrt(mean_squared_error(y_train, y_pred_train))
    rmse_test = np.sqrt(mean_squared_error(y_test, y_pred_test))
    
    r2_train = r2_score(y_train, y_pred_train)
    r2_test = r2_score(y_test, y_pred_test)
    
    corr_train, _ = pearsonr(y_train, y_pred_train)
    corr_test, _ = pearsonr(y_test, y_pred_test)
    
    metrics = {
        'train': {
            'mae': float(mae_train),
            'rmse': float(rmse_train),
            'r2': float(r2_train),
            'pearson': float(corr_train),
            'n_samples': len(y_train)
        },
        'test': {
            'mae': float(mae_test),
            'rmse': float(rmse_test),
            'r2': float(r2_test),
            'pearson': float(corr_test),
            'n_samples': len(y_test)
        },
        'coefficients': model.coef_.tolist(),
        'intercept': float(model.intercept_)
    }
    
    return model, metrics


def print_metrics(tier_name: str, metrics: Dict):
    """Pretty-print training metrics."""
    print(f"\n{'='*60}")
    print(f"  {tier_name} Model Evaluation")
    print(f"{'='*60}")
    print(f"Training Set (n={metrics['train']['n_samples']}):")
    print(f"  MAE:     {metrics['train']['mae']:.4f}")
    print(f"  RMSE:    {metrics['train']['rmse']:.4f}")
    print(f"  R²:      {metrics['train']['r2']:.4f}")
    print(f"  Pearson: {metrics['train']['pearson']:.4f}")
    print(f"\nTest Set (n={metrics['test']['n_samples']}):")
    print(f"  MAE:     {metrics['test']['mae']:.4f}")
    print(f"  RMSE:    {metrics['test']['rmse']:.4f}")
    print(f"  R²:      {metrics['test']['r2']:.4f}")
    print(f"  Pearson: {metrics['test']['pearson']:.4f}")
    print(f"\nModel Parameters:")
    print(f"  Intercept:    {metrics['intercept']:.4f}")
    print(f"  Coefficients: {[f'{c:.4f}' for c in metrics['coefficients']]}")


def compute_ndcg_at_k(y_true: np.ndarray, y_pred: np.ndarray, k: int = 5) -> float:
    """
    Compute NDCG@k for a single query.
    
    NDCG (Normalized Discounted Cumulative Gain) measures ranking quality.
    Higher is better, range [0, 1].
    """
    # Limit k to available samples
    k = min(k, len(y_true))
    if k == 0:
        return 0.0
    
    # Sort by predicted scores (descending)
    order = np.argsort(y_pred)[::-1][:k]
    y_true_sorted = y_true[order]
    
    # DCG@k
    dcg = np.sum((2 ** y_true_sorted - 1) / np.log2(np.arange(2, k + 2)))
    
    # IDCG@k (ideal DCG with perfect ranking)
    ideal_order = np.argsort(y_true)[::-1][:k]
    y_true_ideal = y_true[ideal_order]
    idcg = np.sum((2 ** y_true_ideal - 1) / np.log2(np.arange(2, k + 2)))
    
    if idcg == 0:
        return 0.0
    return dcg / idcg


def train_and_evaluate_ranker_v6(
    X: np.ndarray,
    y: np.ndarray,
    qids: np.ndarray,
    test_size: float,
    random_state: int = 42
) -> Tuple[object, Dict]:
    """
    Train LGBMRanker (LambdaMART) and evaluate with NDCG@k.
    
    Args:
        X: Feature matrix
        y: Judge scores (0-3)
        qids: Question IDs for grouping
        test_size: Test split ratio
        random_state: Random seed
    
    Returns:
        model: Trained LGBMRanker
        metrics: Dict with MAE, RMSE, R², Pearson, NDCG@5, NDCG@10
    """
    if not LIGHTGBM_AVAILABLE or LGBMRanker is None:
        raise ImportError("LightGBM with Ranker not available. Install with: pip install lightgbm")
    
    # Split data by question ID to avoid leaking queries across train/test
    unique_qids = np.unique(qids)
    train_qids, test_qids = train_test_split(
        unique_qids, test_size=test_size, random_state=random_state
    )
    
    train_mask = np.isin(qids, train_qids)
    test_mask = np.isin(qids, test_qids)
    
    X_train, y_train, qids_train = X[train_mask], y[train_mask], qids[train_mask]
    X_test, y_test, qids_test = X[test_mask], y[test_mask], qids[test_mask]
    
    # Compute group sizes (number of samples per query)
    train_groups = [np.sum(qids_train == qid) for qid in np.unique(qids_train)]
    test_groups = [np.sum(qids_test == qid) for qid in np.unique(qids_test)]
    
    # Train LGBMRanker with LambdaMART objective
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
        random_state=random_state,
        verbose=-1
    )
    
    model.fit(
        X_train, y_train,
        group=train_groups,
        eval_set=[(X_test, y_test)],
        eval_group=[test_groups],
        eval_metric='ndcg'
    )
    
    # Predictions
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    
    # Clip predictions to [0, 3] range
    y_pred_train = np.clip(y_pred_train, 0, 3)
    y_pred_test = np.clip(y_pred_test, 0, 3)
    
    # Regression metrics (for comparison with regressor)
    mae_train = mean_absolute_error(y_train, y_pred_train)
    mae_test = mean_absolute_error(y_test, y_pred_test)
    
    rmse_train = np.sqrt(mean_squared_error(y_train, y_pred_train))
    rmse_test = np.sqrt(mean_squared_error(y_test, y_pred_test))
    
    r2_train = r2_score(y_train, y_pred_train)
    r2_test = r2_score(y_test, y_pred_test)
    
    corr_train, _ = pearsonr(y_train, y_pred_train)
    corr_test, _ = pearsonr(y_test, y_pred_test)
    
    # Ranking metrics (NDCG@k)
    ndcg5_train_list = []
    ndcg10_train_list = []
    for qid in np.unique(qids_train):
        mask = qids_train == qid
        ndcg5_train_list.append(compute_ndcg_at_k(y_train[mask], y_pred_train[mask], k=5))
        ndcg10_train_list.append(compute_ndcg_at_k(y_train[mask], y_pred_train[mask], k=10))
    
    ndcg5_test_list = []
    ndcg10_test_list = []
    for qid in np.unique(qids_test):
        mask = qids_test == qid
        ndcg5_test_list.append(compute_ndcg_at_k(y_test[mask], y_pred_test[mask], k=5))
        ndcg10_test_list.append(compute_ndcg_at_k(y_test[mask], y_pred_test[mask], k=10))
    
    metrics = {
        'model_type': 'ranker',
        'train': {
            'mae': float(mae_train),
            'rmse': float(rmse_train),
            'r2': float(r2_train),
            'pearson': float(corr_train),
            'ndcg@5': float(np.mean(ndcg5_train_list)),
            'ndcg@10': float(np.mean(ndcg10_train_list)),
            'n_samples': len(y_train),
            'n_queries': len(np.unique(qids_train))
        },
        'test': {
            'mae': float(mae_test),
            'rmse': float(rmse_test),
            'r2': float(r2_test),
            'pearson': float(corr_test),
            'ndcg@5': float(np.mean(ndcg5_test_list)),
            'ndcg@10': float(np.mean(ndcg10_test_list)),
            'n_samples': len(y_test),
            'n_queries': len(np.unique(qids_test))
        },
        'feature_importance': model.feature_importances_.tolist(),
        'n_estimators': model.n_estimators,
        'learning_rate': model.learning_rate
    }
    
    return model, metrics


def train_and_evaluate_v6(
    X: np.ndarray,
    y: np.ndarray,
    test_size: float,
    model_type: str = "lgbm",
    random_state: int = 42
) -> Tuple[object, Dict]:
    """
    Train calibration model (LightGBM or LinearRegression) and evaluate.
    
    Args:
        model_type: "lgbm" (LGBMRegressor, v0.6) or "linear" (LinearRegression, v0.5)
    
    Returns:
        model: Trained model
        metrics: Dict with MAE, RMSE, R², Pearson correlation
    """
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    # Initialize model
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
            random_state=random_state,
            verbose=-1
        )
    elif model_type == "linear":
        model = LinearRegression()
    else:
        raise ValueError(f"Unknown model_type: {model_type}")
    
    # Train model
    model.fit(X_train, y_train)
    
    # Predictions
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    
    # Clip predictions to [0, 3] range
    y_pred_train = np.clip(y_pred_train, 0, 3)
    y_pred_test = np.clip(y_pred_test, 0, 3)
    
    # Metrics
    mae_train = mean_absolute_error(y_train, y_pred_train)
    mae_test = mean_absolute_error(y_test, y_pred_test)
    
    rmse_train = np.sqrt(mean_squared_error(y_train, y_pred_train))
    rmse_test = np.sqrt(mean_squared_error(y_test, y_pred_test))
    
    r2_train = r2_score(y_train, y_pred_train)
    r2_test = r2_score(y_test, y_pred_test)
    
    corr_train, _ = pearsonr(y_train, y_pred_train)
    corr_test, _ = pearsonr(y_test, y_pred_test)
    
    metrics = {
        'model_type': model_type,
        'train': {
            'mae': float(mae_train),
            'rmse': float(rmse_train),
            'r2': float(r2_train),
            'pearson': float(corr_train),
            'n_samples': len(y_train)
        },
        'test': {
            'mae': float(mae_test),
            'rmse': float(rmse_test),
            'r2': float(r2_test),
            'pearson': float(corr_test),
            'n_samples': len(y_test)
        }
    }
    
    # Add model-specific parameters
    if model_type == "linear":
        metrics['coefficients'] = model.coef_.tolist()
        metrics['intercept'] = float(model.intercept_)
    elif model_type == "lgbm":
        metrics['feature_importance'] = model.feature_importances_.tolist()
        metrics['n_estimators'] = model.n_estimators
        metrics['learning_rate'] = model.learning_rate
    
    return model, metrics


def print_metrics_v6(tier_name: str, metrics: Dict):
    """Pretty-print training metrics for v0.6."""
    model_type = metrics.get('model_type', 'unknown')
    print(f"\n{'='*60}")
    print(f"  {tier_name} Model Evaluation ({model_type.upper()})")
    print(f"{'='*60}")
    
    # Training metrics
    train_info = f"Training Set (n={metrics['train']['n_samples']}"
    if 'n_queries' in metrics['train']:
        train_info += f", queries={metrics['train']['n_queries']}"
    train_info += "):"
    print(train_info)
    print(f"  MAE:     {metrics['train']['mae']:.4f}")
    print(f"  RMSE:    {metrics['train']['rmse']:.4f}")
    print(f"  R²:      {metrics['train']['r2']:.4f}")
    print(f"  Pearson: {metrics['train']['pearson']:.4f}")
    if 'ndcg@5' in metrics['train']:
        print(f"  NDCG@5:  {metrics['train']['ndcg@5']:.4f}")
        print(f"  NDCG@10: {metrics['train']['ndcg@10']:.4f}")
    
    # Test metrics
    test_info = f"\nTest Set (n={metrics['test']['n_samples']}"
    if 'n_queries' in metrics['test']:
        test_info += f", queries={metrics['test']['n_queries']}"
    test_info += "):"
    print(test_info)
    print(f"  MAE:     {metrics['test']['mae']:.4f}")
    print(f"  RMSE:    {metrics['test']['rmse']:.4f}")
    print(f"  R²:      {metrics['test']['r2']:.4f}")
    print(f"  Pearson: {metrics['test']['pearson']:.4f}")
    if 'ndcg@5' in metrics['test']:
        print(f"  NDCG@5:  {metrics['test']['ndcg@5']:.4f}")
        print(f"  NDCG@10: {metrics['test']['ndcg@10']:.4f}")
    
    # Model-specific parameters
    if model_type == "linear":
        print(f"\nLinear Model Parameters:")
        print(f"  Intercept:    {metrics['intercept']:.4f}")
        print(f"  Coefficients: {[f'{c:.4f}' for c in metrics['coefficients']]}")
    elif model_type in ["lgbm", "ranker"]:
        print(f"\nLightGBM Model Parameters:")
        print(f"  n_estimators:   {metrics['n_estimators']}")
        print(f"  learning_rate:  {metrics['learning_rate']}")
        print(f"  Feature importance: {[f'{f:.4f}' for f in metrics['feature_importance']]}")


def main():
    parser = argparse.ArgumentParser(
        description="Train hierarchical calibration models for RAG optimization"
    )
    parser.add_argument(
        '--volume-features',
        type=str,
        default='experiments/results/qwen257b_hipporag2_volume_features.jsonl',
        help='Path to volume features JSONL'
    )
    parser.add_argument(
        '--chapter-features',
        type=str,
        default='experiments/results/qwen257b_hipporag2_chapter_features.jsonl',
        help='Path to chapter features JSONL'
    )
    parser.add_argument(
        '--chunk-features',
        type=str,
        default='experiments/results/qwen257b_hipporag2_chunk_features.jsonl',
        help='Path to chunk features JSONL'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='experiments/calibration_models',
        help='Directory to save trained models'
    )
    parser.add_argument(
        '--test-split',
        type=float,
        default=0.2,
        help='Test set fraction (default: 0.2)'
    )
    parser.add_argument(
        '--random-seed',
        type=int,
        default=42,
        help='Random seed for reproducibility'
    )
    parser.add_argument(
        '--model-type',
        type=str,
        default='lgbm',
        choices=['linear', 'lgbm', 'ranker'],
        help='Model type: linear (v0.5, LinearRegression), lgbm (v0.6, LightGBM Regressor), or ranker (v0.6, LambdaMART)'
    )
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*60)
    print(f"  v0.6 Hierarchical Calibration Model Training ({args.model_type.upper()})")
    print("="*60)
    print(f"Model type:       {args.model_type}")
    print(f"Volume features:  {args.volume_features}")
    print(f"Chapter features: {args.chapter_features}")
    print(f"Chunk features:   {args.chunk_features}")
    print(f"Output directory: {args.output_dir}")
    print(f"Test split:       {args.test_split:.1%}")
    print()
    
    # Load features
    print("Loading features...")
    volume_features = load_features(args.volume_features)
    chapter_features = load_features(args.chapter_features)
    chunk_features = load_features(args.chunk_features)
    print(f"  Volume:  {len(volume_features)} entries")
    print(f"  Chapter: {len(chapter_features)} entries")
    print(f"  Chunk:   {len(chunk_features)} entries")
    
    # Prepare data
    print("\nPreparing feature matrices...")
    X_vol, y_vol, qids_vol = prepare_volume_data(volume_features)
    X_chap, y_chap, qids_chap = prepare_chapter_data(chapter_features)
    X_chunk, y_chunk, qids_chunk = prepare_chunk_data(chunk_features)
    print(f"  Volume:  X={X_vol.shape}, y={y_vol.shape}")
    print(f"  Chapter: X={X_chap.shape}, y={y_chap.shape}")
    print(f"  Chunk:   X={X_chunk.shape}, y={y_chunk.shape}")
    
    # Train Volume model
    print("\n[1/3] Training Volume model...")
    if args.model_type == 'ranker':
        vol_model, vol_metrics = train_and_evaluate_ranker_v6(
            X_vol, y_vol, qids_vol, args.test_split, random_state=args.random_seed
        )
    else:
        vol_model, vol_metrics = train_and_evaluate_v6(
            X_vol, y_vol, args.test_split, model_type=args.model_type, random_state=args.random_seed
        )
    print_metrics_v6("VOLUME", vol_metrics)
    
    # Train Chapter model
    print("\n[2/3] Training Chapter model...")
    if args.model_type == 'ranker':
        chap_model, chap_metrics = train_and_evaluate_ranker_v6(
            X_chap, y_chap, qids_chap, args.test_split, random_state=args.random_seed
        )
    else:
        chap_model, chap_metrics = train_and_evaluate_v6(
            X_chap, y_chap, args.test_split, model_type=args.model_type, random_state=args.random_seed
        )
    print_metrics_v6("CHAPTER", chap_metrics)
    
    # Train Chunk model
    print("\n[3/3] Training Chunk model...")
    if args.model_type == 'ranker':
        chunk_model, chunk_metrics = train_and_evaluate_ranker_v6(
            X_chunk, y_chunk, qids_chunk, args.test_split, random_state=args.random_seed
        )
    else:
        chunk_model, chunk_metrics = train_and_evaluate_v6(
            X_chunk, y_chunk, args.test_split, model_type=args.model_type, random_state=args.random_seed
        )
    print_metrics_v6("CHUNK", chunk_metrics)
    
    # Save models
    print("\n" + "="*60)
    print("Saving models...")
    
    # Model file prefix based on type
    if args.model_type == "lgbm":
        prefix = "lgbm_"
    elif args.model_type == "ranker":
        prefix = "ranker_"
    else:
        prefix = ""
    
    vol_path = output_dir / f"{prefix}volume_model.pkl"
    with open(vol_path, 'wb') as f:
        pickle.dump(vol_model, f)
    print(f"  ✓ {vol_path}")
    
    chap_path = output_dir / f"{prefix}chapter_model.pkl"
    with open(chap_path, 'wb') as f:
        pickle.dump(chap_model, f)
    print(f"  ✓ {chap_path}")
    
    chunk_path = output_dir / f"{prefix}chunk_model.pkl"
    with open(chunk_path, 'wb') as f:
        pickle.dump(chunk_model, f)
    print(f"  ✓ {chunk_path}")
    
    # Save training report
    report = {
        'volume': vol_metrics,
        'chapter': chap_metrics,
        'chunk': chunk_metrics,
        'config': {
            'model_type': args.model_type,
            'test_split': args.test_split,
            'random_seed': args.random_seed,
            'volume_features_path': args.volume_features,
            'chapter_features_path': args.chapter_features,
            'chunk_features_path': args.chunk_features
        }
    }
    
    report_path = output_dir / f"{prefix}training_report.json"
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2)
    print(f"  ✓ {report_path}")
    
    print("\n" + "="*60)
    print("Training complete!")
    print("="*60)
    print("\nNext steps:")
    print("  1. Integrate models into 03_rag_retrievers.py")
    print("  2. Add --use-calibration flag to 04_eval_rag.py")
    print("  3. Run full evaluation with calibrated scoring")
    print()


if __name__ == "__main__":
    main()
