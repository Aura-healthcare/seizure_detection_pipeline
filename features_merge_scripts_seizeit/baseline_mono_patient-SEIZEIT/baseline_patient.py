"""
baseline_patient.py — Per-patient baseline.

Trains the full embedding + classifier pipeline (same architecture/losses as
launch_train.py) separately for EACH patient, using a chronological rotation
split: the patient's rows (already in chronological order in the CSV — sorted
by run then elapsed_s) are cut into `n_folds` contiguous intervals. Each fold
uses 3 of the 4 intervals as train and the remaining one as test, rotating so
every interval is used as test exactly once (4-fold "block" cross-validation).

This is a within-patient baseline: train and test come from the same patient,
so it measures a much easier setting than the cross-patient / held-out-patient
evaluation used by launch_train.py. Useful as a sanity ceiling.

Usage:
    python baseline_patient.py
    python baseline_patient.py --patients sub-001 sub-002 --n-folds 4
    python baseline_patient.py --embedding-epochs 2 --classifier-epochs 2   # quick smoke test
"""

import argparse
import json
import logging
import os
import sys
from datetime import datetime

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from dataset import ContrastiveDataset, TripletDataset, EmbeddingDataset, PKBatchSampler
from eval import eval_classifier_head, generate_umap_visualizations
from launch_train import train_embedding_model, train_classifier_head, save_config_to_json
from config import (
    DATA_CONFIG,
    MODEL_CONFIG,
    EMBEDDING_TRAINING_CONFIG,
    CLASSIFIER_TRAINING_CONFIG,
    EVAL_CONFIG,
    DEVICE_CONFIG,
    LOGGING_CONFIG,
    PATIENT_BASELINE_CONFIG,
)


# ============================================================================
# DATA SPLITTING / PREPROCESSING (per patient, per fold)
# ============================================================================

def _build_fold_dataframes(df_patient, fold_idx, n_folds):
    """Split one patient's rows (chronological CSV order) into n_folds
    contiguous blocks. Block `fold_idx` becomes test, the rest become train."""
    blocks = np.array_split(np.arange(len(df_patient)), n_folds)
    test_block = blocks[fold_idx]
    train_block = np.concatenate([b for i, b in enumerate(blocks) if i != fold_idx])
    train_df = df_patient.iloc[train_block].copy()
    test_df = df_patient.iloc[test_block].copy()
    return train_df, test_df


def _preprocess_fold(train_df, test_df, feature_columns, label_col):
    """Fill NaNs, z-score normalize (stats fit on the train interval only, to
    avoid leaking test-interval information), and undersample train non-seizures."""
    train_df = train_df.copy()
    test_df = test_df.copy()
    train_df[feature_columns] = train_df[feature_columns].fillna(0)
    test_df[feature_columns] = test_df[feature_columns].fillna(0)

    if DATA_CONFIG.get('per_patient_normalization', False):
        baseline = train_df[train_df[label_col] == 0]
        if len(baseline) < 2:
            baseline = train_df
        pmean = baseline[feature_columns].mean()
        pstd = baseline[feature_columns].std().fillna(1).replace(0, 1)
    else:
        pmean = train_df[feature_columns].mean()
        pstd = train_df[feature_columns].std().replace(0, 1)

    train_df[feature_columns] = (train_df[feature_columns] - pmean) / pstd
    test_df[feature_columns] = (test_df[feature_columns] - pmean) / pstd
    train_df[feature_columns] = train_df[feature_columns].fillna(0)
    test_df[feature_columns] = test_df[feature_columns].fillna(0)

    undersampling_ratio = DATA_CONFIG.get('undersampling_ratio', None)
    if undersampling_ratio is not None:
        labels_np = train_df[label_col].to_numpy()
        seizure_idx = np.where(labels_np == 1)[0]
        non_seizure_idx = np.where(labels_np == 0)[0]
        n_target = min(len(seizure_idx) * undersampling_ratio, len(non_seizure_idx))
        rng = np.random.default_rng(42)
        selected_non_seizure_idx = rng.choice(non_seizure_idx, size=n_target, replace=False)
        keep_idx = np.sort(np.concatenate([selected_non_seizure_idx, seizure_idx]))
        train_df = train_df.iloc[keep_idx]

    return train_df, test_df


def _make_datasets(train_df, test_df, feature_columns, label_col, loss_type):
    """Build train (embedding-loss-specific) and plain classifier datasets.
    Patient IDs are all zero — only one patient is involved per run."""
    train_features_np = np.nan_to_num(
        train_df[feature_columns].to_numpy(), nan=0.0, posinf=0.0, neginf=0.0)
    test_features_np = np.nan_to_num(
        test_df[feature_columns].to_numpy(), nan=0.0, posinf=0.0, neginf=0.0)

    train_features = torch.tensor(train_features_np, dtype=torch.float32)
    train_labels = torch.tensor(train_df[label_col].to_numpy(), dtype=torch.long)
    train_pids = torch.zeros(len(train_df), dtype=torch.long)

    test_features = torch.tensor(test_features_np, dtype=torch.float32)
    test_labels = torch.tensor(test_df[label_col].to_numpy(), dtype=torch.long)
    test_pids = torch.zeros(len(test_df), dtype=torch.long)

    if loss_type == 'contrastive':
        train_dataset_embedding = ContrastiveDataset(
            train_features, train_labels, patient_ids=train_pids,
            num_pairs=EMBEDDING_TRAINING_CONFIG['num_pairs'],
        )
    elif loss_type == 'triplet':
        train_dataset_embedding = TripletDataset(train_features, train_labels, train_pids)
    else:  # 'batch_hard_triplet', 'supcon', or 'simple'
        train_dataset_embedding = EmbeddingDataset(train_features, train_labels, train_pids)

    train_dataset_simple = EmbeddingDataset(train_features, train_labels, train_pids)
    test_dataset_simple = EmbeddingDataset(test_features, test_labels, test_pids)

    return train_dataset_embedding, train_dataset_simple, test_dataset_simple


# ============================================================================
# ONE PATIENT, ONE FOLD
# ============================================================================

def run_patient_fold(df, pid, fold_idx, n_folds, feature_columns, results_root, device,
                      embedding_epochs=None, classifier_epochs=None):
    label_col = DATA_CONFIG['label_col']
    pid_col = DATA_CONFIG['patient_id_col']
    loss_type = EMBEDDING_TRAINING_CONFIG['loss_type']

    df_patient = df[df[pid_col] == pid]
    train_df, test_df = _build_fold_dataframes(df_patient, fold_idx, n_folds)

    if len(np.unique(train_df[label_col])) < 2 or len(np.unique(test_df[label_col])) < 2:
        logging.warning(
            f"[{pid} fold {fold_idx}] Skipped — train or test interval is missing a class "
            f"(train={dict(zip(*np.unique(train_df[label_col], return_counts=True)))}, "
            f"test={dict(zip(*np.unique(test_df[label_col], return_counts=True)))})."
        )
        return None

    train_df, test_df = _preprocess_fold(train_df, test_df, feature_columns, label_col)
    train_dataset_embedding, train_dataset_simple, test_dataset_simple = _make_datasets(
        train_df, test_df, feature_columns, label_col, loss_type,
    )

    fold_dir = os.path.join(results_root, pid, f"fold_{fold_idx}")
    os.makedirs(fold_dir, exist_ok=True)

    MODEL_CONFIG['input_dim'] = len(feature_columns)
    DATA_CONFIG['checkpoint_dir'] = os.path.join(
        DATA_CONFIG['_base_checkpoint_dir'], 'patient_baseline', pid, f"fold_{fold_idx}",
    )

    orig_emb_epochs = EMBEDDING_TRAINING_CONFIG['epochs']
    orig_clf_epochs = CLASSIFIER_TRAINING_CONFIG['epochs']
    if embedding_epochs is not None:
        EMBEDDING_TRAINING_CONFIG['epochs'] = embedding_epochs
    if classifier_epochs is not None:
        CLASSIFIER_TRAINING_CONFIG['epochs'] = classifier_epochs

    try:
        save_config_to_json(fold_dir)

        if loss_type == 'supcon':
            pk_sampler = PKBatchSampler(
                train_dataset_embedding.patient_ids,
                P=EMBEDDING_TRAINING_CONFIG['pk_p'],
                K=EMBEDDING_TRAINING_CONFIG['pk_k'],
                labels=train_dataset_embedding.labels,
                max_per_patient=EMBEDDING_TRAINING_CONFIG.get('max_per_patient'),
            )
            train_dataloader_embedding = DataLoader(train_dataset_embedding, batch_sampler=pk_sampler)
        else:
            train_dataloader_embedding = DataLoader(
                train_dataset_embedding,
                batch_size=EMBEDDING_TRAINING_CONFIG['batch_size'],
                shuffle=True,
            )

        train_dataloader_classifier = DataLoader(
            train_dataset_simple, batch_size=CLASSIFIER_TRAINING_CONFIG['batch_size'], shuffle=True,
        )
        test_dataloader_classifier = DataLoader(
            test_dataset_simple, batch_size=CLASSIFIER_TRAINING_CONFIG['batch_size'], shuffle=False,
        )
        train_dataloader_umap = DataLoader(train_dataset_simple, batch_size=128, shuffle=False)
        test_dataloader_umap = DataLoader(test_dataset_simple, batch_size=128, shuffle=False)
        pid_map = {0: pid}

        embedding_model = train_embedding_model(
            train_dataloader_embedding, device, results_dir=fold_dir,
            train_labels=train_dataset_embedding.labels,
        )

        if EVAL_CONFIG['generate_umap']:
            generate_umap_visualizations(
                embedding_model, train_dataloader_umap, device, fold_dir,
                sample_size=EVAL_CONFIG['umap_sample_size'],
                prefix='umap_pretrain',
                test_dataloader=test_dataloader_umap,
                pid_map=pid_map,
                plot_patient_ids=False,
            )

        classifier = train_classifier_head(
            embedding_model, train_dataloader_classifier, test_dataloader_classifier, device, fold_dir,
        )

        if EVAL_CONFIG['generate_umap'] and not CLASSIFIER_TRAINING_CONFIG['freeze_embeddings']:
            generate_umap_visualizations(
                classifier.embedding_model, train_dataloader_umap, device, fold_dir,
                sample_size=EVAL_CONFIG['umap_sample_size'],
                prefix='umap_finetuned',
                test_dataloader=test_dataloader_umap,
                pid_map=pid_map,
                plot_patient_ids=False,
            )

        _, test_metrics = eval_classifier_head(
            classifier, train_dataloader_classifier, test_dataloader_classifier, device, fold_dir,
        )
    finally:
        EMBEDDING_TRAINING_CONFIG['epochs'] = orig_emb_epochs
        CLASSIFIER_TRAINING_CONFIG['epochs'] = orig_clf_epochs

    metrics = {
        'accuracy': test_metrics['accuracy'],
        'precision': test_metrics['precision'],
        'recall': test_metrics['recall'],
        'f1': test_metrics['f1'],
        'roc_auc': test_metrics['roc_auc'],
    }
    with open(os.path.join(fold_dir, 'metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=2)

    return {
        'pid': pid,
        'fold_idx': fold_idx,
        'n_train': len(train_df),
        'n_test': len(test_df),
        'n_seizure_train': int((train_df[label_col] == 1).sum()),
        'n_seizure_test': int((test_df[label_col] == 1).sum()),
        'metrics': metrics,
    }


# ============================================================================
# SUMMARY
# ============================================================================

def _save_summary(all_results, results_root):
    if not all_results:
        logging.warning("No fold produced results — nothing to summarize.")
        return

    rows = [
        {
            'patient_id': r['pid'],
            'fold': r['fold_idx'],
            'n_train': r['n_train'],
            'n_test': r['n_test'],
            'n_seizure_train': r['n_seizure_train'],
            'n_seizure_test': r['n_seizure_test'],
            **r['metrics'],
        }
        for r in all_results
    ]
    df_summary = pd.DataFrame(rows)
    csv_path = os.path.join(results_root, 'summary.csv')
    df_summary.to_csv(csv_path, index=False)

    metric_cols = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
    per_patient = df_summary.groupby('patient_id')[metric_cols].agg(['mean', 'std'])
    overall = df_summary[metric_cols].agg(['mean', 'std'])

    with open(os.path.join(results_root, 'summary.json'), 'w') as f:
        json.dump({
            'per_fold': rows,
            'per_patient_mean_std': json.loads(per_patient.to_json()),
            'overall_mean_std': json.loads(overall.to_json()),
        }, f, indent=2)

    logging.info("=" * 80)
    logging.info("PER-PATIENT BASELINE — SUMMARY")
    logging.info("=" * 80)
    logging.info(f"\n{per_patient}\n")
    logging.info(f"Overall (all patients, all folds):\n{overall}\n")
    logging.info(f"Summary saved to {csv_path} and {os.path.join(results_root, 'summary.json')}")


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Per-patient chronological 4-fold baseline")
    parser.add_argument('--patients', nargs='+', default=None,
                         help="Patient IDs to run (default: every patient in the CSV)")
    parser.add_argument('--n-folds', type=int, default=PATIENT_BASELINE_CONFIG.get('n_folds', 4))
    parser.add_argument('--results-dir', default=None)
    parser.add_argument('--embedding-epochs', type=int, default=None,
                         help="Override EMBEDDING_TRAINING_CONFIG['epochs'] (e.g. for a quick smoke test)")
    parser.add_argument('--classifier-epochs', type=int, default=None,
                         help="Override CLASSIFIER_TRAINING_CONFIG['epochs']")
    args = parser.parse_args()

    logging.basicConfig(level=getattr(logging, LOGGING_CONFIG['level']), format=LOGGING_CONFIG['format'])

    device = (
        torch.device(f"cuda:{DEVICE_CONFIG['cuda_device']}")
        if DEVICE_CONFIG['use_cuda'] and torch.cuda.is_available()
        else torch.device('cpu')
    )
    logging.info(f"Using device: {device}\n")

    DATA_CONFIG['_base_checkpoint_dir'] = DATA_CONFIG['checkpoint_dir']

    csv_path = DATA_CONFIG['data_path']
    if csv_path is None:
        raise ValueError("DATA_CONFIG['data_path'] is not set.")
    logging.info(f"Loading dataset from {csv_path}")
    df = pd.read_csv(csv_path)

    pid_col = DATA_CONFIG['patient_id_col']
    label_col = DATA_CONFIG['label_col']
    drop_cols = [c for c in DATA_CONFIG['drop_columns'] if c in df.columns]
    avail_cols = [c for c in DATA_CONFIG.get('availability_cols', []) if c in df.columns]
    df.drop(columns=drop_cols, inplace=True)

    non_feature = [label_col, pid_col] + avail_cols
    feature_columns_all = [c for c in df.columns if c not in non_feature]

    patients = args.patients or PATIENT_BASELINE_CONFIG.get('patients') or sorted(df[pid_col].unique().tolist())
    n_folds = args.n_folds

    if args.results_dir:
        results_root = args.results_dir
    elif PATIENT_BASELINE_CONFIG.get('results_dir'):
        results_root = PATIENT_BASELINE_CONFIG['results_dir']
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_root = f"./results_patient_baseline_{timestamp}"
    os.makedirs(results_root, exist_ok=True)
    logging.info(f"Results will be saved to: {results_root}\n")
    logging.info(f"Patients: {patients}")
    logging.info(f"Folds per patient: {n_folds} (3 intervals train / 1 interval test, rotated)\n")

    all_results = []
    for pid in patients:
        df_patient_full = df[df[pid_col] == pid]
        if df_patient_full.empty:
            logging.warning(f"No rows found for patient {pid}, skipping.")
            continue

        # Features constant across this patient's own data carry no signal for them.
        stds = df_patient_full[feature_columns_all].std()
        constant = stds[stds == 0].index.tolist()
        feature_columns = [c for c in feature_columns_all if c not in constant]
        if constant:
            logging.info(f"[{pid}] Dropping {len(constant)} constant features")

        for fold_idx in range(n_folds):
            logging.info("=" * 80)
            logging.info(f"PATIENT {pid} — FOLD {fold_idx + 1}/{n_folds}")
            logging.info("=" * 80)
            result = run_patient_fold(
                df, pid, fold_idx, n_folds, feature_columns, results_root, device,
                embedding_epochs=args.embedding_epochs,
                classifier_epochs=args.classifier_epochs,
            )
            if result is not None:
                all_results.append(result)

    _save_summary(all_results, results_root)
    logging.info("PER-PATIENT BASELINE COMPLETE!")


if __name__ == "__main__":
    main()
