"""
run_grid_search.py — Grid search : patients × undersampling_ratio × embedding_dim.

Teste toutes les combinaisons de :
  - patients            : tous les baseline-patient-*.csv trouvés dans DATA_DIR
  - undersampling_ratio : [5, 10, 20, 50]
  - embedding_dim       : [16, 32, 64]

Pour chaque combinaison, lance une cross-validation à 4 plis temporels.

Structure des résultats :
  grid_results/
    <patient_id>/
      ratio_<R>_emb_<E>/
        fold_<k>/          ← modèles, plots, config
        cv_summary.json
        cv_summary.png
    grid_summary.csv       ← tableau comparatif de toutes les configs

Reprise automatique : si cv_summary.json existe déjà, la config est ignorée.

Usage :
    uv run python run_grid_search.py
    uv run python run_grid_search.py --data_dir /chemin/vers/csvs --output grid_results
    uv run python run_grid_search.py --patient_csv /chemin/vers/feat-grid-intersect_01-001.csv
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler

sys.path.insert(0, os.path.dirname(__file__))

from config import (
    BASELINE_CONFIG,
    CLASSIFIER_TRAINING_CONFIG,
    DATA_CONFIG,
    DEVICE_CONFIG,
    EMBEDDING_TRAINING_CONFIG,
    EVAL_CONFIG,
    LOGGING_CONFIG,
    MODEL_CONFIG,
)
from dataset import PKBatchSampler
from eval import eval_classifier_head, generate_umap_visualizations
from launch_train import (
    load_single_patient_dataset,
    save_config_to_json,
    train_classifier_head,
    train_embedding_model,
)
from visualization import plot_cv_summary

# ============================================================================
# PARAMÈTRES DE LA GRILLE
# ============================================================================
UNDERSAMPLING_RATIOS = [5, 10, 20, 50]
EMBEDDING_DIMS       = [16, 32, 64]


# ============================================================================
# UTILITAIRES
# ============================================================================

def find_patient_csvs(data_dir: Path):
    csvs = sorted(data_dir.glob("feat-grid-intersect_*.csv"))
    if not csvs:
        raise FileNotFoundError(f"Aucun fichier feat-grid-intersect_*.csv dans {data_dir}")
    return csvs


def _make_dataloaders(train_ds_emb, train_ds_clf, test_ds_clf, weights, loss_type):
    """Crée les DataLoaders pour embedding et classifieur."""
    if loss_type == 'supcon':
        n_uniq = len(torch.unique(train_ds_emb.patient_ids))
        eff_p  = min(EMBEDDING_TRAINING_CONFIG['pk_p'], n_uniq)
        sampler = PKBatchSampler(
            train_ds_emb.patient_ids, P=eff_p,
            K=EMBEDDING_TRAINING_CONFIG['pk_k'],
            labels=train_ds_emb.labels,
            max_per_patient=EMBEDDING_TRAINING_CONFIG.get('max_per_patient'),
        )
        dl_emb = DataLoader(train_ds_emb, batch_sampler=sampler)
    elif DATA_CONFIG['use_weighted_sampling'] and loss_type != 'contrastive':
        dl_emb = DataLoader(
            train_ds_emb,
            batch_size=EMBEDDING_TRAINING_CONFIG['batch_size'],
            sampler=WeightedRandomSampler(weights, len(weights), replacement=True),
        )
    else:
        dl_emb = DataLoader(
            train_ds_emb,
            batch_size=EMBEDDING_TRAINING_CONFIG['batch_size'],
            shuffle=True,
        )

    dl_train = DataLoader(train_ds_clf, batch_size=CLASSIFIER_TRAINING_CONFIG['batch_size'], shuffle=True)
    dl_test  = DataLoader(test_ds_clf,  batch_size=CLASSIFIER_TRAINING_CONFIG['batch_size'], shuffle=False)
    return dl_emb, dl_train, dl_test


# ============================================================================
# EXÉCUTION D'UNE CONFIGURATION
# ============================================================================

def run_one_config(patient_csv: Path, ratio: int, emb_dim: int,
                   device: torch.device, config_dir: Path) -> Optional[dict]:
    """
    Lance la CV 4-plis pour (patient, ratio, emb_dim).
    Retourne le summary dict, ou None si échec.
    Reprise automatique si cv_summary.json existe déjà.
    """
    summary_path = config_dir / "cv_summary.json"
    if summary_path.exists():
        logging.info(f"    [SKIP] déjà calculé → {config_dir.name}")
        with open(summary_path) as f:
            return json.load(f)

    config_dir.mkdir(parents=True, exist_ok=True)

    # Mise à jour des configs globales pour cette combinaison
    DATA_CONFIG['data_path']               = str(patient_csv)
    DATA_CONFIG['checkpoint_dir']          = str(config_dir / "checkpoints")
    BASELINE_CONFIG['undersampling_ratio'] = ratio
    MODEL_CONFIG['embedding_dim']          = emb_dim
    # patient_ids plots désactivés (inutiles en mono-patient, et réduisent la taille des résultats)

    loss_type = EMBEDDING_TRAINING_CONFIG['loss_type']
    n_folds   = BASELINE_CONFIG['n_folds']
    all_fold_metrics = []

    for fold in range(n_folds):
        fold_dir = config_dir / f"fold_{fold}"
        fold_dir.mkdir(exist_ok=True)

        # Réinitialise input_dim (peut varier si des features constantes diffèrent entre plis)
        MODEL_CONFIG['input_dim'] = None

        try:
            train_ds_emb, _, weights, _ = load_single_patient_dataset(
                csv_path=str(patient_csv), loss_type=loss_type, test_fold=fold)
            train_ds_clf, test_ds_clf, _, pid_map = load_single_patient_dataset(
                csv_path=str(patient_csv), loss_type='simple', test_fold=fold)
        except Exception as e:
            logging.warning(f"    Fold {fold} chargement échoué : {e}")
            continue

        save_config_to_json(str(fold_dir))

        dl_emb, dl_train, dl_test = _make_dataloaders(
            train_ds_emb, train_ds_clf, test_ds_clf, weights, loss_type)

        try:
            emb_model  = train_embedding_model(dl_emb, device, results_dir=str(fold_dir),
                                               train_labels=train_ds_emb.labels)

            dl_umap_train = DataLoader(train_ds_clf, batch_size=128, shuffle=False)
            dl_umap_test  = DataLoader(test_ds_clf,  batch_size=128, shuffle=False)
            generate_umap_visualizations(
                emb_model, dl_umap_train, device, str(fold_dir),
                sample_size=EVAL_CONFIG['umap_sample_size'],
                prefix='umap_pretrain',
                test_dataloader=dl_umap_test,
                pid_map=pid_map,
                plot_patient_ids=False,
            )

            classifier = train_classifier_head(emb_model, dl_train, dl_test, device, str(fold_dir))
            _, test_metrics = eval_classifier_head(classifier, dl_train, dl_test, device, str(fold_dir))
        except Exception as e:
            logging.warning(f"    Fold {fold} entraînement échoué : {e}")
            continue

        if test_metrics is not None:
            roc = test_metrics['roc_auc']
            roc_str = f"{roc:.4f}" if roc is not None else "N/A"
            logging.info(f"    Fold {fold} — AUC: {roc_str}  Recall: {test_metrics['recall']:.4f}  F1: {test_metrics['f1']:.4f}")
            all_fold_metrics.append({
                'fold':      fold,
                'accuracy':  test_metrics['accuracy'],
                'precision': test_metrics['precision'],
                'recall':    test_metrics['recall'],
                'f1':        test_metrics['f1'],
                'roc_auc':   test_metrics['roc_auc'],
            })

    if not all_fold_metrics:
        logging.warning(f"    Aucun fold valide pour {config_dir.name}")
        return None

    accs    = [m['accuracy'] for m in all_fold_metrics]
    f1s     = [m['f1']      for m in all_fold_metrics]
    recalls = [m['recall']  for m in all_fold_metrics]
    aucs    = [m['roc_auc'] for m in all_fold_metrics if m['roc_auc'] is not None]

    summary = {
        'patient':             patient_csv.stem,
        'undersampling_ratio': ratio,
        'embedding_dim':       emb_dim,
        'n_folds':             n_folds,
        'per_fold':            all_fold_metrics,
        'mean_accuracy':  float(np.mean(accs)),
        'std_accuracy':   float(np.std(accs)),
        'mean_recall':    float(np.mean(recalls)),
        'std_recall':     float(np.std(recalls)),
        'mean_f1':        float(np.mean(f1s)),
        'std_f1':         float(np.std(f1s)),
        'mean_roc_auc':   float(np.mean(aucs)) if aucs else None,
        'std_roc_auc':    float(np.std(aucs))  if aucs else None,
    }

    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=4)

    plot_cv_summary(summary, save_path=str(config_dir / "cv_summary.png"))
    return summary


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Grid search baseline mono-patient")
    parser.add_argument("--data_dir", default=None,
                        help="Dossier contenant les baseline-patient-*.csv "
                             "(défaut : dossier du data_path dans config.py)")
    parser.add_argument("--patient_csv", default=None,
                        help="Chemin vers un seul CSV patient à traiter "
                             "(si renseigné, ignore --data_dir et lance la grille sur ce seul fichier)")
    parser.add_argument("--output", default="./grid_results",
                        help="Dossier de sortie (défaut : ./grid_results)")
    args = parser.parse_args()

    logging.basicConfig(level=getattr(logging, LOGGING_CONFIG['level']),
                        format=LOGGING_CONFIG['format'])

    output   = Path(args.output)
    output.mkdir(exist_ok=True)

    device = (torch.device(f"cuda:{DEVICE_CONFIG['cuda_device']}")
              if DEVICE_CONFIG['use_cuda'] and torch.cuda.is_available()
              else torch.device('cpu'))
    logging.info(f"Device : {device}")

    if args.patient_csv:
        patient_csv = Path(args.patient_csv).expanduser().resolve()
        if not patient_csv.exists():
            raise FileNotFoundError(f"Fichier patient introuvable : {patient_csv}")
        patient_csvs = [patient_csv]
        logging.info(f"Mode mono-fichier : 1 patient sélectionné → {patient_csv}")
    else:
        data_dir = Path(args.data_dir) if args.data_dir else Path(DATA_CONFIG['data_path']).parent
        patient_csvs = find_patient_csvs(data_dir)
        logging.info(f"{len(patient_csvs)} patient(s) trouvé(s) dans {data_dir}")
    for p in patient_csvs:
        logging.info(f"  {p.name}")

    total = len(patient_csvs) * len(UNDERSAMPLING_RATIOS) * len(EMBEDDING_DIMS)
    n_folds = BASELINE_CONFIG['n_folds']
    logging.info(f"\nGrille : {len(patient_csvs)} patients × {len(UNDERSAMPLING_RATIOS)} ratios "
                 f"× {len(EMBEDDING_DIMS)} emb_dims = {total} configs × {n_folds} folds = {total * n_folds} runs")
    logging.info(f"Résultats dans : {output.resolve()}\n")

    rows = []
    done = 0

    for patient_csv in patient_csvs:
        patient_id = patient_csv.stem.replace("baseline-patient-", "")

        for ratio in UNDERSAMPLING_RATIOS:
            for emb_dim in EMBEDDING_DIMS:
                done += 1
                config_name = f"ratio_{ratio}_emb_{emb_dim}"
                config_dir  = output / patient_id / config_name

                logging.info(f"\n{'='*70}")
                logging.info(f"[{done}/{total}] Patient {patient_id} | ratio {ratio}:1 | emb_dim {emb_dim}")
                logging.info(f"{'='*70}")

                summary = run_one_config(patient_csv, ratio, emb_dim, device, config_dir)

                if summary:
                    rows.append({
                        'patient':             patient_id,
                        'undersampling_ratio': ratio,
                        'embedding_dim':       emb_dim,
                        'mean_roc_auc':        summary.get('mean_roc_auc'),
                        'std_roc_auc':         summary.get('std_roc_auc'),
                        'mean_recall':         summary.get('mean_recall'),
                        'std_recall':          summary.get('std_recall'),
                        'mean_f1':             summary.get('mean_f1'),
                        'std_f1':              summary.get('std_f1'),
                        'mean_accuracy':       summary.get('mean_accuracy'),
                        'std_accuracy':        summary.get('std_accuracy'),
                    })

                # Sauvegarde intermédiaire après chaque config
                if rows:
                    pd.DataFrame(rows).sort_values(
                        ['patient', 'undersampling_ratio', 'embedding_dim']
                    ).to_csv(output / "grid_summary.csv", index=False)

    # ── Résumé final ─────────────────────────────────────────────────────────
    logging.info(f"\n{'='*70}")
    logging.info("GRID SEARCH TERMINÉ")
    logging.info(f"{'='*70}")

    if not rows:
        logging.warning("Aucun résultat collecté.")
        return

    df = pd.DataFrame(rows)
    df.to_csv(output / "grid_summary.csv", index=False)
    logging.info(f"Tableau complet : {output / 'grid_summary.csv'}")

    # Meilleure config par patient
    logging.info("\nMeilleure config par patient (ROC-AUC moyen) :")
    for patient, grp in df.groupby('patient'):
        valid_grp = grp.dropna(subset=['mean_roc_auc'])
        if valid_grp.empty:
            logging.info(f"  {patient} → aucune config avec ROC-AUC définie")
            continue

        best = valid_grp.loc[valid_grp['mean_roc_auc'].idxmax()]
        logging.info(f"  {patient} → ratio={int(best['undersampling_ratio'])}  "
                     f"emb_dim={int(best['embedding_dim'])}  "
                     f"AUC={best['mean_roc_auc']:.4f} ± {best['std_roc_auc']:.4f}")

    # Meilleure config toutes moyennes confondues
    valid_df = df.dropna(subset=['mean_roc_auc'])
    logging.info(f"\nMeilleure config moyenne tous patients :")
    if valid_df.empty:
        logging.info("  aucune config avec ROC-AUC définie")
        return

    mean_by_config = valid_df.groupby(['undersampling_ratio', 'embedding_dim'])['mean_roc_auc'].mean()
    best_ratio, best_emb = mean_by_config.idxmax()
    logging.info(f"  ratio={best_ratio}  emb_dim={best_emb}  "
                 f"AUC moyen={mean_by_config.max():.4f}")


if __name__ == "__main__":
    main()
