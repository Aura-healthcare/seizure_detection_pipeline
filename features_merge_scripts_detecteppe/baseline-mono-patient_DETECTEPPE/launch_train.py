"""
launch_train.py — Train a contrastive embedding model then fine-tune a classifier.

Embedding loss is controlled by EMBEDDING_TRAINING_CONFIG['loss_type'] in config.py:
  'contrastive'        — pairwise contrastive loss (ContrastiveDataset, cross-patient pairs)
  'triplet'            — offline triplet loss (TripletDataset, cross-patient positives)
  'batch_hard_triplet' — online hard mining (EmbeddingDataset + BatchHardTripletLoss)
"""

# les imports
import warnings
warnings.filterwarnings("ignore", message="Compilation requested for previously compiled argument types")

import pandas as pd
import logging
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import sys
import json
from datetime import datetime
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from model import DeepResidualEmbeddingModel, SeizureClassifier
from loss import ContrastiveLoss, TripletLoss, BatchHardTripletLoss, SupervisedContrastiveLoss
from dataset import ContrastiveDataset, TripletDataset, EmbeddingDataset, PKBatchSampler
class FocalLoss(nn.Module):
    """Focal Loss — réduit le poids des exemples faciles (majoritaires) pour forcer
    le modèle à se concentrer sur les cas difficiles (minoritaires).
    FL(p_t) = -(1 - p_t)^gamma * log(p_t), avec pondération optionnelle par classe."""

    def __init__(self, gamma=2.0, weight=None):
        super().__init__()
        self.gamma = gamma
        self.weight = weight  # tensor de poids par classe, même device que le modèle

    def forward(self, logits, targets):
        log_prob = F.log_softmax(logits, dim=1)
        prob     = log_prob.exp()
        log_pt   = log_prob.gather(1, targets.unsqueeze(1)).squeeze(1)
        pt       = prob.gather(1, targets.unsqueeze(1)).squeeze(1)
        loss     = -((1 - pt) ** self.gamma) * log_pt
        if self.weight is not None:
            loss = loss * self.weight[targets]
        return loss.mean()


from train import (
    train_model,
    train_model_triplet,
    train_model_triplet_hard_negative,
    train_model_supcon,
    train_classifier,
)
from visualization import plot_training_history, plot_cv_summary
from eval import generate_umap_visualizations, eval_classifier_head
from config import (
    DATA_CONFIG,
    MODEL_CONFIG,
    EMBEDDING_TRAINING_CONFIG,
    CLASSIFIER_TRAINING_CONFIG,
    EVAL_CONFIG,
    BASELINE_CONFIG,
    DEVICE_CONFIG,
    LOGGING_CONFIG,
)


# ============================================================================
# CONFIGURATION SAVING
# ============================================================================

# sauvegarde tous les hyperparamètres (configs de données, modèle, entraînement)
# dans un fichier training_config.json dans le dossier résultats
# utile pour la reproductibilité des expériences 

from pathlib import Path

def serialize_for_json(obj):
    if isinstance(obj, Path):
        return str(obj)
    elif isinstance(obj, dict):
        return {k: serialize_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [serialize_for_json(v) for v in obj]
    else:
        return obj
    
import os
import json
import logging
from datetime import datetime
from pathlib import Path
import json
import logging
from datetime import datetime
from pathlib import Path

def save_config_to_json(results_dir):
    # Copie 
    data_config_safe = DATA_CONFIG.copy()

    # s'assurer que le nom du dataset est bien loggé
    if "dataset_name" not in data_config_safe:
        data_path = DATA_CONFIG.get("data_path", None)
        if data_path is not None:
            data_config_safe["dataset_name"] = Path(data_path).name

    # Retirer le chemin absolu
    data_config_safe.pop("data_path", None)

    config_dict = {
        "data_config": serialize_for_json(data_config_safe),
        "model_config": serialize_for_json(MODEL_CONFIG),
        "embedding_training_config": serialize_for_json(EMBEDDING_TRAINING_CONFIG),
        "classifier_training_config": serialize_for_json(CLASSIFIER_TRAINING_CONFIG),
        "baseline_config": serialize_for_json(BASELINE_CONFIG),
        "eval_config": serialize_for_json(EVAL_CONFIG),
        "device_config": serialize_for_json(DEVICE_CONFIG),
        "logging_config": serialize_for_json(LOGGING_CONFIG),
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }

    config_path = Path(results_dir) / "training_config.json"

    with open(config_path, "w") as f:
        json.dump(serialize_for_json(config_dict), f, indent=4)

    logging.info(f"Configuration saved to {config_path}")
    return str(config_path)

# ============================================================================
# DATA LOADING
# ============================================================================
# convertit les identifiants patients (ex: "01-001", "01-002") en entiers (0, 1, 2...)
# nécessaire car PyTorch ne gère pas les chaînes de caractères comme identifiants
def _encode_patient_ids(series, pid_map=None):
    """Map patient IDs (possibly strings like '01-001') to integer codes.

    If pid_map is None, builds a new mapping from the series (for train).
    If pid_map is provided, reuses the same mapping (for test) so that
    integer codes are consistent between train and test.
    Patients in the series that are absent from pid_map get a new code
    appended at the end of pid_map (in-place).

    Returns (codes, pid_map) where pid_map is {int_code: original_string}.
    """
    if pid_map is None:
        codes, uniques = pd.factorize(series)
        pid_map = {i: str(u) for i, u in enumerate(uniques)}
        return codes, pid_map

    reverse = {v: k for k, v in pid_map.items()}
    codes = []
    for pid_str in series.astype(str):
        if pid_str not in reverse:
            new_code = len(pid_map)
            pid_map[new_code] = pid_str
            reverse[pid_str] = new_code
        codes.append(reverse[pid_str])
    return np.array(codes), pid_map


def _per_patient_normalize(df, feature_cols, pid_col, label_col):
    """Z-score each feature relative to the patient's non-seizure baseline.
    Falls back to the patient's overall stats if no non-seizure data exists."""
    df = df.copy()
    for pid, grp in df.groupby(pid_col): # boucle sur les groupes (un groupe = un patient)
        baseline = grp[grp[label_col] == 0] # il prend uniquement les lignes hors-crise -> baseline
        if len(baseline) < 2:
            baseline = grp  # fallback
        pmean = baseline[feature_cols].mean() # moyenne de la baseline (calcule moue,,e colonne par colonne)
        # pmean est une serie indexée par features_cols
        pstd = baseline[feature_cols].std().fillna(1).replace(0, 1) # écart type de la baseline
        # calcule écart type par feature
        # fillna(1) : NaN quand toutes les valeurs de la feature sont NaN pour ce patient (capteur absent)
        # replace(0, 1) : feature constante donc std=0, on remplace par 1 pour éviter division par 0
        df.loc[grp.index, feature_cols] = (grp[feature_cols] - pmean) / pstd # transforme les valeurs du patients en z-score
        # z = x - mu_baseline / sigma_baseline
    return df

# fonction de chargeent principale 
def load_train_test_dataset(csv_path, loss_type='batch_hard_triplet'):
    """Load and preprocess features dataset."""
    if csv_path is None:
        raise ValueError("DATA_PRIVATE_CSV environment variable is not set. Please set it to the path of your data CSV file.")
    logging.info(f"Loading dataset from {csv_path}")
    logging.info(f"Using loss type: {loss_type}")
    df = pd.read_csv(csv_path) # chargement

    # Read column names from config
    pid_col = DATA_CONFIG['patient_id_col']
    label_col = DATA_CONFIG['label_col']
    split_col = DATA_CONFIG['split_col']
    drop_cols = [c for c in DATA_CONFIG['drop_columns'] if c in df.columns] # on récupère les colonnes à drop du fichier config
    avail_cols = [c for c in DATA_CONFIG.get('availability_cols', []) if c in df.columns]

    df.drop(columns=drop_cols, inplace=True) # nettoyage des colonne inutiles
    # inplace = true c'est pour que la suppression soit faite directement dans df, pas de nouveau dataframe crée

    # Split train/test
    train_df = df[df[split_col] == DATA_CONFIG['train_split_name']].drop(columns=[split_col])
    test_df = df[df[split_col] == DATA_CONFIG['test_split_name']].drop(columns=[split_col])

    if DATA_CONFIG['num_patients_train'] is not None:
        patient_ids_list = train_df[pid_col].unique()
        train_df = train_df[train_df[pid_col].isin(
            patient_ids_list[:DATA_CONFIG['num_patients_train']]
        )]

    logging.info(f"Train samples: {len(train_df)}, Test samples: {len(test_df)}")
    logging.info(f"Train label distribution:\n{train_df[label_col].value_counts()}")
    logging.info(f"Test label distribution:\n{test_df[label_col].value_counts()}")

    # Identify feature columns (everything except label, patient_id, and availability flags)
    non_feature = [label_col, pid_col] + avail_cols
    feature_columns = [c for c in train_df.columns if c not in non_feature]

    # Drop constant features
    stds = train_df[feature_columns].std()
    constant = stds[stds == 0].index.tolist()
    # Une feature qui ne varie jamais n'apporte aucune information, on la supprime : 

    if constant:
        logging.info(f"Dropping {len(constant)} constant features: {constant}")
        feature_columns = [c for c in feature_columns if c not in constant]
        train_df.drop(columns=constant, inplace=True)
        test_df.drop(columns=constant, inplace=True)

    # Fill NaN with 0 (for rows where a sensor was unavailable)
    train_df[feature_columns] = train_df[feature_columns].fillna(0)
    test_df[feature_columns] = test_df[feature_columns].fillna(0)

    # Normalisation
    # Soit par patient (voir fonction ci-dessus), soit globale (z-score sur tout le train, appliqué aussi au test)
    if DATA_CONFIG.get('per_patient_normalization', False):
        # si per_patient_normalization == True, le bloc if est exécuté
        # si False ou absent le bloc est ignoré
        logging.info("Applying per-patient normalisation (baseline z-score)...")
        train_df = _per_patient_normalize(train_df, feature_columns, pid_col, label_col) # fonction de z-score par patient déclarée plus haut
        test_df = _per_patient_normalize(test_df, feature_columns, pid_col, label_col)
        # Fill any NaN introduced by patients with constant features
        train_df[feature_columns] = train_df[feature_columns].fillna(0)
        test_df[feature_columns] = test_df[feature_columns].fillna(0)
    else:
        # Global z-score normalisation
        train_mean = train_df[feature_columns].mean()
        train_std = train_df[feature_columns].std().replace(0, 1)
        train_df[feature_columns] = (train_df[feature_columns] - train_mean) / train_std
        test_df[feature_columns] = (test_df[feature_columns] - train_mean) / train_std

    n_features = len(feature_columns)
    logging.info(f"Using {n_features} features")

    # Auto-detect input_dim
    if MODEL_CONFIG['input_dim'] is None:
        MODEL_CONFIG['input_dim'] = n_features
        logging.info(f"Auto-detected input_dim = {n_features}")

    # ============================================================================
    # SMOTE pour le déséquilibre de classe 
    # ============================================================================
    logging.info("Applying SMOTE + Undersampling to balance training classes...")
    train_features_np = train_df[feature_columns].to_numpy()
    train_labels_np = train_df[label_col].to_numpy()
    train_pids_np = train_df[pid_col].to_numpy()
    
    # Retirer les NaN / Inf avant SMOTE
    train_features_np = np.nan_to_num(train_features_np, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Compter la distribution de classe 
    unique, counts = np.unique(train_labels_np, return_counts=True)
    logging.info(f"Before resampling: {dict(zip(unique, counts))}")
    
    # SMOTE: oversample la minorité (crises) pour matcher la majorité à un ratio de ~50% 
    # puis on sous sample la majorité pour réduire la taille des données 
    pipeline = Pipeline([
        ('smote', SMOTE(k_neighbors=3, random_state=42, sampling_strategy=0.5)),  # minorité devient 50% de la majorité
        ('undersampling', RandomUnderSampler(sampling_strategy=0.5, random_state=42))  # garder le ratio à 50%
    ])
    
    train_features_resampled, train_labels_resampled = pipeline.fit_resample(
        train_features_np, train_labels_np
    )
    
    # For patient IDs: repeat the original mapping for synthetic SMOTE samples
    # SMOTE generates indices from the original feature space; we map back to patient IDs
    n_original = len(train_labels_np)
    n_resampled = len(train_labels_resampled)
    
    # Create a patient ID mapping for resampled data
    # Synthetic samples get the patient ID of their parent
    train_pids_resampled = np.zeros(n_resampled, dtype=train_pids_np.dtype)
    train_pids_resampled[:n_original] = train_pids_np
    
    # For synthetic samples beyond original size, assign them to patients of their class
    if n_resampled > n_original:
        seizure_indices = np.where(train_labels_np == 1)[0]
        seizure_pids = train_pids_np[seizure_indices]
        for i in range(n_original, n_resampled):
            # Assign synthetic seizure to a random seizure-class patient
            if train_labels_resampled[i] == 1 and len(seizure_pids) > 0:
                train_pids_resampled[i] = np.random.choice(seizure_pids)
            else:
                # Fallback: pick a random patient from training
                train_pids_resampled[i] = np.random.choice(train_pids_np)
    
    unique_resampled, counts_resampled = np.unique(train_labels_resampled, return_counts=True)
    logging.info(f"After resampling: {dict(zip(unique_resampled, counts_resampled))}")
    logging.info(f"Resampled dataset size: {len(train_labels_resampled)} (was {len(train_labels_np)})")
    
    # Convert to tensors — encode patient IDs to integers.
    # Test reuses the train mapping so that the same patient always gets the same code.
    train_pid_codes, pid_map = _encode_patient_ids(pd.Series(train_pids_resampled))
    test_pid_codes, pid_map = _encode_patient_ids(test_df[pid_col], pid_map=pid_map)

    train_features = torch.tensor(train_features_resampled, dtype=torch.float32)
    train_labels = torch.tensor(train_labels_resampled, dtype=torch.long)
    train_patient_ids = torch.tensor(train_pid_codes, dtype=torch.long)

    test_features = torch.tensor(test_df[feature_columns].to_numpy(), dtype=torch.float32)
    test_labels = torch.tensor(test_df[label_col].to_numpy(), dtype=torch.long)
    test_patient_ids = torch.tensor(test_pid_codes, dtype=torch.long)

    #  Création du dataset selon le type de loss
    if loss_type == 'contrastive':
        train_dataset = ContrastiveDataset(
            train_features,
            train_labels,
            patient_ids=train_patient_ids,
            num_pairs=EMBEDDING_TRAINING_CONFIG['num_pairs'], # paires (ancre, positif/négatif)
        )
        test_dataset = ContrastiveDataset(
            test_features,
            test_labels,
            patient_ids=test_patient_ids,
            num_pairs=1000,
        )
    elif loss_type == 'triplet':
        train_dataset = TripletDataset(train_features, train_labels, train_patient_ids) # triplets (ancre, positif, négatif)
        test_dataset = TripletDataset(test_features, test_labels, test_patient_ids) 
    else:  # 'batch_hard_triplet', 'supcon', or 'simple'
        train_dataset = EmbeddingDataset(train_features, train_labels, train_patient_ids)
        test_dataset = EmbeddingDataset(test_features, test_labels, test_patient_ids) # samples simples (features, label, patient_id)

    patient_counts = pd.Series(train_pid_codes).value_counts()
    weights = pd.Series(train_pid_codes).apply(lambda x: 1.0 / patient_counts[x]).values # Donne plus de poids aux patients sous-représentés pour équilibrer l'échantillonnage

    return train_dataset, test_dataset, weights, pid_map


# ============================================================================
# Patient unique chargement des données (baseline)
# ============================================================================

def load_single_patient_dataset(csv_path, loss_type='batch_hard_triplet', test_fold=None):
    """Charge le CSV d'un seul patient et découpe ses données en plis temporels.

    - Le CSV est censé être ordonné chronologiquement (chaque ligne est une fenêtre de temps de 1s)
    - Données divisées en BASELINE_CONFIG['n_folds'] blocs égaux
    - Le pli `test_fold` est réservé au test et les autres forment le train
    - Si test_fold est None, on utilise BASELINE_CONFIG['test_fold']
    - La normalisation est ajustée sur les données de train uniquement
    """
    if csv_path is None:
        raise ValueError("Aucun chemin de données configuré.")

    n_folds = BASELINE_CONFIG['n_folds']
    if test_fold is None:
        test_fold = BASELINE_CONFIG['test_fold']
    logging.info(f"Chargement dataset single-patient depuis {csv_path}")
    logging.info(f"Stratégie : {n_folds} plis temporels, pli de test = {test_fold}")

    df = pd.read_csv(csv_path)


    # recupère les noms des colonnes clés 
    pid_col = DATA_CONFIG['patient_id_col'] 
    label_col = DATA_CONFIG['label_col']
    split_col = DATA_CONFIG.get('split_col', 'training-split')

    # les listes drop_cols et avail_cols sont filtrées avec if i in df.columns 
    # pour n'inclure que les colonnes réellement présentes
    drop_cols = [c for c in DATA_CONFIG['drop_columns'] if c in df.columns]
    avail_cols = [c for c in DATA_CONFIG.get('availability_cols', []) if c in df.columns]

    # supprimer la colonne de split existante (car on refait le split par plis)
    if split_col in df.columns:
        df.drop(columns=[split_col], inplace=True)
    df.drop(columns=drop_cols, inplace=True)

    # Découpage temporel : fold_idx appartient à {0, 1, ..., n_folds-1}
    n = len(df)
    fold_size = n // n_folds # taille d'un fold
    fold_idx = np.minimum(np.arange(n) // fold_size, n_folds - 1)

    train_df = df[fold_idx != test_fold].copy()
    test_df = df[fold_idx == test_fold].copy()

    logging.info(f"Train : {len(train_df)} samples  |  Test (pli {test_fold}) : {len(test_df)} samples")
    logging.info(f"Distribution train :\n{train_df[label_col].value_counts()}")
    logging.info(f"Distribution test  :\n{test_df[label_col].value_counts()}")

    # Colonnes de features (tout sauf label, patient_id, flags de capteurs)
    non_feature = [label_col, pid_col] + avail_cols
    feature_columns = [c for c in train_df.columns if c not in non_feature]

    # Supprimer les features constantes (sur le train uniquement)
    stds = train_df[feature_columns].std()
    constant = stds[stds == 0].index.tolist()
    if constant:
        logging.info(f"Suppression de {len(constant)} features constantes")
        feature_columns = [c for c in feature_columns if c not in constant]
        train_df.drop(columns=constant, inplace=True)
        test_df.drop(columns=constant, inplace=True)

    # Remplacer Inf par NaN avant fillna : certaines features spectrales/ratios
    # peuvent contenir +Inf/-Inf que fillna(0) ne capture pas
    train_df[feature_columns] = train_df[feature_columns].replace([np.inf, -np.inf], np.nan).fillna(0)
    test_df[feature_columns] = test_df[feature_columns].replace([np.inf, -np.inf], np.nan).fillna(0)

    # Normalisation : stats calculées sur la baseline non-crise du TRAIN uniquement,
    # appliquées ensuite au test pour éviter toute fuite de données.
    if DATA_CONFIG.get('per_patient_normalization', False):
        logging.info("Normalisation z-score sur la baseline non-crise du train...")
        baseline = train_df[train_df[label_col] == 0][feature_columns]
        if len(baseline) < 2:
            baseline = train_df[feature_columns]
        pmean = baseline.mean()
        pstd = baseline.std().fillna(1).replace(0, 1)
        train_df[feature_columns] = (train_df[feature_columns] - pmean) / pstd
        test_df[feature_columns] = (test_df[feature_columns] - pmean) / pstd
        train_df[feature_columns] = train_df[feature_columns].replace([np.inf, -np.inf], np.nan).fillna(0)
        test_df[feature_columns] = test_df[feature_columns].replace([np.inf, -np.inf], np.nan).fillna(0)
    else:
        train_mean = train_df[feature_columns].mean()
        train_std = train_df[feature_columns].std().replace(0, 1)
        train_df[feature_columns] = (train_df[feature_columns] - train_mean) / train_std
        test_df[feature_columns] = (test_df[feature_columns] - train_mean) / train_std
        train_df[feature_columns] = train_df[feature_columns].replace([np.inf, -np.inf], np.nan).fillna(0)
        test_df[feature_columns] = test_df[feature_columns].replace([np.inf, -np.inf], np.nan).fillna(0)

    n_features = len(feature_columns)
    logging.info(f"Nombre de features utilisées : {n_features}")

    if MODEL_CONFIG['input_dim'] is None:
        MODEL_CONFIG['input_dim'] = n_features
        logging.info(f"input_dim auto-détecté = {n_features}")

    # SMOTE est désactivé pour le cas mono-patient : avec peu de crises réelles,
    # les échantillons synthétiques ne généralisent pas et l'AUC test chute à ~0.5
    # L'équilibre des classes est géré par class_weights dans la CrossEntropyLoss
    train_features_np = train_df[feature_columns].to_numpy()
    train_labels_np = train_df[label_col].to_numpy()
    train_pids_np = train_df[pid_col].to_numpy()

    train_features_np = np.nan_to_num(train_features_np, nan=0.0, posinf=0.0, neginf=0.0)

    unique, counts = np.unique(train_labels_np, return_counts=True)
    logging.info(f"Class distribution (no resampling): {dict(zip(unique.tolist(), counts.tolist()))}")

    # Sous-sampling des non-crises : pour chaque crise, on garde 10 non-crises (ratio 1:10)
    seizure_idx = np.where(train_labels_np == 1)[0]
    non_seizure_idx = np.where(train_labels_np == 0)[0]
    if len(seizure_idx) == 0:
        # Pas de crise dans ce fold d'entraînement : on garde tout tel quel
        train_features_resampled = train_features_np
        train_labels_resampled = train_labels_np
        train_pids_resampled = train_pids_np
    else:
        n_target = min(len(seizure_idx) * BASELINE_CONFIG['undersampling_ratio'], len(non_seizure_idx))
        rng = np.random.default_rng(42)
        selected_non_seizure_idx = np.sort(rng.choice(non_seizure_idx, size=n_target, replace=False))
        keep_idx = np.sort(np.concatenate([selected_non_seizure_idx, seizure_idx]))
        train_features_resampled = train_features_np[keep_idx]
        train_labels_resampled = train_labels_np[keep_idx]
        train_pids_resampled = train_pids_np[keep_idx]
    n_sz = (train_labels_resampled == 1).sum()
    n_non_sz = (train_labels_resampled == 0).sum()
    ratio = n_non_sz / n_sz if n_sz > 0 else float('inf')
    unique_resampled, counts_resampled = np.unique(train_labels_resampled, return_counts=True)
    logging.info(f"Class distribution (after subsampling): {dict(zip(unique_resampled.tolist(), counts_resampled.tolist()))} | ratio non-crise/crise : {ratio:.1f}:1")

    # Baseline logistique : diagnostic pour vérifier si le signal est discriminant
    # Si ce score est aussi ~0.5, le problème vient des features, pas du modèle.
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score as _roc_auc_score
    test_features_np = test_df[feature_columns].to_numpy()
    test_labels_np   = test_df[label_col].to_numpy()
    if len(np.unique(train_labels_resampled)) >= 2 and len(np.unique(test_labels_np)) >= 2:
        lr = LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42)
        lr.fit(train_features_resampled, train_labels_resampled)
        lr_probs = lr.predict_proba(test_features_np)[:, 1]
        lr_auc = _roc_auc_score(test_labels_np, lr_probs)
        logging.info(f"[Baseline LR] ROC-AUC sur le test : {lr_auc:.4f}")
    else:
        logging.info("[Baseline LR] ignoré — une seule classe dans le train ou le test")

    # Encodage des patient IDs (même mapping train et test)
    train_pid_codes, pid_map = _encode_patient_ids(pd.Series(train_pids_resampled))
    test_pid_codes, pid_map = _encode_patient_ids(test_df[pid_col], pid_map=pid_map)

    train_features = torch.tensor(train_features_resampled, dtype=torch.float32)
    train_labels = torch.tensor(train_labels_resampled, dtype=torch.long)
    train_patient_ids = torch.tensor(train_pid_codes, dtype=torch.long)

    test_features = torch.tensor(test_df[feature_columns].to_numpy(), dtype=torch.float32)
    test_labels = torch.tensor(test_df[label_col].to_numpy(), dtype=torch.long)
    test_patient_ids = torch.tensor(test_pid_codes, dtype=torch.long)

    # Création du dataset selon la loss
    if loss_type == 'contrastive':
        train_dataset = ContrastiveDataset(train_features, train_labels,
                                           patient_ids=train_patient_ids,
                                           num_pairs=EMBEDDING_TRAINING_CONFIG['num_pairs'])
        test_dataset = ContrastiveDataset(test_features, test_labels,
                                          patient_ids=test_patient_ids, num_pairs=1000)
    elif loss_type == 'triplet':
        train_dataset = TripletDataset(train_features, train_labels, train_patient_ids)
        test_dataset = TripletDataset(test_features, test_labels, test_patient_ids)
    else:  # 'batch_hard_triplet', 'supcon', ou 'simple'
        train_dataset = EmbeddingDataset(train_features, train_labels, train_patient_ids)
        test_dataset = EmbeddingDataset(test_features, test_labels, test_patient_ids)

    # Poids d'échantillonnage basés sur les données rééchantillonnées (après SMOTE)
    label_counts_resampled = np.bincount(train_labels_resampled)
    sample_weights = np.array([1.0 / label_counts_resampled[l] for l in train_labels_resampled], dtype=np.float64)

    return train_dataset, test_dataset, sample_weights, pid_map


# ============================================================================
# PHASE 1: EMBEDDING TRAINING
# ============================================================================

# Phase 1 - entraînement du modèle d'embedding avec la loss choisie (contrastive, triplet, batch hard triplet, ou supcon)
def train_embedding_model(train_dataloader, device, results_dir=None, train_labels=None):
    """Train the embedding model using the loss type set in config."""
    loss_type = EMBEDDING_TRAINING_CONFIG['loss_type']

    logging.info("=" * 80)
    logging.info(f"PHASE 1: Training Embedding Model with {loss_type.upper()} Loss")
    logging.info("=" * 80)

    embedding_model = DeepResidualEmbeddingModel(
        input_dim=MODEL_CONFIG['input_dim'],
        embedding_dim=MODEL_CONFIG['embedding_dim'],
        num_blocks=MODEL_CONFIG['num_residual_blocks'],
        dropout=MODEL_CONFIG.get('embedding_dropout', 0.2),
    ).to(device)

    if loss_type == 'contrastive': 
        loss_fn = ContrastiveLoss() # Rapproche les paires similaires, éloigne les dissimilaires
    elif loss_type == 'triplet':
        loss_fn = TripletLoss(margin=EMBEDDING_TRAINING_CONFIG['margin']) # Rapproche les ancre-positif, éloigne les ancre-négatif d'au moins la marge
    elif loss_type == 'supcon':
        # Compute inverse-frequency class weights so seizure anchors
        # contribute equally to the loss despite ~800:1 imbalance
        # Pour supcon, des poids de classe sont calculés pour compenser le déséquilibre ~800:1 (données normales vs crises)
        class_weights = None
        if train_labels is not None:
            labels_np = train_labels.numpy() if hasattr(train_labels, 'numpy') else train_labels
            unique, counts = np.unique(labels_np, return_counts=True)
            total = counts.sum()
            class_weights = {int(c): total / (len(unique) * count) for c, count in zip(unique, counts)}
            logging.info(f"SupCon class weights: {class_weights}")
        loss_fn = SupervisedContrastiveLoss( # Tous les exemples de la même classe s'attirent, tous les exemples de classes différentes se repoussent
            temperature=EMBEDDING_TRAINING_CONFIG['temperature'],
            class_weights=class_weights,
        )
    else:  # batch_hard_triplet 
        loss_fn = BatchHardTripletLoss(margin=EMBEDDING_TRAINING_CONFIG['margin']) # Cherche le triplet le plus difficile dans le batch pour chaque ancre et applique la triplet loss dessus

    optimizer = optim.Adam(
        embedding_model.parameters(),
        lr=EMBEDDING_TRAINING_CONFIG['learning_rate'],
    )

    scheduler = None
    if EMBEDDING_TRAINING_CONFIG['use_scheduler']:
        scheduler = optim.lr_scheduler.StepLR(
            optimizer,
            step_size=EMBEDDING_TRAINING_CONFIG['scheduler_step_size'],
            gamma=EMBEDDING_TRAINING_CONFIG['scheduler_gamma'],
        )

    os.makedirs(DATA_CONFIG['checkpoint_dir'], exist_ok=True)


    # Sélectionne la bonne fonction d'entraînement selon la config, puis l'appelle avec les bons arguments :
    train_fn = {
        'contrastive': train_model,
        'triplet': train_model_triplet,
        'batch_hard_triplet': train_model_triplet_hard_negative,
        'supcon': train_model_supcon,
        'simple': train_model_triplet_hard_negative,  # même format EmbeddingDataset, phase 1 minimale
    }[loss_type]

    # appel de la fonction d'entraînement qui retourne le modèle entraîné (et l'historique de perte, pas utilisé ici mais utile pour debug ou courbes d'apprentissage)
    embedding_model, _ = train_fn(
        embedding_model,
        train_dataloader,
        optimizer,
        loss_fn,
        device,
        epochs=EMBEDDING_TRAINING_CONFIG['epochs'],
        scheduler=scheduler,
        checkpoint_dir=DATA_CONFIG['checkpoint_dir'],
    )

    logging.info("Embedding model training completed!\n")

    # À la fin, le modèle est sauvegardé dans embedding_model.pth : 
    if results_dir is not None:
        model_save_path = os.path.join(results_dir, 'embedding_model.pth')
        torch.save(embedding_model.state_dict(), model_save_path)
        logging.info(f"Embedding model saved to {model_save_path}")

    return embedding_model


# ============================================================================
# PHASE 2: CLASSIFICATION
# ============================================================================

# Construit et entraîne un classifieur (SeizureClassifier) qui prend en entrée les embeddings appris en Phase 1
def train_classifier_head(embedding_model, train_dataloader, test_dataloader, device, results_dir):
    """Train the classifier head on learned embeddings."""
    logging.info("=" * 80)
    logging.info("PHASE 2: Training Seizure Classifier")
    logging.info("=" * 80)

    classifier = SeizureClassifier(
        embedding_model=embedding_model,
        embedding_dim=MODEL_CONFIG['embedding_dim'],
        num_classes=MODEL_CONFIG['num_classes'],
        hidden_dims=MODEL_CONFIG['classifier_hidden_dims'],
    ).to(device)

    # On peut choisir de geler le modèle d'embedding (seul le classifieur apprend) 
    # ou de fine-tuner aussi tout ou partie du modèle d'embedding
    if not CLASSIFIER_TRAINING_CONFIG['freeze_embeddings']:
        n_blocks = CLASSIFIER_TRAINING_CONFIG.get('unfreeze_last_n_blocks', None)
        if n_blocks is None:
            logging.info("Unfreezing ALL embedding layers for fine-tuning...")
        else:
            logging.info(f"Unfreezing last {n_blocks} residual blocks for fine-tuning...")
        classifier.unfreeze_embedding_model(last_n_blocks=n_blocks)

    # Class-weighted loss to handle seizure imbalance
    all_labels = [] # va contenir tous les labels du train_dataloader 
    for batch in train_dataloader: # train_dataloader renvoie des batchs
        # chaque batch est une tuple mais sa longueur varie selon le dataset utilisé (contrastive, triplet, ou simple) :
        if len(batch) == 3: # Batch à 3 éléments : (x1, x2, labels) ou (x1, labels, patient_ids)
            _, labels, _ = batch
        elif len(batch) == 4: # Batch à 4 éléments : (x1, x2, x3, labels) ou (x1, x2, labels, patient_ids)
            _, _, _, labels = batch
        elif len(batch) == 5: # Batch à 5 éléments : (x1, x2, x3, labels, patient_ids)
            _, _, _, labels, _ = batch

        # dans tous les cas, ignore le reste (_) et récupère les labels, puis les ajoute à la liste all_labels 
        all_labels.extend(labels.numpy())
        #  all_labels = [0, 0, 0, 1, 0, 0, 1, ...]
        

    class_counts = np.bincount(all_labels, minlength=2)
    if class_counts[1] == 0:
        # Fold sans crise : pas de pondération possible
        logging.info("Class weights: désactivé (aucune crise dans le train)")
        criterion = nn.CrossEntropyLoss()
    else:
        # Le weighted sampler gère déjà le déséquilibre dans les batchs.
        # On plafonne le ratio à 50:1 pour éviter que la loss pousse le modèle
        # à tout prédire positif (double correction = collapse en recall=1, precision~1%).
        ratio = min(class_counts[0] / class_counts[1], 50.0)
        class_weights = torch.tensor([1.0, ratio], dtype=torch.float32).to(device)
        logging.info(f"Class weights: [1.0, {ratio:.1f}] (plafonné à 50:1)")
        criterion = nn.CrossEntropyLoss(weight=class_weights)
    # paramètre weight : multiplie la perte par le poids de la vraie classe
    optimizer = optim.Adam( # configure l'optimiseur pour n'entraîner que les paramètres du classifieur ou aussi ceux de l'embedding selon la config
        classifier.parameters() if not CLASSIFIER_TRAINING_CONFIG['freeze_embeddings'] # freeze_embeddings = False → fine-tuning complet : tous les paramètres sont entraînés 
        # entraîne tout le modèle (backbone + embedding + classifieur)
        else classifier.classifier.parameters(), # freeze_embeddings = True → n'entraîne que le classifieur : seuls les paramètres du classifieur sont entraînés, ceux de l'embedding sont gelés
        # entraine seulement tête de classification (pratique piur contrastive learning + head fine-tuning, transfer learning)
        lr=CLASSIFIER_TRAINING_CONFIG['learning_rate'],
    )

    scheduler = None
    if CLASSIFIER_TRAINING_CONFIG['use_scheduler']:
        scheduler = optim.lr_scheduler.StepLR(
            optimizer,
            step_size=CLASSIFIER_TRAINING_CONFIG['scheduler_step_size'],
            gamma=CLASSIFIER_TRAINING_CONFIG['scheduler_gamma'],
        )

    classifier, train_history = train_classifier(
        classifier, train_dataloader, optimizer, criterion, device,
        epochs=CLASSIFIER_TRAINING_CONFIG['epochs'], scheduler=scheduler,
    )
    logging.info("Classifier training completed!\n")

    plot_training_history(
        train_history,
        save_path=os.path.join(results_dir, 'classifier_training_history.png'),
    )

    # Sauvegarder le modèle d'embedding (finetuné) pour que eval.py prenne 
    # les derniers poids plutot que le checkpoint de la phase 1 pre finetuné 
    if not CLASSIFIER_TRAINING_CONFIG['freeze_embeddings']:
        finetuned_path = os.path.join(results_dir, 'embedding_model_finetuned.pth')
        torch.save(classifier.embedding_model.state_dict(), finetuned_path)
        logging.info(f"Fine-tuned embedding saved to {finetuned_path}")

    return classifier


# ============================================================================
# MAIN
# ============================================================================

def main():
    # Configure le système de logs
    # getattr(logging, 'INFO') par exemple récupère dynamiquement la constante logging.INFO depuis une string dans la config
    logging.basicConfig(
        level=getattr(logging, LOGGING_CONFIG['level']),
        format=LOGGING_CONFIG['format'],
    )

    # 1. choix du device (GPU ou CPU) selon la config et la disponibilité du GPU
    # Si la config demande le GPU et qu'un GPU est disponible sur la machine, on l'utilise (ex: cuda:0)
    # Sinon on tombe sur le CPU. Tout le reste du code utilise cette variable device pour savoir où envoyer les tenseurs.
    device = (
        torch.device(f"cuda:{DEVICE_CONFIG['cuda_device']}")
        if DEVICE_CONFIG['use_cuda'] and torch.cuda.is_available()
        else torch.device('cpu')
    )
    logging.info(f"Using device: {device}\n")

    # Si aucun dossier n'est spécifié dans la config, on en crée un automatiquement avec un timestamp (ex: results_20240315_143022)
    # exist_ok=True évite une erreur si le dossier existe déjà.
    if DATA_CONFIG['results_dir'] is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_dir = f"./results_{timestamp}" # 2. création du dossier de résultats (timestampé si non spécifié)
    else:
        results_dir = DATA_CONFIG['results_dir']

    os.makedirs(results_dir, exist_ok=True)
    logging.info(f"Results will be saved to: {results_dir}\n")

    loss_type = EMBEDDING_TRAINING_CONFIG['loss_type']
    n_folds = BASELINE_CONFIG['n_folds']
    all_fold_metrics = []

    # ============================================================================
    # CROSS-VALIDATION : chaque pli passe une fois en test, les autres en train
    # ============================================================================
    for fold in range(n_folds):
        logging.info("")
        logging.info("=" * 80)
        logging.info(f"FOLD {fold + 1}/{n_folds}  (pli de test = {fold})")
        logging.info("=" * 80)

        fold_dir = os.path.join(results_dir, f'fold_{fold}')
        os.makedirs(fold_dir, exist_ok=True)

        # input_dim peut varier si des features constantes diffèrent d'un pli à l'autre ;
        # on le réinitialise pour forcer la re-détection à chaque pli.
        MODEL_CONFIG['input_dim'] = None

        # 3. Chargement des données pour ce pli
        # Deux appels : un pour l'embedding (format adapté à la loss), un pour le classifieur et l'UMAP (samples simples)
        train_dataset_embedding, _, weights, _ = load_single_patient_dataset(
            csv_path=DATA_CONFIG['data_path'],
            loss_type=loss_type,
            test_fold=fold,
        )
        train_dataset_classifier, test_dataset_classifier, _, pid_map = load_single_patient_dataset(
            csv_path=DATA_CONFIG['data_path'],
            loss_type='simple',
            test_fold=fold,
        )

        # Save config AFTER input_dim has been auto-detected
        save_config_to_json(fold_dir)

        # 4. Création des DataLoaders pour l'embedding
        if loss_type == 'supcon':
            n_unique_patients = len(torch.unique(train_dataset_embedding.patient_ids))
            effective_pk_p = min(EMBEDDING_TRAINING_CONFIG['pk_p'], n_unique_patients)
            if effective_pk_p < EMBEDDING_TRAINING_CONFIG['pk_p']:
                logging.info(
                    f"pk_p réduit de {EMBEDDING_TRAINING_CONFIG['pk_p']} à {effective_pk_p} "
                    f"(seulement {n_unique_patients} patient(s) dans le train)"
                )
            pk_sampler = PKBatchSampler(
                train_dataset_embedding.patient_ids,
                P=effective_pk_p,
                K=EMBEDDING_TRAINING_CONFIG['pk_k'],
                labels=train_dataset_embedding.labels,
                max_per_patient=EMBEDDING_TRAINING_CONFIG.get('max_per_patient'),
            )
            train_dataloader_embedding = DataLoader(
                train_dataset_embedding,
                batch_sampler=pk_sampler,
            )
        elif DATA_CONFIG['use_weighted_sampling'] and loss_type != 'contrastive':
            sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)
            train_dataloader_embedding = DataLoader(
                train_dataset_embedding,
                batch_size=EMBEDDING_TRAINING_CONFIG['batch_size'],
                sampler=sampler,
                drop_last=True,
            )
        else:
            train_dataloader_embedding = DataLoader(
                train_dataset_embedding,
                batch_size=EMBEDDING_TRAINING_CONFIG['batch_size'],
                shuffle=True,
                drop_last=True,
            )

        # 5. Dataloaders pour le classifieur et le UMAP
        train_dataloader_classifier = DataLoader(
            train_dataset_classifier,
            batch_size=CLASSIFIER_TRAINING_CONFIG['batch_size'],
            shuffle=True,
        )
        test_dataloader_classifier = DataLoader(
            test_dataset_classifier,
            batch_size=CLASSIFIER_TRAINING_CONFIG['batch_size'],
            shuffle=False,
        )
        train_dataloader_umap = DataLoader(train_dataset_classifier, batch_size=128, shuffle=False)
        test_dataloader_umap = DataLoader(test_dataset_classifier, batch_size=128, shuffle=False)

        # PHASE 1 : embedding
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
            )

        # PHASE 2 : classifieur
        classifier = train_classifier_head(
            embedding_model,
            train_dataloader_classifier,
            test_dataloader_classifier,
            device,
            fold_dir,
        )

        if EVAL_CONFIG['generate_umap'] and not CLASSIFIER_TRAINING_CONFIG['freeze_embeddings']:
            generate_umap_visualizations(
                classifier.embedding_model, train_dataloader_umap, device, fold_dir,
                sample_size=EVAL_CONFIG['umap_sample_size'],
                prefix='umap_finetuned',
                test_dataloader=test_dataloader_umap,
                pid_map=pid_map,
            )

        # Évaluation de ce pli
        _, test_metrics = eval_classifier_head(
            classifier,
            train_dataloader_classifier,
            test_dataloader_classifier,
            device,
            fold_dir,
        )

        if test_metrics is not None:
            roc_auc_str = f"{test_metrics['roc_auc']:.4f}" if test_metrics['roc_auc'] is not None else 'N/A'
            logging.info(f"Fold {fold} — Accuracy: {test_metrics['accuracy']:.4f} | "
                         f"F1: {test_metrics['f1']:.4f} | "
                         f"ROC-AUC: {roc_auc_str}")
            all_fold_metrics.append({
                'fold': fold,
                'accuracy': test_metrics['accuracy'],
                'precision': test_metrics['precision'],
                'recall': test_metrics['recall'],
                'f1': test_metrics['f1'],
                'roc_auc': test_metrics['roc_auc'],
            })

    # ============================================================================
    # RÉSUMÉ CROSS-VALIDATION
    # ============================================================================
    logging.info("")
    logging.info("=" * 80)
    logging.info("CROSS-VALIDATION TERMINÉE")
    logging.info("=" * 80)

    if all_fold_metrics:
        accs    = [m['accuracy'] for m in all_fold_metrics]
        f1s     = [m['f1'] for m in all_fold_metrics]
        recalls = [m['recall'] for m in all_fold_metrics]
        aucs    = [m['roc_auc'] for m in all_fold_metrics if m['roc_auc'] is not None]

        logging.info(f"Accuracy  — mean: {np.mean(accs):.4f}  std: {np.std(accs):.4f}  "
                     f"min: {np.min(accs):.4f}  max: {np.max(accs):.4f}")
        logging.info(f"Recall    — mean: {np.mean(recalls):.4f}  std: {np.std(recalls):.4f}  "
                     f"min: {np.min(recalls):.4f}  max: {np.max(recalls):.4f}")
        logging.info(f"F1        — mean: {np.mean(f1s):.4f}  std: {np.std(f1s):.4f}  "
                     f"min: {np.min(f1s):.4f}  max: {np.max(f1s):.4f}")
        if aucs:
            logging.info(f"ROC-AUC   — mean: {np.mean(aucs):.4f}  std: {np.std(aucs):.4f}  "
                         f"min: {np.min(aucs):.4f}  max: {np.max(aucs):.4f}")

        # Sauvegarde du résumé en JSON
        summary = {
            'n_folds': n_folds,
            'per_fold': all_fold_metrics,
            'mean_accuracy': float(np.mean(accs)),
            'std_accuracy': float(np.std(accs)),
            'mean_recall': float(np.mean(recalls)),
            'std_recall': float(np.std(recalls)),
            'mean_f1': float(np.mean(f1s)),
            'std_f1': float(np.std(f1s)),
            'mean_roc_auc': float(np.mean(aucs)) if aucs else None,
            'std_roc_auc': float(np.std(aucs)) if aucs else None,
        }
        summary_path = os.path.join(results_dir, 'cv_summary.json')
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=4)
        logging.info(f"\nRésumé cross-validation sauvegardé dans {summary_path}")

        plot_cv_summary(summary, save_path=os.path.join(results_dir, 'cv_summary.png'))
        logging.info(f"Visualisation cross-validation sauvegardée dans {results_dir}/cv_summary.png")


if __name__ == "__main__":
    main()
