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
import numpy as np
import os
import sys
import json
from datetime import datetime

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from model import DeepResidualEmbeddingModel, SeizureClassifier
from loss import ContrastiveLoss, TripletLoss, BatchHardTripletLoss, SupervisedContrastiveLoss
from dataset import ContrastiveDataset, TripletDataset, EmbeddingDataset, PKBatchSampler
from train import (
    train_model,
    train_model_triplet,
    train_model_triplet_hard_negative,
    train_model_supcon,
    train_classifier,
)
from visualization import plot_training_history
from eval import generate_umap_visualizations, eval_classifier_head
from config import (
    DATA_CONFIG,
    MODEL_CONFIG,
    EMBEDDING_TRAINING_CONFIG,
    CLASSIFIER_TRAINING_CONFIG,
    EVAL_CONFIG,
    DEVICE_CONFIG,
    LOGGING_CONFIG,
)


# ============================================================================
# CONFIGURATION SAVING
# ============================================================================

# sauvegarde tous les hyperparamètres (confugs de données, modèle, entraînement)
# dans un fichier training_config.json dans le dossier résultats
# utile pour la reproductibilité des expériences 

def save_config_to_json(results_dir):
    config_dict = {
        'data_config': DATA_CONFIG,
        'model_config': MODEL_CONFIG,
        'embedding_training_config': EMBEDDING_TRAINING_CONFIG,
        'classifier_training_config': CLASSIFIER_TRAINING_CONFIG,
        'eval_config': EVAL_CONFIG,
        'device_config': DEVICE_CONFIG,
        'logging_config': LOGGING_CONFIG,
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    config_path = os.path.join(results_dir, 'training_config.json')
    with open(config_path, 'w') as f:
        json.dump(config_dict, f, indent=4)
    logging.info(f"Configuration saved to {config_path}")
    return config_path


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
        pmean = baseline[feature_cols].mean() # moyenne de la baseline (calcule moyenne colonne par colonne)
        # pmean est une serie indexée par features_cols
        pstd = baseline[feature_cols].std().fillna(1).replace(0, 1) # écart type de la baseline
        # calcule écart type par feature
        # fillna(1) : NaN quand toutes les valeurs de la feature sont NaN pour ce patient (capteur absent)
        # replace(0, 1) : feature constante → std=0, on remplace par 1 pour éviter division par 0
        df.loc[grp.index, feature_cols] = (grp[feature_cols] - pmean) / pstd # transforme les valeurs du patients en z-score
        # z = x - mu_baseline / sigma_baseline
    return df

# fonction de chargement principale 
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
    drop_cols = [c for c in DATA_CONFIG['drop_columns'] if c in df.columns] # on récupère les colonnes à drop du fichier config
    avail_cols = [c for c in DATA_CONFIG.get('availability_cols', []) if c in df.columns]

    df.drop(columns=drop_cols, inplace=True) # nettoyage des colonne inutiles
    # inplace = true c'est pour que la suppression soit faite directement dans df, pas de nouveau dataframe crée

    # Split train/test by patient ID (assignment decided in config, not a data column)
    patient_split = DATA_CONFIG['patient_split']
    train_df = df[df[pid_col].isin(patient_split['train'])]
    test_df = df[df[pid_col].isin(patient_split['test'])]
    logging.info(f"Train patients: {patient_split['train']} → {len(train_df)} samples")
    logging.info(f"Test patients: {patient_split['test']} → {len(test_df)} samples")

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
        # si per_patient_normalization == True → le bloc if est exécuté
        # si False ou absent → le bloc est ignoré
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
    # SOUS-ÉCHANTILLONNAGE DES NON-CRISES (train uniquement)
    # ============================================================================
    train_features_np = train_df[feature_columns].to_numpy()
    train_labels_np   = train_df[label_col].to_numpy()
    train_pids_np     = train_df[pid_col].to_numpy()
    train_features_np = np.nan_to_num(train_features_np, nan=0.0, posinf=0.0, neginf=0.0)

    unique, counts = np.unique(train_labels_np, return_counts=True)
    logging.info(f"Class distribution (no resampling): {dict(zip(unique.tolist(), counts.tolist()))}")

    undersampling_ratio = DATA_CONFIG.get('undersampling_ratio', None)
    if undersampling_ratio is not None:
        seizure_idx     = np.where(train_labels_np == 1)[0]
        non_seizure_idx = np.where(train_labels_np == 0)[0]
        n_target = min(len(seizure_idx) * undersampling_ratio, len(non_seizure_idx))
        rng = np.random.default_rng(42)
        selected_non_seizure_idx = np.sort(rng.choice(non_seizure_idx, size=n_target, replace=False))
        keep_idx = np.sort(np.concatenate([selected_non_seizure_idx, seizure_idx]))
        train_features_np = train_features_np[keep_idx]
        train_labels_np   = train_labels_np[keep_idx]
        train_pids_np     = train_pids_np[keep_idx]
        n_sz     = (train_labels_np == 1).sum()
        n_non_sz = (train_labels_np == 0).sum()
        ratio    = n_non_sz / n_sz if n_sz > 0 else float('inf')
        unique_r, counts_r = np.unique(train_labels_np, return_counts=True)
        logging.info(f"Class distribution (after subsampling): {dict(zip(unique_r.tolist(), counts_r.tolist()))} | ratio non-crise/crise : {ratio:.1f}:1")

    # Baseline logistique : diagnostic pour vérifier si le signal est discriminant
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score as _roc_auc_score
    test_features_np = np.nan_to_num(test_df[feature_columns].to_numpy(), nan=0.0, posinf=0.0, neginf=0.0)
    test_labels_np   = test_df[label_col].to_numpy()
    if len(np.unique(test_labels_np)) >= 2:
        lr = LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42)
        lr.fit(train_features_np, train_labels_np)
        lr_auc = _roc_auc_score(test_labels_np, lr.predict_proba(test_features_np)[:, 1])
        logging.info(f"[Baseline LR] ROC-AUC sur le test : {lr_auc:.4f}")

    # Convert to tensors — encode patient IDs to integers.
    # Test reuses the train mapping so that the same patient always gets the same code.
    train_pid_codes, pid_map = _encode_patient_ids(pd.Series(train_pids_np))
    test_pid_codes, pid_map  = _encode_patient_ids(test_df[pid_col], pid_map=pid_map)

    train_features     = torch.tensor(train_features_np, dtype=torch.float32)
    train_labels       = torch.tensor(train_labels_np, dtype=torch.long)
    train_patient_ids  = torch.tensor(train_pid_codes, dtype=torch.long)

    test_features = torch.tensor(test_features_np, dtype=torch.float32)
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
# PHASE 1: EMBEDDING TRAINING
# ============================================================================

# Phase 1 — entraînement du modèle d'embedding avec la loss choisie (contrastive, triplet, batch hard triplet, ou supcon)
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
        

    class_counts = np.bincount(all_labels) # np.bincount compte le nombre d’occurrences de chaque entier
    # class_counts = [5, 2] pour l'exemple au dessus
    class_weights = torch.tensor(1.0 / class_counts, dtype=torch.float32).to(device) # inverse de la fréquence 
    # classe fréquente : petit poids, classe rare : gros poids
    # exemple : class_counts = [5000, 200], class_weights = [0.0002, 0.005]
    # Une erreur sur une crise sera beaucoup plus pénalisée
    
    logging.info(f"Class weights: {class_weights.cpu().numpy()}")

    criterion = nn.CrossEntropyLoss(weight=class_weights) # standard pour la classification multi-classe
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

    # Save the (possibly fine-tuned) embedding model so eval.py picks up the
    # latest weights rather than the pre-fine-tune Phase 1 checkpoint.
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


# 3. Chargement des données (deux fois avec des formats différents : une pour l'embedding, une pour le classifieur)

    # Dataset for embedding training (also auto-detects input_dim)
    # premier chargement produit un dataset au format requis par la loss (paires pour contrastive, triplets pour triplet, etc.)
    train_dataset_embedding, _, weights, _ = load_train_test_dataset( # Le _ ignore le test dataset embedding qui ne servira pas.
        csv_path=DATA_CONFIG['data_path'],
        loss_type=loss_type,
    )

    # Dataset for classifier training and UMAP (always plain features/labels/pids)
    # second charge toujours des samples simples (features, label, patient_id) — c'est ce dont le classifieur et le UMAP ont besoin
    train_dataset_classifier, test_dataset_classifier, _, pid_map = load_train_test_dataset(
        csv_path=DATA_CONFIG['data_path'],
        loss_type='simple',
    )

    ########### Pourquoi charger deux fois ? 
    ###### Le dataset pour l'embedding peut avoir un format spécial (paires, triplets), alors que le classifieur a toujours besoin de samples simples.

    # Save config AFTER input_dim has been auto-detected
    save_config_to_json(results_dir)
    # Fait après le chargement des données parce que load_train_test_dataset peut modifier MODEL_CONFIG['input_dim'] si celui-ci est None (auto-détection du nombre de features)
    # On veut sauvegarder la valeur finale, pas None.


    # 4. Création des DataLoaders pour l'embedding
    # Pour supcon : PKBatchSampler (P patients × K samples par batch)
    # Sinon : WeightedRandomSampler ou shuffle simple

    # Dataloader for embedding training
    if loss_type == 'supcon':
        # Pour supcon, on utilise un sampler custom PKBatchSampler qui garantit que chaque batch contient exactement P patients × K samples
        # C'est critique pour SupCon : il faut que chaque ancre ait des positifs de la même classe dans le batch, issus de patients différents
        pk_sampler = PKBatchSampler(
            train_dataset_embedding.patient_ids,
            P=EMBEDDING_TRAINING_CONFIG['pk_p'],
            K=EMBEDDING_TRAINING_CONFIG['pk_k'],
            labels=train_dataset_embedding.labels,  # class-balanced batches for SupCon
            max_per_patient=EMBEDDING_TRAINING_CONFIG.get('max_per_patient'),
        )
        train_dataloader_embedding = DataLoader(
            train_dataset_embedding,
            batch_sampler=pk_sampler,
        )
    elif DATA_CONFIG['use_weighted_sampling'] and loss_type != 'contrastive':
        # ContrastiveDataset generates pairs internally; WeightedRandomSampler doesn't apply.
        # Pour les autres losses avec weighted sampling activé : 
        # WeightedRandomSampler tire les samples avec une probabilité inversement proportionnelle à la taille du patient 
        # les patients avec peu de données sont vus plus souvent
        # Pas applicable à contrastive car ContrastiveDataset génère ses paires en interne
        sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)
        train_dataloader_embedding = DataLoader(
            train_dataset_embedding,
            batch_size=EMBEDDING_TRAINING_CONFIG['batch_size'],
            sampler=sampler,
        )
    else: # Cas par défaut : shuffle aléatoire classique.
        train_dataloader_embedding = DataLoader(
            train_dataset_embedding,
            batch_size=EMBEDDING_TRAINING_CONFIG['batch_size'],
            shuffle=True,
        )


    # 5. Dataloaders pour le classifieur et le UMAP (4 distincts)
    train_dataloader_classifier = DataLoader( # Train classifieur avec shuffle (important pour l'entraînement)
        train_dataset_classifier,
        batch_size=CLASSIFIER_TRAINING_CONFIG['batch_size'],
        shuffle=True,
    )
    test_dataloader_classifier = DataLoader( # Test classifieur sans shuffle (les métriques ne dépendent pas de l'ordre, mais on veut la reproductibilité)
        test_dataset_classifier,
        batch_size=CLASSIFIER_TRAINING_CONFIG['batch_size'],
        shuffle=False,
    )

    # Les deux UMAP utilisent le même dataset que le classifieur 
    # mais avec un batch size fixe de 128 et sans shuffle 
    # on veut projeter tous les points dans un ordre stable pour les visualisations

    train_dataloader_umap = DataLoader(
        train_dataset_classifier,
        batch_size=128,
        shuffle=False,
    )
    test_dataloader_umap = DataLoader(
        test_dataset_classifier,
        batch_size=128,
        shuffle=False,
    )
    ########################################################################################

    # PHASE 1: Train embedding model
    embedding_model = train_embedding_model(
        # Lance l'entraînement contrastif/triplet/supcon
        train_dataloader_embedding, device, results_dir=results_dir,
        train_labels=train_dataset_embedding.labels, # train_labels est passé uniquement pour le calcul des class weights dans le cas supcon
        # Retourne le modèle entraîné et le sauvegarde dans embedding_model.pth.
    )

    # UMAP (fit on train, project test into same space)
    # Si activé dans la config, génère une visualisation UMAP de l'espace d'embedding avant le fine-tuning du classifieur
    # Le prefix umap_pretrain permet de distinguer ce fichier du UMAP post-fine-tuning
    # Le UMAP est fitté sur le train et le test est projeté dans le même espace.
    if EVAL_CONFIG['generate_umap']:
        generate_umap_visualizations(
            embedding_model, train_dataloader_umap, device, results_dir,
            sample_size=EVAL_CONFIG['umap_sample_size'],
            prefix='umap_pretrain',
            test_dataloader=test_dataloader_umap,
            pid_map=pid_map,
        )

    # PHASE 2: Train classifier
    # Construit et entraîne le SeizureClassifier par-dessus l'embedding
    # Selon la config freeze_embeddings, les poids de l'embedding sont gelés ou fine-tunés
    classifier = train_classifier_head(
        embedding_model,
        train_dataloader_classifier,
        test_dataloader_classifier,
        device,
        results_dir,
    )

    # Post-finetuning UMAP (only when embeddings were unfrozen — otherwise identical to pretrain)
    # Généré seulement si les embeddings ont été dégel és pendant le fine-tuning 
    # sinon l'espace d'embedding n'a pas changé et le UMAP serait identique au précédent
    if EVAL_CONFIG['generate_umap'] and not CLASSIFIER_TRAINING_CONFIG['freeze_embeddings']:
        generate_umap_visualizations(
            classifier.embedding_model, train_dataloader_umap, device, results_dir,
            sample_size=EVAL_CONFIG['umap_sample_size'],
            prefix='umap_finetuned',
            test_dataloader=test_dataloader_umap,
            pid_map=pid_map,
        )

    # Evaluate classifier
    # Calcule accuracy, F1, ROC-AUC sur train et test. Le _ ignore les métriques train (on ne les loggue pas)
    # Sauvegarde aussi des figures de visualisation dans results_dir.
    _, test_metrics = eval_classifier_head(
        classifier,
        train_dataloader_classifier,
        test_dataloader_classifier,
        device,
        results_dir,
    )

    logging.info("=" * 80)
    logging.info("TRAINING COMPLETE!")
    logging.info("=" * 80)

    # Affiche les métriques finales. ROC-AUC est dans un if séparé car elle peut être None si le classifieur ne sort qu'une seule classe sur le test 
    # (cas pathologique mais possible sur de petits splits)
    if test_metrics is not None:
        logging.info(f"Final Test Accuracy: {test_metrics['accuracy']:.4f}")
        logging.info(f"Final Test F1 Score: {test_metrics['f1']:.4f}")
        if test_metrics['roc_auc'] is not None:
            logging.info(f"Final Test ROC-AUC: {test_metrics['roc_auc']:.4f}")
    logging.info(f"\nAll results saved to: {results_dir}")
    logging.info("Generated files: training_config.json, embedding_model.pth, "
                 "classifier_training_history.png, *_evaluation_results.png, umap_plot_*.png")

    return test_metrics


if __name__ == "__main__":
    main()
