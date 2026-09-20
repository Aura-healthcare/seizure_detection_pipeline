"""
eval.py — Evaluate a trained embedding model.

Launch:
    python eval.py --results_dir ./results_20240101_120000 --model ./checkpoints/model_9.pth

The model .pth can also be dropped directly in results_dir and will be auto-detected.

Computes (from PLAN.md):
  - kNN classifier          : train embeddings as gallery, test as query
  - Linear probe            : logistic regression on frozen train embeddings → test
  - LOPO                    : Leave-One-Patient-Out linear probe on full dataset
  - Retrieval               : Precision@K, Recall@K, mAP  (cosine similarity)
  - UMAP visualizations     : colored by label and by patient_id

All metrics are saved to <results_dir>/eval_metrics.json.
"""

import argparse
import base64
import json
import logging
import os
import re
import sys

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    roc_auc_score,
    confusion_matrix,
)
import faiss
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import EmbeddingDataset
from model import DeepResidualEmbeddingModel
from visualization import visualize_umap, plot_evaluation_results, print_evaluation_metrics
from config import EVAL_CONFIG


# ============================================================================
# DATA LOADING
# ============================================================================

def _encode_patient_ids(series, pid_map=None):
    """Map patient IDs to integer codes, reusing an existing mapping if provided."""
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


def _resolve_data_path(data_cfg):
    """Return the CSV path from data_cfg, reconstructing it from AURA_DATA_DIR if missing."""
    if "data_path" in data_cfg:
        return data_cfg["data_path"]
    data_dir = os.environ.get("AURA_DATA_DIR", "")
    if not data_dir:
        raise RuntimeError(
            "data_path absent du training_config.json et AURA_DATA_DIR non défini."
        )
    return os.path.join(data_dir, data_cfg["dataset_name"])


def _per_patient_normalize(df, feature_cols, pid_col, label_col):
    """Z-score each feature relative to the patient's non-seizure baseline."""
    df = df.copy()
    for pid, grp in df.groupby(pid_col):
        baseline = grp[grp[label_col] == 0]
        if len(baseline) < 2:
            baseline = grp
        pmean = baseline[feature_cols].mean()
        pstd = baseline[feature_cols].std().fillna(1).replace(0, 1)
        df.loc[grp.index, feature_cols] = (grp[feature_cols] - pmean) / pstd
    return df


def load_data(config):
    """
    Reload train/test splits using the exact same preprocessing as training.
    Returns numpy arrays: train_feats, train_labels, train_pids, test_feats, test_labels, test_pids.
    """
    data_cfg = config["data_config"]
    pid_col = data_cfg.get("patient_id_col", "patient_id")
    label_col = data_cfg.get("label_col", "label")
    split_col = data_cfg.get("split_col", "split")
    avail_cols = data_cfg.get("availability_cols", [])

    df = pd.read_csv(_resolve_data_path(data_cfg))
    drop_cols = [c for c in data_cfg["drop_columns"] if c in df.columns]
    df.drop(columns=drop_cols, inplace=True)

    train_df = df[df[split_col] == data_cfg["train_split_name"]].drop(columns=[split_col])
    test_df = df[df[split_col] == data_cfg["test_split_name"]].drop(columns=[split_col])

    if data_cfg["num_patients_train"] is not None:
        pids = train_df[pid_col].unique()
        train_df = train_df[train_df[pid_col].isin(pids[: data_cfg["num_patients_train"]])]

    non_feature = [label_col, pid_col] + [c for c in avail_cols if c in train_df.columns]
    feature_cols = [c for c in train_df.columns if c not in non_feature]

    # Drop constant features
    stds = train_df[feature_cols].std()
    constant = stds[stds == 0].index.tolist()
    if constant:
        feature_cols = [c for c in feature_cols if c not in constant]
        train_df.drop(columns=constant, inplace=True)
        test_df.drop(columns=constant, inplace=True)

    # Fill NaN
    train_df[feature_cols] = train_df[feature_cols].fillna(0)
    test_df[feature_cols] = test_df[feature_cols].fillna(0)

    # Normalisation
    if data_cfg.get("per_patient_normalization", False):
        train_df = _per_patient_normalize(train_df, feature_cols, pid_col, label_col)
        test_df = _per_patient_normalize(test_df, feature_cols, pid_col, label_col)
        train_df[feature_cols] = train_df[feature_cols].fillna(0)
        test_df[feature_cols] = test_df[feature_cols].fillna(0)
    else:
        mean = train_df[feature_cols].mean()
        std = train_df[feature_cols].std().replace(0, 1)
        train_df[feature_cols] = (train_df[feature_cols] - mean) / std
        test_df[feature_cols] = (test_df[feature_cols] - mean) / std

    def split_to_tensors(split_df, pid_map=None):
        feats = torch.tensor(split_df[feature_cols].to_numpy(), dtype=torch.float32)
        labels = torch.tensor(split_df[label_col].to_numpy(), dtype=torch.long)
        codes, pid_map = _encode_patient_ids(split_df[pid_col], pid_map=pid_map)
        pids = torch.tensor(codes, dtype=torch.long)
        return feats, labels, pids, pid_map

    train_feats, train_labels, train_pids, pid_map = split_to_tensors(train_df)
    test_feats, test_labels, test_pids, _ = split_to_tensors(test_df, pid_map=pid_map)
    return train_feats, train_labels, train_pids, test_feats, test_labels, test_pids


def load_data_baseline(config, test_fold):
    """Charge les données en mode single-patient avec split temporel par fold.

    Réplique exactement la logique de load_single_patient_dataset() utilisée à
    l'entraînement : même découpage chronologique, même normalisation.
    """
    data_cfg = config["data_config"]
    baseline_cfg = config.get("baseline_config", {})
    n_folds = baseline_cfg.get("n_folds", 4)

    pid_col = data_cfg.get("patient_id_col", "patient_id")
    label_col = data_cfg.get("label_col", "label")
    split_col = data_cfg.get("split_col", "training-split")
    avail_cols = data_cfg.get("availability_cols", [])

    df = pd.read_csv(_resolve_data_path(data_cfg))
    drop_cols = [c for c in data_cfg["drop_columns"] if c in df.columns]
    df.drop(columns=drop_cols, inplace=True)
    if split_col in df.columns:
        df.drop(columns=[split_col], inplace=True)

    # Split temporel identique à load_single_patient_dataset
    n = len(df)
    fold_size = n // n_folds
    fold_idx = np.minimum(np.arange(n) // fold_size, n_folds - 1)
    train_df = df[fold_idx != test_fold].copy()
    test_df = df[fold_idx == test_fold].copy()

    logging.info(f"Baseline split: train={len(train_df)}, test={len(test_df)} (test_fold={test_fold})")

    non_feature = [label_col, pid_col] + [c for c in avail_cols if c in train_df.columns]
    feature_cols = [c for c in train_df.columns if c not in non_feature]

    stds = train_df[feature_cols].std()
    constant = stds[stds == 0].index.tolist()
    if constant:
        feature_cols = [c for c in feature_cols if c not in constant]
        train_df.drop(columns=constant, inplace=True)
        test_df.drop(columns=constant, inplace=True)

    train_df[feature_cols] = train_df[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0)
    test_df[feature_cols] = test_df[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0)

    if data_cfg.get("per_patient_normalization", False):
        baseline = train_df[train_df[label_col] == 0][feature_cols]
        if len(baseline) < 2:
            baseline = train_df[feature_cols]
        pmean = baseline.mean()
        pstd = baseline.std().fillna(1).replace(0, 1)
        train_df[feature_cols] = (train_df[feature_cols] - pmean) / pstd
        test_df[feature_cols] = (test_df[feature_cols] - pmean) / pstd
        train_df[feature_cols] = train_df[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0)
        test_df[feature_cols] = test_df[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0)
    else:
        mean = train_df[feature_cols].mean()
        std = train_df[feature_cols].std().replace(0, 1)
        train_df[feature_cols] = (train_df[feature_cols] - mean) / std
        test_df[feature_cols] = (test_df[feature_cols] - mean) / std
        train_df[feature_cols] = train_df[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0)
        test_df[feature_cols] = test_df[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0)

    def split_to_tensors(split_df, pid_map=None):
        feats = torch.tensor(split_df[feature_cols].to_numpy(), dtype=torch.float32)
        labels = torch.tensor(split_df[label_col].to_numpy(), dtype=torch.long)
        codes, pid_map = _encode_patient_ids(split_df[pid_col], pid_map=pid_map)
        pids = torch.tensor(codes, dtype=torch.long)
        return feats, labels, pids, pid_map

    train_feats, train_labels, train_pids, pid_map = split_to_tensors(train_df)
    test_feats, test_labels, test_pids, _ = split_to_tensors(test_df, pid_map=pid_map)
    return train_feats, train_labels, train_pids, test_feats, test_labels, test_pids


def load_split_metadata(config, n_folds=1, test_fold=0):
    """Reload raw train/test split metadata aligned with embedding extraction order."""
    data_cfg = config["data_config"]
    pid_col = data_cfg.get("patient_id_col", "patient_id")
    label_col = data_cfg.get("label_col", "label")
    split_col = data_cfg.get("split_col", "split")
    seizure_id_col = "seizure-id"

    try:
        df = pd.read_csv(_resolve_data_path(data_cfg))
    except Exception as exc:
        logging.warning("Impossible de relire les métadonnées UMAP: %s", exc)
        return None, None

    if n_folds > 1:
        n = len(df)
        fold_size = n // n_folds
        fold_idx = np.minimum(np.arange(n) // fold_size, n_folds - 1)
        train_df = df[fold_idx != test_fold].copy()
        test_df = df[fold_idx == test_fold].copy()
    else:
        if split_col in df.columns:
            train_df = df[df[split_col] == data_cfg["train_split_name"]].copy()
            test_df = df[df[split_col] == data_cfg["test_split_name"]].copy()
        else:
            train_df = df.copy()
            test_df = df.copy()

    if data_cfg.get("num_patients_train") is not None:
        train_pids = train_df[pid_col].unique()
        train_df = train_df[train_df[pid_col].isin(train_pids[: data_cfg["num_patients_train"]])].copy()

    def _standardize(split_df):
        split_df = split_df.reset_index(drop=False).rename(columns={"index": "sample_id"})
        meta = pd.DataFrame(
            {
                "sample_id": split_df["sample_id"],
                "patient_id": split_df[pid_col].astype(str),
                "label": split_df[label_col].astype(int),
            }
        )
        if seizure_id_col in split_df.columns:
            meta["seizure_id"] = split_df[seizure_id_col]
        else:
            meta["seizure_id"] = pd.NA
        return meta.reset_index(drop=True)

    return _standardize(train_df), _standardize(test_df)


def _build_pid_map_from_metadata(*pid_meta_pairs):
    """Build a code -> original patient_id mapping from aligned arrays and metadata."""
    pid_map = {}
    for pid_codes, meta in pid_meta_pairs:
        if meta is None or len(meta) != len(pid_codes):
            continue
        for code, patient_id in zip(pid_codes, meta["patient_id"].astype(str).to_numpy()):
            pid_map.setdefault(int(code), patient_id)
    return pid_map or None


# ============================================================================
# EMBEDDING EXTRACTION
# ============================================================================

def extract_embeddings(model, features, labels, pids, device, batch_size=512):
    """Run the embedding model in eval mode and return (embeddings, labels, pids) as numpy arrays."""
    dataset = EmbeddingDataset(features, labels, pids)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    model.eval()
    all_embs, all_labels, all_pids = [], [], []

    with torch.no_grad():
        for batch_features, batch_labels, batch_pids in tqdm(loader, desc="Extracting embeddings", leave=False):
            embs = model(batch_features.to(device))
            all_embs.append(embs.cpu().numpy())
            all_labels.append(batch_labels.numpy())
            all_pids.append(batch_pids.numpy())

    return (
        np.concatenate(all_embs),
        np.concatenate(all_labels),
        np.concatenate(all_pids),
    )


# ============================================================================
# kNN CLASSIFIER
# ============================================================================

def _eval_knn_faiss(train_embs, train_labels, test_embs, test_labels, k=5, nprobe=32):
    """Fast approximate kNN using FAISS IVF index (cosine on L2-normed vectors)."""
    # L2-normalize so inner product == cosine similarity
    # la normalisation L2 c'est diviser chaque vecteur par sa propre norme
    # resultat ; tous les vecteurs ont une norme de 1, on a effacé l'info de longueur, on ne garde que la direction
    # formule habituelle de la similarité cosinus c'est A.B / (norme de A x nome de B)
    # sauf que puisqu'on a norme de A et norme de B = 1, le dénominateur vaut 1 et disparait 
    # DONC le produit scalaire A.B correspond directement au cosinus de l'angle
    train_normed = train_embs / (np.linalg.norm(train_embs, axis=1, keepdims=True) + 1e-8) # + 1e-8 pour éviter division par 0
    test_normed = test_embs / (np.linalg.norm(test_embs, axis=1, keepdims=True) + 1e-8)

    train_normed = np.ascontiguousarray(train_normed.astype(np.float32))
    test_normed = np.ascontiguousarray(test_normed.astype(np.float32))

    d = train_normed.shape[1] # d : dimension des vecteurs d'embedding
    n_train = train_normed.shape[0]
    nlist = int(np.sqrt(n_train)) # nlist c'est le nombre de centroides/cluster
    # on prend la racine du nombre de points d'entrainement 
    # ex: √2500 = 50 clusters

    quantizer = faiss.IndexFlatIP(d) # cette ligne crée un quantizer (quantififcateur) qui utilise une recherche exacte par produit scalaire (Inner Product) pour comparer un vecteur aux centroides
    # IndexFlatIP signifie : "Pour trouver le centroïde le plus proche d'un vecteur donné, calcule exactement le produit scalaire entre ce vecteur et TOUS les centroïdes existants"
    # Il ne trouve pas les centroïdes lui-même, il définit seulement la méthode de quantification (recherche exacte via IP). Les centroïdes n'existent pas encore à ce stade !

    # PREPARATION DE L'INDEX FAISS
    index = faiss.IndexIVFFlat(quantizer, d, nlist, faiss.METRIC_INNER_PRODUCT) # cette ligne crée un index FAISS pour la recherche de voisins les plus proches rapide
        # index IndexIVFFlat : c'est un index de type IVF (Inverted File) 
       
    # ENTRAINEMENT DE L'INDEX : FAISS va apprendre les centroides des clusters à partir des données d'entraînement
    index.train(train_normed)
    # Utilise un algorithme comme k-means en interne pour partitionner les données en nlist clusters et calculer les positions des centroïdes (les "centres" de chaque cluster)
    # Après train(), les centroïdes sont définis et stockés dans l'index
    # initialisation : tire nlist vecteurs aléatoires comme centroïdes initiaux, puis itère :
    # 1. assignation : chaque vecteur de train_normed est assigné au centroïde le plus proche (en termes de produit scalaire)
    # 2. mise à jour : chaque centroïde est recalculé comme la moyenne des vecteurs qui lui sont assignés
    # Répète jusqu'à convergence ou un nombre max d'itérations. À la fin, on a nlist centroïdes qui représentent les clusters de l'espace d'embedding.

    # FAISS connait mtn les centroides après le train mais l'index est vide
    # add() permet de ranger chaque vecteur de train_normed dans le bon cluster : 
    # pour chq vecteur, FAISS calcule à quel centroide il est le plus proche et le place dans la liste de ce cluster
    # cluster C1 → [vecteur_4, vecteur_17, vecteur_82, ...]
    # cluster C2 → [vecteur_1, vecteur_9,  vecteur_33, ...]
    # cluster C3 → [vecteur_6, vecteur_44, vecteur_71, ...]
    index.add(train_normed)

    # ATTENTION : les vecteurs de test (test_normed) ne sont jamais ajoutés à l'index
    # Ils ne servent qu'en entrée de .search() — ce sont les requêtes 
    # Les vecteurs de train sont ceux qu'on cherche à retrouver

    index.nprobe = nprobe # index.search() va chercher uniquement dans les listes des nprobe clusters les plus proches de la requête, pas dans tous les vecteurs
    logging.info(f"FAISS IVF index: nlist={nlist}, nprobe={nprobe}")

    distances, indices = index.search(test_normed, k) # prend chaque vecteur de test et cherche ses 5 voisins les plus proches parmi tous les vecteurs de train indexés
    # Elle retourne deux matrices : 
    # indices : Ce sont les positions dans train_normed. indices[0][0] = 4 signifie que le voisin le plus proche du point de test 0 est train_normed[4]
    # exemple : [[  4,  17,  82,   9,  33 ], ...]   ← les 5 voisins du point de test 0

    # distances : Ce sont les scores de similarité cosinus (produit scalaire entre vecteurs normés), triés du plus proche au plus loin
    # Une valeur proche de 1.0 = très similaire, proche de 0 = peu similaire
    # [[ 0.98,  0.91,  0.87,  0.82,  0.79 ], ...]
    # MAIS distances pas utilisé dans la prédiction

    # Majority vote
    neighbor_labels = train_labels[indices]  # (n_test, k)
    preds = np.array([np.bincount(row.astype(int), minlength=2).argmax() for row in neighbor_labels])
    # preds : "Quelle est la classe ?"" bincount compte les votes par classe, puis argmax prend l'index du maximum
    # exemple : 1 vote pour la classe 0, 4 votes pour la classe 1

    # Probability estimate (fraction of positive neighbors)
    probs = neighbor_labels.astype(float).mean(axis=1)
    # probs : "À quel point est-on sûr ?" On convertit les étiquettes en float (0.0 ou 1.0) et on prend la moyenne. 
    # C'est simplement la fraction de voisins positifs. exemple : (1+1+1+1+0)/5 = 0.80.

    return preds, probs


def eval_knn(train_embs, train_labels, test_embs, test_labels, k=5):
    """kNN: use train embeddings as gallery, classify test queries by majority vote.
    Uses FAISS IVF index for fast approximate cosine search."""
    preds, probs = _eval_knn_faiss(train_embs, train_labels, test_embs, test_labels, k)

    precision, recall, f1, _ = precision_recall_fscore_support(
        test_labels, preds, average="binary", zero_division=0
    )
    try:
        roc_auc = float(roc_auc_score(test_labels, probs))
    except ValueError:
        roc_auc = None

    return {
        "accuracy": float(accuracy_score(test_labels, preds)),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "roc_auc": roc_auc,
        "_test_preds": preds,
    }


# ============================================================================
# LINEAR PROBE
# ============================================================================

def fit_linear_probe(train_embs, train_labels):
    """Train the logistic regression used as linear probe."""

    clf = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)
    clf.fit(train_embs, train_labels)
    return clf


def eval_linear_probe(clf, train_embs, test_embs, test_labels):
    """Evaluate a fitted linear probe on test and expose train/test predictions."""

    train_preds = clf.predict(train_embs)

    preds = clf.predict(test_embs) # prédit les labels des embeddings de test en utilisant le classifieur entraîné
    # preds contient les prédictions finales (0 ou 1 selon le modèle binaire)


    probs = clf.predict_proba(test_embs)[:, 1]
    # probs contient les probabilités prédites pour la classe positive (classe 1) pour chaque échantillon de test
    # predict_proba renvoie unematrice de probabilités à 2 colonnes, on prend la colonne de la classe positive


    precision, recall, f1, _ = precision_recall_fscore_support(
        test_labels, preds, average="binary", zero_division=0
    ) # Calcule précision, rappel et score F1 en comparant les labels réels test_labels aux prédictions preds.
    try:
        roc_auc = float(roc_auc_score(test_labels, probs))
    except ValueError:
        roc_auc = None

    return {
        "accuracy": float(accuracy_score(test_labels, preds)),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "roc_auc": roc_auc,
        "_test_preds": preds,
        "_train_preds": train_preds,
    }


# ============================================================================
# LOPO — LEAVE-ONE-PATIENT-OUT
# ============================================================================

def eval_lopo(all_embs, all_labels, all_pids): # embeddings pour tous les échantillons (train + test), labels er patient_ids correspondants
    """
    For each patient: train a logistic regression on all other patients' embeddings,
    evaluate on the held-out patient. Uses the full dataset (train + test combined)
    so every patient is evaluated at least once.

    Returns macro metrics (concatenated predictions across all folds) and per-patient metrics.
    """
    unique_pids = np.unique(all_pids) # extraire liste des patients uniaques présents dans all_pids
    # chaque patent sera traité comme un fold de validation à part entière : on va faire autant de folds que de patients,
    # et à chaque fois on laisse un patient de côté pour le test et on entraîne sur les autres

    per_patient = {} # initialiser un dictionnaire vide pour stocker les métriques par patient

    concat_preds, concat_true = [], []
    # initialise deux listes vides pour stocker les prédictions concaténées et les labels réels concaténés de tous les patients
    # elles serviront à calculer des métroques globales (macro) à la fin, en traitant tous les échantillons comme un seul ensemble une fois tous les folds traités

    for pid in tqdm(unique_pids, desc="LOPO"): # parcourt chaque patient unique et affiche une barre de progression
        mask_test = all_pids == pid # crée un masque booléen pour identifier les échantillons appartenant au patient actuel (pid) 
        # ce masque servira à sélectionner la part de test du fold, composée de tous les échantillons qui appartiennent au patient actuel


        mask_train = ~mask_test # masque inverse pour sélectionner tous les échantillons sauf ceux du patient actuel
        # ce sera la part d'entrainement du fold, composée de tous les échantillons qui n'appartiennent pas au patient actuel

        # Skip degenerate folds
        if mask_train.sum() == 0 or mask_test.sum() == 0: # ignore les folds invalides où il n'y a pas d'échantillon de train ou de test
            continue
        if len(np.unique(all_labels[mask_train])) < 2: # ignore les folds si ne contient pas les 2 classes
            continue

        clf = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42) # crée un classifieur de régression logistique avec les mêmes paramètres que pour le linear probe classique
        
        clf.fit(all_embs[mask_train], all_labels[mask_train]) # entraîne le modèle sur les embeddings et labels de tous les autres patients
        # mask_train sélectionne les échantillons d'entraînement pour ce pli LOPO.


        preds = clf.predict(all_embs[mask_test]) # Prédit les labels sur les embeddings du patient tenu à l'écart.
        # mask_test sélectionne les exemples du patient testé.

        concat_preds.extend(preds) # ajoute les prédictions de ce pli à la liste globale concat_preds.
        concat_true.extend(all_labels[mask_test]) # ajoute les labels vrais de ce pli à la liste globale concat_true

        # Per-patient metrics only when both classes are present in the test fold (otherwise metrics will be misleading)
        if len(np.unique(all_labels[mask_test])) >= 2: # vérifie si le patient testé a des exemples des deux classes dans son ensemble de test.
            p, r, f1, _ = precision_recall_fscore_support(
                all_labels[mask_test], preds, average="binary", zero_division=0
            )
            per_patient[int(pid)] = { # stocke les métriques calculées pour ce patient dans le dictionnaire per_patient
                "accuracy": float(accuracy_score(all_labels[mask_test], preds)),
                "precision": float(p),
                "recall": float(r),
                "f1": float(f1),
                "n_samples": int(mask_test.sum()), # inclut aussi le nombre d'échantillons de test du patient (n_samples)

            }

    concat_preds = np.array(concat_preds)
    concat_true = np.array(concat_true)

    # calcule les métriques globales (macro) en traitant toutes les prédictions et tous les labels concaténés comme un seul ensemble
    # cela donne une mesure d'ensemble sur tous les folds LOPO
    macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(
        concat_true, concat_preds, average="binary", zero_division=0
    )

    # retourne un dictionnaire avec métriques globaux
    return {
        "macro_accuracy": float(accuracy_score(concat_true, concat_preds)),
        "macro_precision": float(macro_p),
        "macro_recall": float(macro_r),
        "macro_f1": float(macro_f1),
        "num_patients_evaluated": len(per_patient),
        "per_patient": per_patient,
    }


# ============================================================================
# RETRIEVAL METRICS
# ============================================================================

def eval_retrieval(query_embs, query_labels, gallery_embs, gallery_labels, k_values=(5, 10, 20)):
    # query_embs : embeddings des exemples de requête (généralement les données de test)
    # gallery_embs : embeddings de la galerie (généralement les données d'entraînement)
    # k_values : liste des valeurs de K pour calculer Precision@K et Recall@K
    """
    For each query (test sample), rank the gallery (train samples) by cosine similarity.
    A gallery item is "relevant" if it shares the same label as the query.

    Uses FAISS for top-K retrieval to avoid building the full similarity matrix.

    Metrics (computed per query, then averaged):
      - Precision@K: fraction of the top-K retrieved items that share the query's label.
        E.g. P@5=0.8 means 4 out of 5 nearest neighbors have the same label.
      - Recall@K: fraction of ALL same-label items in the gallery that appear in the top-K.
        E.g. if there are 1000 seizure items in the gallery and 3 appear in top-20, R@20=0.003.
      - mAP@K: average precision over the ranked top-K list — rewards relevant items
        appearing earlier in the ranking.
    """
    max_k = max(k_values) # # k_values=(5,10,20) → max_k = 20
    # détermine la valeur maximale de K demandée pour la recherche FAISS
    # sert à rechercher le top-K le plus grand en une seule fois 

    logging.info(f"Retrieval: normalizing {len(query_embs)} queries and {len(gallery_embs)} gallery vectors...")
    q_norm = query_embs / (np.linalg.norm(query_embs, axis=1, keepdims=True) + 1e-8) # normalise chaque vecteur pour que sa norme soit 1
    g_norm = gallery_embs / (np.linalg.norm(gallery_embs, axis=1, keepdims=True) + 1e-8) # + 1e-8 évite la division par zéro si un vecteur est nul

    q_norm = np.ascontiguousarray(q_norm.astype(np.float32))
    g_norm = np.ascontiguousarray(g_norm.astype(np.float32))

    d = g_norm.shape[1] # récupère la dimension de chaque vecteur d'embedding

    n_gallery = len(g_norm) # nombre d'éléments dans la galerie
    nlist = int(np.sqrt(n_gallery)) # nombre de clusters pour l'index FAISS, typiquement la racine carrée du nombre d'éléments de la galerie
    nprobe = 32 # nombre de clusters à explorer 
    logging.info(f"Retrieval: building FAISS IVF index (dim={d}, gallery={n_gallery}, nlist={nlist}, nprobe={nprobe})...")
    
    quantizer = faiss.IndexFlatIP(d) # Crée un quantizer FAISS qui compare les vecteurs par produit scalaire (Inner Product)
    # ce quantizer sera utilisé pour assigner les vecteurs aux clusters

    index = faiss.IndexIVFFlat(quantizer, d, nlist, faiss.METRIC_INNER_PRODUCT) # Crée un index IVF plat qui utilise le quantizer défini précédemment, avec la métrique de similarité par produit scalaire
    
    index.train(g_norm) # entraîne l'index FAISS sur les vecteurs de la galerie, FAISS apprend les centroïdes des clusters à partir de ces vecteurs
    
    index.add(g_norm) # ajoute les vecteurs de la galerie à l'index après entraînement 

    index.nprobe = nprobe # définit combien de clusters FAISS doit explorer pendant la recherche
    # Plus nprobe est grand, plus la recherche est précise, mais plus elle est lente
    logging.info(f"Retrieval: searching top-{max_k} for {len(q_norm)} queries...")

    _, indices = index.search(q_norm, max_k) # recherche les max_k voisins les plus proches dans la gallerie pour chaque requête
    # on ignore les distances on ne garde que les indices 
    # exemple : indices.shape → (100, 20)
    # indices[0] → [4821, 312, 7043, 18, 9901, ...]   ← positions des 20 voisins de la requête 0
    logging.info("Retrieval: FAISS search done, computing metrics...")

    # Pre-count relevant items per query label for recall
    label_counts = {}
    for lbl in np.unique(query_labels): # calcule combien d'éléments pertinents existent dans la galerie pour chaque label de requête.
        label_counts[lbl] = int((gallery_labels == lbl).sum())
    # exemple : label_counts = {
    # 0: 6000,   ← 6000 items de label 0 dans la gallery
    # 1: 4000    ← 4000 items de label 1 dans la gallery
    # }
    # on fait ce calcul une seule fois plutot que pour chaque requête 
    # (ces totaux sont utilisés pour normaliser recall et map)

    # initialise des listes pour stocker scores par requêtes :
    prec_at_k = {k: [] for k in k_values}
    rec_at_k = {k: [] for k in k_values}
    ap_list = []

    ##### IMPORTANT
    # pour chaque requête i, récupère les labels des max_k plus proches voisins trouvés
    for i in range(len(query_labels)):
        top_k_labels = gallery_labels[indices[i]]
        # on recupere les vrais labels des 20 voisins trouvés par FAISS pour la requête i
        # suite exemple pour la requête i = 0 (label =1)
        # indices[0]     = [4821, 312, 7043, 18, 9901, 55, 302, ...]
        # top_k_labels   = [1,    0,   1,    1,  0,    1,  0,   ...]


        relevance = (top_k_labels == query_labels[i]).astype(float) 
        # on compare chaque label voisin avec le label de la requête. 1.0 = même label (pertinent), 0.0 = label différent
        # suite example : 
        # relevance = [1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0, ...]

        n_relevant = label_counts.get(query_labels[i], 0)
        # c'est le nombre d'élements dans la gallerie qui ont le même label que notre requete i 
        # dans notre exemple, on a dit que query_labels[0] = 1, donc le premier query est de label positif alors n_relevant = label_counts[1] = 4000
        # si le label n'existe pas dans la galerie, 0 est renvoyé 

        if n_relevant == 0:
            continue
        # Si le label de cette requête n'existe pas du tout dans la gallery, on saute — les métriques seraient indéfinies.

        for k in k_values: #  k = 5, 10, 20
            top_k = relevance[:k]   # on tronque le vecteur de pertinence à k
            prec_at_k[k].append(top_k.mean())
            rec_at_k[k].append(top_k.sum() / n_relevant)

            # exemple
            # avec relevance = [1, 0, 1, 1, 0, 1, 0, 1, 0, 1, ...] et n_relevant = 4000
            # pour k = 5 : Precision@5 = mean([1,0,1,1,0]) = (1 + 0 + 1 + 1 + 0) / 5 = 3/5 = 0.6 
            # Recall@5    = sum([1,0,1,1,0])/4000 = 3/4000 = 0.00075 
            
            # pour k = 10   
            # Precision@10 = 6/10  = 0.60
            # Recall@10    = 6/4000 = 0.0015
            # Le recall est très petit car même 6 pertinents sur 4000 c'est peu. C'est normal et attendu

        # Average Precision (truncated at max_k)
        cumsum = np.cumsum(relevance)
        ranks = np.arange(1, max_k + 1)
        ap = ((cumsum / ranks) * relevance).sum() / n_relevant
        ap_list.append(ap)

        # exemple 
        # Prenons relevance = [1, 0, 1, 1, 0, 0, 1, 0, 0, 0, ...] sur 10 positions :
        # ranks     = [1,  2,  3,  4,  5,  6,  7,  8,  9, 10, ...]
        # cumsum    = [1,  1,  2,  3,  3,  3,  4,  4,  4,  4, ...] (somme  cumulative quand relevance = 1)
        # cumsum/ranks = [1.0, 0.5, 0.67, 0.75, 0.6, 0.5, 0.57, ...]


        # (cumsum/ranks) * relevance : on ne garde que les positions où relevance=1 (les pertinents) :
        # positions pertinentes : rang 1, 3, 4, 7
        # precisions à ces rangs : 1.0,  0.67, 0.75, 0.57
        # AP = moyenne de ces précisions = (1.0 + 0.67 + 0.75 + 0.57) / 4000 (n_relevant)

        # IDEE PRINCIPALE :  si les pertinents arrivent tôt (rangs 1, 2, 3), les précisions sont élevées (1.0, 1.0, 1.0) → AP proche de 1. 
        # S'ils arrivent tard (rangs 18, 19, 20), les précisions sont faibles (0.05, 0.10, 0.15) → AP proche de 0.

    results = {}
    for k in k_values:
        results[f"precision@{k}"] = float(np.mean(prec_at_k[k])) if prec_at_k[k] else 0.0
        results[f"recall@{k}"] = float(np.mean(rec_at_k[k])) if rec_at_k[k] else 0.0
    results["mAP"] = float(np.mean(ap_list)) if ap_list else 0.0
    # On fait la moyenne de chaque métrique sur toutes les requêtes. 
    # exemple : si on avait 100 requêtes, precision@5 est la moyenne des 100 P@5 individuels.
    # C'est la mean de mAP qui lui donne son nom (mean Average Precision).

    return results


# ============================================================================
# CLASSIFIER EVALUATION
# ============================================================================

def evaluate_classifier(classifier, dataloader, device):
    """Run the classifier on a dataloader and return metrics dict."""
    classifier.eval()
    all_preds, all_labels, all_probs = [], [], []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating", leave=False):
            features, labels, _ = batch
            features, labels = features.to(device), labels.to(device)
            logits = classifier(features)
            probs = torch.softmax(logits, dim=1)
            all_preds.extend(torch.argmax(logits, dim=1).cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs[:, 1].cpu().numpy())

    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average="binary", zero_division=0
    )
    try:
        roc_auc = float(roc_auc_score(all_labels, all_probs))
    except ValueError:
        roc_auc = None
        logging.warning("ROC-AUC could not be computed (only one class present)")

    return {
        "accuracy": float(accuracy_score(all_labels, all_preds)),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "roc_auc": roc_auc,
        "confusion_matrix": confusion_matrix(all_labels, all_preds),
        "predictions_probs": np.array(all_probs),
        "true_labels": np.array(all_labels),
    }


def eval_classifier_head(classifier, train_dataloader, test_dataloader, device, results_dir):
    """Evaluate the trained classifier on train and test sets and save plots."""
    train_metrics = None
    if EVAL_CONFIG["evaluate_on_train"]:
        logging.info("Evaluating classifier on training set...")
        train_metrics = evaluate_classifier(classifier, train_dataloader, device)
        print_evaluation_metrics(train_metrics, dataset_name="Train")
        plot_evaluation_results(
            train_metrics, train_metrics["predictions_probs"], train_metrics["true_labels"],
            save_path=os.path.join(results_dir, "train_evaluation_results.png"),
        )

    test_metrics = None
    if EVAL_CONFIG["evaluate_on_test"]:
        logging.info("Evaluating classifier on test set...")
        test_metrics = evaluate_classifier(classifier, test_dataloader, device)
        print_evaluation_metrics(test_metrics, dataset_name="Test")
        plot_evaluation_results(
            test_metrics, test_metrics["predictions_probs"], test_metrics["true_labels"],
            save_path=os.path.join(results_dir, "test_evaluation_results.png"),
        )

    return train_metrics, test_metrics


# ============================================================================
# UMAP VISUALIZATION
# ============================================================================

def _extract_from_dataloader(embedding_model, dataloader, device, sample_size=None):
    """Extract embeddings, labels, patient_ids from a dataloader."""
    embedding_model.eval()
    encoded_data, labels, patient_ids = [], [], []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Generating embeddings", leave=False):
            if len(batch) == 3:       # EmbeddingDataset
                features, label, patient_id = batch
            elif len(batch) == 4:     # ContrastiveDataset
                features, _, _, label = batch
                patient_id = torch.zeros_like(label)
            elif len(batch) == 5:     # TripletDataset
                features, _, _, label, patient_id = batch
            else:
                raise ValueError(f"Unexpected batch format with {len(batch)} elements")

            embeddings = embedding_model(features.to(device))
            encoded_data.extend(embeddings.cpu().numpy())
            labels.extend(label.cpu().numpy())
            patient_ids.extend(patient_id.cpu().numpy())

    encoded_data = np.array(encoded_data)
    labels = np.array(labels)
    patient_ids = np.array(patient_ids)

    if sample_size is not None and len(encoded_data) > sample_size:
        # Keep all seizure samples (up to sample_size), fill the rest with no-seizure.
        # Cap seizure at 10% of sample_size to avoid distorting UMAP neighborhoods.
        minority_idx = np.where(labels == 1)[0]
        majority_idx = np.where(labels != 1)[0]

        max_minority = min(len(minority_idx), sample_size // 10)
        if len(minority_idx) > max_minority:
            minority_idx = np.random.choice(minority_idx, max_minority, replace=False)

        n_majority = sample_size - len(minority_idx)
        if n_majority < len(majority_idx):
            majority_idx = np.random.choice(majority_idx, n_majority, replace=False)

        idx = np.concatenate([minority_idx, majority_idx])
        np.random.shuffle(idx)
        encoded_data, labels, patient_ids = encoded_data[idx], labels[idx], patient_ids[idx]

    return encoded_data, labels, patient_ids


def generate_umap_visualizations(embedding_model, dataloader, device, results_dir,
                                 sample_size=None, prefix='umap', test_dataloader=None,
                                 pid_map=None, plot_patient_ids=True):
    """Generate UMAP plots. UMAP is fit on train data; test is projected into the same space.
    plot_patient_ids=False supprime les plots colorés par patient_id."""
    logging.info(f"Generating UMAP visualizations ({prefix})...")

    encoded_data, labels, patient_ids = _extract_from_dataloader(
        embedding_model, dataloader, device, sample_size)

    if np.isnan(encoded_data).any():
        logging.warning("UMAP ignoré : les embeddings contiennent des NaN "
                        "(le modèle n'a pas convergé — vérifier la loss).")
        return

    test_data, test_labels, test_pids = None, None, None
    if test_dataloader is not None:
        test_data, test_labels, test_pids = _extract_from_dataloader(
            embedding_model, test_dataloader, device, sample_size)
        if test_data is not None and np.isnan(test_data).any():
            logging.warning("Test embeddings contiennent des NaN, projection test ignorée.")
            test_data, test_labels, test_pids = None, None, None

    visualize_umap(encoded_data, labels, patient_ids, results_dir, prefix=prefix,
                   test_data=test_data, test_labels=test_labels, test_patient_ids=test_pids,
                   pid_map=pid_map, plot_patient_ids=plot_patient_ids)
    logging.info("UMAP visualizations completed!\n")


# ============================================================================
# HTML REPORT
# ============================================================================

def _img_tag(path, title=""):
    """Return an <img> tag with the image embedded as base64, or empty string if file missing."""
    if not os.path.exists(path):
        return ""
    with open(path, "rb") as f:
        data = base64.b64encode(f.read()).decode()
    return (
        f'<figure>'
        f'<img src="data:image/png;base64,{data}" alt="{title}">'
        f'<figcaption>{title}</figcaption>'
        f'</figure>'
    )


def _metrics_table(metrics, exclude=("per_patient",)):
    """Render a flat dict of metrics as an HTML table row, skipping excluded keys."""
    rows = ""
    for k, v in metrics.items():
        if k in exclude:
            continue
        if isinstance(v, float):
            rows += f"<tr><td>{k}</td><td>{v:.4f}</td></tr>"
        else:
            rows += f"<tr><td>{k}</td><td>{v}</td></tr>"
    return f'<table><tr><th>Metric</th><th>Value</th></tr>{rows}</table>'


def _config_table(cfg):
    """Render a config dict as an HTML table, skipping nested dicts."""
    rows = ""
    for k, v in cfg.items():
        if not isinstance(v, dict):
            rows += f"<tr><td>{k}</td><td>{v}</td></tr>"
    return f'<table><tr><th>Parameter</th><th>Value</th></tr>{rows}</table>'


def generate_report(results_dir):
    """
    Build a self-contained HTML report from eval_metrics.json, training_config.json,
    and all PNG images found in results_dir. Saved as report.html.
    """
    # Load data
    metrics_path = os.path.join(results_dir, "eval_metrics.json")
    config_path = os.path.join(results_dir, "training_config.json")

    metrics = json.load(open(metrics_path)) if os.path.exists(metrics_path) else {}
    config = json.load(open(config_path)) if os.path.exists(config_path) else {}

    timestamp = config.get("timestamp", "")
    model_cfg = config.get("model_config", {})
    emb_cfg = config.get("embedding_training_config", {})
    clf_cfg = config.get("classifier_training_config", {})

    def section(title, content):
        return f'<section><h2>{title}</h2>{content}</section>'

    def img_row(*paths_titles):
        imgs = "".join(_img_tag(os.path.join(results_dir, p), t) for p, t in paths_titles if p)
        return f'<div class="img-row">{imgs}</div>' if imgs else ""

    def interactive_link(filename, label):
        if not os.path.exists(os.path.join(results_dir, filename)):
            return ""
        return (
            f'<p><a href="{filename}" target="_blank" rel="noopener noreferrer">{label}</a></p>'
        )

    # --- Config section ---
    config_html = (
        "<h3>Model</h3>" + _config_table(model_cfg) +
        "<h3>Embedding Training</h3>" + _config_table(emb_cfg) +
        "<h3>Classifier Training</h3>" + _config_table(clf_cfg)
    )

    # --- Embedding metrics section ---
    emb_metrics_html = ""
    for key, title in [
        ("knn", "kNN Classifier"),
        ("linear_probe", "Linear Probe"),
    ]:
        if key in metrics:
            emb_metrics_html += f"<h3>{title}</h3>" + _metrics_table(metrics[key])

    if "lopo" in metrics:
        lopo = metrics["lopo"]
        macro = {k: v for k, v in lopo.items() if k != "per_patient"}
        emb_metrics_html += "<h3>LOPO (Leave-One-Patient-Out)</h3>" + _metrics_table(macro)

        if lopo.get("per_patient"):
            rows = "".join(
                f"<tr><td>{pid}</td><td>{v['accuracy']:.4f}</td><td>{v['precision']:.4f}</td>"
                f"<td>{v['recall']:.4f}</td><td>{v['f1']:.4f}</td><td>{v['n_samples']}</td></tr>"
                for pid, v in lopo["per_patient"].items()
            )
            emb_metrics_html += (
                "<h4>Per-patient breakdown</h4>"
                "<table><tr><th>Patient</th><th>Accuracy</th><th>Precision</th>"
                "<th>Recall</th><th>F1</th><th>Samples</th></tr>"
                f"{rows}</table>"
            )

    if "retrieval" in metrics:
        emb_metrics_html += "<h3>Retrieval</h3>" + _metrics_table(metrics["retrieval"])

    # --- Images ---
    umap_html = ""
    # Determine which model eval.py used
    finetuned_exists = os.path.exists(os.path.join(results_dir, "embedding_model_finetuned.pth"))
    eval_model_label = "fine-tuned embedding" if finetuned_exists else "Phase 1 embedding (no fine-tuning)"

    for phase_prefix, phase_title in [
        ("umap_pretrain", "Pre-fine-tune (Phase 1 embedding)"),
        ("umap_finetuned", "Post-fine-tune (Phase 2 — unfrozen embedding)"),
        ("umap_eval", f"Eval ({eval_model_label})"),
    ]:
        train_exists = any(os.path.exists(os.path.join(results_dir, f)) for f in (
            f"{phase_prefix}_labels.png", f"{phase_prefix}_patient_ids.png"
        ))
        test_exists = any(os.path.exists(os.path.join(results_dir, f)) for f in (
            f"{phase_prefix}_test_labels.png", f"{phase_prefix}_test_patient_ids.png"
        ))
        if train_exists:
            umap_html += f"<h3>{phase_title} — Train</h3>"
            umap_html += img_row(
                (f"{phase_prefix}_labels.png", "by label"),
                (f"{phase_prefix}_patient_ids.png", "by patient"),
            )
            umap_html += img_row(
                (f"{phase_prefix}_errors.png", "erreurs TP / FP / FN / TN (train)"),
            )
            umap_html += interactive_link(
                f"{phase_prefix}_errors_interactive.html",
                "ouvrir la version interactive des erreurs (train)",
            )
        if test_exists:
            umap_html += f"<h3>{phase_title} — Test (same UMAP space)</h3>"
            umap_html += img_row(
                (f"{phase_prefix}_test_labels.png", "by label"),
                (f"{phase_prefix}_test_patient_ids.png", "by patient"),
            )
            umap_html += img_row(
                (f"{phase_prefix}_test_errors.png", "erreurs TP / FP / FN / TN"),
            )
            umap_html += interactive_link(
                f"{phase_prefix}_test_errors_interactive.html",
                "ouvrir la version interactive des erreurs (test)",
            )
    # --- Seizure error tables (train + test) ---
    def _seizure_table(csv_path):
        if not os.path.exists(csv_path):
            return ""
        try:
            se_df = pd.read_csv(csv_path)
            status_colors = {"missed": "#fdd", "partial": "#ffd", "fully_detected": "#dfd"}
            rows_html = ""
            for _, row in se_df.iterrows():
                bg = status_colors.get(row["status"], "")
                style_attr = f' style="background:{bg}"' if bg else ""
                rows_html += (
                    f"<tr{style_attr}>"
                    f"<td>{row['seizure_id']}</td><td>{row['patient_id']}</td>"
                    f"<td>{row['n_windows']}</td><td>{row['n_TP']}</td><td>{row['n_FN']}</td>"
                    f"<td>{row['detection_rate']:.3f}</td><td><b>{row['status']}</b></td>"
                    f"</tr>"
                )
            return (
                "<table><tr><th>seizure_id</th><th>patient_id</th><th>n_windows</th>"
                "<th>n_TP</th><th>n_FN</th><th>detection_rate</th><th>status</th></tr>"
                f"{rows_html}</table>"
                "<p style='font-size:0.85em;color:#666'>Rouge = manquée · Jaune = partielle · Vert = détectée</p>"
            )
        except Exception:
            return f"<p>Voir <code>{os.path.basename(csv_path)}</code></p>"

    seizure_err_html = ""
    train_tbl = _seizure_table(os.path.join(results_dir, "seizure_errors_train.csv"))
    test_tbl  = _seizure_table(os.path.join(results_dir, "seizure_errors_test.csv"))
    if train_tbl:
        seizure_err_html += "<h3>Train</h3>" + train_tbl
    if test_tbl:
        seizure_err_html += "<h3>Test</h3>" + test_tbl

    classifier_html = img_row(
        ("classifier_training_history.png", "Classifier training history"),
    ) + img_row(
        ("train_evaluation_results.png", "Train — ROC & confusion matrix"),
        ("test_evaluation_results.png", "Test — ROC & confusion matrix"),
    )

    # --- Assemble page ---
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Evaluation Report — {os.path.basename(results_dir)}</title>
<style>
  body {{ font-family: sans-serif; max-width: 1200px; margin: 40px auto; padding: 0 20px; color: #222; }}
  h1 {{ border-bottom: 2px solid #444; padding-bottom: 8px; }}
  h2 {{ margin-top: 40px; border-bottom: 1px solid #ccc; padding-bottom: 4px; color: #333; }}
  h3 {{ margin-top: 24px; color: #555; }}
  h4 {{ margin-top: 16px; color: #666; }}
  table {{ border-collapse: collapse; margin: 12px 0; min-width: 320px; }}
  th, td {{ border: 1px solid #ddd; padding: 7px 14px; text-align: left; }}
  th {{ background: #f4f4f4; font-weight: 600; }}
  tr:nth-child(even) {{ background: #fafafa; }}
  .img-row {{ display: flex; flex-wrap: wrap; gap: 24px; margin: 16px 0; }}
  figure {{ margin: 0; }}
  figcaption {{ text-align: center; font-size: 0.85em; color: #666; margin-top: 6px; }}
  img {{ max-width: 560px; width: 100%; border: 1px solid #ddd; border-radius: 4px; }}
  .meta {{ color: #666; font-size: 0.9em; margin-bottom: 24px; }}
</style>
</head>
<body>
<h1>Evaluation Report</h1>
<p class="meta">Results directory: <code>{results_dir}</code> &nbsp;|&nbsp; Trained: {timestamp}</p>

{section("Training Configuration", config_html)}
{section("Embedding Evaluation Metrics", emb_metrics_html)}
{section("Erreurs par crise (seizure_errors.csv)", seizure_err_html) if seizure_err_html else ""}
{section("UMAP Visualizations", umap_html)}
{section("Classifier Evaluation", classifier_html)}

</body>
</html>"""

    out_path = os.path.join(results_dir, "report.html")
    with open(out_path, "w") as f:
        f.write(html)
    logging.info(f"Report saved to {out_path}")
    return out_path


# ============================================================================
# SEIZURE ERROR REPORT
# ============================================================================

def save_seizure_error_report(config, labels_np, preds, results_dir,
                               n_folds=1, test_fold=0, split="test"):
    """Save seizure_errors_{split}.csv listing per-seizure TP/FN/detection status.

    split="test" ou "train". Relit le CSV brut pour récupérer seizure-id (droppé des features).
    """
    data_cfg = config["data_config"]
    pid_col        = data_cfg.get("patient_id_col", "patient_id")
    label_col      = data_cfg.get("label_col", "label")
    split_col      = data_cfg.get("split_col", "split")
    seizure_id_col = "seizure-id"

    try:
        df = pd.read_csv(_resolve_data_path(data_cfg))
    except Exception as e:
        logging.warning(f"Impossible de lire le CSV pour le rapport seizure : {e}")
        return None

    if seizure_id_col not in df.columns:
        logging.warning(f"Colonne '{seizure_id_col}' absente du CSV — rapport seizure ignoré.")
        return None

    # Reproduire le même découpage que lors du chargement des données
    if n_folds > 1:
        n = len(df)
        fold_size = n // n_folds
        fold_idx = np.minimum(np.arange(n) // fold_size, n_folds - 1)
        if split == "test":
            split_df = df[fold_idx == test_fold].copy().reset_index(drop=True)
        else:
            split_df = df[fold_idx != test_fold].copy().reset_index(drop=True)
    else:
        split_name = data_cfg.get(f"{split}_split_name", split)
        if split_col in df.columns:
            split_df = df[df[split_col] == split_name].copy().reset_index(drop=True)
        else:
            split_df = df.copy().reset_index(drop=True)

    if len(split_df) != len(labels_np):
        logging.warning(
            f"Désalignement ({split}) : split_df={len(split_df)} lignes, labels={len(labels_np)}. "
            "Rapport seizure ignoré."
        )
        return None

    meta = split_df[[pid_col, label_col, seizure_id_col]].copy()
    meta["pred"] = preds

    seizure_meta = meta[meta[label_col] == 1]
    if seizure_meta.empty:
        logging.warning(f"Aucune fenêtre de crise dans le split '{split}' — rapport seizure ignoré.")
        return None

    rows = []
    for sid, grp in seizure_meta.groupby(seizure_id_col, sort=False):
        patient   = grp[pid_col].iloc[0]
        n_windows = len(grp)
        n_tp      = int((grp["pred"] == 1).sum())
        n_fn      = int((grp["pred"] == 0).sum())
        det_rate  = round(n_tp / n_windows, 3) if n_windows > 0 else 0.0
        if n_tp == 0:
            status = "missed"
        elif n_fn == 0:
            status = "fully_detected"
        else:
            status = "partial"
        rows.append({
            "seizure_id":     sid,
            "patient_id":     patient,
            "n_windows":      n_windows,
            "n_TP":           n_tp,
            "n_FN":           n_fn,
            "detection_rate": det_rate,
            "status":         status,
        })

    result_df = pd.DataFrame(rows).sort_values("detection_rate").reset_index(drop=True)
    out_path = os.path.join(results_dir, f"seizure_errors_{split}.csv")
    result_df.to_csv(out_path, index=False)

    n_missed   = (result_df["status"] == "missed").sum()
    n_partial  = (result_df["status"] == "partial").sum()
    n_detected = (result_df["status"] == "fully_detected").sum()
    logging.info(
        f"Rapport seizure ({split}) → {out_path} | "
        f"{n_detected} détectées, {n_partial} partielles, {n_missed} manquées "
        f"({len(result_df)} crises)"
    )
    return out_path


# ============================================================================
# MAIN
# ============================================================================

def _eval_one_fold(args, results_dir):
    """Évalue un seul dossier de fold. Retourne all_results dict."""
    logging.info(f"\n{'='*80}\nÉvaluation : {results_dir}\n{'='*80}")

    # --- Config ---
    config_path = os.path.join(results_dir, "training_config.json")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"training_config.json not found in {results_dir}")
    with open(config_path) as f:
        config = json.load(f)

    # --- Modèle ---
    model_path = args.model
    if model_path is None:
        for candidate in ("embedding_model_finetuned.pth", "embedding_model.pth"):
            candidate_path = os.path.join(results_dir, candidate)
            if os.path.exists(candidate_path):
                model_path = candidate_path
                break
        if model_path is None:
            raise FileNotFoundError(f"No .pth file found in {results_dir}. Pass --model explicitly.")
        logging.info(f"Auto-detected model: {model_path}")

    # --- Device ---
    device_cfg = config.get("device_config", {})
    device = (
        torch.device(f"cuda:{device_cfg.get('cuda_device', 0)}")
        if device_cfg.get("use_cuda", True) and torch.cuda.is_available()
        else torch.device("cpu")
    )
    logging.info(f"Using device: {device}")

    # --- Données ---
    logging.info("Loading data...")
    baseline_cfg = config.get("baseline_config", {})
    n_folds = baseline_cfg.get("n_folds", 1)
    test_fold = 0
    if n_folds > 1:
        fold_match = re.search(r'fold_(\d+)', os.path.abspath(results_dir))
        if fold_match:
            test_fold = int(fold_match.group(1))
            logging.info(f"Baseline single-patient mode: test_fold={test_fold} (inféré depuis le chemin)")
        else:
            test_fold = baseline_cfg.get("test_fold", 0)
            logging.warning(
                f"Impossible d'inférer le fold depuis le chemin ; "
                f"utilisation de baseline_config.test_fold={test_fold}"
            )
        train_feats, train_labels, train_pids, test_feats, test_labels, test_pids = \
            load_data_baseline(config, test_fold)
    else:
        train_feats, train_labels, train_pids, test_feats, test_labels, test_pids = load_data(config)

    # --- Modèle ---
    model_cfg = config["model_config"]
    input_dim = model_cfg.get("input_dim") or train_feats.shape[1]
    model = DeepResidualEmbeddingModel(
        input_dim=input_dim,
        embedding_dim=model_cfg["embedding_dim"],
        num_blocks=model_cfg.get("num_residual_blocks", 3),
        dropout=model_cfg.get("embedding_dropout", 0.2),
    ).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    logging.info(f"Loaded model from {model_path} (input_dim={input_dim})")

    # --- Embeddings ---
    logging.info("Extracting embeddings...")
    train_embs, train_labels_np, train_pids_np = extract_embeddings(
        model, train_feats, train_labels, train_pids, device
    )
    test_embs, test_labels_np, test_pids_np = extract_embeddings(
        model, test_feats, test_labels, test_pids, device
    )
    logging.info(f"Train embeddings: {train_embs.shape}, Test embeddings: {test_embs.shape}")

    train_meta, test_meta = load_split_metadata(config, n_folds=n_folds, test_fold=test_fold)
    if train_meta is not None and len(train_meta) != len(train_labels_np):
        logging.warning(
            "Métadonnées train désalignées: meta=%s, labels=%s. Hover UMAP train désactivé.",
            len(train_meta), len(train_labels_np)
        )
        train_meta = None
    if test_meta is not None and len(test_meta) != len(test_labels_np):
        logging.warning(
            "Métadonnées test désalignées: meta=%s, labels=%s. Hover UMAP test désactivé.",
            len(test_meta), len(test_labels_np)
        )
        test_meta = None

    pid_map = _build_pid_map_from_metadata(
        (train_pids_np, train_meta),
        (test_pids_np, test_meta),
    )

    all_results = {}

    # --- kNN ---
    logging.info(f"\n{'='*60}\nkNN Classifier  (k={args.knn_k})\n{'='*60}")
    knn_metrics = eval_knn(train_embs, train_labels_np, test_embs, test_labels_np, k=args.knn_k)
    knn_test_preds = knn_metrics.pop("_test_preds", None)
    all_results["knn"] = knn_metrics
    for metric, val in knn_metrics.items():
        logging.info(f"  {metric:<12}: {val:.4f}" if val is not None else f"  {metric:<12}: N/A")

    # --- Linear probe ---
    logging.info(f"\n{'='*60}\nLinear Probe\n{'='*60}")
    _null_metrics = {"accuracy": None, "precision": None, "recall": None, "f1": None, "roc_auc": None}
    if len(np.unique(train_labels_np)) < 2:
        logging.warning("Linear probe ignoré : une seule classe dans le train.")
        linear_metrics = _null_metrics
        lp_test_preds  = None
        lp_train_preds = None
    else:
        lp_clf = fit_linear_probe(train_embs, train_labels_np)
        lp_train_preds = lp_clf.predict(train_embs)
        if len(np.unique(test_labels_np)) < 2:
            logging.warning(
                "Métriques linear probe ignorées sur le test : une seule classe dans le test. "
                "Les prédictions train restent disponibles pour le rapport train."
            )
            linear_metrics = _null_metrics
            lp_test_preds = None
        else:
            linear_metrics = eval_linear_probe(lp_clf, train_embs, test_embs, test_labels_np)
            lp_test_preds  = linear_metrics.pop("_test_preds", None)
            lp_train_preds = linear_metrics.pop("_train_preds", lp_train_preds)
    all_results["linear_probe"] = linear_metrics
    for metric, val in linear_metrics.items():
        logging.info(f"  {metric:<12}: {val:.4f}" if val is not None else f"  {metric:<12}: N/A")

    # --- LOPO ---
    logging.info(f"\n{'='*60}\nLOPO (Leave-One-Patient-Out)\n{'='*60}")
    all_embs_lopo = np.concatenate([train_embs, test_embs])
    all_labels_lopo = np.concatenate([train_labels_np, test_labels_np])
    all_pids_lopo = np.concatenate([train_pids_np, test_pids_np])
    lopo_metrics = eval_lopo(all_embs_lopo, all_labels_lopo, all_pids_lopo)
    all_results["lopo"] = lopo_metrics
    logging.info(f"  Patients evaluated : {lopo_metrics['num_patients_evaluated']}")
    logging.info(f"  Macro accuracy     : {lopo_metrics['macro_accuracy']:.4f}")
    logging.info(f"  Macro precision    : {lopo_metrics['macro_precision']:.4f}")
    logging.info(f"  Macro recall       : {lopo_metrics['macro_recall']:.4f}")
    logging.info(f"  Macro F1           : {lopo_metrics['macro_f1']:.4f}")

    # --- Retrieval ---
    logging.info(f"\n{'='*60}\nRetrieval  (K={args.retrieval_k})\n{'='*60}")
    if len(np.unique(train_labels_np)) < 2 or len(np.unique(test_labels_np)) < 2:
        logging.warning("Retrieval ignoré : une seule classe dans le train ou le test.")
        retrieval_metrics = {}
    else:
        retrieval_metrics = eval_retrieval(
            test_embs, test_labels_np, train_embs, train_labels_np, k_values=args.retrieval_k
        )
    all_results["retrieval"] = retrieval_metrics
    for metric, val in retrieval_metrics.items():
        logging.info(f"  {metric:<15}: {val:.4f}")

    # --- Rapport erreurs par crise ---
    test_preds_report  = lp_test_preds  if lp_test_preds  is not None else knn_test_preds
    train_preds_report = lp_train_preds if lp_train_preds is not None else None
    if test_preds_report is not None:
        save_seizure_error_report(
            config, test_labels_np, test_preds_report, results_dir,
            n_folds=n_folds, test_fold=test_fold, split="test",
        )
    if train_preds_report is not None:
        save_seizure_error_report(
            config, train_labels_np, train_preds_report, results_dir,
            n_folds=n_folds, test_fold=test_fold, split="train",
        )

    # --- Sauvegarde ---
    out_path = os.path.join(results_dir, "eval_metrics.json")
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=4)
    logging.info(f"\nMetrics saved to {out_path}")

    # --- UMAP ---
    if not args.no_umap:
        finetuned_exists = os.path.exists(os.path.join(results_dir, "embedding_model_finetuned.pth"))
        umap_prefix = 'umap_eval' if finetuned_exists else 'umap_pretrain'
        logging.info(f"\n{'='*60}\nUMAP Visualizations (prefix={umap_prefix})\n{'='*60}")
        sample = args.umap_sample_size

        def _subsample_keep_minority(embs, labels, pids, n):
            if n is None or len(embs) <= n:
                return embs, labels, pids, np.arange(len(embs))
            minority_idx = np.where(labels == 1)[0]
            majority_idx = np.where(labels != 1)[0]
            max_minority = min(len(minority_idx), n // 10)
            if len(minority_idx) > max_minority:
                minority_idx = np.random.choice(minority_idx, max_minority, replace=False)
            n_majority = n - len(minority_idx)
            if n_majority < len(majority_idx):
                majority_idx = np.random.choice(majority_idx, n_majority, replace=False)
            idx = np.concatenate([minority_idx, majority_idx])
            np.random.shuffle(idx)
            return embs[idx], labels[idx], pids[idx], idx

        train_viz, train_lbl_viz, train_pid_viz, train_viz_idx = _subsample_keep_minority(
            train_embs, train_labels_np, train_pids_np, sample)
        test_viz, test_lbl_viz, test_pid_viz, test_viz_idx = _subsample_keep_minority(
            test_embs, test_labels_np, test_pids_np, sample)

        train_meta_viz = None if train_meta is None else train_meta.iloc[train_viz_idx].reset_index(drop=True)
        test_meta_viz = None if test_meta is None else test_meta.iloc[test_viz_idx].reset_index(drop=True)

        # Prédictions sur le sous-ensemble UMAP — linear probe en priorité, kNN en fallback
        test_preds_viz  = None
        train_preds_viz = None
        if lp_test_preds is not None:
            test_preds_viz = lp_test_preds[test_viz_idx]
        elif knn_test_preds is not None:
            test_preds_viz = knn_test_preds[test_viz_idx]
        if lp_train_preds is not None:
            train_preds_viz = lp_train_preds[train_viz_idx]

        visualize_umap(train_viz, train_lbl_viz, train_pid_viz, results_dir,
                       prefix=umap_prefix,
                       test_data=test_viz, test_labels=test_lbl_viz, test_patient_ids=test_pid_viz,
                       pid_map=pid_map,
                       test_preds=test_preds_viz, train_preds=train_preds_viz,
                       train_metadata=train_meta_viz, test_metadata=test_meta_viz)

    # --- HTML report ---
    generate_report(results_dir)

    return all_results


def _print_cv_summary(fold_results):
    """Affiche un résumé agrégé des métriques kNN et linear probe sur tous les folds."""
    logging.info(f"\n{'='*80}")
    logging.info("RÉSUMÉ CROSS-VALIDATION — métriques test agrégées")
    logging.info(f"{'='*80}")

    for probe_key, probe_label in [("knn", "kNN"), ("linear_probe", "Linear Probe")]:
        metrics_per_fold = [r[probe_key] for r in fold_results.values() if probe_key in r]
        if not metrics_per_fold:
            continue
        logging.info(f"\n  [{probe_label}]")
        for metric in ["accuracy", "precision", "recall", "f1", "roc_auc"]:
            vals = [m[metric] for m in metrics_per_fold if m.get(metric) is not None]
            if vals:
                logging.info(
                    f"    {metric:<12}: mean={np.mean(vals):.4f}  std={np.std(vals):.4f}"
                    f"  min={np.min(vals):.4f}  max={np.max(vals):.4f}"
                )

    # Tableau par fold
    logging.info(f"\n  {'Fold':<8} {'kNN-F1':>8} {'kNN-AUC':>9} {'LP-F1':>8} {'LP-AUC':>9}")
    logging.info(f"  {'-'*46}")
    for fold_dir, res in sorted(fold_results.items()):
        fold_name = os.path.basename(fold_dir)
        knn_f1  = res.get("knn", {}).get("f1")
        knn_auc = res.get("knn", {}).get("roc_auc")
        lp_f1   = res.get("linear_probe", {}).get("f1")
        lp_auc  = res.get("linear_probe", {}).get("roc_auc")
        fmt = lambda v: f"{v:.4f}" if v is not None else "  N/A"
        logging.info(f"  {fold_name:<8} {fmt(knn_f1):>8} {fmt(knn_auc):>9} {fmt(lp_f1):>8} {fmt(lp_auc):>9}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate a trained contrastive embedding model")
    parser.add_argument(
        "--results_dir", required=True,
        help="Dossier contenant training_config.json (fold unique) "
             "ou dossier parent contenant fold_0/, fold_1/... (avec --all_folds)"
    )
    parser.add_argument(
        "--all_folds", action="store_true",
        help="Évalue tous les sous-dossiers fold_X/ trouvés dans --results_dir "
             "et affiche un résumé agrégé."
    )
    parser.add_argument(
        "--model", default=None,
        help="Chemin vers le fichier .pth. Auto-détecté si absent."
    )
    parser.add_argument("--knn_k", type=int, default=5, help="k pour le kNN (défaut: 5)")
    parser.add_argument(
        "--retrieval_k", type=int, nargs="+", default=[5, 10, 20],
        help="Valeurs de K pour les métriques de retrieval (défaut: 5 10 20)"
    )
    parser.add_argument(
        "--umap_sample_size", type=int, default=2000,
        help="Nombre max de points pour le UMAP (défaut: 2000)"
    )
    parser.add_argument("--no_umap", action="store_true", help="Désactive la génération du UMAP")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    if args.all_folds:
        import glob
        fold_dirs = sorted(glob.glob(os.path.join(args.results_dir, "fold_*")))
        fold_dirs = [d for d in fold_dirs if os.path.isdir(d)]
        if not fold_dirs:
            raise FileNotFoundError(
                f"Aucun sous-dossier fold_X/ trouvé dans {args.results_dir}"
            )
        logging.info(f"{len(fold_dirs)} fold(s) trouvé(s) : {[os.path.basename(d) for d in fold_dirs]}")

        fold_results = {}
        for fold_dir in fold_dirs:
            fold_results[fold_dir] = _eval_one_fold(args, fold_dir)

        _print_cv_summary(fold_results)

        # Calcul des métriques agrégées (mean / std sur tous les folds)
        aggregated = {}
        for probe_key in ("knn", "linear_probe", "retrieval"):
            per_fold = [r[probe_key] for r in fold_results.values() if probe_key in r]
            if not per_fold:
                continue
            agg = {}
            all_metric_keys = set(k for m in per_fold for k in m if isinstance(m[k], (int, float)) and m[k] is not None)
            for metric in all_metric_keys:
                vals = [m[metric] for m in per_fold if m.get(metric) is not None]
                if vals:
                    agg[f"mean_{metric}"] = float(np.mean(vals))
                    agg[f"std_{metric}"]  = float(np.std(vals))
            aggregated[probe_key] = agg

        # Sauvegarde du résumé agrégé dans le dossier parent
        summary_path = os.path.join(args.results_dir, "eval_cv_summary.json")
        with open(summary_path, "w") as f:
            json.dump(
                {
                    "aggregated": aggregated,
                    "per_fold": {os.path.basename(k): v for k, v in fold_results.items()},
                },
                f, indent=4
            )
        logging.info(f"\nRésumé cross-validation sauvegardé dans {summary_path}")
    else:
        _eval_one_fold(args, args.results_dir)

    logging.info("\nEvaluation complete!")


if __name__ == "__main__":
    main()
