#!/usr/bin/env python3
"""
Fusionne feat-hrv, feat-acc et les événements de crise en un feat-grid.csv
unifié, par run, pour le dataset SeizeIT2.

Les runs sont associés entre les trois arborescences de fichiers grâce à leur
identifiant BIDS (sub-XXX_ses-YY_task-szMonitoring_run-ZZ), par exemple :
  feats-hrv/<id>_ecg/fast/features/feats_<id>_ecg_fast.csv
  feats-acc/features_ACC_EEG_SD_<id>_mov.csv
  raw-data/.../sub-XXX/ses-YY/eeg/<id>_events.tsv

IMPORTANT : les timestamps des features HRV et du events.tsv sont ancrés sur
le début d'enregistrement anonymisé BIDS (2000-01-01), alors que la colonne
"time" du fichier de features ACC provient de l'en-tête du fichier
*mov.edf*, qui -- contrairement à celui de ecg.edf -- n'a pas été anonymisé
et garde sa vraie date de début d'enregistrement (ex: 2025-02-...). Les deux
fichiers EDF ont la même durée (file_duration), donc ces deux horloges ne
sont pas directement comparables en tant que datetimes absolus, mais chacune
reste une référence valide pour le temps écoulé depuis le début de son
propre fichier. L'alignement se fait donc en secondes écoulées depuis le
début du run :
  - HRV    : interval_start_time (ms) / 1000
  - ACC    : time - time.min(), en secondes
  - events : onset / onset + duration (déjà relatif, en secondes)

Usage:
    python merge_feat_grid.py \
        --hrv-dir   /data2/.../derived-data/feats-hrv \
        --acc-dir   /data2/.../derived-data/feats-acc \
        --raw-dir   /data2/.../raw-data/ds005873-1.1.0 \
        --output-dir output/feat-grid
"""

import argparse
import re
from pathlib import Path

import pandas as pd

# Identifiant complet d'un run BIDS
# Crée un motif pour reconnaître un identifiant de run au format BIDS
# Exemple : sub-001_ses-01_task-szMonitoring_run-03
RUN_ID_RE = re.compile(r"sub-\d+_ses-\d+_task-szMonitoring_run-\d+")


# Même motif mais avec les numéros de sujet/session/run capturés séparément (groupes)
# #our pouvoir reconstruire le chemin vers le events.tsv correspondant
# permet d'extraire séparément le numéro de sujet, de session et de run depuis l'identifiant complet du run
RUN_ID_PARTS_RE = re.compile(r"sub-(\d+)_ses-(\d+)_task-szMonitoring_run-(\d+)")

HRV_TOLERANCE_S = 0.5   # la moitié de la fenêtre glissante HRV (1s)
ACC_TOLERANCE_S = 2.5   # la moitié de la fenêtre glissante ACC (5s)


def parse_args():
    """Déclare et parse les arguments CLI du script."""
    parser = argparse.ArgumentParser(
        description="Fusionne les features HRV, ACC et les événements de crise en une grille temporelle unifiée, pour chaque run du dataset SeizeIT2."
    )
    parser.add_argument("--hrv-dir", required=True, help="Racine des données dérivées feats-hrv")
    parser.add_argument("--acc-dir", required=True, help="Racine des données dérivées feats_acc")
    parser.add_argument("--raw-dir", required=True, help="Racine de l'arborescence raw-data (BIDS) de SeizeIT2")
    parser.add_argument("--output-dir", required=True, help="Dossier de sortie pour les CSV feat-grid par run")
    parser.add_argument("--algo", default="fast", help="Nom du sous-dossier de l'algo de détection QRS pour la HRV (défaut: fast)")
    return parser.parse_args()


def find_hrv_files(hrv_dir: Path, algo: str) -> list:
    """Liste tous les fichiers de features HRV disponibles, tous runs confondus.

    C'est cette liste qui pilote la boucle principale : un run n'est traité
    que si son fichier HRV existe (l'ACC et le events.tsv sont optionnels).
    """
    return sorted(hrv_dir.glob(f"*_ecg/{algo}/features/feats_*_ecg_{algo}.csv"))
# recherche dans le dossier HRV tous les fichiers correpondant au motif 
# classe les résultats par ordre alphabétique et les retourne sous forme de liste


def run_id_from_path(path: Path) -> str:
    """Extrait l'identifiant de run (sub-.../ses-.../run-...) depuis un nom de fichier."""
    match = RUN_ID_RE.search(path.name) # Cherche dans le nom du fichier un motif correspondant à un run BIDS
    if not match:
        raise ValueError(f"Impossible d'extraire un run id depuis {path}") # si rien n'est trouvé, une erreur est levée
    return match.group(0) # retourne la chaîne trouvée (l'identifiant complet du run BIDS)


def find_acc_file(acc_dir: Path, run_id: str) -> Path:
    """Construit le chemin attendu du fichier de features ACC pour ce run."""
    return acc_dir / f"features_ACC_EEG_SD_{run_id}_mov.csv"
# Forme un chemin du type : acc_dir/features_ACC_EEG_SD_sub-001_ses-01..._mov.csv
# (le dossier contient aussi une variante features_ACC_ECGEMG_SD_*, écartée ici)


def find_events_file(raw_dir: Path, run_id: str) -> Path:
    """Construit le chemin attendu du events.tsv BIDS pour ce run."""
    sub, ses, _run = RUN_ID_PARTS_RE.match(run_id).groups() # Extrait les trois parties du run id
    return raw_dir / f"sub-{sub}" / f"ses-{ses}" / "eeg" / f"{run_id}_events.tsv" # Construit le chemin complet du fichier d’événements


def load_hrv(path: Path) -> tuple:
    """Charge les features HRV et calcule le temps écoulé depuis le début du run.

    Retourne (datetime de début du run, dataframe avec la colonne elapsed_s).
    Le datetime de début sert plus tard à reconstruire une colonne
    "timestamp" absolue sur toute la grille de sortie.
    """
    df = pd.read_csv(path) # Lit le fichier CSV avec pandas
    anchor = pd.to_datetime(df["timestamp"].iloc[0]) # Récupère la première valeur de la colonne timestamp

    # interval_start_time est en millisecondes depuis le début du run
    # (0, 1000, 2000, ... pour une fenêtre glissante de 1s) -> on repasse en secondes.
    df["elapsed_s"] = df["interval_start_time"] / 1000.0
    # Convertit interval_start_time de millisecondes en secondes
    # on crée une nouvelle colonne elapsed_s


    df = df.drop(columns=["filename", "timestamp"], errors="ignore") # Supprime les colonnes inutiles
    # Préfixe toutes les colonnes de features par "hrv_" pour éviter les collisions
    # avec les colonnes ACC une fois les deux dataframes fusionnés

    df.columns = [f"hrv_{c}" if c != "elapsed_s" else c for c in df.columns]
    # Renomme toutes les colonnes pour éviter les collisions avec les données ACC
    # Exemple : mean_rr devient hrv_mean_rr

    return anchor, df.sort_values("elapsed_s").reset_index(drop=True)
# Retourne : l’ancre temporelle et le DataFrame trié par temps écoulé


def load_acc(path: Path) -> pd.DataFrame:
    """Charge les features ACC sur un axe "secondes écoulées depuis le début du run".

    La colonne "time" provient de l'horloge (non anonymisée) du mov.edf, qui
    diffère de celle du ecg.edf/events.tsv : seul l'écart par rapport à sa
    propre première ligne est donc exploitable ici.
    """
    df = pd.read_csv(path)
    unnamed = [c for c in df.columns if c.startswith("Unnamed")] 
    # colonne d'index brute laissée par pandas à l'écriture du CSV d'origine, à ignorer
    df = df.drop(columns=unnamed, errors="ignore") # les supprime

    time = pd.to_datetime(df["time"]) # Convertit la colonne time en type datetime

    df["elapsed_s"] = (time - time.min()).dt.total_seconds() # Calcule le temps écoulé depuis le début du fichier ACC
    # C’est la clé pour aligner les données ACC avec les autres

    df = df.drop(columns=["time"]) # supprime la colonne brute de temps
    df.columns = [f"acc_{c}" if c != "elapsed_s" else c for c in df.columns] # Renomme les colonnes avec le préfixe acc_
    return df.sort_values("elapsed_s").reset_index(drop=True) # retourne le DataFrame trié


def load_events(path: Path) -> pd.DataFrame:
    """Charge le events.tsv BIDS et ne garde que les lignes de crise (eventType commençant par "sz").

    Les autres types (ex: "bckg" pour le fond, "impd" pour un contrôle
    d'impédance) ne sont pas des crises et sont donc écartés.
    """
    df = pd.read_csv(path, sep="\t") # lit un fichier TSV, donc avec tabulation comme séparateur
    seizures = df[df["eventType"].str.startswith("sz", na=False)] # Garde seulement les lignes dont le type d’événement commence par sz

    return seizures[["onset", "duration", "eventType"]].reset_index(drop=True)
# Retourne seulement les colonnes utiles : onset, duration,eventType


def annotate_seizures(grid: pd.DataFrame, seizures: pd.DataFrame) -> pd.DataFrame:
    """Ajoute les colonnes label (0/1) et seizure_type à la grille.

    Pour chaque crise, toutes les lignes de la grille dont le temps écoulé
    tombe dans la fenêtre [onset, onset + duration] sont marquées label=1.
    """
    grid["label"] = 0 # Initialise un label à 0 pour dire “pas de crise”
    grid["seizure_type"] = pd.NA # Crée une colonne pour le type de crise, initialement vide

    for _, row in seizures.iterrows(): # Parcourt chaque crise détectée
        mask = (grid["elapsed_s"] >= row["onset"]) & (grid["elapsed_s"] <= row["onset"] + row["duration"])
        # Détermine les secondes de la grille qui sont comprises dans la fenêtre de crise

        grid.loc[mask, "label"] = 1 # marque les lignes concernées comme crise (label=1)
        grid.loc[mask, "seizure_type"] = row["eventType"]
    return grid


def merge_run(hrv_path: Path, acc_path: Path, events_path: Path, run_id: str) -> tuple:
    """Fusionne un seul run (HRV + ACC + events) sur une grille à 1 seconde.

    Retourne (grid_union, grid_intersect) :
      - grid_union    : lignes où HRV OU ACC est disponible
      - grid_intersect: lignes où HRV ET ACC sont disponibles
    """
    hrv_anchor, hrv = load_hrv(hrv_path)
    # ACC et events.tsv sont optionnels : s'ils manquent, on continue quand
    # même en HRV seule (acc_available restera à False / label restera à 0).
    acc = load_acc(acc_path) if acc_path.exists() else pd.DataFrame({"elapsed_s": []}) # charge ACC si le fichier existe, sinon crée un DataFrame vide
    seizures = load_events(events_path) if events_path.exists() else pd.DataFrame(columns=["onset", "duration", "eventType"]) # charge les événements si le fichier existe, sinon crée un DataFrame vide

    # La grille de référence couvre toute la durée disponible (HRV et/ou ACC)
    t_max = hrv["elapsed_s"].max() # intialise t_max avec la durée maximale de HRV
    if not acc.empty: # vérifie si ACC existe 
        t_max = max(t_max, acc["elapsed_s"].max()) # si oui, prend la durée maximale entre HRV et ACC

    # Grille de référence à 1 seconde (1 ligne par seconde), de 0 à t_max (en float pour matcher le
    # dtype des colonnes elapsed_s utilisées par merge_asof)
    grid = pd.DataFrame({"elapsed_s": range(0, int(t_max) + 1)}, dtype="float64")

    # Jointure HRV : on prend la ligne HRV la plus proche de chaque seconde de
    # la grille, à condition qu'elle soit à moins de 500ms (la moitié de la
    # fenêtre glissante de 1s)
    merged = pd.merge_asof(
        grid, hrv, on="elapsed_s", direction="nearest", tolerance=HRV_TOLERANCE_S
    )
    merged["hrv_available"] = merged["hrv_interval_index"].notna() # crée hrv_available, vrai si une feature HRV a été trouvée

    # Jointure ACC : même logique, mais avec une tolérance de 2.5s puisque la
    # fenêtre glissante ACC fait 5s. On ne fait la jointure que si le fichier
    # ACC existait réellement (sinon acc_cols est vide)
    acc_cols = [c for c in acc.columns if c != "elapsed_s"] # liste les colonnes ACC sauf elapsed_s
    if acc_cols: # vérifie s’il y a vraiment des colonnes ACC
        merged = pd.merge_asof(
            merged, acc, on="elapsed_s", direction="nearest", tolerance=ACC_TOLERANCE_S
        )
        merged["acc_available"] = merged[acc_cols[0]].notna() # crée acc_available, vrai si une feature ACC a été trouvée
    else:
        merged["acc_available"] = False

    # Label de crise + métadonnées de traçabilité (timestamp absolu reconstruit
    # à partir de l'ancre HRV, identifiant patient et run pour retrouver
    # l'origine de chaque ligne une fois tous les runs concaténés).
    merged = annotate_seizures(merged, seizures)
    merged["timestamp"] = hrv_anchor + pd.to_timedelta(merged["elapsed_s"], unit="s") # reconstruit un timestamp absolu à partir de l’ancre HRV
    merged["patient_id"] = run_id.split("_")[0] # extrait patient_id, par exemple sub-001
    merged["run_id"] = run_id # ajoute l'identifiant du run

    grid_union = merged[merged["hrv_available"] | merged["acc_available"]].reset_index(drop=True)
    grid_intersect = merged[merged["hrv_available"] & merged["acc_available"]].reset_index(drop=True)
    return grid_union, grid_intersect # retourne les deux grilles


def main():
    """Point d'entrée : parcourt tous les runs trouvés sous --hrv-dir et les fusionne un par un."""
    args = parse_args()

    # convertit les chemins en objets Path
    hrv_dir = Path(args.hrv_dir)
    acc_dir = Path(args.acc_dir)
    raw_dir = Path(args.raw_dir)
    output_dir = Path(args.output_dir) # créer le dossier de sortie si besoin
    output_dir.mkdir(parents=True, exist_ok=True)

    hrv_files = find_hrv_files(hrv_dir, args.algo) # trouve tous les fichiers HRV disponibles, tous runs confondus
    print(f"{len(hrv_files)} fichiers de features HRV trouvés sous {hrv_dir}") # affiche combien ont été trouvés

    n_ok = n_no_acc = n_no_events = n_err = 0 # initialise les compteurs pour le résumé final

    for hrv_path in hrv_files: # boucle sur chaque fichier HRV trouvé
        run_id = run_id_from_path(hrv_path) # extrait l’identifiant de run depuis le nom du fichier HRV
        acc_path = find_acc_file(acc_dir, run_id) # construit le chemin ACC correspondant à ce run
        events_path = find_events_file(raw_dir, run_id) # construit le chemin attendu du fichier events.tsv pour ce run

        if not acc_path.exists(): # si ACC manque, affiche un message et incrémente n_no_acc
            print(f"  [{run_id}] Fichier ACC introuvable ({acc_path}), on continue en HRV seule")
            n_no_acc += 1
        if not events_path.exists(): # si events.tsv manque, affiche un message et incrémente n_no_events.
            print(f"  [{run_id}] events.tsv introuvable ({events_path}), label restera à 0")
            n_no_events += 1

        try:
            grid_union, grid_intersect = merge_run(hrv_path, acc_path, events_path, run_id) # tente de fusionner les données pour ce run
        except Exception as e: # en cas d'erreur afficher l'erreur, incrémenter n_err et continuer avec le run suivant
            print(f"  [{run_id}] ERREUR : {e}")
            n_err += 1
            continue

        grid_union.to_csv(output_dir / f"feat-grid-union_{run_id}.csv", index=False) # écrit la grille union dans un fichier CSV
        grid_intersect.to_csv(output_dir / f"feat-grid-intersect_{run_id}.csv", index=False) # écrit la grille intersection dans un fichier CSV
        n_ok += 1 # incrémente le compteur de runs fusionnés avec succès
        print(f"  [{run_id}] OK -> union={len(grid_union)} lignes, intersect={len(grid_intersect)} lignes")

    print(
        f"\nTerminé : {n_ok} runs fusionnés, {n_no_acc} sans fichier ACC, "
        f"{n_no_events} sans events.tsv, {n_err} erreurs"
    )


if __name__ == "__main__":
    main()

    # Donc, la logique centrale est : prendre HRV comme point de départ, créer une grille à 1 seconde, 
    # y coller ACC par proximité temporelle,
    # puis ajouter les labels de crise.
