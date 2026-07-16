import argparse # pour parser les arguments de la ligne de commande 
import pathlib # pour manipuler les chemins de fichiers 
import sys # pour accéder aux variables système et la liste des chemins 
import tempfile # pour cérer des fichiers temporaires
from datetime import timezone # pour gérer les fuseaux horaires

import numpy as np # bibliothèque pour le calcul numérique
import pandas as pd # pour manipuler les données en dataframes
import pyedflib # pour lire les fichiers EDF


########### CHEMIN DE REPERTOIRES ###########

CURRENT_DIR = pathlib.Path(__file__).resolve().parent # reupère le repertoire courant du script
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))
# Ajoute ces répertoires au chemin Python pour que les imports fonctionnent

# imports custom du projet
from sources.fast import qrs_detector as fast_qrs_detector
from sources.features import compute_features
from main import build_rr_dataframe


DEFAULT_FS = 256 # fréquence d'échantillonnage par défaut en Hz
DEFAULT_CHANNEL = 0 # canal ecg par défaut (index 0)
DEFAULT_OUTPUT_DIR = "script_output" # repertoire de sortie par défaut pour les fichiers générés

##########################################################
# Créer un paser pour les arguments en ligne de commande
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="EDF → QRS → RR intervals → HRV features (sans fichier CSV intermédiaire)."
    )
    parser.add_argument(
        "--file", required=True,
        help="Chemin vers le fichier d'entrée (.edf ou .csv avec colonnes 'timestamp' et 'ecg')."
    )
    # argument obligatoire : chemin du fichier EDF ou CSV à traiter

    parser.add_argument(
        "--fs", type=int, default=DEFAULT_FS,
        help=f"Fréquence d'échantillonnage en Hz (défaut : {DEFAULT_FS})."
    ) # Argument optionnel : fréquence d'échantillonnage (256 Hz par défaut)


    parser.add_argument(
        "--channel", type=int, default=DEFAULT_CHANNEL,
        help=f"Index du canal ECG dans le fichier EDF (défaut : {DEFAULT_CHANNEL})."
    ) # Argument optionnel : canal ECG (0 par défaut)


    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR) # Argument optionnel : répertoire de sortie
    parser.add_argument("--max-nan-interpolation", type=int, default=10) #  Argument optionnel : nombre max de valeurs NaN à interpoler
    parser.add_argument("--min-segment-len", type=int, default=50) # Argument optionnel : longueur min d'un segment
    parser.add_argument("--skip-features", action="store_true") # # Drapeau optionnel : si présent, skip le calcul des features
    return parser.parse_args() # parse et retourne les arguments


##########################################################

# Charge un canal ECG d'un fichier EDF et retourne un DataFrame timestamp/ecg en µV
def load_edf(edf_path: str, channel: int, fs: int) -> pd.DataFrame:
    """Charge un canal ECG d'un fichier EDF et retourne un DataFrame timestamp/ecg en µV."""

    with pyedflib.EdfReader(edf_path) as f: # ouvre le fichier EDF en lecture
        start_dt = f.getStartdatetime().replace(
            year=2000, month=1, day=1, hour=0, minute=0, second=0, tzinfo=timezone.utc
        ) # Récupère l'heure de début et la normalise (date=2000-01-01, UTC)

        # Lit les échantillons du signal ECG du canal spécifié
        samples = f.readSignal(channel)

        #  Récupère l'unité de mesure (V, mV, µV, etc.) 
        unit = f.getPhysicalDimension(channel).strip()

    # Normalisation en µV (mcirovolts)
    unit_lower = unit.lower()
    if unit_lower in ("v",):
        samples = samples * 1e6 # Volts → µV (multiply by 1 million)
    elif unit_lower in ("mv", "millivolt", "millivolts"):
        samples = samples * 1e3 # Millivolts → µV (multiply by 1000)
    elif unit_lower not in ("uv", "µv", "microvolt", "microvolts"):
        print(f"Avertissement : unité '{unit}' inconnue, aucune conversion appliquée.")
    #  Si l'unité est déjà en µV, pas de conversion

    # Crée une série de timestamps espacés régulièrement selon fs
    timestamps = pd.date_range(
        start=start_dt,
        periods=len(samples),
        freq=pd.Timedelta(seconds=1 / fs),
    ) # Par exemple: si fs=256, chaque échantillon est espacé de 1/256 secondes
    
    # Retourne un DataFrame avec timestamps et valeurs ECG en µV : 
    return pd.DataFrame({"timestamp": timestamps, "ecg": samples})


SUPPORTED_EXTENSIONS = (".edf", ".csv") # extensions de fichier d'entrée acceptées


# Charge un CSV (colonnes "timestamp" et "ecg", déjà en µV) et retourne un DataFrame timestamp/ecg
def load_csv(csv_path: str) -> pd.DataFrame:
    """Charge un CSV avec colonnes timestamp/ecg (µV) et retourne un DataFrame timestamp/ecg."""
    df = pd.read_csv(csv_path)

    missing = {"timestamp", "ecg"} - set(df.columns)
    if missing:
        raise ValueError(
            f"Colonnes manquantes dans le CSV : {sorted(missing)} "
            f"(colonnes trouvées : {list(df.columns)})."
        )

    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True) # force un timestamp tz-aware UTC
    df["ecg"] = df["ecg"].astype(float)
    return df[["timestamp", "ecg"]]


def run(args: argparse.Namespace) -> None:
    input_path = pathlib.Path(args.file)
    if not input_path.exists(): # vérifie que le fichier existe, sinon lève une exception
        raise FileNotFoundError(f"Fichier introuvable : {input_path}")

    suffix = input_path.suffix.lower()
    if suffix not in SUPPORTED_EXTENSIONS:
        raise ValueError(
            f"Extension '{suffix}' non supportée : utilisez un fichier .edf ou .csv."
        )

    print(f"Chargement : {input_path}")
    if suffix == ".edf":
        df = load_edf(str(input_path), args.channel, args.fs)
        # charge le fichier EDF avec la fonction définie juste au-dessus
        # et retourne un DataFrame avec les timestamps et les valeurs ECG
    else:
        df = load_csv(str(input_path))
        # charge le CSV (colonnes timestamp/ecg déjà en µV)

    # Affiche des informations sur les données chargées
    print(f"  {len(df):,} échantillons | fs={args.fs} Hz | début={df['timestamp'].iloc[0]}")
    print(df.head(3).to_string(index=False))

    print("\nDétection QRS (algo fast)…")

    # Détecte les pics QRS (battements du cœur) dans l'ECG
    qrs_timestamps = fast_qrs_detector(
        df,
        args.fs,
        max_nan_interpolation=args.max_nan_interpolation,
        min_segment_len=args.min_segment_len,
    )
    print(f"  {len(qrs_timestamps)} QRS détectés")

    
    # Construit un DataFrame avec les intervalles RR (entre chaque battement)
    rr_df = build_rr_dataframe([qrs_timestamps], args.fs)
    rr_df = rr_df[rr_df["rr_interval"] <= 5000] # Filtre : ne garde que les intervalles ≤ 5000 ms (élimine les aberrantes)
    print(f"  {len(rr_df)} intervalles RR (≤ 5000 ms)")

    # Crée le chemin de sortie : script_output/nom_fichier/fast/
    output_dir = pathlib.Path(args.output_dir) / input_path.stem / "fast"

    # Crée le répertoire s'il n'existe pas
    output_dir.mkdir(parents=True, exist_ok=True)

    # Définit le chemin du fichier CSV de sortie
    rr_path = output_dir / f"rr_{input_path.stem}_fast.csv"

    # Sauvegarde les intervalles RR dans un fichier CSV
    rr_df.to_csv(rr_path, index=False, date_format="%Y-%m-%d_%H:%M:%S.%f%z")
    print(f"\nRR intervals : {rr_path}")

    if args.skip_features:
        return  # Si --skip-features est actif, s'arrête ici

    print("\nCalcul des features HRV…")
    # Crée le répertoire pour les features
    features_dir = output_dir / "features"
    features_dir.mkdir(exist_ok=True)

    # Calcule les features HRV (Heart Rate Variability) : 
    features_path = pathlib.Path(compute_features(str(rr_path), str(features_dir))) 

    # Renomme le fichier de "rr_..." à "feats_..." pour clarifier
    if features_path.name.startswith("rr_"):
        renamed = features_path.with_name("feats_" + features_path.name[len("rr_"):])
        features_path.rename(renamed)
        features_path = renamed
    print(f"Features     : {features_path}")


def main() -> None:
    args = parse_args() # parse les arguments de la ligne de commande
    run(args) # exécute le traitement principal avec les arguments


if __name__ == "__main__":
    main() # point d'entrée du script 
