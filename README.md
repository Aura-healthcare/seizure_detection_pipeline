# Script automatise QRS -> RR -> features

Ce dossier contient une CLI pour enchainer automatiquement:

1. detection des positions QRS;
2. conversion en intervalles RR;
3. calcul des features HRV via `seizure_detection_pipeline`.

## Structure

```text
.
├── main.py
└── sources/
    ├── fast.py
    ├── features.py
    └── hamilton.py
```

- `sources/fast.py`: copie locale de l'algorithme QRS actuel.
- `sources/hamilton.py`: detecteur alternatif base sur `scipy.signal.find_peaks`.
- `sources/features.py`: wrapper vers `compute_hrvanalysis_features` du dossier `seizure_detection_pipeline`.

## Prérequis / installation

Depuis la racine du projet:

```bash
python3 -m venv venv
venv/bin/pip install numpy pandas scipy pyedflib
```

Toutes les commandes ci-dessous utilisent `venv/bin/python` — activez le venv (`source venv/bin/activate`) si vous préférez utiliser `python3` directement.

**Important** : le calcul des features HRV (`compute_hrvanalysis_features`, via `sources/features.py`) dépend d'un module `seizure_detection_pipeline` situé en dehors de ce dépôt (`/pipeline-scripts/processing/ecg-to-rr-intervals/pan-tompkins/seizure_detection_pipeline`). Sans accès à ce chemin, seule la détection QRS → RR fonctionnera ; utilisez `--skip-features` (voir plus bas) pour vous arrêter avant le calcul des features.

## Commande principale

Depuis la racine du projet:

```bash
venv/bin/python main.py --algo fast --file example_samples/sub-001_ses-001_run-01_sample100000.csv
```

Ou avec le detecteur alternatif:

```bash
venv/bin/python main.py --algo hamilton --file example_samples/sub-001_ses-001_run-01_sample100000.csv
```

Par defaut, le script suppose:

- frequence ECG: `256 Hz`;
- colonne temps: `time`, `timestamp`, `datetime` ou `date`;
- colonne ECG: `ecgpoint`, `ecg`, `signal`, `value`, ou l'unique colonne numerique disponible;
- sorties dans `script_output/`.

## Options utiles

```bash
venv/bin/python main.py \
  --algo fast \
  --file example_samples/sub-001_ses-001_run-01_sample100000.csv \
  --fs 256 \
  --time-column time \
  --signal-column ecgpoint \
  --output-dir script_output
```

Pour seulement produire les RR sans calculer les features:

```bash
venv/bin/python main.py --algo fast --file example_samples/sub-001_ses-001_run-01_sample100000.csv --skip-features
```

Pour calculer les features depuis un fichier RR deja existant:

```bash
venv/bin/python main.py \
  --algo fast \
  --file example_samples/sub-001_ses-001_run-01_sample100000.csv \
  --rr-file script_output/sub-001_ses-001_run-01_sample100000/fast/rr_sub-001_ses-001_run-01_sample100000_fast.csv
```

## Sorties

Pour un fichier `patient.csv` et `--algo fast`, les sorties sont creees sous:

```text
script_output/patient/fast/
├── rr_patient_fast.csv
└── features/
    └── feats_patient_fast.csv
```

---

# Traitement direct depuis un fichier EDF ou CSV

`edf_to_features.py` fait la même chaîne (QRS → RR → features) mais à partir d'un fichier **EDF** ou d'un **CSV** déjà au format `timestamp`/`ecg`, sans passer par un CSV intermédiaire supplémentaire.

- Pour un fichier `.edf` : le signal est converti en **µV** automatiquement selon l'unité indiquée dans l'en-tête EDF (V, mV ou µV).
- Pour un fichier `.csv` : les colonnes `timestamp` et `ecg` sont attendues telles quelles (valeurs déjà en µV, timestamp parsable par `pandas.to_datetime`) ; une erreur explicite est levée si l'une de ces colonnes est absente.

## Structure du DataFrame interne / du CSV attendu

```
timestamp                      ecg
2000-01-01 00:00:00.000000000  9.017472
2000-01-01 00:00:00.003906250  -12.974441
...
```

## Commande

```bash
venv/bin/python edf_to_features.py --file example_samples/sub-001_ses-001_run-01_sample100000.edf
```

```bash
venv/bin/python edf_to_features.py --file example_samples/sub-001_ses-001_run-01_sample100000.csv
```

Avec les options utiles (EDF) :

```bash
venv/bin/python edf_to_features.py \
  --file example_samples/sub-001_ses-001_run-01_sample100000.edf \
  --fs 256 \
  --channel 0 \
  --output-dir example_samples_output
```

Pour ne produire que les RR sans calculer les features :

```bash
venv/bin/python edf_to_features.py --file example_samples/sub-001_ses-001_run-01_sample100000.edf --skip-features
```

## Options

| Option | Défaut | Description |
|---|---|---|
| `--file` | — | Chemin vers le fichier `.edf` ou `.csv` (obligatoire) |
| `--fs` | `256` | Fréquence d'échantillonnage en Hz |
| `--channel` | `0` | Index du canal ECG dans le fichier EDF (ignoré pour un CSV) |
| `--output-dir` | `script_output` | Dossier de sortie |
| `--max-nan-interpolation` | `10` | Nombre max de NaN consécutifs à interpoler |
| `--min-segment-len` | `50` | Longueur minimale d'un segment valide |
| `--skip-features` | `False` | Si présent, arrête après les RR intervals |

## Sorties

Pour un fichier `ecg.edf` (ou `ecg.csv`) :

```text
script_output/ecg/fast/
├── rr_ecg_fast.csv
└── features/
    └── feats_ecg_fast.csv
```

---

# Traitement en lot : `run_all_edf.sh`

Script qui applique `edf_to_features.py` à tous les fichiers ECG (`.edf` **et** `.csv`) trouvés sous `*/ecg/*` pour une liste de patients, dans le dataset SeizeIT2.

Le dataset SeizeIT2 (`ds005873`) est téléchargeable ici : https://openneuro.org/datasets/ds005873/versions/1.1.0

```bash
./run_all_edf.sh
```

Si vous avez téléchargé le dataset vous-même depuis OpenNeuro (voir lien ci-dessus), pointez vers votre dossier extrait avec `-d`/`--dataset-root` plutôt que d'éditer le script :

```bash
./run_all_edf.sh --dataset-root /chemin/vers/ds005873-1.1.0
```

- `DATASET_ROOT` : racine du dataset (`/data2/datasets/seizeit2-dataset/ds005873-1.1.0` par défaut, chemin interne ; surchargeable via `-d`/`--dataset-root`).
- `PATIENTS` : liste des sous-dossiers `sub-XXX` à traiter (à éditer directement dans le script).
- Les fichiers sont recherchés via `find "$DATASET_ROOT/$sub" -path "*/ecg/*.edf" -o -path "*/ecg/*.csv"`, triés, puis traités un par un avec `venv/bin/python edf_to_features.py --file ...`.
- À la fin, un résumé du nombre de succès/échecs est affiché ; les erreurs individuelles sont loguées sur stderr sans interrompre le traitement des fichiers suivants.

**NOTE** : le dataset sous /data2/datasets/seizeit2-dataset/ds005873-1.1.0 ne contient actuellement que des .edf dans les dossiers ecg/ — le script est prêt à traiter des .csv s'ils apparaissent, mais ce cas n'a pas encore été testé sur des données réelles

---

# Exemples et comparaison EDF vs CSV

## `example_samples/`

Contient un même enregistrement au format `.edf` et `.csv`, utilisé pour vérifier que `edf_to_features.py` produit des résultats identiques quelle que soit la source :

```text
example_samples/
├── sub-001_ses-001_run-01_sample100000.edf
└── sub-001_ses-001_run-01_sample100000.csv
```

## `example_samples_output/`

Résultat de `edf_to_features.py` lancé séparément sur chacun des deux fichiers ci-dessus, un dossier par origine :

```text
example_samples_output/
├── edf_origin/sub-001_ses-001_run-01_sample100000/fast/
│   ├── rr_sub-001_ses-001_run-01_sample100000_fast.csv
│   └── features/feats_sub-001_ses-001_run-01_sample100000_fast.csv
└── csv_origin/sub-001_ses-001_run-01_sample100000/fast/
    ├── rr_sub-001_ses-001_run-01_sample100000_fast.csv
    └── features/feats_sub-001_ses-001_run-01_sample100000_fast.csv
```

## `compare_feats.py`

Compare les deux fichiers `feats_*.csv` (`edf_origin` vs `csv_origin`) pour vérifier que le pipeline donne le même résultat peu importe le format d'entrée.

- Compare colonne par colonne avec une tolérance numérique (`rtol=1e-5`, `atol=1e-8`), en gérant correctement `NaN` et `inf`.
- Ignore les colonnes `filename` et `original_filename`, qui contiennent le chemin du dossier d'origine et diffèrent donc normalement.
- Signale toute différence de nombre de lignes/colonnes et affiche jusqu'à 10 divergences par colonne.

```bash
venv/bin/python compare_feats.py
```

Ou avec des fichiers spécifiques :

```bash
venv/bin/python compare_feats.py example_samples_output/csv_origin/sub-001_ses-001_run-01_sample100000/fast/features/feats_sub-001_ses-001_run-01_sample100000_fast.csv example_samples_output/edf_origin/sub-001_ses-001_run-01_sample100000/fast/features/feats_sub-001_ses-001_run-01_sample100000_fast.csv
```

