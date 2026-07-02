# Script automatise QRS -> RR -> features

Ce dossier contient une CLI pour enchainer automatiquement:

1. detection des positions QRS;
2. conversion en intervalles RR;
3. calcul des features HRV via `seizure_detection_pipeline`.

## Structure

```text
script/
├── main.py
└── sources/
    ├── fast.py
    ├── features.py
    └── hamilton.py
```

- `sources/fast.py`: copie locale de l'algorithme QRS actuel.
- `sources/hamilton.py`: detecteur alternatif base sur `scipy.signal.find_peaks`.
- `sources/features.py`: wrapper vers `compute_hrvanalysis_features` du dossier `seizure_detection_pipeline`.

## Commande principale

Depuis la racine du projet:

```bash
env_jc/bin/python script/main.py --algo fast --file chemin/vers/ecg.csv
```

Ou avec le detecteur alternatif:

```bash
env_jc/bin/python script/main.py --algo hamilton --file chemin/vers/ecg.csv
```

Par defaut, le script suppose:

- frequence ECG: `250 Hz`;
- colonne temps: `time`, `timestamp`, `datetime` ou `date`;
- colonne ECG: `ecgpoint`, `ecg`, `signal`, `value`, ou l'unique colonne numerique disponible;
- sorties dans `script_output/`.

## Options utiles

```bash
env_jc/bin/python script/main.py \
  --algo fast \
  --file chemin/vers/ecg.csv \
  --fs 250 \
  --time-column time \
  --signal-column ecgpoint \
  --output-dir script_output
```

Pour seulement produire les RR sans calculer les features:

```bash
env_jc/bin/python script/main.py --algo fast --file chemin/vers/ecg.csv --skip-features
```

Pour calculer les features depuis un fichier RR deja existant:

```bash
env_jc/bin/python script/main.py \
  --algo fast \
  --file chemin/vers/ecg.csv \
  --rr-file output/rr_intervals_patient_05_V1_JC.csv
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
python edf_to_features.py --file chemin/vers/ecg.edf
```

```bash
python edf_to_features.py --file chemin/vers/ecg.csv
```

Avec les options utiles (EDF) :

```bash
python edf_to_features.py \
  --file chemin/vers/ecg.edf \
  --fs 256 \
  --channel 0 \
  --output-dir script_output
```

Pour ne produire que les RR sans calculer les features :

```bash
python edf_to_features.py --file chemin/vers/ecg.edf --skip-features
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

```bash
./run_all_edf.sh
```

- `DATASET_ROOT` : racine du dataset (`/data2/datasets/seizeit2-dataset/ds005873-1.1.0` par défaut).
- `PATIENTS` : liste des sous-dossiers `sub-XXX` à traiter (à éditer directement dans le script).
- Les fichiers sont recherchés via `find "$DATASET_ROOT/$sub" -path "*/ecg/*.edf" -o -path "*/ecg/*.csv"`, triés, puis traités un par un avec `venv/bin/python edf_to_features.py --file ...`.
- À la fin, un résumé du nombre de succès/échecs est affiché ; les erreurs individuelles sont loguées sur stderr sans interrompre le traitement des fichiers suivants.

# NOTE : le dataset sous /data2/datasets/seizeit2-dataset/ds005873-1.1.0 ne contient actuellement que des .edf dans les dossiers ecg/ — le script est prêt à traiter des .csv s'ils apparaissent, mais ce cas n'a pas encore été testé sur des données réelles

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
python3 compare_feats.py
```

Ou avec des fichiers spécifiques :

```bash
python3 compare_feats.py chemin/vers/feats_edf.csv chemin/vers/feats_csv.csv
```

