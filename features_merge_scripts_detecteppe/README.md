# features_merge_scripts

Scripts de préparation des features Detecteppe : découpage des annotations de crises par patient, fusion des features HRV/ACC sur une grille temporelle commune, puis concaténation de tous les patients en fichiers train/test unifiés.

## Prérequis

- Python **3.10+**
- pandas (`pip install pandas>=2.0`)

## Pipeline

```
seizure-annotations-v1.0.csv
        │
        ▼
split_seizures_by_patient.py        (1 CSV par patient)
        │
        ▼
merge_feat_grid.py  (par patient)   (grille 1s : HRV + ACC + label crise)
        │
        ▼
process_all_patients.sh             (exécute merge_feat_grid.py sur tous les patients train/test)
        │
        ▼
concat_feats_grids.py               (concatène tous les patients → 2 fichiers finaux)
```

---

## 1. `split_seizures_by_patient.py`

Découpe un fichier d'annotations de crises en un CSV par `patient-id`.

### Usage

```bash
python3 split_seizures_by_patient.py \
  --input  data/seizure-annotations-v1.0.csv \
  --outdir seizure_by_patient
```

### Arguments

| Argument | Obligatoire | Description |
|---|---|---|
| `--input` | oui | Chemin vers le fichier d'annotations (CSV) |
| `--outdir` | oui | Dossier de sortie (créé si besoin) |
| `--patient-col` | non | Nom de la colonne patient (défaut: `patient-id`) |

### Sorties

- Un fichier par patient: `seizure-annotations_<patient-id>.csv`
- Les `/` dans les identifiants sont remplacés par `_` pour garantir des noms de fichiers sûrs.

Exemple:
```
seizure_by_patient/
  seizure-annotations_01-003.csv
  seizure-annotations_01-011.csv
  seizure-annotations_01-013.csv
```

---

## 2. `merge_feat_grid.py`

Fusionne les features `feat-hrv`, `feat-acc` et les annotations de crises d'**un seul patient** sur une grille temporelle commune à 1 seconde.

### Usage

```bash
python3 merge_feat_grid.py \
  --hrv              data/train/feat_hrv_01-001.csv \
  --acc              data/train/features_01-001.acc.csv \
  --seizure          data/seizure_by_patient_v1.2/seizure-annotations_01-001.csv \
  --output-union     output_v1.2/feat-grid-union_01-001.csv \
  --output-intersect output_v1.2/feat-grid-intersect_01-001.csv \
  --training-split   train \
  --patient-id       01-001
```

### Arguments

| Argument | Obligatoire | Description |
|---|---|---|
| `--hrv` | oui | CSV de features HRV |
| `--acc` | oui | CSV de features ACC |
| `--seizure` | oui | CSV d'annotations de crises (sortie de `split_seizures_by_patient.py`) |
| `--output-union` | oui | Sortie: grille où HRV **ou** ACC est disponible |
| `--output-intersect` | oui | Sortie: grille où HRV **et** ACC sont disponibles |
| `--training-split` | oui | Label du split (`train`, `val`, `test`, ...) |
| `--patient-id` | oui | Identifiant patient (ex: `01-001`) |

### Fonctionnement

1. Charge HRV et ACC, normalise les timestamps en UTC (préfixe les colonnes `hrv_` / `acc_`).
2. Construit une grille temporelle à 1 seconde couvrant toute la période des deux signaux.
3. Joint HRV (tolérance ±500 ms) et ACC (tolérance ±2500 ms) sur la grille via `merge_asof`.
4. Annote chaque ligne avec `label` (1 si dans une fenêtre de crise) et `seizure-id`.
5. Ajoute les colonnes `training-split` et `patient-id`, convertit l'index en `Europe/Paris`.
6. Écrit deux fichiers :
   - **union** : lignes où HRV ou ACC est présent
   - **intersect** : lignes où HRV et ACC sont présents

---

## 3. `process_all_patients.sh`

Exécute `merge_feat_grid.py` pour l'ensemble des patients train et test.

### Usage

```bash
./process_all_patients.sh [--train "PATIENT1 PATIENT2 ..."] [--test "PATIENT1 PATIENT2 ..."]
```

### Arguments

| Argument | Obligatoire | Description |
|---|---|---|
| `--train` | non | Liste des patients train, séparés par des espaces (entre guillemets). Défaut: `01-001 01-002 01-004 01-005 01-011` |
| `--test` | non | Liste des patients test, séparés par des espaces (entre guillemets). Défaut: `01-003 01-006 01-007 01-009` |
| `-h`, `--help` | non | Affiche l'aide |

Exemple:
```bash
./process_all_patients.sh --train "01-001 01-002 01-004" --test "01-003 01-006"
```

Les chemins d'entrée/sortie (`data/train`, `data/test`, `data/seizure_by_patient_v1.2`, `output_v1.2`) restent à modifier directement en tête de script si nécessaire.

### Sorties

Pour chaque patient, dans `output_v1.2/` :
- `feat-grid-union_<patient-id>.csv`
- `feat-grid-intersect_<patient-id>.csv`

---

## 4. `concat_feats_grids.py`

Concatène tous les fichiers `feat-grid-union_*.csv` et `feat-grid-intersect_*.csv` d'un dossier en deux fichiers uniques.

### Usage

```bash
python3 concat_feats_grids.py --input-dir output_v1.2
```

### Arguments

| Argument | Obligatoire | Description |
|---|---|---|
| `--input-dir` | oui | Dossier contenant les fichiers `feat-grid-union_*.csv` et `feat-grid-intersect_*.csv` |

### Sorties

- `feat-grid-all-union.csv`
- `feat-grid-all-intersect.csv`

---

## Exemple de workflow complet

```bash
# 1. Découper les annotations par patient
python3 split_seizures_by_patient.py \
  --input  data/seizure-annotations-v1.0.csv \
  --outdir data/seizure_by_patient_v1.2

# 2. Fusionner HRV + ACC + annotations pour tous les patients train/test
./process_all_patients.sh

# 3. Concaténer tous les patients en fichiers finaux
python3 concat_feats_grids.py --input-dir output_v1.2
```
