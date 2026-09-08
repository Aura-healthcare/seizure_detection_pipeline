# features_merge_scripts_seizeit

Scripts de préparation des features pour le dataset **SeizeIT2** : fusion des features HRV/ACC et des événements de crise (`events.tsv` BIDS) sur une grille temporelle commune (par run), puis concaténation de tous les runs en deux fichiers unifiés.

## Prérequis

- Python **3.10+**
- pandas (`pip install pandas>=2.0`)

## Pipeline

```
feats-hrv/ (dérivées)  ─┐
feats_acc/ (dérivées)  ─┼──▶  merge_feat_grid.py   (grille 1s : HRV + ACC + label crise, par run)
raw-data/ events.tsv   ─┘              │
                                        ▼
                          concat_feats_grids.py   (concatène tous les runs → 2 fichiers finaux)
```

Les trois sources sont associées automatiquement entre elles par leur identifiant de run BIDS (`sub-XXX_ses-YY_task-szMonitoring_run-ZZ`), extrait du nom de fichier. Il n'y a pas d'étape manuelle de découpage/appariement à faire au préalable.

---

## 1. `merge_feat_grid.py`

Parcourt tous les fichiers de features HRV disponibles sous `--hrv-dir`, retrouve pour chaque run le fichier ACC et le `events.tsv` correspondants par leur nom, et fusionne les trois sur une grille temporelle à 1 seconde **par run**.

### Usage

```bash
python3 merge_feat_grid.py \
  --hrv-dir    /data2/datasets/seizeit2-dataset/derived-data/feats-hrv \
  --acc-dir    /data2/datasets/seizeit2-dataset/derived-data/feats_acc \
  --raw-dir    /data2/datasets/seizeit2-dataset/raw-data/ds005873-1.1.0 \
  --output-dir output/feat-grid
```

### Arguments

| Argument | Obligatoire | Description |
|---|---|---|
| `--hrv-dir` | oui | Racine des données dérivées `feats-hrv` (pilote la découverte des runs à traiter) |
| `--acc-dir` | oui | Racine des données dérivées `feats_acc` |
| `--raw-dir` | oui | Racine de l'arborescence BIDS `raw-data` (pour retrouver les `events.tsv`) |
| `--output-dir` | oui | Dossier de sortie pour les CSV feat-grid, un par run |
| `--algo` | non | Sous-dossier de l'algo de détection QRS pour la HRV (défaut : `fast`) |

### Fonctionnement

1. Liste tous les fichiers de features HRV sous `--hrv-dir` : c'est cette liste qui détermine les runs traités (un run doit avoir sa HRV calculée pour être pris en compte).
2. Pour chaque run, retrouve par nom le fichier ACC (`feats_acc/feats_features_<run-id>_mov.csv`) et le `events.tsv` (`raw-data/.../sub-XXX/ses-YY/eeg/<run-id>_events.tsv`). Si l'un des deux est absent, le run est quand même traité (HRV seule, sans label si le `events.tsv` manque).
3. Aligne HRV, ACC et événements de crise sur une grille à 1 seconde, **en secondes écoulées depuis le début du run** (et non en datetime absolu — voir la note importante dans le docstring de `merge_feat_grid.py` sur les horloges HRV/ACC non comparables).
4. Joint HRV (tolérance ±500 ms) et ACC (tolérance ±2500 ms) sur la grille via `merge_asof`.
5. Annote chaque ligne avec `label` (1 si dans une fenêtre `[onset, onset+duration]` d'un `eventType` commençant par `sz_`) et `seizure_type` (le type exact, ex: `sz_foc_ia_nm`).
6. Ajoute les colonnes `patient_id`, `run_id` et une colonne `timestamp` absolue reconstruite à partir de l'ancre HRV.
7. Écrit, pour chaque run, deux fichiers dans `--output-dir` :
   - `feat-grid-union_<run-id>.csv` : lignes où HRV **ou** ACC est disponible
   - `feat-grid-intersect_<run-id>.csv` : lignes où HRV **et** ACC sont disponibles

---

## 2. `concat_feats_grids.py`

Concatène tous les fichiers `feat-grid-union_*.csv` et `feat-grid-intersect_*.csv` d'un dossier en deux fichiers uniques.

### Usage

```bash
# Concatène tous les runs de tous les patients, sortie dans --input-dir
python3 concat_feats_grids.py --input-dir output/feat-grid

# Sortie dans un dossier différent
python3 concat_feats_grids.py --input-dir output/feat-grid --output-dir output/feat-grid-all

# Ne concaténer que certains patients
python3 concat_feats_grids.py --input-dir output/feat-grid --patients sub-001 sub-002
```

### Arguments

| Argument | Obligatoire | Description |
|---|---|---|
| `--input-dir` | oui | Dossier contenant les fichiers `feat-grid-union_*.csv` et `feat-grid-intersect_*.csv` (la sortie de `merge_feat_grid.py`) |
| `--output-dir` | non | Dossier de sortie pour les fichiers concaténés (défaut : identique à `--input-dir`), créé si besoin |
| `--patients` | non | Liste de `patient_id` (ex: `sub-001 sub-002`) à concaténer (défaut : tous les patients trouvés dans `--input-dir`) |

### Sorties

- `feat-grid-all-union.csv`
- `feat-grid-all-intersect.csv`

---

## Exemple de workflow complet

```bash
# 1. Fusionner HRV + ACC + events.tsv pour tous les runs du dataset
python3 merge_feat_grid.py \
  --hrv-dir /data2/datasets/seizeit2-dataset/derived-data/feats-hrv \
  --acc-dir /data2/datasets/seizeit2-dataset/derived-data/feats_acc \
  --raw-dir /data2/datasets/seizeit2-dataset/raw-data/ds005873-1.1.0 \
  --output-dir output/feat-grid

# 2. Concaténer tous les runs en fichiers finaux
python3 concat_feats_grids.py --input-dir output/feat-grid
```
