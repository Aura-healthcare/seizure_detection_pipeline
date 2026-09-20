Started this work to study how we could could build an embedding which can capture the variability of seizure / non seizure but also the variability inter patients.

This is based on following papers :

SimCLR from google brain https://arxiv.org/abs/2002.05709 
MoCo from facebook ai https://arxiv.org/abs/1911.05722
InfoNCE from google https://arxiv.org/pdf/2004.11362


Interesting paper on contrastive learning : https://arxiv.org/pdf/2010.05113

Contrastive learning in ECG or EEG :  https://pubmed.ncbi.nlm.nih.gov/37028019/

Originally developed on the Teppe dataset (29 HRV + 141 accelerometer features, 9 patients). This folder is now the **SeizeIT** version: it works on the merged feature grid produced by `../merge_feat_grid.py` / `../concat_feats_grids.py` (see `../FEATURES.md`).

## Dataset

Configured in `DATA_CONFIG` of `config.py`:

- CSV: `../mini-dataset2/feat-grid-all-intersect.csv` — HRV features + ACC-norme features (after `drop_columns`, ~14 HRV + 27 ACC), columns `patient_id`, `label` and sensor availability flags `hrv_available` / `acc_available`. There is no `split` column: train/test is assigned per patient via `DATA_CONFIG['patient_split']`.
- 12 patients (sub-001, 002, 011, 012, 021, 022, 030, 031, 040, 044, 046, 050).
  - train: sub-001, 002, 011, 012, 021, 022, 030, 031, 044, 046
  - test: sub-040, sub-050
- Per-patient z-score normalisation on the non-seizure baseline (`per_patient_normalization`), non-seizure undersampling on train only (`undersampling_ratio`, 50:1 currently).
- **Note:** `mini-dataset2/` currently contains `feat-grid-all-intersect_12patients.csv` and `feat-grid-all-union_12patients.csv`; `data_path` in `config.py` must point to one of them (the un-suffixed `feat-grid-all-intersect.csv` does not exist at the moment).

## Usage
Install uv
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Update the conf file `config.py` to experiment.

Launch training of embedding and classification head (cross-patient setting: train patients vs held-out test patients)
```bash
uv run python launch_train.py
```

Per-patient baseline (see [Per-patient baseline](#per-patient-baseline-baseline_patientpy) below)
```bash
uv run python baseline_patient.py
uv run python baseline_patient.py --patients sub-001 sub-002 --n-folds 4
uv run python baseline_patient.py --embedding-epochs 2 --classifier-epochs 2   # quick smoke test
```

Hyper-parameter search with Optuna (writes `best_params.json`, to be re-applied by hand in `config.py`)
```bash
uv run python optuna_search.py --n-trials 30
uv run python optuna_search.py --n-trials 100 --metric f1 --embedding-epochs 20 --classifier-epochs 15
uv run python optuna_search.py --compare-losses --n-trials 20     # one study per loss_type
```

Evaluate a trained model (auto-detects `embedding_model.pth` in the results folder)
```bash
uv run python eval.py --results_dir ./results_20260214_090530

# With explicit model path and custom k values
uv run python eval.py --results_dir ./results_20260214_090530 \
    --model ./checkpoints/model_9.pth \
    --knn_k 10 \
    --retrieval_k 5 10 20 50
```

## Positive / Negative pair strategy (contrastive loss)

Each batch item is a pair with a target distance fed to `ContrastiveLoss`:

| Pair type | How sampled | Target distance |
|-----------|-------------|-----------------|
| **Positive** (50%) | Same label, **different patient** preferred — falls back to same patient if none exist | `0.0` (embeddings should be similar) |
| **Negative** (50%) | Different label, any patient | `1.0` (embeddings should be dissimilar) |

The loss minimises MSE between the cosine similarity of the two embeddings and the target distance, with temperature scaling.

Forcing positives to come from **different patients** is the key design choice: it prevents the model from learning patient identity as a shortcut and instead pushes it to capture what is universal about seizures across patients.

## Triplet pair strategy (triplet loss)

Triplets are built **offline** at dataset construction time (one triplet per anchor sample):

| Role | How selected |
|------|-------------|
| **Anchor** | Every sample in the dataset is used as an anchor once |
| **Positive** | Same label, **different patient** — falls back to same patient only if no cross-patient examples exist |
| **Negative** | Different label, any patient (random) |

The loss pushes `dist(anchor, positive) + margin < dist(anchor, negative)`.

Key difference vs contrastive: triplets are **fixed before training starts**, so they don't adapt as the model improves (easy triplets dominate late in training). This is why `batch_hard_triplet` is generally more effective — it mines hard pairs online from each batch.

## Supervised contrastive strategy — recommended (supcon loss)

Based on [Khosla et al. 2020](https://arxiv.org/abs/2004.11362). Extends contrastive learning to use **all** same-class samples in the batch as positives, not just one.

| Role | How selected |
|------|-------------|
| **Positives** | All samples in the batch with the **same label AND different patient** — same-patient pairs are excluded to prevent learning patient identity as a shortcut |
| **Negatives** | All samples in the batch with a **different label** |

Loss per anchor `i` over a batch of B samples:
```
L_i = -1/|P(i)| · Σ_{p∈P(i)} log( exp(sim(i,p)/τ) / Σ_{a≠i} exp(sim(i,a)/τ) )
```

Key properties:
- **Cross-patient positives only** — same-patient, same-label pairs are excluded from the positive set. This forces the model to find seizure features that are shared across patients rather than relying on within-patient similarity
- **Class-weighted loss** — each anchor's loss is weighted by inverse class frequency, so seizure anchors contribute equally despite ~800:1 imbalance
- Embeddings are **L2-normalised** — only direction matters, not magnitude
- Temperature `τ` (default `0.07`) controls cluster tightness — lower = sharper boundaries
- Unlike triplet losses, every positive in the batch contributes a gradient signal
- Uses `EmbeddingDataset` + `PKBatchSampler` with `max_per_patient=4` — caps per-patient contribution per batch to maximise cross-patient diversity

## Batch hard triplet strategy (batch_hard_triplet loss)

Triplets are mined **online** inside each batch — no dataset-level pre-generation needed.

For every sample in the batch acting as anchor, the loss picks:

| Role | How selected |
|------|-------------|
| **Hardest positive** | Sample in the batch with the **same label** and the **largest** L2 distance to the anchor |
| **Hardest negative** | Sample in the batch with a **different label** and the **smallest** L2 distance to the anchor |

Loss per anchor: `relu(dist(anchor, hardest_pos) - dist(anchor, hardest_neg) + margin)`

**Important:** in the current training loop (`train_model_triplet_hard_negative`), the labels passed to the loss are **`patient_ids`**, not seizure/non-seizure labels. This means the model is trained to **discriminate between patients** first — learning a rich, patient-aware embedding — before being fine-tuned for seizure detection in phase 2. This is the multi-stage transfer learning approach.

Because mining is online, hard triplets are always relative to the model's current state, which prevents easy-triplet saturation. Batch diversity matters: use `PKBatchSampler` (P patients × K samples per batch) to guarantee that both hard positives and hard negatives are available in every batch.

  ┌───────────────┬───────────────────────────────────────┬──────────────────────────────────────────┐
  │               │              contrastive              │                  supcon                  │
  ├───────────────┼───────────────────────────────────────┼──────────────────────────────────────────┤
  │ Dataset       │ ContrastiveDataset (pairwise)         │ EmbeddingDataset (plain samples)         │
  ├───────────────┼───────────────────────────────────────┼──────────────────────────────────────────┤
  │ Batch sampler │ random shuffle                        │ PKBatchSampler (P patients × K samples)  │
  ├───────────────┼───────────────────────────────────────┼──────────────────────────────────────────┤
  │ Training loop │ receives (anchor, contrastive, label) │ receives (features, labels) — full batch │
  ├───────────────┼───────────────────────────────────────┼──────────────────────────────────────────┤
  │ Loss input    │ two embeddings + binary label         │ full batch of embeddings + all labels    │
  └───────────────┴───────────────────────────────────────┴──────────────────────────────────────────┘

## Per-patient baseline (baseline_patient.py)

Within-patient sanity ceiling: the full pipeline (embedding pretraining + classifier head, same architecture and losses as `launch_train.py`) is trained **separately for each patient**.

- The patient's rows (chronological order in the CSV: run, then `elapsed_s`) are cut into `n_folds` (default 4) contiguous blocks.
- Each fold uses 3 blocks as train and the remaining one as test, rotating so every block is test once (block cross-validation).
- Folds where the train or test block has only one class are skipped (that is why some patients have fewer than 4 folds).
- Normalisation statistics are fit on the train blocks only; features constant for a patient are dropped.
- Since only one patient is involved, all patient ids are 0 (the "different patient" positive rule of SupCon has no effect here).
- Options are in `PATIENT_BASELINE_CONFIG` (`patients`, `n_folds`, `results_dir`) or via CLI flags.

Outputs, one folder per run `results_patient_baseline_<timestamp>/`:
```
<pid>/fold_<k>/
    embedding_model.pth, training_config.json, metrics.json
    umap_pretrain_{labels,patient_seizure,test_labels,test_patient_seizure}.png
    classifier_training_history.png, train_/test_evaluation_results.png
summary.csv, summary.json     # per fold, per patient (mean/std) and overall — written at the end of the run
```
Per-fold checkpoints go to `checkpoints/patient_baseline/<pid>/fold_<k>/model_<epoch>.pth`.

Current results (`results_patient_baseline_20260830_182408`, sub-001 and sub-002 only — no `summary.*` yet, the run was not completed for the other patients; checkpoints exist for 10 patients):

| Patient / fold | ROC AUC | Recall | Precision | F1 |
|----------------|---------|--------|-----------|----|
| sub-001 / 1 | 0.858 | 0.781 | 0.006 | 0.011 |
| sub-001 / 2 | 0.921 | 0.832 | 0.017 | 0.032 |
| sub-001 / 3 | 0.964 | 0.923 | 0.003 | 0.005 |
| sub-002 / 0 | 0.828 | 0.624 | 0.056 | 0.103 |
| sub-002 / 1 | 0.941 | 0.813 | 0.016 | 0.031 |
| sub-002 / 2 | 0.938 | 0.929 | 0.034 | 0.066 |
| sub-002 / 3 | 0.905 | 0.795 | 0.032 | 0.061 |

AUC and recall are good but precision is very low: with extreme class imbalance in the test block, the false-positive rate still swamps the few seizures.

## Files

### launch_train.py
Base code to launch a training: phase 1 (embedding, loss chosen by `EMBEDDING_TRAINING_CONFIG['loss_type']`), phase 2 (classifier head), evaluation and UMAPs. Its functions (`train_embedding_model`, `train_classifier_head`, `save_config_to_json`) are reused by `baseline_patient.py` and `optuna_search.py`.

### baseline_patient.py
Per-patient chronological block cross-validation baseline (see above).

### optuna_search.py
Optuna hyper-parameter search over `config.py` (model, loss, sampler, optimizer params); runs the full `launch_train.py` pipeline per trial, optionally one study per loss type (`--compare-losses`).

### config.py
All the configuration dicts: `DATA_CONFIG`, `MODEL_CONFIG`, `EMBEDDING_TRAINING_CONFIG`, `CLASSIFIER_TRAINING_CONFIG`, `PATIENT_BASELINE_CONFIG`, `EVAL_CONFIG`, `DEVICE_CONFIG`, `LOGGING_CONFIG`. Current values come from Optuna trial 17 of `optuna_results_supcon` (ROC AUC 0.8712).

### train.py
Training loops, one per loss (`train_model`, `train_model_triplet`, `train_model_triplet_hard_negative`, `train_model_supcon`) plus `train_classifier`.

### eval.py
Evaluation of a trained embedding: kNN (FAISS), linear probe, LOPO, retrieval metrics (P@K, R@K, mAP), classifier-head evaluation and UMAP generation. Also usable as a CLI (see Usage).

### dataset.py
Everything to build pytorch dataset which load data and create items retrieved in batch (`ContrastiveDataset`, `TripletDataset`, `EmbeddingDataset`, `PKBatchSampler`).

### loss.py
All the losses (contrastive, triplets, batch hard triplet, supervised contrastive).

### model.py
Contains the pytorch embeddings classes (`DeepResidualEmbeddingModel`, `SeizureClassifier`).

### visualization.py
Everything to create UMAPs to visualize embedding space in 2D, training-history and evaluation plots.

### PLAN.md
Experiment plan, training strategy and dated run log / conclusions.

### Generated folders
- `checkpoints/` — `model_<epoch>.pth` from `launch_train.py`, and `checkpoints/patient_baseline/` from the per-patient baseline
- `results_*/` — outputs of each run (metrics, configs, plots, `embedding_model.pth`)
- `configs/` — currently only a `__pycache__`, no config files


## What Laura did

### Step 1
Simple depth model, trial of contrastive loss and triplet loss
Not converging, UMAP and embedding PCA shows mixed seizure/non seizure

### Step 2
Denser model with triplet loss. Need to have residual connexion to avoid vanishing gradients.
But the triplets are not well chosen (not balanced, and no hard enough).

Trial of projection head to reduce the model only during training. Not good for now.

### Step 3
Work on hard tiplets. If the triplets are not hard enough, the model cannot learn.
With BatchHardTripletLoss, triplets are not done in the Dataset, but chosen during the loss computation.

### Observation

Training data need to be balanced (too many non seizure).

TODO: try embedding only with patients to capture patients differences.

This is unsupervised in terms of class (e.g., seizure/no seizure), but weakly supervised using patient identity.
I also increased the learning rate

/!\ It seems that with Laura's implementation, loss cannot go under 1 -> depends on marging

Contrastive works well for separating 3 patients with margin = 1, but not so well on labels 1 or 0
Lowering margin to 0.5 gives an embedding that separates better the labels than the patients.


### Things to experiment
Multi-stage transfer learning approach:                                                                                
  1. Phase 1: Learn rich patient embeddings (discriminate between patients)                                                                       
  2. Phase 2: Fine-tune for seizure detection (unfreeze some layers)


### Update from 29/03/2026 by Laura
- Simple contrastive learning was ineficient. It was only making one particular pair similar or disimilar
- Implement new supervised contrastive loss (from infoNCE paper) which is makes positives more similar than ALL negatives
- PKBatchSampler : P patients × K samples per batch for diverse batches with BatchHardTripletLoss  
- Create eval script with metrics on embedding
- Fine tune embedding with claissifier (unfreeze all embeddings weights)
- Reduced the dimension of the embedding

### Update from 30/08/2026 by Fahira
- Ported the pipeline to the SeizeIT feature grid (12 patients, HRV + ACC-norme features)
- Added `optuna_search.py` and tuned the hyper-parameters (best trial: ROC AUC 0.8712)
- Added `baseline_patient.py`: per-patient chronological 4-fold baseline (first results on sub-001 / sub-002: AUC 0.83–0.96, recall 0.62–0.93, very low precision)