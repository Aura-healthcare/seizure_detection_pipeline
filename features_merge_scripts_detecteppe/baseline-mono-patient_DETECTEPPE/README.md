Started this work to study how we could could build an embedding which can capture the variability of seizure / non seizure but also the variability inter patients.

This is based on following papers :

SimCLR from google brain https://arxiv.org/abs/2002.05709 
MoCo from facebook ai https://arxiv.org/abs/1911.05722
InfoNCE from google https://arxiv.org/pdf/2004.11362


Interesting paper on contrastive learning : https://arxiv.org/pdf/2010.05113

Contrastive learning in ECG or EEG :  https://pubmed.ncbi.nlm.nih.gov/37028019/

Work done on Teppe dataset (29 HRV + 141 accelerometer features, 9 patients).

## Usage

Install uv
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Update the conf file `config.py` to experiment.

Launch training of embedding and classification head
```bash
uv run python launch_train.py
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

Evaluate all 4 folds at once and get an aggregated cross-validation summary
```bash
# Point --results_dir to the parent directory (containing fold_0/, fold_1/, ...)
uv run python eval.py \
    --results_dir ./grid_results/feat-grid-intersect_01-001_triple/ratio_50_emb_64/ \
    --all_folds

# Same, without UMAP generation (faster)
uv run python eval.py \
    --results_dir ./grid_results/feat-grid-intersect_01-001_triple/ratio_50_emb_64/ \
    --all_folds --no_umap
```

`--all_folds` automatically discovers all `fold_X/` subdirectories, evaluates each one,
saves `eval_metrics.json` and `report.html` per fold, then prints a summary table
(kNN-F1, kNN-AUC, LP-F1, LP-AUC per fold + mean/std) and writes
`eval_cv_summary.json` in the parent directory.

> **Note — single-patient (baseline) mode**: `eval.py` automatically detects whether the
> results come from a per-fold temporal split (baseline) rather than a column-based
> train/test split. The test fold is inferred from the directory name (`fold_0` → `test_fold=0`).

> **Note — results portability**: `training_config.json` no longer stores the CSV's
> absolute path (only `dataset_name`), so a results folder can be moved/shared. If
> `data_path` is missing from the config, `eval.py` reconstructs the path from the
> `AURA_DATA_DIR` environment variable (e.g. `AURA_DATA_DIR=/path/to/csvs uv run python eval.py ...`).

> **Note — richer evaluation reports**: in addition to the kNN / linear probe / LOPO /
> retrieval metrics, `eval.py` now also generates:
> - a **per-seizure error report** (`seizure_errors_train.csv`, `seizure_errors_test.csv`)
>   classifying each seizure as `missed` / `partial` / `fully_detected`, embedded in `report.html`;
> - a **UMAP of prediction errors** (TP/FP/FN/TN), in static (`*_errors.png`) and
>   interactive (`*_errors_interactive.html`, hover with `patient_id` / `seizure_id`) form,
>   for both train and test.

---

## Grid Search — `run_grid_search.py`

Runs an exhaustive hyperparameter search using a 4-fold cross-validation for each combination of:

| Parameter | Values tested |
|---|---|
| `undersampling_ratio` | 5, 10, 20, 50 |
| `embedding_dim` | 16, 32, 64 |
| patients | all `baseline-patient-*.csv` files found in `data_dir` |

That's **12 configs × N patients × 4 folds** of training in total.

### Quick start

```bash
# Uses data_dir inferred from DATA_CONFIG['data_path'] in config.py
uv run python run_grid_search.py

# Explicit data and output directories
uv run python run_grid_search.py \
    --data_dir /path/to/csvs \
    --output ./grid_results

# Single patient / single CSV file
uv run python run_grid_search.py \
    --patient_csv /path/to/feat-grid-intersect_01-001.csv \
    --output ./grid_results
```

### Arguments

| Argument | Default | Description |
|---|---|---|
| `--data_dir` | parent folder of `DATA_CONFIG['data_path']` | Folder containing the `baseline-patient-*.csv` files |
| `--patient_csv` | `None` | Runs the grid on a single patient CSV file; if given, `--data_dir` is ignored |
| `--output` | `./grid_results` | Root folder where all results are written |

### Results structure

```
grid_results/
├── grid_summary.csv                  ← comparison table of all configs
└── <patient_id>/
    └── ratio_<R>_emb_<E>/
        ├── cv_summary.json           ← average metrics (AUC, F1, recall…)
        ├── cv_summary.png            ← summary plot
        └── fold_<k>/
            ├── config.json
            ├── embedding_model.pth
            ├── classifier.pth
            └── umap_pretrain*.png
```

### Automatic resume

If `cv_summary.json` already exists in a config's folder, that config is skipped. You can therefore interrupt the script and rerun it without starting over.

### Reading the results

```python
import pandas as pd

df = pd.read_csv("grid_results/grid_summary.csv")

# Best config per patient
df.sort_values("mean_roc_auc", ascending=False).groupby("patient").first()

# Best config on average across all patients
df.groupby(["undersampling_ratio", "embedding_dim"])["mean_roc_auc"].mean().idxmax()
```

The script also prints this summary directly in the logs at the end of the run.

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

**Important — update:** in the current training loop (`train_model_triplet_hard_negative`), the loss now receives the **seizure/non-seizure labels** (`labels`), rather than `patient_ids`. In single-patient baseline mode, `patient_ids` is constant across an entire fold — using it as the signal for `BatchHardTripletLoss` would be degenerate (no cross-class pairs to separate). Learning therefore happens directly on the seizure/non-seizure task from Phase 1 onward, rather than via the patient→seizure multi-stage approach described below (which remains relevant in a multi-patient setup). A `nan_to_num` safeguard also neutralises any residual `Inf`/`NaN` in the features right before the forward pass.

*Original approach (multi-patient, historical): the loss received `patient_ids` as labels, training the model to **discriminate between patients** first — learning a rich, patient-aware embedding — before fine-tuning for seizure detection in phase 2 (multi-stage transfer learning approach).*

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

## Files

### launch_train.py
Contains the base code to launch a training. Runs a 4-fold temporal cross-validation
for the single-patient baseline: each fold's test set is a chronological quarter of the
recording. The classifier's class-weighted loss is capped at a 50:1 ratio (and disabled
entirely when a fold has no seizures) to avoid double-correcting on top of the
`WeightedRandomSampler`, which previously could collapse the classifier to predicting
everything positive.

### eval.py
Evaluation script. Computes kNN, linear probe, LOPO, retrieval metrics and UMAP
visualizations on a trained model. Supports single-fold and `--all_folds` (aggregated
CV summary). Auto-detects baseline single-patient mode from the directory name and
reconstructs the data path from `AURA_DATA_DIR` when needed. Also produces a
per-seizure error report (`missed` / `partial` / `fully_detected`) and UMAP plots
colored by prediction outcome (TP/FP/FN/TN), both static and interactive with hover
metadata — all embedded in `report.html`.

### dataset.py
Everything to build pytorch dataset which load data and create items retrieved in batch.

### loss.py
All the losses (contrastive, triplets).

### model.py
Contains the pytorch embeddings classes.

### visualization.py
Everything to create UMAPs to visualize embedding space in 2D, including error-outcome
UMAPs (TP/FP/FN/TN, static + interactive) and the cross-validation summary plot
(`cv_summary.png`).

### run_grid_search.py
Grid search over hyperparameters (undersampling ratio, embedding dim) across patients.


## What was done so far

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
Laura also increased the learning rate

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

### Update from April / May 2026 by Fahira — single-patient CV, error diagnostics & robustness
- Switched from a global multi-patient train/test split to a 4-fold **temporal
  cross-validation on a single patient** (`launch_train.py`), with per-fold
  `cv_summary.json` / `cv_summary.png` and aggregation across folds
- Fixed a classifier loss bug: uncapped inverse-frequency class weights combined with
  `WeightedRandomSampler` were double-correcting for imbalance and collapsing the
  classifier (recall≈1, precision≈1%) — now capped at 50:1, disabled when a fold has no
  seizures
- `eval.py` now auto-detects single-patient/fold mode from the results directory name
  and supports `--all_folds` for an aggregated cross-validation report
- Added a **per-seizure error report** (missed / partial / fully detected) and
  **UMAP error visualizations** (TP/FP/FN/TN), static and interactive with hover
  metadata (patient_id, seizure_id) — both surfaced in `report.html`
- `training_config.json` no longer stores an absolute data path (portability); `eval.py`
  falls back to the `AURA_DATA_DIR` environment variable to relocate the dataset
- Handled degenerate folds (single class, NaN embeddings) gracefully instead of crashing
- `train_model_triplet_hard_negative` (Phase 1, `batch_hard_triplet` loss) now trains on
  **seizure labels** instead of `patient_ids` — since `patient_ids` is constant within a
  single-patient fold and would give the loss no signal — and guards against residual
  NaN/Inf in features before the forward pass
