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

## Files

### launch_train.py
Contains the base code to launch a training

### dataset.py
Everything to build pytorch dataset which load data and create items retrieved in batch.

### loss.py
All the losses (contrastive, triplets).

### model.py
Contains the pytorch embeddings classes.

### visualization.py
Everything to create UMAPs to visualize embedding space in 2D.


## What I did

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

/!\ It seems that with my implementation, loss cannot go under 1 -> depends on marging

Contrastive works well for separating 3 patients with margin = 1, but not so well on labels 1 or 0
Lowering margin to 0.5 gives an embedding that separates better the labels than the patients.

UMAP on 25 patients with plotly shows complex problem.
Need to do umap for each patient.

### Things to experiment
Multi-stage transfer learning approach:                                                                                
  1. Phase 1: Learn rich patient embeddings (discriminate between patients)                                                                       
  2. Phase 2: Fine-tune for seizure detection (unfreeze some layers)


### Update from 29/03/2026
- Simple contrastive learning was ineficient. It was only making one particular pair similar or disimilar
- Implement new supervised contrastive loss (from infoNCE paper) which is makes positives more similar than ALL negatives
- PKBatchSampler : P patients × K samples per batch for diverse batches with BatchHardTripletLoss  
- Create eval script with metrics on embedding
- Fine tune embedding with claissifier (unfreeze all embeddings weights)
- Reduced the dimension of the embedding