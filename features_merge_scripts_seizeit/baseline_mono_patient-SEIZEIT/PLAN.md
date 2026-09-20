## Plan

1. Pretrain with contrastive learning
2. Fine-tune with: (partial fine tuning since not a lot of datasets or full fine tuning with low learning rate)
	- classification loss
	- contrastive loss (weighted) : L=Lclassification​+λ⋅Lcontrastive​ (λ = 0.1 or 0.2)
3. Evaluate with:
	- kNN : train embedding + classify using k-nearest neighbors in embedding space
	- linear probe : train embedding + freeze and train logistic regression classifier
	- LOPO classifier
	- retrieval metrics : Precision@K, Recall@K, Mean Average Precision (mAP) -> using cosine similarity for distance
4. Visualize embedding:
	- color by patient vs label



## Training strategy:
1. training embedding + total freeze and train classifier
2. training embedding and classifier at same time -> hard
3. training embedding + fine tune with classifier -> best approach


## Positive / Negative pair choice:
Focus first on good contrastive loss (then after triplet, more complex)
We want invariance to:
- patient identity
- baseline cardiac differences
- sensor noise

So positives could be:
- same seizure segment under augmentations
- nearby windows within the same seizure
- maybe similar physiological states

Negatives:
- seizure vs non-seizure (obvious)
- BUT also: non-seizure from different contexts
Pitfall:If we define positives only within-patient, we risk learning patient-specific embeddings, which is exactly what we don’t want.

Consider explicitly encouraging:
- cross-patient positives for seizures
- or patient-adversarial training


## Run log

### 2026-03-31 — Baseline run (results_20260331_234404)
- SupCon loss, embedding_dim=64, 2 residual blocks, temperature=0.07
- PKBatchSampler P=16, K=16 (class-balanced)
- Classifier: 5 epochs, freeze_embeddings=False, lr=1e-4
- Per-patient normalization enabled
- **Results**: Train AUC 0.997 / Test AUC 0.622. UMAP clusters by patient, not seizure — model learns patient identity. Classifier barely detects seizures on test (F1=0.012, recall=0.025).

### 2026-04-01 — Cross-patient diversity + class-weighted SupCon + FAISS eval
- **PKBatchSampler**: added `max_per_patient=4` cap — limits each patient to 4 samples per class per batch, forcing SupCon positives to come from many different patients instead of repeating the same patient 16 times
- **Class-weighted SupCon loss**: each anchor's loss is weighted by inverse class frequency so seizure anchors contribute equally despite ~800:1 imbalance
- **FAISS eval**: replaced sklearn brute-force kNN and full similarity matrix retrieval with FAISS `IndexIVFFlat` (approximate cosine search) — scales to 1M+ embeddings in seconds
- **Classifier**: freeze_embeddings=True


## Conclusion
- 2026-04-01 : after embedding training, the seizure points collapse at the same place (cf UMAP). 
Most probably because for each batch we select 128 non seizure and 128 seizure examples and we mostly select multiple times the same seizures accross all epochs. -> reduce epochs and increase embedding regularization
Also, training the classifier with unfreezed embeddings layers totally change the embedding and the seizure points are more scattered. -> bug in UMAP
-> exclude same patient pairs in the supervised contrastive learning.
-> Rework train / test set? maybe test patient should be in the training set
-> Work on feature to make them patient agnostic?

After analyzing train / test data, 
1. 01-009 dominates the test seizures (2048 out of 2381 = 86%) -> too decisive on model eval
2. test patients' seizures are different from train patients' seizures (max similarity ~0.44) cf dataset_analysis_report.html

We need to focus on the embedding to capture transferable seizure signal.
1. Data augmentation on seizure samples (highest impact)                                                                                         
With only 1347 seizure samples from 4 patients, the model memorizes patient-specific seizure patterns. Augmentation forces it to learn invariant 
features. Concrete: add random noise, feature dropout (mask random features to 0), and mixup between seizure samples from different patients.    
																																				
2. Feature-level attention based on seizure consistency                                                                                          
Your analysis showed which features change consistently during seizures across patients. We can add a learnable feature weighting layer that the
model can use to down-weight patient-specific features and up-weight shared seizure features.                                                    

3. Hard negative mining                                                                                                                          
Currently all non-seizure samples are equal negatives. But the most informative negatives are non-seizure samples that are close to seizure in 
embedding space — these force the model to find the real boundary. We can mine these within each batch.  


## Future ideas

### Patient-adversarial training (not yet implemented)
Add a gradient reversal layer (GRL) to discourage patient-specific features in the embedding:

1. Add a small MLP head on top of embeddings that predicts patient ID
2. Insert a gradient reversal layer before it: during forward pass it's identity, during backward pass it flips the gradient sign
3. Total loss: `L = L_supcon + λ * L_patient_adversarial`
4. Effect: the embedding model is rewarded for SupCon similarity but *punished* for encoding patient identity
5. λ controls the trade-off — start with 0.1, tune on validation

Reference: Ganin et al. "Domain-Adversarial Training of Neural Networks" (2016) — same principle applied to patient domains instead of source/target domains.