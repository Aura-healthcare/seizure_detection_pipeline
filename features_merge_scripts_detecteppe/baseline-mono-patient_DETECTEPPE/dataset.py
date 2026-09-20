import torch
from torch.utils.data import Dataset, Sampler
import numpy as np
import random
from collections import defaultdict

class ContrastiveDataset(Dataset):
    def __init__(self, features, labels, patient_ids=None, num_pairs=1000):
        self.features = features
        self.labels = labels
        self.patient_ids = patient_ids
        self.num_pairs = num_pairs

        # Index by (label, patient) if patient_ids provided, else by label only
        self.label_to_indices = defaultdict(list)
        self.label_patient_to_indices = defaultdict(list)
        for idx, label in enumerate(labels):
            lbl = label.item()
            self.label_to_indices[lbl].append(idx)
            if patient_ids is not None:
                pid = patient_ids[idx].item()
                self.label_patient_to_indices[(lbl, pid)].append(idx)

    def _sample_positive(self, idx1, label):
        """Sample a positive: prefer different patient, same label."""
        if self.patient_ids is not None:
            pid1 = self.patient_ids[idx1].item()
            # Cross-patient candidates: same label, different patient
            cross_patient = [
                i for (lbl, pid), pool in self.label_patient_to_indices.items()
                if lbl == label and pid != pid1
                for i in pool
            ]
            if cross_patient:
                return random.choice(cross_patient)
        # Fallback: within-patient (or no patient info)
        candidates = [i for i in self.label_to_indices[label] if i != idx1]
        return random.choice(candidates) if candidates else idx1

    def __len__(self):
        return self.num_pairs

    def __getitem__(self, idx):
        # 50% positive pair, 50% negative pair
        if np.random.random() < 0.5:
            # Positive pair - same label, prefer cross-patient
            label = random.choice(list(self.label_to_indices.keys()))
            if len(self.label_to_indices[label]) >= 2:
                idx1 = random.choice(self.label_to_indices[label])
                idx2 = self._sample_positive(idx1, label)
                pair_label = 1.0  # positive pair
            else:
                idx1 = np.random.randint(0, len(self.features))
                idx2 = np.random.randint(0, len(self.features))
                pair_label = 1.0 if self.labels[idx1] == self.labels[idx2] else 0.0
        else:
            # Negative pair - different labels
            all_labels = list(self.label_to_indices.keys())
            if len(all_labels) >= 2:
                label1, label2 = random.sample(all_labels, 2)
                idx1 = random.choice(self.label_to_indices[label1])
                idx2 = random.choice(self.label_to_indices[label2])
                pair_label = 0.0  # negative pair
            else:
                idx1 = np.random.randint(0, len(self.features))
                idx2 = np.random.randint(0, len(self.features))
                pair_label = 1.0 if self.labels[idx1] == self.labels[idx2] else 0.0

        anchor = self.features[idx1]
        contrastive = self.features[idx2]
        anchor_label = self.labels[idx1]

        return anchor, contrastive, torch.tensor(pair_label, dtype=torch.float32), anchor_label

class TripletDataset(Dataset):
    def __init__(self, features, labels, patient_ids):
        self.features = features
        self.labels = labels
        self.patient_ids = patient_ids
        
        self.label_patient_index = self._build_index()
        self.triplets = self._create_triplets()
        
    def _build_index(self):
        index = defaultdict(list)
        for i, (lbl, pid) in enumerate(zip(self.labels.tolist(), self.patient_ids.tolist())):
            index[(lbl, pid)].append(i)
        return index

    def _create_triplets(self):
        # Build a label-only index for cross-patient positive lookup
        label_to_all_indices = defaultdict(list)
        for (label, patient), pool in self.label_patient_index.items():
            label_to_all_indices[label].extend(pool)

        triplets = []
        for (label, patient), anchor_pool in self.label_patient_index.items():
            # Negative candidates: different label, any patient
            negative_candidates = []
            for (neg_label, _), neg_pool in self.label_patient_index.items():
                if neg_label != label:
                    negative_candidates.extend(neg_pool)

            if not negative_candidates:
                continue

            # Cross-patient positive candidates: same label, different patient
            cross_patient_positives = [
                i for (pos_label, pos_patient), pos_pool in self.label_patient_index.items()
                if pos_label == label and pos_patient != patient
                for i in pos_pool
            ]
            # Fall back to within-patient if no cross-patient positives exist
            use_cross_patient = len(cross_patient_positives) > 0

            for anchor_idx in anchor_pool:
                if use_cross_patient:
                    positive_pool = cross_patient_positives
                else:
                    positive_pool = [i for i in anchor_pool if i != anchor_idx]

                if not positive_pool:
                    continue

                # Sample one positive and one negative per anchor
                positive_idx = random.choice(positive_pool)
                negative_idx = random.choice(negative_candidates)
                triplets.append((anchor_idx, positive_idx, negative_idx))

        return triplets

    def __len__(self):
        return len(self.triplets)

    def __getitem__(self, idx):
        anchor_idx, positive_idx, negative_idx = self.triplets[idx]
        return (
            self.features[anchor_idx],
            self.features[positive_idx],
            self.features[negative_idx],
            self.labels[anchor_idx],
            self.patient_ids[anchor_idx],
        )

class EmbeddingDataset(Dataset):
    def __init__(self, features, labels, patient_ids):
        self.features = features  # Tensor or array of shape (N, D)
        self.labels = labels      # Tensor or array of shape (N,)
        self.patient_ids = patient_ids  # Tensor or array of shape (N,)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return (
            self.features[idx],
            self.labels[idx],
            self.patient_ids[idx],
        )


class PKBatchSampler(Sampler):
    """
    Yields batches of P*K samples guaranteeing patient diversity.

    When labels are provided (recommended for SupCon), each batch is
    class-balanced: P*K // n_classes samples from each class, spread across
    diverse patients. This ensures every SupCon batch has meaningful
    same-label positives even when classes are heavily imbalanced (e.g. 5%
    seizure rate).

    The max_per_patient parameter caps how many samples a single patient
    contributes per class per batch. Lower values force more cross-patient
    diversity, which prevents the model from learning patient identity as
    a shortcut for same-label similarity.

    Usage (patient-only, for BatchHardTripletLoss):
        sampler = PKBatchSampler(dataset.patient_ids, P=8, K=8)

    Usage (class-balanced, recommended for SupCon):
        sampler = PKBatchSampler(dataset.patient_ids, P=8, K=8, labels=dataset.labels,
                                 max_per_patient=4)
        loader = DataLoader(dataset, batch_sampler=sampler)
    """

    def __init__(self, patient_ids, P=8, K=8, labels=None, max_per_patient=None):
        self.P = P
        self.K = K
        self.max_per_patient = max_per_patient

        self.pid_to_indices = defaultdict(list)
        for idx, pid in enumerate(patient_ids):
            pid = pid.item() if hasattr(pid, 'item') else pid
            self.pid_to_indices[pid].append(idx)

        self.valid_pids = [pid for pid, idxs in self.pid_to_indices.items() if len(idxs) >= K]

        # Class-stratified index: class → patient → [indices]
        if labels is not None:
            self.class_pid_indices = defaultdict(lambda: defaultdict(list))
            for idx, (pid, lbl) in enumerate(zip(patient_ids, labels)):
                pid = pid.item() if hasattr(pid, 'item') else pid
                lbl = int(lbl.item() if hasattr(lbl, 'item') else lbl)
                self.class_pid_indices[lbl][pid].append(idx)
        else:
            self.class_pid_indices = None

    def _draw(self, pool, k):
        """Sample k items without replacement when possible."""
        return random.sample(pool, k) if len(pool) >= k else random.choices(pool, k=k)

    def __iter__(self):
        if self.class_pid_indices is not None:
            yield from self._class_balanced_iter()
        else:
            yield from self._patient_iter()

    def _patient_iter(self):
        pids = self.valid_pids.copy()
        random.shuffle(pids)
        batch = []
        for pid in pids:
            batch.extend(self._draw(self.pid_to_indices[pid], self.K))
            if len(batch) >= self.P * self.K:
                yield batch
                batch = []

    def _class_balanced_iter(self):
        """Yield batches with equal samples per class, maximizing cross-patient diversity."""
        classes = sorted(self.class_pid_indices.keys())
        n_cls = len(classes)
        per_class = (self.P * self.K) // n_cls

        for _ in range(self.__len__()):
            batch = []
            for cls in classes:
                pid_dict = self.class_pid_indices[cls]
                all_class_indices = [i for idxs in pid_dict.values() for i in idxs]

                # Cap per-patient contribution to force cross-patient diversity
                cap = self.max_per_patient if self.max_per_patient is not None else self.K

                # Collect per_class samples, cycling through patients
                pids = list(pid_dict.keys())
                random.shuffle(pids)
                collected = []

                # First pass: draw up to `cap` samples from each patient
                for pid in pids:
                    need = per_class - len(collected)
                    if need <= 0:
                        break
                    collected.extend(self._draw(pid_dict[pid], min(cap, need)))

                # Second pass: if we still need more (few patients), cycle again
                if len(collected) < per_class:
                    random.shuffle(pids)
                    for pid in pids:
                        need = per_class - len(collected)
                        if need <= 0:
                            break
                        collected.extend(self._draw(pid_dict[pid], min(cap, need)))

                # Final fallback: fill from full class pool
                if len(collected) < per_class:
                    collected.extend(random.choices(all_class_indices, k=per_class - len(collected)))

                batch.extend(collected[:per_class])

            random.shuffle(batch)
            yield batch

    def __len__(self):
        if self.class_pid_indices is not None:
            # Base on minority class size so every sample gets roughly one turn per epoch
            classes = list(self.class_pid_indices.keys())
            n_cls = len(classes)
            per_class = (self.P * self.K) // n_cls
            min_class_size = min(
                sum(len(idxs) for idxs in self.class_pid_indices[cls].values())
                for cls in classes
            )
            return max(1, min_class_size // per_class)
        return len(self.valid_pids) // self.P