
"""
Configuration file for contrastive learning seizure detection pipeline.

Dataset: Teppe (14 HRV + 31 ACC features, columns: patient-id, training-split, label)
"""

# ============================================================================
# DATA CONFIGURATION
# ============================================================================
import os 

from pathlib import Path

DATA_ROOT = Path(os.environ.get("AURA_DATA_DIR", ""))

if not DATA_ROOT.exists():
    raise RuntimeError(
        "AURA_DATA_DIR is not set or points to a non-existing directory"
    )

DATA_CONFIG = {
    "dataset_name": "feat-grid-intersect_01-013.csv",
    'data_path': DATA_ROOT / 'feat-grid-intersect_01-013.csv', # CSV n'est plus dans le workspace, on suppose qu'il est à l'extérieur et on lit le chemin depuis une variable d'environnement
    'patient_id_col': 'patient-id',
    'label_col': 'label',
    'split_col': 'training-split',
    'train_split_name': 'train',
    'test_split_name': 'test',
    'drop_columns': ['timestamps', 'hrv_interval_index', 'hrv_interval_start_time'],
    'availability_cols': ['hrv_available', 'acc_available'],  # sensor availability flags

   'checkpoint_dir': './grid_results/feat-grid-intersect_01-013_triple_errors/ratio_50_emb_64/checkpoints',
   'results_dir': './grid_results/feat-grid-intersect_01-013_triple_errors/ratio_50_emb_64',
   'num_patients_train': None,  # Limit training patients (None for all)
   'use_weighted_sampling': True,
   'per_patient_normalization': True,  # z-score features relative to each patient's baseline


    # Features to drop
    'drop_columns': [

    'timestamps',
    'hrv_available',
    'acc_available',

    'hrv_interval_index',
    'hrv_interval_start_time',
    'hrv_sdsd',
    'hrv_pnni_50',
    'hrv_pnni_20',
    'hrv_rmssd',
    'hrv_median_nni',
    'hrv_cvsd',
    'hrv_cvnni',
    'hrv_std_hr',
    'hrv_lf',
    'hrv_hf',
    'hrv_vlf',
    'hrv_lf_hf_ratio',
    'hrv_ratio_sd2_sd1',

    'acc_x_autocorr','acc_x_zero_crossing','acc_x_mean_abs_diff','acc_x_distance',
    'acc_x_sum_abs_diff','acc_x_slope','acc_x_abs_energy','acc_x_pk_pk_distance',
    'acc_x_entropy','acc_x_max','acc_x_std','acc_x_mean','acc_x_spectral_distance',
    'acc_x_wavelet_entropy','acc_x_spectral_entropy','acc_x_power_bandwidth',
    'acc_x_human_range_energy','acc_x_spectral_roll_on','acc_x_spectral_roll_off',
    'acc_x_spectral_variation','acc_x_spectral_slope','acc_x_spectral_kurtosis',
    'acc_x_spectral_decrease','acc_x_spectral_centroid','acc_x_median_frequency',
    'acc_x_max_power_spectrum','acc_x_max_frequency','acc_x_fundamental_frequency',

    'acc_y_autocorr','acc_y_zero_crossing','acc_y_mean_abs_diff','acc_y_distance',
    'acc_y_sum_abs_diff','acc_y_slope','acc_y_abs_energy','acc_y_pk_pk_distance',
    'acc_y_entropy','acc_y_max','acc_y_std','acc_y_mean','acc_y_spectral_distance',
    'acc_y_wavelet_entropy','acc_y_spectral_entropy','acc_y_power_bandwidth',
    'acc_y_human_range_energy','acc_y_spectral_roll_on','acc_y_spectral_roll_off',
    'acc_y_spectral_variation','acc_y_spectral_slope','acc_y_spectral_kurtosis',
    'acc_y_spectral_decrease','acc_y_spectral_centroid','acc_y_median_frequency',
    'acc_y_max_power_spectrum','acc_y_max_frequency','acc_y_fundamental_frequency',

    'acc_z_autocorr','acc_z_zero_crossing','acc_z_mean_abs_diff','acc_z_distance',
    'acc_z_sum_abs_diff','acc_z_slope','acc_z_abs_energy','acc_z_pk_pk_distance',
    'acc_z_entropy','acc_z_max','acc_z_std','acc_z_mean','acc_z_spectral_distance',
    'acc_z_wavelet_entropy','acc_z_spectral_entropy','acc_z_power_bandwidth',
    'acc_z_human_range_energy','acc_z_spectral_roll_on','acc_z_spectral_roll_off',
    'acc_z_spectral_variation','acc_z_spectral_slope','acc_z_spectral_kurtosis',
    'acc_z_spectral_decrease','acc_z_spectral_centroid','acc_z_median_frequency',
    'acc_z_max_power_spectrum','acc_z_max_frequency','acc_z_fundamental_frequency',

    'acc_norme_spectral_roll_on',

    'acc_F_0_1',
    'acc_F_1_2',
    'acc_F_2_3',
    'acc_F_3_4',
    'acc_F_4_5',
    'acc_F_5_6',
    'acc_F_6_7',
    'acc_F_7_8',
    'acc_F_8_9',
    'acc_F_9_10',
    'acc_F_10_11',
    'acc_F_11_12',
    'acc_F_12_13',
    'acc_F_13_14',
    'acc_F_14_15',
    'acc_F_15_16',
    'acc_F_16_17',
    'acc_F_17_18',
    'acc_F_18_19',
    'acc_F_19_20',
    'acc_F_20_21',
    'acc_F_21_22',
    'acc_F_22_23',
    'acc_F_23_24',
    'acc_F_24_25',
    'seizure-id'
],
}


# ============================================================================
# MODEL CONFIGURATION
# ============================================================================

# input_dim is computed automatically from the data in launch_train.py
# Set to None for auto-detection, or override with an int.
MODEL_CONFIG = {
    'input_dim': 45,  #None # Auto-detected from data
    'embedding_dim': 64,  # Reduced from 128 (better for generalisation)
    'num_classes': 2,

    'num_residual_blocks': 2,  # Reduced from 3
    'embedding_dropout': 0.4, # Increased from 0.2 to reduce overfitting on seizure samples

    'classifier_hidden_dims': [64, 32],
    'classifier_dropout': 0.3,
}


# ============================================================================
# TRAINING CONFIGURATION - EMBEDDING MODEL
# ============================================================================

EMBEDDING_TRAINING_CONFIG = {
    'epochs': 30,   # Reduced from 100 — fewer seizure samples get over-compressed with too many epochs
    'batch_size': 20,
    'learning_rate': 1e-3,  # Reduced from 5e-3

    # Loss function options: 'contrastive', 'triplet', 'batch_hard_triplet', 'supcon', 'simple'
    'loss_type': 'batch_hard_triplet',  # 'contrastive', 'triplet', 'batch_hard_triplet', 'supcon', 'simple'

    # Loss function parameters
    'margin': 1.0,              # For triplet and batch_hard_triplet
    'temperature': 0.07,        # Reduced from 0.3 — sharper focus on hard negatives
    'num_pairs': 5000,          # For contrastive dataset
    'pk_p': 5,                 # PKBatchSampler: number of patients per batch (supcon)
    'pk_k': 4,                 # PKBatchSampler: number of samples per patient per batch (supcon)
    'max_per_patient': 4,       # Cap per-patient samples per class per batch (forces cross-patient diversity)

    # Learning rate scheduler
    'use_scheduler': True,
    'scheduler_step_size': 10,
    'scheduler_gamma': 0.5,

    # Checkpointing
    'save_checkpoints': True,
    'checkpoint_frequency': 1,  # Save every N epochs
}


# ============================================================================
# TRAINING CONFIGURATION - CLASSIFIER
# ============================================================================

CLASSIFIER_TRAINING_CONFIG = {
    'epochs': 30,
    'batch_size': 128,
    'learning_rate': 1e-4,

    # Learning rate scheduler
    'use_scheduler': True,
    'scheduler_step_size': 5,
    'scheduler_gamma': 0.5,

    # Fine-tuning options (Two-stage approach)
    'freeze_embeddings': True,
    'unfreeze_last_n_blocks': 1,  # fine-tune uniquement le dernier bloc résiduel
}


# ============================================================================
# BASELINE (SINGLE-PATIENT) CONFIGURATION
# ============================================================================

BASELINE_CONFIG = {
    # Découpe temporelle : le CSV est supposé être en ordre chronologique.
    # n_folds=4 divise les données en 4 blocs égaux ; 3 servent au train, 1 au test.
    'n_folds': 4,
    'test_fold': 3,   # indice du pli utilisé comme test (0-indexé ; 3 = dernier quart)

    # Ratio de sous-échantillonnage des non-crises : pour chaque crise, on garde
    # au maximum `undersampling_ratio` non-crises. Ex : 3 → ratio 3:1 (non-crise/crise).
    'undersampling_ratio': 50,
}


# ============================================================================
# EVALUATION CONFIGURATION
# ============================================================================

EVAL_CONFIG = {
    'evaluate_on_train': True,  # Whether to evaluate on training set
    'evaluate_on_test': True,   # Whether to evaluate on test set
    'generate_umap': True,      # Whether to generate UMAP visualizations
    'umap_sample_size': 2000,   # Number of samples to use for UMAP (None for all)
}


# ============================================================================
# DEVICE CONFIGURATION
# ============================================================================

DEVICE_CONFIG = {
    'use_cuda': True,  # Set to False to force CPU
    'cuda_device': 0,  # GPU device number
}


# ============================================================================
# LOGGING CONFIGURATION
# ============================================================================

LOGGING_CONFIG = {
    'level': 'INFO',  # 'DEBUG', 'INFO', 'WARNING', 'ERROR'
    'format': '%(asctime)s - %(levelname)s - %(message)s',
}
