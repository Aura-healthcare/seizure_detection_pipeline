"""
Configuration file for contrastive learning seizure detection pipeline.

Dataset: mini-dataset2 feat-grid-all-intersect (14 HRV + 27 ACC-norme features,
columns: patient_id, label; no split column — train/test assigned via DATA_CONFIG['patient_split'])
"""

# ============================================================================
# DATA CONFIGURATION
# ============================================================================
import os 

DATA_CONFIG = {
    'data_path': '/home/aura-fahira/data-preparation-seizeit/seizure_detection_pipeline/features_merge_scripts_seizeit/mini-dataset2/feat-grid-all-intersect.csv',
    'patient_id_col': 'patient_id',
    'label_col': 'label',
    'availability_cols': ['hrv_available', 'acc_available'],  # sensor availability flags

    'checkpoint_dir': './checkpoints/',
    'results_dir': None,  # Auto-generated with timestamp if None
    'num_patients_train': None,  # Limit training patients (None for all)
    'use_weighted_sampling': True,
    'per_patient_normalization': True,  # z-score features relative to each patient's baseline

    # Ratio de sous-échantillonnage des non-crises (train uniquement).
    # Pour chaque crise, on garde au maximum `undersampling_ratio` non-crises.
    # Ex : 3 → ratio 3:1 (non-crise/crise). None pour désactiver.
    'undersampling_ratio': 50,  # best from optuna_results_supcon (trial 17, roc_auc=0.8712)

    # Assignation train/test par patient (le dataset n'a pas de colonne de split).
    # Tous les patient_id du dataset doivent être répartis dans une des deux listes.
    'patient_split': {
        'train': ['sub-001', 'sub-002', 'sub-011', 'sub-012', 'sub-021', 'sub-022',
                   'sub-030', 'sub-031', 'sub-044', 'sub-046'],
        'test': ['sub-040', 'sub-050'],
    },


    # Features to drop
    'drop_columns': [

    'elapsed_s',
    'timestamp',
    'seizure_type',
    'run_id',

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

    'acc_EEG_SD_ACC_x_autocorr','acc_EEG_SD_ACC_x_zero_crossing','acc_EEG_SD_ACC_x_mean_abs_diff','acc_EEG_SD_ACC_x_distance',
    'acc_EEG_SD_ACC_x_sum_abs_diff','acc_EEG_SD_ACC_x_slope','acc_EEG_SD_ACC_x_abs_energy','acc_EEG_SD_ACC_x_pk_pk_distance',
    'acc_EEG_SD_ACC_x_entropy','acc_EEG_SD_ACC_x_max','acc_EEG_SD_ACC_x_std','acc_EEG_SD_ACC_x_mean','acc_EEG_SD_ACC_x_spectral_distance',
    'acc_EEG_SD_ACC_x_wavelet_entropy','acc_EEG_SD_ACC_x_spectral_entropy','acc_EEG_SD_ACC_x_power_bandwidth',
    'acc_EEG_SD_ACC_x_human_range_energy','acc_EEG_SD_ACC_x_spectral_roll_on','acc_EEG_SD_ACC_x_spectral_roll_off',
    'acc_EEG_SD_ACC_x_spectral_variation','acc_EEG_SD_ACC_x_spectral_slope','acc_EEG_SD_ACC_x_spectral_kurtosis',
    'acc_EEG_SD_ACC_x_spectral_decrease','acc_EEG_SD_ACC_x_spectral_centroid','acc_EEG_SD_ACC_x_median_frequency',
    'acc_EEG_SD_ACC_x_max_power_spectrum','acc_EEG_SD_ACC_x_max_frequency','acc_EEG_SD_ACC_x_fundamental_frequency',

    'acc_EEG_SD_ACC_y_autocorr','acc_EEG_SD_ACC_y_zero_crossing','acc_EEG_SD_ACC_y_mean_abs_diff','acc_EEG_SD_ACC_y_distance',
    'acc_EEG_SD_ACC_y_sum_abs_diff','acc_EEG_SD_ACC_y_slope','acc_EEG_SD_ACC_y_abs_energy','acc_EEG_SD_ACC_y_pk_pk_distance',
    'acc_EEG_SD_ACC_y_entropy','acc_EEG_SD_ACC_y_max','acc_EEG_SD_ACC_y_std','acc_EEG_SD_ACC_y_mean','acc_EEG_SD_ACC_y_spectral_distance',
    'acc_EEG_SD_ACC_y_wavelet_entropy','acc_EEG_SD_ACC_y_spectral_entropy','acc_EEG_SD_ACC_y_power_bandwidth',
    'acc_EEG_SD_ACC_y_human_range_energy','acc_EEG_SD_ACC_y_spectral_roll_on','acc_EEG_SD_ACC_y_spectral_roll_off',
    'acc_EEG_SD_ACC_y_spectral_variation','acc_EEG_SD_ACC_y_spectral_slope','acc_EEG_SD_ACC_y_spectral_kurtosis',
    'acc_EEG_SD_ACC_y_spectral_decrease','acc_EEG_SD_ACC_y_spectral_centroid','acc_EEG_SD_ACC_y_median_frequency',
    'acc_EEG_SD_ACC_y_max_power_spectrum','acc_EEG_SD_ACC_y_max_frequency','acc_EEG_SD_ACC_y_fundamental_frequency',

    'acc_EEG_SD_ACC_z_autocorr','acc_EEG_SD_ACC_z_zero_crossing','acc_EEG_SD_ACC_z_mean_abs_diff','acc_EEG_SD_ACC_z_distance',
    'acc_EEG_SD_ACC_z_sum_abs_diff','acc_EEG_SD_ACC_z_slope','acc_EEG_SD_ACC_z_abs_energy','acc_EEG_SD_ACC_z_pk_pk_distance',
    'acc_EEG_SD_ACC_z_entropy','acc_EEG_SD_ACC_z_max','acc_EEG_SD_ACC_z_std','acc_EEG_SD_ACC_z_mean','acc_EEG_SD_ACC_z_spectral_distance',
    'acc_EEG_SD_ACC_z_wavelet_entropy','acc_EEG_SD_ACC_z_spectral_entropy','acc_EEG_SD_ACC_z_power_bandwidth',
    'acc_EEG_SD_ACC_z_human_range_energy','acc_EEG_SD_ACC_z_spectral_roll_on','acc_EEG_SD_ACC_z_spectral_roll_off',
    'acc_EEG_SD_ACC_z_spectral_variation','acc_EEG_SD_ACC_z_spectral_slope','acc_EEG_SD_ACC_z_spectral_kurtosis',
    'acc_EEG_SD_ACC_z_spectral_decrease','acc_EEG_SD_ACC_z_spectral_centroid','acc_EEG_SD_ACC_z_median_frequency',
    'acc_EEG_SD_ACC_z_max_power_spectrum','acc_EEG_SD_ACC_z_max_frequency','acc_EEG_SD_ACC_z_fundamental_frequency',

    'acc_EEG_SD_ACC_norme_1_spectral_roll_on',

    'acc_EEG_SD_ACC_F_0_1',
    'acc_EEG_SD_ACC_F_1_2',
    'acc_EEG_SD_ACC_F_2_3',
    'acc_EEG_SD_ACC_F_3_4',
    'acc_EEG_SD_ACC_F_4_5',
    'acc_EEG_SD_ACC_F_5_6',
    'acc_EEG_SD_ACC_F_6_7',
    'acc_EEG_SD_ACC_F_7_8',
    'acc_EEG_SD_ACC_F_8_9',
    'acc_EEG_SD_ACC_F_9_10',
    'acc_EEG_SD_ACC_F_10_11',
    'acc_EEG_SD_ACC_F_11_12',
    'acc_EEG_SD_ACC_F_custom_0_5',
    'acc_EEG_SD_ACC_F_custom_5_8',
    'acc_EEG_SD_ACC_F_custom_8_12',
    'acc_EEG_SD_ACC_F_custom_12_25',
],
}


# ============================================================================
# MODEL CONFIGURATION
# ============================================================================

# input_dim is computed automatically from the data in launch_train.py
# Set to None for auto-detection, or override with an int.
MODEL_CONFIG = {
    'input_dim': None,  # Auto-detected from data (feature set changed with new ECGEMG_ACC dataset)
    'embedding_dim': 32,  # Reduced from 128 (better for generalisation)
    'num_classes': 2,

    'num_residual_blocks': 3,  # best from optuna_results_supcon (trial 17, roc_auc=0.8712)
    'embedding_dropout': 0.4166883665788511,  # best from optuna_results_supcon (trial 17, roc_auc=0.8712)

    'classifier_hidden_dims': [64,32],
    'classifier_dropout': 0.15557567123775703,  # best from optuna_results_supcon (trial 17, roc_auc=0.8712)
}


# ============================================================================
# TRAINING CONFIGURATION - EMBEDDING MODEL
# ============================================================================

EMBEDDING_TRAINING_CONFIG = {
    'epochs': 50,   # Reduced from 100 — fewer seizure samples get over-compressed with too many epochs
    'batch_size': 64,
    'learning_rate': 0.004335340021600169,  # best from optuna_results_supcon (trial 17, roc_auc=0.8712)

    # Loss function options: 'contrastive', 'triplet', 'batch_hard_triplet', 'supcon'
    'loss_type': 'supcon',

    # Loss function parameters
    'margin': 1.0,              # For triplet and batch_hard_triplet
    'temperature': 0.23362258369500222,  # best from optuna_results_supcon (trial 17, roc_auc=0.8712)
    'num_pairs': 5000,          # For contrastive dataset
    'pk_p': 2,                 # PKBatchSampler: number of patients per batch (supcon)
    'pk_k': 16,                 # best from optuna_results_supcon (trial 17, roc_auc=0.8712)
    'max_per_patient': None,       # Cap per-patient samples per class per batch (forces cross-patient diversity)

    # Learning rate scheduler
    'use_scheduler': True,
    'scheduler_step_size': 7,  # best from optuna_results_supcon (trial 17, roc_auc=0.8712)
    'scheduler_gamma': 0.7608825328266291,  # best from optuna_results_supcon (trial 17, roc_auc=0.8712)

    # Checkpointing
    'save_checkpoints': True,
    'checkpoint_frequency': 1,  # Save every N epochs
}


# ============================================================================
# TRAINING CONFIGURATION - CLASSIFIER
# ============================================================================

CLASSIFIER_TRAINING_CONFIG = {
    'epochs': 30,  # Set low for testing (use 20+ for real training)
    'batch_size': 128,
    'learning_rate': 0.003079310150378342,  # best from optuna_results_supcon (trial 17, roc_auc=0.8712)

    # Learning rate scheduler
    'use_scheduler': True,
    'scheduler_step_size': 5,
    'scheduler_gamma': 0.5,

    # Fine-tuning options (Two-stage approach)
    'freeze_embeddings': True,  # Changed: freeze embeddings by default
    'unfreeze_last_n_blocks': None,
}


# ============================================================================
# PER-PATIENT BASELINE CONFIGURATION (baseline_patient.py)
# ============================================================================
# Trains/evaluates the full embedding+classifier pipeline independently for
# each patient. Each patient's rows (already in chronological order in the
# CSV: sorted by run then elapsed_s) are split into `n_folds` contiguous
# intervals; each fold uses one interval as test and the rest as train,
# rotating so every interval is used as test exactly once.

PATIENT_BASELINE_CONFIG = {
    'patients': None,   # None = every patient found in the CSV; else a list like ['sub-001', 'sub-002']
    'n_folds': 4,        # number of contiguous chronological intervals per patient
    'results_dir': None,  # Auto-generated with timestamp if None
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
