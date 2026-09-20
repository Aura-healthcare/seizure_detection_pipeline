import torch
import numpy as np
import logging
from sklearn.decomposition import PCA
from sklearn.metrics import roc_curve
from tqdm import tqdm
from model import EmbeddingModel, DeepResidualEmbeddingModel
import plotly.graph_objects as go
import pandas as pd
from torch.utils.data import DataLoader
from dataset import ContrastiveDataset, TripletDataset, EmbeddingDataset
import umap
import umap.plot
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import Subset
import random
import os
import time

def _lighten_color(rgb, factor=0.55):
    """Mix an RGB colour with white. factor=1 → original, factor=0 → white."""
    return tuple(c + (1.0 - c) * (1.0 - factor) for c in rgb)


def _darken_color(rgb, factor=0.65):
    """Darken an RGB colour. factor=1 → original, factor=0 → black."""
    return tuple(c * factor for c in rgb)


def _plot_umap_patient_seizure(embedding, labels, patient_ids, results_dir, prefix,
                               title_suffix='', pid_map=None):
    """UMAP combinant patient et statut crise : une couleur par patient,
    teinte claire + cercle pour les non-crises, teinte foncée + croix pour les crises."""
    os.makedirs(results_dir, exist_ok=True)

    unique_pids = np.unique(patient_ids)
    n_patients = len(unique_pids)
    cmap = plt.cm.get_cmap('tab10' if n_patients <= 10 else 'tab20')

    fig, ax = plt.subplots(figsize=(13, 9))

    for i, pid in enumerate(unique_pids):
        pid_mask = patient_ids == pid
        base_rgb = cmap(i % cmap.N)[:3]
        light_rgb = _lighten_color(base_rgb, factor=0.55)
        dark_rgb = _darken_color(base_rgb, factor=0.65)
        pid_label = pid_map.get(int(pid), str(pid)) if pid_map else str(pid)

        # Non-crise : couleur claire, marqueur cercle, en arrière-plan
        mask_ns = pid_mask & (labels == 0)
        if mask_ns.any():
            ax.scatter(embedding[mask_ns, 0], embedding[mask_ns, 1],
                       c=[light_rgb], s=15, alpha=0.55, marker='o',
                       label=f'{pid_label} — non-crise')

        # Crise : couleur foncée, marqueur croix, au premier plan
        mask_s = pid_mask & (labels == 1)
        if mask_s.any():
            ax.scatter(embedding[mask_s, 0], embedding[mask_s, 1],
                       c=[dark_rgb], s=70, alpha=0.9, marker='x',
                       linewidths=1.8, zorder=10,
                       label=f'{pid_label} — crise')

    ax.set_title(f'UMAP — crises et non-crises par patient{title_suffix}', fontsize=14)
    ax.set_xlabel('UMAP 1')
    ax.set_ylabel('UMAP 2')

    ncol = max(1, n_patients // 8)
    legend_fontsize = 7 if n_patients > 5 else 9
    ax.legend(markerscale=2, fontsize=legend_fontsize,
              bbox_to_anchor=(1.02, 1), loc='upper left',
              borderaxespad=0., ncol=ncol)

    plt.savefig(os.path.join(results_dir, f'{prefix}_patient_seizure.png'),
                dpi=300, bbox_inches='tight')
    plt.close()


def _plot_umap(embedding, labels, patient_ids, results_dir, prefix, title_suffix='', pid_map=None,
                plot_patient_ids=True):
    """Plot UMAP embedding colored by patient_ids and by label."""
    os.makedirs(results_dir, exist_ok=True)

    # Patient ID plot
    if plot_patient_ids:
        fig, ax = plt.subplots(figsize=(10, 8))
        unique_pids = np.unique(patient_ids)
        cmap = plt.cm.get_cmap('tab10' if len(unique_pids) <= 10 else 'tab20')
        for i, pid in enumerate(unique_pids):
            mask = patient_ids == pid
            ax.scatter(embedding[mask, 0], embedding[mask, 1],
                       s=5, alpha=0.4, c=[cmap(i % cmap.N)], label=str(pid))
        ax.legend(markerscale=3, fontsize=9)
        ax.set_title(f'UMAP — by patient{title_suffix}', fontsize=14)
        ax.set_xlabel('UMAP 1')
        ax.set_ylabel('UMAP 2')
        plt.savefig(os.path.join(results_dir, f'{prefix}_patient_ids.png'), dpi=300, bbox_inches='tight')
        plt.close()

    # Label plot: draw seizure points larger and on top
    str_labels = np.array(['seizure' if l == 1 else 'no seizure' for l in labels])
    fig, ax = plt.subplots(figsize=(10, 8))
    mask_ns = str_labels == 'no seizure'
    ax.scatter(embedding[mask_ns, 0], embedding[mask_ns, 1],
               s=5, alpha=0.3, c='#1f77b4', label='no seizure')
    mask_s = str_labels == 'seizure'
    ax.scatter(embedding[mask_s, 0], embedding[mask_s, 1],
               s=40, alpha=0.9, c='#d62728', edgecolors='black',
               linewidths=0.5, label='seizure', zorder=10)
    ax.legend(markerscale=1.5, fontsize=11)
    ax.set_title(f'UMAP — by label{title_suffix}', fontsize=14)
    ax.set_xlabel('UMAP 1')
    ax.set_ylabel('UMAP 2')
    plt.savefig(os.path.join(results_dir, f'{prefix}_labels.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # Combined patient + seizure status plot
    _plot_umap_patient_seizure(embedding, labels, patient_ids, results_dir, prefix, title_suffix,
                               pid_map=pid_map)


def visualize_umap(encoded_data, labels, patient_ids, results_dir, prefix='umap',
                   test_data=None, test_labels=None, test_patient_ids=None, pid_map=None,
                   plot_patient_ids=True):
    """Generate UMAP visualizations.

    When test_data is provided, UMAP is fit on encoded_data (train) and both
    train and test are transformed into the same space for comparable plots.
    """
    os.makedirs(results_dir, exist_ok=True)

    print("Computing UMAP embedding...")
    mapper = umap.UMAP(random_state=42, metric='cosine', n_neighbors=15, min_dist=0.1)
    train_embedding = mapper.fit_transform(encoded_data)
    print("UMAP computation completed!")

    print("Creating UMAP plots...")
    _plot_umap(train_embedding, labels, patient_ids, results_dir, prefix,
               title_suffix=' (train)', pid_map=pid_map, plot_patient_ids=plot_patient_ids)

    if test_data is not None:
        print("Transforming test data into same UMAP space...")
        test_embedding = mapper.transform(test_data)
        _plot_umap(test_embedding, test_labels, test_patient_ids, results_dir,
                   f'{prefix}_test', title_suffix=' (test)', pid_map=pid_map,
                   plot_patient_ids=plot_patient_ids)

    print("UMAP plots saved!")

def visualize_embeddings(encoded_data, labels, results_dir):
    # Apply PCA to reduce dimensionality of data from embedding_dim -> 3d to make it easier to visualize!
    pca = PCA(n_components=3)
    encoded_data_3d = pca.fit_transform(encoded_data)

    scatter = go.Scatter3d(
        x=encoded_data_3d[:, 0],
        y=encoded_data_3d[:, 1],
        z=encoded_data_3d[:, 2],
        mode='markers',
        marker=dict(size=4, color=labels, colorscale='Viridis', opacity=0.8),
        text=labels, 
        hoverinfo='text',
    )

    # Create layout
    layout = go.Layout(
        title="Encoded and PCA Reduced 3D Scatter Plot",
        scene=dict(
            xaxis=dict(title="PC1"),
            yaxis=dict(title="PC2"),
            zaxis=dict(title="PC3"),
        ),
        width=1000, 
        height=750,
    )

    # Create figure and add scatter plot
    fig = go.Figure(data=[scatter], layout=layout)

    # Save the plot as HTML file instead of showing it
    fig.write_html(os.path.join(results_dir, 'pca_visualization.html'))
    print("PCA visualization saved as 'pca_visualization.html'")
    print("Open this file in your browser to view the interactive plot")


def plot_evaluation_results(metrics, predictions_probs, true_labels, save_path='./evaluation_results.png'):
    """
    Create a combined visualization showing ROC curve and confusion matrix.

    Args:
        metrics: Dictionary containing evaluation metrics including 'roc_auc' and 'confusion_matrix'
        predictions_probs: Array of predicted probabilities for the positive class
        true_labels: Array of true labels
        save_path: Path to save the figure
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Plot 1: ROC Curve
    if metrics['roc_auc'] is not None:
        fpr, tpr, thresholds = roc_curve(true_labels, predictions_probs)

        axes[0].plot(fpr, tpr, linewidth=2, label=f'ROC curve (AUC = {metrics["roc_auc"]:.3f})')
        axes[0].plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random classifier')
        axes[0].set_xlim([0.0, 1.0])
        axes[0].set_ylim([0.0, 1.05])
        axes[0].set_xlabel('False Positive Rate', fontsize=12)
        axes[0].set_ylabel('True Positive Rate', fontsize=12)
        axes[0].set_title('ROC Curve', fontsize=14, fontweight='bold')
        axes[0].legend(loc='lower right', fontsize=10)
        axes[0].grid(True, alpha=0.3)
    else:
        axes[0].text(0.5, 0.5, 'ROC curve not available\n(single class present)',
                     ha='center', va='center', fontsize=12)
        axes[0].set_title('ROC Curve', fontsize=14, fontweight='bold')

    # Plot 2: Confusion Matrix
    cm = metrics['confusion_matrix']
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                cbar_kws={'label': 'Count'}, ax=axes[1],
                annot_kws={'fontsize': 14})
    axes[1].set_xlabel('Predicted Label', fontsize=12)
    axes[1].set_ylabel('True Label', fontsize=12)
    axes[1].set_title('Confusion Matrix', fontsize=14, fontweight='bold')
    axes[1].set_xticklabels(['No Seizure', 'Seizure'])
    axes[1].set_yticklabels(['No Seizure', 'Seizure'], rotation=0)

    # Add metrics text
    metrics_text = (
        f"Accuracy: {metrics['accuracy']:.3f}\n"
        f"Precision: {metrics['precision']:.3f}\n"
        f"Recall: {metrics['recall']:.3f}\n"
        f"F1 Score: {metrics['f1']:.3f}"
    )
    fig.text(0.5, 0.02, metrics_text, ha='center', fontsize=11,
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

    plt.tight_layout(rect=[0, 0.08, 1, 1])

    # Ensure directory exists
    os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)

    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Evaluation results saved to {save_path}")
    return save_path


def plot_training_history(train_history, save_path='./training_history.png'):
    """
    Plot training loss and accuracy over epochs.

    Args:
        train_history: Dictionary with 'loss' and 'accuracy' lists
        save_path: Path to save the figure
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    epochs = range(1, len(train_history['loss']) + 1)

    # Plot loss
    axes[0].plot(epochs, train_history['loss'], 'b-', linewidth=2, marker='o')
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Loss', fontsize=12)
    axes[0].set_title('Training Loss', fontsize=14, fontweight='bold')
    axes[0].grid(True, alpha=0.3)

    # Plot accuracy
    axes[1].plot(epochs, train_history['accuracy'], 'g-', linewidth=2, marker='o')
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('Accuracy', fontsize=12)
    axes[1].set_title('Training Accuracy', fontsize=14, fontweight='bold')
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()

    # Ensure directory exists
    os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)

    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Training history saved to {save_path}")
    return save_path


def plot_distance_distributions(pos_dists, neg_dists, save_path='./distance_distributions.png'):
    """
    Plot distribution of positive and negative distances.

    Args:
        pos_dists: Array of positive pair distances
        neg_dists: Array of negative pair distances
        save_path: Path to save the figure
    """
    plt.figure(figsize=(8, 5))
    plt.hist(pos_dists, alpha=0.6, label='Positive Distances')
    plt.hist(neg_dists, alpha=0.6, label='Negative Distances')
    plt.xlabel('Distance')
    plt.ylabel('Count')
    plt.legend()
    plt.title('Distribution of Positive and Negative Distances')

    # Ensure directory exists
    os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)

    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Distance distributions saved to {save_path}")
    return save_path


def print_evaluation_metrics(metrics, dataset_name="Test"):
    """Pretty print evaluation metrics."""
    logging.info(f"\n{'='*50}")
    logging.info(f"{dataset_name} Set Evaluation Results")
    logging.info(f"{'='*50}")
    logging.info(f"Accuracy:  {metrics['accuracy']:.4f}")
    logging.info(f"Precision: {metrics['precision']:.4f}")
    logging.info(f"Recall:    {metrics['recall']:.4f}")
    logging.info(f"F1 Score:  {metrics['f1']:.4f}")
    if metrics['roc_auc'] is not None:
        logging.info(f"ROC-AUC:   {metrics['roc_auc']:.4f}")
    logging.info(f"\nConfusion Matrix:")
    logging.info(f"{metrics['confusion_matrix']}")
    logging.info(f"{'='*50}\n")


# if __name__ == "__main__":
#     from app import load_train_test_dataset

#     print("Loading dataset...")
#     train_dataset, test_dataset, weights = load_train_test_dataset(csv_path="/Users/laura/Documents/aura/tuh_ecg_features2.csv")
    
#     # Reduce sample size for faster computation
#     sampled_indices = random.sample(list(range(len(train_dataset))), 2000)
#     print(f"Using {len(sampled_indices)} samples for visualization")

#     # Create subset
#     train_subset = Subset(train_dataset, sampled_indices)

#     train_dataloader = DataLoader(train_dataset, batch_size=128, shuffle=True)

#     print("Loading model...")
#     model = DeepResidualEmbeddingModel(input_dim=14, embedding_dim=1024)
#     model.load_state_dict(torch.load('checkpoints/model_9.pth'))
#     model.eval()  # Set to evaluation mode

#     device = 'cpu'
#     if torch.cuda.is_available():
#         device = torch.device('cuda:0')

#     print("Generating embeddings...")
#     encoded_data = []
#     labels = []
#     patient_ids = []
#     with torch.no_grad():
#         for features, label, patient_id in tqdm(train_dataloader, desc="Processing batches"):
#             features, label, patient_id = features.to(device), label.to(device), patient_id.to(device)
#             embeddings = model(features)
#             encoded_data.extend(embeddings.cpu().numpy())
#             labels.extend(label.cpu().numpy())
#             patient_ids.extend(patient_id.cpu().numpy())
    
#     # Convert lists to numpy arrays
#     encoded_data = np.array(encoded_data)
#     labels = np.array(labels)
#     patient_ids = np.array(patient_ids)
    
#     print(f"Generated embeddings shape: {encoded_data.shape}")

#     print("Creating visualizations...")
#     results_dir = f'results_{time.time()}'
#     os.makedirs(results_dir, exist_ok=True)
#     visualize_embeddings(encoded_data,  patient_ids, results_dir)
#     visualize_umap(encoded_data, labels, patient_ids, results_dir)
#     print("All visualizations completed!")