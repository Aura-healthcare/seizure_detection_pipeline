import torch
import torch.nn as nn
import logging
from tqdm import tqdm
import numpy as np
import random
from sklearn.metrics import accuracy_score

def train_model(model, train_dataloader, optimizer, loss_fn, device, epochs, scheduler, checkpoint_dir):
    model.train()
    for epoch in range(epochs):
        epoch_loss = 0
        loss_list = []
        for batch in tqdm(train_dataloader):
            anchor, contrastive, distance, label = batch
            anchor, contrastive, distance, label = anchor.to(device), contrastive.to(device), distance.to(device), label.to(device)
            optimizer.zero_grad()
            
            # Pass features through model to get embeddings
            anchor_embeddings = model(anchor)
            contrastive_embeddings = model(contrastive)
            
            # Pass embeddings to loss function
            loss = loss_fn(anchor_embeddings, contrastive_embeddings, distance)
            epoch_loss += loss.item()
            loss.backward()
            optimizer.step()
            loss_list.append(epoch_loss / len(train_dataloader))
        logging.info(f"Epoch {epoch+1}/{epochs} loss: {epoch_loss/len(train_dataloader)}")
        if scheduler is not None:
            scheduler.step()
        torch.save(model.state_dict(), f'{checkpoint_dir}/model_{epoch}.pth')

    return model, loss_list

def train_model_triplet(model, train_dataloader, optimizer, loss_fn, device, epochs, scheduler, checkpoint_dir):
    model.train()
    for epoch in range(epochs):
        epoch_loss = 0.0
        loss_list = []
        for batch in tqdm(train_dataloader):
            anchor, positive, negative, anchor_label, _ = batch
            anchor, positive, negative, anchor_label = anchor.to(device), positive.to(device), negative.to(device), anchor_label.to(device)
            optimizer.zero_grad()
            
            # Pass features through model to get embeddings
            anchor_embedding = model(anchor, return_projection=False)
            positive_embedding = model(positive, return_projection=False)
            negative_embedding = model(negative, return_projection=False)

            # pos_dist = torch.norm(anchor_embedding - positive_embedding, dim=1)
            # neg_dist = torch.norm(anchor_embedding - negative_embedding, dim=1)
            # logging.info(f"Pos dist avg: {pos_dist.mean().item():.4f}, Neg dist avg: {neg_dist.mean().item():.4f}")
            
            # Pass embeddings to loss function
            loss = loss_fn(anchor_embedding, positive_embedding, negative_embedding)
            epoch_loss += loss.item()
            loss.backward()
            optimizer.step()
            loss_list.append(epoch_loss / len(train_dataloader))
        logging.info(f"Epoch {epoch+1}/{epochs} loss: {epoch_loss/len(train_dataloader)}")
        scheduler.step()

        torch.save(model.state_dict(), f'{checkpoint_dir}/model_{epoch}.pth')

    return model, loss_list

def train_model_triplet_hard_negative(model, train_dataloader, optimizer, loss_fn, device, epochs, scheduler, checkpoint_dir):
    model.train()
    for epoch in range(epochs):
        epoch_loss = 0.0
        num_batches = 0
        loss_list = []
        for batch in tqdm(train_dataloader):
            features, labels, patient_ids = batch
            features, labels, patient_ids = features.to(device), labels.to(device), patient_ids.to(device)
            optimizer.zero_grad()

            # Neutraliser tout Inf/NaN résiduel avant le forward
            features = torch.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
            embeddings = model(features)

            # Pass seizure labels (not patient_ids) so the loss separates
            # seizure vs non-seizure, not patient identity.
            loss = loss_fn(embeddings, labels)
            epoch_loss += loss.item()
            loss.backward()
            optimizer.step()
            num_batches += 1

        avg_epoch_loss = epoch_loss / num_batches
        loss_list.append(avg_epoch_loss)

        logging.info(f"Epoch {epoch+1}/{epochs} loss: {avg_epoch_loss:.4f}")
        scheduler.step()

        torch.save(model.state_dict(), f'{checkpoint_dir}/model_{epoch}.pth')

    return model, loss_list


def train_model_supcon(model, train_dataloader, optimizer, loss_fn, device, epochs, scheduler, checkpoint_dir):
    """Training loop for SupervisedContrastiveLoss.
    Passes seizure labels and patient_ids so that only cross-patient
    same-label pairs are treated as positives.
    """
    model.train()
    for epoch in range(epochs):
        epoch_loss = 0.0
        num_batches = 0
        for batch in tqdm(train_dataloader):
            features, labels, patient_ids = batch
            features = features.to(device)
            labels = labels.to(device)
            patient_ids = patient_ids.to(device)
            optimizer.zero_grad()
            loss = loss_fn(model(features), labels, patient_ids=patient_ids)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            num_batches += 1

        avg_loss = epoch_loss / num_batches
        logging.info(f"Epoch {epoch+1}/{epochs} loss: {avg_loss:.4f}")
        if scheduler is not None:
            scheduler.step()
        torch.save(model.state_dict(), f'{checkpoint_dir}/model_{epoch}.pth')

    return model, []


def compute_distances(model, dataloader, device='cpu', k=10):
    model.eval()
    pos_dists = []
    neg_dists = []
    
    with torch.no_grad():
        for batch in dataloader:
            features, labels, _ = batch
            features, labels = features.to(device), labels.to(device)
            
            embeddings = model(features)
            # Plot the top k closest and farthest embeddings to each other
            pos_dist = torch.norm(embeddings, p=2, dim=1).cpu().numpy()
            neg_dist = torch.norm(embeddings, p=2, dim=1).cpu().numpy()
            sampled_pos_dist = random.sample(list(pos_dist), k)
            sampled_neg_dist = random.sample(list(neg_dist), k)
        
        pos_dists.extend(sampled_pos_dist)
        neg_dists.extend(sampled_neg_dist)

    return np.array(pos_dists), np.array(neg_dists)


def train_classifier(classifier, train_dataloader, optimizer, criterion, device, epochs, scheduler=None):
    """Train the seizure classifier."""
    classifier.train()
    train_history = {'loss': [], 'accuracy': []}

    for epoch in range(epochs):
        epoch_loss = 0.0
        all_preds = []
        all_labels = []

        for batch in tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{epochs}"):
            features, labels, _ = batch
            features, labels = features.to(device), labels.to(device)

            optimizer.zero_grad()

            # Forward pass
            logits = classifier(features)
            loss = criterion(logits, labels)

            # Backward pass
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

            # Collect predictions
            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

        # Compute metrics
        avg_loss = epoch_loss / len(train_dataloader)
        accuracy = accuracy_score(all_labels, all_preds)

        train_history['loss'].append(avg_loss)
        train_history['accuracy'].append(accuracy)

        logging.info(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}")

        if scheduler:
            scheduler.step()

    return classifier, train_history



