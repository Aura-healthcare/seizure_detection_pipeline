import torch
import torch.nn as nn
import torch.nn.functional as F

# Model which transforms a tensor of features into a tensor of embeddings in 256 dimensions
class EmbeddingModel(nn.Module):
    def __init__(self, input_dim, embedding_dim):
        super(EmbeddingModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.bn1 = nn.BatchNorm1d(128)
        self.fc2 = nn.Linear(128, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.fc3 = nn.Linear(256, embedding_dim)
        self.bn3 = nn.BatchNorm1d(embedding_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
        
        # Residual connections
        self.residual1 = nn.Linear(input_dim, 128) if input_dim != 128 else nn.Identity()
        self.residual2 = nn.Linear(128, 256) if 128 != 256 else nn.Identity()
        self.residual3 = nn.Linear(256, embedding_dim) if 256 != embedding_dim else nn.Identity()

    def forward(self, x):
        # First layer with residual
        identity1 = self.residual1(x)
        out1 = self.relu(self.bn1(self.fc1(x)))
        out1 = out1 + identity1  # Residual connection
        out1 = self.dropout(out1)
        
        # Second layer with residual
        identity2 = self.residual2(out1)
        out2 = self.relu(self.bn2(self.fc2(out1)))
        out2 = out2 + identity2  # Residual connection
        out2 = self.dropout(out2)
        
        # Final layer with residual
        identity3 = self.residual3(out2)
        out3 = self.bn3(self.fc3(out2))
        out3 = out3 + identity3  # Residual connection
        
        return out3

# Alternative: Deep Residual Network
class DeepResidualEmbeddingModel(nn.Module):
    def __init__(self, input_dim, embedding_dim, num_blocks=3, dropout=0.2):
        super(DeepResidualEmbeddingModel, self).__init__()

        self.input_layer = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Residual blocks at 256
        self.residual_blocks_256 = nn.ModuleList()
        for i in range(num_blocks):
            self.residual_blocks_256.append(ResidualBlock(256, 256, dropout=dropout))
        
        # Expansion to 512 (if needed)
        if embedding_dim >= 512:
            self.expand_to_512 = nn.Sequential(
                nn.Linear(256, 512),
                nn.BatchNorm1d(512),
                nn.ReLU(),
                nn.Dropout(dropout)
            )

            # Residual blocks at 512
            self.residual_blocks_512 = nn.ModuleList()
            for i in range(num_blocks):
                self.residual_blocks_512.append(ResidualBlock(512, 512, dropout=dropout))
        else:
            self.expand_to_512 = nn.Identity()
            self.residual_blocks_512 = nn.ModuleList()

        # Expansion to 1024 (if needed)
        if embedding_dim >= 1024:
            self.expand_to_1024 = nn.Sequential(
                nn.Linear(512, 1024),
                nn.BatchNorm1d(1024),
                nn.ReLU(),
                nn.Dropout(dropout)
            )
            
            # Residual blocks at 1024
            self.residual_blocks_1024 = nn.ModuleList()
            for i in range(num_blocks):
                self.residual_blocks_1024.append(ResidualBlock(1024, 1024, dropout=dropout))
        else:
            self.expand_to_1024 = nn.Identity()
            self.residual_blocks_1024 = nn.ModuleList()
        
        # Output layer
        if embedding_dim >= 1024:
            self.output_layer = nn.Linear(1024, embedding_dim)
        elif embedding_dim >= 512:
            self.output_layer = nn.Linear(512, embedding_dim)
        else:
            self.output_layer = nn.Linear(256, embedding_dim)

        self.projection_head = nn.Sequential(
            nn.Linear(embedding_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
        
    def forward(self, x, return_projection=False):
        x = self.input_layer(x)
        
        # Process through 256 blocks
        for block in self.residual_blocks_256:
            x = block(x)
        
        # Expand to 512 if needed
        if not isinstance(self.expand_to_512, nn.Identity):
            x = self.expand_to_512(x)
            
            # Process through 512 blocks
            for block in self.residual_blocks_512:
                x = block(x)
        
        # Expand to 1024 if needed
        if not isinstance(self.expand_to_1024, nn.Identity):
            x = self.expand_to_1024(x)

            # Process through 1024 blocks
            for block in self.residual_blocks_1024:
                x = block(x)

        embedding = self.output_layer(x)  # Base embedding
        embedding = F.normalize(embedding, p=2, dim=1)

        if return_projection:
            projection = self.projection_head(embedding)
            projection = F.normalize(projection, dim=1)  # L2-normalize for contrastive loss
            return embedding, projection

        return embedding

class ResidualBlock(nn.Module):
    def __init__(self, in_dim, out_dim, dropout=0.2):
        super(ResidualBlock, self).__init__()

        self.layers = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.BatchNorm1d(out_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(out_dim, out_dim),
            nn.BatchNorm1d(out_dim)
        )

        # Shortcut connection (identity if same dimensions)
        self.shortcut = nn.Linear(in_dim, out_dim) if in_dim != out_dim else nn.Identity()
        self.relu = nn.ReLU()

    def forward(self, x):
        residual = self.shortcut(x)
        out = self.layers(x)
        return self.relu(out + residual)


class SeizureClassifier(nn.Module):
    """Classification head that uses embeddings from embedding model."""
    def __init__(self, embedding_model, embedding_dim, num_classes=2, hidden_dims=None):
        super(SeizureClassifier, self).__init__()
        self.embedding_model = embedding_model

        # Freeze the embedding model by default
        for param in self.embedding_model.parameters():
            param.requires_grad = False

        # Classification head
        if hidden_dims is None:
            hidden_dims = [64, 32]
        layers = []
        in_dim = embedding_dim
        for h in hidden_dims:
            layers += [nn.Linear(in_dim, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(0.3)]
            in_dim = h
        layers.append(nn.Linear(in_dim, num_classes))
        self.classifier = nn.Sequential(*layers)

    def forward(self, x):
        embeddings = self.embedding_model(x)
        return self.classifier(embeddings)

    def unfreeze_embedding_model(self, last_n_blocks=None):
        """Unfreeze embedding model for fine-tuning.

        Args:
            last_n_blocks: If None, unfreeze all layers.
                          If int, only unfreeze the last N residual blocks.
        """
        if last_n_blocks is None:
            # Unfreeze everything
            for param in self.embedding_model.parameters():
                param.requires_grad = True
        else:
            # Partial unfreezing: only unfreeze last N blocks
            # Keep input layers and early blocks frozen

            # Unfreeze output layer
            for param in self.embedding_model.output_layer.parameters():
                param.requires_grad = True

            # Unfreeze last N blocks from each stage (1024, 512, 256)
            if hasattr(self.embedding_model, 'residual_blocks_1024'):
                for block in self.embedding_model.residual_blocks_1024[-last_n_blocks:]:
                    for param in block.parameters():
                        param.requires_grad = True

            if last_n_blocks > 1 and hasattr(self.embedding_model, 'residual_blocks_512'):
                for block in self.embedding_model.residual_blocks_512[-last_n_blocks:]:
                    for param in block.parameters():
                        param.requires_grad = True

            if last_n_blocks > 2 and hasattr(self.embedding_model, 'residual_blocks_256'):
                for block in self.embedding_model.residual_blocks_256[-last_n_blocks:]:
                    for param in block.parameters():
                        param.requires_grad = True

