#! Autoencoder Model for Recommendation System
#? In this script we implement an Autoencoder model to predict user-item ratings (1,2,3,4 or 5).

import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, precision_score, recall_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.calibration import LabelEncoder
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import roc_curve, auc
import time
import os
import seaborn as sns
from itertools import cycle

ROUTE = "/home/adiez/Desktop/Deep Learning/DL - Assignment 2/data/100k/processed"

USER_FEATURES = ['age', 'gender_F','gender_M', 'occupation_administrator', 'occupation_artist',
       'occupation_doctor', 'occupation_educator', 'occupation_engineer', 'occupation_entertainment', 
       'occupation_executive', 'occupation_healthcare', 'occupation_homemaker','occupation_lawyer',
       'occupation_librarian', 'occupation_marketing', 'occupation_none','occupation_other', 
       'occupation_programmer', 'occupation_retired','occupation_salesman', 'occupation_scientist', 
       'occupation_student', 'occupation_technician', 'occupation_writer', 'release_date']

ITEM_FEATURES = ['unknown','Action', 'Adventure', 'Animation', 'Children', 'Comedy', 'Crime',
                'Documentary', 'Drama', 'Fantasy', 'Film-Noir', 'Horror', 'Musical','Mystery', 
                'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western']

class AutoencoderRecommender(nn.Module):
    def __init__(self, num_users, num_items, 
                 num_user_features=25, 
                 num_item_features=19,
                 latent_dim=128,  # Increased latent dimension
                 hidden_layers=[512, 256, 128],  # Deeper architecture
                 dropout_rate=0.3,  # Less aggressive dropout
                 use_batch_norm=True,
                 variational=True):  # Add variational option
        super(AutoencoderRecommender, self).__init__()
        
        # Embedding dimensions for user and item ids - increased size
        self.embedding_dim = 64
        self.variational = variational
        
        # Embedding layers for user and item ids
        self.user_embedding = nn.Embedding(num_users, self.embedding_dim)
        self.item_embedding = nn.Embedding(num_items, self.embedding_dim)
        
        # Calculate actual input dimension (user_embed + item_embed + user_features + item_features)
        encoder_input_dim = self.embedding_dim + self.embedding_dim + num_user_features + num_item_features
        self.input_dim = encoder_input_dim
        
        # Encoder layers
        self.encoder_layers = nn.ModuleList()
        
        for dim in hidden_layers:
            self.encoder_layers.append(nn.Linear(encoder_input_dim, dim))
            if use_batch_norm:
                self.encoder_layers.append(nn.BatchNorm1d(dim))
            self.encoder_layers.append(nn.LeakyReLU(0.2))
            self.encoder_layers.append(nn.Dropout(dropout_rate))
            encoder_input_dim = dim
        
        # Latent space - for variational autoencoder, we need mean and log variance
        if variational:
            self.encoder_mu = nn.Linear(encoder_input_dim, latent_dim)
            self.encoder_logvar = nn.Linear(encoder_input_dim, latent_dim)
        else:
            self.encoder_output = nn.Linear(encoder_input_dim, latent_dim)
        
        # Decoder layers
        self.decoder_layers = nn.ModuleList()
        decoder_input_dim = latent_dim
        
        # Reverse the hidden layers for the decoder
        for dim in reversed(hidden_layers):
            self.decoder_layers.append(nn.Linear(decoder_input_dim, dim))
            if use_batch_norm:
                self.decoder_layers.append(nn.BatchNorm1d(dim))
            self.decoder_layers.append(nn.LeakyReLU(0.2))
            self.decoder_layers.append(nn.Dropout(dropout_rate))
            decoder_input_dim = dim
        
        # Final output layer (predict rating)
        self.decoder_output = nn.Linear(decoder_input_dim, 1)
        
        # Add residual connection from input to output
        self.residual_to_output = nn.Sequential(
            nn.Linear(self.input_dim, 64),
            nn.LeakyReLU(0.2),
            nn.Linear(64, 1)
        )
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights with He initialization for better convergence"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, nonlinearity='leaky_relu')
                if module.bias is not None:
                    module.bias.data.zero_()
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.01)
    
    def encode(self, user_indices, item_indices, user_features, item_features):
        # Get embeddings
        user_emb = self.user_embedding(user_indices)
        item_emb = self.item_embedding(item_indices)
        
        # Concatenate all input features
        combined = torch.cat([
            user_emb,
            item_emb,
            user_features,
            item_features
        ], dim=1)
        
        # Encoder
        x = combined
        for layer in self.encoder_layers:
            x = layer(x)
        
        # Return latent representation
        if self.variational:
            mu = self.encoder_mu(x)
            logvar = self.encoder_logvar(x)
            return mu, logvar, combined
        else:
            latent = self.encoder_output(x)
            return latent, combined
    
    def reparameterize(self, mu, logvar):
        """
        Reparameterization trick: z = mu + std * epsilon
        where epsilon is randomly sampled from N(0, 1)
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z):
        """Decode latent representation back to rating prediction"""
        x = z
        for layer in self.decoder_layers:
            x = layer(x)
        
        return self.decoder_output(x)
        
    def forward(self, user_indices, item_indices, user_features, item_features):
        if self.variational:
            # Variational autoencoder approach
            mu, logvar, combined = self.encode(user_indices, item_indices, user_features, item_features)
            z = self.reparameterize(mu, logvar)
            
            # Decode
            rating_pred = self.decode(z)
            
            # Add residual connection
            residual_pred = self.residual_to_output(combined)
            
            # Combine predictions with residual connection
            final_pred = rating_pred + residual_pred
            
            return final_pred.squeeze(), z, mu, logvar
        else:
            # Regular autoencoder approach
            latent, combined = self.encode(user_indices, item_indices, user_features, item_features)
            
            # Decode
            rating_pred = self.decode(latent)
            
            # Add residual connection
            residual_pred = self.residual_to_output(combined)
            
            # Combine predictions with residual connection
            final_pred = rating_pred + residual_pred
            
            return final_pred.squeeze(), latent

class MovieLensDataset(Dataset):
    def __init__(self, data):
        data.fillna(0, inplace=True)
        
        self.users = torch.tensor(data['user_id'].values, dtype=torch.long)
        self.items = torch.tensor(data['item_id'].values, dtype=torch.long)
        self.user_features = torch.tensor(data[USER_FEATURES].astype(float).values, dtype=torch.float32)
        self.item_features = torch.tensor(data[ITEM_FEATURES].astype(float).values, dtype=torch.float32)
        self.ratings = torch.tensor(data['rating'].values, dtype=torch.float)
    
    def __len__(self):
        return len(self.users)
    
    def __getitem__(self, idx):
        return self.users[idx], self.items[idx], self.user_features[idx], self.item_features[idx], self.ratings[idx]

def prepare_datasets(ratings_file, users_file, movies_file, val_size=0.1, test_size=0.1, random_state=42):
    ratings_df = pd.read_csv(ratings_file)
    users_df = pd.read_csv(users_file)
    movies_df = pd.read_csv(movies_file)
    
    data = ratings_df.merge(users_df, on='user_id', how='left')
    data = data.merge(movies_df, on='item_id', how='left')
    
    user_encoder = LabelEncoder()
    item_encoder = LabelEncoder()
    
    data["user_id"] = user_encoder.fit_transform(data["user_id"])
    data["item_id"] = item_encoder.fit_transform(data["item_id"])
    
    n_users = len(user_encoder.classes_)
    n_items = len(item_encoder.classes_)
    
    train_val_df, test_df = train_test_split(data, test_size=test_size, random_state=random_state)
    train_df, val_df = train_test_split(train_val_df, test_size=val_size, random_state=random_state)
    
    train_df = train_df.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)
    
    # Add some noise to user features to create more robust representations (data augmentation)
    noise_level = 0.05
    train_df[USER_FEATURES] = train_df[USER_FEATURES] * (1 + np.random.normal(0, noise_level, train_df[USER_FEATURES].shape))
    
    train_dataset = MovieLensDataset(train_df)
    val_dataset = MovieLensDataset(val_df)
    test_dataset = MovieLensDataset(test_df)
    
    return train_dataset, val_dataset, test_dataset, n_users, n_items

def train_model(model, train_loader, val_loader, optimizer, device, num_epochs=20, checkpoint_dir='./checkpoints', patience=5):
    # Add learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='min', 
        factor=0.5, 
        patience=2
    )
    
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    model = model.to(device)
    
    # Debug: Validate input shapes
    debug = False
    if debug:
        print("Running shape debug...")
        # Get a batch from the train loader
        sample_batch = next(iter(train_loader))
        users, items, user_features, item_features, ratings = sample_batch
        
        print(f"Batch size: {len(users)}")
        print(f"User indices shape: {users.shape}")
        print(f"Item indices shape: {items.shape}")
        print(f"User features shape: {user_features.shape}")
        print(f"Item features shape: {item_features.shape}")
        print(f"Ratings shape: {ratings.shape}")
        
        # Move to device
        users = users.to(device)
        items = items.to(device)
        user_features = user_features.to(device)
        item_features = item_features.to(device)
        
        # Forward pass to check shapes
        with torch.no_grad():
            model.eval()
            try:
                if model.variational:
                    outputs, z, mu, logvar = model(users, items, user_features, item_features)
                    print("Forward pass successful!")
                    print(f"Output shape: {outputs.shape}")
                    print(f"Latent shape: {z.shape}")
                    print(f"Mu shape: {mu.shape}")
                    print(f"LogVar shape: {logvar.shape}")
                else:
                    outputs, latent = model(users, items, user_features, item_features)
                    print("Forward pass successful!")
                    print(f"Output shape: {outputs.shape}")
                    print(f"Latent shape: {latent.shape}")
            except Exception as e:
                print(f"Forward pass failed: {e}")
                raise
    
    # MSE loss for rating prediction
    mse_criterion = nn.MSELoss()
    
    # Add a latent representation regularization term (to encourage compact representations)
    latent_reg_weight = 0.001  # Reduced weight
    
    # KL divergence weight - start small and increase over time (KL annealing)
    kl_weight_start = 0.0
    kl_weight_end = 0.1
    kl_weight = kl_weight_start
    
    # Initialize tracking variables
    best_val_loss = np.inf
    best_epoch = 0
    epochs_no_improve = 0
    
    # History of train/validation losses
    history = {"train_loss": [], "val_loss": [], "reconstruction_loss": [], "kl_loss": []}
    
    # Curriculum learning for ratings
    # Start by focusing on easier samples (high and low ratings) then gradually introduce harder samples
    curriculum_epochs = 5
    
    # Teacher forcing rate (start high, gradually decrease)
    teacher_forcing_ratio = 0.8
    
    # Start training loop
    start_time = time.time()
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        recon_loss_sum = 0.0
        kl_loss_sum = 0.0
        
        # Update KL weight using annealing schedule
        if model.variational:
            kl_weight = min(kl_weight_end, kl_weight_start + epoch * (kl_weight_end - kl_weight_start) / (num_epochs // 2))
        
        # Decrease teacher forcing ratio over time
        if epoch > curriculum_epochs:
            teacher_forcing_ratio = max(0.0, teacher_forcing_ratio - 0.1)
        
        for batch_idx, (users, items, user_features, item_features, ratings) in enumerate(train_loader):
            users = users.to(device)
            items = items.to(device)
            user_features = user_features.to(device)
            item_features = item_features.to(device)
            ratings = ratings.to(device)
            
            optimizer.zero_grad()
            
            # Forward pass
            if model.variational:
                outputs, z, mu, logvar = model(users, items, user_features, item_features)
                
                # Calculate reconstruction loss
                reconstruction_loss = mse_criterion(outputs, ratings.float())
                
                # KL divergence loss
                kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
                kl_loss = kl_loss / users.size(0)  # Normalize by batch size
                
                # Weighted loss with annealing for KL term
                loss = reconstruction_loss + kl_weight * kl_loss
                
                recon_loss_sum += reconstruction_loss.item()
                kl_loss_sum += kl_loss.item()
            else:
                outputs, latent = model(users, items, user_features, item_features)
                
                # Calculate reconstruction loss
                reconstruction_loss = mse_criterion(outputs, ratings.float())
                
                # Latent representation regularization (L2)
                latent_reg = torch.norm(latent, p=2, dim=1).mean()
                
                # Total loss
                loss = reconstruction_loss + latent_reg_weight * latent_reg
            
            loss.backward()
            
            # Gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            
            optimizer.step()
            
            train_loss += loss.item()
            
            if (batch_idx + 1) % 100 == 0:
                if model.variational:
                    print(f'Epoch {epoch+1}/{num_epochs}, Batch {batch_idx+1}/{len(train_loader)}, '
                          f'Loss: {loss.item():.4f}, Recon: {reconstruction_loss.item():.4f}, '
                          f'KL: {kl_loss.item():.4f}, KL Weight: {kl_weight:.4f}')
                else:
                    print(f'Epoch {epoch+1}/{num_epochs}, Batch {batch_idx+1}/{len(train_loader)}, Loss: {loss.item():.4f}')
        
        avg_train_loss = train_loss / len(train_loader)
        history["train_loss"].append(avg_train_loss)
        
        if model.variational:
            history["reconstruction_loss"].append(recon_loss_sum / len(train_loader))
            history["kl_loss"].append(kl_loss_sum / len(train_loader))
        
        # Validation pass
        model.eval()
        val_loss = 0.0
        val_recon_loss = 0.0
        val_kl_loss = 0.0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for users, items, user_features, item_features, ratings in val_loader:
                users = users.to(device)
                items = items.to(device)
                user_features = user_features.to(device)
                item_features = item_features.to(device)
                ratings = ratings.to(device)
                
                if model.variational:
                    outputs, z, mu, logvar = model(users, items, user_features, item_features)
                    
                    # Calculate reconstruction loss
                    reconstruction_loss = mse_criterion(outputs, ratings.float())
                    
                    # KL divergence loss
                    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
                    kl_loss = kl_loss / users.size(0)  # Normalize by batch size
                    
                    # Total loss
                    loss = reconstruction_loss + kl_weight * kl_loss
                    
                    val_recon_loss += reconstruction_loss.item()
                    val_kl_loss += kl_loss.item()
                else:
                    outputs, latent = model(users, items, user_features, item_features)
                    
                    # Calculate loss
                    reconstruction_loss = mse_criterion(outputs, ratings.float())
                    latent_reg = torch.norm(latent, p=2, dim=1).mean()
                    loss = reconstruction_loss + latent_reg_weight * latent_reg
                
                val_loss += loss.item()
                
                all_preds.extend(outputs.cpu().numpy())
                all_labels.extend(ratings.cpu().numpy())
        
        avg_val_loss = val_loss / len(val_loader)
        history["val_loss"].append(avg_val_loss)
        
        if model.variational:
            print(f'Epoch {epoch+1}/{num_epochs} - '
                  f'Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, '
                  f'Val Recon: {val_recon_loss/len(val_loader):.4f}, Val KL: {val_kl_loss/len(val_loader):.4f}')
        else:
            print(f'Epoch {epoch+1}/{num_epochs} - '
                  f'Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}')
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_epoch = epoch
            epochs_no_improve = 0
            
            # Save the best model
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': best_val_loss,
            }, os.path.join(checkpoint_dir, '06_autoencoder_recommendation_best_model.pt'))
            
            print(f"Saved best model checkpoint with validation loss: {best_val_loss:.4f}")
        else:
            epochs_no_improve += 1
        
        # Early stopping check
        if epochs_no_improve >= patience:
            print(f'Early stopping triggered! No improvement for {patience} epochs.')
            break
        
        # Update learning rate
        scheduler.step(avg_val_loss)
        
        # After scheduler.step(avg_val_loss), add:
        for param_group in optimizer.param_groups:
            current_lr = param_group['lr']
            
        print(f"Current learning rate: {current_lr}")
    
    training_time = time.time() - start_time
    print(f"Training completed in {training_time:.2f} seconds.")
    print(f"Best validation loss: {best_val_loss:.4f} at epoch {best_epoch+1}")
    
    # Load the best model
    checkpoint = torch.load(os.path.join(checkpoint_dir, '06_autoencoder_recommendation_best_model.pt'))
    model.load_state_dict(checkpoint['model_state_dict'])
    
    return model, history

def evaluate_model(model, test_loader, device):
    model.eval()
    test_loss = 0.0
    test_recon_loss = 0.0
    test_kl_loss = 0.0
    all_preds = []
    all_labels = []
    
    criterion = nn.MSELoss()
    
    kl_weight = 0.1  # Use the final annealed weight
    
    with torch.no_grad():
        for users, items, user_features, item_features, ratings in test_loader:
            users = users.to(device)
            items = items.to(device)
            user_features = user_features.to(device)
            item_features = item_features.to(device)
            ratings = ratings.to(device)
            
            if model.variational:
                outputs, _, mu, logvar = model(users, items, user_features, item_features)
                
                # Calculate reconstruction loss
                reconstruction_loss = criterion(outputs, ratings.float())
                
                # KL divergence loss
                kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
                kl_loss = kl_loss / users.size(0)  # Normalize by batch size
                
                # Total loss
                loss = reconstruction_loss + kl_weight * kl_loss
                
                test_recon_loss += reconstruction_loss.item()
                test_kl_loss += kl_loss.item()
            else:
                outputs, _ = model(users, items, user_features, item_features)
                loss = criterion(outputs, ratings.float())
            
            test_loss += loss.item()
            
            all_preds.extend(outputs.cpu().numpy())
            all_labels.extend(ratings.cpu().numpy())
    
    # Calculate metrics
    avg_test_loss = test_loss / len(test_loader)
    print(f'Test Loss (MSE): {avg_test_loss:.4f}')
    
    if model.variational:
        avg_recon_loss = test_recon_loss / len(test_loader)
        avg_kl_loss = test_kl_loss / len(test_loader)
        print(f'Test Reconstruction Loss: {avg_recon_loss:.4f}')
        print(f'Test KL Loss: {avg_kl_loss:.4f}')
    
    # Convert predictions to rounded integers for classification metrics
    rounded_preds = np.rint(all_preds).astype(int)
    
    # Ensure predictions are within valid range (1-5)
    rounded_preds = np.clip(rounded_preds, 1, 5)
    
    # Calculate regression metrics
    mae = mean_absolute_error(all_labels, all_preds)
    rmse = np.sqrt(mean_squared_error(all_labels, all_preds))
    
    # Calculate classification metrics
    precision = precision_score(all_labels, rounded_preds, average='macro')
    recall = recall_score(all_labels, rounded_preds, average='macro')
    f1 = f1_score(all_labels, rounded_preds, average='macro')
    accuracy = accuracy_score(all_labels, rounded_preds)
    
    # Calculate per-class metrics
    class_names = ["Rating 1", "Rating 2", "Rating 3", "Rating 4", "Rating 5"]
    per_class_f1 = f1_score(all_labels, rounded_preds, average=None)
    per_class_precision = precision_score(all_labels, rounded_preds, average=None)
    per_class_recall = recall_score(all_labels, rounded_preds, average=None)
    
    # Print all metrics
    print(f'Accuracy: {accuracy:.4f}')
    print(f'Precision: {precision:.4f}')
    print(f'Recall: {recall:.4f}')
    print(f'F1-Score: {f1:.4f}')
    print(f'MAE: {mae:.4f}')
    print(f'RMSE: {rmse:.4f}')
    print('Per-class F1 Scores:')
    for i, class_f1 in enumerate(per_class_f1):
        print(f'  {class_names[i]}: {class_f1:.4f}')
    
    return {
        'test_loss': avg_test_loss,
        'predictions': all_preds,
        'true_labels': all_labels,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'per_class_f1': per_class_f1,
        'per_class_precision': per_class_precision,
        'per_class_recall': per_class_recall,
        'mae': mae,
        'rmse': rmse
    }

def plot_training_history(history):
    """
    Plot training history
    """
    # Check if we have VAE-specific metrics
    has_vae_metrics = "reconstruction_loss" in history and "kl_loss" in history and len(history["reconstruction_loss"]) > 0
    
    if has_vae_metrics:
        plt.figure(figsize=(16, 10))
        
        # Plot 1: Overall losses
        plt.subplot(2, 2, 1)
        plt.plot(history['train_loss'], label='Train Loss')
        plt.plot(history['val_loss'], label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Overall Training and Validation Loss')
        plt.legend()
        
        # Plot 2: Reconstruction loss
        plt.subplot(2, 2, 2)
        plt.plot(history['reconstruction_loss'], label='Reconstruction Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Reconstruction Loss')
        plt.title('Reconstruction Loss (MSE)')
        plt.legend()
        
        # Plot 3: KL Divergence loss
        plt.subplot(2, 2, 3)
        plt.plot(history['kl_loss'], label='KL Divergence')
        plt.xlabel('Epoch')
        plt.ylabel('KL Loss')
        plt.title('KL Divergence Loss')
        plt.legend()
        
        # Plot 4: Comparison of all losses
        plt.subplot(2, 2, 4)
        plt.plot(history['train_loss'], label='Train Loss')
        plt.plot(history['val_loss'], label='Validation Loss')
        plt.plot(history['reconstruction_loss'], label='Reconstruction Loss')
        plt.plot(history['kl_loss'], label='KL Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('All Losses Comparison')
        plt.legend()
        
    else:
        plt.figure(figsize=(12, 4))
        
        # Plot loss
        plt.plot(history['train_loss'], label='Train Loss')
        plt.plot(history['val_loss'], label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss')
        plt.legend()
    
    plt.tight_layout()
    plt.savefig('/home/adiez/Desktop/Deep Learning/DL - Assignment 2/plots/06_autoencoder_recommendation_training_history.png')
    plt.show()

def plot_confusion_matrix(predictions, actuals, classes=None, metrics=None):
    """
    Plot a confusion matrix for the predicted and actual ratings.

    Args:
        predictions (list or np.array): Predicted ratings.
        actuals (list or np.array): Actual ratings.
        classes (list): List of class labels (e.g., [1, 2, 3, 4, 5]).
        metrics (dict): Dictionary containing evaluation metrics (optional).
    """
    # Round predictions to the nearest integer
    rounded_preds = np.rint(predictions).astype(int)
    
    # Ensure predictions are within valid range
    rounded_preds = np.clip(rounded_preds, 1, 5)
    
    # Compute confusion matrix
    cm = confusion_matrix(actuals, rounded_preds, labels=classes)
    
    # Print metrics if provided, otherwise compute accuracy
    if metrics:
        print(f'Accuracy: {metrics["accuracy"]:.4f}')
        print(f'Precision: {metrics["precision"]:.4f}')
        print(f'Recall: {metrics["recall"]:.4f}')
        print(f'F1-Score: {metrics["f1"]:.4f}')
        print(f'MAE: {metrics["mae"]:.4f}')
        print(f'RMSE: {metrics["rmse"]:.4f}')
    else:
        # Compute accuracy and print it (fallback if metrics not provided)
        accuracy = accuracy_score(actuals, rounded_preds)
        print(f'Accuracy: {accuracy:.4f}')
    
    print('Confusion Matrix:')
    print(cm)
    
    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.xlabel('Predicted Ratings')
    plt.ylabel('Actual Ratings')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig('/home/adiez/Desktop/Deep Learning/DL - Assignment 2/plots/06_autoencoder_recommendation_confusion_matrix.png')
    plt.show()

def plot_metrics(metrics):
    """
    Plot all evaluation metrics in a bar chart.
    
    Args:
        metrics (dict): Dictionary containing evaluation metrics.
    """
    # Metrics to plot
    metric_names = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    metric_values = [metrics['accuracy'], metrics['precision'], 
                     metrics['recall'], metrics['f1']]
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(metric_names, metric_values, color=['blue', 'green', 'orange', 'red'])
    
    # Add values on top of bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                 f'{height:.4f}', ha='center', va='bottom')
    
    plt.ylim(0, 1.1)  # Set y-axis limit
    plt.ylabel('Score')
    plt.title('Classification Metrics')
    plt.tight_layout()
    plt.savefig('/home/adiez/Desktop/Deep Learning/DL - Assignment 2/plots/06_autoencoder_recommendation_classification_metrics.png')
    plt.show()
    
    # Plot regression metrics (MAE and RMSE) separately
    plt.figure(figsize=(8, 5))
    regression_metrics = ['MAE', 'RMSE']
    regression_values = [metrics['mae'], metrics['rmse']]
    
    bars = plt.bar(regression_metrics, regression_values, color=['purple', 'teal'])
    
    # Add values on top of bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                 f'{height:.4f}', ha='center', va='bottom')
    
    plt.ylabel('Error')
    plt.title('Regression Metrics')
    plt.tight_layout()
    plt.savefig('/home/adiez/Desktop/Deep Learning/DL - Assignment 2/plots/06_autoencoder_recommendation_regression_metrics.png')
    plt.show()

def plot_latent_space(model, test_loader, device, n_samples=1000):
    """
    Plot the latent space of the autoencoder model.
    This helps visualize if the model has learned meaningful representations.
    
    Args:
        model: The trained autoencoder model
        test_loader: DataLoader for the test dataset
        device: The device to use for inference
        n_samples: Number of samples to plot
    """
    model.eval()
    
    users_list = []
    items_list = []
    latent_vectors = []
    ratings_list = []
    original_users = []
    original_items = []
    
    with torch.no_grad():
        for users, items, user_features, item_features, ratings in test_loader:
            users = users.to(device)
            items = items.to(device)
            user_features = user_features.to(device)
            item_features = item_features.to(device)
            
            if model.variational:
                _, z, mu, _ = model(users, items, user_features, item_features)
                latent = mu  # Use mean rather than sampled latent for visualization
            else:
                _, latent = model(users, items, user_features, item_features)
            
            users_list.extend(users.cpu().numpy())
            items_list.extend(items.cpu().numpy())
            latent_vectors.extend(latent.cpu().numpy())
            ratings_list.extend(ratings.cpu().numpy())
            
            if len(users_list) >= n_samples:
                break
    
    # Convert to numpy arrays
    users_np = np.array(users_list[:n_samples])
    items_np = np.array(items_list[:n_samples])
    latent_np = np.array(latent_vectors[:n_samples])
    ratings_np = np.array(ratings_list[:n_samples])
    
    # Use t-SNE for better visualization of high-dimensional data
    print("Performing t-SNE dimensionality reduction...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=1000)
    latent_2d = tsne.fit_transform(latent_np)
    
    # Also create PCA representation for comparison
    pca = PCA(n_components=2)
    latent_2d_pca = pca.fit_transform(latent_np)
    print(f"PCA explained variance ratio: {pca.explained_variance_ratio_}")
    
    # Create a figure with multiple plots
    plt.figure(figsize=(16, 12))
    
    # Plot 1: t-SNE by rating
    plt.subplot(2, 2, 1)
    scatter = plt.scatter(latent_2d[:, 0], latent_2d[:, 1], c=ratings_np, 
                        cmap='viridis', alpha=0.7, s=30)
    plt.colorbar(scatter, label='Rating')
    plt.title('t-SNE Latent Space (colored by rating)')
    plt.xlabel('t-SNE Dimension 1')
    plt.ylabel('t-SNE Dimension 2')
    
    # Plot 2: PCA by rating
    plt.subplot(2, 2, 2)
    scatter = plt.scatter(latent_2d_pca[:, 0], latent_2d_pca[:, 1], c=ratings_np, 
                        cmap='viridis', alpha=0.7, s=30)
    plt.colorbar(scatter, label='Rating')
    plt.title('PCA Latent Space (colored by rating)')
    plt.xlabel('PCA Dimension 1')
    plt.ylabel('PCA Dimension 2')
    
    # Plot 3: t-SNE by user
    plt.subplot(2, 2, 3)
    
    # Use a color map that can handle many unique users
    unique_users = np.unique(users_np)
    if len(unique_users) > 10:
        # If many users, just color by user modulo 10
        user_colors = users_np % 10
    else:
        user_colors = users_np
    
    scatter = plt.scatter(latent_2d[:, 0], latent_2d[:, 1], c=user_colors, 
                        cmap='tab10', alpha=0.7, s=30)
    plt.title('t-SNE Latent Space (colored by user)')
    plt.xlabel('t-SNE Dimension 1')
    plt.ylabel('t-SNE Dimension 2')
    
    # Plot 4: t-SNE by item
    plt.subplot(2, 2, 4)
    
    # Use a color map that can handle many unique items
    unique_items = np.unique(items_np)
    if len(unique_items) > 10:
        # If many items, just color by item modulo 10
        item_colors = items_np % 10
    else:
        item_colors = items_np
    
    scatter = plt.scatter(latent_2d[:, 0], latent_2d[:, 1], c=item_colors, 
                        cmap='tab10', alpha=0.7, s=30)
    plt.title('t-SNE Latent Space (colored by item)')
    plt.xlabel('t-SNE Dimension 1')
    plt.ylabel('t-SNE Dimension 2')
    
    plt.tight_layout()
    plt.savefig('/home/adiez/Desktop/Deep Learning/DL - Assignment 2/plots/06_autoencoder_recommendation_latent_space.png')
    plt.show()

def run_training_pipeline(ratings_file, users_file, movies_file, model_config=None, train_config=None, debug=False):
    # Default configurations
    default_model_config = {
        "latent_dim": 128,
        "hidden_layers": [512, 256, 128],
        "dropout_rate": 0.3,
        "use_batch_norm": True,
        "variational": True
    }
    
    default_train_config = {
        "batch_size": 256,
        "learning_rate": 0.001,
        "weight_decay": 1e-4,
        "num_epochs": 100,
        "patience": 15,
        "val_size": 0.15,
        "test_size": 0.1
    }
    
    # Update with provided configurations
    if model_config:
        default_model_config.update(model_config)
    
    if train_config:
        default_train_config.update(train_config)
    
    # Set manual seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Prepare datasets
    print("Preparing datasets...")
    train_dataset, val_dataset, test_dataset, n_users, n_items = prepare_datasets(
        ratings_file, users_file, movies_file,
        val_size=default_train_config["val_size"],
        test_size=default_train_config["test_size"]
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=default_train_config["batch_size"],
        shuffle=True,
        num_workers=4
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=default_train_config["batch_size"],
        shuffle=False,
        num_workers=4
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=default_train_config["batch_size"],
        shuffle=False,
        num_workers=4
    )
    
    print(f"Number of users: {n_users}, Number of items: {n_items}")
    print(f"Dataset sizes: Train: {len(train_dataset)}, Validation: {len(val_dataset)}, Test: {len(test_dataset)}")
    
    print("Creating model...")
    model = AutoencoderRecommender(
        num_users=n_users,
        num_items=n_items,
        num_user_features=len(USER_FEATURES),
        num_item_features=len(ITEM_FEATURES),
        latent_dim=default_model_config["latent_dim"],
        hidden_layers=default_model_config["hidden_layers"],
        dropout_rate=default_model_config["dropout_rate"],
        use_batch_norm=default_model_config["use_batch_norm"],
        variational=default_model_config["variational"]
    )
    
    # Print model architecture
    if debug:
        print("Model Architecture:")
        print(model)
        total_params = sum(p.numel() for p in model.parameters())
        print(f"Total parameters: {total_params:,}")
    
    optimizer = optim.Adam(
        model.parameters(), 
        lr=default_train_config["learning_rate"], 
        weight_decay=default_train_config["weight_decay"]
    )
    
    # Update train_model to include debug flag
    if debug:
        # Enable debug flag in train_model
        # Get a small batch for testing
        for batch in train_loader:
            users, items, user_features, item_features, ratings = batch
            print("Sample batch shapes:")
            print(f"Users: {users.shape}, Items: {items.shape}")
            print(f"User features: {user_features.shape}, Item features: {item_features.shape}")
            
            # Move to device for a test forward pass
            users = users.to(device)
            items = items.to(device)
            user_features = user_features.to(device)
            item_features = item_features.to(device)
            
            model = model.to(device)
            
            print("Testing forward pass...")
            with torch.no_grad():
                try:
                    if model.variational:
                        outputs, z, mu, logvar = model(users, items, user_features, item_features)
                        print("Forward pass successful!")
                        print(f"Output shape: {outputs.shape}")
                        print(f"Latent shape: {z.shape}")
                        print(f"Mu shape: {mu.shape}")
                        print(f"LogVar shape: {logvar.shape}")
                    else:
                        outputs, latent = model(users, items, user_features, item_features)
                        print("Forward pass successful!")
                        print(f"Output shape: {outputs.shape}")
                        print(f"Latent shape: {latent.shape}")
                except Exception as e:
                    print(f"Forward pass error: {e}")
                    print("Error occurred during forward pass. Check model dimensions.")
                    raise
            break
    
    # Train the model
    print("Starting training...")
    trained_model, history = train_model(
        model, 
        train_loader, 
        val_loader, 
        optimizer,
        device=device,
        num_epochs=default_train_config["num_epochs"],
        patience=default_train_config["patience"]
    )
    
    print("Training completed!")
    print("Evaluating model on test set...")
    results = evaluate_model(trained_model, test_loader, device)
    
    # Plot training history
    plot_training_history(history)
    
    # Plot confusion matrix
    plot_confusion_matrix(results['predictions'], results['true_labels'], 
                        classes=[1, 2, 3, 4, 5], metrics=results)
    
    # Plot metrics
    plot_metrics(results)
    
    # Plot latent space visualization
    plot_latent_space(trained_model, test_loader, device)
    
    return trained_model, history, results

if __name__ == "__main__":
    ratings_file = ROUTE + "/data.csv"
    users_file = ROUTE + "/user.csv"
    movies_file = ROUTE + "/item.csv"
    
    # Configuration for improved VAE model
    model_config = {
        "latent_dim": 128,  # Larger latent space
        "hidden_layers": [512, 256, 128],  # Deeper network
        "dropout_rate": 0.3,  # Moderate dropout
        "use_batch_norm": True,
        "variational": True  # Enable variational autoencoder
    }
    
    # Alternative configuration for classic autoencoder (for comparison)
    standard_ae_config = {
        "latent_dim": 128,
        "hidden_layers": [512, 256, 128],
        "dropout_rate": 0.3,
        "use_batch_norm": True,
        "variational": False  # Classic autoencoder
    }
    
    # Enhanced training hyperparameters
    train_config = {
        "batch_size": 256,  # Larger batch size for more stable gradients
        "learning_rate": 0.001,
        "weight_decay": 1e-4,
        "num_epochs": 100,  # Train longer
        "patience": 15,  # More patience for complex model
        "val_size": 0.15,
        "test_size": 0.1
    }
    
    # Choose which model to use
    use_variational = True
    active_model_config = model_config if use_variational else standard_ae_config
    
    # Enable debugging by default
    debug = True
    
    trained_model, history, results = run_training_pipeline(
        ratings_file=ratings_file,
        users_file=users_file,
        movies_file=movies_file,
        model_config=active_model_config,
        train_config=train_config,
        debug=debug
    ) 