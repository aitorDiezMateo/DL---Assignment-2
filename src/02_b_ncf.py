#! Neural Collaborative Filtering (NCF) Model
#? In this script we implement NCF to predict user-item ratings (1,2,3,4 or 5).

import torch
from torch.utils.data import Dataset
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from imblearn.over_sampling import RandomOverSampler
import time
import os
import seaborn as sns

# Define the model
class NCF(nn.Module):
    def __init__(self, num_users, num_items, embedding_dim=64, mlp_dims=[128, 64, 32, 16], 
                dropout_rate=0.2, use_batch_norm=True):
        super(NCF, self).__init__()
        
        # GMF part: Generalized Matrix Factorization
        self.user_embedding_gmf = nn.Embedding(num_users, embedding_dim)
        self.item_embedding_gmf = nn.Embedding(num_items, embedding_dim)
        
        # MLP part: Multilayer Perceptron
        self.user_embedding_mlp = nn.Embedding(num_users, embedding_dim)
        self.item_embedding_mlp = nn.Embedding(num_items, embedding_dim)
        
        # MLP tower structure with configurable layers
        self.mlp_layers = nn.ModuleList()
        input_dim = embedding_dim * 2
        
        # Create tower structure with decreasing dimensions
        for i, dim in enumerate(mlp_dims):
            # Add linear layer
            self.mlp_layers.append(nn.Linear(input_dim, dim))
            
            # Add batch normalization if specified
            if use_batch_norm:
                self.mlp_layers.append(nn.BatchNorm1d(dim))
                
            # Add activation
            self.mlp_layers.append(nn.ReLU())
            
            # Add dropout
            self.mlp_layers.append(nn.Dropout(dropout_rate))
            
            # Update input dimension for next layer
            input_dim = dim
        
        # Final output layer for regression (output a single value)
        self.final_layer = nn.Linear(embedding_dim + mlp_dims[-1], 1)
    
        # NEW: Add output scaling parameters (Uncomment to delete)
        self.scale = nn.Parameter(torch.tensor(4.0))  # 5-1 = 4 rating range
        self.shift = nn.Parameter(torch.tensor(1.0))  # Minimum rating
    
        # Initialize weights
        self._init_weights()
        
    def _init_weights(self):
        """Initialize weights using Xavier/Glorot initialization"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, std=0.01)
    
    def forward(self, user, item):
        # GMF part
        user_gmf = self.user_embedding_gmf(user)
        item_gmf = self.item_embedding_gmf(item)
        gmf_interaction = user_gmf * item_gmf  # Element-wise multiplication
        
        # MLP part
        user_mlp = self.user_embedding_mlp(user)
        item_mlp = self.item_embedding_mlp(item)
        mlp_input = torch.cat([user_mlp, item_mlp], dim=-1)  # Concatenation
        
        # Process through tower structure
        x = mlp_input
        for layer in self.mlp_layers:
            x = layer(x)
        mlp_output = x
        
        # Combine GMF and MLP outputs
        neumf_output = torch.cat([gmf_interaction, mlp_output], dim=-1)
        
        # Regression output
        raw_output = self.final_layer(neumf_output)
                # Apply sigmoid and scale to rating range (1-5)
        scaled_output = torch.sigmoid(raw_output) * self.scale + self.shift
        
        return scaled_output.squeeze()
        
        # return raw_output.squeeze()  # Return continuous ratings (batch_size,)

def prepare_datasets(ratings_file, val_size=0.1, test_size=0.1, random_state=42):
    """
    Prepare train, validation, and test datasets
    """
    # Read ratings data
    ratings_df = pd.read_csv(ratings_file)
    
    # Get number of users and items (assuming IDs start from 0 or are already mapped)
    n_users = ratings_df['user_id'].max() + 1  
    n_items = ratings_df['item_id'].max() + 1
    
    # First split: separate out test set
    train_val_df, test_df = train_test_split(
        ratings_df, test_size=test_size, random_state=random_state, stratify=ratings_df['user_id']
    )
    
    # Second split: separate train and validation sets
    # Adjust validation size to get the right proportion from the remaining data
    adjusted_val_size = val_size / (1 - test_size)
    train_df, val_df = train_test_split(
        train_val_df, test_size=adjusted_val_size, random_state=random_state, 
        stratify=train_val_df['user_id']
    )
    
    # Oversample the minority classes in the training set
    # ros = RandomOverSampler(random_state=random_state)
    # train_df, _ = ros.fit_resample(train_df[['user_id', 'item_id', 'rating_class']], train_df['rating_class'])
    # train_df = pd.DataFrame(train_df, columns=['user_id', 'item_id', 'rating_class'])
    # Reset indices
    train_df = train_df.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)
    
    print(f"Data split sizes: Train: {len(train_df)}, Validation: {len(val_df)}, Test: {len(test_df)}")
    
    # Create custom datasets
    class RatingDataset(torch.utils.data.Dataset):
        def __init__(self, ratings_df):
            self.users = torch.LongTensor(ratings_df['user_id'].values)
            self.items = torch.LongTensor(ratings_df['item_id'].values)
            self.ratings = torch.LongTensor(ratings_df['rating'].values)  # Use 0-indexed classes
            
        def __len__(self):
            return len(self.users)
            
        def __getitem__(self, idx):
            return self.users[idx], self.items[idx], self.ratings[idx]
    
    train_dataset = RatingDataset(train_df)
    val_dataset = RatingDataset(val_df)
    test_dataset = RatingDataset(test_df)
    
    return train_dataset, val_dataset, test_dataset, n_users, n_items

def train_ncf_model(model, train_loader, val_loader, optimizer, 
                    device, num_epochs=10, patience=3, checkpoint_dir='./checkpoints',
                    class_weights=None):
    """
    Train the NCF model for regression
    
    Args:
        model: NCF model
        train_loader: Training data loader
        val_loader: Validation data loader
        optimizer: Optimizer (e.g., Adam)
        criterion: Loss function (e.g., CrossEntropyLoss)
        device: Device to run training on (CPU or GPU)
        num_epochs: Maximum number of training epochs
        patience: Early stopping patience
        checkpoint_dir: Directory to save model checkpoints
        class_weights: Weights for each class for weighted loss
    
    Returns:
        Trained model and training history
    """
    # Create checkpoint directory if it doesn't exist
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Move model to device
    model = model.to(device)
    
    # Use Mean Squared Error (MSE) loss for regression
    criterion = nn.MSELoss()
    
    # Initialize tracking variables
    best_val_loss = float('inf')
    best_epoch = 0
    epochs_no_improve = 0
    
    # History for tracking metrics
    history = {
        'train_loss': [],
        'val_loss': [],
        'val_accuracy': [],
        'val_f1': []
    }
    
    # Start training
    start_time = time.time()
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        
        for batch_idx, (users, items, ratings) in enumerate(train_loader):
            # Move batch to device
            users, items, ratings = users.to(device), items.to(device), ratings.to(device)
            
            # Zero gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(users, items)
            
            # Calculate loss
            loss = criterion(outputs, ratings.float())  # Ensure ratings are float for regression
            
            # Backward pass
            loss.backward()
            
            # Update weights
            optimizer.step()
            
            # Accumulate batch loss
            train_loss += loss.item()
            
            # Print progress every 100 batches
            if (batch_idx + 1) % 100 == 0:
                print(f'Epoch {epoch+1}/{num_epochs}, Batch {batch_idx+1}/{len(train_loader)}, Loss: {loss.item():.4f}')
        
        # Calculate average training loss
        avg_train_loss = train_loss / len(train_loader)
        history['train_loss'].append(avg_train_loss)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for users, items, ratings in val_loader:
                # Move batch to device
                users, items, ratings = users.to(device), items.to(device), ratings.to(device)
                
                # Forward pass
                outputs = model(users, items)
                
                # Calculate loss
                loss = criterion(outputs, ratings.float())
                val_loss += loss.item()
                
                # Store predictions and labels for metrics calculation
                all_preds.extend(outputs.cpu().numpy())
                all_labels.extend(ratings.cpu().numpy())
        
        # Calculate average validation loss
        avg_val_loss = val_loss / len(val_loader)
        
        # Save metrics to history
        history['val_loss'].append(avg_val_loss)
        
        # Print epoch summary
        print(f'Epoch {epoch+1}/{num_epochs} - '
              f'Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}')
        
        # Check if this is the best model so far
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
            }, os.path.join(checkpoint_dir, 'best_model.pt'))
            
            print(f"Saved best model checkpoint with validation loss: {best_val_loss:.4f}")
        else:
            epochs_no_improve += 1
        
        # Early stopping check
        if epochs_no_improve >= patience:
            print(f'Early stopping triggered! No improvement for {patience} epochs.')
            break
    
    # Calculate training time
    training_time = time.time() - start_time
    print(f'Training completed in {training_time/60:.2f} minutes')
    print(f'Best model was from epoch {best_epoch+1} with validation loss {best_val_loss:.4f}')
    
    # Load the best model
    checkpoint = torch.load(os.path.join(checkpoint_dir, 'best_model.pt'))
    model.load_state_dict(checkpoint['model_state_dict'])
    
    return model, history

def evaluate_model(model, test_loader, device):
    """
    Evaluate the model on test data for regression
    """
    model.eval()
    test_loss = 0.0
    all_preds = []
    all_labels = []
    
    criterion = nn.MSELoss()
    
    with torch.no_grad():
        for users, items, ratings in test_loader:
            users, items, ratings = users.to(device), items.to(device), ratings.to(device)
            outputs = model(users, items)
            
            loss = criterion(outputs, ratings.float())
            test_loss += loss.item()
            
            all_preds.extend(outputs.cpu().numpy())
            all_labels.extend(ratings.cpu().numpy())
    
    # Calculate metrics
    avg_test_loss = test_loss / len(test_loader)
    print(f'Test Loss (MSE): {avg_test_loss:.4f}')
    
    return {
        'test_loss': avg_test_loss,
        'predictions': all_preds,
        'true_labels': all_labels
    }

def plot_training_history(history):
    """
    Plot training history
    """
    plt.figure(figsize=(12, 4))
    
    # Plot loss
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.tight_layout()
    plt.savefig('training_history.png')
    plt.show()

def plot_predictions_vs_actuals(predictions, actuals):
    """
    Plot predicted ratings vs. actual ratings for regression
    """
    plt.figure(figsize=(8, 8))
    plt.scatter(actuals, predictions, alpha=0.5, edgecolors='k', color='blue')
    plt.plot([min(actuals), max(actuals)], [min(actuals), max(actuals)], color='red', linestyle='--', linewidth=2)
    plt.xlabel('Actual Ratings')
    plt.ylabel('Predicted Ratings')
    plt.title('Predicted vs. Actual Ratings')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('predictions_vs_actuals.png')
    plt.show()

def plot_confusion_matrix(predictions, actuals, classes=None):
    """
    Plot a confusion matrix for the predicted and actual ratings.

    Args:
        predictions (list or np.array): Predicted ratings.
        actuals (list or np.array): Actual ratings.
        classes (list): List of class labels (e.g., [1, 2, 3, 4, 5]).
    """
    # Round predictions to the nearest integer
    rounded_preds = np.rint(predictions).astype(int)
    
    # Compute confusion matrix
    cm = confusion_matrix(actuals, rounded_preds, labels=classes)
    
    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.xlabel('Predicted Ratings')
    plt.ylabel('Actual Ratings')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png')
    plt.show()

def get_class_weights(train_loader,device):
    all_ratings = []
    for _, _, ratings in train_loader.dataset:
        all_ratings.append(ratings)
    all_ratings = np.array(all_ratings)
    
    # Compute class weights
    class_weights = compute_class_weight(
        class_weight='balanced',
        classes=np.unique(all_ratings),
        y=all_ratings
    )
    
    return torch.FloatTensor(class_weights).to(device)

# Main execution function
def run_training_pipeline(ratings_file, model_config=None, train_config=None):
    """
    Run the complete training pipeline
    """
    # Default configurations
    default_model_config = {
        "embedding_dim": 1024,
        "mlp_dims": [512, 256, 128, 64],
        "dropout_rate": 0.5,
        "use_batch_norm": True
    }
    
    default_train_config = {
        "batch_size": 256,
        "learning_rate": 0.0001,
        "weight_decay": 1e-5,
        "num_epochs": 40,
        "patience": 5,
        "val_size": 0.1,
        "test_size": 0.1
    }
    
    # Update with provided configurations
    if model_config:
        # default_model_config.update(model_config)
        pass
    
    if train_config:
        # default_train_config.update(train_config)
        pass
    
    #Set random seed
    torch.manual_seed(42)
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Prepare datasets
    print("Preparing datasets...")
    train_dataset, val_dataset, test_dataset, n_users, n_items = prepare_datasets(
        ratings_file=ratings_file,
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
    
    # Create model
    print("Creating model...")
    model = NCF(
        num_users=n_users,
        num_items=n_items,
        embedding_dim=default_model_config["embedding_dim"],
        mlp_dims=default_model_config["mlp_dims"],
        dropout_rate=default_model_config["dropout_rate"],
        use_batch_norm=default_model_config["use_batch_norm"]
    )
    
    # # Define loss function and optimizer
    class_weights = get_class_weights(train_loader,device=device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(
        model.parameters(),
        lr=default_train_config["learning_rate"],
        weight_decay=default_train_config["weight_decay"]
    )
    # Define Class Weights

    # Train model
    print("Starting training...")
    trained_model, history = train_ncf_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        # criterion=criterion,
        device=device,
        num_epochs=default_train_config["num_epochs"],
        patience=default_train_config["patience"],
        class_weights=class_weights
    )
    
    # Evaluate model on test set
    print("Evaluating model on test set...")
    evaluation_results = evaluate_model(trained_model, test_loader, device)
    
    plot_training_history(history)

    # plot_predictions_vs_actuals(evaluation_results['predictions'], evaluation_results['true_labels'])
    
    plot_confusion_matrix(evaluation_results['predictions'], evaluation_results['true_labels'], classes=[1, 2, 3, 4, 5])
    
    return trained_model, history, evaluation_results

# Example usage
if __name__ == "__main__":
    # Path to ratings file (e.g., "ml-1m/ratings.csv")
    ratings_file = "/home/adiez/Desktop/Deep Learning/DL - Assignment 2/data/100k/processed/data.csv"
    
    # Custom configurations (optional)
    model_config = {
        "embedding_dim": 64,
        "mlp_dims": [256, 128, 64, 32],
        "dropout_rate": 0.3
    }
    
    train_config = {
        "batch_size": 512,
        "learning_rate": 0.0005,
        "num_epochs": 30,
        "val_size": 0.15,  # 15% for validation
        "test_size": 0.15   # 15% for testing
    }
    
    # Run training pipeline
    model, history, evaluation = run_training_pipeline(
        ratings_file=ratings_file,
        model_config=model_config,
        train_config=train_config
    )