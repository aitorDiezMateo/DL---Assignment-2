#! Neural Collaborative Filtering (NCF) Model for Classification
#? Predicts user-item ratings (1-5) as a classification task

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
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, precision_score, recall_score, mean_absolute_error, mean_squared_error, roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from imblearn.over_sampling import RandomOverSampler
import time
import os
from itertools import cycle

# Define the NCF model for classification
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
        
        # MLP tower structure
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
        
        # Final output layer (NeuMF Layer)
        self.final_layer = nn.Linear(embedding_dim + mlp_dims[-1], 5)
        
    def _init_weights(self):
        # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, std=0.01)
        
        # Special initialization for final layer
        nn.init.xavier_normal_(self.final_layer.weight, gain=0.1)

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
        
        # Get final logits for 5 classes (ratings 1-5)
        final_logits = self.final_layer(neumf_output)
        
        return torch.softmax(final_logits, dim=-1)

def prepare_datasets(ratings_file, val_size=0.1, test_size=0.1, random_state=42):
    # Read ratings data
    ratings_df = pd.read_csv(ratings_file)
    
    # Get number of users and items
    n_users = ratings_df['user_id'].max() + 1  
    n_items = ratings_df['item_id'].max() + 1
    
    # Convert ratings to 0-indexed classes (original ratings are 1-5)
    ratings_df['rating_class'] = ratings_df['rating'] - 1
    
    # Split data into train, validation and test sets
    train_val_df, test_df = train_test_split(
        ratings_df, test_size=test_size, random_state=random_state, stratify=ratings_df['user_id']
    )
    
    adjusted_val_size = val_size / (1 - test_size)
    train_df, val_df = train_test_split(
        train_val_df, test_size=adjusted_val_size, random_state=random_state, 
        stratify=train_val_df['user_id']
    )
    
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
            self.ratings = torch.LongTensor(ratings_df['rating_class'].values)  # Use 0-indexed classes
            
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
    # Create checkpoint directory
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Move model to device
    model = model.to(device)
    
    # Set up loss function
    if class_weights is not None:
        criterion = torch.nn.CrossEntropyLoss(weight=class_weights.to(device))
        print("Using weighted loss function with weights:", class_weights)
    else:
        criterion = torch.nn.CrossEntropyLoss()
    
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
            loss = criterion(outputs, ratings)
            
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
                loss = criterion(outputs, ratings)
                val_loss += loss.item()
                
                # Get predictions
                _, preds = torch.max(outputs, 1)
                
                # Store predictions and labels for metrics
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(ratings.cpu().numpy())
        
        # Calculate validation metrics
        avg_val_loss = val_loss / len(val_loader)
        val_accuracy = accuracy_score(all_labels, all_preds)
        val_f1 = f1_score(all_labels, all_preds, average='weighted')
        
        # Save metrics to history
        history['val_loss'].append(avg_val_loss)
        history['val_accuracy'].append(val_accuracy)
        history['val_f1'].append(val_f1)
        
        # Print epoch summary
        print(f'Epoch {epoch+1}/{num_epochs} - '
              f'Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, '
              f'Val Accuracy: {val_accuracy:.4f}, Val F1: {val_f1:.4f}')
        
        # Save best model
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
            }, os.path.join(checkpoint_dir, '02_ncf_classification_best_model.pt'))
            
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
    checkpoint = torch.load(os.path.join(checkpoint_dir, '02_ncf_classification_best_model.pt'))
    model.load_state_dict(checkpoint['model_state_dict'])
    
    return model, history

def evaluate_model(model, test_loader, criterion, device):
    # Set model to evaluation mode
    model.eval()
    test_loss = 0.0
    all_preds = []
    all_labels = []
    all_outputs_raw = []  # For regression metrics
    all_probabilities = []  # For ROC AUC
    
    with torch.no_grad():
        for users, items, ratings in test_loader:
            users, items, ratings = users.to(device), items.to(device), ratings.to(device)
            outputs = model(users, items)
            
            loss = criterion(outputs, ratings)
            test_loss += loss.item()
            
            # Store probabilities for ROC AUC
            all_probabilities.extend(outputs.cpu().numpy())
            
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(ratings.cpu().numpy())
            
            # Calculate expected ratings (for regression metrics)
            proba_outputs = F.softmax(outputs, dim=1)
            rating_values = torch.tensor([0, 1, 2, 3, 4], device=device).float()
            expected_ratings = torch.sum(proba_outputs * rating_values.unsqueeze(0), dim=1)
            # Convert back to 1-5 scale
            expected_ratings = expected_ratings + 1  
            all_outputs_raw.extend(expected_ratings.cpu().numpy())
    
    # Calculate metrics
    avg_test_loss = test_loss / len(test_loader)
    test_accuracy = accuracy_score(all_labels, all_preds)
    test_precision = precision_score(all_labels, all_preds, average='macro')
    test_recall = recall_score(all_labels, all_preds, average='macro')
    test_f1 = f1_score(all_labels, all_preds, average='weighted')
    
    # Convert to 1-5 scale for regression metrics
    true_ratings = np.array(all_labels) + 1
    
    # Calculate regression metrics
    mae = mean_absolute_error(true_ratings, all_outputs_raw)
    rmse = np.sqrt(mean_squared_error(true_ratings, all_outputs_raw))
    
    # Calculate per-class metrics
    class_names = ["Rating 1", "Rating 2", "Rating 3", "Rating 4", "Rating 5"]
    per_class_f1 = f1_score(all_labels, all_preds, average=None)
    per_class_precision = precision_score(all_labels, all_preds, average=None)
    per_class_recall = recall_score(all_labels, all_preds, average=None)
    
    # Print results
    print(f'Test Loss: {avg_test_loss:.4f}')
    print(f'Test Accuracy: {test_accuracy:.4f}')
    print(f'Test Precision: {test_precision:.4f}')
    print(f'Test Recall: {test_recall:.4f}')
    print(f'Test F1 Score (weighted): {test_f1:.4f}')
    print(f'Test MAE: {mae:.4f}')
    print(f'Test RMSE: {rmse:.4f}')
    print('Per-class F1 Scores:')
    for i, class_f1 in enumerate(per_class_f1):
        print(f'  {class_names[i]}: {class_f1:.4f}')
    print('Confusion Matrix:')
    print(confusion_matrix(all_labels, all_preds))
    
    return {
        'test_loss': avg_test_loss,
        'test_accuracy': test_accuracy,
        'test_precision': test_precision,
        'test_recall': test_recall,
        'test_f1': test_f1,
        'per_class_f1': per_class_f1,
        'per_class_precision': per_class_precision,
        'per_class_recall': per_class_recall,
        'confusion_matrix': confusion_matrix(all_labels, all_preds),
        'mae': mae,
        'rmse': rmse,
        'probabilities': all_probabilities,
        'true_labels': all_labels
    }

def plot_training_history(history):
    # Plot loss curves
    plt.figure(figsize=(12, 4))
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.tight_layout()
    plt.savefig('/home/adiez/Desktop/Deep Learning/DL - Assignment 2/plots/02_ncf_classification_training_history.png')
    plt.show()
    
    # Plot validation metrics
    plt.figure(figsize=(12, 4))
    plt.plot(history['val_accuracy'], label='Accuracy')
    plt.plot(history['val_f1'], label='F1 Score')
    plt.xlabel('Epoch')
    plt.ylabel('Score')
    plt.title('Validation Metrics')
    plt.legend()
    plt.tight_layout()
    plt.savefig('/home/adiez/Desktop/Deep Learning/DL - Assignment 2/plots/02_ncf_classification_validation_metrics.png')
    plt.show()

def plot_confusion_matrix(cm, class_names):
    # Create confusion matrix plot
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    
    # Set labels
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=class_names, yticklabels=class_names,
           ylabel='True Rating',
           xlabel='Predicted Rating')
    
    # Rotate x tick labels
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    
    # Add text annotations
    fmt = 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    
    fig.tight_layout()
    plt.savefig('/home/adiez/Desktop/Deep Learning/DL - Assignment 2/plots/02_ncf_classification_confusion_matrix.png')
    plt.show()
    return ax

def get_class_weights(train_loader,device):
    # Extract all ratings from the training set
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

def plot_metrics(metrics):
    # Plot classification metrics
    metric_names = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    metric_values = [metrics['test_accuracy'], metrics['test_precision'], 
                     metrics['test_recall'], metrics['test_f1']]
    
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
    plt.savefig('/home/adiez/Desktop/Deep Learning/DL - Assignment 2/plots/02_ncf_classification_classification_metrics.png')
    plt.show()
    
    # Plot regression metrics
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
    plt.savefig('/home/adiez/Desktop/Deep Learning/DL - Assignment 2/plots/02_ncf_classification_regression_metrics.png')
    plt.show()

def plot_roc_auc(probabilities, actuals, classes=None):
    # Prepare data for ROC curve plotting
    probabilities = np.array(probabilities)
    actuals = np.array(actuals)
    n_classes = probabilities.shape[1]

    if classes is None:
        classes = [f"Class {i+1}" for i in range(n_classes)]
    elif len(classes) != n_classes:
        raise ValueError("Length of 'classes' must match the number of classes in 'probabilities'")

    # Compute ROC curve and AUC for each class
    fpr, tpr, roc_auc = {}, {}, {}

    plt.figure(figsize=(10, 8))
    colors = cycle(['blue', 'red', 'green', 'purple', 'orange', 'cyan', 'magenta', 'yellow', 'black', 'grey'])

    for i in range(n_classes):
        # Get scores for current class
        class_scores = probabilities[:, i]

        # Create binary labels for current class (One-vs-Rest)
        binary_actuals = (actuals == i).astype(int)

        # Compute ROC curve and AUC
        fpr[i], tpr[i], _ = roc_curve(binary_actuals, class_scores)
        roc_auc[i] = auc(fpr[i], tpr[i])

        # Plot ROC curve for this class
        plt.plot(fpr[i], tpr[i], color=next(colors), lw=2,
                 label=f'{classes[i]} (AUC = {roc_auc[i]:.2f})')

    # Plot diagonal line (random classifier)
    plt.plot([0, 1], [0, 1], 'k--', lw=2, label='Random Chance')

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves (One-vs-Rest)')
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig('/home/adiez/Desktop/Deep Learning/DL - Assignment 2/plots/02_ncf_classification_roc_auc.png')
    plt.show()

def run_training_pipeline(ratings_file, model_config, train_config):
    # Set random seed for reproducibility
    torch.manual_seed(42)
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Prepare datasets
    print("Preparing datasets...")
    train_dataset, val_dataset, test_dataset, n_users, n_items = prepare_datasets(
        ratings_file=ratings_file,
        val_size=train_config["val_size"],
        test_size=train_config["test_size"]
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=train_config["batch_size"],
        shuffle=True,
        num_workers=4
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=train_config["batch_size"],
        shuffle=False,
        num_workers=4
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=train_config["batch_size"],
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
        embedding_dim=model_config["embedding_dim"],
        mlp_dims=model_config["mlp_dims"],
        dropout_rate=model_config["dropout_rate"],
        use_batch_norm=model_config["use_batch_norm"]
    )
    
    # Define loss function and optimizer
    class_weights = get_class_weights(train_loader,device=device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(
        model.parameters(),
        lr=train_config["learning_rate"],
        weight_decay=train_config["weight_decay"]
    )

    # Train model
    print("Starting training...")
    trained_model, history = train_ncf_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        device=device,
        num_epochs=train_config["num_epochs"],
        patience=train_config["patience"],
        class_weights=class_weights
    )
    
    # Evaluate model
    print("Evaluating model on test set...")
    evaluation_results = evaluate_model(trained_model, test_loader, criterion, device)
    
    # Plot results
    plot_training_history(history)
    class_names = ["Rating 1", "Rating 2", "Rating 3", "Rating 4", "Rating 5"]
    plot_confusion_matrix(evaluation_results['confusion_matrix'], class_names)
    plot_metrics(evaluation_results)
    plot_roc_auc(evaluation_results['probabilities'], evaluation_results['true_labels'], class_names)
    
    return trained_model, history, evaluation_results

# Main execution
if __name__ == "__main__":
    # Path to ratings file
    ratings_file = "/home/adiez/Desktop/Deep Learning/DL - Assignment 2/data/100k/processed/data.csv"
    
    # Model configuration
    model_config = {
        "embedding_dim": 64,
        "mlp_dims": [256,128,64,32],
        "dropout_rate": 0.05,
        "use_batch_norm": True
    }
    
    # Training configuration
    train_config = {
        "batch_size": 256,
        "learning_rate": 0.001,
        "num_epochs": 100,
        "weight_decay": 0.0,
        "patience": 5,
        "val_size": 0.1,
        "test_size": 0.1
    }
    
    # Run training pipeline
    model, history, evaluation = run_training_pipeline(
        ratings_file=ratings_file,
        model_config=model_config,
        train_config=train_config
    )