#! Neural Collaborative Filtering (NCF) Model
#? In this script we implement NCF to predict user-item ratings (1,2,3,4 or 5).

from sklearn.calibration import LabelEncoder
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
from sklearn.metrics import roc_curve, auc
from itertools import cycle


ROUTE = "/home/adiez/Desktop/Deep Learning/DL - Assignment 2/data/100k/processed"

USER_FEATURES = ['age', 'gender_F','gender_M', 'occupation_administrator', 'occupation_artist',       'occupation_doctor', 'occupation_educator', 'occupation_engineer', 'occupation_entertainment', 'occupation_executive', 'occupation_healthcare', 'occupation_homemaker','occupation_lawyer','occupation_librarian', 'occupation_marketing', 'occupation_none','occupation_other', 'occupation_programmer', 'occupation_retired','occupation_salesman', 'occupation_scientist', 'occupation_student',
'occupation_technician', 'occupation_writer', 'release_date']

ITEM_FEATURES = ['unknown','Action', 'Adventure', 'Animation', 'Children', 'Comedy', 'Crime','Documentary', 'Drama', 'Fantasy', 'Film-Noir', 'Horror', 'Musical','Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western']

ratings_df = pd.read_csv(ROUTE + "/data.csv")
users_df = pd.read_csv(ROUTE + "/user.csv")
movies_df = pd.read_csv(ROUTE + "/item.csv")

data = ratings_df.merge(users_df, on='user_id', how='left')
data = data.merge(movies_df, on='item_id', how='left')

n_users = data['user_id'].nunique()
n_items = data['item_id'].nunique()

class HybridNCF(nn.Module):
    def __init__(self, num_users, num_items,
                embedding_dim=64, mlp_dims=[128, 64, 32, 16],
                dropout_rate=0.2, use_batch_norm=True,
                num_user_features=25, # User features
                num_item_features=19): # Item features
        super(HybridNCF, self).__init__()
        
        
        # Collaborative Filtering Embeddings
        self.user_embedding_gmf_cf = nn.Embedding(num_users, embedding_dim)
        self.item_embedding_gmf_cf = nn.Embedding(num_items, embedding_dim)
        self.user_embedding_mlp_cf = nn.Embedding(num_users, embedding_dim)
        self.item_embedding_mlp_cf = nn.Embedding(num_items, embedding_dim)
        
        # MLP tower structure
        self.mlp_layers = nn.ModuleList()
        input_dim = (embedding_dim * 2) + num_item_features + num_item_features
        for dim in mlp_dims:
            self.mlp_layers.append(nn.Linear(input_dim, dim))
            if use_batch_norm:
                self.mlp_layers.append(nn.BatchNorm1d(dim))
            self.mlp_layers.append(nn.ReLU())
            self.mlp_layers.append(nn.Dropout(dropout_rate))
            input_dim = dim
            
        # Final prediction layers
        self.gmf_output = nn.Linear(embedding_dim, 1)
        self.mlp_output = nn.Linear(mlp_dims[-1], 1)
        self.final_output = nn.Linear(2, 1)
        
        # Initialize weights
        self._init_weights()
        
    def _init_weights(self):
        """Initialize weights to have reasonable starting values"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    module.bias.data.zero_()
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, std=0.01)
                
    def forward(self, user_indices, item_indices, user_features, item_features):
        # GMF path (element-wise multiplication of user and item embeddings)
        user_embed_gmf = self.user_embedding_gmf_cf(user_indices)
        item_embed_gmf = self.item_embedding_gmf_cf(item_indices)
        gmf_vector = user_embed_gmf * item_embed_gmf
        gmf_output = self.gmf_output(gmf_vector)
        
        # MLP path (concatenation of embeddings and features)
        user_embed_mlp = self.user_embedding_mlp_cf(user_indices)
        item_embed_mlp = self.item_embedding_mlp_cf(item_indices)
        
        
        # Concatenate all inputs for MLP path
        mlp_vector = torch.cat([
            user_embed_mlp,
            item_embed_mlp,
            user_features,
            item_features
        ], dim=1)
        
        # Process through MLP layers
        for layer in self.mlp_layers:
            mlp_vector = layer(mlp_vector)
        
        # MLP output
        mlp_output = self.mlp_output(mlp_vector)
        
        # Combine GMF and MLP outputs
        combined = torch.cat([gmf_output, mlp_output], dim=1)
        prediction = self.final_output(combined)
        
        # For rating prediction (1-5), no activation is needed for regression
        # You could add a sigmoid/tanh + scaling if you want to constrain the range
        # prediction = 1.0 + 4.0 * torch.sigmoid(prediction)  # Scale to [1,5]
        
        return prediction.squeeze()

class MovieLensDataset(Dataset):
    def __init__(self, data):
        self.users = torch.tensor(data['user_id'].values, dtype=torch.long)
        self.items = torch.tensor(data['item_id'].values, dtype=torch.long)
        self.user_features = torch.tensor(data[USER_FEATURES].values, dtype=torch.float)
        self.item_features = torch.tensor(data[ITEM_FEATURES].values, dtype=torch.float)
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
    
    train_dataset = MovieLensDataset(train_df)
    val_dataset = MovieLensDataset(val_df)
    test_dataset = MovieLensDataset(test_df)
    
    return train_dataset, val_dataset, test_dataset, n_users, n_items


def train_model(model, train_loader, val_loader, optimizer, device, num_epochs=20, checkpoint_dir='./checkpoints', patience=3):
    
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    model = model.to(device)
    
    criterion = nn.MSELoss()
    
    # Initialize tracking variables
    best_val_loss = np.inf
    best_epoch = 0
    epochs_no_improve = 0
    
    # History of train/validation losses
    history = {"train_loss": [], "val_loss": []}
    
    # Start training loop
    start_time = time.time()
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        
        for batch_idx, (users, items, user_features, item_features, ratings) in enumerate(train_loader):
            users, items, user_features, item_features, ratings = users.to(device), items.to(device), user_features.to(device), item_features.to(device), ratings.to(device)
            
            
            optimizer.zero_grad()
            
            outputs = model(users, items, user_features, item_features)
            
            loss = criterion(outputs, ratings.float())
            
            loss.backward()
            
            optimizer.step()
            
            train_loss += loss.item()
            
            if (batch_idx + 1) % 100 == 0:
                print(f'Epoch {epoch+1}/{num_epochs}, Batch {batch_idx+1}/{len(train_loader)}, Loss: {loss.item():.4f}')
        
        avg_train_loss = train_loss / len(train_loader)
        history["train_loss"].append(avg_train_loss)
        
        #Validation pass
        model.eval()
        val_loss = 0.0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for users, items, user_features, item_features, ratings in val_loader:
                users, items, user_features, item_features, ratings = users.to(device), items.to(device), user_features.to(device), item_features.to(device), ratings.to(device)
                
                outputs = model(users, items, user_features, item_features)
                
                loss = criterion(outputs, ratings.float())
                
                val_loss += loss.item()
                
                all_preds.extend(outputs.cpu().numpy())
                all_labels.extend(ratings.cpu().numpy())
        
        avg_val_loss = val_loss / len(val_loader)
        history["val_loss"].append(avg_val_loss)
        
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
            }, os.path.join(checkpoint_dir, 'best_model.pt'))
            
            print(f"Saved best model checkpoint with validation loss: {best_val_loss:.4f}")
        else:
            epochs_no_improve += 1
        
        # Early stopping check
        if epochs_no_improve >= patience:
            print(f'Early stopping triggered! No improvement for {patience} epochs.')
            break
    
    training_time = time.time() - start_time
    print(f"Training completed in {training_time:.2f} seconds.")
    print(f"Best validation loss: {best_val_loss:.4f} at epoch {best_epoch+1}")
    
    # Load the best model
    checkpoint = torch.load(os.path.join(checkpoint_dir, 'best_model.pt'))
    model.load_state_dict(checkpoint['model_state_dict'])
    
    return model, history


def evaluate_model(model, test_loader, device):
    model.eval()
    test_loss = 0.0
    all_preds = []
    all_labels = []
    
    
    criterion = nn.MSELoss()
    
    with torch.no_grad():
        for users, items, user_features, item_features, ratings in test_loader:
            users, items, user_features, item_features, ratings = users.to(device), items.to(device), user_features.to(device), item_features.to(device), ratings.to(device)
            
            outputs = model(users, items, user_features, item_features)
            
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
    plt.savefig('/home/adiez/Desktop/Deep Learning/DL - Assignment 2/plots/training_history_reg.png')
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
    
    # Compute accuracy and print it
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
    plt.savefig('/home/adiez/Desktop/Deep Learning/DL - Assignment 2/plots/confusion_matrix_hybrid.png')
    plt.show()

def plot_roc_auc(predictions, actuals, classes=None):
    """
    Plot ROC AUC curve for a multi-class classification problem.

    Args:
        predictions (np.array): Predicted probabilities (shape: [n_samples, n_classes]).
        actuals (np.array): Actual class labels (shape: [n_samples]).
        classes (list): List of class labels.
    """
    n_classes = len(classes)
    
    # Compute ROC curve and AUC for each class
    fpr, tpr, roc_auc = {}, {}, {}
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve((actuals == classes[i]).astype(int), predictions[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    
    # Plot all ROC curves
    plt.figure(figsize=(8, 6))
    colors = cycle(['blue', 'red', 'green', 'purple', 'orange'])  # Adjust for more classes if needed
    for i, color in zip(range(n_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=2,
                 label=f'Class {classes[i]} (AUC = {roc_auc[i]:.2f})')

    # Plot diagonal line (random classifier)
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray', lw=2)

    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC AUC Curve')
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig('/home/adiez/Desktop/Deep Learning/DL - Assignment 2/plots/roc_auc_curve_hybrid.png')
    plt.show()

def run_training_pipeline(ratings_file, users_file, movies_file,model_config=None, train_config=None):
        # Default configurations
    default_model_config = {
        "embedding_dim": 16,
        "mlp_dims": [256, 128, 64],
        "dropout_rate": 0.5,
        "use_batch_norm": True
    }
    
    default_train_config = {
        "batch_size": 64,
        "learning_rate": 0.001,
        "weight_decay": 1e-5,
        "num_epochs": 40,
        "patience": 5,
        "val_size": 0.1,
        "test_size": 0.1
    }
    
    # Update with provided configurations
    if model_config:
        default_model_config.update(model_config)
        pass
    
    if train_config:
        default_train_config.update(train_config)
        pass
    
    #Set manual seed for reproducibility
    torch.manual_seed(42)
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    
    # Prepare datasets
    print("Preparing datasets...")
    train_dataset, val_dataset, test_dataset, n_users, n_items = prepare_datasets(ratings_file, users_file, movies_file)
    
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
    model = HybridNCF(
        num_users=n_users,
        num_items=n_items,
        embedding_dim=default_model_config["embedding_dim"],
        mlp_dims=default_model_config["mlp_dims"],
        dropout_rate=default_model_config["dropout_rate"],
        use_batch_norm=default_model_config["use_batch_norm"],
        num_user_features=len(USER_FEATURES),
        num_item_features=len(ITEM_FEATURES)
    )
    
    optimizer = optim.Adam(
        model.parameters(), 
        lr=default_train_config["learning_rate"], 
        weight_decay=default_train_config["weight_decay"]
    )
    
    # Train the model
    print("Starting training...")
    train_model, history = train_model(
        model, 
        train_loader, 
        val_loader, 
        optimizer,
        device=device,
        num_epochs=default_train_config["num_epochs"],
        patience=default_train_config["patience"]
    )
    
    print("Training completed!")
    print("Evaluatiing model on test set...")
    results = evaluate_model(train_model, test_loader, device)
    
    # Plot training history
    plot_training_history(history)
    
    # Plot confusion matrix
    plot_confusion_matrix(results['predictions'], results['true_labels'], classes=[1, 2, 3, 4, 5])
    
    # Plot ROC AUC curve
    plot_roc_auc(results['predictions'], results['true_labels'], classes=[1, 2, 3, 4, 5])
    
    return train_model, history, results


if __name__ == "__main__":
    ratings_file = ROUTE + "/data.csv"
    users_file = ROUTE + "/user.csv"
    movies_file = ROUTE + "/item.csv"
    
    model_config = {
        "embedding_dim": 16,
        "mlp_dims": [256, 128, 64],
        "dropout_rate": 0.5,
        "use_batch_norm": True
    }
    
    train_config = {
        "batch_size": 64,
        "learning_rate": 0.001,
        "weight_decay": 1e-5,
        "num_epochs": 40,
        "patience": 5,
        "val_size": 0.1,
        "test_size": 0.1
    }
    
    trained_model, history, results = run_training_pipeline(
        ratings_file=ratings_file,
        users_file=users_file,
        movies_file=movies_file,
        model_config=model_config,
        train_config=train_config
    )