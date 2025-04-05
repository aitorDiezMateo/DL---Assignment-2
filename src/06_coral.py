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
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, precision_score, recall_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from imblearn.over_sampling import RandomOverSampler
import time
import os
import seaborn as sns
from sklearn.metrics import roc_curve, auc
from itertools import cycle
from coral_pytorch.layers import CoralLayer
from coral_pytorch.losses import coral_loss
from coral_pytorch.dataset import proba_to_label
from coral_pytorch.dataset import levels_from_labelbatch

torch.manual_seed(42)

ROUTE = "/home/adiez/Desktop/Deep Learning/DL - Assignment 2/data/100k/processed"

USER_FEATURES = ['age', 'gender_F','gender_M', 'occupation_administrator', 'occupation_artist',       'occupation_doctor', 'occupation_educator', 'occupation_engineer', 'occupation_entertainment', 'occupation_executive', 'occupation_healthcare', 'occupation_homemaker','occupation_lawyer','occupation_librarian', 'occupation_marketing', 'occupation_none','occupation_other', 'occupation_programmer', 'occupation_retired','occupation_salesman', 'occupation_scientist', 'occupation_student',
'occupation_technician', 'occupation_writer', 'release_date']

ITEM_FEATURES = ['unknown','Action', 'Adventure', 'Animation', 'Children', 'Comedy', 'Crime','Documentary', 'Drama', 'Fantasy', 'Film-Noir', 'Horror', 'Musical','Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western']

class HybridNCF(nn.Module):
    def __init__(self, num_users, num_items,
                embedding_dim=32,  # Reduced from 64
                mlp_dims=[128, 64], # Simplified architecture
                dropout_rate=0.5,  # Increased from 0.2
                l2_regularization=1e-4,  # Added L2 regularization
                use_batch_norm=True,
                num_user_features=25,
                num_item_features=19,
                num_classes=5): # Added num_classes
        super(HybridNCF, self).__init__()
        
        # Add L2 regularization to embeddings
        self.user_embedding_gmf_cf = nn.Embedding(num_users, embedding_dim, max_norm=1.0)
        self.item_embedding_gmf_cf = nn.Embedding(num_items, embedding_dim, max_norm=1.0)
        self.user_embedding_mlp_cf = nn.Embedding(num_users, embedding_dim, max_norm=1.0)
        self.item_embedding_mlp_cf = nn.Embedding(num_items, embedding_dim, max_norm=1.0)
        
        # MLP tower structure
        self.mlp_layers = nn.ModuleList()
        input_dim = (embedding_dim * 2) + num_user_features + num_item_features
        for dim in mlp_dims:
            self.mlp_layers.append(nn.Linear(input_dim, dim))
            if use_batch_norm:
                self.mlp_layers.append(nn.BatchNorm1d(dim))
            self.mlp_layers.append(nn.ReLU())
            self.mlp_layers.append(nn.Dropout(dropout_rate))
            input_dim = dim
            
        # Final prediction layers
        self.gmf_output = nn.Linear(embedding_dim, mlp_dims[-1]) # Output features from GMF
        self.mlp_output = nn.Linear(mlp_dims[-1], mlp_dims[-1]) # Output features from MLP
        
        # Combine features before final classification layer
        combined_dim = mlp_dims[-1] * 2 
        self.final_output = CoralLayer(combined_dim, num_classes)
        
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
        # GMF path
        user_embed_gmf = self.user_embedding_gmf_cf(user_indices)
        item_embed_gmf = self.item_embedding_gmf_cf(item_indices)
        gmf_vector = user_embed_gmf * item_embed_gmf
        gmf_features = F.relu(self.gmf_output(gmf_vector)) # Get features from GMF
        
        # MLP path
        user_embed_mlp = self.user_embedding_mlp_cf(user_indices)
        item_embed_mlp = self.item_embedding_mlp_cf(item_indices)
        mlp_input_vector = torch.cat([
            user_embed_mlp,
            item_embed_mlp,
            user_features,
            item_features
        ], dim=1)
        mlp_processed_vector = mlp_input_vector
        for layer in self.mlp_layers:
             mlp_processed_vector = layer(mlp_processed_vector)
        mlp_features = F.relu(self.mlp_output(mlp_processed_vector)) # Get features from MLP
        
        # Combine features from GMF and MLP paths
        combined_features = torch.cat([gmf_features, mlp_features], dim=1)
        
        # Final classification layer
        logits = self.final_output(combined_features) 
        # No softmax needed here, CrossEntropyLoss applies it internally
        
        return logits # Return logits for classification loss

class MovieLensDataset(Dataset):
    def __init__(self, data):
        data.fillna(0, inplace=True)
        
        self.users = torch.tensor(data['user_id'].values, dtype=torch.long)
        self.items = torch.tensor(data['item_id'].values, dtype=torch.long)
        self.user_features = torch.tensor(data[USER_FEATURES].astype(float).values, dtype=torch.float32)
        self.item_features = torch.tensor(data[ITEM_FEATURES].astype(float).values, dtype=torch.float32)
        self.ratings = torch.tensor(data['rating'].values - 1, dtype=torch.long)
    
    def __len__(self):
        return len(self.users)
    
    def __getitem__(self, idx):
        return self.users[idx], self.items[idx], self.user_features[idx], self.item_features[idx], self.ratings[idx]

ratings_df = pd.read_csv(ROUTE + "/data.csv")
users_df = pd.read_csv(ROUTE + "/user.csv")
movies_df = pd.read_csv(ROUTE + "/item.csv")

data = ratings_df.merge(users_df, on='user_id', how='left')
data = data.merge(movies_df, on='item_id', how='left')

user_encoder = LabelEncoder()
item_encoder = LabelEncoder()

data["user_id"] = user_encoder.fit_transform(data["user_id"])
data["item_id"] = item_encoder.fit_transform(data["item_id"])

n_users = len(user_encoder.classes_)
n_items = len(item_encoder.classes_)

train_val_df, test_df = train_test_split(data, test_size=0.1, random_state=42)
train_df, val_df = train_test_split(train_val_df, test_size=0.1, random_state=24)

train_df = train_df.reset_index(drop=True)
val_df = val_df.reset_index(drop=True)
test_df = test_df.reset_index(drop=True)

train_dataset = MovieLensDataset(train_df)
val_dataset = MovieLensDataset(val_df)
test_dataset = MovieLensDataset(test_df)

default_model_config = {
    "embedding_dim": 16,
    "mlp_dims": [256, 128, 64],
    "dropout_rate": 0.5,
    "use_batch_norm": True
}

default_train_config = {
    "batch_size": 128,  # Larger batch size for better gradient estimates
    "learning_rate": 0.0005,  # Lower learning rate to prevent overfitting
    "weight_decay": 1e-4,  # Increased weight decay
    "num_epochs": 50,
    "patience": 8,  # More patience for early stopping
    "val_size": 0.15,  # Larger validation set
    "test_size": 0.1
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

train_loader = DataLoader(train_dataset, batch_size=default_train_config["batch_size"], shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=default_train_config["batch_size"], shuffle=False, num_workers=4)
test_loader = DataLoader(test_dataset, batch_size=default_train_config["batch_size"], shuffle=False, num_workers=4)

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
    num_item_features=len(ITEM_FEATURES),
    num_classes=5
)

optimizer = optim.Adam(
    model.parameters(), 
    lr=default_train_config["learning_rate"], 
    weight_decay=default_train_config["weight_decay"]
)

# Train the model
print("Starting training...")


# Evaluate model on test set
# Add learning rate scheduler

checkpoint_dir = "./checkpoints"
num_epochs = default_train_config["num_epochs"]
patience = default_train_config["patience"]

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, 
    mode='min', 
    factor=0.5, 
    patience=2
)

os.makedirs(checkpoint_dir, exist_ok=True)

model = model.to(device)
# For Coral, we don't need class weights in the same way

# Replace CrossEntropyLoss with the coral_loss function
criterion = coral_loss


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
        
        predicted_labels = proba_to_label(outputs)

        # print(predicted_labels)
        levels = levels_from_labelbatch(ratings, num_classes=5)
        
        # Coral loss expects continuous labels, not class indices
        loss = criterion(outputs, levels.to(device))
        
        loss.backward()
        
        # Change
        # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
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
            
            levels = levels_from_labelbatch(ratings, num_classes=5)
            
            loss = criterion(outputs, levels.to(device))
            
            val_loss += loss.item()
            
            # Get predicted class (highest logit) for accuracy/confusion matrix
            _, predicted_classes = torch.max(outputs, 1)
            all_preds.extend(predicted_classes.cpu().numpy())
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
checkpoint = torch.load(os.path.join(checkpoint_dir, 'best_model.pt'))
model.load_state_dict(checkpoint['model_state_dict'])

# Evaluate the model on the test set

model.eval()
test_loss = 0.0
all_preds = []  # Store predicted class indices (argmax)
all_labels = [] # Store true labels
all_probs = []  # Store predicted probabilities for each class

criterion = coral_loss

with torch.no_grad():
    for users, items, user_features, item_features, ratings in test_loader:
        users, items, user_features, item_features, ratings = users.to(device), items.to(device), user_features.to(device), item_features.to(device), ratings.to(device)

        outputs = model(users, items, user_features, item_features) # These are coral logits

        
        levels = levels_from_labelbatch(ratings, num_classes=5)
        
        loss = criterion(outputs, levels.to(device))
        test_loss += loss.item()

        # For Coral, we need to convert logits to class probabilities differently
        probas = torch.sigmoid(outputs)
        all_probs.extend(probas.cpu().numpy())

        # Get predicted class index (levels start at 0)
        predicted_classes = proba_to_label(probas)
        all_preds.extend(predicted_classes.cpu().numpy())
        all_labels.extend(ratings.cpu().numpy())

avg_test_loss = test_loss / len(test_loader)
print(f'Test Loss (Coral): {avg_test_loss:.4f}')

# Convert to numpy arrays
all_preds = np.array(all_preds)
all_labels = np.array(all_labels)

# Calculate accuracy
correct = sum(1 for pred, true in zip(all_preds, all_labels) if pred == true)
total = len(all_preds)
accuracy = correct / total
print(f'Test Accuracy: {accuracy:.4f}')

# Calculate classification metrics
precision = precision_score(all_labels, all_preds, average='macro')
recall = recall_score(all_labels, all_preds, average='macro')
f1 = f1_score(all_labels, all_preds, average='macro')

# Calculate regression metrics - convert 0-4 scale back to 1-5 for better interpretability
all_preds_reg = all_preds + 1
all_labels_reg = all_labels + 1
mae = mean_absolute_error(all_labels_reg, all_preds_reg)
rmse = np.sqrt(mean_squared_error(all_labels_reg, all_preds_reg))

# Print all metrics
print(f'Precision: {precision:.4f}')
print(f'Recall: {recall:.4f}')
print(f'F1-Score: {f1:.4f}')
print(f'MAE: {mae:.4f}')
print(f'RMSE: {rmse:.4f}')

#Confusion matrix
cm = confusion_matrix(all_labels, all_preds)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=[1,2,3,4,5], yticklabels=[1,2,3,4,5])
plt.xlabel('Predicted Ratings')
plt.ylabel('Actual Ratings')
plt.title('Confusion Matrix')
plt.tight_layout()
plt.savefig('/home/adiez/Desktop/Deep Learning/DL - Assignment 2/plots/confusion_matrix_coral.png')
plt.show()

# Plot metrics in bar charts
plt.figure(figsize=(10, 6))
metric_names = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
metric_values = [accuracy, precision, recall, f1]

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
plt.savefig('/home/adiez/Desktop/Deep Learning/DL - Assignment 2/plots/classification_metrics_coral.png')
plt.show()

# Plot regression metrics
plt.figure(figsize=(8, 5))
regression_metrics = ['MAE', 'RMSE']
regression_values = [mae, rmse]

bars = plt.bar(regression_metrics, regression_values, color=['purple', 'teal'])

# Add values on top of bars
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
             f'{height:.4f}', ha='center', va='bottom')

plt.ylabel('Error')
plt.title('Regression Metrics')
plt.tight_layout()
plt.savefig('/home/adiez/Desktop/Deep Learning/DL - Assignment 2/plots/regression_metrics_coral.png')
plt.show()
