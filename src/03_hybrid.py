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