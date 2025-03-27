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
from sklearn.preprocessing import LabelEncoder

class HybridNCF(nn.Module):
    def __init__(self, num_users, num_items,
                 embedding_dim=64, mlp_dims=[128, 64, 32, 16],
                 dropout_rate=0.2, use_batch_norm=True,
                 num_user_features=..., user_feature_embedding_dim=...,  # User features
                 num_item_features=..., item_feature_embedding_dim=...):  # Item features
        super(HybridNCF, self).__init__()

        # Collaborative Filtering Embeddings
        self.user_embedding_gmf_cf = nn.Embedding(num_users, embedding_dim)
        self.item_embedding_gmf_cf = nn.Embedding(num_items, embedding_dim)
        self.user_embedding_mlp_cf = nn.Embedding(num_users, embedding_dim)
        self.item_embedding_mlp_cf = nn.Embedding(num_items, embedding_dim)

        # User Feature Embeddings (Example: one categorical feature)
        self.user_feature_embedding = nn.Embedding(num_user_features, user_feature_embedding_dim)
        self.user_fc = nn.Linear(user_feature_embedding_dim, embedding_dim) # Project to embedding_dim

        # Item Feature Embeddings (Example: one categorical feature)
        self.item_feature_embedding = nn.Embedding(num_item_features, item_feature_embedding_dim)
        self.item_fc = nn.Linear(item_feature_embedding_dim, embedding_dim) # Project to embedding_dim

        # MLP tower structure
        self.mlp_layers = nn.ModuleList()
        input_dim = (embedding_dim * 2) + (embedding_dim * 2) # CF user+item + Feature user+item
        for dim in mlp_dims:
            self.mlp_layers.append(nn.Linear(input_dim, dim))
            if use_batch_norm:
                self.mlp_layers.append(nn.BatchNorm1d(dim))
            self.mlp_layers.append(nn.ReLU())
            self.mlp_layers.append(nn.Dropout(dropout_rate))
            input_dim = dim

        # Final output layer
        self.final_layer = nn.Linear(embedding_dim + mlp_dims[-1], 1) # For regression

        self.scale = nn.Parameter(torch.tensor(4.0))
        self.shift = nn.Parameter(torch.tensor(1.0))

        self._init_weights()
        
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Embedding):
                nn.init.xavier_uniform_(m.weight)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)
                

    def forward(self, user_ids, item_ids, user_features, item_features):        
        user_gmf_cf = self.user_embedding_gmf_cf(user_ids)
        item_gmf_cf = self.item_embedding_gmf_cf(item_ids)
        user_mlp_cf = self.user_embedding_mlp_cf(user_ids)
        item_mlp_cf = self.item_embedding_mlp_cf(item_ids)

        gmf_interaction = user_gmf_cf * item_gmf_cf

        # User Features
        user_feature_emb = self.user_feature_embedding(user_features)
        processed_user_features = F.relu(self.user_fc(user_feature_emb))

        # Item Features
        item_feature_emb = self.item_feature_embedding(item_features)
        processed_item_features = F.relu(self.item_fc(item_feature_emb))

        # MLP Input (Concatenate CF and Feature embeddings)
        mlp_input = torch.cat([user_mlp_cf, item_mlp_cf, processed_user_features, processed_item_features], dim=-1)

        x = mlp_input
        for layer in self.mlp_layers:
            x = layer(x)
        mlp_output = x

        # Combine GMF and MLP outputs
        neumf_output = torch.cat([gmf_interaction, mlp_output], dim=1)

        raw_output = self.final_layer(neumf_output)
        scaled_output = torch.sigmoid(raw_output) * self.scale + self.shift
        return scaled_output.squeeze()

def prepare_datasets(ratings_file, users_file, movies_file, val_size=0.1, test_size=0.1, random_state=42):
    """
    Prepare train, validation, and test datasets for the HybridNCF model.
    Assumes user and item features are already encoded in their respective files.
    
    Args:
        ratings_file (str): Path to ratings CSV file
        users_file (str): Path to users CSV file with encoded features
        movies_file (str): Path to movies CSV file with encoded features
        val_size (float): Proportion of data for validation
        test_size (float): Proportion of data for test
        random_state (int): Random seed for reproducibility
        
    Returns:
        train_dataset, val_dataset, test_dataset, n_users, n_items
    """
    # Read data files
    ratings_df = pd.read_csv(ratings_file)
    users_df = pd.read_csv(users_file)
    movies_df = pd.read_csv(movies_file)
    
    data = ratings_df.merge(users_df, on='user_id', how='left')
    data = data.merge(movies_df, on='item_id', how='left')

    user_encoder = LabelEncoder()
    item_encoder = LabelEncoder()
    
    # Fit and transform the user_id and item_id columns
    ratings_df['user_id'] = user_encoder.fit_transform(ratings_df['user_id'])
    ratings_df['item_id'] = item_encoder.fit_transform(ratings_df['item_id'])
    
    # Get number of users and items (after encoding)
    n_users = len(user_encoder.classes_)
    n_items = len(item_encoder.classes_)
    
    # Split data into train, validation, and test sets
    train_val_df, test_df = train_test_split(data, test_size=test_size, random_state=random_state) #todo: Add stratify
    
    train_df, val_df = train_test_split(train_val_df, test_size=val_size, random_state=random_state) #todo: Add stratify
    
    train_df = train_df.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)
    
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
        
    
    train_dataset = MovieLensDataset(train_df)
    val_dataset = MovieLensDataset(val_df)
    test_dataset = MovieLensDataset(test_df)
    
    return train_dataset, val_dataset, test_dataset, n_users, n_items
