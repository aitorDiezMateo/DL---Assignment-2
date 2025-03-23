
#! Neural Collaborative Filtering (NCF) Model

import torch
from torch.utils.data import Dataset
import pandas as pd

class NCFDataset(Dataset):
    def __init__(self, ratings_file, n_users, n_items):
        self.ratings = pd.read_csv(ratings_file)
        self.users = torch.LongTensor(self.ratings['user_id'].values)
        self.items = torch.LongTensor(self.ratings['item_id'].values)

        self.ratings_tensor = torch.IntTensor(self.ratings['rating'].values)
        
        self.num_users = n_users
        self.num_items = n_items
        
    def __len__(self):
        return len(self.ratings)
    
    def __getitem__(self, idx):
        return self.users[idx], self.items[idx], self.ratings_tensor[idx]


# Define the model
import torch.nn as nn
import torch.nn.functional as F

class NCF(nn.Module):
    def __init__(self, num_users, num_items, embedding_dim, hidden_dim):
        super(NCF, self).__init__()
        
        # GMF part: Generalized Matrix Factorization
        self.user_embedding_gmf = nn.Embedding(num_users, embedding_dim)
        self.item_embedding_gmf = nn.Embedding(num_items, embedding_dim)
        
        # MLP part: Multilayer Perceptron
        self.user_embedding_mlp = nn.Embedding(num_users, embedding_dim)
        self.item_embedding_mlp = nn.Embedding(num_items, embedding_dim)
        
        # MLP hidden layers
        self.mlp_layer1 = nn.Linear(embedding_dim * 2, hidden_dim)
        self.mlp_layer2 = nn.Linear(hidden_dim, hidden_dim)
        self.mlp_output = nn.Linear(hidden_dim, 1)
        
        # GMF part: Element-wise multiplication for interactions
        self.gmf_output = nn.Linear(embedding_dim, 1)

        # Activation functions
        self.relu = nn.ReLU()

    def forward(self, user, item):
        # GMF part
        user_gmf = self.user_embedding_gmf(user)
        item_gmf = self.item_embedding_gmf(item)
        gmf_interaction = user_gmf * item_gmf  # Element-wise multiplication
        
        # MLP part
        user_mlp = self.user_embedding_mlp(user)
        item_mlp = self.item_embedding_mlp(item)
        mlp_input = torch.cat([user_mlp, item_mlp], dim=-1)  # Concatenation
        mlp_output = self.relu(self.mlp_layer1(mlp_input))
        mlp_output = self.relu(self.mlp_layer2(mlp_output))
        mlp_output = self.mlp_output(mlp_output)
        
        # Combine GMF and MLP outputs
        gmf_output = self.gmf_output(gmf_interaction)
        combined_output = gmf_output + mlp_output
        
        return combined_output.squeeze()  # Return scalar prediction