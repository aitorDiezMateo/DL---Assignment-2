import torch
from torch.utils.data import Dataset
import pandas as pd

class UserDataset(Dataset):
    def __init__(self, csv_file):
        # Read the CSV file
        self.user_data = pd.read_csv(csv_file)
        
        # Assuming the CSV has a 'user_id' column, convert it to a tensor
        self.user_ids = torch.tensor(self.user_data['user_id'].values, dtype=torch.long)
        
        self.user_features = torch.tensor(self.user_data.drop(columns='user_id').values, dtype=torch.float32)
        
    def __len__(self):
        return len(self.user_data)
    
    def __getitem__(self, idx):
        # Return the user_id and the user features for a given index
        return self.user_ids[idx], self.user_features[idx]

class ItemDataset(Dataset):
    def __init__(self, csv_file):
        # Read the CSV file
        self.item_data = pd.read_csv(csv_file)
        
        # Assuming the CSV has an 'item_id' column, convert it to a tensor
        self.item_ids = torch.tensor(self.item_data['item_id'].values, dtype=torch.long)
        
        # If you have other item-specific features, you can convert them here
        # For example, if genres are stored as one-hot encoded columns, drop 'item_id' and keep features
        self.item_features = torch.tensor(self.item_data.drop(columns='item_id').values, dtype=torch.float32)
        
    def __len__(self):
        return len(self.item_data)
    
    def __getitem__(self, idx):
        # Return the item_id and the item features for a given index
        return self.item_ids[idx], self.item_features[idx]

class RatingDataset(Dataset):
    def __init__(self, csv_file):
        # Read the CSV file
        self.rating_data = pd.read_csv(csv_file)
        
        # Convert 'user_id', 'item_id' and 'rating' columns to tensors
        self.user_ids = torch.tensor(self.rating_data['user_id'].values, dtype=torch.long)
        self.item_ids = torch.tensor(self.rating_data['item_id'].values, dtype=torch.long)
        self.ratings = torch.tensor(self.rating_data['rating'].values, dtype=torch.float32)
        
    def __len__(self):
        return len(self.rating_data)
    
    def __getitem__(self, idx):
        # Return user_id, item_id, and the rating for a given index
        return self.user_ids[idx], self.item_ids[idx], self.ratings[idx]
