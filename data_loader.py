import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

class HousePriceDataset(Dataset):
    def __init__(self, csv_file):
        df = pd.read_csv(csv_file)

        # Select relevant columns
        self.features = ["area", "bedrooms", "bathrooms", "stories", "parking"]
        self.target = "price"

        # Convert data to PyTorch tensors
        self.X = torch.tensor(df[self.features].values, dtype=torch.float32)
        self.y = torch.tensor(df[self.target].values.reshape(-1, 1), dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

def get_dataloaders(csv_file, batch_size=32, split=0.8):
    dataset = HousePriceDataset(csv_file)
    train_size = int(split * len(dataset))
    
    # Split dataset into training and testing
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, len(dataset) - train_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader

