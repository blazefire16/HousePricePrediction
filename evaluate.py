import torch
from data_loader import get_dataloaders
from model import HousePriceModel
import torch.nn as nn

# Define constants
CSV_FILE = "Housing.csv"
FEATURES = ["area", "bedrooms", "bathrooms", "stories", "parking"]
TARGET = "price"

# Define the split ratio (80% train, 20% test)
SPLIT_RATIO = 0.8  # 80% for training, 20% for testing

# Get data loaders (train and test)
train_loader, test_loader = get_dataloaders(CSV_FILE, FEATURES, TARGET, split=SPLIT_RATIO)

# Load model
model = HousePriceModel(len(FEATURES))  # Initialize model with the number of features
model.load_state_dict(torch.load("house_price_model.pth"))  # Load saved model weights
model.eval()  # Set model to evaluation mode

# Define the loss function
criterion = nn.MSELoss()

# Evaluation function
def evaluate(model, test_loader):
    total_loss = 0
    with torch.no_grad():  # No need to track gradients during evaluation
        for X_batch, y_batch in test_loader:
            predictions = model(X_batch)  # Get model predictions
            loss = criterion(predictions, y_batch)  # Compute loss
            total_loss += loss.item()  # Accumulate the loss
    return total_loss / len(test_loader)  # Average loss over the dataset

# Run evaluation
test_loss = evaluate(model, test_loader)
print(f"Test Loss: {test_loss:.4f}")

