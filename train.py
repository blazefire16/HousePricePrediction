import torch
import torch.optim as optim
import torch.nn as nn
from data_loader import get_dataloaders
from model import HousePriceModel

# Define dataset parameters
CSV_FILE = "Housing.csv"
FEATURES = ["area", "bedrooms", "bathrooms", "stories", "parking"]
TARGET = "price"

# Load data
train_loader, test_loader = get_dataloaders(CSV_FILE)

# Initialize model
model = HousePriceModel(len(FEATURES))  # ✅ Fix: Pass input feature count
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# Train model
epochs = 50
for epoch in range(epochs):
    for X_batch, y_batch in train_loader:
        optimizer.zero_grad()
        predictions = model(X_batch)
        loss = criterion(predictions, y_batch)
        loss.backward()
        optimizer.step()
    if epoch % 10 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item():.4f}")

# Save trained model
torch.save(model.state_dict(), "house_price_model.pth")
print("Model saved!")
