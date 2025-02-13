import torch
from model import HousePriceModel

# Load trained model
model = HousePriceModel(input_size=5)  # Updated for 5 selected features
model.load_state_dict(torch.load("house_price_model.pth"))
model.eval()

# Example house: [area, bedrooms, bathrooms, stories, parking]
sample_house = torch.tensor([[3000, 4, 3, 2, 2]], dtype=torch.float32)  # Example input
predicted_price = model(sample_house).item()

print(f"🏡 Predicted House Price: ${predicted_price:.2f}")
