# House Price Prediction using PyTorch

This project uses a neural network built with PyTorch to predict house prices based on features such as area, number of bedrooms, bathrooms, stories, and parking spaces.

## Files

- `data_loader.py`: Loads and preprocesses the data.
- `model.py`: Defines the neural network model.
- `train.py`: Trains the model on the dataset.
- `evaluate.py`: Evaluates the trained model on the test data.
- `predict.py`: Makes predictions for new input data using the trained model.
- `Housing.csv`: Dataset containing house features and prices.
- `house_price_model.pth`: Trained model (saved after running `train.py`).

## Requirements

- `torch`
- `scikit-learn`
- `pandas`
- `numpy`

You can install the dependencies using pip:

```bash
pip install torch scikit-learn pandas numpy
