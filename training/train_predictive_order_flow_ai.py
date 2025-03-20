# train_predictive_order_flow_ai.py


import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from ml_models.predictive_order_flow_ai import PredictiveOrderFlowAI

class PredictiveOrderFlowTrainer:
    """Trains AI model for predicting order flow & market moves"""

    def __init__(self, model_path="models/order_flow_predictor.pth"):
        self.model = PredictiveOrderFlowAI(input_size=10)
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.criterion = nn.MSELoss()
        self.model_path = model_path

    def train_model(self, data):
        """Trains model using past Level 3 order book data & price moves"""
        market_features = torch.tensor(data[:, :-1], dtype=torch.float32)
        price_movements = torch.tensor(data[:, -1], dtype=torch.float32).unsqueeze(1)  # -1 (down), 0 (neutral), 1 (up)

        for epoch in range(200):
            self.optimizer.zero_grad()
            predictions = self.model(market_features)
            loss = self.criterion(predictions, price_movements)
            loss.backward()
            self.optimizer.step()

            if epoch % 20 == 0:
                print(f"Epoch {epoch}, Loss: {loss.item()}")

        # Save trained model
        torch.save(self.model.state_dict(), self.model_path)
        print("Predictive Order Flow AI Model Trained & Saved.")

# Load training data (replace with real dataset)
def load_training_data():
    df = pd.read_csv("data/historical_order_flow.csv")
    return df.to_numpy()

if __name__ == "__main__":
    data = load_training_data()
    trainer = PredictiveOrderFlowTrainer()
    trainer.train_model(data)

