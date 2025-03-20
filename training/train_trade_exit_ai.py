# train_trade_exit_ai.py


import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from ml_models.trade_exit_ai import TradeExitAI

class TradeExitTrainer:
    """Trains AI model for trade exit timing"""

    def __init__(self, model_path="models/trade_exit_model.pth"):
        self.model = TradeExitAI(input_size=10)  # 10 market features
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.criterion = nn.BCELoss()  # Binary classification loss
        self.model_path = model_path

    def train_model(self, data):
        """Trains model using past market data & successful exit timings"""
        market_features = torch.tensor(data[:, :-1], dtype=torch.float32)
        exit_labels = torch.tensor(data[:, -1], dtype=torch.float32).unsqueeze(1)  # 0 = bad exit, 1 = good exit

        for epoch in range(100):
            self.optimizer.zero_grad()
            predictions = self.model(market_features)
            loss = self.criterion(predictions, exit_labels)
            loss.backward()
            self.optimizer.step()

            if epoch % 10 == 0:
                print(f"Epoch {epoch}, Loss: {loss.item()}")

        # Save trained model
        torch.save(self.model.state_dict(), self.model_path)
        print("Trade Exit AI Model Trained & Saved.")

# Load training data (replace with real historical dataset)
def load_training_data():
    df = pd.read_csv("data/historical_trade_exits.csv")
    return df.to_numpy()

if __name__ == "__main__":
    data = load_training_data()
    trainer = TradeExitTrainer()
    trainer.train_model(data)

