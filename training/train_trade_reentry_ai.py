# train_trade_reentry_ai.py


import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from ml_models.trade_reentry_ai import TradeReEntryAI

class TradeReEntryTrainer:
    """Trains AI model for trade re-entry timing"""

    def __init__(self, model_path="models/trade_reentry_model.pth"):
        self.model = TradeReEntryAI(input_size=10)  # 10 market features
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.criterion = nn.BCELoss()  # Binary classification loss
        self.model_path = model_path

    def train_model(self, data):
        """Trains model using past market data & successful re-entry timings"""
        market_features = torch.tensor(data[:, :-1], dtype=torch.float32)
        reentry_labels = torch.tensor(data[:, -1], dtype=torch.float32).unsqueeze(1)  # 0 = bad re-entry, 1 = good re-entry

        for epoch in range(100):
            self.optimizer.zero_grad()
            predictions = self.model(market_features)
            loss = self.criterion(predictions, reentry_labels)
            loss.backward()
            self.optimizer.step()

            if epoch % 10 == 0:
                print(f"Epoch {epoch}, Loss: {loss.item()}")

        # Save trained model
        torch.save(self.model.state_dict(), self.model_path)
        print("Trade Re-Entry AI Model Trained & Saved.")

# Load training data (replace with real historical dataset)
def load_training_data():
    df = pd.read_csv("data/historical_trade_reentries.csv")
    return df.to_numpy()

if __name__ == "__main__":
    data = load_training_data()
    trainer = TradeReEntryTrainer()
    trainer.train_model(data)

