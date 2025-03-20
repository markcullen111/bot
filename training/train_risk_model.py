# train_risk_model.py


import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from trading.ai_risk_manager import RiskManagementAI

class RiskModelTrainer:
    """Trains the AI risk management model"""

    def __init__(self, model_path="models/risk_model.pth"):
        self.model = RiskManagementAI(input_size=8)  # 8 risk features
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.criterion = nn.MSELoss()
        self.model_path = model_path

    def train_model(self, data):
        """Trains model using historical risk factors & pnl outcomes"""
        market_conditions = torch.tensor(data[:, :-1], dtype=torch.float32)
        risk_targets = torch.tensor(data[:, -1], dtype=torch.float32).unsqueeze(1)

        for epoch in range(100):
            self.optimizer.zero_grad()
            predictions = self.model(market_conditions)
            loss = self.criterion(predictions, risk_targets)
            loss.backward()
            self.optimizer.step()

            if epoch % 10 == 0:
                print(f"Epoch {epoch}, Loss: {loss.item()}")

        # Save the trained model
        torch.save(self.model.state_dict(), self.model_path)
        print("AI Risk Model Trained & Saved.")

# Load historical data (replace with real data source)
def load_training_data():
    df = pd.read_csv("data/historical_risk_data.csv")
    return df.to_numpy()

if __name__ == "__main__":
    data = load_training_data()
    trainer = RiskModelTrainer()
    trainer.train_model(data)

