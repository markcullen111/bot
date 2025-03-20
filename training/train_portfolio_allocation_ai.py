# train_portfolio_allocation_ai.py


import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from portfolio_allocation_ai import PortfolioAllocationAI

class PortfolioAllocationTrainer:
    """Trains AI model for portfolio allocation with enhanced risk management."""
    
    def __init__(self, model_path="models/portfolio_allocation_model.pth", asset_count=5):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = PortfolioAllocationAI(input_size=10, asset_count=asset_count).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.criterion = nn.MSELoss()
        self.model_path = model_path
    
    def train_model(self, data, epochs=200, batch_size=32):
        """Trains model using past market data & risk-adjusted optimal allocations."""
        features = torch.FloatTensor(data[:, :-asset_count]).to(self.device)
        allocations = torch.FloatTensor(data[:, -asset_count:]).to(self.device)
        
        dataset = torch.utils.data.TensorDataset(features, allocations)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        for epoch in range(epochs):
            total_loss = 0
            for market_features, allocation_targets in dataloader:
                self.optimizer.zero_grad()
                predictions = self.model(market_features)
                loss = self.criterion(predictions, allocation_targets)
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()
            
            if epoch % 20 == 0:
                print(f"Epoch {epoch}, Loss: {total_loss / len(dataloader):.6f}")
        
        torch.save(self.model.state_dict(), self.model_path)
        print("Portfolio Allocation AI Model Trained & Saved.")
    
    def load_training_data(self, file_path="data/historical_portfolio_allocations.csv"):
        """Loads and preprocesses training data."""
        df = pd.read_csv(file_path).dropna()
        return df.to_numpy()

if __name__ == "__main__":
    trainer = PortfolioAllocationTrainer(asset_count=5)
    data = trainer.load_training_data()
    trainer.train_model(data)

