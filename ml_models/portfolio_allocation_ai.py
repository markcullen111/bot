# portfolio_allocation_ai.py


import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

class PortfolioAllocationAI(nn.Module):
    """Deep learning model for risk-adjusted portfolio allocation"""
    def __init__(self, input_size, asset_count):
        super().__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, asset_count)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)  # Ensures allocations sum to 1
    
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        return self.softmax(self.fc3(x))

class AIPortfolioAllocator:
    """Manages AI-based portfolio allocation with risk-adjusted optimization"""
    def __init__(self, model_path="models/portfolio_allocation_model.pth", asset_count=5):
        self.model = PortfolioAllocationAI(input_size=10, asset_count=asset_count)
        self.model.load_state_dict(torch.load(model_path))
        self.model.eval()
    
    def allocate_funds(self, market_data):
        """Predicts optimal fund allocation across multiple assets based on risk-adjusted returns"""
        features = np.array(market_data).reshape(1, -1)
        input_tensor = torch.tensor(features, dtype=torch.float32)
        
        with torch.no_grad():
            allocation = self.model(input_tensor).numpy()[0]
        
        return allocation  # Returns percentage allocation per asset
    
    def optimize_for_sharpe_ratio(self, asset_returns, risk_free_rate=0.01):
        """Adjusts portfolio weights to maximize Sharpe ratio"""
        mean_returns = np.mean(asset_returns, axis=0)
        std_dev = np.std(asset_returns, axis=0) + 1e-9  # Prevent division by zero
        sharpe_ratios = (mean_returns - risk_free_rate) / std_dev
        optimal_weights = sharpe_ratios / np.sum(sharpe_ratios)
        return optimal_weights

