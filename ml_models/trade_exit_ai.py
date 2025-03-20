# trade_exit_ai.py


import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

class TradeExitAI(nn.Module):
    """Deep learning model for predicting optimal trade exit timing."""
    
    def __init__(self, input_size):
        super().__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        return self.sigmoid(self.fc3(x))  # Output probability (0-1, where 1 = ideal exit)

class AITradeExit:
    """Manages AI-based trade exits with improved risk-awareness."""
    
    def __init__(self, model_path="models/trade_exit_model.pth"):
        self.model = TradeExitAI(input_size=12)  # Additional features for better decision-making
        self.model.load_state_dict(torch.load(model_path))
        self.model.eval()
    
    def predict_exit_timing(self, market_data):
        """Predicts if current conditions are ideal for trade exit."""
        features = np.array(market_data).reshape(1, -1)
        input_tensor = torch.tensor(features, dtype=torch.float32)
        
        with torch.no_grad():
            probability = self.model(input_tensor).item()
        
        return probability  # Higher probability means better exit timing
    
    def evaluate_risk_adjusted_exit(self, market_data, volatility, trade_pnl):
        """Incorporates volatility and PnL into exit decision."""
        exit_score = self.predict_exit_timing(market_data)
        risk_adjustment = np.clip(volatility * trade_pnl, -0.2, 0.2)  # Adjust based on volatility impact
        return np.clip(exit_score + risk_adjustment, 0, 1)

