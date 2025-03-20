# trade_timing_ai.py


import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

class TradeTimingAI(nn.Module):
    """Deep learning model for predicting the best trade entry timing."""
    
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
        return self.sigmoid(self.fc3(x))  # Output probability (0-1, where 1 = ideal entry)

class AITradeTimer:
    """Manages AI-based trade timing with trend and volatility adjustments."""
    
    def __init__(self, model_path="models/trade_timing_model.pth"):
        self.model = TradeTimingAI(input_size=12)  # Expanded feature set
        self.model.load_state_dict(torch.load(model_path))
        self.model.eval()
    
    def predict_trade_timing(self, market_data):
        """Predicts if current conditions are ideal for trade entry."""
        features = np.array(market_data).reshape(1, -1)
        input_tensor = torch.tensor(features, dtype=torch.float32)
        
        with torch.no_grad():
            probability = self.model(input_tensor).item()
        
        return probability  # Higher probability means better entry timing
    
    def evaluate_volatility_adjusted_entry(self, market_data, volatility, price_momentum):
        """Incorporates volatility and momentum into entry decision."""
        entry_score = self.predict_trade_timing(market_data)
        volatility_adjustment = np.clip(volatility * price_momentum, -0.2, 0.2)  # Adjust based on volatility impact
        return np.clip(entry_score + volatility_adjustment, 0, 1)

