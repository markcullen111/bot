# _risk_manager.py


import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

class RiskManagementAI(nn.Module):
    """Neural network to optimize risk & position sizing"""
    
    def __init__(self, input_size):
        super(RiskManagementAI, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)  # Output: Position size multiplier (0.1 - 1.0)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        return self.sigmoid(self.fc3(x))  # Output between 0-1 (risk adjustment factor)

class AIRiskManager:
    """Manages dynamic position sizing & leverage based on AI predictions"""

    def __init__(self, model_path="models/risk_model.pth"):
        self.model = None
        self.scaler = None
        self.model_path = model_path
        self._load_model()

    def _load_model(self):
        """Loads trained AI model for risk management"""
        try:
            self.model = RiskManagementAI(input_size=8)  # 8 market risk factors
            self.model.load_state_dict(torch.load(self.model_path))
            self.model.eval()
        except Exception as e:
            print(f"Error loading AI risk model: {e}")

    def calculate_risk_adjustment(self, market_conditions):
        """Predicts optimal position size multiplier (0.1 - 1.0)"""
        features = np.array([
            market_conditions['volatility'],
            market_conditions['liquidity'],
            market_conditions['order_flow_imbalance'],
            market_conditions['whale_activity'],
            market_conditions['funding_rate'],
            market_conditions['open_interest_change'],
            market_conditions['market_trend_strength'],
            market_conditions['historical_pnl_volatility']
        ])
        
        features = torch.tensor(features, dtype=torch.float32).unsqueeze(0)
        
        with torch.no_grad():
            risk_multiplier = self.model(features).item()  # Output is between 0.1 - 1.0

        return max(0.1, min(1.0, risk_multiplier))  # Clamp between 10% - 100% exposure

    def adjust_position_size(self, base_size, market_conditions):
        """Dynamically adjust position size based on risk factors"""
        risk_multiplier = self.calculate_risk_adjustment(market_conditions)
        adjusted_size = base_size * risk_multiplier
        return round(adjusted_size, 4)  # Keep position size precise

