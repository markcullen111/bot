# dynamic_risk.py

import numpy as np

class DynamicRiskManager:
    """Dynamic risk manager that adjusts risk exposure based on market conditions."""

    def __init__(self, initial_risk=0.02, volatility_threshold=0.05, drawdown_limit=0.15):
        """
        Args:
            initial_risk (float): Default risk per trade (2% of capital).
            volatility_threshold (float): Adjusts risk if volatility exceeds threshold.
            drawdown_limit (float): Risk reduction when portfolio drawdown exceeds limit.
        """
        self.initial_risk = initial_risk
        self.volatility_threshold = volatility_threshold
        self.drawdown_limit = drawdown_limit
        self.current_drawdown = 0.0  # Track portfolio drawdown

    def adjust_risk(self, volatility, portfolio_drawdown):
        """
        Adjust risk level dynamically based on volatility and portfolio drawdown.

        Args:
            volatility (float): Market volatility.
            portfolio_drawdown (float): Current portfolio drawdown.

        Returns:
            float: Adjusted risk per trade.
        """
        risk = self.initial_risk

        # Reduce risk if volatility is high
        if volatility > self.volatility_threshold:
            risk *= 0.8  # Reduce risk by 20%

        # Further reduce risk if portfolio drawdown exceeds the limit
        if portfolio_drawdown > self.drawdown_limit:
            risk *= 0.5  # Cut risk by 50%

        return max(risk, 0.005)  # Ensure a minimum risk level of 0.5%

    def save(self, filename):
        """Save risk manager state."""
        import json
        state = {
            "initial_risk": self.initial_risk,
            "volatility_threshold": self.volatility_threshold,
            "drawdown_limit": self.drawdown_limit,
            "current_drawdown": self.current_drawdown
        }
        with open(filename, 'w') as f:
            json.dump(state, f)

    def load(self, filename):
        """Load risk manager state."""
        import json
        with open(filename, 'r') as f:
            state = json.load(f)
            self.initial_risk = state["initial_risk"]
            self.volatility_threshold = state["volatility_threshold"]
            self.drawdown_limit = state["drawdown_limit"]
            self.current_drawdown = state["current_drawdown"]

