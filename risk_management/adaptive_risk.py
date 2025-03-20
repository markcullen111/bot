# adaptive_risk.py

import logging

class AdaptiveRiskManager:
    """Manages dynamic risk parameters for trading."""

    def __init__(self):
        """Initializes default risk limits."""
        self.risk_limits = {  # ✅ This ensures `risk_limits` exists
            "max_drawdown_limit": 10,  # Default drawdown limit in %
            "position_limit": 1000,  # Max number of positions
            "max_open_positions": 5,  # Maximum number of concurrent open positions
            "stop_loss": 5,  # Default stop-loss in %
            "max_risk_per_sector": 0.2,  # Maximum risk per sector/asset (20%)
            "max_risk_exposure": 20,  # Maximum portfolio exposure in %
            "leverage_limit": 3,  # Maximum leverage allowed
            "daily_loss_limit": 0.03,  # 3% max daily loss
            "weekly_loss_limit": 0.07,  # 7% max weekly loss
            "correlation_threshold": 0.7  # Correlation threshold for position limits
        }
        logging.info("✅ AdaptiveRiskManager initialized with default risk limits.")

    def set_risk_limits(self, limits):
        """Updates risk limits dynamically."""
        self.risk_limits.update(limits)
        logging.info(f"🔄 Risk limits updated: {self.risk_limits}")

    def get_risk_limits(self):
        """Returns current risk limits."""
        return self.risk_limits
        
    # Add property accessors for UI compatibility
    @property
    def max_open_positions(self):
        """Get maximum number of open positions."""
        return self.risk_limits.get("max_open_positions", 5)
        
    @max_open_positions.setter
    def max_open_positions(self, value):
        """Set maximum number of open positions."""
        self.risk_limits["max_open_positions"] = value
        
    @property
    def max_risk_per_sector(self):
        """Get maximum risk per sector."""
        return self.risk_limits.get("max_risk_per_sector", 0.2)
        
    @max_risk_per_sector.setter
    def max_risk_per_sector(self, value):
        """Set maximum risk per sector."""
        self.risk_limits["max_risk_per_sector"] = value
        
    @property
    def correlation_threshold(self):
        """Get correlation threshold."""
        return self.risk_limits.get("correlation_threshold", 0.7)
        
    @correlation_threshold.setter
    def correlation_threshold(self, value):
        """Set correlation threshold."""
        self.risk_limits["correlation_threshold"] = value
        
    @property
    def max_drawdown_limit(self):
        """Get maximum drawdown limit."""
        return self.risk_limits.get("max_drawdown_limit", 15) / 100.0
        
    @max_drawdown_limit.setter
    def max_drawdown_limit(self, value):
        """Set maximum drawdown limit."""
        self.risk_limits["max_drawdown_limit"] = value * 100.0
        
    @property
    def daily_loss_limit(self):
        """Get daily loss limit."""
        return self.risk_limits.get("daily_loss_limit", 0.03)
        
    @daily_loss_limit.setter
    def daily_loss_limit(self, value):
        """Set daily loss limit."""
        self.risk_limits["daily_loss_limit"] = value
