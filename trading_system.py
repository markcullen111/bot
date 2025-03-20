# trading_system.py

import logging
import json
import os
from risk_management.adaptive_risk import AdaptiveRiskManager  # ✅ Correct Import
from strategies.strategy_system import AdvancedStrategySystem  # ✅ Import strategy system
from risk_management.position_sizing import AdaptivePositionSizer  # ✅ Import position sizer

class TradingSystem:
    """Main trading system for executing and managing trades."""

    def __init__(self, config_path=None, use_mock_db=False, log_level=logging.INFO):
        """Initializes all components of the trading system."""

        self.config_path = os.path.abspath(config_path) if config_path else None
        logging.info(f"🟢 Loading Config from: {self.config_path}")  # ✅ Debugging print

        logging.basicConfig(level=log_level)
        logging.info(f"🔹 Log level set to: {logging.getLevelName(log_level)}")

        # Initialize risk management components
        self.risk_manager = AdaptiveRiskManager()
        self.position_sizer = AdaptivePositionSizer(
            base_risk=0.02,  # Default base risk 2%
            max_risk=0.05,   # Maximum risk 5%
            min_risk=0.01,   # Minimum risk 1%
            initial_capital=10000  # Default starting capital
        )
        
        # Initialize strategy system
        self.strategy_system = AdvancedStrategySystem()
        
        self.active_trades = []
        self.trade_history = []
        self.use_mock_db = use_mock_db
        self.config = {}

        if self.use_mock_db:
            logging.info("🟢 Running Trading System with Mock Database Mode")

        if self.config_path:
            self.load_config(self.config_path)
            
            # Initialize additional components based on config
            self._initialize_components()

        logging.info(f"📄 FINAL CONFIG: {self.config}")  # ✅ Confirm config is loaded
        logging.info("✅ Trading System Initialized with all components.")

    def load_config(self, config_path):
        """Loads trading system configuration from a file and ensures necessary keys exist."""
        logging.info(f"🟢 Attempting to load config from: {config_path}")  # ✅ Print path

        if not os.path.exists(config_path):
            logging.error(f"🚨 Config file NOT found at {config_path}. Creating default config...")
            self._create_default_config(config_path)
            return

        try:
            with open(config_path, "r") as config_file:
                self.config = json.load(config_file)

            logging.info(f"✅ Successfully loaded config from: {config_path}")

        except json.JSONDecodeError as e:
            logging.error(f"🚨 JSON Error in config file: {e}. Resetting config to default.")
            self._create_default_config(config_path)

        except Exception as e:
            logging.error(f"⚠️ Unexpected error loading config: {e}. Using default settings.")
            self._create_default_config(config_path)

        self._validate_config()

    def _create_default_config(self, config_path):
        """Creates a default configuration file."""
        self.config = {
            "risk_limits": {
                "max_drawdown_limit": 10,
                "position_limit": 1000,
                "stop_loss": 5
            },
            "strategies": {
                "active_strategies": ["mean_reversion", "trend_following"]
            },
            "trading": {
                "symbols": ["BTC/USDT", "ETH/USDT"],
                "timeframes": ["1h", "4h"]
            }
        }
        with open(config_path, "w") as f:
            json.dump(self.config, f, indent=4)
        logging.info(f"✅ Default config created at: {config_path}")

    def _validate_config(self):
        """Ensures all required keys exist in config."""
        if "risk_limits" not in self.config:
            logging.warning("⚠️ No 'risk_limits' found in config. Using default values.")
            self.config["risk_limits"] = {
                "max_drawdown_limit": 10,
                "position_limit": 1000,
                "stop_loss": 5
            }

        if "strategies" not in self.config or "active_strategies" not in self.config["strategies"]:
            logging.warning("⚠️ No 'strategies' found in config. Using default strategies.")
            self.config["strategies"] = {"active_strategies": ["mean_reversion", "trend_following"]}
            
        if "trading" not in self.config:
            logging.warning("⚠️ No 'trading' settings found in config. Using defaults.")
            self.config["trading"] = {
                "symbols": ["BTC/USDT", "ETH/USDT"],
                "timeframes": ["1h", "4h"]
            }

        logging.info(f"📄 FINAL CONFIG AFTER VALIDATION: {self.config}")  # ✅ Confirm final config
    
    def _initialize_components(self):
        """Initialize additional components based on configuration"""
        # Initialize strategy system with active strategies
        if self.strategy_system and "strategies" in self.config:
            active_strategies = self.config["strategies"].get("active_strategies", [])
            if active_strategies:
                logging.info(f"Initializing strategy system with strategies: {active_strategies}")
                self.strategy_system.active_strategies = active_strategies
        
        # Configure position sizer based on risk settings
        if self.position_sizer and "risk_limits" in self.config:
            risk_limits = self.config["risk_limits"]
            if "max_drawdown_limit" in risk_limits:
                # Convert percentage to decimal (e.g., 10 -> 0.10)
                max_dd = risk_limits["max_drawdown_limit"] / 100
                self.position_sizer.max_drawdown = max_dd
                
        # Initialize other components as needed
        # ...

    def execute_trade(self, trade):
        """Execute a trade with risk management."""
        # Example implementation
        logging.info(f"Executing trade: {trade}")
        
        # Apply risk management
        if hasattr(self, 'risk_manager'):
            # Check if trade passes risk checks
            risk_check = self.risk_manager.check_risk_limits(trade)
            if not risk_check.get('risk_ok', False):
                logging.warning(f"Trade rejected by risk manager: {risk_check.get('reason')}")
                return False
        
        # Apply position sizing
        if hasattr(self, 'position_sizer'):
            trade_size = self.position_sizer.calculate_position_size(
                price=trade.get('price', 0),
                confidence=trade.get('confidence', 0.5)
            )
            trade['position_size'] = trade_size.get('position_size', 0)
        
        # Execute the trade logic here
        # ...
        
        # Add to active trades
        self.active_trades.append(trade)
        
        return True

    def get_active_trades(self):
        """Return list of active trades."""
        return self.active_trades

    def update_risk_limits(self, new_limits):
        """Update risk management parameters."""
        if hasattr(self, 'risk_manager'):
            self.risk_manager.set_risk_limits(new_limits)
            logging.info(f"Risk limits updated: {new_limits}")
            
        # Update config
        if "risk_limits" in self.config:
            self.config["risk_limits"].update(new_limits)

    def get_risk_limits(self):
        """Get current risk limits."""
        if hasattr(self, 'risk_manager'):
            return self.risk_manager.get_risk_limits()
        return {}

    def get_portfolio_value(self):
        """Get current portfolio value."""
        # Placeholder implementation
        return {
            "total_value": 10500,
            "cash_value": 5000,
            "crypto_value": 5500
        }
        
    def analyze_market(self, symbol):
        """Analyze market conditions for a symbol."""
        # This would normally use real market data and analysis
        return {
            "market_condition": "bull",  # Example market condition
            "volatility": 0.8,           # Example volatility measure
            "trend_strength": 0.65       # Example trend strength
        }
        
    def get_market_data(self, symbol, timeframe='1h', limit=100):
        """Get market data for a symbol and timeframe."""
        # This would normally fetch from database or API
        # Return empty DataFrame for now
        import pandas as pd
        return pd.DataFrame()
        
    def close_position(self, symbol, price, reason=""):
        """Close a position."""
        for i, trade in enumerate(self.active_trades):
            if trade.get('symbol') == symbol:
                closed_trade = self.active_trades.pop(i)
                closed_trade['exit_price'] = price
                closed_trade['exit_reason'] = reason
                self.trade_history.append(closed_trade)
                logging.info(f"Closed position: {symbol} at {price}, reason: {reason}")
                return closed_trade
        return None

# ✅ Standalone test function
if __name__ == "__main__":
    system = TradingSystem(config_path="config.json", use_mock_db=True, log_level=logging.DEBUG)

    test_trade = {
        "trade_id": "T001",
        "symbol": "BTC/USD",
        "action": "buy",
        "price": 40000,
        "amount": 0.5,
        "position_size": 0.5,
        "confidence": 0.8
    }

    system.execute_trade(test_trade)
    print("Active Trades:", system.get_active_trades())

    system.update_risk_limits({"max_drawdown_limit": 15, "position_limit": 1500})
    print("Updated Risk Limits:", system.get_risk_limits())

    print("Portfolio Value:", system.get_portfolio_value())
