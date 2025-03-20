import logging
import json
import os
import pandas as pd

from risk_management.adaptive_risk import AdaptiveRiskManager  # ✅ Correct Import

class TradingSystem:
    """Main trading system for executing and managing trades."""

    def __init__(self, config_path=None, use_mock_db=False, log_level=logging.INFO):
        """Initializes all components of the trading system."""
        self.config_path = os.path.abspath(config_path) if config_path else None
        logging.info(f"🟢 Loading Config from: {self.config_path}")  # ✅ Debugging print

        logging.basicConfig(level=log_level)
        logging.info(f"🔹 Log level set to: {logging.getLevelName(log_level)}")

        self.risk_manager = AdaptiveRiskManager()
        self.active_trades = []
        self.trade_history = []
        self.use_mock_db = use_mock_db
        self.config = {}

        if self.use_mock_db:
            logging.info("🟢 Running Trading System with Mock Database Mode")

        if self.config_path:
            self.load_config(self.config_path)

        logging.info(f"📄 FINAL CONFIG: {self.config}")  # ✅ Confirm config is loaded
        logging.info("✅ Trading System Initialized with Adaptive Risk Manager.")

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

        logging.info(f"📄 FINAL CONFIG AFTER VALIDATION: {self.config}")  # ✅ Confirm final config

    # ----------------- New Methods Added -----------------

    def get_open_positions(self):
        """Get all open positions."""
        if hasattr(self, 'db'):
            return self.db.get_open_trades().to_dict('records')
        return []

    def get_portfolio_history(self, days=30):
        """Get portfolio history."""
        if hasattr(self, 'db'):
            return self.db.get_portfolio_history(days=days)
        return pd.DataFrame()

    def close_position(self, symbol, price, reason):
        """Close a position."""
        # Implementation depends on your execution framework
        pass

    def update_position(self, symbol, modifications):
        """Update position parameters."""
        # Implementation depends on your position management system
        pass

    # -------------------------------------------------------

    # ✅ Standalone test function
    if __name__ == "__main__":
        # Example usage with mock settings for testing purposes
        system = TradingSystem(config_path="config.json", use_mock_db=True, log_level=logging.DEBUG)

        test_trade = {
            "trade_id": "T001",
            "asset": "BTC/USD",
            "position_size": 800,
            "drawdown": 7,
            "leverage_used": 2.5,
            "risk_exposure": 18,
            "stop_loss_triggered": False
        }

        # Assuming execute_trade, get_active_trades, update_risk_limits, and get_portfolio_value
        # are implemented elsewhere in your full trading system.
        try:
            system.execute_trade(test_trade)
            print("Active Trades:", system.get_active_trades())
            system.update_risk_limits({"max_drawdown_limit": 15, "position_limit": 1500})
            print("Updated Risk Limits:", system.config.get("risk_limits", {}))
            print("Portfolio Value:", system.get_portfolio_value())
        except AttributeError:
            logging.warning("Some trading system methods are not implemented for testing.")

