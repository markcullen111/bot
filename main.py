import os
import json
import logging
import sys
from PyQt5.QtWidgets import QApplication  # ✅ Import QApplication first
from trading_system import TradingSystem
from gui.main_window import MainWindow

# ✅ Ensure we pass the correct config path
CONFIG_PATH = os.path.join(os.getcwd(), "config.json")

# ✅ Check if config.json exists, if not, create it
if not os.path.exists(CONFIG_PATH):
    logging.warning(f"🚨 Config file not found at {CONFIG_PATH}. Creating a default one...")
    default_config = {
        "risk_limits": {
            "max_drawdown_limit": 10,
            "position_limit": 1000,
            "stop_loss": 5
        },
        "strategies": {
            "active_strategies": ["mean_reversion", "trend_following"]
        }
    }
    with open(CONFIG_PATH, "w") as f:
        json.dump(default_config, f, indent=4)
    logging.info(f"✅ Default config.json created at {CONFIG_PATH}")

def main():
    logging.info("🚀 Initializing Trading System...")
    trading_system = TradingSystem(config_path=CONFIG_PATH, use_mock_db=True, log_level=logging.INFO)

    # ✅ Fix: Initialize QApplication before any QWidget
    logging.info("🎯 Starting GUI application...")
    app = QApplication(sys.argv)  # ✅ Ensure QApplication is created before any QWidget
    window = MainWindow(trading_system)
    window.show()

    # ✅ Fix: Start the application event loop
    sys.exit(app.exec_())  # ✅ Prevent unexpected crashes

if __name__ == "__main__":
    main()

