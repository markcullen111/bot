#!/usr/bin/env python3
"""
Main entry point for the trading system.
This script initializes and runs all components of the trading system.
"""

import sys
import logging
import signal
from pathlib import Path
from PyQt5.QtWidgets import QApplication
from trading_system import TradingSystem
from gui.main_window import MainWindow
from core.logging_config import setup_logging

def signal_handler(signum, frame):
    """Handle system signals gracefully."""
    logging.info(f"Received signal {signum}. Initiating shutdown...")
    QApplication.quit()

def main():
    """Main entry point for the trading system."""
    
    # Setup logging
    setup_logging(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    # Register signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    try:
        # Initialize QApplication
        app = QApplication(sys.argv)
        
        # Create config directory if it doesn't exist
        config_dir = Path("config")
        config_dir.mkdir(exist_ok=True)
        
        # Initialize trading system
        config_path = config_dir / "config.json"
        if not config_path.exists():
            logger.info("No config file found. Creating from example...")
            example_config = Path("config.example.json")
            if example_config.exists():
                config_path.write_text(example_config.read_text())
            else:
                logger.error("No config.example.json found! Please create a config file.")
                sys.exit(1)
        
        # Initialize trading system
        trading_system = TradingSystem(
            config_path=str(config_path),
            use_mock_db=False,  # Set to True for paper trading
            log_level=logging.INFO
        )
        
        # Create and show main window
        window = MainWindow(trading_system)
        window.show()
        
        # Start the event loop
        exit_code = app.exec_()
        
        # Cleanup
        trading_system.cleanup()
        sys.exit(exit_code)
        
    except Exception as e:
        logger.error(f"Fatal error: {str(e)}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main() 