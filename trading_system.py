# trading_system.py

import logging
import json
import os
from typing import Dict, List, Optional
from datetime import datetime

from core.brokers.base_broker import Order, Position, OrderType, OrderSide
from core.brokers.interactive_brokers import InteractiveBrokers
from core.logging_config import setup_logging, log_order, log_position, log_balance, log_error
from risk_management.adaptive_risk import AdaptiveRiskManager
from strategies.strategy_system import AdvancedStrategySystem
from risk_management.position_sizing import AdaptivePositionSizer

class TradingSystem:
    """Main trading system for executing and managing trades."""

    def __init__(self, config_path=None, use_mock_db=False, log_level=logging.INFO):
        """Initializes all components of the trading system."""
        
        # Setup logging first
        setup_logging(log_level)
        self.logger = logging.getLogger(__name__)
        
        self.config_path = os.path.abspath(config_path) if config_path else None
        self.logger.info(f"🟢 Loading Config from: {self.config_path}")

        # Load configuration
        if self.config_path:
            self.load_config(self.config_path)
        else:
            raise ValueError("Configuration path must be provided")

        # Initialize broker connection
        self.broker = InteractiveBrokers(
            host=self.config.get('broker', {}).get('host', 'localhost'),
            port=self.config.get('broker', {}).get('port', 7497),
            client_id=1
        )
        
        # Connect to broker
        if not self.broker.connect():
            self.logger.error("Failed to connect to broker")
            raise ConnectionError("Could not connect to broker")

        # Initialize risk management components
        self.risk_manager = AdaptiveRiskManager()
        self.position_sizer = AdaptivePositionSizer(
            base_risk=self.config['risk_limits']['risk_per_trade'] / 100,
            max_risk=self.config['risk_limits']['max_position_size_percent'] / 100,
            min_risk=0.001,
            initial_capital=self.broker.get_account_balance()
        )
        
        # Initialize strategy system
        self.strategy_system = AdvancedStrategySystem()
        
        self.active_trades: List[Dict] = []
        self.trade_history: List[Dict] = []
        self.use_mock_db = use_mock_db

        # Log initial account balance
        initial_balance = self.broker.get_account_balance()
        log_balance(initial_balance)
        self.logger.info(f"Initial account balance: ${initial_balance:,.2f}")

    def load_config(self, config_path: str) -> None:
        """Load configuration from JSON file."""
        try:
            with open(config_path, 'r') as f:
                self.config = json.load(f)
            self.logger.info("Configuration loaded successfully")
        except Exception as e:
            self.logger.error(f"Error loading configuration: {str(e)}")
            raise

    def place_trade(self, symbol: str, side: OrderSide, quantity: float, 
                   order_type: OrderType = OrderType.MARKET, 
                   price: Optional[float] = None) -> str:
        """Place a trade with position sizing and risk management."""
        try:
            # Check if market is open
            if not self.broker.is_market_open(symbol):
                msg = f"Market is closed for {symbol}"
                self.logger.warning(msg)
                return msg

            # Get current market data
            market_data = self.broker.get_market_data(symbol)
            
            # Apply risk management checks
            if not self.risk_manager.check_trade(symbol, side, quantity, market_data):
                msg = "Trade rejected by risk management"
                self.logger.warning(msg)
                return msg

            # Calculate position size
            adjusted_quantity = self.position_sizer.calculate_position_size(
                symbol, quantity, market_data['last']
            )

            # Create and place order
            order = Order(
                symbol=symbol,
                order_type=order_type,
                side=side,
                quantity=adjusted_quantity,
                price=price
            )
            
            order_id = self.broker.place_order(order)
            
            # Log the order
            log_order(side.value, order_type.value)
            self.logger.info(f"Placed {side.value} order for {adjusted_quantity} {symbol}")
            
            return order_id

        except Exception as e:
            self.logger.error(f"Error placing trade: {str(e)}")
            log_error("trade_placement")
            raise

    def get_positions(self) -> List[Position]:
        """Get current positions with real-time values."""
        try:
            positions = self.broker.get_positions()
            
            # Log position values
            for pos in positions:
                log_position(pos.symbol, pos.quantity * pos.current_price)
            
            return positions
        except Exception as e:
            self.logger.error(f"Error getting positions: {str(e)}")
            log_error("position_fetch")
            raise

    def update_portfolio_status(self) -> Dict:
        """Update and return current portfolio status."""
        try:
            account_info = self.broker.get_account_info()
            positions = self.get_positions()
            
            # Log current balance
            log_balance(account_info['net_liquidation'])
            
            return {
                'account_value': account_info['net_liquidation'],
                'buying_power': account_info['buying_power'],
                'cash': account_info['cash'],
                'positions': positions
            }
        except Exception as e:
            self.logger.error(f"Error updating portfolio status: {str(e)}")
            log_error("portfolio_update")
            raise

    def run_strategy(self, strategy_name: str) -> None:
        """Run a specific trading strategy."""
        try:
            if strategy_name not in self.config['strategies']['active_strategies']:
                raise ValueError(f"Strategy {strategy_name} not in active strategies")
            
            signals = self.strategy_system.run_strategy(strategy_name)
            
            for signal in signals:
                if self.risk_manager.validate_signal(signal):
                    self.place_trade(
                        symbol=signal['symbol'],
                        side=signal['side'],
                        quantity=signal['quantity'],
                        order_type=signal.get('order_type', OrderType.MARKET),
                        price=signal.get('price')
                    )
        except Exception as e:
            self.logger.error(f"Error running strategy {strategy_name}: {str(e)}")
            log_error("strategy_execution")
            raise

    def cleanup(self) -> None:
        """Cleanup resources and close connections."""
        try:
            self.broker.disconnect()
            self.logger.info("Trading system shutdown complete")
        except Exception as e:
            self.logger.error(f"Error during cleanup: {str(e)}")
            log_error("cleanup")
            raise

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
