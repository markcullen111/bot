# execution/market_maker.py

import logging
import numpy as np
import time
import threading
import queue
import json
import os
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any, Union, Callable

# Import error handling if available
try:
    from error_handling import safe_execute, ErrorCategory, ErrorSeverity, ErrorHandler, TradingSystemError
    HAVE_ERROR_HANDLING = True
except ImportError:
    HAVE_ERROR_HANDLING = False
    logging.warning("Error handling module not available. Using basic error handling in market maker.")

# Import ML model for market making
try:
    from ml_models.market_making_ai import AIMarketMaker
    HAVE_AI_MARKET_MAKER = True
except ImportError:
    try:
        from market_making_ai import AIMarketMaker
        HAVE_AI_MARKET_MAKER = True
    except ImportError:
        HAVE_AI_MARKET_MAKER = False
        logging.warning("AI Market Maker model not available. Using classical algorithms.")

class OrderBookState:
    """
    Represents the current state of the order book for a symbol.
    Tracks bid-ask spreads, depth, and imbalances.
    """
    
    def __init__(self, symbol: str):
        self.symbol = symbol
        self.timestamp = datetime.now()
        self.bids: List[Tuple[float, float]] = []  # Price, Size
        self.asks: List[Tuple[float, float]] = []  # Price, Size
        self.mid_price: float = 0.0
        self.spread: float = 0.0
        self.bid_depth: float = 0.0
        self.ask_depth: float = 0.0
        self.imbalance: float = 0.0  # Positive = more bids, Negative = more asks
        self.volatility: float = 0.0
        self.last_update = datetime.now()
        
    def update(self, bids: List[Tuple[float, float]], asks: List[Tuple[float, float]]) -> None:
        """Update order book state with latest data"""
        self.timestamp = datetime.now()
        self.bids = sorted(bids, key=lambda x: x[0], reverse=True)
        self.asks = sorted(asks, key=lambda x: x[0])
        
        if not self.bids or not self.asks:
            logging.warning(f"Empty bid or ask book for {self.symbol}")
            return
            
        # Calculate mid price and spread
        best_bid = self.bids[0][0]
        best_ask = self.asks[0][0]
        self.mid_price = (best_bid + best_ask) / 2
        self.spread = best_ask - best_bid
        
        # Calculate depth (sum of quantities at top 5 levels)
        self.bid_depth = sum(qty for _, qty in self.bids[:5])
        self.ask_depth = sum(qty for _, qty in self.asks[:5])
        
        # Calculate imbalance
        total_depth = self.bid_depth + self.ask_depth
        if total_depth > 0:
            self.imbalance = (self.bid_depth - self.ask_depth) / total_depth
        
        self.last_update = datetime.now()
        
    def is_stale(self, max_age_seconds: int = 5) -> bool:
        """Check if order book data is stale"""
        age = (datetime.now() - self.last_update).total_seconds()
        return age > max_age_seconds
        
    def to_features(self) -> np.ndarray:
        """Convert order book state to features for ML model"""
        # Extract features from order book for ML model
        features = [
            self.mid_price,
            self.spread,
            self.bid_depth,
            self.ask_depth,
            self.imbalance,
            self.volatility
        ]
        
        # Add top 5 bid and ask levels (price and quantity)
        for i in range(5):
            if i < len(self.bids):
                features.extend(self.bids[i])
            else:
                features.extend([0, 0])
                
            if i < len(self.asks):
                features.extend(self.asks[i])
            else:
                features.extend([0, 0])
                
        return np.array(features)

class InventoryManager:
    """
    Manages market maker's inventory positions and risk exposure.
    Implements inventory rebalancing strategies to maintain target position.
    """
    
    def __init__(self, 
                 symbol: str, 
                 target_inventory: float = 0.0, 
                 max_inventory: float = 1.0, 
                 rebalance_threshold: float = 0.5):
        self.symbol = symbol
        self.target_inventory = target_inventory
        self.max_inventory = max_inventory
        self.rebalance_threshold = rebalance_threshold
        self.current_inventory = 0.0
        self.inventory_cost_basis = 0.0
        self.inventory_value = 0.0
        self.unrealized_pnl = 0.0
        self.realized_pnl = 0.0
        self.position_history = []
        self.last_rebalance = datetime.now()
        
    def update_inventory(self, quantity: float, price: float, is_buy: bool) -> None:
        """Update inventory after a trade"""
        old_inventory = self.current_inventory
        old_cost_basis = self.inventory_cost_basis
        
        if is_buy:
            # Buying increases inventory
            new_position_cost = quantity * price
            self.current_inventory += quantity
            self.inventory_cost_basis += new_position_cost
        else:
            # Selling decreases inventory
            if self.current_inventory > 0:
                # Calculate realized PnL for selling from long inventory
                avg_price = self.inventory_cost_basis / self.current_inventory if self.current_inventory > 0 else 0
                realized_pnl = (price - avg_price) * min(quantity, self.current_inventory)
                self.realized_pnl += realized_pnl
                
                # Reduce cost basis proportionally
                if self.current_inventory > 0:
                    cost_basis_reduction = (quantity / self.current_inventory) * self.inventory_cost_basis
                    self.inventory_cost_basis -= min(cost_basis_reduction, self.inventory_cost_basis)
            
            self.current_inventory -= quantity
            
        # Record position change
        self.position_history.append({
            'timestamp': datetime.now(),
            'old_inventory': old_inventory,
            'new_inventory': self.current_inventory,
            'quantity': quantity,
            'price': price,
            'is_buy': is_buy
        })
        
        logging.info(
            f"Inventory updated - Symbol: {self.symbol}, Action: {'BUY' if is_buy else 'SELL'}, "
            f"Quantity: {quantity}, Price: {price}, New Inventory: {self.current_inventory:.6f}"
        )
        
    def update_value(self, current_price: float) -> None:
        """Update inventory value and unrealized PnL based on current price"""
        self.inventory_value = self.current_inventory * current_price
        
        # Calculate unrealized PnL
        if self.current_inventory != 0:
            avg_price = self.inventory_cost_basis / self.current_inventory if self.current_inventory > 0 else 0
            self.unrealized_pnl = (current_price - avg_price) * self.current_inventory
        else:
            self.unrealized_pnl = 0
            
    def should_rebalance(self) -> bool:
        """Determine if inventory should be rebalanced"""
        # Check if deviation from target exceeds threshold
        deviation = abs(self.current_inventory - self.target_inventory)
        threshold = self.max_inventory * self.rebalance_threshold
        
        # Also check time since last rebalance (avoid too frequent rebalancing)
        time_since_rebalance = (datetime.now() - self.last_rebalance).total_seconds()
        
        return deviation > threshold and time_since_rebalance > 60  # 1 minute minimum between rebalances
        
    def get_rebalance_order(self, current_price: float) -> Optional[Dict[str, Any]]:
        """Generate an order to rebalance inventory toward target"""
        if not self.should_rebalance():
            return None
            
        deviation = self.current_inventory - self.target_inventory
        
        # Calculate order size to move toward target (don't try to rebalance all at once)
        rebalance_size = abs(deviation) * 0.5  # Rebalance 50% of the deviation
        is_buy = deviation < 0  # If current < target, we need to buy
        
        # Adjust price to be aggressive on rebalancing
        # For buys: bid slightly higher, for sells: ask slightly lower
        price_adjustment = 0.0001 * current_price  # 0.01% adjustment
        adjusted_price = current_price + price_adjustment if is_buy else current_price - price_adjustment
        
        self.last_rebalance = datetime.now()
        
        return {
            'symbol': self.symbol,
            'side': 'buy' if is_buy else 'sell',
            'quantity': rebalance_size,
            'price': adjusted_price,
            'order_type': 'limit',
            'time_in_force': 'GTT',  # Good until time (will be cancelled if not filled)
            'purpose': 'inventory_rebalance'
        }
        
    def get_risk_exposure(self) -> Dict[str, float]:
        """Get risk metrics for current inventory"""
        return {
            'inventory': self.current_inventory,
            'inventory_value': self.inventory_value,
            'inventory_pct': self.current_inventory / self.max_inventory if self.max_inventory > 0 else 0,
            'unrealized_pnl': self.unrealized_pnl,
            'realized_pnl': self.realized_pnl,
            'total_pnl': self.unrealized_pnl + self.realized_pnl,
            'deviation_from_target': self.current_inventory - self.target_inventory
        }
        
    def reset_stats(self) -> None:
        """Reset PnL and statistics (useful for daily resets)"""
        self.realized_pnl = 0.0
        self.unrealized_pnl = 0.0

class SpreadOptimizer:
    """
    Optimizes bid-ask spreads based on market conditions and inventory.
    Adjusts spreads based on volatility, liquidity, and inventory position.
    """
    
    def __init__(self, 
                 base_spread_bps: float = 10.0,
                 min_spread_bps: float = 2.0,
                 max_spread_bps: float = 50.0,
                 inventory_impact_factor: float = 1.0,
                 volatility_impact_factor: float = 1.0):
        self.base_spread_bps = base_spread_bps  # Base spread in basis points (0.01%)
        self.min_spread_bps = min_spread_bps    # Minimum spread in basis points
        self.max_spread_bps = max_spread_bps    # Maximum spread in basis points
        self.inventory_impact_factor = inventory_impact_factor
        self.volatility_impact_factor = volatility_impact_factor
        self.last_spreads = {}  # Track last spread by symbol
        self.spread_history = []  # Track spread history for analysis
        
    def calculate_spread(self, 
                       symbol: str,
                       mid_price: float, 
                       volatility: float, 
                       inventory_deviation: float,
                       market_imbalance: float = 0.0) -> Tuple[float, float, float]:
        """
        Calculate optimal bid and ask prices with dynamic spread.
        
        Args:
            symbol: Trading pair symbol
            mid_price: Current mid price
            volatility: Current market volatility (e.g., ATR ratio)
            inventory_deviation: Deviation from target inventory (normalized -1 to 1)
            market_imbalance: Order book imbalance (-1 to 1, positive = more bids)
            
        Returns:
            Tuple of (bid_price, ask_price, spread_bps)
        """
        # Base spread in price terms
        base_spread = (self.base_spread_bps / 10000) * mid_price
        
        # Adjust for volatility
        volatility_adjustment = volatility * self.volatility_impact_factor
        
        # Adjust for inventory - widen spread on side we don't want to trade
        # If inventory_deviation is positive (too much inventory), lower bid and raise ask
        # If negative (too little inventory), raise bid and lower ask
        inventory_adjustment = abs(inventory_deviation) * self.inventory_impact_factor
        
        # Adjust for market imbalance - if more bids than asks, tighten ask spread and vice versa
        imbalance_adjustment = market_imbalance * 0.5  # Apply 50% of imbalance effect
        
        # Calculate final spread
        spread_bps = self.base_spread_bps * (1 + volatility_adjustment + inventory_adjustment)
        spread_bps = max(self.min_spread_bps, min(self.max_spread_bps, spread_bps))
        
        spread = (spread_bps / 10000) * mid_price
        
        # Calculate bid-ask skew based on inventory and imbalance
        skew_factor = inventory_deviation - imbalance_adjustment
        skew_factor = max(-0.9, min(0.9, skew_factor))  # Limit skew to ±90%
        
        # Apply skew to bid and ask prices
        skew_amount = spread * skew_factor * 0.5  # Apply half of skew to each side
        
        bid_price = mid_price - (spread / 2) - skew_amount
        ask_price = mid_price + (spread / 2) - skew_amount
        
        # Store this spread
        self.last_spreads[symbol] = spread_bps
        self.spread_history.append({
            'timestamp': datetime.now(),
            'symbol': symbol,
            'mid_price': mid_price,
            'spread_bps': spread_bps,
            'volatility': volatility,
            'inventory_deviation': inventory_deviation,
            'market_imbalance': market_imbalance
        })
        
        # Keep spread history manageable
        if len(self.spread_history) > 1000:
            self.spread_history = self.spread_history[-1000:]
            
        return bid_price, ask_price, spread_bps
        
    def get_spread_statistics(self, symbol: str = None) -> Dict[str, Any]:
        """Get statistics on recent spreads"""
        if not self.spread_history:
            return {"average_spread_bps": self.base_spread_bps}
            
        if symbol:
            symbol_history = [entry for entry in self.spread_history if entry['symbol'] == symbol]
            if not symbol_history:
                return {"average_spread_bps": self.base_spread_bps}
                
            spreads = [entry['spread_bps'] for entry in symbol_history]
        else:
            spreads = [entry['spread_bps'] for entry in self.spread_history]
            
        return {
            "average_spread_bps": np.mean(spreads),
            "min_spread_bps": np.min(spreads),
            "max_spread_bps": np.max(spreads),
            "current_spread_bps": spreads[-1] if spreads else self.base_spread_bps,
            "spread_volatility": np.std(spreads) if len(spreads) > 1 else 0
        }

class MarketMaker:
    """
    Advanced market making system that provides liquidity across multiple symbols.
    Uses AI model predictions combined with classical market making algorithms.
    Manages inventory, optimizes spreads, and implements risk management.
    """
    
    def __init__(self, trading_system, symbols: List[str] = None, config: Dict[str, Any] = None):
        """
        Initialize the market maker.
        
        Args:
            trading_system: Reference to main trading system
            symbols: List of trading pairs to market make
            config: Market making configuration parameters
        """
        self.trading_system = trading_system
        self.symbols = symbols or []
        self.running = False
        self.thread = None
        self.order_queue = queue.Queue()
        self.cancel_queue = queue.Queue()
        
        # Load config with defaults
        self.config = {
            # Market making parameters
            'base_spread_bps': 10.0,           # Base spread in basis points
            'min_spread_bps': 2.0,             # Minimum spread
            'max_spread_bps': 50.0,            # Maximum spread
            'order_refresh_seconds': 15,       # Time between order refreshes
            'inventory_skew_enabled': True,    # Enable inventory skew
            'volatility_adjustment_enabled': True,  # Enable volatility adjustment
            
            # Order parameters
            'min_order_size': 0.001,           # Minimum order size
            'max_order_size': 0.1,             # Maximum order size
            'order_levels': 1,                 # Number of order levels
            'level_spread_multiplier': 1.5,    # Multiplier for spreads at each level
            
            # Risk parameters
            'max_inventory': 1.0,              # Maximum inventory (base currency)
            'target_inventory': 0.0,           # Target inventory (base currency)
            'inventory_risk_factor': 1.0,      # How much inventory affects spreads
            'rebalance_threshold': 0.5,        # Threshold for inventory rebalancing
            
            # Execution parameters
            'time_in_force': 'GTT',            # Good till time
            'cancel_all_on_stop': True,        # Cancel all orders on stop
            
            # AI model parameters
            'use_ai_model': True,              # Whether to use AI predictions
            'ai_impact_factor': 0.5,           # How much AI affects decisions
            'model_path': 'models/market_making_model.pth',  # Model file
            
            # Performance tracking
            'track_performance': True,         # Whether to track performance
            'target_profit_bps': 5.0,          # Target profit in basis points
            'max_loss_bps': 20.0               # Maximum loss in basis points
        }
        
        # Update with provided config
        if config:
            self.config.update(config)
            
        # Initialize components
        self.order_books = {symbol: OrderBookState(symbol) for symbol in self.symbols}
        self.inventory_managers = {symbol: InventoryManager(
            symbol, 
            self.config['target_inventory'],
            self.config['max_inventory'],
            self.config['rebalance_threshold']
        ) for symbol in self.symbols}
        
        self.spread_optimizer = SpreadOptimizer(
            base_spread_bps=self.config['base_spread_bps'],
            min_spread_bps=self.config['min_spread_bps'],
            max_spread_bps=self.config['max_spread_bps'],
            inventory_impact_factor=self.config['inventory_risk_factor'],
            volatility_impact_factor=1.0
        )
        
        # Active orders tracking
        self.active_orders = {}  # order_id -> order details
        self.last_order_refresh = {symbol: datetime.now() for symbol in self.symbols}
        
        # Performance tracking
        self.trades = []
        self.start_time = None
        self.total_filled_volume = 0.0
        self.total_fees = 0.0
        self.total_pnl = 0.0
        
        # Initialize AI model if available
        self.ai_model = None
        if self.config['use_ai_model'] and HAVE_AI_MARKET_MAKER:
            try:
                self.ai_model = AIMarketMaker()
                logging.info("Market making AI model initialized successfully")
            except Exception as e:
                logging.error(f"Failed to initialize market making AI model: {e}")
                
        # Setup logging
        self.setup_logging()
                
    def setup_logging(self) -> None:
        """Set up dedicated logger for market maker"""
        self.logger = logging.getLogger('market_maker')
        self.logger.setLevel(logging.INFO)
        
        # Create a file handler if it doesn't exist
        if not self.logger.handlers:
            try:
                logs_dir = os.path.join(os.getcwd(), 'logs')
                os.makedirs(logs_dir, exist_ok=True)
                
                log_file = os.path.join(logs_dir, 'market_maker.log')
                file_handler = logging.FileHandler(log_file)
                file_handler.setLevel(logging.INFO)
                
                formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
                file_handler.setFormatter(formatter)
                
                self.logger.addHandler(file_handler)
            except Exception as e:
                logging.error(f"Failed to set up market maker logging: {e}")
                
    def log(self, level: int, message: str) -> None:
        """Log a message with the market maker's logger"""
        self.logger.log(level, message)
        
    def start(self) -> bool:
        """Start the market maker"""
        if self.running:
            logging.warning("Market maker already running")
            return False
            
        self.running = True
        self.start_time = datetime.now()
        
        # Start the market maker thread
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()
        
        logging.info(f"Market maker started for symbols: {', '.join(self.symbols)}")
        return True
        
    def stop(self) -> bool:
        """Stop the market maker"""
        if not self.running:
            logging.warning("Market maker not running")
            return False
            
        logging.info("Stopping market maker...")
        self.running = False
        
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=5.0)
            
        # Cancel all active orders if configured
        if self.config['cancel_all_on_stop']:
            self._cancel_all_orders()
            
        logging.info("Market maker stopped")
        
        # Log performance
        if self.config['track_performance']:
            self._log_performance()
            
        return True
        
    def _run(self) -> None:
        """Main market maker loop"""
        while self.running:
            try:
                # Process each symbol
                for symbol in self.symbols:
                    self._process_symbol(symbol)
                    
                # Process order queue
                self._process_order_queue()
                
                # Process cancel queue
                self._process_cancel_queue()
                
                # Sleep to avoid excessive CPU usage
                time.sleep(0.1)
                
            except Exception as e:
                error_msg = f"Error in market maker main loop: {str(e)}"
                if HAVE_ERROR_HANDLING:
                    ErrorHandler.handle_error(
                        TradingSystemError(error_msg, ErrorCategory.TRADE_EXECUTION, ErrorSeverity.WARNING),
                        context={"component": "MarketMaker", "symbols": self.symbols}
                    )
                else:
                    logging.error(error_msg)
                    
                # Sleep to avoid error spam
                time.sleep(1.0)
                
    def _process_symbol(self, symbol: str) -> None:
        """Process a single symbol for market making"""
        # Check if we need to refresh orders
        if self._should_refresh_orders(symbol):
            try:
                # Update order book
                self._update_order_book(symbol)
                
                # Update inventory value
                self._update_inventory_value(symbol)
                
                # Generate new orders
                orders = self._generate_orders(symbol)
                
                # Cancel existing orders
                self._cancel_existing_orders(symbol)
                
                # Place new orders
                for order in orders:
                    self._place_order(order)
                    
                # Check if we need to rebalance inventory
                self._check_inventory_rebalance(symbol)
                
                # Update last refresh time
                self.last_order_refresh[symbol] = datetime.now()
                
            except Exception as e:
                error_msg = f"Error processing symbol {symbol}: {str(e)}"
                if HAVE_ERROR_HANDLING:
                    ErrorHandler.handle_error(
                        TradingSystemError(error_msg, ErrorCategory.TRADE_EXECUTION, ErrorSeverity.WARNING),
                        context={"component": "MarketMaker", "symbol": symbol}
                    )
                else:
                    logging.error(error_msg)
                
    def _should_refresh_orders(self, symbol: str) -> bool:
        """Determine if orders should be refreshed for a symbol"""
        now = datetime.now()
        time_since_refresh = (now - self.last_order_refresh[symbol]).total_seconds()
        return time_since_refresh >= self.config['order_refresh_seconds']
        
    def _update_order_book(self, symbol: str) -> None:
        """Update the order book for a symbol"""
        try:
            # Get latest order book data
            order_book = self.trading_system.get_order_book(symbol)
            
            if not order_book:
                logging.warning(f"Empty order book received for {symbol}")
                return
                
            # Update order book state
            self.order_books[symbol].update(
                bids=order_book.get('bids', []),
                asks=order_book.get('asks', [])
            )
            
            # Calculate volatility estimate
            self._calculate_volatility(symbol)
            
        except Exception as e:
            error_msg = f"Error updating order book for {symbol}: {str(e)}"
            if HAVE_ERROR_HANDLING:
                ErrorHandler.handle_error(
                    TradingSystemError(error_msg, ErrorCategory.DATA_PROCESSING, ErrorSeverity.WARNING),
                    context={"component": "MarketMaker", "symbol": symbol}
                )
            else:
                logging.error(error_msg)
            
    def _calculate_volatility(self, symbol: str) -> None:
        """Calculate volatility estimate for a symbol"""
        try:
            # Get recent price data for volatility calculation
            data = self.trading_system.get_market_data(symbol, limit=20)
            
            if data is None or data.empty:
                self.order_books[symbol].volatility = 0.02  # Default value
                return
                
            # Calculate using standard deviation of returns
            returns = data['close'].pct_change().dropna()
            if len(returns) > 0:
                volatility = returns.std() * np.sqrt(252)  # Annualized
                self.order_books[symbol].volatility = volatility
            else:
                self.order_books[symbol].volatility = 0.02  # Default value
                
        except Exception as e:
            logging.warning(f"Error calculating volatility for {symbol}: {e}")
            self.order_books[symbol].volatility = 0.02  # Default value
            
    def _update_inventory_value(self, symbol: str) -> None:
        """Update inventory value based on current price"""
        order_book = self.order_books[symbol]
        if order_book.mid_price > 0:
            self.inventory_managers[symbol].update_value(order_book.mid_price)
            
    def _generate_orders(self, symbol: str) -> List[Dict[str, Any]]:
        """Generate market making orders for a symbol"""
        orders = []
        
        # Get current order book
        order_book = self.order_books[symbol]
        
        # If order book is stale, don't generate orders
        if order_book.is_stale():
            logging.warning(f"Order book for {symbol} is stale, skipping order generation")
            return []
            
        # Get inventory manager
        inventory_manager = self.inventory_managers[symbol]
        
        # Calculate inventory deviation (normalized to -1 to 1)
        inventory_deviation = 0.0
        if self.config['max_inventory'] > 0:
            current_deviation = inventory_manager.current_inventory - inventory_manager.target_inventory
            inventory_deviation = current_deviation / self.config['max_inventory']
            
        # Apply AI model if available
        ai_prediction = 0.0
        if self.ai_model and self.config['use_ai_model']:
            try:
                # Convert order book to features
                features = order_book.to_features()
                
                # Get AI prediction (-1 to 1 range)
                action = self.ai_model.predict_action(features.tolist())
                
                # Convert DQN action to adjustment
                if action == 0:  # Buy signal
                    ai_prediction = self.config['ai_impact_factor']
                elif action == 1:  # Sell signal
                    ai_prediction = -self.config['ai_impact_factor']
                # Action 2 is Hold (0)
                
            except Exception as e:
                logging.warning(f"Error getting AI prediction: {e}")
                
        # Adjust inventory deviation with AI prediction
        if self.config['use_ai_model']:
            inventory_deviation = (1 - self.config['ai_impact_factor']) * inventory_deviation + ai_prediction
            
        # Calculate optimal bid and ask prices
        bid_price, ask_price, spread_bps = self.spread_optimizer.calculate_spread(
            symbol,
            order_book.mid_price,
            order_book.volatility,
            inventory_deviation,
            order_book.imbalance
        )
        
        # Calculate base order size
        base_order_size = self._calculate_order_size(symbol, order_book.mid_price, inventory_deviation)
        
        # Generate orders for each level
        for level in range(self.config['order_levels']):
            # Calculate level spread multiplier (increase spread for each level)
            level_multiplier = 1.0 + (level * (self.config['level_spread_multiplier'] - 1.0))
            
            # Calculate level-specific spread
            level_spread = (spread_bps / 10000) * order_book.mid_price * level_multiplier
            
            # Calculate level prices
            level_bid_price = order_book.mid_price - (level_spread / 2)
            level_ask_price = order_book.mid_price + (level_spread / 2)
            
            # Apply inventory skew if enabled
            if self.config['inventory_skew_enabled']:
                # Reduce size on side that would increase inventory imbalance
                bid_size_multiplier = max(0.1, 1.0 - max(0, inventory_deviation))
                ask_size_multiplier = max(0.1, 1.0 + min(0, inventory_deviation))
                
                bid_size = base_order_size * bid_size_multiplier
                ask_size = base_order_size * ask_size_multiplier
            else:
                bid_size = ask_size = base_order_size
                
            # Ensure minimum size
            bid_size = max(self.config['min_order_size'], bid_size)
            ask_size = max(self.config['min_order_size'], ask_size)
            
            # Create bid order
            bid_order = {
                'symbol': symbol,
                'side': 'buy',
                'price': level_bid_price,
                'quantity': bid_size,
                'order_type': 'limit',
                'time_in_force': self.config['time_in_force'],
                'purpose': 'market_making'
            }
            
            # Create ask order
            ask_order = {
                'symbol': symbol,
                'side': 'sell',
                'price': level_ask_price,
                'quantity': ask_size,
                'order_type': 'limit',
                'time_in_force': self.config['time_in_force'],
                'purpose': 'market_making'
            }
            
            orders.extend([bid_order, ask_order])
            
        return orders
        
    def _calculate_order_size(self, symbol: str, price: float, inventory_deviation: float) -> float:
        """Calculate optimal order size based on market conditions and inventory"""
        # Base size calculation
        base_size = (self.config['max_order_size'] + self.config['min_order_size']) / 2
        
        # Adjust for inventory - reduce size as we approach max inventory
        inventory_factor = 1.0
        if abs(inventory_deviation) > 0.5:
            # Reduce size as we approach limits
            inventory_factor = max(0.1, 1.0 - (abs(inventory_deviation) - 0.5) * 2)
            
        # Calculate final size
        order_size = base_size * inventory_factor
        
        # Ensure within limits
        order_size = max(self.config['min_order_size'], min(self.config['max_order_size'], order_size))
        
        return order_size
        
    def _cancel_existing_orders(self, symbol: str) -> None:
        """Cancel existing orders for a symbol"""
        # Find all active orders for this symbol
        orders_to_cancel = []
        
        for order_id, order in list(self.active_orders.items()):
            if order['symbol'] == symbol and order['purpose'] == 'market_making':
                orders_to_cancel.append(order_id)
                
        # Submit cancel requests
        for order_id in orders_to_cancel:
            self.cancel_queue.put(order_id)
            
    def _check_inventory_rebalance(self, symbol: str) -> None:
        """Check if inventory rebalancing is needed"""
        inventory_manager = self.inventory_managers[symbol]
        
        if inventory_manager.should_rebalance():
            order_book = self.order_books[symbol]
            
            # Generate rebalancing order
            rebalance_order = inventory_manager.get_rebalance_order(order_book.mid_price)
            
            if rebalance_order:
                self._place_order(rebalance_order)
                logging.info(f"Generated inventory rebalance order for {symbol}")
                
    def _place_order(self, order: Dict[str, Any]) -> None:
        """Place an order via the order queue"""
        self.order_queue.put(order)
        
    def _process_order_queue(self) -> None:
        """Process pending orders from the queue"""
        # Process up to 10 orders per iteration to avoid blocking
        for _ in range(10):
            try:
                order = self.order_queue.get_nowait()
                
                # Submit order to trading system
                order_id = self._submit_order(order)
                
                if order_id:
                    # Track the order
                    order['order_id'] = order_id
                    self.active_orders[order_id] = order
                    
                self.order_queue.task_done()
                
            except queue.Empty:
                break
                
            except Exception as e:
                logging.error(f"Error processing order queue: {e}")
                
    def _process_cancel_queue(self) -> None:
        """Process pending cancellations from the queue"""
        # Process up to 10 cancellations per iteration
        for _ in range(10):
            try:
                order_id = self.cancel_queue.get_nowait()
                
                # Cancel the order
                self._cancel_order(order_id)
                
                self.cancel_queue.task_done()
                
            except queue.Empty:
                break
                
            except Exception as e:
                logging.error(f"Error processing cancel queue: {e}")
                
    def _submit_order(self, order: Dict[str, Any]) -> Optional[str]:
        """Submit an order to the trading system"""
        try:
            # Submit the order using the trading system's order function
            if hasattr(self.trading_system, 'place_order'):
                order_id = self.trading_system.place_order(
                    symbol=order['symbol'],
                    side=order['side'],
                    order_type=order.get('order_type', 'limit'),
                    price=order['price'],
                    quantity=order['quantity'],
                    time_in_force=order.get('time_in_force', 'GTC')
                )
                
                if order_id:
                    logging.info(
                        f"Placed {order['side']} order for {order['symbol']} - "
                        f"Price: {order['price']}, Quantity: {order['quantity']}"
                    )
                    return order_id
                    
            return None
            
        except Exception as e:
            error_msg = f"Error submitting order: {str(e)}"
            if HAVE_ERROR_HANDLING:
                ErrorHandler.handle_error(
                    TradingSystemError(error_msg, ErrorCategory.TRADE_EXECUTION, ErrorSeverity.WARNING),
                    context={"component": "MarketMaker", "order": order}
                )
            else:
                logging.error(error_msg)
                
            return None
            
    def _cancel_order(self, order_id: str) -> bool:
        """Cancel an order with the trading system"""
        try:
            if order_id in self.active_orders:
                # Call trading system cancel function
                if hasattr(self.trading_system, 'cancel_order'):
                    success = self.trading_system.cancel_order(order_id)
                    
                    if success:
                        # Remove from active orders
                        if order_id in self.active_orders:
                            del self.active_orders[order_id]
                            
                        return True
                        
            return False
            
        except Exception as e:
            error_msg = f"Error cancelling order {order_id}: {str(e)}"
            if HAVE_ERROR_HANDLING:
                ErrorHandler.handle_error(
                    TradingSystemError(error_msg, ErrorCategory.TRADE_EXECUTION, ErrorSeverity.WARNING),
                    context={"component": "MarketMaker", "order_id": order_id}
                )
            else:
                logging.error(error_msg)
                
            return False
            
    def _cancel_all_orders(self) -> None:
        """Cancel all active orders"""
        order_ids = list(self.active_orders.keys())
        
        for order_id in order_ids:
            self._cancel_order(order_id)
            
        logging.info(f"Cancelled {len(order_ids)} active orders")
        
    def handle_fill(self, fill_data: Dict[str, Any]) -> None:
        """Handle trade fill event"""
        try:
            order_id = fill_data.get('order_id')
            symbol = fill_data.get('symbol')
            side = fill_data.get('side')
            price = fill_data.get('price')
            quantity = fill_data.get('quantity')
            fee = fill_data.get('fee', 0.0)
            
            if not all([order_id, symbol, side, price, quantity]):
                logging.warning(f"Incomplete fill data received: {fill_data}")
                return
                
            # Update inventory
            if symbol in self.inventory_managers:
                is_buy = side.lower() == 'buy'
                self.inventory_managers[symbol].update_inventory(quantity, price, is_buy)
                
            # Update active orders
            if order_id in self.active_orders:
                # If order fully filled, remove from active orders
                if fill_data.get('status') == 'filled':
                    del self.active_orders[order_id]
                    
            # Track fill for performance metrics
            self.trades.append({
                'timestamp': datetime.now(),
                'symbol': symbol,
                'side': side,
                'price': price,
                'quantity': quantity,
                'fee': fee,
                'order_id': order_id,
                'purpose': self.active_orders.get(order_id, {}).get('purpose', 'unknown')
            })
            
            # Update performance metrics
            self.total_filled_volume += quantity * price
            self.total_fees += fee
            
            logging.info(
                f"Trade filled - Symbol: {symbol}, Side: {side}, "
                f"Price: {price}, Quantity: {quantity}, Fee: {fee}"
            )
            
        except Exception as e:
            error_msg = f"Error handling fill: {str(e)}"
            if HAVE_ERROR_HANDLING:
                ErrorHandler.handle_error(
                    TradingSystemError(error_msg, ErrorCategory.TRADE_EXECUTION, ErrorSeverity.WARNING),
                    context={"component": "MarketMaker", "fill_data": fill_data}
                )
            else:
                logging.error(error_msg)
                
    def handle_order_update(self, order_update: Dict[str, Any]) -> None:
        """Handle order status update"""
        try:
            order_id = order_update.get('order_id')
            new_status = order_update.get('status')
            
            if not order_id or not new_status:
                return
                
            # Update active orders
            if order_id in self.active_orders:
                if new_status in ['filled', 'cancelled', 'rejected']:
                    del self.active_orders[order_id]
                elif new_status == 'partially_filled':
                    # Update quantity for partially filled order
                    if 'filled_quantity' in order_update:
                        self.active_orders[order_id]['filled_quantity'] = order_update['filled_quantity']
                        
        except Exception as e:
            logging.error(f"Error handling order update: {e}")
            
    def add_symbol(self, symbol: str) -> bool:
        """Add a new symbol for market making"""
        if symbol in self.symbols:
            logging.warning(f"Symbol {symbol} already being market made")
            return False
            
        # Add to symbols list
        self.symbols.append(symbol)
        
        # Initialize components for this symbol
        self.order_books[symbol] = OrderBookState(symbol)
        
        self.inventory_managers[symbol] = InventoryManager(
            symbol, 
            self.config['target_inventory'],
            self.config['max_inventory'],
            self.config['rebalance_threshold']
        )
        
        self.last_order_refresh[symbol] = datetime.now()
        
        logging.info(f"Added symbol {symbol} for market making")
        return True
        
    def remove_symbol(self, symbol: str) -> bool:
        """Remove a symbol from market making"""
        if symbol not in self.symbols:
            logging.warning(f"Symbol {symbol} not being market made")
            return False
            
        # Cancel all orders for this symbol
        self._cancel_symbol_orders(symbol)
        
        # Remove from symbols list
        self.symbols.remove(symbol)
        
        # Clean up components
        if symbol in self.order_books:
            del self.order_books[symbol]
            
        if symbol in self.inventory_managers:
            del self.inventory_managers[symbol]
            
        if symbol in self.last_order_refresh:
            del self.last_order_refresh[symbol]
            
        logging.info(f"Removed symbol {symbol} from market making")
        return True
        
    def _cancel_symbol_orders(self, symbol: str) -> None:
        """Cancel all orders for a specific symbol"""
        order_ids = [
            order_id for order_id, order in self.active_orders.items()
            if order['symbol'] == symbol
        ]
        
        for order_id in order_ids:
            self._cancel_order(order_id)
            
    def update_config(self, new_config: Dict[str, Any]) -> None:
        """Update market making configuration"""
        # Update config
        self.config.update(new_config)
        
        # Update spread optimizer
        self.spread_optimizer = SpreadOptimizer(
            base_spread_bps=self.config['base_spread_bps'],
            min_spread_bps=self.config['min_spread_bps'],
            max_spread_bps=self.config['max_spread_bps'],
            inventory_impact_factor=self.config['inventory_risk_factor'],
            volatility_impact_factor=1.0
        )
        
        # Update inventory managers
        for symbol, manager in self.inventory_managers.items():
            manager.target_inventory = self.config['target_inventory']
            manager.max_inventory = self.config['max_inventory']
            manager.rebalance_threshold = self.config['rebalance_threshold']
            
        logging.info("Market maker configuration updated")
        
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics for the market maker"""
        metrics = {
            'start_time': self.start_time,
            'run_time_seconds': (datetime.now() - self.start_time).total_seconds() if self.start_time else 0,
            'symbols': self.symbols,
            'total_trades': len(self.trades),
            'total_filled_volume': self.total_filled_volume,
            'total_fees': self.total_fees,
            'active_orders_count': len(self.active_orders),
            'inventory': {
                symbol: manager.current_inventory 
                for symbol, manager in self.inventory_managers.items()
            },
            'pnl': {
                symbol: {
                    'unrealized_pnl': manager.unrealized_pnl,
                    'realized_pnl': manager.realized_pnl,
                    'total_pnl': manager.unrealized_pnl + manager.realized_pnl
                }
                for symbol, manager in self.inventory_managers.items()
            },
            'spreads': {
                symbol: self.spread_optimizer.get_spread_statistics(symbol)
                for symbol in self.symbols
            }
        }
        
        # Calculate total PnL
        total_pnl = sum(
            manager.unrealized_pnl + manager.realized_pnl
            for manager in self.inventory_managers.values()
        )
        
        metrics['total_pnl'] = total_pnl
        
        # Calculate PnL as percentage of volume
        if self.total_filled_volume > 0:
            metrics['pnl_percentage'] = (total_pnl / self.total_filled_volume) * 100
        else:
            metrics['pnl_percentage'] = 0.0
            
        return metrics
        
    def _log_performance(self) -> None:
        """Log performance metrics"""
        metrics = self.get_performance_metrics()
        
        runtime = timedelta(seconds=int(metrics['run_time_seconds']))
        
        performance_log = (
            f"Market Maker Performance Summary\n"
            f"-------------------------------\n"
            f"Run time: {runtime}\n"
            f"Symbols: {', '.join(self.symbols)}\n"
            f"Total trades: {metrics['total_trades']}\n"
            f"Total volume: {metrics['total_filled_volume']:.2f}\n"
            f"Total fees: {metrics['total_fees']:.2f}\n"
            f"Total PnL: {metrics['total_pnl']:.2f}\n"
            f"PnL percentage: {metrics['pnl_percentage']:.4f}%\n"
            f"-------------------------------\n"
        )
        
        # Add per-symbol metrics
        for symbol in self.symbols:
            inventory = metrics['inventory'].get(symbol, 0)
            pnl = metrics['pnl'].get(symbol, {})
            spread = metrics['spreads'].get(symbol, {})
            
            performance_log += (
                f"Symbol: {symbol}\n"
                f"  Inventory: {inventory:.6f}\n"
                f"  Unrealized PnL: {pnl.get('unrealized_pnl', 0):.2f}\n"
                f"  Realized PnL: {pnl.get('realized_pnl', 0):.2f}\n"
                f"  Total PnL: {pnl.get('total_pnl', 0):.2f}\n"
                f"  Avg spread (bps): {spread.get('average_spread_bps', 0):.2f}\n"
                f"-------------------------------\n"
            )
            
        # Log to dedicated market maker log
        self.log(logging.INFO, performance_log)
        
        # Also log to main logger
        logging.info(f"Market maker performance summary: {len(self.trades)} trades, PnL: {metrics['total_pnl']:.2f}")
        
    def export_performance_data(self, filepath: str = None) -> bool:
        """Export performance data to file"""
        try:
            # Generate filename if not provided
            if not filepath:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filepath = f"market_maker_performance_{timestamp}.json"
                
            # Get metrics
            metrics = self.get_performance_metrics()
            
            # Add detailed trade history
            metrics['trades'] = self.trades
            
            # Export to file
            with open(filepath, 'w') as f:
                json.dump(metrics, f, indent=4, default=str)
                
            logging.info(f"Exported market maker performance data to {filepath}")
            return True
            
        except Exception as e:
            logging.error(f"Error exporting performance data: {e}")
            return False
            
    def get_inventory_status(self) -> Dict[str, Dict[str, Any]]:
        """Get current inventory status for all symbols"""
        return {
            symbol: manager.get_risk_exposure()
            for symbol, manager in self.inventory_managers.items()
        }
        
    def get_order_book_status(self) -> Dict[str, Dict[str, Any]]:
        """Get current order book status for all symbols"""
        return {
            symbol: {
                "mid_price": book.mid_price,
                "spread": book.spread,
                "imbalance": book.imbalance,
                "volatility": book.volatility,
                "bid_depth": book.bid_depth,
                "ask_depth": book.ask_depth,
                "last_update": book.last_update.isoformat(),
                "is_stale": book.is_stale()
            }
            for symbol, book in self.order_books.items()
        }
        
    def get_active_orders_count(self) -> int:
        """Get count of active orders"""
        return len(self.active_orders)

# Factory function to create a market maker instance
def create_market_maker(trading_system, symbols=None, config=None):
    """Create a market maker instance"""
    market_maker = MarketMaker(trading_system, symbols, config)
    return market_maker
