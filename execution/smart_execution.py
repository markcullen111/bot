# execution/smart_execution.py

import time
import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union, Any, Callable
from datetime import datetime, timedelta
import threading
import json
import traceback
from dataclasses import dataclass
from enum import Enum
import random

try:
    # Import error handling if available
    from core.error_handling import safe_execute, ErrorCategory, ErrorSeverity
    HAVE_ERROR_HANDLING = True
except ImportError:
    HAVE_ERROR_HANDLING = False
    logging.info("Error handling module not available in SmartExecution. Using basic error handling.")

# Execution algorithms
class ExecutionAlgorithm(Enum):
    """Available execution algorithms"""
    MARKET = "market"               # Simple market order execution
    LIMIT = "limit"                 # Simple limit order execution
    TWAP = "twap"                   # Time-Weighted Average Price
    VWAP = "vwap"                   # Volume-Weighted Average Price
    ICEBERG = "iceberg"             # Iceberg/hidden orders
    PEG = "peg"                     # Pegged orders following market
    ADAPTIVE = "adaptive"           # Dynamic adaptation based on market conditions
    SMART = "smart"                 # Smart routing and execution

@dataclass
class OrderBookLevel:
    """Represents a level in the order book"""
    price: float
    quantity: float
    count: Optional[int] = None

@dataclass
class OrderBook:
    """Structured order book representation"""
    bids: List[OrderBookLevel]
    asks: List[OrderBookLevel]
    timestamp: float

    @classmethod
    def from_dict(cls, data: Dict) -> 'OrderBook':
        """Create OrderBook from dictionary data"""
        if not all(k in data for k in ['bids', 'asks']):
            raise ValueError("Invalid order book data: Missing bid or ask data")
            
        # Convert raw data to structured format
        bids = [OrderBookLevel(price=float(b[0]), quantity=float(b[1]), 
                count=int(b[2]) if len(b) > 2 else None) 
                for b in data['bids']]
        asks = [OrderBookLevel(price=float(a[0]), quantity=float(a[1]),
                count=int(a[2]) if len(a) > 2 else None) 
                for a in data['asks']]
                
        timestamp = data.get('timestamp', time.time())
        return cls(bids=bids, asks=asks, timestamp=timestamp)

    def get_mid_price(self) -> float:
        """Calculate mid price from order book"""
        if not self.bids or not self.asks:
            raise ValueError("Cannot calculate mid price: Empty order book")
        return (self.bids[0].price + self.asks[0].price) / 2
        
    def get_bid_ask_spread(self) -> float:
        """Calculate bid-ask spread from order book"""
        if not self.bids or not self.asks:
            raise ValueError("Cannot calculate spread: Empty order book")
        return self.asks[0].price - self.bids[0].price
        
    def get_bid_ask_spread_pct(self) -> float:
        """Calculate bid-ask spread as percentage of mid price"""
        mid_price = self.get_mid_price()
        spread = self.get_bid_ask_spread()
        return (spread / mid_price) * 100
        
    def get_imbalance(self) -> float:
        """Calculate order book imbalance (-1 to 1)"""
        bid_volume = sum(level.quantity for level in self.bids[:5])
        ask_volume = sum(level.quantity for level in self.asks[:5])
        total_volume = bid_volume + ask_volume
        
        if total_volume == 0:
            return 0
            
        return (bid_volume - ask_volume) / total_volume
        
    def get_depth(self, price_range_pct: float = 0.01) -> Tuple[float, float]:
        """Calculate order book depth within price range"""
        mid_price = self.get_mid_price()
        price_range = mid_price * price_range_pct
        
        lower_bound = mid_price - price_range
        upper_bound = mid_price + price_range
        
        bid_depth = sum(level.quantity for level in self.bids 
                      if level.price >= lower_bound)
        ask_depth = sum(level.quantity for level in self.asks 
                      if level.price <= upper_bound)
                      
        return bid_depth, ask_depth

class MarketConditions:
    """Analyzes market conditions for execution optimization"""
    
    def __init__(self, symbol: str, order_book: OrderBook, 
                recent_trades: Optional[List[Dict]] = None,
                price_history: Optional[pd.DataFrame] = None):
        self.symbol = symbol
        self.order_book = order_book
        self.recent_trades = recent_trades or []
        self.price_history = price_history
        
    def get_volatility(self, window: int = 20) -> float:
        """Calculate recent price volatility"""
        if self.price_history is None or len(self.price_history) < window:
            # Fallback estimation based on bid-ask spread
            return self.order_book.get_bid_ask_spread_pct() / 100
            
        returns = self.price_history['close'].pct_change().dropna()
        if len(returns) < 2:
            return 0.01  # Default value
            
        return returns.std() * np.sqrt(window)
        
    def get_trend(self, short_window: int = 20, long_window: int = 50) -> float:
        """Determine market trend strength (-1 to 1)"""
        if self.price_history is None or len(self.price_history) < long_window:
            # Fallback to order book imbalance
            return self.order_book.get_imbalance()
            
        if 'close' not in self.price_history.columns:
            return 0  # No data available
            
        # Calculate EMAs
        short_ema = self.price_history['close'].ewm(span=short_window, adjust=False).mean()
        long_ema = self.price_history['close'].ewm(span=long_window, adjust=False).mean()
        
        if len(short_ema) == 0 or len(long_ema) == 0:
            return 0
            
        # Calculate trend strength
        latest_short = short_ema.iloc[-1]
        latest_long = long_ema.iloc[-1]
        
        # Normalize to -1 to 1 range
        trend = (latest_short / latest_long) - 1
        
        # Clamp to reasonable range
        return max(-1, min(1, trend * 10))
        
    def get_liquidity(self) -> float:
        """Assess market liquidity (0 to 1)"""
        # Calculate book depth
        bid_depth, ask_depth = self.order_book.get_depth(0.01)
        
        # Get recent volume if available
        recent_volume = 0
        if self.price_history is not None and 'volume' in self.price_history.columns:
            recent_volume = self.price_history['volume'].mean()
            
        # Combine metrics (normalized)
        spread_factor = 1 / (1 + self.order_book.get_bid_ask_spread_pct())
        depth_factor = (bid_depth + ask_depth) / 100  # Normalize based on expected depth
        
        # Combine factors with weights
        liquidity = (0.4 * spread_factor) + (0.6 * min(1, depth_factor))
        
        return max(0, min(1, liquidity))
        
    def get_market_impact_estimate(self, order_size: float, side: str) -> float:
        """Estimate market impact of order (price movement %)"""
        mid_price = self.order_book.get_mid_price()
        
        if side.lower() == 'buy':
            # Calculate how far up the book we need to go
            remaining_size = order_size
            avg_execution_price = 0
            total_filled = 0
            
            for level in self.order_book.asks:
                filled = min(remaining_size, level.quantity)
                avg_execution_price += level.price * filled
                total_filled += filled
                remaining_size -= filled
                
                if remaining_size <= 0:
                    break
            
            if total_filled > 0:
                avg_execution_price /= total_filled
                return (avg_execution_price / mid_price) - 1
            return 0
            
        elif side.lower() == 'sell':
            # Calculate how far down the book we need to go
            remaining_size = order_size
            avg_execution_price = 0
            total_filled = 0
            
            for level in self.order_book.bids:
                filled = min(remaining_size, level.quantity)
                avg_execution_price += level.price * filled
                total_filled += filled
                remaining_size -= filled
                
                if remaining_size <= 0:
                    break
            
            if total_filled > 0:
                avg_execution_price /= total_filled
                return 1 - (avg_execution_price / mid_price)
            return 0
            
        return 0

class ExecutionResult:
    """Structured result of order execution"""
    
    def __init__(self, status: str, order_id: Optional[str] = None,
                symbol: Optional[str] = None, side: Optional[str] = None,
                quantity: Optional[float] = None, filled: Optional[float] = None,
                avg_price: Optional[float] = None, target_price: Optional[float] = None,
                slippage: Optional[float] = None, execution_time: Optional[float] = None,
                fees: Optional[float] = None, error: Optional[str] = None,
                details: Optional[Dict] = None):
        self.status = status
        self.order_id = order_id
        self.symbol = symbol
        self.side = side
        self.quantity = quantity
        self.filled = filled or 0
        self.avg_price = avg_price
        self.target_price = target_price
        self.slippage = slippage
        self.execution_time = execution_time
        self.fees = fees or 0
        self.error = error
        self.details = details or {}
        self.timestamp = time.time()
        
    @property
    def is_successful(self) -> bool:
        """Check if execution was successful"""
        return self.status in ['executed', 'partial']
        
    @property
    def fill_percentage(self) -> float:
        """Calculate fill percentage"""
        if not self.quantity or self.quantity == 0:
            return 0
        return (self.filled / self.quantity) * 100
        
    @property
    def slippage_percentage(self) -> float:
        """Calculate slippage as percentage"""
        if not self.target_price or not self.avg_price or self.target_price == 0:
            return 0
            
        if self.side == 'buy':
            return ((self.avg_price / self.target_price) - 1) * 100
        else:  # sell
            return ((self.target_price / self.avg_price) - 1) * 100
            
    def to_dict(self) -> Dict:
        """Convert to dictionary representation"""
        return {
            'status': self.status,
            'order_id': self.order_id,
            'symbol': self.symbol,
            'side': self.side,
            'quantity': self.quantity,
            'filled': self.filled,
            'avg_price': self.avg_price,
            'target_price': self.target_price,
            'slippage': self.slippage,
            'slippage_percentage': self.slippage_percentage,
            'execution_time': self.execution_time,
            'fees': self.fees,
            'fill_percentage': self.fill_percentage,
            'error': self.error,
            'details': self.details,
            'timestamp': self.timestamp
        }
        
    @classmethod
    def from_dict(cls, data: Dict) -> 'ExecutionResult':
        """Create from dictionary"""
        return cls(
            status=data.get('status', 'unknown'),
            order_id=data.get('order_id'),
            symbol=data.get('symbol'),
            side=data.get('side'),
            quantity=data.get('quantity'),
            filled=data.get('filled'),
            avg_price=data.get('avg_price'),
            target_price=data.get('target_price'),
            slippage=data.get('slippage'),
            execution_time=data.get('execution_time'),
            fees=data.get('fees'),
            error=data.get('error'),
            details=data.get('details', {})
        )
    
    @classmethod
    def create_error_result(cls, error_message: str, symbol: Optional[str] = None,
                          side: Optional[str] = None, quantity: Optional[float] = None,
                          target_price: Optional[float] = None) -> 'ExecutionResult':
        """Create an error result"""
        return cls(
            status='failed',
            symbol=symbol,
            side=side,
            quantity=quantity,
            target_price=target_price,
            error=error_message
        )

class ExecutionContext:
    """Execution context with market data and constraints"""
    
    def __init__(self, symbol: str, side: str, quantity: float, target_price: Optional[float] = None,
                 order_book: Optional[OrderBook] = None, price_history: Optional[pd.DataFrame] = None,
                 max_slippage: float = 0.001, time_limit: Optional[float] = None,
                 algorithm: ExecutionAlgorithm = ExecutionAlgorithm.SMART,
                 client_order_id: Optional[str] = None, exchange: Optional[str] = None,
                 parameters: Optional[Dict] = None):
        self.symbol = symbol
        self.side = side.lower()
        self.quantity = quantity
        self.target_price = target_price
        self.order_book = order_book
        self.price_history = price_history
        self.max_slippage = max_slippage
        self.time_limit = time_limit
        self.algorithm = algorithm
        self.client_order_id = client_order_id or f"smart_{int(time.time()*1000)}"
        self.exchange = exchange
        self.parameters = parameters or {}
        self.start_time = time.time()
        
        # Validation
        if side.lower() not in ['buy', 'sell']:
            raise ValueError(f"Invalid side: {side}. Must be 'buy' or 'sell'")
            
        if quantity <= 0:
            raise ValueError(f"Invalid quantity: {quantity}. Must be positive")
            
    def get_market_conditions(self) -> MarketConditions:
        """Get market conditions analysis"""
        return MarketConditions(
            symbol=self.symbol,
            order_book=self.order_book,
            price_history=self.price_history
        )
        
    def get_time_remaining(self) -> Optional[float]:
        """Get remaining execution time in seconds"""
        if not self.time_limit:
            return None
            
        elapsed = time.time() - self.start_time
        remaining = self.time_limit - elapsed
        
        return max(0, remaining)
        
    def has_time_expired(self) -> bool:
        """Check if execution time limit has expired"""
        if not self.time_limit:
            return False
            
        return time.time() - self.start_time > self.time_limit
        
    def calculate_slippage(self, execution_price: float) -> float:
        """Calculate slippage percentage from target price"""
        if not self.target_price or self.target_price == 0:
            return 0
            
        if self.side == 'buy':
            return (execution_price - self.target_price) / self.target_price
        else:  # sell
            return (self.target_price - execution_price) / self.target_price
            
    def is_slippage_acceptable(self, execution_price: float) -> bool:
        """Check if slippage is within acceptable limits"""
        slippage = self.calculate_slippage(execution_price)
        return slippage <= self.max_slippage

class SmartExecution:
    """Advanced order execution with optimal execution algorithms and slippage control"""
    
    def __init__(self, exchange_connector=None, order_router=None, 
                slippage_tolerance: float = 0.002,
                max_retries: int = 3, retry_delay: float = 0.5,
                log_level: int = logging.INFO):
        """
        Initialize SmartExecution with advanced execution capabilities.

        Args:
            exchange_connector: Connector to exchange API
            order_router: Optional order router for multi-exchange execution
            slippage_tolerance: Maximum allowed slippage before canceling an order
            max_retries: Maximum number of retry attempts for failed executions
            retry_delay: Delay between retry attempts in seconds
            log_level: Logging level
        """
        self.exchange_connector = exchange_connector
        self.order_router = order_router
        self.slippage_tolerance = slippage_tolerance
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        
        # Setup logging
        self.logger = logging.getLogger('SmartExecution')
        self.logger.setLevel(log_level)
        
        # Performance metrics
        self.execution_stats = {
            'total_executions': 0,
            'successful_executions': 0,
            'failed_executions': 0,
            'avg_slippage': 0,
            'avg_execution_time': 0
        }
        self.execution_history = []
        
        # Execution algorithms
        self.algorithms = {
            ExecutionAlgorithm.MARKET: self._execute_market_order,
            ExecutionAlgorithm.LIMIT: self._execute_limit_order,
            ExecutionAlgorithm.TWAP: self._execute_twap,
            ExecutionAlgorithm.VWAP: self._execute_vwap,
            ExecutionAlgorithm.ICEBERG: self._execute_iceberg,
            ExecutionAlgorithm.PEG: self._execute_pegged_order,
            ExecutionAlgorithm.ADAPTIVE: self._execute_adaptive,
            ExecutionAlgorithm.SMART: self._execute_smart
        }
        
        # Thread lock for thread safety
        self._lock = threading.RLock()
        
        self.logger.info("SmartExecution initialized successfully")

    def execute_order(self, context: ExecutionContext) -> ExecutionResult:
        """
        Execute an order using the specified algorithm with optimal execution.
        
        Args:
            context: Execution context with order details and constraints
            
        Returns:
            ExecutionResult: Detailed result of the execution
        """
        start_time = time.time()
        
        try:
            # Validate input
            self._validate_execution_context(context)
            
            # Select algorithm
            algorithm = context.algorithm
            if not isinstance(algorithm, ExecutionAlgorithm):
                # Handle string input
                try:
                    algorithm = ExecutionAlgorithm(algorithm)
                except ValueError:
                    self.logger.warning(f"Invalid algorithm: {algorithm}, using SMART")
                    algorithm = ExecutionAlgorithm.SMART
            
            # Get execution function
            execute_func = self.algorithms.get(algorithm, self._execute_smart)
            
            # Execute with retry logic
            for attempt in range(self.max_retries):
                try:
                    # Execute order using selected algorithm
                    self.logger.info(f"Executing {context.side} order for {context.quantity} {context.symbol} using {algorithm.value} algorithm (attempt {attempt+1})")
                    result = execute_func(context)
                    
                    if result.is_successful:
                        break
                        
                    # If execution failed but it's not a critical error, retry
                    if attempt < self.max_retries - 1 and 'retry' in result.details and result.details['retry']:
                        self.logger.warning(f"Execution attempt {attempt+1} failed, retrying in {self.retry_delay}s: {result.error}")
                        time.sleep(self.retry_delay)
                    else:
                        break
                        
                except Exception as e:
                    if attempt < self.max_retries - 1:
                        self.logger.error(f"Execution error on attempt {attempt+1}, retrying: {str(e)}")
                        time.sleep(self.retry_delay)
                    else:
                        raise
            
            # Update execution time
            execution_time = time.time() - start_time
            result.execution_time = execution_time
            
            # Update statistics
            with self._lock:
                self.execution_stats['total_executions'] += 1
                
                if result.is_successful:
                    self.execution_stats['successful_executions'] += 1
                    # Update average metrics
                    n = self.execution_stats['successful_executions']
                    self.execution_stats['avg_slippage'] = ((n - 1) * self.execution_stats['avg_slippage'] + (result.slippage or 0)) / n
                    self.execution_stats['avg_execution_time'] = ((n - 1) * self.execution_stats['avg_execution_time'] + execution_time) / n
                else:
                    self.execution_stats['failed_executions'] += 1
                
                # Add to history (limited to last 100)
                self.execution_history.append(result.to_dict())
                if len(self.execution_history) > 100:
                    self.execution_history.pop(0)
            
            return result
            
        except Exception as e:
            self.logger.error(f"Order execution failed: {str(e)}\n{traceback.format_exc()}")
            
            # Update statistics
            with self._lock:
                self.execution_stats['total_executions'] += 1
                self.execution_stats['failed_executions'] += 1
            
            # Create error result
            return ExecutionResult.create_error_result(
                error_message=str(e),
                symbol=context.symbol,
                side=context.side,
                quantity=context.quantity,
                target_price=context.target_price
            )

    def _validate_execution_context(self, context: ExecutionContext) -> None:
        """Validate execution context before proceeding"""
        # Check if exchange connector is available
        if not self.exchange_connector and not hasattr(self, 'mock_execute_order'):
            raise ValueError("No exchange connector available for execution")
            
        # Check if we have necessary order book data
        if context.order_book is None and context.algorithm not in [ExecutionAlgorithm.MARKET, ExecutionAlgorithm.LIMIT]:
            raise ValueError(f"Order book data required for {context.algorithm.value} algorithm")
            
        # If no target price is provided, try to determine it from order book
        if context.target_price is None and context.order_book is not None:
            mid_price = context.order_book.get_mid_price()
            self.logger.info(f"No target price provided, using mid price: {mid_price}")
            context.target_price = mid_price

    def _execute_market_order(self, context: ExecutionContext) -> ExecutionResult:
        """Execute a simple market order"""
        try:
            # Determine price if not provided
            price = None
            if context.order_book:
                # Use best available price from order book
                if context.side == 'buy':
                    price = context.order_book.asks[0].price
                else:
                    price = context.order_book.bids[0].price
            
            # Execute order
            if self.exchange_connector:
                order_result = self.exchange_connector.create_market_order(
                    symbol=context.symbol,
                    side=context.side,
                    quantity=context.quantity,
                    client_order_id=context.client_order_id
                )
                
                # Process order result
                if order_result.get('status') == 'filled':
                    return ExecutionResult(
                        status='executed',
                        order_id=order_result.get('id'),
                        symbol=context.symbol,
                        side=context.side,
                        quantity=context.quantity,
                        filled=float(order_result.get('filled', context.quantity)),
                        avg_price=float(order_result.get('price')),
                        target_price=context.target_price,
                        slippage=context.calculate_slippage(float(order_result.get('price'))) if context.target_price else None,
                        fees=float(order_result.get('fee', 0)),
                        details=order_result
                    )
                else:
                    return ExecutionResult(
                        status='failed',
                        symbol=context.symbol,
                        side=context.side,
                        quantity=context.quantity,
                        target_price=context.target_price,
                        error=f"Market order failed: {order_result.get('message', 'Unknown error')}",
                        details={'retry': True, 'order_result': order_result}
                    )
            else:
                # Mock execution for testing
                mock_price = context.target_price
                if price and random.random() > 0.7:  # Simulate some slippage
                    slippage_factor = 1 + (random.random() * 0.002 * (-1 if context.side == 'sell' else 1))
                    mock_price = price * slippage_factor
                
                self.logger.info(f"Mock executing market order: {context.side} {context.quantity} {context.symbol} @ ~{mock_price}")
                
                return ExecutionResult(
                    status='executed',
                    order_id=f"mock_{int(time.time()*1000)}",
                    symbol=context.symbol,
                    side=context.side,
                    quantity=context.quantity,
                    filled=context.quantity,
                    avg_price=mock_price,
                    target_price=context.target_price,
                    slippage=context.calculate_slippage(mock_price) if context.target_price else None,
                    fees=context.quantity * mock_price * 0.001  # Mock 0.1% fee
                )
                
        except Exception as e:
            self.logger.error(f"Market order execution error: {str(e)}")
            return ExecutionResult.create_error_result(
                error_message=f"Market order execution failed: {str(e)}",
                symbol=context.symbol,
                side=context.side,
                quantity=context.quantity,
                target_price=context.target_price
            )

    def _execute_limit_order(self, context: ExecutionContext) -> ExecutionResult:
        """Execute a limit order with price constraints"""
        try:
            # Determine limit price
            limit_price = context.target_price
            
            # Adjust price based on context if needed
            if not limit_price and context.order_book:
                if context.side == 'buy':
                    # For buy, use price slightly higher than best ask
                    limit_price = context.order_book.asks[0].price * 1.001
                else:
                    # For sell, use price slightly lower than best bid
                    limit_price = context.order_book.bids[0].price * 0.999
            
            if not limit_price:
                raise ValueError("Cannot determine limit price, no target_price or order_book provided")
            
            # Execute order
            if self.exchange_connector:
                order_result = self.exchange_connector.create_limit_order(
                    symbol=context.symbol,
                    side=context.side,
                    quantity=context.quantity,
                    price=limit_price,
                    client_order_id=context.client_order_id,
                    time_in_force=context.parameters.get('time_in_force', 'GTC')  # Default to GTC
                )
                
                # Wait for fill if time_limit provided
                if context.time_limit:
                    filled_quantity = 0
                    avg_price = 0
                    timeout = time.time() + context.time_limit
                    
                    while time.time() < timeout and filled_quantity < context.quantity:
                        # Check order status
                        order_status = self.exchange_connector.get_order_status(
                            symbol=context.symbol,
                            order_id=order_result['id']
                        )
                        
                        if order_status['status'] in ['filled', 'closed']:
                            filled_quantity = float(order_status['filled'])
                            avg_price = float(order_status['price'])
                            break
                            
                        elif order_status['status'] == 'canceled':
                            break
                            
                        # Check partial fills
                        if float(order_status['filled']) > filled_quantity:
                            filled_quantity = float(order_status['filled'])
                            avg_price = float(order_status['price'])
                            
                            # If partial fill meets our needs, break
                            if filled_quantity / context.quantity >= context.parameters.get('min_fill_ratio', 0.9):
                                break
                                
                        # Sleep to prevent API rate limiting
                        time.sleep(0.5)
                    
                    # If not fully filled after time limit, cancel remaining
                    if filled_quantity < context.quantity:
                        self.exchange_connector.cancel_order(
                            symbol=context.symbol,
                            order_id=order_result['id']
                        )
                        
                        status = 'partial' if filled_quantity > 0 else 'failed'
                        
                        return ExecutionResult(
                            status=status,
                            order_id=order_result['id'],
                            symbol=context.symbol,
                            side=context.side,
                            quantity=context.quantity,
                            filled=filled_quantity,
                            avg_price=avg_price if filled_quantity > 0 else None,
                            target_price=context.target_price,
                            slippage=context.calculate_slippage(avg_price) if filled_quantity > 0 and context.target_price else None,
                            error="Order partially filled" if filled_quantity > 0 else "Order not filled within time limit",
                            details={'original_order': order_result}
                        )
                        
                    else:
                        # Order fully filled
                        return ExecutionResult(
                            status='executed',
                            order_id=order_result['id'],
                            symbol=context.symbol,
                            side=context.side,
                            quantity=context.quantity,
                            filled=filled_quantity,
                            avg_price=avg_price,
                            target_price=context.target_price,
                            slippage=context.calculate_slippage(avg_price) if context.target_price else None,
                            details={'original_order': order_result}
                        )
                
                # If no time limit, just return the initial order result
                return ExecutionResult(
                    status='pending',
                    order_id=order_result['id'],
                    symbol=context.symbol,
                    side=context.side,
                    quantity=context.quantity,
                    target_price=context.target_price,
                    details={'original_order': order_result}
                )
            else:
                # Mock execution for testing
                self.logger.info(f"Mock executing limit order: {context.side} {context.quantity} {context.symbol} @ {limit_price}")
                
                # Simulate fill probability based on price aggressiveness
                fill_probability = 0.7  # Base probability
                
                if context.order_book:
                    mid_price = context.order_book.get_mid_price()
                    
                    # Adjust fill probability based on price aggressiveness
                    if context.side == 'buy':
                        # Higher probability if buying above mid price
                        price_factor = limit_price / mid_price
                        fill_probability *= min(1.5, price_factor)
                    else:
                        # Higher probability if selling below mid price
                        price_factor = mid_price / limit_price
                        fill_probability *= min(1.5, price_factor)
                
                # Clamp probability
                fill_probability = min(0.95, max(0.1, fill_probability))
                
                # Simulate fill
                if random.random() < fill_probability:
                    # Fully filled
                    return ExecutionResult(
                        status='executed',
                        order_id=f"mock_{int(time.time()*1000)}",
                        symbol=context.symbol,
                        side=context.side,
                        quantity=context.quantity,
                        filled=context.quantity,
                        avg_price=limit_price,
                        target_price=context.target_price,
                        slippage=context.calculate_slippage(limit_price) if context.target_price else None,
                        fees=context.quantity * limit_price * 0.001  # Mock 0.1% fee
                    )
                else:
                    # Partially filled or not filled
                    fill_ratio = random.random() * 0.5  # 0-50% fill
                    filled_qty = context.quantity * fill_ratio
                    
                    if filled_qty > 0:
                        return ExecutionResult(
                            status='partial',
                            order_id=f"mock_{int(time.time()*1000)}",
                            symbol=context.symbol,
                            side=context.side,
                            quantity=context.quantity,
                            filled=filled_qty,
                            avg_price=limit_price,
                            target_price=context.target_price,
                            slippage=context.calculate_slippage(limit_price) if context.target_price else None,
                            fees=filled_qty * limit_price * 0.001,  # Mock 0.1% fee
                            error="Order partially filled"
                        )
                    else:
                        return ExecutionResult(
                            status='failed',
                            symbol=context.symbol,
                            side=context.side,
                            quantity=context.quantity,
                            target_price=context.target_price,
                            error="Order not filled within time limit",
                            details={'retry': True}
                        )
                
        except Exception as e:
            self.logger.error(f"Limit order execution error: {str(e)}")
            return ExecutionResult.create_error_result(
                error_message=f"Limit order execution failed: {str(e)}",
                symbol=context.symbol,
                side=context.side,
                quantity=context.quantity,
                target_price=context.target_price
            )
    
    def _execute_twap(self, context: ExecutionContext) -> ExecutionResult:
        """Execute order using Time-Weighted Average Price algorithm"""
        try:
            # Validate TWAP-specific parameters
            if not context.time_limit:
                time_limit = context.parameters.get('time_limit', 300)  # Default 5 min
            else:
                time_limit = context.time_limit
                
            num_slices = context.parameters.get('num_slices', 5)
            randomize = context.parameters.get('randomize', True)
            
            # Calculate time and size per slice
            time_per_slice = time_limit / num_slices
            base_size_per_slice = context.quantity / num_slices
            
            # Track execution
            total_filled = 0
            total_cost = 0
            all_order_ids = []
            
            # Execute slices
            start_time = time.time()
            end_time = start_time + time_limit
            
            for i in range(num_slices):
                # Check if we still have time
                if time.time() >= end_time:
                    break
                    
                # Calculate slice size (optionally randomized)
                if randomize and i < num_slices - 1:
                    # Randomize size except for last slice
                    remaining = context.quantity - total_filled
                    slices_left = num_slices - i
                    
                    if slices_left > 1:
                        max_size = remaining - (slices_left - 1) * (base_size_per_slice * 0.5)
                        min_size = base_size_per_slice * 0.5
                        
                        slice_size = min(max_size, max(min_size, 
                                                     random.uniform(min_size, base_size_per_slice * 1.5)))
                    else:
                        slice_size = remaining
                else:
                    # Use even slices
                    remaining = context.quantity - total_filled
                    slices_left = num_slices - i
                    slice_size = remaining / slices_left
                
                # Round to appropriate precision
                slice_size = round(slice_size, 8)
                
                if slice_size <= 0:
                    continue
                
                # Execute slice using limit orders
                slice_context = ExecutionContext(
                    symbol=context.symbol,
                    side=context.side,
                    quantity=slice_size,
                    target_price=context.target_price,
                    order_book=context.order_book,
                    max_slippage=context.max_slippage,
                    time_limit=time_per_slice * 0.95,  # Slight buffer
                    algorithm=ExecutionAlgorithm.LIMIT,
                    client_order_id=f"{context.client_order_id}_slice_{i+1}",
                    exchange=context.exchange,
                    parameters=context.parameters
                )
                
                slice_result = self._execute_limit_order(slice_context)
                
                # Update totals
                if slice_result.is_successful:
                    total_filled += slice_result.filled
                    
                    if slice_result.avg_price:
                        total_cost += slice_result.filled * slice_result.avg_price
                    
                    if slice_result.order_id:
                        all_order_ids.append(slice_result.order_id)
                
                # Wait until next slice time if not the last slice
                if i < num_slices - 1:
                    next_slice_time = start_time + (i + 1) * time_per_slice
                    wait_time = next_slice_time - time.time()
                    
                    if wait_time > 0:
                        time.sleep(wait_time)
            
            # Calculate results
            if total_filled == 0:
                return ExecutionResult(
                    status='failed',
                    symbol=context.symbol,
                    side=context.side,
                    quantity=context.quantity,
                    target_price=context.target_price,
                    error="TWAP execution failed: No quantity filled",
                    details={'order_ids': all_order_ids}
                )
            
            avg_price = total_cost / total_filled if total_filled > 0 else None
            
            status = 'executed' if total_filled >= context.quantity * 0.99 else 'partial'
            
            return ExecutionResult(
                status=status,
                symbol=context.symbol,
                side=context.side,
                quantity=context.quantity,
                filled=total_filled,
                avg_price=avg_price,
                target_price=context.target_price,
                slippage=context.calculate_slippage(avg_price) if avg_price and context.target_price else None,
                details={
                    'algorithm': 'TWAP',
                    'num_slices': num_slices,
                    'time_limit': time_limit,
                    'order_ids': all_order_ids
                }
            )
            
        except Exception as e:
            self.logger.error(f"TWAP execution error: {str(e)}")
            return ExecutionResult.create_error_result(
                error_message=f"TWAP execution failed: {str(e)}",
                symbol=context.symbol,
                side=context.side,
                quantity=context.quantity,
                target_price=context.target_price
            )

    def _execute_vwap(self, context: ExecutionContext) -> ExecutionResult:
        """Execute order using Volume-Weighted Average Price algorithm"""
        try:
            # This requires historical volume profile data
            if context.price_history is None or 'volume' not in context.price_history.columns:
                self.logger.warning("VWAP execution requires volume data, falling back to TWAP")
                return self._execute_twap(context)
                
            # Get parameters
            if not context.time_limit:
                time_limit = context.parameters.get('time_limit', 3600)  # Default 1 hour
            else:
                time_limit = context.time_limit
                
            num_slices = context.parameters.get('num_slices', 10)
            
            # Calculate volume profile (typical 24h pattern)
            context.price_history['hour'] = pd.to_datetime(context.price_history.index).hour
            volume_profile = context.price_history.groupby('hour')['volume'].sum()
            total_volume = volume_profile.sum()
            
            # Normalize to create volume distribution
            volume_distribution = volume_profile / total_volume
            
            # Get current hour
            current_hour = datetime.now().hour
            
            # Create slices based on time intervals and expected volume
            time_per_slice = time_limit / num_slices
            
            # Track execution
            total_filled = 0
            total_cost = 0
            all_order_ids = []
            
            # Execute slices
            start_time = time.time()
            end_time = start_time + time_limit
            
            for i in range(num_slices):
                # Check if we still have time
                if time.time() >= end_time:
                    break
                    
                # Calculate target hour for this slice
                slice_hour = (current_hour + i) % 24
                
                # Get volume factor for this hour
                volume_factor = volume_distribution.get(slice_hour, 1/24)
                
                # Calculate slice size based on volume profile
                remaining = context.quantity - total_filled
                slices_left = num_slices - i
                
                if slices_left > 1:
                    # Adjust for volume profile, but ensure minimum slice size
                    min_slice = remaining * 0.5 / slices_left
                    target_slice = remaining * volume_factor * (num_slices / slices_left)
                    slice_size = max(min_slice, min(remaining * 0.5, target_slice))
                else:
                    # Last slice gets all remaining quantity
                    slice_size = remaining
                
                # Round to appropriate precision
                slice_size = round(slice_size, 8)
                
                if slice_size <= 0:
                    continue
                
                # Execute slice
                slice_context = ExecutionContext(
                    symbol=context.symbol,
                    side=context.side,
                    quantity=slice_size,
                    target_price=context.target_price,
                    order_book=context.order_book,
                    max_slippage=context.max_slippage,
                    time_limit=time_per_slice * 0.95,  # Slight buffer
                    algorithm=ExecutionAlgorithm.LIMIT if random.random() < 0.7 else ExecutionAlgorithm.MARKET,  # Mix of limit and market
                    client_order_id=f"{context.client_order_id}_vslice_{i+1}",
                    exchange=context.exchange,
                    parameters=context.parameters
                )
                
                # Choose execution method based on volume
                if random.random() < 0.7:  # 70% limit orders
                    slice_result = self._execute_limit_order(slice_context)
                else:  # 30% market orders
                    slice_context.algorithm = ExecutionAlgorithm.MARKET
                    slice_result = self._execute_market_order(slice_context)
                
                # Update totals
                if slice_result.is_successful:
                    total_filled += slice_result.filled
                    
                    if slice_result.avg_price:
                        total_cost += slice_result.filled * slice_result.avg_price
                    
                    if slice_result.order_id:
                        all_order_ids.append(slice_result.order_id)
                
                # Wait until next slice time if not the last slice
                if i < num_slices - 1:
                    next_slice_time = start_time + (i + 1) * time_per_slice
                    wait_time = next_slice_time - time.time()
                    
                    if wait_time > 0:
                        time.sleep(wait_time)
            
            # Calculate results
            if total_filled == 0:
                return ExecutionResult(
                    status='failed',
                    symbol=context.symbol,
                    side=context.side,
                    quantity=context.quantity,
                    target_price=context.target_price,
                    error="VWAP execution failed: No quantity filled",
                    details={'order_ids': all_order_ids}
                )
            
            avg_price = total_cost / total_filled if total_filled > 0 else None
            
            status = 'executed' if total_filled >= context.quantity * 0.99 else 'partial'
            
            return ExecutionResult(
                status=status,
                symbol=context.symbol,
                side=context.side,
                quantity=context.quantity,
                filled=total_filled,
                avg_price=avg_price,
                target_price=context.target_price,
                slippage=context.calculate_slippage(avg_price) if avg_price and context.target_price else None,
                details={
                    'algorithm': 'VWAP',
                    'num_slices': num_slices,
                    'time_limit': time_limit,
                    'order_ids': all_order_ids
                }
            )
            
        except Exception as e:
            self.logger.error(f"VWAP execution error: {str(e)}")
            return ExecutionResult.create_error_result(
                error_message=f"VWAP execution failed: {str(e)}",
                symbol=context.symbol,
                side=context.side,
                quantity=context.quantity,
                target_price=context.target_price
            )

    def _execute_iceberg(self, context: ExecutionContext) -> ExecutionResult:
        """Execute order using Iceberg/hidden order algorithm"""
        try:
            # Get parameters
            visible_qty = context.parameters.get('visible_quantity', context.quantity * 0.1)
            min_qty = context.parameters.get('min_quantity', context.quantity * 0.05)
            
            # Make sure visible quantity is not too small
            visible_qty = max(min_qty, visible_qty)
            
            # Make sure visible quantity is not larger than total
            visible_qty = min(visible_qty, context.quantity)
            
            # If exchange supports iceberg/hidden orders natively
            if self.exchange_connector and hasattr(self.exchange_connector, 'create_iceberg_order'):
                order_result = self.exchange_connector.create_iceberg_order(
                    symbol=context.symbol,
                    side=context.side,
                    quantity=context.quantity,
                    visible_quantity=visible_qty,
                    price=context.target_price,
                    client_order_id=context.client_order_id
                )
                
                # Wait for fill if time_limit provided
                if context.time_limit:
                    filled_quantity = 0
                    avg_price = 0
                    timeout = time.time() + context.time_limit
                    
                    while time.time() < timeout and filled_quantity < context.quantity:
                        # Check order status
                        order_status = self.exchange_connector.get_order_status(
                            symbol=context.symbol,
                            order_id=order_result['id']
                        )
                        
                        if order_status['status'] in ['filled', 'closed']:
                            filled_quantity = float(order_status['filled'])
                            avg_price = float(order_status['price'])
                            break
                            
                        elif order_status['status'] == 'canceled':
                            break
                            
                        # Check partial fills
                        if float(order_status['filled']) > filled_quantity:
                            filled_quantity = float(order_status['filled'])
                            avg_price = float(order_status['price'])
                            
                        # Sleep to prevent API rate limiting
                        time.sleep(0.5)
                    
                    # If not fully filled after time limit, cancel remaining
                    if filled_quantity < context.quantity:
                        self.exchange_connector.cancel_order(
                            symbol=context.symbol,
                            order_id=order_result['id']
                        )
                        
                        status = 'partial' if filled_quantity > 0 else 'failed'
                        
                        return ExecutionResult(
                            status=status,
                            order_id=order_result['id'],
                            symbol=context.symbol,
                            side=context.side,
                            quantity=context.quantity,
                            filled=filled_quantity,
                            avg_price=avg_price if filled_quantity > 0 else None,
                            target_price=context.target_price,
                            slippage=context.calculate_slippage(avg_price) if filled_quantity > 0 and context.target_price else None,
                            error="Order partially filled" if filled_quantity > 0 else "Order not filled within time limit",
                            details={'original_order': order_result, 'algorithm': 'iceberg'}
                        )
                        
                    else:
                        # Order fully filled
                        return ExecutionResult(
                            status='executed',
                            order_id=order_result['id'],
                            symbol=context.symbol,
                            side=context.side,
                            quantity=context.quantity,
                            filled=filled_quantity,
                            avg_price=avg_price,
                            target_price=context.target_price,
                            slippage=context.calculate_slippage(avg_price) if context.target_price else None,
                            details={'original_order': order_result, 'algorithm': 'iceberg'}
                        )
                        
                # If no time limit, just return the initial order result
                return ExecutionResult(
                    status='pending',
                    order_id=order_result['id'],
                    symbol=context.symbol,
                    side=context.side,
                    quantity=context.quantity,
                    target_price=context.target_price,
                    details={'original_order': order_result, 'algorithm': 'iceberg'}
                )
                
            # If exchange doesn't support iceberg, simulate with multiple orders
            else:
                self.logger.info("Exchange connector doesn't support iceberg orders natively, simulating with multiple orders")
                
                time_limit = context.time_limit or context.parameters.get('time_limit', 300)  # Default 5 min
                
                # Track execution
                total_filled = 0
                total_cost = 0
                all_order_ids = []
                
                # Execute chunks until fully filled or time expires
                start_time = time.time()
                end_time = start_time + time_limit
                
                i = 0
                while total_filled < context.quantity and time.time() < end_time:
                    # Calculate remaining quantity
                    remaining = context.quantity - total_filled
                    
                    # Calculate chunk size (visible portion)
                    chunk_size = min(remaining, visible_qty)
                    
                    # Skip if chunk too small
                    if chunk_size < min_qty:
                        chunk_size = remaining
                    
                    # Execute chunk
                    chunk_context = ExecutionContext(
                        symbol=context.symbol,
                        side=context.side,
                        quantity=chunk_size,
                        target_price=context.target_price,
                        order_book=context.order_book,
                        max_slippage=context.max_slippage,
                        time_limit=min(30, end_time - time.time()),  # Max 30 sec per chunk
                        algorithm=ExecutionAlgorithm.LIMIT,
                        client_order_id=f"{context.client_order_id}_chunk_{i+1}",
                        exchange=context.exchange,
                        parameters=context.parameters
                    )
                    
                    chunk_result = self._execute_limit_order(chunk_context)
                    i += 1
                    
                    # Update totals
                    if chunk_result.is_successful:
                        total_filled += chunk_result.filled
                        
                        if chunk_result.avg_price:
                            total_cost += chunk_result.filled * chunk_result.avg_price
                        
                        if chunk_result.order_id:
                            all_order_ids.append(chunk_result.order_id)
                    
                    # If not successful, wait a bit before retrying
                    elif time.time() < end_time:
                        time.sleep(2)
                
                # Calculate results
                if total_filled == 0:
                    return ExecutionResult(
                        status='failed',
                        symbol=context.symbol,
                        side=context.side,
                        quantity=context.quantity,
                        target_price=context.target_price,
                        error="Iceberg execution failed: No quantity filled",
                        details={'order_ids': all_order_ids, 'algorithm': 'iceberg'}
                    )
                
                avg_price = total_cost / total_filled if total_filled > 0 else None
                
                status = 'executed' if total_filled >= context.quantity * 0.99 else 'partial'
                
                return ExecutionResult(
                    status=status,
                    symbol=context.symbol,
                    side=context.side,
                    quantity=context.quantity,
                    filled=total_filled,
                    avg_price=avg_price,
                    target_price=context.target_price,
                    slippage=context.calculate_slippage(avg_price) if avg_price and context.target_price else None,
                    details={
                        'algorithm': 'iceberg',
                        'visible_quantity': visible_qty,
                        'chunks': i,
                        'order_ids': all_order_ids
                    }
                )
                
        except Exception as e:
            self.logger.error(f"Iceberg execution error: {str(e)}")
            return ExecutionResult.create_error_result(
                error_message=f"Iceberg execution failed: {str(e)}",
                symbol=context.symbol,
                side=context.side,
                quantity=context.quantity,
                target_price=context.target_price
            )

    def _execute_pegged_order(self, context: ExecutionContext) -> ExecutionResult:
        """Execute order using pegged order algorithm (following the market)"""
        try:
            # Get parameters
            peg_offset = context.parameters.get('peg_offset', 0.0001)  # Default 1 pip
            update_interval = context.parameters.get('update_interval', 1.0)  # Default 1 second
            max_deviation = context.parameters.get('max_deviation', 0.01)  # Default 1%
            
            # Use time limit or default
            time_limit = context.time_limit or context.parameters.get('time_limit', 300)  # Default 5 min
            
            # Validate that we have order book
            if not context.order_book:
                raise ValueError("Pegged order requires order book data")
            
            # Track execution
            total_filled = 0
            total_cost = 0
            all_order_ids = []
            active_order_id = None
            
            # Start execution loop
            start_time = time.time()
            end_time = start_time + time_limit
            last_update = 0
            
            while total_filled < context.quantity and time.time() < end_time:
                current_time = time.time()
                
                # Get latest order book (if it has update method)
                if hasattr(context.order_book, 'update') and current_time - last_update >= update_interval:
                    context.order_book.update()
                    last_update = current_time
                
                # Calculate current reference price
                ref_price = context.order_book.get_mid_price()
                
                # Calculate target price with offset
                if context.side == 'buy':
                    target_price = ref_price * (1 + peg_offset)
                else:
                    target_price = ref_price * (1 - peg_offset)
                
                # Check if we have an active order that needs updating
                if active_order_id and self.exchange_connector:
                    order_status = self.exchange_connector.get_order_status(
                        symbol=context.symbol,
                        order_id=active_order_id
                    )
                    
                    # Check if order is still open
                    if order_status['status'] in ['open', 'new', 'partially_filled']:
                        current_price = float(order_status['price'])
                        
                        # Check if price deviation is too large
                        deviation = abs(current_price - target_price) / target_price
                        
                        if deviation > max_deviation:
                            # Cancel existing order
                            self.exchange_connector.cancel_order(
                                symbol=context.symbol,
                                order_id=active_order_id
                            )
                            
                            # Update filled amount
                            new_filled = float(order_status['filled'])
                            if new_filled > total_filled:
                                additional_filled = new_filled - total_filled
                                total_filled = new_filled
                                
                                if 'price' in order_status:
                                    total_cost += additional_filled * float(order_status['price'])
                            
                            active_order_id = None
                            
                        else:
                            # Check if order has been partially filled
                            new_filled = float(order_status['filled'])
                            if new_filled > total_filled:
                                additional_filled = new_filled - total_filled
                                total_filled = new_filled
                                
                                if 'price' in order_status:
                                    total_cost += additional_filled * float(order_status['price'])
                            
                            # Continue with current order
                            time.sleep(update_interval)
                            continue
                            
                    else:
                        # Order is done (filled, canceled, etc.)
                        if order_status['status'] == 'filled':
                            total_filled = float(order_status['filled'])
                            avg_price = float(order_status['price'])
                            total_cost = total_filled * avg_price
                            
                            # Order fully filled, we're done
                            break
                            
                        elif order_status['status'] == 'partially_filled':
                            # Update filled amount
                            total_filled = float(order_status['filled'])
                            
                            if 'price' in order_status:
                                total_cost = total_filled * float(order_status['price'])
                        
                        # Clear active order
                        active_order_id = None
                
                # Create new order if we don't have an active one
                if not active_order_id:
                    # Calculate remaining quantity
                    remaining = context.quantity - total_filled
                    
                    # Create new limit order at target price
                    if self.exchange_connector:
                        order_result = self.exchange_connector.create_limit_order(
                            symbol=context.symbol,
                            side=context.side,
                            quantity=remaining,
                            price=target_price,
                            client_order_id=f"{context.client_order_id}_peg_{int(current_time*1000)}",
                            time_in_force='GTC'  # Good-til-cancelled
                        )
                        
                        active_order_id = order_result['id']
                        all_order_ids.append(active_order_id)
                    else:
                        # Mock execution for testing
                        self.logger.info(f"Mock pegged order: {context.side} {remaining} {context.symbol} @ {target_price}")
                        
                        # Simulate fill probability
                        fill_probability = 0.1  # Low probability per interval
                        
                        if random.random() < fill_probability:
                            # Simulate partial fill
                            fill_ratio = random.random() * 0.3  # 0-30% fill per update
                            filled_qty = remaining * fill_ratio
                            
                            if filled_qty > 0:
                                total_filled += filled_qty
                                total_cost += filled_qty * target_price
                                
                                self.logger.info(f"Mock pegged order partial fill: {filled_qty} @ {target_price}")
                
                # Wait before next update
                time.sleep(update_interval)
            
            # Cancel any remaining order
            if active_order_id and self.exchange_connector:
                try:
                    self.exchange_connector.cancel_order(
                        symbol=context.symbol,
                        order_id=active_order_id
                    )
                    
                    # Get final fill status
                    order_status = self.exchange_connector.get_order_status(
                        symbol=context.symbol,
                        order_id=active_order_id
                    )
                    
                    # Update filled amount
                    total_filled = float(order_status['filled'])
                    
                    if 'price' in order_status and total_filled > 0:
                        total_cost = total_filled * float(order_status['price'])
                    
                except Exception as e:
                    self.logger.warning(f"Error canceling order: {str(e)}")
            
            # Calculate results
            if total_filled == 0:
                return ExecutionResult(
                    status='failed',
                    symbol=context.symbol,
                    side=context.side,
                    quantity=context.quantity,
                    target_price=context.target_price,
                    error="Pegged order execution failed: No quantity filled",
                    details={'order_ids': all_order_ids, 'algorithm': 'peg'}
                )
            
            avg_price = total_cost / total_filled if total_filled > 0 else None
            
            status = 'executed' if total_filled >= context.quantity * 0.99 else 'partial'
            
            return ExecutionResult(
                status=status,
                symbol=context.symbol,
                side=context.side,
                quantity=context.quantity,
                filled=total_filled,
                avg_price=avg_price,
                target_price=context.target_price,
                slippage=context.calculate_slippage(avg_price) if avg_price and context.target_price else None,
                details={
                    'algorithm': 'peg',
                    'peg_offset': peg_offset,
                    'order_ids': all_order_ids
                }
            )
            
        except Exception as e:
            self.logger.error(f"Pegged order execution error: {str(e)}")
            return ExecutionResult.create_error_result(
                error_message=f"Pegged order execution failed: {str(e)}",
                symbol=context.symbol,
                side=context.side,
                quantity=context.quantity,
                target_price=context.target_price
            )

    def _execute_adaptive(self, context: ExecutionContext) -> ExecutionResult:
        """Execute order using adaptive algorithm that adjusts based on market conditions"""
        try:
            # Analyze market conditions
            market_conditions = context.get_market_conditions()
            
            # Determine best execution strategy based on market conditions
            volatility = market_conditions.get_volatility()
            liquidity = market_conditions.get_liquidity()
            trend = market_conditions.get_trend()
            
            # Decide execution approach based on conditions
            if volatility > 0.05:  # High volatility
                # Use TWAP for high volatility - spread execution over time
                self.logger.info(f"High volatility detected ({volatility:.4f}), using TWAP algorithm")
                return self._execute_twap(context)
                
            elif liquidity < 0.3:  # Low liquidity
                # Use Iceberg for low liquidity - hide true order size
                self.logger.info(f"Low liquidity detected ({liquidity:.4f}), using Iceberg algorithm")
                return self._execute_iceberg(context)
                
            elif abs(trend) > 0.5:  # Strong trend
                # Use aggressive execution when trend is in our favor, passive when against
                if (context.side == 'buy' and trend > 0) or (context.side == 'sell' and trend < 0):
                    # Trend is against us, be passive with limit orders
                    self.logger.info(f"Strong trend against direction ({trend:.4f}), using passive limit orders")
                    return self._execute_limit_order(context)
                else:
                    # Trend is with us, be more aggressive with market orders
                    self.logger.info(f"Strong trend in our direction ({trend:.4f}), using market orders")
                    return self._execute_market_order(context)
            else:
                # Balanced conditions - use VWAP
                self.logger.info(f"Balanced market conditions, using VWAP algorithm")
                return self._execute_vwap(context)
                
        except Exception as e:
            self.logger.error(f"Adaptive execution error: {str(e)}")
            return ExecutionResult.create_error_result(
                error_message=f"Adaptive execution failed: {str(e)}",
                symbol=context.symbol,
                side=context.side,
                quantity=context.quantity,
                target_price=context.target_price
            )

    def _execute_smart(self, context: ExecutionContext) -> ExecutionResult:
        """
        Execute order using smart routing and optimal execution strategy.
        This is the most advanced algorithm that combines multiple approaches.
        """
        try:
            # Analyze market conditions
            market_impact = 0
            
            if context.order_book:
                market_conditions = context.get_market_conditions()
                market_impact = market_conditions.get_market_impact_estimate(context.quantity, context.side)
            
            # Determine if order size is large relative to market
            large_order = market_impact > 0.01  # More than 1% price impact
            
            # Check if we need to split the order
            if large_order:
                self.logger.info(f"Large order detected with {market_impact*100:.2f}% market impact, splitting execution")
                
                # For large orders, use a mix of algorithms
                if context.time_limit:
                    # If time limit specified, use primarily TWAP/VWAP
                    if context.price_history is not None and 'volume' in context.price_history.columns:
                        return self._execute_vwap(context)
                    else:
                        return self._execute_twap(context)
                else:
                    # If no time limit, use Iceberg to hide size
                    return self._execute_iceberg(context)
            else:
                # For smaller orders, use simpler execution
                if context.target_price:
                    # If target price specified, use limit orders
                    return self._execute_limit_order(context)
                else:
                    # Otherwise use market orders
                    return self._execute_market_order(context)
                    
        except Exception as e:
            self.logger.error(f"Smart execution error: {str(e)}")
            return ExecutionResult.create_error_result(
                error_message=f"Smart execution failed: {str(e)}",
                symbol=context.symbol,
                side=context.side,
                quantity=context.quantity,
                target_price=context.target_price
            )
    
    def get_execution_stats(self) -> Dict:
        """Get execution statistics"""
        with self._lock:
            return self.execution_stats.copy()
    
    def get_execution_history(self) -> List[Dict]:
        """Get recent execution history"""
        with self._lock:
            return self.execution_history.copy()
            
    def reset_stats(self) -> None:
        """Reset execution statistics"""
        with self._lock:
            self.execution_stats = {
                'total_executions': 0,
                'successful_executions': 0,
                'failed_executions': 0,
                'avg_slippage': 0,
                'avg_execution_time': 0
            }
            self.execution_history = []

# Decorate with error handling if available
if HAVE_ERROR_HANDLING:
    SmartExecution.execute_order = safe_execute(
        ErrorCategory.TRADE_EXECUTION, 
        default_return=ExecutionResult(
            status='failed', 
            error='Execution failed due to internal error',
            details={'recovered': True}
        )
    )(SmartExecution.execute_order)
