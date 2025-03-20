# execution_simulator.py

import numpy as np
import pandas as pd
import logging
import time
import random
import threading
import queue
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Union, Optional, Tuple, Any, Callable
from enum import Enum
import json
from collections import deque
import heapq

# Try importing error handling module (but continue if not available)
try:
    from error_handling import safe_execute, ErrorCategory, ErrorSeverity
    HAVE_ERROR_HANDLING = True
except ImportError:
    HAVE_ERROR_HANDLING = False
    logging.info("Error handling module not available in execution simulator. Using basic error handling.")

class OrderStatus(Enum):
    """Order statuses matching real exchange terminology"""
    PENDING = "pending"       # Order submitted but not yet acknowledged
    NEW = "new"               # Order accepted by exchange but not filled
    PARTIALLY_FILLED = "partially_filled"
    FILLED = "filled"
    CANCELED = "canceled"
    REJECTED = "rejected"
    EXPIRED = "expired"

class OrderType(Enum):
    """Standard order types"""
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"
    TAKE_PROFIT = "take_profit"
    TAKE_PROFIT_LIMIT = "take_profit_limit"
    TRAILING_STOP = "trailing_stop"

class TimeInForce(Enum):
    """Time-in-force options"""
    GTC = "gtc"  # Good Till Canceled
    IOC = "ioc"  # Immediate or Cancel
    FOK = "fok"  # Fill or Kill
    GTD = "gtd"  # Good Till Date

class OrderSide(Enum):
    """Order sides"""
    BUY = "buy"
    SELL = "sell"

class ExchangeEvent(Enum):
    """Possible exchange events to simulate"""
    NORMAL = "normal"               # Normal operation
    HIGH_LATENCY = "high_latency"   # High latency response
    TIMEOUT = "timeout"             # Request timeout
    RATE_LIMIT = "rate_limit"       # API rate limit hit
    INSUFFICIENT_FUNDS = "insufficient_funds"
    PRICE_OUTSIDE_LIMITS = "price_outside_limits"
    EXCHANGE_MAINTENANCE = "exchange_maintenance"
    ORDER_BOOK_CHANGE = "order_book_change"  # Sudden order book change
    LIQUIDITY_SPIKE = "liquidity_spike"
    LIQUIDITY_DROP = "liquidity_drop"

class SimulatedOrder:
    """Represents an order in the simulation system"""
    
    def __init__(
        self,
        order_id: str,
        symbol: str,
        side: OrderSide,
        order_type: OrderType,
        quantity: float,
        price: Optional[float] = None,
        stop_price: Optional[float] = None,
        time_in_force: TimeInForce = TimeInForce.GTC,
        client_order_id: Optional[str] = None,
        create_time: Optional[datetime] = None,
        strategy_id: Optional[str] = None,
        leverage: float = 1.0
    ):
        # Core order details
        self.order_id = order_id
        self.client_order_id = client_order_id or f"client_{order_id}"
        self.symbol = symbol
        self.side = side if isinstance(side, OrderSide) else OrderSide(side)
        self.order_type = order_type if isinstance(order_type, OrderType) else OrderType(order_type)
        self.original_quantity = quantity
        self.executed_quantity = 0.0
        self.remaining_quantity = quantity
        self.price = price
        self.stop_price = stop_price
        self.time_in_force = time_in_force if isinstance(time_in_force, TimeInForce) else TimeInForce(time_in_force)
        self.leverage = leverage
        
        # State tracking
        self.status = OrderStatus.PENDING
        self.create_time = create_time or datetime.now()
        self.update_time = self.create_time
        self.strategy_id = strategy_id
        
        # Execution details
        self.fills = []
        self.avg_fill_price = 0.0
        self.commission = 0.0
        self.reject_reason = None
        self.is_closed = False
        
        # Internal tracking
        self._expiry_time = None if self.time_in_force != TimeInForce.GTD else (self.create_time + timedelta(days=1))
        self._triggered = False if self.order_type in [OrderType.STOP, OrderType.STOP_LIMIT] else True
        self._trailing_price = stop_price if self.order_type == OrderType.TRAILING_STOP else None
        
    def update_status(self, status: OrderStatus) -> None:
        """Update order status with timestamp"""
        self.status = status
        self.update_time = datetime.now()
        
        if status in [OrderStatus.FILLED, OrderStatus.CANCELED, OrderStatus.REJECTED, OrderStatus.EXPIRED]:
            self.is_closed = True
    
    def add_fill(self, quantity: float, price: float, timestamp: Optional[datetime] = None) -> None:
        """Add a fill to this order"""
        timestamp = timestamp or datetime.now()
        
        self.fills.append({
            "quantity": quantity,
            "price": price,
            "timestamp": timestamp
        })
        
        self.executed_quantity += quantity
        self.remaining_quantity = max(0, self.original_quantity - self.executed_quantity)
        
        # Update average fill price
        self.avg_fill_price = sum(fill["price"] * fill["quantity"] for fill in self.fills) / self.executed_quantity
        
        # Update status
        if self.remaining_quantity <= 0:
            self.update_status(OrderStatus.FILLED)
        else:
            self.update_status(OrderStatus.PARTIALLY_FILLED)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert order to dictionary representation"""
        return {
            "order_id": self.order_id,
            "client_order_id": self.client_order_id,
            "symbol": self.symbol,
            "side": self.side.value,
            "type": self.order_type.value,
            "quantity": self.original_quantity,
            "executed_quantity": self.executed_quantity,
            "remaining_quantity": self.remaining_quantity,
            "price": self.price,
            "stop_price": self.stop_price,
            "time_in_force": self.time_in_force.value,
            "status": self.status.value,
            "create_time": self.create_time.isoformat(),
            "update_time": self.update_time.isoformat(),
            "strategy_id": self.strategy_id,
            "avg_fill_price": self.avg_fill_price,
            "commission": self.commission,
            "fills": [
                {
                    "quantity": fill["quantity"],
                    "price": fill["price"],
                    "timestamp": fill["timestamp"].isoformat()
                }
                for fill in self.fills
            ],
            "is_closed": self.is_closed,
            "leverage": self.leverage
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SimulatedOrder':
        """Create order from dictionary representation"""
        order = cls(
            order_id=data["order_id"],
            symbol=data["symbol"],
            side=OrderSide(data["side"]),
            order_type=OrderType(data["type"]),
            quantity=data["quantity"],
            price=data.get("price"),
            stop_price=data.get("stop_price"),
            time_in_force=TimeInForce(data["time_in_force"]),
            client_order_id=data["client_order_id"],
            create_time=datetime.fromisoformat(data["create_time"]),
            strategy_id=data.get("strategy_id"),
            leverage=data.get("leverage", 1.0)
        )
        
        # Restore state
        order.status = OrderStatus(data["status"])
        order.executed_quantity = data["executed_quantity"]
        order.remaining_quantity = data["remaining_quantity"]
        order.update_time = datetime.fromisoformat(data["update_time"])
        order.avg_fill_price = data["avg_fill_price"]
        order.commission = data["commission"]
        order.is_closed = data["is_closed"]
        
        # Restore fills
        order.fills = [
            {
                "quantity": fill["quantity"],
                "price": fill["price"],
                "timestamp": datetime.fromisoformat(fill["timestamp"])
            }
            for fill in data["fills"]
        ]
        
        return order

class OrderBook:
    """Simulated order book implementation"""
    
    def __init__(self, symbol: str, tick_size: float = 0.01, depth: int = 10):
        self.symbol = symbol
        self.tick_size = tick_size
        self.depth = depth
        
        # Order book state
        self.bids = []  # List of (price, quantity) tuples, highest first
        self.asks = []  # List of (price, quantity) tuples, lowest first
        
        # Last update time
        self.timestamp = datetime.now()
        
        # Threading safety
        self._lock = threading.RLock()
    
    def update(self, data: Dict[str, Any]) -> None:
        """Update the order book with new data"""
        with self._lock:
            if "bids" in data:
                self.bids = sorted(data["bids"], key=lambda x: x[0], reverse=True)[:self.depth]
            
            if "asks" in data:
                self.asks = sorted(data["asks"], key=lambda x: x[0])[:self.depth]
            
            self.timestamp = datetime.now()
    
    def get_best_bid(self) -> Tuple[float, float]:
        """Get the highest bid price and quantity"""
        with self._lock:
            if not self.bids:
                return (0, 0)
            return self.bids[0]
    
    def get_best_ask(self) -> Tuple[float, float]:
        """Get the lowest ask price and quantity"""
        with self._lock:
            if not self.asks:
                return (float('inf'), 0)
            return self.asks[0]
    
    def get_mid_price(self) -> float:
        """Get the mid price"""
        with self._lock:
            bid, _ = self.get_best_bid()
            ask, _ = self.get_best_ask()
            
            if bid <= 0 or ask >= float('inf'):
                return 0
            
            return (bid + ask) / 2
    
    def get_spread(self) -> float:
        """Get the bid-ask spread"""
        with self._lock:
            bid, _ = self.get_best_bid()
            ask, _ = self.get_best_ask()
            
            if bid <= 0 or ask >= float('inf'):
                return float('inf')
            
            return ask - bid
    
    def get_liquidity(self, levels: int = 3) -> float:
        """Get the available liquidity (sum of bids and asks)"""
        with self._lock:
            bid_liquidity = sum(qty for _, qty in self.bids[:levels])
            ask_liquidity = sum(qty for _, qty in self.asks[:levels])
            return bid_liquidity + ask_liquidity
    
    def execute_market_order(self, side: OrderSide, quantity: float) -> List[Dict[str, Any]]:
        """
        Simulate market order execution and return fills
        
        Args:
            side: Buy or sell
            quantity: Order quantity
            
        Returns:
            List of fill dictionaries (price, quantity, timestamp)
        """
        with self._lock:
            remaining = quantity
            fills = []
            
            if side == OrderSide.BUY:
                # Execute against asks (lowest first)
                for i, (price, qty) in enumerate(self.asks):
                    if remaining <= 0:
                        break
                    
                    # Calculate fill quantity
                    fill_qty = min(remaining, qty)
                    remaining -= fill_qty
                    
                    # Add fill
                    fills.append({
                        "price": price,
                        "quantity": fill_qty,
                        "timestamp": datetime.now()
                    })
                    
                    # Update order book
                    if fill_qty >= qty:
                        # Level fully consumed
                        self.asks[i] = (price, 0)
                    else:
                        # Level partially consumed
                        self.asks[i] = (price, qty - fill_qty)
            else:
                # Execute against bids (highest first)
                for i, (price, qty) in enumerate(self.bids):
                    if remaining <= 0:
                        break
                    
                    # Calculate fill quantity
                    fill_qty = min(remaining, qty)
                    remaining -= fill_qty
                    
                    # Add fill
                    fills.append({
                        "price": price,
                        "quantity": fill_qty,
                        "timestamp": datetime.now()
                    })
                    
                    # Update order book
                    if fill_qty >= qty:
                        # Level fully consumed
                        self.bids[i] = (price, 0)
                    else:
                        # Level partially consumed
                        self.bids[i] = (price, qty - fill_qty)
            
            # Clean up zero quantity levels
            self.bids = [(price, qty) for price, qty in self.bids if qty > 0]
            self.asks = [(price, qty) for price, qty in self.asks if qty > 0]
            
            return fills
    
    def can_execute_market_order(self, side: OrderSide, quantity: float) -> bool:
        """Check if a market order can be fully executed"""
        with self._lock:
            if side == OrderSide.BUY:
                available = sum(qty for _, qty in self.asks)
            else:
                available = sum(qty for _, qty in self.bids)
            
            return available >= quantity
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert order book to dictionary representation"""
        with self._lock:
            return {
                "symbol": self.symbol,
                "timestamp": self.timestamp.isoformat(),
                "bids": self.bids.copy(),
                "asks": self.asks.copy(),
                "mid_price": self.get_mid_price(),
                "spread": self.get_spread()
            }

class MarketImpactModel:
    """
    Simulates market impact of orders on price and liquidity
    
    This model implements temporary and permanent price impact
    based on order size relative to available liquidity.
    """
    
    def __init__(
        self,
        base_volatility: float = 0.001,
        impact_factor: float = 0.1,
        recovery_factor: float = 0.5,
        volatility_factor: float = 0.5
    ):
        self.base_volatility = base_volatility  # Base price volatility (% per tick)
        self.impact_factor = impact_factor      # Price impact scaling factor
        self.recovery_factor = recovery_factor  # How quickly price recovers from impact
        self.volatility_factor = volatility_factor  # Volatility scaling factor
    
    def calculate_price_impact(
        self,
        order_side: OrderSide,
        order_quantity: float,
        market_liquidity: float,
        volatility: float = None
    ) -> Tuple[float, float]:
        """
        Calculate temporary and permanent price impact of an order
        
        Args:
            order_side: Buy or sell
            order_quantity: Order quantity
            market_liquidity: Available market liquidity
            volatility: Current market volatility (default: use base volatility)
            
        Returns:
            Tuple of (temporary_impact, permanent_impact) as percentage
        """
        if volatility is None:
            volatility = self.base_volatility
        
        # Avoid division by zero
        if market_liquidity <= 0:
            market_liquidity = 1.0
        
        # Calculate relative order size
        relative_size = order_quantity / market_liquidity
        
        # Scale impact based on volatility
        volatility_scale = 1.0 + (volatility / self.base_volatility - 1.0) * self.volatility_factor
        
        # Calculate impact (square root model commonly used in academic literature)
        # Higher impact for buy orders (buy pressure increases price more than sell pressure decreases it)
        direction = 1.0 if order_side == OrderSide.BUY else -0.9
        
        # Calculate impacts
        temporary_impact = direction * self.impact_factor * (relative_size ** 0.5) * volatility_scale
        permanent_impact = temporary_impact * self.recovery_factor
        
        return temporary_impact, permanent_impact

class SimulationClock:
    """
    Clock for simulation time management with variable speed
    
    Allows backtesting at accelerated pace while maintaining
    realistic event ordering and causality.
    """
    
    def __init__(self, initial_time: Optional[datetime] = None, time_multiplier: float = 1.0):
        self.start_time_real = datetime.now()
        self.start_time_sim = initial_time or self.start_time_real
        self.time_multiplier = time_multiplier  # >1 means faster simulation
        self.paused = False
        self.pause_time = None
        self.accumulated_pause = timedelta(0)
        
        # Event scheduling
        self.event_queue = []  # Priority queue of (sim_time, event_id, callback)
        self.event_counter = 0
        self._lock = threading.RLock()
    
    def get_time(self) -> datetime:
        """Get current simulation time"""
        with self._lock:
            if self.paused:
                return self.pause_time
            
            elapsed_real = datetime.now() - self.start_time_real - self.accumulated_pause
            elapsed_sim = elapsed_real * self.time_multiplier
            
            return self.start_time_sim + elapsed_sim
    
    def pause(self) -> None:
        """Pause simulation clock"""
        with self._lock:
            if not self.paused:
                self.paused = True
                self.pause_time = self.get_time()
    
    def resume(self) -> None:
        """Resume simulation clock"""
        with self._lock:
            if self.paused:
                now = datetime.now()
                pause_duration = now - (self.start_time_real + 
                                       (self.pause_time - self.start_time_sim) / self.time_multiplier + 
                                       self.accumulated_pause)
                
                self.accumulated_pause += pause_duration
                self.paused = False
                self.pause_time = None
    
    def set_time_multiplier(self, multiplier: float) -> None:
        """Change simulation speed"""
        with self._lock:
            # Get current sim time
            current_time = self.get_time()
            
            # Reset clock with new multiplier
            self.start_time_real = datetime.now() - self.accumulated_pause
            self.start_time_sim = current_time
            self.time_multiplier = multiplier
    
    def schedule_event(self, delay_seconds: float, callback: Callable, *args, **kwargs) -> int:
        """
        Schedule an event to occur after specified delay
        
        Args:
            delay_seconds: Delay in simulation seconds
            callback: Function to call
            args, kwargs: Arguments to pass to callback
            
        Returns:
            Event ID for potential cancellation
        """
        with self._lock:
            # Calculate event time
            current_time = self.get_time()
            event_time = current_time + timedelta(seconds=delay_seconds)
            
            # Create event ID
            self.event_counter += 1
            event_id = self.event_counter
            
            # Create event with callback wrapper
            def event_wrapper():
                try:
                    return callback(*args, **kwargs)
                except Exception as e:
                    logging.error(f"Error in scheduled event {event_id}: {e}")
            
            # Add to priority queue
            heapq.heappush(self.event_queue, (event_time, event_id, event_wrapper))
            
            return event_id
    
    def cancel_event(self, event_id: int) -> bool:
        """Cancel a scheduled event"""
        with self._lock:
            # Find and remove event
            for i, (time, eid, callback) in enumerate(self.event_queue):
                if eid == event_id:
                    self.event_queue.pop(i)
                    heapq.heapify(self.event_queue)  # Restore heap property
                    return True
            
            return False
    
    def process_due_events(self) -> int:
        """
        Process all events that are due to execute
        
        Returns:
            Number of events processed
        """
        with self._lock:
            if self.paused:
                return 0
            
            current_time = self.get_time()
            processed = 0
            
            while self.event_queue and self.event_queue[0][0] <= current_time:
                _, _, callback = heapq.heappop(self.event_queue)
                
                # Execute outside the lock
                self._lock.release()
                try:
                    callback()
                    processed += 1
                except Exception as e:
                    logging.error(f"Error processing scheduled event: {e}")
                finally:
                    self._lock.acquire()
            
            return processed

class ExecutionSimulator:
    """
    Advanced execution simulator for realistic trade execution
    
    Features:
    - Realistic order book simulation
    - Market impact modeling
    - Latency simulation
    - Partial fills
    - Error conditions
    - Fee calculations
    - Support for all major order types
    """
    
    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        db_manager=None,
        risk_manager=None,
        thread_manager=None
    ):
        # Configuration
        self.config = config or self._get_default_config()
        
        # External components
        self.db_manager = db_manager
        self.risk_manager = risk_manager
        self.thread_manager = thread_manager
        
        # Internal state
        self.order_books = {}          # Symbol -> OrderBook
        self.orders = {}               # Order ID -> SimulatedOrder
        self.positions = {}            # Symbol -> Position dict
        self.market_data = {}          # Symbol -> Market data dict
        self.symbols = set()           # Active symbols
        self.clock = SimulationClock() # Simulation clock
        self.running = False           # Simulator running flag
        
        # Market simulation
        self.market_impact_model = MarketImpactModel()
        self.event_probabilities = self._get_event_probabilities()
        
        # Threading
        self._lock = threading.RLock()
        self._order_id_counter = 0
        self._simulation_thread = None
        self._stop_event = threading.Event()
        self._event_log = deque(maxlen=1000)
        
        # Performance tracking
        self.performance_metrics = {
            'orders_submitted': 0,
            'orders_filled': 0,
            'orders_cancelled': 0,
            'orders_rejected': 0,
            'total_volume': 0.0,
            'total_commission': 0.0,
            'avg_slippage': 0.0,
            'avg_latency': 0.0,
            'error_rate': 0.0
        }
        
        # Initialize
        self._initialize()
    
    def _initialize(self) -> None:
        """Initialize simulator state"""
        # Setup logging
        self.logger = logging.getLogger('execution_simulator')
        
        # Initialize with configured symbols
        for symbol in self.config['symbols']:
            self.add_symbol(symbol)
        
        # Schedule recurring events
        self._schedule_recurring_events()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration"""
        return {
            'symbols': ['BTC/USDT', 'ETH/USDT', 'XRP/USDT'],
            'base_fee': 0.001,  # 0.1% base fee
            'tier_discounts': {  # Volume-based fee discounts
                100000: 0.0009,  # 100k volume -> 0.09% fee
                1000000: 0.0007, # 1M volume -> 0.07% fee
                10000000: 0.0005 # 10M volume -> 0.05% fee
            },
            'latency': {
                'base_latency': 0.05,  # 50ms base latency
                'variance': 0.02,      # 20ms variance
                'timeout_prob': 0.001  # 0.1% chance of timeout
            },
            'slippage': {
                'base_slippage': 0.0001,  # 0.01% base slippage
                'impact_factor': 0.05,    # Market impact factor
                'volatility_multiplier': 0.5  # Higher volatility -> higher slippage
            },
            'errors': {
                'base_error_rate': 0.001,  # 0.1% base error rate
                'rate_limit_threshold': 10,  # Rate limit hits after 10 requests/sec
                'maintenance_probability': 0.0001  # 0.01% chance of exchange maintenance
            },
            'liquidity': {
                'base_spread': 0.0005,  # 0.05% base spread
                'depth_factor': 1.2,    # Order book depth factor (higher = more liquidity)
                'refresh_interval': 1.0  # Order book refresh interval (seconds)
            },
            'time_multiplier': 1.0,  # Simulation speed (1.0 = real-time)
            'seed': None  # Random seed for reproducibility
        }
    
    def _get_event_probabilities(self) -> Dict[ExchangeEvent, float]:
        """Get probabilities for different exchange events"""
        return {
            ExchangeEvent.NORMAL: 0.95,
            ExchangeEvent.HIGH_LATENCY: 0.03,
            ExchangeEvent.TIMEOUT: self.config['latency']['timeout_prob'],
            ExchangeEvent.RATE_LIMIT: 0.005,
            ExchangeEvent.INSUFFICIENT_FUNDS: 0.002,
            ExchangeEvent.PRICE_OUTSIDE_LIMITS: 0.002,
            ExchangeEvent.EXCHANGE_MAINTENANCE: self.config['errors']['maintenance_probability'],
            ExchangeEvent.ORDER_BOOK_CHANGE: 0.01,
            ExchangeEvent.LIQUIDITY_SPIKE: 0.001,
            ExchangeEvent.LIQUIDITY_DROP: 0.001
        }
    
    def _generate_order_id(self) -> str:
        """Generate a unique order ID"""
        with self._lock:
            self._order_id_counter += 1
            return f"sim_{int(time.time())}_{self._order_id_counter}"
    
    def _schedule_recurring_events(self) -> None:
        """Schedule recurring simulation events"""
        # Order book updates
        refresh_interval = self.config['liquidity']['refresh_interval']
        self.clock.schedule_event(refresh_interval, self._update_order_books)
        
        # Market data updates
        self.clock.schedule_event(1.0, self._update_market_data)
        
        # Process pending events
        self.clock.schedule_event(0.1, self._process_events)
    
    def _update_order_books(self) -> None:
        """Update all order books with new data"""
        with self._lock:
            for symbol in self.symbols:
                if symbol not in self.order_books:
                    continue
                
                # Get current book
                book = self.order_books[symbol]
                
                # Generate realistic order book update
                mid_price = book.get_mid_price()
                if mid_price <= 0:
                    # Initialize with reasonable default
                    if symbol == 'BTC/USDT':
                        mid_price = 50000.0
                    elif symbol == 'ETH/USDT':
                        mid_price = 3000.0
                    else:
                        mid_price = 100.0
                
                # Apply random price change (normal distribution around current price)
                volatility = self.config['slippage']['base_slippage'] * mid_price
                price_change = np.random.normal(0, volatility)
                new_mid = max(0.01, mid_price + price_change)
                
                # Calculate spread
                base_spread = self.config['liquidity']['base_spread'] * new_mid
                spread = max(book.tick_size, base_spread)
                
                # Calculate best bid/ask
                best_bid = new_mid - spread / 2
                best_ask = new_mid + spread / 2
                
                # Round to tick size
                tick_size = book.tick_size
                best_bid = round(best_bid / tick_size) * tick_size
                best_ask = round(best_ask / tick_size) * tick_size
                
                # Ensure bid < ask
                if best_bid >= best_ask:
                    best_ask = best_bid + tick_size
                
                # Generate bids (price descending)
                depth_factor = self.config['liquidity']['depth_factor']
                bids = []
                for i in range(book.depth):
                    price = best_bid - i * tick_size
                    if price <= 0:
                        break
                    
                    # Higher quantity near the mid price
                    quantity = 10.0 * (depth_factor ** (book.depth - i)) * (0.8 + 0.4 * random.random())
                    bids.append((price, quantity))
                
                # Generate asks (price ascending)
                asks = []
                for i in range(book.depth):
                    price = best_ask + i * tick_size
                    
                    # Higher quantity near the mid price
                    quantity = 10.0 * (depth_factor ** (book.depth - i)) * (0.8 + 0.4 * random.random())
                    asks.append((price, quantity))
                
                # Update order book
                book.update({"bids": bids, "asks": asks})
        
        # Schedule next update
        refresh_interval = self.config['liquidity']['refresh_interval']
        self.clock.schedule_event(refresh_interval, self._update_order_books)
    
    def _update_market_data(self) -> None:
        """Update market data for all symbols"""
        with self._lock:
            for symbol in self.symbols:
                if symbol not in self.order_books:
                    continue
                
                book = self.order_books[symbol]
                mid_price = book.get_mid_price()
                
                if symbol not in self.market_data:
                    # Initialize market data
                    self.market_data[symbol] = {
                        'open': mid_price,
                        'high': mid_price,
                        'low': mid_price,
                        'close': mid_price,
                        'volume': 0.0,
                        'timestamp': self.clock.get_time(),
                        'last_price': mid_price,
                        'daily_change': 0.0,
                        'daily_change_pct': 0.0
                    }
                    continue
                
                # Update market data
                data = self.market_data[symbol]
                data['close'] = mid_price
                data['high'] = max(data['high'], mid_price)
                data['low'] = min(data['low'], mid_price)
                
                # Calculate changes
                data['daily_change'] = mid_price - data['open']
                if data['open'] > 0:
                    data['daily_change_pct'] = data['daily_change'] / data['open'] * 100
                
                # Update timestamp
                data['timestamp'] = self.clock.get_time()
        
        # Schedule next update
        self.clock.schedule_event(1.0, self._update_market_data)
    
    def _process_events(self) -> None:
        """Process simulation events"""
        # Process due events from clock
        self.clock.process_due_events()
        
        # Process open orders
        self._process_open_orders()
        
        # Schedule next processing
        self.clock.schedule_event(0.1, self._process_events)
    
    def _process_open_orders(self) -> None:
        """Process all open orders"""
        with self._lock:
            # Get current time
            current_time = self.clock.get_time()
            
            # Process each order
            for order_id, order in list(self.orders.items()):
                # Skip closed orders
                if order.is_closed:
                    continue
                
                # Check for expired orders
                if order._expiry_time and current_time >= order._expiry_time:
                    order.update_status(OrderStatus.EXPIRED)
                    self._log_event(f"Order {order_id} expired")
                    continue
                
                # Process based on order type
                if order.order_type == OrderType.MARKET:
                    self._process_market_order(order)
                elif order.order_type == OrderType.LIMIT:
                    self._process_limit_order(order)
                elif order.order_type in [OrderType.STOP, OrderType.STOP_LIMIT]:
                    self._process_stop_order(order)
                elif order.order_type in [OrderType.TAKE_PROFIT, OrderType.TAKE_PROFIT_LIMIT]:
                    self._process_take_profit_order(order)
                elif order.order_type == OrderType.TRAILING_STOP:
                    self._process_trailing_stop_order(order)
    
    def _process_market_order(self, order: SimulatedOrder) -> None:
        """Process a market order"""
        # Skip if already being processed
        if order.status != OrderStatus.NEW:
            return
        
        # Get order book
        if order.symbol not in self.order_books:
            order.update_status(OrderStatus.REJECTED)
            order.reject_reason = "Symbol not available"
            self._log_event(f"Market order {order.order_id} rejected: Symbol not available")
            return
        
        book = self.order_books[order.symbol]
        
        # Check if we can execute
        if not book.can_execute_market_order(order.side, order.remaining_quantity):
            # Not enough liquidity for full execution
            if order.time_in_force == TimeInForce.FOK:
                # Fill or Kill - reject
                order.update_status(OrderStatus.REJECTED)
                order.reject_reason = "Not enough liquidity for Fill or Kill order"
                self._log_event(f"Market order {order.order_id} rejected: Not enough liquidity for FOK")
                return
        
        # Execute order
        fills = book.execute_market_order(order.side, order.remaining_quantity)
        
        # Apply fills to order
        for fill in fills:
            # Apply slippage to price
            price = fill["price"]
            quantity = fill["quantity"]
            
            # Add fill to order
            order.add_fill(quantity, price, fill["timestamp"])
            
            # Update metrics
            self.performance_metrics['total_volume'] += quantity * price
        
        # Calculate and add commission
        self._calculate_commission(order)
        
        # Update position
        self._update_position(order)
        
        # Log completion
        if order.status == OrderStatus.FILLED:
            self.performance_metrics['orders_filled'] += 1
            self._log_event(f"Market order {order.order_id} filled: {order.avg_fill_price}")
        elif order.status == OrderStatus.PARTIALLY_FILLED:
            self._log_event(f"Market order {order.order_id} partially filled: {order.executed_quantity}/{order.original_quantity}")
    
    def _process_limit_order(self, order: SimulatedOrder) -> None:
        """Process a limit order"""
        # Skip if already being processed or no price
        if order.status != OrderStatus.NEW or order.price is None:
            return
        
        # Get order book
        if order.symbol not in self.order_books:
            return
        
        book = self.order_books[order.symbol]
        
        # Check if the limit order can be executed
        if order.side == OrderSide.BUY:
            best_ask, ask_qty = book.get_best_ask()
            if best_ask <= order.price:
                # Can execute against asks
                executable_qty = min(order.remaining_quantity, ask_qty)
                
                # Create fill
                order.add_fill(
                    executable_qty,
                    best_ask,
                    self.clock.get_time()
                )
                
                # Update book
                book.asks[0] = (best_ask, ask_qty - executable_qty)
                
                # Clean up
                if book.asks[0][1] <= 0:
                    book.asks.pop(0)
                
                # Calculate and add commission
                self._calculate_commission(order)
                
                # Update position
                self._update_position(order)
                
                self._log_event(f"Limit buy {order.order_id} executed: {executable_qty} @ {best_ask}")
        else:
            # Sell order
            best_bid, bid_qty = book.get_best_bid()
            if best_bid >= order.price:
                # Can execute against bids
                executable_qty = min(order.remaining_quantity, bid_qty)
                
                # Create fill
                order.add_fill(
                    executable_qty,
                    best_bid,
                    self.clock.get_time()
                )
                
                # Update book
                book.bids[0] = (best_bid, bid_qty - executable_qty)
                
                # Clean up
                if book.bids[0][1] <= 0:
                    book.bids.pop(0)
                
                # Calculate and add commission
                self._calculate_commission(order)
                
                # Update position
                self._update_position(order)
                
                self._log_event(f"Limit sell {order.order_id} executed: {executable_qty} @ {best_bid}")
        
        # Handle special time-in-force
        if order.time_in_force == TimeInForce.IOC and order.remaining_quantity > 0:
            # Immediate or Cancel - cancel remaining quantity
            order.update_status(OrderStatus.CANCELED)
            self._log_event(f"IOC order {order.order_id} remaining quantity cancelled")
    
    def _process_stop_order(self, order: SimulatedOrder) -> None:
        """Process a stop or stop-limit order"""
        # Skip if already triggered or no stop price
        if order._triggered or order.stop_price is None:
            return
        
        # Get market data
        if order.symbol not in self.market_data:
            return
        
        market_data = self.market_data[order.symbol]
        last_price = market_data['last_price']
        
        # Check if stop is triggered
        triggered = False
        if order.side == OrderSide.BUY:
            # Buy Stop: Triggers when price goes above stop price
            triggered = last_price >= order.stop_price
        else:
            # Sell Stop: Triggers when price goes below stop price
            triggered = last_price <= order.stop_price
        
        if triggered:
            self._log_event(f"Stop {order.side.value} {order.order_id} triggered at {last_price}")
            
            # Mark as triggered
            order._triggered = True
            
            if order.order_type == OrderType.STOP:
                # Convert to market order
                order.order_type = OrderType.MARKET
                self._process_market_order(order)
            elif order.order_type == OrderType.STOP_LIMIT:
                # Convert to limit order (price should already be set)
                order.order_type = OrderType.LIMIT
                self._process_limit_order(order)
    
    def _process_take_profit_order(self, order: SimulatedOrder) -> None:
        """Process a take-profit or take-profit-limit order"""
        # Similar to stop orders but with inverted trigger logic
        # Skip if already triggered or no stop price
        if order._triggered or order.stop_price is None:
            return
        
        # Get market data
        if order.symbol not in self.market_data:
            return
        
        market_data = self.market_data[order.symbol]
        last_price = market_data['last_price']
        
        # Check if take-profit is triggered
        triggered = False
        if order.side == OrderSide.BUY:
            # Buy Take Profit: Triggers when price goes below stop price
            triggered = last_price <= order.stop_price
        else:
            # Sell Take Profit: Triggers when price goes above stop price
            triggered = last_price >= order.stop_price
        
        if triggered:
            self._log_event(f"Take profit {order.side.value} {order.order_id} triggered at {last_price}")
            
            # Mark as triggered
            order._triggered = True
            
            if order.order_type == OrderType.TAKE_PROFIT:
                # Convert to market order
                order.order_type = OrderType.MARKET
                self._process_market_order(order)
            elif order.order_type == OrderType.TAKE_PROFIT_LIMIT:
                # Convert to limit order (price should already be set)
                order.order_type = OrderType.LIMIT
                self._process_limit_order(order)
    
    def _process_trailing_stop_order(self, order: SimulatedOrder) -> None:
        """Process a trailing stop order"""
        # Skip if already triggered or no trailing price
        if order._triggered or order._trailing_price is None:
            return
        
        # Get market data
        if order.symbol not in self.market_data:
            return
        
        market_data = self.market_data[order.symbol]
        last_price = market_data['last_price']
        
        # Update trailing price if market moves in favorable direction
        if order.side == OrderSide.BUY:
            # For buy trailing stop, stop price moves down as market price moves down
            if last_price < order._trailing_price - order.stop_price:
                # Update trailing reference price
                order._trailing_price = last_price
                self._log_event(f"Trailing buy stop {order.order_id} moved to {last_price + order.stop_price}")
            
            # Check if triggered
            triggered = last_price >= order._trailing_price + order.stop_price
        else:
            # For sell trailing stop, stop price moves up as market price moves up
            if last_price > order._trailing_price + order.stop_price:
                # Update trailing reference price
                order._trailing_price = last_price
                self._log_event(f"Trailing sell stop {order.order_id} moved to {last_price - order.stop_price}")
            
            # Check if triggered
            triggered = last_price <= order._trailing_price - order.stop_price
        
        if triggered:
            self._log_event(f"Trailing stop {order.side.value} {order.order_id} triggered at {last_price}")
            
            # Mark as triggered
            order._triggered = True
            
            # Convert to market order
            order.order_type = OrderType.MARKET
            self._process_market_order(order)
    
    def _calculate_commission(self, order: SimulatedOrder) -> None:
        """Calculate and apply commission to an order"""
        base_fee = self.config['base_fee']
        tier_discounts = self.config['tier_discounts']
        
        # Determine applicable fee rate based on volume tiers
        volume = self.performance_metrics['total_volume']
        fee_rate = base_fee
        
        # Apply tiered discounts
        for tier_volume, tier_fee in sorted(tier_discounts.items()):
            if volume >= tier_volume:
                fee_rate = tier_fee
        
        # Calculate commission
        executed_value = order.executed_quantity * order.avg_fill_price
        commission = executed_value * fee_rate
        
        # Minimal commission to prevent zero fees
        commission = max(commission, 0.000001 * order.avg_fill_price)
        
        # Update order
        order.commission = commission
        
        # Update metrics
        self.performance_metrics['total_commission'] += commission
    
    def _update_position(self, order: SimulatedOrder) -> None:
        """Update position based on order fills"""
        symbol = order.symbol
        
        with self._lock:
            # Initialize position if needed
            if symbol not in self.positions:
                self.positions[symbol] = {
                    'quantity': 0.0,
                    'entry_price': 0.0,
                    'realized_pnl': 0.0,
                    'unrealized_pnl': 0.0,
                    'side': None
                }
            
            position = self.positions[symbol]
            
            # Calculate fill impact
            if order.status in [OrderStatus.FILLED, OrderStatus.PARTIALLY_FILLED]:
                # Get latest fills
                fills = [f for f in order.fills if f['quantity'] > 0]
                
                # Process each fill
                for fill in fills:
                    fill_qty = fill['quantity']
                    fill_price = fill['price']
                    
                    # Calculate impact based on side
                    if order.side == OrderSide.BUY:
                        # Buy order
                        if position['quantity'] < 0:
                            # Reducing short position
                            close_qty = min(abs(position['quantity']), fill_qty)
                            
                            # Calculate realized PnL
                            realized_pnl = close_qty * (position['entry_price'] - fill_price)
                            position['realized_pnl'] += realized_pnl
                            
                            # Reduce position
                            position['quantity'] += close_qty
                            
                            # If some quantity remains, add to position
                            remaining_qty = fill_qty - close_qty
                            if remaining_qty > 0:
                                # Switch to long
                                position['quantity'] = remaining_qty
                                position['entry_price'] = fill_price
                                position['side'] = 'long'
                        else:
                            # Adding to long position or new long
                            if position['quantity'] == 0:
                                # New position
                                position['quantity'] = fill_qty
                                position['entry_price'] = fill_price
                                position['side'] = 'long'
                            else:
                                # Average up/down
                                total_value = position['quantity'] * position['entry_price']
                                total_value += fill_qty * fill_price
                                position['quantity'] += fill_qty
                                position['entry_price'] = total_value / position['quantity']
                    else:
                        # Sell order
                        if position['quantity'] > 0:
                            # Reducing long position
                            close_qty = min(position['quantity'], fill_qty)
                            
                            # Calculate realized PnL
                            realized_pnl = close_qty * (fill_price - position['entry_price'])
                            position['realized_pnl'] += realized_pnl
                            
                            # Reduce position
                            position['quantity'] -= close_qty
                            
                            # If some quantity remains, add to position
                            remaining_qty = fill_qty - close_qty
                            if remaining_qty > 0:
                                # Switch to short
                                position['quantity'] = -remaining_qty
                                position['entry_price'] = fill_price
                                position['side'] = 'short'
                        else:
                            # Adding to short position or new short
                            if position['quantity'] == 0:
                                # New position
                                position['quantity'] = -fill_qty
                                position['entry_price'] = fill_price
                                position['side'] = 'short'
                            else:
                                # Average up/down
                                total_value = abs(position['quantity']) * position['entry_price']
                                total_value += fill_qty * fill_price
                                position['quantity'] -= fill_qty
                                position['entry_price'] = total_value / abs(position['quantity'])
                
                # Clear for next update
                order.fills = []
            
            # Update unrealized PnL
            if position['quantity'] != 0 and symbol in self.market_data:
                current_price = self.market_data[symbol]['close']
                
                if position['quantity'] > 0:
                    # Long position
                    position['unrealized_pnl'] = position['quantity'] * (current_price - position['entry_price'])
                else:
                    # Short position
                    position['unrealized_pnl'] = abs(position['quantity']) * (position['entry_price'] - current_price)
    
    def _log_event(self, message: str) -> None:
        """Log a simulation event"""
        event = {
            'timestamp': self.clock.get_time(),
            'message': message
        }
        
        # Add to event log
        self._event_log.append(event)
        
        # Log to Python logger
        self.logger.info(message)
    
    def _simulate_random_event(self) -> ExchangeEvent:
        """Simulate a random exchange event based on configured probabilities"""
        rand = random.random()
        cumulative_prob = 0
        
        for event, prob in self.event_probabilities.items():
            cumulative_prob += prob
            if rand < cumulative_prob:
                return event
        
        return ExchangeEvent.NORMAL
    
    def _simulate_latency(self) -> float:
        """Simulate network latency"""
        base = self.config['latency']['base_latency']
        variance = self.config['latency']['variance']
        
        # Generate random latency with occasional spikes
        rand = random.random()
        if rand > 0.98:
            # Latency spike (5-10x normal)
            spike_factor = 5 + 5 * random.random()
            return base * spike_factor
        else:
            # Normal latency distribution
            return max(0.001, np.random.normal(base, variance))
    
    def _simulate_slippage(
        self, 
        order_type: OrderType, 
        side: OrderSide, 
        symbol: str, 
        quantity: float, 
        price: Optional[float] = None
    ) -> float:
        """
        Simulate price slippage
        
        Args:
            order_type: Type of order
            side: Buy or sell
            symbol: Trading symbol
            quantity: Order quantity
            price: Order price (if applicable)
            
        Returns:
            Slippage as a percentage (positive = worse price)
        """
        # No slippage for limit orders that aren't marketable
        if order_type == OrderType.LIMIT and price is not None:
            # For buy limit, if price < best ask, no slippage
            # For sell limit, if price > best bid, no slippage
            if symbol in self.order_books:
                book = self.order_books[symbol]
                best_bid, _ = book.get_best_bid()
                best_ask, _ = book.get_best_ask()
                
                if (side == OrderSide.BUY and price < best_ask) or \
                   (side == OrderSide.SELL and price > best_bid):
                    return 0.0
        
        # Base slippage parameters
        base_slippage = self.config['slippage']['base_slippage']
        
        # Get market parameters
        if symbol in self.order_books:
            book = self.order_books[symbol]
            spread = book.get_spread()
            mid_price = book.get_mid_price()
            liquidity = book.get_liquidity()
            
            # Adjust for order size relative to liquidity
            if liquidity > 0:
                # Calculate market impact
                temp_impact, _ = self.market_impact_model.calculate_price_impact(
                    side, quantity, liquidity
                )
                
                # Convert impact to slippage
                return abs(temp_impact)
        
        # Fallback to base slippage if market data not available
        return base_slippage
    
    def add_symbol(self, symbol: str) -> None:
        """Add a symbol to the simulator"""
        with self._lock:
            if symbol in self.symbols:
                return
            
            # Add to active symbols
            self.symbols.add(symbol)
            
            # Create order book
            tick_size = 0.01
            if symbol == 'BTC/USDT':
                tick_size = 0.1
            elif symbol == 'ETH/USDT':
                tick_size = 0.01
                
            self.order_books[symbol] = OrderBook(symbol, tick_size)
            
            # Initialize market data
            if symbol not in self.market_data:
                mid_price = 0
                if symbol == 'BTC/USDT':
                    mid_price = 50000.0
                elif symbol == 'ETH/USDT':
                    mid_price = 3000.0
                else:
                    mid_price = 100.0
                    
                self.market_data[symbol] = {
                    'open': mid_price,
                    'high': mid_price,
                    'low': mid_price,
                    'close': mid_price,
                    'volume': 0.0,
                    'timestamp': self.clock.get_time(),
                    'last_price': mid_price,
                    'daily_change': 0.0,
                    'daily_change_pct': 0.0
                }
    
    def remove_symbol(self, symbol: str) -> None:
        """Remove a symbol from the simulator"""
        with self._lock:
            if symbol not in self.symbols:
                return
            
            # Remove from active symbols
            self.symbols.remove(symbol)
            
            # Remove order book and market data
            if symbol in self.order_books:
                del self.order_books[symbol]
            
            if symbol in self.market_data:
                del self.market_data[symbol]
    
    def start(self) -> None:
        """Start the simulation"""
        with self._lock:
            if self.running:
                return
            
            self.running = True
            self._stop_event.clear()
            
            # Start simulation thread
            self._simulation_thread = threading.Thread(target=self._simulation_loop)
            self._simulation_thread.daemon = True
            self._simulation_thread.start()
            
            self._log_event("Execution simulator started")
    
    def stop(self) -> None:
        """Stop the simulation"""
        with self._lock:
            if not self.running:
                return
            
            self.running = False
            self._stop_event.set()
            
            # Wait for simulation thread to end
            if self._simulation_thread and self._simulation_thread.is_alive():
                self._simulation_thread.join(timeout=2.0)
            
            self._log_event("Execution simulator stopped")
    
    def _simulation_loop(self) -> None:
        """Main simulation loop"""
        try:
            # Resume clock if paused
            if self.clock.paused:
                self.clock.resume()
            
            # Run until stopped
            while self.running and not self._stop_event.is_set():
                # Process events and orders
                self.clock.process_due_events()
                
                # Small delay to prevent 100% CPU usage
                time.sleep(0.01)
                
        except Exception as e:
            self.logger.error(f"Error in simulation loop: {e}")
            self._log_event(f"Simulation error: {e}")
        finally:
            # Pause clock when simulation stops
            self.clock.pause()
    
    def place_order(
        self,
        symbol: str,
        side: Union[OrderSide, str],
        order_type: Union[OrderType, str],
        quantity: float,
        price: Optional[float] = None,
        stop_price: Optional[float] = None,
        time_in_force: Union[TimeInForce, str] = TimeInForce.GTC,
        client_order_id: Optional[str] = None,
        strategy_id: Optional[str] = None,
        leverage: float = 1.0
    ) -> Dict[str, Any]:
        """
        Place an order in the simulator
        
        Args:
            symbol: Trading pair symbol
            side: Buy or sell
            order_type: Order type
            quantity: Order quantity
            price: Order price (required for limit orders)
            stop_price: Stop price (required for stop orders)
            time_in_force: Time in force
            client_order_id: Client-provided order ID
            strategy_id: Strategy identifier
            leverage: Order leverage
            
        Returns:
            Dict with order details and status
        """
        try:
            # Validate inputs
            if not symbol or symbol not in self.symbols:
                return {"status": "error", "message": f"Invalid symbol: {symbol}"}
            
            if quantity <= 0:
                return {"status": "error", "message": "Quantity must be positive"}
            
            # Convert enums if needed
            if isinstance(side, str):
                side = OrderSide(side)
            
            if isinstance(order_type, str):
                order_type = OrderType(order_type)
            
            if isinstance(time_in_force, str):
                time_in_force = TimeInForce(time_in_force)
            
            # Price validation
            if order_type in [OrderType.LIMIT, OrderType.STOP_LIMIT, OrderType.TAKE_PROFIT_LIMIT] and price is None:
                return {"status": "error", "message": "Price is required for limit orders"}
            
            if order_type in [OrderType.STOP, OrderType.STOP_LIMIT, OrderType.TAKE_PROFIT, 
                             OrderType.TAKE_PROFIT_LIMIT, OrderType.TRAILING_STOP] and stop_price is None:
                return {"status": "error", "message": "Stop price is required for stop orders"}
            
            # Generate order ID
            order_id = self._generate_order_id()
            
            # Simulate exchange event
            event = self._simulate_random_event()
            
            if event == ExchangeEvent.EXCHANGE_MAINTENANCE:
                return {"status": "error", "message": "Exchange is in maintenance mode"}
            
            if event == ExchangeEvent.RATE_LIMIT:
                return {"status": "error", "message": "Rate limit exceeded"}
            
            if event == ExchangeEvent.TIMEOUT:
                return {"status": "error", "message": "Request timed out"}
            
            # Simulate latency
            latency = self._simulate_latency()
            time.sleep(latency)
            
            # Update metrics
            self.performance_metrics['avg_latency'] = (
                self.performance_metrics['avg_latency'] * self.performance_metrics['orders_submitted'] + latency
            ) / (self.performance_metrics['orders_submitted'] + 1)
            
            # Create order object
            order = SimulatedOrder(
                order_id=order_id,
                symbol=symbol,
                side=side,
                order_type=order_type,
                quantity=quantity,
                price=price,
                stop_price=stop_price,
                time_in_force=time_in_force,
                client_order_id=client_order_id,
                create_time=self.clock.get_time(),
                strategy_id=strategy_id,
                leverage=leverage
            )
            
            # Set initial status
            order.update_status(OrderStatus.NEW)
            
            # Store order
            with self._lock:
                self.orders[order_id] = order
                self.performance_metrics['orders_submitted'] += 1
            
            # Process immediately if market order
            if order_type == OrderType.MARKET:
                self._process_market_order(order)
            
            self._log_event(f"Order {order_id} placed: {side.value} {quantity} {symbol} at {price or 'market'}")
            
            # Return order info
            return {
                "status": "success",
                "order_id": order_id,
                "client_order_id": client_order_id,
                "symbol": symbol,
                "side": side.value,
                "type": order_type.value,
                "quantity": quantity,
                "price": price,
                "stop_price": stop_price,
                "time_in_force": time_in_force.value,
                "strategy_id": strategy_id,
                "order_status": order.status.value
            }
            
        except Exception as e:
            self.logger.error(f"Error placing order: {e}")
            return {"status": "error", "message": f"Internal error: {str(e)}"}
    
    def cancel_order(self, order_id: str) -> Dict[str, Any]:
        """
        Cancel an open order
        
        Args:
            order_id: Order ID to cancel
            
        Returns:
            Dict with cancellation status
        """
        try:
            # Check if order exists
            if order_id not in self.orders:
                return {"status": "error", "message": "Order not found"}
            
            order = self.orders[order_id]
            
            # Check if order can be cancelled
            if order.is_closed:
                return {
                    "status": "error", 
                    "message": f"Order cannot be cancelled (status: {order.status.value})"
                }
            
            # Simulate exchange event
            event = self._simulate_random_event()
            
            if event == ExchangeEvent.EXCHANGE_MAINTENANCE:
                return {"status": "error", "message": "Exchange is in maintenance mode"}
            
            if event == ExchangeEvent.RATE_LIMIT:
                return {"status": "error", "message": "Rate limit exceeded"}
            
            if event == ExchangeEvent.TIMEOUT:
                return {"status": "error", "message": "Request timed out"}
            
            # Simulate latency
            latency = self._simulate_latency()
            time.sleep(latency)
            
            # Cancel order
            order.update_status(OrderStatus.CANCELED)
            self.performance_metrics['orders_cancelled'] += 1
            
            self._log_event(f"Order {order_id} cancelled")
            
            return {
                "status": "success",
                "order_id": order_id,
                "symbol": order.symbol,
                "side": order.side.value,
                "type": order.order_type.value,
                "quantity": order.original_quantity,
                "executed_quantity": order.executed_quantity,
                "price": order.price,
                "order_status": order.status.value
            }
            
        except Exception as e:
            self.logger.error(f"Error cancelling order: {e}")
            return {"status": "error", "message": f"Internal error: {str(e)}"}
    
    def get_order(self, order_id: str) -> Dict[str, Any]:
        """
        Get order details
        
        Args:
            order_id: Order ID
            
        Returns:
            Dict with order details or error
        """
        try:
            # Check if order exists
            if order_id not in self.orders:
                return {"status": "error", "message": "Order not found"}
            
            order = self.orders[order_id]
            
            # Simulate latency (lighter than place/cancel)
            latency = self._simulate_latency() * 0.5
            time.sleep(latency)
            
            # Return order details
            return {
                "status": "success",
                "order": order.to_dict()
            }
            
        except Exception as e:
            self.logger.error(f"Error getting order: {e}")
            return {"status": "error", "message": f"Internal error: {str(e)}"}
    
    def get_open_orders(self, symbol: Optional[str] = None) -> Dict[str, Any]:
        """
        Get all open orders, optionally filtered by symbol
        
        Args:
            symbol: Filter by symbol (optional)
            
        Returns:
            Dict with open orders list
        """
        try:
            # Simulate latency (lighter than single order query)
            latency = self._simulate_latency() * 0.3
            time.sleep(latency)
            
            # Filter orders
            open_orders = []
            
            with self._lock:
                for order_id, order in self.orders.items():
                    if order.is_closed:
                        continue
                    
                    if symbol and order.symbol != symbol:
                        continue
                    
                    open_orders.append(order.to_dict())
            
            return {
                "status": "success",
                "orders": open_orders
            }
            
        except Exception as e:
            self.logger.error(f"Error getting open orders: {e}")
            return {"status": "error", "message": f"Internal error: {str(e)}"}
    
    def get_position(self, symbol: str) -> Dict[str, Any]:
        """
        Get current position for a symbol
        
        Args:
            symbol: Trading pair symbol
            
        Returns:
            Dict with position details
        """
        try:
            # Check if symbol exists
            if symbol not in self.symbols:
                return {"status": "error", "message": f"Invalid symbol: {symbol}"}
            
            # Simulate latency
            latency = self._simulate_latency() * 0.2
            time.sleep(latency)
            
            # Get position
            with self._lock:
                if symbol not in self.positions:
                    # No position
                    return {
                        "status": "success",
                        "position": {
                            "symbol": symbol,
                            "quantity": 0.0,
                            "entry_price": 0.0,
                            "realized_pnl": 0.0,
                            "unrealized_pnl": 0.0,
                            "side": None
                        }
                    }
                
                position = self.positions[symbol].copy()
                position['symbol'] = symbol
                
                return {
                    "status": "success",
                    "position": position
                }
                
        except Exception as e:
            self.logger.error(f"Error getting position: {e}")
            return {"status": "error", "message": f"Internal error: {str(e)}"}
    
    def get_all_positions(self) -> Dict[str, Any]:
        """
        Get all current positions
        
        Returns:
            Dict with all positions
        """
        try:
            # Simulate latency
            latency = self._simulate_latency() * 0.3
            time.sleep(latency)
            
            # Get all positions
            positions = []
            
            with self._lock:
                for symbol, position in self.positions.items():
                    if position['quantity'] == 0:
                        continue
                    
                    pos_copy = position.copy()
                    pos_copy['symbol'] = symbol
                    positions.append(pos_copy)
            
            return {
                "status": "success",
                "positions": positions
            }
            
        except Exception as e:
            self.logger.error(f"Error getting all positions: {e}")
            return {"status": "error", "message": f"Internal error: {str(e)}"}
    
    def get_order_book(self, symbol: str, depth: int = 10) -> Dict[str, Any]:
        """
        Get order book for a symbol
        
        Args:
            symbol: Trading pair symbol
            depth: Order book depth
            
        Returns:
            Dict with order book data
        """
        try:
            # Check if symbol exists
            if symbol not in self.symbols or symbol not in self.order_books:
                return {"status": "error", "message": f"Invalid symbol: {symbol}"}
            
            # Simulate latency
            latency = self._simulate_latency() * 0.1
            time.sleep(latency)
            
            # Get order book
            with self._lock:
                book = self.order_books[symbol]
                
                # Limit depth
                depth = min(depth, len(book.bids), len(book.asks))
                
                return {
                    "status": "success",
                    "symbol": symbol,
                    "timestamp": book.timestamp.isoformat(),
                    "bids": book.bids[:depth],
                    "asks": book.asks[:depth],
                    "mid_price": book.get_mid_price()
                }
                
        except Exception as e:
            self.logger.error(f"Error getting order book: {e}")
            return {"status": "error", "message": f"Internal error: {str(e)}"}
    
    def get_ticker(self, symbol: str) -> Dict[str, Any]:
        """
        Get ticker data for a symbol
        
        Args:
            symbol: Trading pair symbol
            
        Returns:
            Dict with ticker data
        """
        try:
            # Check if symbol exists
            if symbol not in self.symbols or symbol not in self.market_data:
                return {"status": "error", "message": f"Invalid symbol: {symbol}"}
            
            # Simulate latency
            latency = self._simulate_latency() * 0.05
            time.sleep(latency)
            
            # Get ticker data
            with self._lock:
                data = self.market_data[symbol]
                
                return {
                    "status": "success",
                    "symbol": symbol,
                    "timestamp": data['timestamp'].isoformat(),
                    "last_price": data['close'],
                    "open": data['open'],
                    "high": data['high'],
                    "low": data['low'],
                    "close": data['close'],
                    "volume": data['volume'],
                    "change": data['daily_change'],
                    "change_percent": data['daily_change_pct']
                }
                
        except Exception as e:
            self.logger.error(f"Error getting ticker: {e}")
            return {"status": "error", "message": f"Internal error: {str(e)}"}
    
    def get_account_balance(self) -> Dict[str, Any]:
        """
        Get account balance information
        
        Returns:
            Dict with account balance data
        """
        try:
            # Simulate latency
            latency = self._simulate_latency() * 0.2
            time.sleep(latency)
            
            # Calculate total balance based on positions
            balances = {
                'USDT': 100000.0  # Default balance
            }
            
            # Add balance for each position
            with self._lock:
                for symbol, position in self.positions.items():
                    if position['quantity'] == 0:
                        continue
                    
                    # Extract base currency from symbol
                    base_currency = symbol.split('/')[0]
                    
                    # Add to balances
                    if base_currency not in balances:
                        balances[base_currency] = 0.0
                    
                    balances[base_currency] += position['quantity']
            
            # Total value in USDT
            total_value = balances['USDT']
            
            # Add value of other currencies
            for currency, amount in balances.items():
                if currency == 'USDT':
                    continue
                
                # Get symbol for this currency
                symbol = f"{currency}/USDT"
                if symbol in self.market_data:
                    price = self.market_data[symbol]['close']
                    total_value += amount * price
            
            return {
                "status": "success",
                "balances": balances,
                "total_value_usdt": total_value,
                "pnl": sum(p['realized_pnl'] + p['unrealized_pnl'] for p in self.positions.values())
            }
            
        except Exception as e:
            self.logger.error(f"Error getting account balance: {e}")
            return {"status": "error", "message": f"Internal error: {str(e)}"}
    
    def get_simulation_status(self) -> Dict[str, Any]:
        """
        Get current simulation status and metrics
        
        Returns:
            Dict with simulation status
        """
        with self._lock:
            return {
                "status": "success",
                "running": self.running,
                "clock_time": self.clock.get_time().isoformat(),
                "time_multiplier": self.clock.time_multiplier,
                "paused": self.clock.paused,
                "metrics": self.performance_metrics,
                "symbols": list(self.symbols),
                "open_order_count": len([o for o in self.orders.values() if not o.is_closed]),
                "position_count": len([p for p in self.positions.values() if p['quantity'] != 0])
            }
    
    def set_time_multiplier(self, multiplier: float) -> Dict[str, Any]:
        """
        Set simulation time multiplier
        
        Args:
            multiplier: Time multiplier (1.0 = real-time)
            
        Returns:
            Dict with status
        """
        try:
            if multiplier <= 0:
                return {"status": "error", "message": "Time multiplier must be positive"}
            
            self.clock.set_time_multiplier(multiplier)
            self.config['time_multiplier'] = multiplier
            
            return {"status": "success", "time_multiplier": multiplier}
            
        except Exception as e:
            self.logger.error(f"Error setting time multiplier: {e}")
            return {"status": "error", "message": f"Internal error: {str(e)}"}
    
    def reset_simulation(self) -> Dict[str, Any]:
        """
        Reset simulation state
        
        Returns:
            Dict with status
        """
        try:
            # Stop if running
            was_running = self.running
            if was_running:
                self.stop()
            
            # Reset state
            with self._lock:
                # Clear orders and positions
                self.orders = {}
                self.positions = {}
                
                # Reset metrics
                self.performance_metrics = {
                    'orders_submitted': 0,
                    'orders_filled': 0,
                    'orders_cancelled': 0,
                    'orders_rejected': 0,
                    'total_volume': 0.0,
                    'total_commission': 0.0,
                    'avg_slippage': 0.0,
                    'avg_latency': 0.0,
                    'error_rate': 0.0
                }
                
                # Clear event log
                self._event_log.clear()
                
                # Reset order books and market data
                for symbol in self.symbols:
                    self.remove_symbol(symbol)
                    self.add_symbol(symbol)
                
                # Reset clock
                self.clock = SimulationClock(time_multiplier=self.config['time_multiplier'])
                
                # Schedule recurring events
                self._schedule_recurring_events()
                
                # Reset counters
                self._order_id_counter = 0
            
            # Restart if needed
            if was_running:
                self.start()
            
            return {"status": "success", "message": "Simulation reset"}
            
        except Exception as e:
            self.logger.error(f"Error resetting simulation: {e}")
            return {"status": "error", "message": f"Internal error: {str(e)}"}
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert simulation state to dictionary for persistence
        
        Returns:
            Dict with simulation state
        """
        with self._lock:
            # Create state dictionary
            state = {
                'config': self.config,
                'symbols': list(self.symbols),
                'orders': [order.to_dict() for order in self.orders.values()],
                'positions': {symbol: position.copy() for symbol, position in self.positions.items()},
                'performance_metrics': self.performance_metrics.copy(),
                'time_multiplier': self.clock.time_multiplier,
                'event_log': [
                    {
                        'timestamp': event['timestamp'].isoformat(),
                        'message': event['message']
                    }
                    for event in self._event_log
                ]
            }
            
            return state
    
    @classmethod
    def from_dict(cls, state: Dict[str, Any], db_manager=None, risk_manager=None, thread_manager=None) -> 'ExecutionSimulator':
        """
        Create simulation from saved state
        
        Args:
            state: Saved state dictionary
            db_manager: Database manager instance
            risk_manager: Risk manager instance
            thread_manager: Thread manager instance
            
        Returns:
            ExecutionSimulator instance with restored state
        """
        # Create simulator with saved config
        simulator = cls(
            config=state.get('config'),
            db_manager=db_manager,
            risk_manager=risk_manager,
            thread_manager=thread_manager
        )
        
        # Restore symbols
        for symbol in state.get('symbols', []):
            simulator.add_symbol(symbol)
        
        # Restore orders
        with simulator._lock:
            for order_data in state.get('orders', []):
                order = SimulatedOrder.from_dict(order_data)
                simulator.orders[order.order_id] = order
        
        # Restore positions
        with simulator._lock:
            for symbol, position_data in state.get('positions', {}).items():
                simulator.positions[symbol] = position_data
        
        # Restore metrics
        simulator.performance_metrics = state.get('performance_metrics', simulator.performance_metrics)
        
        # Restore time multiplier
        if 'time_multiplier' in state:
            simulator.set_time_multiplier(state['time_multiplier'])
        
        # Restore event log if possible
        if 'event_log' in state:
            with simulator._lock:
                simulator._event_log = deque(maxlen=1000)
                for event_data in state['event_log']:
                    simulator._event_log.append({
                        'timestamp': datetime.fromisoformat(event_data['timestamp']),
                        'message': event_data['message']
                    })
        
        return simulator
    
    def save_state(self, filename: str) -> bool:
        """
        Save simulation state to file
        
        Args:
            filename: File path to save state
            
        Returns:
            Boolean indicating success
        """
        try:
            # Get state
            state = self.to_dict()
            
            # Save to file
            with open(filename, 'w') as f:
                json.dump(state, f, indent=4)
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error saving simulation state: {e}")
            return False
    
    @classmethod
    def load_state(cls, filename: str, db_manager=None, risk_manager=None, thread_manager=None) -> Optional['ExecutionSimulator']:
        """
        Load simulation state from file
        
        Args:
            filename: File path to load state from
            db_manager: Database manager instance
            risk_manager: Risk manager instance
            thread_manager: Thread manager instance
            
        Returns:
            ExecutionSimulator instance with loaded state or None if failed
        """
        try:
            # Load from file
            with open(filename, 'r') as f:
                state = json.load(f)
            
            # Create simulator from state
            return cls.from_dict(state, db_manager, risk_manager, thread_manager)
            
        except Exception as e:
            logging.error(f"Error loading simulation state: {e}")
            return None

# Decorate with error handling if available
if HAVE_ERROR_HANDLING:
    ExecutionSimulator.place_order = safe_execute(ErrorCategory.TRADE_EXECUTION, 
                                                 {"status": "error", "message": "Order execution failed"})(ExecutionSimulator.place_order)
    ExecutionSimulator.cancel_order = safe_execute(ErrorCategory.TRADE_EXECUTION,
                                                  {"status": "error", "message": "Order cancellation failed"})(ExecutionSimulator.cancel_order)
