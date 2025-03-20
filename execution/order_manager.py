# order_manager.py

import time
import uuid
import logging
import threading
import json
import pandas as pd
from typing import Dict, List, Optional, Union, Tuple, Any, Callable
from datetime import datetime
from enum import Enum
import traceback

# Try to import error handling, with fallback
try:
    from core.error_handling import safe_execute, ErrorCategory, ErrorSeverity, ErrorHandler, TradingSystemError
    HAVE_ERROR_HANDLING = True
except ImportError:
    HAVE_ERROR_HANDLING = False
    logging.warning("Error handling module not available. Using basic error handling.")

class OrderType(Enum):
    """Types of orders supported by the system."""
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"
    TAKE_PROFIT = "take_profit"
    TAKE_PROFIT_LIMIT = "take_profit_limit"
    TRAILING_STOP = "trailing_stop"

class OrderSide(Enum):
    """Trading sides."""
    BUY = "buy"
    SELL = "sell"

class OrderStatus(Enum):
    """Possible states of an order."""
    CREATED = "created"           # Initial state
    VALIDATING = "validating"     # Order being validated by risk checks
    REJECTED = "rejected"         # Order rejected by risk checks
    PENDING = "pending"           # Order queued for submission
    SUBMITTING = "submitting"     # Order being submitted to exchange
    SUBMITTED = "submitted"       # Order successfully submitted to exchange
    OPEN = "open"                 # Order is active on the exchange
    PARTIALLY_FILLED = "partially_filled"  # Order partially executed
    FILLED = "filled"             # Order fully executed
    CANCELLING = "cancelling"     # Order cancellation in progress
    CANCELLED = "cancelled"       # Order successfully cancelled
    EXPIRED = "expired"           # Order expired on the exchange
    FAILED = "failed"             # Order processing failed

class Order:
    """
    Represents a trading order with complete lifecycle tracking.
    
    This class encapsulates all information about an order including:
    - Basic order parameters (symbol, side, type, quantity, price)
    - Current status and execution details
    - Related orders (stop-loss, take-profit)
    - Execution history
    """
    
    def __init__(
        self,
        symbol: str,
        side: Union[OrderSide, str],
        order_type: Union[OrderType, str],
        quantity: float,
        price: Optional[float] = None,
        stop_price: Optional[float] = None,
        client_order_id: Optional[str] = None,
        exchange: str = "binance",
        reduce_only: bool = False,
        time_in_force: str = "GTC",
        leverage: int = 1,
        post_only: bool = False,
        iceberg_quantity: Optional[float] = None,
        strategy_id: Optional[str] = None,
        strategy_name: Optional[str] = None,
        meta: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize an order with all required parameters.
        
        Args:
            symbol: Trading symbol (e.g., 'BTC/USDT')
            side: Order side ('buy' or 'sell')
            order_type: Order type ('market', 'limit', etc.)
            quantity: Order quantity
            price: Order price (required for limit orders)
            stop_price: Trigger price for stop orders
            client_order_id: Custom order ID for tracking
            exchange: Exchange to place order on
            reduce_only: Whether order should only reduce position
            time_in_force: Time in force policy ('GTC', 'IOC', 'FOK')
            leverage: Leverage to use (for margin/futures)
            post_only: Whether order must be maker
            iceberg_quantity: Visible quantity for iceberg orders
            strategy_id: ID of strategy that generated this order
            strategy_name: Name of strategy that generated this order
            meta: Additional metadata for the order
        """
        # Core order info
        self.symbol = symbol
        
        # Convert string enums to proper Enum types if provided as strings
        self.side = side if isinstance(side, OrderSide) else OrderSide(side)
        self.order_type = order_type if isinstance(order_type, OrderType) else OrderType(order_type)
        
        self.quantity = float(quantity)
        self.price = float(price) if price is not None else None
        self.stop_price = float(stop_price) if stop_price is not None else None
        
        # Ensure price is provided for limit orders
        if self.order_type in [OrderType.LIMIT, OrderType.STOP_LIMIT, OrderType.TAKE_PROFIT_LIMIT] and self.price is None:
            raise ValueError(f"Price must be specified for {self.order_type.value} orders")
            
        # Ensure stop price is provided for stop orders
        if self.order_type in [OrderType.STOP, OrderType.STOP_LIMIT, OrderType.TRAILING_STOP] and self.stop_price is None:
            raise ValueError(f"Stop price must be specified for {self.order_type.value} orders")
        
        # Generate a unique client order ID if not provided
        self.client_order_id = client_order_id or f"order_{uuid.uuid4().hex[:12]}_{int(time.time())}"
        
        # Exchange and execution parameters
        self.exchange = exchange
        self.reduce_only = reduce_only
        self.time_in_force = time_in_force
        self.leverage = leverage
        self.post_only = post_only
        self.iceberg_quantity = iceberg_quantity
        
        # Strategy information
        self.strategy_id = strategy_id
        self.strategy_name = strategy_name
        
        # Additional metadata (JSON serializable)
        self.meta = meta or {}
        
        # Order state tracking
        self.status = OrderStatus.CREATED
        self.exchange_order_id = None
        self.filled_quantity = 0
        self.avg_fill_price = None
        self.last_fill_time = None
        self.created_time = datetime.now().timestamp()
        self.updated_time = self.created_time
        self.submitted_time = None
        self.completed_time = None
        
        # Linked orders
        self.parent_order_id = None  # For child orders (SL/TP)
        self.child_order_ids = []    # For parent orders
        
        # Execution history
        self.status_history = [(self.status, self.created_time)]
        self.fill_history = []       # List of fills with prices and quantities
        self.error_history = []      # List of errors encountered during order lifecycle
        
        # Thread safety
        self._lock = threading.RLock()
    
    def update_status(self, new_status: Union[OrderStatus, str], timestamp: Optional[float] = None) -> None:
        """
        Update the order status with thread safety.
        
        Args:
            new_status: New order status
            timestamp: Optional timestamp for the update (defaults to current time)
        """
        with self._lock:
            # Convert string to enum if needed
            if isinstance(new_status, str):
                new_status = OrderStatus(new_status)
                
            # Set timestamp if not provided
            if timestamp is None:
                timestamp = datetime.now().timestamp()
                
            # Update status and timestamps
            old_status = self.status
            self.status = new_status
            self.updated_time = timestamp
            
            # Set special timestamps based on status
            if new_status == OrderStatus.SUBMITTED:
                self.submitted_time = timestamp
                
            if new_status in [OrderStatus.FILLED, OrderStatus.CANCELLED, OrderStatus.REJECTED, OrderStatus.EXPIRED, OrderStatus.FAILED]:
                self.completed_time = timestamp
                
            # Add to status history
            self.status_history.append((new_status, timestamp))
            
            logging.info(f"Order {self.client_order_id} status updated: {old_status.value} -> {new_status.value}")
    
    def update_fill(self, filled_quantity: float, fill_price: float, timestamp: Optional[float] = None) -> None:
        """
        Update order fill information with thread safety.
        
        Args:
            filled_quantity: Additional quantity filled
            fill_price: Price of this fill
            timestamp: Optional timestamp for the fill (defaults to current time)
        """
        with self._lock:
            # Set timestamp if not provided
            if timestamp is None:
                timestamp = datetime.now().timestamp()
                
            # Calculate new total filled quantity
            old_filled_quantity = self.filled_quantity
            new_fill_quantity = min(filled_quantity, self.quantity - old_filled_quantity)
            self.filled_quantity = old_filled_quantity + new_fill_quantity
            
            # Calculate new average fill price
            if self.avg_fill_price is None:
                self.avg_fill_price = fill_price
            else:
                # Weighted average of previous fills and new fill
                self.avg_fill_price = (
                    (old_filled_quantity * self.avg_fill_price + new_fill_quantity * fill_price) / 
                    self.filled_quantity
                )
            
            # Update timestamps
            self.last_fill_time = timestamp
            self.updated_time = timestamp
            
            # Add to fill history
            self.fill_history.append({
                'quantity': new_fill_quantity,
                'price': fill_price,
                'timestamp': timestamp
            })
            
            # Update status if fully filled
            if self.is_fully_filled():
                self.update_status(OrderStatus.FILLED, timestamp)
            elif self.filled_quantity > 0 and not self.is_fully_filled():
                self.update_status(OrderStatus.PARTIALLY_FILLED, timestamp)
                
            logging.info(f"Order {self.client_order_id} fill update: +{new_fill_quantity} @ {fill_price} (Total: {self.filled_quantity}/{self.quantity})")
    
    def add_error(self, error_message: str, error_code: Optional[str] = None, timestamp: Optional[float] = None) -> None:
        """
        Add an error to the order's error history with thread safety.
        
        Args:
            error_message: Description of the error
            error_code: Optional error code from exchange
            timestamp: Optional timestamp (defaults to current time)
        """
        with self._lock:
            # Set timestamp if not provided
            if timestamp is None:
                timestamp = datetime.now().timestamp()
                
            # Add to error history
            self.error_history.append({
                'message': error_message,
                'code': error_code,
                'timestamp': timestamp
            })
            
            # Update timestamp
            self.updated_time = timestamp
            
            logging.warning(f"Order {self.client_order_id} error: {error_message} (Code: {error_code})")
    
    def update_from_exchange(self, exchange_data: Dict[str, Any]) -> None:
        """
        Update order with data from exchange response with thread safety.
        
        Args:
            exchange_data: Order data from exchange API response
        """
        with self._lock:
            # Extract common fields (exchange-specific implementations may differ)
            if 'id' in exchange_data:
                self.exchange_order_id = str(exchange_data['id'])
                
            # Update status if present in data
            if 'status' in exchange_data:
                # Map exchange status to internal status enum
                exchange_status = exchange_data['status'].lower()
                status_mapping = {
                    'new': OrderStatus.OPEN,
                    'open': OrderStatus.OPEN,
                    'partially_filled': OrderStatus.PARTIALLY_FILLED,
                    'filled': OrderStatus.FILLED,
                    'canceled': OrderStatus.CANCELLED,
                    'cancelled': OrderStatus.CANCELLED,
                    'expired': OrderStatus.EXPIRED,
                    'rejected': OrderStatus.REJECTED
                }
                
                if exchange_status in status_mapping:
                    self.update_status(status_mapping[exchange_status])
                    
            # Update fill information if present
            if 'filled' in exchange_data and exchange_data['filled'] > 0:
                filled_qty = float(exchange_data['filled'])
                
                # Only update if we have more fills than currently tracked
                if filled_qty > self.filled_quantity:
                    # Calculate new fill amount
                    new_fill = filled_qty - self.filled_quantity
                    
                    # Get fill price
                    fill_price = float(exchange_data.get('price', 0))
                    
                    # If average price is available, use that
                    if 'average' in exchange_data and exchange_data['average']:
                        fill_price = float(exchange_data['average'])
                        
                    # Update fill information
                    self.update_fill(new_fill, fill_price)
            
            # Update price if modified
            if 'price' in exchange_data and exchange_data['price'] is not None:
                self.price = float(exchange_data['price'])
                
            # Update timestamps if available
            if 'timestamp' in exchange_data and exchange_data['timestamp']:
                timestamp = float(exchange_data['timestamp']) / 1000.0  # Convert from milliseconds
                self.updated_time = timestamp
                
            logging.debug(f"Order {self.client_order_id} updated from exchange data")
    
    def is_active(self) -> bool:
        """Check if the order is currently active on the exchange."""
        with self._lock:
            return self.status in [OrderStatus.OPEN, OrderStatus.PARTIALLY_FILLED, OrderStatus.SUBMITTING, OrderStatus.SUBMITTED]
    
    def is_complete(self) -> bool:
        """Check if the order has reached a terminal state."""
        with self._lock:
            return self.status in [OrderStatus.FILLED, OrderStatus.CANCELLED, OrderStatus.REJECTED, OrderStatus.EXPIRED, OrderStatus.FAILED]
    
    def is_fully_filled(self) -> bool:
        """Check if the order is fully filled."""
        with self._lock:
            # Allow for small floating point differences
            return abs(self.filled_quantity - self.quantity) < 1e-8
    
    def get_remaining_quantity(self) -> float:
        """Get unfilled quantity."""
        with self._lock:
            return max(0, self.quantity - self.filled_quantity)
    
    def get_fill_value(self) -> float:
        """Get total fill value (quantity * avg_price)."""
        with self._lock:
            if self.avg_fill_price is None:
                return 0.0
            return self.filled_quantity * self.avg_fill_price
    
    def get_estimated_value(self) -> float:
        """Get estimated value of the order."""
        with self._lock:
            # For filled portion, use actual fill price
            filled_value = self.get_fill_value()
            
            # For unfilled portion, use order price
            remaining_qty = self.get_remaining_quantity()
            if remaining_qty > 0:
                # Use either order price or stop price depending on order type
                price = self.price if self.price is not None else self.stop_price
                if price is not None:
                    return filled_value + (remaining_qty * price)
                    
            return filled_value
    
    def get_parent_order(self, order_manager: 'OrderManager') -> Optional['Order']:
        """Get parent order if this is a child order."""
        if self.parent_order_id:
            return order_manager.get_order(self.parent_order_id)
        return None
    
    def get_child_orders(self, order_manager: 'OrderManager') -> List['Order']:
        """Get all child orders of this order."""
        return [order_manager.get_order(order_id) for order_id in self.child_order_ids 
                if order_manager.get_order(order_id) is not None]
    
    def add_child_order(self, child_order_id: str) -> None:
        """
        Add a child order ID to this order with thread safety.
        
        Args:
            child_order_id: ID of child order to add
        """
        with self._lock:
            if child_order_id not in self.child_order_ids:
                self.child_order_ids.append(child_order_id)
                self.updated_time = datetime.now().timestamp()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert order to dictionary for serialization."""
        with self._lock:
            return {
                # Core order info
                'client_order_id': self.client_order_id,
                'exchange_order_id': self.exchange_order_id,
                'symbol': self.symbol,
                'side': self.side.value,
                'order_type': self.order_type.value,
                'quantity': self.quantity,
                'price': self.price,
                'stop_price': self.stop_price,
                
                # Exchange parameters
                'exchange': self.exchange,
                'reduce_only': self.reduce_only,
                'time_in_force': self.time_in_force,
                'leverage': self.leverage,
                'post_only': self.post_only,
                'iceberg_quantity': self.iceberg_quantity,
                
                # Strategy information
                'strategy_id': self.strategy_id,
                'strategy_name': self.strategy_name,
                'meta': self.meta,
                
                # Order state
                'status': self.status.value,
                'filled_quantity': self.filled_quantity,
                'avg_fill_price': self.avg_fill_price,
                'last_fill_time': self.last_fill_time,
                
                # Timestamps
                'created_time': self.created_time,
                'updated_time': self.updated_time,
                'submitted_time': self.submitted_time,
                'completed_time': self.completed_time,
                
                # Linked orders
                'parent_order_id': self.parent_order_id,
                'child_order_ids': self.child_order_ids,
                
                # Histories (converted to serializable format)
                'status_history': [(status.value if isinstance(status, OrderStatus) else status, timestamp) 
                                  for status, timestamp in self.status_history],
                'fill_history': self.fill_history,
                'error_history': self.error_history
            }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Order':
        """
        Create an order instance from a dictionary.
        
        Args:
            data: Dictionary representation of an order
        
        Returns:
            Order instance created from the dictionary
        """
        # Create new order instance with core parameters
        order = cls(
            symbol=data['symbol'],
            side=data['side'],
            order_type=data['order_type'],
            quantity=data['quantity'],
            price=data.get('price'),
            stop_price=data.get('stop_price'),
            client_order_id=data['client_order_id'],
            exchange=data.get('exchange', 'binance'),
            reduce_only=data.get('reduce_only', False),
            time_in_force=data.get('time_in_force', 'GTC'),
            leverage=data.get('leverage', 1),
            post_only=data.get('post_only', False),
            iceberg_quantity=data.get('iceberg_quantity'),
            strategy_id=data.get('strategy_id'),
            strategy_name=data.get('strategy_name'),
            meta=data.get('meta', {})
        )
        
        # Set additional fields from the dictionary
        order.exchange_order_id = data.get('exchange_order_id')
        order.status = OrderStatus(data['status'])
        order.filled_quantity = data.get('filled_quantity', 0)
        order.avg_fill_price = data.get('avg_fill_price')
        order.last_fill_time = data.get('last_fill_time')
        
        # Set timestamps
        order.created_time = data.get('created_time', order.created_time)
        order.updated_time = data.get('updated_time', order.updated_time)
        order.submitted_time = data.get('submitted_time')
        order.completed_time = data.get('completed_time')
        
        # Set linked orders
        order.parent_order_id = data.get('parent_order_id')
        order.child_order_ids = data.get('child_order_ids', [])
        
        # Set histories
        if 'status_history' in data:
            order.status_history = [(OrderStatus(status) if isinstance(status, str) else status, timestamp) 
                                  for status, timestamp in data['status_history']]
        
        order.fill_history = data.get('fill_history', [])
        order.error_history = data.get('error_history', [])
        
        return order


class OrderManager:
    """
    Manages all orders in the trading system with comprehensive lifecycle tracking.
    
    This class provides:
    - Order creation, modification, and cancellation
    - Integration with exchange connector for execution
    - Risk checking before order submission
    - Automatic handling of stop-loss and take-profit orders
    - Order status updates and synchronization with exchange
    - Persistence of order data to database
    - Thread-safe operations
    """
    
    def __init__(
        self, 
        exchange_connector=None, 
        risk_manager=None, 
        db_manager=None, 
        thread_manager=None,
        position_manager=None,
        max_order_history: int = 1000,
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize the order manager with required dependencies.
        
        Args:
            exchange_connector: Connector for executing orders on exchanges
            risk_manager: Risk management system for pre-trade checks
            db_manager: Database manager for order persistence
            thread_manager: Thread manager for async operations
            position_manager: Position manager for tracking positions
            max_order_history: Maximum number of completed orders to keep in memory
            config: Additional configuration parameters
        """
        self.exchange_connector = exchange_connector
        self.risk_manager = risk_manager
        self.db_manager = db_manager
        self.thread_manager = thread_manager
        self.position_manager = position_manager
        self.max_order_history = max_order_history
        self.config = config or {}
        
        # Order storage
        self.active_orders = {}  # client_order_id -> Order
        self.completed_orders = {}  # client_order_id -> Order
        
        # Exchange ID mapping
        self.exchange_id_map = {}  # exchange_order_id -> client_order_id
        
        # Thread safety
        self._orders_lock = threading.RLock()
        
        # Track total orders processed for stats
        self.total_orders_created = 0
        self.total_orders_filled = 0
        self.total_orders_cancelled = 0
        self.total_orders_rejected = 0
        self.total_orders_failed = 0
        
        # Internal callbacks
        self._status_callbacks = []  # List of (callback_id, function) for status updates
        self._fill_callbacks = []    # List of (callback_id, function) for fill updates
        
        # Initialize
        self._initialize()
        
        logging.info("Order Manager initialized")
    
    def _initialize(self) -> None:
        """Perform initial setup tasks."""
        # Create exchange connector if None but a name was provided
        if self.exchange_connector is None and 'exchange' in self.config:
            try:
                from execution.exchange_connector import create_exchange_connector
                self.exchange_connector = create_exchange_connector(self.config['exchange'])
                logging.info(f"Created exchange connector for {self.config['exchange']}")
            except (ImportError, Exception) as e:
                logging.warning(f"Failed to create exchange connector: {e}")
        
        # Load active orders from database if available
        self._load_active_orders()
    
    def _load_active_orders(self) -> None:
        """Load active orders from database."""
        if self.db_manager is None:
            return
            
        try:
            # Query active orders
            active_orders_df = self.db_manager.get_open_trades()
            
            if active_orders_df is None or active_orders_df.empty:
                logging.info("No active orders found in database")
                return
                
            # Convert to Order objects
            for _, row in active_orders_df.iterrows():
                try:
                    # Create order from database row
                    order_data = dict(row)
                    
                    # Convert JSON columns if needed
                    for field in ['meta', 'status_history', 'fill_history', 'error_history']:
                        if field in order_data and isinstance(order_data[field], str):
                            try:
                                order_data[field] = json.loads(order_data[field])
                            except (json.JSONDecodeError, ValueError):
                                order_data[field] = {}
                    
                    # Create order and add to active orders
                    order = Order.from_dict(order_data)
                    
                    # Only add if still active (not terminal state)
                    if not order.is_complete():
                        with self._orders_lock:
                            self.active_orders[order.client_order_id] = order
                            
                            # Update exchange ID mapping if available
                            if order.exchange_order_id:
                                self.exchange_id_map[order.exchange_order_id] = order.client_order_id
                except Exception as e:
                    logging.error(f"Error loading order from database: {e}")
                    
            logging.info(f"Loaded {len(self.active_orders)} active orders from database")
            
        except Exception as e:
            logging.error(f"Failed to load active orders from database: {e}")
    
    def register_status_callback(self, callback: Callable[[str, OrderStatus], None]) -> str:
        """
        Register a callback for order status updates.
        
        Args:
            callback: Function to call when order status changes
                     Will be called with (client_order_id, new_status)
        
        Returns:
            str: Callback ID for later removal
        """
        callback_id = str(uuid.uuid4())
        self._status_callbacks.append((callback_id, callback))
        return callback_id
    
    def register_fill_callback(self, callback: Callable[[str, float, float], None]) -> str:
        """
        Register a callback for order fill updates.
        
        Args:
            callback: Function to call when order gets a fill
                     Will be called with (client_order_id, filled_quantity, fill_price)
        
        Returns:
            str: Callback ID for later removal
        """
        callback_id = str(uuid.uuid4())
        self._fill_callbacks.append((callback_id, callback))
        return callback_id
    
    def unregister_callback(self, callback_id: str) -> bool:
        """
        Unregister a previously registered callback.
        
        Args:
            callback_id: ID of callback to remove
            
        Returns:
            bool: True if callback was found and removed
        """
        # Check status callbacks
        for i, (cb_id, _) in enumerate(self._status_callbacks):
            if cb_id == callback_id:
                self._status_callbacks.pop(i)
                return True
                
        # Check fill callbacks
        for i, (cb_id, _) in enumerate(self._fill_callbacks):
            if cb_id == callback_id:
                self._fill_callbacks.pop(i)
                return True
                
        return False
    
    def create_order(
        self,
        symbol: str,
        side: Union[OrderSide, str],
        order_type: Union[OrderType, str],
        quantity: float,
        price: Optional[float] = None,
        stop_price: Optional[float] = None,
        client_order_id: Optional[str] = None,
        exchange: Optional[str] = None,
        reduce_only: bool = False,
        time_in_force: str = "GTC",
        leverage: int = 1,
        post_only: bool = False,
        iceberg_quantity: Optional[float] = None,
        strategy_id: Optional[str] = None,
        strategy_name: Optional[str] = None,
        meta: Optional[Dict[str, Any]] = None,
        stop_loss: Optional[Dict[str, Any]] = None,
        take_profit: Optional[Dict[str, Any]] = None,
        submit_immediately: bool = True,
        attach_to_position: bool = False
    ) -> Tuple[str, Order]:
        """
        Create a new order with comprehensive parameters and error handling.
        
        Args:
            symbol: Trading symbol (e.g., 'BTC/USDT')
            side: Order side ('buy' or 'sell')
            order_type: Order type ('market', 'limit', etc.)
            quantity: Order quantity
            price: Order price (required for limit orders)
            stop_price: Trigger price for stop orders
            client_order_id: Custom order ID (generated if not provided)
            exchange: Exchange to place order on (uses default if None)
            reduce_only: Whether order should only reduce position
            time_in_force: Time in force policy ('GTC', 'IOC', 'FOK')
            leverage: Leverage to use (for margin/futures)
            post_only: Whether order must be maker
            iceberg_quantity: Visible quantity for iceberg orders
            strategy_id: ID of strategy that generated this order
            strategy_name: Name of strategy that generated this order
            meta: Additional metadata for the order
            stop_loss: Parameters for stop-loss order 
                       e.g., {'price': 9800, 'type': 'stop_market'}
            take_profit: Parameters for take-profit order
                         e.g., {'price': 10200, 'type': 'limit'}
            submit_immediately: Whether to submit the order immediately
            attach_to_position: Whether to attach to existing position
            
        Returns:
            tuple: (client_order_id, Order object)
        """
        try:
            # Validate parameters
            if quantity <= 0:
                raise ValueError("Quantity must be greater than zero")
                
            # Use default exchange if not specified
            if exchange is None:
                exchange = self.config.get('default_exchange', 'binance')
            
            # Create the order object
            order = Order(
                symbol=symbol,
                side=side,
                order_type=order_type,
                quantity=quantity,
                price=price,
                stop_price=stop_price,
                client_order_id=client_order_id,
                exchange=exchange,
                reduce_only=reduce_only,
                time_in_force=time_in_force,
                leverage=leverage,
                post_only=post_only,
                iceberg_quantity=iceberg_quantity,
                strategy_id=strategy_id,
                strategy_name=strategy_name,
                meta=meta or {}
            )
            
            # Add to active orders (thread-safe)
            with self._orders_lock:
                self.active_orders[order.client_order_id] = order
                self.total_orders_created += 1
            
            # Log order creation
            logging.info(f"Created order {order.client_order_id}: {symbol} {side} {order_type} {quantity} @ {price}")
            
            # Submit the order if requested
            if submit_immediately:
                self._submit_order_async(order.client_order_id)
                
            # Create stop-loss and take-profit orders if specified
            sl_order_id = None
            tp_order_id = None
            
            if stop_loss and order.order_type not in [OrderType.STOP, OrderType.STOP_LIMIT]:
                sl_order_id = self._create_stop_loss_order(order, stop_loss)
                
            if take_profit and order.order_type not in [OrderType.TAKE_PROFIT, OrderType.TAKE_PROFIT_LIMIT]:
                tp_order_id = self._create_take_profit_order(order, take_profit)
                
            # Attach to position if requested
            if attach_to_position and self.position_manager is not None:
                try:
                    self.position_manager.attach_order_to_position(symbol, order.client_order_id, sl_order_id, tp_order_id)
                except Exception as e:
                    logging.error(f"Error attaching order to position: {e}")
            
            # Store order in database if available
            self._store_order(order)
            
            return order.client_order_id, order
            
        except Exception as e:
            error_msg = f"Error creating order: {e}"
            logging.error(f"{error_msg}\n{traceback.format_exc()}")
            
            if HAVE_ERROR_HANDLING:
                ErrorHandler.handle_error(
                    error=e,
                    category=ErrorCategory.TRADE_EXECUTION,
                    severity=ErrorSeverity.ERROR,
                    context={"operation": "create_order", "symbol": symbol, "side": side}
                )
            
            # Create a rejected order to track the failure
            if not client_order_id:
                client_order_id = f"rejected_{uuid.uuid4().hex[:12]}_{int(time.time())}"
                
            rejected_order = Order(
                symbol=symbol,
                side=side,
                order_type=order_type,
                quantity=quantity,
                price=price,
                stop_price=stop_price,
                client_order_id=client_order_id,
                exchange=exchange,
                meta={"rejection_reason": str(e)}
            )
            
            rejected_order.update_status(OrderStatus.REJECTED)
            rejected_order.add_error(str(e))
            
            # Add to completed orders
            with self._orders_lock:
                self.completed_orders[rejected_order.client_order_id] = rejected_order
                self.total_orders_rejected += 1
            
            # Store rejected order in database
            self._store_order(rejected_order)
            
            return rejected_order.client_order_id, rejected_order
    
    def _create_stop_loss_order(self, parent_order: Order, stop_loss_params: Dict[str, Any]) -> Optional[str]:
        """
        Create a stop-loss order linked to a parent order.
        
        Args:
            parent_order: The parent order
            stop_loss_params: Parameters for the stop-loss order
            
        Returns:
            Optional[str]: Client order ID of the stop-loss order or None if creation failed
        """
        try:
            # Extract parameters
            sl_price = stop_loss_params.get('price')
            if sl_price is None:
                logging.error(f"Stop-loss price not specified for order {parent_order.client_order_id}")
                return None
                
            # Determine order type
            sl_type = stop_loss_params.get('type', 'stop_market')
            order_type = OrderType.STOP
            
            if sl_type == 'stop_limit':
                order_type = OrderType.STOP_LIMIT
                # Limit price defaults to stop price if not specified
                limit_price = stop_loss_params.get('limit_price', sl_price)
            else:
                # For market stop-loss, price is None
                limit_price = None
            
            # Determine side (opposite of parent order)
            sl_side = OrderSide.SELL if parent_order.side == OrderSide.BUY else OrderSide.BUY
            
            # Generate client order ID
            sl_client_order_id = f"sl_{parent_order.client_order_id}"
            
            # Create stop-loss order
            sl_order_id, sl_order = self.create_order(
                symbol=parent_order.symbol,
                side=sl_side,
                order_type=order_type,
                quantity=parent_order.quantity,
                price=limit_price,
                stop_price=sl_price,
                client_order_id=sl_client_order_id,
                exchange=parent_order.exchange,
                reduce_only=True,  # Stop-loss always reduces position
                time_in_force=parent_order.time_in_force,
                leverage=parent_order.leverage,
                strategy_id=parent_order.strategy_id,
                strategy_name=parent_order.strategy_name,
                meta={"parent_order_id": parent_order.client_order_id, "order_type": "stop_loss"},
                submit_immediately=False  # Don't submit until parent is filled
            )
            
            # Link the orders
            sl_order.parent_order_id = parent_order.client_order_id
            parent_order.add_child_order(sl_order_id)
            
            # Update status - stop-loss enters pending state until parent order is filled
            sl_order.update_status(OrderStatus.PENDING)
            
            logging.info(f"Created stop-loss order {sl_order_id} for parent order {parent_order.client_order_id}")
            
            return sl_order_id
            
        except Exception as e:
            logging.error(f"Error creating stop-loss order: {e}")
            
            if HAVE_ERROR_HANDLING:
                ErrorHandler.handle_error(
                    error=e,
                    category=ErrorCategory.TRADE_EXECUTION,
                    severity=ErrorSeverity.ERROR,
                    context={"operation": "create_stop_loss", "parent_order": parent_order.client_order_id}
                )
                
            return None
    
    def _create_take_profit_order(self, parent_order: Order, take_profit_params: Dict[str, Any]) -> Optional[str]:
        """
        Create a take-profit order linked to a parent order.
        
        Args:
            parent_order: The parent order
            take_profit_params: Parameters for the take-profit order
            
        Returns:
            Optional[str]: Client order ID of the take-profit order or None if creation failed
        """
        try:
            # Extract parameters
            tp_price = take_profit_params.get('price')
            if tp_price is None:
                logging.error(f"Take-profit price not specified for order {parent_order.client_order_id}")
                return None
                
            # Determine order type
            tp_type = take_profit_params.get('type', 'limit')
            
            if tp_type == 'take_profit_market':
                order_type = OrderType.TAKE_PROFIT
                limit_price = None
            elif tp_type == 'take_profit_limit':
                order_type = OrderType.TAKE_PROFIT_LIMIT
                # Limit price defaults to take profit price
                limit_price = take_profit_params.get('limit_price', tp_price)
            else:
                # Default to limit order
                order_type = OrderType.LIMIT
                limit_price = tp_price
            
            # Determine side (opposite of parent order)
            tp_side = OrderSide.SELL if parent_order.side == OrderSide.BUY else OrderSide.BUY
            
            # Generate client order ID
            tp_client_order_id = f"tp_{parent_order.client_order_id}"
            
            # Create take-profit order
            tp_order_id, tp_order = self.create_order(
                symbol=parent_order.symbol,
                side=tp_side,
                order_type=order_type,
                quantity=parent_order.quantity,
                price=limit_price,
                stop_price=tp_price if order_type in [OrderType.TAKE_PROFIT, OrderType.TAKE_PROFIT_LIMIT] else None,
                client_order_id=tp_client_order_id,
                exchange=parent_order.exchange,
                reduce_only=True,  # Take-profit always reduces position
                time_in_force=parent_order.time_in_force,
                leverage=parent_order.leverage,
                strategy_id=parent_order.strategy_id,
                strategy_name=parent_order.strategy_name,
                meta={"parent_order_id": parent_order.client_order_id, "order_type": "take_profit"},
                submit_immediately=False  # Don't submit until parent is filled
            )
            
            # Link the orders
            tp_order.parent_order_id = parent_order.client_order_id
            parent_order.add_child_order(tp_order_id)
            
            # Update status - take-profit enters pending state until parent order is filled
            tp_order.update_status(OrderStatus.PENDING)
            
            logging.info(f"Created take-profit order {tp_order_id} for parent order {parent_order.client_order_id}")
            
            return tp_order_id
            
        except Exception as e:
            logging.error(f"Error creating take-profit order: {e}")
            
            if HAVE_ERROR_HANDLING:
                ErrorHandler.handle_error(
                    error=e,
                    category=ErrorCategory.TRADE_EXECUTION,
                    severity=ErrorSeverity.ERROR,
                    context={"operation": "create_take_profit", "parent_order": parent_order.client_order_id}
                )
                
            return None
    
    def _store_order(self, order: Order) -> bool:
        """
        Store an order in the database.
        
        Args:
            order: Order to store
            
        Returns:
            bool: True if storage was successful
        """
        if self.db_manager is None:
            return False
            
        try:
            # Convert order to dict
            order_dict = order.to_dict()
            
            # Convert non-serializable fields to JSON
            for field in ['meta', 'status_history', 'fill_history', 'error_history']:
                if field in order_dict and not isinstance(order_dict[field], str):
                    order_dict[field] = json.dumps(order_dict[field])
            
            # Map to database fields
            trade_data = {
                'order_id': order.client_order_id,
                'exchange_order_id': order.exchange_order_id,
                'symbol': order.symbol,
                'action': order.side.value,
                'price': order.price,
                'amount': order.quantity,
                'stop_loss': order.stop_price if order.order_type in [OrderType.STOP, OrderType.STOP_LIMIT] else None,
                'take_profit': order.price if order.order_type in [OrderType.TAKE_PROFIT, OrderType.TAKE_PROFIT_LIMIT] else None,
                'status': order.status.value,
                'order_type': order.order_type.value,
                'strategy': order.strategy_name,
                'meta': json.dumps(order.meta) if not isinstance(order.meta, str) else order.meta,
                'time': datetime.fromtimestamp(order.created_time),
                'exit_price': order.avg_fill_price if order.status in [OrderStatus.FILLED, OrderStatus.PARTIALLY_FILLED] else None,
                'exit_time': datetime.fromtimestamp(order.last_fill_time) if order.last_fill_time else None,
                'pnl': None,  # Calculate PnL if needed
                'exit_reason': None  # Fill in exit reason if available
            }
            
            # Store or update order
            if order.exchange_order_id:
                # Update existing order
                self.db_manager.update_trade(order.client_order_id, trade_data)
            else:
                # Store new order
                self.db_manager.store_trade(trade_data)
                
            return True
            
        except Exception as e:
            logging.error(f"Error storing order in database: {e}")
            
            if HAVE_ERROR_HANDLING:
                ErrorHandler.handle_error(
                    error=e,
                    category=ErrorCategory.DATABASE,
                    severity=ErrorSeverity.WARNING,
                    context={"operation": "store_order", "order_id": order.client_order_id}
                )
                
            return False
    
    def get_order(self, order_id: str) -> Optional[Order]:
        """
        Get an order by its client_order_id.
        
        Args:
            order_id: Client order ID
            
        Returns:
            Optional[Order]: Order object or None if not found
        """
        with self._orders_lock:
            # Check active orders first
            if order_id in self.active_orders:
                return self.active_orders[order_id]
                
            # Then check completed orders
            if order_id in self.completed_orders:
                return self.completed_orders[order_id]
                
        return None
    
    def get_order_by_exchange_id(self, exchange_id: str) -> Optional[Order]:
        """
        Get an order by its exchange order ID.
        
        Args:
            exchange_id: Exchange order ID
            
        Returns:
            Optional[Order]: Order object or None if not found
        """
        with self._orders_lock:
            # Look up client order ID from exchange ID
            if exchange_id in self.exchange_id_map:
                client_id = self.exchange_id_map[exchange_id]
                return self.get_order(client_id)
                
        # If not found in memory, try to fetch from exchange
        if self.exchange_connector:
            try:
                order_data = self.exchange_connector.fetch_order(exchange_id)
                if order_data and 'clientOrderId' in order_data:
                    return self.get_order(order_data['clientOrderId'])
            except Exception as e:
                logging.error(f"Error fetching order by exchange ID: {e}")
                
        return None
    
    def get_orders(
        self, 
        symbol: Optional[str] = None, 
        status: Optional[Union[OrderStatus, str, List[Union[OrderStatus, str]]]] = None,
        strategy_id: Optional[str] = None,
        strategy_name: Optional[str] = None,
        side: Optional[Union[OrderSide, str]] = None,
        order_type: Optional[Union[OrderType, str]] = None,
        active_only: bool = False,
        completed_only: bool = False,
        start_time: Optional[float] = None,
        end_time: Optional[float] = None,
        include_child_orders: bool = True
    ) -> List[Order]:
        """
        Get orders matching specified criteria.
        
        Args:
            symbol: Filter by symbol
            status: Filter by status (single status or list)
            strategy_id: Filter by strategy ID
            strategy_name: Filter by strategy name
            side: Filter by order side
            order_type: Filter by order type
            active_only: Only include active orders
            completed_only: Only include completed orders
            start_time: Filter by created time >= start_time
            end_time: Filter by created time <= end_time
            include_child_orders: Whether to include child orders (SL/TP)
            
        Returns:
            List[Order]: List of orders matching criteria
        """
        # Convert string enums to proper Enum types if provided as strings
        if isinstance(status, str):
            status = OrderStatus(status)
        elif isinstance(status, list):
            status = [OrderStatus(s) if isinstance(s, str) else s for s in status]
            
        if isinstance(side, str):
            side = OrderSide(side)
            
        if isinstance(order_type, str):
            order_type = OrderType(order_type)
        
        # Start with all orders
        orders = []
        
        with self._orders_lock:
            # Get appropriate order collections based on filters
            if active_only:
                orders_to_check = list(self.active_orders.values())
            elif completed_only:
                orders_to_check = list(self.completed_orders.values())
            else:
                orders_to_check = list(self.active_orders.values()) + list(self.completed_orders.values())
            
            # Apply filters
            filtered_orders = []
            
            for order in orders_to_check:
                # Skip child orders if requested
                if not include_child_orders and order.parent_order_id is not None:
                    continue
                    
                # Apply filters
                if symbol and order.symbol != symbol:
                    continue
                    
                if status:
                    if isinstance(status, list) and order.status not in status:
                        continue
                    elif not isinstance(status, list) and order.status != status:
                        continue
                        
                if strategy_id and order.strategy_id != strategy_id:
                    continue
                    
                if strategy_name and order.strategy_name != strategy_name:
                    continue
                    
                if side and order.side != side:
                    continue
                    
                if order_type and order.order_type != order_type:
                    continue
                    
                if start_time and order.created_time < start_time:
                    continue
                    
                if end_time and order.created_time > end_time:
                    continue
                    
                # All filters passed, add to result
                filtered_orders.append(order)
            
            # Sort by created time (newest first)
            filtered_orders.sort(key=lambda x: x.created_time, reverse=True)
            
            return filtered_orders
    
    def get_active_orders(self, symbol: Optional[str] = None) -> List[Order]:
        """
        Get all active orders, optionally filtered by symbol.
        
        Args:
            symbol: Filter by symbol
            
        Returns:
            List[Order]: List of active orders
        """
        return self.get_orders(symbol=symbol, active_only=True)
    
    def get_orders_dataframe(self, **kwargs) -> pd.DataFrame:
        """
        Get orders as a pandas DataFrame with specified filters.
        
        Args:
            **kwargs: Same filters as get_orders()
            
        Returns:
            pd.DataFrame: DataFrame of orders
        """
        orders = self.get_orders(**kwargs)
        
        if not orders:
            # Return empty dataframe with expected columns
            return pd.DataFrame(columns=[
                'client_order_id', 'exchange_order_id', 'symbol', 'side', 'type',
                'quantity', 'price', 'stop_price', 'status', 'filled_quantity',
                'avg_fill_price', 'created_time', 'updated_time', 'strategy_name'
            ])
            
        # Convert orders to dictionaries for DataFrame
        order_dicts = []
        
        for order in orders:
            order_dict = {
                'client_order_id': order.client_order_id,
                'exchange_order_id': order.exchange_order_id,
                'symbol': order.symbol,
                'side': order.side.value,
                'type': order.order_type.value,
                'quantity': order.quantity,
                'price': order.price,
                'stop_price': order.stop_price,
                'status': order.status.value,
                'filled_quantity': order.filled_quantity,
                'avg_fill_price': order.avg_fill_price,
                'created_time': datetime.fromtimestamp(order.created_time),
                'updated_time': datetime.fromtimestamp(order.updated_time),
                'strategy_name': order.strategy_name
            }
            
            order_dicts.append(order_dict)
            
        # Create DataFrame
        df = pd.DataFrame(order_dicts)
        
        # Set index to client_order_id
        if not df.empty and 'client_order_id' in df.columns:
            df.set_index('client_order_id', inplace=True)
            
        return df
    
    def submit_order(self, order_id: str) -> bool:
        """
        Submit an order to the exchange.
        
        Args:
            order_id: Client order ID to submit
            
        Returns:
            bool: True if submission was successful
        """
        # Get the order
        order = self.get_order(order_id)
        if not order:
            logging.error(f"Order not found: {order_id}")
            return False
            
        # Check if order is already submitted or complete
        if order.status not in [OrderStatus.CREATED, OrderStatus.VALIDATING, OrderStatus.PENDING, OrderStatus.REJECTED]:
            logging.warning(f"Order {order_id} cannot be submitted: current status is {order.status.value}")
            return False
            
        # Update status to validating
        order.update_status(OrderStatus.VALIDATING)
        
        # Perform risk checks
        risk_check_passed = self._perform_risk_checks(order)
        
        if not risk_check_passed:
            order.update_status(OrderStatus.REJECTED)
            order.add_error("Failed risk checks")
            
            # Move to completed orders
            with self._orders_lock:
                self.completed_orders[order_id] = order
                if order_id in self.active_orders:
                    del self.active_orders[order_id]
                self.total_orders_rejected += 1
                
            # Update in database
            self._store_order(order)
            
            # Notify callbacks
            self._notify_status_callbacks(order_id, order.status)
            
            return False
            
        # Risk checks passed, update status to pending
        order.update_status(OrderStatus.PENDING)
        
        # If no exchange connector, we can't proceed
        if not self.exchange_connector:
            order.update_status(OrderStatus.FAILED)
            order.add_error("No exchange connector available")
            
            # Move to completed orders
            with self._orders_lock:
                self.completed_orders[order_id] = order
                if order_id in self.active_orders:
                    del self.active_orders[order_id]
                self.total_orders_failed += 1
                
            # Update in database
            self._store_order(order)
            
            # Notify callbacks
            self._notify_status_callbacks(order_id, order.status)
            
            return False
            
        # Now submit the order to the exchange
        try:
            # Update status to submitting
            order.update_status(OrderStatus.SUBMITTING)
            
            # Convert our order to exchange format
            exchange_order_params = self._convert_to_exchange_format(order)
            
            # Submit to exchange
            exchange_response = self.exchange_connector.create_order(**exchange_order_params)
            
            # Update order with response
            order.update_from_exchange(exchange_response)
            
            # Update exchange ID mapping
            if order.exchange_order_id:
                with self._orders_lock:
                    self.exchange_id_map[order.exchange_order_id] = order.client_order_id
            
            # Update status to submitted
            if order.status == OrderStatus.SUBMITTING:
                order.update_status(OrderStatus.SUBMITTED)
                
            # If order is already filled or completed, move to completed orders
            if order.is_complete():
                with self._orders_lock:
                    self.completed_orders[order_id] = order
                    if order_id in self.active_orders:
                        del self.active_orders[order_id]
                        
                    if order.status == OrderStatus.FILLED:
                        self.total_orders_filled += 1
                        
            # Update in database
            self._store_order(order)
            
            # Notify callbacks
            self._notify_status_callbacks(order_id, order.status)
            
            # If this order was filled, check for child orders to submit
            if order.status == OrderStatus.FILLED and order.child_order_ids:
                self._handle_child_orders(order)
                
            logging.info(f"Order {order_id} submitted successfully: {order.status.value}")
            return True
            
        except Exception as e:
            order.update_status(OrderStatus.FAILED)
            order.add_error(f"Submission failed: {e}")
            
            # Move to completed orders
            with self._orders_lock:
                self.completed_orders[order_id] = order
                if order_id in self.active_orders:
                    del self.active_orders[order_id]
                self.total_orders_failed += 1
                
            # Update in database
            self._store_order(order)
            
            # Notify callbacks
            self._notify_status_callbacks(order_id, order.status)
            
            # Log error
            logging.error(f"Error submitting order {order_id}: {e}")
            
            if HAVE_ERROR_HANDLING:
                ErrorHandler.handle_error(
                    error=e,
                    category=ErrorCategory.TRADE_EXECUTION,
                    severity=ErrorSeverity.ERROR,
                    context={"operation": "submit_order", "order_id": order_id}
                )
                
            return False
    
    def _submit_order_async(self, order_id: str) -> bool:
        """
        Submit an order asynchronously using thread manager if available.
        
        Args:
            order_id: Client order ID to submit
            
        Returns:
            bool: True if submission was queued successfully
        """
        # If thread manager available, submit asynchronously
        if self.thread_manager:
            task_id = f"submit_order_{order_id}"
            
            # Prepare task
            def submit_task():
                return self.submit_order(order_id)
                
            # Submit task
            self.thread_manager.submit_task(
                task_id,
                submit_task,
                priority=1  # High priority for order submission
            )
            
            return True
            
        # Otherwise, submit synchronously
        return self.submit_order(order_id)
    
    def _perform_risk_checks(self, order: Order) -> bool:
        """
        Perform risk checks before submitting an order.
        
        Args:
            order: Order to check
            
        Returns:
            bool: True if risk checks passed
        """
        # Skip risk checks if no risk manager
        if not self.risk_manager:
            return True
            
        try:
            # Get current market data and positions
            market_data = None
            current_position = None
            
            if self.position_manager:
                current_position = self.position_manager.get_position(order.symbol)
                
            # Create risk context
            risk_context = {
                'order': order.to_dict(),
                'current_position': current_position,
                'market_data': market_data
            }
            
            # Check risk limits
            risk_check = self.risk_manager.check_risk_limits(risk_context)
            
            if not risk_check.get('risk_ok', False):
                reason = risk_check.get('reason', 'Unknown risk limit violation')
                logging.warning(f"Order {order.client_order_id} failed risk check: {reason}")
                order.add_error(f"Risk check failed: {reason}")
                return False
                
            return True
            
        except Exception as e:
            logging.error(f"Error during risk check: {e}")
            order.add_error(f"Risk check error: {e}")
            
            if HAVE_ERROR_HANDLING:
                ErrorHandler.handle_error(
                    error=e,
                    category=ErrorCategory.RISK,
                    severity=ErrorSeverity.WARNING,
                    context={"operation": "risk_check", "order_id": order.client_order_id}
                )
                
            # By default, fail closed (reject order if risk check fails)
            return False
    
    def _convert_to_exchange_format(self, order: Order) -> Dict[str, Any]:
        """
        Convert internal order to exchange-specific format.
        
        Args:
            order: Order to convert
            
        Returns:
            dict: Parameters for exchange API
        """
        # Basic parameters
        params = {
            'symbol': order.symbol,
            'type': order.order_type.value,
            'side': order.side.value,
            'amount': order.quantity,
            'clientOrderId': order.client_order_id
        }
        
        # Add price for limit orders
        if order.price is not None:
            params['price'] = order.price
            
        # Add stop price for stop orders
        if order.stop_price is not None:
            params['stopPrice'] = order.stop_price
            
        # Add timeInForce
        if order.time_in_force:
            params['timeInForce'] = order.time_in_force
            
        # Add reduceOnly
        if order.reduce_only:
            params['reduceOnly'] = True
            
        # Add postOnly
        if order.post_only:
            params['postOnly'] = True
            
        # Add iceberg quantity if specified
        if order.iceberg_quantity is not None:
            params['icebergQty'] = order.iceberg_quantity
            
        # Add additional exchange-specific parameters
        if order.exchange == 'binance':
            # Binance-specific order parameters
            if order.order_type == OrderType.TRAILING_STOP:
                params['trailingPercent'] = order.meta.get('trailing_percent')
        elif order.exchange == 'bybit':
            # Bybit-specific order parameters
            pass
        
        return params
    
    def _handle_child_orders(self, parent_order: Order) -> None:
        """
        Handle child orders when parent order is filled.
        
        Args:
            parent_order: Parent order that was filled
        """
        # If no child orders, nothing to do
        if not parent_order.child_order_ids:
            return
            
        logging.info(f"Handling {len(parent_order.child_order_ids)} child orders for {parent_order.client_order_id}")
        
        # Process each child order
        for child_id in parent_order.child_order_ids:
            child_order = self.get_order(child_id)
            
            if not child_order:
                logging.warning(f"Child order {child_id} not found")
                continue
                
            # Skip if not in pending state
            if child_order.status != OrderStatus.PENDING:
                continue
                
            # Submit the child order
            self._submit_order_async(child_id)
    
    def cancel_order(self, order_id: str) -> bool:
        """
        Cancel an order.
        
        Args:
            order_id: Client order ID to cancel
            
        Returns:
            bool: True if cancellation was successful
        """
        # Get the order
        order = self.get_order(order_id)
        if not order:
            logging.error(f"Order not found: {order_id}")
            return False
            
        # Check if order can be cancelled
        if not order.is_active():
            logging.warning(f"Order {order_id} cannot be cancelled: current status is {order.status.value}")
            return False
            
        # If no exchange order ID yet, just mark as cancelled locally
        if not order.exchange_order_id:
            order.update_status(OrderStatus.CANCELLED)
            
            # Move to completed orders
            with self._orders_lock:
                self.completed_orders[order_id] = order
                if order_id in self.active_orders:
                    del self.active_orders[order_id]
                self.total_orders_cancelled += 1
                
            # Update in database
            self._store_order(order)
            
            # Notify callbacks
            self._notify_status_callbacks(order_id, order.status)
            
            return True
            
        # If no exchange connector, we can't proceed
        if not self.exchange_connector:
            logging.error(f"Cannot cancel order {order_id}: no exchange connector available")
            return False
            
        # Now cancel the order on the exchange
        try:
            # Update status to cancelling
            order.update_status(OrderStatus.CANCELLING)
            
            # Cancel on exchange
            cancel_response = self.exchange_connector.cancel_order(
                order.exchange_order_id, 
                symbol=order.symbol
            )
            
            # Update order with response
            order.update_from_exchange(cancel_response)
            
            # Update status to cancelled if not already
            if order.status == OrderStatus.CANCELLING:
                order.update_status(OrderStatus.CANCELLED)
                
            # Move to completed orders
            with self._orders_lock:
                self.completed_orders[order_id] = order
                if order_id in self.active_orders:
                    del self.active_orders[order_id]
                self.total_orders_cancelled += 1
                
            # Update in database
            self._store_order(order)
            
            # Notify callbacks
            self._notify_status_callbacks(order_id, order.status)
            
            logging.info(f"Order {order_id} cancelled successfully")
            return True
            
        except Exception as e:
            order.add_error(f"Cancellation failed: {e}")
            
            # If the order doesn't exist on the exchange, mark as cancelled anyway
            if "order not found" in str(e).lower() or "does not exist" in str(e).lower():
                order.update_status(OrderStatus.CANCELLED)
                
                # Move to completed orders
                with self._orders_lock:
                    self.completed_orders[order_id] = order
                    if order_id in self.active_orders:
                        del self.active_orders[order_id]
                    self.total_orders_cancelled += 1
                    
                # Update in database
                self._store_order(order)
                
                # Notify callbacks
                self._notify_status_callbacks(order_id, order.status)
                
                return True
                
            # Log error
            logging.error(f"Error cancelling order {order_id}: {e}")
            
            if HAVE_ERROR_HANDLING:
                ErrorHandler.handle_error(
                    error=e,
                    category=ErrorCategory.TRADE_EXECUTION,
                    severity=ErrorSeverity.ERROR,
                    context={"operation": "cancel_order", "order_id": order_id}
                )
                
            return False
    
    def cancel_order_async(self, order_id: str) -> bool:
        """
        Cancel an order asynchronously using thread manager if available.
        
        Args:
            order_id: Client order ID to cancel
            
        Returns:
            bool: True if cancellation was queued successfully
        """
        # If thread manager available, cancel asynchronously
        if self.thread_manager:
            task_id = f"cancel_order_{order_id}"
            
            # Prepare task
            def cancel_task():
                return self.cancel_order(order_id)
                
            # Submit task
            self.thread_manager.submit_task(
                task_id,
                cancel_task,
                priority=1  # High priority for order cancellation
            )
            
            return True
            
        # Otherwise, cancel synchronously
        return self.cancel_order(order_id)
    
    def cancel_all_orders(self, symbol: Optional[str] = None) -> Tuple[int, int]:
        """
        Cancel all active orders, optionally filtered by symbol.
        
        Args:
            symbol: Filter by symbol (None for all symbols)
            
        Returns:
            tuple: (number of cancellation requests, number of successful cancellations)
        """
        # Get active orders
        active_orders = self.get_active_orders(symbol)
        
        if not active_orders:
            return 0, 0
            
        # If exchange connector supports batch cancel, use it
        if symbol and self.exchange_connector and hasattr(self.exchange_connector, 'cancel_all_orders'):
            try:
                # Cancel all orders for symbol
                self.exchange_connector.cancel_all_orders(symbol)
                
                # Update all orders
                count = 0
                for order in active_orders:
                    order.update_status(OrderStatus.CANCELLED)
                    
                    # Move to completed orders
                    with self._orders_lock:
                        self.completed_orders[order.client_order_id] = order
                        if order.client_order_id in self.active_orders:
                            del self.active_orders[order.client_order_id]
                        self.total_orders_cancelled += 1
                        
                    # Update in database
                    self._store_order(order)
                    
                    # Notify callbacks
                    self._notify_status_callbacks(order.client_order_id, order.status)
                    
                    count += 1
                    
                logging.info(f"Cancelled {count} orders for symbol {symbol}")
                return len(active_orders), count
                
            except Exception as e:
                logging.error(f"Error cancelling all orders for {symbol}: {e}")
                
                if HAVE_ERROR_HANDLING:
                    ErrorHandler.handle_error(
                        error=e,
                        category=ErrorCategory.TRADE_EXECUTION,
                        severity=ErrorSeverity.ERROR,
                        context={"operation": "cancel_all_orders", "symbol": symbol}
                    )
        
        # Otherwise, cancel orders individually
        success_count = 0
        for order in active_orders:
            if self.cancel_order_async(order.client_order_id):
                success_count += 1
                
        return len(active_orders), success_count
    
    def update_order(self, order_id: str) -> bool:
        """
        Update order status from exchange.
        
        Args:
            order_id: Client order ID to update
            
        Returns:
            bool: True if update was successful
        """
        # Get the order
        order = self.get_order(order_id)
        if not order:
            logging.error(f"Order not found: {order_id}")
            return False
            
        # If no exchange order ID or exchange connector, can't update
        if not order.exchange_order_id or not self.exchange_connector:
            return False
            
        try:
            # Fetch order from exchange
            exchange_order = self.exchange_connector.fetch_order(order.exchange_order_id, symbol=order.symbol)
            
            # Update order with response
            order.update_from_exchange(exchange_order)
            
            # If order is complete, move to completed orders
            if order.is_complete():
                with self._orders_lock:
                    self.completed_orders[order_id] = order
                    if order_id in self.active_orders:
                        del self.active_orders[order_id]
                        
                    if order.status == OrderStatus.FILLED:
                        self.total_orders_filled += 1
                    elif order.status == OrderStatus.CANCELLED:
                        self.total_orders_cancelled += 1
                        
            # Update in database
            self._store_order(order)
            
            # Notify callbacks
            self._notify_status_callbacks(order_id, order.status)
            
            # If this order was filled, check for child orders to submit
            if order.status == OrderStatus.FILLED and order.child_order_ids:
                self._handle_child_orders(order)
                
            return True
            
        except Exception as e:
            logging.error(f"Error updating order {order_id}: {e}")
            
            if HAVE_ERROR_HANDLING:
                ErrorHandler.handle_error(
                    error=e,
                    category=ErrorCategory.TRADE_EXECUTION,
                    severity=ErrorSeverity.WARNING,
                    context={"operation": "update_order", "order_id": order_id}
                )
                
            return False
    
    def update_all_active_orders(self) -> Tuple[int, int]:
        """
        Update all active orders from exchange.
        
        Returns:
            tuple: (number of update requests, number of successful updates)
        """
        active_orders = []
        
        # Get active orders with exchange IDs
        with self._orders_lock:
            active_orders = [order for order in self.active_orders.values() 
                            if order.exchange_order_id is not None]
                            
        if not active_orders:
            return 0, 0
            
        # Update each order
        success_count = 0
        for order in active_orders:
            if self.update_order(order.client_order_id):
                success_count += 1
                
        return len(active_orders), success_count
    
    def _notify_status_callbacks(self, order_id: str, status: OrderStatus) -> None:
        """
        Notify status callbacks of order status change.
        
        Args:
            order_id: Client order ID
            status: New order status
        """
        for _, callback in self._status_callbacks:
            try:
                callback(order_id, status)
            except Exception as e:
                logging.error(f"Error in status callback: {e}")
    
    def _notify_fill_callbacks(self, order_id: str, filled_quantity: float, fill_price: float) -> None:
        """
        Notify fill callbacks of order fill.
        
        Args:
            order_id: Client order ID
            filled_quantity: Amount filled
            fill_price: Fill price
        """
        for _, callback in self._fill_callbacks:
            try:
                callback(order_id, filled_quantity, fill_price)
            except Exception as e:
                logging.error(f"Error in fill callback: {e}")
    
    def cleanup_completed_orders(self, max_age: Optional[float] = None) -> int:
        """
        Clean up old completed orders from memory.
        
        Args:
            max_age: Maximum age in seconds (None to use max_order_history)
            
        Returns:
            int: Number of orders cleaned up
        """
        now = datetime.now().timestamp()
        
        # If max_age not specified, use max_order_history
        if max_age is None and self.max_order_history:
            # Keep the most recent max_order_history orders
            with self._orders_lock:
                if len(self.completed_orders) > self.max_order_history:
                    # Sort by completion time
                    sorted_orders = sorted(
                        self.completed_orders.items(),
                        key=lambda x: x[1].completed_time or x[1].updated_time or 0,
                        reverse=True
                    )
                    
                    # Keep only the most recent ones
                    to_keep = sorted_orders[:self.max_order_history]
                    to_remove = sorted_orders[self.max_order_history:]
                    
                    # Remove old orders
                    for order_id, _ in to_remove:
                        del self.completed_orders[order_id]
                        
                    return len(to_remove)
                    
            return 0
            
        # Remove orders older than max_age
        if max_age:
            to_remove = []
            
            with self._orders_lock:
                for order_id, order in self.completed_orders.items():
                    # Use completion time, updated time, or created time (in that order)
                    order_time = order.completed_time or order.updated_time or order.created_time
                    
                    if now - order_time > max_age:
                        to_remove.append(order_id)
                        
                # Remove old orders
                for order_id in to_remove:
                    del self.completed_orders[order_id]
                    
                return len(to_remove)
                
        return 0
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get order manager statistics.
        
        Returns:
            dict: Order manager statistics
        """
        with self._orders_lock:
            return {
                'active_orders': len(self.active_orders),
                'completed_orders': len(self.completed_orders),
                'total_orders_created': self.total_orders_created,
                'total_orders_filled': self.total_orders_filled,
                'total_orders_cancelled': self.total_orders_cancelled,
                'total_orders_rejected': self.total_orders_rejected,
                'total_orders_failed': self.total_orders_failed
            }
    
    def shutdown(self) -> None:
        """Clean up resources and shutdown the order manager."""
        logging.info("Shutting down order manager...")
        
        # Update all active orders one last time
        self.update_all_active_orders()
        
        # Store all orders in database
        if self.db_manager:
            with self._orders_lock:
                for order in list(self.active_orders.values()) + list(self.completed_orders.values()):
                    self._store_order(order)
                    
        logging.info("Order manager shutdown complete")


# Factory function to create order manager with appropriate dependencies
def create_order_manager(
    trading_system=None,
    exchange_name: Optional[str] = None,
    use_mock: bool = False,
    config: Optional[Dict[str, Any]] = None
) -> OrderManager:
    """
    Create an order manager with appropriate dependencies.
    
    Args:
        trading_system: Optional trading system to extract dependencies from
        exchange_name: Name of exchange to connect to
        use_mock: Whether to use mock implementations
        config: Additional configuration
        
    Returns:
        OrderManager: Configured order manager
    """
    config = config or {}
    
    # Extract dependencies from trading system if provided
    if trading_system:
        exchange_connector = getattr(trading_system, 'exchange_connector', None)
        risk_manager = getattr(trading_system, 'risk_manager', None)
        db_manager = getattr(trading_system, 'db_manager', None)
        thread_manager = getattr(trading_system, 'thread_manager', None)
        position_manager = getattr(trading_system, 'position_manager', None)
    else:
        exchange_connector = None
        risk_manager = None
        db_manager = None
        thread_manager = None
        position_manager = None
    
    # Create exchange connector if not provided
    if not exchange_connector and exchange_name:
        try:
            from execution.exchange_connector import create_exchange_connector
            exchange_connector = create_exchange_connector(exchange_name, use_mock=use_mock)
            logging.info(f"Created exchange connector for {exchange_name}")
        except (ImportError, Exception) as e:
            logging.warning(f"Failed to create exchange connector: {e}")
    
    # Create order manager
    order_manager = OrderManager(
        exchange_connector=exchange_connector,
        risk_manager=risk_manager,
        db_manager=db_manager,
        thread_manager=thread_manager,
        position_manager=position_manager,
        config=config
    )
    
    return order_manager
