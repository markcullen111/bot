# trade.py


import uuid
import time
import logging
import threading
from datetime import datetime
from typing import Dict, Optional, Union, List, Any, Tuple
from enum import Enum
import numpy as np

# Import error handling if available
try:
    from core.error_handling import safe_execute, ErrorCategory, ErrorSeverity
    HAVE_ERROR_HANDLING = True
except ImportError:
    HAVE_ERROR_HANDLING = False
    logging.warning("Error handling module not available in trade.py. Using basic error handling.")

class TradeStatus(Enum):
    """Enumeration of possible trade statuses"""
    PENDING = "pending"      # Order submitted but not yet filled
    OPEN = "open"            # Trade is active/position is open
    CLOSED = "closed"        # Trade completed normally
    CANCELLED = "cancelled"  # Order was cancelled before execution
    REJECTED = "rejected"    # Order was rejected by the exchange
    PARTIALLY_FILLED = "partially_filled"  # Order partially executed
    ERROR = "error"          # Error occurred during trade lifecycle


class TradeDirection(Enum):
    """Enumeration of trade directions"""
    LONG = "buy"
    SHORT = "sell"
    NEUTRAL = "neutral"


class TradeExitReason(Enum):
    """Enumeration of reasons for exiting a trade"""
    TAKE_PROFIT = "take_profit"
    STOP_LOSS = "stop_loss"
    TRAILING_STOP = "trailing_stop"
    MANUAL = "manual"
    STRATEGY_EXIT = "strategy_exit"
    TIME_EXIT = "time_exit"
    RISK_MANAGEMENT = "risk_management"
    VOLATILITY_EXIT = "volatility_exit"
    TECHNICAL_EXIT = "technical_exit"
    SYSTEM_EXIT = "system_exit"
    UNKNOWN = "unknown"


class Trade:
    """
    Comprehensive trade object that represents a trading operation.
    Thread-safe with optimized performance for high-frequency trading.
    """

    def __init__(
        self,
        symbol: str,
        direction: Union[str, TradeDirection],
        entry_price: float,
        quantity: float,
        timestamp: Optional[datetime] = None,
        trade_id: Optional[str] = None,
        strategy: Optional[str] = None,
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None,
        trailing_stop: Optional[float] = None,
        trailing_step: Optional[float] = None,
        time_limit: Optional[int] = None,  # Time limit in seconds
        risk_percentage: Optional[float] = None,
        exchange: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        order_id: Optional[str] = None,
        tags: Optional[List[str]] = None
    ):
        """
        Initialize a new trade with comprehensive tracking of position, risk, and execution details.

        Args:
            symbol: Trading pair symbol (e.g., 'BTC/USDT')
            direction: Trade direction ('buy' for long, 'sell' for short)
            entry_price: Trade entry price
            quantity: Trade size/quantity
            timestamp: Trade creation timestamp (defaults to current time)
            trade_id: Unique trade identifier (generated if not provided)
            strategy: Name of the strategy that generated this trade
            stop_loss: Stop loss price level
            take_profit: Take profit price level
            trailing_stop: Trailing stop distance in price or percentage
            trailing_step: Step size for trailing stop adjustments
            time_limit: Maximum trade duration in seconds
            risk_percentage: Percentage of account risked on this trade
            exchange: Exchange where trade is executed
            metadata: Additional custom data for the trade
            order_id: Exchange order ID if available
            tags: List of tags for classifying the trade
        """
        # Core trade properties
        self.symbol = symbol
        self.direction = direction if isinstance(direction, TradeDirection) else TradeDirection(direction)
        self.entry_price = float(entry_price)
        self.quantity = float(quantity)
        self.timestamp = timestamp or datetime.now()
        self.trade_id = trade_id or str(uuid.uuid4())
        self.strategy = strategy
        self.exchange = exchange or "default"
        
        # Risk management
        self.stop_loss = stop_loss
        self.take_profit = take_profit
        self.initial_stop_loss = stop_loss  # Keep original for calculations
        self.trailing_stop = trailing_stop
        self.trailing_step = trailing_step
        self.time_limit = time_limit
        self.risk_percentage = risk_percentage
        
        # Position tracking
        self.status = TradeStatus.PENDING
        self.order_id = order_id
        self.sl_order_id = None  # Stop loss order ID
        self.tp_order_id = None  # Take profit order ID
        self.exit_price = None
        self.exit_timestamp = None
        self.exit_reason = None
        self.partial_fills = []   # List of partial fills for tracking partial executions
        self.filled_quantity = 0.0
        
        # Performance tracking
        self.unrealized_pnl = 0.0
        self.realized_pnl = 0.0
        self.fees = 0.0
        self.slippage = 0.0
        self.execution_time = 0.0  # Time from order submission to fill
        self.drawdown = 0.0        # Maximum adverse excursion
        self.favorable_excursion = 0.0  # Maximum favorable excursion
        
        # Additional data
        self.metadata = metadata or {}
        self.tags = tags or []
        self.notes = ""
        
        # Performance optimization
        self._price_history = []  # Track price changes for analytics
        self._max_price_history = 1000  # Limit history size for memory efficiency
        self._highest_price = entry_price
        self._lowest_price = entry_price
        
        # Thread safety
        self._lock = threading.RLock()  # Reentrant lock for thread-safe operations
        
        # Lifecycle tracking
        self.created_at = datetime.now()
        self.updated_at = self.created_at
        self.execution_path = []  # Track key events in trade lifecycle
        self._add_execution_event("created")
        
        # Validate the trade on creation
        self._validate()

    def _validate(self) -> bool:
        """
        Validate trade parameters for consistency and correctness.
        
        Returns:
            bool: True if validation passes, False otherwise
        """
        try:
            if not self.symbol or not isinstance(self.symbol, str):
                raise ValueError(f"Invalid symbol: {self.symbol}")
                
            if self.quantity <= 0:
                raise ValueError(f"Quantity must be positive: {self.quantity}")
                
            if self.entry_price <= 0:
                raise ValueError(f"Entry price must be positive: {self.entry_price}")
                
            # Validate stop loss (must be below entry for longs, above for shorts)
            if self.stop_loss is not None:
                if self.direction == TradeDirection.LONG and self.stop_loss >= self.entry_price:
                    raise ValueError(f"Stop loss ({self.stop_loss}) must be below entry price ({self.entry_price}) for long positions")
                elif self.direction == TradeDirection.SHORT and self.stop_loss <= self.entry_price:
                    raise ValueError(f"Stop loss ({self.stop_loss}) must be above entry price ({self.entry_price}) for short positions")
                    
            # Validate take profit (must be above entry for longs, below for shorts)
            if self.take_profit is not None:
                if self.direction == TradeDirection.LONG and self.take_profit <= self.entry_price:
                    raise ValueError(f"Take profit ({self.take_profit}) must be above entry price ({self.entry_price}) for long positions")
                elif self.direction == TradeDirection.SHORT and self.take_profit >= self.entry_price:
                    raise ValueError(f"Take profit ({self.take_profit}) must be below entry price ({self.entry_price}) for short positions")
                    
            return True
            
        except Exception as e:
            logging.error(f"Trade validation error: {e}")
            self.status = TradeStatus.ERROR
            self._add_execution_event("validation_failed", {"error": str(e)})
            return False

    def _add_execution_event(self, event_type: str, data: Optional[Dict[str, Any]] = None) -> None:
        """Thread-safe addition of execution event to trade history"""
        with self._lock:
            self.execution_path.append({
                "timestamp": datetime.now(),
                "event": event_type,
                "data": data or {}
            })
            self.updated_at = datetime.now()

    def update_current_price(self, price: float, timestamp: Optional[datetime] = None) -> Dict[str, Any]:
        """
        Update the trade with current market price to recalculate metrics and check conditions.
        Thread-safe and optimized for high frequency updates.
        
        Args:
            price: Current market price
            timestamp: Timestamp of the price update (defaults to current time)
            
        Returns:
            Dict with trade status updates including triggered conditions
        """
        with self._lock:
            current_timestamp = timestamp or datetime.now()
            
            # Skip if trade is not active
            if self.status != TradeStatus.OPEN and self.status != TradeStatus.PARTIALLY_FILLED:
                return {"status": self.status.value, "price_update": "skipped"}
                
            # Add to price history with sampling to limit memory usage
            if not self._price_history or (current_timestamp - self._price_history[-1][1]).total_seconds() > 1:
                self._price_history.append((price, current_timestamp))
                # Limit history size
                if len(self._price_history) > self._max_price_history:
                    # Keep first entry, last entries, and sample middle for efficiency
                    self._price_history = [
                        self._price_history[0],
                        *self._price_history[-(self._max_price_history-1):]
                    ]
            
            # Track highest/lowest price seen during trade
            self._highest_price = max(self._highest_price, price)
            self._lowest_price = min(self._lowest_price, price)
            
            # Calculate current P&L
            if self.direction == TradeDirection.LONG:
                self.unrealized_pnl = (price - self.entry_price) * self.filled_quantity
                self.drawdown = min(self.drawdown, (self._lowest_price - self.entry_price) * self.filled_quantity)
                self.favorable_excursion = max(self.favorable_excursion, (self._highest_price - self.entry_price) * self.filled_quantity)
            else:  # SHORT
                self.unrealized_pnl = (self.entry_price - price) * self.filled_quantity
                self.drawdown = min(self.drawdown, (self.entry_price - self._highest_price) * self.filled_quantity)
                self.favorable_excursion = max(self.favorable_excursion, (self.entry_price - self._lowest_price) * self.filled_quantity)
            
            # Check for exit conditions
            exit_triggers = self._check_exit_conditions(price, current_timestamp)
            
            # Update trailing stop if needed
            self._update_trailing_stop(price)
            
            return {
                "status": self.status.value,
                "price": price,
                "unrealized_pnl": self.unrealized_pnl,
                "exit_triggers": exit_triggers
            }

    def _check_exit_conditions(self, price: float, timestamp: datetime) -> Dict[str, bool]:
        """
        Check if any exit conditions are triggered by the current price.
        Optimized with early return pattern for performance.
        
        Args:
            price: Current market price
            timestamp: Current timestamp
            
        Returns:
            Dict of triggered exit conditions
        """
        triggers = {
            "take_profit": False,
            "stop_loss": False,
            "trailing_stop": False,
            "time_limit": False
        }
        
        # Skip if not active
        if self.status != TradeStatus.OPEN and self.status != TradeStatus.PARTIALLY_FILLED:
            return triggers
            
        # Check stop loss
        if self.stop_loss is not None:
            if (self.direction == TradeDirection.LONG and price <= self.stop_loss) or \
               (self.direction == TradeDirection.SHORT and price >= self.stop_loss):
                triggers["stop_loss"] = True
                return triggers  # Early return for performance in high-frequency updates
        
        # Check take profit
        if self.take_profit is not None:
            if (self.direction == TradeDirection.LONG and price >= self.take_profit) or \
               (self.direction == TradeDirection.SHORT and price <= self.take_profit):
                triggers["take_profit"] = True
                return triggers  # Early return for performance
        
        # Check time limit
        if self.time_limit is not None:
            trade_duration = (timestamp - self.timestamp).total_seconds()
            if trade_duration >= self.time_limit:
                triggers["time_limit"] = True
                
        return triggers

    def _update_trailing_stop(self, price: float) -> None:
        """
        Update trailing stop based on current price movement.
        
        Args:
            price: Current market price
        """
        if self.trailing_stop is None or self.trailing_step is None:
            return
            
        # For long positions, move stop loss up as price increases
        if self.direction == TradeDirection.LONG:
            # Calculate potential new stop loss
            potential_stop = price - self.trailing_stop
            
            # Update only if price has moved up enough and new stop is higher
            if (self.stop_loss is None or potential_stop > self.stop_loss) and \
               (price >= self.entry_price + self.trailing_step):
                self.stop_loss = max(self.stop_loss or 0, potential_stop)
                self._add_execution_event("trailing_stop_updated", {
                    "new_stop": self.stop_loss, 
                    "price": price
                })
                
        # For short positions, move stop loss down as price decreases
        else:
            # Calculate potential new stop loss
            potential_stop = price + self.trailing_stop
            
            # Update only if price has moved down enough and new stop is lower
            if (self.stop_loss is None or potential_stop < self.stop_loss) and \
               (price <= self.entry_price - self.trailing_step):
                self.stop_loss = min(self.stop_loss or float('inf'), potential_stop)
                self._add_execution_event("trailing_stop_updated", {
                    "new_stop": self.stop_loss,
                    "price": price
                })

    def execute(self, execution_price: Optional[float] = None, quantity: Optional[float] = None) -> bool:
        """
        Execute the trade with the specified price and quantity.
        
        Args:
            execution_price: Actual execution price (defaults to entry price)
            quantity: Execution quantity (defaults to full quantity)
            
        Returns:
            bool: True if execution succeeded, False otherwise
        """
        with self._lock:
            # Skip if not pending
            if self.status != TradeStatus.PENDING and self.status != TradeStatus.PARTIALLY_FILLED:
                logging.warning(f"Cannot execute trade {self.trade_id} with status {self.status}")
                return False
                
            execution_time = datetime.now()
            actual_price = execution_price or self.entry_price
            actual_quantity = quantity or (self.quantity - self.filled_quantity)
            
            # Calculate slippage
            self.slippage = actual_price - self.entry_price if self.direction == TradeDirection.LONG else self.entry_price - actual_price
            
            # Record partial fill
            self.partial_fills.append({
                "timestamp": execution_time,
                "price": actual_price,
                "quantity": actual_quantity,
                "slippage": self.slippage
            })
            
            # Update filled quantity
            self.filled_quantity += actual_quantity
            
            # Update status
            if self.filled_quantity >= self.quantity:
                self.status = TradeStatus.OPEN
            else:
                self.status = TradeStatus.PARTIALLY_FILLED
                
            # Record execution
            self._add_execution_event("executed", {
                "price": actual_price,
                "quantity": actual_quantity,
                "filled_quantity": self.filled_quantity,
                "slippage": self.slippage
            })
            
            # Initialize price tracking
            self._highest_price = actual_price
            self._lowest_price = actual_price
            self._price_history.append((actual_price, execution_time))
            
            return True

    def close(
        self, 
        exit_price: float, 
        exit_reason: Union[str, TradeExitReason] = TradeExitReason.MANUAL,
        timestamp: Optional[datetime] = None,
        quantity: Optional[float] = None,
        fees: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Close the trade with a specified exit price and reason.
        
        Args:
            exit_price: Trade exit price
            exit_reason: Reason for exiting the trade
            timestamp: Exit timestamp (defaults to current time)
            quantity: Quantity to close (defaults to full quantity)
            fees: Transaction fees
            
        Returns:
            Dict with trade closure details including P&L
        """
        with self._lock:
            # Skip if already closed
            if self.status == TradeStatus.CLOSED:
                return {"status": "already_closed", "pnl": self.realized_pnl}
                
            current_timestamp = timestamp or datetime.now()
            actual_quantity = quantity or self.filled_quantity
            
            if actual_quantity > self.filled_quantity:
                logging.warning(f"Close quantity {actual_quantity} exceeds filled quantity {self.filled_quantity}")
                actual_quantity = self.filled_quantity
                
            # Calculate realized P&L
            if self.direction == TradeDirection.LONG:
                trade_pnl = (exit_price - self.entry_price) * actual_quantity
            else:  # SHORT
                trade_pnl = (self.entry_price - exit_price) * actual_quantity
                
            # Account for fees
            if fees is not None:
                self.fees += fees
                trade_pnl -= fees
                
            # Update trade data
            self.exit_price = exit_price
            self.exit_timestamp = current_timestamp
            self.exit_reason = exit_reason if isinstance(exit_reason, TradeExitReason) else TradeExitReason(exit_reason)
            self.realized_pnl += trade_pnl
            self.status = TradeStatus.CLOSED
            
            # Record closure
            self._add_execution_event("closed", {
                "exit_price": exit_price,
                "pnl": trade_pnl,
                "reason": self.exit_reason.value,
                "quantity": actual_quantity,
                "total_pnl": self.realized_pnl
            })
            
            # Calculate trade duration
            duration = (current_timestamp - self.timestamp).total_seconds()
            
            result = {
                "trade_id": self.trade_id,
                "symbol": self.symbol,
                "direction": self.direction.value,
                "entry_price": self.entry_price,
                "exit_price": exit_price,
                "quantity": actual_quantity,
                "pnl": trade_pnl,
                "total_pnl": self.realized_pnl,
                "fees": self.fees,
                "duration": duration,
                "exit_reason": self.exit_reason.value,
                "drawdown": self.drawdown,
                "favorable_excursion": self.favorable_excursion,
                "status": self.status.value
            }
            
            return result

    def cancel(self, reason: Optional[str] = None) -> bool:
        """
        Cancel the trade if it hasn't been executed yet.
        
        Args:
            reason: Reason for cancellation
            
        Returns:
            bool: True if cancellation succeeded, False otherwise
        """
        with self._lock:
            # Can only cancel pending trades
            if self.status != TradeStatus.PENDING:
                logging.warning(f"Cannot cancel trade {self.trade_id} with status {self.status}")
                return False
                
            self.status = TradeStatus.CANCELLED
            
            # Record cancellation
            self._add_execution_event("cancelled", {"reason": reason or "unspecified"})
            
            return True

    def reject(self, reason: str) -> bool:
        """
        Mark the trade as rejected, typically by the exchange.
        
        Args:
            reason: Rejection reason
            
        Returns:
            bool: True if rejection succeeded, False otherwise
        """
        with self._lock:
            # Can only reject pending trades
            if self.status != TradeStatus.PENDING:
                logging.warning(f"Cannot reject trade {self.trade_id} with status {self.status}")
                return False
                
            self.status = TradeStatus.REJECTED
            
            # Record rejection
            self._add_execution_event("rejected", {"reason": reason})
            
            return True

    def get_risk_reward_ratio(self) -> Optional[float]:
        """
        Calculate the risk/reward ratio for this trade.
        
        Returns:
            float: Risk/reward ratio or None if stops/targets not set
        """
        if self.stop_loss is None or self.take_profit is None:
            return None
            
        if self.direction == TradeDirection.LONG:
            risk = self.entry_price - self.stop_loss
            reward = self.take_profit - self.entry_price
        else:  # SHORT
            risk = self.stop_loss - self.entry_price
            reward = self.entry_price - self.take_profit
            
        # Avoid division by zero
        if risk == 0:
            return None
            
        return reward / risk

    def get_position_value(self, current_price: Optional[float] = None) -> float:
        """
        Calculate the current value of the trade position.
        
        Args:
            current_price: Current market price (uses last known price if None)
            
        Returns:
            float: Current position value
        """
        price = current_price
        if price is None and self._price_history:
            price = self._price_history[-1][0]
        if price is None:
            price = self.entry_price
            
        return price * self.filled_quantity

    def get_unrealized_pnl(self, current_price: Optional[float] = None) -> float:
        """
        Calculate the current unrealized P&L for this trade.
        
        Args:
            current_price: Current market price (uses last known price if None)
            
        Returns:
            float: Current unrealized P&L
        """
        if current_price is not None:
            if self.direction == TradeDirection.LONG:
                return (current_price - self.entry_price) * self.filled_quantity
            else:  # SHORT
                return (self.entry_price - current_price) * self.filled_quantity
                
        return self.unrealized_pnl

    def get_total_pnl(self, current_price: Optional[float] = None) -> float:
        """
        Calculate the total P&L (realized + unrealized).
        
        Args:
            current_price: Current market price for unrealized portion
            
        Returns:
            float: Total P&L
        """
        return self.realized_pnl + self.get_unrealized_pnl(current_price)

    def get_trade_duration(self) -> float:
        """
        Calculate the trade duration in seconds.
        
        Returns:
            float: Trade duration in seconds
        """
        end_time = self.exit_timestamp or datetime.now()
        return (end_time - self.timestamp).total_seconds()

    def get_performance_metrics(self, current_price: Optional[float] = None) -> Dict[str, Any]:
        """
        Get comprehensive performance metrics for this trade.
        
        Args:
            current_price: Current market price for unrealized metrics
            
        Returns:
            Dict with various performance metrics
        """
        price = current_price
        if price is None and self._price_history:
            price = self._price_history[-1][0]
        if price is None:
            price = self.entry_price
            
        trade_duration = self.get_trade_duration()
        unrealized_pnl = self.get_unrealized_pnl(price)
        total_pnl = self.realized_pnl + unrealized_pnl
        
        # Calculate returns
        position_value = self.entry_price * self.filled_quantity
        if position_value == 0:
            roi = 0
        else:
            roi = total_pnl / position_value
            
        # Calculate price volatility during trade
        price_changes = []
        for i in range(1, len(self._price_history)):
            price_changes.append(
                (self._price_history[i][0] - self._price_history[i-1][0]) / self._price_history[i-1][0]
            )
        
        volatility = np.std(price_changes) * 100 if price_changes else 0
        
        # Calculate maximum adverse excursion (MAE)
        if self.direction == TradeDirection.LONG:
            mae = (self._lowest_price - self.entry_price) / self.entry_price
        else:
            mae = (self._highest_price - self.entry_price) / self.entry_price
            
        # Calculate maximum favorable excursion (MFE)
        if self.direction == TradeDirection.LONG:
            mfe = (self._highest_price - self.entry_price) / self.entry_price
        else:
            mfe = (self.entry_price - self._lowest_price) / self.entry_price
            
        return {
            "trade_id": self.trade_id,
            "symbol": self.symbol,
            "direction": self.direction.value,
            "status": self.status.value,
            "entry_price": self.entry_price,
            "current_price": price,
            "exit_price": self.exit_price,
            "quantity": self.filled_quantity,
            "unrealized_pnl": unrealized_pnl,
            "realized_pnl": self.realized_pnl,
            "total_pnl": total_pnl,
            "roi": roi * 100,  # Percentage
            "duration": trade_duration,
            "duration_hours": trade_duration / 3600,
            "fees": self.fees,
            "slippage": self.slippage,
            "drawdown": self.drawdown,
            "favorable_excursion": self.favorable_excursion,
            "max_adverse_excursion": mae * 100,  # Percentage
            "max_favorable_excursion": mfe * 100,  # Percentage
            "volatility": volatility,
            "risk_reward_ratio": self.get_risk_reward_ratio(),
            "win_loss": "win" if total_pnl > 0 else "loss" if total_pnl < 0 else "breakeven"
        }

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert trade to dictionary for serialization.
        
        Returns:
            Dict representation of the trade
        """
        with self._lock:
            return {
                "trade_id": self.trade_id,
                "symbol": self.symbol,
                "direction": self.direction.value,
                "entry_price": self.entry_price,
                "quantity": self.quantity,
                "filled_quantity": self.filled_quantity,
                "timestamp": self.timestamp.isoformat(),
                "status": self.status.value,
                "strategy": self.strategy,
                "stop_loss": self.stop_loss,
                "take_profit": self.take_profit,
                "trailing_stop": self.trailing_stop,
                "trailing_step": self.trailing_step,
                "time_limit": self.time_limit,
                "exit_price": self.exit_price,
                "exit_timestamp": self.exit_timestamp.isoformat() if self.exit_timestamp else None,
                "exit_reason": self.exit_reason.value if self.exit_reason else None,
                "order_id": self.order_id,
                "sl_order_id": self.sl_order_id,
                "tp_order_id": self.tp_order_id,
                "unrealized_pnl": self.unrealized_pnl,
                "realized_pnl": self.realized_pnl,
                "fees": self.fees,
                "slippage": self.slippage,
                "drawdown": self.drawdown,
                "favorable_excursion": self.favorable_excursion,
                "risk_percentage": self.risk_percentage,
                "exchange": self.exchange,
                "metadata": self.metadata,
                "tags": self.tags,
                "notes": self.notes,
                "created_at": self.created_at.isoformat(),
                "updated_at": self.updated_at.isoformat(),
                "execution_path": [
                    {
                        "timestamp": event["timestamp"].isoformat(),
                        "event": event["event"],
                        "data": event["data"]
                    }
                    for event in self.execution_path
                ],
                "partial_fills": self.partial_fills
            }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Trade':
        """
        Create a trade instance from a dictionary.
        
        Args:
            data: Dictionary containing trade data
            
        Returns:
            Trade instance
        """
        # Convert timestamp strings to datetime objects
        timestamp = datetime.fromisoformat(data["timestamp"]) if "timestamp" in data else None
        exit_timestamp = datetime.fromisoformat(data["exit_timestamp"]) if data.get("exit_timestamp") else None
        
        # Create trade instance
        trade = cls(
            symbol=data["symbol"],
            direction=data["direction"],
            entry_price=data["entry_price"],
            quantity=data["quantity"],
            timestamp=timestamp,
            trade_id=data.get("trade_id"),
            strategy=data.get("strategy"),
            stop_loss=data.get("stop_loss"),
            take_profit=data.get("take_profit"),
            trailing_stop=data.get("trailing_stop"),
            trailing_step=data.get("trailing_step"),
            time_limit=data.get("time_limit"),
            risk_percentage=data.get("risk_percentage"),
            exchange=data.get("exchange"),
            metadata=data.get("metadata"),
            order_id=data.get("order_id"),
            tags=data.get("tags")
        )
        
        # Restore additional properties
        trade.status = TradeStatus(data.get("status", "pending"))
        trade.exit_price = data.get("exit_price")
        trade.exit_timestamp = exit_timestamp
        if data.get("exit_reason"):
            trade.exit_reason = TradeExitReason(data["exit_reason"])
        trade.sl_order_id = data.get("sl_order_id")
        trade.tp_order_id = data.get("tp_order_id")
        trade.filled_quantity = data.get("filled_quantity", 0)
        trade.unrealized_pnl = data.get("unrealized_pnl", 0)
        trade.realized_pnl = data.get("realized_pnl", 0)
        trade.fees = data.get("fees", 0)
        trade.slippage = data.get("slippage", 0)
        trade.drawdown = data.get("drawdown", 0)
        trade.favorable_excursion = data.get("favorable_excursion", 0)
        trade.notes = data.get("notes", "")
        
        # Restore created/updated timestamps
        if "created_at" in data:
            trade.created_at = datetime.fromisoformat(data["created_at"])
        if "updated_at" in data:
            trade.updated_at = datetime.fromisoformat(data["updated_at"])
            
        # Restore execution path
        if "execution_path" in data:
            trade.execution_path = [
                {
                    "timestamp": datetime.fromisoformat(event["timestamp"]),
                    "event": event["event"],
                    "data": event["data"]
                }
                for event in data["execution_path"]
            ]
            
        # Restore partial fills
        if "partial_fills" in data:
            trade.partial_fills = data["partial_fills"]
            
        return trade

    def get_database_record(self) -> Dict[str, Any]:
        """
        Get a simplified representation for database storage.
        
        Returns:
            Dict with trade data formatted for database storage
        """
        return {
            "trade_id": self.trade_id,
            "symbol": self.symbol,
            "action": self.direction.value,
            "price": self.entry_price,
            "amount": self.quantity,
            "time": self.timestamp,
            "stop_loss": self.stop_loss,
            "take_profit": self.take_profit,
            "status": self.status.value,
            "order_id": self.order_id,
            "sl_order_id": self.sl_order_id,
            "tp_order_id": self.tp_order_id,
            "exit_price": self.exit_price,
            "exit_time": self.exit_timestamp,
            "pnl": self.realized_pnl,
            "exit_reason": self.exit_reason.value if self.exit_reason else None,
            "strategy": self.strategy,
            "exchange": self.exchange,
            "metadata": json.dumps(self.metadata) if self.metadata else None
        }


class TradeManager:
    """
    Manager for tracking and managing multiple trades with optimal performance.
    Thread-safe with batch operations for efficiency.
    """
    
    def __init__(self):
        """Initialize the trade manager"""
        self.trades = {}  # Dict mapping trade_id to Trade objects
        self.active_trades = {}  # Dict of only active trades for faster access
        self._lock = threading.RLock()  # Thread safety for multi-threaded access
        
    def add_trade(self, trade: Trade) -> str:
        """
        Add a trade to the manager.
        
        Args:
            trade: Trade to add
            
        Returns:
            str: Trade ID
        """
        with self._lock:
            trade_id = trade.trade_id
            self.trades[trade_id] = trade
            
            # Add to active trades if appropriate
            if trade.status == TradeStatus.OPEN or trade.status == TradeStatus.PARTIALLY_FILLED:
                self.active_trades[trade_id] = trade
                
            return trade_id
            
    def remove_trade(self, trade_id: str) -> bool:
        """
        Remove a trade from the manager.
        
        Args:
            trade_id: ID of the trade to remove
            
        Returns:
            bool: True if trade was removed, False otherwise
        """
        with self._lock:
            if trade_id in self.trades:
                if trade_id in self.active_trades:
                    del self.active_trades[trade_id]
                del self.trades[trade_id]
                return True
            return False
            
    def get_trade(self, trade_id: str) -> Optional[Trade]:
        """
        Get a trade by ID.
        
        Args:
            trade_id: ID of the trade to get
            
        Returns:
            Trade instance or None if not found
        """
        return self.trades.get(trade_id)
        
    def get_active_trades(self, symbol: Optional[str] = None) -> Dict[str, Trade]:
        """
        Get active trades, optionally filtered by symbol.
        
        Args:
            symbol: Symbol to filter by
            
        Returns:
            Dict mapping trade_id to Trade objects
        """
        with self._lock:
            if symbol is None:
                return self.active_trades.copy()
                
            return {
                trade_id: trade for trade_id, trade in self.active_trades.items()
                if trade.symbol == symbol
            }
            
    def update_prices(self, price_updates: Dict[str, float]) -> Dict[str, Dict[str, Any]]:
        """
        Batch update prices for multiple symbols at once.
        Optimized for performance with multiple active trades.
        
        Args:
            price_updates: Dict mapping symbols to current prices
            
        Returns:
            Dict with updates for each affected trade
        """
        updates = {}
        timestamp = datetime.now()
        
        with self._lock:
            # Group active trades by symbol for efficient updates
            trades_by_symbol = {}
            for trade_id, trade in self.active_trades.items():
                if trade.symbol not in trades_by_symbol:
                    trades_by_symbol[trade.symbol] = []
                trades_by_symbol[trade.symbol].append(trade)
                
            # Process updates for each symbol
            for symbol, price in price_updates.items():
                if symbol in trades_by_symbol:
                    for trade in trades_by_symbol[symbol]:
                        # Update trade
                        trade_update = trade.update_current_price(price, timestamp)
                        updates[trade.trade_id] = trade_update
                        
                        # Check for exit triggers
                        if (trade.status == TradeStatus.OPEN or trade.status == TradeStatus.PARTIALLY_FILLED) and \
                           any(trade_update.get("exit_triggers", {}).values()):
                            # Some exit condition triggered
                            exit_triggers = trade_update["exit_triggers"]
                            
                            # Determine exit reason
                            if exit_triggers.get("stop_loss"):
                                exit_reason = TradeExitReason.STOP_LOSS
                            elif exit_triggers.get("take_profit"):
                                exit_reason = TradeExitReason.TAKE_PROFIT
                            elif exit_triggers.get("trailing_stop"):
                                exit_reason = TradeExitReason.TRAILING_STOP
                            elif exit_triggers.get("time_limit"):
                                exit_reason = TradeExitReason.TIME_EXIT
                            else:
                                exit_reason = TradeExitReason.UNKNOWN
                                
                            # Close the trade
                            close_result = trade.close(price, exit_reason, timestamp)
                            updates[trade.trade_id]["closed"] = close_result
                            
                            # Remove from active trades
                            if trade.trade_id in self.active_trades:
                                del self.active_trades[trade.trade_id]
                        
        return updates
        
    def get_portfolio_value(self, price_updates: Dict[str, float]) -> Dict[str, float]:
        """
        Calculate current portfolio value and metrics.
        
        Args:
            price_updates: Dict mapping symbols to current prices
            
        Returns:
            Dict with portfolio metrics
        """
        with self._lock:
            total_value = 0
            unrealized_pnl = 0
            realized_pnl = 0
            position_values = {}
            
            # Calculate total value and P&L
            for trade_id, trade in self.active_trades.items():
                if trade.symbol in price_updates:
                    price = price_updates[trade.symbol]
                    position_value = trade.get_position_value(price)
                    unrealized_p = trade.get_unrealized_pnl(price)
                    
                    total_value += position_value
                    unrealized_pnl += unrealized_p
                    
                    # Track by symbol
                    if trade.symbol not in position_values:
                        position_values[trade.symbol] = 0
                    position_values[trade.symbol] += position_value
            
            # Add realized P&L from all trades
            for trade_id, trade in self.trades.items():
                realized_pnl += trade.realized_pnl
                
            return {
                "total_value": total_value,
                "unrealized_pnl": unrealized_pnl,
                "realized_pnl": realized_pnl,
                "total_pnl": unrealized_pnl + realized_pnl,
                "position_values": position_values
            }
            
    def get_performance_metrics(self, price_updates: Optional[Dict[str, float]] = None) -> Dict[str, Any]:
        """
        Get comprehensive performance metrics across all trades.
        
        Args:
            price_updates: Dict mapping symbols to current prices
            
        Returns:
            Dict with performance metrics
        """
        with self._lock:
            metrics = {
                "trade_count": len(self.trades),
                "active_count": len(self.active_trades),
                "win_count": 0,
                "loss_count": 0,
                "total_pnl": 0,
                "realized_pnl": 0,
                "unrealized_pnl": 0,
                "win_rate": 0,
                "average_win": 0,
                "average_loss": 0,
                "profit_factor": 0,
                "average_duration": 0,
                "largest_win": 0,
                "largest_loss": 0
            }
            
            if not self.trades:
                return metrics
                
            # Calculate metrics
            win_trades = []
            loss_trades = []
            total_duration = 0
            
            for trade_id, trade in self.trades.items():
                trade_pnl = trade.realized_pnl
                
                # Add unrealized P&L for active trades
                if trade.status in [TradeStatus.OPEN, TradeStatus.PARTIALLY_FILLED]:
                    if price_updates and trade.symbol in price_updates:
                        trade_pnl += trade.get_unrealized_pnl(price_updates[trade.symbol])
                    else:
                        trade_pnl += trade.unrealized_pnl
                
                if trade_pnl > 0:
                    win_trades.append(trade_pnl)
                    metrics["largest_win"] = max(metrics["largest_win"], trade_pnl)
                elif trade_pnl < 0:
                    loss_trades.append(trade_pnl)
                    metrics["largest_loss"] = min(metrics["largest_loss"], trade_pnl)
                    
                metrics["total_pnl"] += trade_pnl
                metrics["realized_pnl"] += trade.realized_pnl
                
                if trade.status in [TradeStatus.OPEN, TradeStatus.PARTIALLY_FILLED]:
                    metrics["unrealized_pnl"] += trade.unrealized_pnl
                    
                # Track duration
                total_duration += trade.get_trade_duration()
                
            # Calculate aggregate metrics
            metrics["win_count"] = len(win_trades)
            metrics["loss_count"] = len(loss_trades)
            
            if metrics["win_count"] + metrics["loss_count"] > 0:
                metrics["win_rate"] = metrics["win_count"] / (metrics["win_count"] + metrics["loss_count"])
                
            if metrics["win_count"] > 0:
                metrics["average_win"] = sum(win_trades) / metrics["win_count"]
                
            if metrics["loss_count"] > 0:
                metrics["average_loss"] = sum(loss_trades) / metrics["loss_count"]
                
            total_losses = abs(sum(loss_trades)) if loss_trades else 0
            total_wins = sum(win_trades) if win_trades else 0
            
            if total_losses > 0:
                metrics["profit_factor"] = total_wins / total_losses
                
            if len(self.trades) > 0:
                metrics["average_duration"] = total_duration / len(self.trades)
                
            return metrics
            
    def clear_closed_trades(self, max_age: Optional[float] = None) -> int:
        """
        Clear closed trades from memory to prevent memory leaks.
        
        Args:
            max_age: Maximum age in seconds to keep closed trades
            
        Returns:
            int: Number of trades cleared
        """
        with self._lock:
            to_remove = []
            now = datetime.now()
            
            for trade_id, trade in self.trades.items():
                if trade.status in [TradeStatus.CLOSED, TradeStatus.CANCELLED, TradeStatus.REJECTED]:
                    # If max_age specified, only remove trades older than max_age
                    if max_age is not None:
                        # Check if trade is old enough
                        if trade.exit_timestamp and (now - trade.exit_timestamp).total_seconds() > max_age:
                            to_remove.append(trade_id)
                    else:
                        # No max_age, remove all closed trades
                        to_remove.append(trade_id)
                        
            # Remove trades
            for trade_id in to_remove:
                del self.trades[trade_id]
                if trade_id in self.active_trades:
                    del self.active_trades[trade_id]
                    
            return len(to_remove)
            
    def to_dict_list(self) -> List[Dict[str, Any]]:
        """
        Convert all trades to a list of dictionaries.
        
        Returns:
            List of trade dictionaries
        """
        with self._lock:
            return [trade.to_dict() for trade in self.trades.values()]
            
    @classmethod
    def from_dict_list(cls, trade_dicts: List[Dict[str, Any]]) -> 'TradeManager':
        """
        Create a trade manager from a list of trade dictionaries.
        
        Args:
            trade_dicts: List of trade dictionaries
            
        Returns:
            TradeManager instance with loaded trades
        """
        manager = cls()
        
        for trade_dict in trade_dicts:
            trade = Trade.from_dict(trade_dict)
            manager.add_trade(trade)
            
        return manager


# Decorate methods with error handling if available
if HAVE_ERROR_HANDLING:
    Trade.update_current_price = safe_execute(ErrorCategory.TRADE_EXECUTION, {})(Trade.update_current_price)
    Trade.execute = safe_execute(ErrorCategory.TRADE_EXECUTION, False)(Trade.execute)
    Trade.close = safe_execute(ErrorCategory.TRADE_EXECUTION, {})(Trade.close)
    TradeManager.update_prices = safe_execute(ErrorCategory.TRADE_EXECUTION, {})(TradeManager.update_prices)
