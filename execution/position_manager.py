# execution/position_manager.py

import threading
import time
import uuid
import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Union
from decimal import Decimal, ROUND_DOWN, ROUND_UP
from datetime import datetime, timedelta
import json

# Import error handling if available
try:
    from core.error_handling import safe_execute, ErrorCategory, ErrorSeverity
    HAVE_ERROR_HANDLING = True
except ImportError:
    HAVE_ERROR_HANDLING = False
    logging.warning("Error handling module not available. Using basic error handling.")

class Position:
    """
    Represents a trading position with comprehensive metadata and risk metrics.
    
    Thread-safe position object with complete tracking of position lifecycle,
    including partial fills, average entry price calculation, and risk metrics.
    """
    
    def __init__(
        self,
        position_id: str,
        symbol: str,
        side: str,  # 'long' or 'short'
        entry_price: float,
        quantity: float,
        timestamp: Optional[datetime] = None,
        leverage: float = 1.0,
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None,
        strategy: Optional[str] = None,
        exchange: str = "binance",
        order_ids: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        # Core position data
        self.position_id = position_id
        self.symbol = symbol
        self.side = side.lower()
        self.initial_quantity = float(quantity)
        self.current_quantity = float(quantity)
        self.leverage = float(leverage)
        
        # Entry data with support for multiple fills
        self.fills = [{"price": float(entry_price), "quantity": float(quantity), "timestamp": timestamp or datetime.now()}]
        self._recalculate_average_entry()
        
        # Exit conditions
        self.stop_loss = float(stop_loss) if stop_loss is not None else None
        self.take_profit = float(take_profit) if take_profit is not None else None
        
        # Exit data
        self.exit_price = None
        self.exit_timestamp = None
        self.exit_reason = None
        self.is_open = True
        self.pnl = 0.0
        self.pnl_percentage = 0.0
        self.realized_pnl = 0.0
        
        # Risk metrics
        self.max_drawdown = 0.0
        self.peak_unrealized_pnl = 0.0
        self.max_adverse_excursion = 0.0  # Worst price movement against position
        self.max_favorable_excursion = 0.0  # Best price movement in favor of position
        
        # Tracking data
        self.strategy = strategy
        self.exchange = exchange
        self.order_ids = order_ids or []
        self.metadata = metadata or {}
        self.entry_timestamp = timestamp or datetime.now()
        self.last_update_timestamp = self.entry_timestamp
        self.duration = timedelta(0)
        
        # Optional advanced settings
        self.trailing_stop = None
        self.trailing_stop_activation = None
        self.trailing_stop_distance = None
        self.scale_out_levels = []  # For partial exit targets
        self.scale_in_levels = []  # For adding to position
        
        # Locking for thread safety
        self._lock = threading.RLock()
        
    def _recalculate_average_entry(self) -> None:
        """Recalculate average entry price based on all fills."""
        with self._lock:
            if not self.fills:
                self.entry_price = 0.0
                return
                
            total_cost = sum(fill["price"] * fill["quantity"] for fill in self.fills)
            total_quantity = sum(fill["quantity"] for fill in self.fills)
            
            if total_quantity > 0:
                self.entry_price = total_cost / total_quantity
            else:
                self.entry_price = 0.0
    
    def add_fill(self, price: float, quantity: float, timestamp: Optional[datetime] = None) -> None:
        """Add a new fill to the position (for partial fills or adding to position)."""
        with self._lock:
            self.fills.append({
                "price": float(price),
                "quantity": float(quantity),
                "timestamp": timestamp or datetime.now()
            })
            self.current_quantity += float(quantity)
            self._recalculate_average_entry()
            self.last_update_timestamp = timestamp or datetime.now()
    
    def update_market_price(self, current_price: float, timestamp: Optional[datetime] = None) -> Dict[str, Any]:
        """
        Update position with current market price and calculate metrics.
        
        Args:
            current_price: Current market price
            timestamp: Timestamp of the update (default: now)
            
        Returns:
            dict: Updated position metrics
        """
        with self._lock:
            now = timestamp or datetime.now()
            self.last_update_timestamp = now
            self.duration = now - self.entry_timestamp
            
            # Calculate unrealized PnL
            if self.side == 'long':
                price_diff = float(current_price) - self.entry_price
            else:  # short
                price_diff = self.entry_price - float(current_price)
                
            # Calculate raw PnL
            self.pnl = price_diff * self.current_quantity * self.leverage
            
            # Calculate PnL as percentage of position value
            position_value = self.entry_price * self.current_quantity
            if position_value > 0:
                self.pnl_percentage = (self.pnl / position_value) * 100
            
            # Update peak PnL and drawdown metrics
            if self.pnl > self.peak_unrealized_pnl:
                self.peak_unrealized_pnl = self.pnl
            
            # Calculate drawdown from peak
            if self.peak_unrealized_pnl > 0:
                current_drawdown = (self.peak_unrealized_pnl - self.pnl) / self.peak_unrealized_pnl
                self.max_drawdown = max(self.max_drawdown, current_drawdown)
            
            # Update max adverse/favorable excursion
            if self.side == 'long':
                adverse_move = self.entry_price - float(current_price)
                favorable_move = float(current_price) - self.entry_price
            else:
                adverse_move = float(current_price) - self.entry_price
                favorable_move = self.entry_price - float(current_price)
                
            self.max_adverse_excursion = max(self.max_adverse_excursion, adverse_move)
            self.max_favorable_excursion = max(self.max_favorable_excursion, favorable_move)
            
            # Check for stop loss / take profit triggers
            exit_triggered = False
            exit_reason = None
            
            if self.stop_loss is not None:
                if (self.side == 'long' and float(current_price) <= self.stop_loss) or \
                   (self.side == 'short' and float(current_price) >= self.stop_loss):
                    exit_triggered = True
                    exit_reason = "stop_loss"
            
            if self.take_profit is not None:
                if (self.side == 'long' and float(current_price) >= self.take_profit) or \
                   (self.side == 'short' and float(current_price) <= self.take_profit):
                    exit_triggered = True
                    exit_reason = "take_profit"
            
            # Check trailing stop if active
            if self.trailing_stop is not None:
                if self.side == 'long':
                    # For long positions, trailing stop activates above a certain price
                    if (self.trailing_stop_activation is None or 
                        float(current_price) >= self.trailing_stop_activation):
                        # Update trailing stop if price improves
                        new_trailing_stop = float(current_price) - self.trailing_stop_distance
                        if self.trailing_stop is None or new_trailing_stop > self.trailing_stop:
                            self.trailing_stop = new_trailing_stop
                        
                        # Check if current price has fallen to the trailing stop
                        if float(current_price) <= self.trailing_stop:
                            exit_triggered = True
                            exit_reason = "trailing_stop"
                else:
                    # For short positions, trailing stop activates below a certain price
                    if (self.trailing_stop_activation is None or 
                        float(current_price) <= self.trailing_stop_activation):
                        # Update trailing stop if price improves
                        new_trailing_stop = float(current_price) + self.trailing_stop_distance
                        if self.trailing_stop is None or new_trailing_stop < self.trailing_stop:
                            self.trailing_stop = new_trailing_stop
                        
                        # Check if current price has risen to the trailing stop
                        if float(current_price) >= self.trailing_stop:
                            exit_triggered = True
                            exit_reason = "trailing_stop"
            
            return {
                "position_id": self.position_id,
                "current_price": float(current_price),
                "pnl": self.pnl,
                "pnl_percentage": self.pnl_percentage,
                "exit_triggered": exit_triggered,
                "exit_reason": exit_reason,
                "updated_at": now
            }
    
    def close_position(
        self, 
        exit_price: float, 
        exit_reason: str = "manual", 
        timestamp: Optional[datetime] = None,
        partial_quantity: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Close the position or a portion of it and calculate final metrics.
        
        Args:
            exit_price: Exit price
            exit_reason: Reason for closing position
            timestamp: Timestamp of the close (default: now)
            partial_quantity: If provided, close only this much of the position
            
        Returns:
            dict: Final position metrics
        """
        with self._lock:
            now = timestamp or datetime.now()
            
            # Handle partial close
            if partial_quantity is not None and partial_quantity < self.current_quantity:
                close_quantity = float(partial_quantity)
                self.current_quantity -= close_quantity
                
                # Calculate PnL for partial close
                if self.side == 'long':
                    price_diff = float(exit_price) - self.entry_price
                else:  # short
                    price_diff = self.entry_price - float(exit_price)
                
                partial_pnl = price_diff * close_quantity * self.leverage
                self.realized_pnl += partial_pnl
                
                return {
                    "position_id": self.position_id,
                    "partial_close": True,
                    "close_quantity": close_quantity,
                    "remaining_quantity": self.current_quantity,
                    "exit_price": float(exit_price),
                    "realized_pnl": partial_pnl,
                    "total_realized_pnl": self.realized_pnl,
                    "exit_reason": exit_reason,
                    "closed_at": now
                }
            
            # Full close
            self.exit_price = float(exit_price)
            self.exit_timestamp = now
            self.exit_reason = exit_reason
            self.is_open = False
            self.duration = now - self.entry_timestamp
            
            # Calculate final PnL
            if self.side == 'long':
                price_diff = float(exit_price) - self.entry_price
            else:  # short
                price_diff = self.entry_price - float(exit_price)
                
            self.pnl = price_diff * self.current_quantity * self.leverage
            self.realized_pnl += self.pnl
            
            position_value = self.entry_price * self.initial_quantity
            if position_value > 0:
                self.pnl_percentage = (self.pnl / position_value) * 100
            
            return {
                "position_id": self.position_id,
                "symbol": self.symbol,
                "side": self.side,
                "entry_price": self.entry_price,
                "exit_price": self.exit_price,
                "quantity": self.initial_quantity,
                "leverage": self.leverage,
                "pnl": self.pnl,
                "pnl_percentage": self.pnl_percentage,
                "realized_pnl": self.realized_pnl,
                "max_drawdown": self.max_drawdown,
                "duration_seconds": self.duration.total_seconds(),
                "entry_timestamp": self.entry_timestamp,
                "exit_timestamp": self.exit_timestamp,
                "exit_reason": self.exit_reason,
                "strategy": self.strategy
            }
    
    def set_trailing_stop(
        self, 
        distance: float, 
        activation_price: Optional[float] = None
    ) -> None:
        """
        Set a trailing stop loss for the position.
        
        Args:
            distance: Distance from current price for trailing stop
            activation_price: Price at which trailing stop becomes active (optional)
        """
        with self._lock:
            self.trailing_stop_distance = float(distance)
            if activation_price is not None:
                self.trailing_stop_activation = float(activation_price)
            else:
                self.trailing_stop_activation = None
            self.trailing_stop = None  # Will be set when price reaches activation level
    
    def update_stop_loss(self, new_stop_loss: float) -> None:
        """Update stop loss price."""
        with self._lock:
            self.stop_loss = float(new_stop_loss)
    
    def update_take_profit(self, new_take_profit: float) -> None:
        """Update take profit price."""
        with self._lock:
            self.take_profit = float(new_take_profit)
    
    def add_order_id(self, order_id: str) -> None:
        """Add order ID to position tracking."""
        with self._lock:
            if order_id not in self.order_ids:
                self.order_ids.append(order_id)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert position to dictionary for serialization."""
        with self._lock:
            return {
                "position_id": self.position_id,
                "symbol": self.symbol,
                "side": self.side,
                "entry_price": self.entry_price,
                "current_quantity": self.current_quantity,
                "initial_quantity": self.initial_quantity,
                "leverage": self.leverage,
                "stop_loss": self.stop_loss,
                "take_profit": self.take_profit,
                "trailing_stop": self.trailing_stop,
                "trailing_stop_activation": self.trailing_stop_activation,
                "trailing_stop_distance": self.trailing_stop_distance,
                "is_open": self.is_open,
                "pnl": self.pnl,
                "pnl_percentage": self.pnl_percentage,
                "realized_pnl": self.realized_pnl,
                "max_drawdown": self.max_drawdown,
                "max_adverse_excursion": self.max_adverse_excursion,
                "max_favorable_excursion": self.max_favorable_excursion,
                "entry_timestamp": self.entry_timestamp.isoformat() if self.entry_timestamp else None,
                "exit_timestamp": self.exit_timestamp.isoformat() if self.exit_timestamp else None,
                "last_update_timestamp": self.last_update_timestamp.isoformat() if self.last_update_timestamp else None,
                "exit_reason": self.exit_reason,
                "duration_seconds": self.duration.total_seconds() if self.duration else 0,
                "strategy": self.strategy,
                "exchange": self.exchange,
                "order_ids": self.order_ids,
                "fills": [
                    {
                        "price": fill["price"],
                        "quantity": fill["quantity"],
                        "timestamp": fill["timestamp"].isoformat() if fill["timestamp"] else None
                    }
                    for fill in self.fills
                ],
                "metadata": self.metadata
            }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Position':
        """Create position from dictionary."""
        # Convert ISO timestamps to datetime objects
        entry_timestamp = datetime.fromisoformat(data["entry_timestamp"]) if data.get("entry_timestamp") else None
        
        # Create position object
        position = cls(
            position_id=data["position_id"],
            symbol=data["symbol"],
            side=data["side"],
            entry_price=data["entry_price"],
            quantity=data["initial_quantity"],
            timestamp=entry_timestamp,
            leverage=data.get("leverage", 1.0),
            stop_loss=data.get("stop_loss"),
            take_profit=data.get("take_profit"),
            strategy=data.get("strategy"),
            exchange=data.get("exchange", "binance"),
            order_ids=data.get("order_ids", []),
            metadata=data.get("metadata", {})
        )
        
        # Update position with additional data
        position.current_quantity = data.get("current_quantity", data["initial_quantity"])
        position.is_open = data.get("is_open", True)
        position.pnl = data.get("pnl", 0.0)
        position.pnl_percentage = data.get("pnl_percentage", 0.0)
        position.realized_pnl = data.get("realized_pnl", 0.0)
        position.max_drawdown = data.get("max_drawdown", 0.0)
        position.max_adverse_excursion = data.get("max_adverse_excursion", 0.0)
        position.max_favorable_excursion = data.get("max_favorable_excursion", 0.0)
        
        # Handle exit data
        if data.get("exit_timestamp"):
            position.exit_timestamp = datetime.fromisoformat(data["exit_timestamp"])
        position.exit_price = data.get("exit_price")
        position.exit_reason = data.get("exit_reason")
        
        # Handle trailing stop
        position.trailing_stop = data.get("trailing_stop")
        position.trailing_stop_activation = data.get("trailing_stop_activation")
        position.trailing_stop_distance = data.get("trailing_stop_distance")
        
        # Handle fills if provided
        if "fills" in data and isinstance(data["fills"], list):
            position.fills = []
            for fill in data["fills"]:
                fill_copy = fill.copy()
                if "timestamp" in fill_copy and fill_copy["timestamp"]:
                    fill_copy["timestamp"] = datetime.fromisoformat(fill_copy["timestamp"])
                position.fills.append(fill_copy)
            
            # Recalculate average entry price
            position._recalculate_average_entry()
        
        # Update timestamps
        if data.get("last_update_timestamp"):
            position.last_update_timestamp = datetime.fromisoformat(data["last_update_timestamp"])
        
        # Update duration
        if position.exit_timestamp and position.entry_timestamp:
            position.duration = position.exit_timestamp - position.entry_timestamp
        elif position.last_update_timestamp and position.entry_timestamp:
            position.duration = position.last_update_timestamp - position.entry_timestamp
        
        return position


class PositionManager:
    """
    Comprehensive position management system with performance optimizations.
    
    Features:
    - Thread-safe position tracking
    - Database persistence
    - Risk metrics calculation
    - Portfolio aggregation
    - High-performance position updates
    - Efficient querying and filtering
    """
    
    def __init__(self, db_manager=None, risk_manager=None):
        """
        Initialize position manager.
        
        Args:
            db_manager: Database manager for persistence
            risk_manager: Risk manager for risk calculations
        """
        self.db_manager = db_manager
        self.risk_manager = risk_manager
        
        # Position tracking
        self.positions = {}
        self.historical_positions = {}
        self.positions_lock = threading.RLock()
        
        # Indexing for fast queries
        self._symbol_index = {}
        self._strategy_index = {}
        self._exchange_index = {}
        
        # Performance optimization
        self._position_cache = {}
        self._portfolio_stats_cache = None
        self._portfolio_stats_timestamp = None
        self._cache_ttl = 5  # Cache TTL in seconds
        
        # Portfolio tracking
        self.total_equity = 0.0
        self.initial_equity = 0.0
        self.realized_pnl = 0.0
        self.unrealized_pnl = 0.0
        
        # Initialize from database if available
        self._initialize_from_database()
        
        logging.info("Position manager initialized")

    def _initialize_from_database(self) -> None:
        """Initialize positions from database if available."""
        if self.db_manager is None:
            return
            
        try:
            # Fetch open positions from database
            open_positions_df = self.db_manager.get_open_trades()
            
            if open_positions_df is None or open_positions_df.empty:
                logging.info("No open positions found in database")
                return
                
            # Convert DataFrame to list of dictionaries
            open_positions = open_positions_df.to_dict('records')
            
            # Create position objects
            with self.positions_lock:
                for pos_data in open_positions:
                    try:
                        # Map database fields to position fields
                        position_data = {
                            "position_id": str(pos_data.get("id") or pos_data.get("trade_id", uuid.uuid4())),
                            "symbol": pos_data.get("symbol"),
                            "side": "long" if pos_data.get("action") == "buy" else "short",
                            "entry_price": float(pos_data.get("price", 0)),
                            "initial_quantity": float(pos_data.get("amount", 0)),
                            "current_quantity": float(pos_data.get("amount", 0)),
                            "leverage": float(pos_data.get("leverage", 1.0)),
                            "stop_loss": float(pos_data.get("stop_loss")) if pos_data.get("stop_loss") else None,
                            "take_profit": float(pos_data.get("take_profit")) if pos_data.get("take_profit") else None,
                            "strategy": pos_data.get("strategy"),
                            "exchange": pos_data.get("exchange", "binance"),
                            "order_ids": [pos_data.get("order_id")] if pos_data.get("order_id") else [],
                            "entry_timestamp": pd.to_datetime(pos_data.get("time")).to_pydatetime() if pd.notna(pos_data.get("time")) else datetime.now(),
                            "is_open": pos_data.get("status") == "open"
                        }
                        
                        # Create position
                        position = Position.from_dict(position_data)
                        
                        # Add to tracking
                        self.positions[position.position_id] = position
                        self._add_to_indices(position)
                        
                    except Exception as e:
                        logging.error(f"Error creating position from database: {e}")
            
            logging.info(f"Loaded {len(self.positions)} open positions from database")
            
        except Exception as e:
            logging.error(f"Error initializing positions from database: {e}")

    def create_position(
        self,
        symbol: str,
        side: str,
        entry_price: float,
        quantity: float,
        timestamp: Optional[datetime] = None,
        leverage: float = 1.0,
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None,
        strategy: Optional[str] = None,
        exchange: str = "binance",
        order_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        position_id: Optional[str] = None
    ) -> Position:
        """
        Create and track a new position.
        
        Args:
            symbol: Trading pair symbol
            side: 'long' or 'short'
            entry_price: Entry price
            quantity: Position size
            timestamp: Entry timestamp (optional)
            leverage: Position leverage (default: 1.0)
            stop_loss: Stop loss price (optional)
            take_profit: Take profit price (optional)
            strategy: Strategy name (optional)
            exchange: Exchange name (default: "binance")
            order_id: Order ID (optional)
            metadata: Additional position metadata (optional)
            position_id: Custom position ID (optional)
            
        Returns:
            Position: The created position object
        """
        position_id = position_id or str(uuid.uuid4())
        
        # Validate inputs
        if not symbol or not side or not entry_price or not quantity:
            raise ValueError("Symbol, side, entry price and quantity are required")
        
        if side.lower() not in ['long', 'short']:
            raise ValueError("Side must be 'long' or 'short'")
        
        if float(quantity) <= 0:
            raise ValueError("Quantity must be positive")
        
        if float(leverage) <= 0:
            raise ValueError("Leverage must be positive")
        
        # Create position
        position = Position(
            position_id=position_id,
            symbol=symbol,
            side=side,
            entry_price=entry_price,
            quantity=quantity,
            timestamp=timestamp,
            leverage=leverage,
            stop_loss=stop_loss,
            take_profit=take_profit,
            strategy=strategy,
            exchange=exchange,
            order_ids=[order_id] if order_id else [],
            metadata=metadata
        )
        
        # Add to tracking
        with self.positions_lock:
            self.positions[position_id] = position
            self._add_to_indices(position)
            
            # Invalidate caches
            self._invalidate_caches()
        
        # Store in database if available
        if self.db_manager is not None:
            try:
                trade_data = {
                    "symbol": symbol,
                    "action": "buy" if side.lower() == "long" else "sell",
                    "price": float(entry_price),
                    "amount": float(quantity),
                    "stop_loss": float(stop_loss) if stop_loss is not None else None,
                    "take_profit": float(take_profit) if take_profit is not None else None,
                    "status": "open",
                    "order_id": order_id,
                    "strategy": strategy,
                    "leverage": float(leverage)
                }
                
                trade_id = self.db_manager.store_trade(trade_data)
                
                if trade_id:
                    # Update position with database ID
                    position.metadata["db_id"] = trade_id
            except Exception as e:
                logging.error(f"Error storing position in database: {e}")
        
        logging.info(f"Created position {position_id} for {symbol} ({side})")
        return position

    def close_position(
        self,
        position_id: str,
        exit_price: float,
        exit_reason: str = "manual",
        timestamp: Optional[datetime] = None,
        partial_quantity: Optional[float] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Close a position or partially close it.
        
        Args:
            position_id: ID of the position to close
            exit_price: Exit price
            exit_reason: Reason for closing
            timestamp: Exit timestamp (optional)
            partial_quantity: If provided, close only this much of the position
            
        Returns:
            dict: Position closing details or None if position not found
        """
        with self.positions_lock:
            if position_id not in self.positions:
                logging.warning(f"Position {position_id} not found")
                return None
                
            position = self.positions[position_id]
            
            # Close position
            result = position.close_position(
                exit_price=exit_price,
                exit_reason=exit_reason,
                timestamp=timestamp,
                partial_quantity=partial_quantity
            )
            
            # If fully closed, move to historical positions
            if not position.is_open:
                # Update in database
                if self.db_manager is not None:
                    try:
                        db_id = position.metadata.get("db_id")
                        
                        if db_id:
                            update_data = {
                                "status": "closed",
                                "exit_price": float(exit_price),
                                "exit_time": timestamp or datetime.now(),
                                "pnl": float(position.pnl),
                                "exit_reason": exit_reason
                            }
                            
                            self.db_manager.update_trade(db_id, update_data)
                    except Exception as e:
                        logging.error(f"Error updating position in database: {e}")
                
                # Move to historical positions
                self.historical_positions[position_id] = position
                
                # Remove from active tracking
                del self.positions[position_id]
                self._remove_from_indices(position)
                
                # Update realized PnL
                self.realized_pnl += position.pnl
                
                logging.info(f"Closed position {position_id} with PnL: {position.pnl:.2f}")
            else:
                # Partial close, update in database
                if self.db_manager is not None and partial_quantity is not None:
                    try:
                        db_id = position.metadata.get("db_id")
                        
                        if db_id:
                            update_data = {
                                "amount": float(position.current_quantity)
                            }
                            
                            self.db_manager.update_trade(db_id, update_data)
                    except Exception as e:
                        logging.error(f"Error updating position in database: {e}")
                
                logging.info(f"Partially closed position {position_id}, remaining quantity: {position.current_quantity:.4f}")
            
            # Invalidate caches
            self._invalidate_caches()
            
            return result

    def update_positions(self, market_data: Dict[str, Dict[str, float]]) -> List[Dict[str, Any]]:
        """
        Update all positions with the latest market data.
        
        Args:
            market_data: Dictionary mapping symbols to market data
                        {"BTC/USDT": {"price": 50000.0, "timestamp": datetime}}
            
        Returns:
            list: List of triggered stop loss/take profit events
        """
        triggered_events = []
        
        with self.positions_lock:
            # Reset unrealized PnL
            self.unrealized_pnl = 0.0
            
            # Update each position
            for position_id, position in list(self.positions.items()):
                # Skip if symbol not in market data
                if position.symbol not in market_data:
                    continue
                
                # Get current price and timestamp
                current_price = market_data[position.symbol].get("price")
                timestamp = market_data[position.symbol].get("timestamp")
                
                if current_price is None:
                    continue
                
                # Update position
                update_result = position.update_market_price(current_price, timestamp)
                
                # Add to unrealized PnL
                self.unrealized_pnl += position.pnl
                
                # Check for triggered exits
                if update_result["exit_triggered"]:
                    # Add to triggered events
                    triggered_events.append({
                        "position_id": position_id,
                        "symbol": position.symbol,
                        "current_price": current_price,
                        "trigger_type": update_result["exit_reason"],
                        "side": position.side,
                        "entry_price": position.entry_price,
                        "quantity": position.current_quantity
                    })
            
            # Update total equity
            self.total_equity = self.initial_equity + self.realized_pnl + self.unrealized_pnl
            
            # Invalidate portfolio stats cache
            self._portfolio_stats_cache = None
        
        return triggered_events

    def get_position(self, position_id: str) -> Optional[Position]:
        """Get a specific position by ID."""
        with self.positions_lock:
            return self.positions.get(position_id)

    def get_positions_by_symbol(self, symbol: str) -> List[Position]:
        """Get all positions for a specific symbol."""
        with self.positions_lock:
            return [self.positions[pos_id] for pos_id in self._symbol_index.get(symbol, [])]

    def get_positions_by_strategy(self, strategy: str) -> List[Position]:
        """Get all positions for a specific strategy."""
        with self.positions_lock:
            return [self.positions[pos_id] for pos_id in self._strategy_index.get(strategy, [])]

    def get_positions_by_exchange(self, exchange: str) -> List[Position]:
        """Get all positions for a specific exchange."""
        with self.positions_lock:
            return [self.positions[pos_id] for pos_id in self._exchange_index.get(exchange, [])]

    def get_all_positions(self, include_closed: bool = False) -> List[Position]:
        """
        Get all positions.
        
        Args:
            include_closed: Whether to include closed positions
            
        Returns:
            list: List of positions
        """
        with self.positions_lock:
            # Get active positions
            positions = list(self.positions.values())
            
            # Include closed positions if requested
            if include_closed:
                positions.extend(self.historical_positions.values())
                
            return positions

    def get_all_positions_dict(self, include_closed: bool = False) -> List[Dict[str, Any]]:
        """
        Get all positions as dictionaries.
        
        Args:
            include_closed: Whether to include closed positions
            
        Returns:
            list: List of position dictionaries
        """
        positions = self.get_all_positions(include_closed)
        return [position.to_dict() for position in positions]

    def get_portfolio_stats(self, recalculate: bool = False) -> Dict[str, Any]:
        """
        Get overall portfolio statistics.
        
        Args:
            recalculate: Force recalculation even if cached
            
        Returns:
            dict: Portfolio statistics
        """
        # Check cache
        current_time = time.time()
        if (not recalculate and 
            self._portfolio_stats_cache is not None and 
            self._portfolio_stats_timestamp is not None and
            current_time - self._portfolio_stats_timestamp < self._cache_ttl):
            return self._portfolio_stats_cache
        
        with self.positions_lock:
            # Get all positions
            positions = list(self.positions.values())
            historical = list(self.historical_positions.values())
            
            # Calculate stats
            total_open_positions = len(positions)
            total_positions_all_time = total_open_positions + len(historical)
            
            # Calculate success rate
            profitable_positions = sum(1 for pos in historical if pos.pnl > 0)
            success_rate = profitable_positions / len(historical) if historical else 0
            
            # Calculate exposure by symbol
            exposure_by_symbol = {}
            for position in positions:
                symbol = position.symbol
                position_value = position.entry_price * position.current_quantity * position.leverage
                
                if symbol not in exposure_by_symbol:
                    exposure_by_symbol[symbol] = 0
                
                exposure_by_symbol[symbol] += position_value
            
            # Calculate exposure by strategy
            exposure_by_strategy = {}
            for position in positions:
                strategy = position.strategy or "unknown"
                position_value = position.entry_price * position.current_quantity * position.leverage
                
                if strategy not in exposure_by_strategy:
                    exposure_by_strategy[strategy] = 0
                
                exposure_by_strategy[strategy] += position_value
            
            # Calculate PnL statistics
            if historical:
                avg_profit = sum(pos.pnl for pos in historical if pos.pnl > 0) / profitable_positions if profitable_positions else 0
                avg_loss = sum(abs(pos.pnl) for pos in historical if pos.pnl <= 0) / (len(historical) - profitable_positions) if len(historical) - profitable_positions else 0
                profit_factor = avg_profit / avg_loss if avg_loss > 0 else float('inf')
                
                # Calculate durations
                durations = [pos.duration.total_seconds() for pos in historical if pos.duration]
                avg_duration = sum(durations) / len(durations) if durations else 0
                
                # Calculate max consecutive wins/losses
                ordered_positions = sorted(historical, key=lambda p: p.exit_timestamp or datetime.now())
                
                max_consecutive_wins = 0
                max_consecutive_losses = 0
                current_consecutive_wins = 0
                current_consecutive_losses = 0
                
                for position in ordered_positions:
                    if position.pnl > 0:
                        current_consecutive_wins += 1
                        current_consecutive_losses = 0
                        max_consecutive_wins = max(max_consecutive_wins, current_consecutive_wins)
                    else:
                        current_consecutive_losses += 1
                        current_consecutive_wins = 0
                        max_consecutive_losses = max(max_consecutive_losses, current_consecutive_losses)
            else:
                avg_profit = 0
                avg_loss = 0
                profit_factor = 0
                avg_duration = 0
                max_consecutive_wins = 0
                max_consecutive_losses = 0
            
            # Calculate total exposure
            total_exposure = sum(position.entry_price * position.current_quantity * position.leverage for position in positions)
            
            # Calculate exposure percentage
            exposure_percentage = (total_exposure / self.total_equity) * 100 if self.total_equity > 0 else 0
            
            # Build stats dictionary
            stats = {
                "total_equity": self.total_equity,
                "initial_equity": self.initial_equity,
                "realized_pnl": self.realized_pnl,
                "unrealized_pnl": self.unrealized_pnl,
                "total_open_positions": total_open_positions,
                "total_positions_all_time": total_positions_all_time,
                "success_rate": success_rate,
                "exposure_by_symbol": exposure_by_symbol,
                "exposure_by_strategy": exposure_by_strategy,
                "avg_profit": avg_profit,
                "avg_loss": avg_loss,
                "profit_factor": profit_factor,
                "avg_duration_seconds": avg_duration,
                "max_consecutive_wins": max_consecutive_wins,
                "max_consecutive_losses": max_consecutive_losses,
                "total_exposure": total_exposure,
                "exposure_percentage": exposure_percentage,
                "calculated_at": datetime.now()
            }
            
            # Update cache
            self._portfolio_stats_cache = stats
            self._portfolio_stats_timestamp = current_time
            
            return stats

    def update_position_risk_parameters(
        self,
        position_id: str,
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None
    ) -> bool:
        """
        Update risk parameters for a position.
        
        Args:
            position_id: Position ID
            stop_loss: New stop loss price (optional)
            take_profit: New take profit price (optional)
            
        Returns:
            bool: Success status
        """
        with self.positions_lock:
            if position_id not in self.positions:
                logging.warning(f"Position {position_id} not found")
                return False
                
            position = self.positions[position_id]
            
            # Update stop loss if provided
            if stop_loss is not None:
                position.update_stop_loss(stop_loss)
                
            # Update take profit if provided
            if take_profit is not None:
                position.update_take_profit(take_profit)
            
            # Update in database
            if self.db_manager is not None:
                try:
                    db_id = position.metadata.get("db_id")
                    
                    if db_id:
                        update_data = {}
                        
                        if stop_loss is not None:
                            update_data["stop_loss"] = float(stop_loss)
                            
                        if take_profit is not None:
                            update_data["take_profit"] = float(take_profit)
                        
                        if update_data:
                            self.db_manager.update_trade(db_id, update_data)
                except Exception as e:
                    logging.error(f"Error updating position risk parameters in database: {e}")
            
            return True

    def set_position_trailing_stop(
        self,
        position_id: str,
        distance: float,
        activation_price: Optional[float] = None
    ) -> bool:
        """
        Set trailing stop for a position.
        
        Args:
            position_id: Position ID
            distance: Distance from current price for trailing stop
            activation_price: Price at which trailing stop becomes active (optional)
            
        Returns:
            bool: Success status
        """
        with self.positions_lock:
            if position_id not in self.positions:
                logging.warning(f"Position {position_id} not found")
                return False
                
            position = self.positions[position_id]
            
            # Set trailing stop
            position.set_trailing_stop(distance, activation_price)
            
            # Not updating database as trailing stop logic is managed by the position manager
            
            return True

    def set_initial_equity(self, equity: float) -> None:
        """Set initial equity value."""
        self.initial_equity = float(equity)
        self.total_equity = self.initial_equity + self.realized_pnl + self.unrealized_pnl

    def add_to_position(
        self,
        position_id: str,
        price: float,
        quantity: float,
        timestamp: Optional[datetime] = None,
        order_id: Optional[str] = None
    ) -> bool:
        """
        Add to an existing position.
        
        Args:
            position_id: Position ID
            price: Entry price for additional quantity
            quantity: Additional quantity
            timestamp: Timestamp (optional)
            order_id: Order ID (optional)
            
        Returns:
            bool: Success status
        """
        with self.positions_lock:
            if position_id not in self.positions:
                logging.warning(f"Position {position_id} not found")
                return False
                
            position = self.positions[position_id]
            
            # Add fill
            position.add_fill(price, quantity, timestamp)
            
            # Add order ID if provided
            if order_id:
                position.add_order_id(order_id)
            
            # Update in database
            if self.db_manager is not None:
                try:
                    db_id = position.metadata.get("db_id")
                    
                    if db_id:
                        update_data = {
                            "amount": float(position.current_quantity)
                        }
                        
                        self.db_manager.update_trade(db_id, update_data)
                except Exception as e:
                    logging.error(f"Error updating position in database: {e}")
            
            # Invalidate caches
            self._invalidate_caches()
            
            return True

    def bulk_update_positions(self, market_data_batch: Dict[str, Dict[str, float]]) -> List[Dict[str, Any]]:
        """
        Update all positions in a single batch operation.
        
        Optimized version for high-frequency updates.
        
        Args:
            market_data_batch: Dictionary mapping symbols to market data
            
        Returns:
            list: List of triggered stop loss/take profit events
        """
        # Note: This implementation already updates all positions in a batch operation,
        # so we can just delegate to the standard update method
        return self.update_positions(market_data_batch)

    def scan_for_exit_signals(self, strategy_signals: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Scan for exit signals from strategies.
        
        Args:
            strategy_signals: Dictionary mapping symbols to strategy signals
            
        Returns:
            list: List of positions with exit signals
        """
        exit_signals = []
        
        with self.positions_lock:
            for position_id, position in self.positions.items():
                symbol = position.symbol
                
                # Skip if symbol not in signals
                if symbol not in strategy_signals:
                    continue
                
                # Get signal for this symbol
                signal = strategy_signals[symbol]
                
                # Check for exit signal
                if position.side == 'long' and signal.get('action') == 'sell':
                    exit_signals.append({
                        "position_id": position_id,
                        "symbol": symbol,
                        "side": position.side,
                        "signal_type": "strategy_exit",
                        "entry_price": position.entry_price,
                        "current_price": signal.get('price', None),
                        "quantity": position.current_quantity
                    })
                elif position.side == 'short' and signal.get('action') == 'buy':
                    exit_signals.append({
                        "position_id": position_id,
                        "symbol": symbol,
                        "side": position.side,
                        "signal_type": "strategy_exit",
                        "entry_price": position.entry_price,
                        "current_price": signal.get('price', None),
                        "quantity": position.current_quantity
                    })
        
        return exit_signals

    def get_exposure_summary(self) -> Dict[str, Any]:
        """
        Get summary of current exposure across different dimensions.
        
        Returns:
            dict: Exposure summary
        """
        with self.positions_lock:
            # Calculate total exposure
            total_exposure = sum(position.entry_price * position.current_quantity * position.leverage 
                               for position in self.positions.values())
            
            # Calculate exposure by symbol
            exposure_by_symbol = {}
            for position in self.positions.values():
                symbol = position.symbol
                position_value = position.entry_price * position.current_quantity * position.leverage
                
                if symbol not in exposure_by_symbol:
                    exposure_by_symbol[symbol] = 0
                
                exposure_by_symbol[symbol] += position_value
            
            # Calculate exposure by strategy
            exposure_by_strategy = {}
            for position in self.positions.values():
                strategy = position.strategy or "unknown"
                position_value = position.entry_price * position.current_quantity * position.leverage
                
                if strategy not in exposure_by_strategy:
                    exposure_by_strategy[strategy] = 0
                
                exposure_by_strategy[strategy] += position_value
            
            # Calculate exposure by side
            long_exposure = sum(position.entry_price * position.current_quantity * position.leverage 
                              for position in self.positions.values() 
                              if position.side == 'long')
            
            short_exposure = sum(position.entry_price * position.current_quantity * position.leverage 
                               for position in self.positions.values() 
                               if position.side == 'short')
            
            # Calculate net exposure
            net_exposure = long_exposure - short_exposure
            
            # Calculate exposure ratios
            exposure_to_equity = total_exposure / self.total_equity if self.total_equity > 0 else 0
            net_exposure_to_equity = net_exposure / self.total_equity if self.total_equity > 0 else 0
            
            return {
                "total_exposure": total_exposure,
                "long_exposure": long_exposure,
                "short_exposure": short_exposure,
                "net_exposure": net_exposure,
                "exposure_by_symbol": exposure_by_symbol,
                "exposure_by_strategy": exposure_by_strategy,
                "exposure_to_equity": exposure_to_equity,
                "net_exposure_to_equity": net_exposure_to_equity
            }

    def export_positions(self, format: str = "json") -> Union[str, pd.DataFrame]:
        """
        Export positions to JSON or DataFrame.
        
        Args:
            format: Export format ("json" or "dataframe")
            
        Returns:
            str or DataFrame: Exported positions
        """
        positions_list = self.get_all_positions_dict(include_closed=True)
        
        if format.lower() == "json":
            return json.dumps(positions_list, indent=2)
        else:
            return pd.DataFrame(positions_list)

    def import_positions(self, data: Union[str, pd.DataFrame, List[Dict[str, Any]]]) -> int:
        """
        Import positions from JSON, DataFrame, or list of dictionaries.
        
        Args:
            data: Position data to import
            
        Returns:
            int: Number of positions imported
        """
        positions_list = []
        
        # Convert input to list of dictionaries
        if isinstance(data, str):
            # Parse JSON string
            try:
                positions_list = json.loads(data)
            except json.JSONDecodeError:
                logging.error("Invalid JSON data")
                return 0
        elif isinstance(data, pd.DataFrame):
            # Convert DataFrame to list of dictionaries
            positions_list = data.to_dict('records')
        elif isinstance(data, list):
            # Already a list
            positions_list = data
        else:
            logging.error("Unsupported data format")
            return 0
        
        # Import positions
        imported_count = 0
        
        with self.positions_lock:
            for pos_data in positions_list:
                try:
                    position = Position.from_dict(pos_data)
                    
                    if position.is_open:
                        # Add to open positions
                        self.positions[position.position_id] = position
                        self._add_to_indices(position)
                    else:
                        # Add to historical positions
                        self.historical_positions[position.position_id] = position
                    
                    imported_count += 1
                except Exception as e:
                    logging.error(f"Error importing position: {e}")
            
            # Invalidate caches
            self._invalidate_caches()
        
        return imported_count

    def calculate_drawdown(self) -> Dict[str, float]:
        """
        Calculate portfolio drawdown metrics.
        
        Returns:
            dict: Drawdown metrics
        """
        # Calculate max equity
        max_equity = self.initial_equity
        current_equity = self.total_equity
        
        # Calculate drawdown
        absolute_drawdown = max_equity - current_equity if max_equity > current_equity else 0
        percentage_drawdown = (absolute_drawdown / max_equity) * 100 if max_equity > 0 else 0
        
        return {
            "max_equity": max_equity,
            "current_equity": current_equity,
            "absolute_drawdown": absolute_drawdown,
            "percentage_drawdown": percentage_drawdown
        }

    def calculate_profit_loss(self) -> Dict[str, float]:
        """
        Calculate profit and loss metrics.
        
        Returns:
            dict: Profit and loss metrics
        """
        return {
            "realized_pnl": self.realized_pnl,
            "unrealized_pnl": self.unrealized_pnl,
            "total_pnl": self.realized_pnl + self.unrealized_pnl,
            "roi_percentage": ((self.realized_pnl + self.unrealized_pnl) / self.initial_equity) * 100 if self.initial_equity > 0 else 0
        }

    def _add_to_indices(self, position: Position) -> None:
        """Add position to indices for fast lookups."""
        # Add to symbol index
        if position.symbol not in self._symbol_index:
            self._symbol_index[position.symbol] = set()
        self._symbol_index[position.symbol].add(position.position_id)
        
        # Add to strategy index
        strategy = position.strategy or "unknown"
        if strategy not in self._strategy_index:
            self._strategy_index[strategy] = set()
        self._strategy_index[strategy].add(position.position_id)
        
        # Add to exchange index
        if position.exchange not in self._exchange_index:
            self._exchange_index[position.exchange] = set()
        self._exchange_index[position.exchange].add(position.position_id)

    def _remove_from_indices(self, position: Position) -> None:
        """Remove position from indices."""
        # Remove from symbol index
        if position.symbol in self._symbol_index:
            self._symbol_index[position.symbol].discard(position.position_id)
            if not self._symbol_index[position.symbol]:
                del self._symbol_index[position.symbol]
        
        # Remove from strategy index
        strategy = position.strategy or "unknown"
        if strategy in self._strategy_index:
            self._strategy_index[strategy].discard(position.position_id)
            if not self._strategy_index[strategy]:
                del self._strategy_index[strategy]
        
        # Remove from exchange index
        if position.exchange in self._exchange_index:
            self._exchange_index[position.exchange].discard(position.position_id)
            if not self._exchange_index[position.exchange]:
                del self._exchange_index[position.exchange]

    def _invalidate_caches(self) -> None:
        """Invalidate caches."""
        self._position_cache = {}
        self._portfolio_stats_cache = None

    def update_portfolio_snapshot(self) -> bool:
        """
        Create a snapshot of current portfolio state and store in database.
        
        Returns:
            bool: Success status
        """
        if self.db_manager is None:
            return False
            
        try:
            # Calculate portfolio values
            portfolio_data = {
                "total_value": self.total_equity,
                "cash_value": self.total_equity - self.unrealized_pnl,  # Approximate cash value
                "crypto_value": self.unrealized_pnl,  # Current position values
                "pnl_daily": 0  # Placeholder, would need to track daily changes
            }
            
            # Store snapshot
            success = self.db_manager.store_portfolio_snapshot(portfolio_data)
            return success
        except Exception as e:
            logging.error(f"Error creating portfolio snapshot: {e}")
            return False

    def get_position_performance_metrics(self, position_id: str) -> Optional[Dict[str, Any]]:
        """
        Get detailed performance metrics for a specific position.
        
        Args:
            position_id: Position ID
            
        Returns:
            dict: Performance metrics or None if position not found
        """
        with self.positions_lock:
            # Check if position exists
            position = self.positions.get(position_id)
            
            if position is None:
                # Check historical positions
                position = self.historical_positions.get(position_id)
                
            if position is None:
                return None
            
            # Calculate metrics
            entry_value = position.entry_price * position.initial_quantity
            
            if position.is_open:
                # Position still open, we'll need current price
                # This is limited since we don't have current price in this context
                current_value = entry_value + position.pnl
                unrealized_roi = (position.pnl / entry_value) * 100 if entry_value > 0 else 0
                
                return {
                    "position_id": position_id,
                    "symbol": position.symbol,
                    "side": position.side,
                    "entry_price": position.entry_price,
                    "entry_value": entry_value,
                    "current_value": current_value,
                    "unrealized_pnl": position.pnl,
                    "unrealized_roi": unrealized_roi,
                    "duration": position.duration.total_seconds(),
                    "max_drawdown": position.max_drawdown,
                    "max_adverse_excursion": position.max_adverse_excursion,
                    "max_favorable_excursion": position.max_favorable_excursion,
                    "is_open": True
                }
            else:
                # Closed position
                exit_value = position.exit_price * position.initial_quantity
                realized_roi = (position.pnl / entry_value) * 100 if entry_value > 0 else 0
                
                return {
                    "position_id": position_id,
                    "symbol": position.symbol,
                    "side": position.side,
                    "entry_price": position.entry_price,
                    "exit_price": position.exit_price,
                    "entry_value": entry_value,
                    "exit_value": exit_value,
                    "realized_pnl": position.pnl,
                    "realized_roi": realized_roi,
                    "duration": position.duration.total_seconds(),
                    "max_drawdown": position.max_drawdown,
                    "max_adverse_excursion": position.max_adverse_excursion,
                    "max_favorable_excursion": position.max_favorable_excursion,
                    "is_open": False,
                    "exit_reason": position.exit_reason
                }

    def get_historical_performance(self, days: int = 30) -> Dict[str, Any]:
        """
        Get historical performance statistics over a given time period.
        
        Args:
            days: Number of days to look back
            
        Returns:
            dict: Historical performance metrics
        """
        with self.positions_lock:
            # Calculate start date
            start_date = datetime.now() - timedelta(days=days)
            
            # Get positions that were open or closed during this period
            relevant_positions = []
            
            # Check historical positions
            for position in self.historical_positions.values():
                if position.exit_timestamp and position.exit_timestamp >= start_date:
                    relevant_positions.append(position)
            
            # Include currently open positions
            relevant_positions.extend(self.positions.values())
            
            # No positions in time period
            if not relevant_positions:
                return {
                    "period_days": days,
                    "total_trades": 0,
                    "win_rate": 0,
                    "profit_factor": 0,
                    "total_pnl": 0,
                    "roi": 0,
                    "sharpe_ratio": 0,
                    "max_drawdown": 0,
                    "avg_trade_duration": 0
                }
            
            # Calculate metrics
            closed_positions = [p for p in relevant_positions if not p.is_open]
            total_trades = len(closed_positions)
            winning_trades = sum(1 for p in closed_positions if p.pnl > 0)
            win_rate = winning_trades / total_trades if total_trades > 0 else 0
            
            total_profit = sum(p.pnl for p in closed_positions if p.pnl > 0)
            total_loss = sum(abs(p.pnl) for p in closed_positions if p.pnl < 0)
            profit_factor = total_profit / total_loss if total_loss > 0 else float('inf')
            
            total_pnl = sum(p.pnl for p in closed_positions)
            
            # Calculate ROI (assuming initial_equity was stable)
            roi = (total_pnl / self.initial_equity) * 100 if self.initial_equity > 0 else 0
            
            # Calculate Sharpe ratio (simplified)
            daily_returns = []
            
            # Group trades by day
            trades_by_day = {}
            for position in closed_positions:
                if position.exit_timestamp:
                    day = position.exit_timestamp.date()
                    if day not in trades_by_day:
                        trades_by_day[day] = []
                    trades_by_day[day].append(position)
            
            # Calculate daily returns
            for day, positions in trades_by_day.items():
                daily_pnl = sum(p.pnl for p in positions)
                daily_return = daily_pnl / self.initial_equity if self.initial_equity > 0 else 0
                daily_returns.append(daily_return)
            
            if daily_returns:
                avg_return = sum(daily_returns) / len(daily_returns)
                std_dev = np.std(daily_returns) if len(daily_returns) > 1 else 0.01
                sharpe_ratio = (avg_return / std_dev) * np.sqrt(252) if std_dev > 0 else 0
            else:
                sharpe_ratio = 0
            
            # Calculate max drawdown
            max_drawdown = max((p.max_drawdown for p in relevant_positions), default=0)
            
            # Calculate average trade duration
            durations = [p.duration.total_seconds() for p in closed_positions if p.duration]
            avg_duration = sum(durations) / len(durations) if durations else 0
            
            return {
                "period_days": days,
                "total_trades": total_trades,
                "win_rate": win_rate,
                "profit_factor": profit_factor,
                "total_pnl": total_pnl,
                "roi": roi,
                "sharpe_ratio": sharpe_ratio,
                "max_drawdown": max_drawdown,
                "avg_trade_duration": avg_duration
            }

    def sync_with_database(self) -> bool:
        """
        Synchronize position manager state with database.
        
        Returns:
            bool: Success status
        """
        if self.db_manager is None:
            logging.warning("Cannot sync with database: database manager not initialized")
            return False
            
        try:
            # Get open positions from database
            db_positions = self.db_manager.get_open_trades()
            
            if db_positions is None or db_positions.empty:
                logging.info("No open positions found in database")
                return True
                
            # Convert to dictionary format
            db_positions_dict = {}
            for _, row in db_positions.iterrows():
                position_id = str(row.get("id") or row.get("trade_id", ""))
                if position_id:
                    db_positions_dict[position_id] = row.to_dict()
            
            # Get current positions
            with self.positions_lock:
                current_positions = set(self.positions.keys())
                db_position_ids = set(db_positions_dict.keys())
                
                # Positions to add (in database but not in memory)
                positions_to_add = db_position_ids - current_positions
                
                # Positions to remove (in memory but not in database)
                positions_to_remove = current_positions - db_position_ids
                
                # Update existing positions
                for position_id in current_positions.intersection(db_position_ids):
                    # Update metadata
                    self.positions[position_id].metadata["db_id"] = position_id
                
                # Add new positions
                for position_id in positions_to_add:
                    db_position = db_positions_dict[position_id]
                    
                    # Create position object
                    try:
                        position_data = {
                            "position_id": position_id,
                            "symbol": db_position.get("symbol"),
                            "side": "long" if db_position.get("action") == "buy" else "short",
                            "entry_price": float(db_position.get("price", 0)),
                            "initial_quantity": float(db_position.get("amount", 0)),
                            "current_quantity": float(db_position.get("amount", 0)),
                            "leverage": float(db_position.get("leverage", 1.0)),
                            "stop_loss": float(db_position.get("stop_loss")) if db_position.get("stop_loss") else None,
                            "take_profit": float(db_position.get("take_profit")) if db_position.get("take_profit") else None,
                            "strategy": db_position.get("strategy"),
                            "exchange": db_position.get("exchange", "binance"),
                            "order_ids": [db_position.get("order_id")] if db_position.get("order_id") else [],
                            "entry_timestamp": pd.to_datetime(db_position.get("time")).to_pydatetime() if pd.notna(db_position.get("time")) else datetime.now(),
                            "is_open": db_position.get("status") == "open"
                        }
                        
                        position = Position.from_dict(position_data)
                        position.metadata["db_id"] = position_id
                        
                        # Add to tracking
                        self.positions[position_id] = position
                        self._add_to_indices(position)
                        
                    except Exception as e:
                        logging.error(f"Error creating position from database: {e}")
                
                # Remove positions not in database
                for position_id in positions_to_remove:
                    position = self.positions[position_id]
                    
                    # Move to historical positions
                    self.historical_positions[position_id] = position
                    
                    # Remove from active tracking
                    del self.positions[position_id]
                    self._remove_from_indices(position)
                
                # Invalidate caches
                self._invalidate_caches()
                
            return True
            
        except Exception as e:
            logging.error(f"Error synchronizing with database: {e}")
            return False

    def add_metadata_to_position(self, position_id: str, metadata: Dict[str, Any]) -> bool:
        """
        Add metadata to a position.
        
        Args:
            position_id: Position ID
            metadata: Metadata to add
            
        Returns:
            bool: Success status
        """
        with self.positions_lock:
            if position_id in self.positions:
                position = self.positions[position_id]
            elif position_id in self.historical_positions:
                position = self.historical_positions[position_id]
            else:
                logging.warning(f"Position {position_id} not found")
                return False
            
            # Update metadata
            position.metadata.update(metadata)
            
            return True

    def get_active_position_count(self) -> int:
        """Get number of active positions."""
        with self.positions_lock:
            return len(self.positions)

    def get_historical_position_count(self) -> int:
        """Get number of historical positions."""
        with self.positions_lock:
            return len(self.historical_positions)

    def reset(self) -> None:
        """Reset position manager state."""
        with self.positions_lock:
            self.positions = {}
            self.historical_positions = {}
            self._symbol_index = {}
            self._strategy_index = {}
            self._exchange_index = {}
            self._position_cache = {}
            self._portfolio_stats_cache = None
            self._portfolio_stats_timestamp = None
            self.total_equity = 0.0
            self.initial_equity = 0.0
            self.realized_pnl = 0.0
            self.unrealized_pnl = 0.0


# Apply error handling decorator if available
if HAVE_ERROR_HANDLING:
    PositionManager.create_position = safe_execute(ErrorCategory.TRADE_EXECUTION, None)(PositionManager.create_position)
    PositionManager.close_position = safe_execute(ErrorCategory.TRADE_EXECUTION, None)(PositionManager.close_position)
    PositionManager.update_positions = safe_execute(ErrorCategory.DATA_PROCESSING, [])(PositionManager.update_positions)
    PositionManager.bulk_update_positions = safe_execute(ErrorCategory.DATA_PROCESSING, [])(PositionManager.bulk_update_positions)
    PositionManager.sync_with_database = safe_execute(ErrorCategory.DATABASE, False)(PositionManager.sync_with_database)
    Position.update_market_price = safe_execute(ErrorCategory.DATA_PROCESSING, {})(Position.update_market_price)
    Position.close_position = safe_execute(ErrorCategory.TRADE_EXECUTION, {})(Position.close_position)
