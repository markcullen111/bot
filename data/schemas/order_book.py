# order_book.py
import numpy as np
import pandas as pd
import time
from datetime import datetime
from sortedcontainers import SortedDict
from collections import defaultdict
import threading
import logging
from typing import Dict, List, Tuple, Optional, Any, Union, Callable

# Setup logging
logger = logging.getLogger('order_book')

class Price:
    """
    Efficient price representation with necessary precision.
    Using a dedicated class for price handling to ensure consistent price operations.
    """
    __slots__ = ('_value')
    
    def __init__(self, value: float):
        """
        Initialize price with float value, ensuring proper precision.
        
        Args:
            value: Price value
        """
        self._value = float(value)
        
    def __hash__(self) -> int:
        """Hash implementation for using Price as dictionary key."""
        # Use quantization to a fixed precision to handle floating point equality issues
        return hash(round(self._value, 10))
        
    def __eq__(self, other) -> bool:
        """Equality comparison with precision handling."""
        if isinstance(other, Price):
            return abs(self._value - other._value) < 1e-10
        return abs(self._value - float(other)) < 1e-10
        
    def __lt__(self, other) -> bool:
        """Less than comparison."""
        if isinstance(other, Price):
            return self._value < other._value
        return self._value < float(other)
        
    def __gt__(self, other) -> bool:
        """Greater than comparison."""
        if isinstance(other, Price):
            return self._value > other._value
        return self._value > float(other)
    
    def __float__(self) -> float:
        """Convert to float."""
        return self._value
    
    def __repr__(self) -> str:
        """String representation."""
        return f"{self._value:.8f}"


class OrderBookLevel:
    """
    Represents a single price level in the order book.
    Optimized for quick updates and minimal memory usage.
    """
    __slots__ = ('price', 'amount', 'order_count', 'price_float', 'orders')
    
    def __init__(self, price: Union[float, Price], amount: float, order_count: int = 1):
        """
        Initialize order book level.
        
        Args:
            price: Price level
            amount: Total amount at this price level
            order_count: Number of orders at this price level
        """
        self.price = price if isinstance(price, Price) else Price(price)
        self.price_float = float(self.price)
        self.amount = float(amount)
        self.order_count = order_count
        self.orders = {}  # Only used for L3 order books
        
    def __repr__(self) -> str:
        """String representation."""
        return f"OrderBookLevel({self.price_float:.8f}, {self.amount:.8f}, {self.order_count})"


class OrderBook:
    """
    Fast and memory-efficient order book implementation for real-time trading.
    Supports L1, L2, and L3 order books with comprehensive analytics.
    
    Features:
    - Thread-safe operations for concurrent access
    - Efficient storage with sorted containers
    - Low-latency updates and lookups
    - Comprehensive market microstructure analytics
    - Support for multiple exchanges and symbols
    - Optimized for algorithmic trading applications
    """
    
    def __init__(self, symbol: str, exchange: str = "default", depth: int = 100, level: int = 2):
        """
        Initialize order book for a specific trading pair.
        
        Args:
            symbol: Trading pair symbol (e.g., 'BTC/USD')
            exchange: Exchange name
            depth: Maximum depth to track
            level: Order book level (1=top only, 2=price aggregated, 3=order by order)
        """
        self.symbol = symbol
        self.exchange = exchange
        self.depth = depth
        self.level = level
        
        # Use SortedDict for efficient order book structures
        # Bids are sorted in descending order (highest price first)
        # Asks are sorted in ascending order (lowest price first)
        self.bids = SortedDict(lambda k: -float(k))  # Price -> OrderBookLevel
        self.asks = SortedDict()  # Price -> OrderBookLevel
        
        # For L3 order books
        self.orders = {}  # Order ID -> (Price, Amount, Side)
        
        # Timestamps
        self.timestamp = None
        self.last_update = None
        
        # Sequence number for consistency checking
        self.sequence = 0
        
        # Statistics and analytics
        self.spread = None
        self.mid_price = None
        self.vwap = None
        self.imbalance = 0.0
        
        # Thread safety
        self._lock = threading.RLock()
        
        # Performance metrics
        self.update_count = 0
        self.update_time_sum = 0
        self.update_time_max = 0
        
        # Snapshot handling
        self.last_snapshot_time = 0
        self.updates_since_snapshot = 0
        
        # Event callbacks
        self.on_update_callbacks = []
        
        logger.info(f"Initialized {exchange} order book for {symbol} with depth {depth} at level {level}")
    
    def update(self, bids: List[List[float]], asks: List[List[float]], 
               timestamp: Optional[float] = None, check_consistency: bool = True) -> bool:
        """
        Update the order book with new bids and asks.
        Thread-safe implementation for concurrent access.
        
        Args:
            bids: List of [price, amount] pairs
            asks: List of [price, amount] pairs
            timestamp: Update timestamp (defaults to current time)
            check_consistency: Whether to check for consistency with previous state
            
        Returns:
            bool: True if update was successful
        """
        update_start_time = time.time()
        
        with self._lock:
            if timestamp is None:
                timestamp = time.time()
                
            # Update sequence number
            new_sequence = self.sequence + 1
            
            if check_consistency:
                # Simple consistency check based on timestamp
                if self.timestamp and timestamp < self.timestamp:
                    logger.warning(f"Out of order update for {self.symbol}: "
                                  f"new ts={timestamp}, last ts={self.timestamp}")
                    return False
            
            # Process bids
            new_bids = {}
            for price_amount in bids:
                if len(price_amount) >= 2:
                    price, amount = price_amount[:2]
                    price_obj = Price(price)
                    
                    if amount > 0:
                        # Add or update price level
                        order_count = 1
                        if len(price_amount) > 2:
                            order_count = int(price_amount[2])
                        new_bids[price_obj] = OrderBookLevel(price_obj, amount, order_count)
            
            # Process asks
            new_asks = {}
            for price_amount in asks:
                if len(price_amount) >= 2:
                    price, amount = price_amount[:2]
                    price_obj = Price(price)
                    
                    if amount > 0:
                        # Add or update price level
                        order_count = 1
                        if len(price_amount) > 2:
                            order_count = int(price_amount[2])
                        new_asks[price_obj] = OrderBookLevel(price_obj, amount, order_count)
            
            # Replace order book data
            self.bids = SortedDict({p: new_bids[p] for p in sorted(new_bids.keys(), reverse=True)[:self.depth]})
            self.asks = SortedDict({p: new_asks[p] for p in sorted(new_asks.keys())[:self.depth]})
            
            # Update timestamps
            self.timestamp = timestamp
            self.last_update = time.time()
            self.sequence = new_sequence
            
            # Update analytics
            self._update_analytics()
            
            # Performance tracking
            self.update_count += 1
            update_time = time.time() - update_start_time
            self.update_time_sum += update_time
            self.update_time_max = max(self.update_time_max, update_time)
            
            # Trigger callbacks
            self._notify_update_callbacks()
            
            return True

    def apply_delta(self, bids: List[List[float]], asks: List[List[float]], 
                   timestamp: Optional[float] = None, sequence: Optional[int] = None) -> bool:
        """
        Apply incremental updates to the order book.
        More efficient than full updates for exchanges that support deltas.
        
        Args:
            bids: List of [price, amount] pairs
            asks: List of [price, amount] pairs
            timestamp: Update timestamp (defaults to current time)
            sequence: Sequence number for consistency checking
            
        Returns:
            bool: True if update was successful
        """
        update_start_time = time.time()
        
        with self._lock:
            if timestamp is None:
                timestamp = time.time()
                
            # Check sequence if provided
            if sequence is not None and self.sequence > 0:
                if sequence <= self.sequence:
                    logger.warning(f"Out of sequence update for {self.symbol}: "
                                  f"new seq={sequence}, current seq={self.sequence}")
                    return False
                elif sequence > self.sequence + 1:
                    logger.warning(f"Sequence gap for {self.symbol}: "
                                  f"new seq={sequence}, current seq={self.sequence}")
                    self.updates_since_snapshot += 1
                    # Request a snapshot if too many updates since last snapshot
                    if self.updates_since_snapshot > 1000:
                        logger.warning(f"Too many updates since last snapshot for {self.symbol}, "
                                      f"snapshot should be requested")
                        return False
                
                self.sequence = sequence
            else:
                self.sequence += 1
                
            # Process bid updates
            for price_amount in bids:
                if len(price_amount) >= 2:
                    price, amount = price_amount[:2]
                    price_obj = Price(price)
                    
                    if amount > 0:
                        # Add or update price level
                        order_count = 1
                        if len(price_amount) > 2:
                            order_count = int(price_amount[2])
                        self.bids[price_obj] = OrderBookLevel(price_obj, amount, order_count)
                    else:
                        # Remove price level
                        self.bids.pop(price_obj, None)
            
            # Process ask updates
            for price_amount in asks:
                if len(price_amount) >= 2:
                    price, amount = price_amount[:2]
                    price_obj = Price(price)
                    
                    if amount > 0:
                        # Add or update price level
                        order_count = 1
                        if len(price_amount) > 2:
                            order_count = int(price_amount[2])
                        self.asks[price_obj] = OrderBookLevel(price_obj, amount, order_count)
                    else:
                        # Remove price level
                        self.asks.pop(price_obj, None)
            
            # Update timestamps
            self.timestamp = timestamp
            self.last_update = time.time()
            
            # Update analytics
            self._update_analytics()
            
            # Performance tracking
            self.update_count += 1
            self.updates_since_snapshot += 1
            update_time = time.time() - update_start_time
            self.update_time_sum += update_time
            self.update_time_max = max(self.update_time_max, update_time)
            
            # Trigger callbacks
            self._notify_update_callbacks()
            
            return True

    def process_l3_update(self, data: Dict[str, Any]) -> bool:
        """
        Process Level 3 (order by order) updates.
        Specialized for exchanges that provide full order book data.
        
        Args:
            data: Update data containing order information
            
        Returns:
            bool: True if update was successful
        """
        if self.level < 3:
            logger.warning(f"L3 update received for L{self.level} order book")
            return False
            
        with self._lock:
            timestamp = data.get('timestamp', time.time())
            
            # Process updates based on the type
            update_type = data.get('type', '')
            
            if update_type == 'received':
                # New order received, not yet in the book
                pass
                
            elif update_type == 'open':
                # Order added to the book
                order_id = data.get('order_id')
                side = data.get('side')
                price = Price(float(data.get('price', 0)))
                amount = float(data.get('remaining_size', 0))
                
                if order_id and side and amount > 0:
                    # Store order
                    self.orders[order_id] = (price, amount, side)
                    
                    # Update order book
                    book_side = self.bids if side == 'buy' else self.asks
                    
                    if price in book_side:
                        level = book_side[price]
                        level.amount += amount
                        level.order_count += 1
                        level.orders[order_id] = amount
                    else:
                        level = OrderBookLevel(price, amount, 1)
                        level.orders[order_id] = amount
                        book_side[price] = level
                
            elif update_type == 'done':
                # Order removed from the book
                order_id = data.get('order_id')
                
                if order_id in self.orders:
                    price, amount, side = self.orders.pop(order_id)
                    book_side = self.bids if side == 'buy' else self.asks
                    
                    if price in book_side:
                        level = book_side[price]
                        level.amount -= amount
                        level.order_count -= 1
                        level.orders.pop(order_id, None)
                        
                        # Remove level if empty
                        if level.amount <= 0 or level.order_count <= 0:
                            book_side.pop(price)
                
            elif update_type == 'match':
                # Trade occurred
                maker_order_id = data.get('maker_order_id')
                size = float(data.get('size', 0))
                
                if maker_order_id in self.orders:
                    price, old_amount, side = self.orders[maker_order_id]
                    new_amount = max(0, old_amount - size)
                    
                    # Update order
                    self.orders[maker_order_id] = (price, new_amount, side)
                    
                    # Update book
                    book_side = self.bids if side == 'buy' else self.asks
                    if price in book_side:
                        level = book_side[price]
                        level.amount -= size
                        level.orders[maker_order_id] = new_amount
                        
                        # Remove order if filled
                        if new_amount <= 0:
                            level.orders.pop(maker_order_id, None)
                            level.order_count -= 1
                        
                        # Remove level if empty
                        if level.amount <= 0 or level.order_count <= 0:
                            book_side.pop(price)
                
            elif update_type == 'change':
                # Order size changed
                order_id = data.get('order_id')
                new_size = float(data.get('new_size', 0))
                
                if order_id in self.orders:
                    price, old_amount, side = self.orders[order_id]
                    size_delta = new_size - old_amount
                    
                    # Update order
                    self.orders[order_id] = (price, new_size, side)
                    
                    # Update book
                    book_side = self.bids if side == 'buy' else self.asks
                    if price in book_side:
                        level = book_side[price]
                        level.amount += size_delta
                        level.orders[order_id] = new_size
                        
                        # Remove level if empty
                        if level.amount <= 0:
                            book_side.pop(price)
            
            # Update timestamps
            self.timestamp = timestamp
            self.last_update = time.time()
            
            # Update analytics
            self._update_analytics()
            
            # Trigger callbacks
            self._notify_update_callbacks()
            
            return True

    def snapshot_received(self) -> None:
        """
        Mark that a snapshot has been received.
        Resets the updates counter for tracking consistency.
        """
        with self._lock:
            self.last_snapshot_time = time.time()
            self.updates_since_snapshot = 0

    def register_update_callback(self, callback: Callable) -> None:
        """
        Register a callback function to be called on order book updates.
        
        Args:
            callback: Function to call on updates
        """
        if callback not in self.on_update_callbacks:
            self.on_update_callbacks.append(callback)

    def unregister_update_callback(self, callback: Callable) -> None:
        """
        Unregister a callback function.
        
        Args:
            callback: Function to remove
        """
        if callback in self.on_update_callbacks:
            self.on_update_callbacks.remove(callback)

    def _notify_update_callbacks(self) -> None:
        """Notify all registered callbacks of updates."""
        for callback in self.on_update_callbacks:
            try:
                callback(self)
            except Exception as e:
                logger.error(f"Error in order book update callback: {e}")

    def _update_analytics(self) -> None:
        """Update order book analytics."""
        if not self.bids or not self.asks:
            return
            
        # Get best bid and ask
        best_bid_price = float(next(iter(self.bids)))
        best_ask_price = float(next(iter(self.asks)))
        
        # Calculate spread and mid price
        self.spread = best_ask_price - best_bid_price
        self.mid_price = (best_bid_price + best_ask_price) / 2
        
        # Calculate volume-weighted average price (VWAP)
        total_bid_value = sum(float(p) * level.amount for p, level in self.bids.items())
        total_bid_volume = sum(level.amount for level in self.bids.values())
        
        total_ask_value = sum(float(p) * level.amount for p, level in self.asks.items())
        total_ask_volume = sum(level.amount for level in self.asks.values())
        
        if total_bid_volume > 0 and total_ask_volume > 0:
            bid_vwap = total_bid_value / total_bid_volume
            ask_vwap = total_ask_value / total_ask_volume
            self.vwap = (bid_vwap + ask_vwap) / 2
        
        # Calculate bid-ask imbalance
        total_volume = total_bid_volume + total_ask_volume
        if total_volume > 0:
            self.imbalance = (total_bid_volume - total_ask_volume) / total_volume

    def get_average_update_time(self) -> float:
        """
        Get average time for order book updates.
        
        Returns:
            float: Average update time in milliseconds
        """
        if self.update_count > 0:
            return (self.update_time_sum / self.update_count) * 1000
        return 0

    def get_max_update_time(self) -> float:
        """
        Get maximum time for order book updates.
        
        Returns:
            float: Maximum update time in milliseconds
        """
        return self.update_time_max * 1000

    def get_bid_levels(self, depth: int = None) -> List[List[float]]:
        """
        Get bid levels from the order book.
        
        Args:
            depth: Maximum number of levels to return
            
        Returns:
            List of [price, amount] pairs
        """
        if depth is None:
            depth = self.depth
            
        with self._lock:
            return [[float(price), level.amount] for price, level in list(self.bids.items())[:depth]]

    def get_ask_levels(self, depth: int = None) -> List[List[float]]:
        """
        Get ask levels from the order book.
        
        Args:
            depth: Maximum number of levels to return
            
        Returns:
            List of [price, amount] pairs
        """
        if depth is None:
            depth = self.depth
            
        with self._lock:
            return [[float(price), level.amount] for price, level in list(self.asks.items())[:depth]]

    def get_top_bid(self) -> Optional[Tuple[float, float]]:
        """
        Get the highest bid price and amount.
        
        Returns:
            Tuple of (price, amount) or None if no bids
        """
        with self._lock:
            if not self.bids:
                return None
            price = next(iter(self.bids))
            return float(price), self.bids[price].amount

    def get_top_ask(self) -> Optional[Tuple[float, float]]:
        """
        Get the lowest ask price and amount.
        
        Returns:
            Tuple of (price, amount) or None if no asks
        """
        with self._lock:
            if not self.asks:
                return None
            price = next(iter(self.asks))
            return float(price), self.asks[price].amount

    def get_price_for_volume(self, volume: float, side: str) -> Optional[float]:
        """
        Calculate the price needed to buy/sell a specific volume.
        
        Args:
            volume: Volume to buy/sell
            side: 'buy' or 'sell'
            
        Returns:
            float: Average execution price or None if not enough liquidity
        """
        with self._lock:
            book_side = self.asks if side == 'buy' else self.bids
            
            remaining_volume = volume
            total_cost = 0.0
            
            for price, level in book_side.items():
                price_float = float(price)
                available = level.amount
                
                if remaining_volume <= available:
                    # Enough liquidity at this level
                    total_cost += price_float * remaining_volume
                    return total_cost / volume
                    
                # Partial fill at this level
                total_cost += price_float * available
                remaining_volume -= available
                
            # Not enough liquidity
            return None

    def get_liquidity_in_range(self, price_start: float, price_end: float) -> Dict[str, float]:
        """
        Calculate the liquidity available in a price range.
        
        Args:
            price_start: Start price
            price_end: End price
            
        Returns:
            Dict with 'bid_volume', 'ask_volume', 'total_volume'
        """
        with self._lock:
            bid_volume = sum(
                level.amount for price, level in self.bids.items()
                if price_start <= float(price) <= price_end
            )
            
            ask_volume = sum(
                level.amount for price, level in self.asks.items()
                if price_start <= float(price) <= price_end
            )
            
            return {
                'bid_volume': bid_volume,
                'ask_volume': ask_volume,
                'total_volume': bid_volume + ask_volume
            }

    def estimate_slippage(self, volume: float, side: str) -> Dict[str, float]:
        """
        Estimate slippage for a given volume.
        
        Args:
            volume: Volume to buy/sell
            side: 'buy' or 'sell'
            
        Returns:
            Dict with slippage metrics
        """
        with self._lock:
            if not self.bids or not self.asks:
                return {'slippage_bps': 0, 'avg_price': 0, 'best_price': 0}
                
            book_side = self.asks if side == 'buy' else self.bids
            best_price = float(next(iter(book_side)))
            
            # Calculate execution price
            price = self.get_price_for_volume(volume, side)
            
            if price is None:
                return {'slippage_bps': float('inf'), 'avg_price': 0, 'best_price': best_price}
                
            # Calculate slippage in basis points
            if side == 'buy':
                slippage_bps = (price - best_price) / best_price * 10000
            else:
                slippage_bps = (best_price - price) / best_price * 10000
                
            return {
                'slippage_bps': slippage_bps,
                'avg_price': price,
                'best_price': best_price
            }

    def calculate_market_impact(self, volume: float, side: str) -> Dict[str, float]:
        """
        Calculate the market impact of a trade.
        
        Args:
            volume: Volume to buy/sell
            side: 'buy' or 'sell'
            
        Returns:
            Dict with market impact metrics
        """
        with self._lock:
            # Get current mid price
            if not self.mid_price:
                return {'price_impact_bps': 0, 'expected_price': 0}
                
            # Calculate execution price
            price = self.get_price_for_volume(volume, side)
            
            if price is None:
                return {'price_impact_bps': float('inf'), 'expected_price': 0}
                
            # Calculate impact in basis points relative to mid price
            if side == 'buy':
                impact_bps = (price - self.mid_price) / self.mid_price * 10000
            else:
                impact_bps = (self.mid_price - price) / self.mid_price * 10000
                
            return {
                'price_impact_bps': impact_bps,
                'expected_price': price
            }

    def get_order_book_imbalance(self, levels: int = 10) -> float:
        """
        Calculate order book imbalance based on a specific number of levels.
        
        Args:
            levels: Number of levels to include
            
        Returns:
            float: Imbalance ratio (-1.0 to 1.0)
        """
        with self._lock:
            bid_volume = sum(level.amount for _, level in list(self.bids.items())[:levels])
            ask_volume = sum(level.amount for _, level in list(self.asks.items())[:levels])
            
            total_volume = bid_volume + ask_volume
            if total_volume == 0:
                return 0
                
            return (bid_volume - ask_volume) / total_volume

    def calculate_vwap_deviation(self) -> float:
        """
        Calculate deviation between mid price and VWAP.
        
        Returns:
            float: VWAP deviation in basis points
        """
        with self._lock:
            if not self.mid_price or not self.vwap:
                return 0
                
            return (self.vwap - self.mid_price) / self.mid_price * 10000

    def to_dataframe(self) -> Dict[str, pd.DataFrame]:
        """
        Convert order book to pandas DataFrames for analysis.
        
        Returns:
            Dict with 'bids' and 'asks' DataFrames
        """
        with self._lock:
            bids_df = pd.DataFrame([
                {'price': float(price), 'amount': level.amount, 'orders': level.order_count}
                for price, level in self.bids.items()
            ])
            
            asks_df = pd.DataFrame([
                {'price': float(price), 'amount': level.amount, 'orders': level.order_count}
                for price, level in self.asks.items()
            ])
            
            if not bids_df.empty:
                bids_df = bids_df.sort_values('price', ascending=False)
                
            if not asks_df.empty:
                asks_df = asks_df.sort_values('price', ascending=True)
                
            return {'bids': bids_df, 'asks': asks_df}

    def get_status(self) -> Dict[str, Any]:
        """
        Get order book status.
        
        Returns:
            Dict with order book status information
        """
        with self._lock:
            return {
                'symbol': self.symbol,
                'exchange': self.exchange,
                'timestamp': self.timestamp,
                'last_update': self.last_update,
                'bid_levels': len(self.bids),
                'ask_levels': len(self.asks),
                'top_bid': self.get_top_bid(),
                'top_ask': self.get_top_ask(),
                'spread': self.spread,
                'mid_price': self.mid_price,
                'vwap': self.vwap,
                'imbalance': self.imbalance,
                'update_count': self.update_count,
                'avg_update_time_ms': self.get_average_update_time(),
                'max_update_time_ms': self.get_max_update_time(),
                'updates_since_snapshot': self.updates_since_snapshot
            }

    def reset(self) -> None:
        """Reset the order book."""
        with self._lock:
            self.bids = SortedDict(lambda k: -float(k))
            self.asks = SortedDict()
            self.orders = {}
            self.timestamp = None
            self.last_update = None
            self.sequence = 0
            self.spread = None
            self.mid_price = None
            self.vwap = None
            self.imbalance = 0.0
            self.update_count = 0
            self.update_time_sum = 0
            self.update_time_max = 0
            self.last_snapshot_time = 0
            self.updates_since_snapshot = 0
            
            logger.info(f"Order book reset for {self.symbol} on {self.exchange}")

    def is_valid(self) -> bool:
        """
        Check if the order book is valid.
        
        Returns:
            bool: True if order book is valid
        """
        with self._lock:
            # Check if order book has both bids and asks
            if not self.bids or not self.asks:
                return False
                
            # Check if best bid is less than best ask
            if self.bids and self.asks:
                best_bid = float(next(iter(self.bids)))
                best_ask = float(next(iter(self.asks)))
                if best_bid >= best_ask:
                    logger.warning(f"Invalid order book for {self.symbol}: best_bid={best_bid} >= best_ask={best_ask}")
                    return False
                    
            # Check timestamp
            if self.timestamp is None:
                return False
                
            # Check for stale order book
            if self.last_update and time.time() - self.last_update > 300:  # 5 minutes
                logger.warning(f"Stale order book for {self.symbol}, last update was {time.time() - self.last_update:.1f}s ago")
                return False
                
            return True

    def __str__(self) -> str:
        """String representation of the order book."""
        with self._lock:
            top_bid = self.get_top_bid()
            top_ask = self.get_top_ask()
            
            top_bid_str = f"{top_bid[0]:.8f} @ {top_bid[1]:.8f}" if top_bid else "None"
            top_ask_str = f"{top_ask[0]:.8f} @ {top_ask[1]:.8f}" if top_ask else "None"
            
            return (f"OrderBook({self.symbol} @ {self.exchange}, "
                   f"Top Bid: {top_bid_str}, Top Ask: {top_ask_str}, "
                   f"Levels: {len(self.bids)}/{len(self.asks)})")


class OrderBookManager:
    """
    Manages multiple order books for different symbols and exchanges.
    Provides a centralized interface for order book operations.
    """
    
    def __init__(self):
        """Initialize order book manager."""
        self.order_books = {}  # (exchange, symbol) -> OrderBook
        self._lock = threading.RLock()
        logger.info("Order book manager initialized")
        
    def get_order_book(self, symbol: str, exchange: str = "default", 
                      create_if_missing: bool = True, depth: int = 100, level: int = 2) -> Optional[OrderBook]:
        """
        Get order book for a specific symbol and exchange.
        
        Args:
            symbol: Trading pair symbol
            exchange: Exchange name
            create_if_missing: Create order book if it doesn't exist
            depth: Maximum depth for new order book
            level: Order book level for new order book
            
        Returns:
            OrderBook or None if not found
        """
        key = (exchange, symbol)
        
        with self._lock:
            if key in self.order_books:
                return self.order_books[key]
                
            if create_if_missing:
                order_book = OrderBook(symbol, exchange, depth, level)
                self.order_books[key] = order_book
                return order_book
                
        return None
        
    def update_order_book(self, symbol: str, exchange: str, bids: List[List[float]], 
                         asks: List[List[float]], timestamp: Optional[float] = None) -> bool:
        """
        Update order book for a specific symbol and exchange.
        
        Args:
            symbol: Trading pair symbol
            exchange: Exchange name
            bids: List of [price, amount] pairs
            asks: List of [price, amount] pairs
            timestamp: Update timestamp
            
        Returns:
            bool: True if update was successful
        """
        order_book = self.get_order_book(symbol, exchange)
        if order_book:
            return order_book.update(bids, asks, timestamp)
        return False
        
    def apply_delta(self, symbol: str, exchange: str, bids: List[List[float]], 
                   asks: List[List[float]], timestamp: Optional[float] = None,
                   sequence: Optional[int] = None) -> bool:
        """
        Apply incremental updates to order book.
        
        Args:
            symbol: Trading pair symbol
            exchange: Exchange name
            bids: List of [price, amount] pairs
            asks: List of [price, amount] pairs
            timestamp: Update timestamp
            sequence: Sequence number
            
        Returns:
            bool: True if update was successful
        """
        order_book = self.get_order_book(symbol, exchange)
        if order_book:
            return order_book.apply_delta(bids, asks, timestamp, sequence)
        return False
        
    def process_l3_update(self, symbol: str, exchange: str, data: Dict[str, Any]) -> bool:
        """
        Process Level 3 order book update.
        
        Args:
            symbol: Trading pair symbol
            exchange: Exchange name
            data: Update data
            
        Returns:
            bool: True if update was successful
        """
        order_book = self.get_order_book(symbol, exchange, level=3)
        if order_book:
            return order_book.process_l3_update(data)
        return False
        
    def snapshot_received(self, symbol: str, exchange: str) -> None:
        """
        Mark that a snapshot has been received.
        
        Args:
            symbol: Trading pair symbol
            exchange: Exchange name
        """
        order_book = self.get_order_book(symbol, exchange, create_if_missing=False)
        if order_book:
            order_book.snapshot_received()
            
    def get_all_symbols(self) -> List[Tuple[str, str]]:
        """
        Get all symbols and exchanges.
        
        Returns:
            List of (exchange, symbol) tuples
        """
        with self._lock:
            return list(self.order_books.keys())
            
    def reset_order_book(self, symbol: str, exchange: str) -> bool:
        """
        Reset order book for a specific symbol and exchange.
        
        Args:
            symbol: Trading pair symbol
            exchange: Exchange name
            
        Returns:
            bool: True if reset was successful
        """
        key = (exchange, symbol)
        
        with self._lock:
            if key in self.order_books:
                self.order_books[key].reset()
                return True
        return False
        
    def remove_order_book(self, symbol: str, exchange: str) -> bool:
        """
        Remove order book for a specific symbol and exchange.
        
        Args:
            symbol: Trading pair symbol
            exchange: Exchange name
            
        Returns:
            bool: True if removal was successful
        """
        key = (exchange, symbol)
        
        with self._lock:
            if key in self.order_books:
                del self.order_books[key]
                return True
        return False
        
    def reset_all(self) -> None:
        """Reset all order books."""
        with self._lock:
            for order_book in self.order_books.values():
                order_book.reset()
                
    def get_status(self) -> Dict[str, Any]:
        """
        Get status of all order books.
        
        Returns:
            Dict with status information
        """
        with self._lock:
            return {
                f"{exchange}_{symbol}": order_book.get_status()
                for (exchange, symbol), order_book in self.order_books.items()
            }


# Global instance for easy access
order_book_manager = OrderBookManager()


def get_order_book(symbol: str, exchange: str = "default") -> Optional[OrderBook]:
    """
    Get order book for a specific symbol and exchange.
    Convenience function for accessing the global order book manager.
    
    Args:
        symbol: Trading pair symbol
        exchange: Exchange name
        
    Returns:
        OrderBook or None if not found
    """
    return order_book_manager.get_order_book(symbol, exchange)


def update_order_book(symbol: str, exchange: str, bids: List[List[float]], 
                     asks: List[List[float]], timestamp: Optional[float] = None) -> bool:
    """
    Update order book for a specific symbol and exchange.
    Convenience function for accessing the global order book manager.
    
    Args:
        symbol: Trading pair symbol
        exchange: Exchange name
        bids: List of [price, amount] pairs
        asks: List of [price, amount] pairs
        timestamp: Update timestamp
        
    Returns:
        bool: True if update was successful
    """
    return order_book_manager.update_order_book(symbol, exchange, bids, asks, timestamp)
