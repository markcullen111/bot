# exchange_connector.py

import os
import time
import json
import hmac
import hashlib
import base64
import logging
import threading
import queue
import ccxt
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Union, Tuple, Callable
from datetime import datetime, timedelta
from functools import wraps
from concurrent.futures import ThreadPoolExecutor
from dotenv import load_dotenv

# Import error handling if available
try:
    from error_handling import safe_execute, ErrorCategory, ErrorSeverity
    HAVE_ERROR_HANDLING = True
except ImportError:
    HAVE_ERROR_HANDLING = False
    logging.warning("Error handling module not available. Using basic error handling.")

# Configure logging
logger = logging.getLogger("exchange_connector")

# Load environment variables
load_dotenv()

# Rate limiting decorator
def rate_limited(max_per_second: float):
    """Decorator to rate limit function calls"""
    min_interval = 1.0 / max_per_second
    last_called = [0.0]
    lock = threading.Lock()
    
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            with lock:
                elapsed = time.time() - last_called[0]
                wait_time = min_interval - elapsed
                if wait_time > 0:
                    time.sleep(wait_time)
                last_called[0] = time.time()
            return func(*args, **kwargs)
        return wrapper
    return decorator

class ExchangeRateLimiter:
    """Rate limiter for exchange API calls with dynamic backoff"""
    
    def __init__(self, max_calls: int = 10, time_frame: float = 1.0, safety_factor: float = 0.8):
        """
        Initialize rate limiter.
        
        Args:
            max_calls: Maximum calls allowed in time_frame
            time_frame: Time frame in seconds
            safety_factor: Reduce max calls by this factor for safety
        """
        self.max_calls = int(max_calls * safety_factor)
        self.time_frame = time_frame
        self.calls = []
        self.lock = threading.Lock()
        self.backoff_time = 0.0
        
    def __call__(self, func):
        """Make this class callable as a decorator"""
        @wraps(func)
        def wrapper(*args, **kwargs):
            with self.lock:
                now = time.time()
                
                # Clear old calls
                self.calls = [call_time for call_time in self.calls if now - call_time <= self.time_frame]
                
                # Check if we need to wait for rate limit
                if len(self.calls) >= self.max_calls:
                    # Wait until the oldest call falls out of the time frame
                    wait_time = self.time_frame - (now - self.calls[0]) + self.backoff_time
                    if wait_time > 0:
                        time.sleep(wait_time)
                    now = time.time()  # Update current time
                    self.calls = self.calls[1:]  # Remove oldest call
                
                # Add this call to the list
                self.calls.append(now)
                
                # Attempt to call the function
                try:
                    result = func(*args, **kwargs)
                    # Gradually reduce backoff time on success
                    self.backoff_time = max(0.0, self.backoff_time - 0.01)
                    return result
                except Exception as e:
                    # Check for rate limit errors and increase backoff
                    if "rate limit" in str(e).lower() or "too many requests" in str(e).lower():
                        self.backoff_time += 0.5  # Increase backoff time
                        logger.warning(f"Rate limit hit, increasing backoff to {self.backoff_time}s")
                    raise
                
        return wrapper

class ExchangeConnectionPool:
    """Thread-safe pool of exchange connections to reduce overhead"""
    
    def __init__(self, exchange_id: str, api_key: str, api_secret: str, max_connections: int = 5):
        """
        Initialize connection pool.
        
        Args:
            exchange_id: CCXT exchange ID
            api_key: API key
            api_secret: API secret
            max_connections: Maximum concurrent connections
        """
        self.exchange_id = exchange_id
        self.api_key = api_key
        self.api_secret = api_secret
        self.max_connections = max_connections
        self.connections = queue.Queue(maxsize=max_connections)
        self.lock = threading.Lock()
        
        # Create initial connections
        for _ in range(max_connections):
            conn = self._create_connection()
            self.connections.put(conn)
    
    def _create_connection(self):
        """Create a new exchange connection"""
        exchange_class = getattr(ccxt, self.exchange_id)
        exchange = exchange_class({
            'apiKey': self.api_key,
            'secret': self.api_secret,
            'enableRateLimit': True,
            'options': {
                'adjustForTimeDifference': True
            }
        })
        # Load markets on initialization
        exchange.load_markets()
        return exchange
    
    def get_connection(self, timeout: float = 10.0):
        """Get a connection from the pool with timeout"""
        try:
            return _ConnectionContext(self, self.connections.get(timeout=timeout))
        except queue.Empty:
            raise TimeoutError("Timeout waiting for available exchange connection")
    
    def return_connection(self, connection):
        """Return a connection to the pool"""
        try:
            self.connections.put(connection, block=False)
        except queue.Full:
            # Pool is full, close this connection
            pass

class _ConnectionContext:
    """Context manager for safely using connections from the pool"""
    
    def __init__(self, pool, connection):
        self.pool = pool
        self.connection = connection
        
    def __enter__(self):
        return self.connection
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.pool.return_connection(self.connection)

class ExchangeConnector:
    """
    Comprehensive exchange connector for cryptocurrency trading with failover support.
    
    Features:
    - Multiple exchange support with unified interface
    - Connection pooling for efficiency and thread safety
    - Rate limiting with dynamic backoff
    - Comprehensive error handling and retry logic
    - Market data standardization
    - Order execution with validation 
    - Position tracking
    - Historical data retrieval
    """
    
    # Supported exchanges with their configurations
    SUPPORTED_EXCHANGES = {
        'binance': {
            'max_requests_per_second': 10,
            'has_websocket': True,
            'maker_fee': 0.001,
            'taker_fee': 0.001,
            'has_margin': True,
            'timeframes': ['1m', '3m', '5m', '15m', '30m', '1h', '2h', '4h', '6h', '8h', '12h', '1d', '3d', '1w', '1M']
        },
        'coinbase': {
            'max_requests_per_second': 5,
            'has_websocket': True,
            'maker_fee': 0.005,
            'taker_fee': 0.005,
            'has_margin': False,
            'timeframes': ['1m', '5m', '15m', '1h', '6h', '1d']
        },
        'kraken': {
            'max_requests_per_second': 3,
            'has_websocket': True,
            'maker_fee': 0.0016,
            'taker_fee': 0.0026,
            'has_margin': True,
            'timeframes': ['1m', '5m', '15m', '30m', '1h', '4h', '1d', '1w', '2w']
        },
        'kucoin': {
            'max_requests_per_second': 8,
            'has_websocket': True,
            'maker_fee': 0.001,
            'taker_fee': 0.001,
            'has_margin': True,
            'timeframes': ['1m', '3m', '5m', '15m', '30m', '1h', '2h', '4h', '6h', '8h', '12h', '1d', '1w']
        },
        'ftx': {
            'max_requests_per_second': 8,
            'has_websocket': True,
            'maker_fee': 0.0002,
            'taker_fee': 0.0007,
            'has_margin': True,
            'timeframes': ['15s', '1m', '5m', '15m', '1h', '4h', '1d', '3d', '1w', '2w', '1M']
        },
        'okex': {
            'max_requests_per_second': 6,
            'has_websocket': True,
            'maker_fee': 0.0008,
            'taker_fee': 0.001,
            'has_margin': True,
            'timeframes': ['1m', '3m', '5m', '15m', '30m', '1h', '2h', '4h', '6h', '12h', '1d', '1w', '1M', '3M', '6M', '1y']
        }
    }
    
    def __init__(self, 
                 primary_exchange: str = None, 
                 secondary_exchange: str = None, 
                 max_retries: int = 3, 
                 retry_delay: float = 1.0, 
                 use_testnet: bool = False,
                 validate_orders: bool = True,
                 connection_timeout: float = 10.0,
                 max_connections: int = 5,
                 api_credentials: Dict[str, Dict[str, str]] = None):
        """
        Initialize the exchange connector.
        
        Args:
            primary_exchange: Primary exchange ID (default: from env var EXCHANGE_ID)
            secondary_exchange: Secondary exchange ID for failover
            max_retries: Maximum number of retries for API calls
            retry_delay: Base delay between retries (will be exponentially increased)
            use_testnet: Use exchange testnet instead of production
            validate_orders: Validate orders before submission
            connection_timeout: Timeout for getting a connection from the pool
            max_connections: Maximum connections per exchange
            api_credentials: Dictionary of exchange credentials (overrides env vars)
        """
        # Load default exchange from environment if not provided
        if primary_exchange is None:
            primary_exchange = os.getenv("EXCHANGE_ID", "binance").lower()
            
        # Validate exchanges
        if primary_exchange not in self.SUPPORTED_EXCHANGES:
            raise ValueError(f"Unsupported primary exchange: {primary_exchange}")
            
        if secondary_exchange and secondary_exchange not in self.SUPPORTED_EXCHANGES:
            raise ValueError(f"Unsupported secondary exchange: {secondary_exchange}")
            
        self.primary_exchange = primary_exchange
        self.secondary_exchange = secondary_exchange
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.use_testnet = use_testnet
        self.validate_orders = validate_orders
        self.connection_timeout = connection_timeout
        
        # Initialize API credentials
        self.api_credentials = {}
        
        # Load from provided credentials or environment variables
        if api_credentials:
            self.api_credentials = api_credentials
        else:
            self._load_credentials_from_env()
            
        # Create connection pools
        self.connection_pools = {}
        self._init_connection_pools(max_connections)
        
        # Create rate limiters
        self.rate_limiters = {}
        self._init_rate_limiters()
        
        # Cache for market info
        self.markets_cache = {}
        self.symbols_cache = {}
        self.last_market_fetch = {}
        
        # Tracked orders and positions
        self.tracked_orders = {}  # order_id -> order_details
        self.positions = {}       # symbol -> position_details
        
        # Thread safety
        self.orders_lock = threading.RLock()
        self.positions_lock = threading.RLock()
        self.cache_lock = threading.RLock()
        
        # Initialize markets cache for primary exchange
        self._fetch_markets(self.primary_exchange)
        
        logger.info(f"Exchange connector initialized with primary exchange: {primary_exchange}")
    
    def _load_credentials_from_env(self):
        """Load API credentials from environment variables"""
        # Check for primary exchange credentials
        primary_key = os.getenv(f"{self.primary_exchange.upper()}_API_KEY")
        primary_secret = os.getenv(f"{self.primary_exchange.upper()}_API_SECRET")
        
        if not primary_key or not primary_secret:
            raise ValueError(f"API credentials for {self.primary_exchange} not found in environment variables")
            
        self.api_credentials[self.primary_exchange] = {
            'api_key': primary_key,
            'api_secret': primary_secret
        }
        
        # Check for secondary exchange credentials if applicable
        if self.secondary_exchange:
            secondary_key = os.getenv(f"{self.secondary_exchange.upper()}_API_KEY")
            secondary_secret = os.getenv(f"{self.secondary_exchange.upper()}_API_SECRET")
            
            if secondary_key and secondary_secret:
                self.api_credentials[self.secondary_exchange] = {
                    'api_key': secondary_key,
                    'api_secret': secondary_secret
                }
            else:
                logger.warning(f"Secondary exchange {self.secondary_exchange} credentials not found, disabling failover")
                self.secondary_exchange = None
    
    def _init_connection_pools(self, max_connections: int):
        """Initialize connection pools for configured exchanges"""
        # Create pool for primary exchange
        if self.primary_exchange in self.api_credentials:
            creds = self.api_credentials[self.primary_exchange]
            self.connection_pools[self.primary_exchange] = ExchangeConnectionPool(
                self.primary_exchange,
                creds['api_key'],
                creds['api_secret'],
                max_connections
            )
            
        # Create pool for secondary exchange if configured
        if self.secondary_exchange and self.secondary_exchange in self.api_credentials:
            creds = self.api_credentials[self.secondary_exchange]
            self.connection_pools[self.secondary_exchange] = ExchangeConnectionPool(
                self.secondary_exchange,
                creds['api_key'],
                creds['api_secret'],
                max_connections
            )
    
    def _init_rate_limiters(self):
        """Initialize rate limiters for configured exchanges"""
        # Create rate limiter for primary exchange
        if self.primary_exchange in self.SUPPORTED_EXCHANGES:
            max_calls = self.SUPPORTED_EXCHANGES[self.primary_exchange]['max_requests_per_second']
            self.rate_limiters[self.primary_exchange] = ExchangeRateLimiter(max_calls)
            
        # Create rate limiter for secondary exchange if configured
        if self.secondary_exchange and self.secondary_exchange in self.SUPPORTED_EXCHANGES:
            max_calls = self.SUPPORTED_EXCHANGES[self.secondary_exchange]['max_requests_per_second']
            self.rate_limiters[self.secondary_exchange] = ExchangeRateLimiter(max_calls)
    
    def _get_rate_limiter(self, exchange_id: str):
        """Get rate limiter for specified exchange"""
        return self.rate_limiters.get(exchange_id, lambda x: x)  # Default to no-op
    
    def _execute_with_retry(self, exchange_id: str, method_name: str, *args, **kwargs):
        """Execute an exchange method with retries and failover"""
        primary_failed = False
        current_exchange = exchange_id
        
        for attempt in range(self.max_retries):
            try:
                # Get rate limiter for current exchange
                rate_limiter = self._get_rate_limiter(current_exchange)
                
                # Get a connection from the pool
                with self.connection_pools[current_exchange].get_connection(self.connection_timeout) as connection:
                    # Get the method to call
                    method = getattr(connection, method_name)
                    
                    # Apply rate limiting
                    @rate_limiter
                    def rate_limited_call():
                        return method(*args, **kwargs)
                    
                    # Execute the method
                    result = rate_limited_call()
                    return result
                    
            except Exception as e:
                # Log the error
                logger.warning(f"Error executing {method_name} on {current_exchange} (attempt {attempt+1}/{self.max_retries}): {str(e)}")
                
                # If this is the primary exchange and we have a secondary, try failover
                if current_exchange == self.primary_exchange and self.secondary_exchange and not primary_failed:
                    logger.info(f"Failing over to secondary exchange: {self.secondary_exchange}")
                    current_exchange = self.secondary_exchange
                    primary_failed = True
                    continue
                    
                # If we're on the last attempt, raise the exception
                if attempt == self.max_retries - 1:
                    if HAVE_ERROR_HANDLING:
                        from error_handling import TradingSystemError
                        raise TradingSystemError(
                            f"Failed to execute {method_name} after {self.max_retries} attempts",
                            ErrorCategory.NETWORK,
                            ErrorSeverity.ERROR,
                            e
                        )
                    else:
                        raise
                        
                # Otherwise wait and retry
                wait_time = self.retry_delay * (2 ** attempt)  # Exponential backoff
                logger.info(f"Retrying in {wait_time:.2f}s...")
                time.sleep(wait_time)
    
    def _fetch_markets(self, exchange_id: str, force_refresh: bool = False):
        """Fetch markets data for an exchange with caching"""
        current_time = time.time()
        
        with self.cache_lock:
            # Check if we need to refresh the cache
            last_fetch = self.last_market_fetch.get(exchange_id, 0)
            cache_age = current_time - last_fetch
            
            # Cache for 1 hour unless forced to refresh
            if exchange_id in self.markets_cache and cache_age < 3600 and not force_refresh:
                return self.markets_cache[exchange_id]
                
            # Fetch markets from exchange
            markets = self._execute_with_retry(exchange_id, 'fetch_markets')
            
            # Update cache
            self.markets_cache[exchange_id] = markets
            self.last_market_fetch[exchange_id] = current_time
            
            # Update symbols cache
            self.symbols_cache[exchange_id] = [market['symbol'] for market in markets]
            
            return markets
    
    def get_ticker(self, symbol: str, exchange_id: str = None):
        """
        Get current ticker data for a symbol.
        
        Args:
            symbol: Trading pair symbol (e.g., 'BTC/USDT')
            exchange_id: Exchange to use (default: primary exchange)
            
        Returns:
            Ticker data dictionary
        """
        exchange_id = exchange_id or self.primary_exchange
        
        # Validate symbol
        self._validate_symbol(symbol, exchange_id)
        
        # Fetch ticker
        ticker = self._execute_with_retry(exchange_id, 'fetch_ticker', symbol)
        
        return ticker
    
    def get_order_book(self, symbol: str, limit: int = 20, exchange_id: str = None):
        """
        Get order book for a symbol.
        
        Args:
            symbol: Trading pair symbol (e.g., 'BTC/USDT')
            limit: Number of orders to fetch on each side
            exchange_id: Exchange to use (default: primary exchange)
            
        Returns:
            Order book dictionary with 'bids' and 'asks'
        """
        exchange_id = exchange_id or self.primary_exchange
        
        # Validate symbol
        self._validate_symbol(symbol, exchange_id)
        
        # Fetch order book
        order_book = self._execute_with_retry(exchange_id, 'fetch_order_book', symbol, limit)
        
        return order_book
    
    def get_trades(self, symbol: str, since: int = None, limit: int = 100, exchange_id: str = None):
        """
        Get recent trades for a symbol.
        
        Args:
            symbol: Trading pair symbol (e.g., 'BTC/USDT')
            since: Timestamp in ms to fetch trades from
            limit: Maximum number of trades to fetch
            exchange_id: Exchange to use (default: primary exchange)
            
        Returns:
            List of trade dictionaries
        """
        exchange_id = exchange_id or self.primary_exchange
        
        # Validate symbol
        self._validate_symbol(symbol, exchange_id)
        
        # Fetch trades
        trades = self._execute_with_retry(exchange_id, 'fetch_trades', symbol, since, limit)
        
        return trades
    
    def get_ohlcv(self, symbol: str, timeframe: str = '1h', since: int = None, 
                 limit: int = 100, exchange_id: str = None) -> pd.DataFrame:
        """
        Get OHLCV candlestick data.
        
        Args:
            symbol: Trading pair symbol (e.g., 'BTC/USDT')
            timeframe: Candlestick timeframe (e.g., '1m', '1h', '1d')
            since: Timestamp in ms to fetch data from
            limit: Maximum number of candles to fetch
            exchange_id: Exchange to use (default: primary exchange)
            
        Returns:
            DataFrame with columns: timestamp, open, high, low, close, volume
        """
        exchange_id = exchange_id or self.primary_exchange
        
        # Validate symbol
        self._validate_symbol(symbol, exchange_id)
        
        # Validate timeframe
        self._validate_timeframe(timeframe, exchange_id)
        
        # Fetch OHLCV data
        ohlcv = self._execute_with_retry(exchange_id, 'fetch_ohlcv', symbol, timeframe, since, limit)
        
        # Convert to DataFrame
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)
        
        return df
    
    def get_balance(self, exchange_id: str = None):
        """
        Get account balance.
        
        Args:
            exchange_id: Exchange to use (default: primary exchange)
            
        Returns:
            Balance dictionary
        """
        exchange_id = exchange_id or self.primary_exchange
        
        # Fetch balance
        balance = self._execute_with_retry(exchange_id, 'fetch_balance')
        
        return balance
    
    def create_market_order(self, symbol: str, side: str, amount: float, params: Dict = None,
                          exchange_id: str = None):
        """
        Create a market order.
        
        Args:
            symbol: Trading pair symbol (e.g., 'BTC/USDT')
            side: Order side ('buy' or 'sell')
            amount: Order amount in base currency
            params: Additional parameters for the exchange
            exchange_id: Exchange to use (default: primary exchange)
            
        Returns:
            Order information dictionary
        """
        exchange_id = exchange_id or self.primary_exchange
        params = params or {}
        
        # Validate inputs
        self._validate_symbol(symbol, exchange_id)
        self._validate_side(side)
        self._validate_amount(amount, symbol, exchange_id)
        
        # Validate order if enabled
        if self.validate_orders:
            self._validate_market_order(symbol, side, amount, exchange_id)
        
        # Create order
        order = self._execute_with_retry(exchange_id, 'create_market_order', symbol, side, amount, params)
        
        # Track order
        self._track_order(order, exchange_id)
        
        return order
    
    def create_limit_order(self, symbol: str, side: str, amount: float, price: float, 
                         params: Dict = None, exchange_id: str = None):
        """
        Create a limit order.
        
        Args:
            symbol: Trading pair symbol (e.g., 'BTC/USDT')
            side: Order side ('buy' or 'sell')
            amount: Order amount in base currency
            price: Order price in quote currency
            params: Additional parameters for the exchange
            exchange_id: Exchange to use (default: primary exchange)
            
        Returns:
            Order information dictionary
        """
        exchange_id = exchange_id or self.primary_exchange
        params = params or {}
        
        # Validate inputs
        self._validate_symbol(symbol, exchange_id)
        self._validate_side(side)
        self._validate_amount(amount, symbol, exchange_id)
        self._validate_price(price, symbol, exchange_id)
        
        # Validate order if enabled
        if self.validate_orders:
            self._validate_limit_order(symbol, side, amount, price, exchange_id)
        
        # Create order
        order = self._execute_with_retry(exchange_id, 'create_limit_order', symbol, side, amount, price, params)
        
        # Track order
        self._track_order(order, exchange_id)
        
        return order
    
    def cancel_order(self, order_id: str, symbol: str, params: Dict = None, exchange_id: str = None):
        """
        Cancel an order.
        
        Args:
            order_id: Order ID
            symbol: Trading pair symbol (e.g., 'BTC/USDT')
            params: Additional parameters for the exchange
            exchange_id: Exchange to use (default: primary exchange)
            
        Returns:
            Cancellation result
        """
        # If order is tracked, use the exchange it was created on
        with self.orders_lock:
            if order_id in self.tracked_orders:
                exchange_id = self.tracked_orders[order_id].get('exchange_id', exchange_id)
        
        exchange_id = exchange_id or self.primary_exchange
        params = params or {}
        
        # Validate symbol
        self._validate_symbol(symbol, exchange_id)
        
        # Cancel order
        result = self._execute_with_retry(exchange_id, 'cancel_order', order_id, symbol, params)
        
        # Update tracked order
        with self.orders_lock:
            if order_id in self.tracked_orders:
                self.tracked_orders[order_id]['status'] = 'canceled'
        
        return result
    
    def fetch_order(self, order_id: str, symbol: str, params: Dict = None, exchange_id: str = None):
        """
        Fetch order information.
        
        Args:
            order_id: Order ID
            symbol: Trading pair symbol (e.g., 'BTC/USDT')
            params: Additional parameters for the exchange
            exchange_id: Exchange to use (default: primary exchange)
            
        Returns:
            Order information
        """
        # If order is tracked, use the exchange it was created on
        with self.orders_lock:
            if order_id in self.tracked_orders:
                exchange_id = self.tracked_orders[order_id].get('exchange_id', exchange_id)
        
        exchange_id = exchange_id or self.primary_exchange
        params = params or {}
        
        # Validate symbol
        self._validate_symbol(symbol, exchange_id)
        
        # Fetch order
        order = self._execute_with_retry(exchange_id, 'fetch_order', order_id, symbol, params)
        
        # Update tracked order
        self._update_tracked_order(order, exchange_id)
        
        return order
    
    def fetch_open_orders(self, symbol: str = None, since: int = None, limit: int = None, 
                         params: Dict = None, exchange_id: str = None):
        """
        Fetch open orders.
        
        Args:
            symbol: Trading pair symbol (e.g., 'BTC/USDT')
            since: Timestamp in ms to fetch orders from
            limit: Maximum number of orders to fetch
            params: Additional parameters for the exchange
            exchange_id: Exchange to use (default: primary exchange)
            
        Returns:
            List of open order information
        """
        exchange_id = exchange_id or self.primary_exchange
        params = params or {}
        
        # Validate symbol if provided
        if symbol:
            self._validate_symbol(symbol, exchange_id)
        
        # Fetch open orders
        orders = self._execute_with_retry(exchange_id, 'fetch_open_orders', symbol, since, limit, params)
        
        # Update tracked orders
        for order in orders:
            self._update_tracked_order(order, exchange_id)
        
        return orders
    
    def fetch_closed_orders(self, symbol: str = None, since: int = None, limit: int = None, 
                          params: Dict = None, exchange_id: str = None):
        """
        Fetch closed orders.
        
        Args:
            symbol: Trading pair symbol (e.g., 'BTC/USDT')
            since: Timestamp in ms to fetch orders from
            limit: Maximum number of orders to fetch
            params: Additional parameters for the exchange
            exchange_id: Exchange to use (default: primary exchange)
            
        Returns:
            List of closed order information
        """
        exchange_id = exchange_id or self.primary_exchange
        params = params or {}
        
        # Validate symbol if provided
        if symbol:
            self._validate_symbol(symbol, exchange_id)
        
        # Fetch closed orders
        orders = self._execute_with_retry(exchange_id, 'fetch_closed_orders', symbol, since, limit, params)
        
        # Update tracked orders
        for order in orders:
            self._update_tracked_order(order, exchange_id)
        
        return orders
    
    def _validate_symbol(self, symbol: str, exchange_id: str):
        """Validate that a symbol is supported by the exchange"""
        # Ensure markets are loaded
        self._fetch_markets(exchange_id)
        
        # Check if symbol is in cache
        if symbol not in self.symbols_cache.get(exchange_id, []):
            raise ValueError(f"Symbol {symbol} not supported by exchange {exchange_id}")
    
    def _validate_timeframe(self, timeframe: str, exchange_id: str):
        """Validate that a timeframe is supported by the exchange"""
        if exchange_id not in self.SUPPORTED_EXCHANGES:
            raise ValueError(f"Unsupported exchange: {exchange_id}")
            
        if timeframe not in self.SUPPORTED_EXCHANGES[exchange_id]['timeframes']:
            raise ValueError(f"Timeframe {timeframe} not supported by exchange {exchange_id}")
    
    def _validate_side(self, side: str):
        """Validate order side"""
        if side not in ['buy', 'sell']:
            raise ValueError(f"Invalid order side: {side}. Must be 'buy' or 'sell'")
    
    def _validate_amount(self, amount: float, symbol: str, exchange_id: str):
        """Validate order amount"""
        if amount <= 0:
            raise ValueError(f"Invalid order amount: {amount}. Must be greater than 0")
            
        # Get market limits
        self._fetch_markets(exchange_id)
        
        markets = self.markets_cache.get(exchange_id, [])
        market = next((m for m in markets if m['symbol'] == symbol), None)
        
        if market:
            limits = market.get('limits', {})
            amount_limits = limits.get('amount', {})
            
            min_amount = amount_limits.get('min')
            max_amount = amount_limits.get('max')
            
            if min_amount and amount < min_amount:
                raise ValueError(f"Order amount {amount} is less than minimum {min_amount} for {symbol}")
                
            if max_amount and amount > max_amount:
                raise ValueError(f"Order amount {amount} is greater than maximum {max_amount} for {symbol}")
    
    def _validate_price(self, price: float, symbol: str, exchange_id: str):
        """Validate order price"""
        if price <= 0:
            raise ValueError(f"Invalid order price: {price}. Must be greater than 0")
            
        # Get market limits
        self._fetch_markets(exchange_id)
        
        markets = self.markets_cache.get(exchange_id, [])
        market = next((m for m in markets if m['symbol'] == symbol), None)
        
        if market:
            limits = market.get('limits', {})
            price_limits = limits.get('price', {})
            
            min_price = price_limits.get('min')
            max_price = price_limits.get('max')
            
            if min_price and price < min_price:
                raise ValueError(f"Order price {price} is less than minimum {min_price} for {symbol}")
                
            if max_price and price > max_price:
                raise ValueError(f"Order price {price} is greater than maximum {max_price} for {symbol}")
    
    def _validate_market_order(self, symbol: str, side: str, amount: float, exchange_id: str):
        """
        Validate market order parameters.
        
        This performs additional validation beyond basic parameter checks,
        such as checking available balance.
        """
        # Get account balance
        balance = self.get_balance(exchange_id)
        
        # Get market info
        self._fetch_markets(exchange_id)
        markets = self.markets_cache.get(exchange_id, [])
        market = next((m for m in markets if m['symbol'] == symbol), None)
        
        if not market:
            return  # Skip validation if market info not available
            
        # Extract base and quote currencies
        base_currency = market['base']
        quote_currency = market['quote']
        
        if side == 'buy':
            # For buy orders, check if we have enough quote currency
            ticker = self.get_ticker(symbol, exchange_id)
            estimated_cost = amount * ticker['ask'] * 1.01  # Add 1% buffer for price movement
            
            available_quote = balance.get(quote_currency, {}).get('free', 0)
            if available_quote < estimated_cost:
                raise ValueError(f"Insufficient {quote_currency} balance for buy order. Need {estimated_cost}, have {available_quote}")
        else:
            # For sell orders, check if we have enough base currency
            available_base = balance.get(base_currency, {}).get('free', 0)
            if available_base < amount:
                raise ValueError(f"Insufficient {base_currency} balance for sell order. Need {amount}, have {available_base}")
    
    def _validate_limit_order(self, symbol: str, side: str, amount: float, price: float, exchange_id: str):
        """
        Validate limit order parameters.
        
        This performs additional validation beyond basic parameter checks,
        such as checking available balance.
        """
        # Get account balance
        balance = self.get_balance(exchange_id)
        
        # Get market info
        self._fetch_markets(exchange_id)
        markets = self.markets_cache.get(exchange_id, [])
        market = next((m for m in markets if m['symbol'] == symbol), None)
        
        if not market:
            return  # Skip validation if market info not available
            
        # Extract base and quote currencies
        base_currency = market['base']
        quote_currency = market['quote']
        
        if side == 'buy':
            # For buy orders, check if we have enough quote currency
            cost = amount * price
            
            available_quote = balance.get(quote_currency, {}).get('free', 0)
            if available_quote < cost:
                raise ValueError(f"Insufficient {quote_currency} balance for buy order. Need {cost}, have {available_quote}")
        else:
            # For sell orders, check if we have enough base currency
            available_base = balance.get(base_currency, {}).get('free', 0)
            if available_base < amount:
                raise ValueError(f"Insufficient {base_currency} balance for sell order. Need {amount}, have {available_base}")
    
    def _track_order(self, order: Dict, exchange_id: str):
        """Track an order for position updates"""
        if not order or 'id' not in order:
            return
            
        with self.orders_lock:
            # Add exchange_id to order for tracking purposes
            order['exchange_id'] = exchange_id
            
            # Store order in tracking map
            self.tracked_orders[order['id']] = order
            
            # Update positions if filled
            if order.get('status') == 'closed':
                self._update_position_from_order(order)
    
    def _update_tracked_order(self, order: Dict, exchange_id: str):
        """Update a tracked order with new information"""
        if not order or 'id' not in order:
            return
            
        with self.orders_lock:
            # Add exchange_id to order for tracking purposes
            order['exchange_id'] = exchange_id
            
            # Update or add order in tracking map
            self.tracked_orders[order['id']] = order
            
            # Update positions if filled
            if order.get('status') == 'closed':
                self._update_position_from_order(order)
    
    def _update_position_from_order(self, order: Dict):
        """Update position tracking based on order information"""
        symbol = order.get('symbol')
        if not symbol:
            return
            
        with self.positions_lock:
            # Initialize position if not exists
            if symbol not in self.positions:
                self.positions[symbol] = {
                    'amount': 0.0,
                    'cost': 0.0,
                    'side': None,
                    'last_update': time.time()
                }
                
            position = self.positions[symbol]
            
            # Update position based on order
            side = order.get('side')
            amount = order.get('amount', 0)
            cost = order.get('cost', 0)
            
            if side == 'buy':
                position['amount'] += amount
                position['cost'] += cost
            elif side == 'sell':
                position['amount'] -= amount
                position['cost'] -= cost
                
            # Update position side
            if position['amount'] > 0:
                position['side'] = 'long'
            elif position['amount'] < 0:
                position['side'] = 'short'
            else:
                position['side'] = None
                
            # Update timestamp
            position['last_update'] = time.time()
    
    def get_position(self, symbol: str):
        """Get current position for a symbol"""
        with self.positions_lock:
            return self.positions.get(symbol, {
                'amount': 0.0,
                'cost': 0.0,
                'side': None,
                'last_update': None
            })
    
    def get_all_positions(self):
        """Get all current positions"""
        with self.positions_lock:
            return self.positions.copy()
    
    def get_tracked_orders(self, symbol: str = None, status: str = None):
        """
        Get tracked orders with optional filtering.
        
        Args:
            symbol: Filter by symbol
            status: Filter by order status
            
        Returns:
            Dictionary of tracked orders
        """
        with self.orders_lock:
            filtered_orders = self.tracked_orders.copy()
            
            # Filter by symbol
            if symbol:
                filtered_orders = {
                    order_id: order for order_id, order in filtered_orders.items()
                    if order.get('symbol') == symbol
                }
                
            # Filter by status
            if status:
                filtered_orders = {
                    order_id: order for order_id, order in filtered_orders.items()
                    if order.get('status') == status
                }
                
            return filtered_orders
    
    def get_exchange_fee(self, symbol: str, order_type: str = 'limit', side: str = 'buy', amount: float = 1.0, 
                       price: float = None, exchange_id: str = None):
        """
        Calculate exchange fee for an order.
        
        Args:
            symbol: Trading pair symbol
            order_type: Order type ('limit' or 'market')
            side: Order side ('buy' or 'sell')
            amount: Order amount
            price: Order price (for limit orders)
            exchange_id: Exchange to use
            
        Returns:
            Dictionary with fee information
        """
        exchange_id = exchange_id or self.primary_exchange
        
        # Validate inputs
        self._validate_symbol(symbol, exchange_id)
        
        # Try to get fee from exchange
        try:
            with self.connection_pools[exchange_id].get_connection() as connection:
                fee = connection.calculate_fee(symbol, order_type, side, amount, price)
                return fee
        except Exception as e:
            # Fall back to pre-configured fees if calculation fails
            if exchange_id in self.SUPPORTED_EXCHANGES:
                config = self.SUPPORTED_EXCHANGES[exchange_id]
                
                # Determine if maker or taker fee applies
                fee_type = 'maker' if order_type == 'limit' else 'taker'
                fee_rate = config.get(f'{fee_type}_fee', 0.001)
                
                # Calculate cost
                cost = amount
                if price:
                    cost = amount * price
                    
                # Calculate fee
                fee_amount = cost * fee_rate
                
                # Get market info to determine fee currency
                self._fetch_markets(exchange_id)
                markets = self.markets_cache.get(exchange_id, [])
                market = next((m for m in markets if m['symbol'] == symbol), None)
                
                fee_currency = market['quote'] if market else None
                
                return {
                    'cost': fee_amount,
                    'currency': fee_currency,
                    'rate': fee_rate,
                    'type': fee_type
                }
                
            # If all else fails, return a default fee
            return {
                'cost': amount * 0.001,
                'currency': None,
                'rate': 0.001,
                'type': 'taker'
            }
    
    def get_supported_timeframes(self, exchange_id: str = None):
        """Get supported timeframes for an exchange"""
        exchange_id = exchange_id or self.primary_exchange
        
        if exchange_id in self.SUPPORTED_EXCHANGES:
            return self.SUPPORTED_EXCHANGES[exchange_id]['timeframes']
            
        return []
    
    def get_supported_exchanges(self):
        """Get list of supported exchanges"""
        return list(self.SUPPORTED_EXCHANGES.keys())
    
    def is_exchange_supported(self, exchange_id: str):
        """Check if an exchange is supported"""
        return exchange_id in self.SUPPORTED_EXCHANGES
    
    def synchronize_positions(self, exchange_id: str = None):
        """
        Synchronize position tracking with exchange balances.
        
        This ensures position tracking is accurate by comparing with
        actual exchange balances.
        
        Args:
            exchange_id: Exchange to use (default: primary exchange)
            
        Returns:
            Dictionary of updated positions
        """
        exchange_id = exchange_id or self.primary_exchange
        
        # Get account balance
        balance = self.get_balance(exchange_id)
        
        # Get markets info
        self._fetch_markets(exchange_id)
        markets = self.markets_cache.get(exchange_id, [])
        
        with self.positions_lock:
            # For each market, check if we have a balance
            for market in markets:
                symbol = market['symbol']
                base = market['base']
                quote = market['quote']
                
                # Check if we have a balance in the base currency
                base_balance = balance.get(base, {}).get('total', 0)
                if base_balance > 0:
                    # Get or create position
                    if symbol not in self.positions:
                        self.positions[symbol] = {
                            'amount': 0.0,
                            'cost': 0.0,
                            'side': None,
                            'last_update': time.time()
                        }
                        
                    # Update position amount
                    self.positions[symbol]['amount'] = base_balance
                    
                    # Estimate cost based on current price
                    ticker = self.get_ticker(symbol, exchange_id)
                    if ticker:
                        self.positions[symbol]['cost'] = base_balance * ticker['close']
                        
                    # Update side
                    self.positions[symbol]['side'] = 'long' if base_balance > 0 else None
                    
                    # Update timestamp
                    self.positions[symbol]['last_update'] = time.time()
            
            return self.positions.copy()
    
    def get_market_info(self, symbol: str, exchange_id: str = None):
        """
        Get detailed market information for a symbol.
        
        Args:
            symbol: Trading pair symbol
            exchange_id: Exchange to use
            
        Returns:
            Dictionary with market information
        """
        exchange_id = exchange_id or self.primary_exchange
        
        # Validate symbol
        self._validate_symbol(symbol, exchange_id)
        
        # Get markets info
        self._fetch_markets(exchange_id)
        markets = self.markets_cache.get(exchange_id, [])
        
        # Find market for symbol
        market = next((m for m in markets if m['symbol'] == symbol), None)
        
        return market
    
    def get_exchange_status(self, exchange_id: str = None):
        """
        Check if an exchange is operational.
        
        Args:
            exchange_id: Exchange to check
            
        Returns:
            Dictionary with status information
        """
        exchange_id = exchange_id or self.primary_exchange
        
        try:
            # Try to fetch time from exchange
            with self.connection_pools[exchange_id].get_connection() as connection:
                time_response = connection.fetch_time()
                
                # Calculate time offset
                local_time = int(time.time() * 1000)
                time_offset = time_response - local_time
                
                return {
                    'status': 'ok',
                    'time_offset': time_offset,
                    'connected': True
                }
        except Exception as e:
            return {
                'status': 'error',
                'message': str(e),
                'connected': False
            }
    
    def reset_connection(self, exchange_id: str = None):
        """
        Reset exchange connection.
        
        This can be useful if there are connection issues.
        
        Args:
            exchange_id: Exchange to reset
            
        Returns:
            True if reset successful, False otherwise
        """
        exchange_id = exchange_id or self.primary_exchange
        
        try:
            # Replace connection pool
            if exchange_id in self.api_credentials:
                creds = self.api_credentials[exchange_id]
                
                # Create new pool
                max_connections = self.connection_pools[exchange_id].max_connections
                
                # Replace pool
                self.connection_pools[exchange_id] = ExchangeConnectionPool(
                    exchange_id,
                    creds['api_key'],
                    creds['api_secret'],
                    max_connections
                )
                
                # Reset markets cache
                with self.cache_lock:
                    if exchange_id in self.markets_cache:
                        del self.markets_cache[exchange_id]
                    if exchange_id in self.symbols_cache:
                        del self.symbols_cache[exchange_id]
                    if exchange_id in self.last_market_fetch:
                        del self.last_market_fetch[exchange_id]
                        
                # Reload markets
                self._fetch_markets(exchange_id, force_refresh=True)
                
                return True
        except Exception as e:
            logger.error(f"Error resetting connection for {exchange_id}: {str(e)}")
            
        return False
        
    def close(self):
        """Close all connections and cleanup"""
        logger.info("Closing exchange connector...")
        
        # Nothing specific needed for cleanup
        logger.info("Exchange connector closed")

# Export a factory function for creating exchange connectors
def create_exchange_connector(**kwargs):
    """Create and initialize an exchange connector"""
    return ExchangeConnector(**kwargs)

# If running directly, test functionality
if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # Create connector
    connector = ExchangeConnector()
    
    # Test market data retrieval
    symbol = "BTC/USDT"
    print(f"Testing market data retrieval for {symbol}...")
    
    try:
        # Get ticker
        ticker = connector.get_ticker(symbol)
        print(f"Ticker: {ticker['last']}")
        
        # Get OHLCV
        ohlcv = connector.get_ohlcv(symbol, timeframe='1h', limit=5)
        print(f"Recent OHLCV data:\n{ohlcv}")
        
        # Get order book
        order_book = connector.get_order_book(symbol, limit=5)
        print(f"Order book: {len(order_book['bids'])} bids, {len(order_book['asks'])} asks")
        
        print("Tests completed successfully")
    except Exception as e:
        print(f"Error during testing: {str(e)}")
