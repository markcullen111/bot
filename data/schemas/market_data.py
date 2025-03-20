# market_data.py

import os
import time
import json
import logging
import threading
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union, Any, Callable
from datetime import datetime, timedelta
from functools import lru_cache
import asyncio
import aiohttp
import ccxt
import ccxt.async_support as ccxt_async
from threading import RLock
from dataclasses import dataclass, field

# Try to import error handling
try:
    from error_handling import safe_execute, ErrorCategory, ErrorSeverity, ErrorHandler, TradingSystemError
    HAVE_ERROR_HANDLING = True
except ImportError:
    HAVE_ERROR_HANDLING = False
    logging.warning("Error handling module not available. Using basic error handling.")

# Define constants
DEFAULT_TIMEFRAMES = {
    '1m': 60,
    '3m': 180,
    '5m': 300,
    '15m': 900,
    '30m': 1800,
    '1h': 3600,
    '2h': 7200,
    '4h': 14400,
    '6h': 21600,
    '8h': 28800,
    '12h': 43200,
    '1d': 86400,
    '3d': 259200,
    '1w': 604800,
    '1M': 2592000
}

EXCHANGE_RATE_LIMITS = {
    'binance': {'calls_per_second': 10, 'calls_per_minute': 1200},
    'coinbase': {'calls_per_second': 3, 'calls_per_minute': 100},
    'kraken': {'calls_per_second': 1, 'calls_per_minute': 60},
    'bybit': {'calls_per_second': 5, 'calls_per_minute': 300},
    'kucoin': {'calls_per_second': 5, 'calls_per_minute': 180},
    'default': {'calls_per_second': 1, 'calls_per_minute': 60}
}

DEFAULT_EXCHANGE = 'binance'
DEFAULT_CACHE_EXPIRY = 60  # 60 seconds
DATA_BATCH_SIZE = 1000     # Number of candles per API request

# Technical indicator parameters
DEFAULT_INDICATOR_PARAMS = {
    'sma_periods': [20, 50, 200],
    'ema_periods': [9, 21, 55, 200],
    'rsi_period': 14,
    'macd_params': {'fast': 12, 'slow': 26, 'signal': 9},
    'bbands_params': {'period': 20, 'std_dev': 2.0},
    'atr_period': 14
}

@dataclass
class CacheEntry:
    """Dataclass for cache entries with expiry handling"""
    data: pd.DataFrame
    timestamp: float = field(default_factory=time.time)
    expiry: float = DEFAULT_CACHE_EXPIRY
    
    def is_expired(self) -> bool:
        """Check if the cache entry has expired"""
        return (time.time() - self.timestamp) > self.expiry
        
    def update(self, data: pd.DataFrame) -> None:
        """Update cache entry with new data"""
        self.data = data
        self.timestamp = time.time()

class RateLimiter:
    """Rate limiter for API calls to prevent hitting exchange limits"""
    
    def __init__(self, calls_per_second: int = 1, calls_per_minute: int = 60):
        self.calls_per_second = calls_per_second
        self.calls_per_minute = calls_per_minute
        self.second_bucket = []
        self.minute_bucket = []
        self.lock = threading.RLock()
        
    async def acquire(self):
        """Acquire permission to make an API call, waiting if necessary"""
        while True:
            with self.lock:
                # Clean expired timestamps
                current_time = time.time()
                self.second_bucket = [t for t in self.second_bucket if current_time - t < 1.0]
                self.minute_bucket = [t for t in self.minute_bucket if current_time - t < 60.0]
                
                # Check if we can make a request
                if (len(self.second_bucket) < self.calls_per_second and 
                    len(self.minute_bucket) < self.calls_per_minute):
                    # Add timestamps
                    self.second_bucket.append(current_time)
                    self.minute_bucket.append(current_time)
                    return
            
            # Wait before checking again
            await asyncio.sleep(0.1)
            
    def sync_acquire(self):
        """Synchronous version of acquire for non-async code"""
        while True:
            with self.lock:
                # Clean expired timestamps
                current_time = time.time()
                self.second_bucket = [t for t in self.second_bucket if current_time - t < 1.0]
                self.minute_bucket = [t for t in self.minute_bucket if current_time - t < 60.0]
                
                # Check if we can make a request
                if (len(self.second_bucket) < self.calls_per_second and 
                    len(self.minute_bucket) < self.calls_per_minute):
                    # Add timestamps
                    self.second_bucket.append(current_time)
                    self.minute_bucket.append(current_time)
                    return
            
            # Wait before checking again
            time.sleep(0.1)

class MarketDataManager:
    """
    Comprehensive market data manager that efficiently fetches, processes, and caches 
    market data from multiple exchanges with robust error handling and optimization.
    
    Features:
    - Multi-exchange support with automatic fallback
    - Efficient caching with size limits and LRU eviction
    - Technical indicator calculation
    - Real-time and historical data
    - Order book depth analysis
    - Asynchronous batch fetching
    - Thread-safe operations
    - Rate limit management
    - Database integration
    - Robust error handling and recovery
    """
    
    def __init__(
        self, 
        db_manager=None, 
        cache_size: int = 100, 
        default_exchange: str = DEFAULT_EXCHANGE,
        enable_live_updates: bool = False,
        max_retries: int = 3,
        cache_expiry: float = DEFAULT_CACHE_EXPIRY,
        indicator_params: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize the market data manager.
        
        Args:
            db_manager: Database manager for persistent storage
            cache_size: Maximum number of symbols to cache in memory
            default_exchange: Default exchange to use for data fetching
            enable_live_updates: Whether to enable real-time updates
            max_retries: Maximum number of retries for failed API calls
            cache_expiry: Default cache expiry time in seconds
            indicator_params: Custom parameters for technical indicators
        """
        self.db_manager = db_manager
        self.cache_size = cache_size
        self.default_exchange = default_exchange
        self.enable_live_updates = enable_live_updates
        self.max_retries = max_retries
        self.default_cache_expiry = cache_expiry
        self.indicator_params = indicator_params or DEFAULT_INDICATOR_PARAMS
        
        # Initialize caches with thread safety
        self._kline_cache = {}  # Symbol, timeframe -> DataFrame cache
        self._orderbook_cache = {}  # Symbol -> OrderBook cache
        self._ticker_cache = {}  # Symbol -> Ticker cache
        self._cache_locks = {}  # Symbol -> Lock for thread safety
        self._global_lock = RLock()  # Global lock for cache operations
        
        # Initialize exchanges
        self._exchanges = {}  # Exchange name -> ccxt instance
        self._async_exchanges = {}  # Exchange name -> ccxt async instance
        self._exchange_locks = {}  # Exchange name -> Lock
        
        # Initialize rate limiters
        self._rate_limiters = {}  # Exchange name -> RateLimiter
        
        # Set up performance metrics
        self._api_call_count = 0
        self._cache_hit_count = 0
        self._cache_miss_count = 0
        self._api_error_count = 0
        self._performance_metrics = {
            'api_calls': 0,
            'cache_hits': 0, 
            'cache_misses': 0,
            'api_errors': 0,
            'avg_response_time': 0,
            'data_points_processed': 0
        }
        
        # Live update handling
        self._live_update_tasks = {}  # Symbol -> Task
        self._live_update_callbacks = {}  # Symbol -> List[Callback]
        self._shutdown_flag = False
        
        # Initialize default exchange
        self._init_default_exchange()
        
        logging.info(f"Market Data Manager initialized with {cache_size} cache slots")
    
    def _init_default_exchange(self):
        """Initialize the default exchange instance"""
        try:
            # Create exchange instance
            exchange_class = getattr(ccxt, self.default_exchange)
            exchange = exchange_class({
                'enableRateLimit': True,
                'timeout': 30000,  # 30 seconds timeout
            })
            
            # Set up rate limiter
            rate_limits = EXCHANGE_RATE_LIMITS.get(
                self.default_exchange, 
                EXCHANGE_RATE_LIMITS['default']
            )
            
            rate_limiter = RateLimiter(
                calls_per_second=rate_limits['calls_per_second'],
                calls_per_minute=rate_limits['calls_per_minute']
            )
            
            # Store instances
            with self._global_lock:
                self._exchanges[self.default_exchange] = exchange
                self._rate_limiters[self.default_exchange] = rate_limiter
                self._exchange_locks[self.default_exchange] = RLock()
            
            logging.info(f"Initialized default exchange: {self.default_exchange}")
            
        except Exception as e:
            logging.error(f"Error initializing default exchange: {e}")
            if HAVE_ERROR_HANDLING:
                ErrorHandler.handle_error(
                    error=e,
                    category=ErrorCategory.NETWORK,
                    severity=ErrorSeverity.WARNING,
                    context={"exchange": self.default_exchange}
                )
    
    async def _init_async_exchange(self, exchange_id):
        """Initialize an async exchange instance"""
        try:
            # Create async exchange instance
            exchange_class = getattr(ccxt_async, exchange_id)
            exchange = exchange_class({
                'enableRateLimit': True,
                'timeout': 30000,  # 30 seconds timeout
            })
            
            # Store instance
            with self._global_lock:
                self._async_exchanges[exchange_id] = exchange
            
            return exchange
            
        except Exception as e:
            logging.error(f"Error initializing async exchange {exchange_id}: {e}")
            if HAVE_ERROR_HANDLING:
                ErrorHandler.handle_error(
                    error=e,
                    category=ErrorCategory.NETWORK,
                    severity=ErrorSeverity.WARNING,
                    context={"exchange": exchange_id}
                )
            return None
    
    def _get_cache_lock(self, symbol):
        """Get or create a lock for a specific symbol"""
        with self._global_lock:
            if symbol not in self._cache_locks:
                self._cache_locks[symbol] = RLock()
            return self._cache_locks[symbol]
    
    def _get_exchange_lock(self, exchange_id):
        """Get or create a lock for a specific exchange"""
        with self._global_lock:
            if exchange_id not in self._exchange_locks:
                self._exchange_locks[exchange_id] = RLock()
            return self._exchange_locks[exchange_id]
    
    def _get_rate_limiter(self, exchange_id):
        """Get or create a rate limiter for a specific exchange"""
        with self._global_lock:
            if exchange_id not in self._rate_limiters:
                rate_limits = EXCHANGE_RATE_LIMITS.get(
                    exchange_id, 
                    EXCHANGE_RATE_LIMITS['default']
                )
                
                self._rate_limiters[exchange_id] = RateLimiter(
                    calls_per_second=rate_limits['calls_per_second'],
                    calls_per_minute=rate_limits['calls_per_minute']
                )
            
            return self._rate_limiters[exchange_id]
    
    def _update_performance_metrics(self, metric, value=1):
        """Update performance metrics"""
        with self._global_lock:
            if metric in self._performance_metrics:
                self._performance_metrics[metric] += value
    
    def get_performance_metrics(self):
        """Get current performance metrics"""
        with self._global_lock:
            return self._performance_metrics.copy()
    
    def _prune_cache_if_needed(self):
        """Prune cache if it exceeds the size limit"""
        with self._global_lock:
            # Check if we need to prune kline cache
            if len(self._kline_cache) > self.cache_size:
                # Sort cache entries by last access time
                sorted_entries = sorted(
                    self._kline_cache.items(),
                    key=lambda x: x[1].timestamp
                )
                
                # Remove oldest entries
                for key, _ in sorted_entries[:len(sorted_entries) - self.cache_size]:
                    del self._kline_cache[key]
                
                logging.debug(f"Pruned market data cache to {self.cache_size} entries")
    
    def _get_key(self, symbol, timeframe):
        """Generate cache key from symbol and timeframe"""
        return f"{symbol}_{timeframe}"
    
    def _get_from_cache(self, symbol, timeframe):
        """Get data from cache if available and not expired"""
        key = self._get_key(symbol, timeframe)
        
        with self._global_lock:
            if key in self._kline_cache:
                cache_entry = self._kline_cache[key]
                
                if not cache_entry.is_expired():
                    self._update_performance_metrics('cache_hits')
                    return cache_entry.data.copy()
                
                # Entry expired
                self._update_performance_metrics('cache_misses')
        
        return None
    
    def _store_in_cache(self, symbol, timeframe, data, expiry=None):
        """Store data in cache with expiry"""
        key = self._get_key(symbol, timeframe)
        expiry = expiry or self.default_cache_expiry
        
        with self._global_lock:
            if key in self._kline_cache:
                # Update existing entry
                self._kline_cache[key].update(data)
                self._kline_cache[key].expiry = expiry
            else:
                # Create new entry
                self._kline_cache[key] = CacheEntry(data=data.copy(), expiry=expiry)
            
            # Prune cache if needed
            self._prune_cache_if_needed()
    
    def _store_in_database(self, symbol, timeframe, data):
        """Store market data in database"""
        if self.db_manager is None:
            return False
        
        try:
            # Make a defensive copy and add timeframe column
            df = data.copy()
            df['timeframe'] = timeframe
            
            # Store in database
            self.db_manager.store_market_data(df, symbol)
            return True
            
        except Exception as e:
            logging.error(f"Error storing market data in database: {e}")
            if HAVE_ERROR_HANDLING:
                ErrorHandler.handle_error(
                    error=e,
                    category=ErrorCategory.DATABASE,
                    severity=ErrorSeverity.WARNING,
                    context={
                        "symbol": symbol,
                        "timeframe": timeframe,
                        "operation": "store"
                    }
                )
            return False
    
    def _load_from_database(self, symbol, timeframe, limit=None, start_time=None, end_time=None):
        """Load market data from database"""
        if self.db_manager is None:
            return None
        
        try:
            # Load from database
            df = self.db_manager.get_market_data(
                symbol, 
                timeframe=timeframe,
                limit=limit,
                start_time=start_time,
                end_time=end_time
            )
            
            if df is not None and not df.empty:
                return df
                
            return None
            
        except Exception as e:
            logging.error(f"Error loading market data from database: {e}")
            if HAVE_ERROR_HANDLING:
                ErrorHandler.handle_error(
                    error=e,
                    category=ErrorCategory.DATABASE,
                    severity=ErrorSeverity.WARNING,
                    context={
                        "symbol": symbol,
                        "timeframe": timeframe,
                        "operation": "load"
                    }
                )
            return None
    
    def _calculate_indicators(self, df):
        """Calculate technical indicators for DataFrame"""
        if df is None or df.empty:
            return df
        
        # Make a copy to avoid modifying original data
        data = df.copy()
        
        try:
            # SMA calculations
            for period in self.indicator_params['sma_periods']:
                data[f'sma_{period}'] = data['close'].rolling(window=period).mean()
            
            # EMA calculations
            for period in self.indicator_params['ema_periods']:
                data[f'ema_{period}'] = data['close'].ewm(span=period, adjust=False).mean()
            
            # RSI calculation
            rsi_period = self.indicator_params['rsi_period']
            delta = data['close'].diff()
            gain = delta.where(delta > 0, 0).rolling(window=rsi_period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=rsi_period).mean()
            
            # Avoid division by zero
            rs = gain / loss.replace(0, np.finfo(float).eps)
            data['rsi'] = 100 - (100 / (1 + rs))
            
            # MACD calculation
            macd_params = self.indicator_params['macd_params']
            data['ema_short'] = data['close'].ewm(span=macd_params['fast'], adjust=False).mean()
            data['ema_long'] = data['close'].ewm(span=macd_params['slow'], adjust=False).mean()
            data['macd'] = data['ema_short'] - data['ema_long']
            data['macd_signal'] = data['macd'].ewm(span=macd_params['signal'], adjust=False).mean()
            data['macd_hist'] = data['macd'] - data['macd_signal']
            
            # Bollinger Bands
            bbands_params = self.indicator_params['bbands_params']
            period = bbands_params['period']
            std_dev = bbands_params['std_dev']
            
            data['bb_middle'] = data['close'].rolling(window=period).mean()
            data['bb_std'] = data['close'].rolling(window=period).std()
            data['bb_upper'] = data['bb_middle'] + (data['bb_std'] * std_dev)
            data['bb_lower'] = data['bb_middle'] - (data['bb_std'] * std_dev)
            
            # ATR calculation
            atr_period = self.indicator_params['atr_period']
            high_low = data['high'] - data['low']
            high_close = abs(data['high'] - data['close'].shift())
            low_close = abs(data['low'] - data['close'].shift())
            
            tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
            data['atr'] = tr.rolling(window=atr_period).mean()
            
            # Update performance metrics
            self._update_performance_metrics(
                'data_points_processed', 
                len(data) * len(self.indicator_params['sma_periods'] + self.indicator_params['ema_periods'] + 5)
            )
            
            return data
            
        except Exception as e:
            logging.error(f"Error calculating indicators: {e}")
            if HAVE_ERROR_HANDLING:
                ErrorHandler.handle_error(
                    error=e,
                    category=ErrorCategory.DATA_PROCESSING,
                    severity=ErrorSeverity.WARNING,
                    context={"operation": "calculate_indicators"}
                )
            return df  # Return original data on error
    
    def _process_klines(self, klines):
        """Process raw klines data into a DataFrame"""
        if not klines:
            return pd.DataFrame()
        
        # Create DataFrame
        df = pd.DataFrame(klines, columns=[
            'timestamp', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_volume', 'trades', 'taker_buy_base',
            'taker_buy_quote', 'ignore'
        ])
        
        # Convert types
        numeric_columns = ['open', 'high', 'low', 'close', 'volume', 'quote_volume']
        for col in numeric_columns:
            df[col] = pd.to_numeric(df[col])
        
        # Convert timestamps to datetime
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        
        # Set timestamp as index
        df.set_index('timestamp', inplace=True)
        
        # Sort by timestamp
        df.sort_index(inplace=True)
        
        return df
    
    async def _fetch_klines_async(self, exchange_id, symbol, timeframe, since=None, limit=None):
        """Fetch klines asynchronously with rate limiting and error handling"""
        exchange = None
        
        try:
            # Get or initialize async exchange
            if exchange_id in self._async_exchanges:
                exchange = self._async_exchanges[exchange_id]
            else:
                exchange = await self._init_async_exchange(exchange_id)
                
            if exchange is None:
                raise Exception(f"Failed to initialize async exchange {exchange_id}")
            
            # Get rate limiter
            rate_limiter = self._get_rate_limiter(exchange_id)
            
            # Wait for rate limit
            await rate_limiter.acquire()
            
            # Update metrics
            self._update_performance_metrics('api_calls')
            
            # Track call time for performance measurement
            start_time = time.time()
            
            # Fetch klines
            klines = await exchange.fetch_ohlcv(
                symbol=symbol,
                timeframe=timeframe,
                since=since,
                limit=limit
            )
            
            # Update response time metric
            elapsed = time.time() - start_time
            with self._global_lock:
                old_avg = self._performance_metrics['avg_response_time']
                old_count = self._performance_metrics['api_calls']
                new_avg = ((old_avg * (old_count - 1)) + elapsed) / old_count
                self._performance_metrics['avg_response_time'] = new_avg
            
            return klines
            
        except Exception as e:
            self._update_performance_metrics('api_errors')
            
            # Log error
            logging.error(f"Error fetching klines from {exchange_id} for {symbol} {timeframe}: {e}")
            
            if HAVE_ERROR_HANDLING:
                ErrorHandler.handle_error(
                    error=e,
                    category=ErrorCategory.NETWORK,
                    severity=ErrorSeverity.WARNING,
                    context={
                        "exchange": exchange_id,
                        "symbol": symbol,
                        "timeframe": timeframe,
                        "operation": "fetch_klines_async"
                    }
                )
            
            return []
        finally:
            # Clean up exchange connection
            if exchange is not None and exchange_id not in self._async_exchanges:
                try:
                    await exchange.close()
                except:
                    pass
    
    async def _fetch_historical_klines_async(self, exchange_id, symbol, timeframe, start_time, end_time, batch_size=DATA_BATCH_SIZE):
        """Fetch historical klines in batches to handle large date ranges"""
        all_klines = []
        
        try:
            # Calculate batch parameters
            tf_seconds = DEFAULT_TIMEFRAMES.get(timeframe, 3600)  # Default to 1h
            batch_timespan = batch_size * tf_seconds * 1000  # in milliseconds
            
            # Convert start_time and end_time to milliseconds timestamp
            if isinstance(start_time, (datetime, pd.Timestamp)):
                start_ms = int(start_time.timestamp() * 1000)
            else:
                start_ms = start_time
                
            if isinstance(end_time, (datetime, pd.Timestamp)):
                end_ms = int(end_time.timestamp() * 1000)
            else:
                end_ms = end_time
                
            # Fetch data in batches
            current_start = start_ms
            
            while current_start < end_ms:
                # Calculate batch end (capped at end_time)
                batch_end = min(current_start + batch_timespan, end_ms)
                
                # Fetch batch
                batch = await self._fetch_klines_async(
                    exchange_id=exchange_id,
                    symbol=symbol,
                    timeframe=timeframe,
                    since=current_start,
                    limit=batch_size
                )
                
                if batch:
                    all_klines.extend(batch)
                    
                    # Update start time for next batch
                    # Use the timestamp of the last candle + one timeframe
                    last_timestamp = batch[-1][0]
                    current_start = last_timestamp + tf_seconds * 1000
                    
                    # Add a small delay to prevent hitting rate limits
                    await asyncio.sleep(0.1)
                else:
                    # No data returned, move to next batch to avoid infinite loop
                    current_start = current_start + batch_timespan
                
                # Check if we've collected all data we need
                if len(all_klines) >= DATA_BATCH_SIZE * 10:  # Arbitrary large but reasonable limit
                    logging.warning(f"Large data fetch detected for {symbol}, truncating")
                    break
            
            return all_klines
            
        except Exception as e:
            logging.error(f"Error fetching historical klines: {e}")
            if HAVE_ERROR_HANDLING:
                ErrorHandler.handle_error(
                    error=e,
                    category=ErrorCategory.NETWORK,
                    severity=ErrorSeverity.WARNING,
                    context={
                        "exchange": exchange_id,
                        "symbol": symbol,
                        "timeframe": timeframe,
                        "operation": "fetch_historical_klines_async"
                    }
                )
            return all_klines  # Return whatever we've collected so far
    
    def fetch_market_data(
        self,
        symbol: str,
        timeframe: str = '1h',
        limit: int = None,
        start_time: Optional[Union[datetime, int]] = None,
        end_time: Optional[Union[datetime, int]] = None,
        include_indicators: bool = True,
        refresh_cache: bool = False,
        exchange_id: str = None,
        store_in_db: bool = True
    ) -> pd.DataFrame:
        """
        Fetch market data (OHLCV) for a symbol with comprehensive caching and error handling.
        
        This method tries to fetch data from:
        1. Memory cache (if not expired and refresh_cache is False)
        2. Database (if available and not in cache)
        3. Exchange API (if not in database or refresh_cache is True)
        
        Args:
            symbol: Trading pair symbol (e.g., "BTC/USDT")
            timeframe: Candlestick timeframe (e.g., "1h", "4h", "1d")
            limit: Maximum number of candles to fetch
            start_time: Start time for historical data
            end_time: End time for historical data
            include_indicators: Whether to calculate technical indicators
            refresh_cache: Whether to force refresh cached data
            exchange_id: Exchange to fetch data from (uses default if None)
            store_in_db: Whether to store fetched data in database
            
        Returns:
            DataFrame with market data and indicators (if requested)
        """
        # Normalize symbol (some exchanges use different formats)
        symbol = symbol.upper()
        exchange_id = exchange_id or self.default_exchange
        
        # Default limit
        if limit is None:
            limit = 100
            
        # Get symbol lock for thread safety
        symbol_lock = self._get_cache_lock(symbol)
        
        with symbol_lock:
            # Try to get from cache first if refresh not forced
            if not refresh_cache:
                cached_data = self._get_from_cache(symbol, timeframe)
                if cached_data is not None:
                    # Apply limit
                    if limit and len(cached_data) > limit:
                        cached_data = cached_data.iloc[-limit:]
                        
                    # Calculate indicators if needed
                    if include_indicators:
                        return self._calculate_indicators(cached_data)
                    else:
                        return cached_data
            
            # Try to get from database if available
            if self.db_manager is not None and not refresh_cache:
                db_data = self._load_from_database(
                    symbol, 
                    timeframe, 
                    limit=limit,
                    start_time=start_time,
                    end_time=end_time
                )
                
                if db_data is not None and not db_data.empty:
                    # Store in cache for future use
                    self._store_in_cache(symbol, timeframe, db_data)
                    
                    # Calculate indicators if needed
                    if include_indicators:
                        return self._calculate_indicators(db_data)
                    else:
                        return db_data
            
            # Fetch from exchange API
            try:
                # Run async fetch in event loop
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                
                try:
                    if start_time and end_time:
                        # Fetch historical data in batches
                        klines = loop.run_until_complete(
                            self._fetch_historical_klines_async(
                                exchange_id=exchange_id,
                                symbol=symbol,
                                timeframe=timeframe,
                                start_time=start_time,
                                end_time=end_time
                            )
                        )
                    else:
                        # Fetch recent data
                        klines = loop.run_until_complete(
                            self._fetch_klines_async(
                                exchange_id=exchange_id,
                                symbol=symbol,
                                timeframe=timeframe,
                                limit=limit
                            )
                        )
                finally:
                    loop.close()
                
                # Process klines
                df = self._process_klines(klines)
                
                if df.empty:
                    logging.warning(f"No data returned from exchange for {symbol} {timeframe}")
                    return pd.DataFrame()
                
                # Store in cache
                self._store_in_cache(symbol, timeframe, df)
                
                # Store in database if requested
                if store_in_db and self.db_manager is not None:
                    self._store_in_database(symbol, timeframe, df)
                
                # Calculate indicators if needed
                if include_indicators:
                    return self._calculate_indicators(df)
                else:
                    return df
                    
            except Exception as e:
                logging.error(f"Error fetching market data: {e}")
                if HAVE_ERROR_HANDLING:
                    ErrorHandler.handle_error(
                        error=e,
                        category=ErrorCategory.DATA_PROCESSING,
                        severity=ErrorSeverity.ERROR,
                        context={
                            "symbol": symbol,
                            "timeframe": timeframe,
                            "exchange": exchange_id
                        }
                    )
                return pd.DataFrame()
    
    @lru_cache(maxsize=100)
    def get_available_symbols(self, exchange_id=None):
        """
        Get list of available trading symbols from an exchange.
        
        Args:
            exchange_id: Exchange to query (uses default if None)
            
        Returns:
            List of available symbols
        """
        exchange_id = exchange_id or self.default_exchange
        
        try:
            # Get exchange
            exchange = self._exchanges.get(exchange_id)
            
            if exchange is None:
                # Initialize exchange
                exchange_class = getattr(ccxt, exchange_id)
                exchange = exchange_class({
                    'enableRateLimit': True,
                    'timeout': 30000,
                })
                self._exchanges[exchange_id] = exchange
            
            # Get rate limiter
            rate_limiter = self._get_rate_limiter(exchange_id)
            
            # Wait for rate limit
            rate_limiter.sync_acquire()
            
            # Load markets
            markets = exchange.load_markets()
            
            # Extract symbols
            symbols = list(markets.keys())
            
            return symbols
            
        except Exception as e:
            logging.error(f"Error getting available symbols from {exchange_id}: {e}")
            if HAVE_ERROR_HANDLING:
                ErrorHandler.handle_error(
                    error=e,
                    category=ErrorCategory.NETWORK,
                    severity=ErrorSeverity.WARNING,
                    context={"exchange": exchange_id}
                )
            return []
    
    async def _fetch_orderbook_async(self, exchange_id, symbol, limit=None):
        """Fetch order book asynchronously with rate limiting and error handling"""
        exchange = None
        
        try:
            # Get or initialize async exchange
            if exchange_id in self._async_exchanges:
                exchange = self._async_exchanges[exchange_id]
            else:
                exchange = await self._init_async_exchange(exchange_id)
                
            if exchange is None:
                raise Exception(f"Failed to initialize async exchange {exchange_id}")
            
            # Get rate limiter
            rate_limiter = self._get_rate_limiter(exchange_id)
            
            # Wait for rate limit
            await rate_limiter.acquire()
            
            # Update metrics
            self._update_performance_metrics('api_calls')
            
            # Fetch order book
            orderbook = await exchange.fetch_order_book(symbol, limit=limit)
            
            return orderbook
            
        except Exception as e:
            self._update_performance_metrics('api_errors')
            
            # Log error
            logging.error(f"Error fetching order book from {exchange_id} for {symbol}: {e}")
            
            if HAVE_ERROR_HANDLING:
                ErrorHandler.handle_error(
                    error=e,
                    category=ErrorCategory.NETWORK,
                    severity=ErrorSeverity.WARNING,
                    context={
                        "exchange": exchange_id,
                        "symbol": symbol,
                        "operation": "fetch_orderbook_async"
                    }
                )
            
            return None
        finally:
            # Clean up exchange connection
            if exchange is not None and exchange_id not in self._async_exchanges:
                try:
                    await exchange.close()
                except:
                    pass
    
    def fetch_order_book(
        self,
        symbol: str,
        limit: int = 20,
        refresh_cache: bool = False,
        exchange_id: str = None
    ) -> Dict:
        """
        Fetch order book for a symbol with caching.
        
        Args:
            symbol: Trading pair symbol (e.g., "BTC/USDT")
            limit: Depth of the order book
            refresh_cache: Whether to force refresh cached data
            exchange_id: Exchange to fetch data from (uses default if None)
            
        Returns:
            Dictionary with order book data (asks, bids)
        """
        # Normalize symbol
        symbol = symbol.upper()
        exchange_id = exchange_id or self.default_exchange
        
        # Get symbol lock for thread safety
        symbol_lock = self._get_cache_lock(symbol)
        
        with symbol_lock:
            # Try to get from cache first if refresh not forced
            if not refresh_cache:
                with self._global_lock:
                    if symbol in self._orderbook_cache:
                        cache_entry = self._orderbook_cache[symbol]
                        
                        if not cache_entry.is_expired():
                            self._update_performance_metrics('cache_hits')
                            return cache_entry.data.copy()
                        
                        # Entry expired
                        self._update_performance_metrics('cache_misses')
            
            # Fetch from exchange API
            try:
                # Run async fetch in event loop
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                
                try:
                    orderbook = loop.run_until_complete(
                        self._fetch_orderbook_async(
                            exchange_id=exchange_id,
                            symbol=symbol,
                            limit=limit
                        )
                    )
                finally:
                    loop.close()
                
                if orderbook is None:
                    logging.warning(f"No order book data returned for {symbol}")
                    return None
                
                # Store in cache
                with self._global_lock:
                    self._orderbook_cache[symbol] = CacheEntry(
                        data=orderbook,
                        expiry=5  # Short expiry for order book (5 seconds)
                    )
                
                return orderbook
                    
            except Exception as e:
                logging.error(f"Error fetching order book: {e}")
                if HAVE_ERROR_HANDLING:
                    ErrorHandler.handle_error(
                        error=e,
                        category=ErrorCategory.DATA_PROCESSING,
                        severity=ErrorSeverity.WARNING,
                        context={
                            "symbol": symbol,
                            "exchange": exchange_id
                        }
                    )
                return None
    
    def analyze_order_book(self, symbol: str, limit: int = 20, exchange_id: str = None) -> Dict:
        """
        Analyze order book to extract useful information for trading decisions.
        
        Args:
            symbol: Trading pair symbol (e.g., "BTC/USDT")
            limit: Depth of the order book
            exchange_id: Exchange to fetch data from (uses default if None)
            
        Returns:
            Dictionary with order book analysis
        """
        # Fetch order book
        orderbook = self.fetch_order_book(
            symbol=symbol,
            limit=limit,
            refresh_cache=True,  # Always get fresh data for analysis
            exchange_id=exchange_id
        )
        
        if orderbook is None:
            return None
        
        try:
            # Extract asks and bids
            asks = np.array(orderbook['asks'])
            bids = np.array(orderbook['bids'])
            
            # Calculate spread
            best_ask = asks[0][0]
            best_bid = bids[0][0]
            spread = best_ask - best_bid
            spread_percentage = (spread / best_bid) * 100
            
            # Calculate cumulative volume
            ask_prices = asks[:, 0]
            ask_volumes = asks[:, 1]
            bid_prices = bids[:, 0]
            bid_volumes = bids[:, 1]
            
            # Calculate order imbalance
            ask_value = np.sum(ask_prices * ask_volumes)
            bid_value = np.sum(bid_prices * bid_volumes)
            order_imbalance = (bid_value - ask_value) / (bid_value + ask_value)
            
            # Calculate price impact for market orders
            # How much would price move for a $10k market order?
            market_order_amount = 10000  # in quote currency
            
            # Simulate market buy
            remaining = market_order_amount
            executed_price = 0
            executed_amount = 0
            
            for i in range(len(asks)):
                ask_price = asks[i][0]
                ask_volume = asks[i][1]
                ask_value = ask_price * ask_volume
                
                if remaining <= ask_value:
                    # Can execute the remaining amount at this level
                    execution_volume = remaining / ask_price
                    executed_price += remaining
                    executed_amount += execution_volume
                    remaining = 0
                    break
                else:
                    # Execute the entire level and continue
                    executed_price += ask_value
                    executed_amount += ask_volume
                    remaining -= ask_value
            
            avg_buy_price = executed_price / executed_amount if executed_amount > 0 else best_ask
            buy_slippage = ((avg_buy_price / best_ask) - 1) * 100
            
            # Simulate market sell
            remaining = market_order_amount
            executed_price = 0
            executed_amount = 0
            
            for i in range(len(bids)):
                bid_price = bids[i][0]
                bid_volume = bids[i][1]
                bid_value = bid_price * bid_volume
                
                if remaining <= bid_value:
                    # Can execute the remaining amount at this level
                    execution_volume = remaining / bid_price
                    executed_price += remaining
                    executed_amount += execution_volume
                    remaining = 0
                    break
                else:
                    # Execute the entire level and continue
                    executed_price += bid_value
                    executed_amount += bid_volume
                    remaining -= bid_value
            
            avg_sell_price = executed_price / executed_amount if executed_amount > 0 else best_bid
            sell_slippage = ((best_bid / avg_sell_price) - 1) * 100
            
            return {
                'symbol': symbol,
                'best_ask': best_ask,
                'best_bid': best_bid,
                'spread': spread,
                'spread_percentage': spread_percentage,
                'order_imbalance': order_imbalance,
                'buy_slippage': buy_slippage,
                'sell_slippage': sell_slippage,
                'ask_depth': np.sum(ask_volumes),
                'bid_depth': np.sum(bid_volumes),
                'ask_liquidity': ask_value,
                'bid_liquidity': bid_value,
                'timestamp': datetime.now()
            }
            
        except Exception as e:
            logging.error(f"Error analyzing order book: {e}")
            if HAVE_ERROR_HANDLING:
                ErrorHandler.handle_error(
                    error=e,
                    category=ErrorCategory.DATA_PROCESSING,
                    severity=ErrorSeverity.WARNING,
                    context={
                        "symbol": symbol,
                        "exchange": exchange_id,
                        "operation": "analyze_order_book"
                    }
                )
            return None
    
    async def _fetch_ticker_async(self, exchange_id, symbol):
        """Fetch ticker asynchronously with rate limiting and error handling"""
        exchange = None
        
        try:
            # Get or initialize async exchange
            if exchange_id in self._async_exchanges:
                exchange = self._async_exchanges[exchange_id]
            else:
                exchange = await self._init_async_exchange(exchange_id)
                
            if exchange is None:
                raise Exception(f"Failed to initialize async exchange {exchange_id}")
            
            # Get rate limiter
            rate_limiter = self._get_rate_limiter(exchange_id)
            
            # Wait for rate limit
            await rate_limiter.acquire()
            
            # Update metrics
            self._update_performance_metrics('api_calls')
            
            # Fetch ticker
            ticker = await exchange.fetch_ticker(symbol)
            
            return ticker
            
        except Exception as e:
            self._update_performance_metrics('api_errors')
            
            # Log error
            logging.error(f"Error fetching ticker from {exchange_id} for {symbol}: {e}")
            
            if HAVE_ERROR_HANDLING:
                ErrorHandler.handle_error(
                    error=e,
                    category=ErrorCategory.NETWORK,
                    severity=ErrorSeverity.WARNING,
                    context={
                        "exchange": exchange_id,
                        "symbol": symbol,
                        "operation": "fetch_ticker_async"
                    }
                )
            
            return None
        finally:
            # Clean up exchange connection
            if exchange is not None and exchange_id not in self._async_exchanges:
                try:
                    await exchange.close()
                except:
                    pass
    
    def fetch_ticker(
        self, 
        symbol: str, 
        refresh_cache: bool = False,
        exchange_id: str = None
    ) -> Dict:
        """
        Fetch current ticker for a symbol with caching.
        
        Args:
            symbol: Trading pair symbol (e.g., "BTC/USDT")
            refresh_cache: Whether to force refresh cached data
            exchange_id: Exchange to fetch data from (uses default if None)
            
        Returns:
            Dictionary with ticker data
        """
        # Normalize symbol
        symbol = symbol.upper()
        exchange_id = exchange_id or self.default_exchange
        
        # Get symbol lock for thread safety
        symbol_lock = self._get_cache_lock(symbol)
        
        with symbol_lock:
            # Try to get from cache first if refresh not forced
            if not refresh_cache:
                with self._global_lock:
                    if symbol in self._ticker_cache:
                        cache_entry = self._ticker_cache[symbol]
                        
                        if not cache_entry.is_expired():
                            self._update_performance_metrics('cache_hits')
                            return cache_entry.data.copy()
                        
                        # Entry expired
                        self._update_performance_metrics('cache_misses')
            
            # Fetch from exchange API
            try:
                # Run async fetch in event loop
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                
                try:
                    ticker = loop.run_until_complete(
                        self._fetch_ticker_async(
                            exchange_id=exchange_id,
                            symbol=symbol
                        )
                    )
                finally:
                    loop.close()
                
                if ticker is None:
                    logging.warning(f"No ticker data returned for {symbol}")
                    return None
                
                # Store in cache
                with self._global_lock:
                    self._ticker_cache[symbol] = CacheEntry(
                        data=ticker,
                        expiry=5  # Short expiry for ticker (5 seconds)
                    )
                
                return ticker
                    
            except Exception as e:
                logging.error(f"Error fetching ticker: {e}")
                if HAVE_ERROR_HANDLING:
                    ErrorHandler.handle_error(
                        error=e,
                        category=ErrorCategory.DATA_PROCESSING,
                        severity=ErrorSeverity.WARNING,
                        context={
                            "symbol": symbol,
                            "exchange": exchange_id
                        }
                    )
                return None
    
    def start_live_updates(self, symbol: str, timeframe: str = '1m', callback: Callable = None):
        """
        Start live market data updates for a symbol.
        
        Args:
            symbol: Trading pair symbol (e.g., "BTC/USDT")
            timeframe: Candlestick timeframe (e.g., "1m", "5m")
            callback: Function to call with updated data
            
        Returns:
            bool: Whether live updates were successfully started
        """
        if not self.enable_live_updates:
            logging.warning("Live updates are disabled")
            return False
        
        # Normalize symbol
        symbol = symbol.upper()
        
        # Check if already updating
        key = f"{symbol}_{timeframe}"
        with self._global_lock:
            if key in self._live_update_tasks:
                # Already updating this symbol/timeframe
                if callback and callback not in self._live_update_callbacks.get(key, []):
                    # Add callback
                    if key not in self._live_update_callbacks:
                        self._live_update_callbacks[key] = []
                    self._live_update_callbacks[key].append(callback)
                return True
        
        # Create update task
        try:
            # Create and start task
            def update_task():
                while not self._shutdown_flag:
                    try:
                        # Fetch latest data
                        data = self.fetch_market_data(
                            symbol=symbol,
                            timeframe=timeframe,
                            limit=100,
                            refresh_cache=True
                        )
                        
                        # Call callbacks
                        with self._global_lock:
                            callbacks = self._live_update_callbacks.get(key, [])
                            
                        for cb in callbacks:
                            try:
                                cb(data)
                            except Exception as e:
                                logging.error(f"Error in live update callback: {e}")
                        
                        # Sleep until next update
                        # For 1m, we update every 15s
                        # For other timeframes, we update at 1/4 of the timeframe interval
                        if timeframe == '1m':
                            time.sleep(15)
                        else:
                            # Get timeframe in seconds
                            tf_seconds = DEFAULT_TIMEFRAMES.get(timeframe, 3600)
                            sleep_time = max(15, tf_seconds / 4)  # At least 15 seconds
                            time.sleep(sleep_time)
                            
                    except Exception as e:
                        logging.error(f"Error in live update task: {e}")
                        time.sleep(5)  # Sleep on error
                        
                logging.info(f"Live update task for {symbol} {timeframe} stopped")
            
            # Start task in a new thread
            thread = threading.Thread(target=update_task, daemon=True)
            thread.start()
            
            # Store task
            with self._global_lock:
                self._live_update_tasks[key] = thread
                
                # Store callback
                if callback:
                    if key not in self._live_update_callbacks:
                        self._live_update_callbacks[key] = []
                    self._live_update_callbacks[key].append(callback)
            
            logging.info(f"Started live updates for {symbol} {timeframe}")
            return True
            
        except Exception as e:
            logging.error(f"Error starting live updates: {e}")
            if HAVE_ERROR_HANDLING:
                ErrorHandler.handle_error(
                    error=e,
                    category=ErrorCategory.DATA_PROCESSING,
                    severity=ErrorSeverity.WARNING,
                    context={
                        "symbol": symbol,
                        "timeframe": timeframe,
                        "operation": "start_live_updates"
                    }
                )
            return False
    
    def stop_live_updates(self, symbol: str = None, timeframe: str = None):
        """
        Stop live market data updates.
        
        Args:
            symbol: Trading pair symbol (or None to stop all)
            timeframe: Candlestick timeframe (or None to stop all for the symbol)
            
        Returns:
            bool: Whether live updates were successfully stopped
        """
        with self._global_lock:
            keys_to_stop = []
            
            if symbol and timeframe:
                # Stop specific symbol/timeframe
                key = f"{symbol.upper()}_{timeframe}"
                keys_to_stop.append(key)
            elif symbol:
                # Stop all timeframes for symbol
                symbol = symbol.upper()
                for key in list(self._live_update_tasks.keys()):
                    if key.startswith(f"{symbol}_"):
                        keys_to_stop.append(key)
            else:
                # Stop all
                keys_to_stop = list(self._live_update_tasks.keys())
            
            # Stop tasks
            for key in keys_to_stop:
                if key in self._live_update_tasks:
                    # Thread will exit on next iteration
                    if key in self._live_update_callbacks:
                        del self._live_update_callbacks[key]
                    # Note: Thread will exit when _shutdown_flag is True
                
            return True
    
    def clear_cache(self, symbol: str = None, timeframe: str = None):
        """
        Clear market data cache.
        
        Args:
            symbol: Trading pair symbol (or None to clear all)
            timeframe: Candlestick timeframe (or None to clear all for the symbol)
            
        Returns:
            int: Number of cache entries cleared
        """
        with self._global_lock:
            if symbol and timeframe:
                # Clear specific symbol/timeframe
                key = self._get_key(symbol.upper(), timeframe)
                if key in self._kline_cache:
                    del self._kline_cache[key]
                    return 1
                return 0
            elif symbol:
                # Clear all timeframes for symbol
                symbol = symbol.upper()
                count = 0
                keys_to_remove = []
                
                for key in self._kline_cache:
                    if key.startswith(f"{symbol}_"):
                        keys_to_remove.append(key)
                        count += 1
                
                for key in keys_to_remove:
                    del self._kline_cache[key]
                    
                # Also clear orderbook and ticker cache
                if symbol in self._orderbook_cache:
                    del self._orderbook_cache[symbol]
                    count += 1
                    
                if symbol in self._ticker_cache:
                    del self._ticker_cache[symbol]
                    count += 1
                    
                return count
            else:
                # Clear all
                count = len(self._kline_cache) + len(self._orderbook_cache) + len(self._ticker_cache)
                self._kline_cache.clear()
                self._orderbook_cache.clear()
                self._ticker_cache.clear()
                return count
    
    def shutdown(self):
        """
        Shutdown the market data manager, closing all connections and stopping updates.
        """
        logging.info("Shutting down market data manager...")
        
        # Stop live updates
        self._shutdown_flag = True
        self.stop_live_updates()
        
        # Close all exchanges
        with self._global_lock:
            # Close sync exchanges
            for exchange_id, exchange in self._exchanges.items():
                try:
                    if hasattr(exchange, 'close'):
                        exchange.close()
                except:
                    pass
            
            # Close async exchanges
            for exchange_id, exchange in self._async_exchanges.items():
                try:
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    
                    try:
                        loop.run_until_complete(exchange.close())
                    finally:
                        loop.close()
                except:
                    pass
            
            # Clear caches
            self._kline_cache.clear()
            self._orderbook_cache.clear()
            self._ticker_cache.clear()
            
            logging.info("Market data manager shutdown complete")

# Decorate methods with error handling if available
if HAVE_ERROR_HANDLING:
    MarketDataManager.fetch_market_data = safe_execute(
        ErrorCategory.DATA_PROCESSING, 
        default_return=pd.DataFrame()
    )(MarketDataManager.fetch_market_data)
    
    MarketDataManager.fetch_order_book = safe_execute(
        ErrorCategory.DATA_PROCESSING, 
        default_return=None
    )(MarketDataManager.fetch_order_book)
    
    MarketDataManager.fetch_ticker = safe_execute(
        ErrorCategory.DATA_PROCESSING, 
        default_return=None
    )(MarketDataManager.fetch_ticker)
    
    MarketDataManager.analyze_order_book = safe_execute(
        ErrorCategory.DATA_PROCESSING, 
        default_return=None
    )(MarketDataManager.analyze_order_book)
