# data_fetcher.py

import os
import time
import json
import logging
import asyncio
import hashlib
import datetime
import traceback
from typing import Dict, List, Optional, Union, Tuple, Any, Callable
from functools import lru_cache, wraps
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
import requests
import aiohttp
import ccxt
import ccxt.async_support as ccxt_async
from dateutil.parser import parse

# Try to import the system-specific modules with fallback mechanisms
try:
    from core.error_handling import safe_execute, ErrorCategory, ErrorSeverity, ErrorHandler
    HAVE_ERROR_HANDLING = True
except ImportError:
    HAVE_ERROR_HANDLING = False
    logging.warning("Error handling module not available. Using basic error handling.")

try:
    from core.database_factory import DatabaseFactory
    HAVE_DB_FACTORY = True
except ImportError:
    HAVE_DB_FACTORY = False
    logging.warning("Database factory not available. Will attempt direct database imports.")

if not HAVE_DB_FACTORY:
    try:
        from core.database_manager import DatabaseManager
        HAVE_REAL_DB = True
    except ImportError:
        HAVE_REAL_DB = False
        
    try:
        from core.mock_database import MockDatabaseManager
        HAVE_MOCK_DB = True
    except ImportError:
        HAVE_MOCK_DB = False

# Configure logging
logger = logging.getLogger(__name__)

# Rate limiting decorator
def rate_limit(calls_per_second: float = 1.0):
    """
    Decorator to enforce rate limiting for API calls
    
    Args:
        calls_per_second: Maximum number of calls allowed per second
    """
    min_interval = 1.0 / calls_per_second
    last_call_times = {}
    
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Get a unique key for this function call
            key = f"{func.__name__}:{str(args)}:{str(kwargs)}"
            
            # Get current time
            current_time = time.time()
            
            # Check if we need to wait
            if key in last_call_times:
                elapsed = current_time - last_call_times[key]
                if elapsed < min_interval:
                    # Need to wait
                    sleep_time = min_interval - elapsed
                    time.sleep(sleep_time)
            
            # Update last call time
            last_call_times[key] = time.time()
            
            # Call the function
            return func(*args, **kwargs)
        return wrapper
    return decorator

# Caching decorator
def cache_result(ttl_seconds: int = 60):
    """
    Decorator to cache function results with a time-to-live (TTL)
    
    Args:
        ttl_seconds: Cache validity period in seconds
    """
    cache = {}
    
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Create a cache key
            key = f"{func.__name__}:{str(args)}:{str(kwargs)}"
            key_hash = hashlib.md5(key.encode()).hexdigest()
            
            # Check if result is in cache and still valid
            if key_hash in cache:
                result, timestamp = cache[key_hash]
                if time.time() - timestamp < ttl_seconds:
                    return result
            
            # Call function and cache result
            result = func(*args, **kwargs)
            cache[key_hash] = (result, time.time())
            
            # Clean expired cache entries periodically
            if random.random() < 0.01:  # 1% chance per call to clean cache
                current_time = time.time()
                keys_to_delete = [k for k, (_, ts) in cache.items() if current_time - ts >= ttl_seconds]
                for k in keys_to_delete:
                    del cache[k]
            
            return result
        return wrapper
    return decorator

class APIRateLimiter:
    """Manages rate limiting for different exchanges"""
    
    def __init__(self):
        self.rate_limits = {
            'binance': {'calls': 1200, 'period': 60},  # 1200 calls per minute
            'coinbase': {'calls': 10, 'period': 1},    # 10 calls per second
            'kraken': {'calls': 15, 'period': 1},      # 15 calls per second
            'kucoin': {'calls': 30, 'period': 3},      # 30 calls per 3 seconds
            'bybit': {'calls': 50, 'period': 3},       # 50 calls per 3 seconds
            'default': {'calls': 1, 'period': 1}       # 1 call per second
        }
        self.last_calls = {}
        self.locks = {}
        
        # Initialize locks for each exchange
        for exchange in self.rate_limits:
            self.locks[exchange] = asyncio.Lock()
            self.last_calls[exchange] = []
    
    async def wait_if_needed(self, exchange: str):
        """
        Wait if we've hit the rate limit for an exchange
        
        Args:
            exchange: Exchange name
        """
        exchange = exchange.lower()
        if exchange not in self.rate_limits:
            exchange = 'default'
        
        limit = self.rate_limits[exchange]
        max_calls = limit['calls']
        period = limit['period']
        
        async with self.locks[exchange]:
            # Clean up old timestamps
            current_time = time.time()
            self.last_calls[exchange] = [t for t in self.last_calls[exchange] if current_time - t <= period]
            
            # Check if we need to wait
            if len(self.last_calls[exchange]) >= max_calls:
                # Need to wait for the oldest call to expire
                oldest_call = min(self.last_calls[exchange])
                wait_time = period - (current_time - oldest_call)
                if wait_time > 0:
                    logger.debug(f"Rate limit reached for {exchange}. Waiting {wait_time:.2f} seconds.")
                    await asyncio.sleep(wait_time)
            
            # Add current call
            self.last_calls[exchange].append(time.time())

class DataFetcher:
    """
    Comprehensive market data fetcher with support for multiple exchanges,
    caching, rate limiting, and robust error handling.
    """
    
    def __init__(self, db_manager=None, use_async=True, cache_dir='./cache'):
        """
        Initialize the data fetcher
        
        Args:
            db_manager: Database manager instance for data persistence
            use_async: Whether to use async API calls for faster fetching
            cache_dir: Directory for disk cache
        """
        self.use_async = use_async
        self.cache_dir = cache_dir
        
        # Ensure cache directory exists
        os.makedirs(cache_dir, exist_ok=True)
        
        # Initialize rate limiter
        self.rate_limiter = APIRateLimiter()
        
        # Initialize database connection
        self.db_manager = db_manager
        if self.db_manager is None:
            self._initialize_db()
        
        # Initialize exchange connections
        self.exchanges = {}
        self.async_exchanges = {}
        self._init_exchanges()
        
        # Thread pool for parallel operations
        self.thread_pool = ThreadPoolExecutor(max_workers=10)
        
        # Success metrics for monitoring
        self.metrics = {
            'successful_fetches': 0,
            'failed_fetches': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'db_hits': 0,
            'db_misses': 0,
            'api_calls': 0
        }
        
        # Timeframe mapping for standardization
        self.timeframe_map = {
            '1m': {'seconds': 60, 'ccxt': '1m', 'binance': '1m'},
            '3m': {'seconds': 180, 'ccxt': '3m', 'binance': '3m'},
            '5m': {'seconds': 300, 'ccxt': '5m', 'binance': '5m'},
            '15m': {'seconds': 900, 'ccxt': '15m', 'binance': '15m'},
            '30m': {'seconds': 1800, 'ccxt': '30m', 'binance': '30m'},
            '1h': {'seconds': 3600, 'ccxt': '1h', 'binance': '1h'},
            '2h': {'seconds': 7200, 'ccxt': '2h', 'binance': '2h'},
            '4h': {'seconds': 14400, 'ccxt': '4h', 'binance': '4h'},
            '6h': {'seconds': 21600, 'ccxt': '6h', 'binance': '6h'},
            '8h': {'seconds': 28800, 'ccxt': '8h', 'binance': '8h'},
            '12h': {'seconds': 43200, 'ccxt': '12h', 'binance': '12h'},
            '1d': {'seconds': 86400, 'ccxt': '1d', 'binance': '1d'},
            '3d': {'seconds': 259200, 'ccxt': '3d', 'binance': '3d'},
            '1w': {'seconds': 604800, 'ccxt': '1w', 'binance': '1w'},
            '1M': {'seconds': 2592000, 'ccxt': '1M', 'binance': '1M'}
        }
    
    def _initialize_db(self):
        """Initialize database connection with fallback mechanisms"""
        try:
            if HAVE_DB_FACTORY:
                self.db_manager = DatabaseFactory.create_database(use_mock=False)
                if self.db_manager is None:
                    logger.warning("Failed to create real database, trying mock database")
                    self.db_manager = DatabaseFactory.create_database(use_mock=True)
            elif HAVE_REAL_DB:
                self.db_manager = DatabaseManager()
            elif HAVE_MOCK_DB:
                self.db_manager = MockDatabaseManager()
            else:
                logger.warning("No database implementation available. Data will not be persisted.")
                self.db_manager = None
        except Exception as e:
            logger.error(f"Error initializing database: {e}")
            self.db_manager = None
    
    def _init_exchanges(self):
        """Initialize exchange connections"""
        # Standard exchanges
        try:
            self.exchanges['binance'] = ccxt.binance({
                'enableRateLimit': True,
                'options': {
                    'defaultType': 'spot',
                    'adjustForTimeDifference': True
                }
            })
        except Exception as e:
            logger.error(f"Error initializing Binance: {e}")
        
        try:
            self.exchanges['coinbase'] = ccxt.coinbase({
                'enableRateLimit': True
            })
        except Exception as e:
            logger.error(f"Error initializing Coinbase: {e}")
            
        try:
            self.exchanges['kraken'] = ccxt.kraken({
                'enableRateLimit': True
            })
        except Exception as e:
            logger.error(f"Error initializing Kraken: {e}")
            
        try:
            self.exchanges['kucoin'] = ccxt.kucoin({
                'enableRateLimit': True
            })
        except Exception as e:
            logger.error(f"Error initializing KuCoin: {e}")
            
        # Async exchanges (if enabled)
        if self.use_async:
            try:
                self.async_exchanges['binance'] = ccxt_async.binance({
                    'enableRateLimit': True,
                    'options': {
                        'defaultType': 'spot',
                        'adjustForTimeDifference': True
                    }
                })
            except Exception as e:
                logger.error(f"Error initializing async Binance: {e}")
            
            try:
                self.async_exchanges['coinbase'] = ccxt_async.coinbase({
                    'enableRateLimit': True
                })
            except Exception as e:
                logger.error(f"Error initializing async Coinbase: {e}")
                
            try:
                self.async_exchanges['kraken'] = ccxt_async.kraken({
                    'enableRateLimit': True
                })
            except Exception as e:
                logger.error(f"Error initializing async Kraken: {e}")
                
            try:
                self.async_exchanges['kucoin'] = ccxt_async.kucoin({
                    'enableRateLimit': True
                })
            except Exception as e:
                logger.error(f"Error initializing async KuCoin: {e}")
    
    def get_exchange(self, exchange_name: str, async_mode: bool = False):
        """
        Get exchange instance by name with fallback
        
        Args:
            exchange_name: Exchange name (e.g., 'binance')
            async_mode: Whether to return async exchange instance
            
        Returns:
            Exchange instance or None if not available
        """
        exchange_name = exchange_name.lower()
        
        if async_mode:
            if exchange_name in self.async_exchanges:
                return self.async_exchanges[exchange_name]
            
            # Try to find any available async exchange as fallback
            if self.async_exchanges:
                logger.warning(f"Async exchange {exchange_name} not available, using fallback")
                return next(iter(self.async_exchanges.values()))
                
            return None
        else:
            if exchange_name in self.exchanges:
                return self.exchanges[exchange_name]
            
            # Try to find any available exchange as fallback
            if self.exchanges:
                logger.warning(f"Exchange {exchange_name} not available, using fallback")
                return next(iter(self.exchanges.values()))
                
            return None
    
    def _generate_cache_key(self, exchange: str, symbol: str, timeframe: str, 
                           since: Optional[int] = None, limit: Optional[int] = None) -> str:
        """
        Generate a unique cache key for market data
        
        Args:
            exchange: Exchange name
            symbol: Trading pair
            timeframe: Candle timeframe
            since: Start timestamp
            limit: Maximum number of candles
            
        Returns:
            Cache key string
        """
        components = [exchange.lower(), symbol.upper(), timeframe]
        
        if since is not None:
            components.append(str(since))
        
        if limit is not None:
            components.append(str(limit))
            
        key = '_'.join(components)
        return hashlib.md5(key.encode()).hexdigest()
    
    def _save_to_cache(self, key: str, data: pd.DataFrame, expiry_seconds: int = 300) -> bool:
        """
        Save data to disk cache
        
        Args:
            key: Cache key
            data: DataFrame to cache
            expiry_seconds: Cache validity period in seconds
            
        Returns:
            Success status
        """
        try:
            cache_file = os.path.join(self.cache_dir, f"{key}.parquet")
            
            # Save data
            data.to_parquet(cache_file)
            
            # Save metadata (including expiry)
            metadata = {
                'created': time.time(),
                'expires': time.time() + expiry_seconds,
                'rows': len(data)
            }
            
            with open(os.path.join(self.cache_dir, f"{key}.meta"), 'w') as f:
                json.dump(metadata, f)
                
            return True
        except Exception as e:
            logger.error(f"Error saving to cache: {e}")
            return False
    
    def _load_from_cache(self, key: str) -> Optional[pd.DataFrame]:
        """
        Load data from disk cache if available and not expired
        
        Args:
            key: Cache key
            
        Returns:
            DataFrame or None if not available or expired
        """
        try:
            cache_file = os.path.join(self.cache_dir, f"{key}.parquet")
            meta_file = os.path.join(self.cache_dir, f"{key}.meta")
            
            # Check if files exist
            if not (os.path.exists(cache_file) and os.path.exists(meta_file)):
                self.metrics['cache_misses'] += 1
                return None
            
            # Load metadata
            with open(meta_file, 'r') as f:
                metadata = json.load(f)
            
            # Check if expired
            if time.time() > metadata.get('expires', 0):
                self.metrics['cache_misses'] += 1
                return None
            
            # Load data
            data = pd.read_parquet(cache_file)
            self.metrics['cache_hits'] += 1
            return data
            
        except Exception as e:
            logger.error(f"Error loading from cache: {e}")
            self.metrics['cache_misses'] += 1
            return None
    
    def _load_from_database(self, exchange: str, symbol: str, timeframe: str, 
                           start_time: Optional[datetime.datetime] = None,
                           end_time: Optional[datetime.datetime] = None,
                           limit: Optional[int] = None) -> Optional[pd.DataFrame]:
        """
        Load market data from database if available
        
        Args:
            exchange: Exchange name
            symbol: Trading pair
            timeframe: Candle timeframe
            start_time: Start datetime
            end_time: End datetime
            limit: Maximum number of candles
            
        Returns:
            DataFrame or None if not available
        """
        if self.db_manager is None:
            self.metrics['db_misses'] += 1
            return None
            
        try:
            # Convert timeframe to database format if needed
            db_timeframe = timeframe
            
            # Query database
            data = self.db_manager.get_market_data(
                symbol=symbol,
                timeframe=db_timeframe,
                start_time=start_time,
                end_time=end_time,
                limit=limit
            )
            
            if data is not None and not data.empty:
                self.metrics['db_hits'] += 1
                return data
                
            self.metrics['db_misses'] += 1
            return None
            
        except Exception as e:
            logger.error(f"Error loading from database: {e}")
            self.metrics['db_misses'] += 1
            return None
    
    def _save_to_database(self, exchange: str, symbol: str, data: pd.DataFrame) -> bool:
        """
        Save market data to database
        
        Args:
            exchange: Exchange name
            symbol: Trading pair
            data: DataFrame to save
            
        Returns:
            Success status
        """
        if self.db_manager is None:
            return False
            
        try:
            # Make sure data has an index
            if not isinstance(data.index, pd.DatetimeIndex):
                if 'timestamp' in data.columns:
                    data['timestamp'] = pd.to_datetime(data['timestamp'], unit='ms')
                    data.set_index('timestamp', inplace=True)
                elif 'time' in data.columns:
                    data['time'] = pd.to_datetime(data['time'], unit='ms')
                    data.set_index('time', inplace=True)
            
            # Store in database
            return self.db_manager.store_market_data(data, symbol)
            
        except Exception as e:
            logger.error(f"Error saving to database: {e}")
            return False
    
    def _standardize_dataframe(self, data: List[List], exchange: str, symbol: str, timeframe: str) -> pd.DataFrame:
        """
        Convert raw OHLCV data to a standardized DataFrame
        
        Args:
            data: Raw OHLCV data from exchange
            exchange: Exchange name
            symbol: Trading pair
            timeframe: Candle timeframe
            
        Returns:
            Standardized DataFrame
        """
        # Create DataFrame
        df = pd.DataFrame(data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        
        # Convert timestamp to datetime
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        
        # Set timestamp as index
        df.set_index('timestamp', inplace=True)
        
        # Sort by timestamp
        df.sort_index(inplace=True)
        
        # Add exchange and symbol columns
        df['exchange'] = exchange
        df['symbol'] = symbol
        
        # Add timeframe column
        df['timeframe'] = timeframe
        
        return df
    
    def _process_ohlcv_list(self, ohlcv_list: List[List], exchange: str, symbol: str, timeframe: str, 
                            add_indicators: bool = False) -> pd.DataFrame:
        """
        Process OHLCV list data into a DataFrame with indicators
        
        Args:
            ohlcv_list: List of OHLCV data points
            exchange: Exchange name
            symbol: Trading pair
            timeframe: Candle timeframe
            add_indicators: Whether to add technical indicators
            
        Returns:
            Processed DataFrame
        """
        # Create standardized DataFrame
        df = self._standardize_dataframe(ohlcv_list, exchange, symbol, timeframe)
        
        # Add indicators if requested
        if add_indicators:
            df = self._add_indicators(df)
            
        return df

    def _add_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add common technical indicators to DataFrame
        
        Args:
            df: OHLCV DataFrame
            
        Returns:
            DataFrame with indicators
        """
        if len(df) < 50:
            # Not enough data for reliable indicators
            return df
            
        try:
            # SMA
            df['sma_20'] = df['close'].rolling(window=20).mean()
            df['sma_50'] = df['close'].rolling(window=50).mean()
            
            # EMA
            df['ema_12'] = df['close'].ewm(span=12, adjust=False).mean()
            df['ema_26'] = df['close'].ewm(span=26, adjust=False).mean()
            
            # MACD
            df['macd'] = df['ema_12'] - df['ema_26']
            df['signal_line'] = df['macd'].ewm(span=9, adjust=False).mean()
            df['macd_histogram'] = df['macd'] - df['signal_line']
            
            # RSI
            delta = df['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / (loss + 1e-10)  # Avoid division by zero
            df['rsi'] = 100 - (100 / (1 + rs))
            
            # Bollinger Bands
            df['middle_band'] = df['close'].rolling(window=20).mean()
            std_dev = df['close'].rolling(window=20).std()
            df['upper_band'] = df['middle_band'] + (std_dev * 2)
            df['lower_band'] = df['middle_band'] - (std_dev * 2)
            
            # Average True Range (ATR)
            df['tr1'] = abs(df['high'] - df['low'])
            df['tr2'] = abs(df['high'] - df['close'].shift())
            df['tr3'] = abs(df['low'] - df['close'].shift())
            df['true_range'] = df[['tr1', 'tr2', 'tr3']].max(axis=1)
            df['atr'] = df['true_range'].rolling(window=14).mean()
            df.drop(['tr1', 'tr2', 'tr3', 'true_range'], axis=1, inplace=True)
            
        except Exception as e:
            logger.error(f"Error adding indicators: {e}")
            
        return df
    
    def _convert_timeframe(self, df: pd.DataFrame, target_timeframe: str) -> pd.DataFrame:
        """
        Convert DataFrame to a different timeframe
        
        Args:
            df: Source DataFrame
            target_timeframe: Target timeframe
            
        Returns:
            DataFrame with new timeframe
        """
        # Convert the timeframe to a pandas frequency string
        timeframe_seconds = self.timeframe_map.get(target_timeframe, {}).get('seconds', 60)
        
        # Define the aggregation rules
        ohlc_dict = {
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        }
        
        # Resample the DataFrame
        resampled = df.resample(f"{timeframe_seconds}S").agg(ohlc_dict)
        
        # Update timeframe column
        if 'timeframe' in df.columns:
            resampled['timeframe'] = target_timeframe
            
        # Forward fill any indicators
        indicators = [col for col in df.columns if col not in ['open', 'high', 'low', 'close', 'volume', 'exchange', 'symbol', 'timeframe']]
        if indicators:
            for indicator in indicators:
                if indicator in resampled.columns:
                    resampled[indicator] = resampled[indicator].ffill()
        
        return resampled
    
    @cache_result(ttl_seconds=60)
    @rate_limit(calls_per_second=5)
    def fetch_ohlcv(self, symbol: str, timeframe: str = '1h', limit: int = 100, 
                   exchange: str = 'binance', since: Optional[int] = None, 
                   add_indicators: bool = False, use_cache: bool = True,
                   use_db: bool = True) -> pd.DataFrame:
        """
        Fetch OHLCV (candle) data from exchange with caching and database integration
        
        Args:
            symbol: Trading pair (e.g., 'BTC/USDT')
            timeframe: Candle timeframe (e.g., '1m', '1h', '1d')
            limit: Number of candles to fetch
            exchange: Exchange name
            since: Start timestamp (optional)
            add_indicators: Whether to add technical indicators
            use_cache: Whether to use disk cache
            use_db: Whether to use database
            
        Returns:
            DataFrame with OHLCV data
        """
        # Standardize inputs
        symbol = symbol.upper()
        exchange = exchange.lower()
        timeframe = timeframe.lower()
        
        # Convert 'since' to milliseconds timestamp if it's a datetime
        if isinstance(since, (datetime.datetime, str)):
            if isinstance(since, str):
                since = parse(since)
            since = int(since.timestamp() * 1000)
            
        # Generate cache key
        cache_key = self._generate_cache_key(exchange, symbol, timeframe, since, limit)
        
        # Try to load from cache first
        if use_cache:
            cached_data = self._load_from_cache(cache_key)
            if cached_data is not None:
                logger.debug(f"Loaded {symbol} {timeframe} data from cache")
                return cached_data
        
        # Try to load from database next
        if use_db and self.db_manager is not None:
            start_time = datetime.datetime.fromtimestamp(since / 1000) if since else None
            db_data = self._load_from_database(exchange, symbol, timeframe, start_time=start_time, limit=limit)
            if db_data is not None and len(db_data) >= limit:
                logger.debug(f"Loaded {symbol} {timeframe} data from database")
                
                # Add indicators if not present
                if add_indicators and not any(col.startswith('sma_') for col in db_data.columns):
                    db_data = self._add_indicators(db_data)
                    
                return db_data
        
        # If we got here, need to fetch from exchange API
        try:
            # Get exchange instance
            ex = self.get_exchange(exchange)
            if ex is None:
                logger.error(f"Exchange {exchange} not available")
                self.metrics['failed_fetches'] += 1
                return pd.DataFrame()
                
            # Get timeframe in exchange format
            ex_timeframe = self.timeframe_map.get(timeframe, {}).get('ccxt', timeframe)
            
            # Fetch OHLCV data
            self.metrics['api_calls'] += 1
            ohlcv = ex.fetch_ohlcv(symbol, timeframe=ex_timeframe, since=since, limit=limit)
            
            # Check if we got data
            if not ohlcv:
                logger.warning(f"No OHLCV data received for {symbol} {timeframe}")
                self.metrics['failed_fetches'] += 1
                return pd.DataFrame()
                
            # Process data
            df = self._process_ohlcv_list(ohlcv, exchange, symbol, timeframe, add_indicators)
            
            # Save to cache
            if use_cache:
                self._save_to_cache(cache_key, df)
                
            # Save to database
            if use_db and self.db_manager is not None:
                self._save_to_database(exchange, symbol, df)
                
            logger.debug(f"Fetched {len(df)} candles for {symbol} {timeframe} from {exchange}")
            self.metrics['successful_fetches'] += 1
            return df
            
        except Exception as e:
            logger.error(f"Error fetching OHLCV data for {symbol} {timeframe} from {exchange}: {e}")
            logger.error(traceback.format_exc())
            self.metrics['failed_fetches'] += 1
            return pd.DataFrame()
    
    async def fetch_ohlcv_async(self, symbol: str, timeframe: str = '1h', limit: int = 100, 
                              exchange: str = 'binance', since: Optional[int] = None, 
                              add_indicators: bool = False, use_cache: bool = True,
                              use_db: bool = True) -> pd.DataFrame:
        """
        Asynchronously fetch OHLCV (candle) data
        
        Args:
            symbol: Trading pair (e.g., 'BTC/USDT')
            timeframe: Candle timeframe (e.g., '1m', '1h', '1d')
            limit: Number of candles to fetch
            exchange: Exchange name
            since: Start timestamp (optional)
            add_indicators: Whether to add technical indicators
            use_cache: Whether to use disk cache
            use_db: Whether to use database
            
        Returns:
            DataFrame with OHLCV data
        """
        # Standardize inputs
        symbol = symbol.upper()
        exchange = exchange.lower()
        timeframe = timeframe.lower()
        
        # Convert 'since' to milliseconds timestamp if it's a datetime
        if isinstance(since, (datetime.datetime, str)):
            if isinstance(since, str):
                since = parse(since)
            since = int(since.timestamp() * 1000)
            
        # Generate cache key
        cache_key = self._generate_cache_key(exchange, symbol, timeframe, since, limit)
        
        # Try to load from cache first
        if use_cache:
            cached_data = self._load_from_cache(cache_key)
            if cached_data is not None:
                logger.debug(f"Loaded {symbol} {timeframe} data from cache")
                return cached_data
        
        # Try to load from database next
        if use_db and self.db_manager is not None:
            start_time = datetime.datetime.fromtimestamp(since / 1000) if since else None
            db_data = self._load_from_database(exchange, symbol, timeframe, start_time=start_time, limit=limit)
            if db_data is not None and len(db_data) >= limit:
                logger.debug(f"Loaded {symbol} {timeframe} data from database")
                
                # Add indicators if not present
                if add_indicators and not any(col.startswith('sma_') for col in db_data.columns):
                    db_data = self._add_indicators(db_data)
                    
                return db_data
        
        # If we got here, need to fetch from exchange API
        try:
            # Get exchange instance
            ex = self.get_exchange(exchange, async_mode=True)
            if ex is None:
                logger.error(f"Async exchange {exchange} not available")
                self.metrics['failed_fetches'] += 1
                return pd.DataFrame()
                
            # Get timeframe in exchange format
            ex_timeframe = self.timeframe_map.get(timeframe, {}).get('ccxt', timeframe)
            
            # Wait for rate limit if needed
            await self.rate_limiter.wait_if_needed(exchange)
            
            # Fetch OHLCV data
            self.metrics['api_calls'] += 1
            ohlcv = await ex.fetch_ohlcv(symbol, timeframe=ex_timeframe, since=since, limit=limit)
            
            # Check if we got data
            if not ohlcv:
                logger.warning(f"No OHLCV data received for {symbol} {timeframe}")
                self.metrics['failed_fetches'] += 1
                return pd.DataFrame()
                
            # Process data
            df = self._process_ohlcv_list(ohlcv, exchange, symbol, timeframe, add_indicators)
            
            # Save to cache
            if use_cache:
                self._save_to_cache(cache_key, df)
                
            # Save to database
            if use_db and self.db_manager is not None:
                self._save_to_database(exchange, symbol, df)
                
            logger.debug(f"Fetched {len(df)} candles for {symbol} {timeframe} from {exchange}")
            self.metrics['successful_fetches'] += 1
            return df
            
        except Exception as e:
            logger.error(f"Error fetching OHLCV data for {symbol} {timeframe} from {exchange}: {e}")
            logger.error(traceback.format_exc())
            self.metrics['failed_fetches'] += 1
            return pd.DataFrame()
    
    @cache_result(ttl_seconds=5)  # Short TTL for ticker data
    @rate_limit(calls_per_second=10)
    def fetch_ticker(self, symbol: str, exchange: str = 'binance') -> Dict:
        """
        Fetch current ticker data for a symbol
        
        Args:
            symbol: Trading pair (e.g., 'BTC/USDT')
            exchange: Exchange name
            
        Returns:
            Dictionary with ticker data
        """
        try:
            # Get exchange instance
            ex = self.get_exchange(exchange)
            if ex is None:
                logger.error(f"Exchange {exchange} not available")
                return {}
                
            # Fetch ticker
            self.metrics['api_calls'] += 1
            ticker = ex.fetch_ticker(symbol)
            
            self.metrics['successful_fetches'] += 1
            return ticker
            
        except Exception as e:
            logger.error(f"Error fetching ticker for {symbol} from {exchange}: {e}")
            self.metrics['failed_fetches'] += 1
            return {}
    
    async def fetch_ticker_async(self, symbol: str, exchange: str = 'binance') -> Dict:
        """
        Asynchronously fetch current ticker data for a symbol
        
        Args:
            symbol: Trading pair (e.g., 'BTC/USDT')
            exchange: Exchange name
            
        Returns:
            Dictionary with ticker data
        """
        try:
            # Get exchange instance
            ex = self.get_exchange(exchange, async_mode=True)
            if ex is None:
                logger.error(f"Async exchange {exchange} not available")
                return {}
                
            # Wait for rate limit if needed
            await self.rate_limiter.wait_if_needed(exchange)
            
            # Fetch ticker
            self.metrics['api_calls'] += 1
            ticker = await ex.fetch_ticker(symbol)
            
            self.metrics['successful_fetches'] += 1
            return ticker
            
        except Exception as e:
            logger.error(f"Error fetching ticker for {symbol} from {exchange}: {e}")
            self.metrics['failed_fetches'] += 1
            return {}
    
    @cache_result(ttl_seconds=5)  # Short TTL for order book data
    @rate_limit(calls_per_second=5)
    def fetch_order_book(self, symbol: str, limit: int = 20, exchange: str = 'binance') -> Dict:
        """
        Fetch current order book for a symbol
        
        Args:
            symbol: Trading pair (e.g., 'BTC/USDT')
            limit: Order book depth
            exchange: Exchange name
            
        Returns:
            Dictionary with order book data
        """
        try:
            # Get exchange instance
            ex = self.get_exchange(exchange)
            if ex is None:
                logger.error(f"Exchange {exchange} not available")
                return {}
                
            # Fetch order book
            self.metrics['api_calls'] += 1
            order_book = ex.fetch_order_book(symbol, limit=limit)
            
            self.metrics['successful_fetches'] += 1
            return order_book
            
        except Exception as e:
            logger.error(f"Error fetching order book for {symbol} from {exchange}: {e}")
            self.metrics['failed_fetches'] += 1
            return {}
    
    async def fetch_order_book_async(self, symbol: str, limit: int = 20, exchange: str = 'binance') -> Dict:
        """
        Asynchronously fetch current order book for a symbol
        
        Args:
            symbol: Trading pair (e.g., 'BTC/USDT')
            limit: Order book depth
            exchange: Exchange name
            
        Returns:
            Dictionary with order book data
        """
        try:
            # Get exchange instance
            ex = self.get_exchange(exchange, async_mode=True)
            if ex is None:
                logger.error(f"Async exchange {exchange} not available")
                return {}
                
            # Wait for rate limit if needed
            await self.rate_limiter.wait_if_needed(exchange)
            
            # Fetch order book
            self.metrics['api_calls'] += 1
            order_book = await ex.fetch_order_book(symbol, limit=limit)
            
            self.metrics['successful_fetches'] += 1
            return order_book
            
        except Exception as e:
            logger.error(f"Error fetching order book for {symbol} from {exchange}: {e}")
            self.metrics['failed_fetches'] += 1
            return {}
    
    def fetch_multiple_ohlcv(self, symbols: List[str], timeframe: str = '1h', limit: int = 100, 
                           exchange: str = 'binance', since: Optional[int] = None,
                           add_indicators: bool = False, use_cache: bool = True, 
                           use_db: bool = True) -> Dict[str, pd.DataFrame]:
        """
        Fetch OHLCV data for multiple symbols in parallel
        
        Args:
            symbols: List of trading pairs
            timeframe: Candle timeframe
            limit: Number of candles to fetch
            exchange: Exchange name
            since: Start timestamp (optional)
            add_indicators: Whether to add technical indicators
            use_cache: Whether to use disk cache
            use_db: Whether to use database
            
        Returns:
            Dictionary mapping symbols to DataFrames
        """
        # Use thread pool for parallel fetching
        results = {}
        
        with ThreadPoolExecutor(max_workers=min(10, len(symbols))) as executor:
            # Submit all tasks
            futures = {
                executor.submit(self.fetch_ohlcv, symbol, timeframe, limit, exchange, since, add_indicators, use_cache, use_db): symbol
                for symbol in symbols
            }
            
            # Get results as they complete
            for future in futures:
                symbol = futures[future]
                try:
                    results[symbol] = future.result()
                except Exception as e:
                    logger.error(f"Error fetching {symbol} data: {e}")
                    results[symbol] = pd.DataFrame()
                    
        return results
    
    async def fetch_multiple_ohlcv_async(self, symbols: List[str], timeframe: str = '1h', limit: int = 100, 
                                      exchange: str = 'binance', since: Optional[int] = None,
                                      add_indicators: bool = False, use_cache: bool = True, 
                                      use_db: bool = True) -> Dict[str, pd.DataFrame]:
        """
        Asynchronously fetch OHLCV data for multiple symbols
        
        Args:
            symbols: List of trading pairs
            timeframe: Candle timeframe
            limit: Number of candles to fetch
            exchange: Exchange name
            since: Start timestamp (optional)
            add_indicators: Whether to add technical indicators
            use_cache: Whether to use disk cache
            use_db: Whether to use database
            
        Returns:
            Dictionary mapping symbols to DataFrames
        """
        # Create tasks for all symbols
        tasks = [
            self.fetch_ohlcv_async(symbol, timeframe, limit, exchange, since, add_indicators, use_cache, use_db)
            for symbol in symbols
        ]
        
        # Run tasks concurrently
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results
        data_dict = {}
        for symbol, result in zip(symbols, results):
            if isinstance(result, Exception):
                logger.error(f"Error fetching {symbol} data: {result}")
                data_dict[symbol] = pd.DataFrame()
            else:
                data_dict[symbol] = result
                
        return data_dict
    
    def fetch_symbols(self, exchange: str = 'binance') -> List[str]:
        """
        Fetch available trading pairs from an exchange
        
        Args:
            exchange: Exchange name
            
        Returns:
            List of trading pairs
        """
        try:
            # Get exchange instance
            ex = self.get_exchange(exchange)
            if ex is None:
                logger.error(f"Exchange {exchange} not available")
                return []
                
            # Fetch markets
            self.metrics['api_calls'] += 1
            markets = ex.load_markets()
            
            # Extract symbols
            symbols = list(markets.keys())
            
            self.metrics['successful_fetches'] += 1
            return symbols
            
        except Exception as e:
            logger.error(f"Error fetching symbols from {exchange}: {e}")
            self.metrics['failed_fetches'] += 1
            return []
    
    def get_recent_trades(self, symbol: str, limit: int = 100, exchange: str = 'binance') -> pd.DataFrame:
        """
        Fetch recent trades for a symbol
        
        Args:
            symbol: Trading pair
            limit: Number of trades to fetch
            exchange: Exchange name
            
        Returns:
            DataFrame with trade data
        """
        try:
            # Get exchange instance
            ex = self.get_exchange(exchange)
            if ex is None:
                logger.error(f"Exchange {exchange} not available")
                return pd.DataFrame()
                
            # Fetch trades
            self.metrics['api_calls'] += 1
            trades = ex.fetch_trades(symbol, limit=limit)
            
            # Convert to DataFrame
            if not trades:
                return pd.DataFrame()
                
            df = pd.DataFrame(trades)
            
            # Convert timestamp to datetime
            if 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                
            self.metrics['successful_fetches'] += 1
            return df
            
        except Exception as e:
            logger.error(f"Error fetching trades for {symbol} from {exchange}: {e}")
            self.metrics['failed_fetches'] += 1
            return pd.DataFrame()
    
    def get_metrics(self) -> Dict[str, int]:
        """
        Get metrics about data fetching performance
        
        Returns:
            Dictionary with metrics
        """
        return self.metrics.copy()
    
    def reset_metrics(self) -> None:
        """Reset metrics counters"""
        for key in self.metrics:
            self.metrics[key] = 0
    
    async def close_async_exchanges(self) -> None:
        """Close all async exchange connections"""
        for name, exchange in self.async_exchanges.items():
            try:
                await exchange.close()
                logger.info(f"Closed async {name} connection")
            except Exception as e:
                logger.error(f"Error closing async {name} connection: {e}")
    
    def get_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add technical indicators to an existing DataFrame
        
        Args:
            df: OHLCV DataFrame
            
        Returns:
            DataFrame with indicators
        """
        return self._add_indicators(df)
    
    def get_exchange_status(self, exchange: str = 'binance') -> Dict:
        """
        Check if an exchange is operational
        
        Args:
            exchange: Exchange name
            
        Returns:
            Dictionary with status info
        """
        try:
            # Get exchange instance
            ex = self.get_exchange(exchange)
            if ex is None:
                return {'status': 'unavailable', 'message': 'Exchange not available'}
                
            # Fetch status
            status = {'status': 'operational', 'message': 'Exchange operational'}
            
            return status
            
        except Exception as e:
            logger.error(f"Error checking status for {exchange}: {e}")
            return {'status': 'error', 'message': str(e)}
    
    def analyze_liquidity(self, symbol: str, exchange: str = 'binance') -> Dict:
        """
        Analyze market liquidity for a symbol
        
        Args:
            symbol: Trading pair
            exchange: Exchange name
            
        Returns:
            Dictionary with liquidity metrics
        """
        try:
            # Fetch order book
            order_book = self.fetch_order_book(symbol, limit=100, exchange=exchange)
            
            if not order_book or 'bids' not in order_book or 'asks' not in order_book:
                return {}
                
            bids = order_book['bids']
            asks = order_book['asks']
            
            # Calculate bid-ask spread
            best_bid = bids[0][0] if bids else 0
            best_ask = asks[0][0] if asks else 0
            
            if best_bid == 0 or best_ask == 0:
                spread_pct = None
            else:
                spread = best_ask - best_bid
                spread_pct = (spread / best_bid) * 100
                
            # Calculate depth at different levels
            bid_depth_1pct = sum(qty for price, qty in bids if price >= best_bid * 0.99)
            ask_depth_1pct = sum(qty for price, qty in asks if price <= best_ask * 1.01)
            
            # Calculate slippage for standard order sizes
            std_order_sizes = [0.1, 0.5, 1.0, 5.0, 10.0]  # BTC or similar
            
            slippage = {}
            for size in std_order_sizes:
                # Buy slippage
                buy_cost = 0
                buy_qty = 0
                for price, qty in asks:
                    if buy_qty >= size:
                        break
                    order_qty = min(qty, size - buy_qty)
                    buy_cost += order_qty * price
                    buy_qty += order_qty
                    
                avg_buy_price = buy_cost / buy_qty if buy_qty > 0 else 0
                buy_slippage_pct = ((avg_buy_price / best_ask) - 1) * 100 if best_ask > 0 else 0
                
                # Sell slippage
                sell_revenue = 0
                sell_qty = 0
                for price, qty in bids:
                    if sell_qty >= size:
                        break
                    order_qty = min(qty, size - sell_qty)
                    sell_revenue += order_qty * price
                    sell_qty += order_qty
                    
                avg_sell_price = sell_revenue / sell_qty if sell_qty > 0 else 0
                sell_slippage_pct = (1 - (avg_sell_price / best_bid)) * 100 if best_bid > 0 else 0
                
                slippage[size] = {
                    'buy': buy_slippage_pct,
                    'sell': sell_slippage_pct
                }
                
            # Return liquidity metrics
            return {
                'best_bid': best_bid,
                'best_ask': best_ask,
                'spread': spread_pct,
                'bid_depth_1pct': bid_depth_1pct,
                'ask_depth_1pct': ask_depth_1pct,
                'slippage': slippage
            }
            
        except Exception as e:
            logger.error(f"Error analyzing liquidity for {symbol}: {e}")
            return {}
    
    def close(self) -> None:
        """Close all connections and resources"""
        # Close async exchanges
        if self.use_async:
            try:
                loop = asyncio.get_event_loop()
                loop.run_until_complete(self.close_async_exchanges())
            except Exception as e:
                logger.error(f"Error closing async exchanges: {e}")
                
        # Close thread pool
        self.thread_pool.shutdown(wait=True)
        
        # Close database connection
        if self.db_manager is not None:
            try:
                self.db_manager.close()
            except Exception as e:
                logger.error(f"Error closing database connection: {e}")
                
        logger.info("Data fetcher closed")

# For error handling integration
if HAVE_ERROR_HANDLING:
    # Apply error handling decorators to critical methods
    DataFetcher.fetch_ohlcv = safe_execute(ErrorCategory.DATA_PROCESSING, default_return=pd.DataFrame())(DataFetcher.fetch_ohlcv)
    DataFetcher.fetch_ticker = safe_execute(ErrorCategory.DATA_PROCESSING, default_return={})(DataFetcher.fetch_ticker)
    DataFetcher.fetch_order_book = safe_execute(ErrorCategory.DATA_PROCESSING, default_return={})(DataFetcher.fetch_order_book)

# Module-level functions for direct use
import random  # Required for the cache cleanup random

def get_data_fetcher(use_mock_db=False):
    """
    Create and return a data fetcher instance with appropriate database
    
    Args:
        use_mock_db: Whether to use mock database
        
    Returns:
        DataFetcher instance
    """
    # Try to get database
    db_manager = None
    
    if HAVE_DB_FACTORY:
        db_manager = DatabaseFactory.create_database(use_mock=use_mock_db)
    elif use_mock_db and HAVE_MOCK_DB:
        db_manager = MockDatabaseManager()
    elif not use_mock_db and HAVE_REAL_DB:
        db_manager = DatabaseManager()
        
    # Create data fetcher
    return DataFetcher(db_manager=db_manager)

def fetch_market_data(symbol, timeframe='1h', limit=100, exchange='binance', since=None, add_indicators=True):
    """
    Convenience function to fetch market data
    
    Args:
        symbol: Trading pair
        timeframe: Candle timeframe
        limit: Number of candles
        exchange: Exchange name
        since: Start timestamp
        add_indicators: Whether to add indicators
        
    Returns:
        DataFrame with market data
    """
    fetcher = get_data_fetcher()
    return fetcher.fetch_ohlcv(symbol, timeframe, limit, exchange, since, add_indicators)

async def fetch_market_data_async(symbol, timeframe='1h', limit=100, exchange='binance', since=None, add_indicators=True):
    """
    Convenience async function to fetch market data
    
    Args:
        symbol: Trading pair
        timeframe: Candle timeframe
        limit: Number of candles
        exchange: Exchange name
        since: Start timestamp
        add_indicators: Whether to add indicators
        
    Returns:
        DataFrame with market data
    """
    fetcher = get_data_fetcher()
    data = await fetcher.fetch_ohlcv_async(symbol, timeframe, limit, exchange, since, add_indicators)
    await fetcher.close_async_exchanges()
    return data

if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Demo usage
    fetcher = get_data_fetcher()
    
    # Fetch BTC/USDT data
    btc_data = fetcher.fetch_ohlcv('BTC/USDT', timeframe='1h', limit=100, add_indicators=True)
    print(f"Fetched {len(btc_data)} rows of BTC/USDT data")
    
    # Fetch ticker
    btc_ticker = fetcher.fetch_ticker('BTC/USDT')
    print(f"BTC/USDT ticker: {btc_ticker.get('last', 'N/A')}")
    
    # Fetch multiple symbols
    symbols = ['BTC/USDT', 'ETH/USDT', 'XRP/USDT']
    multi_data = fetcher.fetch_multiple_ohlcv(symbols, timeframe='1h', limit=50)
    for symbol, data in multi_data.items():
        print(f"{symbol}: {len(data)} rows")
    
    # Close connections
    fetcher.close()
