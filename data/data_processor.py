# data_processor.py

import os
import numpy as np
import pandas as pd
import logging
import time
from typing import Dict, List, Optional, Tuple, Union, Callable, Any
from datetime import datetime, timedelta
import threading
import concurrent.futures
from functools import lru_cache
import warnings

# Try importing optional dependencies
try:
    import talib
    TALIB_AVAILABLE = True
except ImportError:
    TALIB_AVAILABLE = False
    warnings.warn("TA-Lib not available. Using pandas-based calculations as fallback.")

try:
    from error_handling import safe_execute, ErrorCategory, ErrorSeverity
    CUSTOM_ERROR_HANDLING = True
except ImportError:
    CUSTOM_ERROR_HANDLING = False
    warnings.warn("Custom error handling module not available. Using standard error handling.")

# Configure logging
logger = logging.getLogger(__name__)


class DataProcessor:
    """
    Comprehensive data processing pipeline for market data.
    
    Handles loading, cleaning, feature engineering, and specialized 
    transformations for trading strategies and ML models.
    """
    
    def __init__(
        self, 
        base_timeframe: str = '1h',
        feature_set: str = 'standard',
        cache_size: int = 100,
        use_parallel: bool = True,
        max_workers: int = None,
        db_manager = None
    ):
        """
        Initialize the data processor.
        
        Args:
            base_timeframe (str): Base timeframe for data processing ('1m', '1h', '1d', etc.)
            feature_set (str): Feature set to generate ('minimal', 'standard', 'extended', 'ml')
            cache_size (int): Size of LRU cache for processed data
            use_parallel (bool): Whether to use parallel processing for heavy computations
            max_workers (int): Maximum number of worker threads (None = auto)
            db_manager: Optional database manager for data persistence
        """
        self.base_timeframe = base_timeframe
        self.feature_set = feature_set
        self.cache_size = cache_size
        self.use_parallel = use_parallel
        self.max_workers = max_workers
        self.db_manager = db_manager
        
        # Data caches with locks for thread safety
        self._raw_data_cache = {}
        self._processed_data_cache = {}
        self._feature_cache = {}
        self._cache_lock = threading.RLock()
        
        # Thread pool for parallel processing
        if self.use_parallel:
            self.thread_pool = concurrent.futures.ThreadPoolExecutor(max_workers=max_workers)
        
        # Timeframe resolution in minutes
        self.tf_resolution = {
            '1m': 1, '3m': 3, '5m': 5, '15m': 15, '30m': 30,
            '1h': 60, '2h': 120, '4h': 240, '6h': 360, '8h': 480, '12h': 720,
            '1d': 1440, '3d': 4320, '1w': 10080, '1M': 43200
        }
        
        # Feature calculation functions by category
        self.feature_functions = {
            'trend': self._calculate_trend_features,
            'momentum': self._calculate_momentum_features,
            'volatility': self._calculate_volatility_features,
            'volume': self._calculate_volume_features,
            'support_resistance': self._calculate_support_resistance,
            'pattern': self._calculate_pattern_features,
            'sentiment': self._calculate_sentiment_features,
            'custom': self._calculate_custom_features
        }
        
        # Feature sets (combinations of feature categories)
        self.feature_sets = {
            'minimal': ['trend', 'volatility'],
            'standard': ['trend', 'momentum', 'volatility', 'volume'],
            'extended': ['trend', 'momentum', 'volatility', 'volume', 'support_resistance', 'pattern'],
            'ml': ['trend', 'momentum', 'volatility', 'volume', 'support_resistance', 'pattern', 'sentiment', 'custom']
        }
        
        # Initialize worker status
        self._system_ready = True
        logger.info(f"DataProcessor initialized with {feature_set} feature set and {base_timeframe} base timeframe")
        
    def process_data(
        self, 
        data: Union[pd.DataFrame, str, Dict[str, Any]],
        symbol: str,
        timeframe: Optional[str] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        include_indicators: bool = True,
        fillna: bool = True,
        for_training: bool = False,
        custom_features: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Process data from various sources and generate trading features.
        
        Args:
            data: DataFrame, file path, or API parameters for data loading
            symbol: Trading pair symbol (e.g., 'BTC/USDT')
            timeframe: Target timeframe for processing (defaults to base_timeframe)
            start_time: Start time for data filtering
            end_time: End time for data filtering
            include_indicators: Whether to include technical indicators
            fillna: Whether to fill missing values
            for_training: Whether to prepare data for ML training
            custom_features: Optional list of custom features to include
        
        Returns:
            Processed DataFrame with requested features
        """
        timeframe = timeframe or self.base_timeframe
        cache_key = f"{symbol}_{timeframe}_{start_time}_{end_time}"
        
        # Check cache first for processed data
        with self._cache_lock:
            if cache_key in self._processed_data_cache:
                cached_data = self._processed_data_cache[cache_key]
                logger.debug(f"Using cached processed data for {symbol} {timeframe}")
                return cached_data.copy()
        
        # Load and prepare data
        try:
            # Load data from source
            df = self._load_data(data, symbol, timeframe, start_time, end_time)
            
            if df is None or df.empty:
                logger.warning(f"No data available for {symbol} {timeframe}")
                return pd.DataFrame()
            
            # Ensure DataFrame has proper OHLCV columns
            df = self._validate_and_standardize_dataframe(df)
            
            # Store raw data in cache
            with self._cache_lock:
                self._raw_data_cache[cache_key] = df.copy()
            
            # Clean and preprocess data
            df = self._preprocess_data(df, fillna=fillna)
            
            # Generate requested features
            if include_indicators:
                df = self._generate_features(df, custom_features)
            
            # Prepare for ML training if requested
            if for_training:
                df = self._prepare_for_ml(df)
            
            # Cache processed result
            with self._cache_lock:
                self._processed_data_cache[cache_key] = df.copy()
                
                # Limit cache size
                if len(self._processed_data_cache) > self.cache_size:
                    # Remove oldest item
                    oldest_key = next(iter(self._processed_data_cache))
                    del self._processed_data_cache[oldest_key]
            
            return df
            
        except Exception as e:
            error_msg = f"Error processing data for {symbol} {timeframe}: {str(e)}"
            
            if CUSTOM_ERROR_HANDLING:
                from error_handling import TradingSystemError
                logger.error(error_msg)
                raise TradingSystemError(
                    message=error_msg,
                    category=ErrorCategory.DATA_PROCESSING,
                    severity=ErrorSeverity.ERROR,
                    original_exception=e
                )
            else:
                logger.error(error_msg)
                raise RuntimeError(error_msg) from e
    
    def _load_data(
        self, 
        data: Union[pd.DataFrame, str, Dict[str, Any]],
        symbol: str,
        timeframe: str,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None
    ) -> pd.DataFrame:
        """
        Load market data from various sources.
        
        Args:
            data: Data source (DataFrame, file path, or API parameters)
            symbol: Trading pair symbol
            timeframe: Target timeframe
            start_time: Start time for filtering
            end_time: End time for filtering
            
        Returns:
            DataFrame with standardized market data
        """
        # If data is already a DataFrame, use it directly
        if isinstance(data, pd.DataFrame):
            df = data.copy()
            
        # If data is a string, treat it as a file path
        elif isinstance(data, str):
            if os.path.exists(data):
                if data.endswith('.csv'):
                    df = pd.read_csv(data, parse_dates=True)
                elif data.endswith('.parquet'):
                    df = pd.read_parquet(data)
                elif data.endswith('.pickle') or data.endswith('.pkl'):
                    df = pd.read_pickle(data)
                else:
                    raise ValueError(f"Unsupported file format: {data}")
            else:
                raise FileNotFoundError(f"File not found: {data}")
                
        # If data is a dictionary, treat it as API parameters
        elif isinstance(data, dict):
            # Try to fetch from database if manager is available
            if self.db_manager is not None:
                try:
                    df = self.db_manager.get_market_data(
                        symbol=symbol,
                        timeframe=timeframe,
                        limit=data.get('limit', 1000),
                        start_time=start_time,
                        end_time=end_time
                    )
                except Exception as e:
                    logger.error(f"Database fetch failed: {e}, trying backup methods")
                    df = self._fetch_from_api(symbol, timeframe, start_time, end_time, data)
            else:
                df = self._fetch_from_api(symbol, timeframe, start_time, end_time, data)
                
        else:
            raise ValueError("Data must be a DataFrame, file path, or API parameters")
            
        # Apply time filters if provided and not already applied
        if not df.empty:
            if isinstance(df.index, pd.DatetimeIndex):
                time_indexed = True
            else:
                time_indexed = False
                # Try to convert 'time' or 'timestamp' column to datetime index
                for col in ['time', 'timestamp', 'date']:
                    if col in df.columns:
                        try:
                            df[col] = pd.to_datetime(df[col])
                            df.set_index(col, inplace=True)
                            time_indexed = True
                            break
                        except Exception:
                            pass
            
            if time_indexed:
                if start_time:
                    df = df[df.index >= pd.to_datetime(start_time)]
                if end_time:
                    df = df[df.index <= pd.to_datetime(end_time)]
                    
            # Sort by time
            df = df.sort_index()
            
        return df
    
    def _fetch_from_api(
        self, 
        symbol: str, 
        timeframe: str, 
        start_time: Optional[datetime], 
        end_time: Optional[datetime],
        params: Dict[str, Any]
    ) -> pd.DataFrame:
        """
        Fetch data from exchange API using ccxt.
        
        This method attempts to load historical market data from an exchange API
        with proper error handling and retry logic.
        
        Args:
            symbol: Trading pair symbol
            timeframe: Target timeframe
            start_time: Start time for filtering
            end_time: End time for filtering
            params: Additional API parameters
            
        Returns:
            DataFrame with API data
        """
        try:
            import ccxt
            
            # Get exchange name from params or use default
            exchange_name = params.get('exchange', 'binance')
            
            # Initialize exchange
            exchange_class = getattr(ccxt, exchange_name)
            exchange = exchange_class({
                'enableRateLimit': True,
                'options': {'defaultType': 'future'}  # Use future market by default
            })
            
            # Set timeframe if supported
            if timeframe not in exchange.timeframes:
                available_timeframes = list(exchange.timeframes.keys())
                logger.warning(f"Timeframe {timeframe} not supported by {exchange_name}. Available: {available_timeframes}")
                # Try to find closest timeframe
                closest = self._find_closest_timeframe(timeframe, available_timeframes)
                logger.info(f"Using closest available timeframe: {closest}")
                timeframe = closest
            
            # Calculate start/end times in milliseconds
            since = int(start_time.timestamp() * 1000) if start_time else None
            until = int(end_time.timestamp() * 1000) if end_time else None
            
            # Initialize result accumulator
            all_candles = []
            
            # Paginate API requests if needed
            limit = params.get('limit', 1000)
            current_since = since
            
            # Set maximum number of API requests to prevent infinite loops
            max_requests = params.get('max_requests', 10)
            request_count = 0
            
            while request_count < max_requests:
                request_count += 1
                
                try:
                    # Fetch OHLCV data from exchange
                    candles = exchange.fetch_ohlcv(
                        symbol=symbol,
                        timeframe=timeframe,
                        since=current_since,
                        limit=limit
                    )
                    
                    if not candles or len(candles) == 0:
                        break
                        
                    all_candles.extend(candles)
                    
                    # Update since for next request
                    last_candle_time = candles[-1][0]
                    if last_candle_time == current_since:
                        break  # No new data
                    current_since = last_candle_time + 1
                    
                    # Check if we've reached the end time
                    if until and current_since >= until:
                        break
                        
                    # Respect rate limits
                    time.sleep(exchange.rateLimit / 1000)
                    
                except ccxt.ExchangeError as e:
                    logger.error(f"Exchange error: {e}")
                    break
                except Exception as e:
                    logger.error(f"API fetch error: {e}")
                    break
            
            # Convert to DataFrame
            if not all_candles:
                return pd.DataFrame()
                
            df = pd.DataFrame(
                all_candles,
                columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']
            )
            
            # Convert timestamp to datetime index
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            
            return df
            
        except ImportError:
            logger.error("ccxt library not available for API fetching")
            return pd.DataFrame()
        except Exception as e:
            logger.error(f"Error fetching data from API: {e}")
            return pd.DataFrame()
    
    def _find_closest_timeframe(self, target: str, available: List[str]) -> str:
        """Find the closest available timeframe to the target."""
        if target in available:
            return target
            
        # Get target minutes
        target_minutes = self.tf_resolution.get(target)
        if not target_minutes:
            return available[0]  # Default to first available
            
        # Find closest match
        closest = available[0]
        min_diff = float('inf')
        
        for tf in available:
            tf_minutes = self.tf_resolution.get(tf)
            if not tf_minutes:
                continue
                
            diff = abs(tf_minutes - target_minutes)
            if diff < min_diff:
                min_diff = diff
                closest = tf
                
        return closest
    
    def _validate_and_standardize_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Ensure DataFrame has proper OHLCV columns and formatting.
        
        Args:
            df: Input DataFrame
            
        Returns:
            Standardized DataFrame with OHLCV columns
        """
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        
        # Check for required columns (case-insensitive)
        df_columns_lower = [col.lower() for col in df.columns]
        
        # Create mapping from standardized column names to actual column names
        col_mapping = {}
        for std_col in required_columns:
            for i, col in enumerate(df_columns_lower):
                if col == std_col or col.endswith(std_col):
                    col_mapping[std_col] = df.columns[i]
                    break
        
        # Check if all required columns were found
        missing_columns = [col for col in required_columns if col not in col_mapping]
        
        if missing_columns:
            # Try to infer missing columns if possible
            if 'close' in col_mapping and 'open' in missing_columns:
                col_mapping['open'] = col_mapping['close']
                missing_columns.remove('open')
                
            if 'high' in missing_columns and 'low' in missing_columns and 'close' in col_mapping:
                col_mapping['high'] = col_mapping['close']
                col_mapping['low'] = col_mapping['close']
                missing_columns.remove('high')
                missing_columns.remove('low')
                
            if 'volume' in missing_columns:
                # Create dummy volume column
                df['volume'] = 0
                col_mapping['volume'] = 'volume'
                missing_columns.remove('volume')
                
            if missing_columns:
                raise ValueError(f"Missing required columns: {missing_columns}")
        
        # Standardize column names
        renamed_columns = {col_mapping[std_col]: std_col for std_col in required_columns}
        df = df.rename(columns=renamed_columns)
        
        # Ensure numeric data types
        for col in required_columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            
        # Ensure DataFrame is properly sorted
        if isinstance(df.index, pd.DatetimeIndex):
            df = df.sort_index()
        
        return df
    
    def _preprocess_data(self, df: pd.DataFrame, fillna: bool = True) -> pd.DataFrame:
        """
        Clean and preprocess market data.
        
        Args:
            df: Input DataFrame
            fillna: Whether to fill missing values
            
        Returns:
            Preprocessed DataFrame
        """
        # Make a copy to avoid modifying the original
        df = df.copy()
        
        # Remove duplicates
        df = df[~df.index.duplicated(keep='last')]
        
        # Handle missing values
        if fillna:
            # Forward fill most values
            df.fillna(method='ffill', inplace=True)
            
            # For remaining NaNs, use reasonable defaults
            df['open'].fillna(df['close'], inplace=True)
            df['high'].fillna(df['close'], inplace=True)
            df['low'].fillna(df['close'], inplace=True)
            df['close'].fillna(method='bfill', inplace=True)
            df['volume'].fillna(0, inplace=True)
        
        # Ensure all OHLC values are consistent
        df['high'] = df[['high', 'open', 'close']].max(axis=1)
        df['low'] = df[['low', 'open', 'close']].min(axis=1)
        
        # Add percentage change
        df['returns'] = df['close'].pct_change()
        df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
        
        # Add date features
        if isinstance(df.index, pd.DatetimeIndex):
            df['day_of_week'] = df.index.dayofweek
            df['hour_of_day'] = df.index.hour
            df['month'] = df.index.month
            
        return df
    
    def _generate_features(
        self, 
        df: pd.DataFrame, 
        custom_features: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Generate trading features based on configuration.
        
        Args:
            df: Preprocessed DataFrame
            custom_features: Optional list of custom features to include
            
        Returns:
            DataFrame with added features
        """
        if df.empty:
            return df
        
        # Get feature categories for current feature set
        if self.feature_set in self.feature_sets:
            categories = self.feature_sets[self.feature_set]
        else:
            # Default to standard features
            categories = self.feature_sets['standard']
            
        # Make a copy of the DataFrame
        df_with_features = df.copy()
        
        # Generate features in parallel if enabled
        if self.use_parallel and len(df) > 1000:
            # Submit feature calculation tasks
            futures = {}
            for category in categories:
                if category in self.feature_functions:
                    func = self.feature_functions[category]
                    futures[category] = self.thread_pool.submit(func, df)
                    
            # Collect results
            for category, future in futures.items():
                try:
                    category_df = future.result(timeout=30)  # 30 second timeout
                    if category_df is not None and not category_df.empty:
                        # Merge with main DataFrame
                        df_with_features = pd.concat([df_with_features, category_df], axis=1)
                except concurrent.futures.TimeoutError:
                    logger.warning(f"Feature calculation timeout for category '{category}'")
                except Exception as e:
                    logger.error(f"Error calculating {category} features: {e}")
        else:
            # Sequential calculation
            for category in categories:
                if category in self.feature_functions:
                    try:
                        func = self.feature_functions[category]
                        category_df = func(df)
                        if category_df is not None and not category_df.empty:
                            # Merge with main DataFrame
                            df_with_features = pd.concat([df_with_features, category_df], axis=1)
                    except Exception as e:
                        logger.error(f"Error calculating {category} features: {e}")
        
        # Add custom features if requested
        if custom_features and 'custom' in categories:
            try:
                custom_df = self._calculate_custom_features(df, feature_list=custom_features)
                if custom_df is not None and not custom_df.empty:
                    df_with_features = pd.concat([df_with_features, custom_df], axis=1)
            except Exception as e:
                logger.error(f"Error calculating custom features: {e}")
                
        # Remove duplicate columns (in case of overlaps between feature categories)
        df_with_features = df_with_features.loc[:, ~df_with_features.columns.duplicated()]
        
        return df_with_features
    
    def _calculate_trend_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate trend-related features.
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with trend features
        """
        if df.empty:
            return pd.DataFrame()
            
        # Initialize result DataFrame
        result = pd.DataFrame(index=df.index)
        
        # Simple Moving Averages
        for period in [5, 10, 20, 50, 100, 200]:
            if len(df) >= period:
                col_name = f'sma_{period}'
                result[col_name] = df['close'].rolling(window=period).mean()
                
                # SMA-based features
                if period in [20, 50]:
                    # Price relative to SMA
                    result[f'price_to_sma_{period}'] = df['close'] / result[col_name]
                    
                    # SMA Crossovers (lagging by design)
                    if period == 50 and 'sma_20' in result.columns:
                        result['sma_20_50_cross'] = np.where(
                            result['sma_20'] > result['sma_50'], 1, 
                            np.where(result['sma_20'] < result['sma_50'], -1, 0)
                        )
        
        # Exponential Moving Averages
        for period in [5, 12, 26, 50, 200]:
            if len(df) >= period:
                col_name = f'ema_{period}'
                
                if TALIB_AVAILABLE:
                    try:
                        result[col_name] = talib.EMA(df['close'].values, timeperiod=period)
                    except Exception:
                        result[col_name] = df['close'].ewm(span=period, adjust=False).mean()
                else:
                    result[col_name] = df['close'].ewm(span=period, adjust=False).mean()
                    
                # EMA-based features
                if period in [12, 26]:
                    result[f'price_to_ema_{period}'] = df['close'] / result[col_name]
        
        # MACD
        if 'ema_12' in result.columns and 'ema_26' in result.columns:
            result['macd'] = result['ema_12'] - result['ema_26']
            result['macd_signal'] = result['macd'].ewm(span=9, adjust=False).mean()
            result['macd_hist'] = result['macd'] - result['macd_signal']
        
        # Ichimoku Cloud (simplified)
        if len(df) >= 52:
            # Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2
            tenkan_sen_high = df['high'].rolling(window=9).max()
            tenkan_sen_low = df['low'].rolling(window=9).min()
            result['tenkan_sen'] = (tenkan_sen_high + tenkan_sen_low) / 2
            
            # Kijun-sen (Base Line): (26-period high + 26-period low)/2
            kijun_sen_high = df['high'].rolling(window=26).max()
            kijun_sen_low = df['low'].rolling(window=26).min()
            result['kijun_sen'] = (kijun_sen_high + kijun_sen_low) / 2
            
            # Senkou Span A (Leading Span A): (Conversion Line + Base Line)/2
            result['senkou_span_a'] = ((result['tenkan_sen'] + result['kijun_sen']) / 2).shift(26)
            
            # Senkou Span B (Leading Span B): (52-period high + 52-period low)/2
            senkou_span_b_high = df['high'].rolling(window=52).max()
            senkou_span_b_low = df['low'].rolling(window=52).min()
            result['senkou_span_b'] = ((senkou_span_b_high + senkou_span_b_low) / 2).shift(26)
            
            # Chikou Span (Lagging Span): Close price shifted back 26 periods
            result['chikou_span'] = df['close'].shift(-26)
        
        # Trend strength indicators
        if 'ema_50' in result.columns and 'ema_200' in result.columns:
            # Golden/Death Cross
            result['golden_cross'] = np.where(
                (result['ema_50'] > result['ema_200']) & 
                (result['ema_50'].shift(1) <= result['ema_200'].shift(1)), 
                1, 0
            )
            
            result['death_cross'] = np.where(
                (result['ema_50'] < result['ema_200']) & 
                (result['ema_50'].shift(1) >= result['ema_200'].shift(1)), 
                1, 0
            )
            
            # ADX (Average Directional Index) - simplified version
            plus_dm = df['high'].diff()
            minus_dm = df['low'].diff(-1) * -1
            
            plus_dm = np.where((plus_dm > minus_dm) & (plus_dm > 0), plus_dm, 0)
            minus_dm = np.where((minus_dm > plus_dm) & (minus_dm > 0), minus_dm, 0)
            
            tr = np.maximum(
                df['high'] - df['low'],
                np.maximum(
                    abs(df['high'] - df['close'].shift(1)),
                    abs(df['low'] - df['close'].shift(1))
                )
            )
            
            atr_14 = pd.Series(tr).rolling(window=14).mean().values
            
            plus_di = 100 * pd.Series(plus_dm).rolling(window=14).mean() / atr_14
            minus_di = 100 * pd.Series(minus_dm).rolling(window=14).mean() / atr_14
            
            dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
            result['adx'] = pd.Series(dx).rolling(window=14).mean()
            
            # Trend direction based on ADX
            result['trend_strength'] = np.where(
                result['adx'] > 25,
                np.where(plus_di > minus_di, 1, -1),
                0
            )
        
        return result
    
    def _calculate_momentum_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate momentum-related features.
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with momentum features
        """
        if df.empty:
            return pd.DataFrame()
            
        # Initialize result DataFrame
        result = pd.DataFrame(index=df.index)
        
        # RSI (Relative Strength Index)
        if TALIB_AVAILABLE:
            try:
                result['rsi_14'] = talib.RSI(df['close'].values, timeperiod=14)
            except Exception as e:
                logger.warning(f"TA-Lib RSI calculation failed: {e}. Using pandas.")
                rsi_pandas = True
        else:
            rsi_pandas = True
            
        if not TALIB_AVAILABLE or rsi_pandas:
            # Calculate RSI using pandas
            delta = df['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            
            rs = gain / loss.replace(0, np.finfo(float).eps)  # Avoid division by zero
            result['rsi_14'] = 100 - (100 / (1 + rs))
        
        # Stochastic Oscillator
        if TALIB_AVAILABLE:
            try:
                result['stoch_k'], result['stoch_d'] = talib.STOCH(
                    df['high'].values, df['low'].values, df['close'].values,
                    fastk_period=14, slowk_period=3, slowd_period=3
                )
            except Exception as e:
                logger.warning(f"TA-Lib Stochastic calculation failed: {e}. Using pandas.")
                stoch_pandas = True
        else:
            stoch_pandas = True
            
        if not TALIB_AVAILABLE or stoch_pandas:
            # Calculate Stochastic using pandas
            low_14 = df['low'].rolling(window=14).min()
            high_14 = df['high'].rolling(window=14).max()
            result['stoch_k'] = 100 * ((df['close'] - low_14) / (high_14 - low_14))
            result['stoch_d'] = result['stoch_k'].rolling(window=3).mean()
        
        # Rate of Change (ROC)
        for period in [5, 10, 20]:
            result[f'roc_{period}'] = df['close'].pct_change(periods=period) * 100
        
        # Money Flow Index (MFI)
        if TALIB_AVAILABLE:
            try:
                result['mfi'] = talib.MFI(
                    df['high'].values, df['low'].values, 
                    df['close'].values, df['volume'].values,
                    timeperiod=14
                )
            except Exception as e:
                logger.warning(f"TA-Lib MFI calculation failed: {e}. Using pandas.")
                mfi_pandas = True
        else:
            mfi_pandas = True
            
        if not TALIB_AVAILABLE or mfi_pandas:
            # Calculate MFI using pandas
            typical_price = (df['high'] + df['low'] + df['close']) / 3
            money_flow = typical_price * df['volume']
            
            # Get positive and negative money flow
            positive_flow = money_flow.where(typical_price > typical_price.shift(1), 0).rolling(window=14).sum()
            negative_flow = money_flow.where(typical_price < typical_price.shift(1), 0).rolling(window=14).sum()
            
            # Calculate ratio and MFI
            money_ratio = positive_flow / negative_flow.replace(0, np.finfo(float).eps)
            result['mfi'] = 100 - (100 / (1 + money_ratio))
        
        # Williams %R
        if TALIB_AVAILABLE:
            try:
                result['willr'] = talib.WILLR(
                    df['high'].values, df['low'].values, df['close'].values,
                    timeperiod=14
                )
            except Exception as e:
                logger.warning(f"TA-Lib WILLR calculation failed: {e}. Using pandas.")
                willr_pandas = True
        else:
            willr_pandas = True
            
        if not TALIB_AVAILABLE or willr_pandas:
            # Calculate Williams %R using pandas
            highest_high = df['high'].rolling(window=14).max()
            lowest_low = df['low'].rolling(window=14).min()
            result['willr'] = -100 * ((highest_high - df['close']) / (highest_high - lowest_low))
        
        # Calculate momentum signal
        if 'rsi_14' in result.columns:
            # Oversold/Overbought signals
            result['oversold'] = np.where(result['rsi_14'] < 30, 1, 0)
            result['overbought'] = np.where(result['rsi_14'] > 70, 1, 0)
        
        if 'stoch_k' in result.columns and 'stoch_d' in result.columns:
            # Stochastic Crossover signals
            result['stoch_cross_up'] = np.where(
                (result['stoch_k'] > result['stoch_d']) & 
                (result['stoch_k'].shift(1) <= result['stoch_d'].shift(1)),
                1, 0
            )
            
            result['stoch_cross_down'] = np.where(
                (result['stoch_k'] < result['stoch_d']) & 
                (result['stoch_k'].shift(1) >= result['stoch_d'].shift(1)),
                1, 0
            )
        
        return result
    
    def _calculate_volatility_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate volatility-related features.
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with volatility features
        """
        if df.empty:
            return pd.DataFrame()
            
        # Initialize result DataFrame
        result = pd.DataFrame(index=df.index)
        
        # ATR (Average True Range)
        if TALIB_AVAILABLE:
            try:
                result['atr'] = talib.ATR(
                    df['high'].values, df['low'].values, df['close'].values,
                    timeperiod=14
                )
            except Exception as e:
                logger.warning(f"TA-Lib ATR calculation failed: {e}. Using pandas.")
                atr_pandas = True
        else:
            atr_pandas = True
            
        if not TALIB_AVAILABLE or atr_pandas:
            # Calculate ATR using pandas
            high_low = df['high'] - df['low']
            high_close = np.abs(df['high'] - df['close'].shift())
            low_close = np.abs(df['low'] - df['close'].shift())
            
            ranges = pd.concat([high_low, high_close, low_close], axis=1)
            true_range = ranges.max(axis=1)
            result['atr'] = true_range.rolling(window=14).mean()
        
        # Normalized ATR (ATR%)
        result['atr_pct'] = result['atr'] / df['close'] * 100
        
        # Bollinger Bands
        if TALIB_AVAILABLE:
            try:
                result['bb_upper'], result['bb_middle'], result['bb_lower'] = talib.BBANDS(
                    df['close'].values,
                    timeperiod=20,
                    nbdevup=2,
                    nbdevdn=2,
                    matype=0
                )
            except Exception as e:
                logger.warning(f"TA-Lib BBANDS calculation failed: {e}. Using pandas.")
                bb_pandas = True
        else:
            bb_pandas = True
            
        if not TALIB_AVAILABLE or bb_pandas:
            # Calculate Bollinger Bands using pandas
            result['bb_middle'] = df['close'].rolling(window=20).mean()
            std_dev = df['close'].rolling(window=20).std()
            result['bb_upper'] = result['bb_middle'] + (std_dev * 2)
            result['bb_lower'] = result['bb_middle'] - (std_dev * 2)
        
        # Bollinger Band %B
        result['bb_width'] = (result['bb_upper'] - result['bb_lower']) / result['bb_middle']
        result['bb_pct_b'] = (df['close'] - result['bb_lower']) / (result['bb_upper'] - result['bb_lower'])
        
        # Keltner Channels
        if 'atr' in result.columns:
            result['kc_middle'] = df['close'].rolling(window=20).mean()
            result['kc_upper'] = result['kc_middle'] + (result['atr'] * 2)
            result['kc_lower'] = result['kc_middle'] - (result['atr'] * 2)
        
        # Historical volatility (standard deviation of log returns)
        for period in [10, 20, 50]:
            if len(df) >= period:
                result[f'volatility_{period}'] = df['log_returns'].rolling(window=period).std() * np.sqrt(252) * 100
        
        # Volatility Regime (based on historical volatility)
        if 'volatility_20' in result.columns:
            # Calculate long-term volatility (using 100-day lookback if available, otherwise 50-day)
            if 'volatility_50' in result.columns:
                long_vol = result['volatility_50']
            else:
                long_vol = result['volatility_20'].rolling(window=min(50, len(df))).mean()
                
            # Classify volatility regime
            result['volatility_ratio'] = result['volatility_20'] / long_vol
            result['volatility_regime'] = np.where(
                result['volatility_ratio'] > 1.2, 2,  # High volatility
                np.where(
                    result['volatility_ratio'] < 0.8, 0,  # Low volatility
                    1  # Normal volatility
                )
            )
        
        # Squeeze Momentum Indicator (combining Bollinger Bands and Keltner Channels)
        if ('bb_lower' in result.columns and 'bb_upper' in result.columns and
            'kc_lower' in result.columns and 'kc_upper' in result.columns):
            
            # Squeeze condition: Bollinger Bands inside Keltner Channels
            result['squeeze_on'] = np.where(
                (result['bb_lower'] > result['kc_lower']) & 
                (result['bb_upper'] < result['kc_upper']),
                1, 0
            )
            
            # Momentum calculation
            highest_high = df['high'].rolling(window=20).max()
            lowest_low = df['low'].rolling(window=20).min()
            
            result['squeeze_momentum'] = (
                (df['close'] - ((highest_high + lowest_low) / 2)) +
                (df['high'] - df['low'])
            )
        
        return result
    
    def _calculate_volume_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate volume-related features.
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with volume features
        """
        if df.empty or 'volume' not in df.columns:
            return pd.DataFrame()
            
        # Initialize result DataFrame
        result = pd.DataFrame(index=df.index)
        
        # Volume Moving Averages
        for period in [10, 20, 50]:
            result[f'volume_sma_{period}'] = df['volume'].rolling(window=period).mean()
        
        # Volume Relative to Moving Average
        result['volume_ratio_20'] = df['volume'] / result['volume_sma_20']
        
        # On-Balance Volume (OBV)
        if TALIB_AVAILABLE:
            try:
                result['obv'] = talib.OBV(df['close'].values, df['volume'].values)
            except Exception as e:
                logger.warning(f"TA-Lib OBV calculation failed: {e}. Using pandas.")
                obv_pandas = True
        else:
            obv_pandas = True
            
        if not TALIB_AVAILABLE or obv_pandas:
            # Calculate OBV using pandas
            close_diff = df['close'].diff()
            obv = [0]
            
            for i in range(1, len(df)):
                if close_diff.iloc[i] > 0:
                    obv.append(obv[-1] + df['volume'].iloc[i])
                elif close_diff.iloc[i] < 0:
                    obv.append(obv[-1] - df['volume'].iloc[i])
                else:
                    obv.append(obv[-1])
                    
            result['obv'] = obv
        
        # OBV Moving Average
        result['obv_sma'] = result['obv'].rolling(window=20).mean()
        
        # Volume Price Trend (VPT)
        vpt = [0]
        for i in range(1, len(df)):
            vpt.append(vpt[-1] + df['volume'].iloc[i] * (df['close'].iloc[i] - df['close'].iloc[i-1]) / df['close'].iloc[i-1])
        result['vpt'] = vpt
        
        # Accumulation/Distribution Line (ADL)
        mfm = ((df['close'] - df['low']) - (df['high'] - df['close'])) / (df['high'] - df['low'])
        mfm = mfm.replace([np.inf, -np.inf], 0)  # Replace infinities with 0
        mfv = mfm * df['volume']
        result['adl'] = mfv.cumsum()
        
        # Chaikin Money Flow (CMF)
        result['cmf'] = mfv.rolling(window=21).sum() / df['volume'].rolling(window=21).sum()
        
        # Volume Volatility (standard deviation of volume)
        result['volume_volatility'] = df['volume'].rolling(window=20).std() / result['volume_sma_20']
        
        # High Volume Bars
        result['high_volume'] = np.where(result['volume_ratio_20'] > 2, 1, 0)
        
        # Calculate money flow
        typical_price = (df['high'] + df['low'] + df['close']) / 3
        money_flow = typical_price * df['volume']
        
        # Positive and Negative Money Flow
        positive_money_flow = money_flow.where(typical_price > typical_price.shift(1), 0).rolling(window=14).sum()
        negative_money_flow = money_flow.where(typical_price < typical_price.shift(1), 0).rolling(window=14).sum()
        
        # Money Flow Ratio and Money Flow Index
        money_flow_ratio = positive_money_flow / negative_money_flow.replace(0, np.finfo(float).eps)
        result['mfi'] = 100 - (100 / (1 + money_flow_ratio))
        
        # Volume Force Index
        result['force_index'] = df['close'].diff() * df['volume']
        result['force_index_ema'] = result['force_index'].ewm(span=13, adjust=False).mean()
        
        return result
    
    def _calculate_support_resistance(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate support and resistance features.
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with support/resistance features
        """
        if df.empty or len(df) < 20:
            return pd.DataFrame()
            
        # Initialize result DataFrame
        result = pd.DataFrame(index=df.index)
        
        # Find local peaks and troughs
        def find_extrema(data, window=10, extrema_type='peak'):
            """Find local peaks or troughs with a given window."""
            result = np.zeros_like(data)
            
            for i in range(window, len(data) - window):
                if extrema_type == 'peak':
                    if data[i] == max(data[i-window:i+window+1]):
                        result[i] = 1
                else:  # trough
                    if data[i] == min(data[i-window:i+window+1]):
                        result[i] = 1
                        
            return result
        
        # Find local maxima and minima
        highs = df['high'].values
        lows = df['low'].values
        
        peaks = find_extrema(highs, window=10, extrema_type='peak')
        troughs = find_extrema(lows, window=10, extrema_type='trough')
        
        # Store peak and trough positions
        result['is_peak'] = peaks
        result['is_trough'] = troughs
        
        # Create rolling support and resistance levels
        lookback = min(100, len(df) - 1)
        
        # Find support (lowest low in past N periods)
        result['support'] = df['low'].rolling(window=lookback, min_periods=1).min()
        
        # Find resistance (highest high in past N periods)
        result['resistance'] = df['high'].rolling(window=lookback, min_periods=1).max()
        
        # Distance to support and resistance
        result['dist_to_support'] = (df['close'] - result['support']) / df['close']
        result['dist_to_resistance'] = (result['resistance'] - df['close']) / df['close']
        
        # Support/Resistance Strength
        
        # Create rolling windows for peak detection
        peak_series = pd.Series(peaks, index=df.index)
        trough_series = pd.Series(troughs, index=df.index)
        
        # Calculate historical support/resistance hits
        support_touch = np.where(
            df['low'] <= result['support'] * 1.01,  # Within 1% of support
            1, 0
        )
        resistance_touch = np.where(
            df['high'] >= result['resistance'] * 0.99,  # Within 1% of resistance
            1, 0
        )
        
        # Calculate hit frequency
        result['support_hits'] = pd.Series(support_touch).rolling(window=lookback).sum()
        result['resistance_hits'] = pd.Series(resistance_touch).rolling(window=lookback).sum()
        
        # Calculate strength by hit frequency
        result['support_strength'] = np.clip(result['support_hits'] / 5, 0, 1)  # Max strength at 5+ hits
        result['resistance_strength'] = np.clip(result['resistance_hits'] / 5, 0, 1)  # Max strength at 5+ hits
        
        # Support/Resistance Breakout signals
        result['support_break'] = np.where(
            (df['close'] < result['support']) & 
            (df['close'].shift(1) >= result['support'].shift(1)),
            1, 0
        )
        
        result['resistance_break'] = np.where(
            (df['close'] > result['resistance']) & 
            (df['close'].shift(1) <= result['resistance'].shift(1)),
            1, 0
        )
        
        # Support/Resistance Bounce signals
        result['support_bounce'] = np.where(
            (df['low'] <= result['support'] * 1.01) & 
            (df['close'] > result['support'] * 1.01),
            1, 0
        )
        
        result['resistance_bounce'] = np.where(
            (df['high'] >= result['resistance'] * 0.99) & 
            (df['close'] < result['resistance'] * 0.99),
            1, 0
        )
        
        return result
    
    def _calculate_pattern_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate chart pattern features.
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with pattern features
        """
        if df.empty or len(df) < 30:
            return pd.DataFrame()
            
        # Initialize result DataFrame
        result = pd.DataFrame(index=df.index)
        
        # Candlestick patterns using TA-Lib if available
        if TALIB_AVAILABLE:
            try:
                # Bullish reversal patterns
                result['hammer'] = talib.CDLHAMMER(
                    df['open'].values, df['high'].values, 
                    df['low'].values, df['close'].values
                )
                
                result['inverted_hammer'] = talib.CDLINVERTEDHAMMER(
                    df['open'].values, df['high'].values, 
                    df['low'].values, df['close'].values
                )
                
                result['morning_star'] = talib.CDLMORNINGSTAR(
                    df['open'].values, df['high'].values, 
                    df['low'].values, df['close'].values
                )
                
                result['bullish_engulfing'] = talib.CDLENGULFING(
                    df['open'].values, df['high'].values, 
                    df['low'].values, df['close'].values
                )
                
                # Bearish reversal patterns
                result['hanging_man'] = talib.CDLHANGINGMAN(
                    df['open'].values, df['high'].values, 
                    df['low'].values, df['close'].values
                )
                
                result['shooting_star'] = talib.CDLSHOOTINGSTAR(
                    df['open'].values, df['high'].values, 
                    df['low'].values, df['close'].values
                )
                
                result['evening_star'] = talib.CDLEVENINGSTAR(
                    df['open'].values, df['high'].values, 
                    df['low'].values, df['close'].values
                )
                
                result['bearish_engulfing'] = talib.CDLENGULFING(
                    df['open'].values, df['high'].values, 
                    df['low'].values, df['close'].values
                )
            except Exception as e:
                logger.warning(f"TA-Lib candlestick pattern calculation failed: {e}")
        
        # Simple DIY patterns for when TA-Lib is not available
        
        # Calculate basic candlestick properties
        result['body_size'] = abs(df['close'] - df['open'])
        result['upper_shadow'] = df['high'] - df[['open', 'close']].max(axis=1)
        result['lower_shadow'] = df[['open', 'close']].min(axis=1) - df['low']
        result['body_to_range'] = result['body_size'] / (df['high'] - df['low'])
        
        # Doji (tiny body)
        result['doji'] = np.where(
            result['body_size'] < (df['high'] - df['low']) * 0.1,
            1, 0
        )
        
        # Hammer and Hanging Man (small body, little/no upper shadow, long lower shadow)
        hammer_condition = (
            (result['body_size'] < (df['high'] - df['low']) * 0.3) &  # Small body
            (result['upper_shadow'] < result['body_size'] * 0.5) &    # Little/no upper shadow
            (result['lower_shadow'] > result['body_size'] * 2)        # Long lower shadow
        )
        
        if 'hammer' not in result.columns:
            # Bullish only if appearing in a downtrend
            down_trend = df['close'].rolling(window=10).mean().diff() < 0
            result['hammer'] = np.where(hammer_condition & down_trend, 1, 0)
        
        if 'hanging_man' not in result.columns:
            # Bearish only if appearing in an uptrend
            up_trend = df['close'].rolling(window=10).mean().diff() > 0
            result['hanging_man'] = np.where(hammer_condition & up_trend, 1, 0)
        
        # Shooting Star (small body, little/no lower shadow, long upper shadow)
        shooting_star_condition = (
            (result['body_size'] < (df['high'] - df['low']) * 0.3) &  # Small body
            (result['lower_shadow'] < result['body_size'] * 0.5) &    # Little/no lower shadow
            (result['upper_shadow'] > result['body_size'] * 2)        # Long upper shadow
        )
        
        if 'shooting_star' not in result.columns:
            # Bearish only if appearing in an uptrend
            up_trend = df['close'].rolling(window=10).mean().diff() > 0
            result['shooting_star'] = np.where(shooting_star_condition & up_trend, 1, 0)
        
        # Bullish and Bearish Engulfing
        if 'bullish_engulfing' not in result.columns or 'bearish_engulfing' not in result.columns:
            bullish_engulfing = (
                (df['close'] > df['open']) &                            # Current bar is up
                (df['close'].shift(1) < df['open'].shift(1)) &          # Previous bar is down
                (df['close'] > df['open'].shift(1)) &                   # Current close > previous open
                (df['open'] < df['close'].shift(1))                     # Current open < previous close
            )
            
            bearish_engulfing = (
                (df['close'] < df['open']) &                            # Current bar is down
                (df['close'].shift(1) > df['open'].shift(1)) &          # Previous bar is up
                (df['close'] < df['open'].shift(1)) &                   # Current close < previous open
                (df['open'] > df['close'].shift(1))                     # Current open > previous close
            )
            
            result['bullish_engulfing'] = np.where(bullish_engulfing, 1, 0)
            result['bearish_engulfing'] = np.where(bearish_engulfing, 1, 0)
        
        # Combine pattern signals
        result['bullish_pattern'] = np.where(
            (result['hammer'] > 0) | 
            (result['inverted_hammer'] > 0) | 
            (result['morning_star'] > 0) | 
            (result['bullish_engulfing'] > 0),
            1, 0
        )
        
        result['bearish_pattern'] = np.where(
            (result['hanging_man'] > 0) | 
            (result['shooting_star'] > 0) | 
            (result['evening_star'] > 0) | 
            (result['bearish_engulfing'] > 0),
            1, 0
        )
        
        return result
    
    def _calculate_sentiment_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate sentiment-related features.
        
        This is a placeholder for actual sentiment analysis
        with integration points for news/social data.
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with sentiment features
        """
        # For a real implementation, this would integrate with
        # news APIs, social sentiment analysis, etc.
        
        # Initialize empty result
        result = pd.DataFrame(index=df.index)
        
        # Add placeholder sentiment metrics
        result['sentiment_score'] = 0
        result['sentiment_volume'] = 0
        
        return result
    
    def _calculate_custom_features(
        self, 
        df: pd.DataFrame, 
        feature_list: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Calculate custom user-defined features.
        
        Args:
            df: Input DataFrame
            feature_list: List of custom features to calculate
            
        Returns:
            DataFrame with custom features
        """
        if df.empty:
            return pd.DataFrame()
            
        # Initialize result DataFrame
        result = pd.DataFrame(index=df.index)
        
        # Define custom feature calculation functions
        custom_feature_functions = {
            # Volatility Adjusted Bollinger Bands
            'vbb': lambda df: self._calc_volatility_adjusted_bb(df),
            
            # Heikin-Ashi candles
            'heikin_ashi': lambda df: self._calc_heikin_ashi(df),
            
            # Volume Weighted Average Price (VWAP)
            'vwap': lambda df: self._calc_vwap(df),
            
            # Market Regime Detection
            'market_regime': lambda df: self._calc_market_regime(df),
            
            # Hull Moving Average
            'hull_ma': lambda df: self._calc_hull_ma(df, period=20),
            
            # Relative Vigor Index (RVI)
            'rvi': lambda df: self._calc_rvi(df),
            
            # Mean Reversion Z-Score
            'zscore': lambda df: self._calc_zscore(df),

            # KST Oscillator
            'kst': lambda df: self._calc_kst(df),
            
            # Elder Ray Index
            'elder_ray': lambda df: self._calc_elder_ray(df)
        }
        
        # If no specific features requested, calculate them all
        if feature_list is None:
            feature_list = list(custom_feature_functions.keys())
            
        # Calculate requested features
        for feature in feature_list:
            if feature in custom_feature_functions:
                try:
                    # Calculate feature and merge with result
                    feature_df = custom_feature_functions[feature](df)
                    if feature_df is not None and not feature_df.empty:
                        for col in feature_df.columns:
                            result[col] = feature_df[col]
                except Exception as e:
                    logger.error(f"Error calculating custom feature '{feature}': {e}")
        
        return result
    
    # Custom feature calculation methods
    
    def _calc_volatility_adjusted_bb(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate volatility-adjusted Bollinger Bands."""
        result = pd.DataFrame(index=df.index)
        
        # Calculate standard Bollinger Bands
        result['vbb_middle'] = df['close'].rolling(window=20).mean()
        
        # Use ATR for adjusting standard deviation
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())
        
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = ranges.max(axis=1)
        atr = true_range.rolling(window=14).mean()
        
        # Normalize the ATR to standard deviation scale
        atr_normalized = atr / (df['close'].rolling(window=20).std()) * 2
        
        # Apply the ATR multiplier to the standard deviation
        std_dev = df['close'].rolling(window=20).std()
        result['vbb_upper'] = result['vbb_middle'] + (std_dev * atr_normalized)
        result['vbb_lower'] = result['vbb_middle'] - (std_dev * atr_normalized)
        
        return result
    
    def _calc_heikin_ashi(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate Heikin-Ashi candlesticks."""
        result = pd.DataFrame(index=df.index)
        
        # Calculate Heikin-Ashi candles
        result['ha_close'] = (df['open'] + df['high'] + df['low'] + df['close']) / 4
        
        # First value of ha_open is the same as regular open
        result['ha_open'] = pd.Series(index=df.index)
        result['ha_open'].iloc[0] = df['open'].iloc[0]
        
        # Rest of ha_open values are average of prior ha_open and prior ha_close
        for i in range(1, len(df)):
            result['ha_open'].iloc[i] = (result['ha_open'].iloc[i-1] + result['ha_close'].iloc[i-1]) / 2
            
        # Calculate high and low values
        result['ha_high'] = df[['high', 'open', 'close']].max(axis=1)
        result['ha_low'] = df[['low', 'open', 'close']].min(axis=1)
        
        # Calculate trend based on Heikin-Ashi
        result['ha_trend'] = np.where(
            result['ha_close'] > result['ha_open'],
            1,  # Bullish
            np.where(
                result['ha_close'] < result['ha_open'],
                -1,  # Bearish
                0    # Neutral
            )
        )
        
        # Calculate trend strength
        result['ha_trend_strength'] = abs(result['ha_close'] - result['ha_open']) / (result['ha_high'] - result['ha_low'])
        
        return result
    
    def _calc_vwap(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate Volume Weighted Average Price (VWAP)."""
        result = pd.DataFrame(index=df.index)
        
        # Calculate VWAP
        if 'volume' in df.columns:
            # Calculate typical price
            typical_price = (df['high'] + df['low'] + df['close']) / 3
            
            # Calculate volume * typical price
            vol_tp = typical_price * df['volume']
            
            # Calculate cumulative values
            cum_vol_tp = vol_tp.cumsum()
            cum_vol = df['volume'].cumsum()
            
            # Calculate VWAP
            result['vwap'] = cum_vol_tp / cum_vol
            
            # Calculate VWAP bands (similar to Bollinger Bands but based on VWAP)
            vwap_std = (typical_price - result['vwap']) ** 2
            vwap_std = (vwap_std * df['volume']).cumsum() / cum_vol
            vwap_std = np.sqrt(vwap_std)
            
            result['vwap_upper'] = result['vwap'] + (vwap_std * 2)
            result['vwap_lower'] = result['vwap'] - (vwap_std * 2)
            
            # Calculate distance from VWAP
            result['vwap_distance'] = (df['close'] - result['vwap']) / result['vwap'] * 100
        
        return result
    
    def _calc_market_regime(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate market regime indicators."""
        result = pd.DataFrame(index=df.index)
        
        # Calculate different EMAs for regime detection
        result['ema_20'] = df['close'].ewm(span=20, adjust=False).mean()
        result['ema_100'] = df['close'].ewm(span=100, adjust=False).mean()
        
        # Calculate volatility
        returns = df['close'].pct_change()
        result['volatility'] = returns.rolling(window=20).std() * np.sqrt(252) * 100
        
        # Calculate average volatility
        result['avg_volatility'] = result['volatility'].rolling(window=50).mean()
        
        # Determine regime based on trend and volatility
        trend = np.where(result['ema_20'] > result['ema_100'], 1, -1)
        
        # High volatility threshold (1.5x average)
        vol_ratio = result['volatility'] / result['avg_volatility']
        high_vol = vol_ratio > 1.5
        
        # Regime classification:
        # 1: Bull trend, normal volatility
        # 2: Bull trend, high volatility
        # -1: Bear trend, normal volatility
        # -2: Bear trend, high volatility
        # 0: Neutral/ranging
        
        result['regime'] = np.where(
            abs(result['ema_20'] / result['ema_100'] - 1) < 0.02,
            0,  # Neutral/ranging when EMAs are close together
            np.where(
                high_vol,
                trend * 2,  # High volatility regime
                trend      # Normal volatility regime
            )
        )
        
        return result
    
    def _calc_hull_ma(self, df: pd.DataFrame, period: int = 20) -> pd.DataFrame:
        """Calculate Hull Moving Average."""
        result = pd.DataFrame(index=df.index)
        
        # Calculate weighted moving averages
        wma_half_period = df['close'].rolling(window=period//2).apply(
            lambda x: sum([(i+1) * x.iloc[i] for i in range(len(x))]) / sum(range(1, len(x)+1)),
            raw=True
        )
        
        wma_full_period = df['close'].rolling(window=period).apply(
            lambda x: sum([(i+1) * x.iloc[i] for i in range(len(x))]) / sum(range(1, len(x)+1)),
            raw=True
        )
        
        # Calculate raw Hull MA
        raw_hma = 2 * wma_half_period - wma_full_period
        
        # Calculate final Hull MA
        sqrt_period = int(np.sqrt(period))
        result[f'hull_ma_{period}'] = raw_hma.rolling(window=sqrt_period).apply(
            lambda x: sum([(i+1) * x.iloc[i] for i in range(len(x))]) / sum(range(1, len(x)+1)),
            raw=True
        )
        
        return result
    
    def _calc_rvi(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate Relative Vigor Index (RVI)."""
        result = pd.DataFrame(index=df.index)
        
        # Calculate numerator (close - open)
        numerator = df['close'] - df['open']
        
        # Calculate denominator (high - low)
        denominator = df['high'] - df['low']
        
        # Calculate 10-period average of numerator and denominator
        num_avg = numerator.rolling(window=10).mean()
        den_avg = denominator.rolling(window=10).mean()
        
        # Calculate RVI
        result['rvi'] = num_avg / den_avg
        
        # Calculate RVI signal line (4-period moving average)
        result['rvi_signal'] = result['rvi'].rolling(window=4).mean()
        
        return result
    
    def _calc_zscore(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate mean reversion Z-score."""
        result = pd.DataFrame(index=df.index)
        
        # Calculate 20-period rolling mean and std dev
        mean_20 = df['close'].rolling(window=20).mean()
        std_20 = df['close'].rolling(window=20).std()
        
        # Calculate z-score
        result['zscore'] = (df['close'] - mean_20) / std_20
        
        # Calculate mean reversion signals
        result['oversold'] = np.where(result['zscore'] < -2, 1, 0)  # Z-score < -2 (oversold)
        result['overbought'] = np.where(result['zscore'] > 2, 1, 0)  # Z-score > 2 (overbought)
        
        return result
    
    def _calc_kst(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate Know Sure Thing (KST) oscillator."""
        result = pd.DataFrame(index=df.index)
        
        # Calculate ROC values with different periods
        roc10 = df['close'].pct_change(periods=10)
        roc15 = df['close'].pct_change(periods=15)
        roc20 = df['close'].pct_change(periods=20)
        roc30 = df['close'].pct_change(periods=30)
        
        # Calculate moving averages of ROC values
        ma10 = roc10.rolling(window=10).mean()
        ma10 = ma10 * 1  # Weight
        
        ma15 = roc15.rolling(window=10).mean()
        ma15 = ma15 * 2  # Weight
        
        ma20 = roc20.rolling(window=10).mean()
        ma20 = ma20 * 3  # Weight
        
        ma30 = roc30.rolling(window=15).mean()
        ma30 = ma30 * 4  # Weight
        
        # Calculate KST
        result['kst'] = ma10 + ma15 + ma20 + ma30
        
        # Calculate signal line (9-period average of KST)
        result['kst_signal'] = result['kst'].rolling(window=9).mean()
        
        # Calculate histogram
        result['kst_hist'] = result['kst'] - result['kst_signal']
        
        return result
    
    def _calc_elder_ray(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate Elder Ray Index."""
        result = pd.DataFrame(index=df.index)
        
        # Calculate 13-period EMA
        ema13 = df['close'].ewm(span=13, adjust=False).mean()
        
        # Calculate Bull Power and Bear Power
        result['bull_power'] = df['high'] - ema13
        result['bear_power'] = df['low'] - ema13
        
        # Calculate combined Elder Force Index
        result['elder_force'] = result['bull_power'] + result['bear_power']
        
        return result
    
    def _prepare_for_ml(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare data for machine learning models.
        
        Args:
            df: Input DataFrame with features
            
        Returns:
            DataFrame prepared for ML training/inference
        """
        if df.empty:
            return df
            
        # Make copy to avoid modifying original
        ml_df = df.copy()
        
        # Remove any non-numeric columns
        numeric_cols = ml_df.select_dtypes(include=[np.number]).columns
        ml_df = ml_df[numeric_cols]
        
        # Forward fill any remaining NaN values
        ml_df = ml_df.ffill()
        
        # Replace any remaining NaNs with 0
        ml_df = ml_df.fillna(0)
        
        # Replace infinite values with large numbers
        ml_df = ml_df.replace([np.inf, -np.inf], [1e9, -1e9])
        
        # Add target variable (next period return)
        ml_df['target_next_return'] = df['returns'].shift(-1)
        
        # Add binary classification target
        ml_df['target_direction'] = np.where(ml_df['target_next_return'] > 0, 1, 0)
        
        # Add multi-class target (significant up, up, flat, down, significant down)
        std_return = ml_df['returns'].rolling(window=20).std()
        ml_df['target_class'] = np.where(
            ml_df['target_next_return'] > std_return,
            2,  # Significant up
            np.where(
                ml_df['target_next_return'] > 0,
                1,  # Up
                np.where(
                    ml_df['target_next_return'] > -std_return,
                    0,  # Flat
                    np.where(
                        ml_df['target_next_return'] > -2 * std_return,
                        -1,  # Down
                        -2   # Significant down
                    )
                )
            )
        )
        
        return ml_df
    
    def resample_timeframe(
        self, 
        df: pd.DataFrame, 
        target_timeframe: str,
        volume_weighted: bool = True
    ) -> pd.DataFrame:
        """
        Resample data to a different timeframe.
        
        Args:
            df: Input DataFrame with OHLCV data
            target_timeframe: Target timeframe (e.g., '1h', '4h', '1d')
            volume_weighted: Whether to use volume-weighted resampling
            
        Returns:
            Resampled DataFrame
        """
        if df.empty:
            return df
            
        # Ensure df has a datetime index
        if not isinstance(df.index, pd.DatetimeIndex):
            # Try to convert index to datetime
            try:
                df.index = pd.to_datetime(df.index)
            except Exception as e:
                raise ValueError(f"Cannot convert index to datetime: {e}")
                
        # Convert timeframe string to pandas resampling rule
        tf_map = {
            '1m': '1min', '3m': '3min', '5m': '5min', '15m': '15min', '30m': '30min',
            '1h': '1H', '2h': '2H', '4h': '4H', '6h': '6H', '8h': '8H', '12h': '12H',
            '1d': '1D', '3d': '3D', '1w': '1W', '1M': '1M'
        }
        
        if target_timeframe not in tf_map:
            raise ValueError(f"Unsupported timeframe: {target_timeframe}")
            
        resample_rule = tf_map[target_timeframe]
        
        # Define resampling functions
        if volume_weighted and 'volume' in df.columns:
            # Volume-weighted resampling for OHLCV data
            resampled = df.resample(resample_rule).agg({
                'open': 'first',
                'high': 'max',
                'low': 'min',
                'close': 'last',
                'volume': 'sum'
            })
        else:
            # Standard resampling for OHLCV data
            resampled = df.resample(resample_rule).agg({
                'open': 'first',
                'high': 'max',
                'low': 'min',
                'close': 'last',
                'volume': 'sum'
            })
            
        # Calculate returns for resampled data
        resampled['returns'] = resampled['close'].pct_change()
        
        return resampled
    
    def get_feature_importance(
        self, 
        df: pd.DataFrame, 
        target_col: str = 'returns'
    ) -> pd.DataFrame:
        """
        Calculate feature importance using a Random Forest model.
        
        Args:
            df: DataFrame with features
            target_col: Target column for importance calculation
            
        Returns:
            DataFrame with feature importance scores
        """
        try:
            from sklearn.ensemble import RandomForestRegressor
            from sklearn.preprocessing import StandardScaler
            
            # Prepare the data
            features = df.select_dtypes(include=[np.number]).copy()
            
            # Remove target from features if it exists
            if target_col in features.columns:
                y = features[target_col].shift(-1)  # Next period's value
                features = features.drop(columns=[target_col])
            else:
                y = df['close'].pct_change().shift(-1)  # Default to next period's return
                
            # Forward fill NaN values
            features = features.ffill()
            
            # Drop rows with NaN
            valid_idx = ~(features.isna().any(axis=1) | y.isna())
            X = features[valid_idx]
            y = y[valid_idx]
            
            # Standardize features
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            # Train a Random Forest model
            model = RandomForestRegressor(n_estimators=100, random_state=42)
            model.fit(X_scaled, y)
            
            # Get feature importance
            importance = model.feature_importances_
            
            # Create DataFrame with importance scores
            importance_df = pd.DataFrame({
                'Feature': X.columns,
                'Importance': importance
            }).sort_values('Importance', ascending=False)
            
            return importance_df
            
        except ImportError:
            logger.warning("scikit-learn not available. Feature importance calculation skipped.")
            return pd.DataFrame()
        except Exception as e:
            logger.error(f"Error calculating feature importance: {e}")
            return pd.DataFrame()
    
    def save_processed_data(
        self, 
        df: pd.DataFrame, 
        filepath: str, 
        format: str = 'csv'
    ) -> bool:
        """
        Save processed data to disk.
        
        Args:
            df: DataFrame to save
            filepath: Path to save the data
            format: File format ('csv', 'parquet', 'pickle')
            
        Returns:
            bool: Success status
        """
        if df.empty:
            logger.warning("Cannot save empty DataFrame")
            return False
            
        try:
            # Create directory if it doesn't exist
            directory = os.path.dirname(filepath)
            if directory and not os.path.exists(directory):
                os.makedirs(directory)
                
            # Save in requested format
            if format.lower() == 'csv':
                df.to_csv(filepath)
            elif format.lower() == 'parquet':
                df.to_parquet(filepath)
            elif format.lower() in ['pickle', 'pkl']:
                df.to_pickle(filepath)
            else:
                raise ValueError(f"Unsupported file format: {format}")
                
            logger.info(f"Data saved to {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving data: {e}")
            return False
    
    def get_system_status(self) -> Dict[str, Any]:
        """
        Get system status and information.
        
        Returns:
            Dict with system status info
        """
        return {
            'system_ready': self._system_ready,
            'base_timeframe': self.base_timeframe,
            'feature_set': self.feature_set,
            'talib_available': TALIB_AVAILABLE,
            'use_parallel': self.use_parallel,
            'max_workers': self.max_workers,
            'custom_error_handling': CUSTOM_ERROR_HANDLING,
            'cache_size': self.cache_size,
            'db_manager_available': self.db_manager is not None,
            'raw_cache_size': len(self._raw_data_cache),
            'processed_cache_size': len(self._processed_data_cache),
            'feature_cache_size': len(self._feature_cache)
        }
        
    def clear_cache(self, cache_type: str = 'all') -> None:
        """
        Clear data caches.
        
        Args:
            cache_type: Type of cache to clear ('raw', 'processed', 'feature', 'all')
        """
        with self._cache_lock:
            if cache_type in ['raw', 'all']:
                self._raw_data_cache.clear()
                
            if cache_type in ['processed', 'all']:
                self._processed_data_cache.clear()
                
            if cache_type in ['feature', 'all']:
                self._feature_cache.clear()
                
        logger.info(f"Cleared {cache_type} cache")
        
    def shutdown(self) -> None:
        """Clean up resources when shutting down."""
        try:
            # Clear all caches
            self.clear_cache('all')
            
            # Shutdown thread pool if using parallel processing
            if self.use_parallel and hasattr(self, 'thread_pool'):
                self.thread_pool.shutdown(wait=True)
                
            logger.info("DataProcessor shutdown complete")
            
        except Exception as e:
            logger.error(f"Error during DataProcessor shutdown: {e}")
