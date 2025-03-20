# feature_engineering.py

import numpy as np
import pandas as pd
import logging
import time
from typing import Dict, List, Tuple, Union, Optional, Callable, Any
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression
import talib
from joblib import Parallel, delayed
import threading
import warnings
import os
import sys

# Add project root to path for imports
current_path = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_path, os.pardir))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Try to import error handling
try:
    from core.error_handling import safe_execute, ErrorCategory, ErrorSeverity
    HAVE_ERROR_HANDLING = True
except ImportError:
    HAVE_ERROR_HANDLING = False
    logging.warning("Error handling module not available. Using basic error handling.")

# Configure logging
logger = logging.getLogger("feature_engineering")

class FeatureGenerator:
    """
    Comprehensive feature engineering for algorithmic trading models.
    
    This class generates, transforms and selects features from market data,
    optimized for machine learning models with parallel processing support.
    """
    
    def __init__(
        self, 
        use_talib: bool = True,
        n_jobs: int = -1,
        normalize_method: str = 'standard',
        cache_dir: Optional[str] = None,
        thread_manager = None
    ):
        """
        Initialize the feature generator.
        
        Args:
            use_talib: Whether to use TA-Lib for technical indicators
            n_jobs: Number of parallel jobs (-1 for all available cores)
            normalize_method: Method for feature normalization ('standard', 'minmax', 'robust', None)
            cache_dir: Directory to cache computed features (None = no caching)
            thread_manager: Optional thread manager for task execution
        """
        self.use_talib = use_talib
        self.n_jobs = n_jobs
        self.normalize_method = normalize_method
        self.cache_dir = cache_dir
        self.thread_manager = thread_manager
        self.feature_cache = {}
        self.scalers = {}
        self._lock = threading.RLock()
        
        # Check TA-Lib installation
        if self.use_talib:
            try:
                talib_available = talib.MA is not None
            except AttributeError:
                talib_available = False
                
            if not talib_available:
                logger.warning("TA-Lib not available. Falling back to pandas methods.")
                self.use_talib = False
        
        # Create cache directory if it doesn't exist
        if self.cache_dir:
            os.makedirs(self.cache_dir, exist_ok=True)
            
        # Suppress warnings from feature calculation
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        
        logger.info(f"FeatureGenerator initialized (use_talib={use_talib}, n_jobs={n_jobs})")
    
    def _get_cache_path(self, symbol: str, timeframe: str, feature_set: str) -> str:
        """Get path for cached features."""
        if not self.cache_dir:
            return None
        
        return os.path.join(
            self.cache_dir, 
            f"{symbol}_{timeframe}_{feature_set}_{pd.Timestamp.now().strftime('%Y%m%d')}.pkl"
        )
    
    def _try_load_cache(self, cache_path: str) -> Optional[pd.DataFrame]:
        """Try to load features from cache."""
        if not cache_path or not os.path.exists(cache_path):
            return None
            
        try:
            df = pd.read_pickle(cache_path)
            logger.info(f"Loaded features from cache: {cache_path}")
            return df
        except Exception as e:
            logger.warning(f"Failed to load from cache ({cache_path}): {e}")
            return None
    
    def _save_to_cache(self, df: pd.DataFrame, cache_path: str) -> bool:
        """Save features to cache."""
        if not cache_path:
            return False
            
        try:
            df.to_pickle(cache_path)
            logger.info(f"Saved features to cache: {cache_path}")
            return True
        except Exception as e:
            logger.warning(f"Failed to save to cache: {e}")
            return False
    
    def generate_features(
        self, 
        df: pd.DataFrame, 
        feature_sets: List[str] = ['price', 'volume', 'volatility', 'trend', 'momentum', 'cycle'],
        symbol: str = None,
        timeframe: str = None,
        custom_features: Optional[Dict[str, Callable]] = None,
        use_cache: bool = True,
        progress_callback: Optional[Callable[[int], None]] = None
    ) -> pd.DataFrame:
        """
        Generate all features for the given market data.
        
        Args:
            df: DataFrame with OHLCV market data
            feature_sets: List of feature sets to generate
            symbol: Trading pair symbol (for caching)
            timeframe: Timeframe (for caching)
            custom_features: Dictionary of custom feature functions 
            use_cache: Whether to use cache
            progress_callback: Optional callback for progress updates
            
        Returns:
            DataFrame with original data and generated features
        """
        if df is None or df.empty:
            logger.warning("Empty DataFrame provided. No features generated.")
            return pd.DataFrame()
            
        # Make a copy to avoid modifying the original
        df = df.copy()
        
        # Ensure DataFrame has proper columns
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        for col in required_columns:
            if col not in df.columns:
                logger.error(f"Required column '{col}' not found in DataFrame")
                raise ValueError(f"Required column '{col}' not found in DataFrame")
        
        # Setup cache
        cache_key = f"{symbol}_{timeframe}_{'-'.join(sorted(feature_sets))}"
        cache_path = None
        
        if use_cache and symbol and timeframe:
            cache_path = self._get_cache_path(symbol, timeframe, '-'.join(sorted(feature_sets)))
            
            # Try to load from cache first
            cached_df = self._try_load_cache(cache_path)
            if cached_df is not None and len(cached_df) >= len(df):
                return cached_df.loc[df.index]
        
        # Track initial column count for progress calculation
        initial_col_count = len(df.columns)
        total_progress_steps = len(feature_sets) + 2  # +2 for cleanup and scaling
        progress_step = 0
        
        def update_progress():
            nonlocal progress_step
            progress_step += 1
            if progress_callback:
                progress_percent = min(100, int((progress_step / total_progress_steps) * 100))
                progress_callback(progress_percent)
        
        # Generate features for each feature set
        for feature_set in feature_sets:
            start_time = time.time()
            
            if feature_set.lower() == 'price':
                df = self._generate_price_features(df)
            elif feature_set.lower() == 'volume':
                df = self._generate_volume_features(df)
            elif feature_set.lower() == 'volatility':
                df = self._generate_volatility_features(df)
            elif feature_set.lower() == 'trend':
                df = self._generate_trend_features(df)
            elif feature_set.lower() == 'momentum':
                df = self._generate_momentum_features(df)
            elif feature_set.lower() == 'cycle':
                df = self._generate_cycle_features(df)
            elif feature_set.lower() == 'pattern':
                df = self._generate_pattern_features(df)
            elif feature_set.lower() == 'orderbook':
                df = self._generate_orderbook_features(df)
            elif feature_set.lower() == 'sentiment':
                df = self._generate_sentiment_features(df)
            elif feature_set.lower() == 'fundamental':
                df = self._generate_fundamental_features(df)
            elif feature_set.lower() == 'statistical':
                df = self._generate_statistical_features(df)
            else:
                logger.warning(f"Unknown feature set: {feature_set}")
            
            elapsed = time.time() - start_time
            logger.debug(f"Generated {feature_set} features in {elapsed:.2f} seconds")
            update_progress()
        
        # Add custom features if provided
        if custom_features:
            start_time = time.time()
            df = self._add_custom_features(df, custom_features)
            elapsed = time.time() - start_time
            logger.debug(f"Generated custom features in {elapsed:.2f} seconds")
        
        # Clean up features
        start_time = time.time()
        df = self._cleanup_features(df)
        elapsed = time.time() - start_time
        logger.debug(f"Cleaned up features in {elapsed:.2f} seconds")
        update_progress()
        
        # Apply normalization if requested
        if self.normalize_method:
            start_time = time.time()
            df = self._normalize_features(df, symbol, timeframe)
            elapsed = time.time() - start_time
            logger.debug(f"Normalized features in {elapsed:.2f} seconds")
        update_progress()
        
        # Save to cache
        if cache_path and len(df) > 0:
            self._save_to_cache(df, cache_path)
        
        # Log feature generation stats
        feature_count = len(df.columns) - initial_col_count
        logger.info(f"Generated {feature_count} features for {symbol} {timeframe}")
        
        return df
    
    def _generate_price_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate price-based features."""
        # Original price features
        df['log_return'] = np.log(df['close'] / df['close'].shift(1))
        
        # Price ratios
        df['close_to_open'] = df['close'] / df['open']
        df['high_to_open'] = df['high'] / df['open']
        df['low_to_open'] = df['low'] / df['open']
        df['high_to_low'] = df['high'] / df['low']
        
        # Candle-based features
        df['body_size'] = abs(df['close'] - df['open']) / df['open']
        df['upper_shadow'] = (df['high'] - df[['open', 'close']].max(axis=1)) / df['open']
        df['lower_shadow'] = (df[['open', 'close']].min(axis=1) - df['low']) / df['open']
        df['is_bullish'] = (df['close'] > df['open']).astype(float)
        
        # Price velocities (1st derivatives)
        windows = [2, 3, 5, 10, 20]
        for window in windows:
            df[f'price_velocity_{window}'] = df['close'].pct_change(window)
            
        # Price accelerations (2nd derivatives)
        for window in windows:
            df[f'price_acceleration_{window}'] = df[f'price_velocity_{window}'].diff()
        
        # Relative price positions
        for window in [5, 10, 20, 50, 100, 200]:
            if len(df) >= window:
                df[f'rel_position_{window}'] = (df['close'] - df['low'].rolling(window).min()) / \
                                               (df['high'].rolling(window).max() - df['low'].rolling(window).min())
        
        return df
    
    def _generate_volume_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate volume-based features."""
        # Volume changes
        df['volume_change'] = df['volume'].pct_change()
        df['volume_change_abs'] = df['volume_change'].abs()
        
        # Volume moving averages and ratios
        for window in [5, 10, 20, 50]:
            if len(df) >= window:
                df[f'volume_ma_{window}'] = df['volume'].rolling(window).mean()
                df[f'volume_ratio_{window}'] = df['volume'] / df[f'volume_ma_{window}']
                
                # Relative volume
                df[f'rel_volume_{window}'] = df['volume'] / df['volume'].rolling(window).mean()
                
                # Volume oscillators (similar to MACD but for volume)
                if window < 20:
                    df[f'vol_osc_{window}_20'] = (df[f'volume_ma_{window}'] - df['volume_ma_20']) / df['volume_ma_20']
        
        # Money flow (volume * price direction)
        df['money_flow'] = df['volume'] * ((df['close'] - df['low'] - (df['high'] - df['close'])) / (df['high'] - df['low'] + 1e-10))
        
        # On-Balance Volume (OBV)
        df['obv_signal'] = np.where(df['close'] > df['close'].shift(1), 1, 
                                   np.where(df['close'] < df['close'].shift(1), -1, 0))
        df['obv_raw'] = df['volume'] * df['obv_signal']
        df['obv'] = df['obv_raw'].cumsum()
        
        # Volume-weighted price metrics
        df['vwap_daily'] = (df['volume'] * (df['high'] + df['low'] + df['close']) / 3).cumsum() / df['volume'].cumsum()
        
        # Volume-price trend features
        df['price_up_vol_up'] = ((df['close'] > df['close'].shift(1)) & (df['volume'] > df['volume'].shift(1))).astype(float)
        df['price_up_vol_down'] = ((df['close'] > df['close'].shift(1)) & (df['volume'] < df['volume'].shift(1))).astype(float)
        df['price_down_vol_up'] = ((df['close'] < df['close'].shift(1)) & (df['volume'] > df['volume'].shift(1))).astype(float)
        df['price_down_vol_down'] = ((df['close'] < df['close'].shift(1)) & (df['volume'] < df['volume'].shift(1))).astype(float)
        
        return df
    
    def _generate_volatility_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate volatility-based features."""
        # Basic volatility metrics
        df['hl_ratio'] = df['high'] / df['low']
        
        # Calculate price range metrics
        df['true_range'] = np.maximum(
            df['high'] - df['low'],
            np.maximum(
                abs(df['high'] - df['close'].shift(1)),
                abs(df['low'] - df['close'].shift(1))
            )
        )
        
        # Average True Range (ATR) and normalized variants
        windows = [5, 10, 14, 20]
        for window in windows:
            if len(df) >= window:
                # Use TA-Lib if available (faster)
                if self.use_talib:
                    df[f'atr_{window}'] = talib.ATR(df['high'].values, df['low'].values, 
                                                    df['close'].values, timeperiod=window)
                else:
                    df[f'atr_{window}'] = df['true_range'].rolling(window=window).mean()
                
                # Normalized ATR (relative to price)
                df[f'natr_{window}'] = df[f'atr_{window}'] / df['close'] * 100
                
                # ATR ratio (current ATR vs recent ATR)
                if window > 5:
                    df[f'atr_ratio_{window}_5'] = df[f'atr_5'] / df[f'atr_{window}']
        
        # Historical volatility using standard deviation of returns
        for window in [5, 10, 20, 50]:
            if len(df) >= window:
                df[f'volatility_{window}'] = df['log_return'].rolling(window).std() * np.sqrt(252)
                
                # Volatility ratio (current vs. historical)
                if window > 10:
                    df[f'volatility_ratio_{window}_10'] = df['volatility_10'] / df[f'volatility_{window}']
        
        # Garman-Klass volatility
        df['garman_klass_vol'] = np.sqrt(
            0.5 * np.log(df['high'] / df['low'])**2 - 
            (2 * np.log(2) - 1) * np.log(df['close'] / df['open'])**2
        )
        
        # Bollinger Bands
        for window in [10, 20, 50]:
            if len(df) >= window:
                if self.use_talib:
                    upper, middle, lower = talib.BBANDS(df['close'].values, timeperiod=window, nbdevup=2, nbdevdn=2)
                    df[f'bb_upper_{window}'] = upper
                    df[f'bb_middle_{window}'] = middle
                    df[f'bb_lower_{window}'] = lower
                else:
                    rolling_mean = df['close'].rolling(window=window).mean()
                    rolling_std = df['close'].rolling(window=window).std()
                    df[f'bb_upper_{window}'] = rolling_mean + (rolling_std * 2)
                    df[f'bb_middle_{window}'] = rolling_mean
                    df[f'bb_lower_{window}'] = rolling_mean - (rolling_std * 2)
                
                # BB width (volatility indicator)
                df[f'bb_width_{window}'] = (df[f'bb_upper_{window}'] - df[f'bb_lower_{window}']) / df[f'bb_middle_{window}']
                
                # BB position (where current price is within bands)
                df[f'bb_pos_{window}'] = (df['close'] - df[f'bb_lower_{window}']) / (df[f'bb_upper_{window}'] - df[f'bb_lower_{window}'])
        
        # Volatility breakout features
        for window in [10, 20]:
            if len(df) >= window:
                df[f'close_gt_upper_{window}'] = (df['close'] > df[f'bb_upper_{window}']).astype(float)
                df[f'close_lt_lower_{window}'] = (df['close'] < df[f'bb_lower_{window}']).astype(float)
        
        # Identify volatility regime
        if len(df) >= 50:
            long_term_vol = df['volatility_50'].rolling(window=50).mean()
            df['high_vol_regime'] = (df['volatility_20'] > long_term_vol * 1.1).astype(float)
            df['low_vol_regime'] = (df['volatility_20'] < long_term_vol * 0.9).astype(float)
            
        # Price gap features
        df['gap_up'] = (df['open'] > df['close'].shift(1)).astype(float)
        df['gap_down'] = (df['open'] < df['close'].shift(1)).astype(float)
        df['gap_size'] = abs(df['open'] - df['close'].shift(1)) / df['close'].shift(1)
        
        return df
    
    def _generate_trend_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate trend-based features."""
        # Simple Moving Averages (SMA)
        for window in [5, 10, 20, 50, 100, 200]:
            if len(df) >= window:
                if self.use_talib:
                    df[f'sma_{window}'] = talib.SMA(df['close'].values, timeperiod=window)
                else:
                    df[f'sma_{window}'] = df['close'].rolling(window=window).mean()
                
                # Distance from moving average (%)
                df[f'dist_sma_{window}'] = (df['close'] / df[f'sma_{window}'] - 1) * 100
        
        # Exponential Moving Averages (EMA)
        for window in [5, 10, 20, 50, 100, 200]:
            if len(df) >= window:
                if self.use_talib:
                    df[f'ema_{window}'] = talib.EMA(df['close'].values, timeperiod=window)
                else:
                    df[f'ema_{window}'] = df['close'].ewm(span=window, adjust=False).mean()
                
                # Distance from EMA (%)
                df[f'dist_ema_{window}'] = (df['close'] / df[f'ema_{window}'] - 1) * 100
        
        # Moving Average Crossovers
        ma_pairs = [(5, 10), (10, 20), (20, 50), (50, 200)]
        for short, long in ma_pairs:
            if len(df) >= long:
                # MA crossover signals
                df[f'ema_{short}_gt_{long}'] = (df[f'ema_{short}'] > df[f'ema_{long}']).astype(float)
                df[f'sma_{short}_gt_{long}'] = (df[f'sma_{short}'] > df[f'sma_{long}']).astype(float)
                
                # Crossover events
                df[f'ema_{short}_{long}_cross_up'] = ((df[f'ema_{short}'] > df[f'ema_{long}']) & 
                                                     (df[f'ema_{short}'].shift(1) <= df[f'ema_{long}'].shift(1))).astype(float)
                df[f'ema_{short}_{long}_cross_down'] = ((df[f'ema_{short}'] < df[f'ema_{long}']) & 
                                                       (df[f'ema_{short}'].shift(1) >= df[f'ema_{long}'].shift(1))).astype(float)
        
        # MACD
        if len(df) >= 26:
            if self.use_talib:
                df['macd'], df['macd_signal'], df['macd_hist'] = talib.MACD(
                    df['close'].values, fastperiod=12, slowperiod=26, signalperiod=9
                )
            else:
                ema12 = df['close'].ewm(span=12, adjust=False).mean()
                ema26 = df['close'].ewm(span=26, adjust=False).mean()
                df['macd'] = ema12 - ema26
                df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
                df['macd_hist'] = df['macd'] - df['macd_signal']
            
            # MACD crossovers
            df['macd_cross_up'] = ((df['macd'] > df['macd_signal']) & 
                                   (df['macd'].shift(1) <= df['macd_signal'].shift(1))).astype(float)
            df['macd_cross_down'] = ((df['macd'] < df['macd_signal']) & 
                                     (df['macd'].shift(1) >= df['macd_signal'].shift(1))).astype(float)
        
        # ADX (Average Directional Index) - Trend strength
        if len(df) >= 14 and self.use_talib:
            df['adx'] = talib.ADX(df['high'].values, df['low'].values, df['close'].values, timeperiod=14)
            df['adx_trend_strength'] = np.where(df['adx'] < 20, 0,  # No trend
                                               np.where(df['adx'] < 40, 1,  # Moderate trend
                                                       2))  # Strong trend
        
        # Linear regression-based trend features
        for window in [20, 50]:
            if len(df) >= window:
                # Use parallel processing for linear regression
                results = self._parallel_regression(df['close'], window)
                df[f'linear_slope_{window}'] = results['slope']
                df[f'linear_r2_{window}'] = results['r2']
                df[f'linear_intercept_{window}'] = results['intercept']
        
        # Price channels
        for window in [10, 20, 50]:
            if len(df) >= window:
                df[f'upper_channel_{window}'] = df['high'].rolling(window).max()
                df[f'lower_channel_{window}'] = df['low'].rolling(window).min()
                
                # Channel width and relative position
                channel_width = df[f'upper_channel_{window}'] - df[f'lower_channel_{window}']
                df[f'channel_width_{window}'] = channel_width / df['close']
                df[f'channel_position_{window}'] = (df['close'] - df[f'lower_channel_{window}']) / channel_width
        
        # Directional indicators
        if len(df) >= 14 and self.use_talib:
            df['plus_di'] = talib.PLUS_DI(df['high'].values, df['low'].values, df['close'].values, timeperiod=14)
            df['minus_di'] = talib.MINUS_DI(df['high'].values, df['low'].values, df['close'].values, timeperiod=14)
            df['di_diff'] = df['plus_di'] - df['minus_di']
            df['di_sum'] = df['plus_di'] + df['minus_di']
            df['di_ratio'] = df['plus_di'] / df['minus_di'].replace(0, np.nan)
        
        # Parabolic SAR
        if self.use_talib:
            df['sar'] = talib.SAR(df['high'].values, df['low'].values, acceleration=0.02, maximum=0.2)
            df['sar_trend'] = (df['close'] > df['sar']).astype(float)
            df['sar_dist'] = (df['close'] / df['sar'] - 1) * 100
            
        return df
    
    def _generate_momentum_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate momentum-based features."""
        # Rate of Change (ROC)
        for window in [1, 5, 10, 20, 50]:
            if len(df) >= window:
                if self.use_talib:
                    df[f'roc_{window}'] = talib.ROC(df['close'].values, timeperiod=window)
                else:
                    df[f'roc_{window}'] = df['close'].pct_change(window) * 100
        
        # Relative Strength Index (RSI)
        for window in [6, 14, 20]:
            if len(df) >= window:
                if self.use_talib:
                    df[f'rsi_{window}'] = talib.RSI(df['close'].values, timeperiod=window)
                else:
                    delta = df['close'].diff()
                    gain = delta.clip(lower=0)
                    loss = -delta.clip(upper=0)
                    avg_gain = gain.rolling(window=window).mean()
                    avg_loss = loss.rolling(window=window).mean()
                    rs = avg_gain / avg_loss.replace(0, np.nan)
                    df[f'rsi_{window}'] = 100 - (100 / (1 + rs))
        
        # RSI-based features
        if 'rsi_14' in df.columns:
            df['rsi_overbought'] = (df['rsi_14'] > 70).astype(float)
            df['rsi_oversold'] = (df['rsi_14'] < 30).astype(float)
            df['rsi_cross_50_up'] = ((df['rsi_14'] > 50) & (df['rsi_14'].shift(1) < 50)).astype(float)
            df['rsi_cross_50_down'] = ((df['rsi_14'] < 50) & (df['rsi_14'].shift(1) > 50)).astype(float)
        
        # Stochastic Oscillator
        if len(df) >= 14:
            if self.use_talib:
                df['stoch_k'], df['stoch_d'] = talib.STOCH(
                    df['high'].values, df['low'].values, df['close'].values,
                    fastk_period=14, slowk_period=3, slowk_matype=0, slowd_period=3, slowd_matype=0
                )
            else:
                high_14 = df['high'].rolling(14).max()
                low_14 = df['low'].rolling(14).min()
                df['stoch_k'] = 100 * (df['close'] - low_14) / (high_14 - low_14)
                df['stoch_d'] = df['stoch_k'].rolling(3).mean()
            
            # Stochastic crossovers
            df['stoch_cross_up'] = ((df['stoch_k'] > df['stoch_d']) & 
                                    (df['stoch_k'].shift(1) <= df['stoch_d'].shift(1))).astype(float)
            df['stoch_cross_down'] = ((df['stoch_k'] < df['stoch_d']) & 
                                      (df['stoch_k'].shift(1) >= df['stoch_d'].shift(1))).astype(float)
            
            # Stochastic overbought/oversold
            df['stoch_overbought'] = (df['stoch_k'] > 80).astype(float)
            df['stoch_oversold'] = (df['stoch_k'] < 20).astype(float)
        
        # Commodity Channel Index (CCI)
        if len(df) >= 20 and self.use_talib:
            df['cci'] = talib.CCI(df['high'].values, df['low'].values, df['close'].values, timeperiod=20)
            df['cci_overbought'] = (df['cci'] > 100).astype(float)
            df['cci_oversold'] = (df['cci'] < -100).astype(float)
        
        # Williams %R
        if len(df) >= 14 and self.use_talib:
            df['willr'] = talib.WILLR(df['high'].values, df['low'].values, df['close'].values, timeperiod=14)
            df['willr_overbought'] = (df['willr'] > -20).astype(float)
            df['willr_oversold'] = (df['willr'] < -80).astype(float)
        
        # Money Flow Index (MFI)
        if len(df) >= 14 and self.use_talib:
            df['mfi'] = talib.MFI(df['high'].values, df['low'].values, df['close'].values, 
                                 df['volume'].values, timeperiod=14)
            df['mfi_overbought'] = (df['mfi'] > 80).astype(float)
            df['mfi_oversold'] = (df['mfi'] < 20).astype(float)
        
        # Ultimate Oscillator
        if len(df) >= 28 and self.use_talib:
            df['ultosc'] = talib.ULTOSC(df['high'].values, df['low'].values, df['close'].values, 
                                       timeperiod1=7, timeperiod2=14, timeperiod3=28)
            df['ultosc_overbought'] = (df['ultosc'] > 70).astype(float)
            df['ultosc_oversold'] = (df['ultosc'] < 30).astype(float)
        
        # Chande Momentum Oscillator (CMO)
        if len(df) >= 14 and self.use_talib:
            df['cmo'] = talib.CMO(df['close'].values, timeperiod=14)
            df['cmo_overbought'] = (df['cmo'] > 50).astype(float)
            df['cmo_oversold'] = (df['cmo'] < -50).astype(float)
        
        # Balance of Power (BOP)
        if self.use_talib:
            df['bop'] = talib.BOP(df['open'].values, df['high'].values, df['low'].values, df['close'].values)
            df['bop_positive'] = (df['bop'] > 0).astype(float)
            df['bop_strong_positive'] = (df['bop'] > 0.5).astype(float)
            df['bop_strong_negative'] = (df['bop'] < -0.5).astype(float)
        
        # Awesome Oscillator
        if len(df) >= 34:
            median_price = (df['high'] + df['low']) / 2
            df['ao'] = median_price.rolling(5).mean() - median_price.rolling(34).mean()
            df['ao_positive'] = (df['ao'] > 0).astype(float)
            df['ao_cross_zero_up'] = ((df['ao'] > 0) & (df['ao'].shift(1) < 0)).astype(float)
            df['ao_cross_zero_down'] = ((df['ao'] < 0) & (df['ao'].shift(1) > 0)).astype(float)
        
        return df
    
    def _generate_cycle_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate cycle-based features."""
        # Hilbert Transform - Dominant Cycle Period
        if len(df) >= 50 and self.use_talib:
            try:
                df['ht_dcperiod'] = talib.HT_DCPERIOD(df['close'].values)
                df['ht_dcphase'] = talib.HT_DCPHASE(df['close'].values)
                sine, lead_sine = talib.HT_SINE(df['close'].values)
                df['ht_sine'] = sine
                df['ht_leadsine'] = lead_sine
                
                # Sine wave crossovers
                df['ht_sine_cross_up'] = ((df['ht_sine'] > df['ht_leadsine']) & 
                                         (df['ht_sine'].shift(1) <= df['ht_leadsine'].shift(1))).astype(float)
                df['ht_sine_cross_down'] = ((df['ht_sine'] < df['ht_leadsine']) & 
                                           (df['ht_sine'].shift(1) >= df['ht_leadsine'].shift(1))).astype(float)
                
                # HT Trendmode
                df['ht_trendmode'] = talib.HT_TRENDMODE(df['close'].values)
            except Exception as e:
                logger.warning(f"Failed to calculate Hilbert Transform features: {e}")
        
        # Simple cyclical features based on sin/cos transformations of time
        # These can help models detect time-based patterns
        df['day_sin'] = np.sin(df.index.dayofweek * (2 * np.pi / 7))
        df['day_cos'] = np.cos(df.index.dayofweek * (2 * np.pi / 7))
        
        if hasattr(df.index, 'hour'):
            df['hour_sin'] = np.sin(df.index.hour * (2 * np.pi / 24))
            df['hour_cos'] = np.cos(df.index.hour * (2 * np.pi / 24))
        
        # Distance to nearest N-day high/low
        peaks_windows = [5, 10, 20]
        for window in peaks_windows:
            if len(df) >= window:
                # Rolling window max/min
                df[f'rolling_max_{window}'] = df['high'].rolling(window).max()
                df[f'rolling_min_{window}'] = df['low'].rolling(window).min()
                
                # Distance to high/low as percentage
                df[f'dist_to_max_{window}'] = (df[f'rolling_max_{window}'] - df['close']) / df['close'] * 100
                df[f'dist_to_min_{window}'] = (df['close'] - df[f'rolling_min_{window}']) / df['close'] * 100
                
                # New high/low signals
                df[f'new_high_{window}'] = (df['high'] > df['high'].shift(1).rolling(window-1).max()).astype(float)
                df[f'new_low_{window}'] = (df['low'] < df['low'].shift(1).rolling(window-1).min()).astype(float)
        
        # Cycles based on price moving average crossovers
        if len(df) >= 50:  # Ensure enough data for longer cycles
            # Detect "Bull/Bear" market regime based on 50/200 MA
            if 'sma_50' in df.columns and 'sma_200' in df.columns:
                df['bull_market'] = (df['sma_50'] > df['sma_200']).astype(float)
                df['golden_cross'] = ((df['sma_50'] > df['sma_200']) & 
                                     (df['sma_50'].shift(1) <= df['sma_200'].shift(1))).astype(float)
                df['death_cross'] = ((df['sma_50'] < df['sma_200']) & 
                                     (df['sma_50'].shift(1) >= df['sma_200'].shift(1))).astype(float)
            
            # Detect "Correction" phases in bull market
            if 'sma_20' in df.columns and 'sma_50' in df.columns:
                df['correction_phase'] = ((df['sma_50'] > df['sma_200']) & 
                                       (df['sma_20'] < df['sma_50'])).astype(float)
            
            # Detect "Rally" phases in bear market
            if 'sma_20' in df.columns and 'sma_50' in df.columns and 'sma_200' in df.columns:
                df['rally_phase'] = ((df['sma_50'] < df['sma_200']) & 
                                  (df['sma_20'] > df['sma_50'])).astype(float)
        
        return df
    
    def _generate_pattern_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate candlestick pattern features."""
        if not self.use_talib or len(df) < 10:
            return df
            
        # Candlestick pattern detection functions
        pattern_funcs = [
            ('doji', talib.CDLDOJI),
            ('hammer', talib.CDLHAMMER),
            ('shooting_star', talib.CDLSHOOTINGSTAR),
            ('engulfing', talib.CDLENGULFING),
            ('morning_star', talib.CDLMORNINGSTAR),
            ('evening_star', talib.CDLEVENINGSTAR),
            ('three_white_soldiers', talib.CDL3WHITESOLDIERS),
            ('three_black_crows', talib.CDL3BLACKCROWS),
            ('hanging_man', talib.CDLHANGINGMAN),
            ('harami', talib.CDLHARAMI),
            ('harami_cross', talib.CDLHARAMICROSS),
            ('marubozu', talib.CDLMARUBOZU),
            ('belt_hold', talib.CDLBELTHOLD),
            ('piercing', talib.CDLPIERCING),
            ('dark_cloud_cover', talib.CDLDARKCLOUDCOVER)
        ]
        
        # Generate candlestick pattern features
        try:
            for pattern_name, pattern_func in pattern_funcs:
                df[f'pattern_{pattern_name}'] = pattern_func(
                    df['open'].values, df['high'].values, df['low'].values, df['close'].values
                )
                
                # Normalize pattern values (some TA-Lib functions return -100, 0, or 100)
                if df[f'pattern_{pattern_name}'].max() > 1 or df[f'pattern_{pattern_name}'].min() < -1:
                    df[f'pattern_{pattern_name}'] = df[f'pattern_{pattern_name}'] / 100
        except Exception as e:
            logger.warning(f"Error generating candlestick patterns: {e}")
        
        # Aggregated pattern signals
        pattern_cols = [col for col in df.columns if 'pattern_' in col]
        if pattern_cols:
            # Bullish pattern count
            df['bullish_patterns'] = df[pattern_cols].clip(lower=0).sum(axis=1)
            # Bearish pattern count
            df['bearish_patterns'] = df[pattern_cols].clip(upper=0).abs().sum(axis=1)
            # Net pattern signal
            df['net_pattern_signal'] = df['bullish_patterns'] - df['bearish_patterns']
        
        return df
    
    def _generate_orderbook_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate order book features."""
        # Order book features are typically available only for live trading
        # But we can infer some limited information from OHLC data
        
        # Simulate order book pressure using price/volume data
        df['buy_pressure'] = np.where(df['close'] > df['open'], df['volume'], 0)
        df['sell_pressure'] = np.where(df['close'] < df['open'], df['volume'], 0)
        
        # Calculate relative buy/sell pressure
        total_volume = df['buy_pressure'] + df['sell_pressure']
        df['rel_buy_pressure'] = df['buy_pressure'] / total_volume.replace(0, 1)
        df['rel_sell_pressure'] = df['sell_pressure'] / total_volume.replace(0, 1)
        
        # Simple relative strength between buying and selling
        df['buy_sell_imbalance'] = (df['rel_buy_pressure'] - df['rel_sell_pressure'])
        
        # Calculate volume-price correlation (5-day rolling)
        if len(df) >= 5:
            returns = df['close'].pct_change()
            volumes = df['volume'].pct_change()
            df['volume_price_corr'] = returns.rolling(5).corr(volumes)
        
        # If additional orderbook columns exist, use them
        orderbook_fields = ['bid_count', 'ask_count', 'bid_volume', 'ask_volume', 
                           'bid_ask_spread', 'bid_ask_ratio']
        
        for field in orderbook_fields:
            if field in df.columns:
                # Generate features based on actual orderbook data
                if 'ratio' in field or 'spread' in field:
                    # These are already ratio features
                    pass
                elif len(df) >= 5:
                    # Calculate moving averages for orderbook metrics
                    df[f'{field}_ma5'] = df[field].rolling(5).mean()
                    # Calculate ratio to moving average
                    df[f'{field}_ratio'] = df[field] / df[f'{field}_ma5']
        
        return df
    
    def _generate_sentiment_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate sentiment features."""
        # Sentiment features are typically derived from external data
        # But we can try to infer market sentiment from price and volume
        
        # Sentiment fields that may be in the dataframe already
        sentiment_fields = ['sentiment_score', 'sentiment_magnitude', 
                           'positive_mentions', 'negative_mentions', 'neutral_mentions',
                           'fear_greed_index', 'social_volume', 'social_engagement',
                           'news_sentiment', 'twitter_sentiment', 'reddit_sentiment']
        
        existing_fields = [f for f in sentiment_fields if f in df.columns]
        
        if existing_fields:
            # Generate features based on existing sentiment data
            for field in existing_fields:
                if len(df) >= 5:
                    # Moving average
                    df[f'{field}_ma5'] = df[field].rolling(5).mean()
                    # Z-score (standardized)
                    df[f'{field}_zscore'] = (df[field] - df[field].rolling(20).mean()) / df[field].rolling(20).std()
        else:
            # Infer "sentiment" from price/volume data
            # Green candles with increasing volume might indicate positive sentiment
            df['inferred_bullish'] = ((df['close'] > df['open']) & 
                                     (df['volume'] > df['volume'].shift(1))).astype(float)
            
            # Red candles with increasing volume might indicate negative sentiment
            df['inferred_bearish'] = ((df['close'] < df['open']) & 
                                     (df['volume'] > df['volume'].shift(1))).astype(float)
            
            # Create a simple sentiment score [-1, 1]
            if 'inferred_bullish' in df.columns and 'inferred_bearish' in df.columns:
                df['inferred_sentiment'] = df['inferred_bullish'] - df['inferred_bearish']
                
                # Moving averages of sentiment
                if len(df) >= 5:
                    df['inferred_sentiment_ma5'] = df['inferred_sentiment'].rolling(5).mean()
                if len(df) >= 10:
                    df['inferred_sentiment_ma10'] = df['inferred_sentiment'].rolling(10).mean()
        
        return df
    
    def _generate_fundamental_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate fundamental features."""
        # Fundamental features typically come from external data sources
        # Here we'll check if any fundamental data already exists in the dataframe
        
        fundamental_fields = ['market_cap', 'volume_24h', 'supply', 'max_supply', 
                             'circulating_supply', 'dominance', 'active_addresses',
                             'transactions', 'fees', 'difficulty']
        
        existing_fields = [f for f in fundamental_fields if f in df.columns]
        
        if existing_fields:
            # Calculate derived features
            if 'market_cap' in existing_fields and 'volume_24h' in existing_fields:
                # Volume to market cap ratio (turnover)
                df['volume_to_mcap'] = df['volume_24h'] / df['market_cap']
            
            if 'supply' in existing_fields and 'max_supply' in existing_fields:
                # Supply ratio
                df['supply_ratio'] = df['supply'] / df['max_supply']
            
            # Calculate moving averages and relative values
            for field in existing_fields:
                if len(df) >= 7:  # At least a week of data
                    df[f'{field}_ma7'] = df[field].rolling(7).mean()
                    df[f'{field}_ratio'] = df[field] / df[f'{field}_ma7']
                
                if len(df) >= 30:  # At least a month of data
                    df[f'{field}_ma30'] = df[field].rolling(30).mean()
                    df[f'{field}_monthly_change'] = (df[field] / df[field].shift(30) - 1) * 100
        
        return df
    
    def _generate_statistical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate statistical features."""
        # These are more advanced statistical features that might help ML models
        
        # Z-score of price (how many std devs from moving mean)
        for window in [20, 50]:
            if len(df) >= window:
                df[f'close_zscore_{window}'] = (df['close'] - df['close'].rolling(window).mean()) / df['close'].rolling(window).std()
                
                # Z-score categories
                df[f'zscore_extreme_high_{window}'] = (df[f'close_zscore_{window}'] > 2).astype(float)
                df[f'zscore_extreme_low_{window}'] = (df[f'close_zscore_{window}'] < -2).astype(float)
        
        # Outlier detection based on median absolute deviation (MAD)
        for window in [20, 50]:
            if len(df) >= window:
                rolling_median = df['close'].rolling(window).median()
                rolling_mad = (df['close'] - rolling_median).abs().rolling(window).median()
                
                # 3 MADs is a common threshold for outliers
                df[f'price_outlier_{window}'] = (abs(df['close'] - rolling_median) > (3 * rolling_mad)).astype(float)
        
        # Autoregressive features - past N day returns
        for lag in [1, 2, 3, 5, 10]:
            df[f'return_lag_{lag}'] = df['log_return'].shift(lag)
        
        # Return autocorrelation
        if len(df) >= 10:
            df['return_autocorr_1'] = df['log_return'].rolling(10).apply(
                lambda x: x.autocorr(lag=1), raw=False
            )
        
        # Volume autocorrelation
        if len(df) >= 10:
            df['volume_autocorr_1'] = df['volume'].pct_change().rolling(10).apply(
                lambda x: x.autocorr(lag=1), raw=False
            )
        
        # Skew and Kurtosis of returns
        if len(df) >= 20:
            df['return_skew_20'] = df['log_return'].rolling(20).skew()
            df['return_kurt_20'] = df['log_return'].rolling(20).kurt()
            
            # Normalized versions
            df['return_skew_zscore'] = (df['return_skew_20'] - df['return_skew_20'].rolling(100).mean()) / df['return_skew_20'].rolling(100).std()
            df['return_kurt_zscore'] = (df['return_kurt_20'] - df['return_kurt_20'].rolling(100).mean()) / df['return_kurt_20'].rolling(100).std()
        
        # Realized volatility (annualized)
        if 'log_return' in df.columns:
            for window in [5, 10, 20, 50]:
                if len(df) >= window:
                    # Annualized standard deviation
                    df[f'realized_vol_{window}'] = df['log_return'].rolling(window).std() * np.sqrt(252)
                    
                    # "Normalized" volatility (relative to recent volatility)
                    if window < 50 and len(df) >= 50:
                        df[f'vol_ratio_{window}_50'] = df[f'realized_vol_{window}'] / df['realized_vol_50']
        
        # Analyze the distribution of returns
        if len(df) >= 100:  # Need substantial data for distribution analysis
            returns = df['log_return'].dropna()
            if len(returns) >= 100:
                # Calculate percentiles of the return distribution
                percentiles = [0.01, 0.05, 0.25, 0.75, 0.95, 0.99]
                perc_values = returns.quantile(percentiles).values
                
                # Check if return is in tail
                for i, p in enumerate(percentiles):
                    if p < 0.5:  # Lower tail
                        df[f'return_lt_{int(p*100)}p'] = (df['log_return'] < perc_values[i]).astype(float)
                    else:  # Upper tail
                        df[f'return_gt_{int(p*100)}p'] = (df['log_return'] > perc_values[i]).astype(float)
        
        return df
    
    def _add_custom_features(self, df: pd.DataFrame, custom_features: Dict[str, Callable]) -> pd.DataFrame:
        """Add custom features defined by user functions."""
        if not custom_features:
            return df
            
        # Make a copy for safety
        df_custom = df.copy()
        
        # Process each custom feature
        for feature_name, feature_func in custom_features.items():
            try:
                start_time = time.time()
                df_custom[feature_name] = feature_func(df_custom)
                
                elapsed = time.time() - start_time
                logger.debug(f"Generated custom feature '{feature_name}' in {elapsed:.2f} seconds")
                
            except Exception as e:
                logger.error(f"Error computing custom feature '{feature_name}': {e}")
                
        return df_custom
    
    def _cleanup_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean up features by handling NaN values and infinity."""
        # Make a copy for safety
        df_clean = df.copy()
        
        # Replace inf/-inf with NaN
        df_clean = df_clean.replace([np.inf, -np.inf], np.nan)
        
        # Fill NaN values with appropriate methods for different feature types
        numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
        
        # Strategy: Forward fill, then backward fill, then fill with medians
        if not numeric_cols.empty:
            # Start with forward filling (for time series)
            df_clean[numeric_cols] = df_clean[numeric_cols].ffill()
            
            # Then backward fill
            df_clean[numeric_cols] = df_clean[numeric_cols].bfill()
            
            # For any remaining NaNs, fill with column median
            for col in numeric_cols:
                median_val = df_clean[col].median()
                if pd.isna(median_val):  # If median is also NaN, use 0
                    median_val = 0
                df_clean[col] = df_clean[col].fillna(median_val)
        
        return df_clean
    
    def _normalize_features(self, df: pd.DataFrame, symbol: str = None, timeframe: str = None) -> pd.DataFrame:
        """Normalize features using the specified method."""
        if not self.normalize_method:
            return df
            
        # Get columns to normalize (exclude non-numeric)
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        # Exclude certain columns we don't want to normalize
        exclude_cols = ['open', 'high', 'low', 'close', 'volume']
        cols_to_normalize = [col for col in numeric_cols if col not in exclude_cols]
        
        if not cols_to_normalize:
            return df
            
        # Generate a key for this dataset
        scaler_key = f"{symbol}_{timeframe}" if symbol and timeframe else "default"
        
        # Get or create scaler
        with self._lock:
            if scaler_key not in self.scalers:
                if self.normalize_method == 'standard':
                    scaler = StandardScaler()
                elif self.normalize_method == 'minmax':
                    scaler = MinMaxScaler()
                elif self.normalize_method == 'robust':
                    scaler = RobustScaler()
                else:
                    logger.warning(f"Unknown normalization method: {self.normalize_method}")
                    return df
                    
                self.scalers[scaler_key] = scaler
            else:
                scaler = self.scalers[scaler_key]
        
        # Make a copy for safety
        df_norm = df.copy()
        
        # Fit and transform
        try:
            values = df_norm[cols_to_normalize].values
            normalized = scaler.fit_transform(values)
            df_norm[cols_to_normalize] = normalized
        except Exception as e:
            logger.error(f"Error during feature normalization: {e}")
            return df
            
        return df_norm
    
    def _parallel_regression(self, series: pd.Series, window: int) -> dict:
        """Calculate linear regression features in parallel."""
        # Prepare results
        results = {
            'slope': np.full(len(series), np.nan),
            'r2': np.full(len(series), np.nan),
            'intercept': np.full(len(series), np.nan)
        }
        
        if len(series) < window:
            return results
            
        # Get valid indices (where we have enough data for the window)
        valid_indices = range(window - 1, len(series))
        
        # Helper function for each window regression
        def _process_window(i):
            if i < window - 1:
                return None
                
            # Get window data
            y = series.iloc[i-window+1:i+1].values
            X = np.arange(window).reshape(-1, 1)
            
            try:
                # Calculate regression with numpy for efficiency
                X_mean = np.mean(X)
                y_mean = np.mean(y)
                
                # Slope calculation
                numerator = np.sum((X - X_mean) * (y - y_mean))
                denominator = np.sum((X - X_mean) ** 2)
                slope = numerator / denominator
                
                # Intercept calculation
                intercept = y_mean - slope * X_mean
                
                # R-squared calculation
                y_pred = slope * X + intercept
                ss_total = np.sum((y - y_mean) ** 2)
                ss_residual = np.sum((y - y_pred) ** 2)
                r2 = 1 - (ss_residual / ss_total)
                
                return i, slope, r2, intercept
            except:
                return i, np.nan, np.nan, np.nan
        
        # Use joblib for parallel computation
        n_jobs = min(self.n_jobs if self.n_jobs > 0 else os.cpu_count(), 
                    os.cpu_count(), len(valid_indices))
        results_list = Parallel(n_jobs=n_jobs)(
            delayed(_process_window)(i) for i in valid_indices
        )
        
        # Process results
        for result in results_list:
            if result is not None:
                i, slope, r2, intercept = result
                results['slope'][i] = slope
                results['r2'][i] = r2
                results['intercept'][i] = intercept
                
        # Convert results to Series
        for key in results:
            results[key] = pd.Series(results[key], index=series.index)
            
        return results
    
    def select_features(
        self, 
        X: pd.DataFrame, 
        y: pd.Series, 
        n_features: int = 20, 
        method: str = 'f_regression'
    ) -> List[str]:
        """
        Select top features based on statistical tests.
        
        Args:
            X: Feature matrix
            y: Target variable
            n_features: Number of features to select
            method: Feature selection method ('f_regression', 'mutual_info')
            
        Returns:
            List of selected feature names
        """
        # Handle NaNs
        X = X.fillna(0)
        y = y.fillna(0)
        
        # Select scorer based on method
        if method == 'f_regression':
            scorer = f_regression
        elif method == 'mutual_info':
            scorer = mutual_info_regression
        else:
            raise ValueError(f"Unknown feature selection method: {method}")
        
        # Select features
        selector = SelectKBest(score_func=scorer, k=min(n_features, X.shape[1]))
        selector.fit(X, y)
        
        # Get selected feature names
        feature_idx = selector.get_support(indices=True)
        feature_names = X.columns[feature_idx].tolist()
        
        return feature_names
    
    def reduce_dimensions(
        self, 
        X: pd.DataFrame, 
        n_components: int = 10, 
        method: str = 'pca'
    ) -> pd.DataFrame:
        """
        Reduce dimensionality of feature matrix.
        
        Args:
            X: Feature matrix
            n_components: Number of components to keep
            method: Dimension reduction method ('pca' only for now)
            
        Returns:
            DataFrame with reduced dimensions
        """
        # Handle NaNs
        X = X.fillna(0)
        
        # Only PCA supported for now
        if method != 'pca':
            raise ValueError(f"Unsupported dimension reduction method: {method}")
        
        # Apply PCA
        pca = PCA(n_components=min(n_components, X.shape[1]))
        reduced = pca.fit_transform(X)
        
        # Convert to DataFrame
        reduced_df = pd.DataFrame(
            reduced, 
            index=X.index,
            columns=[f'pc_{i+1}' for i in range(reduced.shape[1])]
        )
        
        # Log explained variance
        explained_var = sum(pca.explained_variance_ratio_) * 100
        logger.info(f"PCA with {n_components} components explains {explained_var:.2f}% of variance")
        
        return reduced_df
    
    def get_feature_importance(
        self, 
        model, 
        feature_names: List[str]
    ) -> Dict[str, float]:
        """
        Get feature importance from a trained model.
        
        Args:
            model: Trained model with feature_importances_ attribute
            feature_names: List of feature names
            
        Returns:
            Dictionary mapping feature names to importance scores
        """
        if not hasattr(model, 'feature_importances_'):
            raise ValueError("Model does not have feature_importances_ attribute")
            
        importances = model.feature_importances_
        
        if len(importances) != len(feature_names):
            raise ValueError(f"Feature importance length ({len(importances)}) does not match feature names ({len(feature_names)})")
            
        # Create dictionary of feature importances
        importance_dict = dict(zip(feature_names, importances))
        
        # Sort by importance
        importance_dict = {k: v for k, v in sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)}
        
        return importance_dict
    
    def save_scalers(self, path: str) -> bool:
        """
        Save feature scalers to disk.
        
        Args:
            path: Directory path to save scalers
            
        Returns:
            True if successful, False otherwise
        """
        if not self.scalers:
            logger.warning("No scalers to save")
            return False
            
        os.makedirs(path, exist_ok=True)
        
        try:
            import joblib
            
            for key, scaler in self.scalers.items():
                scaler_path = os.path.join(path, f"scaler_{key}.joblib")
                joblib.dump(scaler, scaler_path)
                
            logger.info(f"Saved {len(self.scalers)} scalers to {path}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving scalers: {e}")
            return False
    
    def load_scalers(self, path: str) -> bool:
        """
        Load feature scalers from disk.
        
        Args:
            path: Directory path to load scalers from
            
        Returns:
            True if successful, False otherwise
        """
        if not os.path.exists(path):
            logger.warning(f"Scaler directory {path} does not exist")
            return False
            
        try:
            import joblib
            
            # Find all scaler files
            scaler_files = [f for f in os.listdir(path) if f.startswith("scaler_") and f.endswith(".joblib")]
            
            if not scaler_files:
                logger.warning(f"No scaler files found in {path}")
                return False
                
            # Load each scaler
            with self._lock:
                self.scalers = {}
                
                for file in scaler_files:
                    key = file.replace("scaler_", "").replace(".joblib", "")
                    scaler_path = os.path.join(path, file)
                    self.scalers[key] = joblib.load(scaler_path)
                    
            logger.info(f"Loaded {len(self.scalers)} scalers from {path}")
            return True
            
        except Exception as e:
            logger.error(f"Error loading scalers: {e}")
            return False
    
    def clear_cache(self) -> bool:
        """
        Clear cache directory.
        
        Returns:
            True if successful, False otherwise
        """
        if not self.cache_dir:
            logger.warning("No cache directory set")
            return False
            
        if not os.path.exists(self.cache_dir):
            logger.warning(f"Cache directory {self.cache_dir} does not exist")
            return False
            
        try:
            # Clear all files in cache directory
            file_count = 0
            for file in os.listdir(self.cache_dir):
                file_path = os.path.join(self.cache_dir, file)
                if os.path.isfile(file_path):
                    os.remove(file_path)
                    file_count += 1
                    
            logger.info(f"Cleared {file_count} files from cache directory {self.cache_dir}")
            return True
            
        except Exception as e:
            logger.error(f"Error clearing cache: {e}")
            return False

# Apply error handling decorator if available
if HAVE_ERROR_HANDLING:
    safe_generate_features = safe_execute(ErrorCategory.DATA_PROCESSING)(FeatureGenerator.generate_features)
    FeatureGenerator.generate_features = safe_generate_features

# Create a function to get feature generator with appropriate thread manager
def get_feature_generator(thread_manager=None, use_talib=True, cache_dir="data/feature_cache"):
    """
    Get a properly configured feature generator instance.
    
    Args:
        thread_manager: Optional thread manager for parallel processing
        use_talib: Whether to use TA-Lib
        cache_dir: Directory for caching features
        
    Returns:
        Configured FeatureGenerator instance
    """
    # Ensure cache directory exists
    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)
        
    # Create feature generator
    feature_gen = FeatureGenerator(
        use_talib=use_talib,
        n_jobs=-1,  # Use all available cores
        normalize_method='standard',
        cache_dir=cache_dir,
        thread_manager=thread_manager
    )
    
    return feature_gen

# Example usage function
def generate_all_features(
    df: pd.DataFrame, 
    symbol: str = None, 
    timeframe: str = None,
    thread_manager = None,
    progress_callback = None
) -> pd.DataFrame:
    """
    Generate all available features for the given market data.
    
    Args:
        df: DataFrame with OHLCV market data
        symbol: Trading pair symbol (for caching)
        timeframe: Timeframe (for caching)
        thread_manager: Optional thread manager
        progress_callback: Optional progress callback
        
    Returns:
        DataFrame with original data and all generated features
    """
    # Get feature generator
    fg = get_feature_generator(thread_manager)
    
    # Generate all feature sets
    feature_sets = ['price', 'volume', 'volatility', 'trend', 'momentum', 'cycle']
    
    # Add pattern features if TA-Lib is available
    if fg.use_talib:
        feature_sets.append('pattern')
    
    # Generate features
    return fg.generate_features(
        df, 
        feature_sets=feature_sets, 
        symbol=symbol, 
        timeframe=timeframe,
        progress_callback=progress_callback
    )
