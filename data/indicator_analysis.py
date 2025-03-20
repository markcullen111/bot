# indicator_analysis.py

import numpy as np
import pandas as pd
import math
import logging
from typing import Dict, List, Union, Optional, Tuple, Any
from scipy.signal import argrelextrema, find_peaks
import time
from concurrent.futures import ThreadPoolExecutor
import traceback

# Try to import error handling module
try:
    from error_handling import safe_execute, ErrorCategory, ErrorSeverity
    HAVE_ERROR_HANDLING = True
except ImportError:
    HAVE_ERROR_HANDLING = False
    logging.warning("Error handling module not available, using basic error handling in indicator analysis")

class IndicatorAnalysis:
    """
    Advanced technical indicator analysis engine for AI-driven trading.
    Optimized for performance with vectorized operations and parallel processing.
    
    Features:
    - Calculates and caches key technical indicators
    - Detects market structure and patterns
    - Generates trading signals with confidence scores
    - Provides market regime classification
    - Thread-safe for concurrent analysis
    """
    
    def __init__(self, df: Optional[pd.DataFrame] = None, cache_enabled: bool = True, parallel: bool = True):
        """
        Initialize the indicator analysis engine.
        
        Args:
            df: Optional pandas DataFrame with OHLCV data
            cache_enabled: Whether to cache indicator calculations for performance
            parallel: Whether to use parallel processing for heavy calculations
        """
        self.df = df.copy() if df is not None else pd.DataFrame()
        self.cache_enabled = cache_enabled
        self.parallel = parallel
        self._reset_state()
        
        # Performance tracking
        self.calculation_times = {}
        
        # Threading resources
        self.executor = ThreadPoolExecutor(max_workers=4) if parallel else None
        
        logging.debug("IndicatorAnalysis initialized with cache_enabled=%s, parallel=%s", 
                     cache_enabled, parallel)
    
    def __del__(self):
        """Cleanup resources on object destruction"""
        if self.executor:
            self.executor.shutdown(wait=False)
    
    def _reset_state(self):
        """Reset internal state - called when data changes"""
        # Analysis results
        self.signals = {}  # Trading signals
        self.patterns = {}  # Chart patterns
        self.market_regime = None  # Current market regime
        
        # Indicator cache
        self._cache = {}
        
        # Status tracking
        self._cache_hits = 0
        self._cache_misses = 0
    
    def set_data(self, df: pd.DataFrame) -> bool:
        """
        Set the dataframe for analysis with validation.
        
        Args:
            df: pandas DataFrame with OHLCV data
            
        Returns:
            bool: Success status
        """
        if df is None or not isinstance(df, pd.DataFrame):
            logging.error("Invalid dataframe provided to indicator analysis")
            return False
            
        if df.empty:
            logging.warning("Empty dataframe provided to indicator analysis")
            return False
        
        required_columns = ['open', 'high', 'low', 'close']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            logging.error(f"DataFrame missing required columns: {missing_columns}")
            return False
        
        # Copy the dataframe to avoid modifying the original
        self.df = df.copy()
        
        # Reset state for new data
        self._reset_state()
        
        return True
    
    def _get_from_cache(self, key: str) -> Optional[Any]:
        """Get item from cache if available and enabled"""
        if not self.cache_enabled:
            self._cache_misses += 1
            return None
            
        value = self._cache.get(key)
        if value is not None:
            self._cache_hits += 1
            return value
            
        self._cache_misses += 1
        return None
    
    def _add_to_cache(self, key: str, value: Any) -> None:
        """Add item to cache if enabled"""
        if self.cache_enabled:
            self._cache[key] = value
    
    def _time_operation(self, operation_name: str) -> callable:
        """Decorator for timing operations"""
        def decorator(func):
            def wrapper(*args, **kwargs):
                start_time = time.time()
                result = func(*args, **kwargs)
                end_time = time.time()
                
                if operation_name in self.calculation_times:
                    # Running average
                    prev_avg = self.calculation_times[operation_name]['avg']
                    prev_count = self.calculation_times[operation_name]['count']
                    new_avg = (prev_avg * prev_count + (end_time - start_time)) / (prev_count + 1)
                    
                    self.calculation_times[operation_name] = {
                        'avg': new_avg,
                        'min': min(self.calculation_times[operation_name]['min'], end_time - start_time),
                        'max': max(self.calculation_times[operation_name]['max'], end_time - start_time),
                        'count': prev_count + 1
                    }
                else:
                    self.calculation_times[operation_name] = {
                        'avg': end_time - start_time,
                        'min': end_time - start_time,
                        'max': end_time - start_time,
                        'count': 1
                    }
                
                return result
            return wrapper
        return decorator
    
    @_time_operation("calculate_rsi")
    def calculate_rsi(self, period: int = 14, column: str = 'close') -> pd.Series:
        """
        Calculate Relative Strength Index (RSI) with performance optimization.
        
        Args:
            period: RSI calculation period
            column: Price column to use
            
        Returns:
            Series containing RSI values
        """
        cache_key = f"rsi_{period}_{column}"
        cached = self._get_from_cache(cache_key)
        if cached is not None:
            return cached
            
        if self.df.empty:
            return pd.Series()
            
        if column not in self.df.columns:
            logging.warning(f"Column {column} not found in dataframe for RSI calculation")
            return pd.Series()
            
        # RSI calculation - modified from 'ta' library for performance
        delta = self.df[column].diff()
        
        # Separate gains and losses
        gains = delta.copy()
        losses = delta.copy()
        gains[gains < 0] = 0
        losses[losses > 0] = 0
        losses = -losses  # Make losses positive
        
        # Use exponential moving average for efficiency
        avg_gain = gains.ewm(alpha=1/period, min_periods=period).mean()
        avg_loss = losses.ewm(alpha=1/period, min_periods=period).mean()
        
        # Calculate RS and RSI
        rs = avg_gain / avg_loss.replace(0, np.finfo(float).eps)  # Avoid division by zero
        rsi = 100 - (100 / (1 + rs))
        
        self._add_to_cache(cache_key, rsi)
        return rsi
    
    @_time_operation("calculate_macd")
    def calculate_macd(self, fast_period: int = 12, slow_period: int = 26, signal_period: int = 9, 
                      column: str = 'close') -> pd.DataFrame:
        """
        Calculate MACD Indicator with performance optimization.
        
        Args:
            fast_period: Fast EMA period
            slow_period: Slow EMA period
            signal_period: Signal line period
            column: Price column to use
            
        Returns:
            DataFrame with MACD, Signal and Histogram
        """
        cache_key = f"macd_{fast_period}_{slow_period}_{signal_period}_{column}"
        cached = self._get_from_cache(cache_key)
        if cached is not None:
            return cached
            
        if self.df.empty:
            return pd.DataFrame()
            
        if column not in self.df.columns:
            logging.warning(f"Column {column} not found in dataframe for MACD calculation")
            return pd.DataFrame()
            
        try:
            # Use pandas built-in ewm for optimal performance
            ema_fast = self.df[column].ewm(span=fast_period, adjust=False).mean()
            ema_slow = self.df[column].ewm(span=slow_period, adjust=False).mean()
            
            # Calculate MACD line and signal line
            macd_line = ema_fast - ema_slow
            signal_line = macd_line.ewm(span=signal_period, adjust=False).mean()
            histogram = macd_line - signal_line
            
            # Return as DataFrame
            result = pd.DataFrame({
                'macd': macd_line,
                'signal': signal_line,
                'histogram': histogram
            })
            
            self._add_to_cache(cache_key, result)
            return result
            
        except Exception as e:
            logging.error(f"Error calculating MACD: {e}")
            return pd.DataFrame()
    
    @_time_operation("calculate_bollinger_bands")
    def calculate_bollinger_bands(self, period: int = 20, std_dev: float = 2.0, 
                                column: str = 'close') -> pd.DataFrame:
        """
        Calculate Bollinger Bands with performance optimization.
        
        Args:
            period: Moving average period
            std_dev: Number of standard deviations
            column: Price column to use
            
        Returns:
            DataFrame with upper, middle, and lower bands
        """
        cache_key = f"bbands_{period}_{std_dev}_{column}"
        cached = self._get_from_cache(cache_key)
        if cached is not None:
            return cached
            
        if self.df.empty:
            return pd.DataFrame()
            
        if column not in self.df.columns:
            logging.warning(f"Column {column} not found in dataframe for Bollinger Bands calculation")
            return pd.DataFrame()
            
        try:
            # Calculate middle band (SMA)
            middle_band = self.df[column].rolling(window=period).mean()
            
            # Calculate standard deviation
            std = self.df[column].rolling(window=period).std()
            
            # Calculate upper and lower bands
            upper_band = middle_band + (std * std_dev)
            lower_band = middle_band - (std * std_dev)
            
            # Calculate bandwidth and %B
            # Add epsilon to avoid division by zero
            bandwidth = (upper_band - lower_band) / (middle_band + 1e-9)
            percent_b = (self.df[column] - lower_band) / (upper_band - lower_band + 1e-9)
            
            result = pd.DataFrame({
                'upper': upper_band,
                'middle': middle_band,
                'lower': lower_band,
                'bandwidth': bandwidth,
                'percent_b': percent_b
            })
            
            self._add_to_cache(cache_key, result)
            return result
            
        except Exception as e:
            logging.error(f"Error calculating Bollinger Bands: {e}")
            return pd.DataFrame()
    
    @_time_operation("calculate_atr")
    def calculate_atr(self, period: int = 14) -> pd.Series:
        """
        Calculate Average True Range for volatility analysis with optimized performance.
        
        Args:
            period: ATR period
            
        Returns:
            Series with ATR values
        """
        cache_key = f"atr_{period}"
        cached = self._get_from_cache(cache_key)
        if cached is not None:
            return cached
            
        if self.df.empty or len(self.df) <= period:
            return pd.Series()
            
        try:
            # Use vectorized operations for performance
            high_low = self.df['high'] - self.df['low']
            high_close = np.abs(self.df['high'] - self.df['close'].shift())
            low_close = np.abs(self.df['low'] - self.df['close'].shift())
            
            # Stack arrays horizontally and compute row-wise maximum
            ranges = np.vstack([high_low.values, high_close.values, low_close.values])
            true_range = pd.Series(np.max(ranges, axis=0), index=self.df.index)
            
            # For first row, TR is simply High-Low
            true_range.iloc[0] = high_low.iloc[0]
            
            # Calculate rolling average
            atr = true_range.rolling(window=period).mean()
            
            self._add_to_cache(cache_key, atr)
            return atr
            
        except Exception as e:
            logging.error(f"Error calculating ATR: {e}")
            return pd.Series()
    
    @_time_operation("calculate_stochastic")
    def calculate_stochastic(self, k_period: int = 14, d_period: int = 3) -> pd.DataFrame:
        """
        Calculate Stochastic Oscillator with optimized performance.
        
        Args:
            k_period: %K period
            d_period: %D period
            
        Returns:
            DataFrame with %K and %D values
        """
        cache_key = f"stoch_{k_period}_{d_period}"
        cached = self._get_from_cache(cache_key)
        if cached is not None:
            return cached
            
        if self.df.empty or len(self.df) <= k_period:
            return pd.DataFrame()
            
        try:
            # Calculate %K
            low_min = self.df['low'].rolling(window=k_period).min()
            high_max = self.df['high'].rolling(window=k_period).max()
            
            # Handle division by zero
            range_val = high_max - low_min
            
            # Using numpy where for vectorized operation
            # If range is zero, replace with small value to prevent division by zero
            range_val = np.where(range_val == 0, 1e-9, range_val)
            
            k = 100 * ((self.df['close'] - low_min) / range_val)
            
            # Calculate %D (moving average of %K)
            d = k.rolling(window=d_period).mean()
            
            result = pd.DataFrame({'k': k, 'd': d})
            
            self._add_to_cache(cache_key, result)
            return result
            
        except Exception as e:
            logging.error(f"Error calculating Stochastic Oscillator: {e}")
            return pd.DataFrame()
    
    @_time_operation("calculate_volume_analysis")
    def calculate_volume_analysis(self) -> Dict[str, pd.Series]:
        """
        Perform volume analysis to detect volume patterns with optimized calculations.
        
        Returns:
            Dictionary of volume indicators
        """
        cache_key = "volume_analysis"
        cached = self._get_from_cache(cache_key)
        if cached is not None:
            return cached
            
        if self.df.empty or 'volume' not in self.df.columns:
            return {}
            
        try:
            # Calculate average volume
            avg_volume = self.df['volume'].rolling(window=20).mean()
            
            # Detect volume spikes (volume > 2x average)
            volume_spike = self.df['volume'] > (avg_volume * 2)
            
            # Calculate volume rate of change
            volume_roc = self.df['volume'].pct_change() * 100
            
            # Calculate On-Balance Volume (OBV)
            # This requires a more optimized approach as it can't be fully vectorized
            price_change = self.df['close'].diff()
            obv = pd.Series(0, index=self.df.index)
            
            # First value is zero
            obv.iloc[0] = 0
            
            # Vectorized calculation
            obv = obv.add(
                np.where(price_change > 0, self.df['volume'], 
                         np.where(price_change < 0, -self.df['volume'], 0))
            ).cumsum()
            
            # Calculate Chaikin Money Flow (CMF)
            period = 20
            mf_multiplier = (
                (self.df['close'] - self.df['low']) - (self.df['high'] - self.df['close'])
            ) / (self.df['high'] - self.df['low'] + 1e-9)  # Add small value to prevent division by zero
            
            mf_volume = mf_multiplier * self.df['volume']
            cmf = mf_volume.rolling(window=period).sum() / self.df['volume'].rolling(window=period).sum()
            
            result = {
                'avg_volume': avg_volume,
                'volume_spike': volume_spike,
                'volume_roc': volume_roc,
                'obv': obv,
                'cmf': cmf
            }
            
            self._add_to_cache(cache_key, result)
            return result
            
        except Exception as e:
            logging.error(f"Error calculating volume analysis: {e}")
            return {}
    
    @_time_operation("calculate_ichimoku")
    def calculate_ichimoku(self, tenkan_period: int = 9, kijun_period: int = 26,
                          senkou_period: int = 52, displacement: int = 26) -> pd.DataFrame:
        """
        Calculate Ichimoku Cloud components with optimized calculations.
        
        Args:
            tenkan_period: Tenkan-sen (Conversion Line) period
            kijun_period: Kijun-sen (Base Line) period
            senkou_period: Senkou Span B period
            displacement: Displacement period for Senkou Span
            
        Returns:
            DataFrame with Ichimoku components
        """
        cache_key = f"ichimoku_{tenkan_period}_{kijun_period}_{senkou_period}_{displacement}"
        cached = self._get_from_cache(cache_key)
        if cached is not None:
            return cached
            
        if self.df.empty or len(self.df) <= max(tenkan_period, kijun_period, senkou_period):
            return pd.DataFrame()
            
        try:
            # Tenkan-sen (Conversion Line)
            tenkan_high = self.df['high'].rolling(window=tenkan_period).max()
            tenkan_low = self.df['low'].rolling(window=tenkan_period).min()
            tenkan_sen = (tenkan_high + tenkan_low) / 2
            
            # Kijun-sen (Base Line)
            kijun_high = self.df['high'].rolling(window=kijun_period).max()
            kijun_low = self.df['low'].rolling(window=kijun_period).min()
            kijun_sen = (kijun_high + kijun_low) / 2
            
            # Senkou Span A (Leading Span A)
            senkou_span_a = ((tenkan_sen + kijun_sen) / 2).shift(displacement)
            
            # Senkou Span B (Leading Span B)
            senkou_high = self.df['high'].rolling(window=senkou_period).max()
            senkou_low = self.df['low'].rolling(window=senkou_period).min()
            senkou_span_b = ((senkou_high + senkou_low) / 2).shift(displacement)
            
            # Chikou Span (Lagging Span)
            chikou_span = self.df['close'].shift(-displacement)
            
            result = pd.DataFrame({
                'tenkan_sen': tenkan_sen,
                'kijun_sen': kijun_sen,
                'senkou_span_a': senkou_span_a,
                'senkou_span_b': senkou_span_b,
                'chikou_span': chikou_span
            })
            
            self._add_to_cache(cache_key, result)
            return result
            
        except Exception as e:
            logging.error(f"Error calculating Ichimoku Cloud: {e}")
            return pd.DataFrame()
    
    @_time_operation("calculate_heikin_ashi")
    def calculate_heikin_ashi(self) -> pd.DataFrame:
        """
        Calculate Heikin-Ashi candlesticks for trend visualization.
        
        Returns:
            DataFrame with Heikin-Ashi OHLC
        """
        cache_key = "heikin_ashi"
        cached = self._get_from_cache(cache_key)
        if cached is not None:
            return cached
            
        if self.df.empty:
            return pd.DataFrame()
            
        try:
            ha_close = (self.df['open'] + self.df['high'] + self.df['low'] + self.df['close']) / 4
            
            # Initialize ha_open series
            ha_open = pd.Series(index=self.df.index)
            
            # First value is the average of first open and close
            ha_open.iloc[0] = (self.df['open'].iloc[0] + self.df['close'].iloc[0]) / 2
            
            # Rest of ha_open values
            for i in range(1, len(self.df)):
                ha_open.iloc[i] = (ha_open.iloc[i-1] + ha_close.iloc[i-1]) / 2
            
            # Calculate ha_high and ha_low
            ha_high = pd.concat([self.df['high'], ha_open, ha_close], axis=1).max(axis=1)
            ha_low = pd.concat([self.df['low'], ha_open, ha_close], axis=1).min(axis=1)
            
            result = pd.DataFrame({
                'ha_open': ha_open,
                'ha_high': ha_high,
                'ha_low': ha_low,
                'ha_close': ha_close
            })
            
            self._add_to_cache(cache_key, result)
            return result
            
        except Exception as e:
            logging.error(f"Error calculating Heikin-Ashi: {e}")
            return pd.DataFrame()
    
    @_time_operation("detect_support_resistance")
    def detect_support_resistance(self, window: int = 20, sensitivity: float = 0.02,
                                column: str = 'close') -> Dict[str, List[Dict[str, Any]]]:
        """
        Detect support and resistance levels with improved algorithm and optimization.
        
        Args:
            window: Window for local extrema detection
            sensitivity: Minimum price difference (as % of price)
            column: Price column to use
            
        Returns:
            Dict with support and resistance levels
        """
        cache_key = f"sup_res_{window}_{sensitivity}_{column}"
        cached = self._get_from_cache(cache_key)
        if cached is not None:
            return cached
            
        if self.df.empty or len(self.df) < window * 2:
            return {'support': [], 'resistance': []}
            
        try:
            # Find local minima and maxima using scipy's find_peaks for better detection
            # Invert for finding minima
            price = self.df[column].values
            
            # Find maxima (resistance)
            # Use distance parameter to ensure peaks are separated by at least window/2
            maxima_idx, _ = find_peaks(price, distance=window//2)
            
            # Find minima (support)
            minima_idx, _ = find_peaks(-price, distance=window//2)
            
            # Price level clustering function with optimization
            def cluster_levels(indices, price_accessor, tolerance):
                if not len(indices):
                    return []
                    
                # Extract price levels
                levels = [{'price': price_accessor(idx), 'indices': [idx], 'count': 1} 
                          for idx in indices]
                
                # Sort by price for efficient clustering
                levels.sort(key=lambda x: x['price'])
                
                # Cluster close levels
                clustered = []
                current_cluster = levels[0]
                
                for level in levels[1:]:
                    # If close to current cluster, merge
                    if abs(level['price'] - current_cluster['price']) / current_cluster['price'] < tolerance:
                        current_cluster['indices'].extend(level['indices'])
                        current_cluster['count'] += level['count']
                        # Update price to average of all points in cluster
                        current_cluster['price'] = sum(price_accessor(idx) for idx in current_cluster['indices']) / len(current_cluster['indices'])
                    else:
                        # New cluster
                        clustered.append(current_cluster)
                        current_cluster = level
                
                # Add last cluster
                clustered.append(current_cluster)
                
                # Sort by strength (count) and return
                return sorted(clustered, key=lambda x: x['count'], reverse=True)
            
            # Get levels
            resistance_levels = cluster_levels(
                maxima_idx, 
                lambda idx: price[idx], 
                sensitivity
            )
            
            support_levels = cluster_levels(
                minima_idx, 
                lambda idx: price[idx], 
                sensitivity
            )
            
            # Add extra information to levels
            for level in resistance_levels + support_levels:
                # How recent is the most recent touch
                level['recency'] = len(self.df) - max(level['indices'])
                # Add strength score based on count and recency
                level['strength'] = level['count'] * (1.0 - level['recency'] / len(self.df))
            
            result = {
                'support': support_levels,
                'resistance': resistance_levels
            }
            
            self._add_to_cache(cache_key, result)
            return result
            
        except Exception as e:
            logging.error(f"Error detecting support/resistance: {e}")
            return {'support': [], 'resistance': []}
    
    @_time_operation("detect_trend")
    def detect_trend(self, period: int = 50) -> Dict[str, Any]:
        """
        Detect market trend using multiple indicators with optimized calculations.
        
        Args:
            period: Period for trend analysis
            
        Returns:
            Dict with trend analysis
        """
        cache_key = f"trend_{period}"
        cached = self._get_from_cache(cache_key)
        if cached is not None:
            return cached
            
        if self.df.empty or len(self.df) < period:
            return {'trend': 'unknown', 'strength': 0, 'score': 0, 'details': {}}
            
        try:
            # Use multiple indicators for robust trend detection
            # 1. Moving averages
            ma_short = self.df['close'].rolling(window=20).mean()
            ma_medium = self.df['close'].rolling(window=50).mean()
            ma_long = self.df['close'].rolling(window=200).mean()
            
            # Get latest values
            current_close = self.df['close'].iloc[-1]
            current_ma_short = ma_short.iloc[-1]
            current_ma_medium = ma_medium.iloc[-1]
            current_ma_long = ma_long.iloc[-1]
            
            # 2. Price relative to moving averages
            price_vs_short = 1 if current_close > current_ma_short else -1
            price_vs_medium = 1 if current_close > current_ma_medium else -1
            price_vs_long = 1 if current_close > current_ma_long else -1
            
            # 3. Moving average alignment
            ma_aligned_bullish = 1 if (current_ma_short > current_ma_medium) and (current_ma_medium > current_ma_long) else 0
            ma_aligned_bearish = -1 if (current_ma_short < current_ma_medium) and (current_ma_medium < current_ma_long) else 0
            
            # 4. Price slope
            recent_prices = self.df['close'].iloc[-period:]
            slope = (recent_prices.iloc[-1] - recent_prices.iloc[0]) / recent_prices.iloc[0]
            slope_signal = 1 if slope > 0 else (-1 if slope < 0 else 0)
            
            # 5. Add RSI for trend strength confirmation
            rsi = self.calculate_rsi()
            if not rsi.empty:
                current_rsi = rsi.iloc[-1]
                rsi_signal = 1 if current_rsi > 60 else (-1 if current_rsi < 40 else 0)
            else:
                rsi_signal = 0
                
            # 6. Add MACD for trend confirmation
            macd_data = self.calculate_macd()
            if not macd_data.empty:
                macd_signal = 1 if macd_data['histogram'].iloc[-1] > 0 else -1
            else:
                macd_signal = 0
            
            # Combine all signals
            trend_signals = {
                'price_vs_short': price_vs_short,
                'price_vs_medium': price_vs_medium,
                'price_vs_long': price_vs_long,
                'ma_aligned_bullish': ma_aligned_bullish,
                'ma_aligned_bearish': ma_aligned_bearish,
                'slope': slope_signal,
                'rsi': rsi_signal,
                'macd': macd_signal
            }
            
            # Calculate trend score with weighted signals
            weights = {
                'price_vs_short': 0.15,
                'price_vs_medium': 0.15,
                'price_vs_long': 0.2,
                'ma_aligned_bullish': 0.1,
                'ma_aligned_bearish': 0.1,
                'slope': 0.1,
                'rsi': 0.1,
                'macd': 0.1
            }
            
            trend_score = sum(signal * weights[name] for name, signal in trend_signals.items())
            
            # Normalize score to -1 to 1 range
            trend_score = max(-1, min(1, trend_score))
            
            # Determine trend category
            if trend_score >= 0.6:
                trend = 'strong_bullish'
            elif trend_score >= 0.2:
                trend = 'bullish'
            elif trend_score <= -0.6:
                trend = 'strong_bearish'
            elif trend_score <= -0.2:
                trend = 'bearish'
            else:
                trend = 'neutral'
                
            # Calculate trend strength (0-1)
            strength = abs(trend_score)
            
            result = {
                'trend': trend,
                'strength': strength,
                'score': trend_score,
                'details': trend_signals
            }
            
            self._add_to_cache(cache_key, result)
            return result
            
        except Exception as e:
            logging.error(f"Error detecting trend: {e}")
            return {'trend': 'unknown', 'strength': 0, 'score': 0, 'details': {}}
    
    @_time_operation("detect_volatility_regime")
    def detect_volatility_regime(self, short_period: int = 5, long_period: int = 20) -> Dict[str, Any]:
        """
        Detect market volatility regime with optimized calculations.
        
        Args:
            short_period: Period for short-term volatility
            long_period: Period for long-term volatility
            
        Returns:
            Dict with volatility analysis
        """
        cache_key = f"vol_regime_{short_period}_{long_period}"
        cached = self._get_from_cache(cache_key)
        if cached is not None:
            return cached
            
        if self.df.empty or len(self.df) < long_period:
            return {'volatility': 'unknown', 'level': 0, 'expanding': False}
            
        try:
            # Calculate ATR for volatility measurement
            atr = self.calculate_atr(period=14)
            
            if atr.empty:
                return {'volatility': 'unknown', 'level': 0, 'expanding': False}
            
            # Calculate ATR as percentage of price for normalization
            atr_pct = atr / self.df['close'] * 100
            
            # Calculate short and long-term volatility
            vol_short = atr_pct.rolling(window=short_period).mean().iloc[-1]
            vol_long = atr_pct.rolling(window=long_period).mean().iloc[-1]
            
            # Compare current volatility to historical
            historical_vol = atr_pct.mean()
            vol_ratio = vol_short / historical_vol if historical_vol > 0 else 1.0
            
            # Determine volatility regime
            if vol_ratio > 2.0:
                regime = 'extreme'
                level = 1.0
            elif vol_ratio > 1.5:
                regime = 'high'
                level = 0.75
            elif vol_ratio > 1.0:
                regime = 'elevated'
                level = 0.5
            elif vol_ratio > 0.7:
                regime = 'normal'
                level = 0.25
            else:
                regime = 'low'
                level = 0.0
                
            # Check if volatility is expanding or contracting
            expanding = vol_short > vol_long
            
            result = {
                'volatility': regime,
                'level': level,
                'expanding': expanding,
                'atr': atr.iloc[-1],
                'atr_pct': atr_pct.iloc[-1],
                'vol_ratio': vol_ratio,
                'vol_short': vol_short,
                'vol_long': vol_long
            }
            
            self._add_to_cache(cache_key, result)
            return result
            
        except Exception as e:
            logging.error(f"Error detecting volatility: {e}")
            return {'volatility': 'unknown', 'level': 0, 'expanding': False}
    
    @_time_operation("analyze_market_structure")
    def analyze_market_structure(self) -> Dict[str, Any]:
        """
        Analyze market structure to detect patterns and market regime with optimized approach.
        
        Returns:
            Dict with market structure analysis
        """
        cache_key = "market_structure"
        cached = self._get_from_cache(cache_key)
        if cached is not None:
            return cached
            
        if self.df.empty:
            return {
                'market_regime': 'unknown',
                'patterns': {},
                'momentum': 0,
                'sentiment': 'neutral',
                'cache_status': 'empty_data'
            }
            
        try:
            # Use multi-processing if enabled for heavy calculations
            if self.parallel and self.executor:
                # Prepare futures for parallel execution
                trend_future = self.executor.submit(self.detect_trend)
                volatility_future = self.executor.submit(self.detect_volatility_regime)
                patterns_future = self.executor.submit(self.detect_chart_patterns)
                
                # Calculate momentum indicators (lighter calculations)
                rsi = self.calculate_rsi()
                macd_data = self.calculate_macd()
                stoch_data = self.calculate_stochastic()
                
                # Get results from futures
                trend_analysis = trend_future.result()
                volatility_analysis = volatility_future.result()
                patterns = patterns_future.result()
                
            else:
                # Sequential execution
                trend_analysis = self.detect_trend()
                volatility_analysis = self.detect_volatility_regime()
                patterns = self.detect_chart_patterns()
                
                rsi = self.calculate_rsi()
                macd_data = self.calculate_macd()
                stoch_data = self.calculate_stochastic()
            
            # Get momentum indicators
            latest_rsi = rsi.iloc[-1] if not rsi.empty else 50
            latest_macd = macd_data['histogram'].iloc[-1] if not macd_data.empty else 0
            latest_stoch_k = stoch_data['k'].iloc[-1] if not stoch_data.empty else 50
            
            # Normalize momentum indicators to -1 to 1 range
            norm_rsi = (latest_rsi - 50) / 50  # -1 to 1
            norm_macd = np.tanh(latest_macd)  # Squash to -1 to 1
            norm_stoch = (latest_stoch_k - 50) / 50  # -1 to 1
            
            # Combine for momentum score
            momentum_score = (norm_rsi + norm_macd + norm_stoch) / 3
            
            # Determine market regime
            if trend_analysis['trend'] in ['strong_bullish', 'bullish'] and volatility_analysis['volatility'] in ['normal', 'low']:
                market_regime = 'trending_bullish'
            elif trend_analysis['trend'] in ['strong_bearish', 'bearish'] and volatility_analysis['volatility'] in ['normal', 'low']:
                market_regime = 'trending_bearish'
            elif trend_analysis['trend'] == 'neutral' and volatility_analysis['volatility'] in ['normal', 'low']:
                market_regime = 'ranging'
            elif volatility_analysis['volatility'] in ['high', 'extreme']:
                if trend_analysis['trend'] in ['strong_bullish', 'bullish']:
                    market_regime = 'volatile_bullish'
                elif trend_analysis['trend'] in ['strong_bearish', 'bearish']:
                    market_regime = 'volatile_bearish'
                else:
                    market_regime = 'volatile_neutral'
            else:
                market_regime = 'undefined'
                
            # Determine overall sentiment
            if momentum_score > 0.3:
                sentiment = 'bullish'
            elif momentum_score < -0.3:
                sentiment = 'bearish'
            else:
                sentiment = 'neutral'
                
            # Store results for later use
            self.market_regime = market_regime
            self.patterns = patterns
            
            result = {
                'market_regime': market_regime,
                'trend': trend_analysis,
                'volatility': volatility_analysis,
                'momentum': momentum_score,
                'sentiment': sentiment,
                'patterns': patterns,
                'timestamp': pd.Timestamp.now().isoformat(),
                'cache_status': 'calculated'
            }
            
            self._add_to_cache(cache_key, result)
            return result
            
        except Exception as e:
            logging.error(f"Error analyzing market structure: {e}")
            logging.error(traceback.format_exc())
            return {
                'market_regime': 'unknown',
                'patterns': {},
                'momentum': 0,
                'sentiment': 'neutral',
                'error': str(e),
                'cache_status': 'error'
            }
    
    @_time_operation("detect_chart_patterns")
    def detect_chart_patterns(self) -> Dict[str, Dict[str, Any]]:
        """
        Detect common chart patterns with pattern recognition algorithms.
        
        Returns:
            Dict of detected patterns
        """
        cache_key = "chart_patterns"
        cached = self._get_from_cache(cache_key)
        if cached is not None:
            return cached
            
        if self.df.empty or len(self.df) < 30:
            return {}
            
        try:
            patterns = {}
            
            # Detect double top/bottom 
            double_patterns = self._detect_double_patterns()
            patterns.update(double_patterns)
            
            # Detect head and shoulders
            hs_patterns = self._detect_head_and_shoulders()
            patterns.update(hs_patterns)
            
            # Detect triangle patterns
            triangle_patterns = self._detect_triangle_patterns()
            patterns.update(triangle_patterns)
            
            # Detect trend continuation patterns
            cont_patterns = self._detect_continuation_patterns()
            patterns.update(cont_patterns)
            
            self._add_to_cache(cache_key, patterns)
            return patterns
            
        except Exception as e:
            logging.error(f"Error detecting chart patterns: {e}")
            return {}
    
    def _detect_double_patterns(self) -> Dict[str, Dict[str, Any]]:
        """
        Helper to detect double top/bottom patterns with improved algorithm.
        
        Returns:
            Dict of detected double patterns
        """
        if self.df.empty or len(self.df) < 20:
            return {}
            
        try:
            # Find local extrema using scipy's find_peaks for better detection
            price_high = self.df['high'].values
            price_low = self.df['low'].values
            
            # Detect peaks (for double top)
            # We need minimum height to qualify as a peak and minimum distance between peaks
            peak_height = np.std(price_high) * 0.5  # Dynamic height based on volatility
            peak_indices, _ = find_peaks(price_high, height=np.median(price_high), distance=5, prominence=peak_height)
            
            # Detect valleys (for double bottom)
            # Invert the data to find valleys
            valley_height = np.std(price_low) * 0.5
            valley_indices, _ = find_peaks(-price_low, height=-np.median(price_low), distance=5, prominence=valley_height)
            
            patterns = {}
            
            # Check for double top - more rigorous criteria
            if len(peak_indices) >= 2:
                # Get last two peaks
                last_two_peaks = peak_indices[-2:]
                
                # Check if peaks are similar height (within 2%)
                peak1_val = price_high[last_two_peaks[0]]
                peak2_val = price_high[last_two_peaks[1]]
                
                peak_similar = abs(peak1_val - peak2_val) / peak1_val < 0.02
                
                # Check if there's a significant valley between peaks
                price_between = self.df['low'].iloc[last_two_peaks[0]:last_two_peaks[1]]
                if not price_between.empty:
                    min_between = price_between.min()
                    valley_depth = (peak1_val - min_between) / peak1_val
                    
                    # Valid double top needs significant valley and similar peak heights
                    if peak_similar and valley_depth > 0.03:
                        patterns['double_top'] = {
                            'confidence': 0.7 + (valley_depth * 2),  # Higher confidence with deeper valley
                            'first_peak': self.df.index[last_two_peaks[0]],
                            'second_peak': self.df.index[last_two_peaks[1]],
                            'valley_depth': valley_depth
                        }
            
            # Check for double bottom - similar approach
            if len(valley_indices) >= 2:
                # Get last two valleys
                last_two_valleys = valley_indices[-2:]
                
                # Check if valleys are similar height (within 2%)
                valley1_val = price_low[last_two_valleys[0]]
                valley2_val = price_low[last_two_valleys[1]]
                
                valley_similar = abs(valley1_val - valley2_val) / valley1_val < 0.02
                
                # Check if there's a significant peak between valleys
                price_between = self.df['high'].iloc[last_two_valleys[0]:last_two_valleys[1]]
                if not price_between.empty:
                    max_between = price_between.max()
                    peak_height = (max_between - valley1_val) / valley1_val
                    
                    # Valid double bottom needs significant peak and similar valley depths
                    if valley_similar and peak_height > 0.03:
                        patterns['double_bottom'] = {
                            'confidence': 0.7 + (peak_height * 2),  # Higher confidence with taller peak
                            'first_bottom': self.df.index[last_two_valleys[0]],
                            'second_bottom': self.df.index[last_two_valleys[1]],
                            'peak_height': peak_height
                        }
            
            return patterns
            
        except Exception as e:
            logging.error(f"Error detecting double patterns: {e}")
            return {}
    
    def _detect_head_and_shoulders(self) -> Dict[str, Dict[str, Any]]:
        """
        Helper to detect head and shoulders patterns.
        
        Returns:
            Dict with head and shoulders patterns
        """
        if self.df.empty or len(self.df) < 40:
            return {}
            
        try:
            patterns = {}
            
            # Find local maxima and minima
            price_high = self.df['high'].values
            price_low = self.df['low'].values
            
            # For head and shoulders pattern we need at least 5 extrema points
            # (left shoulder peak, left valley, head peak, right valley, right shoulder peak)
            peak_height = np.std(price_high) * 0.5
            peak_indices, peak_props = find_peaks(price_high, height=np.median(price_high), distance=5, prominence=peak_height)
            
            valley_height = np.std(price_low) * 0.5
            valley_indices, valley_props = find_peaks(-price_low, height=-np.median(price_low), distance=5, prominence=valley_height)
            
            # We need minimum 3 peaks and 2 valleys for H&S
            if len(peak_indices) >= 3 and len(valley_indices) >= 2:
                # Check the last 5 extrema points in chronological order
                extrema = []
                for peak_idx in peak_indices:
                    extrema.append(('peak', peak_idx, price_high[peak_idx]))
                
                for valley_idx in valley_indices:
                    extrema.append(('valley', valley_idx, price_low[valley_idx]))
                
                # Sort by index position
                extrema.sort(key=lambda x: x[1])
                
                # Get last 7 extrema points if available (to cover full pattern)
                extrema = extrema[-7:]
                
                # Check for regular H&S pattern (bearish reversal)
                # Sequence: peak, valley, higher peak, valley, lower peak
                for i in range(len(extrema) - 4):
                    if (extrema[i][0] == 'peak' and 
                        extrema[i+1][0] == 'valley' and 
                        extrema[i+2][0] == 'peak' and 
                        extrema[i+3][0] == 'valley' and 
                        extrema[i+4][0] == 'peak'):
                        
                        # Check the pattern criteria
                        left_shoulder = extrema[i][2]
                        head = extrema[i+2][2]
                        right_shoulder = extrema[i+4][2]
                        
                        # Head should be higher than shoulders
                        if head > left_shoulder and head > right_shoulder:
                            # Shoulders should be roughly at same height (within 10%)
                            if abs(left_shoulder - right_shoulder) / left_shoulder < 0.1:
                                # Valid H&S pattern
                                patterns['head_and_shoulders'] = {
                                    'confidence': 0.8,
                                    'left_shoulder': self.df.index[extrema[i][1]],
                                    'head': self.df.index[extrema[i+2][1]],
                                    'right_shoulder': self.df.index[extrema[i+4][1]],
                                    'neckline': min(price_low[extrema[i+1][1]], price_low[extrema[i+3][1]])
                                }
                                break
                
                # Check for inverse H&S pattern (bullish reversal)
                # Sequence: valley, peak, deeper valley, peak, higher valley
                for i in range(len(extrema) - 4):
                    if (extrema[i][0] == 'valley' and 
                        extrema[i+1][0] == 'peak' and 
                        extrema[i+2][0] == 'valley' and 
                        extrema[i+3][0] == 'peak' and 
                        extrema[i+4][0] == 'valley'):
                        
                        # Check the pattern criteria
                        left_shoulder = extrema[i][2]
                        head = extrema[i+2][2]
                        right_shoulder = extrema[i+4][2]
                        
                        # Head should be lower than shoulders
                        if head < left_shoulder and head < right_shoulder:
                            # Shoulders should be roughly at same height (within 10%)
                            if abs(left_shoulder - right_shoulder) / left_shoulder < 0.1:
                                # Valid inverse H&S pattern
                                patterns['inverse_head_and_shoulders'] = {
                                    'confidence': 0.8,
                                    'left_shoulder': self.df.index[extrema[i][1]],
                                    'head': self.df.index[extrema[i+2][1]],
                                    'right_shoulder': self.df.index[extrema[i+4][1]],
                                    'neckline': max(price_high[extrema[i+1][1]], price_high[extrema[i+3][1]])
                                }
                                break
            
            return patterns
            
        except Exception as e:
            logging.error(f"Error detecting head and shoulders: {e}")
            return {}
    
    def _detect_triangle_patterns(self) -> Dict[str, Dict[str, Any]]:
        """
        Helper to detect triangle patterns (ascending, descending, symmetric).
        
        Returns:
            Dict with triangle patterns
        """
        if self.df.empty or len(self.df) < 30:
            return {}
            
        try:
            patterns = {}
            
            # Get high and low prices
            highs = self.df['high'].values
            lows = self.df['low'].values
            
            # Find local maxima and minima
            peak_indices, _ = find_peaks(highs, distance=5, prominence=np.std(highs) * 0.5)
            valley_indices, _ = find_peaks(-lows, distance=5, prominence=np.std(lows) * 0.5)
            
            # Need at least 3 points to form a valid triangle
            if len(peak_indices) >= 3 and len(valley_indices) >= 3:
                # Get last peaks and valleys (most recent 10-15 candles)
                recent_peaks = [idx for idx in peak_indices if idx >= len(highs) - 30]
                recent_valleys = [idx for idx in valley_indices if idx >= len(lows) - 30]
                
                if len(recent_peaks) >= 3 and len(recent_valleys) >= 3:
                    # Linear regression on peaks and valleys
                    peak_x = np.array(recent_peaks)
                    peak_y = highs[recent_peaks]
                    peak_slope, peak_intercept = np.polyfit(peak_x, peak_y, 1)
                    
                    valley_x = np.array(recent_valleys)
                    valley_y = lows[recent_valleys]
                    valley_slope, valley_intercept = np.polyfit(valley_x, valley_y, 1)
                    
                    # Check for triangle patterns
                    # Symmetric triangle: peak slope down, valley slope up
                    if peak_slope < -0.0001 and valley_slope > 0.0001:
                        patterns['symmetric_triangle'] = {
                            'confidence': 0.7,
                            'peak_slope': peak_slope,
                            'valley_slope': valley_slope,
                            'converging_point': int((valley_intercept - peak_intercept) / (peak_slope - valley_slope))
                        }
                    # Ascending triangle: peak slope flat, valley slope up
                    elif abs(peak_slope) < 0.0001 and valley_slope > 0.0001:
                        patterns['ascending_triangle'] = {
                            'confidence': 0.7,
                            'resistance_level': np.mean(peak_y),
                            'valley_slope': valley_slope
                        }
                    # Descending triangle: peak slope down, valley slope flat
                    elif peak_slope < -0.0001 and abs(valley_slope) < 0.0001:
                        patterns['descending_triangle'] = {
                            'confidence': 0.7,
                            'support_level': np.mean(valley_y),
                            'peak_slope': peak_slope
                        }
            
            return patterns
            
        except Exception as e:
            logging.error(f"Error detecting triangle patterns: {e}")
            return {}
    
    def _detect_continuation_patterns(self) -> Dict[str, Dict[str, Any]]:
        """
        Helper to detect trend continuation patterns like flags and pennants.
        
        Returns:
            Dict with continuation patterns
        """
        if self.df.empty or len(self.df) < 30:
            return {}
            
        try:
            patterns = {}
            
            # Get trend direction first
            trend = self.detect_trend()
            if trend['trend'] in ['neutral', 'unknown']:
                return {}  # No clear trend for continuation patterns
                
            is_uptrend = trend['trend'] in ['bullish', 'strong_bullish']
            
            # Look for sharp move (flag pole) followed by consolidation
            # Calculate rate of change
            roc = self.df['close'].pct_change(5) * 100  # 5-period rate of change
            
            # Find significant moves (potential flag poles)
            threshold = np.std(roc) * 2
            significant_moves = []
            
            for i in range(10, len(roc)):
                if is_uptrend and roc.iloc[i] > threshold:
                    significant_moves.append(('up', i))
                elif not is_uptrend and roc.iloc[i] < -threshold:
                    significant_moves.append(('down', i))
            
            # Check for consolidation after each move
            for direction, start_idx in significant_moves:
                # Skip if too close to the end
                if start_idx + 10 >= len(self.df):
                    continue
                    
                # Get consolidation period (10 candles after significant move)
                consol_high = self.df['high'].iloc[start_idx:start_idx+10]
                consol_low = self.df['low'].iloc[start_idx:start_idx+10]
                
                # Calculate linear regression for consolidation highs and lows
                x = np.arange(len(consol_high))
                high_slope, _ = np.polyfit(x, consol_high, 1)
                low_slope, _ = np.polyfit(x, consol_low, 1)
                
                # Flag pattern: parallel channels in counter-trend direction
                if (direction == 'up' and -0.001 < high_slope < 0 and -0.001 < low_slope < 0) or \
                   (direction == 'down' and 0 < high_slope < 0.001 and 0 < low_slope < 0.001):
                    
                    # Calculate channel width (should be relatively narrow)
                    channel_width = np.mean(consol_high - consol_low) / np.mean(consol_low)
                    
                    if channel_width < 0.03:  # Narrow channel
                        patterns['flag'] = {
                            'confidence': 0.7,
                            'direction': direction,
                            'pole_start': self.df.index[start_idx - 5],
                            'flag_start': self.df.index[start_idx],
                            'flag_end': self.df.index[start_idx + 9]
                        }
                        break
                    else:
                        # Wider channel is still a flag but with lower confidence
                        patterns['flag'] = {
                            'confidence': 0.5,
                            'direction': direction,
                            'pole_start': self.df.index[start_idx - 5],
                            'flag_start': self.df.index[start_idx],
                            'flag_end': self.df.index[start_idx + 9]
                        }
                        break
                
                # Pennant pattern: converging triangle after strong move
                elif (direction == 'up' and high_slope < 0 and low_slope > 0) or \
                     (direction == 'down' and high_slope > 0 and low_slope < 0):
                    
                    patterns['pennant'] = {
                        'confidence': 0.7,
                        'direction': direction,
                        'pole_start': self.df.index[start_idx - 5],
                        'pennant_start': self.df.index[start_idx],
                        'pennant_end': self.df.index[start_idx + 9]
                    }
                    break
            
            return patterns
            
        except Exception as e:
            logging.error(f"Error detecting continuation patterns: {e}")
            return {}
    
    @_time_operation("generate_signals")
    def generate_signals(self) -> Dict[str, Any]:
        """
        Generate trading signals based on comprehensive indicator analysis.
        Optimized for performance and signal quality.
        
        Returns:
            Dict with trading signals and confidence
        """
        cache_key = "signals"
        cached = self._get_from_cache(cache_key)
        if cached is not None:
            return cached
            
        if self.df.empty:
            return {'signals': {}, 'final': 0, 'confidence': 0}
            
        try:
            # Calculate key indicators
            if self.parallel and self.executor:
                # Parallel execution for heavy calculations
                rsi_future = self.executor.submit(self.calculate_rsi)
                macd_future = self.executor.submit(self.calculate_macd)
                bb_future = self.executor.submit(self.calculate_bollinger_bands)
                volume_future = self.executor.submit(self.calculate_volume_analysis)
                trend_future = self.executor.submit(self.detect_trend)
                
                # Get results
                rsi = rsi_future.result()
                macd_data = macd_future.result()
                bb_data = bb_future.result()
                volume_data = volume_future.result()
                trend_analysis = trend_future.result()
                
            else:
                # Sequential execution
                rsi = self.calculate_rsi()
                macd_data = self.calculate_macd()
                bb_data = self.calculate_bollinger_bands()
                volume_data = self.calculate_volume_analysis()
                trend_analysis = self.detect_trend()
            
            # RSI signals - more nuanced
            rsi_signal = 0
            if not rsi.empty:
                latest_rsi = rsi.iloc[-1]
                prev_rsi = rsi.iloc[-2] if len(rsi) > 1 else latest_rsi
                
                if latest_rsi < 30:
                    # Oversold - bullish
                    rsi_signal = 1 - (latest_rsi / 30)  # Stronger signal for more oversold
                elif latest_rsi > 70:
                    # Overbought - bearish
                    rsi_signal = -1 * ((latest_rsi - 70) / 30)  # Stronger signal for more overbought
                elif latest_rsi < 50 and prev_rsi > latest_rsi:
                    # Falling momentum in lower half - slightly bearish
                    rsi_signal = -0.2
                elif latest_rsi > 50 and prev_rsi < latest_rsi:
                    # Rising momentum in upper half - slightly bullish
                    rsi_signal = 0.2
                
            # MACD signals - more sophisticated signal calculation
            macd_signal = 0
            if not macd_data.empty:
                # Get recent values
                recent_macd = macd_data['macd'].iloc[-10:] if len(macd_data) >= 10 else macd_data['macd']
                recent_signal = macd_data['signal'].iloc[-10:] if len(macd_data) >= 10 else macd_data['signal']
                recent_hist = macd_data['histogram'].iloc[-10:] if len(macd_data) >= 10 else macd_data['histogram']
                
                # MACD crossover (recent)
                if macd_data['macd'].iloc[-1] > macd_data['signal'].iloc[-1] and macd_data['macd'].iloc[-2] <= macd_data['signal'].iloc[-2]:
                    macd_signal = 1  # Bullish crossover
                elif macd_data['macd'].iloc[-1] < macd_data['signal'].iloc[-1] and macd_data['macd'].iloc[-2] >= macd_data['signal'].iloc[-2]:
                    macd_signal = -1  # Bearish crossover
                    
                # MACD histogram direction
                hist_direction = macd_data['histogram'].iloc[-1] - macd_data['histogram'].iloc[-2]
                if hist_direction > 0 and macd_data['histogram'].iloc[-1] < 0:
                    # Negative histogram but improving (potential bullish divergence)
                    macd_signal += 0.5
                elif hist_direction < 0 and macd_data['histogram'].iloc[-1] > 0:
                    # Positive histogram but deteriorating (potential bearish divergence)
                    macd_signal -= 0.5
                    
                # MACD divergence detection (basic)
                if len(self.df) >= 20:
                    price_direction = 1 if self.df['close'].iloc[-1] > self.df['close'].iloc[-10] else -1
                    macd_direction = 1 if recent_macd.iloc[-1] > recent_macd.iloc[0] else -1
                    
                    if price_direction != macd_direction:
                        # Divergence detected
                        macd_signal += 0.5 * -price_direction  # Opposite signal to price direction
                
            # Bollinger Band signals with enhanced rules
            bb_signal = 0
            if not bb_data.empty:
                # Price in relation to bands
                latest_close = self.df['close'].iloc[-1]
                
                # Calculate % distance from middle band
                middle = bb_data['middle'].iloc[-1]
                upper = bb_data['upper'].iloc[-1]
                lower = bb_data['lower'].iloc[-1]
                
                # Normalize to -1 to 1 range within the bands
                band_position = (latest_close - middle) / (upper - middle) if latest_close > middle else (latest_close - middle) / (middle - lower)
                
                # Band position gives a basic mean reversion signal
                bb_signal = -band_position * 0.5  # Scaled to give less weight
                
                # Price crossing bands (stronger signals)
                if latest_close < lower:
                    bb_signal = 1  # Price below lower band - bullish
                elif latest_close > upper:
                    bb_signal = -1  # Price above upper band - bearish
                    
                # Band contraction/expansion signals
                bandwidth = bb_data['bandwidth'].iloc[-1]
                avg_bandwidth = bb_data['bandwidth'].iloc[-20:].mean() if len(bb_data) >= 20 else bandwidth
                
                if bandwidth < avg_bandwidth * 0.8:
                    # Bands are contracting - potential breakout coming
                    # This is a neutral signal, but increases volatility expectation
                    bb_signal *= 0.5  # Reduce current signal since breakout direction is uncertain
                
            # Volume signals with enhanced analysis
            volume_signal = 0
            if volume_data and len(volume_data) > 0:
                # Volume spike confirmation
                if 'volume_spike' in volume_data and not volume_data['volume_spike'].empty:
                    recent_spikes = volume_data['volume_spike'].iloc[-5:]
                    if recent_spikes.any():
                        # Direction depends on price action during the spike
                        recent_price_action = self.df['close'].iloc[-5:] - self.df['open'].iloc[-5:]
                        
                        # Get price action during volume spikes
                        spike_indices = recent_spikes[recent_spikes].index
                        if not spike_indices.empty:
                            # Check price direction during spikes
                            spike_price_actions = [self.df['close'].loc[idx] > self.df['open'].loc[idx] for idx in spike_indices]
                            # Majority direction
                            if sum(spike_price_actions) > len(spike_price_actions) / 2:
                                volume_signal = 0.7  # Bullish volume spike
                            else:
                                volume_signal = -0.7  # Bearish volume spike
                
                # On-Balance Volume trend
                if 'obv' in volume_data and not volume_data['obv'].empty:
                    recent_obv = volume_data['obv'].iloc[-10:] if len(volume_data['obv']) >= 10 else volume_data['obv']
                    obv_slope = np.polyfit(np.arange(len(recent_obv)), recent_obv, 1)[0]
                    
                    if obv_slope > 0:
                        volume_signal += 0.3  # Bullish OBV trend
                    elif obv_slope < 0:
                        volume_signal -= 0.3  # Bearish OBV trend
                        
                # Money flow signals
                if 'cmf' in volume_data and not volume_data['cmf'].empty:
                    cmf = volume_data['cmf'].iloc[-1]
                    if cmf > 0.2:
                        volume_signal += 0.5  # Strong bullish money flow
                    elif cmf < -0.2:
                        volume_signal -= 0.5  # Strong bearish money flow
            
            # Trend analysis signal
            trend_signal = 0
            if trend_analysis:
                if trend_analysis['trend'] == 'strong_bullish':
                    trend_signal = 1.0
                elif trend_analysis['trend'] == 'bullish':
                    trend_signal = 0.5
                elif trend_analysis['trend'] == 'strong_bearish':
                    trend_signal = -1.0
                elif trend_analysis['trend'] == 'bearish':
                    trend_signal = -0.5
                    
                # Scale by strength
                trend_signal *= trend_analysis['strength']
                
            # Combine all signals with weights and normalizations
            signals = {
                'rsi': rsi_signal,
                'macd': macd_signal,
                'bollinger': bb_signal,
                'volume': volume_signal,
                'trend': trend_signal
            }
            
            # Calculate weighted signal with adaptive weights
            # Adjust weights based on market regime
            market_structure = self.analyze_market_structure()
            market_regime = market_structure.get('market_regime', 'undefined')
            
            # Default weights
            weights = {
                'rsi': 0.15,
                'macd': 0.2,
                'bollinger': 0.15,
                'volume': 0.1,
                'trend': 0.4
            }
            
            # Adjust weights based on market regime
            if 'trending' in market_regime:
                # In trending markets, give more weight to trend and MACD
                weights['trend'] = 0.5
                weights['macd'] = 0.25
                weights['rsi'] = 0.1
                weights['bollinger'] = 0.1
                weights['volume'] = 0.05
            elif 'ranging' in market_regime:
                # In ranging markets, give more weight to RSI and BB
                weights['trend'] = 0.2
                weights['macd'] = 0.15
                weights['rsi'] = 0.3
                weights['bollinger'] = 0.25
                weights['volume'] = 0.1
            elif 'volatile' in market_regime:
                # In volatile markets, volume becomes more important
                weights['trend'] = 0.3
                weights['macd'] = 0.15
                weights['rsi'] = 0.15
                weights['bollinger'] = 0.15
                weights['volume'] = 0.25
                
            # Normalize weights to sum to 1
            weight_sum = sum(weights.values())
            if weight_sum != 1.0:
                for k in weights:
                    weights[k] /= weight_sum
            
            # Calculate weighted signal
            weighted_signal = sum(signals[k] * weights[k] for k in signals)
            
            # Ensure signal is between -1 and 1
            weighted_signal = max(-1, min(1, weighted_signal))
            
            # Determine final signal
            if weighted_signal > 0.3:
                final_signal = 1  # Buy
            elif weighted_signal < -0.3:
                final_signal = -1  # Sell
            else:
                final_signal = 0  # Hold
                
            # Calculate confidence based on signal strength and agreement
            signal_values = [signals[k] for k in signals]
            
            # Calculate agreement ratio (how many signals agree with the final direction)
            if weighted_signal > 0:
                agreement = sum(1 for s in signal_values if s > 0) / len(signal_values) 
            elif weighted_signal < 0:
                agreement = sum(1 for s in signal_values if s < 0) / len(signal_values)
            else:
                agreement = sum(1 for s in signal_values if abs(s) < 0.2) / len(signal_values)
                
            # Confidence combines signal strength and agreement
            confidence = abs(weighted_signal) * 0.7 + agreement * 0.3
            
            # Adjust confidence based on market regime
            if market_regime in ['trending_bullish', 'trending_bearish']:
                # Higher confidence in clear trend regimes
                confidence *= 1.1
            elif 'volatile' in market_regime:
                # Lower confidence in volatile regimes
                confidence *= 0.85
                
            # Cap confidence at 1.0
            confidence = min(confidence, 1.0)
            
            # Store signals
            self.signals = {
                'individual': signals,
                'weights': weights,
                'weighted': weighted_signal,
                'final': final_signal,
                'confidence': confidence,
                'market_regime': market_regime,
                'timestamp': pd.Timestamp.now().isoformat()
            }
            
            self._add_to_cache(cache_key, self.signals)
            return self.signals
            
        except Exception as e:
            logging.error(f"Error generating signals: {e}")
            logging.error(traceback.format_exc())
            return {
                'individual': {},
                'weights': {},
                'weighted': 0,
                'final': 0,
                'confidence': 0,
                'error': str(e)
            }
    
    def get_complete_analysis(self, symbol: str = None) -> Dict[str, Any]:
        """
        Get complete market analysis and signals with performance optimization.
        
        Args:
            symbol: Optional symbol name for the analysis
            
        Returns:
            Dict with all analysis results
        """
        if self.df.empty:
            return {"error": "No data available for analysis"}
            
        try:
            start_time = time.time()
            
            # Use parallel execution for heavy calculations
            if self.parallel and self.executor:
                # Submit tasks to thread pool
                indicators_future = self.executor.submit(self._calculate_all_indicators)
                structure_future = self.executor.submit(self.analyze_market_structure)
                signals_future = self.executor.submit(self.generate_signals)
                support_resistance_future = self.executor.submit(self.detect_support_resistance)
                
                # Get results
                indicators = indicators_future.result()
                market_structure = structure_future.result()
                signals = signals_future.result()
                support_resistance = support_resistance_future.result()
            else:
                # Sequential execution
                indicators = self._calculate_all_indicators()
                market_structure = self.analyze_market_structure()
                signals = self.generate_signals()
                support_resistance = self.detect_support_resistance()
            
            # Combine all analysis
            analysis_summary = {
                "symbol": symbol or "Unknown",
                "last_price": self.df['close'].iloc[-1] if not self.df.empty else None,
                "indicators": indicators,
                "market_structure": market_structure,
                "signals": signals,
                "support_resistance": support_resistance,
                "timestamp": pd.Timestamp.now().isoformat(),
                "execution_time": time.time() - start_time,
                "cache_stats": {
                    "hits": self._cache_hits,
                    "misses": self._cache_misses,
                    "hit_ratio": self._cache_hits / (self._cache_hits + self._cache_misses) if (self._cache_hits + self._cache_misses) > 0 else 0
                }
            }
            
            return analysis_summary
            
        except Exception as e:
            logging.error(f"Error in market analysis: {e}")
            logging.error(traceback.format_exc())
            return {
                "error": f"Analysis failed: {str(e)}",
                "stack_trace": traceback.format_exc()
            }
    
    def _calculate_all_indicators(self) -> Dict[str, Any]:
        """Calculate all basic indicators for the summary"""
        indicators = {}
        
        # Calculate key indicators
        rsi = self.calculate_rsi()
        macd_data = self.calculate_macd()
        bb_data = self.calculate_bollinger_bands()
        stoch_data = self.calculate_stochastic()
        atr = self.calculate_atr()
        
        # Extract latest values where available
        if not rsi.empty:
            indicators['rsi'] = rsi.iloc[-1]
            
        if not macd_data.empty:
            indicators['macd'] = {
                'macd': macd_data['macd'].iloc[-1],
                'signal': macd_data['signal'].iloc[-1],
                'histogram': macd_data['histogram'].iloc[-1]
            }
            
        if not bb_data.empty:
            indicators['bollinger_bands'] = {
                'upper': bb_data['upper'].iloc[-1],
                'middle': bb_data['middle'].iloc[-1],
                'lower': bb_data['lower'].iloc[-1],
                'bandwidth': bb_data['bandwidth'].iloc[-1],
                'percent_b': bb_data['percent_b'].iloc[-1]
            }
            
        if not stoch_data.empty:
            indicators['stochastic'] = {
                'k': stoch_data['k'].iloc[-1],
                'd': stoch_data['d'].iloc[-1]
            }
            
        if not atr.empty:
            indicators['atr'] = atr.iloc[-1]
            
        return indicators
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics for the indicator calculations"""
        return {
            'calculation_times': self.calculation_times,
            'cache_stats': {
                'hits': self._cache_hits,
                'misses': self._cache_misses,
                'hit_ratio': self._cache_hits / (self._cache_hits + self._cache_misses) if (self._cache_hits + self._cache_misses) > 0 else 0
            }
        }

# Apply error handling decorator if available
if HAVE_ERROR_HANDLING:
    safe_indicator_analysis = safe_execute(ErrorCategory.DATA_PROCESSING, default_return={})(IndicatorAnalysis.get_complete_analysis)
    IndicatorAnalysis.get_complete_analysis = safe_indicator_analysis
