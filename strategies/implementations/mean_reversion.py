# strategies/implementations/mean_reversion.py

import numpy as np
import pandas as pd
import logging
import time
from typing import Dict, List, Tuple, Optional, Any, Union
from scipy import stats

class MeanReversionStrategy:
    """
    Advanced mean reversion strategy with adaptive Bollinger Bands and statistical validation.
    
    Features:
    - Adaptive Bollinger Bands with dynamic standard deviation multiplier
    - Statistical confirmation of mean reversion tendency
    - RSI filtering for enhanced signal quality
    - Dynamic entry/exit zones based on price distribution
    - Advanced risk management with volatility-adjusted stops
    - Proprietary overbought/oversold confirmation
    - Volume-based signal enhancement
    """
    
    def __init__(self) -> None:
        """Initialize the strategy with default parameters."""
        # Default strategy parameters - will be overridden by optimization
        self.default_params = {
            # Bollinger Band parameters
            'window': 20,
            'std_dev': 2.0,
            'adaptive_bands': True,
            'min_std_dev': 1.5,
            'max_std_dev': 3.0,
            
            # RSI parameters
            'rsi_period': 14,
            'rsi_overbought': 70,
            'rsi_oversold': 30,
            
            # Volatility parameters
            'atr_period': 14,
            'volatility_factor': 1.0,
            
            # Entry/exit parameters
            'entry_boundary': 0.05,  # % inside the bands for entry
            'exit_level': 0.5,       # % reversion to mean for exit
            'min_mean_cross': False, # Wait for mean crossing for exit
            
            # Statistical parameters
            'stat_window': 100,      # Window for statistical tests
            'confidence_level': 0.95, # Confidence for mean reversion tests
            
            # Risk management
            'stop_loss_atr_multiplier': 1.5,
            'take_profit_atr_multiplier': 2.0,
            'trailing_exit': True,
            
            # Volume confirmation
            'volume_filter': True,
            'volume_std_dev': 1.5,
            
            # Position sizing
            'position_size_pct': 0.02,
            'max_position_size_pct': 0.05,
            'size_factor': 1.0
        }
        
        # Strategy state variables
        self.params = self.default_params.copy()
        self.is_initialized = False
        self.current_position = 0
        self.entry_price = None
        self.stop_loss = None
        self.take_profit = None
        self.entry_time = None
        self.signals_history = []
        
        # Statistical testing
        self.is_mean_reverting = False
        self.mean_reversion_score = 0.0
        
        # Performance tracking
        self.trades = []
        self.performance_metrics = {
            'win_rate': 0.0,
            'avg_win': 0.0,
            'avg_loss': 0.0,
            'profit_factor': 0.0,
            'largest_win': 0.0,
            'largest_loss': 0.0,
            'avg_bars_held': 0
        }
        
        # Cache for expensive calculations
        self._calculation_cache = {}
        self._last_calc_time = {}
        
        # Logging setup
        self.logger = logging.getLogger(__name__)
        self.logger.info("Mean Reversion Strategy initialized")
        
    def set_parameters(self, params: Dict[str, Any]) -> None:
        """
        Update strategy parameters.
        
        Args:
            params: Dictionary of parameters to update
        """
        # Update only provided parameters
        for key, value in params.items():
            if key in self.default_params:
                self.params[key] = value
            else:
                self.logger.warning(f"Unknown parameter: {key}")
                
        # Clear calculation cache when parameters change
        self._calculation_cache = {}
        self._last_calc_time = {}
        
        self.logger.info(f"Strategy parameters updated: {params}")
        
    def initialize(self, historical_data: pd.DataFrame) -> None:
        """
        Initialize strategy with historical data for indicator calculation.
        
        Args:
            historical_data: DataFrame with market data
        """
        if historical_data is None or historical_data.empty:
            self.logger.error("Cannot initialize strategy with empty data")
            return
            
        try:
            # Ensure we have enough data for statistical tests
            required_length = max(
                self.params['stat_window'],
                self.params['window'] + 50
            )
            
            if len(historical_data) < required_length:
                self.logger.warning(
                    f"Historical data may be insufficient for reliable signals. "
                    f"Got {len(historical_data)} bars, recommended {required_length}"
                )
            
            # Calculate indicators for the historical data
            self._calculate_indicators(historical_data)
            
            # Perform mean reversion statistical tests
            self._test_mean_reversion(historical_data)
            
            self.is_initialized = True
            self.logger.info(f"Strategy successfully initialized with {len(historical_data)} bars of historical data")
            self.logger.info(f"Mean reversion test result: {self.is_mean_reverting} (score: {self.mean_reversion_score:.2f})")
            
        except Exception as e:
            self.logger.error(f"Error initializing strategy: {e}")
            raise
            
    def _test_mean_reversion(self, data: pd.DataFrame) -> None:
        """
        Test if the price series exhibits mean reversion characteristics.
        
        Args:
            data: DataFrame with market data (OHLCV)
        """
        # Use the most recent data for the test
        test_window = min(self.params['stat_window'], len(data))
        price_series = data['close'].iloc[-test_window:].values
        
        try:
            # Method 1: Augmented Dickey-Fuller test for stationarity
            from statsmodels.tsa.stattools import adfuller
            adf_result = adfuller(price_series)
            adf_pvalue = adf_result[1]
            
            # Method 2: Hurst Exponent for mean reversion
            # (H < 0.5 indicates mean reversion, H > 0.5 indicates trending)
            hurst_exp = self._calculate_hurst_exponent(price_series)
            
            # Method 3: Variance ratio test
            # (ratio < 1 suggests mean reversion)
            var_ratio = self._calculate_variance_ratio(price_series)
            
            # Method 4: Autocorrelation (negative lag-1 autocorrelation suggests mean reversion)
            autocorr = np.corrcoef(price_series[:-1], price_series[1:])[0, 1]
            
            # Combine test results into a single score
            # ADF p-value: lower is better for mean reversion
            # Hurst: lower is better for mean reversion (< 0.5)
            # Variance ratio: lower is better for mean reversion (< 1.0)
            # Autocorrelation: lower (negative) is better for mean reversion
            
            # Normalize and combine scores
            adf_score = 1.0 - min(adf_pvalue, 0.99)  # Higher score for lower p-value
            hurst_score = 1.0 - min(hurst_exp, 0.99)  # Higher score for lower Hurst exponent
            var_ratio_score = 1.0 - min(var_ratio, 0.99) if var_ratio < 1.5 else 0.0
            autocorr_score = 0.5 - min(autocorr, 0.5)  # Higher score for negative autocorrelation
            
            # Weighted combination (customize weights based on empirical performance)
            self.mean_reversion_score = (
                0.4 * adf_score +
                0.3 * hurst_score +
                0.2 * var_ratio_score +
                0.1 * autocorr_score
            )
            
            # Determine if the series is mean-reverting based on score threshold
            self.is_mean_reverting = self.mean_reversion_score > 0.6
            
            # Log detailed results
            self.logger.info(f"Mean reversion tests: ADF p-value={adf_pvalue:.4f}, Hurst={hurst_exp:.4f}, "
                           f"Variance ratio={var_ratio:.4f}, Autocorr={autocorr:.4f}")
            self.logger.info(f"Mean reversion score: {self.mean_reversion_score:.4f} (is_mean_reverting: {self.is_mean_reverting})")
            
        except Exception as e:
            self.logger.error(f"Error in mean reversion test: {e}")
            # Default to true if testing fails
            self.is_mean_reverting = True
            self.mean_reversion_score = 0.6
            
    def _calculate_hurst_exponent(self, price_series: np.ndarray) -> float:
        """
        Calculate the Hurst exponent for a time series.
        H < 0.5: mean-reverting
        H = 0.5: random walk
        H > 0.5: trending
        
        Args:
            price_series: numpy array of prices
            
        Returns:
            Hurst exponent
        """
        # Convert to numpy array
        ts = np.array(price_series)
        
        # Create the range of lag values
        lags = range(2, min(50, len(ts) // 4))
        
        # Calculate the array of the variances of the lagged differences
        tau = [np.sqrt(np.std(np.subtract(ts[lag:], ts[:-lag]))) for lag in lags]
        
        # Use a log-log regression to estimate the Hurst Exponent
        poly = np.polyfit(np.log(lags), np.log(tau), 1)
        
        # Return the Hurst exponent
        return poly[0]
        
    def _calculate_variance_ratio(self, price_series: np.ndarray, lag: int = 5) -> float:
        """
        Calculate the variance ratio for a time series.
        Ratio < 1 suggests mean reversion.
        
        Args:
            price_series: numpy array of prices
            lag: lag period for variance ratio
            
        Returns:
            Variance ratio
        """
        # Calculate returns
        returns = np.diff(np.log(price_series))
        
        # Calculate variances
        var_short = np.var(returns)
        var_long = np.var(np.sum(returns[range(0, len(returns), lag)], axis=0)) / lag
        
        # Return the variance ratio
        return var_long / var_short if var_short > 0 else 1.0
        
    def _calculate_indicators(self, data: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        Calculate all required technical indicators.
        
        Args:
            data: DataFrame with market data (OHLCV)
            
        Returns:
            Dictionary with calculated indicators
        """
        # Start with an empty indicators dictionary
        indicators = {}
        
        # Get parameters
        window = self.params['window']
        std_dev = self.params['std_dev']
        rsi_period = self.params['rsi_period']
        atr_period = self.params['atr_period']
        
        # Create cache key
        cache_key = f"{window}_{std_dev}_{rsi_period}_{atr_period}_{len(data)}"
        
        # Check if we have cached calculations
        if cache_key in self._calculation_cache:
            # Check if cache is recent (within 1 minute)
            last_calc = self._last_calc_time.get(cache_key, 0)
            if time.time() - last_calc < 60:
                return self._calculation_cache[cache_key]
        
        try:
            # Calculate moving average
            indicators['sma'] = data['close'].rolling(window=window).mean()
            indicators['ema'] = data['close'].ewm(span=window, adjust=False).mean()
            
            # Calculate standard deviation
            indicators['std'] = data['close'].rolling(window=window).std()
            
            # Calculate Bollinger Bands with adaptive or fixed std dev
            if self.params['adaptive_bands']:
                # Use linear regression of volatility to predict optimal std dev multiplier
                volatility_ratio = indicators['std'] / indicators['std'].rolling(window=100).mean()
                
                # Scale std_dev between min and max based on recent volatility
                adaptive_multiplier = np.clip(
                    std_dev * volatility_ratio.fillna(1.0),
                    self.params['min_std_dev'],
                    self.params['max_std_dev']
                )
                
                # Calculate adaptive bands
                indicators['upper_band'] = indicators['sma'] + (indicators['std'] * adaptive_multiplier)
                indicators['lower_band'] = indicators['sma'] - (indicators['std'] * adaptive_multiplier)
                indicators['std_dev_multiplier'] = adaptive_multiplier
            else:
                # Fixed standard deviation multiplier
                indicators['upper_band'] = indicators['sma'] + (indicators['std'] * std_dev)
                indicators['lower_band'] = indicators['sma'] - (indicators['std'] * std_dev)
                indicators['std_dev_multiplier'] = pd.Series([std_dev] * len(data), index=data.index)
            
            # Calculate RSI
            delta = data['close'].diff()
            gain = delta.where(delta > 0, 0).fillna(0)
            loss = -delta.where(delta < 0, 0).fillna(0)
            
            avg_gain = gain.rolling(window=rsi_period).mean()
            avg_loss = loss.rolling(window=rsi_period).mean()
            
            rs = avg_gain / avg_loss.replace(0, np.finfo(float).eps)
            indicators['rsi'] = 100 - (100 / (1 + rs))
            
            # Calculate ATR for volatility
            high = data['high']
            low = data['low']
            close = data['close']
            
            tr1 = abs(high - low)
            tr2 = abs(high - close.shift())
            tr3 = abs(low - close.shift())
            
            tr = pd.DataFrame({'tr1': tr1, 'tr2': tr2, 'tr3': tr3}).max(axis=1)
            indicators['atr'] = tr.rolling(window=atr_period).mean()
            
            # Calculate percentage distance from bands
            price = data['close']
            upper = indicators['upper_band']
            lower = indicators['lower_band']
            middle = indicators['sma']
            
            # The percentage distance from the price to the upper band (0 at upper band, 1 at middle)
            indicators['upper_band_distance'] = (upper - price) / (upper - middle)
            
            # The percentage distance from the price to the lower band (0 at lower band, 1 at middle)
            indicators['lower_band_distance'] = (price - lower) / (middle - lower)
            
            # Calculate z-score (how many standard deviations from the mean)
            indicators['z_score'] = (price - middle) / indicators['std']
            
            # Calculate price rate of change
            indicators['roc'] = price.pct_change(periods=window // 2) * 100
            
            # Calculate volume indicators if volume data is available
            if 'volume' in data.columns:
                # Volume moving average
                indicators['volume_sma'] = data['volume'].rolling(window=window).mean()
                
                # Volume standard deviation for identifying unusual volume
                vol_std = data['volume'].rolling(window=window).std()
                indicators['volume_z_score'] = (data['volume'] - indicators['volume_sma']) / vol_std
                
                # On-Balance Volume (OBV)
                obv = pd.Series(0, index=data.index)
                for i in range(1, len(data)):
                    if data['close'].iloc[i] > data['close'].iloc[i-1]:
                        obv.iloc[i] = obv.iloc[i-1] + data['volume'].iloc[i]
                    elif data['close'].iloc[i] < data['close'].iloc[i-1]:
                        obv.iloc[i] = obv.iloc[i-1] - data['volume'].iloc[i]
                    else:
                        obv.iloc[i] = obv.iloc[i-1]
                
                indicators['obv'] = obv
                
                # Chaikin Money Flow (CMF)
                money_flow_multiplier = ((data['close'] - data['low']) - (data['high'] - data['close'])) / (data['high'] - data['low'])
                money_flow_volume = money_flow_multiplier * data['volume']
                indicators['cmf'] = money_flow_volume.rolling(window=window).sum() / data['volume'].rolling(window=window).sum()
            
            # Store in cache
            self._calculation_cache[cache_key] = indicators
            self._last_calc_time[cache_key] = time.time()
            
            return indicators
            
        except Exception as e:
            self.logger.error(f"Error calculating indicators: {e}")
            raise
            
    def generate_signal(self, current_data: pd.DataFrame) -> Dict[str, Any]:
        """
        Generate trading signals based on current market data.
        
        Args:
            current_data: DataFrame with recent market data
            
        Returns:
            Dictionary with signal details
        """
        if not self.is_initialized or current_data is None or current_data.empty:
            return {'signal': 0, 'strength': 0, 'metadata': {}}
            
        try:
            # Calculate indicators
            indicators = self._calculate_indicators(current_data)
            
            # Get latest values
            latest_idx = current_data.index[-1]
            price = current_data['close'].iloc[-1]
            
            sma = indicators['sma'].iloc[-1]
            ema = indicators['ema'].iloc[-1]
            upper_band = indicators['upper_band'].iloc[-1]
            lower_band = indicators['lower_band'].iloc[-1]
            std = indicators['std'].iloc[-1]
            rsi = indicators['rsi'].iloc[-1]
            atr = indicators['atr'].iloc[-1]
            z_score = indicators['z_score'].iloc[-1]
            std_dev_multiplier = indicators['std_dev_multiplier'].iloc[-1]
            
            # Check for volume confirmation if enabled and available
            volume_confirmed = True
            if self.params['volume_filter'] and 'volume' in current_data.columns:
                volume_z_score = indicators['volume_z_score'].iloc[-1]
                cmf = indicators.get('cmf', pd.Series([0])).iloc[-1]
                
                # For mean reversion, we want to see high volume at extremes
                if price < lower_band:
                    # For buying at lower band, we want high volume and positive money flow
                    volume_confirmed = (
                        volume_z_score > self.params['volume_std_dev'] or
                        cmf > 0
                    )
                elif price > upper_band:
                    # For selling at upper band, we want high volume and negative money flow
                    volume_confirmed = (
                        volume_z_score > self.params['volume_std_dev'] or
                        cmf < 0
                    )
            
            # Initialize signal variables
            signal = 0
            signal_strength = 0
            
            # Define entry zones with some buffer inside the bands
            upper_entry = upper_band * (1 - self.params['entry_boundary'])
            lower_entry = lower_band * (1 + self.params['entry_boundary'])
            
            # Initialize conditions dict for metadata
            conditions = {
                "price": price,
                "sma": sma,
                "upper_band": upper_band,
                "lower_band": lower_band,
                "rsi": rsi,
                "z_score": z_score,
                "volume_confirmed": volume_confirmed,
                "is_mean_reverting": self.is_mean_reverting,
                "mean_reversion_score": self.mean_reversion_score
            }
            
            # Mean reversion logic: we're most interested when price touches or exceeds the bands
            # The core concept is to fade extreme moves that are likely to revert to the mean
            
            # Only generate signals if the asset is showing mean reversion characteristics
            if self.is_mean_reverting:
                # Signal for entering at the lower band (buy)
                if price <= lower_entry:
                    # Additional confirmation with RSI (oversold)
                    if rsi <= self.params['rsi_oversold'] and volume_confirmed:
                        signal = 1  # Buy signal
                        
                        # Signal strength increases with distance from band and oversold conditions
                        z_score_factor = min(abs(z_score) / 3, 1.0)  # Normalize z-score to 0-1 scale
                        rsi_factor = (self.params['rsi_oversold'] - rsi) / self.params['rsi_oversold']
                        
                        signal_strength = 0.5 + (0.25 * z_score_factor) + (0.25 * rsi_factor)
                        signal_strength = min(signal_strength, 1.0)  # Cap at 1.0
                        
                        conditions["lower_band_trigger"] = True
                        conditions["rsi_oversold"] = True
                        
                        self.logger.info(f"BUY signal: price={price:.2f}, lower_band={lower_band:.2f}, "
                                       f"rsi={rsi:.2f}, strength={signal_strength:.2f}")
                
                # Signal for entering at the upper band (sell)
                elif price >= upper_entry:
                    # Additional confirmation with RSI (overbought)
                    if rsi >= self.params['rsi_overbought'] and volume_confirmed:
                        signal = -1  # Sell signal
                        
                        # Signal strength increases with distance from band and overbought conditions
                        z_score_factor = min(abs(z_score) / 3, 1.0)  # Normalize z-score to 0-1 scale
                        rsi_factor = (rsi - self.params['rsi_overbought']) / (100 - self.params['rsi_overbought'])
                        
                        signal_strength = 0.5 + (0.25 * z_score_factor) + (0.25 * rsi_factor)
                        signal_strength = min(signal_strength, 1.0)  # Cap at 1.0
                        
                        conditions["upper_band_trigger"] = True
                        conditions["rsi_overbought"] = True
                        
                        self.logger.info(f"SELL signal: price={price:.2f}, upper_band={upper_band:.2f}, "
                                       f"rsi={rsi:.2f}, strength={signal_strength:.2f}")
            
            # Exit signal logic for existing positions
            if self.current_position != 0:
                # Exit conditions will depend on the current position
                exit_signal = False
                
                if self.current_position > 0:  # Long position
                    # Exit when price crosses the mean (SMA) if min_mean_cross is True
                    if self.params['min_mean_cross'] and price >= sma:
                        exit_signal = True
                        conditions["mean_cross_exit"] = True
                    # Otherwise, exit when price retraces a percentage toward the mean
                    elif not self.params['min_mean_cross']:
                        # Calculate percentage retracement from lower band to mean
                        retracement = (price - lower_band) / (sma - lower_band) if (sma - lower_band) > 0 else 0
                        if retracement >= self.params['exit_level']:
                            exit_signal = True
                            conditions["retracement_exit"] = True
                    
                else:  # Short position
                    # Exit when price crosses the mean (SMA) if min_mean_cross is True
                    if self.params['min_mean_cross'] and price <= sma:
                        exit_signal = True
                        conditions["mean_cross_exit"] = True
                    # Otherwise, exit when price retraces a percentage toward the mean
                    elif not self.params['min_mean_cross']:
                        # Calculate percentage retracement from upper band to mean
                        retracement = (upper_band - price) / (upper_band - sma) if (upper_band - sma) > 0 else 0
                        if retracement >= self.params['exit_level']:
                            exit_signal = True
                            conditions["retracement_exit"] = True
                
                # If an exit signal is triggered, set opposite of current position
                if exit_signal:
                    signal = -self.current_position
                    signal_strength = 0.8  # Strong exit signal
                    self.logger.info(f"EXIT signal for {self.current_position} position: price={price:.2f}, sma={sma:.2f}")
            
            # Set stop loss and take profit levels if entering a new position
            if signal != 0 and self.current_position == 0:
                self.entry_price = price
                self.entry_time = latest_idx
                
                # Calculate ATR-based stop loss and take profit
                stop_loss_distance = atr * self.params['stop_loss_atr_multiplier']
                take_profit_distance = atr * self.params['take_profit_atr_multiplier']
                
                if signal > 0:  # Long position
                    # For long positions, stop loss below entry, take profit above
                    self.stop_loss = price - stop_loss_distance
                    self.take_profit = price + take_profit_distance
                else:  # Short position
                    # For short positions, stop loss above entry, take profit below
                    self.stop_loss = price + stop_loss_distance
                    self.take_profit = price - take_profit_distance
            
            # Store signal in history
            self.signals_history.append({
                'time': latest_idx,
                'price': price,
                'signal': signal,
                'strength': signal_strength,
                'z_score': z_score,
                'rsi': rsi
            })
            
            # Trim signals history to last 100 entries
            if len(self.signals_history) > 100:
                self.signals_history = self.signals_history[-100:]
            
            # Calculate position size based on signal strength, z-score, and volatility
            position_size = self._calculate_position_size(price, signal_strength, z_score, std_dev_multiplier)
            
            # Create signal dictionary
            signal_dict = {
                'signal': signal,
                'strength': signal_strength,
                'position_size': position_size,
                'stop_loss': self.stop_loss,
                'take_profit': self.take_profit,
                'metadata': {
                    'strategy': 'mean_reversion',
                    'price': price,
                    'sma': sma,
                    'ema': ema,
                    'upper_band': upper_band,
                    'lower_band': lower_band,
                    'rsi': rsi,
                    'atr': atr,
                    'z_score': z_score,
                    'std_dev_multiplier': std_dev_multiplier,
                    'is_mean_reverting': self.is_mean_reverting,
                    'mean_reversion_score': self.mean_reversion_score,
                    'conditions': conditions
                }
            }
            
            return signal_dict
            
        except Exception as e:
            self.logger.error(f"Error generating signal: {e}")
            # Return no signal in case of error
            return {'signal': 0, 'strength': 0, 'metadata': {'error': str(e)}}
            
    def _calculate_position_size(self, price: float, signal_strength: float, z_score: float, std_dev_multiplier: float) -> float:
        """
        Calculate position size based on signal strength, z-score, and volatility.
        
        Args:
            price: Current price
            signal_strength: Signal strength (0-1)
            z_score: Current z-score (standard deviations from mean)
            std_dev_multiplier: Current standard deviation multiplier
            
        Returns:
            Position size as percentage of account
        """
        # Start with base position size
        position_size = self.params['position_size_pct']
        
        # Scale by signal strength
        position_size *= signal_strength
        
        # Scale by absolute z-score (stronger signals at extremes)
        # More extreme z-scores warrant larger positions
        z_score_abs = abs(z_score)
        z_score_factor = min(z_score_abs / 3.0, 1.0)  # Cap at 1.0
        position_size *= (1.0 + z_score_factor * 0.5)  # Up to 50% increase based on z-score
        
        # Adjust for volatility - lower position size in higher volatility
        # std_dev_multiplier is higher in volatile markets
        vol_adjustment = 1.0 / std_dev_multiplier
        position_size *= vol_adjustment
        
        # Apply size factor from parameters
        position_size *= self.params['size_factor']
        
        # Cap at maximum position size
        position_size = min(position_size, self.params['max_position_size_pct'])
        
        return position_size
        
    def update_position(self, position: int, price: float, time_index: Any) -> None:
        """
        Update current position based on trade execution.
        
        Args:
            position: New position (-1, 0, 1)
            price: Execution price
            time_index: Timestamp of execution
        """
        # If position is changing, log the trade
        if position != self.current_position:
            # If we had a position before, record the trade
            if self.current_position != 0:
                trade = {
                    'entry_time': self.entry_time,
                    'entry_price': self.entry_price,
                    'exit_time': time_index,
                    'exit_price': price,
                    'position': self.current_position,
                    'pnl': self.current_position * (price - self.entry_price) / self.entry_price,
                    'bars_held': (time_index - self.entry_time).total_seconds() / 60 if hasattr(time_index, 'total_seconds') else 0
                }
                self.trades.append(trade)
                self.logger.info(f"Trade closed: {trade}")
                
                # Update performance metrics
                self._update_performance_metrics()
            
            # If entering a new position, record entry details
            if position != 0:
                self.entry_price = price
                self.entry_time = time_index
                self.logger.info(f"New position: {position} at price {price}")
            
        # Update current position
        self.current_position = position
        
    def check_stop_loss(self, current_price: float) -> bool:
        """
        Check if stop loss has been hit.
        
        Args:
            current_price: Latest market price
            
        Returns:
            True if stop loss is hit, False otherwise
        """
        if self.current_position == 0 or self.stop_loss is None:
            return False
            
        if (self.current_position > 0 and current_price < self.stop_loss) or \
           (self.current_position < 0 and current_price > self.stop_loss):
            self.logger.info(f"Stop loss hit: {self.stop_loss}")
            return True
            
        return False
        
    def check_take_profit(self, current_price: float) -> bool:
        """
        Check if take profit has been hit.
        
        Args:
            current_price: Latest market price
            
        Returns:
            True if take profit is hit, False otherwise
        """
        if self.current_position == 0 or self.take_profit is None:
            return False
            
        if (self.current_position > 0 and current_price >= self.take_profit) or \
           (self.current_position < 0 and current_price <= self.take_profit):
            self.logger.info(f"Take profit hit: {self.take_profit}")
            return True
            
        return False
        
    def _update_performance_metrics(self) -> None:
        """Update strategy performance metrics after a completed trade."""
        if not self.trades:
            return
            
        # Calculate performance metrics
        wins = [trade for trade in self.trades if trade['pnl'] > 0]
        losses = [trade for trade in self.trades if trade['pnl'] <= 0]
        
        self.performance_metrics['win_rate'] = len(wins) / len(self.trades) if self.trades else 0
        
        self.performance_metrics['avg_win'] = sum(trade['pnl'] for trade in wins) / len(wins) if wins else 0
        self.performance_metrics['avg_loss'] = sum(trade['pnl'] for trade in losses) / len(losses) if losses else 0
        
        total_wins = sum(trade['pnl'] for trade in wins)
        total_losses = abs(sum(trade['pnl'] for trade in losses))
        self.performance_metrics['profit_factor'] = total_wins / total_losses if total_losses > 0 else float('inf')
        
        self.performance_metrics['largest_win'] = max([trade['pnl'] for trade in wins]) if wins else 0
        self.performance_metrics['largest_loss'] = min([trade['pnl'] for trade in losses]) if losses else 0
        
        self.performance_metrics['avg_bars_held'] = sum(trade['bars_held'] for trade in self.trades) / len(self.trades)
        
        self.logger.info(f"Updated performance metrics: {self.performance_metrics}")
        
    def get_performance_metrics(self) -> Dict[str, float]:
        """
        Get current performance metrics.
        
        Returns:
            Dictionary with performance metrics
        """
        return self.performance_metrics
        
    def reset(self) -> None:
        """Reset the strategy to initial state."""
        self.current_position = 0
        self.entry_price = None
        self.stop_loss = None
        self.take_profit = None
        self.entry_time = None
        self.signals_history = []
        
        # Clear calculation cache
        self._calculation_cache = {}
        self._last_calc_time = {}
        
        self.logger.info("Strategy reset to initial state")

# Factory function to create the strategy
def create_strategy() -> MeanReversionStrategy:
    """Create a new instance of the MeanReversionStrategy."""
    return MeanReversionStrategy()
