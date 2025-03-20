# Strategy_Generator.py

import logging
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Union, Callable, Tuple, Set
from enum import Enum
import traceback
from datetime import datetime, timedelta

# Import error handling if available
try:
    from error_handling import safe_execute, ErrorCategory, ErrorSeverity, TradingSystemError
    HAVE_ERROR_HANDLING = True
except ImportError:
    HAVE_ERROR_HANDLING = False
    logging.warning("Error handling module not available. Using basic error handling in StrategyGenerator.")

# Configure module logger
logger = logging.getLogger(__name__)

class StrategyType(Enum):
    """Enumeration of supported strategy types for type safety and validation."""
    TREND_FOLLOWING = "trend_following"
    MEAN_REVERSION = "mean_reversion"
    BREAKOUT = "breakout"
    MULTI_TIMEFRAME = "multi_timeframe"
    VOLUME_BASED = "volume_based"
    ML_BASED = "ml_based"
    CUSTOM = "custom"


class SignalDirection(Enum):
    """Enumeration of possible trading signal directions."""
    BUY = 1
    SELL = -1
    HOLD = 0


class StrategyGenerator:
    """
    Factory class for generating and configuring trading strategies.
    
    This class implements various algorithmic trading strategies including trend following,
    mean reversion, breakout patterns, and more. Each strategy is implemented as a method
    that processes market data and returns trading signals.
    
    Attributes:
        strategies (Dict[str, Callable]): Mapping of strategy names to strategy functions
        strategy_params (Dict[str, Dict[str, Any]]): Default parameters for each strategy
        optimized_params (Dict[str, Dict[str, Any]]): Optimized parameters from backtesting
    """

    def __init__(self) -> None:
        """
        Initialize the StrategyGenerator with predefined strategies and default parameters.
        
        Sets up the mapping between strategy names and their implementation functions,
        and initializes storage for strategy parameters.
        """
        # Register available strategies
        self.strategies: Dict[str, Callable] = {
            'trend_following': self.trend_following_strategy,
            'mean_reversion': self.mean_reversion_strategy,
            'breakout': self.breakout_strategy,
            'multi_timeframe': self.multi_timeframe_strategy,
            'volume_based': self.volume_based_strategy,
            'ml_based': self.ml_based_strategy
        }
        
        # Initialize parameter storage
        self.strategy_params: Dict[str, Dict[str, Any]] = {}
        self.optimized_params: Dict[str, Dict[str, Any]] = {}
        
        # Initialize default parameters for each strategy
        self._initialize_default_parameters()
        
        logger.info(f"StrategyGenerator initialized with {len(self.strategies)} strategies")

    def _initialize_default_parameters(self) -> None:
        """
        Initialize default parameters for all strategies.
        
        Sets up sensible default values for each strategy's parameters, which can
        later be overridden by optimization or user configuration.
        """
        # Trend Following Strategy Parameters
        self.strategy_params['trend_following'] = {
            'short_window': 10,     # Short-term moving average period
            'medium_window': 50,    # Medium-term moving average period
            'long_window': 100,     # Long-term moving average period
            'signal_threshold': 0.0, # Signal threshold for crossovers
            'stop_loss_pct': 0.02,  # Stop loss percentage
            'take_profit_pct': 0.05, # Take profit percentage
            'trailing_stop_pct': 0.015, # Trailing stop percentage
            'use_atr_stops': True,  # Use ATR for stop calculation
            'atr_multiplier': 2.0,  # Multiplier for ATR stops
            'exit_after_bars': 20,  # Maximum holding period in bars
            'volatility_filter': True, # Apply volatility filter
            'volume_filter': True,  # Apply volume filter
            'minimum_bars': 3       # Minimum bars for confirmation
        }
        
        # Mean Reversion Strategy Parameters
        self.strategy_params['mean_reversion'] = {
            'window': 20,           # Period for calculating mean
            'std_dev': 2.0,         # Standard deviation multiplier for bands
            'entry_threshold': 2.0, # Entry threshold in standard deviations
            'exit_threshold': 0.5,  # Exit threshold in standard deviations
            'max_holding_period': 10, # Maximum holding period in bars
            'stop_loss_pct': 0.02,  # Stop loss percentage
            'take_profit_pct': 0.04, # Take profit percentage
            'mean_type': 'sma',     # Type of mean calculation (sma, ema, wma)
            'oversold_rsi': 30,     # RSI level considered oversold
            'overbought_rsi': 70,   # RSI level considered overbought
            'use_volume_filter': True, # Apply volume confirmation
            'min_volatility': 0.01, # Minimum volatility for entry
            'max_volatility': 0.05  # Maximum volatility for entry
        }
        
        # Breakout Strategy Parameters
        self.strategy_params['breakout'] = {
            'window': 50,           # Period for support/resistance
            'volume_window': 10,    # Period for volume moving average
            'volume_threshold': 1.5, # Volume increase threshold
            'entry_confirmation_bars': 2, # Bars needed for entry confirmation
            'stop_loss_pct': 0.03,  # Stop loss percentage
            'take_profit_pct': 0.06, # Take profit percentage
            'atr_periods': 14,      # Periods for ATR calculation
            'atr_multiplier': 2.5,  # Multiplier for ATR-based targets
            'use_dynamic_stops': True, # Use dynamic stop loss
            'false_breakout_filter': True, # Filter false breakouts
            'min_consolidation_periods': 5, # Minimum consolidation before breakout
            'max_spread_pct': 0.005 # Maximum bid-ask spread percentage
        }
        
        # Multi-Timeframe Strategy Parameters
        self.strategy_params['multi_timeframe'] = {
            'short_window': 20,     # Short-term timeframe MA period
            'long_window': 50,      # Long-term timeframe MA period
            'primary_timeframe': '1h', # Primary analysis timeframe
            'confirmation_timeframe': '4h', # Confirmation timeframe
            'trend_timeframe': '1d', # Trend determination timeframe
            'alignment_required': True, # Require timeframe alignment
            'min_trend_strength': 0.6, # Minimum trend strength (0-1)
            'stop_loss_pct': 0.025, # Stop loss percentage
            'take_profit_pct': 0.05, # Take profit percentage
            'use_multi_entry': True, # Use multiple entry conditions
            'use_volume_confirmation': True, # Use volume confirmation
            'exit_on_trend_shift': True, # Exit when trend shifts
            'early_exit_threshold': 0.3 # Early exit threshold
        }
        
        # Volume-Based Strategy Parameters
        self.strategy_params['volume_based'] = {
            'volume_window': 20,    # Period for volume moving average
            'volume_threshold': 1.8, # Volume spike threshold
            'price_window': 10,     # Period for price average
            'entry_delay': 1,       # Bars to wait after signal before entry
            'stop_loss_pct': 0.02,  # Stop loss percentage
            'take_profit_pct': 0.04, # Take profit percentage
            'use_obv': True,        # Use On-Balance Volume
            'use_vwap': True,       # Use Volume-Weighted Average Price
            'obv_smoothing': 3,     # OBV smoothing period
            'min_price_move': 0.005, # Minimum price move requirement
            'consolidation_threshold': 0.3, # Volume consolidation threshold
            'use_price_action_filter': True # Filter based on price action
        }
        
        # ML-Based Strategy Parameters
        self.strategy_params['ml_based'] = {
            'prediction_threshold': 0.65, # Prediction confidence threshold
            'ensemble_weighting': {  # Weights for different models
                'xgboost': 0.4,
                'lstm': 0.3,
                'random_forest': 0.3
            },
            'feature_importance_threshold': 0.05, # Minimum feature importance
            'retrain_interval': 7,   # Days between model retraining
            'stop_loss_pct': 0.025,  # Stop loss percentage
            'take_profit_pct': 0.05, # Take profit percentage
            'trailing_stop_pct': 0.02, # Trailing stop percentage
            'use_adaptive_thresholds': True, # Adjust thresholds based on volatility
            'min_samples_for_signal': 10, # Minimum data points needed
            'prediction_horizon': 3, # Future bars for prediction
            'use_market_regime_filter': True # Adjust strategy based on market regime
        }
        
    def get_strategy_parameters(self, strategy_name: str) -> Dict[str, Any]:
        """
        Get the current parameters for a specific strategy.
        
        Args:
            strategy_name (str): Name of the strategy to retrieve parameters for
            
        Returns:
            Dict[str, Any]: Dictionary of parameter names and values
            
        Raises:
            ValueError: If the specified strategy does not exist
        """
        if strategy_name not in self.strategies:
            error_msg = f"Strategy '{strategy_name}' not found. Available strategies: {list(self.strategies.keys())}"
            logger.error(error_msg)
            raise ValueError(error_msg)
            
        # Use optimized parameters if available, otherwise default parameters
        if strategy_name in self.optimized_params:
            return self.optimized_params[strategy_name].copy()
        
        # Ensure default parameters exist
        if strategy_name not in self.strategy_params:
            self._initialize_default_parameters()
            
        return self.strategy_params[strategy_name].copy()

    def set_strategy_parameters(self, strategy_name: str, parameters: Dict[str, Any]) -> None:
        """
        Set custom parameters for a specific strategy.
        
        Args:
            strategy_name (str): Name of the strategy to update parameters for
            parameters (Dict[str, Any]): Dictionary of parameter names and values to update
            
        Raises:
            ValueError: If the specified strategy does not exist
        """
        if strategy_name not in self.strategies:
            error_msg = f"Strategy '{strategy_name}' not found. Available strategies: {list(self.strategies.keys())}"
            logger.error(error_msg)
            raise ValueError(error_msg)
            
        # Ensure we have default parameters initialized
        if strategy_name not in self.strategy_params:
            self._initialize_default_parameters()
            
        # Validate parameters (optional but recommended)
        self._validate_strategy_parameters(strategy_name, parameters)
        
        # Update optimized parameters
        if strategy_name not in self.optimized_params:
            self.optimized_params[strategy_name] = self.strategy_params[strategy_name].copy()
            
        # Update parameters
        self.optimized_params[strategy_name].update(parameters)
        
        logger.info(f"Updated parameters for strategy '{strategy_name}'")

    def _validate_strategy_parameters(self, strategy_name: str, parameters: Dict[str, Any]) -> None:
        """
        Validate parameters for a specific strategy.
        
        Args:
            strategy_name (str): Name of the strategy to validate parameters for
            parameters (Dict[str, Any]): Dictionary of parameter names and values to validate
            
        Raises:
            ValueError: If any parameter is invalid
        """
        # Basic validation based on strategy type
        if strategy_name == 'trend_following':
            if 'short_window' in parameters and 'medium_window' in parameters:
                if parameters['short_window'] >= parameters['medium_window']:
                    raise ValueError("short_window must be less than medium_window")
                    
            if 'medium_window' in parameters and 'long_window' in parameters:
                if parameters['medium_window'] >= parameters['long_window']:
                    raise ValueError("medium_window must be less than long_window")
        
        elif strategy_name == 'mean_reversion':
            if 'std_dev' in parameters and parameters['std_dev'] <= 0:
                raise ValueError("std_dev must be positive")
                
            if 'window' in parameters and parameters['window'] < 2:
                raise ValueError("window must be at least 2")
                
        elif strategy_name == 'breakout':
            if 'window' in parameters and parameters['window'] < 5:
                raise ValueError("window must be at least 5 for reliable support/resistance levels")
                
        # Common parameter validations
        for param_name in ['stop_loss_pct', 'take_profit_pct', 'trailing_stop_pct']:
            if param_name in parameters:
                if parameters[param_name] <= 0:
                    raise ValueError(f"{param_name} must be positive")
                elif parameters[param_name] > 0.5:  # 50% would be extreme
                    raise ValueError(f"{param_name} must be less than 0.5 (50%)")

    def generate_strategy(self, strategy_name: str, data: pd.DataFrame, 
                          parameters: Optional[Dict[str, Any]] = None) -> pd.DataFrame:
        """
        Generate trading signals using the specified strategy.
        
        Args:
            strategy_name (str): Name of the strategy to use
            data (pd.DataFrame): Market data with OHLCV columns
            parameters (Optional[Dict[str, Any]]): Custom parameters for this execution
                                                 (overrides stored parameters)
            
        Returns:
            pd.DataFrame: DataFrame with original data and added signals
            
        Raises:
            ValueError: If the specified strategy does not exist
            RuntimeError: If signal generation fails
        """
        # Validate inputs
        if strategy_name not in self.strategies:
            error_msg = f"Strategy '{strategy_name}' not found. Available strategies: {list(self.strategies.keys())}"
            logger.error(error_msg)
            raise ValueError(error_msg)
            
        if data is None or data.empty:
            error_msg = "Cannot generate signals: input data is empty"
            logger.error(error_msg)
            raise ValueError(error_msg)
            
        # Ensure required columns exist
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        missing_columns = [col for col in required_columns if col not in data.columns]
        if missing_columns:
            error_msg = f"Data missing required columns: {missing_columns}"
            logger.error(error_msg)
            raise ValueError(error_msg)
            
        # Get parameters (use provided, or optimized, or default)
        if parameters is not None:
            # Use provided parameters with validation
            self._validate_strategy_parameters(strategy_name, parameters)
            params = self.get_strategy_parameters(strategy_name)
            params.update(parameters)
        else:
            # Use stored parameters
            params = self.get_strategy_parameters(strategy_name)
            
        logger.debug(f"Generating signals for strategy '{strategy_name}' with parameters: {params}")
        
        try:
            # Execute strategy function
            strategy_func = self.strategies[strategy_name]
            start_time = datetime.now()
            
            # Execute with error handling if available
            if HAVE_ERROR_HANDLING:
                result = safe_execute(ErrorCategory.STRATEGY, default_return=pd.DataFrame(index=data.index))(
                    strategy_func)(data, params)
            else:
                result = strategy_func(data, params)
                
            execution_time = (datetime.now() - start_time).total_seconds()
            logger.debug(f"Strategy '{strategy_name}' executed in {execution_time:.3f} seconds")
            
            # Ensure result is a DataFrame with the same index as input data
            if not isinstance(result, pd.DataFrame):
                error_msg = f"Strategy function returned {type(result)}, expected DataFrame"
                logger.error(error_msg)
                raise TypeError(error_msg)
                
            if not all(idx in result.index for idx in data.index):
                logger.warning("Strategy result missing some indices from input data")
                
            return result
            
        except Exception as e:
            error_msg = f"Error generating signals for strategy '{strategy_name}': {str(e)}"
            logger.error(f"{error_msg}\n{traceback.format_exc()}")
            raise RuntimeError(error_msg) from e

    def trend_following_strategy(self, data: pd.DataFrame, params: Dict[str, Any]) -> pd.DataFrame:
        """
        Implements an advanced trend following strategy using multiple moving averages.
        
        This strategy generates buy signals when shorter-term moving averages cross above
        longer-term moving averages, and sell signals when shorter-term averages cross below
        longer-term averages, with additional filters for volatility and volume.
        
        Args:
            data (pd.DataFrame): Market data with OHLCV columns
            params (Dict[str, Any]): Strategy parameters
            
        Returns:
            pd.DataFrame: DataFrame with original data and added signals column
        """
        # Create copy of data to avoid modifying the original
        df = data.copy()
        
        # Extract parameters
        short_window = params.get('short_window', 10)
        medium_window = params.get('medium_window', 50)
        long_window = params.get('long_window', 100)
        signal_threshold = params.get('signal_threshold', 0.0)
        use_atr_stops = params.get('use_atr_stops', True)
        atr_multiplier = params.get('atr_multiplier', 2.0)
        volatility_filter = params.get('volatility_filter', True)
        volume_filter = params.get('volume_filter', True)
        minimum_bars = params.get('minimum_bars', 3)
        
        # Initialize signals DataFrame
        signals = pd.DataFrame(index=df.index)
        signals['position'] = 0
        
        # Calculate moving averages
        df['short_ma'] = df['close'].ewm(span=short_window, adjust=False).mean()
        df['medium_ma'] = df['close'].ewm(span=medium_window, adjust=False).mean()
        df['long_ma'] = df['close'].ewm(span=long_window, adjust=False).mean()
        
        # Calculate additional indicators for filters
        if volatility_filter or use_atr_stops:
            # Calculate ATR
            high_low = df['high'] - df['low']
            high_close = (df['high'] - df['close'].shift()).abs()
            low_close = (df['low'] - df['close'].shift()).abs()
            tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
            df['atr'] = tr.rolling(window=14).mean()
            
            # Normalize ATR as percentage of price
            df['atr_pct'] = df['atr'] / df['close']
            
        if volume_filter:
            # Calculate volume moving average
            df['volume_ma'] = df['volume'].rolling(window=20).mean()
            df['volume_ratio'] = df['volume'] / df['volume_ma']
        
        # Generate signals with confirmation count
        long_signal_count = 0
        short_signal_count = 0
        
        for i in range(1, len(df)):
            # Skip if not enough data for moving averages
            if i <= long_window:
                continue
                
            # Base trend signal conditions
            bullish_alignment = (df['short_ma'].iloc[i] > df['medium_ma'].iloc[i] > df['long_ma'].iloc[i])
            bearish_alignment = (df['short_ma'].iloc[i] < df['medium_ma'].iloc[i] < df['long_ma'].iloc[i])
            
            # Short-term crossover signals (faster response)
            bullish_crossover = (df['short_ma'].iloc[i] > df['medium_ma'].iloc[i]) and \
                               (df['short_ma'].iloc[i-1] <= df['medium_ma'].iloc[i-1])
                               
            bearish_crossover = (df['short_ma'].iloc[i] < df['medium_ma'].iloc[i]) and \
                               (df['short_ma'].iloc[i-1] >= df['medium_ma'].iloc[i-1])
            
            # Apply filters
            valid_signal = True
            
            if volatility_filter:
                # Check if volatility is within acceptable range
                min_volatility = 0.005  # 0.5% minimum volatility
                max_volatility = 0.03   # 3% maximum volatility
                current_volatility = df['atr_pct'].iloc[i]
                
                valid_signal = valid_signal and (min_volatility <= current_volatility <= max_volatility)
                
            if volume_filter:
                # Check for sufficient volume
                min_volume_ratio = 0.8  # At least 80% of average volume
                current_volume_ratio = df['volume_ratio'].iloc[i]
                
                valid_signal = valid_signal and (current_volume_ratio >= min_volume_ratio)
            
            # Count consecutive signal days for confirmation
            if bullish_alignment or bullish_crossover:
                long_signal_count += 1
                short_signal_count = 0
            elif bearish_alignment or bearish_crossover:
                short_signal_count += 1
                long_signal_count = 0
            else:
                # No clear signal
                long_signal_count = 0
                short_signal_count = 0
            
            # Generate final signals with confirmation
            if valid_signal:
                if long_signal_count >= minimum_bars:
                    signals['position'].iloc[i] = 1  # Buy signal
                elif short_signal_count >= minimum_bars:
                    signals['position'].iloc[i] = -1  # Sell signal
        
        # Calculate stop loss and take profit levels if required
        if use_atr_stops:
            signals['stop_loss'] = np.nan
            signals['take_profit'] = np.nan
            
            # Calculate stop loss and take profit levels for each signal
            for i in range(len(signals)):
                if signals['position'].iloc[i] == 1:  # Buy signal
                    # ATR-based stop loss for long positions
                    signals['stop_loss'].iloc[i] = df['close'].iloc[i] - (df['atr'].iloc[i] * atr_multiplier)
                    # Take profit based on risk-reward ratio (typically 2:1 or 3:1)
                    risk = df['close'].iloc[i] - signals['stop_loss'].iloc[i]
                    signals['take_profit'].iloc[i] = df['close'].iloc[i] + (risk * 2)
                    
                elif signals['position'].iloc[i] == -1:  # Sell signal
                    # ATR-based stop loss for short positions
                    signals['stop_loss'].iloc[i] = df['close'].iloc[i] + (df['atr'].iloc[i] * atr_multiplier)
                    # Take profit based on risk-reward ratio
                    risk = signals['stop_loss'].iloc[i] - df['close'].iloc[i]
                    signals['take_profit'].iloc[i] = df['close'].iloc[i] - (risk * 2)
        
        # Add key indicators to the signals DataFrame for analysis
        signals['short_ma'] = df['short_ma']
        signals['medium_ma'] = df['medium_ma'] 
        signals['long_ma'] = df['long_ma']
        if 'atr' in df.columns:
            signals['atr'] = df['atr']
        
        # Add signal strength metric (distance between fast and slow MAs)
        signals['signal_strength'] = (df['short_ma'] - df['long_ma']) / df['close']
        
        return signals

    def mean_reversion_strategy(self, data: pd.DataFrame, params: Dict[str, Any]) -> pd.DataFrame:
        """
        Implements an advanced mean reversion strategy using Bollinger Bands and RSI.
        
        This strategy generates buy signals when price moves below the lower Bollinger Band
        and RSI indicates oversold conditions, and sell signals when price moves above the
        upper Bollinger Band and RSI indicates overbought conditions.
        
        Args:
            data (pd.DataFrame): Market data with OHLCV columns
            params (Dict[str, Any]): Strategy parameters
            
        Returns:
            pd.DataFrame: DataFrame with original data and added signals column
        """
        # Create copy of data to avoid modifying the original
        df = data.copy()
        
        # Extract parameters
        window = params.get('window', 20)
        std_dev = params.get('std_dev', 2.0)
        entry_threshold = params.get('entry_threshold', 2.0)
        exit_threshold = params.get('exit_threshold', 0.5)
        max_holding_period = params.get('max_holding_period', 10)
        mean_type = params.get('mean_type', 'sma')
        oversold_rsi = params.get('oversold_rsi', 30)
        overbought_rsi = params.get('overbought_rsi', 70)
        use_volume_filter = params.get('use_volume_filter', True)
        min_volatility = params.get('min_volatility', 0.01)
        max_volatility = params.get('max_volatility', 0.05)
        
        # Initialize signals DataFrame
        signals = pd.DataFrame(index=df.index)
        signals['position'] = 0
        
        # Calculate Bollinger Bands
        if mean_type == 'sma':
            df['rolling_mean'] = df['close'].rolling(window=window).mean()
        elif mean_type == 'ema':
            df['rolling_mean'] = df['close'].ewm(span=window, adjust=False).mean()
        elif mean_type == 'wma':
            # Weighted moving average (linear weights)
            weights = np.arange(1, window + 1)
            df['rolling_mean'] = df['close'].rolling(window=window).apply(
                lambda x: np.sum(weights * x) / np.sum(weights), raw=True)
        else:
            # Default to SMA if invalid type
            df['rolling_mean'] = df['close'].rolling(window=window).mean()
            
        df['rolling_std'] = df['close'].rolling(window=window).std()
        df['upper_band'] = df['rolling_mean'] + (df['rolling_std'] * entry_threshold)
        df['lower_band'] = df['rolling_mean'] - (df['rolling_std'] * entry_threshold)
        df['upper_exit_band'] = df['rolling_mean'] + (df['rolling_std'] * exit_threshold)
        df['lower_exit_band'] = df['rolling_mean'] - (df['rolling_std'] * exit_threshold)
        
        # Calculate RSI for confirmation
        delta = df['close'].diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        avg_gain = gain.rolling(window=14).mean()
        avg_loss = loss.rolling(window=14).mean()
        rs = avg_gain / avg_loss.replace(0, np.finfo(float).eps)  # Avoid division by zero
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # Calculate volatility
        df['volatility'] = df['rolling_std'] / df['rolling_mean']
        
        # Calculate volume indicators if needed
        if use_volume_filter:
            df['volume_sma'] = df['volume'].rolling(window=window).mean()
            df['volume_ratio'] = df['volume'] / df['volume_sma']
        
        # Generate signals with state tracking
        current_position = 0
        position_entry_index = None
        
        for i in range(window, len(df)):
            # Skip if not enough data
            if np.isnan(df['rolling_mean'].iloc[i]) or np.isnan(df['rsi'].iloc[i]):
                continue
                
            # Current volatility
            current_volatility = df['volatility'].iloc[i]
            volatility_in_range = min_volatility <= current_volatility <= max_volatility
            
            # Volume filter
            volume_ok = True
            if use_volume_filter:
                volume_ok = df['volume_ratio'].iloc[i] >= 0.8  # At least 80% of average volume
            
            # Check for trade exit
            if current_position == 1:  # Currently long
                # Exit if price crosses above the mean or upper exit band
                price_exit_long = df['close'].iloc[i] >= df['upper_exit_band'].iloc[i]
                # Exit if RSI becomes overbought
                rsi_exit_long = df['rsi'].iloc[i] >= overbought_rsi
                # Exit if holding period exceeded
                max_holding_exit = (position_entry_index is not None) and (i - position_entry_index >= max_holding_period)
                
                if price_exit_long or rsi_exit_long or max_holding_exit:
                    signals['position'].iloc[i] = 0  # Exit signal
                    current_position = 0
                    position_entry_index = None
                else:
                    signals['position'].iloc[i] = current_position  # Maintain position
                    
            elif current_position == -1:  # Currently short
                # Exit if price crosses below the mean or lower exit band
                price_exit_short = df['close'].iloc[i] <= df['lower_exit_band'].iloc[i]
                # Exit if RSI becomes oversold
                rsi_exit_short = df['rsi'].iloc[i] <= oversold_rsi
                # Exit if holding period exceeded
                max_holding_exit = (position_entry_index is not None) and (i - position_entry_index >= max_holding_period)
                
                if price_exit_short or rsi_exit_short or max_holding_exit:
                    signals['position'].iloc[i] = 0  # Exit signal
                    current_position = 0
                    position_entry_index = None
                else:
                    signals['position'].iloc[i] = current_position  # Maintain position
                    
            else:  # No current position
                # Check for oversold conditions (long entry)
                price_below_lower = df['close'].iloc[i] < df['lower_band'].iloc[i]
                rsi_oversold = df['rsi'].iloc[i] < oversold_rsi
                
                # Check for overbought conditions (short entry)
                price_above_upper = df['close'].iloc[i] > df['upper_band'].iloc[i]
                rsi_overbought = df['rsi'].iloc[i] > overbought_rsi
                
                # Generate signals
                if price_below_lower and rsi_oversold and volatility_in_range and volume_ok:
                    signals['position'].iloc[i] = 1  # Buy signal
                    current_position = 1
                    position_entry_index = i
                elif price_above_upper and rsi_overbought and volatility_in_range and volume_ok:
                    signals['position'].iloc[i] = -1  # Sell signal
                    current_position = -1
                    position_entry_index = i
        
        # Add key indicators to the signals DataFrame for analysis
        signals['rolling_mean'] = df['rolling_mean']
        signals['upper_band'] = df['upper_band']
        signals['lower_band'] = df['lower_band']
        signals['rsi'] = df['rsi']
        signals['volatility'] = df['volatility']
        
        # Add mean reversion strength metric (distance from mean normalized by bands)
        signals['reversion_strength'] = (df['close'] - df['rolling_mean']) / (df['upper_band'] - df['rolling_mean'])
        
        return signals

    def breakout_strategy(self, data: pd.DataFrame, params: Dict[str, Any]) -> pd.DataFrame:
        """
        Implements an advanced breakout strategy with support/resistance levels and filters.
        
        This strategy identifies support and resistance levels, and generates signals when
        price breaks through these levels with volume confirmation and volatility-based
        targets.
        
        Args:
            data (pd.DataFrame): Market data with OHLCV columns
            params (Dict[str, Any]): Strategy parameters
            
        Returns:
            pd.DataFrame: DataFrame with original data and added signals column
        """
        # Create copy of data to avoid modifying the original
        df = data.copy()
        
        # Extract parameters
        window = params.get('window', 50)
        volume_window = params.get('volume_window', 10) 
        volume_threshold = params.get('volume_threshold', 1.5)
        entry_confirmation_bars = params.get('entry_confirmation_bars', 2)
        atr_periods = params.get('atr_periods', 14)
        atr_multiplier = params.get('atr_multiplier', 2.5)
        use_dynamic_stops = params.get('use_dynamic_stops', True)
        false_breakout_filter = params.get('false_breakout_filter', True)
        min_consolidation_periods = params.get('min_consolidation_periods', 5)
        
        # Initialize signals DataFrame
        signals = pd.DataFrame(index=df.index)
        signals['position'] = 0
        
        # Calculate support and resistance levels
        df['rolling_high'] = df['high'].rolling(window=window).max()
        df['rolling_low'] = df['low'].rolling(window=window).min()
        
        # Identify consolidation periods (sideways markets)
        df['price_range'] = (df['rolling_high'] - df['rolling_low']) / df['close']
        df['is_consolidating'] = df['price_range'] < 0.05  # 5% range for consolidation
        
        # Calculate volume indicators
        df['volume_ma'] = df['volume'].rolling(window=volume_window).mean()
        df['volume_ratio'] = df['volume'] / df['volume_ma']
        
        # ATR for volatility-based targets
        high_low = df['high'] - df['low']
        high_close = abs(df['high'] - df['close'].shift())
        low_close = abs(df['low'] - df['close'].shift())
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        df['atr'] = true_range.rolling(window=atr_periods).mean()
        
        # Identify false breakout filter if enabled
        if false_breakout_filter:
            # Calculate ADX for trend strength
            plus_dm = df['high'].diff()
            minus_dm = df['low'].diff(-1)
            plus_dm = plus_dm.where((plus_dm > 0) & (plus_dm > minus_dm), 0)
            minus_dm = minus_dm.where((minus_dm > 0) & (minus_dm > plus_dm), 0)
            
            tr_smoothed = true_range.ewm(alpha=1/atr_periods, adjust=False).mean()
            plus_di = 100 * plus_dm.ewm(alpha=1/atr_periods, adjust=False).mean() / tr_smoothed
            minus_di = 100 * minus_dm.ewm(alpha=1/atr_periods, adjust=False).mean() / tr_smoothed
            dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di).replace(0, np.finfo(float).eps)
            df['adx'] = dx.ewm(alpha=1/atr_periods, adjust=False).mean()
        
        # Generate signals with confirmation tracking
        breakout_up_count = 0
        breakout_down_count = 0
        consolidation_count = 0
        
        for i in range(window, len(df)):
            # Skip if not enough data
            if np.isnan(df['rolling_high'].iloc[i-1]) or np.isnan(df['atr'].iloc[i]):
                continue
                
            # Check for sufficient consolidation period before breakout
            if df['is_consolidating'].iloc[i]:
                consolidation_count += 1
            else:
                consolidation_count = 0
                
            sufficient_consolidation = consolidation_count >= min_consolidation_periods
            
            # Volume confirmation
            volume_confirmation = df['volume_ratio'].iloc[i] >= volume_threshold
            
            # Trend strength filter for false breakouts
            trend_strength_ok = True
            if false_breakout_filter:
                trend_strength_ok = df['adx'].iloc[i] > 25  # ADX above 25 indicates trend
            
            # Check for breakouts
            breakout_up = df['close'].iloc[i] > df['rolling_high'].iloc[i-1]
            breakout_down = df['close'].iloc[i] < df['rolling_low'].iloc[i-1]
            
            # Count consecutive breakout bars for confirmation
            if breakout_up and volume_confirmation and trend_strength_ok:
                breakout_up_count += 1
                breakout_down_count = 0
            elif breakout_down and volume_confirmation and trend_strength_ok:
                breakout_down_count += 1
                breakout_up_count = 0
            else:
                # Reset counters if no breakout
                breakout_up_count = 0
                breakout_down_count = 0
            
            # Generate signals with confirmation
            if sufficient_consolidation:
                if breakout_up_count >= entry_confirmation_bars:
                    signals['position'].iloc[i] = 1  # Buy signal
                elif breakout_down_count >= entry_confirmation_bars:
                    signals['position'].iloc[i] = -1  # Sell signal
        
        # Calculate stop loss and take profit levels if required
        if use_dynamic_stops:
            signals['stop_loss'] = np.nan
            signals['take_profit'] = np.nan
            
            for i in range(len(signals)):
                if signals['position'].iloc[i] == 1:  # Buy signal
                    # ATR-based stop loss below the breakout level
                    resistance_level = df['rolling_high'].iloc[i-1]
                    signals['stop_loss'].iloc[i] = resistance_level - (df['atr'].iloc[i] * atr_multiplier / 2)
                    # Take profit based on ATR
                    signals['take_profit'].iloc[i] = df['close'].iloc[i] + (df['atr'].iloc[i] * atr_multiplier)
                    
                elif signals['position'].iloc[i] == -1:  # Sell signal
                    # ATR-based stop loss above the breakout level
                    support_level = df['rolling_low'].iloc[i-1]
                    signals['stop_loss'].iloc[i] = support_level + (df['atr'].iloc[i] * atr_multiplier / 2)
                    # Take profit based on ATR
                    signals['take_profit'].iloc[i] = df['close'].iloc[i] - (df['atr'].iloc[i] * atr_multiplier)
        
        # Add key indicators to the signals DataFrame for analysis
        signals['rolling_high'] = df['rolling_high']
        signals['rolling_low'] = df['rolling_low']
        signals['atr'] = df['atr']
        signals['volume_ratio'] = df['volume_ratio']
        if false_breakout_filter:
            signals['adx'] = df['adx']
        
        # Add breakout strength metric (distance from level normalized by ATR)
        signals['breakout_strength'] = np.zeros(len(signals))
        for i in range(1, len(signals)):
            if signals['position'].iloc[i] == 1:  # Buy breakout
                signals['breakout_strength'].iloc[i] = (df['close'].iloc[i] - df['rolling_high'].iloc[i-1]) / df['atr'].iloc[i]
            elif signals['position'].iloc[i] == -1:  # Sell breakout
                signals['breakout_strength'].iloc[i] = (df['rolling_low'].iloc[i-1] - df['close'].iloc[i]) / df['atr'].iloc[i]
        
        return signals

    def multi_timeframe_strategy(self, data: pd.DataFrame, params: Dict[str, Any]) -> pd.DataFrame:
        """
        Implements a multi-timeframe trading strategy that aligns signals across timeframes.
        
        This strategy evaluates moving averages across different timeframes and generates
        signals when there is alignment across multiple timeframes, enhancing signal quality.
        
        Note: This implementation assumes the input data is already resampled to the shortest timeframe
        and includes columns for each timeframe's indicators. In practice, you would resample the
        data to different timeframes before analysis.
        
        Args:
            data (pd.DataFrame): Market data with OHLCV columns
            params (Dict[str, Any]): Strategy parameters
            
        Returns:
            pd.DataFrame: DataFrame with original data and added signals column
        """
        # Create copy of data to avoid modifying the original
        df = data.copy()
        
        # Extract parameters
        short_window = params.get('short_window', 20)
        long_window = params.get('long_window', 50)
        primary_timeframe = params.get('primary_timeframe', '1h')
        confirmation_timeframe = params.get('confirmation_timeframe', '4h')
        trend_timeframe = params.get('trend_timeframe', '1d')
        alignment_required = params.get('alignment_required', True)
        min_trend_strength = params.get('min_trend_strength', 0.6)
        exit_on_trend_shift = params.get('exit_on_trend_shift', True)
        
        # Initialize signals DataFrame
        signals = pd.DataFrame(index=df.index)
        signals['position'] = 0
        
        # In a real implementation, you would have resampled data for each timeframe
        # Here we simulate this by creating EMAs with different windows to represent timeframes
        
        # Primary timeframe indicators (e.g., 1h)
        df['primary_short_ema'] = df['close'].ewm(span=short_window, adjust=False).mean()
        df['primary_long_ema'] = df['close'].ewm(span=long_window, adjust=False).mean()
        
        # Confirmation timeframe indicators (e.g., 4h)
        df['confirm_short_ema'] = df['close'].ewm(span=short_window*4, adjust=False).mean()
        df['confirm_long_ema'] = df['close'].ewm(span=long_window*4, adjust=False).mean()
        
        # Trend timeframe indicators (e.g., 1d)
        df['trend_ema'] = df['close'].ewm(span=long_window*24, adjust=False).mean()
        
        # Calculate trend strength
        df['trend_strength'] = (df['trend_ema'].pct_change(periods=20) * 100).abs()
        
        # Generate signals based on multi-timeframe alignment
        for i in range(long_window*24, len(df)):
            # Skip if not enough data
            if any(np.isnan(df.iloc[i][col]) for col in 
                  ['primary_short_ema', 'primary_long_ema', 'confirm_short_ema', 'confirm_long_ema', 'trend_ema']):
                continue
                
            # Check trend strength
            sufficient_trend = df['trend_strength'].iloc[i] >= min_trend_strength
            
            # Primary timeframe signal
            primary_bullish = df['primary_short_ema'].iloc[i] > df['primary_long_ema'].iloc[i]
            primary_bearish = df['primary_short_ema'].iloc[i] < df['primary_long_ema'].iloc[i]
            
            # Confirmation timeframe signal
            confirm_bullish = df['confirm_short_ema'].iloc[i] > df['confirm_long_ema'].iloc[i]
            confirm_bearish = df['confirm_short_ema'].iloc[i] < df['confirm_long_ema'].iloc[i]
            
            # Trend timeframe direction
            trend_bullish = df['close'].iloc[i] > df['trend_ema'].iloc[i]
            trend_bearish = df['close'].iloc[i] < df['trend_ema'].iloc[i]
            
            # Generate signals with timeframe alignment
            if alignment_required:
                # Require alignment across all timeframes
                if primary_bullish and confirm_bullish and trend_bullish and sufficient_trend:
                    signals['position'].iloc[i] = 1  # Buy signal
                elif primary_bearish and confirm_bearish and trend_bearish and sufficient_trend:
                    signals['position'].iloc[i] = -1  # Sell signal
                elif exit_on_trend_shift:
                    # Exit signals on trend misalignment
                    if (signals['position'].iloc[i-1] == 1 and 
                        not (primary_bullish and confirm_bullish and trend_bullish)):
                        signals['position'].iloc[i] = 0  # Exit long
                    elif (signals['position'].iloc[i-1] == -1 and 
                          not (primary_bearish and confirm_bearish and trend_bearish)):
                        signals['position'].iloc[i] = 0  # Exit short
                    else:
                        signals['position'].iloc[i] = signals['position'].iloc[i-1]  # Maintain position
                else:
                    # Maintain previous position
                    signals['position'].iloc[i] = signals['position'].iloc[i-1]
            else:
                # More flexible approach - primary and at least one other timeframe
                if primary_bullish and (confirm_bullish or trend_bullish) and sufficient_trend:
                    signals['position'].iloc[i] = 1  # Buy signal
                elif primary_bearish and (confirm_bearish or trend_bearish) and sufficient_trend:
                    signals['position'].iloc[i] = -1  # Sell signal
                elif exit_on_trend_shift:
                    # Exit signals on trend shift
                    if (signals['position'].iloc[i-1] == 1 and 
                        not (primary_bullish and (confirm_bullish or trend_bullish))):
                        signals['position'].iloc[i] = 0  # Exit long
                    elif (signals['position'].iloc[i-1] == -1 and 
                          not (primary_bearish and (confirm_bearish or trend_bearish))):
                        signals['position'].iloc[i] = 0  # Exit short
                    else:
                        signals['position'].iloc[i] = signals['position'].iloc[i-1]  # Maintain position
                else:
                    # Maintain previous position
                    signals['position'].iloc[i] = signals['position'].iloc[i-1]
        
        # Add key indicators to the signals DataFrame for analysis
        signals['primary_short_ema'] = df['primary_short_ema']
        signals['primary_long_ema'] = df['primary_long_ema']
        signals['confirm_short_ema'] = df['confirm_short_ema']
        signals['confirm_long_ema'] = df['confirm_long_ema']
        signals['trend_ema'] = df['trend_ema']
        signals['trend_strength'] = df['trend_strength']
        
        # Add alignment score metric
        signals['alignment_score'] = np.zeros(len(signals))
        for i in range(len(signals)):
            # Calculate alignment score (0-3 for timeframes aligned)
            primary_direction = 1 if df['primary_short_ema'].iloc[i] > df['primary_long_ema'].iloc[i] else -1
            confirm_direction = 1 if df['confirm_short_ema'].iloc[i] > df['confirm_long_ema'].iloc[i] else -1
            trend_direction = 1 if df['close'].iloc[i] > df['trend_ema'].iloc[i] else -1
            
            # Count aligned timeframes in the same direction
            aligned_count = (
                (primary_direction == 1 and confirm_direction == 1 and trend_direction == 1) or
                (primary_direction == -1 and confirm_direction == -1 and trend_direction == -1)
            )
            
            # Scale to -1 to 1 range
            if aligned_count:
                signals['alignment_score'].iloc[i] = primary_direction
            else:
                # Partial alignment scores
                if primary_direction == confirm_direction:
                    signals['alignment_score'].iloc[i] = primary_direction * 0.66  # 2/3 aligned
                elif primary_direction == trend_direction:
                    signals['alignment_score'].iloc[i] = primary_direction * 0.66  # 2/3 aligned
                else:
                    signals['alignment_score'].iloc[i] = primary_direction * 0.33  # 1/3 aligned
        
        return signals

    def volume_based_strategy(self, data: pd.DataFrame, params: Dict[str, Any]) -> pd.DataFrame:
        """
        Implements a volume-based trading strategy utilizing volume indicators and price action.
        
        This strategy generates signals based on volume spikes, on-balance volume (OBV),
        and price movements, with filters for volume quality and volatility.
        
        Args:
            data (pd.DataFrame): Market data with OHLCV columns
            params (Dict[str, Any]): Strategy parameters
            
        Returns:
            pd.DataFrame: DataFrame with original data and added signals column
        """
        # Create copy of data to avoid modifying the original
        df = data.copy()
        
        # Extract parameters
        volume_window = params.get('volume_window', 20)
        volume_threshold = params.get('volume_threshold', 1.8)
        price_window = params.get('price_window', 10)
        entry_delay = params.get('entry_delay', 1)
        use_obv = params.get('use_obv', True)
        use_vwap = params.get('use_vwap', True)
        obv_smoothing = params.get('obv_smoothing', 3)
        min_price_move = params.get('min_price_move', 0.005)
        use_price_action_filter = params.get('use_price_action_filter', True)
        
        # Initialize signals DataFrame
        signals = pd.DataFrame(index=df.index)
        signals['position'] = 0
        
        # Calculate volume indicators
        df['volume_sma'] = df['volume'].rolling(window=volume_window).mean()
        df['volume_ratio'] = df['volume'] / df['volume_sma']
        
        # Calculate price indicators
        df['price_sma'] = df['close'].rolling(window=price_window).mean()
        df['price_change'] = df['close'].pct_change()
        df['price_change_abs'] = df['price_change'].abs()
        
        # Calculate On-Balance Volume (OBV) if required
        if use_obv:
            obv = np.zeros(len(df))
            obv[0] = df['volume'].iloc[0]
            
            for i in range(1, len(df)):
                if df['close'].iloc[i] > df['close'].iloc[i-1]:
                    obv[i] = obv[i-1] + df['volume'].iloc[i]
                elif df['close'].iloc[i] < df['close'].iloc[i-1]:
                    obv[i] = obv[i-1] - df['volume'].iloc[i]
                else:
                    obv[i] = obv[i-1]
                    
            df['obv'] = obv
            df['obv_ema'] = df['obv'].ewm(span=obv_smoothing).mean()
            df['obv_slope'] = df['obv_ema'].pct_change(periods=5)
        
        # Calculate VWAP (Volume-Weighted Average Price) if required
        if use_vwap:
            # Calculate daily VWAP (assumes data is sorted by date/time)
            df['date'] = pd.to_datetime(df.index).date
            df['cumulative_volume'] = df.groupby('date')['volume'].cumsum()
            df['cumulative_vp'] = df.groupby('date').apply(
                lambda x: (x['typical_price'] * x['volume']).cumsum()).reset_index(level=0, drop=True)
            df['vwap'] = df['cumulative_vp'] / df['cumulative_volume'].replace(0, np.nan)
            
        # Generate signals
        for i in range(volume_window + entry_delay, len(df)):
            # Skip if not enough data
            if np.isnan(df['volume_sma'].iloc[i]) or np.isnan(df['price_sma'].iloc[i]):
                continue
                
            # Check for volume spike
            volume_spike = df['volume_ratio'].iloc[i-entry_delay] >= volume_threshold
            
            # Check for significant price movement
            price_move_significant = df['price_change_abs'].iloc[i-entry_delay] >= min_price_move
            
            # Price action filters
            price_action_ok = True
            if use_price_action_filter:
                # Check if price candle has a strong body (not a doji)
                body_size = abs(df['close'].iloc[i-entry_delay] - df['open'].iloc[i-entry_delay])
                range_size = df['high'].iloc[i-entry_delay] - df['low'].iloc[i-entry_delay]
                body_to_range_ratio = body_size / range_size if range_size > 0 else 0
                
                price_action_ok = body_to_range_ratio >= 0.5  # Body is at least 50% of range
            
            # OBV confirmation
            obv_confirmation = True
            if use_obv:
                # Check OBV slope
                obv_bullish = df['obv_slope'].iloc[i-entry_delay] > 0.01  # 1% increase
                obv_bearish = df['obv_slope'].iloc[i-entry_delay] < -0.01  # 1% decrease
                
                # Update confirmation status based on OBV
                if df['price_change'].iloc[i-entry_delay] > 0:  # Bullish price move
                    obv_confirmation = obv_bullish
                elif df['price_change'].iloc[i-entry_delay] < 0:  # Bearish price move
                    obv_confirmation = obv_bearish
            
            # VWAP confirmation
            vwap_confirmation = True
            if use_vwap:
                price_above_vwap = df['close'].iloc[i-entry_delay] > df['vwap'].iloc[i-entry_delay]
                price_below_vwap = df['close'].iloc[i-entry_delay] < df['vwap'].iloc[i-entry_delay]
                
                # Update confirmation status based on VWAP
                if df['price_change'].iloc[i-entry_delay] > 0:  # Bullish price move
                    vwap_confirmation = price_above_vwap
                elif df['price_change'].iloc[i-entry_delay] < 0:  # Bearish price move
                    vwap_confirmation = price_below_vwap
            
            # Generate signals
            if volume_spike and price_move_significant and price_action_ok and obv_confirmation and vwap_confirmation:
                if df['price_change'].iloc[i-entry_delay] > 0:
                    signals['position'].iloc[i] = 1  # Buy signal
                elif df['price_change'].iloc[i-entry_delay] < 0:
                    signals['position'].iloc[i] = -1  # Sell signal
        
        # Add key indicators to the signals DataFrame for analysis
        signals['volume_ratio'] = df['volume_ratio']
        signals['price_change'] = df['price_change']
        if use_obv:
            signals['obv'] = df['obv']
            signals['obv_ema'] = df['obv_ema']
            signals['obv_slope'] = df['obv_slope']
        if use_vwap:
            signals['vwap'] = df['vwap']
        
        # Add volume spike strength metric
        signals['volume_strength'] = np.zeros(len(signals))
        for i in range(len(signals)):
            if i < volume_window:
                continue
                
            if signals['position'].iloc[i] != 0:
                signals['volume_strength'].iloc[i] = df['volume_ratio'].iloc[i-entry_delay] * df['price_change_abs'].iloc[i-entry_delay]
        
        return signals

    def ml_based_strategy(self, data: pd.DataFrame, params: Dict[str, Any]) -> pd.DataFrame:
        """
        Implements a machine learning-based trading strategy using model predictions.
        
        This strategy uses predictions from machine learning models (assumed to be
        pre-trained and available) to generate trading signals, with confidence
        thresholds and risk management.
        
        Note: This implementation provides a framework for integrating machine learning models.
        In practice, you would need to load and use actual trained models.
        
        Args:
            data (pd.DataFrame): Market data with OHLCV columns
            params (Dict[str, Any]): Strategy parameters
            
        Returns:
            pd.DataFrame: DataFrame with original data and added signals column
        """
        # Create copy of data to avoid modifying the original
        df = data.copy()
        
        # Extract parameters
        prediction_threshold = params.get('prediction_threshold', 0.65)
        ensemble_weighting = params.get('ensemble_weighting', {
            'xgboost': 0.4,
            'lstm': 0.3,
            'random_forest': 0.3
        })
        min_samples_for_signal = params.get('min_samples_for_signal', 10)
        use_adaptive_thresholds = params.get('use_adaptive_thresholds', True)
        use_market_regime_filter = params.get('use_market_regime_filter', True)
        
        # Initialize signals DataFrame
        signals = pd.DataFrame(index=df.index)
        signals['position'] = 0
        
        # In a real implementation, here you would:
        # 1. Prepare features for the models
        # 2. Load the pre-trained models
        # 3. Generate predictions from each model
        
        # Simulate model predictions for demonstration
        # In practice, these would come from actual model inference
        rng = np.random.RandomState(42)  # Fixed seed for reproducibility
        
        # Simulate predictions from different models
        xgboost_preds = rng.normal(0, 0.3, size=len(df)).cumsum() / 100
        lstm_preds = rng.normal(0, 0.2, size=len(df)).cumsum() / 100
        rf_preds = rng.normal(0, 0.25, size=len(df)).cumsum() / 100
        
        # Normalize predictions to -1 to 1 range
        xgboost_preds = np.tanh(xgboost_preds)
        lstm_preds = np.tanh(lstm_preds)
        rf_preds = np.tanh(rf_preds)
        
        # Store predictions in DataFrame
        df['xgboost_pred'] = xgboost_preds
        df['lstm_pred'] = lstm_preds
        df['rf_pred'] = rf_preds
        
        # Calculate ensemble prediction with weighting
        df['ensemble_pred'] = (
            df['xgboost_pred'] * ensemble_weighting.get('xgboost', 0.33) +
            df['lstm_pred'] * ensemble_weighting.get('lstm', 0.33) +
            df['rf_pred'] * ensemble_weighting.get('random_forest', 0.33)
        )
        
        # Calculate prediction confidence (absolute value of prediction)
        df['prediction_confidence'] = df['ensemble_pred'].abs()
        
        # Detect market regime if required
        if use_market_regime_filter:
            # Simple market regime detection using volatility and trend
            df['returns'] = df['close'].pct_change()
            df['volatility'] = df['returns'].rolling(window=20).std()
            df['trend'] = df['close'].rolling(window=50).mean().pct_change(20)
            
            # Determine market regime
            df['regime'] = 'neutral'
            df.loc[(df['volatility'] > df['volatility'].quantile(0.7)) & 
                  (df['trend'].abs() < 0.01), 'regime'] = 'volatile_sideways'
            df.loc[(df['volatility'] < df['volatility'].quantile(0.3)) & 
                  (df['trend'].abs() < 0.01), 'regime'] = 'calm_sideways'
            df.loc[(df['volatility'] > df['volatility'].quantile(0.5)) & 
                  (df['trend'] > 0.01), 'regime'] = 'volatile_uptrend'
            df.loc[(df['volatility'] > df['volatility'].quantile(0.5)) & 
                  (df['trend'] < -0.01), 'regime'] = 'volatile_downtrend'
            df.loc[(df['volatility'] < df['volatility'].quantile(0.5)) & 
                  (df['trend'] > 0.01), 'regime'] = 'calm_uptrend'
            df.loc[(df['volatility'] < df['volatility'].quantile(0.5)) & 
                  (df['trend'] < -0.01), 'regime'] = 'calm_downtrend'
        
        # Adjust threshold based on market regime if required
        if use_adaptive_thresholds and use_market_regime_filter:
            df['adjusted_threshold'] = prediction_threshold
            
            # Higher threshold for volatile markets
            df.loc[df['regime'].str.contains('volatile'), 'adjusted_threshold'] = prediction_threshold * 1.2
            
            # Lower threshold for calm trending markets
            df.loc[df['regime'].isin(['calm_uptrend', 'calm_downtrend']), 'adjusted_threshold'] = prediction_threshold * 0.9
        else:
            df['adjusted_threshold'] = prediction_threshold
        
        # Generate signals based on predictions
        for i in range(min_samples_for_signal, len(df)):
            # Get current threshold
            current_threshold = df['adjusted_threshold'].iloc[i]
            
            # Check prediction confidence
            if df['prediction_confidence'].iloc[i] >= current_threshold:
                if df['ensemble_pred'].iloc[i] > 0:
                    signals['position'].iloc[i] = 1  # Buy signal
                elif df['ensemble_pred'].iloc[i] < 0:
                    signals['position'].iloc[i] = -1  # Sell signal
        
        # Add key indicators to the signals DataFrame for analysis
        signals['ensemble_pred'] = df['ensemble_pred']
        signals['prediction_confidence'] = df['prediction_confidence']
        if use_market_regime_filter:
            signals['regime'] = df['regime']
            signals['volatility'] = df['volatility']
        
        # Add model disagreement metric
        signals['model_disagreement'] = np.zeros(len(signals))
        for i in range(len(signals)):
            # Calculate standard deviation of predictions to measure disagreement
            model_preds = [df['xgboost_pred'].iloc[i], df['lstm_pred'].iloc[i], df['rf_pred'].iloc[i]]
            signals['model_disagreement'].iloc[i] = np.std(model_preds)
        
        return signals

    def add_custom_strategy(self, name: str, strategy_function: Callable) -> bool:
        """
        Add a custom strategy to the available strategies.
        
        Args:
            name (str): Name of the custom strategy
            strategy_function (Callable): Strategy function that takes data and parameters
                                        and returns a DataFrame with signals
                                        
        Returns:
            bool: True if strategy was added successfully, False otherwise
        """
        if name in self.strategies:
            logger.warning(f"Strategy '{name}' already exists. Use a different name.")
            return False
        
        # Validate strategy function signature
        try:
            # Create minimal test data
            test_data = pd.DataFrame({
                'open': [1.0, 2.0, 3.0],
                'high': [1.1, 2.1, 3.1],
                'low': [0.9, 1.9, 2.9],
                'close': [1.0, 2.0, 3.0],
                'volume': [100, 200, 300]
            })
            
            # Test strategy function with empty parameters
            result = strategy_function(test_data, {})
            
            # Check result
            if not isinstance(result, pd.DataFrame):
                logger.error(f"Custom strategy '{name}' function must return a DataFrame")
                return False
                
            if 'position' not in result.columns:
                logger.error(f"Custom strategy '{name}' must return a DataFrame with 'position' column")
                return False
                
        except Exception as e:
            logger.error(f"Error validating custom strategy '{name}': {str(e)}")
            return False
            
        # Add strategy to available strategies
        self.strategies[name] = strategy_function
        
        # Initialize default parameters (empty dictionary)
        self.strategy_params[name] = {}
        
        logger.info(f"Custom strategy '{name}' added successfully")
        return True

    def get_available_strategies(self) -> Dict[str, Dict[str, Any]]:
        """
        Get information about all available strategies.
        
        Returns:
            Dict[str, Dict[str, Any]]: Dictionary mapping strategy names to metadata
        """
        strategies_info = {}
        
        for name in self.strategies.keys():
            # Determine strategy type (built-in or custom)
            is_builtin = name in ['trend_following', 'mean_reversion', 'breakout', 
                                 'multi_timeframe', 'volume_based', 'ml_based']
            
            # Get parameters
            params = self.get_strategy_parameters(name)
            
            # Create strategy metadata
            strategies_info[name] = {
                'type': 'built-in' if is_builtin else 'custom',
                'parameters': params,
                'optimized': name in self.optimized_params
            }
            
        return strategies_info

    def reset_parameters(self, strategy_name: Optional[str] = None) -> None:
        """
        Reset strategy parameters to default values.
        
        Args:
            strategy_name (Optional[str]): Name of the strategy to reset, or None to reset all
            
        Raises:
            ValueError: If the specified strategy does not exist
        """
        if strategy_name is not None:
            if strategy_name not in self.strategies:
                error_msg = f"Strategy '{strategy_name}' not found. Available strategies: {list(self.strategies.keys())}"
                logger.error(error_msg)
                raise ValueError(error_msg)
                
            # Reset specific strategy
            if strategy_name in self.optimized_params:
                del self.optimized_params[strategy_name]
                logger.info(f"Parameters for strategy '{strategy_name}' reset to defaults")
        else:
            # Reset all strategies
            self.optimized_params = {}
            logger.info("All strategy parameters reset to defaults")
