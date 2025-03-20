# strategies/implementations/trend_following.py

import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Tuple, Optional, Any, Union
import time

class TrendFollowingStrategy:
    """
    Advanced trend following strategy using multiple indicators and adaptive parameters.
    
    Features:
    - Multi-timeframe moving average crossover system
    - Trend strength confirmation with ADX
    - Adaptive parameters based on market volatility
    - Dynamic position sizing integration
    - Advanced entry/exit logic with multiple confirmation signals
    - Volatility-based stop loss and take profit
    """
    
    def __init__(self) -> None:
        """Initialize the strategy with default parameters."""
        # Default strategy parameters - will be overridden by optimization
        self.default_params = {
            # Moving average parameters
            'short_window': 10,
            'medium_window': 50,
            'long_window': 200,
            
            # Trend confirmation
            'adx_period': 14,
            'adx_threshold': 25,
            
            # Volatility adjustment
            'volatility_adjustment': True,
            'atr_period': 14,
            'atr_multiplier': 2.0,
            
            # Entry/exit parameters
            'entry_confirmation_window': 3,
            'exit_confirmation_window': 2,
            
            # Risk management
            'stop_loss_atr_multiplier': 2.0,
            'take_profit_atr_multiplier': 4.0,
            'trailing_stop_activation': 0.02,  # 2% in profit to activate
            'trailing_stop_distance': 2.5,     # ATR multiplier for trailing stop
            
            # Position sizing (if not using external risk manager)
            'position_size_pct': 0.02,         # 2% risk per trade
            'max_position_size_pct': 0.05      # 5% maximum position size
        }
        
        # Strategy state variables
        self.params = self.default_params.copy()
        self.is_initialized = False
        self.current_position = 0
        self.entry_price = None
        self.stop_loss = None
        self.take_profit = None
        self.trailing_stop = None
        self.entry_time = None
        self.confirmation_counter = 0
        self.exit_counter = 0
        self.last_signal = 0
        
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
        self.logger.info("Trend Following Strategy initialized")
        
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
            # Ensure we have enough data for longest window calculations
            required_length = max(
                self.params['long_window'] + 100,
                self.params['adx_period'] + 100
            )
            
            if len(historical_data) < required_length:
                self.logger.warning(
                    f"Historical data may be insufficient for reliable signals. "
                    f"Got {len(historical_data)} bars, recommended {required_length}"
                )
            
            # Pre-calculate indicators
            self._calculate_indicators(historical_data)
            
            self.is_initialized = True
            self.logger.info("Strategy successfully initialized with historical data")
            
        except Exception as e:
            self.logger.error(f"Error initializing strategy: {e}")
            raise
            
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
        short_window = self.params['short_window']
        medium_window = self.params['medium_window']
        long_window = self.params['long_window']
        adx_period = self.params['adx_period']
        atr_period = self.params['atr_period']
        
        # Create cache key
        cache_key = f"{short_window}_{medium_window}_{long_window}_{adx_period}_{atr_period}_{len(data)}"
        
        # Check if we have cached calculations
        if cache_key in self._calculation_cache:
            # Check if cache is recent (within 1 minute)
            last_calc = self._last_calc_time.get(cache_key, 0)
            if time.time() - last_calc < 60:
                return self._calculation_cache[cache_key]
        
        try:
            # Calculate moving averages
            indicators['sma_short'] = data['close'].rolling(window=short_window).mean()
            indicators['sma_medium'] = data['close'].rolling(window=medium_window).mean()
            indicators['sma_long'] = data['close'].rolling(window=long_window).mean()
            
            # Calculate EMAs which often perform better in trending markets
            indicators['ema_short'] = data['close'].ewm(span=short_window, adjust=False).mean()
            indicators['ema_medium'] = data['close'].ewm(span=medium_window, adjust=False).mean()
            indicators['ema_long'] = data['close'].ewm(span=long_window, adjust=False).mean()
            
            # Calculate ADX for trend strength
            high = data['high']
            low = data['low']
            close = data['close']
            
            # True Range
            tr1 = abs(high - low)
            tr2 = abs(high - close.shift())
            tr3 = abs(low - close.shift())
            tr = pd.DataFrame({'tr1': tr1, 'tr2': tr2, 'tr3': tr3}).max(axis=1)
            atr = tr.rolling(window=atr_period).mean()
            indicators['atr'] = atr
            
            # Plus Directional Movement (+DM)
            plus_dm = high.diff()
            plus_dm = plus_dm.where((plus_dm > 0) & (plus_dm > -low.diff()), 0)
            
            # Minus Directional Movement (-DM)
            minus_dm = low.diff()
            minus_dm = minus_dm.where((minus_dm > 0) & (minus_dm > high.diff()), 0)
            
            # Smooth +DM and -DM with Wilder's smoothing technique
            smoothed_plus_dm = plus_dm.rolling(window=adx_period).sum()
            smoothed_minus_dm = minus_dm.rolling(window=adx_period).sum()
            
            # Directional Indicators (+DI and -DI)
            plus_di = 100 * (smoothed_plus_dm / atr.rolling(window=adx_period).sum())
            minus_di = 100 * (smoothed_minus_dm / atr.rolling(window=adx_period).sum())
            
            # Directional Movement Index (DX)
            dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
            
            # Average Directional Index (ADX)
            indicators['adx'] = dx.rolling(window=adx_period).mean()
            indicators['plus_di'] = plus_di
            indicators['minus_di'] = minus_di
            
            # Calculate MACD for additional trend confirmation
            ema12 = data['close'].ewm(span=12, adjust=False).mean()
            ema26 = data['close'].ewm(span=26, adjust=False).mean()
            indicators['macd'] = ema12 - ema26
            indicators['macd_signal'] = indicators['macd'].ewm(span=9, adjust=False).mean()
            indicators['macd_hist'] = indicators['macd'] - indicators['macd_signal']
            
            # Calculate Rate of Change (ROC) for momentum confirmation
            indicators['roc'] = data['close'].pct_change(periods=10) * 100
            
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
            close = current_data['close'].iloc[-1]
            
            sma_short = indicators['sma_short'].iloc[-1]
            sma_medium = indicators['sma_medium'].iloc[-1]
            sma_long = indicators['sma_long'].iloc[-1]
            
            ema_short = indicators['ema_short'].iloc[-1]
            ema_medium = indicators['ema_medium'].iloc[-1]
            ema_long = indicators['ema_long'].iloc[-1]
            
            adx = indicators['adx'].iloc[-1]
            plus_di = indicators['plus_di'].iloc[-1]
            minus_di = indicators['minus_di'].iloc[-1]
            
            macd = indicators['macd'].iloc[-1]
            macd_signal = indicators['macd_signal'].iloc[-1]
            macd_hist = indicators['macd_hist'].iloc[-1]
            
            roc = indicators['roc'].iloc[-1]
            atr = indicators['atr'].iloc[-1]
            
            # Get previous values for crossover detection
            prev_ema_short = indicators['ema_short'].iloc[-2]
            prev_ema_medium = indicators['ema_medium'].iloc[-2]
            
            # 1. Check for strong trend (ADX > threshold)
            strong_trend = adx > self.params['adx_threshold']
            
            # 2. Determine trend direction based on moving averages
            uptrend = (
                (ema_short > ema_medium) and 
                (ema_medium > ema_long) and
                (plus_di > minus_di)
            )
            
            downtrend = (
                (ema_short < ema_medium) and 
                (ema_medium < ema_long) and
                (minus_di > plus_di)
            )
            
            # 3. Check for crossovers
            bullish_crossover = (
                (ema_short > ema_medium) and (prev_ema_short <= prev_ema_medium)
            )
            
            bearish_crossover = (
                (ema_short < ema_medium) and (prev_ema_short >= prev_ema_medium)
            )
            
            # 4. Check for MACD confirmation
            macd_bullish = macd > macd_signal
            macd_bearish = macd < macd_signal
            
            # 5. Calculate signal and strength
            signal = 0
            signal_strength = 0
            
            # Initialize conditions list for metadata
            conditions = {}
            
            # Bullish signal logic
            if strong_trend and uptrend:
                # Strong uptrend conditions
                buy_conditions = [
                    bullish_crossover,
                    macd_bullish,
                    roc > 0,
                    close > sma_medium
                ]
                
                conditions["strong_trend"] = strong_trend
                conditions["uptrend"] = uptrend
                conditions["bullish_crossover"] = bullish_crossover
                conditions["macd_bullish"] = macd_bullish
                conditions["positive_roc"] = roc > 0
                conditions["above_medium_ma"] = close > sma_medium
                
                # Count confirmed conditions
                confirmed_conditions = sum(buy_conditions)
                
                # Require at least 2 confirmations for a buy signal
                if confirmed_conditions >= 2:
                    signal = 1
                    # Scale strength based on number of confirmations (0.5-1.0)
                    signal_strength = 0.5 + (0.5 * confirmed_conditions / len(buy_conditions))
                    
                    # Increment confirmation counter for entry filtering
                    self.confirmation_counter += 1
                    self.exit_counter = 0
                else:
                    # Not enough confirmations
                    self.confirmation_counter = 0
                    
            # Bearish signal logic
            elif strong_trend and downtrend:
                # Strong downtrend conditions
                sell_conditions = [
                    bearish_crossover,
                    macd_bearish,
                    roc < 0,
                    close < sma_medium
                ]
                
                conditions["strong_trend"] = strong_trend
                conditions["downtrend"] = downtrend
                conditions["bearish_crossover"] = bearish_crossover
                conditions["macd_bearish"] = macd_bearish
                conditions["negative_roc"] = roc < 0
                conditions["below_medium_ma"] = close < sma_medium
                
                # Count confirmed conditions
                confirmed_conditions = sum(sell_conditions)
                
                # Require at least 2 confirmations for a sell signal
                if confirmed_conditions >= 2:
                    signal = -1
                    # Scale strength based on number of confirmations (0.5-1.0)
                    signal_strength = 0.5 + (0.5 * confirmed_conditions / len(sell_conditions))
                    
                    # Increment confirmation counter for entry filtering
                    self.confirmation_counter += 1
                    self.exit_counter = 0
                else:
                    # Not enough confirmations
                    self.confirmation_counter = 0
            
            # Exit signal logic for existing positions
            elif self.current_position != 0:
                # Check exit conditions based on current position
                if self.current_position > 0:  # Long position
                    # Exit long if trend weakens or reverses
                    exit_conditions = [
                        bearish_crossover,
                        ema_short < ema_medium,
                        macd_bearish,
                        roc < 0
                    ]
                    
                    conditions["bearish_crossover"] = bearish_crossover
                    conditions["below_medium_ema"] = ema_short < ema_medium
                    conditions["macd_bearish"] = macd_bearish
                    conditions["negative_roc"] = roc < 0
                    
                    confirmed_exits = sum(exit_conditions)
                    
                    if confirmed_exits >= 2:
                        self.exit_counter += 1
                    else:
                        self.exit_counter = 0
                        
                else:  # Short position
                    # Exit short if trend weakens or reverses
                    exit_conditions = [
                        bullish_crossover,
                        ema_short > ema_medium,
                        macd_bullish,
                        roc > 0
                    ]
                    
                    conditions["bullish_crossover"] = bullish_crossover
                    conditions["above_medium_ema"] = ema_short > ema_medium
                    conditions["macd_bullish"] = macd_bullish
                    conditions["positive_roc"] = roc > 0
                    
                    confirmed_exits = sum(exit_conditions)
                    
                    if confirmed_exits >= 2:
                        self.exit_counter += 1
                    else:
                        self.exit_counter = 0
            
            # Apply filter for entry confirmation
            if signal != 0 and self.confirmation_counter < self.params['entry_confirmation_window']:
                # Need more confirmation bars before acting
                self.logger.debug(f"Waiting for entry confirmation: {self.confirmation_counter}/{self.params['entry_confirmation_window']}")
                signal = 0
                signal_strength = 0
                
            # Check for exit signal based on exit counter
            if self.current_position != 0 and self.exit_counter >= self.params['exit_confirmation_window']:
                # Exit signal confirmed
                signal = -self.current_position  # Opposite of current position
                signal_strength = 0.8  # Strong exit signal
                self.logger.info(f"Exit signal confirmed after {self.exit_counter} confirmation bars")
            
            # Set stop loss and take profit levels if entering a new position
            if signal != 0 and self.current_position == 0:
                self.entry_price = close
                self.entry_time = latest_idx
                
                # Calculate ATR-based stop loss and take profit
                stop_loss_distance = atr * self.params['stop_loss_atr_multiplier']
                take_profit_distance = atr * self.params['take_profit_atr_multiplier']
                
                if signal > 0:  # Long position
                    self.stop_loss = close - stop_loss_distance
                    self.take_profit = close + take_profit_distance
                else:  # Short position
                    self.stop_loss = close + stop_loss_distance
                    self.take_profit = close - take_profit_distance
                    
                # Initialize trailing stop
                self.trailing_stop = None
            
            # Update trailing stop if in a position and in profit
            if self.current_position != 0 and self.entry_price is not None:
                # Check if we're in profit enough to activate trailing stop
                activation_threshold = self.entry_price * (1 + self.params['trailing_stop_activation'] * self.current_position)
                
                if (self.current_position > 0 and close > activation_threshold) or \
                   (self.current_position < 0 and close < activation_threshold):
                    # Calculate trailing stop level
                    trailing_distance = atr * self.params['trailing_stop_distance']
                    
                    if self.current_position > 0:  # Long position
                        new_stop = close - trailing_distance
                        # Only update if new stop is higher than current
                        if self.trailing_stop is None or new_stop > self.trailing_stop:
                            self.trailing_stop = new_stop
                    else:  # Short position
                        new_stop = close + trailing_distance
                        # Only update if new stop is lower than current
                        if self.trailing_stop is None or new_stop < self.trailing_stop:
                            self.trailing_stop = new_stop
            
            # Store last signal
            self.last_signal = signal
            
            # Calculate adaptive position size if needed
            position_size = self._calculate_position_size(current_data, signal_strength)
            
            # Create signal dictionary
            signal_dict = {
                'signal': signal,
                'strength': signal_strength,
                'position_size': position_size,
                'stop_loss': self.stop_loss,
                'take_profit': self.take_profit,
                'trailing_stop': self.trailing_stop,
                'metadata': {
                    'strategy': 'trend_following',
                    'adx': adx,
                    'atr': atr,
                    'ema_short': ema_short,
                    'ema_medium': ema_medium,
                    'ema_long': ema_long,
                    'macd': macd,
                    'macd_signal': macd_signal,
                    'roc': roc,
                    'confirmation_counter': self.confirmation_counter,
                    'exit_counter': self.exit_counter,
                    'conditions': conditions
                }
            }
            
            return signal_dict
            
        except Exception as e:
            self.logger.error(f"Error generating signal: {e}")
            # Return no signal in case of error
            return {'signal': 0, 'strength': 0, 'metadata': {'error': str(e)}}
            
    def _calculate_position_size(self, data: pd.DataFrame, signal_strength: float) -> float:
        """
        Calculate appropriate position size based on volatility and signal strength.
        
        Args:
            data: Current market data
            signal_strength: Strength of the trading signal (0-1)
            
        Returns:
            Position size as a percentage of account
        """
        # Start with base position size
        position_size = self.params['position_size_pct']
        
        # Scale by signal strength
        position_size *= signal_strength
        
        # If volatility adjustment is enabled
        if self.params['volatility_adjustment']:
            try:
                # Get ATR as percentage of price
                atr = self._calculation_cache[list(self._calculation_cache.keys())[-1]]['atr'].iloc[-1]
                close = data['close'].iloc[-1]
                atr_pct = atr / close
                
                # Calculate volatility ratio (current vs historical)
                historical_atr_pct = atr_pct.mean() if isinstance(atr_pct, pd.Series) else atr_pct
                vol_ratio = atr_pct / historical_atr_pct if historical_atr_pct > 0 else 1.0
                
                # Adjust position size inversely to volatility
                position_size /= max(1.0, vol_ratio)
                
            except Exception as e:
                self.logger.warning(f"Error in volatility adjustment: {e}")
        
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
            
            # Reset counters
            self.confirmation_counter = 0
            self.exit_counter = 0
            
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
            
        # Check trailing stop first if active
        if self.trailing_stop is not None:
            if (self.current_position > 0 and current_price < self.trailing_stop) or \
               (self.current_position < 0 and current_price > self.trailing_stop):
                self.logger.info(f"Trailing stop hit: {self.trailing_stop}")
                return True
        
        # Check regular stop loss
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
        self.trailing_stop = None
        self.entry_time = None
        self.confirmation_counter = 0
        self.exit_counter = 0
        self.last_signal = 0
        
        # Clear calculation cache
        self._calculation_cache = {}
        self._last_calc_time = {}
        
        self.logger.info("Strategy reset to initial state")

# Factory function to create the strategy
def create_strategy() -> TrendFollowingStrategy:
    """Create a new instance of the TrendFollowingStrategy."""
    return TrendFollowingStrategy()
