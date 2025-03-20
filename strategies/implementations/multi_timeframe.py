# strategies/implementations/multi_timeframe.py

import numpy as np
import pandas as pd
import logging
import time
from typing import Dict, List, Optional, Union, Tuple, Any, Callable
from datetime import datetime, timedelta

class MultiTimeframeStrategy:
    """
    Advanced multi-timeframe trading strategy with trend alignment and momentum detection.
    
    This strategy:
    1. Analyzes price action across multiple timeframes (typically 15m, 1h, 4h, 1d)
    2. Identifies trend alignment and divergence between timeframes
    3. Uses weighted signals from each timeframe based on importance
    4. Adapts to changing market conditions by adjusting timeframe weights
    5. Implements advanced entry/exit mechanisms based on trend strength and momentum
    """
    
    def __init__(self, params: Optional[Dict[str, Any]] = None):
        """
        Initialize the multi-timeframe strategy with configurable parameters.
        
        Args:
            params: Strategy parameters dictionary
        """
        # Default parameters
        self.default_params = {
            # Timeframe configuration
            "timeframes": ["15m", "1h", "4h", "1d"],  # Timeframes to analyze
            "tf_weights": [0.1, 0.2, 0.3, 0.4],       # Default weights for timeframes
            "adaptive_weights": True,                  # Adjust weights based on performance
            
            # Trend detection
            "fast_ma_periods": {"15m": 9, "1h": 12, "4h": 9, "1d": 9},
            "slow_ma_periods": {"15m": 21, "1h": 24, "4h": 21, "1d": 21},
            "trend_ma_periods": {"15m": 50, "1h": 50, "4h": 50, "1d": 50},
            "ma_type": "ema",                         # "sma", "ema", or "wma"
            
            # Momentum indicators
            "rsi_period": 14,                         # RSI calculation period
            "rsi_overbought": 70,                     # RSI overbought threshold
            "rsi_oversold": 30,                       # RSI oversold threshold
            "macd_fast": 12,                          # MACD fast period
            "macd_slow": 26,                          # MACD slow period
            "macd_signal": 9,                         # MACD signal period
            
            # Signal generation
            "min_timeframe_alignment": 3,             # Minimum timeframes that must align
            "entry_timeout": 3,                       # Bars to wait before canceling entry signal
            "require_momentum_confirmation": True,    # Require momentum indicator confirmation
            
            # Risk management
            "risk_per_trade": 0.02,                   # Risk per trade (% of portfolio)
            "stop_loss_atr": 2.0,                     # Stop loss distance in ATR
            "stop_loss_timeframe": "1h",              # Timeframe for stop loss calculation
            "take_profit_atr": 4.0,                   # Take profit distance in ATR
            "trailing_stop": True,                    # Enable trailing stop
            "trailing_activation": 0.02,              # Activate trailing at this profit (%)
            "trailing_distance": 2.0,                 # Trailing stop distance in ATR
            
            # Exit rules
            "timeframe_divergence_exit": True,        # Exit on timeframe trend divergence
            "exit_on_trend_change": True,             # Exit when trend changes on higher timeframes
            "preserve_profits": True,                 # Take partial profits at targets
            "profit_taking_levels": [0.025, 0.05, 0.1],  # Profit targets for scaling out
            "profit_taking_percentages": [0.25, 0.25, 0.25],  # Percentage to take at each level
            
            # Performance adaptation
            "win_streak_factor": 0.1,                 # Increase size on win streaks
            "loss_streak_factor": 0.2,                # Decrease size on loss streaks
            "max_win_streak_increase": 0.5,           # Maximum position size increase from streak
            "max_loss_streak_decrease": 0.7           # Maximum position size decrease from streak
        }
        
        # Update with provided parameters
        self.params = self.default_params.copy()
        if params:
            self.params.update(params)
            
        self.name = "Advanced Multi-Timeframe Strategy"
        self.description = "Multi-timeframe analysis with trend alignment and momentum confirmation"
        
        # State variables
        self.position = 0
        self.entry_price = 0
        self.stop_loss = 0
        self.take_profit = 0
        self.trailing_stop_price = None
        self.active_signals = {}  # Signals by timeframe
        self.entry_signal_time = None
        self.current_timeframe_data = {}  # Store data for each timeframe
        self.timeframe_trends = {}  # Current trend for each timeframe
        self.partial_exits_taken = []  # Track partial profit exits
        
        # Performance tracking
        self.trades = []
        self.win_streak = 0
        self.loss_streak = 0
        self.wins = 0
        self.losses = 0
        self.realized_pnl = 0
        self.adaptive_tf_weights = self.params["tf_weights"].copy()
        
        logging.debug(f"Initialized {self.name} with parameters: {self.params}")
        
    def generate_signals(self, data_dict: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        Generate trading signals based on multi-timeframe analysis.
        
        Args:
            data_dict: Dictionary of DataFrames with OHLCV data for each timeframe
            
        Returns:
            DataFrame with signals (1 for buy, -1 for sell, 0 for hold)
        """
        try:
            # Verify input data
            self._validate_input_data(data_dict)
            
            # Get primary timeframe for signal generation
            primary_tf = self.params["timeframes"][1]  # Usually 1h
            primary_data = data_dict[primary_tf]
            
            # Create signals DataFrame based on primary timeframe
            signals = pd.DataFrame(index=primary_data.index)
            signals['position'] = 0
            
            # Update timeframe data cache
            self.current_timeframe_data = data_dict
            
            # Calculate indicators for all timeframes if not present
            for tf, df in data_dict.items():
                data_dict[tf] = self._calculate_indicators(df, tf)
                
            # Detect trend for each timeframe
            for tf in self.params["timeframes"]:
                if tf in data_dict:
                    self.timeframe_trends[tf] = self._detect_trend(data_dict[tf], tf)
            
            # Analyze trend alignment across timeframes
            trend_alignment, aligned_direction = self._analyze_trend_alignment()
            
            # Track previous position for detecting changes
            previous_position = self.position
            
            # Generate signals for each bar in primary timeframe
            for i in range(len(primary_data)):
                current_bar = primary_data.iloc[i]
                current_time = primary_data.index[i]
                
                # Skip if not enough data
                if i < 50:  # Need sufficient history for indicators
                    continue
                
                # Check current trend alignment at this bar
                current_alignment, direction = self._analyze_trend_alignment_at_index(data_dict, i)
                
                # Generate entry signals
                entry_signal = 0
                if current_alignment >= self.params["min_timeframe_alignment"]:
                    # Check momentum confirmation if required
                    momentum_confirmed = not self.params["require_momentum_confirmation"] or \
                                        self._confirm_momentum(data_dict, i, direction)
                    
                    if momentum_confirmed:
                        entry_signal = 1 if direction == "bullish" else -1
                        
                        # Set entry signal time
                        if self.entry_signal_time is None and self.position == 0:
                            self.entry_signal_time = current_time
                            
                # Process entry signals
                if entry_signal != 0 and self.position == 0:
                    # Check if entry signal is still valid (not timed out)
                    if self.entry_signal_time is not None:
                        # Calculate bars since entry signal
                        bars_since_signal = (current_time - self.entry_signal_time).total_seconds() / \
                                          self._get_timeframe_seconds(primary_tf)
                        
                        if bars_since_signal <= self.params["entry_timeout"]:
                            # Valid entry signal
                            signals.iloc[i]['position'] = entry_signal
                            self.position = entry_signal
                            self.entry_price = current_bar['close']
                            
                            # Calculate stop loss and take profit
                            self._calculate_exit_levels(data_dict, i)
                            
                            logging.info(f"New position: {self.position} at price {self.entry_price}, "
                                       f"Stop: {self.stop_loss}, Target: {self.take_profit}")
                        else:
                            # Entry signal timed out
                            self.entry_signal_time = None
                
                # Process exits for existing positions
                elif self.position != 0:
                    # Check for exit conditions
                    exit_signal = self._check_exit_conditions(data_dict, i)
                    
                    if exit_signal:
                        signals.iloc[i]['position'] = -self.position  # Reverse position to exit
                        
                        # Record trade result
                        exit_price = current_bar['close']
                        pnl = (exit_price - self.entry_price) * self.position
                        pnl_pct = pnl / self.entry_price
                        
                        trade_result = {
                            'entry_time': self.entry_signal_time,
                            'exit_time': current_time,
                            'entry_price': self.entry_price,
                            'exit_price': exit_price,
                            'position': self.position,
                            'pnl': pnl,
                            'pnl_pct': pnl_pct,
                            'exit_reason': exit_signal
                        }
                        
                        self.trades.append(trade_result)
                        self.realized_pnl += pnl_pct
                        
                        # Update streaks
                        if pnl_pct > 0:
                            self.wins += 1
                            self.win_streak += 1
                            self.loss_streak = 0
                        else:
                            self.losses += 1
                            self.loss_streak += 1
                            self.win_streak = 0
                            
                        # Update adaptive timeframe weights if enabled
                        if self.params["adaptive_weights"] and len(self.trades) >= 5:
                            self._update_timeframe_weights()
                        
                        # Reset position tracking
                        self.position = 0
                        self.entry_price = 0
                        self.stop_loss = 0
                        self.take_profit = 0
                        self.trailing_stop_price = None
                        self.entry_signal_time = None
                        self.partial_exits_taken = []
                        
                    else:
                        # Check for partial profit taking
                        if self.params["preserve_profits"]:
                            self._check_partial_profit_taking(signals, i, current_bar)
                            
                        # Update trailing stop if enabled
                        if self.params["trailing_stop"] and self.trailing_stop_price is not None:
                            self._update_trailing_stop(current_bar)
            
            return signals
            
        except Exception as e:
            logging.error(f"Error generating multi-timeframe signals: {str(e)}")
            import traceback
            logging.error(traceback.format_exc())
            
            # Return empty signals DataFrame
            if primary_tf in data_dict:
                return pd.DataFrame(index=data_dict[primary_tf].index, columns=['position'])
            else:
                return pd.DataFrame(columns=['position'])
                
    def _validate_input_data(self, data_dict: Dict[str, pd.DataFrame]) -> None:
        """Validate input data for each timeframe."""
        required_timeframes = set(self.params["timeframes"])
        provided_timeframes = set(data_dict.keys())
        
        # Check that at least 2 timeframes are available
        if len(provided_timeframes) < 2:
            raise ValueError(f"At least 2 timeframes required, got {len(provided_timeframes)}")
            
        # Warn about missing timeframes
        missing_timeframes = required_timeframes - provided_timeframes
        if missing_timeframes:
            logging.warning(f"Missing data for timeframes: {missing_timeframes}")
            
        # Check that each DataFrame has required columns
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        for tf, df in data_dict.items():
            missing_columns = set(required_columns) - set(df.columns)
            if missing_columns:
                raise ValueError(f"Missing columns {missing_columns} in timeframe {tf}")
                
    def _calculate_indicators(self, df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
        """Calculate technical indicators for a timeframe."""
        # Create copy to avoid modifying original
        data = df.copy()
        
        # Get parameters for this timeframe
        fast_period = self.params["fast_ma_periods"].get(timeframe, 9)
        slow_period = self.params["slow_ma_periods"].get(timeframe, 21)
        trend_period = self.params["trend_ma_periods"].get(timeframe, 50)
        ma_type = self.params["ma_type"]
        
        # Calculate moving averages
        if ma_type == "sma":
            data[f'fast_ma'] = data['close'].rolling(window=fast_period).mean()
            data[f'slow_ma'] = data['close'].rolling(window=slow_period).mean()
            data[f'trend_ma'] = data['close'].rolling(window=trend_period).mean()
        elif ma_type == "ema":
            data[f'fast_ma'] = data['close'].ewm(span=fast_period, adjust=False).mean()
            data[f'slow_ma'] = data['close'].ewm(span=slow_period, adjust=False).mean()
            data[f'trend_ma'] = data['close'].ewm(span=trend_period, adjust=False).mean()
        elif ma_type == "wma":
            # Weighted moving average
            weights = np.arange(1, fast_period + 1)
            data[f'fast_ma'] = data['close'].rolling(window=fast_period).apply(
                lambda x: np.sum(weights * x) / weights.sum(), raw=True)
                
            weights = np.arange(1, slow_period + 1)
            data[f'slow_ma'] = data['close'].rolling(window=slow_period).apply(
                lambda x: np.sum(weights * x) / weights.sum(), raw=True)
                
            weights = np.arange(1, trend_period + 1)
            data[f'trend_ma'] = data['close'].rolling(window=trend_period).apply(
                lambda x: np.sum(weights * x) / weights.sum(), raw=True)
        
        # Calculate RSI
        delta = data['close'].diff()
        gain = delta.where(delta > 0, 0).rolling(window=self.params["rsi_period"]).mean()
        loss = -delta.where(delta < 0, 0).rolling(window=self.params["rsi_period"]).mean()
        rs = gain / loss
        data['rsi'] = 100 - (100 / (1 + rs))
        
        # Calculate MACD
        macd_fast = data['close'].ewm(span=self.params["macd_fast"], adjust=False).mean()
        macd_slow = data['close'].ewm(span=self.params["macd_slow"], adjust=False).mean()
        data['macd'] = macd_fast - macd_slow
        data['macd_signal'] = data['macd'].ewm(span=self.params["macd_signal"], adjust=False).mean()
        data['macd_hist'] = data['macd'] - data['macd_signal']
        
        # Calculate ATR
        high_low = data['high'] - data['low']
        high_close = abs(data['high'] - data['close'].shift())
        low_close = abs(data['low'] - data['close'].shift())
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        data['atr'] = tr.rolling(window=14).mean()
        
        # Calculate additional momentum indicators
        # Rate of Change (ROC)
        data['roc'] = data['close'].pct_change(10) * 100
        
        # Average Directional Index (ADX) for trend strength
        plus_dm = data['high'].diff()
        minus_dm = data['low'].shift().diff(-1)
        plus_dm[plus_dm < 0] = 0
        minus_dm[minus_dm < 0] = 0
        
        tr14 = tr.rolling(window=14).sum()
        plus_di14 = 100 * (plus_dm.rolling(window=14).sum() / tr14)
        minus_di14 = abs(100 * (minus_dm.rolling(window=14).sum() / tr14))
        dx = 100 * abs(plus_di14 - minus_di14) / (plus_di14 + minus_di14)
        data['adx'] = dx.rolling(window=14).mean()
        data['plus_di'] = plus_di14
        data['minus_di'] = minus_di14
        
        return data
        
    def _detect_trend(self, df: pd.DataFrame, timeframe: str) -> Dict[str, Any]:
        """
        Detect trend direction and strength for a timeframe.
        
        Returns:
            Dict with trend information including direction and strength
        """
        if df.empty or len(df) < 50:
            return {"direction": "neutral", "strength": 0, "duration": 0}
            
        # Get latest values for trend analysis
        latest = df.iloc[-1]
        
        # Determine trend based on moving average relationships
        price_above_fast = latest['close'] > latest['fast_ma']
        price_above_slow = latest['close'] > latest['slow_ma']
        price_above_trend = latest['close'] > latest['trend_ma']
        fast_above_slow = latest['fast_ma'] > latest['slow_ma']
        fast_above_trend = latest['fast_ma'] > latest['trend_ma']
        slow_above_trend = latest['slow_ma'] > latest['trend_ma']
        
        # Count bullish factors
        bullish_factors = sum([
            price_above_fast, 
            price_above_slow, 
            price_above_trend,
            fast_above_slow,
            fast_above_trend,
            slow_above_trend
        ])
        
        # Determine trend direction
        if bullish_factors >= 5:
            direction = "bullish"
        elif bullish_factors <= 1:
            direction = "bearish"
        else:
            direction = "neutral"
            
        # Calculate trend strength using ADX
        strength = latest['adx'] / 100.0  # Normalize to 0-1 range
        
        # Calculate trend duration
        duration = 0
        current_direction = direction
        for i in range(len(df) - 2, 0, -1):
            bar = df.iloc[i]
            
            # Simplified trend check for duration calculation
            bar_direction = "neutral"
            if bar['fast_ma'] > bar['slow_ma'] and bar['slow_ma'] > bar['trend_ma']:
                bar_direction = "bullish"
            elif bar['fast_ma'] < bar['slow_ma'] and bar['slow_ma'] < bar['trend_ma']:
                bar_direction = "bearish"
                
            if bar_direction == current_direction:
                duration += 1
            else:
                break
                
        # Calculate additional momentum characteristics
        momentum = 0
        
        # RSI contribution to momentum
        if direction == "bullish":
            rsi_momentum = (latest['rsi'] - 50) / 50.0  # 0 to 1 for bullish
            momentum += max(0, rsi_momentum)
        else:
            rsi_momentum = (50 - latest['rsi']) / 50.0  # 0 to 1 for bearish
            momentum += max(0, rsi_momentum)
            
        # MACD contribution to momentum
        macd_strength = abs(latest['macd_hist']) / (abs(latest['macd']) + 1e-10)
        if (direction == "bullish" and latest['macd_hist'] > 0) or \
           (direction == "bearish" and latest['macd_hist'] < 0):
            momentum += macd_strength
            
        # ROC contribution to momentum
        roc_norm = latest['roc'] / 10.0  # Normalize ROC
        if (direction == "bullish" and roc_norm > 0) or \
           (direction == "bearish" and roc_norm < 0):
            momentum += min(1.0, abs(roc_norm))
            
        # Normalize momentum to 0-1
        momentum = min(1.0, momentum / 3.0)
        
        return {
            "direction": direction,
            "strength": strength,
            "duration": duration,
            "momentum": momentum,
            "price_above_fast": price_above_fast,
            "price_above_slow": price_above_slow,
            "price_above_trend": price_above_trend,
            "fast_above_slow": fast_above_slow,
            "slow_above_trend": slow_above_trend,
            "adx": latest['adx'],
            "rsi": latest['rsi'],
            "macd_hist": latest['macd_hist']
        }
        
    def _analyze_trend_alignment(self) -> Tuple[int, str]:
        """
        Analyze trend alignment across timeframes.
        
        Returns:
            Tuple of (number of aligned timeframes, dominant trend direction)
        """
        if not self.timeframe_trends:
            return 0, "neutral"
            
        # Count trends by direction
        bullish_count = 0
        bearish_count = 0
        neutral_count = 0
        
        # Calculate weighted vote for each timeframe
        bullish_weight = 0
        bearish_weight = 0
        
        for i, tf in enumerate(self.params["timeframes"]):
            if tf not in self.timeframe_trends:
                continue
                
            trend = self.timeframe_trends[tf]
            weight = self.adaptive_tf_weights[i] if i < len(self.adaptive_tf_weights) else 0.1
            
            if trend["direction"] == "bullish":
                bullish_count += 1
                bullish_weight += weight * (0.5 + 0.5 * trend["strength"])
            elif trend["direction"] == "bearish":
                bearish_count += 1
                bearish_weight += weight * (0.5 + 0.5 * trend["strength"])
            else:
                neutral_count += 1
                
        # Determine dominant trend
        if bullish_weight > bearish_weight * 1.5:  # Require significant bullish dominance
            dominant_trend = "bullish"
            aligned_count = bullish_count
        elif bearish_weight > bullish_weight * 1.5:  # Require significant bearish dominance
            dominant_trend = "bearish"
            aligned_count = bearish_count
        else:
            dominant_trend = "neutral"
            aligned_count = 0
            
        return aligned_count, dominant_trend
        
    def _analyze_trend_alignment_at_index(self, data_dict: Dict[str, pd.DataFrame], 
                                         index: int) -> Tuple[int, str]:
        """
        Analyze trend alignment across timeframes at a specific index.
        
        Args:
            data_dict: Dictionary of DataFrames with indicator data
            index: Index position in primary timeframe
            
        Returns:
            Tuple of (number of aligned timeframes, dominant trend direction)
        """
        # Get primary timeframe
        primary_tf = self.params["timeframes"][1]  # Usually 1h
        
        if primary_tf not in data_dict:
            primary_tf = list(data_dict.keys())[0]
            
        primary_time = data_dict[primary_tf].index[index]
        
        # Count trends by direction
        bullish_count = 0
        bearish_count = 0
        neutral_count = 0
        
        # Calculate weighted vote for each timeframe
        bullish_weight = 0
        bearish_weight = 0
        
        for i, tf in enumerate(self.params["timeframes"]):
            if tf not in data_dict:
                continue
                
            # Find the corresponding bar in this timeframe
            tf_data = data_dict[tf]
            
            # Find the last bar in this timeframe that is before or at the primary time
            tf_index = tf_data.index.get_indexer([primary_time], method='pad')[0]
            
            if tf_index < 0 or tf_index >= len(tf_data):
                continue
                
            # Get bar data
            bar = tf_data.iloc[tf_index]
            
            # Determine trend
            trend_direction = "neutral"
            trend_strength = 0
            
            # Simplified trend check
            price_above_fast = bar['close'] > bar['fast_ma']
            price_above_slow = bar['close'] > bar['slow_ma']
            price_above_trend = bar['close'] > bar['trend_ma']
            fast_above_slow = bar['fast_ma'] > bar['slow_ma']
            fast_above_trend = bar['fast_ma'] > bar['trend_ma']
            slow_above_trend = bar['slow_ma'] > bar['trend_ma']
            
            # Count bullish factors
            bullish_factors = sum([
                price_above_fast, 
                price_above_slow, 
                price_above_trend,
                fast_above_slow,
                fast_above_trend,
                slow_above_trend
            ])
            
            # Determine trend direction
            if bullish_factors >= 5:
                trend_direction = "bullish"
            elif bullish_factors <= 1:
                trend_direction = "bearish"
                
            # Get trend strength from ADX
            if 'adx' in bar:
                trend_strength = bar['adx'] / 100.0
            
            # Add to counts
            weight = self.adaptive_tf_weights[i] if i < len(self.adaptive_tf_weights) else 0.1
            
            if trend_direction == "bullish":
                bullish_count += 1
                bullish_weight += weight * (0.5 + 0.5 * trend_strength)
            elif trend_direction == "bearish":
                bearish_count += 1
                bearish_weight += weight * (0.5 + 0.5 * trend_strength)
            else:
                neutral_count += 1
                
        # Determine dominant trend
        if bullish_weight > bearish_weight * 1.5:  # Require significant bullish dominance
            dominant_trend = "bullish"
            aligned_count = bullish_count
        elif bearish_weight > bullish_weight * 1.5:  # Require significant bearish dominance
            dominant_trend = "bearish"
            aligned_count = bearish_count
        else:
            dominant_trend = "neutral"
            aligned_count = 0
            
        return aligned_count, dominant_trend
        
    def _confirm_momentum(self, data_dict: Dict[str, pd.DataFrame], index: int, 
                         direction: str) -> bool:
        """
        Confirm trend with momentum indicators.
        
        Args:
            data_dict: Dictionary of DataFrames with indicator data
            index: Index position in primary timeframe
            direction: Expected trend direction
            
        Returns:
            bool: True if momentum confirms the trend direction
        """
        # Get primary timeframe
        primary_tf = self.params["timeframes"][1]  # Usually 1h
        
        if primary_tf not in data_dict:
            primary_tf = list(data_dict.keys())[0]
            
        primary_data = data_dict[primary_tf]
        current_bar = primary_data.iloc[index]
        
        # Check momentum indicators
        rsi_confirms = False
        macd_confirms = False
        
        # RSI confirmation
        if direction == "bullish":
            # Bullish momentum criteria: RSI above 50 and rising
            if current_bar['rsi'] > 50 and current_bar['rsi'] > primary_data.iloc[index-1]['rsi']:
                rsi_confirms = True
        else:  # bearish
            # Bearish momentum criteria: RSI below 50 and falling
            if current_bar['rsi'] < 50 and current_bar['rsi'] < primary_data.iloc[index-1]['rsi']:
                rsi_confirms = True
                
        # MACD confirmation
        if direction == "bullish":
            # Bullish momentum criteria: MACD histogram positive and increasing
            if current_bar['macd_hist'] > 0 and current_bar['macd_hist'] > primary_data.iloc[index-1]['macd_hist']:
                macd_confirms = True
        else:  # bearish
            # Bearish momentum criteria: MACD histogram negative and decreasing
            if current_bar['macd_hist'] < 0 and current_bar['macd_hist'] < primary_data.iloc[index-1]['macd_hist']:
                macd_confirms = True
                
        # Require at least one momentum indicator to confirm
        return rsi_confirms or macd_confirms
        
    def _calculate_exit_levels(self, data_dict: Dict[str, pd.DataFrame], index: int) -> None:
        """Calculate stop loss and take profit levels."""
        # Get stop loss timeframe
        sl_timeframe = self.params["stop_loss_timeframe"]
        
        if sl_timeframe not in data_dict:
            sl_timeframe = self.params["timeframes"][1]  # Default to 1h
            
        if sl_timeframe not in data_dict:
            sl_timeframe = list(data_dict.keys())[0]
            
        # Get ATR from the stop loss timeframe
        sl_data = data_dict[sl_timeframe]
        
        # Find the corresponding index in the stop loss timeframe
        primary_tf = self.params["timeframes"][1]
        primary_time = data_dict[primary_tf].index[index]
        
        sl_index = sl_data.index.get_indexer([primary_time], method='pad')[0]
        
        if sl_index < 0 or sl_index >= len(sl_data):
            # Fallback to primary timeframe
            atr = data_dict[primary_tf].iloc[index]['atr']
        else:
            atr = sl_data.iloc[sl_index]['atr']
            
        # Calculate stop loss
        if self.position > 0:  # Long position
            self.stop_loss = self.entry_price - (atr * self.params["stop_loss_atr"])
            self.take_profit = self.entry_price + (atr * self.params["take_profit_atr"])
        else:  # Short position
            self.stop_loss = self.entry_price + (atr * self.params["stop_loss_atr"])
            self.take_profit = self.entry_price - (atr * self.params["take_profit_atr"])
            
        # Initialize trailing stop at the same level as stop loss
        self.trailing_stop_price = self.stop_loss
        
    def _check_exit_conditions(self, data_dict: Dict[str, pd.DataFrame], index: int) -> Optional[str]:
        """
        Check for exit conditions.
        
        Args:
            data_dict: Dictionary of DataFrames with indicator data
            index: Index position in primary timeframe
            
        Returns:
            str: Exit reason if should exit, None otherwise
        """
        # Get primary timeframe
        primary_tf = self.params["timeframes"][1]  # Usually 1h
        
        if primary_tf not in data_dict:
            primary_tf = list(data_dict.keys())[0]
            
        primary_data = data_dict[primary_tf]
        current_bar = primary_data.iloc[index]
        
        # Check stop loss
        if self.position > 0:  # Long position
            if current_bar['low'] <= self.trailing_stop_price:
                return "stop_loss"
        else:  # Short position
            if current_bar['high'] >= self.trailing_stop_price:
                return "stop_loss"
                
        # Check take profit
        if self.position > 0:  # Long position
            if current_bar['high'] >= self.take_profit:
                return "take_profit"
        else:  # Short position
            if current_bar['low'] <= self.take_profit:
                return "take_profit"
                
        # Check trend change exit on higher timeframes if enabled
        if self.params["exit_on_trend_change"]:
            # Focus on the higher timeframes
            higher_tfs = self.params["timeframes"][2:]  # Usually 4h and 1d
            
            for tf in higher_tfs:
                if tf not in data_dict:
                    continue
                    
                # Find the corresponding bar in this timeframe
                tf_data = data_dict[tf]
                primary_time = primary_data.index[index]
                
                tf_index = tf_data.index.get_indexer([primary_time], method='pad')[0]
                
                if tf_index < 1 or tf_index >= len(tf_data):
                    continue
                    
                # Get current and previous trend
                current_trend = self._detect_trend_at_index(tf_data, tf_index)
                prev_trend = self._detect_trend_at_index(tf_data, tf_index - 1)
                
                # Check for trend change
                if current_trend["direction"] != prev_trend["direction"]:
                    # Exit if trend changed to opposite direction
                    if self.position > 0 and current_trend["direction"] == "bearish":
                        return f"trend_change_{tf}"
                    elif self.position < 0 and current_trend["direction"] == "bullish":
                        return f"trend_change_{tf}"
                        
        # Check timeframe divergence exit if enabled
        if self.params["timeframe_divergence_exit"]:
            # Check if timeframes are no longer aligned
            alignment, direction = self._analyze_trend_alignment_at_index(data_dict, index)
            
            if alignment < self.params["min_timeframe_alignment"]:
                return "timeframe_divergence"
                
            # Also exit if dominant trend direction opposes our position
            if self.position > 0 and direction == "bearish":
                return "opposing_trend"
            elif self.position < 0 and direction == "bullish":
                return "opposing_trend"
                
        return None
        
    def _detect_trend_at_index(self, df: pd.DataFrame, index: int) -> Dict[str, Any]:
        """Detect trend at a specific index."""
        if index < 0 or index >= len(df):
            return {"direction": "neutral", "strength": 0}
            
        # Get bar data
        bar = df.iloc[index]
        
        # Determine trend based on moving average relationships
        price_above_fast = bar['close'] > bar['fast_ma']
        price_above_slow = bar['close'] > bar['slow_ma']
        price_above_trend = bar['close'] > bar['trend_ma']
        fast_above_slow = bar['fast_ma'] > bar['slow_ma']
        fast_above_trend = bar['fast_ma'] > bar['trend_ma']
        slow_above_trend = bar['slow_ma'] > bar['trend_ma']
        
        # Count bullish factors
        bullish_factors = sum([
            price_above_fast, 
            price_above_slow, 
            price_above_trend,
            fast_above_slow,
            fast_above_trend,
            slow_above_trend
        ])
        
        # Determine trend direction
        if bullish_factors >= 5:
            direction = "bullish"
        elif bullish_factors <= 1:
            direction = "bearish"
        else:
            direction = "neutral"
            
        # Calculate trend strength using ADX
        strength = bar['adx'] / 100.0 if 'adx' in bar else 0.5
        
        return {
            "direction": direction,
            "strength": strength,
            "adx": bar['adx'] if 'adx' in bar else 0,
            "rsi": bar['rsi'] if 'rsi' in bar else 50
        }
        
    def _update_trailing_stop(self, current_bar: pd.Series) -> None:
        """Update trailing stop price if profit exceeds activation threshold."""
        current_price = current_bar['close']
        profit = 0
        
        if self.position > 0:  # Long position
            profit = (current_price - self.entry_price) / self.entry_price
            
            # Check if profit exceeds activation threshold
            if profit >= self.params["trailing_activation"]:
                # Calculate new potential stop level
                atr = current_bar['atr'] if 'atr' in current_bar else 0
                potential_stop = current_price - (atr * self.params["trailing_distance"])
                
                # Update if new stop is higher than current stop
                if potential_stop > self.trailing_stop_price:
                    self.trailing_stop_price = potential_stop
                    logging.debug(f"Updated trailing stop to {self.trailing_stop_price}")
                    
        elif self.position < 0:  # Short position
            profit = (self.entry_price - current_price) / self.entry_price
            
            # Check if profit exceeds activation threshold
            if profit >= self.params["trailing_activation"]:
                # Calculate new potential stop level
                atr = current_bar['atr'] if 'atr' in current_bar else 0
                potential_stop = current_price + (atr * self.params["trailing_distance"])
                
                # Update if new stop is lower than current stop
                if potential_stop < self.trailing_stop_price:
                    self.trailing_stop_price = potential_stop
                    logging.debug(f"Updated trailing stop to {self.trailing_stop_price}")
                    
    def _check_partial_profit_taking(self, signals: pd.DataFrame, index: int, 
                                   current_bar: pd.Series) -> None:
        """Check for partial profit taking opportunities."""
        current_price = current_bar['close']
        
        # Calculate current profit percentage
        if self.position > 0:  # Long position
            profit_pct = (current_price - self.entry_price) / self.entry_price
        else:  # Short position
            profit_pct = (self.entry_price - current_price) / self.entry_price
            
        # Check profit levels
        for i, target in enumerate(self.params["profit_taking_levels"]):
            # Skip if already taken profit at this level
            if i in self.partial_exits_taken:
                continue
                
            # Take partial profit if threshold reached
            if profit_pct >= target:
                # Percentage of position to exit
                exit_percentage = self.params["profit_taking_percentages"][i]
                
                # Generate partial exit signal
                signals.iloc[index]['position'] = -self.position * exit_percentage
                
                # Register partial exit
                self.partial_exits_taken.append(i)
                
                # Reduce position size
                self.position = self.position * (1 - exit_percentage)
                
                logging.info(f"Partial profit taking ({exit_percentage:.1%}) at {profit_pct:.2%} profit, "
                           f"remaining position: {self.position}")
                break  # Only take one partial exit per bar
                
    def _update_timeframe_weights(self) -> None:
        """Update adaptive timeframe weights based on performance."""
        # Need at least a few trades for analysis
        if len(self.trades) < 5:
            return
            
        # Analyze recent trades
        recent_trades = self.trades[-min(len(self.trades), 10):]
        
        # Calculate win rate by timeframe
        tf_performance = {}
        
        for tf in self.params["timeframes"]:
            tf_performance[tf] = {
                'wins': 0,
                'losses': 0,
                'win_rate': 0,
                'total_pnl': 0
            }
            
        # Attribute trades to timeframes that triggered them
        for trade in recent_trades:
            # Skip trades without exit reason
            if 'exit_reason' not in trade:
                continue
                
            # Extract timeframe from exit reason if present
            tf = None
            for exit_tf in self.params["timeframes"]:
                if f"trend_change_{exit_tf}" in trade['exit_reason']:
                    tf = exit_tf
                    break
                    
            # If no specific timeframe identified, use primary timeframe
            if tf is None:
                tf = self.params["timeframes"][1]  # Usually 1h
                
            # Record trade result
            if trade['pnl_pct'] > 0:
                tf_performance[tf]['wins'] += 1
            else:
                tf_performance[tf]['losses'] += 1
                
            tf_performance[tf]['total_pnl'] += trade['pnl_pct']
            
        # Calculate win rates and normalize to weights
        total_performance = 0
        
        for tf in self.params["timeframes"]:
            if tf not in tf_performance:
                continue
                
            perf = tf_performance[tf]
            total_trades = perf['wins'] + perf['losses']
            
            if total_trades > 0:
                perf['win_rate'] = perf['wins'] / total_trades
                
                # Use win rate and average PnL for performance score
                if perf['total_pnl'] != 0:
                    avg_pnl = perf['total_pnl'] / total_trades
                    perf['score'] = perf['win_rate'] * (1 + abs(avg_pnl) * 10)
                else:
                    perf['score'] = perf['win_rate']
                    
                total_performance += perf['score']
            else:
                perf['win_rate'] = 0
                perf['score'] = 0
                
        # Convert scores to weights
        if total_performance > 0:
            for i, tf in enumerate(self.params["timeframes"]):
                if tf not in tf_performance:
                    continue
                    
                # Calculate weight (min 10%, max 50%)
                weight = 0.1 + 0.4 * (tf_performance[tf]['score'] / total_performance)
                
                # Update weight if different enough
                if i < len(self.adaptive_tf_weights) and abs(weight - self.adaptive_tf_weights[i]) > 0.05:
                    self.adaptive_tf_weights[i] = weight
                    
            # Normalize weights to sum to 1
            total_weight = sum(self.adaptive_tf_weights)
            self.adaptive_tf_weights = [w / total_weight for w in self.adaptive_tf_weights]
            
            logging.info(f"Updated timeframe weights: {self.adaptive_tf_weights}")
            
    def calculate_position_size(self, account_balance: float, current_price: float) -> float:
        """
        Calculate optimal position size based on risk parameters.
        
        Args:
            account_balance: Current account balance
            current_price: Current asset price
            
        Returns:
            float: Position size in base currency
        """
        # Base risk based on account balance
        risk_amount = account_balance * self.params["risk_per_trade"]
        
        # Calculate stop loss distance
        if not self.stop_loss or not self.entry_price:
            # Default to 1.5% if no stop loss calculated yet
            stop_distance = current_price * 0.015
        else:
            stop_distance = abs(self.entry_price - self.stop_loss)
            
        # Calculate position size
        if stop_distance > 0:
            position_size = risk_amount / stop_distance
        else:
            # Fallback to a safe default
            position_size = risk_amount / (current_price * 0.015)
            
        # Apply streak adjustments
        streak_factor = 1.0
        
        if self.win_streak > 0:
            # Increase size on winning streaks
            streak_increase = min(self.win_streak * self.params["win_streak_factor"], 
                               self.params["max_win_streak_increase"])
            streak_factor = 1.0 + streak_increase
        elif self.loss_streak > 0:
            # Decrease size on losing streaks
            streak_decrease = min(self.loss_streak * self.params["loss_streak_factor"], 
                               self.params["max_loss_streak_decrease"])
            streak_factor = max(0.3, 1.0 - streak_decrease)
            
        # Apply adjustment
        adjusted_size = position_size * streak_factor
        
        # Limit to a reasonable percentage of account
        return min(adjusted_size, account_balance * 0.5 / current_price)
        
    def _get_timeframe_seconds(self, timeframe: str) -> int:
        """Convert timeframe string to seconds."""
        if not timeframe:
            return 3600  # Default to 1h
            
        # Extract number and unit
        import re
        match = re.match(r'(\d+)([mhdw])', timeframe)
        
        if not match:
            return 3600  # Default to 1h
            
        num, unit = match.groups()
        num = int(num)
        
        # Convert to seconds
        if unit == 'm':
            return num * 60
        elif unit == 'h':
            return num * 3600
        elif unit == 'd':
            return num * 86400
        elif unit == 'w':
            return num * 604800
        else:
            return 3600  # Default to 1h
            
    def update_params(self, new_params: Dict[str, Any]) -> None:
        """
        Update strategy parameters.
        
        Args:
            new_params: New parameter values
        """
        self.params.update(new_params)
        logging.info(f"Updated multi-timeframe strategy parameters: {new_params}")
        
    def get_params(self) -> Dict[str, Any]:
        """Get current strategy parameters."""
        return self.params.copy()
        
    def get_info(self) -> Dict[str, Any]:
        """Get strategy information including current state."""
        # Calculate performance metrics
        win_rate = self.wins / (self.wins + self.losses) if (self.wins + self.losses) > 0 else 0
        
        return {
            "name": self.name,
            "description": self.description,
            "position": self.position,
            "entry_price": self.entry_price,
            "stop_loss": self.stop_loss,
            "take_profit": self.take_profit,
            "trailing_stop": self.trailing_stop_price,
            "trades_count": len(self.trades),
            "win_rate": win_rate,
            "win_streak": self.win_streak,
            "loss_streak": self.loss_streak,
            "realized_pnl": self.realized_pnl,
            "params": self.params,
            "adaptive_weights": self.adaptive_tf_weights
        }
        
    def get_performance(self) -> Dict[str, Any]:
        """Get detailed performance statistics."""
        if not self.trades:
            return {
                "trades_count": 0,
                "win_rate": 0,
                "avg_win": 0,
                "avg_loss": 0,
                "profit_factor": 0,
                "realized_pnl": 0
            }
            
        # Calculate performance metrics
        wins = [t['pnl_pct'] for t in self.trades if t['pnl_pct'] > 0]
        losses = [t['pnl_pct'] for t in self.trades if t['pnl_pct'] <= 0]
        
        win_count = len(wins)
        loss_count = len(losses)
        total_count = len(self.trades)
        
        win_rate = win_count / total_count if total_count > 0 else 0
        
        avg_win = sum(wins) / win_count if win_count > 0 else 0
        avg_loss = sum(losses) / loss_count if loss_count > 0 else 0
        
        # Calculate profit factor
        total_wins = sum(wins) if wins else 0
        total_losses = abs(sum(losses)) if losses else 0
        profit_factor = total_wins / total_losses if total_losses else float('inf')
        
        return {
            "trades_count": total_count,
            "win_count": win_count,
            "loss_count": loss_count,
            "win_rate": win_rate,
            "avg_win": avg_win,
            "avg_loss": avg_loss,
            "profit_factor": profit_factor,
            "realized_pnl": self.realized_pnl,
            "win_streak": self.win_streak,
            "loss_streak": self.loss_streak,
            "adaptive_weights": self.adaptive_tf_weights
        }

# Module functions for strategy creation and registration
def create_strategy(params=None):
    """Factory function to create a multi-timeframe strategy instance."""
    return MultiTimeframeStrategy(params)
