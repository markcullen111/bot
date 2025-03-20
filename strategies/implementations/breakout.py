# strategies/implementations/breakout.py

import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Optional, Union, Tuple, Any

class BreakoutStrategy:
    """
    Advanced breakout strategy with multiple confirmation filters and adaptive parameters.
    
    This strategy:
    1. Identifies key support/resistance levels using multi-timeframe analysis
    2. Detects breakouts with volume and momentum confirmation
    3. Uses volatility-based position sizing and risk management
    4. Adapts parameters to current market regime (trending/ranging/volatile)
    5. Implements smart entry/exit logic with trailing stops
    """
    
    def __init__(self, params: Optional[Dict[str, Any]] = None):
        """
        Initialize the breakout strategy with configurable parameters.
        
        Args:
            params: Strategy parameters dictionary
        """
        # Default parameters
        self.default_params = {
            # Breakout detection
            "lookback_period": 20,        # Period for identifying key levels
            "breakout_threshold": 0.005,  # Min breakout size (% of price)
            "consolidation_periods": 5,   # Min periods of consolidation before breakout
            
            # Confirmation filters
            "volume_multiplier": 1.5,     # Volume increase required for confirmation
            "momentum_threshold": 0.3,    # Minimum RSI change for momentum confirmation
            "false_breakout_guard": True, # Enable protection against false breakouts
            
            # Market regime filters
            "trending_threshold": 0.6,    # ADX threshold for trending market
            "volatility_filter": True,    # Enable volatility-based filtering
            "max_volatility": 0.04,       # Maximum allowed ATR/price for entry
            
            # Risk management
            "risk_per_trade": 0.02,       # Risk per trade (% of portfolio)
            "stop_loss_atr": 1.5,         # Stop loss distance in ATR
            "take_profit_atr": 3.0,       # Take profit distance in ATR
            "trailing_stop": True,        # Enable trailing stop
            "trailing_distance": 2.0,     # Trailing stop distance in ATR
            
            # Exit parameters
            "profit_target": 0.05,        # Profit target (% of price)
            "max_holding_periods": 20,    # Maximum holding periods
            "exit_on_opposite": True      # Exit on opposite breakout signal
        }
        
        # Update with provided parameters
        self.params = self.default_params.copy()
        if params:
            self.params.update(params)
            
        self.name = "Advanced Breakout Strategy"
        self.description = "Multi-timeframe breakout detection with volume and momentum confirmation"
        self.position = 0
        self.entry_price = 0
        self.stop_loss = 0
        self.take_profit = 0
        self.holding_periods = 0
        self.key_levels = []
        self.last_signal = None
        self.trailing_stop_price = None
        
        logging.debug(f"Initialized {self.name} with parameters: {self.params}")
        
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Generate trading signals based on breakout strategy.
        
        Args:
            data: OHLCV DataFrame with additional indicators
            
        Returns:
            DataFrame with signals (1 for buy, -1 for sell, 0 for hold)
        """
        try:
            if data is None or data.empty or len(data) < self.params["lookback_period"] + 10:
                logging.warning("Insufficient data for breakout strategy")
                return pd.DataFrame(index=data.index if data is not None else [], columns=['position'])
                
            # Create signals DataFrame
            signals = pd.DataFrame(index=data.index)
            signals['position'] = 0
            
            # Calculate required indicators if not present
            data = self._calculate_indicators(data)
            
            # Identify key support and resistance levels
            self.key_levels = self._identify_key_levels(data)
            
            # Detect market regime
            market_regime = self._detect_market_regime(data)
            
            # Skip signals generation in extremely volatile markets if filter enabled
            if self.params["volatility_filter"] and market_regime == "extreme_volatility":
                logging.info("Skipping breakout signals due to extreme volatility")
                return signals
                
            # Detect breakouts
            for i in range(self.params["lookback_period"] + 10, len(data)):
                # Check current price against key levels
                current_bar = data.iloc[i]
                prev_bar = data.iloc[i-1]
                
                # Check for resistance breakout (buy signal)
                resistance_breakout = self._detect_resistance_breakout(data, i)
                
                # Check for support breakout (sell signal)
                support_breakout = self._detect_support_breakout(data, i)
                
                # Apply confirmation filters
                if resistance_breakout and self._confirm_breakout(data, i, "resistance"):
                    signals.iloc[i]['position'] = 1
                    self.last_signal = {"type": "buy", "price": current_bar['close'], "index": i}
                elif support_breakout and self._confirm_breakout(data, i, "support"):
                    signals.iloc[i]['position'] = -1
                    self.last_signal = {"type": "sell", "price": current_bar['close'], "index": i}
                else:
                    # Check for exit signals
                    if self.position != 0:
                        self.holding_periods += 1
                        
                        # Exit based on maximum holding periods
                        if self.holding_periods >= self.params["max_holding_periods"]:
                            signals.iloc[i]['position'] = -self.position  # Reverse position to exit
                            self.position = 0
                            self.holding_periods = 0
                            logging.info(f"Exit signal due to max holding periods at price {current_bar['close']}")
                            continue
                            
                        # Exit based on opposite breakout signal
                        if self.params["exit_on_opposite"]:
                            if self.position > 0 and support_breakout:
                                signals.iloc[i]['position'] = -1  # Sell to exit
                                self.position = 0
                                self.holding_periods = 0
                                logging.info(f"Exit long position due to support breakout at price {current_bar['close']}")
                            elif self.position < 0 and resistance_breakout:
                                signals.iloc[i]['position'] = 1  # Buy to exit
                                self.position = 0
                                self.holding_periods = 0
                                logging.info(f"Exit short position due to resistance breakout at price {current_bar['close']}")
                
                # Update position tracking
                if signals.iloc[i]['position'] != 0:
                    self.position = signals.iloc[i]['position']
                    self.entry_price = current_bar['close']
                    self.holding_periods = 0
                    
                    # Calculate stop loss and take profit levels
                    atr = current_bar['atr'] if 'atr' in current_bar else self._calculate_atr(data, i)
                    
                    if self.position > 0:  # Long position
                        self.stop_loss = self.entry_price - (atr * self.params["stop_loss_atr"])
                        self.take_profit = self.entry_price + (atr * self.params["take_profit_atr"])
                        self.trailing_stop_price = self.stop_loss
                    else:  # Short position
                        self.stop_loss = self.entry_price + (atr * self.params["stop_loss_atr"])
                        self.take_profit = self.entry_price - (atr * self.params["take_profit_atr"])
                        self.trailing_stop_price = self.stop_loss
                        
                    logging.info(f"New position: {self.position} at price {self.entry_price}, Stop: {self.stop_loss}, Target: {self.take_profit}")
                    
                # Check for stop loss, take profit or trailing stop
                elif self.position != 0:
                    # Update trailing stop if enabled
                    if self.params["trailing_stop"]:
                        self._update_trailing_stop(current_bar)
                        
                    # Check stop loss hit
                    if (self.position > 0 and current_bar['low'] <= self.trailing_stop_price) or \
                       (self.position < 0 and current_bar['high'] >= self.trailing_stop_price):
                        signals.iloc[i]['position'] = -self.position  # Reverse position to exit
                        logging.info(f"Stop loss hit at price {current_bar['close']}, trailing stop: {self.trailing_stop_price}")
                        self.position = 0
                        self.holding_periods = 0
                        
                    # Check take profit hit
                    elif (self.position > 0 and current_bar['high'] >= self.take_profit) or \
                         (self.position < 0 and current_bar['low'] <= self.take_profit):
                        signals.iloc[i]['position'] = -self.position  # Reverse position to exit
                        logging.info(f"Take profit hit at price {current_bar['close']}, target: {self.take_profit}")
                        self.position = 0
                        self.holding_periods = 0
            
            return signals
            
        except Exception as e:
            logging.error(f"Error generating breakout signals: {str(e)}")
            import traceback
            logging.error(traceback.format_exc())
            return pd.DataFrame(index=data.index, columns=['position'])
            
    def _calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate required technical indicators for the strategy."""
        df = data.copy()
        
        # Calculate ATR if not present
        if 'atr' not in df.columns:
            high_low = df['high'] - df['low']
            high_close = abs(df['high'] - df['close'].shift(1))
            low_close = abs(df['low'] - df['close'].shift(1))
            tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
            df['atr'] = tr.rolling(window=14).mean()
        
        # Calculate ADX for trend strength if not present
        if 'adx' not in df.columns:
            # True Range
            tr = pd.DataFrame()
            tr['h-l'] = df['high'] - df['low']
            tr['h-pc'] = abs(df['high'] - df['close'].shift(1))
            tr['l-pc'] = abs(df['low'] - df['close'].shift(1))
            tr['tr'] = tr[['h-l', 'h-pc', 'l-pc']].max(axis=1)
            
            # Directional Movement
            df['up_move'] = df['high'] - df['high'].shift(1)
            df['down_move'] = df['low'].shift(1) - df['low']
            
            df['plus_dm'] = np.where((df['up_move'] > df['down_move']) & (df['up_move'] > 0), df['up_move'], 0)
            df['minus_dm'] = np.where((df['down_move'] > df['up_move']) & (df['down_move'] > 0), df['down_move'], 0)
            
            # Smooth the True Range and Directional Movement
            period = 14
            df['tr'] = tr['tr'].rolling(window=period).mean()
            df['plus_di'] = 100 * (df['plus_dm'].rolling(window=period).mean() / df['tr'])
            df['minus_di'] = 100 * (df['minus_dm'].rolling(window=period).mean() / df['tr'])
            
            # Calculate ADX
            df['dx'] = 100 * abs(df['plus_di'] - df['minus_di']) / (df['plus_di'] + df['minus_di'])
            df['adx'] = df['dx'].rolling(window=period).mean()
        
        # Calculate RSI if not present
        if 'rsi' not in df.columns:
            delta = df['close'].diff()
            gain = delta.where(delta > 0, 0).rolling(window=14).mean()
            loss = -delta.where(delta < 0, 0).rolling(window=14).mean()
            rs = gain / loss
            df['rsi'] = 100 - (100 / (1 + rs))
        
        # Calculate volume moving average if not present
        if 'volume_ma' not in df.columns:
            df['volume_ma'] = df['volume'].rolling(window=20).mean()
        
        # Calculate price volatility as percentage of price
        df['volatility'] = df['atr'] / df['close']
        
        return df
        
    def _identify_key_levels(self, data: pd.DataFrame) -> List[Dict[str, float]]:
        """
        Identify key support and resistance levels using swing highs/lows and volume.
        
        Returns:
            List of dictionaries with level info including price, strength, and type
        """
        levels = []
        
        # Get subset of data for analysis
        lookback = min(len(data), 200)  # Use at most last 200 bars
        df = data.iloc[-lookback:].copy()
        
        # Find local highs and lows (swing points)
        for i in range(self.params["lookback_period"], len(df) - self.params["lookback_period"]):
            # Current bar
            curr = df.iloc[i]
            
            # Check for swing high (resistance)
            is_swing_high = True
            for j in range(1, self.params["lookback_period"] + 1):
                if df.iloc[i-j]['high'] > curr['high'] or df.iloc[i+j]['high'] > curr['high']:
                    is_swing_high = False
                    break
                    
            if is_swing_high:
                # Calculate level strength based on touches and volume
                strength = self._calculate_level_strength(df, i, curr['high'], "resistance")
                
                levels.append({
                    'price': curr['high'],
                    'type': 'resistance',
                    'strength': strength,
                    'touched': 0,  # Will track touches during live trading
                    'volume': curr['volume']
                })
                
            # Check for swing low (support)
            is_swing_low = True
            for j in range(1, self.params["lookback_period"] + 1):
                if df.iloc[i-j]['low'] < curr['low'] or df.iloc[i+j]['low'] < curr['low']:
                    is_swing_low = False
                    break
                    
            if is_swing_low:
                # Calculate level strength based on touches and volume
                strength = self._calculate_level_strength(df, i, curr['low'], "support")
                
                levels.append({
                    'price': curr['low'],
                    'type': 'support',
                    'strength': strength,
                    'touched': 0,  # Will track touches during live trading
                    'volume': curr['volume']
                })
        
        # Cluster similar levels
        clustered_levels = self._cluster_levels(levels)
        
        # Sort levels by strength (descending)
        clustered_levels.sort(key=lambda x: x['strength'], reverse=True)
        
        # Take top N levels
        return clustered_levels[:10]
    
    def _calculate_level_strength(self, data: pd.DataFrame, idx: int, price: float, 
                                 level_type: str) -> float:
        """Calculate strength of a support/resistance level."""
        # Base strength
        strength = 1.0
        
        # Add strength based on volume at the level
        volume_ratio = data.iloc[idx]['volume'] / data['volume'].mean()
        strength += min(volume_ratio - 1, 2.0)  # Cap additional strength at 2.0
        
        # Add strength based on multiple touches
        price_range = price * 0.003  # 0.3% range for considering price touches
        
        touches = 0
        for i in range(len(data)):
            if i == idx:
                continue
                
            if level_type == "resistance":
                if abs(data.iloc[i]['high'] - price) <= price_range:
                    touches += 1
            else:  # support
                if abs(data.iloc[i]['low'] - price) <= price_range:
                    touches += 1
                    
        strength += min(touches * 0.5, 3.0)  # Add up to 3.0 for touches
        
        # Add strength if level aligns with round numbers
        round_numbers = [0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 25.0, 50.0, 100.0, 250.0, 500.0, 1000.0]
        for round_num in round_numbers:
            if abs((price / round_num) % 1) < 0.03 or abs((price / round_num) % 1) > 0.97:
                strength += 1.0
                break
                
        return strength
        
    def _cluster_levels(self, levels: List[Dict[str, float]]) -> List[Dict[str, float]]:
        """Cluster similar price levels together to avoid duplication."""
        if not levels:
            return []
            
        # Sort by price
        sorted_levels = sorted(levels, key=lambda x: x['price'])
        
        clustered = []
        current_cluster = [sorted_levels[0]]
        
        for i in range(1, len(sorted_levels)):
            current_level = sorted_levels[i]
            prev_level = sorted_levels[i-1]
            
            # Check if current level is close to previous level
            if (current_level['price'] - prev_level['price']) / prev_level['price'] < 0.01:
                # Add to current cluster
                current_cluster.append(current_level)
            else:
                # Process the completed cluster
                if current_cluster:
                    # Merge cluster into a single level
                    merged_level = self._merge_cluster(current_cluster)
                    clustered.append(merged_level)
                    
                # Start a new cluster
                current_cluster = [current_level]
                
        # Process the last cluster
        if current_cluster:
            merged_level = self._merge_cluster(current_cluster)
            clustered.append(merged_level)
            
        return clustered
        
    def _merge_cluster(self, cluster: List[Dict[str, float]]) -> Dict[str, float]:
        """Merge a cluster of similar levels into a single level."""
        # Average price weighted by strength
        total_weight = sum(level['strength'] for level in cluster)
        weighted_price = sum(level['price'] * level['strength'] for level in cluster) / total_weight
        
        # Determine cluster type (support or resistance)
        support_count = sum(1 for level in cluster if level['type'] == 'support')
        resistance_count = len(cluster) - support_count
        cluster_type = 'support' if support_count > resistance_count else 'resistance'
        
        # Combined strength (sum of all strengths, capped)
        total_strength = min(sum(level['strength'] for level in cluster), 10.0)
        
        # Maximum volume
        max_volume = max(level['volume'] for level in cluster)
        
        return {
            'price': weighted_price,
            'type': cluster_type,
            'strength': total_strength,
            'touched': 0,
            'volume': max_volume
        }
        
    def _detect_market_regime(self, data: pd.DataFrame) -> str:
        """
        Detect current market regime (trending, ranging, volatile).
        
        Returns:
            str: Market regime type
        """
        # Get last 30 bars or all available data
        lookback = min(30, len(data))
        recent_data = data.iloc[-lookback:]
        
        # Check trend strength using ADX
        adx = recent_data['adx'].iloc[-1] if 'adx' in recent_data.columns else 0
        
        # Check volatility
        volatility = recent_data['volatility'].iloc[-1] if 'volatility' in recent_data.columns else 0
        
        # Determine regime
        if volatility > self.params["max_volatility"] * 1.5:
            regime = "extreme_volatility"
        elif adx > self.params["trending_threshold"]:
            # Determine trend direction
            plus_di = recent_data['plus_di'].iloc[-1] if 'plus_di' in recent_data.columns else 0
            minus_di = recent_data['minus_di'].iloc[-1] if 'minus_di' in recent_data.columns else 0
            
            if plus_di > minus_di:
                regime = "uptrend"
            else:
                regime = "downtrend"
        else:
            regime = "ranging"
            
        logging.debug(f"Market regime detected: {regime}, ADX: {adx:.2f}, Volatility: {volatility:.4f}")
        return regime
        
    def _detect_resistance_breakout(self, data: pd.DataFrame, idx: int) -> bool:
        """Detect breakout above resistance level."""
        current_bar = data.iloc[idx]
        
        # Check for breakout against identified key levels
        for level in self.key_levels:
            if level['type'] != 'resistance':
                continue
                
            # Calculate breakout threshold based on ATR and level strength
            atr = current_bar['atr'] if 'atr' in current_bar else self._calculate_atr(data, idx)
            threshold = max(self.params["breakout_threshold"], atr / current_bar['close'] * 0.5)
            
            # Check for breakout with sufficient margin
            if current_bar['close'] > level['price'] * (1 + threshold):
                # Check for consolidation before breakout
                if self._check_consolidation_before_breakout(data, idx, level['price']):
                    return True
                    
        return False
        
    def _detect_support_breakout(self, data: pd.DataFrame, idx: int) -> bool:
        """Detect breakout below support level."""
        current_bar = data.iloc[idx]
        
        # Check for breakout against identified key levels
        for level in self.key_levels:
            if level['type'] != 'support':
                continue
                
            # Calculate breakout threshold based on ATR and level strength
            atr = current_bar['atr'] if 'atr' in current_bar else self._calculate_atr(data, idx)
            threshold = max(self.params["breakout_threshold"], atr / current_bar['close'] * 0.5)
            
            # Check for breakout with sufficient margin
            if current_bar['close'] < level['price'] * (1 - threshold):
                # Check for consolidation before breakout
                if self._check_consolidation_before_breakout(data, idx, level['price']):
                    return True
                    
        return False
        
    def _check_consolidation_before_breakout(self, data: pd.DataFrame, idx: int, level_price: float) -> bool:
        """Check if price consolidated before breakout."""
        # Need enough bars for checking consolidation
        if idx < self.params["consolidation_periods"]:
            return False
            
        # Get bars before current bar
        lookback_bars = data.iloc[idx-self.params["consolidation_periods"]:idx]
        
        # Check if price was near the level for consecutive periods
        for _, bar in lookback_bars.iterrows():
            # Allow for some wiggle room (0.5% of level price)
            if abs(bar['close'] - level_price) / level_price > 0.005:
                return False
                
        return True
        
    def _confirm_breakout(self, data: pd.DataFrame, idx: int, breakout_type: str) -> bool:
        """Apply confirmation filters to avoid false breakouts."""
        current_bar = data.iloc[idx]
        prev_bar = data.iloc[idx-1]
        
        # Volume confirmation
        volume_increased = current_bar['volume'] > prev_bar['volume'] * self.params["volume_multiplier"]
        
        # Momentum confirmation using RSI
        momentum_confirmed = False
        if 'rsi' in current_bar and 'rsi' in prev_bar:
            rsi_change = abs(current_bar['rsi'] - prev_bar['rsi'])
            momentum_confirmed = rsi_change > self.params["momentum_threshold"]
        
        # Check for false breakout pattern if enabled
        if self.params["false_breakout_guard"]:
            # For resistance breakout (long signal)
            if breakout_type == "resistance":
                # Avoid buying if price closed in lower half of the range
                bar_range = current_bar['high'] - current_bar['low']
                if bar_range > 0 and (current_bar['high'] - current_bar['close']) / bar_range > 0.6:
                    logging.debug("False breakout detected (closed in lower half of range)")
                    return False
            
            # For support breakout (short signal)
            elif breakout_type == "support":
                # Avoid selling if price closed in upper half of the range
                bar_range = current_bar['high'] - current_bar['low']
                if bar_range > 0 and (current_bar['close'] - current_bar['low']) / bar_range > 0.6:
                    logging.debug("False breakout detected (closed in upper half of range)")
                    return False
        
        # Apply market regime filter
        market_regime = self._detect_market_regime(data)
        regime_confirmed = True
        
        if breakout_type == "resistance":
            # Only take long signals in uptrend or ranging markets
            if market_regime == "downtrend":
                regime_confirmed = False
        else:  # support breakout
            # Only take short signals in downtrend or ranging markets
            if market_regime == "uptrend":
                regime_confirmed = False
                
        # Combined confirmation
        confirmed = volume_increased and momentum_confirmed and regime_confirmed
        
        if not confirmed:
            logging.debug(f"Breakout not confirmed. Volume: {volume_increased}, "
                         f"Momentum: {momentum_confirmed}, Regime: {regime_confirmed}")
                         
        return confirmed
        
    def _calculate_atr(self, data: pd.DataFrame, idx: int, period: int = 14) -> float:
        """Calculate Average True Range."""
        if idx < period:
            return 0
            
        true_ranges = []
        for i in range(idx - period + 1, idx + 1):
            high_low = data.iloc[i]['high'] - data.iloc[i]['low']
            high_close = abs(data.iloc[i]['high'] - data.iloc[i-1]['close']) if i > 0 else 0
            low_close = abs(data.iloc[i]['low'] - data.iloc[i-1]['close']) if i > 0 else 0
            
            true_ranges.append(max(high_low, high_close, low_close))
            
        return sum(true_ranges) / len(true_ranges)
        
    def _update_trailing_stop(self, current_bar: pd.Series) -> None:
        """Update trailing stop price."""
        if self.position > 0:  # Long position
            # Calculate new potential stop level
            atr = current_bar['atr'] if 'atr' in current_bar else 0
            potential_stop = current_bar['close'] - (atr * self.params["trailing_distance"])
            
            # Update if new stop is higher than current stop
            if potential_stop > self.trailing_stop_price:
                self.trailing_stop_price = potential_stop
                logging.debug(f"Updated trailing stop to {self.trailing_stop_price}")
                
        elif self.position < 0:  # Short position
            # Calculate new potential stop level
            atr = current_bar['atr'] if 'atr' in current_bar else 0
            potential_stop = current_bar['close'] + (atr * self.params["trailing_distance"])
            
            # Update if new stop is lower than current stop
            if potential_stop < self.trailing_stop_price:
                self.trailing_stop_price = potential_stop
                logging.debug(f"Updated trailing stop to {self.trailing_stop_price}")
                
    def calculate_position_size(self, account_balance: float, current_price: float) -> float:
        """
        Calculate optimal position size based on risk parameters.
        
        Args:
            account_balance: Current account balance
            current_price: Current asset price
            
        Returns:
            float: Position size in base currency
        """
        # Risk amount based on account balance
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
            
        # Apply volatility adjustment if available
        if hasattr(self, 'current_volatility') and self.current_volatility:
            # Reduce position size in high volatility
            vol_factor = max(0.3, 1.0 - (self.current_volatility * 10))
            position_size *= vol_factor
            
        return position_size
        
    def update_params(self, new_params: Dict[str, Any]) -> None:
        """
        Update strategy parameters.
        
        Args:
            new_params: New parameter values
        """
        self.params.update(new_params)
        logging.info(f"Updated breakout strategy parameters: {new_params}")
        
    def get_params(self) -> Dict[str, Any]:
        """Get current strategy parameters."""
        return self.params.copy()
        
    def get_info(self) -> Dict[str, Any]:
        """Get strategy information including current state."""
        return {
            "name": self.name,
            "description": self.description,
            "params": self.params,
            "position": self.position,
            "entry_price": self.entry_price,
            "stop_loss": self.stop_loss,
            "take_profit": self.take_profit,
            "trailing_stop": self.trailing_stop_price,
            "holding_periods": self.holding_periods,
            "key_levels_count": len(self.key_levels)
        }

# Module functions for strategy creation and registration
def create_strategy(params=None):
    """Factory function to create a breakout strategy instance."""
    return BreakoutStrategy(params)
