# strategies/implementations/trend_following.py

import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Tuple, Optional, Any, Union
import time
from dataclasses import dataclass
from datetime import datetime

from core.brokers.base_broker import OrderType, OrderSide

@dataclass
class TrendFollowingParams:
    fast_period: int = 10
    slow_period: int = 30
    atr_period: int = 14
    atr_multiplier: float = 2.0
    position_size: float = 1.0
    max_positions: int = 5

class TrendFollowingStrategy:
    """
    Trend Following Strategy using moving averages and ATR for position management.
    
    The strategy:
    1. Uses two moving averages (fast and slow) to identify trends
    2. Enters long when fast MA crosses above slow MA
    3. Enters short when fast MA crosses below slow MA
    4. Uses ATR-based trailing stops for exit
    """
    
    def __init__(self, params: Optional[TrendFollowingParams] = None):
        """Initialize strategy with parameters."""
        self.params = params or TrendFollowingParams()
        self.positions: Dict[str, Dict] = {}
        self.signals: List[Dict] = []
        
        # Strategy state variables
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
        
    def calculate_indicators(self, data: pd.DataFrame) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Calculate strategy indicators."""
        # Calculate moving averages
        fast_ma = data['close'].rolling(window=self.params.fast_period).mean()
        slow_ma = data['close'].rolling(window=self.params.slow_period).mean()
        
        # Calculate ATR
        high_low = data['high'] - data['low']
        high_close = np.abs(data['high'] - data['close'].shift())
        low_close = np.abs(data['low'] - data['close'].shift())
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = ranges.max(axis=1)
        atr = true_range.rolling(window=self.params.atr_period).mean()
        
        return fast_ma, slow_ma, atr
    
    def generate_signals(self, symbol: str, data: pd.DataFrame) -> List[Dict]:
        """Generate trading signals based on price action."""
        if len(data) < self.params.slow_period + 1:  # Need at least slow_period + 1 data points
            return []
        
        # Calculate indicators
        fast_ma, slow_ma, atr = self.calculate_indicators(data)
        
        signals = []
        current_price = data['close'].iloc[-1]
        timestamp = data.index[-1]
        
        # Get previous and current MA values
        prev_fast = fast_ma.iloc[-2]
        prev_slow = slow_ma.iloc[-2]
        curr_fast = fast_ma.iloc[-1]
        curr_slow = slow_ma.iloc[-1]
        
        # Check if we have an existing position
        position = self.positions.get(symbol)
        
        if position:
            # Calculate trailing stop
            current_atr = atr.iloc[-1]
            stop_distance = current_atr * self.params.atr_multiplier
            
            if position['side'] == 'LONG':
                trailing_stop = current_price - stop_distance
                if trailing_stop > position['stop_loss']:
                    position['stop_loss'] = trailing_stop
                
                # Check if stop loss is hit
                if current_price < position['stop_loss']:
                    signals.append(self._create_signal(
                        symbol, OrderSide.SELL, current_price,
                        timestamp, "TRAILING_STOP"
                    ))
                    self.positions.pop(symbol)
                
                # Check for trend reversal
                elif curr_fast < curr_slow and prev_fast > prev_slow:
                    signals.append(self._create_signal(
                        symbol, OrderSide.SELL, current_price,
                        timestamp, "TREND_REVERSAL"
                    ))
                    self.positions.pop(symbol)
            
            elif position['side'] == 'SHORT':
                trailing_stop = current_price + stop_distance
                if trailing_stop < position['stop_loss']:
                    position['stop_loss'] = trailing_stop
                
                # Check if stop loss is hit
                if current_price > position['stop_loss']:
                    signals.append(self._create_signal(
                        symbol, OrderSide.BUY, current_price,
                        timestamp, "TRAILING_STOP"
                    ))
                    self.positions.pop(symbol)
                
                # Check for trend reversal
                elif curr_fast > curr_slow and prev_fast < prev_slow:
                    signals.append(self._create_signal(
                        symbol, OrderSide.BUY, current_price,
                        timestamp, "TREND_REVERSAL"
                    ))
                    self.positions.pop(symbol)
        
        else:
            # Check entry conditions
            if len(self.positions) < self.params.max_positions:
                # Bullish crossover
                if curr_fast > curr_slow and prev_fast <= prev_slow:
                    stop_loss = current_price - (atr.iloc[-1] * self.params.atr_multiplier)
                    signals.append(self._create_signal(
                        symbol, OrderSide.BUY, current_price,
                        timestamp, "TREND_ENTRY"
                    ))
                    self.positions[symbol] = {
                        'side': 'LONG',
                        'entry_price': current_price,
                        'stop_loss': stop_loss,
                        'entry_time': timestamp
                    }
                
                # Bearish crossover
                elif curr_fast < curr_slow and prev_fast >= prev_slow:
                    stop_loss = current_price + (atr.iloc[-1] * self.params.atr_multiplier)
                    signals.append(self._create_signal(
                        symbol, OrderSide.SELL, current_price,
                        timestamp, "TREND_ENTRY"
                    ))
                    self.positions[symbol] = {
                        'side': 'SHORT',
                        'entry_price': current_price,
                        'stop_loss': stop_loss,
                        'entry_time': timestamp
                    }
        
        return signals
    
    def _create_signal(self, symbol: str, side: OrderSide, 
                      price: float, timestamp: datetime, 
                      signal_type: str) -> Dict:
        """Create a trading signal."""
        return {
            'symbol': symbol,
            'side': side,
            'order_type': OrderType.MARKET,
            'price': price,
            'quantity': self.params.position_size,
            'timestamp': timestamp,
            'signal_type': signal_type
        }
    
    def update_parameters(self, new_params: Dict) -> None:
        """Update strategy parameters."""
        for key, value in new_params.items():
            if hasattr(self.params, key):
                setattr(self.params, key, value)
    
    def get_positions(self) -> Dict[str, Dict]:
        """Get current positions."""
        return self.positions.copy()
    
    def reset(self) -> None:
        """Reset strategy state."""
        self.positions.clear()
        self.signals.clear()
        
        # Reset strategy state variables
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
        
        # Clear calculation cache
        self._calculation_cache = {}
        self._last_calc_time = {}
        
        self.logger.info("Strategy reset to initial state")

# Factory function to create the strategy
def create_strategy() -> TrendFollowingStrategy:
    """Create a new instance of the TrendFollowingStrategy."""
    return TrendFollowingStrategy()
