# strategy_system

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
import torch
import torch.nn as nn
from collections import deque
import random
import logging
from datetime import datetime, timedelta

class StrategyLearner:
    def __init__(self, state_size=60, action_size=3):
        self.state_size = state_size
        self.action_size = action_size  # Buy, Sell, Hold
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95  # Discount rate
        self.epsilon = 1.0  # Exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.batch_size = 32
        self.model = self._build_model()
        self.scaler = StandardScaler()
        self.strategy_performance = {}

    def _build_model(self):
        model = Sequential([
            LSTM(64, input_shape=(self.state_size, 11), return_sequences=True),
            Dropout(0.2),
            LSTM(32, return_sequences=False),
            Dense(16, activation='relu'),
            Dense(self.action_size, activation='linear')
        ])
        model.compile(loss='mse', optimizer='adam')
        return model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if random.random() <= self.epsilon:
            return random.randrange(self.action_size)
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])

    def replay(self, batch_size):
        if len(self.memory) < batch_size:
            return
            
        minibatch = random.sample(self.memory, batch_size)
        states = np.zeros((batch_size, self.state_size, 11))
        next_states = np.zeros((batch_size, self.state_size, 11))
        
        for i, (state, action, reward, next_state, done) in enumerate(minibatch):
            states[i] = state
            next_states[i] = next_state
            
        targets = self.model.predict(states)
        next_targets = self.model.predict(next_states)
        
        for i, (state, action, reward, next_state, done) in enumerate(minibatch):
            if done:
                targets[i][action] = reward
            else:
                targets[i][action] = reward + self.gamma * np.amax(next_targets[i])
                
        self.model.fit(states, targets, epochs=1, verbose=0)
        
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

class StrategyGenerator:
    def __init__(self):
        self.strategies = {
            'trend_following': self.trend_following_strategy,
            'mean_reversion': self.mean_reversion_strategy,
            'breakout': self.breakout_strategy,
            'multi_timeframe': self.multi_timeframe_strategy,
            'volume_based': self.volume_based_strategy
        }
        self.strategy_params = {}
        self.optimized_params = {}

    def trend_following_strategy(self, data, params):
        """Advanced trend following strategy"""
        signals = pd.DataFrame(index=data.index)
        signals['position'] = 0

        # Calculate multiple EMAs
        short_ema = data['close'].ewm(span=params['short_window']).mean()
        medium_ema = data['close'].ewm(span=params['medium_window']).mean()
        long_ema = data['close'].ewm(span=params['long_window']).mean()

        # Generate signals
        signals['position'] = np.where(
            (short_ema > medium_ema) & (medium_ema > long_ema), 
            1,  # Buy signal
            np.where(
                (short_ema < medium_ema) & (medium_ema < long_ema),
                -1,  # Sell signal
                0  # Hold
            )
        )

        return signals

    def mean_reversion_strategy(self, data, params):
        """Advanced mean reversion strategy"""
        signals = pd.DataFrame(index=data.index)
        signals['position'] = 0

        # Calculate Bollinger Bands
        rolling_mean = data['close'].rolling(window=params['window']).mean()
        rolling_std = data['close'].rolling(window=params['window']).std()
        upper_band = rolling_mean + (rolling_std * params['std_dev'])
        lower_band = rolling_mean - (rolling_std * params['std_dev'])

        # RSI for confirmation
        delta = data['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))

        # Generate signals
        signals['position'] = np.where(
            (data['close'] < lower_band) & (rsi < 30),
            1,  # Buy signal
            np.where(
                (data['close'] > upper_band) & (rsi > 70),
                -1,  # Sell signal
                0  # Hold
            )
        )

        return signals

    def breakout_strategy(self, data, params):
        """Advanced breakout strategy"""
        signals = pd.DataFrame(index=data.index)
        signals['position'] = 0

        # Calculate support and resistance levels
        rolling_high = data['high'].rolling(window=params['window']).max()
        rolling_low = data['low'].rolling(window=params['window']).min()
        
        # Volume confirmation
        volume_ma = data['volume'].rolling(window=params['volume_window']).mean()
        volume_trigger = data['volume'] > volume_ma * params['volume_threshold']

        # ATR for volatility adjustment
        high_low = data['high'] - data['low']
        high_close = abs(data['high'] - data['close'].shift())
        low_close = abs(data['low'] - data['close'].shift())
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = ranges.max(axis=1)
        atr = true_range.rolling(window=14).mean()

        # Generate signals with dynamic thresholds
        signals['position'] = np.where(
            (data['close'] > rolling_high.shift()) & volume_trigger & (data['close'] > data['close'].shift() + atr),
            1,  # Buy signal
            np.where(
                (data['close'] < rolling_low.shift()) & volume_trigger & (data['close'] < data['close'].shift() - atr),
                -1,  # Sell signal
                0  # Hold
            )
        )

        return signals

    def multi_timeframe_strategy(self, data, params):
        """Multi-timeframe momentum strategy"""
        signals = pd.DataFrame(index=data.index)
        signals['position'] = 0

        # Calculate indicators for multiple timeframes
        timeframes = ['1h', '4h', '1d']
        trends = {}

        for tf in timeframes:
            # Resample data to timeframe
            resampled = data['close'].resample(tf).ohlc()
            
            # Calculate EMAs
            ema_short = resampled['close'].ewm(span=params['short_window']).mean()
            ema_long = resampled['close'].ewm(span=params['long_window']).mean()
            
            # Determine trend
            trends[tf] = ema_short > ema_long

        # Generate signals based on trend alignment
        signals['position'] = np.where(
            all(trends.values()),  # All timeframes showing uptrend
            1,  # Buy signal
            np.where(
                all(not trend for trend in trends.values()),  # All timeframes showing downtrend
                -1,  # Sell signal
                0  # Hold
            )
        )

        return signals

    def volume_based_strategy(self, data, params):
        """Advanced volume-based strategy"""
        signals = pd.DataFrame(index=data.index)
        signals['position'] = 0

        # Calculate volume indicators
        volume_ma = data['volume'].rolling(window=params['volume_window']).mean()
        obv = (np.sign(data['close'].diff()) * data['volume']).cumsum()
        vwap = (data['volume'] * data['close']).cumsum() / data['volume'].cumsum()

        # Money Flow Index
        typical_price = (data['high'] + data['low'] + data['close']) / 3
        money_flow = typical_price * data['volume']
        positive_flow = money_flow.where(typical_price > typical_price.shift(), 0).rolling(14).sum()
        negative_flow = money_flow.where(typical_price < typical_price.shift(), 0).rolling(14).sum()
        mfi = 100 - (100 / (1 + positive_flow / negative_flow))

        # Generate signals
        signals['position'] = np.where(
            (data['volume'] > volume_ma * params['volume_threshold']) &
            (obv.diff() > 0) & (mfi < 20) & (data['close'] > vwap),
            1,  # Buy signal
            np.where(
                (data['volume'] > volume_ma * params['volume_threshold']) &
                (obv.diff() < 0) & (mfi > 80) & (data['close'] < vwap),
                -1,  # Sell signal
                0  # Hold
            )
        )

        return signals

class StrategyOptimizer:
    def __init__(self, strategy_generator):
        self.strategy_generator = strategy_generator
        self.reinforcement_learner = StrategyLearner()
        self.best_params = {}
        self.strategy_performance = {}

    def optimize_strategy(self, data, strategy_name, param_grid, metric='sharpe'):
        """Optimize strategy parameters using machine learning"""
        best_score = float('-inf')
        best_params = None

        for params in self._generate_param_combinations(param_grid):
            # Generate signals using current parameters
            signals = self.strategy_generator.strategies[strategy_name](data, params)
            
            # Calculate strategy performance
            performance = self._calculate_performance(data, signals, metric)
            
            # Update best parameters if performance is better
            if performance > best_score:
                best_score = performance
                best_params = params

            # Store experience for reinforcement learning
            state = self._prepare_state(data)
            action = self._params_to_action(params)
            reward = performance
            next_state = self._prepare_state(data.shift(-1))
            done = False

            self.reinforcement_learner.remember(state, action, reward, next_state, done)

        # Train reinforcement learner
        self.reinforcement_learner.replay(32)

        return best_params, best_score

    def _generate_param_combinations(self, param_grid):
        """Generate combinations of parameters for grid search"""
        from itertools import product
        
        keys = param_grid.keys()
        values = param_grid.values()
        
        for instance in product(*values):
            yield dict(zip(keys, instance))

    def _calculate_performance(self, data, signals, metric='sharpe'):
        """Calculate strategy performance metrics"""
        returns = data['close'].pct_change() * signals['position'].shift()
        
        if metric == 'sharpe':
            return np.sqrt(252) * returns.mean() / returns.std()
        elif metric == 'sortino':
            downside_returns = returns[returns < 0]
            return np.sqrt(252) * returns.mean() / downside_returns.std()
        elif metric == 'max_drawdown':
            cumulative_returns = (1 + returns).cumprod()
            return (cumulative_returns / cumulative_returns.expanding().max() - 1).min()
        
        return 0

    def _prepare_state(self, data):
        """Prepare state data for reinforcement learning"""
        features = [
            'close', 'volume', 'high', 'low',
            'sma_20', 'sma_50', 'rsi', 'macd',
            'bollinger_upper', 'bollinger_lower', 'atr'
        ]
        
        state = np.zeros((1, self.reinforcement_learner.state_size, len(features)))
        
        for i, feature in enumerate(features):
            if feature in data.columns:
                state[0, :, i] = data[feature].values[-self.reinforcement_learner.state_size:]
            
        return self.reinforcement_learner.scaler.fit_transform(state)

    def _params_to_action(self, params):
        """Convert strategy parameters to discrete action"""
        # Simplified conversion - can be enhanced based on specific parameters
        return sum(hash(str(v)) for v in params.values()) % self.reinforcement_learner.action_size
        


def get_strategy_parameters(self, strategy_name):
    """Get current parameters for a strategy"""
    if strategy_name == 'trend_following':
        return {
            'short_window': 10,  # Default values, will be overridden if set
            'medium_window': 50,
            'long_window': 100,
            'stop_loss_pct': 0.02,
            'take_profit_pct': 0.05
        }
    elif strategy_name == 'mean_reversion':
        return {
            'window': 20,
            'std_dev': 2.0,
            'stop_loss_pct': 0.02,
            'take_profit_pct': 0.04
        }
    # Add parameters for other strategies...
    return {}

class StrategyExecutor:
    def __init__(self, strategy_generator, strategy_optimizer):
        self.strategy_generator = strategy_generator
        self.strategy_optimizer = strategy_optimizer
        self.active_strategies = {}
        self.performance_history = {}

    def execute_strategies(self, data, strategies=None):
        """Execute multiple strategies with position sizing"""
        if strategies is None:
            strategies = list(self.strategy_generator.strategies.keys())

        combined_signals = pd.DataFrame(index=data.index)
        combined_signals['position'] = 0

        for strategy_name in strategies:
            # Get optimized parameters
            if strategy_name not in self.strategy_optimizer.best_params:
                self.optimize_strategy(data, strategy_name)

            # Generate signals using optimized parameters
            signals = self.strategy_generator.strategies[strategy_name](
                data, 
                self.strategy_optimizer.best_params[strategy_name]
            )

            # Weight signals based on strategy performance
            weight = self._calculate_strategy_weight(strategy_name)
            combined_signals['position'] += signals['position'] * weight

        return combined_signals

    def optimize_strategy(self, data, strategy_name):
        """Optimize strategy parameters"""
        param_grid = self._get_param_grid(strategy_name)
        best_params, score = self.strategy_optimizer.optimize_strategy(
            data, 
            strategy_name, 
            param_grid
        )
        
        self.strategy_optimizer.best_params[strategy_name] = best_params
        self.performance_history[strategy_name] = score

    def _get_param_grid(self, strategy_name):
        """Get parameter grid for strategy optimization"""
        if strategy_name == 'trend_following':
            return {
                'short_window': range(5, 30, 5),
                'medium_window': range(15, 60, 5),
                'long_window': range(30, 120, 10)
            }
        elif strategy_name == 'mean_reversion':
            return {
                'window': range(10, 50, 5),
                'std_dev': [1.5, 2.0, 2.5, 3.0]
            }
        elif strategy_name == 'breakout':
            return {
                'window': range(20, 100, 10),
                'volume_window': range(5, 30, 5),
                'volume_threshold': [1.5, 2.0, 2.5, 3.0]
            }
        elif strategy_name == 'multi_timeframe':
            return {
                'short_window': range(5, 30, 5),
                'long_window': range(20, 100, 10)
            }
        elif strategy_name == 'volume_based':
            return {
                'volume_window': range(10, 50, 5),
                'volume_threshold': [1.2, 1.5, 1.8, 2.0]
            }
        return {}

    def _calculate_strategy_weight(self, strategy_name):
        """Calculate strategy weight based on performance"""
        if not self.performance_history:
            return 1.0 / len(self.strategy_generator.strategies)
            
        strategy_score = self.performance_history.get(strategy_name, 0)
        total_score = sum(self.performance_history.values())
        
        if total_score == 0:
            return 0
            
        return strategy_score / total_score

    def update_strategy_weights(self, recent_performance):
        """Update strategy weights based on recent performance"""
        for strategy_name, performance in recent_performance.items():
            if strategy_name in self.performance_history:
                # Exponential moving average of performance
                self.performance_history[strategy_name] = (
                    0.9 * self.performance_history[strategy_name] +
                    0.1 * performance
                )

class AdvancedStrategySystem:
    def __init__(self):
        self.strategy_generator = StrategyGenerator()
        self.strategy_optimizer = StrategyOptimizer(self.strategy_generator)
        self.strategy_executor = StrategyExecutor(
            self.strategy_generator,
            self.strategy_optimizer
        )
        self.active_strategies = []
        self.last_optimization = None
        self.optimization_interval = timedelta(hours=24)

    def initialize(self, data):
        """Initialize the strategy system"""
        self.active_strategies = list(self.strategy_generator.strategies.keys())
        self.optimize_all_strategies(data)

    def optimize_all_strategies(self, data):
        """Optimize all active strategies"""
        for strategy_name in self.active_strategies:
            self.strategy_executor.optimize_strategy(data, strategy_name)
        self.last_optimization = datetime.now()

    def execute(self, data):
        """Execute trading strategies"""
        # Check if reoptimization is needed
        if (self.last_optimization is not None and 
            (datetime.now() - self.last_optimization) > self.optimization_interval):
            self.optimize_all_strategies(data)

        # Execute strategies
        signals = self.strategy_executor.execute_strategies(data, self.active_strategies)
        
        # Calculate position sizes
        position_sizes = self._calculate_position_sizes(signals, data)
        
        return {
            'signals': signals,
            'position_sizes': position_sizes,
            'active_strategies': self.active_strategies,
            'strategy_weights': self._get_strategy_weights()
        }

    def _calculate_position_sizes(self, signals, data):
        """Calculate position sizes based on signals and risk management"""
        position_sizes = pd.Series(index=signals.index, data=0.0)
        
        # Risk per trade (percentage of portfolio)
        risk_per_trade = 0.02
        
        # Calculate ATR for volatility-based sizing
        high_low = data['high'] - data['low']
        high_close = abs(data['high'] - data['close'].shift())
        low_close = abs(data['low'] - data['close'].shift())
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = ranges.max(axis=1)
        atr = true_range.rolling(window=14).mean()
        
        for i in range(len(signals)):
            if signals['position'][i] != 0:
                # Base position size on ATR
                volatility_factor = 1 / (atr[i] / data['close'][i])
                
                # Adjust for signal strength
                signal_strength = abs(signals['position'][i])
                
                # Calculate final position size
                position_sizes[i] = risk_per_trade * volatility_factor * signal_strength
                
                # Cap maximum position size
                position_sizes[i] = min(position_sizes[i], 0.1)  # Max 10% of portfolio

        return position_sizes

    def _get_strategy_weights(self):
        """Get current strategy weights"""
        return {
            strategy: self.strategy_executor._calculate_strategy_weight(strategy)
            for strategy in self.active_strategies
        }

    def update_performance(self, performance_metrics):
        """Update strategy performance and adjust weights"""
        self.strategy_executor.update_strategy_weights(performance_metrics)

class StrategyPerformanceMonitor:
    def __init__(self):
        self.strategy_metrics = {}
        self.rolling_window = 20

    def update_metrics(self, signals, actual_returns):
        """Update strategy performance metrics"""
        metrics = {}
        
        # Calculate returns
        strategy_returns = signals * actual_returns
        
        # Calculate metrics
        metrics['sharpe'] = self._calculate_sharpe(strategy_returns)
        metrics['sortino'] = self._calculate_sortino(strategy_returns)
        metrics['max_drawdown'] = self._calculate_max_drawdown(strategy_returns)
        metrics['win_rate'] = self._calculate_win_rate(strategy_returns)
        
        return metrics

    def _calculate_sharpe(self, returns):
        """Calculate Sharpe ratio"""
        if len(returns) < 2:
            return 0
        return np.sqrt(252) * returns.mean() / returns.std()

    def _calculate_sortino(self, returns):
        """Calculate Sortino ratio"""
        if len(returns) < 2:
            return 0
        downside_returns = returns[returns < 0]
        if len(downside_returns) < 1:
            return 0
        return np.sqrt(252) * returns.mean() / downside_returns.std()

    def _calculate_max_drawdown(self, returns):
        """Calculate maximum drawdown"""
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdowns = cumulative / running_max - 1
        return drawdowns.min()

    def _calculate_win_rate(self, returns):
        """Calculate win rate"""
        if len(returns) < 1:
            return 0
        wins = len(returns[returns > 0])
        return wins / len(returns)

def create_strategy_system():
    """Create and initialize the complete strategy system"""
    strategy_system = AdvancedStrategySystem()
    return strategy_system
