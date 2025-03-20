# strategy_executor.py


import logging
import time
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Union, Optional, Any, Callable, Set
import threading
from collections import defaultdict, deque

# Configure module logger
logger = logging.getLogger(__name__)

class StrategyExecutor:
    """
    Executes multiple trading strategies with performance-based weighting, position
    sizing, and risk management. Manages the lifecycle of strategy signals from
    generation to execution decision.
    
    This class serves as the orchestration layer between strategy generation and
    actual trade execution, ensuring that strategies are executed according to their
    performance metrics, risk parameters, and market conditions.
    
    Attributes:
        strategy_generator: Component that generates strategy instances
        strategy_optimizer: Component that optimizes strategy parameters
        active_strategies: Dict mapping strategy names to their instances
        strategy_weights: Dict mapping strategy names to their weights
        strategy_metrics: Dict mapping strategy names to performance metrics
        position_manager: Optional position manager for tracking positions
        market_regime: Current detected market regime (e.g., trending, volatile)
        last_execution: Timestamp of last strategy execution
        execution_interval: Minimum time between strategy executions
        execution_lock: Thread lock for concurrent execution safety
        signal_history: Dict storing recent signals for each strategy
        position_sizing: Maximum position size as percentage of portfolio
        risk_scaling: Whether to use dynamic risk scaling
        ml_enhancement: Whether to enhance signals with ML predictions
    """
    
    def __init__(
        self, 
        strategy_generator: Any, 
        strategy_optimizer: Any,
        position_manager: Optional[Any] = None,
        execution_interval: int = 60,  # Seconds between executions
        position_sizing: float = 0.02,  # Default 2% position size
        risk_scaling: bool = True,
        ml_enhancement: bool = True
    ) -> None:
        """
        Initialize the Strategy Executor with dependencies and configuration.
        
        Args:
            strategy_generator: Component that generates strategy instances
            strategy_optimizer: Component that optimizes strategy parameters
            position_manager: Optional position manager for tracking positions
            execution_interval: Minimum seconds between strategy executions
            position_sizing: Maximum position size as percentage of portfolio
            risk_scaling: Whether to use dynamic risk scaling based on metrics
            ml_enhancement: Whether to enhance signals with ML predictions
        
        Raises:
            ValueError: If essential dependencies are not provided
        """
        # Validate dependencies
        if strategy_generator is None:
            raise ValueError("Strategy generator must be provided")
        if strategy_optimizer is None:
            raise ValueError("Strategy optimizer must be provided")
            
        # Initialize components
        self.strategy_generator = strategy_generator
        self.strategy_optimizer = strategy_optimizer
        self.position_manager = position_manager
        
        # Configuration
        self.execution_interval = execution_interval
        self.position_sizing = position_sizing
        self.risk_scaling = risk_scaling
        self.ml_enhancement = ml_enhancement
        
        # Internal state
        self.active_strategies: Dict[str, Any] = {}
        self.strategy_weights: Dict[str, float] = {}
        self.strategy_metrics: Dict[str, Dict[str, float]] = {}
        self.market_regime: str = "unknown"
        self.last_execution: Optional[datetime] = None
        self.execution_lock = threading.RLock()  # Reentrant lock for nested calls
        
        # Signal history stores recent signals for smoothing
        self.signal_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=10))
        
        # Performance tracking
        self.cumulative_pnl: float = 0.0
        self.win_count: int = 0
        self.loss_count: int = 0
        self.execution_times: List[float] = []
        
        logger.info("Strategy Executor initialized with %d strategies", 
                    len(self.active_strategies))
    
    def add_strategy(self, strategy_name: str, weight: float = 1.0) -> bool:
        """
        Add a strategy to the executor's active strategies with specified weight.
        
        Args:
            strategy_name: Name of the strategy to add
            weight: Initial weight for the strategy (0.0 - 1.0)
            
        Returns:
            bool: True if strategy was successfully added, False otherwise
            
        Raises:
            ValueError: If weight is outside valid range
        """
        if not 0.0 <= weight <= 1.0:
            raise ValueError(f"Strategy weight must be between 0.0 and 1.0, got {weight}")
            
        with self.execution_lock:
            try:
                # Create strategy instance from generator
                if hasattr(self.strategy_generator, 'strategies') and strategy_name in self.strategy_generator.strategies:
                    # Add to active strategies
                    self.active_strategies[strategy_name] = self.strategy_generator.strategies[strategy_name]
                    self.strategy_weights[strategy_name] = weight
                    
                    # Initialize metrics
                    self.strategy_metrics[strategy_name] = {
                        'win_rate': 0.5,            # Default win rate
                        'profit_factor': 1.0,       # Default profit factor
                        'sharpe_ratio': 0.0,        # Default Sharpe ratio
                        'max_drawdown': 0.0,        # Default max drawdown
                        'expectancy': 0.0,          # Default expectancy
                        'trades_count': 0,          # No trades yet
                        'last_updated': datetime.now().timestamp()
                    }
                    
                    logger.info("Added strategy '%s' with weight %.2f", strategy_name, weight)
                    return True
                else:
                    logger.error("Strategy '%s' not found in strategy generator", strategy_name)
                    return False
            except Exception as e:
                logger.exception("Error adding strategy '%s': %s", strategy_name, str(e))
                return False
    
    def remove_strategy(self, strategy_name: str) -> bool:
        """
        Remove a strategy from the executor's active strategies.
        
        Args:
            strategy_name: Name of the strategy to remove
            
        Returns:
            bool: True if strategy was successfully removed, False otherwise
        """
        with self.execution_lock:
            if strategy_name in self.active_strategies:
                # Remove from active collections
                del self.active_strategies[strategy_name]
                self.strategy_weights.pop(strategy_name, None)
                self.strategy_metrics.pop(strategy_name, None)
                
                # Clear signal history
                if strategy_name in self.signal_history:
                    del self.signal_history[strategy_name]
                
                logger.info("Removed strategy '%s'", strategy_name)
                
                # Normalize remaining weights
                self._normalize_weights()
                
                return True
            else:
                logger.warning("Attempted to remove non-existent strategy '%s'", strategy_name)
                return False
    
    def update_strategy_weight(self, strategy_name: str, weight: float) -> bool:
        """
        Update the weight of a specific strategy.
        
        Args:
            strategy_name: Name of the strategy to update
            weight: New weight for the strategy (0.0 - 1.0)
            
        Returns:
            bool: True if weight was successfully updated, False otherwise
            
        Raises:
            ValueError: If weight is outside valid range
        """
        if not 0.0 <= weight <= 1.0:
            raise ValueError(f"Strategy weight must be between 0.0 and 1.0, got {weight}")
            
        with self.execution_lock:
            if strategy_name in self.active_strategies:
                self.strategy_weights[strategy_name] = weight
                
                # Normalize all weights
                self._normalize_weights()
                
                logger.info("Updated weight for strategy '%s' to %.2f", 
                           strategy_name, self.strategy_weights[strategy_name])
                return True
            else:
                logger.warning("Attempted to update weight for non-existent strategy '%s'", 
                              strategy_name)
                return False
    
    def _normalize_weights(self) -> None:
        """
        Normalize strategy weights to ensure they sum to 1.0.
        Called after adding, removing, or updating strategy weights.
        
        This maintains the relative importance of strategies while ensuring
        their combined weight doesn't exceed 100%.
        """
        if not self.strategy_weights:
            return
            
        # Get sum of weights
        total_weight = sum(self.strategy_weights.values())
        
        # Normalize if sum is not 0
        if total_weight > 0:
            for strategy_name in self.strategy_weights:
                self.strategy_weights[strategy_name] /= total_weight
        else:
            # Equal weights if all weights are 0
            equal_weight = 1.0 / len(self.strategy_weights)
            for strategy_name in self.strategy_weights:
                self.strategy_weights[strategy_name] = equal_weight
                
        logger.debug("Normalized strategy weights: %s", 
                    {k: round(v, 3) for k, v in self.strategy_weights.items()})
    
    def update_strategy_metrics(self, strategy_name: str, metrics: Dict[str, float]) -> bool:
        """
        Update performance metrics for a specific strategy.
        
        Args:
            strategy_name: Name of the strategy to update
            metrics: Dictionary of performance metrics
            
        Returns:
            bool: True if metrics were successfully updated, False otherwise
        """
        with self.execution_lock:
            if strategy_name in self.active_strategies:
                # Update metrics with validation
                for metric, value in metrics.items():
                    if metric in self.strategy_metrics[strategy_name]:
                        # Validate metric values
                        if metric == 'win_rate' and not 0 <= value <= 1:
                            logger.warning("Invalid win_rate value: %.2f, clamping to [0,1]", value)
                            value = max(0, min(1, value))
                        elif metric == 'max_drawdown' and value > 0:
                            # Max drawdown should be negative or zero
                            logger.warning("Max drawdown should be negative: %.2f, correcting", value)
                            value = -abs(value)
                            
                        self.strategy_metrics[strategy_name][metric] = value
                    
                # Update timestamp
                self.strategy_metrics[strategy_name]['last_updated'] = datetime.now().timestamp()
                
                # If risk scaling is enabled, recalculate weights based on performance
                if self.risk_scaling:
                    self._adjust_weights_by_performance()
                
                logger.info("Updated metrics for strategy '%s'", strategy_name)
                return True
            else:
                logger.warning("Attempted to update metrics for non-existent strategy '%s'", 
                              strategy_name)
                return False
    
    def _adjust_weights_by_performance(self) -> None:
        """
        Dynamically adjust strategy weights based on performance metrics.
        Strategies with better performance get higher weights.
        
        This method uses a composite score considering multiple performance factors:
        - Win rate (higher is better)
        - Profit factor (higher is better)
        - Sharpe ratio (higher is better)
        - Max drawdown (less negative is better)
        """
        if not self.active_strategies:
            return
            
        # Calculate performance scores
        performance_scores = {}
        for strategy_name, metrics in self.strategy_metrics.items():
            # Skip strategies with no trades
            if metrics.get('trades_count', 0) < 10:  # Minimum trades for reliable metrics
                performance_scores[strategy_name] = 0.5  # Neutral score
                continue
                
            # Composite score calculation (customize weights as needed)
            score = (
                0.3 * metrics.get('win_rate', 0.5) +
                0.3 * min(3.0, metrics.get('profit_factor', 1.0)) / 3.0 +
                0.2 * min(3.0, max(0, metrics.get('sharpe_ratio', 0))) / 3.0 +
                0.2 * (1.0 + min(0, metrics.get('max_drawdown', 0)))  # Convert drawdown to 0-1 scale
            )
            
            # Ensure score is in valid range
            performance_scores[strategy_name] = max(0.1, min(1.0, score))
        
        # Update weights based on performance scores
        if performance_scores:
            # Get total score
            total_score = sum(performance_scores.values())
            
            if total_score > 0:
                # Assign weights proportional to scores
                for strategy_name, score in performance_scores.items():
                    self.strategy_weights[strategy_name] = score / total_score
            
            logger.debug("Adjusted weights by performance: %s", 
                        {k: round(v, 3) for k, v in self.strategy_weights.items()})
    
    def execute_strategies(
        self, 
        data: pd.DataFrame, 
        symbol: str, 
        timeframe: str = '1h',
        market_state: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Execute all active strategies on given market data and return combined signals.
        
        This method orchestrates the full strategy execution process:
        1. Check if enough time has passed since last execution
        2. Update market regime if provided
        3. Execute each active strategy
        4. Combine signals with strategy weights
        5. Apply ML enhancements if enabled
        6. Calculate position size based on signal strength
        7. Apply risk management constraints
        
        Args:
            data: DataFrame containing market data (OHLCV and indicators)
            symbol: Trading symbol (e.g., 'BTC/USD')
            timeframe: Data timeframe (e.g., '1h', '4h', '1d')
            market_state: Optional dict with market regime and volatility info
            
        Returns:
            Dict containing execution results with keys:
                - 'signal': Combined weighted signal (-1 to 1)
                - 'position_size': Recommended position size
                - 'strategy_signals': Dict of individual strategy signals
                - 'confidence': Signal confidence score (0 to 1)
                - 'execution_time': Time taken to execute strategies
        
        Raises:
            ValueError: If data is empty or missing required columns
        """
        execution_start = time.time()
        logger.debug("Executing strategies for %s (%s)", symbol, timeframe)
        
        # Validate input data
        if data is None or data.empty:
            raise ValueError("Empty dataframe provided to strategy executor")
            
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        missing_columns = [col for col in required_columns if col not in data.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns in data: {missing_columns}")
        
        # Check if enough time has passed since last execution
        current_time = datetime.now()
        if (self.last_execution is not None and 
            (current_time - self.last_execution).total_seconds() < self.execution_interval):
            logger.debug("Skipping execution due to interval constraints")
            return {
                'signal': 0,
                'position_size': 0,
                'strategy_signals': {},
                'confidence': 0,
                'execution_time': 0,
                'status': 'skipped_interval'
            }
        
        with self.execution_lock:
            try:
                # Update market regime if provided
                if market_state is not None and 'market_regime' in market_state:
                    self.market_regime = market_state['market_regime']
                    
                # Execute each active strategy
                strategy_signals = {}
                for strategy_name, strategy in self.active_strategies.items():
                    start_time = time.time()
                    try:
                        # Get strategy parameters from optimizer if available
                        params = self._get_strategy_parameters(strategy_name, market_state)
                        
                        # Execute strategy
                        if hasattr(strategy, '__call__'):
                            # Strategy is a function
                            signals = strategy(data, params)
                        else:
                            # Strategy is an object with execute method
                            signals = strategy(data, params)
                            
                        # Extract signal from last row
                        if isinstance(signals, pd.DataFrame) and 'position' in signals.columns:
                            signal_value = float(signals['position'].iloc[-1])
                        else:
                            logger.warning("Strategy '%s' returned invalid signal format", 
                                          strategy_name)
                            signal_value = 0
                        
                        # Apply signal smoothing
                        smoothed_signal = self._apply_signal_smoothing(strategy_name, signal_value)
                        
                        # Store signal
                        strategy_signals[strategy_name] = {
                            'raw_signal': signal_value,
                            'smoothed_signal': smoothed_signal,
                            'weight': self.strategy_weights.get(strategy_name, 0),
                            'execution_time': time.time() - start_time
                        }
                        
                    except Exception as e:
                        logger.exception("Error executing strategy '%s': %s", 
                                       strategy_name, str(e))
                        strategy_signals[strategy_name] = {
                            'raw_signal': 0,
                            'smoothed_signal': 0,
                            'weight': 0,
                            'execution_time': time.time() - start_time,
                            'error': str(e)
                        }
                
                # Combine signals using weighted average
                combined_signal, signal_confidence = self._combine_strategy_signals(strategy_signals)
                
                # Apply ML enhancement if enabled
                if self.ml_enhancement and market_state is not None:
                    combined_signal, signal_confidence = self._enhance_signal_with_ml(
                        combined_signal, signal_confidence, market_state
                    )
                
                # Calculate position size based on signal strength and confidence
                position_size = self._calculate_position_size(
                    combined_signal, signal_confidence, market_state
                )
                
                # Apply risk management constraints
                position_size = self._apply_risk_constraints(
                    position_size, combined_signal, symbol, market_state
                )
                
                # Record execution time
                execution_time = time.time() - execution_start
                self.execution_times.append(execution_time)
                if len(self.execution_times) > 100:
                    self.execution_times.pop(0)
                
                # Update last execution time
                self.last_execution = current_time
                
                logger.info(
                    "Executed strategies for %s: signal=%.2f, position_size=%.4f, confidence=%.2f",
                    symbol, combined_signal, position_size, signal_confidence
                )
                
                return {
                    'signal': combined_signal,
                    'position_size': position_size,
                    'strategy_signals': strategy_signals,
                    'confidence': signal_confidence,
                    'execution_time': execution_time,
                    'status': 'success'
                }
                
            except Exception as e:
                logger.exception("Error in strategy execution: %s", str(e))
                return {
                    'signal': 0,
                    'position_size': 0,
                    'strategy_signals': {},
                    'confidence': 0,
                    'execution_time': time.time() - execution_start,
                    'status': 'error',
                    'error': str(e)
                }
    
    def _get_strategy_parameters(
        self, 
        strategy_name: str,
        market_state: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Get optimized parameters for a specific strategy.
        
        Args:
            strategy_name: Name of the strategy
            market_state: Optional market state information
            
        Returns:
            Dict of strategy parameters
        """
        # Check if optimizer has parameters for this strategy
        if (hasattr(self.strategy_optimizer, 'best_params') and 
            strategy_name in self.strategy_optimizer.best_params):
            return self.strategy_optimizer.best_params[strategy_name]
        
        # If strategy_optimizer has a get_strategy_parameters method, use it
        if hasattr(self.strategy_optimizer, 'get_strategy_parameters'):
            return self.strategy_optimizer.get_strategy_parameters(strategy_name, market_state)
        
        # Default parameters based on strategy type
        if strategy_name == 'trend_following':
            return {
                'short_window': 10,
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
        elif strategy_name == 'breakout':
            return {
                'window': 50,
                'volume_window': 10,
                'volume_threshold': 1.5,
                'stop_loss_pct': 0.03,
                'take_profit_pct': 0.06
            }
        else:
            # Generic default parameters
            return {
                'window': 20,
                'threshold': 1.5,
                'stop_loss_pct': 0.02,
                'take_profit_pct': 0.05
            }
    
    def _apply_signal_smoothing(self, strategy_name: str, signal: float) -> float:
        """
        Apply signal smoothing to reduce noise and false signals.
        
        Args:
            strategy_name: Name of the strategy
            signal: Raw signal value
            
        Returns:
            Smoothed signal value
        """
        # Add signal to history
        self.signal_history[strategy_name].append(signal)
        
        # Apply exponential smoothing
        if len(self.signal_history[strategy_name]) > 1:
            alpha = 0.3  # Smoothing factor
            history = list(self.signal_history[strategy_name])
            
            # Apply more weight to recent signals
            weights = [alpha * (1 - alpha) ** i for i in range(len(history))]
            weights.reverse()  # Most recent gets highest weight
            
            # Normalize weights
            weights_sum = sum(weights)
            weights = [w / weights_sum for w in weights]
            
            # Weighted average
            smoothed_signal = sum(s * w for s, w in zip(history, weights))
            return smoothed_signal
        else:
            return signal
    
    def _combine_strategy_signals(
        self, 
        strategy_signals: Dict[str, Dict[str, Any]]
    ) -> Tuple[float, float]:
        """
        Combine signals from multiple strategies using weighted average.
        
        Args:
            strategy_signals: Dict mapping strategy names to their signal data
            
        Returns:
            Tuple of (combined signal, confidence)
        """
        if not strategy_signals:
            return 0.0, 0.0
            
        weighted_sum = 0.0
        total_weight = 0.0
        signal_agreement = 0.0  # Measure of signal agreement
        valid_signals = 0
        
        for strategy_name, signal_data in strategy_signals.items():
            weight = signal_data['weight']
            signal = signal_data['smoothed_signal']
            
            if weight > 0:
                weighted_sum += signal * weight
                total_weight += weight
                valid_signals += 1
                
        # If no valid signals, return neutral
        if total_weight == 0 or valid_signals == 0:
            return 0.0, 0.0
            
        # Calculate combined signal
        combined_signal = weighted_sum / total_weight
        
        # Calculate signal agreement/confidence
        for strategy_name, signal_data in strategy_signals.items():
            if signal_data['weight'] > 0:
                # How much does this signal agree with the combined signal
                agreement = 1.0 - (abs(signal_data['smoothed_signal'] - combined_signal) / 2.0)
                signal_agreement += agreement * signal_data['weight']
                
        confidence = signal_agreement / total_weight
        
        # Add signal strength to confidence
        signal_strength = abs(combined_signal)
        confidence = (confidence + signal_strength) / 2.0
        
        return combined_signal, confidence
    
    def _enhance_signal_with_ml(
        self, 
        signal: float, 
        confidence: float,
        market_state: Dict[str, Any]
    ) -> Tuple[float, float]:
        """
        Enhance trading signal using ML predictions if available.
        
        Args:
            signal: Combined strategy signal
            confidence: Signal confidence
            market_state: Market state information including ML predictions
            
        Returns:
            Tuple of (enhanced signal, enhanced confidence)
        """
        # Return original signal if no ML predictions
        if 'ml_predictions' not in market_state:
            return signal, confidence
            
        ml_predictions = market_state['ml_predictions']
        
        # Extract relevant predictions
        direction_prediction = ml_predictions.get('direction', 0)  # -1 to 1
        timing_quality = ml_predictions.get('timing_quality', 0.5)  # 0 to 1
        
        # Combine strategy signal with ML prediction
        enhanced_signal = 0.7 * signal + 0.3 * direction_prediction
        
        # Clamp to valid range
        enhanced_signal = max(-1.0, min(1.0, enhanced_signal))
        
        # Enhance confidence based on timing quality
        enhanced_confidence = 0.7 * confidence + 0.3 * timing_quality
        
        return enhanced_signal, enhanced_confidence
    
    def _calculate_position_size(
        self, 
        signal: float, 
        confidence: float,
        market_state: Optional[Dict[str, Any]] = None
    ) -> float:
        """
        Calculate position size based on signal strength, confidence, and market state.
        
        Args:
            signal: Trading signal (-1 to 1)
            confidence: Signal confidence (0 to 1)
            market_state: Optional market state information
            
        Returns:
            Position size as a fraction of portfolio
        """
        # Base position size from configuration
        base_size = self.position_sizing
        
        # Scale by signal strength
        signal_strength = abs(signal)
        
        # Scale by confidence
        position_size = base_size * signal_strength * confidence
        
        # Adjust for market regime
        if market_state and 'market_regime' in market_state:
            regime = market_state['market_regime']
            
            # Reduce size in high volatility regimes
            if 'volatile' in regime:
                position_size *= 0.7
            # Reduce size in uncertain regimes
            elif regime == 'ranging' or regime == 'unclear':
                position_size *= 0.8
                
        # Adjust for market volatility if available
        if market_state and 'volatility' in market_state:
            volatility = market_state['volatility']
            volatility_factor = 1.0
            
            if volatility > 1.5:  # High volatility
                volatility_factor = 0.7
            elif volatility > 1.0:  # Above average
                volatility_factor = 0.85
            
            position_size *= volatility_factor
        
        # Ensure minimum position size if signal is non-zero
        if signal != 0 and position_size > 0:
            position_size = max(0.001, position_size)  # Minimum position size
            
        # Cap maximum position size
        position_size = min(position_size, 0.2)  # Maximum 20% of portfolio
        
        return position_size
    
    def _apply_risk_constraints(
        self, 
        position_size: float, 
        signal: float,
        symbol: str,
        market_state: Optional[Dict[str, Any]] = None
    ) -> float:
        """
        Apply risk management constraints to position size.
        
        Args:
            position_size: Calculated position size
            signal: Trading signal (-1 to 1)
            symbol: Trading symbol
            market_state: Optional market state information
            
        Returns:
            Adjusted position size
        """
        # Return zero if signal is zero
        if signal == 0:
            return 0.0
            
        # Check if position manager is available
        if self.position_manager is not None:
            # Check current exposure
            try:
                current_exposure = self.position_manager.get_total_exposure()
                max_exposure = 0.8  # Maximum total exposure
                
                # If adding this position would exceed max exposure, reduce size
                if current_exposure + position_size > max_exposure:
                    available_exposure = max(0, max_exposure - current_exposure)
                    position_size = min(position_size, available_exposure)
                    logger.debug("Reduced position size due to exposure constraint: %.4f", 
                                position_size)
            except Exception as e:
                logger.warning("Error checking exposure, using original position size: %s", str(e))
                
            # Check existing position for this symbol
            try:
                symbol_position = self.position_manager.get_position(symbol)
                
                # If signal direction is opposite to current position, allow full size
                if symbol_position and ((symbol_position['side'] == 'long' and signal < 0) or
                                        (symbol_position['side'] == 'short' and signal > 0)):
                    return position_size
                    
                # If adding to existing position, check position limits
                if symbol_position and ((symbol_position['side'] == 'long' and signal > 0) or
                                        (symbol_position['side'] == 'short' and signal < 0)):
                    max_symbol_exposure = 0.3  # Maximum exposure per symbol
                    current_symbol_exposure = abs(symbol_position['size'])
                    
                    if current_symbol_exposure + position_size > max_symbol_exposure:
                        available_exposure = max(0, max_symbol_exposure - current_symbol_exposure)
                        position_size = min(position_size, available_exposure)
                        logger.debug("Reduced position size due to symbol exposure constraint: %.4f", 
                                    position_size)
            except Exception as e:
                logger.warning("Error checking symbol position, using original position size: %s", 
                              str(e))
        
        # Apply risk factor based on market state
        if market_state:
            # Reduce size during high-risk periods
            if market_state.get('manipulation_detected', False):
                position_size *= 0.5
                logger.debug("Reduced position size due to manipulation detection: %.4f", 
                            position_size)
                
            # Reduce size during earnings/news events
            if market_state.get('high_impact_event', False):
                position_size *= 0.7
                logger.debug("Reduced position size due to high impact event: %.4f", 
                            position_size)
        
        return position_size
    
    def get_active_strategies(self) -> Dict[str, float]:
        """
        Get active strategies with their weights.
        
        Returns:
            Dict mapping strategy names to their weights
        """
        with self.execution_lock:
            # Return a copy to avoid external modification
            return {k: v for k, v in self.strategy_weights.items() if k in self.active_strategies}
    
    def get_performance_metrics(self) -> Dict[str, Dict[str, float]]:
        """
        Get performance metrics for all strategies.
        
        Returns:
            Dict mapping strategy names to their performance metrics
        """
        with self.execution_lock:
            # Return a copy to avoid external modification
            return {k: v.copy() for k, v in self.strategy_metrics.items() if k in self.active_strategies}
    
    def get_execution_stats(self) -> Dict[str, Any]:
        """
        Get execution statistics.
        
        Returns:
            Dict containing execution statistics
        """
        with self.execution_lock:
            avg_time = sum(self.execution_times) / len(self.execution_times) if self.execution_times else 0
            return {
                'average_execution_time': avg_time,
                'win_rate': self.win_count / (self.win_count + self.loss_count) if (self.win_count + self.loss_count) > 0 else 0,
                'total_trades': self.win_count + self.loss_count,
                'cumulative_pnl': self.cumulative_pnl,
            }
    
    def update_trade_result(self, trade_result: Dict[str, Any]) -> None:
        """
        Update executor with trade result for performance tracking.
        
        Args:
            trade_result: Dict containing trade result information
        """
        with self.execution_lock:
            # Extract relevant information
            strategy_name = trade_result.get('strategy', None)
            pnl = trade_result.get('pnl', 0)
            is_win = pnl > 0
            
            # Update overall stats
            self.cumulative_pnl += pnl
            if is_win:
                self.win_count += 1
            else:
                self.loss_count += 1
                
            # Update strategy metrics if strategy is specified
            if strategy_name and strategy_name in self.strategy_metrics:
                metrics = self.strategy_metrics[strategy_name]
                
                # Update trades count
                metrics['trades_count'] = metrics.get('trades_count', 0) + 1
                
                # Update win rate
                total_trades = metrics['trades_count']
                old_wins = metrics.get('win_rate', 0.5) * (total_trades - 1)
                new_wins = old_wins + (1 if is_win else 0)
                metrics['win_rate'] = new_wins / total_trades if total_trades > 0 else 0.5
                
                # Other metrics would typically be calculated in a more sophisticated way
                # with a full performance evaluation
                
                logger.debug("Updated metrics for strategy '%s' after trade: win_rate=%.2f, trades=%d", 
                           strategy_name, metrics['win_rate'], metrics['trades_count'])
