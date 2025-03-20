# market_analysis.py

import numpy as np
import pandas as pd
import logging
import time
import threading
from typing import Dict, List, Optional, Tuple, Union, Any
from enum import Enum
from datetime import datetime, timedelta
import traceback
from scipy import stats
from sklearn.cluster import DBSCAN
import talib
from talib import abstract
import matplotlib.pyplot as plt
from concurrent.futures import ThreadPoolExecutor

# Try to import error handling module for consistent error management
try:
    from core.error_handling import safe_execute, ErrorCategory, ErrorSeverity, ErrorHandler
    HAVE_ERROR_HANDLING = True
except ImportError:
    HAVE_ERROR_HANDLING = False
    logging.warning("Error handling module not available. Using basic error handling.")

# Define market regime enum for consistent classification
class MarketRegime(Enum):
    STRONG_BULL = "strong_bull"
    BULL = "bull"
    SIDEWAYS = "sideways"
    BEAR = "bear"
    STRONG_BEAR = "strong_bear"
    VOLATILE_BULL = "volatile_bull"
    VOLATILE_BEAR = "volatile_bear"
    VOLATILE_SIDEWAYS = "volatile_sideways"
    UNKNOWN = "unknown"

class MarketStructure(Enum):
    UPTREND = "uptrend"
    DOWNTREND = "downtrend"
    RANGE = "range"
    ACCUMULATION = "accumulation"
    DISTRIBUTION = "distribution"
    EXPANSION = "expansion"
    CONTRACTION = "contraction"
    UNKNOWN = "unknown"

class SupportResistanceType(Enum):
    SUPPORT = "support"
    RESISTANCE = "resistance"
    MAJOR_SUPPORT = "major_support"
    MAJOR_RESISTANCE = "major_resistance"
    DYNAMIC_SUPPORT = "dynamic_support"
    DYNAMIC_RESISTANCE = "dynamic_resistance"

class PatternType(Enum):
    DOUBLE_TOP = "double_top"
    DOUBLE_BOTTOM = "double_bottom"
    HEAD_AND_SHOULDERS = "head_and_shoulders"
    INV_HEAD_AND_SHOULDERS = "inverse_head_and_shoulders"
    TRIANGLE = "triangle"
    WEDGE = "wedge"
    CHANNEL = "channel"
    FLAG = "flag"
    NONE = "none"

class MarketAnalysis:
    """
    Comprehensive market analysis system for detecting regimes, patterns, and generating trading signals.
    Integrates with ML components and provides inputs for strategy systems and risk management.
    """
    
    def __init__(self, database_manager=None, thread_manager=None, risk_manager=None, config=None):
        """
        Initialize the market analysis system.
        
        Args:
            database_manager: Database for storing and retrieving market data
            thread_manager: Thread manager for parallel processing
            risk_manager: Risk management system for risk adjustments
            config: Configuration dictionary with analysis parameters
        """
        self.db = database_manager
        self.thread_manager = thread_manager
        self.risk_manager = risk_manager
        self.config = config or {}
        
        # Default configuration
        self.config.setdefault("lookback_periods", {
            "short": 20,     # Short-term analysis (1-5 days)
            "medium": 50,    # Medium-term analysis (1-4 weeks)
            "long": 200      # Long-term analysis (1-6 months)
        })
        
        self.config.setdefault("volatility_thresholds", {
            "low": 0.5,      # Low volatility threshold (relative to historical)
            "medium": 1.0,   # Normal volatility
            "high": 2.0      # High volatility threshold
        })
        
        self.config.setdefault("support_resistance", {
            "sensitivity": 0.03,  # Price level sensitivity (3% of price)
            "cluster_distance": 0.01,  # Distance for clustering levels (1% of price)
            "strength_threshold": 3,  # Minimum touches to consider significant
            "lookback_days": 90    # Days to look back for S/R levels
        })
        
        # Market regimes for each analyzed symbol
        self.market_regimes = {}
        self.market_structures = {}
        self.volatility_levels = {}
        self.sr_levels = {}
        self.patterns = {}
        
        # Cache for analysis results to avoid redundant calculations
        self.analysis_cache = {}
        self.cache_expiry = {}
        self.cache_timeout = 300  # 5 minutes
        
        # Thread safety
        self._lock = threading.RLock()
        
        # Technical indicators used for analysis
        self.indicators = [
            'SMA', 'EMA', 'RSI', 'MACD', 'BBANDS', 'ATR', 'OBV', 
            'STOCH', 'ADX', 'CCI', 'MOM', 'ROC', 'WILLR'
        ]
        
        logging.info("Market analysis system initialized")
    
    def analyze_market(self, symbol: str, timeframe: str = '1h', update_cache: bool = False) -> Dict[str, Any]:
        """
        Perform comprehensive market analysis for a symbol.
        
        Args:
            symbol: Trading pair symbol (e.g., "BTC/USD")
            timeframe: Data timeframe (e.g., '1h', '4h', '1d')
            update_cache: Force update of cached analysis
            
        Returns:
            Dictionary with comprehensive market analysis results
        """
        cache_key = f"{symbol}_{timeframe}"
        
        # Check if we have cached results
        if not update_cache and cache_key in self.analysis_cache:
            cache_time = self.cache_expiry.get(cache_key, 0)
            if time.time() < cache_time:
                return self.analysis_cache[cache_key]
        
        try:
            # Get market data
            data = self._get_market_data(symbol, timeframe)
            if data is None or data.empty:
                logging.warning(f"No data available for {symbol} on {timeframe} timeframe")
                return self._generate_empty_analysis(symbol, timeframe)
            
            # Process market data to ensure all required columns exist
            data = self._preprocess_data(data)
            
            # Perform analysis components
            with ThreadPoolExecutor(max_workers=3) as executor:
                # Start parallel analysis tasks
                regime_future = executor.submit(self._analyze_market_regime, data, symbol)
                structure_future = executor.submit(self._analyze_market_structure, data, symbol)
                volatility_future = executor.submit(self._analyze_volatility, data, symbol)
                
                # Get results
                market_regime = regime_future.result()
                market_structure = structure_future.result()
                volatility_info = volatility_future.result()
            
            # Sequential analyses that depend on previous results
            sr_levels = self._identify_support_resistance(data, symbol)
            patterns = self._detect_chart_patterns(data, symbol)
            signals = self._generate_signals(data, market_regime, market_structure, volatility_info, sr_levels, patterns)
            
            # Calculate additional metrics
            metrics = self._calculate_market_metrics(data, symbol)
            
            # Combine all analysis components
            analysis = {
                "symbol": symbol,
                "timeframe": timeframe,
                "timestamp": datetime.now().isoformat(),
                "last_price": float(data['close'].iloc[-1]) if 'close' in data else None,
                "market_regime": market_regime,
                "market_structure": market_structure,
                "volatility": volatility_info,
                "support_resistance": sr_levels,
                "patterns": patterns,
                "signals": signals,
                "metrics": metrics
            }
            
            # Cache the results
            with self._lock:
                self.analysis_cache[cache_key] = analysis
                self.cache_expiry[cache_key] = time.time() + self.cache_timeout
                
                # Update internal state
                self.market_regimes[symbol] = market_regime
                self.market_structures[symbol] = market_structure
                self.volatility_levels[symbol] = volatility_info
                self.sr_levels[symbol] = sr_levels
                self.patterns[symbol] = patterns
            
            return analysis
            
        except Exception as e:
            logging.error(f"Error analyzing market for {symbol}: {str(e)}\n{traceback.format_exc()}")
            return self._generate_empty_analysis(symbol, timeframe)
    
    def get_market_regime(self, symbol: str) -> str:
        """
        Get the current market regime for a symbol.
        
        Args:
            symbol: Trading pair symbol
            
        Returns:
            Market regime string (e.g., "bull", "bear", "sideways")
        """
        return self.market_regimes.get(symbol, MarketRegime.UNKNOWN.value)
    
    def get_volatility_level(self, symbol: str) -> Dict[str, Any]:
        """
        Get the volatility level for a symbol.
        
        Args:
            symbol: Trading pair symbol
            
        Returns:
            Volatility information dictionary
        """
        return self.volatility_levels.get(symbol, {"level": "unknown", "value": 1.0})
    
    def get_support_resistance_levels(self, symbol: str) -> Dict[str, List[Dict[str, Any]]]:
        """
        Get support and resistance levels for a symbol.
        
        Args:
            symbol: Trading pair symbol
            
        Returns:
            Dictionary with support and resistance levels
        """
        return self.sr_levels.get(symbol, {"support": [], "resistance": []})
    
    def get_key_price_levels(self, symbol: str, current_price: float) -> Dict[str, float]:
        """
        Get key price levels for a symbol relative to current price.
        
        Args:
            symbol: Trading pair symbol
            current_price: Current market price
            
        Returns:
            Dictionary with key price levels
        """
        sr_levels = self.get_support_resistance_levels(symbol)
        
        # Find nearest support and resistance
        nearest_support = None
        nearest_resistance = None
        support_distance = float('inf')
        resistance_distance = float('inf')
        
        for level in sr_levels.get("support", []):
            price = level.get("price", 0)
            if current_price > price and (current_price - price) < support_distance:
                support_distance = current_price - price
                nearest_support = price
        
        for level in sr_levels.get("resistance", []):
            price = level.get("price", 0)
            if current_price < price and (price - current_price) < resistance_distance:
                resistance_distance = price - current_price
                nearest_resistance = price
        
        return {
            "current_price": current_price,
            "nearest_support": nearest_support,
            "nearest_resistance": nearest_resistance,
            "support_distance_pct": (support_distance / current_price) * 100 if nearest_support else None,
            "resistance_distance_pct": (resistance_distance / current_price) * 100 if nearest_resistance else None
        }
    
    def is_near_key_level(self, symbol: str, current_price: float, threshold_pct: float = 0.5) -> Dict[str, bool]:
        """
        Check if price is near a key support or resistance level.
        
        Args:
            symbol: Trading pair symbol
            current_price: Current market price
            threshold_pct: Threshold percentage to consider "near" (e.g., 0.5%)
            
        Returns:
            Dictionary indicating proximity to key levels
        """
        key_levels = self.get_key_price_levels(symbol, current_price)
        
        near_support = False
        near_resistance = False
        
        if key_levels["nearest_support"] is not None:
            support_distance_pct = key_levels["support_distance_pct"]
            if support_distance_pct <= threshold_pct:
                near_support = True
        
        if key_levels["nearest_resistance"] is not None:
            resistance_distance_pct = key_levels["resistance_distance_pct"]
            if resistance_distance_pct <= threshold_pct:
                near_resistance = True
        
        return {
            "near_support": near_support,
            "near_resistance": near_resistance,
            "support_level": key_levels["nearest_support"],
            "resistance_level": key_levels["nearest_resistance"]
        }
    
    def calculate_optimal_stop_loss(self, symbol: str, entry_price: float, position_type: str) -> Dict[str, float]:
        """
        Calculate optimal stop loss levels based on market analysis.
        
        Args:
            symbol: Trading pair symbol
            entry_price: Trade entry price
            position_type: 'long' or 'short'
            
        Returns:
            Dictionary with stop loss levels and recommendations
        """
        try:
            # Get market analysis components
            volatility = self.get_volatility_level(symbol)
            sr_levels = self.get_support_resistance_levels(symbol)
            
            # Get latest ATR for volatility-based stop
            data = self._get_market_data(symbol, '1h', limit=50)
            if data is None or data.empty:
                return {"error": "No data available"}
            
            # Calculate ATR if not already in data
            if 'atr' not in data.columns:
                if len(data) >= 14:  # Minimum required for ATR calculation
                    data['atr'] = talib.ATR(data['high'].values, data['low'].values, 
                                            data['close'].values, timeperiod=14)
            
            atr_value = data['atr'].iloc[-1] if 'atr' in data.columns and not pd.isna(data['atr'].iloc[-1]) else entry_price * 0.01
            
            # Volatility-based stop (ATR multiple)
            atr_multiple = 2.0  # Default
            
            # Adjust based on volatility
            vol_level = volatility.get("level", "medium")
            if vol_level == "low":
                atr_multiple = 1.5
            elif vol_level == "high":
                atr_multiple = 3.0
            
            # Calculate volatility-based stop
            if position_type.lower() == 'long':
                vol_stop = entry_price - (atr_value * atr_multiple)
            else:
                vol_stop = entry_price + (atr_value * atr_multiple)
            
            # Support/resistance based stop
            sr_stop = None
            
            if position_type.lower() == 'long':
                # For long positions, find nearest support below entry
                supports = [level["price"] for level in sr_levels.get("support", [])]
                supports = [s for s in supports if s < entry_price]
                if supports:
                    sr_stop = max(supports)  # Highest support below entry
            else:
                # For short positions, find nearest resistance above entry
                resistances = [level["price"] for level in sr_levels.get("resistance", [])]
                resistances = [r for r in resistances if r > entry_price]
                if resistances:
                    sr_stop = min(resistances)  # Lowest resistance above entry
            
            # Structure-based stop (use swing high/low)
            structure_stop = None
            
            # Get recent swing points
            data_tail = data.tail(50)  # Use last 50 candles
            if position_type.lower() == 'long':
                # Find recent swing low
                swing_lows = self._find_swing_points(data_tail, swing_type='low', window=5)
                if swing_lows:
                    structure_stop = max([point["price"] for point in swing_lows if point["price"] < entry_price], default=None)
            else:
                # Find recent swing high
                swing_highs = self._find_swing_points(data_tail, swing_type='high', window=5)
                if swing_highs:
                    structure_stop = min([point["price"] for point in swing_highs if point["price"] > entry_price], default=None)
            
            # Combine approaches for final recommendation
            recommended_stop = vol_stop  # Default to volatility-based
            
            # If we have S/R based stop, consider it
            if sr_stop is not None:
                if position_type.lower() == 'long' and sr_stop > vol_stop:
                    recommended_stop = vol_stop  # More conservative
                elif position_type.lower() == 'short' and sr_stop < vol_stop:
                    recommended_stop = vol_stop  # More conservative
                else:
                    recommended_stop = sr_stop
            
            # If we have structure-based stop and it's more conservative, use it
            if structure_stop is not None:
                if position_type.lower() == 'long' and structure_stop > recommended_stop:
                    recommended_stop = structure_stop
                elif position_type.lower() == 'short' and structure_stop < recommended_stop:
                    recommended_stop = structure_stop
            
            # Calculate stop distances
            stop_distance = abs(recommended_stop - entry_price)
            stop_distance_pct = (stop_distance / entry_price) * 100
            
            return {
                "recommended_stop": recommended_stop,
                "volatility_stop": vol_stop,
                "sr_stop": sr_stop,
                "structure_stop": structure_stop,
                "stop_distance": stop_distance,
                "stop_distance_pct": stop_distance_pct,
                "atr_value": atr_value
            }
            
        except Exception as e:
            logging.error(f"Error calculating optimal stop loss: {str(e)}\n{traceback.format_exc()}")
            
            # Fallback to simple percentage-based stop
            fallback_pct = 0.02  # 2% default stop
            if position_type.lower() == 'long':
                stop_price = entry_price * (1 - fallback_pct)
            else:
                stop_price = entry_price * (1 + fallback_pct)
                
            return {
                "recommended_stop": stop_price,
                "stop_distance_pct": fallback_pct * 100,
                "error": f"Used fallback calculation due to error: {str(e)}"
            }
    
    def calculate_optimal_take_profit(self, symbol: str, entry_price: float, position_type: str, 
                                     stop_loss: Optional[float] = None) -> Dict[str, Any]:
        """
        Calculate optimal take profit levels based on market analysis.
        
        Args:
            symbol: Trading pair symbol
            entry_price: Trade entry price
            position_type: 'long' or 'short'
            stop_loss: Stop loss price (optional)
            
        Returns:
            Dictionary with take profit levels and recommendations
        """
        try:
            # Get market analysis components
            volatility = self.get_volatility_level(symbol)
            sr_levels = self.get_support_resistance_levels(symbol)
            market_regime = self.get_market_regime(symbol)
            
            # Calculate risk-reward based targets
            risk_reward_targets = {}
            
            if stop_loss is not None:
                # Calculate risk in price terms
                risk = abs(entry_price - stop_loss)
                
                # Calculate targets for different R:R ratios
                for rr in [1, 1.5, 2, 3, 5]:
                    if position_type.lower() == 'long':
                        target = entry_price + (risk * rr)
                    else:
                        target = entry_price - (risk * rr)
                    
                    risk_reward_targets[f"rr_{rr}"] = target
            
            # Support/resistance based targets
            sr_target = None
            
            if position_type.lower() == 'long':
                # For long positions, find resistances above entry
                resistances = [level["price"] for level in sr_levels.get("resistance", [])]
                resistances = [r for r in resistances if r > entry_price]
                if resistances:
                    sr_target = min(resistances)  # Closest resistance above entry
            else:
                # For short positions, find supports below entry
                supports = [level["price"] for level in sr_levels.get("support", [])]
                supports = [s for s in supports if s < entry_price]
                if supports:
                    sr_target = max(supports)  # Closest support below entry
            
            # Adjust based on market regime
            regime_factor = 1.0  # Default
            
            if "bull" in market_regime and position_type.lower() == 'long':
                regime_factor = 1.5  # More optimistic targets in bull market for longs
            elif "bear" in market_regime and position_type.lower() == 'short':
                regime_factor = 1.5  # More optimistic targets in bear market for shorts
            elif "sideways" in market_regime:
                regime_factor = 0.8  # More conservative in sideways markets
            
            # Get volatility adjusted target
            vol_level = volatility.get("value", 1.0)
            vol_target_pct = 0.05 * vol_level * regime_factor  # Base 5% adjusted by volatility and regime
            
            if position_type.lower() == 'long':
                vol_target = entry_price * (1 + vol_target_pct)
            else:
                vol_target = entry_price * (1 - vol_target_pct)
            
            # Determine recommended target
            recommended_target = vol_target  # Default
            
            # If we have S/R based target, consider it
            if sr_target is not None:
                if position_type.lower() == 'long':
                    # For longs, take the lesser of volatility and S/R targets (more conservative)
                    recommended_target = min(vol_target, sr_target)
                else:
                    # For shorts, take the greater of volatility and S/R targets (more conservative)
                    recommended_target = max(vol_target, sr_target)
            
            # Check if any R:R target is close to S/R level (within 2%)
            rr_target_key = None
            if sr_target is not None and risk_reward_targets:
                for key, target in risk_reward_targets.items():
                    if abs(target - sr_target) / sr_target < 0.02:  # Within 2%
                        rr_target_key = key
                        recommended_target = target  # Use R:R target that aligns with S/R
                        break
            
            # Calculate target distance
            target_distance = abs(recommended_target - entry_price)
            target_distance_pct = (target_distance / entry_price) * 100
            
            # Calculate reward-risk ratio if stop_loss provided
            reward_risk_ratio = None
            if stop_loss is not None:
                risk = abs(entry_price - stop_loss)
                reward = abs(recommended_target - entry_price)
                reward_risk_ratio = reward / risk if risk > 0 else None
            
            return {
                "recommended_target": recommended_target,
                "volatility_target": vol_target,
                "sr_target": sr_target,
                "risk_reward_targets": risk_reward_targets,
                "rr_target_aligned": rr_target_key,
                "target_distance": target_distance,
                "target_distance_pct": target_distance_pct,
                "reward_risk_ratio": reward_risk_ratio,
                "market_regime": market_regime
            }
            
        except Exception as e:
            logging.error(f"Error calculating optimal take profit: {str(e)}\n{traceback.format_exc()}")
            
            # Fallback to simple percentage-based target
            fallback_pct = 0.05  # 5% default target
            if position_type.lower() == 'long':
                target_price = entry_price * (1 + fallback_pct)
            else:
                target_price = entry_price * (1 - fallback_pct)
                
            return {
                "recommended_target": target_price,
                "target_distance_pct": fallback_pct * 100,
                "error": f"Used fallback calculation due to error: {str(e)}"
            }
    
    def get_trading_recommendation(self, symbol: str, timeframe: str = '1h') -> Dict[str, Any]:
        """
        Get a comprehensive trading recommendation based on market analysis.
        
        Args:
            symbol: Trading pair symbol
            timeframe: Data timeframe
            
        Returns:
            Dictionary with trading recommendations
        """
        # Get full market analysis
        analysis = self.analyze_market(symbol, timeframe)
        
        # Extract key components
        market_regime = analysis.get("market_regime", "unknown")
        signals = analysis.get("signals", {})
        
        # Determine overall recommendation
        signal_strength = signals.get("overall_signal", 0)
        
        if signal_strength > 0.7:
            action = "strong_buy"
        elif signal_strength > 0.3:
            action = "buy"
        elif signal_strength < -0.7:
            action = "strong_sell"
        elif signal_strength < -0.3:
            action = "sell"
        else:
            action = "hold"
        
        # Get current price
        current_price = analysis.get("last_price")
        
        # Calculate stop loss if we have a directional recommendation
        stop_loss = None
        take_profit = None
        
        if action in ["buy", "strong_buy"]:
            stop_loss_data = self.calculate_optimal_stop_loss(symbol, current_price, "long")
            stop_loss = stop_loss_data.get("recommended_stop")
            take_profit_data = self.calculate_optimal_take_profit(symbol, current_price, "long", stop_loss)
            take_profit = take_profit_data.get("recommended_target")
        elif action in ["sell", "strong_sell"]:
            stop_loss_data = self.calculate_optimal_stop_loss(symbol, current_price, "short")
            stop_loss = stop_loss_data.get("recommended_stop")
            take_profit_data = self.calculate_optimal_take_profit(symbol, current_price, "short", stop_loss)
            take_profit = take_profit_data.get("recommended_target")
        
        # Calculate risk-reward ratio
        risk_reward = None
        if stop_loss is not None and take_profit is not None:
            risk = abs(current_price - stop_loss)
            reward = abs(take_profit - current_price)
            risk_reward = reward / risk if risk > 0 else None
        
        # Adjust position size based on market conditions
        position_size_factor = 1.0  # Default
        
        # More conservative in volatile markets
        volatility = analysis.get("volatility", {})
        vol_level = volatility.get("level", "medium")
        
        if vol_level == "high":
            position_size_factor *= 0.7  # Reduce position size in high volatility
        elif vol_level == "low":
            position_size_factor *= 1.2  # Increase position size in low volatility
        
        # More conservative in counter-trend trades
        if ("bull" in market_regime and action in ["sell", "strong_sell"]) or \
           ("bear" in market_regime and action in ["buy", "strong_buy"]):
            position_size_factor *= 0.8  # Reduce size for counter-trend trades
        
        # Recommended position size
        base_position_size = 0.02  # Base 2% of capital
        recommended_position_size = base_position_size * position_size_factor
        
        # Additional context for the recommendation
        context = []
        
        if "bull" in market_regime and action in ["buy", "strong_buy"]:
            context.append("Trade is aligned with bullish market regime")
        elif "bear" in market_regime and action in ["sell", "strong_sell"]:
            context.append("Trade is aligned with bearish market regime")
        elif "sideways" in market_regime:
            context.append("Market is in a sideways regime - consider tighter stops")
        
        if vol_level == "high":
            context.append("High volatility detected - position size reduced")
        
        if risk_reward is not None and risk_reward < 1.5:
            context.append("Risk-reward ratio below recommended minimum of 1.5")
        
        # Create recommendation
        recommendation = {
            "symbol": symbol,
            "timeframe": timeframe,
            "market_regime": market_regime,
            "action": action,
            "confidence": abs(signal_strength),
            "current_price": current_price,
            "stop_loss": stop_loss,
            "take_profit": take_profit,
            "risk_reward_ratio": risk_reward,
            "recommended_position_size": recommended_position_size,
            "position_size_factor": position_size_factor,
            "context": context,
            "timestamp": datetime.now().isoformat()
        }
        
        return recommendation
    
    def calculate_correlation_matrix(self, symbols: List[str], timeframe: str = '1d', 
                                    period: int = 30) -> Dict[str, Any]:
        """
        Calculate correlation matrix between multiple symbols.
        
        Args:
            symbols: List of trading pair symbols
            timeframe: Data timeframe
            period: Correlation calculation period in days
            
        Returns:
            Dictionary with correlation matrix and related information
        """
        try:
            if not symbols or len(symbols) < 2:
                return {"error": "Need at least two symbols for correlation"}
            
            # Collect closing prices for all symbols
            prices = {}
            
            for symbol in symbols:
                data = self._get_market_data(symbol, timeframe, limit=period)
                if data is not None and not data.empty and 'close' in data.columns:
                    prices[symbol] = data['close']
            
            if len(prices) < 2:
                return {"error": "Insufficient data for correlation calculation"}
            
            # Create DataFrame with all prices
            df = pd.DataFrame(prices)
            
            # Calculate correlation matrix
            correlation_matrix = df.corr(method='pearson')
            
            # Convert to dictionary format for easier consumption
            corr_dict = {}
            for symbol1 in correlation_matrix.index:
                corr_dict[symbol1] = {}
                for symbol2 in correlation_matrix.columns:
                    corr_dict[symbol1][symbol2] = correlation_matrix.loc[symbol1, symbol2]
            
            # Identify highly correlated pairs (>0.7) and anti-correlated pairs (<-0.7)
            high_correlation = []
            anti_correlation = []
            
            for symbol1 in symbols:
                for symbol2 in symbols:
                    if symbol1 >= symbol2:  # Avoid duplicates and self-correlations
                        continue
                    
                    if symbol1 in corr_dict and symbol2 in corr_dict[symbol1]:
                        corr_value = corr_dict[symbol1][symbol2]
                        
                        if corr_value > 0.7:
                            high_correlation.append({
                                "pair": (symbol1, symbol2),
                                "correlation": corr_value
                            })
                        elif corr_value < -0.7:
                            anti_correlation.append({
                                "pair": (symbol1, symbol2),
                                "correlation": corr_value
                            })
            
            return {
                "matrix": corr_dict,
                "high_correlation": high_correlation,
                "anti_correlation": anti_correlation,
                "period": period,
                "timeframe": timeframe,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logging.error(f"Error calculating correlation matrix: {str(e)}\n{traceback.format_exc()}")
            return {"error": f"Correlation calculation failed: {str(e)}"}
    
    def get_market_bias(self, symbols: List[str] = None) -> Dict[str, str]:
        """
        Get overall market bias across multiple symbols or the whole market.
        
        Args:
            symbols: List of symbols to analyze (if None, use all analyzed symbols)
            
        Returns:
            Dictionary with market bias information
        """
        if symbols is None:
            symbols = list(self.market_regimes.keys())
        
        if not symbols:
            return {"overall_bias": "unknown", "confidence": 0.0}
        
        # Count different regimes
        regime_counts = {
            "bull": 0, "strong_bull": 0, "volatile_bull": 0,
            "bear": 0, "strong_bear": 0, "volatile_bear": 0,
            "sideways": 0, "volatile_sideways": 0,
            "unknown": 0
        }
        
        for symbol in symbols:
            regime = self.market_regimes.get(symbol, "unknown")
            
            # Increment appropriate counter
            if regime in regime_counts:
                regime_counts[regime] += 1
            else:
                regime_counts["unknown"] += 1
        
        # Calculate bullish and bearish counts
        bullish_count = regime_counts["bull"] + regime_counts["strong_bull"] + regime_counts["volatile_bull"]
        bearish_count = regime_counts["bear"] + regime_counts["strong_bear"] + regime_counts["volatile_bear"]
        sideways_count = regime_counts["sideways"] + regime_counts["volatile_sideways"]
        unknown_count = regime_counts["unknown"]
        
        total_count = len(symbols)
        
        # Determine overall bias
        overall_bias = "unknown"
        confidence = 0.0
        
        if total_count > 0:
            if bullish_count > bearish_count and bullish_count > sideways_count:
                if bullish_count / total_count > 0.7:
                    overall_bias = "strongly_bullish"
                    confidence = bullish_count / total_count
                else:
                    overall_bias = "bullish"
                    confidence = bullish_count / total_count
            elif bearish_count > bullish_count and bearish_count > sideways_count:
                if bearish_count / total_count > 0.7:
                    overall_bias = "strongly_bearish"
                    confidence = bearish_count / total_count
                else:
                    overall_bias = "bearish"
                    confidence = bearish_count / total_count
            elif sideways_count > bullish_count and sideways_count > bearish_count:
                overall_bias = "sideways"
                confidence = sideways_count / total_count
            else:
                # No clear bias
                overall_bias = "mixed"
                strongest = max(bullish_count, bearish_count, sideways_count)
                confidence = strongest / total_count
        
        return {
            "overall_bias": overall_bias,
            "confidence": confidence,
            "regime_counts": regime_counts,
            "symbols_analyzed": len(symbols),
            "timestamp": datetime.now().isoformat()
        }
    
    def clear_cache(self, symbol: str = None) -> bool:
        """
        Clear analysis cache for a symbol or all symbols.
        
        Args:
            symbol: Symbol to clear cache for, or None to clear all
            
        Returns:
            True if cache was cleared, False otherwise
        """
        with self._lock:
            if symbol is None:
                # Clear all cache
                self.analysis_cache = {}
                self.cache_expiry = {}
                return True
            
            # Clear specific symbol
            keys_to_remove = []
            for key in self.analysis_cache:
                if key.startswith(f"{symbol}_"):
                    keys_to_remove.append(key)
            
            for key in keys_to_remove:
                self.analysis_cache.pop(key, None)
                self.cache_expiry.pop(key, None)
            
            return len(keys_to_remove) > 0
    
    def _get_market_data(self, symbol: str, timeframe: str = '1h', limit: int = 500) -> Optional[pd.DataFrame]:
        """Get market data from database or other source."""
        try:
            # If we have a database manager, use it
            if self.db is not None:
                # Try to get data from database
                data = self.db.get_market_data(symbol, timeframe=timeframe, limit=limit)
                
                if data is not None and not data.empty:
                    return data
            
            # Fallback to mock data for testing
            logging.warning(f"No data available from database for {symbol}, using mock data")
            return self._generate_mock_data(symbol, limit)
            
        except Exception as e:
            logging.error(f"Error getting market data for {symbol}: {str(e)}")
            return None
    
    def _generate_mock_data(self, symbol: str, n_periods: int = 500) -> pd.DataFrame:
        """Generate mock market data for testing."""
        np.random.seed(42)  # For reproducibility
        
        # Start with a base price
        base_price = 100.0 if "BTC" not in symbol else 20000.0
        
        # Generate time index
        end_date = pd.Timestamp.now()
        start_date = end_date - pd.Timedelta(days=n_periods)
        dates = pd.date_range(start=start_date, end=end_date, periods=n_periods)
        
        # Generate price series with random walk
        changes = np.random.normal(0, 0.02, n_periods)
        prices = base_price * np.cumprod(1 + changes)
        
        # Generate OHLCV data
        data = pd.DataFrame({
            'open': prices * np.random.uniform(0.99, 1.01, n_periods),
            'high': prices * np.random.uniform(1.01, 1.03, n_periods),
            'low': prices * np.random.uniform(0.97, 0.99, n_periods),
            'close': prices,
            'volume': np.random.uniform(100, 1000, n_periods) * prices
        }, index=dates)
        
        return data
    
    def _preprocess_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Preprocess market data to ensure it has all required columns and indicators."""
        # Ensure we have a copy to avoid modifying the original
        df = data.copy()
        
        # Make sure we have OHLCV columns
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        for col in required_columns:
            if col not in df.columns:
                if col == 'volume':
                    # Generate mock volume if not available
                    df['volume'] = df['close'] * np.random.uniform(100, 1000, len(df))
                else:
                    # For price columns, use close if available, otherwise generate random data
                    if 'close' in df.columns:
                        df[col] = df['close']
                    else:
                        raise ValueError(f"Required column {col} not available in data")
        
        # Calculate common indicators for analysis
        try:
            # Moving averages
            df['sma_20'] = talib.SMA(df['close'].values, timeperiod=20)
            df['sma_50'] = talib.SMA(df['close'].values, timeperiod=50)
            df['sma_200'] = talib.SMA(df['close'].values, timeperiod=200)
            df['ema_20'] = talib.EMA(df['close'].values, timeperiod=20)
            
            # RSI
            df['rsi'] = talib.RSI(df['close'].values, timeperiod=14)
            
            # MACD
            macd, signal, hist = talib.MACD(df['close'].values)
            df['macd'] = macd
            df['macd_signal'] = signal
            df['macd_hist'] = hist
            
            # Bollinger Bands
            upper, middle, lower = talib.BBANDS(df['close'].values, timeperiod=20)
            df['bb_upper'] = upper
            df['bb_middle'] = middle
            df['bb_lower'] = lower
            
            # ATR for volatility
            df['atr'] = talib.ATR(df['high'].values, df['low'].values, df['close'].values, timeperiod=14)
            
            # ADX for trend strength
            df['adx'] = talib.ADX(df['high'].values, df['low'].values, df['close'].values, timeperiod=14)
            
            # Volume indicators
            df['obv'] = talib.OBV(df['close'].values, df['volume'].values)
            
            # Add momentum indicators
            df['mom'] = talib.MOM(df['close'].values, timeperiod=10)
            
            # Stochastic
            df['stoch_k'], df['stoch_d'] = talib.STOCH(df['high'].values, df['low'].values, df['close'].values)
            
            # Returns and volatility
            df['returns'] = df['close'].pct_change()
            df['volatility'] = df['returns'].rolling(window=20).std() * np.sqrt(252)  # Annualized
            
        except Exception as e:
            logging.error(f"Error calculating indicators: {str(e)}")
            # Continue with available data
        
        # Fill NaN values that may have been introduced
        df.fillna(method='bfill', inplace=True)
        
        return df
    
    def _analyze_market_regime(self, data: pd.DataFrame, symbol: str) -> str:
        """Analyze market regime (bull/bear/sideways) based on trend and volatility."""
        try:
            # Check if we have enough data
            if len(data) < 50:
                return MarketRegime.UNKNOWN.value
            
            # Get moving averages for trend detection
            if 'sma_20' not in data.columns or 'sma_50' not in data.columns or 'sma_200' not in data.columns:
                sma_20 = talib.SMA(data['close'].values, timeperiod=20)
                sma_50 = talib.SMA(data['close'].values, timeperiod=50)
                sma_200 = talib.SMA(data['close'].values, timeperiod=200)
            else:
                sma_20 = data['sma_20'].values
                sma_50 = data['sma_50'].values
                sma_200 = data['sma_200'].values
            
            # Get last values
            close = data['close'].iloc[-1]
            ma_20 = sma_20[-1]
            ma_50 = sma_50[-1]
            ma_200 = sma_200[-1]
            
            # Determine trend conditions
            price_above_ma20 = close > ma_20
            price_above_ma50 = close > ma_50
            price_above_ma200 = close > ma_200
            ma20_above_ma50 = ma_20 > ma_50
            ma50_above_ma200 = ma_50 > ma_200
            
            # Calculate trend score (-1 to 1)
            trend_indicators = [
                price_above_ma20, price_above_ma50, price_above_ma200,
                ma20_above_ma50, ma50_above_ma200
            ]
            trend_score = sum(1 if indicator else -1 for indicator in trend_indicators) / len(trend_indicators)
            
            # Calculate ADX for trend strength if not already in data
            if 'adx' not in data.columns:
                adx = talib.ADX(data['high'].values, data['low'].values, data['close'].values, timeperiod=14)[-1]
            else:
                adx = data['adx'].iloc[-1]
            
            # Calculate ATR for volatility if not already in data
            if 'atr' not in data.columns:
                atr = talib.ATR(data['high'].values, data['low'].values, data['close'].values, timeperiod=14)[-1]
            else:
                atr = data['atr'].iloc[-1]
            
            # Normalize ATR as percentage of price
            atr_pct = atr / close
            
            # Get historical ATR percentage for comparison
            if 'atr' not in data.columns:
                historical_atr = talib.ATR(data['high'].values, data['low'].values, data['close'].values, timeperiod=14)
            else:
                historical_atr = data['atr'].values
            
            historical_atr_pct = historical_atr / data['close'].values
            avg_atr_pct = np.nanmean(historical_atr_pct[-30:])  # Average over last 30 periods
            
            # Calculate volatility ratio (current vs historical)
            volatility_ratio = atr_pct / avg_atr_pct if avg_atr_pct > 0 else 1.0
            
            # Determine market regime
            if trend_score > 0.6 and adx > 25:
                if volatility_ratio > 1.5:
                    regime = MarketRegime.VOLATILE_BULL.value
                else:
                    regime = MarketRegime.STRONG_BULL.value if trend_score > 0.8 else MarketRegime.BULL.value
            elif trend_score < -0.6 and adx > 25:
                if volatility_ratio > 1.5:
                    regime = MarketRegime.VOLATILE_BEAR.value
                else:
                    regime = MarketRegime.STRONG_BEAR.value if trend_score < -0.8 else MarketRegime.BEAR.value
            else:
                if volatility_ratio > 1.5:
                    regime = MarketRegime.VOLATILE_SIDEWAYS.value
                else:
                    regime = MarketRegime.SIDEWAYS.value
            
            # Update internal state
            self.market_regimes[symbol] = regime
            
            return regime
            
        except Exception as e:
            logging.error(f"Error analyzing market regime: {str(e)}\n{traceback.format_exc()}")
            return MarketRegime.UNKNOWN.value
    
    def _analyze_market_structure(self, data: pd.DataFrame, symbol: str) -> str:
        """Analyze market structure (trend/range/accumulation/distribution)."""
        try:
            # Check if we have enough data
            if len(data) < 50:
                return MarketStructure.UNKNOWN.value
            
            # Get recent data (last 50 periods)
            recent_data = data.iloc[-50:].copy()
            
            # Find swing points (highs and lows)
            swing_highs = self._find_swing_points(recent_data, swing_type='high')
            swing_lows = self._find_swing_points(recent_data, swing_type='low')
            
            if not swing_highs or not swing_lows:
                return MarketStructure.UNKNOWN.value
            
            # Analyze swing patterns
            higher_highs = self._check_higher_swing_points(swing_highs)
            higher_lows = self._check_higher_swing_points(swing_lows)
            lower_highs = self._check_lower_swing_points(swing_highs)
            lower_lows = self._check_lower_swing_points(swing_lows)
            
            # Determine structure based on swing patterns
            if higher_highs and higher_lows:
                structure = MarketStructure.UPTREND.value
            elif lower_highs and lower_lows:
                structure = MarketStructure.DOWNTREND.value
            elif higher_lows and lower_highs:
                structure = MarketStructure.ACCUMULATION.value
            elif lower_lows and higher_highs:
                structure = MarketStructure.DISTRIBUTION.value
            else:
                # Check for range
                high_range = max(point["price"] for point in swing_highs)
                low_range = min(point["price"] for point in swing_lows)
                price_range = (high_range - low_range) / low_range
                
                if price_range < 0.05:  # Less than 5% range
                    structure = MarketStructure.RANGE.value
                else:
                    # Check for expansion/contraction
                    recent_atr = recent_data['atr'].iloc[-1] if 'atr' in recent_data.columns else None
                    past_atr = recent_data['atr'].iloc[0] if 'atr' in recent_data.columns else None
                    
                    if recent_atr is not None and past_atr is not None:
                        if recent_atr > past_atr * 1.3:  # 30% increase in ATR
                            structure = MarketStructure.EXPANSION.value
                        elif recent_atr < past_atr * 0.7:  # 30% decrease in ATR
                            structure = MarketStructure.CONTRACTION.value
                        else:
                            structure = MarketStructure.RANGE.value
                    else:
                        structure = MarketStructure.RANGE.value
            
            # Update internal state
            self.market_structures[symbol] = structure
            
            return structure
            
        except Exception as e:
            logging.error(f"Error analyzing market structure: {str(e)}\n{traceback.format_exc()}")
            return MarketStructure.UNKNOWN.value
    
    def _analyze_volatility(self, data: pd.DataFrame, symbol: str) -> Dict[str, Any]:
        """Analyze market volatility levels and characteristics."""
        try:
            # Check if we have enough data
            if len(data) < 30:
                return {"level": "unknown", "value": 1.0}
            
            # Calculate ATR if not already in data
            if 'atr' not in data.columns:
                atr = talib.ATR(data['high'].values, data['low'].values, data['close'].values, timeperiod=14)
            else:
                atr = data['atr'].values
            
            # Normalize ATR as percentage of price
            close_prices = data['close'].values
            atr_pct = atr / close_prices
            
            # Get recent and historical ATR percentage
            recent_atr_pct = np.nanmean(atr_pct[-5:])  # Average of last 5 periods
            historical_atr_pct = np.nanmean(atr_pct[-30:-5])  # Average of periods 30 to 5 days ago
            
            # Calculate volatility ratio
            volatility_ratio = recent_atr_pct / historical_atr_pct if historical_atr_pct > 0 else 1.0
            
            # Calculate standard deviation of returns if returns column exists
            if 'returns' in data.columns:
                returns = data['returns'].dropna()
                std_dev = returns.std()
                std_dev_annualized = std_dev * np.sqrt(252)  # Annualized volatility
            else:
                # Calculate returns
                returns = data['close'].pct_change().dropna()
                std_dev = returns.std()
                std_dev_annualized = std_dev * np.sqrt(252)  # Annualized volatility
            
            # Determine volatility level
            vol_thresholds = self.config["volatility_thresholds"]
            
            if volatility_ratio < vol_thresholds["low"]:
                level = "low"
            elif volatility_ratio > vol_thresholds["high"]:
                level = "high"
            else:
                level = "medium"
            
            # Check if volatility is expanding or contracting
            vol_change = recent_atr_pct / historical_atr_pct - 1.0
            
            direction = "stable"
            if vol_change > 0.2:  # 20% increase
                direction = "expanding"
            elif vol_change < -0.2:  # 20% decrease
                direction = "contracting"
            
            volatility_info = {
                "level": level,
                "value": volatility_ratio,
                "direction": direction,
                "recent_atr_pct": float(recent_atr_pct),
                "historical_atr_pct": float(historical_atr_pct),
                "std_dev_annualized": float(std_dev_annualized),
                "change_pct": float(vol_change)
            }
            
            # Update internal state
            self.volatility_levels[symbol] = volatility_info
            
            return volatility_info
            
        except Exception as e:
            logging.error(f"Error analyzing volatility: {str(e)}\n{traceback.format_exc()}")
            return {"level": "unknown", "value": 1.0}
    
    def _identify_support_resistance(self, data: pd.DataFrame, symbol: str) -> Dict[str, List[Dict[str, Any]]]:
        """Identify support and resistance levels using price clustering."""
        try:
            # Check if we have enough data
            if len(data) < 30:
                return {"support": [], "resistance": []}
            
            # Get configuration parameters
            config = self.config["support_resistance"]
            sensitivity = config["sensitivity"]
            cluster_distance = config["cluster_distance"]
            strength_threshold = config["strength_threshold"]
            
            # Get high/low/close prices
            highs = data['high'].values
            lows = data['low'].values
            closes = data['close'].values
            
            # Find swing highs and lows
            swing_highs = self._find_swing_points(data, swing_type='high')
            swing_lows = self._find_swing_points(data, swing_type='low')
            
            # Get recent price for reference
            recent_price = closes[-1]
            
            # Collect all potential resistance levels (swing highs)
            resistance_points = [point["price"] for point in swing_highs]
            
            # Collect all potential support levels (swing lows)
            support_points = [point["price"] for point in swing_lows]
            
            # Cluster nearby levels
            resistance_levels = self._cluster_price_levels(resistance_points, cluster_distance, recent_price)
            support_levels = self._cluster_price_levels(support_points, cluster_distance, recent_price)
            
            # Add moving averages as potential dynamic S/R levels
            if 'sma_50' in data.columns and 'sma_200' in data.columns:
                sma_50 = data['sma_50'].iloc[-1]
                sma_200 = data['sma_200'].iloc[-1]
                
                # Add as dynamic support/resistance based on price location
                if recent_price > sma_50:
                    support_levels.append({
                        "price": sma_50,
                        "strength": 3,  # Medium strength
                        "touches": 0,  # No specific touches
                        "type": SupportResistanceType.DYNAMIC_SUPPORT.value
                    })
                else:
                    resistance_levels.append({
                        "price": sma_50,
                        "strength": 3,
                        "touches": 0,
                        "type": SupportResistanceType.DYNAMIC_RESISTANCE.value
                    })
                    
                if recent_price > sma_200:
                    support_levels.append({
                        "price": sma_200,
                        "strength": 4,  # Stronger level
                        "touches": 0,
                        "type": SupportResistanceType.DYNAMIC_SUPPORT.value
                    })
                else:
                    resistance_levels.append({
                        "price": sma_200,
                        "strength": 4,
                        "touches": 0,
                        "type": SupportResistanceType.DYNAMIC_RESISTANCE.value
                    })
            
            # Sort levels by price
            support_levels = sorted(support_levels, key=lambda x: x["price"], reverse=True)
            resistance_levels = sorted(resistance_levels, key=lambda x: x["price"])
            
            # Update internal state
            self.sr_levels[symbol] = {
                "support": support_levels,
                "resistance": resistance_levels
            }
            
            return {
                "support": support_levels,
                "resistance": resistance_levels
            }
            
        except Exception as e:
            logging.error(f"Error identifying support/resistance: {str(e)}\n{traceback.format_exc()}")
            return {"support": [], "resistance": []}
    
    def _detect_chart_patterns(self, data: pd.DataFrame, symbol: str) -> Dict[str, Any]:
        """Detect chart patterns such as double top/bottom, head & shoulders, etc."""
        try:
            # Check if we have enough data
            if len(data) < 50:
                return {"patterns": []}
            
            # Get recent data (last 50 periods)
            recent_data = data.iloc[-50:].copy()
            
            # Find swing points (highs and lows)
            swing_highs = self._find_swing_points(recent_data, swing_type='high')
            swing_lows = self._find_swing_points(recent_data, swing_type='low')
            
            detected_patterns = []
            
            # Detect double top
            double_top = self._detect_double_top(swing_highs, recent_data)
            if double_top:
                detected_patterns.append(double_top)
            
            # Detect double bottom
            double_bottom = self._detect_double_bottom(swing_lows, recent_data)
            if double_bottom:
                detected_patterns.append(double_bottom)
            
            # Detect head and shoulders
            head_shoulders = self._detect_head_and_shoulders(swing_highs, swing_lows, recent_data)
            if head_shoulders:
                detected_patterns.append(head_shoulders)
            
            # Detect inverse head and shoulders
            inv_head_shoulders = self._detect_inverse_head_and_shoulders(swing_lows, swing_highs, recent_data)
            if inv_head_shoulders:
                detected_patterns.append(inv_head_shoulders)
            
            # Other patterns could be added here
            
            # Update patterns dictionary with detected patterns
            patterns = {
                "patterns": detected_patterns,
                "count": len(detected_patterns)
            }
            
            # Update internal state
            self.patterns[symbol] = patterns
            
            return patterns
            
        except Exception as e:
            logging.error(f"Error detecting chart patterns: {str(e)}\n{traceback.format_exc()}")
            return {"patterns": []}
    
    def _generate_signals(self, data: pd.DataFrame, market_regime: str, market_structure: str, 
                         volatility: Dict[str, Any], sr_levels: Dict[str, List[Dict[str, Any]]], 
                         patterns: Dict[str, Any]) -> Dict[str, Any]:
        """Generate trading signals based on comprehensive analysis."""
        try:
            # Initialize signal components
            trend_signal = 0.0
            momentum_signal = 0.0
            volatility_signal = 0.0
            support_resistance_signal = 0.0
            pattern_signal = 0.0
            
            # Get recent price
            if data.empty or 'close' not in data.columns:
                return {"overall_signal": 0.0}
                
            recent_price = data['close'].iloc[-1]
            
            # 1. Trend signal based on market regime
            if "bull" in market_regime:
                trend_multiplier = 1.0
                if "strong" in market_regime:
                    trend_multiplier = 1.5
                elif "volatile" in market_regime:
                    trend_multiplier = 0.7
                trend_signal = 0.5 * trend_multiplier
            elif "bear" in market_regime:
                trend_multiplier = 1.0
                if "strong" in market_regime:
                    trend_multiplier = 1.5
                elif "volatile" in market_regime:
                    trend_multiplier = 0.7
                trend_signal = -0.5 * trend_multiplier
            else:
                # Sideways or unknown regime
                trend_signal = 0.0
            
            # 2. Momentum signal based on indicators
            # RSI
            if 'rsi' in data.columns:
                rsi = data['rsi'].iloc[-1]
                if rsi > 70:
                    rsi_signal = -0.5  # Overbought
                elif rsi < 30:
                    rsi_signal = 0.5   # Oversold
                else:
                    rsi_signal = 0.0
            else:
                rsi_signal = 0.0
                
            # MACD
            if 'macd' in data.columns and 'macd_signal' in data.columns:
                macd = data['macd'].iloc[-1]
                macd_signal_line = data['macd_signal'].iloc[-1]
                
                if macd > macd_signal_line:
                    macd_signal = 0.3
                else:
                    macd_signal = -0.3
            else:
                macd_signal = 0.0
                
            # Combined momentum signal
            momentum_signal = (rsi_signal + macd_signal) / 2.0
            
            # 3. Volatility signal
            vol_level = volatility.get("level", "medium")
            vol_direction = volatility.get("direction", "stable")
            
            if vol_level == "high":
                if vol_direction == "expanding":
                    volatility_signal = -0.3  # High expanding volatility - cautious
                else:
                    volatility_signal = -0.1  # High stable/contracting volatility - less cautious
            elif vol_level == "low":
                volatility_signal = 0.1  # Low volatility - slightly bullish
            else:
                volatility_signal = 0.0  # Medium volatility - neutral
            
            # 4. Support/Resistance signal
            # Find nearest support and resistance
            nearest_support = None
            nearest_resistance = None
            support_distance = float('inf')
            resistance_distance = float('inf')
            
            for level in sr_levels.get("support", []):
                price = level.get("price", 0)
                if recent_price > price and (recent_price - price) < support_distance:
                    support_distance = recent_price - price
                    nearest_support = price
            
            for level in sr_levels.get("resistance", []):
                price = level.get("price", 0)
                if recent_price < price and (price - recent_price) < resistance_distance:
                    resistance_distance = price - recent_price
                    nearest_resistance = price
            
            # Calculate normalized distances
            if nearest_support is not None:
                support_distance_pct = support_distance / recent_price
            else:
                support_distance_pct = 1.0
                
            if nearest_resistance is not None:
                resistance_distance_pct = resistance_distance / recent_price
            else:
                resistance_distance_pct = 1.0
            
            # Signal based on relative distance to support/resistance
            if support_distance_pct < resistance_distance_pct:
                # Closer to support - bullish
                sr_ratio = resistance_distance_pct / support_distance_pct
                support_resistance_signal = min(0.5, sr_ratio * 0.1)  # Cap at 0.5
            else:
                # Closer to resistance - bearish
                sr_ratio = support_distance_pct / resistance_distance_pct
                support_resistance_signal = max(-0.5, -sr_ratio * 0.1)  # Cap at -0.5
            
            # 5. Pattern signal
            for pattern in patterns.get("patterns", []):
                pattern_type = pattern.get("type")
                confidence = pattern.get("confidence", 0.5)
                
                # Adjust signal based on pattern type
                if pattern_type == PatternType.DOUBLE_TOP.value:
                    pattern_signal -= 0.5 * confidence
                elif pattern_type == PatternType.DOUBLE_BOTTOM.value:
                    pattern_signal += 0.5 * confidence
                elif pattern_type == PatternType.HEAD_AND_SHOULDERS.value:
                    pattern_signal -= 0.6 * confidence
                elif pattern_type == PatternType.INV_HEAD_AND_SHOULDERS.value:
                    pattern_signal += 0.6 * confidence
                # Add more patterns as needed
            
            # Combine all signals with weights
            weights = {
                "trend": 0.3,
                "momentum": 0.25,
                "volatility": 0.1,
                "support_resistance": 0.25,
                "pattern": 0.1
            }
            
            overall_signal = (
                trend_signal * weights["trend"] +
                momentum_signal * weights["momentum"] +
                volatility_signal * weights["volatility"] +
                support_resistance_signal * weights["support_resistance"] +
                pattern_signal * weights["pattern"]
            )
            
            # Limit signal to range [-1, 1]
            overall_signal = max(-1.0, min(1.0, overall_signal))
            
            return {
                "overall_signal": overall_signal,
                "components": {
                    "trend": trend_signal,
                    "momentum": momentum_signal,
                    "volatility": volatility_signal,
                    "support_resistance": support_resistance_signal,
                    "pattern": pattern_signal
                },
                "weights": weights
            }
            
        except Exception as e:
            logging.error(f"Error generating signals: {str(e)}\n{traceback.format_exc()}")
            return {"overall_signal": 0.0}
    
    def _calculate_market_metrics(self, data: pd.DataFrame, symbol: str) -> Dict[str, Any]:
        """Calculate various market metrics for additional context."""
        try:
            metrics = {}
            
            # Check if we have enough data
            if data.empty or len(data) < 30:
                return metrics
            
            # Recent performance metrics
            if 'close' in data.columns:
                current_price = data['close'].iloc[-1]
                day_ago_price = data['close'].iloc[-2] if len(data) > 1 else current_price
                week_ago_idx = -7 if len(data) >= 7 else -len(data)
                week_ago_price = data['close'].iloc[week_ago_idx]
                month_ago_idx = -30 if len(data) >= 30 else -len(data)
                month_ago_price = data['close'].iloc[month_ago_idx]
                
                metrics["price_change_1d"] = (current_price / day_ago_price - 1) * 100
                metrics["price_change_1w"] = (current_price / week_ago_price - 1) * 100
                metrics["price_change_1m"] = (current_price / month_ago_price - 1) * 100
            
            # Volatility metrics
            if 'returns' in data.columns:
                returns = data['returns'].dropna()
                metrics["volatility_daily"] = returns.std() * 100  # Daily standard deviation as percentage
                metrics["volatility_annualized"] = returns.std() * np.sqrt(252) * 100  # Annualized volatility
            
            # Average volume
            if 'volume' in data.columns:
                metrics["avg_volume_10d"] = data['volume'].iloc[-10:].mean()
                metrics["volume_change"] = (data['volume'].iloc[-1] / metrics["avg_volume_10d"] - 1) * 100
            
            # RSI and momentum extremes
            if 'rsi' in data.columns:
                rsi = data['rsi'].iloc[-1]
                metrics["rsi"] = rsi
                metrics["overbought"] = rsi > 70
                metrics["oversold"] = rsi < 30
            
            # Trend strength
            if 'adx' in data.columns:
                adx = data['adx'].iloc[-1]
                metrics["adx"] = adx
                metrics["strong_trend"] = adx > 25
            
            # Distance to moving averages
            if 'sma_20' in data.columns and 'sma_50' in data.columns and 'sma_200' in data.columns:
                sma_20 = data['sma_20'].iloc[-1]
                sma_50 = data['sma_50'].iloc[-1]
                sma_200 = data['sma_200'].iloc[-1]
                
                metrics["price_vs_sma20"] = (current_price / sma_20 - 1) * 100
                metrics["price_vs_sma50"] = (current_price / sma_50 - 1) * 100
                metrics["price_vs_sma200"] = (current_price / sma_200 - 1) * 100
                
                # Check for golden cross (SMA 50 crosses above SMA 200)
                if len(data) > 2:
                    prev_sma_50 = data['sma_50'].iloc[-2]
                    prev_sma_200 = data['sma_200'].iloc[-2]
                    
                    metrics["golden_cross"] = (sma_50 > sma_200) and (prev_sma_50 <= prev_sma_200)
                    metrics["death_cross"] = (sma_50 < sma_200) and (prev_sma_50 >= prev_sma_200)
            
            return metrics
            
        except Exception as e:
            logging.error(f"Error calculating market metrics: {str(e)}\n{traceback.format_exc()}")
            return {}
    
    def _find_swing_points(self, data: pd.DataFrame, swing_type: str = 'high', window: int = 5) -> List[Dict[str, Any]]:
        """Find swing high or low points in the data."""
        if len(data) < window * 2 + 1:
            return []
        
        points = []
        
        if swing_type == 'high':
            # For swing highs
            for i in range(window, len(data) - window):
                is_swing_high = True
                for j in range(1, window + 1):
                    if data['high'].iloc[i] <= data['high'].iloc[i - j] or \
                       data['high'].iloc[i] <= data['high'].iloc[i + j]:
                        is_swing_high = False
                        break
                
                if is_swing_high:
                    points.append({
                        "index": i,
                        "timestamp": data.index[i],
                        "price": data['high'].iloc[i],
                        "type": swing_type
                    })
        else:
            # For swing lows
            for i in range(window, len(data) - window):
                is_swing_low = True
                for j in range(1, window + 1):
                    if data['low'].iloc[i] >= data['low'].iloc[i - j] or \
                       data['low'].iloc[i] >= data['low'].iloc[i + j]:
                        is_swing_low = False
                        break
                
                if is_swing_low:
                    points.append({
                        "index": i,
                        "timestamp": data.index[i],
                        "price": data['low'].iloc[i],
                        "type": swing_type
                    })
        
        return points
    
    def _check_higher_swing_points(self, points: List[Dict[str, Any]]) -> bool:
        """Check if swing points are making higher highs or higher lows."""
        if len(points) < 2:
            return False
        
        # Sort points by index
        sorted_points = sorted(points, key=lambda x: x["index"])
        
        # Check if we have at least 2 points making higher highs/lows
        higher_count = 0
        
        for i in range(1, len(sorted_points)):
            if sorted_points[i]["price"] > sorted_points[i-1]["price"]:
                higher_count += 1
        
        # Consider it a valid pattern if majority of swings are higher
        return higher_count >= (len(sorted_points) - 1) // 2
    
    def _check_lower_swing_points(self, points: List[Dict[str, Any]]) -> bool:
        """Check if swing points are making lower highs or lower lows."""
        if len(points) < 2:
            return False
        
        # Sort points by index
        sorted_points = sorted(points, key=lambda x: x["index"])
        
        # Check if we have at least 2 points making lower highs/lows
        lower_count = 0
        
        for i in range(1, len(sorted_points)):
            if sorted_points[i]["price"] < sorted_points[i-1]["price"]:
                lower_count += 1
        
        # Consider it a valid pattern if majority of swings are lower
        return lower_count >= (len(sorted_points) - 1) // 2
    
    def _cluster_price_levels(self, price_points: List[float], cluster_distance: float, 
                             reference_price: float) -> List[Dict[str, Any]]:
        """Cluster nearby price levels and calculate their strength."""
        if not price_points:
            return []
        
        # Convert to numpy array for clustering
        prices_array = np.array(price_points).reshape(-1, 1)
        
        # Scale for clustering based on cluster_distance parameter
        scaled_distance = cluster_distance * reference_price
        
        # Use DBSCAN for clustering
        clustering = DBSCAN(eps=scaled_distance, min_samples=1).fit(prices_array)
        
        # Get cluster labels
        labels = clustering.labels_
        
        # Group points by cluster
        clusters = {}
        
        for i, label in enumerate(labels):
            if label not in clusters:
                clusters[label] = []
            clusters[label].append(price_points[i])
        
        # Calculate cluster centers and strength
        levels = []
        
        for label, cluster_points in clusters.items():
            # Average price for the cluster
            avg_price = sum(cluster_points) / len(cluster_points)
            
            # Number of touches as strength indicator
            strength = len(cluster_points)
            
            # Determine level type
            if abs(avg_price - reference_price) / reference_price < 0.01:
                # Very close to current price - could be either support or resistance
                type_label = SupportResistanceType.SUPPORT.value if avg_price < reference_price else SupportResistanceType.RESISTANCE.value
            else:
                if avg_price < reference_price:
                    # Below current price - support
                    type_label = SupportResistanceType.MAJOR_SUPPORT.value if strength >= 3 else SupportResistanceType.SUPPORT.value
                else:
                    # Above current price - resistance
                    type_label = SupportResistanceType.MAJOR_RESISTANCE.value if strength >= 3 else SupportResistanceType.RESISTANCE.value
            
            levels.append({
                "price": avg_price,
                "strength": strength,
                "touches": len(cluster_points),
                "type": type_label
            })
        
        return levels
    
    def _detect_double_top(self, swing_highs: List[Dict[str, Any]], data: pd.DataFrame) -> Optional[Dict[str, Any]]:
        """Detect double top pattern."""
        if len(swing_highs) < 2:
            return None
        
        # Sort swing highs by index (chronological order)
        sorted_highs = sorted(swing_highs, key=lambda x: x["index"])
        
        # Need at least two swing highs
        if len(sorted_highs) < 2:
            return None
        
        # Look at the most recent two swing highs
        high1 = sorted_highs[-2]
        high2 = sorted_highs[-1]
        
        # Check if they're at similar price levels (within 1%)
        price_diff_pct = abs(high1["price"] - high2["price"]) / high1["price"]
        
        if price_diff_pct <= 0.01:
            # Check if there's a significant valley between them
            valley_found = False
            min_price = float('inf')
            
            # Find the lowest point between the two highs
            for i in range(high1["index"] + 1, high2["index"]):
                if data['low'].iloc[i] < min_price:
                    min_price = data['low'].iloc[i]
                    
            # Valley should be at least 2% below the tops
            if (high1["price"] - min_price) / high1["price"] >= 0.02:
                valley_found = True
            
            if valley_found:
                # Calculate pattern height for target
                height = high1["price"] - min_price
                target = min_price - height  # Projected target
                
                # Calculate confidence based on criteria
                confidence = 0.5
                
                # Adjust confidence based on price similarity
                if price_diff_pct < 0.005:  # Very similar tops
                    confidence += 0.1
                
                # Adjust confidence based on time between tops
                time_between = high2["index"] - high1["index"]
                if 5 <= time_between <= 20:  # Ideal time range
                    confidence += 0.1
                
                # Adjust confidence based on pattern completion
                recent_index = data.index[-1]
                high2_index = high2["timestamp"]
                
                # If neckline recently broken, higher confidence
                neckline_broken = False
                if len(data) > high2["index"]:
                    for i in range(high2["index"] + 1, len(data)):
                        if data['close'].iloc[i] < min_price:
                            neckline_broken = True
                            break
                
                if neckline_broken:
                    confidence += 0.2
                
                return {
                    "type": PatternType.DOUBLE_TOP.value,
                    "confidence": min(1.0, confidence),
                    "points": [high1, high2],
                    "neckline": min_price,
                    "target": target,
                    "neckline_broken": neckline_broken
                }
        
        return None
    
    def _detect_double_bottom(self, swing_lows: List[Dict[str, Any]], data: pd.DataFrame) -> Optional[Dict[str, Any]]:
        """Detect double bottom pattern."""
        if len(swing_lows) < 2:
            return None
        
        # Sort swing lows by index (chronological order)
        sorted_lows = sorted(swing_lows, key=lambda x: x["index"])
        
        # Need at least two swing lows
        if len(sorted_lows) < 2:
            return None
        
        # Look at the most recent two swing lows
        low1 = sorted_lows[-2]
        low2 = sorted_lows[-1]
        
        # Check if they're at similar price levels (within 1%)
        price_diff_pct = abs(low1["price"] - low2["price"]) / low1["price"]
        
        if price_diff_pct <= 0.01:
            # Check if there's a significant peak between them
            peak_found = False
            max_price = 0
            
            # Find the highest point between the two lows
            for i in range(low1["index"] + 1, low2["index"]):
                if data['high'].iloc[i] > max_price:
                    max_price = data['high'].iloc[i]
                    
            # Peak should be at least 2% above the bottoms
            if (max_price - low1["price"]) / low1["price"] >= 0.02:
                peak_found = True
            
            if peak_found:
                # Calculate pattern height for target
                height = max_price - low1["price"]
                target = max_price + height  # Projected target
                
                # Calculate confidence based on criteria
                confidence = 0.5
                
                # Adjust confidence based on price similarity
                if price_diff_pct < 0.005:  # Very similar bottoms
                    confidence += 0.1
                
                # Adjust confidence based on time between bottoms
                time_between = low2["index"] - low1["index"]
                if 5 <= time_between <= 20:  # Ideal time range
                    confidence += 0.1
                
                # Adjust confidence based on pattern completion
                neckline_broken = False
                if len(data) > low2["index"]:
                    for i in range(low2["index"] + 1, len(data)):
                        if data['close'].iloc[i] > max_price:
                            neckline_broken = True
                            break
                
                if neckline_broken:
                    confidence += 0.2
                
                return {
                    "type": PatternType.DOUBLE_BOTTOM.value,
                    "confidence": min(1.0, confidence),
                    "points": [low1, low2],
                    "neckline": max_price,
                    "target": target,
                    "neckline_broken": neckline_broken
                }
        
        return None
    
    def _detect_head_and_shoulders(self, swing_highs: List[Dict[str, Any]], swing_lows: List[Dict[str, Any]], 
                                  data: pd.DataFrame) -> Optional[Dict[str, Any]]:
        """Detect head and shoulders pattern."""
        if len(swing_highs) < 3 or len(swing_lows) < 2:
            return None
        
        # Sort swing highs and lows by index
        sorted_highs = sorted(swing_highs, key=lambda x: x["index"])
        sorted_lows = sorted(swing_lows, key=lambda x: x["index"])
        
        # Need at least three recent swing highs with a higher middle one
        if len(sorted_highs) < 3:
            return None
        
        # Look at the three most recent swing highs
        left_shoulder = sorted_highs[-3]
        head = sorted_highs[-2]
        right_shoulder = sorted_highs[-1]
        
        # Head should be higher than both shoulders
        if not (head["price"] > left_shoulder["price"] and head["price"] > right_shoulder["price"]):
            return None
        
        # Shoulders should be at similar heights (within 5%)
        shoulder_diff_pct = abs(left_shoulder["price"] - right_shoulder["price"]) / left_shoulder["price"]
        if shoulder_diff_pct > 0.05:
            return None
        
        # Find neckline based on lows between the three peaks
        neckline_points = []
        
        for low in sorted_lows:
            if left_shoulder["index"] < low["index"] < head["index"] or \
               head["index"] < low["index"] < right_shoulder["index"]:
                neckline_points.append(low)
        
        if len(neckline_points) < 2:
            return None
        
        # Calculate neckline (linear regression of lows)
        x = [point["index"] for point in neckline_points]
        y = [point["price"] for point in neckline_points]
        
        slope, intercept, _, _, _ = stats.linregress(x, y)
        
        # Calculate neckline at right shoulder position
        neckline_price = slope * right_shoulder["index"] + intercept
        
        # Calculate pattern height and target
        height = head["price"] - neckline_price
        target = neckline_price - height
        
        # Check if neckline is broken
        neckline_broken = False
        for i in range(right_shoulder["index"] + 1, len(data)):
            if data['close'].iloc[i] < neckline_price:
                neckline_broken = True
                break
        
        # Calculate confidence
        confidence = 0.5
        
        # Adjust confidence based on shoulder symmetry
        if shoulder_diff_pct < 0.03:
            confidence += 0.1
        
        # Adjust confidence based on pattern completion
        if neckline_broken:
            confidence += 0.2
        
        # Adjust confidence based on volume pattern
        volume_confirms = False
        if 'volume' in data.columns:
            # Volume should be higher on left shoulder than head, and lower on right shoulder
            left_vol = data['volume'].iloc[left_shoulder["index"]]
            head_vol = data['volume'].iloc[head["index"]]
            right_vol = data['volume'].iloc[right_shoulder["index"]]
            
            if left_vol > head_vol and head_vol > right_vol:
                volume_confirms = True
                confidence += 0.1
        
        return {
            "type": PatternType.HEAD_AND_SHOULDERS.value,
            "confidence": min(1.0, confidence),
            "points": [left_shoulder, head, right_shoulder],
            "neckline": neckline_price,
            "target": target,
            "neckline_broken": neckline_broken,
            "volume_confirms": volume_confirms
        }
    
    def _detect_inverse_head_and_shoulders(self, swing_lows: List[Dict[str, Any]], swing_highs: List[Dict[str, Any]], 
                                         data: pd.DataFrame) -> Optional[Dict[str, Any]]:
        """Detect inverse head and shoulders pattern."""
        if len(swing_lows) < 3 or len(swing_highs) < 2:
            return None
        
        # Sort swing highs and lows by index
        sorted_lows = sorted(swing_lows, key=lambda x: x["index"])
        sorted_highs = sorted(swing_highs, key=lambda x: x["index"])
        
        # Need at least three recent swing lows with a lower middle one
        if len(sorted_lows) < 3:
            return None
        
        # Look at the three most recent swing lows
        left_shoulder = sorted_lows[-3]
        head = sorted_lows[-2]
        right_shoulder = sorted_lows[-1]
        
        # Head should be lower than both shoulders
        if not (head["price"] < left_shoulder["price"] and head["price"] < right_shoulder["price"]):
            return None
        
        # Shoulders should be at similar heights (within 5%)
        shoulder_diff_pct = abs(left_shoulder["price"] - right_shoulder["price"]) / left_shoulder["price"]
        if shoulder_diff_pct > 0.05:
            return None
        
        # Find neckline based on highs between the three troughs
        neckline_points = []
        
        for high in sorted_highs:
            if left_shoulder["index"] < high["index"] < head["index"] or \
               head["index"] < high["index"] < right_shoulder["index"]:
                neckline_points.append(high)
        
        if len(neckline_points) < 2:
            return None
        
        # Calculate neckline (linear regression of highs)
        x = [point["index"] for point in neckline_points]
        y = [point["price"] for point in neckline_points]
        
        slope, intercept, _, _, _ = stats.linregress(x, y)
        
        # Calculate neckline at right shoulder position
        neckline_price = slope * right_shoulder["index"] + intercept
        
        # Calculate pattern height and target
        height = neckline_price - head["price"]
        target = neckline_price + height
        
        # Check if neckline is broken
        neckline_broken = False
        for i in range(right_shoulder["index"] + 1, len(data)):
            if data['close'].iloc[i] > neckline_price:
                neckline_broken = True
                break
        
        # Calculate confidence
        confidence = 0.5
        
        # Adjust confidence based on shoulder symmetry
        if shoulder_diff_pct < 0.03:
            confidence += 0.1
        
        # Adjust confidence based on pattern completion
        if neckline_broken:
            confidence += 0.2
        
        # Adjust confidence based on volume pattern
        volume_confirms = False
        if 'volume' in data.columns:
            # Volume should increase on right shoulder and breakout
            right_vol = data['volume'].iloc[right_shoulder["index"]]
            head_vol = data['volume'].iloc[head["index"]]
            
            if right_vol > head_vol:
                volume_confirms = True
                confidence += 0.1
        
        return {
            "type": PatternType.INV_HEAD_AND_SHOULDERS.value,
            "confidence": min(1.0, confidence),
            "points": [left_shoulder, head, right_shoulder],
            "neckline": neckline_price,
            "target": target,
            "neckline_broken": neckline_broken,
            "volume_confirms": volume_confirms
        }
    
    def _generate_empty_analysis(self, symbol: str, timeframe: str) -> Dict[str, Any]:
        """Generate empty analysis result for error cases."""
        return {
            "symbol": symbol,
            "timeframe": timeframe,
            "timestamp": datetime.now().isoformat(),
            "last_price": None,
            "market_regime": MarketRegime.UNKNOWN.value,
            "market_structure": MarketStructure.UNKNOWN.value,
            "volatility": {"level": "unknown", "value": 1.0},
            "support_resistance": {"support": [], "resistance": []},
            "patterns": {"patterns": [], "count": 0},
            "signals": {"overall_signal": 0.0},
            "metrics": {},
            "error": "No data available for analysis"
        }

# Apply error handling decorator if available
if HAVE_ERROR_HANDLING:
    MarketAnalysis.analyze_market = safe_execute(
        ErrorCategory.DATA_PROCESSING, 
        default_return={"error": "Analysis failed", "market_regime": MarketRegime.UNKNOWN.value},
        severity=ErrorSeverity.ERROR
    )(MarketAnalysis.analyze_market)
