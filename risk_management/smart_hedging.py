# smart_hedging.py

import numpy as np
import pandas as pd
import logging
import time
import threading
import os
import json
from typing import Dict, List, Optional, Tuple, Union, Any
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor

# Try to import error handling, use basic error handling if not available
try:
    from error_handling import safe_execute, ErrorCategory, ErrorSeverity, ErrorHandler, TradingSystemError
    HAVE_ERROR_HANDLING = True
except ImportError:
    HAVE_ERROR_HANDLING = False
    logging.warning("Error handling module not available for smart hedging. Using basic error handling.")

# Try to import exchange connector, use ccxt directly if not available
try:
    from execution.exchange_connector import ExchangeConnector
    HAVE_EXCHANGE_CONNECTOR = True
except ImportError:
    HAVE_EXCHANGE_CONNECTOR = False
    try:
        import ccxt
        logging.warning("Using ccxt directly for hedging operations.")
    except ImportError:
        logging.error("Neither exchange_connector nor ccxt available. Hedging will be simulated.")

class HedgeType:
    """Enum-like class for hedge types"""
    INVERSE = "inverse"             # Opposite position in same asset
    FUTURES = "futures"             # Hedge using futures contracts
    OPTIONS = "options"             # Hedge using options
    CROSS_ASSET = "cross_asset"     # Hedge using correlated asset
    STABLECOIN = "stablecoin"       # Convert to stablecoin
    DELTA_NEUTRAL = "delta_neutral" # Delta neutral strategy
    
class HedgePosition:
    """Represents an active hedge position"""
    
    def __init__(
        self, 
        hedge_id: str,
        original_position_id: str,
        symbol: str, 
        hedge_symbol: str,
        hedge_type: str,
        entry_price: float,
        size: float,
        order_ids: List[str],
        entry_time: datetime,
        expiration_time: Optional[datetime] = None,
        status: str = "active",
        metadata: Optional[Dict[str, Any]] = None
    ):
        self.hedge_id = hedge_id
        self.original_position_id = original_position_id
        self.symbol = symbol
        self.hedge_symbol = hedge_symbol
        self.hedge_type = hedge_type
        self.entry_price = entry_price
        self.size = size
        self.order_ids = order_ids
        self.entry_time = entry_time
        self.expiration_time = expiration_time
        self.status = status
        self.metadata = metadata or {}
        self.exit_price = None
        self.exit_time = None
        self.pnl = None
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage"""
        return {
            "hedge_id": self.hedge_id,
            "original_position_id": self.original_position_id,
            "symbol": self.symbol,
            "hedge_symbol": self.hedge_symbol,
            "hedge_type": self.hedge_type,
            "entry_price": self.entry_price,
            "size": self.size,
            "order_ids": self.order_ids,
            "entry_time": self.entry_time.isoformat() if self.entry_time else None,
            "expiration_time": self.expiration_time.isoformat() if self.expiration_time else None,
            "status": self.status,
            "exit_price": self.exit_price,
            "exit_time": self.exit_time.isoformat() if self.exit_time else None,
            "pnl": self.pnl,
            "metadata": self.metadata
        }
        
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'HedgePosition':
        """Create from dictionary"""
        # Convert ISO format strings back to datetime objects
        entry_time = datetime.fromisoformat(data["entry_time"]) if data.get("entry_time") else None
        expiration_time = datetime.fromisoformat(data["expiration_time"]) if data.get("expiration_time") else None
        exit_time = datetime.fromisoformat(data["exit_time"]) if data.get("exit_time") else None
        
        # Create instance
        hedge = cls(
            hedge_id=data["hedge_id"],
            original_position_id=data["original_position_id"],
            symbol=data["symbol"],
            hedge_symbol=data["hedge_symbol"],
            hedge_type=data["hedge_type"],
            entry_price=data["entry_price"],
            size=data["size"],
            order_ids=data["order_ids"],
            entry_time=entry_time,
            expiration_time=expiration_time,
            status=data["status"],
            metadata=data.get("metadata", {})
        )
        
        # Set additional fields
        hedge.exit_price = data.get("exit_price")
        hedge.exit_time = exit_time
        hedge.pnl = data.get("pnl")
        
        return hedge

class SmartHedging:
    """Advanced AI-driven hedging system to reduce market risk dynamically with minimal slippage"""

    def __init__(
        self, 
        trading_system=None,
        config=None, 
        exchange_id: str = "binance",
        api_key: Optional[str] = None,
        api_secret: Optional[str] = None,
        volatility_threshold: float = 0.025,
        volume_threshold: float = 1.5,
        sentiment_threshold: float = -0.4,
        correlation_threshold: float = 0.7,
        check_interval: int = 60,  # seconds
        max_hedge_ratio: float = 0.8,
        preferred_hedge_types: Optional[List[str]] = None,
        persistence_path: str = "data/hedges"
    ):
        """
        Initialize the smart hedging system.
        
        Args:
            trading_system: Reference to the main trading system
            config: Configuration dictionary
            exchange_id: ID of the exchange to use
            api_key: API key for exchange (if not in config)
            api_secret: API secret for exchange (if not in config)
            volatility_threshold: Volatility threshold for hedge trigger
            volume_threshold: Volume spike threshold for hedge trigger
            sentiment_threshold: Sentiment threshold for hedge trigger
            correlation_threshold: Correlation threshold for cross-asset hedging
            check_interval: How often to check risk conditions (seconds)
            max_hedge_ratio: Maximum hedge size as ratio of position
            preferred_hedge_types: Preferred hedge types in order
            persistence_path: Path to store hedge positions
        """
        self.trading_system = trading_system
        self.config = config or {}
        
        # Initialize exchange connection
        self.exchange_id = exchange_id
        self.api_key = api_key
        self.api_secret = api_secret
        self._initialize_exchange()
        
        # Risk thresholds
        self.volatility_threshold = volatility_threshold
        self.volume_threshold = volume_threshold
        self.sentiment_threshold = sentiment_threshold
        self.correlation_threshold = correlation_threshold
        
        # Hedging parameters
        self.check_interval = check_interval
        self.max_hedge_ratio = max_hedge_ratio
        self.preferred_hedge_types = preferred_hedge_types or [
            HedgeType.INVERSE, 
            HedgeType.FUTURES, 
            HedgeType.CROSS_ASSET, 
            HedgeType.STABLECOIN
        ]
        
        # State tracking
        self.active_hedges = {}
        self.hedge_history = []
        self.risk_cache = {}
        self.persistence_path = persistence_path
        
        # Ensure persistence directory exists
        os.makedirs(persistence_path, exist_ok=True)
        
        # Asset correlations (to be updated)
        self.asset_correlations = {}
        
        # Market data cache to reduce API calls
        self.market_data_cache = {}
        self.cache_expiry = {}
        
        # Background monitoring thread
        self.monitoring = False
        self.monitor_thread = None
        self.thread_lock = threading.RLock()
        
        # Performance monitoring
        self.performance_metrics = {
            "num_hedges_created": 0,
            "num_hedges_closed": 0,
            "avg_hedge_duration": 0,
            "total_hedge_pnl": 0,
            "slippage_total": 0,
            "api_calls": 0
        }
        
        # Load existing hedges if available
        self._load_hedges()
        
        logging.info(f"Smart Hedging initialized with {exchange_id} exchange")

    def _initialize_exchange(self) -> None:
        """Initialize exchange connection with fallbacks"""
        if HAVE_EXCHANGE_CONNECTOR:
            try:
                # Use the application's exchange connector
                self.exchange = ExchangeConnector(
                    exchange_id=self.exchange_id,
                    api_key=self.api_key or self._get_api_key(),
                    api_secret=self.api_secret or self._get_api_secret()
                )
                logging.info(f"Using ExchangeConnector for {self.exchange_id}")
                return
            except Exception as e:
                logging.warning(f"Failed to initialize ExchangeConnector: {e}")
        
        # Fallback to direct ccxt usage
        try:
            exchange_class = getattr(ccxt, self.exchange_id)
            self.exchange = exchange_class({
                'apiKey': self.api_key or self._get_api_key(),
                'secret': self.api_secret or self._get_api_secret(),
                'enableRateLimit': True,
                'options': {
                    'defaultType': 'future'  # Use futures by default for hedging
                }
            })
            logging.info(f"Using direct ccxt connection for {self.exchange_id}")
        except Exception as e:
            logging.error(f"Failed to initialize exchange: {e}")
            # Create a simulated exchange
            self.exchange = self._create_simulated_exchange()
            logging.warning("Using simulated exchange for hedging operations")

    def _get_api_key(self) -> Optional[str]:
        """Get API key from configuration"""
        if self.trading_system and hasattr(self.trading_system, 'config'):
            return self.trading_system.config.get('exchanges', {}).get(self.exchange_id, {}).get('api_key')
        
        return self.config.get('exchanges', {}).get(self.exchange_id, {}).get('api_key')

    def _get_api_secret(self) -> Optional[str]:
        """Get API secret from configuration"""
        if self.trading_system and hasattr(self.trading_system, 'config'):
            return self.trading_system.config.get('exchanges', {}).get(self.exchange_id, {}).get('api_secret')
        
        return self.config.get('exchanges', {}).get(self.exchange_id, {}).get('api_secret')

    def _create_simulated_exchange(self) -> Any:
        """Create a simulated exchange for testing"""
        class SimulatedExchange:
            def __init__(self):
                self.orders = []
                
            def create_market_sell_order(self, symbol, amount, params={}):
                order_id = f"sim_{len(self.orders) + 1}"
                price = 40000 - (np.random.random() * 100)  # Simulated price
                order = {
                    'id': order_id,
                    'symbol': symbol,
                    'type': 'market',
                    'side': 'sell',
                    'amount': amount,
                    'price': price,
                    'timestamp': datetime.now().timestamp() * 1000,
                    'status': 'closed'
                }
                self.orders.append(order)
                return order
                
            def create_market_buy_order(self, symbol, amount, params={}):
                order_id = f"sim_{len(self.orders) + 1}"
                price = 40000 + (np.random.random() * 100)  # Simulated price
                order = {
                    'id': order_id,
                    'symbol': symbol,
                    'type': 'market',
                    'side': 'buy',
                    'amount': amount,
                    'price': price,
                    'timestamp': datetime.now().timestamp() * 1000,
                    'status': 'closed'
                }
                self.orders.append(order)
                return order
                
            def fetch_ticker(self, symbol):
                return {
                    'symbol': symbol,
                    'bid': 40000 - (np.random.random() * 50),
                    'ask': 40000 + (np.random.random() * 50),
                    'last': 40000 + (np.random.random() * 20 - 10)
                }
                
            def fetch_order(self, id, symbol=None):
                for order in self.orders:
                    if order['id'] == id:
                        return order
                return None
                
        return SimulatedExchange()

    def start_monitoring(self) -> bool:
        """Start background risk monitoring thread"""
        with self.thread_lock:
            if self.monitoring:
                logging.warning("Risk monitoring already running")
                return False
            
            self.monitoring = True
            self.monitor_thread = threading.Thread(
                target=self._risk_monitoring_loop,
                daemon=True
            )
            self.monitor_thread.start()
            logging.info("Smart hedging risk monitoring started")
            return True

    def stop_monitoring(self) -> bool:
        """Stop background risk monitoring"""
        with self.thread_lock:
            if not self.monitoring:
                logging.warning("Risk monitoring not running")
                return False
                
            self.monitoring = False
            if self.monitor_thread and self.monitor_thread.is_alive():
                # Give it a moment to shut down gracefully
                self.monitor_thread.join(timeout=2)
                
            logging.info("Smart hedging risk monitoring stopped")
            return True

    def _risk_monitoring_loop(self) -> None:
        """Background thread that monitors risk and triggers hedges"""
        while self.monitoring:
            try:
                # Get all active positions from trading system
                active_positions = self._get_active_positions()
                
                # Check each position for risk
                for position_id, position in active_positions.items():
                    # Skip positions that are already hedged
                    if position_id in [h.original_position_id for h in self.active_hedges.values()]:
                        continue
                        
                    # Get symbol
                    symbol = position.get('symbol')
                    
                    # Get market data for risk assessment
                    market_data = self._get_market_data(symbol)
                    
                    # Detect risk conditions
                    if self.detect_risk_conditions(market_data):
                        logging.info(f"High risk detected for {symbol}, creating hedge")
                        
                        # Calculate optimal hedge size
                        position_size = position.get('size', 0)
                        hedge_size = self._calculate_hedge_size(position, market_data)
                        
                        # Execute hedge
                        self.hedge_position(
                            symbol=symbol,
                            position_id=position_id,
                            hedge_size=hedge_size
                        )
                
                # Check existing hedges for adjustment or closure
                self._manage_existing_hedges()
                
            except Exception as e:
                logging.error(f"Error in risk monitoring loop: {e}")
            
            # Sleep until next check
            time.sleep(self.check_interval)

    def _get_active_positions(self) -> Dict[str, Dict[str, Any]]:
        """Get active positions from trading system with fallbacks"""
        if self.trading_system and hasattr(self.trading_system, 'open_positions'):
            return self.trading_system.open_positions
            
        # Simulated positions for testing
        return {
            "pos1": {
                "position_id": "pos1",
                "symbol": "BTC/USDT",
                "size": 0.1,
                "entry_price": 40000,
                "current_price": 39500,
                "side": "long"
            }
        }

    def _get_market_data(self, symbol: str, force_refresh: bool = False) -> pd.DataFrame:
        """
        Get market data for a symbol with caching to reduce API calls.
        
        Args:
            symbol: Trading pair
            force_refresh: Force refresh even if cache is valid
            
        Returns:
            DataFrame with OHLCV data
        """
        # Check cache first
        current_time = time.time()
        cache_key = f"{symbol}_1h"
        
        if (not force_refresh and 
            cache_key in self.market_data_cache and 
            cache_key in self.cache_expiry and 
            current_time < self.cache_expiry[cache_key]):
            return self.market_data_cache[cache_key]
            
        # Get data from trading system if available
        if self.trading_system and hasattr(self.trading_system, 'get_market_data'):
            try:
                df = self.trading_system.get_market_data(symbol, timeframe='1h', limit=30)
                if not df.empty:
                    # Cache the data
                    self.market_data_cache[cache_key] = df
                    # Set expiry to 5 minutes
                    self.cache_expiry[cache_key] = current_time + 300
                    return df
            except Exception as e:
                logging.warning(f"Error getting market data from trading system: {e}")
        
        # Fallback to direct exchange API
        try:
            # Increment API call counter
            self.performance_metrics["api_calls"] += 1
            
            # Fetch OHLCV data
            ohlcv = self.exchange.fetch_ohlcv(symbol, '1h', limit=30)
            
            # Convert to DataFrame
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            
            # Cache the data
            self.market_data_cache[cache_key] = df
            # Set expiry to 5 minutes
            self.cache_expiry[cache_key] = current_time + 300
            
            return df
            
        except Exception as e:
            logging.error(f"Error fetching market data: {e}")
            
            # Return empty DataFrame with expected columns
            return pd.DataFrame(columns=['open', 'high', 'low', 'close', 'volume'])

    @safe_execute(ErrorCategory.RISK, False) if HAVE_ERROR_HANDLING else lambda f: f
    def detect_risk_conditions(self, df: pd.DataFrame) -> bool:
        """
        Detects market conditions that indicate high risk using multiple indicators.
        
        Args:
            df: Market data DataFrame
            
        Returns:
            True if hedging is needed, False otherwise
        """
        if df.empty:
            logging.warning("Empty dataframe provided to risk detection")
            return False
            
        try:
            # Calculate volatility using True Range
            high_low = df['high'] - df['low']
            high_close = abs(df['high'] - df['close'].shift())
            low_close = abs(df['low'] - df['close'].shift())
            true_ranges = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
            atr = true_ranges.rolling(14).mean()
            
            # Extract last values
            last_row = df.iloc[-1]
            current_price = last_row['close']
            current_atr = atr.iloc[-1]
            
            # Normalize ATR as percentage of price
            volatility = current_atr / current_price
            
            # Volume analysis
            volume_ma = df['volume'].rolling(10).mean()
            volume_ratio = last_row['volume'] / volume_ma.iloc[-1]
            volume_spike = volume_ratio > self.volume_threshold
            
            # Momentum indicators
            price_change_1h = (current_price / df['close'].iloc[-2] - 1) * 100
            price_change_24h = (current_price / df['close'].iloc[-24] - 1) * 100 if len(df) >= 24 else 0
            
            # Price moves outside Bollinger Bands
            sma20 = df['close'].rolling(20).mean()
            std20 = df['close'].rolling(20).std()
            upper_band = sma20 + (std20 * 2)
            lower_band = sma20 - (std20 * 2)
            price_above_bands = current_price > upper_band.iloc[-1]
            price_below_bands = current_price < lower_band.iloc[-1]
            
            # Detect bearish reversal patterns
            last_3_candles = df.iloc[-3:]
            bearish_engulfing = (
                last_3_candles['close'].iloc[-1] < last_3_candles['open'].iloc[-1] and
                last_3_candles['close'].iloc[-2] > last_3_candles['open'].iloc[-2] and
                last_3_candles['close'].iloc[-1] < last_3_candles['open'].iloc[-2] and
                last_3_candles['open'].iloc[-1] > last_3_candles['close'].iloc[-2]
            )
            
            # Get sentiment score if available
            sentiment_score = self._get_sentiment_score(df.index[-1].strftime('%Y-%m-%d'))
            
            # Update risk cache
            symbol = df.attrs.get('symbol', 'unknown')
            self.risk_cache[symbol] = {
                'timestamp': datetime.now(),
                'volatility': volatility,
                'volume_ratio': volume_ratio,
                'price_change_1h': price_change_1h,
                'price_change_24h': price_change_24h,
                'sentiment': sentiment_score,
                'above_bands': price_above_bands,
                'below_bands': price_below_bands,
                'bearish_pattern': bearish_engulfing
            }
            
            # Combine risk factors with weights
            risk_detected = False
            
            # Volatility risk
            if volatility > self.volatility_threshold:
                logging.info(f"High volatility detected: {volatility:.4f} > {self.volatility_threshold:.4f}")
                risk_detected = True
                
            # Volume spike risk
            if volume_spike:
                logging.info(f"Volume spike detected: {volume_ratio:.2f}x normal volume")
                risk_detected = True
                
            # Rapid price change risk
            if abs(price_change_1h) > 3.0:
                logging.info(f"Rapid price change detected: {price_change_1h:.2f}% in 1 hour")
                risk_detected = True
                
            # Negative sentiment risk
            if sentiment_score < self.sentiment_threshold:
                logging.info(f"Negative sentiment detected: {sentiment_score:.2f}")
                risk_detected = True
                
            # Price outside bands risk (stronger signal)
            if price_above_bands or price_below_bands:
                logging.info(f"Price outside Bollinger Bands: {'above' if price_above_bands else 'below'}")
                risk_detected = True
                
            # Bearish pattern risk
            if bearish_engulfing:
                logging.info("Bearish engulfing pattern detected")
                risk_detected = True
                
            return risk_detected
            
        except Exception as e:
            logging.error(f"Error detecting risk conditions: {e}")
            return False

    def _get_sentiment_score(self, date_str: str) -> float:
        """
        Get market sentiment score for a specific date.
        Integrates with sentiment analysis module if available.
        
        Args:
            date_str: Date string in YYYY-MM-DD format
            
        Returns:
            Sentiment score from -1 (extremely negative) to 1 (extremely positive)
        """
        # Try to get sentiment from trading system
        if (self.trading_system and 
            hasattr(self.trading_system, 'sentiment_analyzer') and
            hasattr(self.trading_system.sentiment_analyzer, 'get_daily_sentiment')):
            try:
                return self.trading_system.sentiment_analyzer.get_daily_sentiment(date_str)
            except Exception as e:
                logging.warning(f"Failed to get sentiment from system: {e}")
        
        # Simple simulation based on day of week (weekends more negative)
        try:
            date = datetime.strptime(date_str, '%Y-%m-%d')
            day_of_week = date.weekday()
            
            # Weekend effect (Friday-Sunday more negative)
            if day_of_week >= 4:  # Friday to Sunday
                base_sentiment = -0.2
            else:
                base_sentiment = 0.1
                
            # Add randomness
            sentiment = base_sentiment + (np.random.random() * 0.4 - 0.2)
            
            # Ensure range -1 to 1
            return max(-1.0, min(1.0, sentiment))
            
        except Exception as e:
            logging.error(f"Error calculating sentiment: {e}")
            return 0.0  # Neutral fallback

    def _calculate_hedge_size(self, position: Dict[str, Any], market_data: pd.DataFrame) -> float:
        """
        Calculate optimal hedge size based on position and market conditions.
        
        Args:
            position: Position information
            market_data: Market data for risk assessment
            
        Returns:
            Optimal hedge size
        """
        position_size = position.get('size', 0)
        
        # Insufficient position size
        if position_size <= 0:
            return 0
            
        # Get symbol
        symbol = position.get('symbol', '')
        
        # Get risk factors
        volatility = 0.03  # Default
        if symbol in self.risk_cache:
            volatility = self.risk_cache[symbol].get('volatility', 0.03)
            
        # Base hedge ratio starts at 50%
        hedge_ratio = 0.5
        
        # Adjust ratio based on volatility
        if volatility > 0.05:  # Very high volatility
            hedge_ratio = 0.8
        elif volatility > 0.03:  # High volatility
            hedge_ratio = 0.7
        elif volatility < 0.01:  # Low volatility
            hedge_ratio = 0.3
            
        # Calculate hedge size
        hedge_size = position_size * min(hedge_ratio, self.max_hedge_ratio)
        
        logging.info(f"Calculated hedge size: {hedge_size:.4f} from position size {position_size:.4f} (ratio: {hedge_ratio:.2f})")
        return hedge_size

    @safe_execute(ErrorCategory.TRADE_EXECUTION, None) if HAVE_ERROR_HANDLING else lambda f: f
    def hedge_position(
        self, 
        symbol: str, 
        position_id: str = None, 
        hedge_size: float = None, 
        hedge_type: Optional[str] = None,
        expiration_hours: Optional[int] = None
    ) -> Optional[str]:
        """
        Place a hedge for a specific position with smart execution.
        
        Args:
            symbol: Trading pair to hedge
            position_id: Original position ID
            hedge_size: Size to hedge (if None, will be calculated)
            hedge_type: Type of hedge to use
            expiration_hours: Hours until hedge expires (None = no expiration)
            
        Returns:
            Hedge ID if successful, None otherwise
        """
        # Generate position ID if not provided
        if not position_id:
            position_id = f"pos_{int(time.time())}"
            
        # Validate symbol
        if not symbol or '/' not in symbol:
            logging.error(f"Invalid symbol format: {symbol}")
            return None
            
        # Get position details
        position = self._get_position_by_id(position_id)
        if not position and hedge_size is None:
            logging.error(f"Position {position_id} not found and no hedge size specified")
            return None
            
        # Calculate hedge size if not provided
        if hedge_size is None:
            market_data = self._get_market_data(symbol)
            hedge_size = self._calculate_hedge_size(position, market_data)
            
        # No hedging needed
        if hedge_size <= 0:
            logging.warning(f"Zero or negative hedge size calculated for {symbol}")
            return None
            
        # Determine optimal hedge type
        hedge_method = self._select_hedge_method(symbol, hedge_type)
        
        try:
            # Generate unique hedge ID
            hedge_id = f"hedge_{int(time.time())}_{symbol.split('/')[0]}"
            
            # Execute the appropriate hedge strategy
            if hedge_method == HedgeType.INVERSE:
                result = self._execute_inverse_hedge(symbol, hedge_size)
            elif hedge_method == HedgeType.FUTURES:
                result = self._execute_futures_hedge(symbol, hedge_size)
            elif hedge_method == HedgeType.CROSS_ASSET:
                result = self._execute_cross_asset_hedge(symbol, hedge_size)
            elif hedge_method == HedgeType.STABLECOIN:
                result = self._execute_stablecoin_hedge(symbol, hedge_size)
            else:
                result = self._execute_inverse_hedge(symbol, hedge_size)  # Default fallback
                
            if not result:
                logging.error(f"Failed to execute {hedge_method} hedge for {symbol}")
                return None
                
            # Create hedge position object
            entry_price = result.get('price', 0)
            order_ids = [result.get('id')] if result.get('id') else []
            hedge_symbol = result.get('symbol', symbol)
            
            # Calculate expiration time if provided
            expiration_time = None
            if expiration_hours:
                expiration_time = datetime.now() + timedelta(hours=expiration_hours)
                
            # Create hedge position
            hedge_position = HedgePosition(
                hedge_id=hedge_id,
                original_position_id=position_id,
                symbol=symbol,
                hedge_symbol=hedge_symbol,
                hedge_type=hedge_method,
                entry_price=entry_price,
                size=hedge_size,
                order_ids=order_ids,
                entry_time=datetime.now(),
                expiration_time=expiration_time,
                status="active",
                metadata={"execution_method": "smart_hedging"}
            )
            
            # Store hedge
            self.active_hedges[hedge_id] = hedge_position
            
            # Update metrics
            self.performance_metrics["num_hedges_created"] += 1
            
            # Persist hedges
            self._save_hedges()
            
            logging.info(f"Hedge {hedge_id} created for {symbol} ({hedge_method}) with size {hedge_size}")
            return hedge_id
            
        except Exception as e:
            logging.error(f"Error placing hedge for {symbol}: {e}")
            return None

    def _get_position_by_id(self, position_id: str) -> Optional[Dict[str, Any]]:
        """Get position by ID from trading system"""
        if self.trading_system and hasattr(self.trading_system, 'open_positions'):
            if position_id in self.trading_system.open_positions:
                return self.trading_system.open_positions[position_id]
                
        return None

    def _select_hedge_method(self, symbol: str, preferred_type: Optional[str] = None) -> str:
        """
        Select the optimal hedge method based on available options and symbol.
        
        Args:
            symbol: Trading pair
            preferred_type: Preferred hedge type (if None, will select automatically)
            
        Returns:
            Hedge type to use
        """
        # Use preferred type if specified and supported
        if preferred_type and preferred_type in self.preferred_hedge_types:
            return preferred_type
            
        # Get the base and quote assets
        try:
            base_asset, quote_asset = symbol.split('/')
        except ValueError:
            base_asset, quote_asset = symbol, "USDT"
            
        # Check futures availability
        futures_available = self._check_futures_available(symbol)
        
        # Check if inverse is available
        inverse_available = True  # Assume market orders are always possible
        
        # Check for strongly correlated assets
        correlated_asset = self._find_correlated_asset(base_asset)
        cross_asset_available = correlated_asset is not None
        
        # Decide based on availability and preferences
        for hedge_type in self.preferred_hedge_types:
            if hedge_type == HedgeType.FUTURES and futures_available:
                return HedgeType.FUTURES
            elif hedge_type == HedgeType.INVERSE and inverse_available:
                return HedgeType.INVERSE
            elif hedge_type == HedgeType.CROSS_ASSET and cross_asset_available:
                return HedgeType.CROSS_ASSET
            elif hedge_type == HedgeType.STABLECOIN:
                return HedgeType.STABLECOIN
                
        # Default to inverse as fallback
        return HedgeType.INVERSE

    def _check_futures_available(self, symbol: str) -> bool:
        """Check if futures contracts are available for the symbol"""
        try:
            # Try to ping futures API endpoints 
            # This is exchange-specific so just return True for now
            return True
        except Exception:
            return False

    def _find_correlated_asset(self, base_asset: str) -> Optional[str]:
        """Find a strongly correlated asset for cross-asset hedging"""
        # Known correlation pairs (this would be dynamically updated)
        correlations = {
            "BTC": ["ETH", "BCH", "LTC"],
            "ETH": ["BTC", "LTC", "UNI"],
            "SOL": ["ETH", "AVAX"],
            "XRP": ["XLM", "ADA"]
        }
        
        if base_asset in correlations:
            return correlations[base_asset][0]  # Return first (strongest) correlation
            
        return None

    def _execute_inverse_hedge(self, symbol: str, size: float) -> Optional[Dict[str, Any]]:
        """
        Execute an inverse hedge (opposite position in same asset).
        
        Args:
            symbol: Trading pair
            size: Size to hedge
            
        Returns:
            Order result or None if failed
        """
        try:
            logging.info(f"Executing inverse hedge for {symbol} with size {size}")
            
            # Execute sell order
            order = self.exchange.create_market_sell_order(symbol, size)
            
            # Track hedge execution
            self.performance_metrics["api_calls"] += 1
            
            return order
        except Exception as e:
            logging.error(f"Error executing inverse hedge: {e}")
            return None

    def _execute_futures_hedge(self, symbol: str, size: float) -> Optional[Dict[str, Any]]:
        """
        Execute a futures hedge.
        
        Args:
            symbol: Trading pair
            size: Size to hedge
            
        Returns:
            Order result or None if failed
        """
        try:
            # Convert spot symbol to futures symbol if needed
            futures_symbol = self._get_futures_symbol(symbol)
            
            logging.info(f"Executing futures hedge for {futures_symbol} with size {size}")
            
            # Execute futures short
            if hasattr(self.exchange, 'create_market_sell_order'):
                # Most exchanges
                order = self.exchange.create_market_sell_order(
                    futures_symbol, 
                    size,
                    {'type': 'future'}
                )
            else:
                # Fallback with params
                order = self.exchange.create_order(
                    futures_symbol,
                    'market',
                    'sell',
                    size,
                    params={'type': 'future', 'reduceOnly': False}
                )
                
            # Track hedge execution
            self.performance_metrics["api_calls"] += 1
            
            return order
        except Exception as e:
            logging.error(f"Error executing futures hedge: {e}")
            return None

    def _get_futures_symbol(self, spot_symbol: str) -> str:
        """Convert spot symbol to futures symbol format"""
        # Different exchanges have different futures symbol formats
        if self.exchange_id == 'binance':
            base, quote = spot_symbol.split('/')
            return f"{base}{quote}_PERP"  # Example: BTCUSDT_PERP
        elif self.exchange_id == 'bitmex':
            base, quote = spot_symbol.split('/')
            return f"{base}{quote}:USDT"  # Example: BTCUSD:USDT
        else:
            # Default: just use the spot symbol
            return spot_symbol

    def _execute_cross_asset_hedge(self, symbol: str, size: float) -> Optional[Dict[str, Any]]:
        """
        Execute a cross-asset hedge using a correlated asset.
        
        Args:
            symbol: Trading pair
            size: Size to hedge
            
        Returns:
            Order result or None if failed
        """
        try:
            # Extract base asset
            base_asset = symbol.split('/')[0]
            
            # Find correlated asset
            correlated_asset = self._find_correlated_asset(base_asset)
            if not correlated_asset:
                logging.warning(f"No correlated asset found for {base_asset}")
                return None
                
            # Create correlated symbol
            quote_asset = symbol.split('/')[1]
            correlated_symbol = f"{correlated_asset}/{quote_asset}"
            
            # Get price ratio
            base_ticker = self.exchange.fetch_ticker(symbol)
            correlated_ticker = self.exchange.fetch_ticker(correlated_symbol)
            
            price_ratio = base_ticker['last'] / correlated_ticker['last']
            
            # Calculate hedge size
            correlated_size = size * price_ratio
            
            logging.info(f"Executing cross-asset hedge with {correlated_symbol} size {correlated_size:.6f}")
            
            # Execute sell order
            order = self.exchange.create_market_sell_order(correlated_symbol, correlated_size)
            
            # Modify order to include original symbol for reference
            order['cross_hedged_symbol'] = symbol
            order['symbol'] = correlated_symbol
            
            # Track hedge execution
            self.performance_metrics["api_calls"] += 2  # Two API calls (tickers)
            
            return order
        except Exception as e:
            logging.error(f"Error executing cross-asset hedge: {e}")
            return None

    def _execute_stablecoin_hedge(self, symbol: str, size: float) -> Optional[Dict[str, Any]]:
        """
        Execute a stablecoin hedge (convert to stablecoin).
        
        Args:
            symbol: Trading pair
            size: Size to hedge
            
        Returns:
            Order result or None if failed
        """
        try:
            logging.info(f"Executing stablecoin hedge for {symbol} with size {size}")
            
            # Just sell to stablecoin (same as inverse hedge)
            order = self.exchange.create_market_sell_order(symbol, size)
            
            # Track hedge execution
            self.performance_metrics["api_calls"] += 1
            
            return order
        except Exception as e:
            logging.error(f"Error executing stablecoin hedge: {e}")
            return None

    def _manage_existing_hedges(self) -> None:
        """Manage existing hedges (check for expiration, adjust, etc.)"""
        current_time = datetime.now()
        hedges_to_close = []
        
        for hedge_id, hedge in list(self.active_hedges.items()):
            # Check for expired hedges
            if hedge.expiration_time and current_time > hedge.expiration_time:
                logging.info(f"Hedge {hedge_id} has expired, closing")
                hedges_to_close.append(hedge_id)
                continue
                
            # Check if original position is closed
            original_position = self._get_position_by_id(hedge.original_position_id)
            if not original_position:
                logging.info(f"Original position {hedge.original_position_id} closed, closing hedge {hedge_id}")
                hedges_to_close.append(hedge_id)
                continue
                
            # Check if risk conditions are no longer present
            market_data = self._get_market_data(hedge.symbol)
            if not self.detect_risk_conditions(market_data):
                logging.info(f"Risk conditions no longer present for {hedge.symbol}, closing hedge {hedge_id}")
                hedges_to_close.append(hedge_id)
                continue
                
        # Close hedges identified for closure
        for hedge_id in hedges_to_close:
            self.close_hedge(hedge_id)

    @safe_execute(ErrorCategory.TRADE_EXECUTION, False) if HAVE_ERROR_HANDLING else lambda f: f
    def close_hedge(self, hedge_id: str) -> bool:
        """
        Close a specific hedge position.
        
        Args:
            hedge_id: ID of the hedge to close
            
        Returns:
            True if successful, False otherwise
        """
        if hedge_id not in self.active_hedges:
            logging.warning(f"Hedge {hedge_id} not found")
            return False
            
        hedge = self.active_hedges[hedge_id]
        
        try:
            logging.info(f"Closing hedge {hedge_id} for {hedge.symbol}")
            
            # Different closure method based on hedge type
            if hedge.hedge_type == HedgeType.INVERSE or hedge.hedge_type == HedgeType.STABLECOIN:
                # For inverse/stablecoin hedges, place a buy order
                order = self.exchange.create_market_buy_order(hedge.symbol, hedge.size)
            elif hedge.hedge_type == HedgeType.FUTURES:
                # For futures hedges, close position
                futures_symbol = self._get_futures_symbol(hedge.symbol)
                order = self.exchange.create_market_buy_order(
                    futures_symbol, 
                    hedge.size,
                    {'type': 'future', 'reduceOnly': True}
                )
            elif hedge.hedge_type == HedgeType.CROSS_ASSET:
                # For cross-asset hedges, buy back the correlated asset
                order = self.exchange.create_market_buy_order(hedge.hedge_symbol, hedge.size)
            else:
                # Default fallback
                order = self.exchange.create_market_buy_order(hedge.symbol, hedge.size)
                
            # Update hedge status
            hedge.status = "closed"
            hedge.exit_time = datetime.now()
            
            # Attempt to get exit price
            if order and 'price' in order:
                hedge.exit_price = order['price']
                
                # Calculate PnL if possible
                if hedge.entry_price:
                    if hedge.hedge_type in [HedgeType.INVERSE, HedgeType.FUTURES, HedgeType.STABLECOIN]:
                        # For short hedges, profit when price goes down
                        hedge.pnl = (hedge.entry_price - hedge.exit_price) * hedge.size
                    else:
                        # For complex hedges, simplified PnL
                        hedge.pnl = 0
            
            # Move to history
            self.hedge_history.append(hedge)
            del self.active_hedges[hedge_id]
            
            # Update performance metrics
            self.performance_metrics["num_hedges_closed"] += 1
            if hedge.pnl:
                self.performance_metrics["total_hedge_pnl"] += hedge.pnl
                
            # Calculate avg hedge duration if we have exit time and entry time
            if hedge.exit_time and hedge.entry_time:
                duration_hours = (hedge.exit_time - hedge.entry_time).total_seconds() / 3600
                
                # Update running average
                current_avg = self.performance_metrics["avg_hedge_duration"]
                closed_count = self.performance_metrics["num_hedges_closed"]
                
                if closed_count > 1:
                    # Update running average
                    self.performance_metrics["avg_hedge_duration"] = (
                        (current_avg * (closed_count - 1) + duration_hours) / closed_count
                    )
                else:
                    # First closed hedge
                    self.performance_metrics["avg_hedge_duration"] = duration_hours
            
            # Persist hedges
            self._save_hedges()
            
            # Track API call
            self.performance_metrics["api_calls"] += 1
            
            logging.info(f"Hedge {hedge_id} closed successfully")
            return True
            
        except Exception as e:
            logging.error(f"Error closing hedge {hedge_id}: {e}")
            return False

    def close_all_hedges(self) -> int:
        """
        Close all active hedges.
        
        Returns:
            Number of hedges successfully closed
        """
        closed_count = 0
        
        for hedge_id in list(self.active_hedges.keys()):
            if self.close_hedge(hedge_id):
                closed_count += 1
                
        return closed_count

    def get_hedge_status(self, hedge_id: str) -> Optional[Dict[str, Any]]:
        """
        Get the status of a specific hedge.
        
        Args:
            hedge_id: ID of the hedge
            
        Returns:
            Dictionary with hedge status or None if not found
        """
        if hedge_id in self.active_hedges:
            return self.active_hedges[hedge_id].to_dict()
            
        # Check history
        for hedge in self.hedge_history:
            if hedge.hedge_id == hedge_id:
                return hedge.to_dict()
                
        return None

    def get_active_hedges(self) -> List[Dict[str, Any]]:
        """
        Get all active hedges.
        
        Returns:
            List of active hedges
        """
        return [hedge.to_dict() for hedge in self.active_hedges.values()]

    def get_performance_metrics(self) -> Dict[str, Any]:
        """
        Get performance metrics for the hedging system.
        
        Returns:
            Dictionary with performance metrics
        """
        return self.performance_metrics.copy()

    def _save_hedges(self) -> bool:
        """
        Save hedge positions to persistent storage.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            # Create data dictionary
            data = {
                "active_hedges": [hedge.to_dict() for hedge in self.active_hedges.values()],
                "hedge_history": [hedge.to_dict() for hedge in self.hedge_history],
                "last_updated": datetime.now().isoformat(),
                "performance_metrics": self.performance_metrics
            }
            
            # Save to file
            file_path = os.path.join(self.persistence_path, "hedge_positions.json")
            with open(file_path, 'w') as f:
                json.dump(data, f, indent=2, default=str)
                
            return True
            
        except Exception as e:
            logging.error(f"Error saving hedges: {e}")
            return False

    def _load_hedges(self) -> bool:
        """
        Load hedge positions from persistent storage.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            file_path = os.path.join(self.persistence_path, "hedge_positions.json")
            
            if not os.path.exists(file_path):
                logging.info("No saved hedges found")
                return False
                
            with open(file_path, 'r') as f:
                data = json.load(f)
                
            # Load active hedges
            self.active_hedges = {}
            for hedge_data in data.get("active_hedges", []):
                hedge = HedgePosition.from_dict(hedge_data)
                self.active_hedges[hedge.hedge_id] = hedge
                
            # Load hedge history
            self.hedge_history = []
            for hedge_data in data.get("hedge_history", []):
                hedge = HedgePosition.from_dict(hedge_data)
                self.hedge_history.append(hedge)
                
            # Load performance metrics
            if "performance_metrics" in data:
                self.performance_metrics.update(data["performance_metrics"])
                
            logging.info(f"Loaded {len(self.active_hedges)} active hedges and {len(self.hedge_history)} historical hedges")
            return True
            
        except Exception as e:
            logging.error(f"Error loading hedges: {e}")
            return False

    def update_asset_correlations(self) -> None:
        """Update asset correlation matrix for cross-asset hedging"""
        try:
            # This would load correlation data from trading system or calculate it
            # For now, we'll use some hard-coded correlations
            self.asset_correlations = {
                ("BTC", "ETH"): 0.85,
                ("BTC", "LTC"): 0.75,
                ("BTC", "BCH"): 0.72,
                ("ETH", "LTC"): 0.68,
                ("XRP", "XLM"): 0.71,
                ("SOL", "ETH"): 0.65
            }
            
            logging.info("Asset correlations updated")
            
        except Exception as e:
            logging.error(f"Error updating asset correlations: {e}")

def create_smart_hedging(trading_system=None, config=None, **kwargs):
    """
    Factory function to create a SmartHedging instance.
    
    Args:
        trading_system: Reference to trading system
        config: Configuration dictionary
        kwargs: Additional parameters for SmartHedging
        
    Returns:
        Configured SmartHedging instance
    """
    # Create the hedging system
    hedging = SmartHedging(trading_system=trading_system, config=config, **kwargs)
    
    # Start monitoring if requested
    if kwargs.get('start_monitoring', True):
        hedging.start_monitoring()
        
    return hedging
