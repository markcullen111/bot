from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Union
from dataclasses import dataclass
from enum import Enum

class OrderType(Enum):
    MARKET = "MARKET"
    LIMIT = "LIMIT"
    STOP = "STOP"
    STOP_LIMIT = "STOP_LIMIT"

class OrderSide(Enum):
    BUY = "BUY"
    SELL = "SELL"

@dataclass
class Position:
    symbol: str
    quantity: float
    avg_price: float
    current_price: float
    unrealized_pnl: float
    realized_pnl: float

@dataclass
class Order:
    symbol: str
    order_type: OrderType
    side: OrderSide
    quantity: float
    price: Optional[float] = None
    stop_price: Optional[float] = None
    order_id: Optional[str] = None
    status: str = "PENDING"

class BaseBroker(ABC):
    """Base class for all broker implementations."""
    
    @abstractmethod
    def connect(self) -> bool:
        """Establish connection to the broker."""
        pass
    
    @abstractmethod
    def disconnect(self) -> bool:
        """Disconnect from the broker."""
        pass
    
    @abstractmethod
    def get_account_info(self) -> Dict:
        """Get account information including balance, equity, margin, etc."""
        pass
    
    @abstractmethod
    def get_positions(self) -> List[Position]:
        """Get current positions."""
        pass
    
    @abstractmethod
    def place_order(self, order: Order) -> str:
        """Place a new order and return order ID."""
        pass
    
    @abstractmethod
    def cancel_order(self, order_id: str) -> bool:
        """Cancel an existing order."""
        pass
    
    @abstractmethod
    def get_order_status(self, order_id: str) -> str:
        """Get the current status of an order."""
        pass
    
    @abstractmethod
    def get_market_data(self, symbol: str) -> Dict:
        """Get real-time market data for a symbol."""
        pass
    
    @abstractmethod
    def get_historical_data(self, symbol: str, timeframe: str, start: str, end: str) -> List[Dict]:
        """Get historical market data."""
        pass
    
    @abstractmethod
    def get_account_balance(self) -> float:
        """Get current account balance."""
        pass
    
    @abstractmethod
    def get_portfolio_value(self) -> float:
        """Get total portfolio value including positions."""
        pass
    
    def is_market_open(self, symbol: str) -> bool:
        """Check if the market is open for a given symbol."""
        pass 