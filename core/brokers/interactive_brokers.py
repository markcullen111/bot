from typing import Dict, List, Optional
import logging
from ib_insync import IB, Stock, MarketOrder, LimitOrder, StopOrder, util
from datetime import datetime, timedelta

from .base_broker import BaseBroker, Order, Position, OrderType, OrderSide

class InteractiveBrokers(BaseBroker):
    """Interactive Brokers implementation of the broker interface."""
    
    def __init__(self, host: str = 'localhost', port: int = 7497, client_id: int = 1):
        """Initialize IB connection parameters.
        
        Args:
            host: TWS/IB Gateway hostname
            port: TWS/IB Gateway port (7497 for TWS paper, 7496 for TWS live)
            client_id: Unique client ID
        """
        self.host = host
        self.port = port
        self.client_id = client_id
        self.ib = IB()
        self.connected = False
        
    def connect(self) -> bool:
        """Connect to Interactive Brokers TWS or Gateway."""
        try:
            self.ib.connect(self.host, self.port, clientId=self.client_id)
            self.connected = True
            logging.info(f"Successfully connected to IB on {self.host}:{self.port}")
            return True
        except Exception as e:
            logging.error(f"Failed to connect to IB: {str(e)}")
            return False
    
    def disconnect(self) -> bool:
        """Disconnect from Interactive Brokers."""
        try:
            self.ib.disconnect()
            self.connected = False
            return True
        except Exception as e:
            logging.error(f"Error disconnecting from IB: {str(e)}")
            return False
    
    def get_account_info(self) -> Dict:
        """Get account information."""
        if not self.connected:
            raise ConnectionError("Not connected to IB")
        
        account = self.ib.managedAccounts()[0]
        values = self.ib.accountValues(account)
        
        return {
            'net_liquidation': float(next((v.value for v in values if v.tag == 'NetLiquidation'), 0)),
            'buying_power': float(next((v.value for v in values if v.tag == 'BuyingPower'), 0)),
            'cash': float(next((v.value for v in values if v.tag == 'TotalCashValue'), 0)),
            'margin': float(next((v.value for v in values if v.tag == 'MaintMarginReq'), 0))
        }
    
    def get_positions(self) -> List[Position]:
        """Get current positions."""
        if not self.connected:
            raise ConnectionError("Not connected to IB")
            
        positions = []
        for pos in self.ib.positions():
            contract = pos.contract
            ticker = self.ib.reqMktData(contract)
            self.ib.sleep(1)  # Wait for market data
            
            positions.append(Position(
                symbol=contract.symbol,
                quantity=pos.position,
                avg_price=pos.avgCost,
                current_price=ticker.marketPrice(),
                unrealized_pnl=pos.unrealizedPNL,
                realized_pnl=pos.realizedPNL
            ))
        
        return positions
    
    def place_order(self, order: Order) -> str:
        """Place a new order."""
        if not self.connected:
            raise ConnectionError("Not connected to IB")
            
        contract = Stock(order.symbol, 'SMART', 'USD')
        
        # Convert our order type to IB order
        if order.order_type == OrderType.MARKET:
            ib_order = MarketOrder(
                'BUY' if order.side == OrderSide.BUY else 'SELL',
                order.quantity
            )
        elif order.order_type == OrderType.LIMIT:
            ib_order = LimitOrder(
                'BUY' if order.side == OrderSide.BUY else 'SELL',
                order.quantity,
                order.price
            )
        elif order.order_type == OrderType.STOP:
            ib_order = StopOrder(
                'BUY' if order.side == OrderSide.BUY else 'SELL',
                order.quantity,
                order.stop_price
            )
        
        trade = self.ib.placeOrder(contract, ib_order)
        self.ib.sleep(1)  # Give time for order to be processed
        
        return str(trade.order.orderId)
    
    def cancel_order(self, order_id: str) -> bool:
        """Cancel an existing order."""
        if not self.connected:
            raise ConnectionError("Not connected to IB")
            
        trades = self.ib.trades()
        trade = next((t for t in trades if str(t.order.orderId) == order_id), None)
        
        if trade:
            self.ib.cancelOrder(trade.order)
            return True
        return False
    
    def get_order_status(self, order_id: str) -> str:
        """Get current order status."""
        if not self.connected:
            raise ConnectionError("Not connected to IB")
            
        trades = self.ib.trades()
        trade = next((t for t in trades if str(t.order.orderId) == order_id), None)
        
        if trade:
            return trade.orderStatus.status
        return "NOT_FOUND"
    
    def get_market_data(self, symbol: str) -> Dict:
        """Get real-time market data."""
        if not self.connected:
            raise ConnectionError("Not connected to IB")
            
        contract = Stock(symbol, 'SMART', 'USD')
        ticker = self.ib.reqMktData(contract)
        self.ib.sleep(1)  # Wait for market data
        
        return {
            'bid': ticker.bid,
            'ask': ticker.ask,
            'last': ticker.last,
            'volume': ticker.volume,
            'high': ticker.high,
            'low': ticker.low,
            'close': ticker.close
        }
    
    def get_historical_data(self, symbol: str, timeframe: str, start: str, end: str) -> List[Dict]:
        """Get historical market data."""
        if not self.connected:
            raise ConnectionError("Not connected to IB")
            
        contract = Stock(symbol, 'SMART', 'USD')
        bars = self.ib.reqHistoricalData(
            contract,
            endDateTime=end,
            durationStr=timeframe,
            barSizeSetting='1 min',
            whatToShow='TRADES',
            useRTH=True
        )
        
        return [
            {
                'timestamp': bar.date,
                'open': bar.open,
                'high': bar.high,
                'low': bar.low,
                'close': bar.close,
                'volume': bar.volume
            }
            for bar in bars
        ]
    
    def get_account_balance(self) -> float:
        """Get current account balance."""
        account_info = self.get_account_info()
        return account_info['net_liquidation']
    
    def get_portfolio_value(self) -> float:
        """Get total portfolio value."""
        return self.get_account_balance()
    
    def is_market_open(self, symbol: str) -> bool:
        """Check if the market is open."""
        if not self.connected:
            raise ConnectionError("Not connected to IB")
            
        contract = Stock(symbol, 'SMART', 'USD')
        details = self.ib.reqContractDetails(contract)[0]
        
        # Get current time in exchange timezone
        exchange_tz = details.timeZoneId
        current_time = datetime.now(tz=exchange_tz)
        
        # Check if current time is within trading hours
        trading_hours = details.liquidHours
        return self._is_within_trading_hours(current_time, trading_hours)
    
    def _is_within_trading_hours(self, current_time: datetime, trading_hours: str) -> bool:
        """Helper method to check if current time is within trading hours."""
        # Parse trading hours string and check if current time falls within any trading window
        # This is a simplified implementation - you would need to properly parse the trading_hours string
        return True  # Placeholder implementation 