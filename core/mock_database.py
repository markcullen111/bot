# mock_database.py
import pandas as pd
import logging

class MockDatabaseManager:
    """Mock database manager for development without a real database"""
    
    def __init__(self):
        logging.info("Mock database manager initialized")
        self.market_data = {}
        self.trades = []
        self.portfolio_history = []
        
    def store_market_data(self, df, symbol):
        """Mock storing market data"""
        if df is not None and not df.empty:
            self.market_data[symbol] = df.copy()
            return True
        return False
        
    def get_market_data(self, symbol, timeframe='1h', limit=100, start_time=None, end_time=None):
        """Mock retrieving market data"""
        if symbol in self.market_data:
            df = self.market_data[symbol].copy()
            if limit and len(df) > limit:
                return df.iloc[-limit:]
            return df
        return pd.DataFrame()
        
    def store_trade(self, trade_data):
        """Mock storing trade"""
        trade_id = len(self.trades) + 1
        trade_data['id'] = trade_id
        self.trades.append(trade_data)
        return trade_id
        
    def update_trade(self, trade_id, update_data):
        """Mock updating trade"""
        for trade in self.trades:
            if trade.get('id') == trade_id:
                trade.update(update_data)
                return True
        return False
        
    def get_open_trades(self, symbol=None):
        """Mock getting open trades"""
        open_trades = [trade for trade in self.trades if trade.get('status') == 'open']
        if symbol:
            open_trades = [trade for trade in open_trades if trade.get('symbol') == symbol]
        return pd.DataFrame(open_trades)
        
    def store_portfolio_snapshot(self, portfolio_data):
        """Mock storing portfolio snapshot"""
        self.portfolio_history.append(portfolio_data)
        return True
        
    def get_portfolio_history(self, days=30):
        """Mock getting portfolio history"""
        return pd.DataFrame(self.portfolio_history[-days:])
        
    def store_settings(self, settings_id, settings_data):
        """Mock storing settings"""
        return True
        
    def get_settings(self, settings_id):
        """Mock getting settings"""
        return {}
        
    def close(self):
        """Mock closing connection"""
        logging.info("Mock database connection closed")
