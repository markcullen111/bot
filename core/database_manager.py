# database_manager
import psycopg2
import pandas as pd
from sqlalchemy import create_engine
import logging
from datetime import datetime, timedelta
import os
from dotenv import load_dotenv
import json

class DatabaseManager:
    """Database manager for TimescaleDB integration"""
    
    def __init__(self):
        load_dotenv()
        
        # Get database connection details from environment variables
        self.db_host = os.getenv("DB_HOST", "localhost")
        self.db_port = os.getenv("DB_PORT", "5432")
        self.db_name = os.getenv("DB_NAME", "crypto_trading")
        self.db_user = os.getenv("DB_USER", "trading_user")
        self.db_password = os.getenv("DB_PASSWORD", "")
        
        # Connection objects
        self.conn = None
        self.engine = None
        
        # Initialize the connection
        self._initialize_connection()
        
    def _initialize_connection(self):
        """Initialize database connection"""
        try:
            # Create SQLAlchemy engine for pandas operations
            connection_string = f"postgresql://{self.db_user}:{self.db_password}@{self.db_host}:{self.db_port}/{self.db_name}"
            self.engine = create_engine(connection_string)
            
            # Create psycopg2 connection for other operations
            self.conn = psycopg2.connect(
                host=self.db_host,
                port=self.db_port,
                database=self.db_name,
                user=self.db_user,
                password=self.db_password
            )
            
            # Create necessary tables if they don't exist
            self._create_tables()
            
            logging.info("Database connection established successfully")
            return True
            
        except Exception as e:
            logging.error(f"Error connecting to database: {e}")
            return False
            
    def _create_tables(self):
        """Create necessary tables if they don't exist"""
        try:
            cursor = self.conn.cursor()
            
            # Create market data table (hypertable for time series)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS market_data (
                    time TIMESTAMPTZ NOT NULL,
                    symbol VARCHAR(20) NOT NULL,
                    open FLOAT,
                    high FLOAT,
                    low FLOAT,
                    close FLOAT,
                    volume FLOAT,
                    sma_20 FLOAT,
                    sma_50 FLOAT,
                    rsi FLOAT,
                    macd FLOAT,
                    signal_line FLOAT,
                    bollinger_upper FLOAT, 
                    bollinger_lower FLOAT,
                    atr FLOAT
                );
            """)
            
            # Check if it's already a hypertable, if not convert it
            cursor.execute("SELECT * FROM pg_extension WHERE extname = 'timescaledb';")
            if cursor.fetchone():
                cursor.execute("""
                    SELECT create_hypertable('market_data', 'time', if_not_exists => TRUE);
                """)
            
            # Create trades table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS trades (
                    id SERIAL PRIMARY KEY,
                    time TIMESTAMPTZ NOT NULL,
                    symbol VARCHAR(20) NOT NULL,
                    action VARCHAR(10) NOT NULL,
                    price FLOAT NOT NULL,
                    amount FLOAT NOT NULL,
                    stop_loss FLOAT,
                    take_profit FLOAT,
                    status VARCHAR(20) DEFAULT 'open',
                    order_id VARCHAR(100),
                    sl_order_id VARCHAR(100),
                    tp_order_id VARCHAR(100),
                    exit_price FLOAT,
                    exit_time TIMESTAMPTZ,
                    pnl FLOAT,
                    exit_reason VARCHAR(50),
                    strategy VARCHAR(50)
                );
            """)
            
            # Create portfolio history table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS portfolio_history (
                    time TIMESTAMPTZ NOT NULL,
                    total_value FLOAT NOT NULL,
                    cash_value FLOAT NOT NULL,
                    crypto_value FLOAT NOT NULL,
                    pnl_daily FLOAT,
                    PRIMARY KEY (time)
                );
            """)
            
            # Create model performance table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS model_performance (
                    time TIMESTAMPTZ NOT NULL,
                    model_name VARCHAR(50) NOT NULL,
                    metric_name VARCHAR(50) NOT NULL,
                    metric_value FLOAT NOT NULL,
                    PRIMARY KEY (time, model_name, metric_name)
                );
            """)
            
            # Create strategy performance table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS strategy_performance (
                    time TIMESTAMPTZ NOT NULL,
                    strategy_name VARCHAR(50) NOT NULL,
                    win_rate FLOAT,
                    profit_factor FLOAT,
                    sharpe_ratio FLOAT,
                    sortino_ratio FLOAT,
                    max_drawdown FLOAT,
                    total_trades INT,
                    average_trade FLOAT,
                    PRIMARY KEY (time, strategy_name)
                );
            """)
            
            # Create settings table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS settings (
                    id VARCHAR(50) PRIMARY KEY,
                    settings JSONB NOT NULL,
                    updated_at TIMESTAMPTZ NOT NULL
                );
            """)
            
            # Commit changes
            self.conn.commit()
            logging.info("Database tables created successfully")
            
        except Exception as e:
            self.conn.rollback()
            logging.error(f"Error creating tables: {e}")
            
    def store_market_data(self, df, symbol):
        """Store market data dataframe to database"""
        try:
            if df is None or df.empty:
                return False
                
            # Make a copy of the dataframe
            df_copy = df.copy()
            
            # Add symbol column
            df_copy['symbol'] = symbol
            
            # Reset index to make timestamp a column
            df_copy = df_copy.reset_index()
            df_copy.rename(columns={'index': 'time', 'timestamp': 'time'}, inplace=True)
            
            # Insert data using pandas
            df_copy.to_sql('market_data', self.engine, if_exists='append', index=False)
            
            logging.info(f"Stored {len(df_copy)} rows of market data for {symbol}")
            return True
            
        except Exception as e:
            logging.error(f"Error storing market data: {e}")
            return False
            
    def get_market_data(self, symbol, timeframe='1h', limit=100, start_time=None, end_time=None):
        """Get market data from database"""
        try:
            query = "SELECT * FROM market_data WHERE symbol = %s"
            params = [symbol]
            
            # Add time range filters if provided
            if start_time:
                query += " AND time >= %s"
                params.append(start_time)
                
            if end_time:
                query += " AND time <= %s"
                params.append(end_time)
                
            # Order by time and limit the results
            query += " ORDER BY time DESC LIMIT %s"
            params.append(limit)
            
            # Execute query
            df = pd.read_sql_query(query, self.engine, params=params)
            
            # Set time as index
            if not df.empty:
                df.set_index('time', inplace=True)
                
            return df
            
        except Exception as e:
            logging.error(f"Error fetching market data: {e}")
            return pd.DataFrame()
            
    def store_trade(self, trade_data):
        """Store trade to database"""
        try:
            cursor = self.conn.cursor()
            
            # Insert trade
            cursor.execute("""
                INSERT INTO trades (time, symbol, action, price, amount, stop_loss, take_profit, 
                                  status, order_id, sl_order_id, tp_order_id, strategy)
                VALUES (NOW(), %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                RETURNING id;
            """, (
                trade_data['symbol'],
                trade_data['action'],
                trade_data['price'],
                trade_data['amount'],
                trade_data.get('stop_loss'),
                trade_data.get('take_profit'),
                trade_data.get('status', 'open'),
                trade_data.get('order_id'),
                trade_data.get('sl_order_id'),
                trade_data.get('tp_order_id'),
                trade_data.get('strategy')
            ))
            
            trade_id = cursor.fetchone()[0]
            self.conn.commit()
            
            logging.info(f"Trade stored with ID: {trade_id}")
            return trade_id
            
        except Exception as e:
            self.conn.rollback()
            logging.error(f"Error storing trade: {e}")
            return None
            
    def update_trade(self, trade_id, update_data):
        """Update trade in database"""
        try:
            cursor = self.conn.cursor()
            
            # Build dynamic update query
            fields = []
            params = []
            
            for key, value in update_data.items():
                fields.append(f"{key} = %s")
                params.append(value)
                
            # Add trade_id to params
            params.append(trade_id)
            
            # Execute update
            cursor.execute(f"""
                UPDATE trades 
                SET {', '.join(fields)}
                WHERE id = %s
            """, params)
            
            self.conn.commit()
            logging.info(f"Trade {trade_id} updated")
            return True
            
        except Exception as e:
            self.conn.rollback()
            logging.error(f"Error updating trade: {e}")
            return False
            
    def get_open_trades(self, symbol=None):
        """Get all open trades"""
        try:
            query = "SELECT * FROM trades WHERE status = 'open'"
            params = []
            
            if symbol:
                query += " AND symbol = %s"
                params.append(symbol)
                
            # Execute query
            df = pd.read_sql_query(query, self.engine, params=params)
            return df
            
        except Exception as e:
            logging.error(f"Error getting open trades: {e}")
            return pd.DataFrame()
            
    def store_portfolio_snapshot(self, portfolio_data):
        """Store portfolio snapshot"""
        try:
            cursor = self.conn.cursor()
            
            cursor.execute("""
                INSERT INTO portfolio_history (time, total_value, cash_value, crypto_value, pnl_daily)
                VALUES (NOW(), %s, %s, %s, %s);
            """, (
                portfolio_data['total_value'],
                portfolio_data['cash_value'],
                portfolio_data['crypto_value'],
                portfolio_data.get('pnl_daily', 0)
            ))
            
            self.conn.commit()
            logging.info("Portfolio snapshot stored")
            return True
            
        except Exception as e:
            self.conn.rollback()
            logging.error(f"Error storing portfolio snapshot: {e}")
            return False
            
    def get_portfolio_history(self, days=30):
        """Get portfolio history"""
        try:
            query = f"""
                SELECT * FROM portfolio_history 
                WHERE time >= NOW() - INTERVAL '{days} days'
                ORDER BY time ASC;
            """
            
            df = pd.read_sql_query(query, self.engine)
            return df
            
        except Exception as e:
            logging.error(f"Error getting portfolio history: {e}")
            return pd.DataFrame()
            
    def store_model_performance(self, model_name, metrics):
        """Store model performance metrics"""
        try:
            cursor = self.conn.cursor()
            
            for metric_name, metric_value in metrics.items():
                cursor.execute("""
                    INSERT INTO model_performance (time, model_name, metric_name, metric_value)
                    VALUES (NOW(), %s, %s, %s);
                """, (model_name, metric_name, metric_value))
                
            self.conn.commit()
            logging.info(f"Performance metrics stored for model: {model_name}")
            return True
            
        except Exception as e:
            self.conn.rollback()
            logging.error(f"Error storing model performance: {e}")
            return False
            
    def store_strategy_performance(self, strategy_name, performance_data):
        """Store strategy performance"""
        try:
            cursor = self.conn.cursor()
            
            cursor.execute("""
                INSERT INTO strategy_performance 
                (time, strategy_name, win_rate, profit_factor, sharpe_ratio, 
                 sortino_ratio, max_drawdown, total_trades, average_trade)
                VALUES (NOW(), %s, %s, %s, %s, %s, %s, %s, %s);
            """, (
                strategy_name,
                performance_data.get('win_rate'),
                performance_data.get('profit_factor'),
                performance_data.get('sharpe_ratio'),
                performance_data.get('sortino_ratio'),
                performance_data.get('max_drawdown'),
                performance_data.get('total_trades'),
                performance_data.get('average_trade')
            ))
            
            self.conn.commit()
            logging.info(f"Performance data stored for strategy: {strategy_name}")
            return True
            
        except Exception as e:
            self.conn.rollback()
            logging.error(f"Error storing strategy performance: {e}")
            return False
            
    def store_settings(self, settings_id, settings_data):
        """Store application settings"""
        try:
            cursor = self.conn.cursor()
            
            cursor.execute("""
                INSERT INTO settings (id, settings, updated_at)
                VALUES (%s, %s, NOW())
                ON CONFLICT (id) DO UPDATE
                SET settings = EXCLUDED.settings, updated_at = NOW();
            """, (settings_id, json.dumps(settings_data)))
            
            self.conn.commit()
            logging.info(f"Settings stored: {settings_id}")
            return True
            
        except Exception as e:
            self.conn.rollback()
            logging.error(f"Error storing settings: {e}")
            return False
            
    def get_settings(self, settings_id):
        """Get application settings"""
        try:
            cursor = self.conn.cursor()
            
            cursor.execute("""
                SELECT settings FROM settings WHERE id = %s;
            """, (settings_id,))
            
            result = cursor.fetchone()
            
            if result:
                return json.loads(result[0])
            else:
                return {}
                
        except Exception as e:
            logging.error(f"Error getting settings: {e}")
            return {}
            
    def close(self):
        """Close database connection"""
        try:
            if self.conn:
                self.conn.close()
            logging.info("Database connection closed")
            
        except Exception as e:
            logging.error(f"Error closing database connection: {e}")
            
    def __del__(self):
        """Destructor to ensure connection is closed"""
        self.close()
