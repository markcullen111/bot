import logging
import logging.handlers
import os
from datetime import datetime
from pathlib import Path
from pythonjsonlogger import jsonlogger
from prometheus_client import Counter, Gauge, start_http_server

# Create logs directory if it doesn't exist
log_dir = Path("logs")
log_dir.mkdir(exist_ok=True)

# Prometheus metrics
ORDERS_TOTAL = Counter('trading_orders_total', 'Total number of orders placed', ['side', 'type'])
POSITION_VALUE = Gauge('trading_position_value', 'Current position value', ['symbol'])
ACCOUNT_BALANCE = Gauge('trading_account_balance', 'Current account balance')
ERROR_COUNT = Counter('trading_errors_total', 'Total number of errors', ['type'])

class CustomJsonFormatter(jsonlogger.JsonFormatter):
    def add_fields(self, log_record, record, message_dict):
        super(CustomJsonFormatter, self).add_fields(log_record, record, message_dict)
        log_record['timestamp'] = datetime.utcnow().isoformat()
        log_record['level'] = record.levelname
        log_record['logger'] = record.name

def setup_logging(level=logging.INFO):
    """Setup logging configuration for production environment."""
    
    # Create formatters
    json_formatter = CustomJsonFormatter(
        '%(timestamp)s %(level)s %(name)s %(message)s'
    )
    console_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Create handlers
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(console_formatter)
    
    # File handler for all logs
    file_handler = logging.handlers.RotatingFileHandler(
        filename=log_dir / "trading.log",
        maxBytes=10*1024*1024,  # 10MB
        backupCount=5
    )
    file_handler.setFormatter(json_formatter)
    
    # File handler for errors only
    error_handler = logging.handlers.RotatingFileHandler(
        filename=log_dir / "errors.log",
        maxBytes=10*1024*1024,  # 10MB
        backupCount=5
    )
    error_handler.setFormatter(json_formatter)
    error_handler.setLevel(logging.ERROR)
    
    # Setup root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    
    # Remove any existing handlers
    root_logger.handlers = []
    
    # Add handlers
    root_logger.addHandler(console_handler)
    root_logger.addHandler(file_handler)
    root_logger.addHandler(error_handler)
    
    # Start Prometheus metrics server
    try:
        start_http_server(8000)
        logging.info("Started Prometheus metrics server on port 8000")
    except Exception as e:
        logging.error(f"Failed to start Prometheus metrics server: {str(e)}")

def log_order(side: str, order_type: str):
    """Log order metrics to Prometheus."""
    ORDERS_TOTAL.labels(side=side, type=order_type).inc()

def log_position(symbol: str, value: float):
    """Log position value to Prometheus."""
    POSITION_VALUE.labels(symbol=symbol).set(value)

def log_balance(balance: float):
    """Log account balance to Prometheus."""
    ACCOUNT_BALANCE.set(balance)

def log_error(error_type: str):
    """Log error metrics to Prometheus."""
    ERROR_COUNT.labels(type=error_type).inc() 