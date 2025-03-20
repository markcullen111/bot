# error_handling.py

import logging
import sys
import traceback
from enum import Enum
from typing import Any, Callable, Dict, Optional, Type, Union, TypeVar

# Define return type for decorated functions
T = TypeVar('T')

class ErrorSeverity(Enum):
    """Enum defining error severity levels for consistent handling."""
    DEBUG = 10
    INFO = 20
    WARNING = 30
    ERROR = 40
    CRITICAL = 50

class ErrorCategory(Enum):
    """Enum defining error categories for classification."""
    DATABASE = "database_error"
    NETWORK = "network_error"
    MODEL_LOADING = "model_loading_error"
    DATA_PROCESSING = "data_processing_error"
    STRATEGY = "strategy_error"
    RISK = "risk_management_error"
    TRADE_EXECUTION = "trade_execution_error"
    SYSTEM = "system_error"
    UNKNOWN = "unknown_error"

class TradingSystemError(Exception):
    """Base exception class for all trading system errors."""
    
    def __init__(
        self, 
        message: str, 
        category: ErrorCategory = ErrorCategory.UNKNOWN, 
        severity: ErrorSeverity = ErrorSeverity.ERROR,
        original_exception: Optional[Exception] = None
    ):
        self.message = message
        self.category = category
        self.severity = severity
        self.original_exception = original_exception
        super().__init__(message)

class ErrorHandler:
    """Centralized error handling for the trading system."""
    
    # Dictionary mapping error categories to recovery functions
    recovery_strategies: Dict[ErrorCategory, Callable] = {}
    
    # Dictionary holding default return values by error category
    default_returns: Dict[ErrorCategory, Any] = {}
    
    @classmethod
    def register_recovery_strategy(cls, category: ErrorCategory, strategy: Callable) -> None:
        """
        Register a recovery strategy for a specific error category.
        
        Args:
            category: The error category to register the strategy for
            strategy: The recovery function to call when this type of error occurs
        """
        cls.recovery_strategies[category] = strategy
        
    @classmethod
    def set_default_return(cls, category: ErrorCategory, default_value: Any) -> None:
        """
        Set a default return value for a specific error category.
        
        Args:
            category: The error category to set a default for
            default_value: The value to return when this type of error occurs
        """
        cls.default_returns[category] = default_value
    
    @classmethod
    def handle_error(
        cls,
        error: Union[Exception, TradingSystemError],
        category: Optional[ErrorCategory] = None,
        severity: Optional[ErrorSeverity] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> Optional[Any]:
        """
        Handle an error with appropriate logging and recovery.
        
        Args:
            error: The exception that occurred
            category: Override error category (for standard exceptions)
            severity: Override error severity
            context: Additional context information for logging
            
        Returns:
            Result from recovery strategy if available, otherwise None
        """
        # Extract information from the error
        if isinstance(error, TradingSystemError):
            message = error.message
            error_category = error.category
            error_severity = error.severity
            original_exception = error.original_exception
        else:
            message = str(error)
            error_category = category or ErrorCategory.UNKNOWN
            error_severity = severity or ErrorSeverity.ERROR
            original_exception = error
        
        # Prepare context for logging
        context_str = ''
        if context:
            context_str = ' | Context: ' + ', '.join(f"{k}={v}" for k, v in context.items())
            
        # Determine log level based on severity
        log_level = error_severity.value if isinstance(error_severity, ErrorSeverity) else logging.ERROR
        
        # Log the error with appropriate severity
        log_message = f"{error_category.value if isinstance(error_category, ErrorCategory) else 'ERROR'}: {message}{context_str}"
        
        logging.log(log_level, log_message)
        
        # Log traceback for ERROR and CRITICAL
        if log_level >= logging.ERROR:
            if original_exception:
                logging.error(f"Original exception traceback:\n{traceback.format_exception(type(original_exception), original_exception, original_exception.__traceback__)}")
            else:
                logging.error(f"Exception traceback:\n{traceback.format_exc()}")
        
        # Try to run recovery strategy
        if error_category in cls.recovery_strategies:
            try:
                recovery_func = cls.recovery_strategies[error_category]
                return recovery_func(error, context)
            except Exception as recovery_error:
                logging.error(f"Recovery strategy failed: {str(recovery_error)}")
        
        # Return default value if defined
        if error_category in cls.default_returns:
            return cls.default_returns[error_category]
            
        return None

def safe_execute(
    error_category: ErrorCategory, 
    default_return: Any = None,
    severity: ErrorSeverity = ErrorSeverity.ERROR
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """
    Decorator for safely executing functions with standardized error handling.
    
    Args:
        error_category: Category of error to use if this function fails
        default_return: Value to return if function fails
        severity: Severity level for logging errors
        
    Returns:
        Decorated function with error handling
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        def wrapper(*args, **kwargs) -> T:
            try:
                return func(*args, **kwargs)
            except Exception as e:
                context = {
                    "function": func.__name__,
                    "args": str(args),
                    "kwargs": str(kwargs)
                }
                
                # Register default return for this category if provided
                if default_return is not None:
                    ErrorHandler.set_default_return(error_category, default_return)
                
                # Handle the error and get recovery result
                result = ErrorHandler.handle_error(
                    error=e,
                    category=error_category,
                    severity=severity,
                    context=context
                )
                
                return result if result is not None else default_return
                
        return wrapper
    return decorator

# Register some default recovery strategies
def _database_recovery(error: Exception, context: Dict[str, Any]) -> Any:
    """Default recovery strategy for database errors."""
    logging.info("Attempting database connection recovery...")
    # In a real implementation, this would try to reconnect to the database
    return None

# Register the strategies
ErrorHandler.register_recovery_strategy(ErrorCategory.DATABASE, _database_recovery)

# Set some default return values
ErrorHandler.set_default_return(ErrorCategory.DATABASE, {})
ErrorHandler.set_default_return(ErrorCategory.NETWORK, {})
ErrorHandler.set_default_return(ErrorCategory.DATA_PROCESSING, [])
