# database_factory.py

import os
import logging
from typing import Union, Optional, Dict, Any

# Import both database implementations
try:
    from database_manager import DatabaseManager
    REAL_DB_AVAILABLE = True
except ImportError:
    REAL_DB_AVAILABLE = False
    logging.warning("Real database implementation not available. Will use mock database.")

try:
    from mock_database import MockDatabaseManager
    MOCK_DB_AVAILABLE = True
except ImportError:
    MOCK_DB_AVAILABLE = False
    logging.error("Mock database implementation not available. Database functionality will be limited.")

class DatabaseFactory:
    """
    Factory class for creating database connections with automatic fallback.
    Provides a unified interface for database operations regardless of backend.
    """
    
    @staticmethod
    def create_database(use_mock: bool = False, connection_params: Optional[Dict[str, Any]] = None) -> Union['DatabaseManager', 'MockDatabaseManager', None]:
        """
        Creates an appropriate database connection with fallback mechanisms.
        
        Args:
            use_mock (bool): If True, uses mock database regardless of real DB availability
            connection_params (dict): Optional connection parameters for real database
            
        Returns:
            Database manager instance or None if no implementation is available
        """
        connection_params = connection_params or {}
        
        # Case 1: User explicitly requested mock database
        if use_mock:
            if MOCK_DB_AVAILABLE:
                logging.info("Using mock database as requested")
                return MockDatabaseManager()
            else:
                logging.error("Mock database requested but implementation not available")
                return None
        
        # Case 2: Try to use real database first
        if REAL_DB_AVAILABLE:
            try:
                db_manager = DatabaseManager()
                
                # Test connection by executing a simple query
                test_success = db_manager._test_connection()
                if test_success:
                    logging.info("Successfully connected to real database")
                    return db_manager
                else:
                    logging.warning("Real database connection test failed")
                    raise Exception("Database connection test failed")
                    
            except Exception as e:
                logging.error(f"Error connecting to real database: {str(e)}")
                
                # Fall back to mock database if available
                if MOCK_DB_AVAILABLE:
                    logging.warning("Falling back to mock database")
                    return MockDatabaseManager()
                else:
                    logging.error("No database implementation available")
                    return None
        
        # Case 3: Real DB not available, try mock
        elif MOCK_DB_AVAILABLE:
            logging.warning("Real database not available, using mock database")
            return MockDatabaseManager()
        
        # Case 4: No database implementation available
        else:
            logging.error("No database implementation available")
            return None

# Modify DatabaseManager to add connection testing capabilities 
if REAL_DB_AVAILABLE:
    # Add method to DatabaseManager class
    def _test_connection(self):
        """Tests database connection by executing a simple query."""
        try:
            cursor = self.conn.cursor()
            cursor.execute("SELECT 1")
            result = cursor.fetchone()
            return result is not None and result[0] == 1
        except Exception as e:
            logging.error(f"Database connection test failed: {str(e)}")
            return False
            
    # Monkey patch the method into DatabaseManager
    setattr(DatabaseManager, '_test_connection', _test_connection)
