# system_integration.py

import os
import sys
import logging
import importlib
from typing import Dict, List, Any, Optional, Union, Set, Type

class SystemIntegration:
    """
    System-wide integration manager to ensure all components work together.
    Handles dynamic module loading, dependency resolution, and component verification.
    """
    
    def __init__(self):
        self.components = {}
        self.dependencies = {}
        self.import_paths = []
        self.verify_imports()
        
    def verify_imports(self) -> None:
        """Verifies that all necessary modules can be imported."""
        # Configure import paths
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.abspath(os.path.join(current_dir, os.pardir))
        
        # Add potential module locations to path
        paths_to_add = [
            current_dir,
            project_root,
            os.path.join(project_root, "models"),
            os.path.join(project_root, "ml_models"),
            os.path.join(project_root, "gui")
        ]
        
        for path in paths_to_add:
            if path not in sys.path and os.path.exists(path):
                sys.path.insert(0, path)
                self.import_paths.append(path)
                
        logging.info(f"Import paths configured: {', '.join(self.import_paths)}")
        
    def register_component(self, component_name: str, component: Any, dependencies: List[str] = None) -> None:
        """
        Register a component with the integration system.
        
        Args:
            component_name: Name of the component
            component: The component instance
            dependencies: List of component names this component depends on
        """
        self.components[component_name] = component
        self.dependencies[component_name] = dependencies or []
        logging.info(f"Registered component: {component_name}")
        
    def get_component(self, component_name: str) -> Optional[Any]:
        """
        Get a registered component by name.
        
        Args:
            component_name: Name of the component
            
        Returns:
            Component instance or None if not found
        """
        return self.components.get(component_name)
        
    def verify_dependencies(self) -> Dict[str, List[str]]:
        """
        Verify that all component dependencies are satisfied.
        
        Returns:
            Dict mapping component names to lists of missing dependencies
        """
        missing_dependencies = {}
        
        for component_name, deps in self.dependencies.items():
            missing = [dep for dep in deps if dep not in self.components]
            if missing:
                missing_dependencies[component_name] = missing
                
        if missing_dependencies:
            for component, missing in missing_dependencies.items():
                logging.warning(f"Component '{component}' is missing dependencies: {', '.join(missing)}")
        else:
            logging.info("All component dependencies are satisfied")
            
        return missing_dependencies
        
    def load_module(self, module_name: str) -> Optional[Any]:
        """
        Dynamically load a module with error handling.
        
        Args:
            module_name: Name of the module to load
            
        Returns:
            Loaded module or None if failed
        """
        try:
            return importlib.import_module(module_name)
        except ImportError as e:
            # Try alternative import paths
            for alt_name in [f".{module_name}", f"..{module_name}", module_name.split('.')[-1]]:
                try:
                    return importlib.import_module(alt_name)
                except ImportError:
                    continue
                    
            logging.error(f"Failed to import module {module_name}: {e}")
            return None
            
    def initialize_trading_system(self, config_path: str = None, use_mock_db: bool = False) -> Any:
        """
        Initialize the complete trading system with all components.
        
        Args:
            config_path: Path to configuration file
            use_mock_db: Whether to use mock database
            
        Returns:
            Initialized trading system or None if initialization failed
        """
        try:
            # Import core modules
            from trading_system import TradingSystem
            
            # Try to import database factory
            try:
                from database_factory import DatabaseFactory
                have_db_factory = True
            except ImportError:
                have_db_factory = False
                logging.warning("Database factory not available, falling back to direct initialization")
                
            # Initialize database
            if have_db_factory:
                db = DatabaseFactory.create_database(use_mock=use_mock_db)
            else:
                if use_mock_db:
                    try:
                        from mock_database import MockDatabaseManager
                        db = MockDatabaseManager()
                    except ImportError:
                        logging.error("Mock database not available")
                        db = None
                else:
                    try:
                        from database_manager import DatabaseManager
                        db = DatabaseManager()
                    except ImportError:
                        logging.error("Real database not available")
                        try:
                            from mock_database import MockDatabaseManager
                            db = MockDatabaseManager()
                            logging.warning("Falling back to mock database")
                        except ImportError:
                            db = None
            
            # Initialize trading system
            trading_system = TradingSystem(
                config_path=config_path,
                use_mock_db=use_mock_db,
                log_level=logging.INFO
            )
            
            # Set database if available
            if db is not None:
                trading_system.db = db
                
            # Register components
            self.register_component("trading_system", trading_system)
            
            if hasattr(trading_system, "strategy_system"):
                self.register_component("strategy_system", trading_system.strategy_system, ["trading_system"])
                
            if hasattr(trading_system, "risk_manager"):
                self.register_component("risk_manager", trading_system.risk_manager, ["trading_system"])
                
            # Try to initialize AI components
            try:
                # Import AI decision engine
                from ai_master_engine import AIDecisionEngine
                ai_engine = AIDecisionEngine()
                self.register_component("ai_engine", ai_engine)
                
                # Make AI available to trading system
                trading_system.ai_engine = ai_engine
                
            except ImportError:
                logging.warning("AI decision engine not available")
            
            # Try to initialize thread manager
            try:
                from thread_manager import ThreadManager
                thread_manager = ThreadManager(max_workers=4)
                self.register_component("thread_manager", thread_manager)
                
                # Make thread manager available to trading system
                trading_system.thread_manager = thread_manager
                
            except ImportError:
                logging.warning("Thread manager not available")
            
            # Verify all dependencies
            self.verify_dependencies()
            
            return trading_system
            
        except Exception as e:
            logging.error(f"Error initializing trading system: {e}")
            import traceback
            logging.error(traceback.format_exc())
            return None
            
    def check_model_availability(self) -> Dict[str, bool]:
        """
        Check which AI models are available.
        
        Returns:
            Dict mapping model names to availability status
        """
        models = {
            "market_making": "market_making_ai.py",
            "portfolio_allocation": "portfolio_allocation_ai.py",
            "trade_timing": "trade_timing_ai.py",
            "trade_exit": "trade_exit_ai.py",
            "trade_reentry": "trade_reentry_ai.py",
            "predictive_order_flow": "predictive_order_flow_ai.py",
            "risk_manager": "risk_manager.py"
        }
        
        availability = {}
        
        for model_name, filename in models.items():
            # Check model files
            file_found = False
            
            for path in self.import_paths:
                model_path = os.path.join(path, filename)
                if os.path.exists(model_path):
                    file_found = True
                    break
                    
                # Also check in ml_models subdirectory
                ml_models_path = os.path.join(path, "ml_models", filename)
                if os.path.exists(ml_models_path):
                    file_found = True
                    break
                    
                # Check in models subdirectory
                models_path = os.path.join(path, "models", filename)
                if os.path.exists(models_path):
                    file_found = True
                    break
            
            # Check model weights
            weights_found = False
            model_files = {
                "market_making": "market_making_model.pth",
                "portfolio_allocation": "portfolio_allocation_model.pth",
                "trade_timing": "trade_timing_model.pth",
                "trade_exit": "trade_exit_model.pth",
                "trade_reentry": "trade_reentry_model.pth",
                "predictive_order_flow": "order_flow_predictor.pth",
                "risk_manager": "risk_model.pth"
            }
            
            if model_name in model_files:
                weight_file = model_files[model_name]
                
                for path in self.import_paths:
                    # Check in models directory
                    weight_path = os.path.join(path, "models", weight_file)
                    if os.path.exists(weight_path):
                        weights_found = True
                        break
            
            availability[model_name] = file_found and weights_found
            
        return availability
        
    def shutdown(self) -> None:
        """Shutdown all components in the correct order."""
        # Get dependency order for shutdown (reverse of initialization)
        components_order = self._get_shutdown_order()
        
        # Shutdown components in order
        for component_name in components_order:
            component = self.components.get(component_name)
            if component is None:
                continue
                
            logging.info(f"Shutting down component: {component_name}")
            
            # Call shutdown method if available
            if hasattr(component, "shutdown"):
                try:
                    component.shutdown()
                except Exception as e:
                    logging.error(f"Error shutting down {component_name}: {e}")
            elif hasattr(component, "close"):
                try:
                    component.close()
                except Exception as e:
                    logging.error(f"Error closing {component_name}: {e}")
        
        logging.info("All components shut down")
        
    def _get_shutdown_order(self) -> List[str]:
        """
        Determine the order for shutting down components based on dependencies.
        
        Returns:
            List of component names in shutdown order
        """
        # Build reverse dependency graph
        reverse_deps = {comp: [] for comp in self.components}
        
        for comp, deps in self.dependencies.items():
            for dep in deps:
                if dep in reverse_deps:
                    reverse_deps[dep].append(comp)
        
        # Perform topological sort
        visited = set()
        temp_visited = set()
        order = []
        
        def visit(component):
            if component in temp_visited:
                # Circular dependency detected
                return
                
            if component in visited:
                return
                
            temp_visited.add(component)
            
            # Visit dependencies
            for dep in reverse_deps.get(component, []):
                visit(dep)
                
            temp_visited.remove(component)
            visited.add(component)
            order.append(component)
            
        # Visit all components
        for component in self.components:
            if component not in visited:
                visit(component)
                
        # Return components in topological order
        return order

# Create a singleton instance
system_integration = SystemIntegration()
