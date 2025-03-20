# adaptive_learning.py

import numpy as np
import pandas as pd
import logging
import time
import pickle
import os
import json
import threading
import datetime
from typing import Dict, List, Tuple, Optional, Any, Union, Callable
from enum import Enum

# Specialized optimization libraries
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import mean_squared_error, accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import TimeSeriesSplit, cross_val_score

# Try to import optional optimization libraries
try:
    from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
    HYPEROPT_AVAILABLE = True
except ImportError:
    HYPEROPT_AVAILABLE = False
    logging.warning("Hyperopt not available. Advanced parameter optimization will use fallback methods.")

try:
    from bayes_opt import BayesianOptimization
    BAYESOPT_AVAILABLE = True
except ImportError:
    BAYESOPT_AVAILABLE = False
    logging.warning("Bayesian Optimization not available. Will use alternative optimization methods.")

# Try to import our error handling system
try:
    from error_handling import safe_execute, ErrorCategory, ErrorSeverity, ErrorHandler, TradingSystemError
    HAVE_ERROR_HANDLING = True
except ImportError:
    HAVE_ERROR_HANDLING = False
    logging.warning("Error handling module not available. Using simplified error handling.")

# Market regime detection
class MarketRegime(Enum):
    """Enum for different market regimes"""
    BULL_TREND = "bull_trend"
    BEAR_TREND = "bear_trend"
    BULL_VOLATILE = "bull_volatile"
    BEAR_VOLATILE = "bear_volatile"
    SIDEWAYS = "sideways"
    SIDEWAYS_VOLATILE = "sideways_volatile"
    UNKNOWN = "unknown"

class AdaptiveModel:
    """
    Base class for adaptive models that continuously learn from market data
    and trading results.
    """
    
    def __init__(self, model_name: str, model_type: str, feature_columns: List[str], target_column: str):
        """
        Initialize the adaptive model.
        
        Args:
            model_name: Unique identifier for the model
            model_type: Type of model ('classification' or 'regression')
            feature_columns: List of feature column names used for training
            target_column: Target column name for predictions
        """
        self.model_name = model_name
        self.model_type = model_type
        self.feature_columns = feature_columns
        self.target_column = target_column
        
        # Model and data processing components
        self.model = None
        self.feature_scaler = StandardScaler()
        self.target_scaler = None if model_type == 'classification' else StandardScaler()
        
        # Training history and performance tracking
        self.training_history = []
        self.performance_metrics = {}
        self.feature_importance = {}
        
        # Lock for thread safety during training and prediction
        self._lock = threading.RLock()
        
        # Model status
        self.is_trained = False
        self.last_training_time = None
        self.total_training_samples = 0
        self.version = 1
        
        # Initialize the model
        self._initialize_model()
    
    def _initialize_model(self):
        """Initialize the underlying machine learning model"""
        with self._lock:
            if self.model_type == 'classification':
                self.model = RandomForestClassifier(
                    n_estimators=100,
                    max_depth=10,
                    min_samples_leaf=5,
                    random_state=42
                )
            else:  # regression
                self.model = RandomForestRegressor(
                    n_estimators=100,
                    max_depth=10,
                    min_samples_leaf=5,
                    random_state=42
                )
    
    def train(self, data: pd.DataFrame, incremental: bool = False) -> Dict[str, Any]:
        """
        Train the model on new data.
        
        Args:
            data: DataFrame containing features and target
            incremental: If True, update model with new data; otherwise retrain from scratch
            
        Returns:
            Dictionary with training results and performance metrics
        """
        with self._lock:
            start_time = time.time()
            
            # Basic validation
            for col in self.feature_columns + [self.target_column]:
                if col not in data.columns:
                    raise ValueError(f"Column '{col}' not found in training data")
            
            # Extract features and target
            X = data[self.feature_columns].copy()
            y = data[self.target_column].copy()
            
            # Handle missing values
            X = X.fillna(method='ffill').fillna(0)
            if self.model_type == 'regression':
                y = y.fillna(method='ffill').fillna(0)
            
            # Scale features
            if not incremental or not self.is_trained:
                # Fit scaler on the new data
                X_scaled = self.feature_scaler.fit_transform(X)
                if self.target_scaler and self.model_type == 'regression':
                    y_scaled = self.target_scaler.fit_transform(y.values.reshape(-1, 1)).ravel()
                else:
                    y_scaled = y
            else:
                # Use existing scaler
                X_scaled = self.feature_scaler.transform(X)
                if self.target_scaler and self.model_type == 'regression':
                    y_scaled = self.target_scaler.transform(y.values.reshape(-1, 1)).ravel()
                else:
                    y_scaled = y
            
            # Train the model
            if incremental and self.is_trained and hasattr(self.model, 'partial_fit'):
                # Incremental training for models that support it
                classes = np.unique(y) if self.model_type == 'classification' else None
                self.model.partial_fit(X_scaled, y_scaled, classes=classes)
            else:
                # Full training
                self.model.fit(X_scaled, y_scaled)
            
            # Calculate performance metrics
            if self.model_type == 'classification':
                y_pred = self.model.predict(X_scaled)
                metrics = {
                    'accuracy': accuracy_score(y_scaled, y_pred),
                    'precision': precision_score(y_scaled, y_pred, average='weighted', zero_division=0),
                    'recall': recall_score(y_scaled, y_pred, average='weighted', zero_division=0),
                    'f1': f1_score(y_scaled, y_pred, average='weighted', zero_division=0)
                }
            else:  # regression
                y_pred = self.model.predict(X_scaled)
                metrics = {
                    'rmse': np.sqrt(mean_squared_error(y_scaled, y_pred)),
                    'mape': np.mean(np.abs((y_scaled - y_pred) / (y_scaled + 1e-10))) * 100,
                    'r2': self.model.score(X_scaled, y_scaled)
                }
            
            # Calculate feature importance
            if hasattr(self.model, 'feature_importances_'):
                self.feature_importance = dict(zip(self.feature_columns, self.model.feature_importances_))
            
            # Update training history
            self.training_history.append({
                'timestamp': datetime.datetime.now(),
                'samples': len(X),
                'metrics': metrics,
                'incremental': incremental
            })
            
            # Update model status
            self.is_trained = True
            self.last_training_time = datetime.datetime.now()
            self.total_training_samples += len(X)
            self.version += 1
            self.performance_metrics = metrics
            
            training_time = time.time() - start_time
            
            return {
                'success': True,
                'model_name': self.model_name,
                'samples_trained': len(X),
                'total_samples': self.total_training_samples,
                'training_time': training_time,
                'metrics': metrics,
                'feature_importance': self.feature_importance
            }
    
    def predict(self, data: pd.DataFrame) -> np.ndarray:
        """
        Generate predictions from the model.
        
        Args:
            data: DataFrame containing features to predict from
            
        Returns:
            NumPy array of predictions
        """
        with self._lock:
            if not self.is_trained:
                raise ValueError(f"Model '{self.model_name}' is not trained yet")
            
            # Extract and validate features
            if isinstance(data, pd.DataFrame):
                # Find which features are available
                available_features = [col for col in self.feature_columns if col in data.columns]
                missing_features = [col for col in self.feature_columns if col not in data.columns]
                
                if missing_features:
                    logging.warning(f"Missing features for model {self.model_name}: {missing_features}")
                    
                    # Create dummy data for missing features
                    for col in missing_features:
                        data[col] = 0
                
                X = data[self.feature_columns].copy()
            else:
                # Numpy array or list - assume correct features in correct order
                X = np.asarray(data)
                if len(X.shape) == 1:
                    X = X.reshape(1, -1)
                
                if X.shape[1] != len(self.feature_columns):
                    raise ValueError(f"Expected {len(self.feature_columns)} features, got {X.shape[1]}")
            
            # Handle missing values
            if isinstance(X, pd.DataFrame):
                X = X.fillna(method='ffill').fillna(0)
            else:
                X = np.nan_to_num(X)
            
            # Scale features
            X_scaled = self.feature_scaler.transform(X)
            
            # Make predictions
            predictions = self.model.predict(X_scaled)
            
            # If regression, inverse transform if needed
            if self.model_type == 'regression' and self.target_scaler:
                predictions = self.target_scaler.inverse_transform(predictions.reshape(-1, 1)).ravel()
            
            return predictions
    
    def predict_proba(self, data: pd.DataFrame) -> Optional[np.ndarray]:
        """
        Generate probability predictions from classification models.
        
        Args:
            data: DataFrame containing features to predict from
            
        Returns:
            NumPy array of class probabilities or None if not a classification model
        """
        with self._lock:
            if self.model_type != 'classification':
                return None
                
            if not self.is_trained:
                raise ValueError(f"Model '{self.model_name}' is not trained yet")
            
            # Extract features (reuse code from predict method)
            if isinstance(data, pd.DataFrame):
                # Find which features are available
                available_features = [col for col in self.feature_columns if col in data.columns]
                missing_features = [col for col in self.feature_columns if col not in data.columns]
                
                if missing_features:
                    logging.warning(f"Missing features for model {self.model_name}: {missing_features}")
                    
                    # Create dummy data for missing features
                    for col in missing_features:
                        data[col] = 0
                
                X = data[self.feature_columns].copy()
            else:
                # Numpy array or list - assume correct features in correct order
                X = np.asarray(data)
                if len(X.shape) == 1:
                    X = X.reshape(1, -1)
                
                if X.shape[1] != len(self.feature_columns):
                    raise ValueError(f"Expected {len(self.feature_columns)} features, got {X.shape[1]}")
            
            # Handle missing values
            if isinstance(X, pd.DataFrame):
                X = X.fillna(method='ffill').fillna(0)
            else:
                X = np.nan_to_num(X)
            
            # Scale features
            X_scaled = self.feature_scaler.transform(X)
            
            # Make probability predictions
            if hasattr(self.model, 'predict_proba'):
                return self.model.predict_proba(X_scaled)
            else:
                return None
    
    def evaluate(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Evaluate model performance on a validation dataset.
        
        Args:
            data: DataFrame containing features and target for evaluation
            
        Returns:
            Dictionary with evaluation metrics
        """
        with self._lock:
            if not self.is_trained:
                raise ValueError(f"Model '{self.model_name}' is not trained yet")
            
            # Extract features and target
            X = data[self.feature_columns].copy()
            y = data[self.target_column].copy()
            
            # Handle missing values
            X = X.fillna(method='ffill').fillna(0)
            if self.model_type == 'regression':
                y = y.fillna(method='ffill').fillna(0)
            
            # Scale features
            X_scaled = self.feature_scaler.transform(X)
            if self.target_scaler and self.model_type == 'regression':
                y_scaled = self.target_scaler.transform(y.values.reshape(-1, 1)).ravel()
            else:
                y_scaled = y
            
            # Generate predictions
            y_pred = self.model.predict(X_scaled)
            
            # Calculate metrics
            if self.model_type == 'classification':
                metrics = {
                    'accuracy': accuracy_score(y_scaled, y_pred),
                    'precision': precision_score(y_scaled, y_pred, average='weighted', zero_division=0),
                    'recall': recall_score(y_scaled, y_pred, average='weighted', zero_division=0),
                    'f1': f1_score(y_scaled, y_pred, average='weighted', zero_division=0)
                }
            else:  # regression
                metrics = {
                    'rmse': np.sqrt(mean_squared_error(y_scaled, y_pred)),
                    'mape': np.mean(np.abs((y_scaled - y_pred) / (y_scaled + 1e-10))) * 100,
                    'r2': self.model.score(X_scaled, y_scaled)
                }
            
            return {
                'model_name': self.model_name,
                'samples_evaluated': len(X),
                'metrics': metrics
            }
    
    def save(self, directory: str) -> str:
        """
        Save the model to disk.
        
        Args:
            directory: Directory to save the model to
            
        Returns:
            Path to the saved model file
        """
        with self._lock:
            os.makedirs(directory, exist_ok=True)
            
            # Create a model info dictionary
            model_info = {
                'model_name': self.model_name,
                'model_type': self.model_type,
                'feature_columns': self.feature_columns,
                'target_column': self.target_column,
                'is_trained': self.is_trained,
                'last_training_time': self.last_training_time.isoformat() if self.last_training_time else None,
                'total_training_samples': self.total_training_samples,
                'version': self.version,
                'feature_importance': self.feature_importance,
                'performance_metrics': self.performance_metrics,
                'training_history': [
                    {
                        'timestamp': entry['timestamp'].isoformat(),
                        'samples': entry['samples'],
                        'metrics': entry['metrics'],
                        'incremental': entry['incremental']
                    }
                    for entry in self.training_history
                ]
            }
            
            # Save model info
            info_path = os.path.join(directory, f"{self.model_name}_info.json")
            with open(info_path, 'w') as f:
                json.dump(model_info, f, indent=2)
            
            # Save the model components
            model_path = os.path.join(directory, f"{self.model_name}_model.pkl")
            with open(model_path, 'wb') as f:
                pickle.dump(self.model, f)
            
            # Save the scalers
            feature_scaler_path = os.path.join(directory, f"{self.model_name}_feature_scaler.pkl")
            with open(feature_scaler_path, 'wb') as f:
                pickle.dump(self.feature_scaler, f)
            
            if self.target_scaler:
                target_scaler_path = os.path.join(directory, f"{self.model_name}_target_scaler.pkl")
                with open(target_scaler_path, 'wb') as f:
                    pickle.dump(self.target_scaler, f)
            
            return model_path
    
    @classmethod
    def load(cls, directory: str, model_name: str) -> 'AdaptiveModel':
        """
        Load a model from disk.
        
        Args:
            directory: Directory to load the model from
            model_name: Name of the model to load
            
        Returns:
            Loaded AdaptiveModel instance
        """
        # Load model info
        info_path = os.path.join(directory, f"{model_name}_info.json")
        with open(info_path, 'r') as f:
            model_info = json.load(f)
        
        # Create model instance
        model = cls(
            model_name=model_info['model_name'],
            model_type=model_info['model_type'],
            feature_columns=model_info['feature_columns'],
            target_column=model_info['target_column']
        )
        
        # Load model components
        model_path = os.path.join(directory, f"{model_name}_model.pkl")
        with open(model_path, 'rb') as f:
            model.model = pickle.load(f)
        
        # Load feature scaler
        feature_scaler_path = os.path.join(directory, f"{model_name}_feature_scaler.pkl")
        with open(feature_scaler_path, 'rb') as f:
            model.feature_scaler = pickle.load(f)
        
        # Load target scaler if exists
        target_scaler_path = os.path.join(directory, f"{model_name}_target_scaler.pkl")
        if os.path.exists(target_scaler_path):
            with open(target_scaler_path, 'rb') as f:
                model.target_scaler = pickle.load(f)
        
        # Restore model status
        model.is_trained = model_info['is_trained']
        model.last_training_time = datetime.datetime.fromisoformat(model_info['last_training_time']) if model_info['last_training_time'] else None
        model.total_training_samples = model_info['total_training_samples']
        model.version = model_info['version']
        model.feature_importance = model_info['feature_importance']
        model.performance_metrics = model_info['performance_metrics']
        
        # Restore training history
        model.training_history = [
            {
                'timestamp': datetime.datetime.fromisoformat(entry['timestamp']),
                'samples': entry['samples'],
                'metrics': entry['metrics'],
                'incremental': entry['incremental']
            }
            for entry in model_info['training_history']
        ]
        
        return model


class ParameterOptimizer:
    """
    Optimizes strategy parameters based on backtesting results
    using advanced optimization techniques.
    """
    
    def __init__(self, optimization_method: str = 'bayesian'):
        """
        Initialize the parameter optimizer.
        
        Args:
            optimization_method: Method to use for optimization 
                ('bayesian', 'hyperopt', 'grid_search', or 'random_search')
        """
        self.optimization_method = optimization_method
        
        # Check if preferred method is available, otherwise fallback
        if optimization_method == 'bayesian' and not BAYESOPT_AVAILABLE:
            logging.warning("Bayesian optimization not available, falling back to grid search")
            self.optimization_method = 'grid_search'
        elif optimization_method == 'hyperopt' and not HYPEROPT_AVAILABLE:
            logging.warning("Hyperopt not available, falling back to grid search")
            self.optimization_method = 'grid_search'
        
        # Optimization results history
        self.optimization_history = []
        self.best_parameters = {}
        self.best_scores = {}
        
        # Parallelization settings
        self.n_jobs = 1
        
        # Threading lock
        self._lock = threading.RLock()
    
    def optimize(self, 
                 strategy_name: str, 
                 param_space: Dict[str, Any], 
                 objective_func: Callable, 
                 max_evals: int = 50,
                 direction: str = 'maximize') -> Dict[str, Any]:
        """
        Run parameter optimization for a strategy.
        
        Args:
            strategy_name: Name of the strategy to optimize
            param_space: Dictionary defining the parameter space
            objective_func: Function that evaluates a parameter set and returns a score
            max_evals: Maximum number of evaluations
            direction: Optimization direction ('maximize' or 'minimize')
            
        Returns:
            Dictionary with optimization results
        """
        with self._lock:
            start_time = time.time()
            
            # Standardize direction
            maximize = direction == 'maximize'
            
            # Select optimization method
            if self.optimization_method == 'bayesian' and BAYESOPT_AVAILABLE:
                best_params, best_score = self._bayesian_optimization(
                    param_space, objective_func, max_evals, maximize)
            elif self.optimization_method == 'hyperopt' and HYPEROPT_AVAILABLE:
                best_params, best_score = self._hyperopt_optimization(
                    param_space, objective_func, max_evals, maximize)
            elif self.optimization_method == 'random_search':
                best_params, best_score = self._random_search(
                    param_space, objective_func, max_evals, maximize)
            else:  # grid_search (default fallback)
                best_params, best_score = self._grid_search(
                    param_space, objective_func, max_evals, maximize)
            
            # Save results
            self.best_parameters[strategy_name] = best_params
            self.best_scores[strategy_name] = best_score
            
            # Add to optimization history
            self.optimization_history.append({
                'timestamp': datetime.datetime.now(),
                'strategy': strategy_name,
                'method': self.optimization_method,
                'max_evals': max_evals,
                'best_params': best_params,
                'best_score': best_score,
                'duration': time.time() - start_time
            })
            
            return {
                'strategy': strategy_name,
                'best_parameters': best_params,
                'best_score': best_score,
                'evaluations': max_evals,
                'method': self.optimization_method,
                'duration': time.time() - start_time
            }
    
    def _bayesian_optimization(self, param_space, objective_func, max_evals, maximize):
        """Run Bayesian optimization"""
        # Convert param space to bounds for BayesianOptimization
        bounds = {}
        param_types = {}
        
        for param, values in param_space.items():
            if isinstance(values, (list, tuple)):
                # Categorical parameter - BayesianOptimization doesn't support categorical
                # We'll handle this by using a numeric proxy
                param_types[param] = 'categorical'
                bounds[param] = (0, len(values) - 1)
            elif isinstance(values, dict) and 'min' in values and 'max' in values:
                param_types[param] = values.get('type', 'float')
                bounds[param] = (values['min'], values['max'])
            else:
                # Unsupported parameter type
                raise ValueError(f"Unsupported parameter space for {param}: {values}")
        
        # Create wrapper function for objective
        def bayesian_objective(**params):
            # Convert numeric proxies back to categorical values
            actual_params = {}
            for param, value in params.items():
                if param_types.get(param) == 'categorical':
                    # Convert to integer index
                    idx = int(value)
                    # Get the actual categorical value
                    actual_params[param] = param_space[param][idx]
                elif param_types.get(param) == 'int':
                    # Round to integer
                    actual_params[param] = int(round(value))
                else:
                    actual_params[param] = value
            
            # Call objective function
            score = objective_func(actual_params)
            return score if maximize else -score
        
        # Run optimization
        optimizer = BayesianOptimization(
            f=bayesian_objective,
            pbounds=bounds,
            random_state=42
        )
        
        optimizer.maximize(
            init_points=5,
            n_iter=max_evals - 5
        )
        
        # Get best parameters
        best_params = optimizer.max['params']
        best_score = optimizer.max['target']
        
        # Convert back to actual parameter values
        actual_best_params = {}
        for param, value in best_params.items():
            if param_types.get(param) == 'categorical':
                idx = int(value)
                actual_best_params[param] = param_space[param][idx]
            elif param_types.get(param) == 'int':
                actual_best_params[param] = int(round(value))
            else:
                actual_best_params[param] = value
        
        return actual_best_params, best_score if maximize else -best_score
    
    def _hyperopt_optimization(self, param_space, objective_func, max_evals, maximize):
        """Run optimization using hyperopt"""
        # Convert param space to hyperopt space
        space = {}
        
        for param, values in param_space.items():
            if isinstance(values, (list, tuple)):
                # Categorical parameter
                space[param] = hp.choice(param, values)
            elif isinstance(values, dict) and 'min' in values and 'max' in values:
                if values.get('type') == 'int':
                    space[param] = hp.randint(param, values['min'], values['max'] + 1)
                else:
                    space[param] = hp.uniform(param, values['min'], values['max'])
            else:
                # Unsupported parameter type
                raise ValueError(f"Unsupported parameter space for {param}: {values}")
        
        # Create wrapper function for objective
        def hyperopt_objective(params):
            score = objective_func(params)
            return {'loss': -score if maximize else score, 'status': STATUS_OK}
        
        # Run optimization
        trials = Trials()
        best = fmin(
            fn=hyperopt_objective,
            space=space,
            algo=tpe.suggest,
            max_evals=max_evals,
            trials=trials,
            rstate=np.random.RandomState(42)
        )
        
        # Get best parameters
        best_params = best
        
        # Convert choice indices back to actual values
        for param, value in best_params.items():
            if isinstance(param_space[param], (list, tuple)):
                best_params[param] = param_space[param][value]
        
        # Get best score
        best_score = -trials.best_trial['result']['loss'] if maximize else trials.best_trial['result']['loss']
        
        return best_params, best_score
    
    def _random_search(self, param_space, objective_func, max_evals, maximize):
        """Run random search optimization"""
        import random
        
        # Generate random parameter combinations
        best_score = float('-inf') if maximize else float('inf')
        best_params = None
        
        for _ in range(max_evals):
            # Generate random parameters
            params = {}
            for param, values in param_space.items():
                if isinstance(values, (list, tuple)):
                    # Categorical parameter
                    params[param] = random.choice(values)
                elif isinstance(values, dict) and 'min' in values and 'max' in values:
                    if values.get('type') == 'int':
                        params[param] = random.randint(values['min'], values['max'])
                    else:
                        params[param] = random.uniform(values['min'], values['max'])
                else:
                    # Unsupported parameter type
                    raise ValueError(f"Unsupported parameter space for {param}: {values}")
            
            # Evaluate
            score = objective_func(params)
            
            # Update best
            if (maximize and score > best_score) or (not maximize and score < best_score):
                best_score = score
                best_params = params
        
        return best_params, best_score
    
    def _grid_search(self, param_space, objective_func, max_evals, maximize):
        """Run grid search optimization"""
        # Generate grid search points (with max_evals limit)
        best_score = float('-inf') if maximize else float('inf')
        best_params = None
        
        # Calculate grid points per dimension based on max_evals
        n_dims = len(param_space)
        points_per_dim = max(2, int(np.power(max_evals, 1/n_dims)))
        
        # Generate parameter grid
        param_grid = {}
        
        for param, values in param_space.items():
            if isinstance(values, (list, tuple)):
                # Categorical parameter
                # If too many categories, sample uniformly
                if len(values) > points_per_dim:
                    indices = np.linspace(0, len(values) - 1, points_per_dim, dtype=int)
                    param_grid[param] = [values[i] for i in indices]
                else:
                    param_grid[param] = values
            elif isinstance(values, dict) and 'min' in values and 'max' in values:
                if values.get('type') == 'int':
                    param_grid[param] = np.linspace(
                        values['min'], values['max'], points_per_dim, dtype=int).tolist()
                else:
                    param_grid[param] = np.linspace(
                        values['min'], values['max'], points_per_dim).tolist()
            else:
                # Unsupported parameter type
                raise ValueError(f"Unsupported parameter space for {param}: {values}")
        
        # Generate all combinations (up to max_evals)
        from itertools import product
        
        params_list = list(product(*[param_grid[param] for param in param_space]))
        
        # Limit number of evaluations
        if len(params_list) > max_evals:
            import random
            random.shuffle(params_list)
            params_list = params_list[:max_evals]
        
        # Evaluate all combinations
        for param_values in params_list:
            params = dict(zip(param_space.keys(), param_values))
            
            # Evaluate
            score = objective_func(params)
            
            # Update best
            if (maximize and score > best_score) or (not maximize and score < best_score):
                best_score = score
                best_params = params
        
        return best_params, best_score


class MarketRegimeDetector:
    """
    Detects market regimes (trend, volatility, etc.) to enable adaptive
    parameter adjustments based on market conditions.
    """
    
    def __init__(self, lookback_window: int = 50):
        """
        Initialize the market regime detector.
        
        Args:
            lookback_window: Number of periods to use for market regime detection
        """
        self.lookback_window = lookback_window
        self.current_regime = MarketRegime.UNKNOWN
        self.regime_history = []
        self.regime_probabilities = {}
    
    def detect_regime(self, market_data: pd.DataFrame) -> MarketRegime:
        """
        Detect current market regime based on price action and volatility.
        
        Args:
            market_data: DataFrame with 'open', 'high', 'low', 'close', 'volume' columns
            
        Returns:
            Detected market regime
        """
        if len(market_data) < self.lookback_window:
            return MarketRegime.UNKNOWN
        
        # Calculate indicators for regime detection
        data = market_data.copy()
        
        # 1. Trend indicators
        data['sma_20'] = data['close'].rolling(window=20).mean()
        data['sma_50'] = data['close'].rolling(window=50).mean()
        
        # Calculate average directional movement over the lookback window
        data['returns'] = data['close'].pct_change()
        data['direction'] = (data['close'] > data['close'].shift(1)).astype(int) * 2 - 1  # +1 or -1
        
        # 2. Volatility indicators
        data['daily_range'] = (data['high'] - data['low']) / data['close']
        data['atr'] = self._calculate_atr(data)
        data['volatility'] = data['returns'].rolling(window=20).std() * np.sqrt(252)
        
        # Focus on the lookback window
        recent_data = data.iloc[-self.lookback_window:]
        
        # Average trend slope
        trend_slope = (recent_data['close'].iloc[-1] - recent_data['close'].iloc[0]) / recent_data['close'].iloc[0]
        
        # Volatility level (normalized)
        volatility_level = recent_data['volatility'].mean() / recent_data['volatility'].median()
        
        # Trend strength
        trend_strength = np.abs(recent_data['direction'].sum()) / len(recent_data)
        
        # Moving average alignment
        ma_aligned = (recent_data['sma_20'] > recent_data['sma_50']).sum() / len(recent_data)
        
        # Determine regime based on trend and volatility
        # High trend strength + low volatility = trending market
        # Low trend strength + high volatility = volatile directionless market
        # Low trend strength + low volatility = sideways market
        
        # Set thresholds
        trend_threshold = 0.3
        volatility_threshold = 1.5
        
        # Calculate regime probabilities
        self.regime_probabilities = {
            MarketRegime.BULL_TREND.value: 0.0,
            MarketRegime.BEAR_TREND.value: 0.0,
            MarketRegime.BULL_VOLATILE.value: 0.0,
            MarketRegime.BEAR_VOLATILE.value: 0.0,
            MarketRegime.SIDEWAYS.value: 0.0,
            MarketRegime.SIDEWAYS_VOLATILE.value: 0.0
        }
        
        if trend_strength > trend_threshold:
            # Trending market
            if trend_slope > 0:
                # Bullish trend
                if volatility_level > volatility_threshold:
                    regime = MarketRegime.BULL_VOLATILE
                    self.regime_probabilities[MarketRegime.BULL_VOLATILE.value] = 0.7
                    self.regime_probabilities[MarketRegime.BULL_TREND.value] = 0.3
                else:
                    regime = MarketRegime.BULL_TREND
                    self.regime_probabilities[MarketRegime.BULL_TREND.value] = 0.8
                    self.regime_probabilities[MarketRegime.BULL_VOLATILE.value] = 0.2
            else:
                # Bearish trend
                if volatility_level > volatility_threshold:
                    regime = MarketRegime.BEAR_VOLATILE
                    self.regime_probabilities[MarketRegime.BEAR_VOLATILE.value] = 0.7
                    self.regime_probabilities[MarketRegime.BEAR_TREND.value] = 0.3
                else:
                    regime = MarketRegime.BEAR_TREND
                    self.regime_probabilities[MarketRegime.BEAR_TREND.value] = 0.8
                    self.regime_probabilities[MarketRegime.BEAR_VOLATILE.value] = 0.2
        else:
            # Sideways market
            if volatility_level > volatility_threshold:
                regime = MarketRegime.SIDEWAYS_VOLATILE
                self.regime_probabilities[MarketRegime.SIDEWAYS_VOLATILE.value] = 0.8
                self.regime_probabilities[MarketRegime.SIDEWAYS.value] = 0.2
            else:
                regime = MarketRegime.SIDEWAYS
                self.regime_probabilities[MarketRegime.SIDEWAYS.value] = 0.8
                self.regime_probabilities[MarketRegime.SIDEWAYS_VOLATILE.value] = 0.2
        
        # Store current regime and add to history
        self.current_regime = regime
        self.regime_history.append({
            'timestamp': data.index[-1] if isinstance(data.index, pd.DatetimeIndex) else datetime.datetime.now(),
            'regime': regime,
            'trend_slope': trend_slope,
            'trend_strength': trend_strength,
            'volatility_level': volatility_level,
            'probabilities': self.regime_probabilities.copy()
        })
        
        return regime
    
    def _calculate_atr(self, data: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate Average True Range"""
        high = data['high']
        low = data['low']
        close = data['close'].shift(1)
        
        tr1 = high - low
        tr2 = abs(high - close)
        tr3 = abs(low - close)
        
        tr = pd.DataFrame({'tr1': tr1, 'tr2': tr2, 'tr3': tr3}).max(axis=1)
        atr = tr.rolling(window=period).mean()
        
        return atr
    
    def get_regime_history(self, n_periods: int = None) -> List[Dict[str, Any]]:
        """
        Get market regime history.
        
        Args:
            n_periods: Number of most recent periods to return (None = all)
            
        Returns:
            List of regime history entries
        """
        if n_periods is None:
            return self.regime_history
            
        return self.regime_history[-n_periods:]
    
    def get_regime_adjustment(self, strategy_type: str) -> Dict[str, float]:
        """
        Get parameter adjustment factors based on current market regime.
        
        Args:
            strategy_type: Type of strategy ('trend_following', 'mean_reversion', etc.)
            
        Returns:
            Dictionary of parameter adjustment factors
        """
        adjustments = {}
        
        # Base adjustments by strategy type and regime
        if strategy_type == 'trend_following':
            if self.current_regime in [MarketRegime.BULL_TREND, MarketRegime.BEAR_TREND]:
                # Increase trend following sensitivity in trending markets
                adjustments['short_window'] = 0.8  # Shorter window = more sensitive
                adjustments['medium_window'] = 0.9
                adjustments['long_window'] = 0.9
                adjustments['stop_loss_pct'] = 1.2  # Wider stop loss in trending markets
            elif self.current_regime in [MarketRegime.BULL_VOLATILE, MarketRegime.BEAR_VOLATILE]:
                # Decrease sensitivity in volatile markets
                adjustments['short_window'] = 1.2  # Longer window = less sensitive
                adjustments['medium_window'] = 1.1
                adjustments['long_window'] = 1.1
                adjustments['stop_loss_pct'] = 0.8  # Tighter stop loss in volatile markets
            else:  # Sideways
                # Neutral adjustments
                adjustments['short_window'] = 1.0
                adjustments['medium_window'] = 1.0
                adjustments['long_window'] = 1.0
                adjustments['stop_loss_pct'] = 0.9  # Slightly tighter stops in sideways markets
                
        elif strategy_type == 'mean_reversion':
            if self.current_regime in [MarketRegime.BULL_TREND, MarketRegime.BEAR_TREND]:
                # Decrease mean reversion sensitivity in trending markets
                adjustments['window'] = 1.2  # Longer window = less sensitive
                adjustments['std_dev'] = 1.2  # Wider bands = fewer signals
                adjustments['stop_loss_pct'] = 0.9  # Tighter stops in trending markets
            elif self.current_regime in [MarketRegime.SIDEWAYS, MarketRegime.SIDEWAYS_VOLATILE]:
                # Increase sensitivity in sideways markets
                adjustments['window'] = 0.8  # Shorter window = more sensitive
                adjustments['std_dev'] = 0.9  # Narrower bands = more signals
                adjustments['stop_loss_pct'] = 1.1  # Wider stops in sideways markets
            else:  # Volatile trending
                # Neutral settings
                adjustments['window'] = 1.0
                adjustments['std_dev'] = 1.0
                adjustments['stop_loss_pct'] = 1.0
                
        elif strategy_type == 'breakout':
            if self.current_regime in [MarketRegime.SIDEWAYS_VOLATILE, MarketRegime.BULL_VOLATILE, MarketRegime.BEAR_VOLATILE]:
                # Increase sensitivity in volatile markets
                adjustments['window'] = 0.7  # Shorter window = more sensitive
                adjustments['volume_threshold'] = 0.8  # Lower threshold = more signals
                adjustments['stop_loss_pct'] = 0.8  # Tighter stops in volatile markets
            else:  # Trending or sideways
                # Decrease sensitivity in non-volatile markets
                adjustments['window'] = 1.2  # Longer window = less sensitive
                adjustments['volume_threshold'] = 1.2  # Higher threshold = fewer signals
                adjustments['stop_loss_pct'] = 1.0  # Normal stops
                
        # Default adjustment for any parameter not explicitly set
        return adjustments


class BacktestLearner:
    """
    Enhanced adaptive learning system for backtesting and optimization.
    Learns from historical market data to optimize trading strategies.
    """
    
    def __init__(self, models_directory: str = "models"):
        """
        Initialize the backtest learner with advanced optimization capabilities.
        
        Args:
            models_directory: Directory to save trained models
        """
        # Parameter optimization
        self.param_optimizer = ParameterOptimizer()
        
        # Market regime detection
        self.regime_detector = MarketRegimeDetector()
        
        # Adaptive models
        self.models = {}
        self.models_directory = models_directory
        
        # Market data cache
        self.market_data = {}
        
        # Optimization history
        self.optimization_history = []
        
        # Create models directory if it doesn't exist
        os.makedirs(models_directory, exist_ok=True)
        
        # Thread lock for concurrent operations
        self._lock = threading.RLock()
    
    def load_data(self, data: pd.DataFrame, symbol: str = None):
        """
        Load historical market data for backtesting.
        
        Args:
            data: DataFrame containing market data
            symbol: Symbol/identifier for the data
        """
        if not isinstance(data, pd.DataFrame):
            raise ValueError("Data must be a pandas DataFrame")
            
        # Validate required columns
        required_columns = ['open', 'high', 'low', 'close']
        missing_columns = [col for col in required_columns if col not in data.columns]
        
        if missing_columns:
            raise ValueError(f"Data is missing required columns: {missing_columns}")
            
        # Make a copy to avoid modifying the original data
        with self._lock:
            self.market_data[symbol if symbol else 'default'] = data.copy()
            
            # Detect market regime
            self.regime_detector.detect_regime(data)
            
            logging.info(f"Loaded {len(data)} periods of market data" +
                        (f" for {symbol}" if symbol else ""))
    
    def optimize_parameters(self,
                           strategy_name: str,
                           param_space: Dict[str, Any],
                           target_metric: str = 'sharpe_ratio',
                           max_evals: int = 50) -> Dict[str, Any]:
        """
        Optimize strategy parameters using backtesting results.
        
        Args:
            strategy_name: Name of the strategy to optimize
            param_space: Dictionary defining the parameter space
            target_metric: Metric to optimize ('sharpe_ratio', 'returns', 'win_rate', etc.)
            max_evals: Maximum number of evaluations
            
        Returns:
            Dictionary with optimization results including best parameters and score
        """
        with self._lock:
            if not self.market_data:
                raise ValueError("No market data loaded for backtesting")
                
            # Get data for backtesting
            data = next(iter(self.market_data.values()))
            
            # Define objective function
            def objective_func(params):
                # Run backtest with given parameters
                result = self._run_backtest(strategy_name, params, data)
                
                # Extract target metric
                if target_metric not in result:
                    raise ValueError(f"Target metric '{target_metric}' not found in backtest results")
                    
                return result[target_metric]
            
            # Run optimization
            result = self.param_optimizer.optimize(
                strategy_name=strategy_name,
                param_space=param_space,
                objective_func=objective_func,
                max_evals=max_evals,
                direction='maximize'  # Assuming higher is better for all metrics
            )
            
            # Add to optimization history
            self.optimization_history.append({
                'timestamp': datetime.datetime.now(),
                'strategy': strategy_name,
                'target_metric': target_metric,
                'best_params': result['best_parameters'],
                'best_score': result['best_score']
            })
            
            # Adjust parameters based on current market regime
            adjusted_params = self._adjust_params_for_regime(
                strategy_name, result['best_parameters'])
            
            result['adjusted_parameters'] = adjusted_params
            result['market_regime'] = self.regime_detector.current_regime.value
            
            return result
    
    def _run_backtest(self, strategy_name: str, params: Dict[str, Any], data: pd.DataFrame) -> Dict[str, float]:
        """
        Run a backtest for a strategy with given parameters.
        
        Args:
            strategy_name: Name of the strategy to backtest
            params: Strategy parameters
            data: Market data for backtesting
            
        Returns:
            Dictionary with backtest metrics
        """
        # Create a simple moving average crossover strategy for demonstration
        # In a real system, this would use the actual strategy logic
        if strategy_name == 'trend_following':
            # Extract parameters with defaults
            short_window = params.get('short_window', 10)
            long_window = params.get('long_window', 50)
            
            # Calculate signals
            data = data.copy()
            data['short_ma'] = data['close'].rolling(window=short_window).mean()
            data['long_ma'] = data['close'].rolling(window=long_window).mean()
            
            # Generate signals
            data['signal'] = 0
            data.loc[data['short_ma'] > data['long_ma'], 'signal'] = 1
            data.loc[data['short_ma'] < data['long_ma'], 'signal'] = -1
            
            # Calculate returns
            data['returns'] = data['close'].pct_change()
            data['strategy_returns'] = data['signal'].shift(1) * data['returns']
            
            # Calculate metrics
            total_return = (1 + data['strategy_returns'].fillna(0)).prod() - 1
            annual_return = (1 + total_return) ** (252 / len(data)) - 1
            
            daily_returns = data['strategy_returns'].fillna(0)
            sharpe_ratio = daily_returns.mean() / daily_returns.std() * np.sqrt(252) if daily_returns.std() > 0 else 0
            
            win_rate = len(daily_returns[daily_returns > 0]) / len(daily_returns[daily_returns != 0]) if len(daily_returns[daily_returns != 0]) > 0 else 0
            
            # Calculate max drawdown
            cumulative_returns = (1 + daily_returns).cumprod()
            running_max = cumulative_returns.cummax()
            drawdown = (cumulative_returns / running_max - 1)
            max_drawdown = drawdown.min()
            
            return {
                'total_return': total_return,
                'annual_return': annual_return,
                'sharpe_ratio': sharpe_ratio,
                'win_rate': win_rate,
                'max_drawdown': max_drawdown
            }
            
        elif strategy_name == 'mean_reversion':
            # Extract parameters with defaults
            window = params.get('window', 20)
            std_dev = params.get('std_dev', 2.0)
            
            # Calculate signals
            data = data.copy()
            data['sma'] = data['close'].rolling(window=window).mean()
            data['std'] = data['close'].rolling(window=window).std()
            data['upper'] = data['sma'] + (data['std'] * std_dev)
            data['lower'] = data['sma'] - (data['std'] * std_dev)
            
            # Generate signals
            data['signal'] = 0
            data.loc[data['close'] < data['lower'], 'signal'] = 1  # Buy when price below lower band
            data.loc[data['close'] > data['upper'], 'signal'] = -1  # Sell when price above upper band
            
            # Calculate returns
            data['returns'] = data['close'].pct_change()
            data['strategy_returns'] = data['signal'].shift(1) * data['returns']
            
            # Calculate metrics (same as above)
            total_return = (1 + data['strategy_returns'].fillna(0)).prod() - 1
            annual_return = (1 + total_return) ** (252 / len(data)) - 1
            
            daily_returns = data['strategy_returns'].fillna(0)
            sharpe_ratio = daily_returns.mean() / daily_returns.std() * np.sqrt(252) if daily_returns.std() > 0 else 0
            
            win_rate = len(daily_returns[daily_returns > 0]) / len(daily_returns[daily_returns != 0]) if len(daily_returns[daily_returns != 0]) > 0 else 0
            
            # Calculate max drawdown
            cumulative_returns = (1 + daily_returns).cumprod()
            running_max = cumulative_returns.cummax()
            drawdown = (cumulative_returns / running_max - 1)
            max_drawdown = drawdown.min()
            
            return {
                'total_return': total_return,
                'annual_return': annual_return,
                'sharpe_ratio': sharpe_ratio,
                'win_rate': win_rate,
                'max_drawdown': max_drawdown
            }
            
        else:
            # For simplicity, return random metrics for unsupported strategies
            import random
            return {
                'total_return': random.uniform(0.1, 0.5),
                'annual_return': random.uniform(0.05, 0.3),
                'sharpe_ratio': random.uniform(0.5, 2.0),
                'win_rate': random.uniform(0.4, 0.7),
                'max_drawdown': random.uniform(-0.2, -0.05)
            }
    
    def _adjust_params_for_regime(self, strategy_type: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Adjust strategy parameters based on current market regime.
        
        Args:
            strategy_type: Type of strategy ('trend_following', 'mean_reversion', etc.)
            params: Original optimized parameters
            
        Returns:
            Adjusted parameters for current market regime
        """
        # Get adjustment factors
        adjustment_factors = self.regime_detector.get_regime_adjustment(strategy_type)
        
        # Apply adjustments
        adjusted_params = params.copy()
        
        for param, value in params.items():
            if param in adjustment_factors:
                # Apply adjustment factor if parameter is numeric
                if isinstance(value, (int, float)):
                    factor = adjustment_factors[param]
                    
                    if isinstance(value, int):
                        # For integers, round after adjustment
                        adjusted_params[param] = int(round(value * factor))
                    else:
                        # For floats, apply directly
                        adjusted_params[param] = value * factor
        
        return adjusted_params
    
    def create_adaptive_model(self, model_name: str, model_type: str,
                            feature_columns: List[str], target_column: str) -> AdaptiveModel:
        """
        Create a new adaptive model for continuous learning.
        
        Args:
            model_name: Unique identifier for the model
            model_type: Type of model ('classification' or 'regression')
            feature_columns: List of feature column names
            target_column: Target column name
            
        Returns:
            Created AdaptiveModel instance
        """
        with self._lock:
            if model_name in self.models:
                raise ValueError(f"Model '{model_name}' already exists")
                
            model = AdaptiveModel(
                model_name=model_name,
                model_type=model_type,
                feature_columns=feature_columns,
                target_column=target_column
            )
            
            self.models[model_name] = model
            return model
    
    def get_model(self, model_name: str) -> Optional[AdaptiveModel]:
        """
        Get an adaptive model by name.
        
        Args:
            model_name: Name of the model
            
        Returns:
            AdaptiveModel instance or None if not found
        """
        return self.models.get(model_name)
    
    def train_model(self, model_name: str, data: pd.DataFrame,
                  incremental: bool = False) -> Dict[str, Any]:
        """
        Train an adaptive model on new data.
        
        Args:
            model_name: Name of the model to train
            data: DataFrame containing features and target
            incremental: Whether to update model incrementally
            
        Returns:
            Dictionary with training results
        """
        with self._lock:
            model = self.get_model(model_name)
            
            if model is None:
                raise ValueError(f"Model '{model_name}' not found")
                
            result = model.train(data, incremental)
            
            # Save model
            model.save(self.models_directory)
            
            return result
    
    def predict_with_model(self, model_name: str, data: pd.DataFrame) -> np.ndarray:
        """
        Generate predictions using an adaptive model.
        
        Args:
            model_name: Name of the model
            data: DataFrame containing features
            
        Returns:
            NumPy array of predictions
        """
        with self._lock:
            model = self.get_model(model_name)
            
            if model is None:
                raise ValueError(f"Model '{model_name}' not found")
                
            return model.predict(data)
    
    def evaluate_model(self, model_name: str, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Evaluate an adaptive model's performance.
        
        Args:
            model_name: Name of the model
            data: DataFrame containing features and target
            
        Returns:
            Dictionary with evaluation results
        """
        with self._lock:
            model = self.get_model(model_name)
            
            if model is None:
                raise ValueError(f"Model '{model_name}' not found")
                
            return model.evaluate(data)
    
    def get_current_market_regime(self) -> Dict[str, Any]:
        """
        Get information about the current market regime.
        
        Returns:
            Dictionary with regime information
        """
        with self._lock:
            return {
                'regime': self.regime_detector.current_regime.value,
                'probabilities': self.regime_detector.regime_probabilities,
                'history': self.regime_detector.get_regime_history(5)  # Last 5 periods
            }
    
    def predict_strategy_performance(self, strategy_name: str, 
                                    parameters: Dict[str, Any]) -> Dict[str, float]:
        """
        Predict the performance of a strategy with given parameters
        in the current market regime.
        
        Args:
            strategy_name: Name of the strategy
            parameters: Strategy parameters
            
        Returns:
            Dictionary with predicted performance metrics
        """
        with self._lock:
            if not self.market_data:
                raise ValueError("No market data loaded for prediction")
                
            # Get data
            data = next(iter(self.market_data.values()))
            
            # Try to find a backtest performance model
            model_name = f"{strategy_name}_performance"
            model = self.get_model(model_name)
            
            if model is not None and model.is_trained:
                # Prepare feature data
                feature_data = pd.DataFrame([parameters])
                
                # Add market regime features
                regime_features = {
                    f"regime_{regime.value}": 1 if regime == self.regime_detector.current_regime else 0
                    for regime in MarketRegime
                }
                
                for key, value in regime_features.items():
                    feature_data[key] = value
                
                # Generate prediction
                prediction = model.predict(feature_data)
                
                # Format as dictionary
                if isinstance(prediction, np.ndarray):
                    prediction = prediction[0]
                    
                return {
                    'predicted_sharpe': float(prediction),
                    'confidence': 0.8,  # Placeholder
                    'market_regime': self.regime_detector.current_regime.value
                }
            else:
                # Fall back to backtest
                result = self._run_backtest(strategy_name, parameters, data)
                
                # Add confidence level
                result['confidence'] = 0.6  # Lower confidence since not from ML model
                result['market_regime'] = self.regime_detector.current_regime.value
                
                return result
    
    def save_state(self, filename: str = "backtest_learner_state.pkl") -> bool:
        """
        Save the state of the backtest learner.
        
        Args:
            filename: File name to save state to
            
        Returns:
            True if successful, False otherwise
        """
        with self._lock:
            try:
                # Save models
                for model_name, model in self.models.items():
                    model.save(self.models_directory)
                
                # Save other state (excluding models and large data)
                state = {
                    'optimization_history': self.optimization_history,
                    'regime_detector': {
                        'current_regime': self.regime_detector.current_regime.value,
                        'regime_history': self.regime_detector.regime_history
                    }
                }
                
                with open(filename, 'wb') as f:
                    pickle.dump(state, f)
                    
                logging.info(f"Backtest learner state saved to {filename}")
                return True
                
            except Exception as e:
                logging.error(f"Error saving backtest learner state: {e}")
                return False
    
    def load_state(self, filename: str = "backtest_learner_state.pkl") -> bool:
        """
        Load the state of the backtest learner.
        
        Args:
            filename: File name to load state from
            
        Returns:
            True if successful, False otherwise
        """
        with self._lock:
            try:
                # Load state
                with open(filename, 'rb') as f:
                    state = pickle.load(f)
                
                # Restore state
                self.optimization_history = state['optimization_history']
                
                # Restore regime detector state
                self.regime_detector.current_regime = MarketRegime(state['regime_detector']['current_regime'])
                self.regime_detector.regime_history = state['regime_detector']['regime_history']
                
                # Load models
                model_files = os.listdir(self.models_directory)
                model_info_files = [f for f in model_files if f.endswith('_info.json')]
                
                for info_file in model_info_files:
                    model_name = info_file.replace('_info.json', '')
                    
                    try:
                        model = AdaptiveModel.load(self.models_directory, model_name)
                        self.models[model_name] = model
                    except Exception as e:
                        logging.error(f"Error loading model {model_name}: {e}")
                
                logging.info(f"Backtest learner state loaded from {filename}")
                return True
                
            except Exception as e:
                logging.error(f"Error loading backtest learner state: {e}")
                return False

# Decorate with error handling if available
if HAVE_ERROR_HANDLING:
    # Apply safe_execute decorator to key methods
    BacktestLearner.optimize_parameters = safe_execute(
        ErrorCategory.DATA_PROCESSING, {'best_parameters': {}, 'best_score': 0})(BacktestLearner.optimize_parameters)
    
    BacktestLearner.predict_strategy_performance = safe_execute(
        ErrorCategory.DATA_PROCESSING, {'predicted_sharpe': 0, 'confidence': 0})(BacktestLearner.predict_strategy_performance)

def create_backtest_learner() -> BacktestLearner:
    """Factory function to create a BacktestLearner instance with proper initialization."""
    return BacktestLearner()
