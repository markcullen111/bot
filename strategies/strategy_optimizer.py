# strategy_optimizer.py

import logging
import os
import time
import numpy as np
import pandas as pd
import json
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Any, Optional, Union, Callable, Iterator, TypeVar
import threading
import queue
import traceback
from sklearn.model_selection import TimeSeriesSplit, ParameterGrid
from scipy.optimize import minimize

# Type definitions for improved type checking
ParameterType = TypeVar('ParameterType', int, float, bool, str)
ParameterSpace = Dict[str, List[ParameterType]]
OptimizationResult = Dict[str, Any]
StrategyParameters = Dict[str, ParameterType]
PerformanceMetrics = Dict[str, float]

# Import error handling if available
try:
    from error_handling import safe_execute, ErrorCategory, ErrorSeverity, TradingSystemError
    HAVE_ERROR_HANDLING = True
except ImportError:
    HAVE_ERROR_HANDLING = False
    logging.warning("Error handling module not available. Using basic error handling.")

# Import thread management if available
try:
    from thread_manager import ThreadManager
    HAVE_THREAD_MANAGER = True
except ImportError:
    HAVE_THREAD_MANAGER = False
    logging.warning("Thread manager not available. Using single-threaded optimization.")

# Optional ML imports with graceful degradation
try:
    import optuna
    from optuna.samplers import TPESampler
    HAVE_OPTUNA = True
except ImportError:
    HAVE_OPTUNA = False
    logging.warning("Optuna not available. Bayesian optimization will be limited.")

try:
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.preprocessing import StandardScaler
    HAVE_SKLEARN = True
except ImportError:
    HAVE_SKLEARN = False
    logging.warning("Scikit-learn not available. ML-based optimization will be limited.")

try:
    from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
    HAVE_HYPEROPT = True
except ImportError:
    HAVE_HYPEROPT = False
    logging.warning("Hyperopt not available. Advanced Bayesian optimization will be limited.")

class OptimizationMethod:
    """Enumeration of supported optimization methods."""
    GRID_SEARCH = "grid_search"
    BAYESIAN = "bayesian"
    GENETIC = "genetic"
    WALK_FORWARD = "walk_forward"
    ML_GUIDED = "ml_guided"
    RANDOM_SEARCH = "random_search"


class StrategyOptimizer:
    """
    Advanced trading strategy optimizer with multiple optimization algorithms
    and comprehensive integration with the broader trading system.
    
    This class implements various optimization techniques for finding optimal
    strategy parameters based on historical performance, risk metrics, and
    machine learning predictions.
    
    Attributes:
        strategy_generator: Reference to the strategy generator component
        thread_manager: Optional thread manager for parallel optimization
        optimization_history: Dictionary storing optimization results
        parameter_importance: Tracked parameter importance metrics
        validation_periods: Number of periods for out-of-sample validation
        max_optimization_time: Maximum time allowed for optimization (seconds)
        optimization_lock: Thread lock for synchronization
        current_optimizations: Currently running optimization tasks
    """
    
    def __init__(
        self, 
        strategy_generator: Any, 
        thread_manager: Optional[Any] = None,
        validation_periods: int = 3,
        max_optimization_time: int = 3600,
        optimization_metrics: Optional[List[str]] = None
    ):
        """
        Initialize the strategy optimizer with necessary components and configuration.
        
        Args:
            strategy_generator: Reference to the strategy generator module
            thread_manager: Optional thread manager for parallel optimization
            validation_periods: Number of periods for out-of-sample validation
            max_optimization_time: Maximum time allowed for optimization (seconds)
            optimization_metrics: List of metrics to use for optimization (defaults to ["sharpe_ratio", "sortino_ratio", "profit_factor"])
        """
        self.strategy_generator = strategy_generator
        self.thread_manager = thread_manager
        
        # Optimization settings
        self.validation_periods = validation_periods
        self.max_optimization_time = max_optimization_time
        self.optimization_metrics = optimization_metrics or ["sharpe_ratio", "sortino_ratio", "profit_factor"]
        
        # Result tracking
        self.optimization_history: Dict[str, List[Dict[str, Any]]] = {}
        self.parameter_importance: Dict[str, Dict[str, float]] = {}
        
        # Thread safety
        self.optimization_lock = threading.RLock()
        self.current_optimizations: Dict[str, Dict[str, Any]] = {}
        
        # Performance tracking
        self.performance_metrics: Dict[str, Dict[str, float]] = {}
        
        # Logging configuration
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        
        # Parameter constraints storage
        self.parameter_constraints: Dict[str, Dict[str, Any]] = {}
        
        # Initialize data store directory
        self.data_dir = os.path.join("data", "optimization")
        os.makedirs(self.data_dir, exist_ok=True)
        
        self.logger.info("Strategy optimizer initialized successfully")
    
    def optimize_strategy(
        self, 
        strategy_name: str, 
        historical_data: pd.DataFrame, 
        parameter_space: ParameterSpace,
        optimization_method: str = OptimizationMethod.BAYESIAN,
        metric: str = "sharpe_ratio",
        max_evals: int = 100,
        use_threading: bool = True,
        callback: Optional[Callable[[Dict[str, Any]], None]] = None
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Optimize strategy parameters using the specified optimization method.
        
        Args:
            strategy_name: Name of the strategy to optimize
            historical_data: DataFrame containing historical market data
            parameter_space: Dictionary mapping parameter names to possible values
            optimization_method: Optimization algorithm to use
            metric: Primary metric to optimize (e.g., "sharpe_ratio")
            max_evals: Maximum number of parameter combinations to evaluate
            use_threading: Whether to use threaded optimization
            callback: Optional callback function for progress updates
            
        Returns:
            Tuple containing (best_parameters, optimization_results)
        """
        optimization_id = f"{strategy_name}_optimization_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Validate inputs
        self._validate_optimization_inputs(strategy_name, historical_data, parameter_space, metric)
        
        # Check if strategy exists
        if not hasattr(self.strategy_generator, 'strategies') or strategy_name not in self.strategy_generator.strategies:
            error_msg = f"Strategy '{strategy_name}' not found in strategy generator"
            self.logger.error(error_msg)
            if HAVE_ERROR_HANDLING:
                raise TradingSystemError(
                    message=error_msg,
                    category=ErrorCategory.STRATEGY,
                    severity=ErrorSeverity.ERROR
                )
            else:
                raise ValueError(error_msg)
        
        # If threading is enabled and thread manager is available, run asynchronously
        if use_threading and self.thread_manager and HAVE_THREAD_MANAGER:
            self.logger.info(f"Starting threaded optimization for {strategy_name} with {optimization_method}")
            
            # Store optimization metadata
            with self.optimization_lock:
                self.current_optimizations[optimization_id] = {
                    "strategy_name": strategy_name,
                    "status": "starting",
                    "start_time": datetime.now(),
                    "method": optimization_method,
                    "metric": metric,
                    "progress": 0
                }
            
            # Submit optimization task to thread manager
            self.thread_manager.submit_task(
                task_id=optimization_id,
                func=self._optimization_thread_wrapper,
                args=(
                    optimization_id, 
                    strategy_name, 
                    historical_data.copy(), 
                    parameter_space,
                    optimization_method,
                    metric,
                    max_evals,
                    callback
                ),
                priority=2,  # Medium priority
                timeout=self.max_optimization_time
            )
            
            # Return initial state and optimization ID
            return {}, {"optimization_id": optimization_id, "status": "running"}
        
        # Otherwise, run synchronously
        self.logger.info(f"Starting synchronous optimization for {strategy_name} with {optimization_method}")
        return self._run_optimization(
            strategy_name, 
            historical_data, 
            parameter_space,
            optimization_method,
            metric,
            max_evals,
            callback
        )
    
    def _optimization_thread_wrapper(
        self, 
        optimization_id: str,
        strategy_name: str,
        historical_data: pd.DataFrame,
        parameter_space: Dict[str, List[Any]],
        optimization_method: str,
        metric: str,
        max_evals: int,
        callback: Optional[Callable] = None,
        progress_callback: Optional[Callable] = None
    ) -> Dict[str, Any]:
        """
        Wrapper for threaded optimization to handle exceptions and update status.
        
        Args:
            optimization_id: Unique identifier for this optimization run
            strategy_name: Name of the strategy to optimize
            historical_data: DataFrame containing historical market data
            parameter_space: Dictionary mapping parameter names to possible values
            optimization_method: Optimization algorithm to use
            metric: Primary metric to optimize
            max_evals: Maximum number of parameter combinations to evaluate
            callback: Optional callback function for progress updates
            progress_callback: Thread manager progress callback
            
        Returns:
            Dictionary containing optimization results
        """
        try:
            # Update status to running
            with self.optimization_lock:
                if optimization_id in self.current_optimizations:
                    self.current_optimizations[optimization_id]["status"] = "running"
            
            # Setup progress tracking
            def track_progress(progress: int):
                """Update progress for both thread manager and callback"""
                if progress_callback:
                    progress_callback(progress)
                
                with self.optimization_lock:
                    if optimization_id in self.current_optimizations:
                        self.current_optimizations[optimization_id]["progress"] = progress
                
                if callback:
                    callback({"strategy_name": strategy_name, "progress": progress})
            
            # Run optimization
            best_params, results = self._run_optimization(
                strategy_name, 
                historical_data, 
                parameter_space,
                optimization_method,
                metric,
                max_evals,
                track_progress
            )
            
            # Update status to completed
            with self.optimization_lock:
                if optimization_id in self.current_optimizations:
                    self.current_optimizations[optimization_id].update({
                        "status": "completed",
                        "end_time": datetime.now(),
                        "results": results,
                        "best_params": best_params,
                        "progress": 100
                    })
            
            # Save results to disk
            self._save_optimization_results(optimization_id, strategy_name, best_params, results)
            
            return {
                "optimization_id": optimization_id,
                "strategy_name": strategy_name,
                "best_params": best_params,
                "results": results,
                "status": "completed"
            }
            
        except Exception as e:
            # Log the exception
            self.logger.error(f"Error during optimization {optimization_id}: {str(e)}")
            self.logger.error(traceback.format_exc())
            
            # Update status to failed
            with self.optimization_lock:
                if optimization_id in self.current_optimizations:
                    self.current_optimizations[optimization_id].update({
                        "status": "failed",
                        "end_time": datetime.now(),
                        "error": str(e)
                    })
            
            # Re-raise if using error handling, otherwise return error dict
            if HAVE_ERROR_HANDLING:
                raise TradingSystemError(
                    message=f"Optimization failed: {str(e)}",
                    category=ErrorCategory.STRATEGY,
                    severity=ErrorSeverity.ERROR,
                    original_exception=e
                )
            else:
                return {
                    "optimization_id": optimization_id,
                    "strategy_name": strategy_name,
                    "status": "failed",
                    "error": str(e)
                }
    
    def _run_optimization(
        self, 
        strategy_name: str,
        historical_data: pd.DataFrame,
        parameter_space: Dict[str, List[Any]],
        optimization_method: str,
        metric: str,
        max_evals: int,
        progress_callback: Optional[Callable] = None
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Execute the optimization process with the specified method.
        
        Args:
            strategy_name: Name of the strategy to optimize
            historical_data: DataFrame containing historical market data
            parameter_space: Dictionary mapping parameter names to possible values
            optimization_method: Optimization algorithm to use
            metric: Primary metric to optimize
            max_evals: Maximum number of parameter combinations to evaluate
            progress_callback: Optional callback for progress updates
            
        Returns:
            Tuple containing (best_parameters, optimization_results)
        """
        self.logger.info(f"Running {optimization_method} optimization for {strategy_name}")
        start_time = time.time()
        
        # Prepare data
        train_data, test_data = self._split_data_for_optimization(historical_data)
        
        # Initialize optimization tracking
        evaluations = []
        best_score = float('-inf')
        best_params = None
        
        # Select optimization method
        if optimization_method == OptimizationMethod.GRID_SEARCH:
            best_params, evaluations = self._run_grid_search(
                strategy_name, train_data, test_data, parameter_space, metric, max_evals, progress_callback
            )
        elif optimization_method == OptimizationMethod.BAYESIAN and HAVE_OPTUNA:
            best_params, evaluations = self._run_bayesian_optimization(
                strategy_name, train_data, test_data, parameter_space, metric, max_evals, progress_callback
            )
        elif optimization_method == OptimizationMethod.GENETIC:
            best_params, evaluations = self._run_genetic_algorithm(
                strategy_name, train_data, test_data, parameter_space, metric, max_evals, progress_callback
            )
        elif optimization_method == OptimizationMethod.WALK_FORWARD:
            best_params, evaluations = self._run_walk_forward_optimization(
                strategy_name, historical_data, parameter_space, metric, max_evals, progress_callback
            )
        elif optimization_method == OptimizationMethod.ML_GUIDED and HAVE_SKLEARN:
            best_params, evaluations = self._run_ml_guided_optimization(
                strategy_name, train_data, test_data, parameter_space, metric, max_evals, progress_callback
            )
        elif optimization_method == OptimizationMethod.RANDOM_SEARCH:
            best_params, evaluations = self._run_random_search(
                strategy_name, train_data, test_data, parameter_space, metric, max_evals, progress_callback
            )
        else:
            # Fallback to grid search if method not available
            self.logger.warning(f"Optimization method {optimization_method} not available or missing dependencies. Falling back to grid search.")
            best_params, evaluations = self._run_grid_search(
                strategy_name, train_data, test_data, parameter_space, metric, max_evals, progress_callback
            )
        
        # Calculate execution time
        execution_time = time.time() - start_time
        
        # Store results in optimization history
        result = {
            "strategy": strategy_name,
            "method": optimization_method,
            "metric": metric,
            "best_score": best_score if best_score > float('-inf') else None,
            "best_params": best_params,
            "evaluations": len(evaluations),
            "execution_time": execution_time,
            "timestamp": datetime.now().isoformat()
        }
        
        with self.optimization_lock:
            if strategy_name not in self.optimization_history:
                self.optimization_history[strategy_name] = []
            self.optimization_history[strategy_name].append(result)
        
        # Calculate parameter importance if we have enough evaluations
        if len(evaluations) >= 10:
            self._calculate_parameter_importance(strategy_name, evaluations)
        
        # Run final validation on best parameters
        if best_params:
            validation_metrics = self._validate_parameters(strategy_name, best_params, test_data)
            result["validation_metrics"] = validation_metrics
        
        self.logger.info(f"Optimization completed for {strategy_name}. Best parameters: {best_params}")
        
        return best_params, result
    
    def _run_grid_search(
        self, 
        strategy_name: str,
        train_data: pd.DataFrame,
        test_data: pd.DataFrame,
        parameter_space: Dict[str, List[Any]],
        metric: str,
        max_evals: int,
        progress_callback: Optional[Callable] = None
    ) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
        """
        Run grid search optimization for strategy parameters.
        
        Args:
            strategy_name: Name of the strategy to optimize
            train_data: Training data for optimization
            test_data: Test data for validation
            parameter_space: Dictionary mapping parameter names to possible values
            metric: Primary metric to optimize
            max_evals: Maximum number of parameter combinations to evaluate
            progress_callback: Optional callback for progress updates
            
        Returns:
            Tuple containing (best_parameters, list_of_evaluations)
        """
        self.logger.info(f"Running grid search for {strategy_name} with max evaluations: {max_evals}")
        
        # Create parameter grid
        param_grid = list(ParameterGrid(parameter_space))
        
        # Limit evaluations if needed
        if len(param_grid) > max_evals:
            self.logger.warning(f"Parameter grid size ({len(param_grid)}) exceeds max_evals ({max_evals}). Sampling grid.")
            np.random.shuffle(param_grid)
            param_grid = param_grid[:max_evals]
        
        # Track best parameters
        best_score = float('-inf')
        best_params = None
        evaluations = []
        
        # Evaluate each parameter combination
        for i, params in enumerate(param_grid):
            try:
                # Execute strategy with current parameters
                performance = self._evaluate_parameters(strategy_name, params, train_data)
                
                # Update progress
                if progress_callback:
                    progress = int((i + 1) * 100 / len(param_grid))
                    progress_callback(progress)
                
                # Extract optimization metric
                if metric in performance:
                    score = performance[metric]
                    
                    # Record evaluation
                    eval_record = {
                        "params": params.copy(),
                        "metrics": performance,
                        "score": score
                    }
                    evaluations.append(eval_record)
                    
                    # Update best parameters if score is better
                    if score > best_score:
                        best_score = score
                        best_params = params.copy()
                        self.logger.debug(f"New best parameters found: {best_params} with score: {best_score}")
            
            except Exception as e:
                self.logger.warning(f"Error evaluating parameters {params}: {str(e)}")
                continue
        
        self.logger.info(f"Grid search completed with {len(evaluations)} evaluations")
        return best_params, evaluations
    
    def _run_bayesian_optimization(
        self, 
        strategy_name: str,
        train_data: pd.DataFrame,
        test_data: pd.DataFrame,
        parameter_space: Dict[str, List[Any]],
        metric: str,
        max_evals: int,
        progress_callback: Optional[Callable] = None
    ) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
        """
        Run Bayesian optimization for strategy parameters using Optuna.
        
        Args:
            strategy_name: Name of the strategy to optimize
            train_data: Training data for optimization
            test_data: Test data for validation
            parameter_space: Dictionary mapping parameter names to possible values
            metric: Primary metric to optimize
            max_evals: Maximum number of parameter combinations to evaluate
            progress_callback: Optional callback for progress updates
            
        Returns:
            Tuple containing (best_parameters, list_of_evaluations)
        """
        if not HAVE_OPTUNA:
            self.logger.warning("Optuna not available. Falling back to grid search.")
            return self._run_grid_search(
                strategy_name, train_data, test_data, parameter_space, metric, max_evals, progress_callback
            )
        
        self.logger.info(f"Running Bayesian optimization for {strategy_name} with max evaluations: {max_evals}")
        
        # Store evaluations for tracking
        evaluations = []
        
        # Define the objective function for optimization
        def objective(trial):
            # Generate parameters for this trial
            params = {}
            for param_name, param_values in parameter_space.items():
                if isinstance(param_values, list):
                    if all(isinstance(x, int) for x in param_values):
                        params[param_name] = trial.suggest_int(
                            param_name, min(param_values), max(param_values)
                        )
                    elif all(isinstance(x, float) for x in param_values):
                        params[param_name] = trial.suggest_float(
                            param_name, min(param_values), max(param_values)
                        )
                    elif all(isinstance(x, bool) for x in param_values):
                        params[param_name] = trial.suggest_categorical(param_name, param_values)
                    else:
                        params[param_name] = trial.suggest_categorical(param_name, param_values)
                else:
                    # Handle range specifications
                    if isinstance(param_values, dict) and 'min' in param_values and 'max' in param_values:
                        if param_values.get('type') == 'int':
                            params[param_name] = trial.suggest_int(
                                param_name, param_values['min'], param_values['max']
                            )
                        else:
                            params[param_name] = trial.suggest_float(
                                param_name, param_values['min'], param_values['max']
                            )
            
            # Evaluate parameters
            try:
                performance = self._evaluate_parameters(strategy_name, params, train_data)
                
                # Extract optimization metric
                if metric in performance:
                    score = performance[metric]
                    
                    # Record evaluation
                    eval_record = {
                        "params": params.copy(),
                        "metrics": performance,
                        "score": score,
                        "trial": trial.number
                    }
                    evaluations.append(eval_record)
                    
                    # Update progress
                    if progress_callback:
                        progress = int((len(evaluations)) * 100 / max_evals)
                        progress_callback(min(progress, 100))
                    
                    return score
                else:
                    return float('-inf')
            
            except Exception as e:
                self.logger.warning(f"Error evaluating parameters {params}: {str(e)}")
                return float('-inf')
        
        # Create Optuna study
        study = optuna.create_study(
            direction="maximize",
            sampler=TPESampler(seed=42)
        )
        
        # Run optimization
        study.optimize(objective, n_trials=max_evals)
        
        # Get best parameters
        best_params = study.best_params
        
        self.logger.info(f"Bayesian optimization completed with {len(evaluations)} evaluations")
        return best_params, evaluations
    
    def _run_genetic_algorithm(
        self, 
        strategy_name: str,
        train_data: pd.DataFrame,
        test_data: pd.DataFrame,
        parameter_space: Dict[str, List[Any]],
        metric: str,
        max_evals: int,
        progress_callback: Optional[Callable] = None
    ) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
        """
        Run genetic algorithm optimization for strategy parameters.
        
        Args:
            strategy_name: Name of the strategy to optimize
            train_data: Training data for optimization
            test_data: Test data for validation
            parameter_space: Dictionary mapping parameter names to possible values
            metric: Primary metric to optimize
            max_evals: Maximum number of parameter combinations to evaluate
            progress_callback: Optional callback for progress updates
            
        Returns:
            Tuple containing (best_parameters, list_of_evaluations)
        """
        self.logger.info(f"Running genetic algorithm for {strategy_name} with max evaluations: {max_evals}")
        
        # Track evaluations
        evaluations = []
        
        # Define genetic algorithm parameters
        population_size = min(50, max_evals // 4)
        generations = max_evals // population_size
        mutation_rate = 0.1
        crossover_rate = 0.7
        elitism_count = max(2, population_size // 10)
        
        # Function to create a random individual
        def create_individual():
            """Generate random parameters from parameter space"""
            individual = {}
            for param_name, param_values in parameter_space.items():
                if isinstance(param_values, list):
                    individual[param_name] = np.random.choice(param_values)
                elif isinstance(param_values, dict) and 'min' in param_values and 'max' in param_values:
                    if param_values.get('type') == 'int':
                        individual[param_name] = np.random.randint(
                            param_values['min'], param_values['max'] + 1
                        )
                    else:
                        individual[param_name] = np.random.uniform(
                            param_values['min'], param_values['max']
                        )
            return individual
        
        # Function to evaluate fitness of an individual
        def evaluate_fitness(individual):
            """Evaluate fitness of parameter set"""
            try:
                performance = self._evaluate_parameters(strategy_name, individual, train_data)
                
                # Extract optimization metric
                if metric in performance:
                    score = performance[metric]
                    
                    # Record evaluation
                    eval_record = {
                        "params": individual.copy(),
                        "metrics": performance,
                        "score": score
                    }
                    evaluations.append(eval_record)
                    
                    return score
                else:
                    return float('-inf')
            
            except Exception as e:
                self.logger.warning(f"Error evaluating parameters {individual}: {str(e)}")
                return float('-inf')
        
        # Function to perform crossover between two individuals
        def crossover(parent1, parent2):
            """Perform crossover between two parents"""
            if np.random.random() > crossover_rate:
                return parent1.copy(), parent2.copy()
            
            child1, child2 = {}, {}
            for param_name in parameter_space.keys():
                if np.random.random() < 0.5:
                    child1[param_name] = parent1[param_name]
                    child2[param_name] = parent2[param_name]
                else:
                    child1[param_name] = parent2[param_name]
                    child2[param_name] = parent1[param_name]
            
            return child1, child2
        
        # Function to mutate an individual
        def mutate(individual):
            """Randomly mutate parameters"""
            mutated = individual.copy()
            for param_name, param_values in parameter_space.items():
                if np.random.random() < mutation_rate:
                    if isinstance(param_values, list):
                        mutated[param_name] = np.random.choice(param_values)
                    elif isinstance(param_values, dict) and 'min' in param_values and 'max' in param_values:
                        if param_values.get('type') == 'int':
                            mutated[param_name] = np.random.randint(
                                param_values['min'], param_values['max'] + 1
                            )
                        else:
                            mutated[param_name] = np.random.uniform(
                                param_values['min'], param_values['max']
                            )
            return mutated
        
        # Initialize population
        population = [create_individual() for _ in range(population_size)]
        
        # Track best individual
        best_individual = None
        best_fitness = float('-inf')
        
        # Run genetic algorithm
        for generation in range(generations):
            # Evaluate fitness of all individuals
            fitness_scores = []
            for individual in population:
                fitness = evaluate_fitness(individual)
                fitness_scores.append((individual, fitness))
            
            # Sort by fitness
            fitness_scores.sort(key=lambda x: x[1], reverse=True)
            
            # Update best individual
            if fitness_scores[0][1] > best_fitness:
                best_individual = fitness_scores[0][0].copy()
                best_fitness = fitness_scores[0][1]
                self.logger.debug(f"New best individual found: {best_individual} with fitness: {best_fitness}")
            
            # Report progress
            if progress_callback:
                progress = int((generation + 1) * 100 / generations)
                progress_callback(progress)
            
            # Check if we've reached max evaluations
            if len(evaluations) >= max_evals:
                self.logger.info(f"Reached max evaluations ({max_evals}). Stopping.")
                break
            
            # Create next generation
            next_population = []
            
            # Elitism: Carry over best individuals
            next_population.extend([x[0].copy() for x in fitness_scores[:elitism_count]])
            
            # Create rest of population through selection, crossover, and mutation
            while len(next_population) < population_size:
                # Tournament selection
                parent1 = self._tournament_selection(fitness_scores, tournament_size=3)
                parent2 = self._tournament_selection(fitness_scores, tournament_size=3)
                
                # Crossover
                child1, child2 = crossover(parent1, parent2)
                
                # Mutation
                child1 = mutate(child1)
                child2 = mutate(child2)
                
                # Add to next generation
                next_population.append(child1)
                if len(next_population) < population_size:
                    next_population.append(child2)
            
            # Update population
            population = next_population
        
        self.logger.info(f"Genetic algorithm completed with {len(evaluations)} evaluations")
        return best_individual, evaluations
    
    def _tournament_selection(self, fitness_scores, tournament_size=3):
        """
        Perform tournament selection to choose a parent.
        
        Args:
            fitness_scores: List of (individual, fitness) tuples
            tournament_size: Number of individuals in each tournament
            
        Returns:
            Selected individual
        """
        indices = np.random.choice(len(fitness_scores), tournament_size, replace=False)
        tournament = [fitness_scores[i] for i in indices]
        tournament.sort(key=lambda x: x[1], reverse=True)
        return tournament[0][0].copy()
    
    def _run_walk_forward_optimization(
        self, 
        strategy_name: str,
        historical_data: pd.DataFrame,
        parameter_space: Dict[str, List[Any]],
        metric: str,
        max_evals: int,
        progress_callback: Optional[Callable] = None
    ) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
        """
        Run walk-forward optimization for robust parameter selection.
        
        Args:
            strategy_name: Name of the strategy to optimize
            historical_data: DataFrame containing historical market data
            parameter_space: Dictionary mapping parameter names to possible values
            metric: Primary metric to optimize
            max_evals: Maximum number of parameter combinations to evaluate
            progress_callback: Optional callback for progress updates
            
        Returns:
            Tuple containing (best_parameters, list_of_evaluations)
        """
        self.logger.info(f"Running walk-forward optimization for {strategy_name}")
        
        # Configuration
        n_splits = 5  # Number of time periods for walk-forward testing
        
        # Time-series cross-validation
        tscv = TimeSeriesSplit(n_splits=n_splits)
        
        # Track results for each fold
        fold_results = []
        all_evaluations = []
        
        # Track overall progress
        total_progress = 0
        
        # Perform walk-forward optimization
        for i, (train_index, test_index) in enumerate(tscv.split(historical_data)):
            self.logger.info(f"Running fold {i+1}/{n_splits}")
            
            # Split data
            train_data = historical_data.iloc[train_index].copy()
            test_data = historical_data.iloc[test_index].copy()
            
            # Run grid search on this fold with reduced max_evals
            fold_max_evals = max_evals // n_splits
            best_params, evaluations = self._run_grid_search(
                strategy_name, 
                train_data, 
                test_data, 
                parameter_space, 
                metric, 
                fold_max_evals,
                lambda p: None  # No progress callback for individual folds
            )
            
            # Validate best parameters on test data
            if best_params:
                test_performance = self._evaluate_parameters(strategy_name, best_params, test_data)
                
                # Store fold results
                fold_results.append({
                    "fold": i,
                    "best_params": best_params,
                    "train_metrics": evaluations[-1]["metrics"] if evaluations else {},
                    "test_metrics": test_performance
                })
                
                # Add fold tag to evaluations and store
                for eval_data in evaluations:
                    eval_data["fold"] = i
                all_evaluations.extend(evaluations)
            
            # Update progress
            total_progress += 100 // n_splits
            if progress_callback:
                progress_callback(min(total_progress, 100))
        
        # Aggregate results across folds to find most stable parameters
        if fold_results:
            # Find parameter combinations that performed well across all folds
            best_overall_params = self._aggregate_walk_forward_results(fold_results, metric)
            
            self.logger.info(f"Walk-forward optimization completed with {len(all_evaluations)} evaluations")
            return best_overall_params, all_evaluations
        else:
            self.logger.warning("No valid fold results. Falling back to grid search.")
            return self._run_grid_search(
                strategy_name, 
                historical_data, 
                pd.DataFrame(), 
                parameter_space, 
                metric, 
                max_evals, 
                progress_callback
            )
    
    def _aggregate_walk_forward_results(
        self, 
        fold_results: List[Dict[str, Any]], 
        metric: str
    ) -> Dict[str, Any]:
        """
        Aggregate walk-forward optimization results to find robust parameters.
        
        Args:
            fold_results: List of results from each fold
            metric: Primary metric to optimize
            
        Returns:
            Best overall parameters
        """
        self.logger.info("Aggregating walk-forward optimization results")
        
        # Extract all parameter combinations
        all_params = [fold["best_params"] for fold in fold_results]
        
        # Count parameter frequency
        param_counts = {}
        
        for params in all_params:
            param_str = json.dumps(params, sort_keys=True)
            if param_str not in param_counts:
                param_counts[param_str] = {
                    "params": params,
                    "count": 0,
                    "train_metrics": [],
                    "test_metrics": []
                }
            param_counts[param_str]["count"] += 1
        
        # Add metrics for each parameter combination
        for fold in fold_results:
            param_str = json.dumps(fold["best_params"], sort_keys=True)
            if "train_metrics" in fold and metric in fold["train_metrics"]:
                param_counts[param_str]["train_metrics"].append(fold["train_metrics"][metric])
            if "test_metrics" in fold and metric in fold["test_metrics"]:
                param_counts[param_str]["test_metrics"].append(fold["test_metrics"][metric])
        
        # Calculate average metrics
        for param_key in param_counts:
            param_data = param_counts[param_key]
            if param_data["train_metrics"]:
                param_data["avg_train_metric"] = np.mean(param_data["train_metrics"])
            else:
                param_data["avg_train_metric"] = float('-inf')
                
            if param_data["test_metrics"]:
                param_data["avg_test_metric"] = np.mean(param_data["test_metrics"])
            else:
                param_data["avg_test_metric"] = float('-inf')
        
        # Sort by frequency, then by average test metric, then by average train metric
        sorted_params = sorted(
            param_counts.values(), 
            key=lambda x: (
                x["count"], 
                x["avg_test_metric"] if x["avg_test_metric"] != float('-inf') else -999, 
                x["avg_train_metric"] if x["avg_train_metric"] != float('-inf') else -999
            ), 
            reverse=True
        )
        
        if sorted_params:
            best_params = sorted_params[0]["params"]
            self.logger.info(f"Best overall parameters: {best_params}")
            return best_params
        else:
            self.logger.warning("No valid parameters found across folds")
            return {}
    
    def _run_ml_guided_optimization(
        self, 
        strategy_name: str,
        train_data: pd.DataFrame,
        test_data: pd.DataFrame,
        parameter_space: Dict[str, List[Any]],
        metric: str,
        max_evals: int,
        progress_callback: Optional[Callable] = None
    ) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
        """
        Run ML-guided optimization that learns from past evaluations to guide search.
        
        Args:
            strategy_name: Name of the strategy to optimize
            train_data: Training data for optimization
            test_data: Test data for validation
            parameter_space: Dictionary mapping parameter names to possible values
            metric: Primary metric to optimize
            max_evals: Maximum number of parameter combinations to evaluate
            progress_callback: Optional callback for progress updates
            
        Returns:
            Tuple containing (best_parameters, list_of_evaluations)
        """
        if not HAVE_SKLEARN:
            self.logger.warning("Scikit-learn not available. Falling back to grid search.")
            return self._run_grid_search(
                strategy_name, train_data, test_data, parameter_space, metric, max_evals, progress_callback
            )
        
        self.logger.info(f"Running ML-guided optimization for {strategy_name} with max evaluations: {max_evals}")
        
        # Store evaluations
        evaluations = []
        
        # Initial random exploration (30% of max_evals)
        initial_evals = max(5, int(max_evals * 0.3))
        random_params = []
        
        for _ in range(initial_evals):
            params = {}
            for param_name, param_values in parameter_space.items():
                if isinstance(param_values, list):
                    params[param_name] = np.random.choice(param_values)
                elif isinstance(param_values, dict) and 'min' in param_values and 'max' in param_values:
                    if param_values.get('type') == 'int':
                        params[param_name] = np.random.randint(
                            param_values['min'], param_values['max'] + 1
                        )
                    else:
                        params[param_name] = np.random.uniform(
                            param_values['min'], param_values['max']
                        )
            random_params.append(params)
        
        # Evaluate initial random parameters
        for i, params in enumerate(random_params):
            try:
                performance = self._evaluate_parameters(strategy_name, params, train_data)
                
                # Extract optimization metric
                if metric in performance:
                    score = performance[metric]
                    
                    # Record evaluation
                    eval_record = {
                        "params": params.copy(),
                        "metrics": performance,
                        "score": score
                    }
                    evaluations.append(eval_record)
            except Exception as e:
                self.logger.warning(f"Error evaluating parameters {params}: {str(e)}")
                continue
            
            # Update progress
            if progress_callback:
                progress = int((i + 1) * 100 / max_evals)
                progress_callback(min(progress, 30))
        
        # Track best parameters
        best_params = None
        best_score = float('-inf')
        
        if evaluations:
            # Find best parameters so far
            best_eval = max(evaluations, key=lambda x: x["score"])
            best_params = best_eval["params"]
            best_score = best_eval["score"]
        
        # Generate grid for prediction
        param_grid = list(ParameterGrid(parameter_space))
        
        # Limit grid size for prediction
        if len(param_grid) > 1000:
            np.random.shuffle(param_grid)
            param_grid = param_grid[:1000]
        
        # ML-guided search for remaining evaluations
        remaining_evals = max_evals - len(evaluations)
        
        for iteration in range(remaining_evals):
            if len(evaluations) < 5:
                # Not enough data to train a model, use random search
                params = {}
                for param_name, param_values in parameter_space.items():
                    if isinstance(param_values, list):
                        params[param_name] = np.random.choice(param_values)
                    elif isinstance(param_values, dict) and 'min' in param_values and 'max' in param_values:
                        if param_values.get('type') == 'int':
                            params[param_name] = np.random.randint(
                                param_values['min'], param_values['max'] + 1
                            )
                        else:
                            params[param_name] = np.random.uniform(
                                param_values['min'], param_values['max']
                            )
            else:
                # Train model to predict performance
                X, y = self._prepare_ml_training_data(evaluations, parameter_space)
                
                # Scale features
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(X)
                
                # Train model
                model = RandomForestRegressor(n_estimators=50, random_state=42)
                model.fit(X_scaled, y)
                
                # Generate predictions for parameter grid
                X_grid = self._prepare_ml_prediction_data(param_grid, parameter_space)
                X_grid_scaled = scaler.transform(X_grid)
                predictions = model.predict(X_grid_scaled)
                
                # Find best predicted parameters
                best_idx = np.argmax(predictions)
                params = param_grid[best_idx]
                
                # Remove from grid to avoid duplicates
                param_grid.pop(best_idx)
            
            # Evaluate selected parameters
            try:
                performance = self._evaluate_parameters(strategy_name, params, train_data)
                
                # Extract optimization metric
                if metric in performance:
                    score = performance[metric]
                    
                    # Record evaluation
                    eval_record = {
                        "params": params.copy(),
                        "metrics": performance,
                        "score": score
                    }
                    evaluations.append(eval_record)
                    
                    # Update best parameters if score is better
                    if score > best_score:
                        best_score = score
                        best_params = params.copy()
                        self.logger.debug(f"New best parameters found: {best_params} with score: {best_score}")
            
            except Exception as e:
                self.logger.warning(f"Error evaluating parameters {params}: {str(e)}")
                continue
            
            # Update progress
            if progress_callback:
                progress = 30 + int((iteration + 1) * 70 / remaining_evals)
                progress_callback(min(progress, 100))
        
        self.logger.info(f"ML-guided optimization completed with {len(evaluations)} evaluations")
        return best_params, evaluations
    
    def _prepare_ml_training_data(
        self, 
        evaluations: List[Dict[str, Any]], 
        parameter_space: Dict[str, List[Any]]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare training data for ML-guided optimization.
        
        Args:
            evaluations: List of evaluation results
            parameter_space: Dictionary mapping parameter names to possible values
            
        Returns:
            Tuple containing (X, y) arrays for training
        """
        # Extract features and targets
        X = []
        y = []
        
        for eval_data in evaluations:
            # Extract parameters
            params = eval_data["params"]
            features = []
            
            # Convert parameters to numerical features
            for param_name in sorted(parameter_space.keys()):
                if param_name in params:
                    value = params[param_name]
                    
                    # Convert categorical parameters to numerical
                    if isinstance(value, str) or isinstance(value, bool):
                        # For categorical values, create one-hot encoding
                        possible_values = parameter_space[param_name]
                        if isinstance(possible_values, list):
                            for pv in possible_values:
                                features.append(1.0 if value == pv else 0.0)
                        else:
                            # Just use as is if we don't have a list of possible values
                            features.append(float(value))
                    else:
                        # For numerical values, use as is
                        features.append(float(value))
                else:
                    # Missing parameter, use default or middle value
                    possible_values = parameter_space[param_name]
                    if isinstance(possible_values, list):
                        # Use middle of range for numerical, or zeros for categorical
                        if all(isinstance(x, (int, float)) for x in possible_values):
                            features.append(float(np.mean(possible_values)))
                        else:
                            features.extend([0.0] * len(possible_values))
                    elif isinstance(possible_values, dict) and 'min' in possible_values and 'max' in possible_values:
                        features.append(float(np.mean([possible_values['min'], possible_values['max']])))
            
            X.append(features)
            y.append(eval_data["score"])
        
        return np.array(X), np.array(y)
    
    def _prepare_ml_prediction_data(
        self, 
        param_grid: List[Dict[str, Any]], 
        parameter_space: Dict[str, List[Any]]
    ) -> np.ndarray:
        """
        Prepare prediction data for ML-guided optimization.
        
        Args:
            param_grid: List of parameter combinations
            parameter_space: Dictionary mapping parameter names to possible values
            
        Returns:
            X array for prediction
        """
        # Extract features
        X = []
        
        for params in param_grid:
            features = []
            
            # Convert parameters to numerical features
            for param_name in sorted(parameter_space.keys()):
                if param_name in params:
                    value = params[param_name]
                    
                    # Convert categorical parameters to numerical
                    if isinstance(value, str) or isinstance(value, bool):
                        # For categorical values, create one-hot encoding
                        possible_values = parameter_space[param_name]
                        if isinstance(possible_values, list):
                            for pv in possible_values:
                                features.append(1.0 if value == pv else 0.0)
                        else:
                            # Just use as is if we don't have a list of possible values
                            features.append(float(value))
                    else:
                        # For numerical values, use as is
                        features.append(float(value))
                else:
                    # Missing parameter, use default or middle value
                    possible_values = parameter_space[param_name]
                    if isinstance(possible_values, list):
                        # Use middle of range for numerical, or zeros for categorical
                        if all(isinstance(x, (int, float)) for x in possible_values):
                            features.append(float(np.mean(possible_values)))
                        else:
                            features.extend([0.0] * len(possible_values))
                    elif isinstance(possible_values, dict) and 'min' in possible_values and 'max' in possible_values:
                        features.append(float(np.mean([possible_values['min'], possible_values['max']])))
            
            X.append(features)
        
        return np.array(X)
    
    def _run_random_search(
        self, 
        strategy_name: str,
        train_data: pd.DataFrame,
        test_data: pd.DataFrame,
        parameter_space: Dict[str, List[Any]],
        metric: str,
        max_evals: int,
        progress_callback: Optional[Callable] = None
    ) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
        """
        Run random search optimization for strategy parameters.
        
        Args:
            strategy_name: Name of the strategy to optimize
            train_data: Training data for optimization
            test_data: Test data for validation
            parameter_space: Dictionary mapping parameter names to possible values
            metric: Primary metric to optimize
            max_evals: Maximum number of parameter combinations to evaluate
            progress_callback: Optional callback for progress updates
            
        Returns:
            Tuple containing (best_parameters, list_of_evaluations)
        """
        self.logger.info(f"Running random search for {strategy_name} with max evaluations: {max_evals}")
        
        # Track best parameters
        best_score = float('-inf')
        best_params = None
        evaluations = []
        
        # Evaluate random parameter combinations
        for i in range(max_evals):
            # Generate random parameters
            params = {}
            for param_name, param_values in parameter_space.items():
                if isinstance(param_values, list):
                    params[param_name] = np.random.choice(param_values)
                elif isinstance(param_values, dict) and 'min' in param_values and 'max' in param_values:
                    if param_values.get('type') == 'int':
                        params[param_name] = np.random.randint(
                            param_values['min'], param_values['max'] + 1
                        )
                    else:
                        params[param_name] = np.random.uniform(
                            param_values['min'], param_values['max']
                        )
            
            try:
                # Execute strategy with current parameters
                performance = self._evaluate_parameters(strategy_name, params, train_data)
                
                # Update progress
                if progress_callback:
                    progress = int((i + 1) * 100 / max_evals)
                    progress_callback(progress)
                
                # Extract optimization metric
                if metric in performance:
                    score = performance[metric]
                    
                    # Record evaluation
                    eval_record = {
                        "params": params.copy(),
                        "metrics": performance,
                        "score": score
                    }
                    evaluations.append(eval_record)
                    
                    # Update best parameters if score is better
                    if score > best_score:
                        best_score = score
                        best_params = params.copy()
                        self.logger.debug(f"New best parameters found: {best_params} with score: {best_score}")
            
            except Exception as e:
                self.logger.warning(f"Error evaluating parameters {params}: {str(e)}")
                continue
        
        self.logger.info(f"Random search completed with {len(evaluations)} evaluations")
        return best_params, evaluations
    
    def _evaluate_parameters(
        self, 
        strategy_name: str, 
        parameters: Dict[str, Any], 
        data: pd.DataFrame
    ) -> Dict[str, float]:
        """
        Evaluate a set of strategy parameters on historical data.
        
        Args:
            strategy_name: Name of the strategy to evaluate
            parameters: Strategy parameters to evaluate
            data: Historical market data
            
        Returns:
            Dictionary of performance metrics
        """
        if data.empty:
            raise ValueError("Empty data provided for parameter evaluation")
        
        # Get strategy function
        strategy_func = self.strategy_generator.strategies.get(strategy_name)
        if not strategy_func:
            raise ValueError(f"Strategy '{strategy_name}' not found")
        
        # Generate signals using strategy with specified parameters
        signals = strategy_func(data, parameters)
        
        # Calculate performance metrics
        metrics = self._calculate_performance_metrics(data, signals)
        
        return metrics
    
    def _calculate_performance_metrics(
        self, 
        data: pd.DataFrame, 
        signals: pd.DataFrame
    ) -> Dict[str, float]:
        """
        Calculate performance metrics for a strategy.
        
        Args:
            data: Historical market data
            signals: Strategy signals dataframe
            
        Returns:
            Dictionary of performance metrics
        """
        # Calculate returns based on signals
        if 'close' not in data.columns:
            raise ValueError("Price data missing 'close' column")
            
        if 'position' not in signals.columns:
            raise ValueError("Signals missing 'position' column")
        
        # Calculate price returns
        data['returns'] = data['close'].pct_change()
        
        # Calculate strategy returns (shifted signals to avoid lookahead bias)
        strategy_returns = signals['position'].shift(1) * data['returns']
        
        # Remove NaN values
        strategy_returns = strategy_returns.dropna()
        
        # Error handling for empty returns
        if len(strategy_returns) == 0:
            return {
                "sharpe_ratio": float('-inf'),
                "sortino_ratio": float('-inf'),
                "max_drawdown": 1.0,
                "profit_factor": 0.0,
                "win_rate": 0.0,
                "total_return": -1.0
            }
        
        # Calculate metrics
        metrics = {}
        
        # Total return
        metrics["total_return"] = (1 + strategy_returns).prod() - 1
        
        # Sharpe ratio (annualized)
        if strategy_returns.std() > 0:
            metrics["sharpe_ratio"] = strategy_returns.mean() / strategy_returns.std() * np.sqrt(252)
        else:
            metrics["sharpe_ratio"] = 0
        
        # Sortino ratio (annualized, using downside deviation)
        downside_returns = strategy_returns[strategy_returns < 0]
        if len(downside_returns) > 0 and downside_returns.std() > 0:
            metrics["sortino_ratio"] = strategy_returns.mean() / downside_returns.std() * np.sqrt(252)
        else:
            metrics["sortino_ratio"] = 0
        
        # Maximum drawdown
        cumulative_returns = (1 + strategy_returns).cumprod()
        peak = cumulative_returns.expanding().max()
        drawdown = (cumulative_returns / peak - 1)
        metrics["max_drawdown"] = drawdown.min() if len(drawdown) > 0 else 0
        
        # Profit factor
        positive_returns = strategy_returns[strategy_returns > 0].sum()
        negative_returns = abs(strategy_returns[strategy_returns < 0].sum())
        if negative_returns > 0:
            metrics["profit_factor"] = positive_returns / negative_returns
        else:
            metrics["profit_factor"] = 0 if positive_returns == 0 else float('inf')
        
        # Win rate
        trades = strategy_returns[signals['position'].shift(1) != 0]
        if len(trades) > 0:
            metrics["win_rate"] = len(trades[trades > 0]) / len(trades)
        else:
            metrics["win_rate"] = 0
        
        # Number of trades
        metrics["trade_count"] = len(trades)
        
        # Average trade return
        metrics["avg_trade"] = trades.mean() if len(trades) > 0 else 0
        
        return metrics
    
    def _validate_parameters(
        self, 
        strategy_name: str, 
        parameters: Dict[str, Any], 
        validation_data: pd.DataFrame
    ) -> Dict[str, float]:
        """
        Validate a set of strategy parameters on out-of-sample data.
        
        Args:
            strategy_name: Name of the strategy to validate
            parameters: Strategy parameters to validate
            validation_data: Out-of-sample validation data
            
        Returns:
            Dictionary of validation metrics
        """
        if validation_data.empty:
            self.logger.warning("Empty validation data provided")
            return {}
        
        try:
            # Evaluate parameters on validation data
            validation_metrics = self._evaluate_parameters(strategy_name, parameters, validation_data)
            
            return validation_metrics
        except Exception as e:
            self.logger.error(f"Error validating parameters: {str(e)}")
            return {}
    
    def _split_data_for_optimization(
        self, 
        data: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Split data into training and validation sets for optimization.
        
        Args:
            data: Historical market data
            
        Returns:
            Tuple containing (train_data, validation_data)
        """
        # Validate data
        if data.empty:
            raise ValueError("Empty data provided for splitting")
        
        # Determine split point (80% train, 20% validation)
        split_idx = int(len(data) * 0.8)
        
        # Split data
        train_data = data.iloc[:split_idx].copy()
        validation_data = data.iloc[split_idx:].copy()
        
        return train_data, validation_data
    
    def _calculate_parameter_importance(self, strategy_name: str, evaluations: List[Dict[str, Any]]) -> Dict[str, float]:
        """
        Calculate parameter importance based on evaluation results.
        
        Args:
            strategy_name: Name of the strategy
            evaluations: List of evaluation results
            
        Returns:
            Dictionary mapping parameter names to importance scores
        """
        if not evaluations:
            return {}
        
        try:
            # Extract parameters and scores
            params_list = []
            scores = []
            
            for eval_data in evaluations:
                params_list.append(eval_data["params"])
                scores.append(eval_data["score"])
            
            if not params_list:
                return {}
            
            # Convert to DataFrame
            params_df = pd.DataFrame(params_list)
            
            # Calculate importance
            importance = {}
            
            for column in params_df.columns:
                # Check if parameter is numeric
                if pd.api.types.is_numeric_dtype(params_df[column]):
                    # Calculate correlation with score
                    correlation = np.corrcoef(params_df[column], scores)[0, 1]
                    importance[column] = abs(correlation)
                else:
                    # For categorical parameters, compare variance of scores for each value
                    values = params_df[column].unique()
                    if len(values) > 1:
                        # Calculate score variance for each value
                        variances = []
                        for value in values:
                            value_scores = [scores[i] for i, params in enumerate(params_list) if params[column] == value]
                            if len(value_scores) > 1:
                                variances.append(np.var(value_scores))
                        
                        if variances:
                            # Higher variance means more importance
                            importance[column] = np.mean(variances) / np.mean(scores)
                        else:
                            importance[column] = 0
                    else:
                        importance[column] = 0
            
            # Normalize importance
            if importance:
                total = sum(importance.values())
                if total > 0:
                    importance = {k: v / total for k, v in importance.items()}
            
            # Store importance
            with self.optimization_lock:
                self.parameter_importance[strategy_name] = importance
            
            return importance
            
        except Exception as e:
            self.logger.error(f"Error calculating parameter importance: {str(e)}")
            return {}
    
    def _save_optimization_results(
        self, 
        optimization_id: str,
        strategy_name: str,
        best_params: Dict[str, Any],
        results: Dict[str, Any]
    ) -> None:
        """
        Save optimization results to disk.
        
        Args:
            optimization_id: Unique identifier for the optimization run
            strategy_name: Name of the strategy
            best_params: Best parameters found
            results: Optimization results
        """
        try:
            # Create output directory
            output_dir = os.path.join(self.data_dir, "results")
            os.makedirs(output_dir, exist_ok=True)
            
            # Create output file
            output_file = os.path.join(output_dir, f"{optimization_id}.json")
            
            # Prepare output data
            output_data = {
                "optimization_id": optimization_id,
                "strategy_name": strategy_name,
                "timestamp": datetime.now().isoformat(),
                "best_params": best_params,
                "results": results
            }
            
            # Save to file
            with open(output_file, "w") as f:
                json.dump(output_data, f, indent=4)
                
            self.logger.info(f"Optimization results saved to {output_file}")
            
        except Exception as e:
            self.logger.error(f"Error saving optimization results: {str(e)}")
    
    def load_optimization_results(self, optimization_id: str) -> Dict[str, Any]:
        """
        Load optimization results from disk.
        
        Args:
            optimization_id: Unique identifier for the optimization run
            
        Returns:
            Dictionary containing optimization results
        """
        try:
            # Find output file
            output_dir = os.path.join(self.data_dir, "results")
            output_file = os.path.join(output_dir, f"{optimization_id}.json")
            
            if not os.path.exists(output_file):
                self.logger.warning(f"Optimization results file not found: {output_file}")
                return {}
            
            # Load from file
            with open(output_file, "r") as f:
                output_data = json.load(f)
                
            return output_data
            
        except Exception as e:
            self.logger.error(f"Error loading optimization results: {str(e)}")
            return {}
    
    def get_optimization_status(self, optimization_id: str) -> Dict[str, Any]:
        """
        Get status of an optimization run.
        
        Args:
            optimization_id: Unique identifier for the optimization run
            
        Returns:
            Dictionary containing optimization status
        """
        with self.optimization_lock:
            if optimization_id in self.current_optimizations:
                return self.current_optimizations[optimization_id].copy()
        
        # Check if results exist
        results = self.load_optimization_results(optimization_id)
        if results:
            return {
                "optimization_id": optimization_id,
                "strategy_name": results.get("strategy_name", ""),
                "status": "completed",
                "results": results.get("results", {}),
                "best_params": results.get("best_params", {})
            }
        
        return {"optimization_id": optimization_id, "status": "not_found"}
    
    def get_optimization_history(self, strategy_name: Optional[str] = None) -> Dict[str, List[Dict[str, Any]]]:
        """
        Get history of optimization runs.
        
        Args:
            strategy_name: Optional strategy name to filter history
            
        Returns:
            Dictionary mapping strategy names to lists of optimization results
        """
        with self.optimization_lock:
            if strategy_name:
                if strategy_name in self.optimization_history:
                    return {strategy_name: self.optimization_history[strategy_name]}
                return {}
            else:
                return self.optimization_history.copy()
    
    def get_parameter_importance(self, strategy_name: Optional[str] = None) -> Dict[str, Dict[str, float]]:
        """
        Get parameter importance scores.
        
        Args:
            strategy_name: Optional strategy name to filter importance scores
            
        Returns:
            Dictionary mapping strategy names to parameter importance scores
        """
        with self.optimization_lock:
            if strategy_name:
                if strategy_name in self.parameter_importance:
                    return {strategy_name: self.parameter_importance[strategy_name]}
                return {}
            else:
                return self.parameter_importance.copy()
    
    def _validate_optimization_inputs(
        self, 
        strategy_name: str,
        historical_data: pd.DataFrame,
        parameter_space: Dict[str, List[Any]],
        metric: str
    ) -> None:
        """
        Validate inputs for optimization.
        
        Args:
            strategy_name: Name of the strategy to optimize
            historical_data: DataFrame containing historical market data
            parameter_space: Dictionary mapping parameter names to possible values
            metric: Primary metric to optimize
            
        Raises:
            ValueError: If inputs are invalid
        """
        # Check strategy name
        if not strategy_name:
            raise ValueError("Strategy name cannot be empty")
        
        # Check historical data
        if historical_data is None or historical_data.empty:
            raise ValueError("Historical data cannot be empty")
        
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        missing_columns = [col for col in required_columns if col not in historical_data.columns]
        if missing_columns:
            raise ValueError(f"Historical data missing required columns: {missing_columns}")
        
        # Check parameter space
        if not parameter_space:
            raise ValueError("Parameter space cannot be empty")
        
        # Check optimization metric
        valid_metrics = [
            "sharpe_ratio", "sortino_ratio", "max_drawdown", 
            "profit_factor", "win_rate", "total_return"
        ]
        if metric not in valid_metrics:
            raise ValueError(f"Invalid optimization metric: {metric}. Valid metrics: {valid_metrics}")
    
    def set_parameter_constraints(
        self, 
        strategy_name: str, 
        constraints: Dict[str, Dict[str, Any]]
    ) -> None:
        """
        Set constraints for strategy parameters.
        
        Args:
            strategy_name: Name of the strategy
            constraints: Dictionary mapping parameter names to constraint specifications
        """
        with self.optimization_lock:
            self.parameter_constraints[strategy_name] = constraints
            
        self.logger.info(f"Parameter constraints set for {strategy_name}: {constraints}")
    
    def clear_optimization_history(self, strategy_name: Optional[str] = None) -> None:
        """
        Clear optimization history.
        
        Args:
            strategy_name: Optional strategy name to clear history for
        """
        with self.optimization_lock:
            if strategy_name:
                if strategy_name in self.optimization_history:
                    self.optimization_history[strategy_name] = []
                if strategy_name in self.parameter_importance:
                    self.parameter_importance[strategy_name] = {}
            else:
                self.optimization_history = {}
                self.parameter_importance = {}
    
    def cancel_optimization(self, optimization_id: str) -> bool:
        """
        Cancel an ongoing optimization run.
        
        Args:
            optimization_id: Unique identifier for the optimization run
            
        Returns:
            True if optimization was cancelled, False otherwise
        """
        # Check if optimization is running
        with self.optimization_lock:
            if optimization_id in self.current_optimizations:
                if self.current_optimizations[optimization_id]["status"] in ["starting", "running"]:
                    self.current_optimizations[optimization_id]["status"] = "cancelled"
                    
                    # Cancel task in thread manager if available
                    if self.thread_manager and HAVE_THREAD_MANAGER:
                        self.thread_manager.cancel_task(optimization_id)
                    
                    self.logger.info(f"Optimization {optimization_id} cancelled")
                    return True
        
        return False
