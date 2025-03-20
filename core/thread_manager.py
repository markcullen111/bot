# thread_manager.py

import concurrent.futures
import threading
import queue
import logging
import time
import uuid
import traceback
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from PyQt5.QtCore import QObject, pyqtSignal, QThread, QTimer

# Import error handling
try:
    from error_handling import safe_execute, ErrorCategory, ErrorSeverity
    HAVE_ERROR_HANDLING = True
except ImportError:
    HAVE_ERROR_HANDLING = False
    logging.warning("Error handling module not available. Using basic error handling.")

class ThreadTask:
    """
    Represents a task to be executed by a worker thread with enhanced tracking.
    
    Attributes:
        task_id (str): Unique identifier for the task
        func (callable): Function to execute
        args (tuple): Positional arguments for the function
        kwargs (dict): Keyword arguments for the function
        priority (int): Priority level (lower number = higher priority)
        result (Any): Result of task execution
        error (Exception): Error if execution failed
        status (str): Current status of the task
        start_time (float): Timestamp when task started
        end_time (float): Timestamp when task completed
        timeout (float): Maximum execution time in seconds
        progress (int): Progress percentage (0-100)
    """
    
    def __init__(
        self, 
        task_id: str, 
        func: Callable, 
        args: Optional[Tuple] = None, 
        kwargs: Optional[Dict[str, Any]] = None, 
        priority: int = 0,
        timeout: Optional[float] = None
    ):
        self.task_id = task_id
        self.func = func
        self.args = args or ()
        self.kwargs = kwargs or {}
        self.priority = priority  # Lower value means higher priority
        self.timeout = timeout
        
        # Task state
        self.result = None
        self.error = None
        self.status = "pending"
        self.start_time = None
        self.end_time = None
        self.progress = 0
        self._lock = threading.RLock()  # Reentrant lock for thread safety
        
    def __lt__(self, other) -> bool:
        """Comparison for priority queue."""
        return self.priority < other.priority
        
    def update_status(self, status: str) -> None:
        """Thread-safe status update."""
        with self._lock:
            self.status = status
            
    def update_progress(self, progress: int) -> None:
        """Thread-safe progress update."""
        with self._lock:
            self.progress = max(0, min(100, progress))
            
    def get_execution_time(self) -> Optional[float]:
        """Get task execution time if completed."""
        with self._lock:
            if self.start_time and self.end_time:
                return self.end_time - self.start_time
            elif self.start_time:
                return time.time() - self.start_time
            return None
            
    def is_expired(self) -> bool:
        """Check if task has exceeded its timeout."""
        if not self.timeout or not self.start_time:
            return False
            
        with self._lock:
            current_time = time.time()
            return (current_time - self.start_time) > self.timeout

class TaskResult:
    """
    Result of an executed task with comprehensive metadata.
    
    Attributes:
        task_id (str): Identifier of the completed task
        success (bool): Whether task completed successfully
        result (Any): Return value from the task function
        error (Exception): Exception if task failed
        execution_time (float): Time taken to execute the task
        status (str): Final status of the task
        progress (int): Final progress value (0-100)
    """
    
    def __init__(
        self, 
        task_id: str, 
        success: bool, 
        result: Any = None, 
        error: Optional[Exception] = None, 
        execution_time: Optional[float] = None,
        status: str = "completed",
        progress: int = 100
    ):
        self.task_id = task_id
        self.success = success
        self.result = result
        self.error = error
        self.execution_time = execution_time
        self.status = status
        self.progress = progress

class Worker(QThread):
    """
    Worker thread that processes tasks from a queue with comprehensive error handling.
    
    Signals:
        task_complete (TaskResult): Emitted when a task completes
        task_progress (str, int): Emitted to report task progress
    """
    
    task_complete = pyqtSignal(object)  # Signal emitted when task is complete
    task_progress = pyqtSignal(str, int)  # Signal emitted when task reports progress
    
    def __init__(self, task_queue, result_queue, worker_id=None):
        super().__init__()
        self.task_queue = task_queue
        self.result_queue = result_queue
        self.worker_id = worker_id or str(uuid.uuid4())
        self.running = True
        self.current_task = None
        self._current_task_lock = threading.RLock()
        self._exception_log = []  # Track exceptions for diagnostics
        
    def run(self):
        """Run the worker thread with enhanced error handling."""
        logging.info(f"Worker {self.worker_id} started")
        
        while self.running:
            try:
                # Get task from queue with timeout
                try:
                    task = self.task_queue.get(timeout=0.5)
                except queue.Empty:
                    continue  # No tasks in queue, just continue
                
                # Check if task should be skipped (cancelled)
                if task.status == "cancelled":
                    self.task_queue.task_done()
                    continue
                
                # Set current task (thread-safe)
                with self._current_task_lock:
                    self.current_task = task
                
                # Update task status
                task.update_status("running")
                task.start_time = time.time()
                
                try:
                    # Create progress callback for the task
                    def progress_callback(progress):
                        task.update_progress(progress)
                        self.task_progress.emit(task.task_id, progress)
                    
                    # Add progress callback to kwargs if function accepts it
                    if "progress_callback" in task.kwargs:
                        task.kwargs["progress_callback"] = progress_callback
                    
                    # Execute task with timeout monitoring in separate thread
                    result = self._execute_with_timeout(task)
                    
                    task.result = result
                    task.update_status("completed")
                    success = True
                    
                except TimeoutError:
                    task.error = "Task execution timed out"
                    task.update_status("timeout")
                    logging.warning(f"Task {task.task_id} timed out after {task.timeout} seconds")
                    success = False
                    
                except Exception as e:
                    task.error = str(e)
                    task.update_status("failed")
                    self._exception_log.append((time.time(), str(e), traceback.format_exc()))
                    logging.error(f"Task {task.task_id} failed: {e}\n{traceback.format_exc()}")
                    success = False
                
                # Record completion time
                task.end_time = time.time()
                execution_time = task.get_execution_time()
                
                # Create result object
                task_result = TaskResult(
                    task.task_id,
                    success,
                    task.result,
                    task.error,
                    execution_time,
                    task.status,
                    task.progress
                )
                
                # Put result in result queue
                self.result_queue.put(task_result)
                
                # Emit signal
                self.task_complete.emit(task_result)
                
                # Mark task as done
                self.task_queue.task_done()
                
                # Clear current task
                with self._current_task_lock:
                    self.current_task = None
                
            except Exception as e:
                # Critical error in worker itself
                logging.error(f"Critical worker error: {e}\n{traceback.format_exc()}")
                self._exception_log.append((time.time(), f"Worker error: {str(e)}", traceback.format_exc()))
                
                # Sleep briefly to prevent CPU spiking if there's a persistent error
                time.sleep(0.1)
        
        logging.info(f"Worker {self.worker_id} stopped")
    
    def _execute_with_timeout(self, task):
        """Execute task with timeout handling using a separate execution thread."""
        if not task.timeout:
            # No timeout specified, execute directly
            return task.func(*task.args, **task.kwargs)
        
        # Use a result container for thread communication
        result_container = {"result": None, "exception": None, "completed": False}
        execution_lock = threading.Lock()
        
        def execute_task():
            try:
                result = task.func(*task.args, **task.kwargs)
                with execution_lock:
                    if not result_container["completed"]:
                        result_container["result"] = result
                        result_container["completed"] = True
            except Exception as e:
                with execution_lock:
                    if not result_container["completed"]:
                        result_container["exception"] = e
                        result_container["completed"] = True
        
        # Start execution in a separate thread
        execution_thread = threading.Thread(target=execute_task)
        execution_thread.daemon = True
        execution_thread.start()
        
        # Wait for completion or timeout
        execution_thread.join(timeout=task.timeout)
        
        with execution_lock:
            if result_container["completed"]:
                if result_container["exception"]:
                    raise result_container["exception"]
                return result_container["result"]
            else:
                result_container["completed"] = True
                raise TimeoutError(f"Task execution timed out after {task.timeout} seconds")
    
    def stop(self):
        """Stop the worker safely."""
        self.running = False
        self.wait()
        
    def get_current_task(self):
        """Get current task info safely."""
        with self._current_task_lock:
            return self.current_task
            
    def get_exception_log(self):
        """Get the exception log for diagnostics."""
        return self._exception_log.copy()

class ThreadManager(QObject):
    """
    Manager for background threads and tasks with comprehensive error handling.
    
    Signals:
        task_started (str): Signal emitted when task starts
        task_completed (object): Signal emitted when task completes
        task_progress (str, int): Signal emitted to update progress
        worker_error (str): Signal emitted when a worker encounters an error
    """
    
    task_started = pyqtSignal(str)  # Signal emitted when task starts
    task_completed = pyqtSignal(object)  # Signal emitted when task completes
    task_progress = pyqtSignal(str, int)  # Signal emitted to update progress
    worker_error = pyqtSignal(str)  # Signal emitted when worker encounters error
    
    def __init__(self, max_workers=4, parent=None):
        super().__init__(parent)
        
        # Task queues and tracking
        self.task_queue = queue.PriorityQueue()
        self.result_queue = queue.Queue()
        self.task_map = {}  # Maps task_id to task
        self.task_map_lock = threading.RLock()  # Thread-safe access to task map
        
        # Initialize workers
        self.max_workers = max_workers
        self.workers = []
        self.workers_lock = threading.RLock()  # Thread-safe access to workers list
        self._initialize_workers()
        
        # Result processing timer
        self.result_timer = QTimer(self)
        self.result_timer.timeout.connect(self._process_results)
        self.result_timer.start(100)  # Check every 100ms
        
        # Health check timer
        self.health_timer = QTimer(self)
        self.health_timer.timeout.connect(self._check_worker_health)
        self.health_timer.start(10000)  # Check every 10 seconds
        
        # Statistics
        self.stats = {
            "tasks_submitted": 0,
            "tasks_completed": 0,
            "tasks_failed": 0,
            "tasks_cancelled": 0
        }
        self.stats_lock = threading.RLock()
        
        logging.info(f"Thread manager initialized with {max_workers} workers")
        
    def _initialize_workers(self):
        """Initialize worker threads with thread safety."""
        with self.workers_lock:
            for i in range(self.max_workers):
                worker = Worker(self.task_queue, self.result_queue, worker_id=i)
                worker.task_complete.connect(self._on_task_complete)
                worker.task_progress.connect(self._on_task_progress)
                worker.start()
                self.workers.append(worker)
                
    def _check_worker_health(self):
        """Periodically check worker health and restart any dead workers."""
        with self.workers_lock:
            for i, worker in enumerate(self.workers):
                if not worker.isRunning():
                    logging.warning(f"Worker {worker.worker_id} is not running, restarting")
                    
                    # Stop the old worker completely
                    try:
                        worker.stop()
                    except Exception as e:
                        logging.error(f"Error stopping worker: {e}")
                    
                    # Create and start a new worker
                    new_worker = Worker(self.task_queue, self.result_queue, worker_id=worker.worker_id)
                    new_worker.task_complete.connect(self._on_task_complete)
                    new_worker.task_progress.connect(self._on_task_progress)
                    new_worker.start()
                    
                    # Replace in the workers list
                    self.workers[i] = new_worker
                    
                    # Emit error signal
                    self.worker_error.emit(f"Worker {worker.worker_id} was restarted")
                    
            # Check if we need to add more workers (in case some were removed)
            while len(self.workers) < self.max_workers:
                worker_id = len(self.workers)
                worker = Worker(self.task_queue, self.result_queue, worker_id=worker_id)
                worker.task_complete.connect(self._on_task_complete)
                worker.task_progress.connect(self._on_task_progress)
                worker.start()
                self.workers.append(worker)
                logging.info(f"Added new worker {worker_id}")
            
    def _on_task_complete(self, task_result):
        """Handle task completion from worker signal."""
        # This method is connected to the worker's signal
        # and will be called in the worker's thread
        self.task_completed.emit(task_result)
        
        # Update statistics
        with self.stats_lock:
            self.stats["tasks_completed"] += 1
            if not task_result.success:
                self.stats["tasks_failed"] += 1
        
    def _on_task_progress(self, task_id, progress):
        """Handle task progress updates from worker."""
        self.task_progress.emit(task_id, progress)
        
    def _process_results(self):
        """Process completed tasks from result queue with thread safety."""
        # This method is called by a timer in the main thread
        try:
            while not self.result_queue.empty():
                task_result = self.result_queue.get_nowait()
                
                # Remove from task map
                with self.task_map_lock:
                    if task_result.task_id in self.task_map:
                        del self.task_map[task_result.task_id]
                    
                self.result_queue.task_done()
                
        except queue.Empty:
            pass
            
    def submit_task(
        self, 
        task_id: str, 
        func: Callable, 
        args: Optional[Tuple] = None, 
        kwargs: Optional[Dict[str, Any]] = None, 
        priority: int = 0,
        timeout: Optional[float] = None
    ) -> str:
        """
        Submit a task to be executed by a worker thread with enhanced error checking.
        
        Args:
            task_id: Unique identifier for the task
            func: Function to execute
            args: Positional arguments for the function
            kwargs: Keyword arguments for the function
            priority: Priority level (lower number = higher priority)
            timeout: Maximum execution time in seconds
            
        Returns:
            task_id: The ID of the submitted task
        """
        # Generate ID if not provided
        if not task_id:
            task_id = str(uuid.uuid4())
        
        with self.task_map_lock:
            if task_id in self.task_map:
                logging.warning(f"Task {task_id} already exists, replacing")
                
            # Create task
            task = ThreadTask(task_id, func, args, kwargs, priority, timeout)
            
            # Add to tracking map
            self.task_map[task_id] = task
            
            # Add to queue
            self.task_queue.put(task)
            
            # Update statistics
            with self.stats_lock:
                self.stats["tasks_submitted"] += 1
        
        # Emit signal
        self.task_started.emit(task_id)
        
        return task_id
        
    def update_progress(self, task_id: str, progress: int) -> bool:
        """
        Update progress of a task.
        
        Args:
            task_id: ID of the task
            progress: Progress percentage (0-100)
            
        Returns:
            bool: True if progress was updated, False otherwise
        """
        if not 0 <= progress <= 100:
            return False
            
        with self.task_map_lock:
            if task_id in self.task_map:
                task = self.task_map[task_id]
                task.update_progress(progress)
                self.task_progress.emit(task_id, progress)
                return True
                
        return False
        
    def cancel_task(self, task_id: str) -> bool:
        """
        Cancel a task if possible.
        
        Args:
            task_id: ID of the task to cancel
            
        Returns:
            bool: True if task was cancelled, False otherwise
        """
        with self.task_map_lock:
            if task_id in self.task_map:
                task = self.task_map[task_id]
                
                # Can only cancel if not already running
                if task.status == "pending":
                    # Remove from queue (not directly possible with PriorityQueue)
                    # So we mark it as canceled and ignore it when it comes up
                    task.update_status("cancelled")
                    
                    # Update statistics
                    with self.stats_lock:
                        self.stats["tasks_cancelled"] += 1
                        
                    return True
                    
        return False
        
    def cancel_all_tasks(self) -> int:
        """
        Cancel all pending tasks.
        
        Returns:
            int: Number of tasks cancelled
        """
        cancelled_count = 0
        
        with self.task_map_lock:
            for task_id, task in list(self.task_map.items()):
                if task.status == "pending":
                    task.update_status("cancelled")
                    cancelled_count += 1
        
        # Update statistics
        with self.stats_lock:
            self.stats["tasks_cancelled"] += cancelled_count
            
        return cancelled_count
        
    def get_task_status(self, task_id: str) -> Optional[str]:
        """
        Get status of a task.
        
        Args:
            task_id: ID of the task
            
        Returns:
            str: Task status or None if task not found
        """
        with self.task_map_lock:
            if task_id in self.task_map:
                return self.task_map[task_id].status
        return None
        
    def get_task_progress(self, task_id: str) -> Optional[int]:
        """
        Get progress of a task.
        
        Args:
            task_id: ID of the task
            
        Returns:
            int: Task progress (0-100) or None if task not found
        """
        with self.task_map_lock:
            if task_id in self.task_map:
                return self.task_map[task_id].progress
        return None
        
    def get_active_tasks(self) -> Dict[str, Dict[str, Any]]:
        """
        Get list of active tasks with detailed information.
        
        Returns:
            dict: Dictionary mapping task_id to task info
        """
        active_tasks = {}
        
        with self.task_map_lock:
            for task_id, task in self.task_map.items():
                if task.status in ["pending", "running"]:
                    active_tasks[task_id] = {
                        "status": task.status,
                        "progress": task.progress,
                        "priority": task.priority,
                        "execution_time": task.get_execution_time(),
                        "has_timeout": task.timeout is not None,
                        "is_expired": task.is_expired()
                    }
                    
        return active_tasks
        
    def get_all_tasks(self) -> Dict[str, Dict[str, Any]]:
        """
        Get information about all tasks.
        
        Returns:
            dict: Dictionary mapping task_id to task info
        """
        all_tasks = {}
        
        with self.task_map_lock:
            for task_id, task in self.task_map.items():
                all_tasks[task_id] = {
                    "status": task.status,
                    "progress": task.progress,
                    "priority": task.priority,
                    "execution_time": task.get_execution_time(),
                    "has_timeout": task.timeout is not None,
                    "is_expired": task.is_expired()
                }
                    
        return all_tasks
        
    def clear_completed_tasks(self) -> int:
        """
        Clear completed tasks from internal tracking.
        
        Returns:
            int: Number of tasks cleared
        """
        to_remove = []
        
        with self.task_map_lock:
            for task_id, task in list(self.task_map.items()):
                if task.status in ["completed", "failed", "cancelled", "timeout"]:
                    to_remove.append(task_id)
                    
            for task_id in to_remove:
                del self.task_map[task_id]
                
        return len(to_remove)
        
    def get_stats(self) -> Dict[str, int]:
        """
        Get task processing statistics.
        
        Returns:
            dict: Dictionary with task statistics
        """
        with self.stats_lock:
            return self.stats.copy()
            
    def reset_stats(self) -> None:
        """Reset task statistics."""
        with self.stats_lock:
            for key in self.stats:
                self.stats[key] = 0
        
    def shutdown(self) -> None:
        """Shutdown the thread manager safely."""
        logging.info("Shutting down thread manager...")
        
        # Stop timers
        self.result_timer.stop()
        self.health_timer.stop()
        
        # Cancel all pending tasks
        self.cancel_all_tasks()
        
        # Wait for tasks to be marked as cancelled
        time.sleep(0.1)
        
        # Stop all workers
        with self.workers_lock:
            for worker in self.workers:
                try:
                    worker.stop()
                except Exception as e:
                    logging.error(f"Error stopping worker {worker.worker_id}: {e}")
            
            # Clear workers list
            self.workers.clear()
            
        # Clear all queues
        while not self.task_queue.empty():
            try:
                self.task_queue.get_nowait()
                self.task_queue.task_done()
            except queue.Empty:
                break
                
        while not self.result_queue.empty():
            try:
                self.result_queue.get_nowait()
                self.result_queue.task_done()
            except queue.Empty:
                break
                
        # Clear task map
        with self.task_map_lock:
            self.task_map.clear()
            
        logging.info("Thread manager shut down successfully")

# Decorate thread submission functions with error handling if available
if HAVE_ERROR_HANDLING:
    # Example functions decorated with error handling
    @safe_execute(ErrorCategory.DATA_PROCESSING, default_return=None)
    def process_market_data_safe(thread_manager, symbol, timeframe='1h', limit=1000):
        """Process market data in a background thread with error handling."""
        return process_market_data(thread_manager, symbol, timeframe, limit)
