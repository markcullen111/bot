# gui/widgets/ai_control_widget.py

from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, 
                           QComboBox, QTabWidget, QTextEdit, QProgressBar, QGroupBox, 
                           QFormLayout, QCheckBox, QDoubleSpinBox, QMessageBox)
from PyQt5.QtCore import Qt, pyqtSignal, QTimer

import logging
import os
import time
from datetime import datetime

class AIControlWidget(QWidget):
    """
    Widget for controlling and monitoring the autonomous AI system
    """
    
    ai_training_requested = pyqtSignal(dict)  # Signal for AI training request
    ai_toggle_requested = pyqtSignal(bool)    # Signal for enabling/disabling AI
    
    def __init__(self, trading_system, parent=None):
        super().__init__(parent)
        
        self.trading_system = trading_system
        self.thread_manager = getattr(trading_system, 'thread_manager', None)
        self.training_task_id = None
        
        # Initialize UI
        self._init_ui()
        
        # Update timer
        self.update_timer = QTimer(self)
        self.update_timer.timeout.connect(self.update_status)
        self.update_timer.start(5000)  # Update every 5 seconds
        
        # Initial status update
        self.update_status()
        
    def _init_ui(self):
        """Initialize UI components"""
        layout = QVBoxLayout(self)
        
        # AI Status and Control Section
        status_group = QGroupBox("AI System Status")
        status_layout = QFormLayout(status_group)
        
        # Status indicators
        self.status_label = QLabel("Inactive")
        self.status_label.setStyleSheet("color: gray; font-weight: bold;")
        status_layout.addRow("Status:", self.status_label)
        
        self.models_label = QLabel("0 / 7")
        status_layout.addRow("Active Models:", self.models_label)
        
        self.last_trained_label = QLabel("Never")
        status_layout.addRow("Last Trained:", self.last_trained_label)
        
        # Enable/Disable AI
        self.ai_toggle_btn = QPushButton("Enable AI System")
        self.ai_toggle_btn.setCheckable(True)
        self.ai_toggle_btn.toggled.connect(self._toggle_ai)
        
        # Create control buttons layout
        control_layout = QHBoxLayout()
        control_layout.addWidget(self.ai_toggle_btn)
        
        self.reset_ai_btn = QPushButton("Reset AI")
        self.reset_ai_btn.clicked.connect(self._reset_ai)
        control_layout.addWidget(self.reset_ai_btn)
        
        # Add to status layout
        status_layout.addRow("", control_layout)
        
        # Add status group to main layout
        layout.addWidget(status_group)
        
        # Training Section
        training_group = QGroupBox("AI Training")
        training_layout = QFormLayout(training_group)
        
        # Model selection
        self.model_combo = QComboBox()
        self.model_combo.addItems([
            "All Models", "Market Making", "Order Flow Prediction", 
            "Trade Timing", "Trade Exit", "Portfolio Allocation", "Risk Manager"
        ])
        training_layout.addRow("Model:", self.model_combo)
        
        # Training options
        options_layout = QHBoxLayout()
        
        self.use_gpu_cb = QCheckBox("Use GPU")
        self.use_gpu_cb.setChecked(True)
        options_layout.addWidget(self.use_gpu_cb)
        
        self.auto_optimize_cb = QCheckBox("Auto-optimize")
        self.auto_optimize_cb.setChecked(True)
        options_layout.addWidget(self.auto_optimize_cb)
        
        training_layout.addRow("Options:", options_layout)
        
        # Training progress
        self.training_progress = QProgressBar()
        training_layout.addRow("Progress:", self.training_progress)
        
        # Training buttons
        train_btn_layout = QHBoxLayout()
        
        self.start_training_btn = QPushButton("Start Training")
        self.start_training_btn.clicked.connect(self._start_training)
        train_btn_layout.addWidget(self.start_training_btn)
        
        self.cancel_training_btn = QPushButton("Cancel")
        self.cancel_training_btn.setEnabled(False)
        self.cancel_training_btn.clicked.connect(self._cancel_training)
        train_btn_layout.addWidget(self.cancel_training_btn)
        
        training_layout.addRow("", train_btn_layout)
        
        # Add training group to main layout
        layout.addWidget(training_group)
        
        # Performance metrics
        metrics_group = QGroupBox("AI Performance")
        metrics_layout = QFormLayout(metrics_group)
        
        self.win_rate_label = QLabel("0.0%")
        metrics_layout.addRow("Win Rate:", self.win_rate_label)
        
        self.accuracy_label = QLabel("0.0%")
        metrics_layout.addRow("Model Accuracy:", self.accuracy_label)
        
        self.profit_contribution_label = QLabel("0.0%")
        metrics_layout.addRow("Profit Contribution:", self.profit_contribution_label)
        
        layout.addWidget(metrics_group)
        
        # Status log
        self.status_text = QTextEdit()
        self.status_text.setReadOnly(True)
        self.status_text.setMaximumHeight(100)
        layout.addWidget(self.status_text)
        
        # Set layout
        self.setLayout(layout)
        
    def update_status(self):
        """Update AI status information"""
        try:
            # Check for AI components
            has_ai_components = self._check_ai_components()
            ai_enabled = getattr(self.trading_system, 'ai_enabled', False)
            
            # Update status indicators
            if ai_enabled:
                self.status_label.setText("Active")
                self.status_label.setStyleSheet("color: green; font-weight: bold;")
                self.ai_toggle_btn.setChecked(True)
                self.ai_toggle_btn.setText("Disable AI System")
            else:
                self.status_label.setText("Inactive")
                self.status_label.setStyleSheet("color: gray; font-weight: bold;")
                self.ai_toggle_btn.setChecked(False)
                self.ai_toggle_btn.setText("Enable AI System")
            
            # Count available models
            available_models = self._count_available_models()
            self.models_label.setText(f"{available_models} / 7")
            
            # Update last trained time
            last_trained = getattr(self.trading_system, 'last_ai_training', None)
            if last_trained:
                self.last_trained_label.setText(last_trained.strftime("%Y-%m-%d %H:%M"))
            
            # Update performance metrics
            self._update_performance_metrics()
            
            # Check ongoing training
            self._check_training_progress()
            
        except Exception as e:
            logging.error(f"Error updating AI status: {e}")
            self.log_message(f"Error updating status: {str(e)}", "error")
    
    def _check_ai_components(self):
        """Check if AI components are available"""
        ai_components = [
            hasattr(self.trading_system, 'ai_engine'),
            hasattr(self.trading_system, 'market_making_ai'),
            hasattr(self.trading_system, 'predictive_order_flow_ai'),
            hasattr(self.trading_system, 'portfolio_allocation_ai'),
            hasattr(self.trading_system, 'trade_timing_ai'),
            hasattr(self.trading_system, 'trade_exit_ai'),
            hasattr(self.trading_system, 'risk_manager') and 
            hasattr(self.trading_system.risk_manager, 'model')
        ]
        
        return any(ai_components)
    
    def _count_available_models(self):
        """Count available AI models"""
        count = 0
        model_dirs = ["models", "ml_models"]
        model_files = [
            "market_making_model.pth",
            "portfolio_allocation_model.pth", 
            "order_flow_predictor.pth",
            "trade_timing_model.pth",
            "trade_exit_model.pth",
            "trade_reentry_model.pth",
            "risk_model.pth"
        ]
        
        for model_file in model_files:
            # Check in possible model directories
            for model_dir in model_dirs:
                path = os.path.join(model_dir, model_file)
                if os.path.exists(path):
                    count += 1
                    break
        
        return count
    
    def _update_performance_metrics(self):
        """Update AI performance metrics"""
        try:
            # These would actually be retrieved from the trading system in a real implementation
            # For now, we use placeholder values
            
            # Check if we can get metrics from the trading system
            if hasattr(self.trading_system, 'get_ai_metrics'):
                metrics = self.trading_system.get_ai_metrics()
                
                # Update labels
                self.win_rate_label.setText(f"{metrics.get('win_rate', 0.0) * 100:.1f}%")
                self.accuracy_label.setText(f"{metrics.get('accuracy', 0.0) * 100:.1f}%")
                self.profit_contribution_label.setText(f"{metrics.get('profit_contribution', 0.0) * 100:.1f}%")
            else:
                # Use placeholder values
                win_rate = getattr(self.trading_system, 'ai_win_rate', 0.0)
                accuracy = getattr(self.trading_system, 'ai_accuracy', 0.0) 
                profit_contribution = getattr(self.trading_system, 'ai_profit_contribution', 0.0)
                
                # Update labels with placeholders
                self.win_rate_label.setText(f"{win_rate * 100:.1f}%")
                self.accuracy_label.setText(f"{accuracy * 100:.1f}%")
                self.profit_contribution_label.setText(f"{profit_contribution * 100:.1f}%")
        
        except Exception as e:
            logging.error(f"Error updating AI metrics: {e}")
    
    def _toggle_ai(self, enabled):
        """Enable or disable AI controller"""
        try:
            if enabled:
                # Confirm enabling
                reply = QMessageBox.question(
                    self, "Enable AI Controller", 
                    "Are you sure you want to enable the autonomous AI controller?",
                    QMessageBox.Yes | QMessageBox.No,
                    QMessageBox.No
                )
                
                if reply != QMessageBox.Yes:
                    self.ai_toggle_btn.setChecked(False)
                    return
                
                # Try to enable AI
                success = True
                if hasattr(self.trading_system, 'enable_ai'):
                    success = self.trading_system.enable_ai()
                else:
                    # Fallback if method not available
                    setattr(self.trading_system, 'ai_enabled', True)
                
                if success:
                    self.log_message("AI system enabled")
                else:
                    self.log_message("Failed to enable AI system", "error")
                    self.ai_toggle_btn.setChecked(False)
            else:
                # Disable AI
                if hasattr(self.trading_system, 'disable_ai'):
                    self.trading_system.disable_ai()
                else:
                    # Fallback if method not available
                    setattr(self.trading_system, 'ai_enabled', False)
                
                self.log_message("AI system disabled")
            
            # Emit signal
            self.ai_toggle_requested.emit(enabled)
            
            # Update status
            self.update_status()
                
        except Exception as e:
            logging.error(f"Error toggling AI: {e}")
            self.log_message(f"Error toggling AI: {str(e)}", "error")
            self.ai_toggle_btn.setChecked(False)
    
    def _reset_ai(self):
        """Reset AI models to initial state"""
        try:
            # Confirm reset
            reply = QMessageBox.question(
                self, "Reset AI System", 
                "Are you sure you want to reset the AI system? This will clear learned patterns.",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No
            )
            
            if reply != QMessageBox.Yes:
                return
            
            # Call reset function if available
            if hasattr(self.trading_system, 'reset_ai'):
                success = self.trading_system.reset_ai()
                if success:
                    self.log_message("AI system reset successfully")
                else:
                    self.log_message("Failed to reset AI system", "error")
            else:
                self.log_message("Reset function not available", "warning")
            
            # Update status
            self.update_status()
                
        except Exception as e:
            logging.error(f"Error resetting AI: {e}")
            self.log_message(f"Error resetting AI: {str(e)}", "error")
    
    def _start_training(self):
        """Start AI model training"""
        try:
            # Get training parameters
            model = self.model_combo.currentText()
            use_gpu = self.use_gpu_cb.isChecked()
            auto_optimize = self.auto_optimize_cb.isChecked()
            
            # Check if thread manager is available
            if not self.thread_manager:
                QMessageBox.warning(self, "Training Error", "Thread manager not available")
                return
            
            # Confirm training
            reply = QMessageBox.question(
                self, "Start Training", 
                f"Start training {model}?\nThis may take several minutes.",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No
            )
            
            if reply != QMessageBox.Yes:
                return
            
            # Prepare training function
            def training_task(model_name, use_gpu, auto_optimize, progress_callback=None):
                """Background task for model training"""
                try:
                    # Simulated training steps
                    total_steps = 10
                    for step in range(total_steps):
                        # Report progress
                        if progress_callback:
                            progress = (step + 1) * 100 // total_steps
                            progress_callback(progress)
                        
                        # Simulate training step
                        time.sleep(0.5)
                    
                    # In a real implementation, this would call the appropriate training function
                    # based on the selected model
                    
                    # Record training time
                    setattr(self.trading_system, 'last_ai_training', datetime.now())
                    
                    return {"status": "success", "model": model_name}
                except Exception as e:
                    logging.error(f"Training error: {e}")
                    return {"status": "error", "error": str(e)}
            
            # Submit training task
            self.training_task_id = self.thread_manager.submit_task(
                task_id=f"train_{model.lower().replace(' ', '_')}",
                func=training_task,
                args=(model, use_gpu, auto_optimize),
                priority=2  # Medium priority
            )
            
            # Update UI
            self.start_training_btn.setEnabled(False)
            self.cancel_training_btn.setEnabled(True)
            self.training_progress.setValue(0)
            
            self.log_message(f"Started training {model}")
                
        except Exception as e:
            logging.error(f"Error starting training: {e}")
            self.log_message(f"Error starting training: {str(e)}", "error")
    
    def _cancel_training(self):
        """Cancel ongoing training"""
        try:
            if not self.training_task_id or not self.thread_manager:
                return
            
            # Try to cancel task
            cancelled = self.thread_manager.cancel_task(self.training_task_id)
            
            if cancelled:
                self.log_message("Training cancelled")
            else:
                self.log_message("Could not cancel training", "warning")
            
            # Reset UI
            self.start_training_btn.setEnabled(True)
            self.cancel_training_btn.setEnabled(False)
            self.training_progress.setValue(0)
            
            # Clear task ID
            self.training_task_id = None
                
        except Exception as e:
            logging.error(f"Error cancelling training: {e}")
            self.log_message(f"Error cancelling training: {str(e)}", "error")
    
    def _check_training_progress(self):
        """Check progress of ongoing training task"""
        if not self.training_task_id or not self.thread_manager:
            return
        
        # Get task status
        status = self.thread_manager.get_task_status(self.training_task_id)
        progress = self.thread_manager.get_task_progress(self.training_task_id)
        
        if status == "completed":
            self.log_message("Training completed successfully")
            self.start_training_btn.setEnabled(True)
            self.cancel_training_btn.setEnabled(False)
            self.training_progress.setValue(100)
            self.training_task_id = None
            
            # Update status after training
            self.update_status()
            
        elif status == "failed":
            self.log_message("Training failed", "error")
            self.start_training_btn.setEnabled(True)
            self.cancel_training_btn.setEnabled(False)
            self.training_task_id = None
            
        elif status == "running" and progress is not None:
            self.training_progress.setValue(progress)
    
    def log_message(self, message, level="info"):
        """Add message to status log with timestamp"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        
        # Format based on message level
        if level == "error":
            formatted = f"<span style='color:red'>[{timestamp}] ERROR: {message}</span>"
        elif level == "warning":
            formatted = f"<span style='color:orange'>[{timestamp}] WARNING: {message}</span>"
        else:
            formatted = f"[{timestamp}] {message}"
        
        # Add to log
        self.status_text.append(formatted)
        
        # Scroll to bottom
        scroll_bar = self.status_text.verticalScrollBar()
        scroll_bar.setValue(scroll_bar.maximum())
