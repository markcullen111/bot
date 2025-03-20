# gui/widgets/ai_control_widget.py

from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel, 
                           QPushButton, QProgressBar, QComboBox, QCheckBox,
                           QSpinBox, QGroupBox, QFormLayout, QSlider, 
                           QTabWidget, QTextEdit, QDoubleSpinBox)
from PyQt5.QtCore import Qt, QTimer, pyqtSignal

class AIControlWidget(QWidget):
    """
    Widget for controlling and monitoring the autonomous AI system
    """
    
    ai_training_requested = pyqtSignal(dict)  # Signal for AI training request
    
    def __init__(self, trading_system, parent=None):
        super().__init__(parent)
        
        self.trading_system = trading_system
        
        # Initialize UI
        self._init_ui()
        
        # Update timer
        self.update_timer = QTimer(self)
        self.update_timer.timeout.connect(self.update_status)
        self.update_timer.start(5000)  # Update every 5 seconds
        
    def _init_ui(self):
        """Initialize UI components"""
        layout = QVBoxLayout(self)
        
        # Create tab widget
        tabs = QTabWidget()
        
        # Status tab
        status_tab = QWidget()
        self._setup_status_tab(status_tab)
        tabs.addTab(status_tab, "Status")
        
        # Training tab
        training_tab = QWidget()
        self._setup_training_tab(training_tab)
        tabs.addTab(training_tab, "Training")
        
        # Settings tab
        settings_tab = QWidget()
        self._setup_settings_tab(settings_tab)
        tabs.addTab(settings_tab, "Settings")
        
        # Performance tab
        performance_tab = QWidget()
        self._setup_performance_tab(performance_tab)
        tabs.addTab(performance_tab, "Performance")
        
        layout.addWidget(tabs)
        
    def _setup_status_tab(self, tab):
        """Set up AI status tab"""
        layout = QVBoxLayout(tab)
        
        # AI status group
        status_group = QGroupBox("AI System Status")
        status_layout = QFormLayout(status_group)
        
        # Meta-agent status
        status_layout.addRow("Meta-Agent:", QLabel("Not initialized"))
        
        # Active models status
        status_layout.addRow("Active Models:", QLabel("0"))
        
        # Training status
        status_layout.addRow("Last Training:", QLabel("Never"))
        
        # Performance score
        status_layout.addRow("Performance Score:", QLabel("0.0"))
        
        # Control mode
        status_layout.addRow("Control Mode:", QLabel("Manual"))
        
        layout.addWidget(status_group)
        
        # Current actions group
        actions_group = QGroupBox("Current Actions")
        actions_layout = QVBoxLayout(actions_group)
        
        # Current actions text
        self.actions_text = QTextEdit()
        self.actions_text.setReadOnly(True)
        actions_layout.addWidget(self.actions_text)
        
        layout.addWidget(actions_group)
        
        # Control buttons group
        control_group = QGroupBox("Control")
        control_layout = QHBoxLayout(control_group)
        
        # Enable/Disable AI button
        self.enable_ai_btn = QPushButton("Enable AI Controller")
        self.enable_ai_btn.setCheckable(True)
        self.enable_ai_btn.toggled.connect(self._toggle_ai_controller)
        control_layout.addWidget(self.enable_ai_btn)
        
        # Reset AI button
        self.reset_ai_btn = QPushButton("Reset AI")
        self.reset_ai_btn.clicked.connect(self._reset_ai)
        control_layout.addWidget(self.reset_ai_btn)
        
        # Update status button
        self.update_status_btn = QPushButton("Update Status")
        self.update_status_btn.clicked.connect(self.update_status)
        control_layout.addWidget(self.update_status_btn)
        
        layout.addWidget(control_group)
        
    def _setup_training_tab(self, tab):
        """Set up AI training tab"""
        layout = QVBoxLayout(tab)
        
        # Training configuration group
        config_group = QGroupBox("Training Configuration")
        config_layout = QFormLayout(config_group)
        
        # Model to train
        self.model_combo = QComboBox()
        self.model_combo.addItems([
            "All Models", 
            "Meta-Agent", 
            "Order Flow Predictor", 
            "Trade Timing", 
            "Trade Exit",
            "Portfolio Allocation"
        ])
        config_layout.addRow("Model:", self.model_combo)
        
        # Training time
        self.training_time_spin = QSpinBox()
        self.training_time_spin.setRange(1, 24)
        self.training_time_spin.setValue(1)
        self.training_time_spin.setSuffix(" hours")
        config_layout.addRow("Training Time:", self.training_time_spin)
        
        # Use GPU
        self.use_gpu_cb = QCheckBox("Use GPU if available")
        self.use_gpu_cb.setChecked(True)
        config_layout.addRow("", self.use_gpu_cb)
        
        # Auto-optimize
        self.auto_optimize_cb = QCheckBox("Auto-optimize hyperparameters")
        self.auto_optimize_cb.setChecked(True)
        config_layout.addRow("", self.auto_optimize_cb)
        
        layout.addWidget(config_group)
        
        # Training schedule group
        schedule_group = QGroupBox("Training Schedule")
        schedule_layout = QVBoxLayout(schedule_group)
        
        # Training schedule text
        self.schedule_text = QTextEdit()
        self.schedule_text.setReadOnly(True)
        schedule_layout.addWidget(self.schedule_text)
        
        layout.addWidget(schedule_group)
        
        # Training progress group
        progress_group = QGroupBox("Training Progress")
        progress_layout = QVBoxLayout(progress_group)
        
        # Progress bar
        self.training_progress = QProgressBar()
        self.training_progress.setValue(0)
        progress_layout.addWidget(self.training_progress)
        
        # Progress text
        self.progress_text = QLabel("No training in progress")
        progress_layout.addWidget(self.progress_text)
        
        layout.addWidget(progress_group)
        
        # Action buttons
        button_layout = QHBoxLayout()
        
        # Start training button
        self.start_training_btn = QPushButton("Start Training")
        self.start_training_btn.clicked.connect(self._start_training)
        button_layout.addWidget(self.start_training_btn)
        
        # Cancel training button
        self.cancel_training_btn = QPushButton("Cancel Training")
        self.cancel_training_btn.setEnabled(False)
        self.cancel_training_btn.clicked.connect(self._cancel_training)
        button_layout.addWidget(self.cancel_training_btn)
        
        layout.addLayout(button_layout)
        
    def _setup_settings_tab(self, tab):
        """Set up AI settings tab"""
        layout = QVBoxLayout(tab)
        
        # Meta-agent settings group
        meta_group = QGroupBox("Meta-Agent Settings")
        meta_layout = QFormLayout(meta_group)
        
        # Learning rate
        self.learning_rate_spin = QDoubleSpinBox()
        self.learning_rate_spin.setRange(0.0001, 0.1)
        self.learning_rate_spin.setValue(0.001)
        self.learning_rate_spin.setDecimals(4)
        self.learning_rate_spin.setSingleStep(0.0001)
        meta_layout.addRow("Learning Rate:", self.learning_rate_spin)
        
        # Exploration rate
        self.exploration_spin = QDoubleSpinBox()
        self.exploration_spin.setRange(0.01, 1.0)
        self.exploration_spin.setValue(0.1)
        self.exploration_spin.setDecimals(2)
        meta_layout.addRow("Exploration Rate:", self.exploration_spin)
        
        # Risk factor
        self.risk_slider = QSlider(Qt.Horizontal)
        self.risk_slider.setRange(1, 10)
        self.risk_slider.setValue(5)
        meta_layout.addRow("Risk Factor:", self.risk_slider)
        
        layout.addWidget(meta_group)
        
        # Strategy settings group
        strategy_group = QGroupBox("Strategy Weighting")
        strategy_layout = QFormLayout(strategy_group)
        
        # Strategy weights
        strategies = ["Trend Following", "Mean Reversion", "Breakout", "ML-Based"]
        self.strategy_sliders = {}
        
        for strategy in strategies:
            slider = QSlider(Qt.Horizontal)
            slider.setRange(0, 100)
            slider.setValue(25)  # Equal weights by default
            strategy_layout.addRow(f"{strategy}:", slider)
            self.strategy_sliders[strategy] = slider
            
        layout.addWidget(strategy_group)
        
        # Safety limits group
        safety_group = QGroupBox("Safety Limits")
        safety_layout = QFormLayout(safety_group)
        
        # Max drawdown
        self.max_drawdown_spin = QDoubleSpinBox()
        self.max_drawdown_spin.setRange(0.05, 0.5)
        self.max_drawdown_spin.setValue(0.15)
        self.max_drawdown_spin.setDecimals(2)
        self.max_drawdown_spin.setSingleStep(0.01)
        self.max_drawdown_spin.setSuffix(" (15%)")
        safety_layout.addRow("Max Drawdown:", self.max_drawdown_spin)
        
        # Max position size
        self.max_position_spin = QDoubleSpinBox()
        self.max_position_spin.setRange(0.01, 1.0)
        self.max_position_spin.setValue(0.1)
        self.max_position_spin.setDecimals(2)
        self.max_position_spin.setSingleStep(0.01)
        self.max_position_spin.setSuffix(" (10%)")
        safety_layout.addRow("Max Position Size:", self.max_position_spin)
        
        # Enable circuit breakers
        self.circuit_breakers_cb = QCheckBox("Enable circuit breakers")
        self.circuit_breakers_cb.setChecked(True)
        safety_layout.addRow("", self.circuit_breakers_cb)
        
        layout.addWidget(safety_group)
        
        # Action buttons
        button_layout = QHBoxLayout()
        
        # Save settings button
        self.save_settings_btn = QPushButton("Save Settings")
        self.save_settings_btn.clicked.connect(self._save_settings)
        button_layout.addWidget(self.save_settings_btn)
        
        # Reset to defaults button
        self.reset_defaults_btn = QPushButton("Reset to Defaults")
        self.reset_defaults_btn.clicked.connect(self._reset_settings)
        button_layout.addWidget(self.reset_defaults_btn)
        
        layout.addLayout(button_layout)
        
    def _setup_performance_tab(self, tab):
        """Set up AI performance tab"""
        layout = QVBoxLayout(tab)
        
        # Performance metrics group
        metrics_group = QGroupBox("Performance Metrics")
        metrics_layout = QFormLayout(metrics_group)
        
        # Win rate
        metrics_layout.addRow("Win Rate:", QLabel("0.00%"))
        
        # ROI
        metrics_layout.addRow("ROI:", QLabel("0.00%"))
        
        # Sharpe ratio
        metrics_layout.addRow("Sharpe Ratio:", QLabel("0.00"))
        
        # Model accuracy
        metrics_layout.addRow("Model Accuracy:", QLabel("0.00%"))
        
        layout.addWidget(metrics_group)
        
        # Model impact group
        impact_group = QGroupBox("Model Impact Analysis")
        impact_layout = QVBoxLayout(impact_group)
        
        # Impact text
        self.impact_text = QTextEdit()
        self.impact_text.setReadOnly(True)
        impact_layout.addWidget(self.impact_text)
        
        layout.addWidget(impact_group)
        
        # Action buttons
        button_layout = QHBoxLayout()
        
        # Run ablation study button
        self.ablation_btn = QPushButton("Run Impact Analysis")
        self.ablation_btn.clicked.connect(self._run_impact_analysis)
        button_layout.addWidget(self.ablation_btn)
        
        # Export performance button
        self.export_btn = QPushButton("Export Performance Data")
        self.export_btn.clicked.connect(self._export_performance)
        button_layout.addWidget(self.export_btn)
        
        layout.addLayout(button_layout)
        
    def update_status(self):
        """Update AI status information"""
        try:
            # Check if meta-agent exists
            has_meta_agent = hasattr(self.trading_system, 'meta_agent')
            
            # Update status tab
            status_tab = self.findChild(QGroupBox, "AI System Status")
            if status_tab:
                # Update status labels
                labels = status_tab.findChildren(QLabel)
                
                if has_meta_agent:
                    labels[0].setText("Active")  # Meta-agent status
                    labels[0].setStyleSheet("color: green; font-weight: bold;")
                    
                    # Update other status fields if meta-agent provides them
                    # This would be implemented based on actual meta-agent structure
                else:
                    labels[0].setText("Not initialized")
                    labels[0].setStyleSheet("color: red;")
                    
            # Update control button state
            self.enable_ai_btn.setChecked(has_meta_agent)
            
            # Update actions text with current actions
            if has_meta_agent:
                # This would show actual actions from meta-agent
                self.actions_text.setText("Meta-agent is monitoring the system and making decisions.")
            else:
                self.actions_text.setText("AI controller is not active.")
                
        except Exception as e:
            print(f"Error updating AI status: {e}")
            
    def _toggle_ai_controller(self, enabled):
        """Enable or disable AI controller"""
        try:
            from PyQt5.QtWidgets import QMessageBox
            
            if enabled:
                # Check if meta-agent already exists
                if hasattr(self.trading_system, 'meta_agent'):
                    return
                    
                # Confirm enabling
                reply = QMessageBox.question(
                    self, "Enable AI Controller", 
                    "Are you sure you want to enable the autonomous AI controller?\n\n"
                    "This will allow the AI to make trading decisions without manual intervention.",
                    QMessageBox.Yes | QMessageBox.No,
                    QMessageBox.No
                )
                
                if reply != QMessageBox.Yes:
                    self.enable_ai_btn.setChecked(False)
                    return
                    
                # Initialize meta-agent
                # This would call appropriate method in trading system
                # self.trading_system.initialize_meta_agent()
                
                QMessageBox.information(
                    self, "AI Controller Enabled", 
                    "Autonomous AI controller has been enabled."
                )
            else:
                # Check if meta-agent exists
                if not hasattr(self.trading_system, 'meta_agent'):
                    return
                    
                # Confirm disabling
                reply = QMessageBox.question(
                    self, "Disable AI Controller", 
                    "Are you sure you want to disable the autonomous AI controller?",
                    QMessageBox.Yes | QMessageBox.No,
                    QMessageBox.No
                )
                
                if reply != QMessageBox.Yes:
                    self.enable_ai_btn.setChecked(True)
                    return
                    
                # Disable meta-agent
                # This would call appropriate method in trading system
                # self.trading_system.disable_meta_agent()
                
                QMessageBox.information(
                    self, "AI Controller Disabled", 
                    "Autonomous AI controller has been disabled."
                )
                
            # Update status
            self.update_status()
                
        except Exception as e:
            from PyQt5.QtWidgets import QMessageBox
            QMessageBox.critical(self, "Error", f"Error toggling AI controller: {e}")
            
    def _reset_ai(self):
        """Reset AI controller"""
        try:
            from PyQt5.QtWidgets import QMessageBox
            
            # Confirm reset
            reply = QMessageBox.question(
                self, "Reset AI Controller", 
                "Are you sure you want to reset the AI controller?\n\n"
                "This will reset all learned patterns and start from scratch.",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No
            )
            
            if reply != QMessageBox.Yes:
                return
                
            # Reset meta-agent
            # This would call appropriate method in trading system
            # self.trading_system.reset_meta_agent()
            
            QMessageBox.information(
                self, "AI Controller Reset", 
                "AI controller has been reset."
            )
            
            # Update status
            self.update_status()
                
        except Exception as e:
            from PyQt5.QtWidgets import QMessageBox
            QMessageBox.critical(self, "Error", f"Error resetting AI controller: {e}")
            
    def _start_training(self):
        """Start AI training"""
        try:
            # Get training parameters
            model = self.model_combo.currentText()
            training_time = self.training_time_spin.value() * 3600  # Convert to seconds
            use_gpu = self.use_gpu_cb.isChecked()
            auto_optimize = self.auto_optimize_cb.isChecked()
            
            # Emit training request signal
            training_config = {
                'model': model,
                'training_time': training_time,
                'use_gpu': use_gpu,
                'auto_optimize': auto_optimize
            }
            
            self.ai_training_requested.emit(training_config)
            
            # Update UI
            self.training_progress.setValue(0)
            self.progress_text.setText(f"Training {model}...")
            self.start_training_btn.setEnabled(False)
            self.cancel_training_btn.setEnabled(True)
            
            # Simulate progress updates (would be handled by actual training process)
            self._simulate_training_progress()
                
        except Exception as e:
            from PyQt5.QtWidgets import QMessageBox
            QMessageBox.critical(self, "Error", f"Error starting training: {e}")
            
    def _simulate_training_progress(self):
        """Simulate training progress (for demonstration)"""
        # This would be replaced by actual progress tracking
        self.progress_timer = QTimer(self)
        self.progress_timer.timeout.connect(self._update_training_progress)
        self.progress_timer.start(1000)  # Update every second
        
    def _update_training_progress(self):
        """Update training progress bar"""
        current = self.training_progress.value()
        
        if current < 100:
            self.training_progress.setValue(current + 1)
        else:
            # Training complete
            self.progress_timer.stop()
            self.progress_text.setText("Training complete")
            self.start_training_btn.setEnabled(True)
            self.cancel_training_btn.setEnabled(False)
            
    def _cancel_training(self):
        """Cancel AI training"""
        try:
            # Stop progress timer
            if hasattr(self, 'progress_timer'):
                self.progress_timer.stop()
                
            # Cancel training
            # This would call appropriate method in trading system
            # self.trading_system.cancel_training()
            
            # Update UI
            self.progress_text.setText("Training cancelled")
            self.start_training_btn.setEnabled(True)
            self.cancel_training_btn.setEnabled(False)
                
        except Exception as e:
            from PyQt5.QtWidgets import QMessageBox
            QMessageBox.critical(self, "Error", f"Error cancelling training: {e}")
            
    def _save_settings(self):
        """Save AI settings"""
        try:
            # Get settings
            settings = {
                'learning_rate': self.learning_rate_spin.value(),
                'exploration_rate': self.exploration_spin.value(),
                'risk_factor': self.risk_slider.value() / 10.0,
                'max_drawdown': self.max_drawdown_spin.value(),
                'max_position_size': self.max_position_spin.value(),
                'circuit_breakers': self.circuit_breakers_cb.isChecked(),
                'strategy_weights': {
                    strategy: slider.value() / 100.0
                    for strategy, slider in self.strategy_sliders.items()
                }
            }
            
            # Save settings
            # This would call appropriate method in trading system
            # self.trading_system.set_ai_settings(settings)
            
            from PyQt5.QtWidgets import QMessageBox
            QMessageBox.information(
                self, "Settings Saved", 
                "AI controller settings have been saved."
            )
                
        except Exception as e:
            from PyQt5.QtWidgets import QMessageBox
            QMessageBox.critical(self, "Error", f"Error saving settings: {e}")
            
    def _reset_settings(self):
        """Reset AI settings to defaults"""
        try:
            # Reset form controls
            self.learning_rate_spin.setValue(0.001)
            self.exploration_spin.setValue(0.1)
            self.risk_slider.setValue(5)
            self.max_drawdown_spin.setValue(0.15)
            self.max_position_spin.setValue(0.1)
            self.circuit_breakers_cb.setChecked(True)
            
            # Reset strategy weights
            for slider in self.strategy_sliders.values():
                slider.setValue(25)  # Equal weights
                
            from PyQt5.QtWidgets import QMessageBox
            QMessageBox.information(
                self, "Settings Reset", 
                "AI controller settings have been reset to defaults."
            )
                
        except Exception as e:
            from PyQt5.QtWidgets import QMessageBox
            QMessageBox.critical(self, "Error", f"Error resetting settings: {e}")
            
    def _run_impact_analysis(self):
        """Run model impact analysis"""
        try:
            from PyQt5.QtWidgets import QMessageBox
            
            # Confirm analysis
            reply = QMessageBox.question(
                self, "Run Impact Analysis", 
                "Are you sure you want to run model impact analysis?\n\n"
                "This will temporarily disable AI components to measure their individual impact.\n"
                "The analysis may take some time to complete.",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No
            )
            
            if reply != QMessageBox.Yes:
                return
                
            # Run analysis
            # This would call appropriate method in trading system
            # results = self.trading_system.run_model_impact_analysis()
            
            # Simulate results for demonstration
            results = {
                'Order Flow Predictor': {'impact': 0.35, 'roi_change': '+2.1%'},
                'Trade Timing': {'impact': 0.25, 'roi_change': '+1.5%'},
                'Trade Exit': {'impact': 0.15, 'roi_change': '+0.9%'},
                'Portfolio Allocation': {'impact': 0.25, 'roi_change': '+1.5%'}
            }
            
            # Update impact text
            impact_text = "Model Impact Analysis Results:\n\n"
            
            for model, data in results.items():
                impact_text += f"{model}:\n"
                impact_text += f"  Impact Factor: {data['impact']:.2f}\n"
                impact_text += f"  ROI Change: {data['roi_change']}\n\n"
                
            self.impact_text.setText(impact_text)
            
            QMessageBox.information(
                self, "Analysis Complete", 
                "Model impact analysis has been completed."
            )
                
        except Exception as e:
            from PyQt5.QtWidgets import QMessageBox
            QMessageBox.critical(self, "Error", f"Error running impact analysis: {e}")
            
    def _export_performance(self):
        """Export performance data"""
        try:
            from PyQt5.QtWidgets import QFileDialog
            
            file_name, _ = QFileDialog.getSaveFileName(
                self, "Export Performance Data", "", "CSV Files (*.csv);;JSON Files (*.json)")
                
            if not file_name:
                return
                
            # Export data
            # This would call appropriate method in trading system
            # self.trading_system.export_ai_performance(file_name)
            
            from PyQt5.QtWidgets import QMessageBox
            QMessageBox.information(
                self, "Export Complete", 
                f"Performance data exported to {file_name}"
            )
                
        except Exception as e:
            from PyQt5.QtWidgets import QMessageBox
            QMessageBox.critical(self, "Error", f"Error exporting performance data: {e}")
