# settings_dialog.py

import os
import json
import logging
from typing import Dict, Any, Optional
from PyQt5.QtWidgets import (QDialog, QTabWidget, QVBoxLayout, QHBoxLayout, QFormLayout, 
                           QLabel, QLineEdit, QCheckBox, QSpinBox, QDoubleSpinBox, 
                           QPushButton, QComboBox, QGroupBox, QFileDialog, QMessageBox,
                           QDialogButtonBox, QWidget, QSizePolicy, QTextEdit, QScrollArea)
from PyQt5.QtCore import Qt, QSettings, pyqtSignal, QSize
from PyQt5.QtGui import QFont, QIcon

class SettingsDialog(QDialog):
    """
    Comprehensive settings dialog for configuring all aspects of the trading system.
    Provides tabbed interface for various settings categories with validation and persistence.
    """
    
    # Signal emitted when settings are applied
    settings_changed = pyqtSignal(dict)
    
    def __init__(self, trading_system, parent=None):
        super().__init__(parent)
        self.trading_system = trading_system
        self.original_settings = {}
        self.current_settings = {}
        
        # Initialize UI
        self.setWindowTitle("Trading System Settings")
        self.setMinimumSize(800, 600)
        self.setWindowIcon(QIcon("icons/settings.png") if os.path.exists("icons/settings.png") else None)
        
        self._init_ui()
        self._load_current_settings()
        
    def _init_ui(self):
        """Initialize the UI components"""
        # Main layout
        main_layout = QVBoxLayout(self)
        
        # Create tab widget
        self.tab_widget = QTabWidget()
        
        # Add tabs for different settings categories
        self.general_tab = self._create_general_tab()
        self.tab_widget.addTab(self.general_tab, "General")
        
        self.trading_tab = self._create_trading_tab()
        self.tab_widget.addTab(self.trading_tab, "Trading")
        
        self.strategy_tab = self._create_strategy_tab()
        self.tab_widget.addTab(self.strategy_tab, "Strategies")
        
        self.risk_tab = self._create_risk_tab()
        self.tab_widget.addTab(self.risk_tab, "Risk Management")
        
        self.ai_tab = self._create_ai_tab()
        self.tab_widget.addTab(self.ai_tab, "AI & ML")
        
        self.data_tab = self._create_data_tab()
        self.tab_widget.addTab(self.data_tab, "Data & API")
        
        self.appearance_tab = self._create_appearance_tab()
        self.tab_widget.addTab(self.appearance_tab, "Appearance")
        
        # Add tab widget to main layout
        main_layout.addWidget(self.tab_widget)
        
        # Button box
        button_box = QDialogButtonBox(QDialogButtonBox.Save | QDialogButtonBox.Cancel | QDialogButtonBox.Apply | QDialogButtonBox.RestoreDefaults)
        button_box.accepted.connect(self._on_save)
        button_box.rejected.connect(self.reject)
        button_box.button(QDialogButtonBox.Apply).clicked.connect(self._on_apply)
        button_box.button(QDialogButtonBox.RestoreDefaults).clicked.connect(self._on_restore_defaults)
        
        main_layout.addWidget(button_box)
        
        # Dictionary to keep track of all settings controls
        self.controls = {}
        
    def _create_scrollable_widget(self, layout):
        """Create a scrollable widget containing the given layout"""
        # Create a widget to hold the layout
        widget = QWidget()
        widget.setLayout(layout)
        
        # Create a scroll area and set the widget as its content
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setWidget(widget)
        
        return scroll_area
        
    def _create_general_tab(self):
        """Create the general settings tab"""
        layout = QVBoxLayout()
        
        # Application section
        app_group = QGroupBox("Application")
        app_layout = QFormLayout()
        
        # Auto-save settings
        self.auto_save_check = QCheckBox("Enabled")
        app_layout.addRow("Auto-save Settings:", self.auto_save_check)
        self.controls["auto_save_settings"] = self.auto_save_check
        
        # Auto-save interval
        self.auto_save_interval = QSpinBox()
        self.auto_save_interval.setRange(1, 60)
        self.auto_save_interval.setSuffix(" minutes")
        app_layout.addRow("Auto-save Interval:", self.auto_save_interval)
        self.controls["auto_save_interval"] = self.auto_save_interval
        
        # Start minimized
        self.start_minimized_check = QCheckBox()
        app_layout.addRow("Start Minimized:", self.start_minimized_check)
        self.controls["start_minimized"] = self.start_minimized_check
        
        # Minimize to tray
        self.minimize_to_tray_check = QCheckBox()
        app_layout.addRow("Minimize to Tray:", self.minimize_to_tray_check)
        self.controls["minimize_to_tray"] = self.minimize_to_tray_check
        
        # Confirm on exit
        self.confirm_exit_check = QCheckBox()
        app_layout.addRow("Confirm on Exit:", self.confirm_exit_check)
        self.controls["confirm_exit"] = self.confirm_exit_check
        
        app_group.setLayout(app_layout)
        layout.addWidget(app_group)
        
        # Logging section
        log_group = QGroupBox("Logging")
        log_layout = QFormLayout()
        
        # Log level
        self.log_level_combo = QComboBox()
        self.log_level_combo.addItems(["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"])
        log_layout.addRow("Log Level:", self.log_level_combo)
        self.controls["log_level"] = self.log_level_combo
        
        # Log file path
        log_file_layout = QHBoxLayout()
        self.log_file_edit = QLineEdit()
        log_file_layout.addWidget(self.log_file_edit)
        self.browse_log_btn = QPushButton("Browse...")
        self.browse_log_btn.clicked.connect(self._browse_log_file)
        log_file_layout.addWidget(self.browse_log_btn)
        log_layout.addRow("Log File:", log_file_layout)
        self.controls["log_file"] = self.log_file_edit
        
        # Log file size
        self.log_size_spin = QSpinBox()
        self.log_size_spin.setRange(1, 100)
        self.log_size_spin.setSuffix(" MB")
        log_layout.addRow("Max Log Size:", self.log_size_spin)
        self.controls["max_log_size"] = self.log_size_spin
        
        # Log rotation
        self.log_rotation_spin = QSpinBox()
        self.log_rotation_spin.setRange(1, 20)
        self.log_rotation_spin.setSuffix(" files")
        log_layout.addRow("Log Rotation:", self.log_rotation_spin)
        self.controls["log_rotation"] = self.log_rotation_spin
        
        log_group.setLayout(log_layout)
        layout.addWidget(log_group)
        
        # Performance section
        perf_group = QGroupBox("Performance")
        perf_layout = QFormLayout()
        
        # Thread count
        self.thread_count_spin = QSpinBox()
        self.thread_count_spin.setRange(1, 16)
        self.thread_count_spin.setSuffix(" threads")
        perf_layout.addRow("Worker Threads:", self.thread_count_spin)
        self.controls["worker_threads"] = self.thread_count_spin
        
        # Update interval
        self.update_interval_spin = QSpinBox()
        self.update_interval_spin.setRange(100, 10000)
        self.update_interval_spin.setSingleStep(100)
        self.update_interval_spin.setSuffix(" ms")
        perf_layout.addRow("UI Update Interval:", self.update_interval_spin)
        self.controls["ui_update_interval"] = self.update_interval_spin
        
        # Chart update limit
        self.chart_update_spin = QSpinBox()
        self.chart_update_spin.setRange(1, 60)
        self.chart_update_spin.setSuffix(" seconds")
        perf_layout.addRow("Chart Update Limit:", self.chart_update_spin)
        self.controls["chart_update_limit"] = self.chart_update_spin
        
        # Use hardware acceleration
        self.hardware_accel_check = QCheckBox()
        perf_layout.addRow("Use Hardware Acceleration:", self.hardware_accel_check)
        self.controls["use_hardware_acceleration"] = self.hardware_accel_check
        
        perf_group.setLayout(perf_layout)
        layout.addWidget(perf_group)
        
        # Add stretch to bottom
        layout.addStretch()
        
        # Create scrollable area
        return self._create_scrollable_widget(layout)
        
    def _create_trading_tab(self):
        """Create the trading settings tab"""
        layout = QVBoxLayout()
        
        # Trading pairs section
        pairs_group = QGroupBox("Trading Pairs")
        pairs_layout = QVBoxLayout()
        
        # Trading pairs
        self.trading_pairs_edit = QTextEdit()
        self.trading_pairs_edit.setPlaceholderText("Enter one trading pair per line, e.g.:\nBTC/USDT\nETH/USDT")
        pairs_layout.addWidget(QLabel("Active Trading Pairs:"))
        pairs_layout.addWidget(self.trading_pairs_edit)
        self.controls["trading_pairs"] = self.trading_pairs_edit
        
        pairs_group.setLayout(pairs_layout)
        layout.addWidget(pairs_group)
        
        # Trading modes section
        modes_group = QGroupBox("Trading Modes")
        modes_layout = QFormLayout()
        
        # Trading mode
        self.trading_mode_combo = QComboBox()
        self.trading_mode_combo.addItems(["Paper Trading", "Live Trading", "Backtest"])
        modes_layout.addRow("Trading Mode:", self.trading_mode_combo)
        self.controls["trading_mode"] = self.trading_mode_combo
        
        # Auto trading
        self.auto_trading_check = QCheckBox()
        modes_layout.addRow("Auto Trading:", self.auto_trading_check)
        self.controls["auto_trading"] = self.auto_trading_check
        
        # Default order size
        self.default_order_size = QDoubleSpinBox()
        self.default_order_size.setRange(0.0001, 100)
        self.default_order_size.setDecimals(4)
        self.default_order_size.setSingleStep(0.01)
        modes_layout.addRow("Default Order Size:", self.default_order_size)
        self.controls["default_order_size"] = self.default_order_size
        
        # Default leverage
        self.default_leverage = QDoubleSpinBox()
        self.default_leverage.setRange(1, 100)
        self.default_leverage.setSingleStep(1)
        modes_layout.addRow("Default Leverage:", self.default_leverage)
        self.controls["default_leverage"] = self.default_leverage
        
        modes_group.setLayout(modes_layout)
        layout.addWidget(modes_group)
        
        # Order execution section
        exec_group = QGroupBox("Order Execution")
        exec_layout = QFormLayout()
        
        # Market/limit default
        self.order_type_combo = QComboBox()
        self.order_type_combo.addItems(["Market", "Limit", "Stop-Limit"])
        exec_layout.addRow("Default Order Type:", self.order_type_combo)
        self.controls["default_order_type"] = self.order_type_combo
        
        # Slippage tolerance
        self.slippage_spin = QDoubleSpinBox()
        self.slippage_spin.setRange(0, 1)
        self.slippage_spin.setDecimals(3)
        self.slippage_spin.setSingleStep(0.001)
        self.slippage_spin.setValue(0.001)
        exec_layout.addRow("Slippage Tolerance:", self.slippage_spin)
        self.controls["slippage_tolerance"] = self.slippage_spin
        
        # Execution delay
        self.execution_delay_spin = QDoubleSpinBox()
        self.execution_delay_spin.setRange(0, 10)
        self.execution_delay_spin.setDecimals(2)
        self.execution_delay_spin.setSingleStep(0.1)
        self.execution_delay_spin.setSuffix(" seconds")
        exec_layout.addRow("Execution Delay:", self.execution_delay_spin)
        self.controls["execution_delay"] = self.execution_delay_spin
        
        # Use smart execution
        self.smart_execution_check = QCheckBox()
        exec_layout.addRow("Use Smart Execution:", self.smart_execution_check)
        self.controls["use_smart_execution"] = self.smart_execution_check
        
        exec_group.setLayout(exec_layout)
        layout.addWidget(exec_group)
        
        # Trading schedule section
        schedule_group = QGroupBox("Trading Schedule")
        schedule_layout = QFormLayout()
        
        # Trade 24/7
        self.trade_24_7_check = QCheckBox()
        schedule_layout.addRow("Trade 24/7:", self.trade_24_7_check)
        self.controls["trade_24_7"] = self.trade_24_7_check
        
        # Trading hours
        self.trading_hours_edit = QLineEdit()
        self.trading_hours_edit.setPlaceholderText("e.g. 09:00-17:00")
        schedule_layout.addRow("Trading Hours:", self.trading_hours_edit)
        self.controls["trading_hours"] = self.trading_hours_edit
        
        schedule_group.setLayout(schedule_layout)
        layout.addWidget(schedule_group)
        
        # Add stretch to bottom
        layout.addStretch()
        
        # Create scrollable area
        return self._create_scrollable_widget(layout)
    
    def _create_strategy_tab(self):
        """Create the strategy settings tab"""
        layout = QVBoxLayout()
        
        # General strategy settings
        general_group = QGroupBox("General Strategy Settings")
        general_layout = QFormLayout()
        
        # Strategy mode
        self.strategy_mode_combo = QComboBox()
        self.strategy_mode_combo.addItems([
            "Single Strategy", 
            "Multiple Strategies", 
            "AI-Driven Selection"
        ])
        general_layout.addRow("Strategy Mode:", self.strategy_mode_combo)
        self.controls["strategy_mode"] = self.strategy_mode_combo
        
        # Max active strategies
        self.max_strategies_spin = QSpinBox()
        self.max_strategies_spin.setRange(1, 20)
        self.max_strategies_spin.setValue(5)
        general_layout.addRow("Max Active Strategies:", self.max_strategies_spin)
        self.controls["max_active_strategies"] = self.max_strategies_spin
        
        # Strategy refresh interval
        self.strategy_refresh_spin = QSpinBox()
        self.strategy_refresh_spin.setRange(1, 60)
        self.strategy_refresh_spin.setValue(15)
        self.strategy_refresh_spin.setSuffix(" minutes")
        general_layout.addRow("Strategy Refresh Interval:", self.strategy_refresh_spin)
        self.controls["strategy_refresh_interval"] = self.strategy_refresh_spin
        
        # Auto-optimize strategies
        self.auto_optimize_check = QCheckBox()
        general_layout.addRow("Auto-optimize Strategies:", self.auto_optimize_check)
        self.controls["auto_optimize_strategies"] = self.auto_optimize_check
        
        general_group.setLayout(general_layout)
        layout.addWidget(general_group)
        
        # Strategy weights
        weights_group = QGroupBox("Strategy Weights")
        weights_layout = QFormLayout()
        
        # Built-in strategy weights
        strategies = ["Trend Following", "Mean Reversion", "Breakout", "Multi-Timeframe", "Volume-Based"]
        
        self.strategy_weights = {}
        for strategy in strategies:
            weight_spin = QDoubleSpinBox()
            weight_spin.setRange(0, 1)
            weight_spin.setDecimals(2)
            weight_spin.setSingleStep(0.1)
            weight_spin.setValue(0.2)  # Default equal weight
            weights_layout.addRow(f"{strategy}:", weight_spin)
            self.strategy_weights[strategy.lower().replace('-', '_')] = weight_spin
            self.controls[f"strategy_weight_{strategy.lower().replace('-', '_')}"] = weight_spin
        
        weights_group.setLayout(weights_layout)
        layout.addWidget(weights_group)
        
        # Strategy parameters
        params_group = QGroupBox("Default Strategy Parameters")
        params_layout = QFormLayout()
        
        # Trend following parameters
        tf_group = QGroupBox("Trend Following")
        tf_layout = QFormLayout()
        
        self.tf_short_window = QSpinBox()
        self.tf_short_window.setRange(5, 50)
        self.tf_short_window.setValue(10)
        tf_layout.addRow("Short Window:", self.tf_short_window)
        self.controls["tf_short_window"] = self.tf_short_window
        
        self.tf_long_window = QSpinBox()
        self.tf_long_window.setRange(20, 200)
        self.tf_long_window.setValue(50)
        tf_layout.addRow("Long Window:", self.tf_long_window)
        self.controls["tf_long_window"] = self.tf_long_window
        
        tf_group.setLayout(tf_layout)
        params_layout.addRow(tf_group)
        
        # Mean reversion parameters
        mr_group = QGroupBox("Mean Reversion")
        mr_layout = QFormLayout()
        
        self.mr_window = QSpinBox()
        self.mr_window.setRange(10, 100)
        self.mr_window.setValue(20)
        mr_layout.addRow("Window:", self.mr_window)
        self.controls["mr_window"] = self.mr_window
        
        self.mr_std_dev = QDoubleSpinBox()
        self.mr_std_dev.setRange(1, 3)
        self.mr_std_dev.setDecimals(2)
        self.mr_std_dev.setSingleStep(0.1)
        self.mr_std_dev.setValue(2.0)
        mr_layout.addRow("Std Dev:", self.mr_std_dev)
        self.controls["mr_std_dev"] = self.mr_std_dev
        
        mr_group.setLayout(mr_layout)
        params_layout.addRow(mr_group)
        
        params_group.setLayout(params_layout)
        layout.addWidget(params_group)
        
        # Add stretch to bottom
        layout.addStretch()
        
        # Create scrollable area
        return self._create_scrollable_widget(layout)
    
    def _create_risk_tab(self):
        """Create the risk management settings tab"""
        layout = QVBoxLayout()
        
        # Position sizing
        pos_group = QGroupBox("Position Sizing")
        pos_layout = QFormLayout()
        
        # Base risk per trade
        self.base_risk_spin = QDoubleSpinBox()
        self.base_risk_spin.setRange(0.001, 0.1)
        self.base_risk_spin.setDecimals(3)
        self.base_risk_spin.setSingleStep(0.001)
        self.base_risk_spin.setValue(0.02)
        pos_layout.addRow("Base Risk Per Trade:", self.base_risk_spin)
        self.controls["base_risk"] = self.base_risk_spin
        
        # Max risk per trade
        self.max_risk_spin = QDoubleSpinBox()
        self.max_risk_spin.setRange(0.001, 0.2)
        self.max_risk_spin.setDecimals(3)
        self.max_risk_spin.setSingleStep(0.001)
        self.max_risk_spin.setValue(0.05)
        pos_layout.addRow("Max Risk Per Trade:", self.max_risk_spin)
        self.controls["max_risk"] = self.max_risk_spin
        
        # Min risk per trade
        self.min_risk_spin = QDoubleSpinBox()
        self.min_risk_spin.setRange(0.001, 0.05)
        self.min_risk_spin.setDecimals(3)
        self.min_risk_spin.setSingleStep(0.001)
        self.min_risk_spin.setValue(0.01)
        pos_layout.addRow("Min Risk Per Trade:", self.min_risk_spin)
        self.controls["min_risk"] = self.min_risk_spin
        
        # Kelly criterion toggle
        self.kelly_check = QCheckBox()
        pos_layout.addRow("Use Kelly Criterion:", self.kelly_check)
        self.controls["use_kelly"] = self.kelly_check
        
        # Kelly fractional setting
        self.kelly_fraction_spin = QDoubleSpinBox()
        self.kelly_fraction_spin.setRange(0.1, 1.0)
        self.kelly_fraction_spin.setDecimals(1)
        self.kelly_fraction_spin.setSingleStep(0.1)
        self.kelly_fraction_spin.setValue(0.5)  # Default to half-Kelly
        pos_layout.addRow("Kelly Fraction:", self.kelly_fraction_spin)
        self.controls["kelly_fraction"] = self.kelly_fraction_spin
        
        # Position volatility adjustment
        self.vol_adjustment_check = QCheckBox()
        self.vol_adjustment_check.setChecked(True)
        pos_layout.addRow("Adjust for Volatility:", self.vol_adjustment_check)
        self.controls["volatility_adjustment"] = self.vol_adjustment_check
        
        pos_group.setLayout(pos_layout)
        layout.addWidget(pos_group)
        
        # Risk limits
        limits_group = QGroupBox("Risk Limits")
        limits_layout = QFormLayout()
        
        # Maximum positions
        self.max_positions_spin = QSpinBox()
        self.max_positions_spin.setRange(1, 50)
        self.max_positions_spin.setValue(10)
        limits_layout.addRow("Max Open Positions:", self.max_positions_spin)
        self.controls["max_open_positions"] = self.max_positions_spin
        
        # Max risk per sector/asset
        self.max_sector_risk_spin = QDoubleSpinBox()
        self.max_sector_risk_spin.setRange(0.01, 0.5)
        self.max_sector_risk_spin.setDecimals(2)
        self.max_sector_risk_spin.setSingleStep(0.01)
        self.max_sector_risk_spin.setValue(0.2)
        limits_layout.addRow("Max Risk Per Asset:", self.max_sector_risk_spin)
        self.controls["max_risk_per_asset"] = self.max_sector_risk_spin
        
        # Daily loss limit
        self.daily_loss_spin = QDoubleSpinBox()
        self.daily_loss_spin.setRange(0.01, 0.2)
        self.daily_loss_spin.setDecimals(2)
        self.daily_loss_spin.setSingleStep(0.01)
        self.daily_loss_spin.setValue(0.03)
        limits_layout.addRow("Daily Loss Limit:", self.daily_loss_spin)
        self.controls["daily_loss_limit"] = self.daily_loss_spin
        
        # Max drawdown limit
        self.max_drawdown_spin = QDoubleSpinBox()
        self.max_drawdown_spin.setRange(0.05, 0.5)
        self.max_drawdown_spin.setDecimals(2)
        self.max_drawdown_spin.setSingleStep(0.01)
        self.max_drawdown_spin.setValue(0.15)
        limits_layout.addRow("Max Drawdown Limit:", self.max_drawdown_spin)
        self.controls["max_drawdown_limit"] = self.max_drawdown_spin
        
        # Correlation threshold
        self.correlation_spin = QDoubleSpinBox()
        self.correlation_spin.setRange(0.3, 0.9)
        self.correlation_spin.setDecimals(2)
        self.correlation_spin.setSingleStep(0.05)
        self.correlation_spin.setValue(0.7)
        limits_layout.addRow("Correlation Threshold:", self.correlation_spin)
        self.controls["correlation_threshold"] = self.correlation_spin
        
        limits_group.setLayout(limits_layout)
        layout.addWidget(limits_group)
        
        # Hedging settings
        hedge_group = QGroupBox("Hedging Settings")
        hedge_layout = QFormLayout()
        
        # Auto-hedging
        self.auto_hedge_check = QCheckBox()
        hedge_layout.addRow("Auto-Hedging:", self.auto_hedge_check)
        self.controls["auto_hedging"] = self.auto_hedge_check
        
        # Hedge trigger threshold
        self.hedge_threshold_spin = QDoubleSpinBox()
        self.hedge_threshold_spin.setRange(0.01, 0.2)
        self.hedge_threshold_spin.setDecimals(2)
        self.hedge_threshold_spin.setSingleStep(0.01)
        self.hedge_threshold_spin.setValue(0.05)
        hedge_layout.addRow("Hedge Trigger Threshold:", self.hedge_threshold_spin)
        self.controls["hedge_threshold"] = self.hedge_threshold_spin
        
        # Hedge size
        self.hedge_size_spin = QDoubleSpinBox()
        self.hedge_size_spin.setRange(0.1, 1.0)
        self.hedge_size_spin.setDecimals(2)
        self.hedge_size_spin.setSingleStep(0.1)
        self.hedge_size_spin.setValue(0.5)
        hedge_layout.addRow("Hedge Size:", self.hedge_size_spin)
        self.controls["hedge_size"] = self.hedge_size_spin
        
        hedge_group.setLayout(hedge_layout)
        layout.addWidget(hedge_group)
        
        # Circuit breakers
        circuit_group = QGroupBox("Circuit Breakers")
        circuit_layout = QFormLayout()
        
        # Enable circuit breakers
        self.circuit_breakers_check = QCheckBox()
        self.circuit_breakers_check.setChecked(True)
        circuit_layout.addRow("Enable Circuit Breakers:", self.circuit_breakers_check)
        self.controls["enable_circuit_breakers"] = self.circuit_breakers_check
        
        # Daily loss breaker
        self.daily_loss_breaker_spin = QDoubleSpinBox()
        self.daily_loss_breaker_spin.setRange(0.03, 0.3)
        self.daily_loss_breaker_spin.setDecimals(2)
        self.daily_loss_breaker_spin.setSingleStep(0.01)
        self.daily_loss_breaker_spin.setValue(0.07)
        circuit_layout.addRow("Daily Loss Breaker:", self.daily_loss_breaker_spin)
        self.controls["daily_loss_breaker"] = self.daily_loss_breaker_spin
        
        # Flash crash detection
        self.flash_crash_check = QCheckBox()
        self.flash_crash_check.setChecked(True)
        circuit_layout.addRow("Flash Crash Detection:", self.flash_crash_check)
        self.controls["flash_crash_detection"] = self.flash_crash_check
        
        circuit_group.setLayout(circuit_layout)
        layout.addWidget(circuit_group)
        
        # Add stretch to bottom
        layout.addStretch()
        
        # Create scrollable area
        return self._create_scrollable_widget(layout)
        
    def _create_ai_tab(self):
        """Create the AI & ML settings tab"""
        layout = QVBoxLayout()
        
        # General AI settings
        general_group = QGroupBox("General AI Settings")
        general_layout = QFormLayout()
        
        # Enable AI
        self.enable_ai_check = QCheckBox()
        general_layout.addRow("Enable AI:", self.enable_ai_check)
        self.controls["enable_ai"] = self.enable_ai_check
        
        # AI confidence threshold
        self.ai_confidence_spin = QDoubleSpinBox()
        self.ai_confidence_spin.setRange(0.5, 0.95)
        self.ai_confidence_spin.setDecimals(2)
        self.ai_confidence_spin.setSingleStep(0.05)
        self.ai_confidence_spin.setValue(0.7)
        general_layout.addRow("Confidence Threshold:", self.ai_confidence_spin)
        self.controls["ai_confidence_threshold"] = self.ai_confidence_spin
        
        # AI control mode
        self.ai_control_combo = QComboBox()
        self.ai_control_combo.addItems([
            "Advisory (Suggestions Only)", 
            "Semi-Autonomous (Require Confirmation)", 
            "Fully Autonomous"
        ])
        general_layout.addRow("AI Control Mode:", self.ai_control_combo)
        self.controls["ai_control_mode"] = self.ai_control_combo
        
        # Use GPU for inference
        self.gpu_inference_check = QCheckBox()
        general_layout.addRow("Use GPU for Inference:", self.gpu_inference_check)
        self.controls["use_gpu_inference"] = self.gpu_inference_check
        
        general_group.setLayout(general_layout)
        layout.addWidget(general_group)
        
        # Model settings
        model_group = QGroupBox("Model Settings")
        model_layout = QFormLayout()
        
        # Order flow model enabled
        self.order_flow_model_check = QCheckBox()
        self.order_flow_model_check.setChecked(True)
        model_layout.addRow("Order Flow Model:", self.order_flow_model_check)
        self.controls["enable_order_flow_model"] = self.order_flow_model_check
        
        # Trade timing model enabled
        self.trade_timing_model_check = QCheckBox()
        self.trade_timing_model_check.setChecked(True)
        model_layout.addRow("Trade Timing Model:", self.trade_timing_model_check)
        self.controls["enable_trade_timing_model"] = self.trade_timing_model_check
        
        # Trade exit model enabled
        self.trade_exit_model_check = QCheckBox()
        self.trade_exit_model_check.setChecked(True)
        model_layout.addRow("Trade Exit Model:", self.trade_exit_model_check)
        self.controls["enable_trade_exit_model"] = self.trade_exit_model_check
        
        # Portfolio allocation model enabled
        self.portfolio_model_check = QCheckBox()
        self.portfolio_model_check.setChecked(True)
        model_layout.addRow("Portfolio Allocation Model:", self.portfolio_model_check)
        self.controls["enable_portfolio_model"] = self.portfolio_model_check
        
        # Risk model enabled
        self.risk_model_check = QCheckBox()
        self.risk_model_check.setChecked(True)
        model_layout.addRow("Risk Management Model:", self.risk_model_check)
        self.controls["enable_risk_model"] = self.risk_model_check
        
        model_group.setLayout(model_layout)
        layout.addWidget(model_group)
        
        # Training settings
        training_group = QGroupBox("Training Settings")
        training_layout = QFormLayout()
        
        # Auto-training
        self.auto_training_check = QCheckBox()
        training_layout.addRow("Auto-Training:", self.auto_training_check)
        self.controls["auto_training"] = self.auto_training_check
        
        # Training schedule
        self.training_schedule_combo = QComboBox()
        self.training_schedule_combo.addItems([
            "Daily", 
            "Weekly", 
            "Monthly", 
            "On Signal Degradation"
        ])
        training_layout.addRow("Training Schedule:", self.training_schedule_combo)
        self.controls["training_schedule"] = self.training_schedule_combo
        
        # Training time limit
        self.training_time_spin = QSpinBox()
        self.training_time_spin.setRange(1, 24)
        self.training_time_spin.setValue(2)
        self.training_time_spin.setSuffix(" hours")
        training_layout.addRow("Training Time Limit:", self.training_time_spin)
        self.controls["training_time_limit"] = self.training_time_spin
        
        # Use GPU for training
        self.gpu_training_check = QCheckBox()
        self.gpu_training_check.setChecked(True)
        training_layout.addRow("Use GPU for Training:", self.gpu_training_check)
        self.controls["use_gpu_training"] = self.gpu_training_check
        
        training_group.setLayout(training_layout)
        layout.addWidget(training_group)
        
        # Model Path settings
        path_group = QGroupBox("Model Paths")
        path_layout = QFormLayout()
        
        # Models directory
        models_dir_layout = QHBoxLayout()
        self.models_dir_edit = QLineEdit()
        models_dir_layout.addWidget(self.models_dir_edit)
        self.browse_models_btn = QPushButton("Browse...")
        self.browse_models_btn.clicked.connect(self._browse_models_dir)
        models_dir_layout.addWidget(self.browse_models_btn)
        path_layout.addRow("Models Directory:", models_dir_layout)
        self.controls["models_directory"] = self.models_dir_edit
        
        path_group.setLayout(path_layout)
        layout.addWidget(path_group)
        
        # Add stretch to bottom
        layout.addStretch()
        
        # Create scrollable area
        return self._create_scrollable_widget(layout)
        
    def _create_data_tab(self):
        """Create the data & API settings tab"""
        layout = QVBoxLayout()
        
        # Data source settings
        source_group = QGroupBox("Data Sources")
        source_layout = QFormLayout()
        
        # Primary data source
        self.primary_source_combo = QComboBox()
        self.primary_source_combo.addItems([
            "Binance", 
            "Coinbase", 
            "Kraken", 
            "Bybit", 
            "Custom API"
        ])
        source_layout.addRow("Primary Data Source:", self.primary_source_combo)
        self.controls["primary_data_source"] = self.primary_source_combo
        
        # Secondary data source
        self.secondary_source_combo = QComboBox()
        self.secondary_source_combo.addItems([
            "None", 
            "Binance", 
            "Coinbase", 
            "Kraken", 
            "Bybit", 
            "Custom API"
        ])
        source_layout.addRow("Secondary Data Source:", self.secondary_source_combo)
        self.controls["secondary_data_source"] = self.secondary_source_combo
        
        # Data update interval
        self.data_interval_spin = QSpinBox()
        self.data_interval_spin.setRange(1, 60)
        self.data_interval_spin.setValue(5)
        self.data_interval_spin.setSuffix(" seconds")
        source_layout.addRow("Data Update Interval:", self.data_interval_spin)
        self.controls["data_update_interval"] = self.data_interval_spin
        
        # Use websocket
        self.websocket_check = QCheckBox()
        self.websocket_check.setChecked(True)
        source_layout.addRow("Use WebSocket:", self.websocket_check)
        self.controls["use_websocket"] = self.websocket_check
        
        source_group.setLayout(source_layout)
        layout.addWidget(source_group)
        
        # API settings
        api_group = QGroupBox("API Settings")
        api_layout = QVBoxLayout()
        
        # Binance API
        binance_group = QGroupBox("Binance")
        binance_layout = QFormLayout()
        
        self.binance_api_key_edit = QLineEdit()
        self.binance_api_key_edit.setEchoMode(QLineEdit.Password)
        binance_layout.addRow("API Key:", self.binance_api_key_edit)
        self.controls["binance_api_key"] = self.binance_api_key_edit
        
        self.binance_api_secret_edit = QLineEdit()
        self.binance_api_secret_edit.setEchoMode(QLineEdit.Password)
        binance_layout.addRow("API Secret:", self.binance_api_secret_edit)
        self.controls["binance_api_secret"] = self.binance_api_secret_edit
        
        binance_group.setLayout(binance_layout)
        api_layout.addWidget(binance_group)
        
        # Coinbase API
        coinbase_group = QGroupBox("Coinbase")
        coinbase_layout = QFormLayout()
        
        self.coinbase_api_key_edit = QLineEdit()
        self.coinbase_api_key_edit.setEchoMode(QLineEdit.Password)
        coinbase_layout.addRow("API Key:", self.coinbase_api_key_edit)
        self.controls["coinbase_api_key"] = self.coinbase_api_key_edit
        
        self.coinbase_api_secret_edit = QLineEdit()
        self.coinbase_api_secret_edit.setEchoMode(QLineEdit.Password)
        coinbase_layout.addRow("API Secret:", self.coinbase_api_secret_edit)
        self.controls["coinbase_api_secret"] = self.coinbase_api_secret_edit
        
        self.coinbase_passphrase_edit = QLineEdit()
        self.coinbase_passphrase_edit.setEchoMode(QLineEdit.Password)
        coinbase_layout.addRow("Passphrase:", self.coinbase_passphrase_edit)
        self.controls["coinbase_passphrase"] = self.coinbase_passphrase_edit
        
        coinbase_group.setLayout(coinbase_layout)
        api_layout.addWidget(coinbase_group)
        
        api_group.setLayout(api_layout)
        layout.addWidget(api_group)
        
        # Database settings
        db_group = QGroupBox("Database Settings")
        db_layout = QFormLayout()
        
        # Use database
        self.use_db_check = QCheckBox()
        db_layout.addRow("Use Database:", self.use_db_check)
        self.controls["use_database"] = self.use_db_check
        
        # Database type
        self.db_type_combo = QComboBox()
        self.db_type_combo.addItems([
            "TimescaleDB", 
            "SQLite", 
            "MySQL", 
            "PostgreSQL", 
            "Mock (In-Memory)"
        ])
        db_layout.addRow("Database Type:", self.db_type_combo)
        self.controls["database_type"] = self.db_type_combo
        
        # Database host
        self.db_host_edit = QLineEdit()
        self.db_host_edit.setText("localhost")
        db_layout.addRow("Database Host:", self.db_host_edit)
        self.controls["database_host"] = self.db_host_edit
        
        # Database port
        self.db_port_spin = QSpinBox()
        self.db_port_spin.setRange(1, 65535)
        self.db_port_spin.setValue(5432)
        db_layout.addRow("Database Port:", self.db_port_spin)
        self.controls["database_port"] = self.db_port_spin
        
        # Database name
        self.db_name_edit = QLineEdit()
        self.db_name_edit.setText("crypto_trading")
        db_layout.addRow("Database Name:", self.db_name_edit)
        self.controls["database_name"] = self.db_name_edit
        
        # Database user
        self.db_user_edit = QLineEdit()
        self.db_user_edit.setText("trading_user")
        db_layout.addRow("Database User:", self.db_user_edit)
        self.controls["database_user"] = self.db_user_edit
        
        # Database password
        self.db_password_edit = QLineEdit()
        self.db_password_edit.setEchoMode(QLineEdit.Password)
        db_layout.addRow("Database Password:", self.db_password_edit)
        self.controls["database_password"] = self.db_password_edit
        
        db_group.setLayout(db_layout)
        layout.addWidget(db_group)
        
        # Data retention
        retention_group = QGroupBox("Data Retention")
        retention_layout = QFormLayout()
        
        # Market data retention
        self.market_data_retention_spin = QSpinBox()
        self.market_data_retention_spin.setRange(1, 365)
        self.market_data_retention_spin.setValue(30)
        self.market_data_retention_spin.setSuffix(" days")
        retention_layout.addRow("Market Data Retention:", self.market_data_retention_spin)
        self.controls["market_data_retention"] = self.market_data_retention_spin
        
        # Order history retention
        self.order_retention_spin = QSpinBox()
        self.order_retention_spin.setRange(1, 365)
        self.order_retention_spin.setValue(90)
        self.order_retention_spin.setSuffix(" days")
        retention_layout.addRow("Order History Retention:", self.order_retention_spin)
        self.controls["order_history_retention"] = self.order_retention_spin
        
        retention_group.setLayout(retention_layout)
        layout.addWidget(retention_group)
        
        # Add stretch to bottom
        layout.addStretch()
        
        # Create scrollable area
        return self._create_scrollable_widget(layout)
        
    def _create_appearance_tab(self):
        """Create the appearance settings tab"""
        layout = QVBoxLayout()
        
        # Theme settings
        theme_group = QGroupBox("Theme")
        theme_layout = QFormLayout()
        
        # Application theme
        self.theme_combo = QComboBox()
        self.theme_combo.addItems([
            "System Default", 
            "Light", 
            "Dark", 
            "Blue", 
            "Custom"
        ])
        theme_layout.addRow("Application Theme:", self.theme_combo)
        self.controls["app_theme"] = self.theme_combo
        
        # Chart theme
        self.chart_theme_combo = QComboBox()
        self.chart_theme_combo.addItems([
            "Light", 
            "Dark", 
            "TradingView", 
            "Custom"
        ])
        theme_layout.addRow("Chart Theme:", self.chart_theme_combo)
        self.controls["chart_theme"] = self.chart_theme_combo
        
        theme_group.setLayout(theme_layout)
        layout.addWidget(theme_group)
        
        # Font settings
        font_group = QGroupBox("Fonts")
        font_layout = QFormLayout()
        
        # Font size
        self.font_size_spin = QSpinBox()
        self.font_size_spin.setRange(8, 24)
        self.font_size_spin.setValue(10)
        font_layout.addRow("Font Size:", self.font_size_spin)
        self.controls["font_size"] = self.font_size_spin
        
        # Font family
        self.font_family_combo = QComboBox()
        self.font_family_combo.addItems([
            "System Default", 
            "Arial", 
            "Helvetica", 
            "Times New Roman", 
            "Courier New", 
            "Roboto", 
            "Open Sans"
        ])
        font_layout.addRow("Font Family:", self.font_family_combo)
        self.controls["font_family"] = self.font_family_combo
        
        font_group.setLayout(font_layout)
        layout.addWidget(font_group)
        
        # Chart settings
        chart_group = QGroupBox("Chart Settings")
        chart_layout = QFormLayout()
        
        # Default timeframe
        self.timeframe_combo = QComboBox()
        self.timeframe_combo.addItems([
            "1m", 
            "5m", 
            "15m", 
            "30m", 
            "1h", 
            "4h", 
            "1d", 
            "1w"
        ])
        self.timeframe_combo.setCurrentText("1h")
        chart_layout.addRow("Default Timeframe:", self.timeframe_combo)
        self.controls["default_timeframe"] = self.timeframe_combo
        
        # Default chart type
        self.chart_type_combo = QComboBox()
        self.chart_type_combo.addItems([
            "Candlestick", 
            "OHLC", 
            "Line", 
            "Area", 
            "Heikin Ashi"
        ])
        chart_layout.addRow("Default Chart Type:", self.chart_type_combo)
        self.controls["default_chart_type"] = self.chart_type_combo
        
        # Show grid
        self.show_grid_check = QCheckBox()
        self.show_grid_check.setChecked(True)
        chart_layout.addRow("Show Grid:", self.show_grid_check)
        self.controls["show_grid"] = self.show_grid_check
        
        # Show volume
        self.show_volume_check = QCheckBox()
        self.show_volume_check.setChecked(True)
        chart_layout.addRow("Show Volume:", self.show_volume_check)
        self.controls["show_volume"] = self.show_volume_check
        
        chart_group.setLayout(chart_layout)
        layout.addWidget(chart_group)
        
        # Layout settings
        layout_group = QGroupBox("Layout")
        layout_layout = QFormLayout()
        
        # Remember window position and size
        self.remember_window_check = QCheckBox()
        self.remember_window_check.setChecked(True)
        layout_layout.addRow("Remember Window Size:", self.remember_window_check)
        self.controls["remember_window_size"] = self.remember_window_check
        
        # Save layout
        self.save_layout_check = QCheckBox()
        self.save_layout_check.setChecked(True)
        layout_layout.addRow("Save Widget Layout:", self.save_layout_check)
        self.controls["save_widget_layout"] = self.save_layout_check
        
        layout_group.setLayout(layout_layout)
        layout.addWidget(layout_group)
        
        # Add stretch to bottom
        layout.addStretch()
        
        # Create scrollable area
        return self._create_scrollable_widget(layout)
        
    def _browse_log_file(self):
        """Browse for log file path"""
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Select Log File", "", "Log Files (*.log);;All Files (*)"
        )
        if file_path:
            self.log_file_edit.setText(file_path)
            
    def _browse_models_dir(self):
        """Browse for models directory"""
        dir_path = QFileDialog.getExistingDirectory(
            self, "Select Models Directory", ""
        )
        if dir_path:
            self.models_dir_edit.setText(dir_path)
            
    def _load_current_settings(self):
        """Load current settings from the trading system"""
        try:
            # Get current settings
            if hasattr(self.trading_system, 'config'):
                self.original_settings = self._get_current_settings()
                self.current_settings = self.original_settings.copy()
                
                # Update UI controls with current settings
                self._update_controls_from_settings(self.current_settings)
                
        except Exception as e:
            logging.error(f"Error loading settings: {e}")
            QMessageBox.warning(self, "Settings Error", f"Error loading settings: {e}")
            
    def _get_current_settings(self) -> Dict[str, Any]:
        """Get current settings from the trading system"""
        settings = {}
        
        # Extract settings from the trading system config
        if hasattr(self.trading_system, 'config'):
            # General settings
            if 'general' in self.trading_system.config:
                general = self.trading_system.config['general']
                settings.update({
                    'auto_save_settings': general.get('auto_save_settings', True),
                    'auto_save_interval': general.get('auto_save_interval', 5),
                    'start_minimized': general.get('start_minimized', False),
                    'minimize_to_tray': general.get('minimize_to_tray', False),
                    'confirm_exit': general.get('confirm_exit', True)
                })
                
            # Logging settings
            if 'logging' in self.trading_system.config:
                logging_config = self.trading_system.config['logging']
                settings.update({
                    'log_level': logging_config.get('level', 'INFO'),
                    'log_file': logging_config.get('file', 'logs/trading.log'),
                    'max_log_size': logging_config.get('max_size_mb', 10),
                    'log_rotation': logging_config.get('rotation_count', 5)
                })
                
            # Performance settings
            if 'performance' in self.trading_system.config:
                perf = self.trading_system.config['performance']
                settings.update({
                    'worker_threads': perf.get('worker_threads', 4),
                    'ui_update_interval': perf.get('ui_update_interval', 1000),
                    'chart_update_limit': perf.get('chart_update_limit', 10),
                    'use_hardware_acceleration': perf.get('use_hardware_acceleration', True)
                })
                
            # Trading settings
            if 'trading' in self.trading_system.config:
                trading = self.trading_system.config['trading']
                settings.update({
                    'trading_mode': trading.get('mode', 'Paper Trading'),
                    'auto_trading': trading.get('auto_trading', False),
                    'default_order_size': trading.get('default_order_size', 0.01),
                    'default_order_type': trading.get('default_order_type', 'Limit'),
                    'default_leverage': trading.get('default_leverage', 1),
                    'slippage_tolerance': trading.get('slippage_tolerance', 0.001),
                    'execution_delay': trading.get('execution_delay', 0.1),
                    'use_smart_execution': trading.get('use_smart_execution', True),
                    'trade_24_7': trading.get('trade_24_7', True),
                    'trading_hours': trading.get('trading_hours', '09:00-17:00')
                })
                
                # Trading pairs
                if 'symbols' in trading:
                    settings['trading_pairs'] = '\n'.join(trading['symbols'])
                
            # Strategy settings
            if 'strategies' in self.trading_system.config:
                strategies = self.trading_system.config['strategies']
                settings.update({
                    'strategy_mode': strategies.get('mode', 'Multiple Strategies'),
                    'max_active_strategies': strategies.get('max_active', 5),
                    'strategy_refresh_interval': strategies.get('refresh_interval', 15),
                    'auto_optimize_strategies': strategies.get('auto_optimize', False)
                })
                
                # Strategy weights
                if 'weights' in strategies:
                    for strategy, weight in strategies['weights'].items():
                        settings[f"strategy_weight_{strategy}"] = weight
                        
            # Risk settings
            if 'risk_limits' in self.trading_system.config:
                risk = self.trading_system.config['risk_limits']
                settings.update({
                    'base_risk': risk.get('base_risk', 0.02),
                    'max_risk': risk.get('max_risk', 0.05),
                    'min_risk': risk.get('min_risk', 0.01),
                    'max_drawdown_limit': risk.get('max_drawdown_limit', 0.15),
                    'daily_loss_limit': risk.get('daily_loss_limit', 0.03),
                    'max_open_positions': risk.get('max_open_positions', 10),
                    'max_risk_per_asset': risk.get('max_risk_per_asset', 0.2),
                    'correlation_threshold': risk.get('correlation_threshold', 0.7),
                    'use_kelly': risk.get('use_kelly', False),
                    'kelly_fraction': risk.get('kelly_fraction', 0.5),
                    'volatility_adjustment': risk.get('volatility_adjustment', True),
                    'auto_hedging': risk.get('auto_hedging', False),
                    'hedge_threshold': risk.get('hedge_threshold', 0.05),
                    'hedge_size': risk.get('hedge_size', 0.5),
                    'enable_circuit_breakers': risk.get('enable_circuit_breakers', True),
                    'daily_loss_breaker': risk.get('daily_loss_breaker', 0.07),
                    'flash_crash_detection': risk.get('flash_crash_detection', True)
                })
                
            # AI settings
            if 'ai' in self.trading_system.config:
                ai = self.trading_system.config['ai']
                settings.update({
                    'enable_ai': ai.get('enable', False),
                    'ai_confidence_threshold': ai.get('confidence_threshold', 0.7),
                    'ai_control_mode': ai.get('control_mode', 'Advisory (Suggestions Only)'),
                    'use_gpu_inference': ai.get('use_gpu_inference', False),
                    'enable_order_flow_model': ai.get('enable_order_flow_model', True),
                    'enable_trade_timing_model': ai.get('enable_trade_timing_model', True),
                    'enable_trade_exit_model': ai.get('enable_trade_exit_model', True),
                    'enable_portfolio_model': ai.get('enable_portfolio_model', True),
                    'enable_risk_model': ai.get('enable_risk_model', True),
                    'auto_training': ai.get('auto_training', False),
                    'training_schedule': ai.get('training_schedule', 'Weekly'),
                    'training_time_limit': ai.get('training_time_limit', 2),
                    'use_gpu_training': ai.get('use_gpu_training', True),
                    'models_directory': ai.get('models_directory', 'models')
                })
                
            # Data API settings
            if 'data' in self.trading_system.config:
                data = self.trading_system.config['data']
                settings.update({
                    'primary_data_source': data.get('primary_source', 'Binance'),
                    'secondary_data_source': data.get('secondary_source', 'None'),
                    'data_update_interval': data.get('update_interval', 5),
                    'use_websocket': data.get('use_websocket', True),
                    'market_data_retention': data.get('market_data_retention', 30),
                    'order_history_retention': data.get('order_history_retention', 90)
                })
                
                # API keys
                if 'api_keys' in data:
                    api_keys = data['api_keys']
                    if 'binance' in api_keys:
                        settings.update({
                            'binance_api_key': api_keys['binance'].get('key', ''),
                            'binance_api_secret': api_keys['binance'].get('secret', '')
                        })
                    if 'coinbase' in api_keys:
                        settings.update({
                            'coinbase_api_key': api_keys['coinbase'].get('key', ''),
                            'coinbase_api_secret': api_keys['coinbase'].get('secret', ''),
                            'coinbase_passphrase': api_keys['coinbase'].get('passphrase', '')
                        })
                
                # Database settings
                if 'database' in data:
                    db = data['database']
                    settings.update({
                        'use_database': db.get('use_database', True),
                        'database_type': db.get('type', 'TimescaleDB'),
                        'database_host': db.get('host', 'localhost'),
                        'database_port': db.get('port', 5432),
                        'database_name': db.get('name', 'crypto_trading'),
                        'database_user': db.get('user', 'trading_user'),
                        'database_password': db.get('password', '')
                    })
                    
            # Appearance settings
            if 'appearance' in self.trading_system.config:
                app = self.trading_system.config['appearance']
                settings.update({
                    'app_theme': app.get('theme', 'System Default'),
                    'chart_theme': app.get('chart_theme', 'Dark'),
                    'font_size': app.get('font_size', 10),
                    'font_family': app.get('font_family', 'System Default'),
                    'default_timeframe': app.get('default_timeframe', '1h'),
                    'default_chart_type': app.get('default_chart_type', 'Candlestick'),
                    'show_grid': app.get('show_grid', True),
                    'show_volume': app.get('show_volume', True),
                    'remember_window_size': app.get('remember_window_size', True),
                    'save_widget_layout': app.get('save_widget_layout', True)
                })
        
        return settings
        
    def _update_controls_from_settings(self, settings):
        """Update UI controls from settings dictionary"""
        for setting_name, value in settings.items():
            if setting_name in self.controls:
                control = self.controls[setting_name]
                
                if isinstance(control, QCheckBox):
                    control.setChecked(bool(value))
                elif isinstance(control, QSpinBox):
                    control.setValue(int(value))
                elif isinstance(control, QDoubleSpinBox):
                    control.setValue(float(value))
                elif isinstance(control, QComboBox):
                    index = control.findText(str(value))
                    if index >= 0:
                        control.setCurrentIndex(index)
                elif isinstance(control, QLineEdit):
                    control.setText(str(value))
                elif isinstance(control, QTextEdit):
                    control.setText(str(value))
                    
    def _get_settings_from_controls(self):
        """Get settings from UI controls"""
        settings = {}
        
        for setting_name, control in self.controls.items():
            if isinstance(control, QCheckBox):
                settings[setting_name] = control.isChecked()
            elif isinstance(control, QSpinBox):
                settings[setting_name] = control.value()
            elif isinstance(control, QDoubleSpinBox):
                settings[setting_name] = control.value()
            elif isinstance(control, QComboBox):
                settings[setting_name] = control.currentText()
            elif isinstance(control, QLineEdit):
                settings[setting_name] = control.text()
            elif isinstance(control, QTextEdit):
                settings[setting_name] = control.toPlainText()
                
        return settings
        
    def _validate_settings(self, settings):
        """Validate settings before applying"""
        # Check risk parameters
        if settings.get('min_risk', 0) > settings.get('base_risk', 0):
            return False, "Min risk cannot be greater than base risk"
            
        if settings.get('base_risk', 0) > settings.get('max_risk', 0):
            return False, "Base risk cannot be greater than max risk"
            
        # Check database settings
        if settings.get('use_database', False) and not settings.get('database_host', ''):
            return False, "Database host cannot be empty when using database"
            
        # Check log file path
        log_file = settings.get('log_file', '')
        if log_file:
            log_dir = os.path.dirname(log_file)
            if log_dir and not os.path.exists(log_dir):
                try:
                    os.makedirs(log_dir, exist_ok=True)
                except Exception as e:
                    return False, f"Cannot create log directory: {str(e)}"
                    
        # Check models directory
        models_dir = settings.get('models_directory', '')
        if models_dir and not os.path.exists(models_dir):
            try:
                os.makedirs(models_dir, exist_ok=True)
            except Exception as e:
                return False, f"Cannot create models directory: {str(e)}"
                
        # Trading hours format (if not 24/7)
        if not settings.get('trade_24_7', True):
            trading_hours = settings.get('trading_hours', '')
            if trading_hours and not self._validate_trading_hours(trading_hours):
                return False, "Invalid trading hours format (use HH:MM-HH:MM)"
                
        # All validation passed
        return True, ""
        
    def _validate_trading_hours(self, trading_hours):
        """Validate trading hours format (HH:MM-HH:MM)"""
        import re
        pattern = r'^([0-1][0-9]|2[0-3]):([0-5][0-9])-([0-1][0-9]|2[0-3]):([0-5][0-9])$'
        return bool(re.match(pattern, trading_hours))
        
    def _apply_settings(self, settings):
        """Apply settings to the trading system"""
        try:
            # Create a new config structure
            config = {
                'general': {
                    'auto_save_settings': settings.get('auto_save_settings', True),
                    'auto_save_interval': settings.get('auto_save_interval', 5),
                    'start_minimized': settings.get('start_minimized', False),
                    'minimize_to_tray': settings.get('minimize_to_tray', False),
                    'confirm_exit': settings.get('confirm_exit', True)
                },
                'logging': {
                    'level': settings.get('log_level', 'INFO'),
                    'file': settings.get('log_file', 'logs/trading.log'),
                    'max_size_mb': settings.get('max_log_size', 10),
                    'rotation_count': settings.get('log_rotation', 5)
                },
                'performance': {
                    'worker_threads': settings.get('worker_threads', 4),
                    'ui_update_interval': settings.get('ui_update_interval', 1000),
                    'chart_update_limit': settings.get('chart_update_limit', 10),
                    'use_hardware_acceleration': settings.get('use_hardware_acceleration', True)
                },
                'trading': {
                    'mode': settings.get('trading_mode', 'Paper Trading'),
                    'auto_trading': settings.get('auto_trading', False),
                    'default_order_size': settings.get('default_order_size', 0.01),
                    'default_order_type': settings.get('default_order_type', 'Limit'),
                    'default_leverage': settings.get('default_leverage', 1),
                    'slippage_tolerance': settings.get('slippage_tolerance', 0.001),
                    'execution_delay': settings.get('execution_delay', 0.1),
                    'use_smart_execution': settings.get('use_smart_execution', True),
                    'trade_24_7': settings.get('trade_24_7', True),
                    'trading_hours': settings.get('trading_hours', '09:00-17:00'),
                    'symbols': settings.get('trading_pairs', '').strip().split('\n')
                },
                'strategies': {
                    'mode': settings.get('strategy_mode', 'Multiple Strategies'),
                    'max_active': settings.get('max_active_strategies', 5),
                    'refresh_interval': settings.get('strategy_refresh_interval', 15),
                    'auto_optimize': settings.get('auto_optimize_strategies', False),
                    'weights': {}
                },
                'risk_limits': {
                    'base_risk': settings.get('base_risk', 0.02),
                    'max_risk': settings.get('max_risk', 0.05),
                    'min_risk': settings.get('min_risk', 0.01),
                    'max_drawdown_limit': settings.get('max_drawdown_limit', 0.15),
                    'daily_loss_limit': settings.get('daily_loss_limit', 0.03),
                    'max_open_positions': settings.get('max_open_positions', 10),
                    'max_risk_per_asset': settings.get('max_risk_per_asset', 0.2),
                    'correlation_threshold': settings.get('correlation_threshold', 0.7),
                    'use_kelly': settings.get('use_kelly', False),
                    'kelly_fraction': settings.get('kelly_fraction', 0.5),
                    'volatility_adjustment': settings.get('volatility_adjustment', True),
                    'auto_hedging': settings.get('auto_hedging', False),
                    'hedge_threshold': settings.get('hedge_threshold', 0.05),
                    'hedge_size': settings.get('hedge_size', 0.5),
                    'enable_circuit_breakers': settings.get('enable_circuit_breakers', True),
                    'daily_loss_breaker': settings.get('daily_loss_breaker', 0.07),
                    'flash_crash_detection': settings.get('flash_crash_detection', True)
                },
                'ai': {
                    'enable': settings.get('enable_ai', False),
                    'confidence_threshold': settings.get('ai_confidence_threshold', 0.7),
                    'control_mode': settings.get('ai_control_mode', 'Advisory (Suggestions Only)'),
                    'use_gpu_inference': settings.get('use_gpu_inference', False),
                    'enable_order_flow_model': settings.get('enable_order_flow_model', True),
                    'enable_trade_timing_model': settings.get('enable_trade_timing_model', True),
                    'enable_trade_exit_model': settings.get('enable_trade_exit_model', True),
                    'enable_portfolio_model': settings.get('enable_portfolio_model', True),
                    'enable_risk_model': settings.get('enable_risk_model', True),
                    'auto_training': settings.get('auto_training', False),
                    'training_schedule': settings.get('training_schedule', 'Weekly'),
                    'training_time_limit': settings.get('training_time_limit', 2),
                    'use_gpu_training': settings.get('use_gpu_training', True),
                    'models_directory': settings.get('models_directory', 'models')
                },
                'data': {
                    'primary_source': settings.get('primary_data_source', 'Binance'),
                    'secondary_source': settings.get('secondary_data_source', 'None'),
                    'update_interval': settings.get('data_update_interval', 5),
                    'use_websocket': settings.get('use_websocket', True),
                    'market_data_retention': settings.get('market_data_retention', 30),
                    'order_history_retention': settings.get('order_history_retention', 90),
                    'api_keys': {
                        'binance': {
                            'key': settings.get('binance_api_key', ''),
                            'secret': settings.get('binance_api_secret', '')
                        },
                        'coinbase': {
                            'key': settings.get('coinbase_api_key', ''),
                            'secret': settings.get('coinbase_api_secret', ''),
                            'passphrase': settings.get('coinbase_passphrase', '')
                        }
                    },
                    'database': {
                        'use_database': settings.get('use_database', True),
                        'type': settings.get('database_type', 'TimescaleDB'),
                        'host': settings.get('database_host', 'localhost'),
                        'port': settings.get('database_port', 5432),
                        'name': settings.get('database_name', 'crypto_trading'),
                        'user': settings.get('database_user', 'trading_user'),
                        'password': settings.get('database_password', '')
                    }
                },
                'appearance': {
                    'theme': settings.get('app_theme', 'System Default'),
                    'chart_theme': settings.get('chart_theme', 'Dark'),
                    'font_size': settings.get('font_size', 10),
                    'font_family': settings.get('font_family', 'System Default'),
                    'default_timeframe': settings.get('default_timeframe', '1h'),
                    'default_chart_type': settings.get('default_chart_type', 'Candlestick'),
                    'show_grid': settings.get('show_grid', True),
                    'show_volume': settings.get('show_volume', True),
                    'remember_window_size': settings.get('remember_window_size', True),
                    'save_widget_layout': settings.get('save_widget_layout', True)
                }
            }
            
            # Extract strategy weights
            for key, value in settings.items():
                if key.startswith('strategy_weight_'):
                    strategy_name = key[len('strategy_weight_'):]
                    config['strategies']['weights'][strategy_name] = value
            
            # Apply config to trading system
            self.trading_system.config = config
            
            # Apply specific settings directly to components
            
            # Update risk manager settings
            if hasattr(self.trading_system, 'risk_manager'):
                risk_limits = config['risk_limits']
                self.trading_system.risk_manager.set_risk_limits(risk_limits)
                
            # Update database settings if needed
            if hasattr(self.trading_system, 'db'):
                # Handle database reconnect if connection settings have changed
                # This would be implemented in the actual trading system
                pass
                
            # Save the config to disk
            self.trading_system.save_config()
            
            return True
            
        except Exception as e:
            logging.error(f"Error applying settings: {e}")
            return False
        
    def _on_save(self):
        """Handle save button click"""
        # Get current settings from controls
        settings = self._get_settings_from_controls()
        
        # Validate settings
        valid, message = self._validate_settings(settings)
        if not valid:
            QMessageBox.warning(self, "Validation Error", message)
            return
            
        # Apply settings
        if self._apply_settings(settings):
            # Update current settings
            self.current_settings = settings
            self.original_settings = settings.copy()
            
            # Emit settings changed signal
            self.settings_changed.emit(settings)
            
            # Close dialog
            self.accept()
        else:
            QMessageBox.critical(self, "Settings Error", "Failed to apply settings")
        
    def _on_apply(self):
        """Handle apply button click"""
        # Get current settings from controls
        settings = self._get_settings_from_controls()
        
        # Validate settings
        valid, message = self._validate_settings(settings)
        if not valid:
            QMessageBox.warning(self, "Validation Error", message)
            return
            
        # Apply settings
        if self._apply_settings(settings):
            # Update current settings
            self.current_settings = settings
            self.original_settings = settings.copy()
            
            # Emit settings changed signal
            self.settings_changed.emit(settings)
            
            QMessageBox.information(self, "Settings Applied", "Settings have been applied successfully")
        else:
            QMessageBox.critical(self, "Settings Error", "Failed to apply settings")
        
    def _on_restore_defaults(self):
        """Handle restore defaults button click"""
        # Confirm with user
        reply = QMessageBox.question(
            self, "Restore Defaults", 
            "Are you sure you want to restore all settings to default values?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )
        
        if reply != QMessageBox.Yes:
            return
            
        # Get default settings
        default_settings = self._get_default_settings()
        
        # Update controls
        self._update_controls_from_settings(default_settings)
        
        # Update current settings
        self.current_settings = default_settings
        
    def _get_default_settings(self):
        """Get default settings"""
        # Default settings for all categories
        return {
            # General settings
            'auto_save_settings': True,
            'auto_save_interval': 5,
            'start_minimized': False,
            'minimize_to_tray': False,
            'confirm_exit': True,
            
            # Logging settings
            'log_level': 'INFO',
            'log_file': 'logs/trading.log',
            'max_log_size': 10,
            'log_rotation': 5,
            
            # Performance settings
            'worker_threads': 4,
            'ui_update_interval': 1000,
            'chart_update_limit': 10,
            'use_hardware_acceleration': True,
            
            # Trading settings
            'trading_mode': 'Paper Trading',
            'auto_trading': False,
            'default_order_size': 0.01,
            'default_order_type': 'Limit',
            'default_leverage': 1,
            'slippage_tolerance': 0.001,
            'execution_delay': 0.1,
            'use_smart_execution': True,
            'trade_24_7': True,
            'trading_hours': '09:00-17:00',
            'trading_pairs': 'BTC/USDT\nETH/USDT',
            
            # Strategy settings
            'strategy_mode': 'Multiple Strategies',
            'max_active_strategies': 5,
            'strategy_refresh_interval': 15,
            'auto_optimize_strategies': False,
            'strategy_weight_trend_following': 0.2,
            'strategy_weight_mean_reversion': 0.2,
            'strategy_weight_breakout': 0.2,
            'strategy_weight_multi_timeframe': 0.2,
            'strategy_weight_volume_based': 0.2,
            
            # Strategy parameters
            'tf_short_window': 10,
            'tf_long_window': 50,
            'mr_window': 20,
            'mr_std_dev': 2.0,
            
            # Risk settings
            'base_risk': 0.02,
            'max_risk': 0.05,
            'min_risk': 0.01,
            'max_drawdown_limit': 0.15,
            'daily_loss_limit': 0.03,
            'max_open_positions': 10,
            'max_risk_per_asset': 0.2,
            'correlation_threshold': 0.7,
            'use_kelly': False,
            'kelly_fraction': 0.5,
            'volatility_adjustment': True,
            'auto_hedging': False,
            'hedge_threshold': 0.05,
            'hedge_size': 0.5,
            'enable_circuit_breakers': True,
            'daily_loss_breaker': 0.07,
            'flash_crash_detection': True,
            
            # AI settings
            'enable_ai': False,
            'ai_confidence_threshold': 0.7,
            'ai_control_mode': 'Advisory (Suggestions Only)',
            'use_gpu_inference': False,
            'enable_order_flow_model': True,
            'enable_trade_timing_model': True,
            'enable_trade_exit_model': True,
            'enable_portfolio_model': True,
            'enable_risk_model': True,
            'auto_training': False,
            'training_schedule': 'Weekly',
            'training_time_limit': 2,
            'use_gpu_training': True,
            'models_directory': 'models',
            
            # Data API settings
            'primary_data_source': 'Binance',
            'secondary_data_source': 'None',
            'data_update_interval': 5,
            'use_websocket': True,
            'market_data_retention': 30,
            'order_history_retention': 90,
            'binance_api_key': '',
            'binance_api_secret': '',
            'coinbase_api_key': '',
            'coinbase_api_secret': '',
            'coinbase_passphrase': '',
            
            # Database settings
            'use_database': True,
            'database_type': 'TimescaleDB',
            'database_host': 'localhost',
            'database_port': 5432,
            'database_name': 'crypto_trading',
            'database_user': 'trading_user',
            'database_password': '',
            
            # Appearance settings
            'app_theme': 'System Default',
            'chart_theme': 'Dark',
            'font_size': 10,
            'font_family': 'System Default',
            'default_timeframe': '1h',
            'default_chart_type': 'Candlestick',
            'show_grid': True,
            'show_volume': True,
            'remember_window_size': True,
            'save_widget_layout': True
        }
        
    def closeEvent(self, event):
        """Handle dialog close event"""
        # Check if settings have changed
        if self.current_settings != self.original_settings:
            reply = QMessageBox.question(
                self, "Unsaved Changes", 
                "You have unsaved changes. Do you want to save them?",
                QMessageBox.Save | QMessageBox.Discard | QMessageBox.Cancel,
                QMessageBox.Save
            )
            
            if reply == QMessageBox.Save:
                self._on_save()
                event.accept()
            elif reply == QMessageBox.Cancel:
                event.ignore()
            else:
                event.accept()
        else:
            event.accept()
            
    def sizeHint(self):
        """Suggested size for the dialog"""
        return QSize(900, 700)
