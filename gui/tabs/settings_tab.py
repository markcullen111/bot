# settings_tab.py

import os
import json
import logging
import copy
from typing import Dict, Any, List, Optional, Tuple, Callable

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGridLayout, QLabel, 
    QLineEdit, QComboBox, QCheckBox, QSpinBox, QDoubleSpinBox,
    QPushButton, QTabWidget, QScrollArea, QGroupBox, QFormLayout,
    QFileDialog, QMessageBox, QFrame, QSizePolicy, QSlider,
    QRadioButton, QButtonGroup, QTextEdit, QToolButton, QMenu
)
from PyQt5.QtCore import Qt, QSettings, pyqtSignal, QTimer
from PyQt5.QtGui import QIcon, QFont, QColor, QPalette

class SettingsTab(QWidget):
    """
    Settings management tab for trading system configuration.
    Allows users to configure all aspects of the trading system.
    """
    
    # Signal emitted when settings are changed
    settings_changed = pyqtSignal(dict)
    
    def __init__(self, trading_system, parent=None):
        super().__init__(parent)
        
        self.trading_system = trading_system
        
        # Initialize local settings storage
        self.settings = {}
        self.original_settings = {}  # For change tracking
        self.is_dirty = False
        
        # Load settings from system
        self._load_settings()
        
        # Setup UI
        self._setup_ui()
        
        # Auto-save timer
        self.autosave_timer = QTimer(self)
        self.autosave_timer.timeout.connect(self._autosave_settings)
        self.autosave_timer.start(30000)  # Auto-save every 30 seconds if dirty
        
    def _load_settings(self):
        """Load settings from trading system"""
        try:
            # Get settings from trading system config
            if hasattr(self.trading_system, 'config'):
                self.settings = copy.deepcopy(self.trading_system.config)
            else:
                self.settings = {}
                
            # If settings are empty, load defaults
            if not self.settings:
                self._load_default_settings()
                
            # Make a copy of original settings
            self.original_settings = copy.deepcopy(self.settings)
            self.is_dirty = False
            
            logging.info("Settings loaded successfully")
            
        except Exception as e:
            logging.error(f"Error loading settings: {e}")
            self._load_default_settings()
            
    def _load_default_settings(self):
        """Load default settings"""
        self.settings = {
            "general": {
                "theme": "system",
                "log_level": "INFO",
                "auto_save": True,
                "startup_check": True,
                "confirm_exit": True
            },
            "trading": {
                "trading_enabled": False,
                "test_mode": True,
                "symbols": ["BTC/USDT", "ETH/USDT", "SOL/USDT"],
                "default_timeframe": "1h",
                "max_open_positions": 5,
                "base_order_size": 0.01,
                "order_type": "market"
            },
            "risk_limits": {
                "max_drawdown_limit": 10,
                "position_limit": 1000,
                "stop_loss": 5,
                "max_risk_exposure": 20,
                "leverage_limit": 3
            },
            "strategies": {
                "active_strategies": ["trend_following", "mean_reversion"],
                "strategy_weights": {
                    "trend_following": 0.5,
                    "mean_reversion": 0.5
                },
                "optimization_interval": 24  # Hours
            },
            "database": {
                "use_mock_db": True,
                "db_host": "localhost",
                "db_port": 5432,
                "db_name": "crypto_trading",
                "db_user": "trading_user"
            },
            "ai_settings": {
                "use_ai": True,
                "model_confidence_threshold": 0.7,
                "learning_rate": 0.001,
                "auto_update_models": False,
                "exploration_rate": 0.1,
                "use_reinforcement_learning": True
            },
            "connection": {
                "api_key": "",
                "api_secret": "",
                "exchange": "binance",
                "testnet": True,
                "connection_timeout": 30
            },
            "notifications": {
                "enable_notifications": True,
                "trade_notifications": True,
                "error_notifications": True,
                "daily_summary": True,
                "email_notifications": False,
                "email_address": ""
            },
            "ui": {
                "chart_style": "candles",
                "default_indicators": ["sma", "volume", "rsi"],
                "refresh_interval": 5,  # Seconds
                "show_orderbook": True,
                "dark_mode": False,
                "font_size": "medium"
            }
        }
        
        # Make a copy of original settings
        self.original_settings = copy.deepcopy(self.settings)
        self.is_dirty = False
        
    def _setup_ui(self):
        """Set up the settings tab UI"""
        main_layout = QVBoxLayout(self)
        
        # Create tabs for different settings categories
        tabs = QTabWidget()
        
        # Create and add category tabs
        self._add_general_tab(tabs)
        self._add_trading_tab(tabs)
        self._add_risk_tab(tabs)
        self._add_strategies_tab(tabs)
        self._add_database_tab(tabs)
        self._add_ai_tab(tabs)
        self._add_connection_tab(tabs)
        self._add_notifications_tab(tabs)
        self._add_ui_tab(tabs)
        
        main_layout.addWidget(tabs)
        
        # Add bottom action buttons
        actions_layout = QHBoxLayout()
        
        # Save button
        self.save_button = QPushButton("Save Settings")
        self.save_button.setIcon(QIcon.fromTheme("document-save"))
        self.save_button.clicked.connect(self._save_settings)
        self.save_button.setEnabled(False)  # Disabled until changes are made
        actions_layout.addWidget(self.save_button)
        
        # Reset button
        self.reset_button = QPushButton("Reset to Defaults")
        self.reset_button.setIcon(QIcon.fromTheme("edit-clear"))
        self.reset_button.clicked.connect(self._reset_to_defaults)
        actions_layout.addWidget(self.reset_button)
        
        # Import/Export buttons
        self.import_button = QPushButton("Import Settings")
        self.import_button.setIcon(QIcon.fromTheme("document-open"))
        self.import_button.clicked.connect(self._import_settings)
        actions_layout.addWidget(self.import_button)
        
        self.export_button = QPushButton("Export Settings")
        self.export_button.setIcon(QIcon.fromTheme("document-save-as"))
        self.export_button.clicked.connect(self._export_settings)
        actions_layout.addWidget(self.export_button)
        
        main_layout.addLayout(actions_layout)
        
        # Status label
        self.status_label = QLabel("")
        self.status_label.setAlignment(Qt.AlignCenter)
        main_layout.addWidget(self.status_label)
        
    def _add_general_tab(self, tabs):
        """Add general settings tab"""
        general_tab = QWidget()
        layout = QVBoxLayout(general_tab)
        
        # Create form layout for settings
        form = QFormLayout()
        
        # Theme setting
        self.theme_combo = QComboBox()
        self.theme_combo.addItems(["system", "light", "dark"])
        current_theme = self.settings.get("general", {}).get("theme", "system")
        self.theme_combo.setCurrentText(current_theme)
        self.theme_combo.currentTextChanged.connect(
            lambda text: self._update_setting("general", "theme", text)
        )
        form.addRow("Theme:", self.theme_combo)
        
        # Log level
        self.log_level_combo = QComboBox()
        self.log_level_combo.addItems(["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"])
        current_log_level = self.settings.get("general", {}).get("log_level", "INFO")
        self.log_level_combo.setCurrentText(current_log_level)
        self.log_level_combo.currentTextChanged.connect(
            lambda text: self._update_setting("general", "log_level", text)
        )
        form.addRow("Log Level:", self.log_level_combo)
        
        # Auto save
        self.auto_save_check = QCheckBox()
        self.auto_save_check.setChecked(self.settings.get("general", {}).get("auto_save", True))
        self.auto_save_check.stateChanged.connect(
            lambda state: self._update_setting("general", "auto_save", state == Qt.Checked)
        )
        form.addRow("Auto Save:", self.auto_save_check)
        
        # Startup check
        self.startup_check = QCheckBox()
        self.startup_check.setChecked(self.settings.get("general", {}).get("startup_check", True))
        self.startup_check.stateChanged.connect(
            lambda state: self._update_setting("general", "startup_check", state == Qt.Checked)
        )
        form.addRow("System Check on Startup:", self.startup_check)
        
        # Confirm exit
        self.confirm_exit_check = QCheckBox()
        self.confirm_exit_check.setChecked(self.settings.get("general", {}).get("confirm_exit", True))
        self.confirm_exit_check.stateChanged.connect(
            lambda state: self._update_setting("general", "confirm_exit", state == Qt.Checked)
        )
        form.addRow("Confirm on Exit:", self.confirm_exit_check)
        
        # Add form to layout
        layout.addLayout(form)
        
        # Add spacer to push settings to top
        layout.addStretch()
        
        # Add tab
        tabs.addTab(general_tab, "General")
        
    def _add_trading_tab(self, tabs):
        """Add trading settings tab"""
        trading_tab = QScrollArea()
        trading_tab.setWidgetResizable(True)
        
        trading_content = QWidget()
        layout = QVBoxLayout(trading_content)
        
        # Enable trading
        trading_group = QGroupBox("Trading Settings")
        trading_form = QFormLayout(trading_group)
        
        # Trading enabled
        self.trading_enabled_check = QCheckBox()
        self.trading_enabled_check.setChecked(self.settings.get("trading", {}).get("trading_enabled", False))
        self.trading_enabled_check.stateChanged.connect(
            lambda state: self._update_setting("trading", "trading_enabled", state == Qt.Checked)
        )
        trading_form.addRow("Enable Trading:", self.trading_enabled_check)
        
        # Test mode
        self.test_mode_check = QCheckBox()
        self.test_mode_check.setChecked(self.settings.get("trading", {}).get("test_mode", True))
        self.test_mode_check.stateChanged.connect(
            lambda state: self._update_setting("trading", "test_mode", state == Qt.Checked)
        )
        trading_form.addRow("Test Mode (Paper Trading):", self.test_mode_check)
        
        # Trading symbols
        symbols_layout = QVBoxLayout()
        
        # Current symbols list
        self.symbols_list = QTextEdit()
        symbols = self.settings.get("trading", {}).get("symbols", ["BTC/USDT"])
        self.symbols_list.setPlainText("\n".join(symbols))
        self.symbols_list.setMaximumHeight(100)
        self.symbols_list.textChanged.connect(self._update_symbols)
        symbols_layout.addWidget(self.symbols_list)
        
        # Help text
        help_label = QLabel("Enter one trading pair per line (e.g., BTC/USDT)")
        help_label.setStyleSheet("color: gray; font-size: 10px;")
        symbols_layout.addWidget(help_label)
        
        trading_form.addRow("Trading Symbols:", symbols_layout)
        
        # Default timeframe
        self.timeframe_combo = QComboBox()
        self.timeframe_combo.addItems(["1m", "5m", "15m", "30m", "1h", "4h", "1d"])
        current_timeframe = self.settings.get("trading", {}).get("default_timeframe", "1h")
        self.timeframe_combo.setCurrentText(current_timeframe)
        self.timeframe_combo.currentTextChanged.connect(
            lambda text: self._update_setting("trading", "default_timeframe", text)
        )
        trading_form.addRow("Default Timeframe:", self.timeframe_combo)
        
        # Max open positions
        self.max_positions_spin = QSpinBox()
        self.max_positions_spin.setRange(1, 100)
        self.max_positions_spin.setValue(self.settings.get("trading", {}).get("max_open_positions", 5))
        self.max_positions_spin.valueChanged.connect(
            lambda value: self._update_setting("trading", "max_open_positions", value)
        )
        trading_form.addRow("Max Open Positions:", self.max_positions_spin)
        
        # Base order size
        self.order_size_spin = QDoubleSpinBox()
        self.order_size_spin.setRange(0.001, 100)
        self.order_size_spin.setDecimals(4)
        self.order_size_spin.setSingleStep(0.001)
        self.order_size_spin.setValue(self.settings.get("trading", {}).get("base_order_size", 0.01))
        self.order_size_spin.valueChanged.connect(
            lambda value: self._update_setting("trading", "base_order_size", value)
        )
        trading_form.addRow("Base Order Size:", self.order_size_spin)
        
        # Order type
        self.order_type_combo = QComboBox()
        self.order_type_combo.addItems(["market", "limit"])
        current_order_type = self.settings.get("trading", {}).get("order_type", "market")
        self.order_type_combo.setCurrentText(current_order_type)
        self.order_type_combo.currentTextChanged.connect(
            lambda text: self._update_setting("trading", "order_type", text)
        )
        trading_form.addRow("Default Order Type:", self.order_type_combo)
        
        # Add trading group to layout
        layout.addWidget(trading_group)
        
        # Trading session settings
        session_group = QGroupBox("Trading Sessions")
        session_form = QFormLayout(session_group)
        
        # 24/7 trading or scheduled
        self.trading_schedule_check = QCheckBox("Use Trading Schedule")
        self.trading_schedule_check.setChecked(self.settings.get("trading", {}).get("use_schedule", False))
        self.trading_schedule_check.stateChanged.connect(
            lambda state: self._update_setting("trading", "use_schedule", state == Qt.Checked)
        )
        session_form.addRow(self.trading_schedule_check)
        
        # Trading hours (simplified for this example)
        trading_hours_layout = QHBoxLayout()
        
        self.trading_start_spin = QSpinBox()
        self.trading_start_spin.setRange(0, 23)
        self.trading_start_spin.setValue(self.settings.get("trading", {}).get("trading_start_hour", 9))
        self.trading_start_spin.valueChanged.connect(
            lambda value: self._update_setting("trading", "trading_start_hour", value)
        )
        
        self.trading_end_spin = QSpinBox()
        self.trading_end_spin.setRange(0, 23)
        self.trading_end_spin.setValue(self.settings.get("trading", {}).get("trading_end_hour", 17))
        self.trading_end_spin.valueChanged.connect(
            lambda value: self._update_setting("trading", "trading_end_hour", value)
        )
        
        trading_hours_layout.addWidget(QLabel("From:"))
        trading_hours_layout.addWidget(self.trading_start_spin)
        trading_hours_layout.addWidget(QLabel(":00  To:"))
        trading_hours_layout.addWidget(self.trading_end_spin)
        trading_hours_layout.addWidget(QLabel(":00"))
        
        session_form.addRow("Trading Hours:", trading_hours_layout)
        
        # Add session group to layout
        layout.addWidget(session_group)
        
        # Additional trading settings
        advanced_group = QGroupBox("Advanced Trading Settings")
        advanced_form = QFormLayout(advanced_group)
        
        # Slippage tolerance
        self.slippage_spin = QDoubleSpinBox()
        self.slippage_spin.setRange(0, 5)
        self.slippage_spin.setDecimals(2)
        self.slippage_spin.setSingleStep(0.01)
        self.slippage_spin.setSuffix("%")
        self.slippage_spin.setValue(self.settings.get("trading", {}).get("slippage_tolerance", 0.2))
        self.slippage_spin.valueChanged.connect(
            lambda value: self._update_setting("trading", "slippage_tolerance", value)
        )
        advanced_form.addRow("Slippage Tolerance:", self.slippage_spin)
        
        # Execution delay
        self.execution_delay_spin = QDoubleSpinBox()
        self.execution_delay_spin.setRange(0, 5)
        self.execution_delay_spin.setDecimals(2)
        self.execution_delay_spin.setSingleStep(0.01)
        self.execution_delay_spin.setSuffix(" sec")
        self.execution_delay_spin.setValue(self.settings.get("trading", {}).get("execution_delay", 0.1))
        self.execution_delay_spin.valueChanged.connect(
            lambda value: self._update_setting("trading", "execution_delay", value)
        )
        advanced_form.addRow("Execution Delay:", self.execution_delay_spin)
        
        # Add advanced group to layout
        layout.addWidget(advanced_group)
        
        # Add spacer
        layout.addStretch()
        
        # Set content widget
        trading_tab.setWidget(trading_content)
        
        # Add tab
        tabs.addTab(trading_tab, "Trading")
        
    def _update_symbols(self):
        """Update the trading symbols from the text edit"""
        # Split text by newlines and remove empty lines
        symbols_text = self.symbols_list.toPlainText()
        symbols = [s.strip() for s in symbols_text.split('\n') if s.strip()]
        
        # Update settings
        self._update_setting("trading", "symbols", symbols)
        
    def _add_risk_tab(self, tabs):
        """Add risk management settings tab"""
        risk_tab = QScrollArea()
        risk_tab.setWidgetResizable(True)
        
        risk_content = QWidget()
        layout = QVBoxLayout(risk_content)
        
        # Risk limits group
        limits_group = QGroupBox("Risk Limits")
        limits_form = QFormLayout(limits_group)
        
        # Max drawdown limit
        self.drawdown_spin = QDoubleSpinBox()
        self.drawdown_spin.setRange(1, 50)
        self.drawdown_spin.setDecimals(1)
        self.drawdown_spin.setSingleStep(0.5)
        self.drawdown_spin.setSuffix("%")
        self.drawdown_spin.setValue(self.settings.get("risk_limits", {}).get("max_drawdown_limit", 10))
        self.drawdown_spin.valueChanged.connect(
            lambda value: self._update_setting("risk_limits", "max_drawdown_limit", value)
        )
        limits_form.addRow("Max Drawdown Limit:", self.drawdown_spin)
        
        # Position limit
        self.position_limit_spin = QSpinBox()
        self.position_limit_spin.setRange(100, 10000)
        self.position_limit_spin.setSingleStep(100)
        self.position_limit_spin.setValue(self.settings.get("risk_limits", {}).get("position_limit", 1000))
        self.position_limit_spin.valueChanged.connect(
            lambda value: self._update_setting("risk_limits", "position_limit", value)
        )
        limits_form.addRow("Position Limit:", self.position_limit_spin)
        
        # Stop loss
        self.stop_loss_spin = QDoubleSpinBox()
        self.stop_loss_spin.setRange(1, 20)
        self.stop_loss_spin.setDecimals(1)
        self.stop_loss_spin.setSingleStep(0.5)
        self.stop_loss_spin.setSuffix("%")
        self.stop_loss_spin.setValue(self.settings.get("risk_limits", {}).get("stop_loss", 5))
        self.stop_loss_spin.valueChanged.connect(
            lambda value: self._update_setting("risk_limits", "stop_loss", value)
        )
        limits_form.addRow("Default Stop Loss:", self.stop_loss_spin)
        
        # Max risk exposure
        self.risk_exposure_spin = QDoubleSpinBox()
        self.risk_exposure_spin.setRange(5, 50)
        self.risk_exposure_spin.setDecimals(1)
        self.risk_exposure_spin.setSingleStep(1)
        self.risk_exposure_spin.setSuffix("%")
        self.risk_exposure_spin.setValue(self.settings.get("risk_limits", {}).get("max_risk_exposure", 20))
        self.risk_exposure_spin.valueChanged.connect(
            lambda value: self._update_setting("risk_limits", "max_risk_exposure", value)
        )
        limits_form.addRow("Max Risk Exposure:", self.risk_exposure_spin)
        
        # Leverage limit
        self.leverage_spin = QDoubleSpinBox()
        self.leverage_spin.setRange(1, 10)
        self.leverage_spin.setDecimals(1)
        self.leverage_spin.setSingleStep(0.5)
        self.leverage_spin.setValue(self.settings.get("risk_limits", {}).get("leverage_limit", 3))
        self.leverage_spin.valueChanged.connect(
            lambda value: self._update_setting("risk_limits", "leverage_limit", value)
        )
        limits_form.addRow("Max Leverage:", self.leverage_spin)
        
        # Add limits group to layout
        layout.addWidget(limits_group)
        
        # Position sizing group
        sizing_group = QGroupBox("Position Sizing")
        sizing_form = QFormLayout(sizing_group)
        
        # Base risk per trade
        self.base_risk_spin = QDoubleSpinBox()
        self.base_risk_spin.setRange(0.1, 5)
        self.base_risk_spin.setDecimals(2)
        self.base_risk_spin.setSingleStep(0.1)
        self.base_risk_spin.setSuffix("%")
        self.base_risk_spin.setValue(self.settings.get("risk_limits", {}).get("base_risk", 2))
        self.base_risk_spin.valueChanged.connect(
            lambda value: self._update_setting("risk_limits", "base_risk", value)
        )
        sizing_form.addRow("Base Risk Per Trade:", self.base_risk_spin)
        
        # Max risk per trade
        self.max_risk_spin = QDoubleSpinBox()
        self.max_risk_spin.setRange(0.5, 10)
        self.max_risk_spin.setDecimals(2)
        self.max_risk_spin.setSingleStep(0.1)
        self.max_risk_spin.setSuffix("%")
        self.max_risk_spin.setValue(self.settings.get("risk_limits", {}).get("max_risk", 5))
        self.max_risk_spin.valueChanged.connect(
            lambda value: self._update_setting("risk_limits", "max_risk", value)
        )
        sizing_form.addRow("Max Risk Per Trade:", self.max_risk_spin)
        
        # Dynamic sizing
        self.dynamic_sizing_check = QCheckBox()
        self.dynamic_sizing_check.setChecked(self.settings.get("risk_limits", {}).get("dynamic_sizing", True))
        self.dynamic_sizing_check.stateChanged.connect(
            lambda state: self._update_setting("risk_limits", "dynamic_sizing", state == Qt.Checked)
        )
        sizing_form.addRow("Dynamic Position Sizing:", self.dynamic_sizing_check)
        
        # Kelly criterion
        self.kelly_check = QCheckBox()
        self.kelly_check.setChecked(self.settings.get("risk_limits", {}).get("use_kelly", False))
        self.kelly_check.stateChanged.connect(
            lambda state: self._update_setting("risk_limits", "use_kelly", state == Qt.Checked)
        )
        sizing_form.addRow("Use Kelly Criterion:", self.kelly_check)
        
        # Kelly fraction
        self.kelly_fraction_spin = QDoubleSpinBox()
        self.kelly_fraction_spin.setRange(0.1, 1.0)
        self.kelly_fraction_spin.setDecimals(1)
        self.kelly_fraction_spin.setSingleStep(0.1)
        self.kelly_fraction_spin.setValue(self.settings.get("risk_limits", {}).get("kelly_fraction", 0.5))
        self.kelly_fraction_spin.valueChanged.connect(
            lambda value: self._update_setting("risk_limits", "kelly_fraction", value)
        )
        sizing_form.addRow("Kelly Fraction:", self.kelly_fraction_spin)
        
        # Add sizing group to layout
        layout.addWidget(sizing_group)
        
        # Risk controls group
        controls_group = QGroupBox("Risk Controls")
        controls_form = QFormLayout(controls_group)
        
        # Circuit breakers
        self.circuit_breakers_check = QCheckBox()
        self.circuit_breakers_check.setChecked(self.settings.get("risk_limits", {}).get("circuit_breakers", True))
        self.circuit_breakers_check.stateChanged.connect(
            lambda state: self._update_setting("risk_limits", "circuit_breakers", state == Qt.Checked)
        )
        controls_form.addRow("Enable Circuit Breakers:", self.circuit_breakers_check)
        
        # Daily loss limit
        self.daily_loss_spin = QDoubleSpinBox()
        self.daily_loss_spin.setRange(1, 20)
        self.daily_loss_spin.setDecimals(1)
        self.daily_loss_spin.setSingleStep(0.5)
        self.daily_loss_spin.setSuffix("%")
        self.daily_loss_spin.setValue(self.settings.get("risk_limits", {}).get("daily_loss_limit", 5))
        self.daily_loss_spin.valueChanged.connect(
            lambda value: self._update_setting("risk_limits", "daily_loss_limit", value)
        )
        controls_form.addRow("Daily Loss Limit:", self.daily_loss_spin)
        
        # Volatility adjustment
        self.volatility_check = QCheckBox()
        self.volatility_check.setChecked(self.settings.get("risk_limits", {}).get("volatility_adjustment", True))
        self.volatility_check.stateChanged.connect(
            lambda state: self._update_setting("risk_limits", "volatility_adjustment", state == Qt.Checked)
        )
        controls_form.addRow("Adjust for Volatility:", self.volatility_check)
        
        # Auto deleveraging
        self.deleveraging_check = QCheckBox()
        self.deleveraging_check.setChecked(self.settings.get("risk_limits", {}).get("auto_deleveraging", True))
        self.deleveraging_check.stateChanged.connect(
            lambda state: self._update_setting("risk_limits", "auto_deleveraging", state == Qt.Checked)
        )
        controls_form.addRow("Auto Deleveraging:", self.deleveraging_check)
        
        # Hedging mode
        self.hedging_check = QCheckBox()
        self.hedging_check.setChecked(self.settings.get("risk_limits", {}).get("hedging_mode", False))
        self.hedging_check.stateChanged.connect(
            lambda state: self._update_setting("risk_limits", "hedging_mode", state == Qt.Checked)
        )
        controls_form.addRow("Enable Hedging:", self.hedging_check)
        
        # Add controls group to layout
        layout.addWidget(controls_group)
        
        # Add spacer
        layout.addStretch()
        
        # Set content widget
        risk_tab.setWidget(risk_content)
        
        # Add tab
        tabs.addTab(risk_tab, "Risk Management")
        
    def _add_strategies_tab(self, tabs):
        """Add strategies settings tab"""
        strategies_tab = QScrollArea()
        strategies_tab.setWidgetResizable(True)
        
        strategies_content = QWidget()
        layout = QVBoxLayout(strategies_content)
        
        # Active strategies group
        active_group = QGroupBox("Active Strategies")
        active_layout = QVBoxLayout(active_group)
        
        # Strategy checkboxes
        strategies = ["trend_following", "mean_reversion", "breakout", "multi_timeframe", "volume_based"]
        active_strategies = self.settings.get("strategies", {}).get("active_strategies", [])
        
        self.strategy_checkboxes = {}
        for strategy in strategies:
            checkbox = QCheckBox(strategy.replace('_', ' ').title())
            checkbox.setChecked(strategy in active_strategies)
            checkbox.stateChanged.connect(
                lambda state, s=strategy: self._update_active_strategies(s, state == Qt.Checked)
            )
            active_layout.addWidget(checkbox)
            self.strategy_checkboxes[strategy] = checkbox
            
        # Add active group to layout
        layout.addWidget(active_group)
        
        # Strategy weights group
        weights_group = QGroupBox("Strategy Weights")
        weights_form = QFormLayout(weights_group)
        
        # Create sliders for strategy weights
        self.weight_sliders = {}
        strategy_weights = self.settings.get("strategies", {}).get("strategy_weights", {})
        
        for strategy in strategies:
            slider = QSlider(Qt.Horizontal)
            slider.setRange(0, 100)
            slider.setValue(int(strategy_weights.get(strategy, 50) * 100))
            
            # Display current value
            value_label = QLabel(f"{slider.value() / 100:.2f}")
            slider.valueChanged.connect(
                lambda value, label=value_label, s=strategy: (
                    label.setText(f"{value / 100:.2f}"),
                    self._update_strategy_weight(s, value / 100)
                )
            )
            
            slider_layout = QHBoxLayout()
            slider_layout.addWidget(slider)
            slider_layout.addWidget(value_label)
            
            weights_form.addRow(f"{strategy.replace('_', ' ').title()}:", slider_layout)
            self.weight_sliders[strategy] = slider
            
        # Add weights group to layout
        layout.addWidget(weights_group)
        
        # Optimization settings group
        optimization_group = QGroupBox("Optimization Settings")
        optimization_form = QFormLayout(optimization_group)
        
        # Optimization interval
        self.optimization_interval_spin = QSpinBox()
        self.optimization_interval_spin.setRange(1, 168)  # 1 hour to 7 days
        self.optimization_interval_spin.setSuffix(" hours")
        self.optimization_interval_spin.setValue(self.settings.get("strategies", {}).get("optimization_interval", 24))
        self.optimization_interval_spin.valueChanged.connect(
            lambda value: self._update_setting("strategies", "optimization_interval", value)
        )
        optimization_form.addRow("Optimization Interval:", self.optimization_interval_spin)
        
        # Auto-optimization
        self.auto_optimize_check = QCheckBox()
        self.auto_optimize_check.setChecked(self.settings.get("strategies", {}).get("auto_optimize", True))
        self.auto_optimize_check.stateChanged.connect(
            lambda state: self._update_setting("strategies", "auto_optimize", state == Qt.Checked)
        )
        optimization_form.addRow("Auto-optimize Strategies:", self.auto_optimize_check)
        
        # Performance metric for optimization
        self.metric_combo = QComboBox()
        self.metric_combo.addItems(["sharpe", "sortino", "total_return", "win_rate", "profit_factor"])
        current_metric = self.settings.get("strategies", {}).get("optimization_metric", "sharpe")
        self.metric_combo.setCurrentText(current_metric)
        self.metric_combo.currentTextChanged.connect(
            lambda text: self._update_setting("strategies", "optimization_metric", text)
        )
        optimization_form.addRow("Optimization Metric:", self.metric_combo)
        
        # Training data period
        self.training_period_spin = QSpinBox()
        self.training_period_spin.setRange(7, 365)
        self.training_period_spin.setSuffix(" days")
        self.training_period_spin.setValue(self.settings.get("strategies", {}).get("training_period", 90))
        self.training_period_spin.valueChanged.connect(
            lambda value: self._update_setting("strategies", "training_period", value)
        )
        optimization_form.addRow("Training Data Period:", self.training_period_spin)
        
        # Add optimization group to layout
        layout.addWidget(optimization_group)
        
        # Advanced strategy settings
        advanced_group = QGroupBox("Advanced Strategy Settings")
        advanced_form = QFormLayout(advanced_group)
        
        # Combined strategy mode
        self.combined_mode_combo = QComboBox()
        self.combined_mode_combo.addItems(["weighted", "voting", "ensemble", "best_performer"])
        current_mode = self.settings.get("strategies", {}).get("combined_mode", "weighted")
        self.combined_mode_combo.setCurrentText(current_mode)
        self.combined_mode_combo.currentTextChanged.connect(
            lambda text: self._update_setting("strategies", "combined_mode", text)
        )
        advanced_form.addRow("Combined Strategy Mode:", self.combined_mode_combo)
        
        # Strategy switching threshold
        self.switch_threshold_spin = QDoubleSpinBox()
        self.switch_threshold_spin.setRange(0.1, 5.0)
        self.switch_threshold_spin.setDecimals(2)
        self.switch_threshold_spin.setSingleStep(0.1)
        self.switch_threshold_spin.setValue(self.settings.get("strategies", {}).get("switch_threshold", 1.0))
        self.switch_threshold_spin.valueChanged.connect(
            lambda value: self._update_setting("strategies", "switch_threshold", value)
        )
        advanced_form.addRow("Strategy Switching Threshold:", self.switch_threshold_spin)
        
        # Add advanced group to layout
        layout.addWidget(advanced_group)
        
        # Add spacer
        layout.addStretch()
        
        # Set content widget
        strategies_tab.setWidget(strategies_content)
        
        # Add tab
        tabs.addTab(strategies_tab, "Strategies")
        
    def _update_active_strategies(self, strategy, active):
        """Update the list of active strategies"""
        strategies = self.settings.get("strategies", {}).get("active_strategies", [])
        
        if active and strategy not in strategies:
            strategies.append(strategy)
        elif not active and strategy in strategies:
            strategies.remove(strategy)
            
        self._update_setting("strategies", "active_strategies", strategies)
        
    def _update_strategy_weight(self, strategy, weight):
        """Update the weight of a strategy"""
        weights = self.settings.get("strategies", {}).get("strategy_weights", {})
        weights[strategy] = weight
        self._update_setting("strategies", "strategy_weights", weights)
        
    def _add_database_tab(self, tabs):
        """Add database settings tab"""
        db_tab = QWidget()
        layout = QVBoxLayout(db_tab)
        
        # Database type group
        type_group = QGroupBox("Database Type")
        type_layout = QVBoxLayout(type_group)
        
        # Mock database option
        self.mock_db_check = QCheckBox("Use Mock Database (for development/testing)")
        self.mock_db_check.setChecked(self.settings.get("database", {}).get("use_mock_db", True))
        self.mock_db_check.stateChanged.connect(
            lambda state: self._update_setting("database", "use_mock_db", state == Qt.Checked)
        )
        type_layout.addWidget(self.mock_db_check)
        
        # Add type group to layout
        layout.addWidget(type_group)
        
        # Connection settings group
        connection_group = QGroupBox("Database Connection")
        connection_form = QFormLayout(connection_group)
        
        # Host
        self.db_host_edit = QLineEdit(self.settings.get("database", {}).get("db_host", "localhost"))
        self.db_host_edit.textChanged.connect(
            lambda text: self._update_setting("database", "db_host", text)
        )
        connection_form.addRow("Host:", self.db_host_edit)
        
        # Port
        self.db_port_spin = QSpinBox()
        self.db_port_spin.setRange(1, 65535)
        self.db_port_spin.setValue(self.settings.get("database", {}).get("db_port", 5432))
        self.db_port_spin.valueChanged.connect(
            lambda value: self._update_setting("database", "db_port", value)
        )
        connection_form.addRow("Port:", self.db_port_spin)
        
        # Database name
        self.db_name_edit = QLineEdit(self.settings.get("database", {}).get("db_name", "crypto_trading"))
        self.db_name_edit.textChanged.connect(
            lambda text: self._update_setting("database", "db_name", text)
        )
        connection_form.addRow("Database Name:", self.db_name_edit)
        
        # Username
        self.db_user_edit = QLineEdit(self.settings.get("database", {}).get("db_user", "trading_user"))
        self.db_user_edit.textChanged.connect(
            lambda text: self._update_setting("database", "db_user", text)
        )
        connection_form.addRow("Username:", self.db_user_edit)
        
        # Password
        self.db_password_edit = QLineEdit(self.settings.get("database", {}).get("db_password", ""))
        self.db_password_edit.setEchoMode(QLineEdit.Password)
        self.db_password_edit.textChanged.connect(
            lambda text: self._update_setting("database", "db_password", text)
        )
        connection_form.addRow("Password:", self.db_password_edit)
        
        # Add connection group to layout
        layout.addWidget(connection_group)
        
        # Test connection button
        self.test_connection_btn = QPushButton("Test Connection")
        self.test_connection_btn.clicked.connect(self._test_db_connection)
        layout.addWidget(self.test_connection_btn)
        
        # Data management group
        data_group = QGroupBox("Data Management")
        data_form = QFormLayout(data_group)
        
        # Auto-cleanup
        self.auto_cleanup_check = QCheckBox()
        self.auto_cleanup_check.setChecked(self.settings.get("database", {}).get("auto_cleanup", True))
        self.auto_cleanup_check.stateChanged.connect(
            lambda state: self._update_setting("database", "auto_cleanup", state == Qt.Checked)
        )
        data_form.addRow("Auto-cleanup Old Data:", self.auto_cleanup_check)
        
        # Data retention
        self.data_retention_spin = QSpinBox()
        self.data_retention_spin.setRange(7, 365)
        self.data_retention_spin.setSuffix(" days")
        self.data_retention_spin.setValue(self.settings.get("database", {}).get("data_retention", 90))
        self.data_retention_spin.valueChanged.connect(
            lambda value: self._update_setting("database", "data_retention", value)
        )
        data_form.addRow("Data Retention Period:", self.data_retention_spin)
        
        # Add data group to layout
        layout.addWidget(data_group)
        
        # Add spacer
        layout.addStretch()
        
        # Add tab
        tabs.addTab(db_tab, "Database")
        
    def _test_db_connection(self):
        """Test database connection"""
        # This would be implemented with actual database connection test
        # For now, just show a message
        if self.mock_db_check.isChecked():
            QMessageBox.information(self, "Connection Test", "Mock database is enabled. No connection test needed.")
        else:
            QMessageBox.information(self, "Connection Test", "Database connection test would be performed here.")
        
    def _add_ai_tab(self, tabs):
        """Add AI settings tab"""
        ai_tab = QScrollArea()
        ai_tab.setWidgetResizable(True)
        
        ai_content = QWidget()
        layout = QVBoxLayout(ai_content)
        
        # General AI settings
        general_group = QGroupBox("General AI Settings")
        general_form = QFormLayout(general_group)
        
        # Enable AI
        self.use_ai_check = QCheckBox()
        self.use_ai_check.setChecked(self.settings.get("ai_settings", {}).get("use_ai", True))
        self.use_ai_check.stateChanged.connect(
            lambda state: self._update_setting("ai_settings", "use_ai", state == Qt.Checked)
        )
        general_form.addRow("Enable AI Components:", self.use_ai_check)
        
        # Confidence threshold
        self.confidence_spin = QDoubleSpinBox()
        self.confidence_spin.setRange(0.1, 0.95)
        self.confidence_spin.setDecimals(2)
        self.confidence_spin.setSingleStep(0.05)
        self.confidence_spin.setValue(self.settings.get("ai_settings", {}).get("model_confidence_threshold", 0.7))
        self.confidence_spin.valueChanged.connect(
            lambda value: self._update_setting("ai_settings", "model_confidence_threshold", value)
        )
        general_form.addRow("Confidence Threshold:", self.confidence_spin)
        
        # Auto-update models
        self.auto_update_check = QCheckBox()
        self.auto_update_check.setChecked(self.settings.get("ai_settings", {}).get("auto_update_models", False))
        self.auto_update_check.stateChanged.connect(
            lambda state: self._update_setting("ai_settings", "auto_update_models", state == Qt.Checked)
        )
        general_form.addRow("Auto-update Models:", self.auto_update_check)
        
        # Add general group to layout
        layout.addWidget(general_group)
        
        # Model settings group
        model_group = QGroupBox("Model Settings")
        model_form = QFormLayout(model_group)
        
        # Learning rate
        self.learning_rate_spin = QDoubleSpinBox()
        self.learning_rate_spin.setRange(0.0001, 0.1)
        self.learning_rate_spin.setDecimals(4)
        self.learning_rate_spin.setSingleStep(0.0001)
        self.learning_rate_spin.setValue(self.settings.get("ai_settings", {}).get("learning_rate", 0.001))
        self.learning_rate_spin.valueChanged.connect(
            lambda value: self._update_setting("ai_settings", "learning_rate", value)
        )
        model_form.addRow("Learning Rate:", self.learning_rate_spin)
        
        # Exploration rate
        self.exploration_rate_spin = QDoubleSpinBox()
        self.exploration_rate_spin.setRange(0.01, 0.5)
        self.exploration_rate_spin.setDecimals(2)
        self.exploration_rate_spin.setSingleStep(0.01)
        self.exploration_rate_spin.setValue(self.settings.get("ai_settings", {}).get("exploration_rate", 0.1))
        self.exploration_rate_spin.valueChanged.connect(
            lambda value: self._update_setting("ai_settings", "exploration_rate", value)
        )
        model_form.addRow("Exploration Rate:", self.exploration_rate_spin)
        
        # Reinforcement learning
        self.rl_check = QCheckBox()
        self.rl_check.setChecked(self.settings.get("ai_settings", {}).get("use_reinforcement_learning", True))
        self.rl_check.stateChanged.connect(
            lambda state: self._update_setting("ai_settings", "use_reinforcement_learning", state == Qt.Checked)
        )
        model_form.addRow("Use Reinforcement Learning:", self.rl_check)
        
        # Add model group to layout
        layout.addWidget(model_group)
        
        # AI components group
        components_group = QGroupBox("AI Components")
        components_layout = QVBoxLayout(components_group)
        
        # Component checkboxes
        ai_components = {
            "market_making": "Market Making AI",
            "portfolio_allocation": "Portfolio Allocation",
            "predictive_order_flow": "Order Flow Prediction",
            "trade_timing": "Trade Timing",
            "trade_exit": "Trade Exit",
            "trade_reentry": "Trade Re-entry"
        }
        
        self.component_checkboxes = {}
        enabled_components = self.settings.get("ai_settings", {}).get("enabled_components", list(ai_components.keys()))
        
        for component_id, component_name in ai_components.items():
            checkbox = QCheckBox(component_name)
            checkbox.setChecked(component_id in enabled_components)
            checkbox.stateChanged.connect(
                lambda state, c=component_id: self._update_ai_component(c, state == Qt.Checked)
            )
            components_layout.addWidget(checkbox)
            self.component_checkboxes[component_id] = checkbox
            
        # Add components group to layout
        layout.addWidget(components_group)
        
        # Model training group
        training_group = QGroupBox("Model Training")
        training_form = QFormLayout(training_group)
        
        # Training frequency
        self.training_freq_combo = QComboBox()
        self.training_freq_combo.addItems(["daily", "weekly", "monthly", "manual"])
        current_freq = self.settings.get("ai_settings", {}).get("training_frequency", "weekly")
        self.training_freq_combo.setCurrentText(current_freq)
        self.training_freq_combo.currentTextChanged.connect(
            lambda text: self._update_setting("ai_settings", "training_frequency", text)
        )
        training_form.addRow("Training Frequency:", self.training_freq_combo)
        
        # Use GPU
        self.gpu_check = QCheckBox()
        self.gpu_check.setChecked(self.settings.get("ai_settings", {}).get("use_gpu", True))
        self.gpu_check.stateChanged.connect(
            lambda state: self._update_setting("ai_settings", "use_gpu", state == Qt.Checked)
        )
        training_form.addRow("Use GPU for Training:", self.gpu_check)
        
        # Add training group to layout
        layout.addWidget(training_group)
        
        # Add spacer
        layout.addStretch()
        
        # Set content widget
        ai_tab.setWidget(ai_content)
        
        # Add tab
        tabs.addTab(ai_tab, "AI/ML")
        
    def _update_ai_component(self, component_id, enabled):
        """Update the enabled AI components"""
        components = self.settings.get("ai_settings", {}).get("enabled_components", [])
        
        if enabled and component_id not in components:
            components.append(component_id)
        elif not enabled and component_id in components:
            components.remove(component_id)
            
        self._update_setting("ai_settings", "enabled_components", components)
        
    def _add_connection_tab(self, tabs):
        """Add connection settings tab"""
        connection_tab = QWidget()
        layout = QVBoxLayout(connection_tab)
        
        # Exchange settings group
        exchange_group = QGroupBox("Exchange Settings")
        exchange_form = QFormLayout(exchange_group)
        
        # Exchange selection
        self.exchange_combo = QComboBox()
        self.exchange_combo.addItems(["binance", "coinbase", "kraken", "ftx", "bybit", "kucoin"])
        current_exchange = self.settings.get("connection", {}).get("exchange", "binance")
        self.exchange_combo.setCurrentText(current_exchange)
        self.exchange_combo.currentTextChanged.connect(
            lambda text: self._update_setting("connection", "exchange", text)
        )
        exchange_form.addRow("Exchange:", self.exchange_combo)
        
        # Testnet
        self.testnet_check = QCheckBox()
        self.testnet_check.setChecked(self.settings.get("connection", {}).get("testnet", True))
        self.testnet_check.stateChanged.connect(
            lambda state: self._update_setting("connection", "testnet", state == Qt.Checked)
        )
        exchange_form.addRow("Use Testnet:", self.testnet_check)
        
        # API key
        self.api_key_edit = QLineEdit(self.settings.get("connection", {}).get("api_key", ""))
        self.api_key_edit.textChanged.connect(
            lambda text: self._update_setting("connection", "api_key", text)
        )
        exchange_form.addRow("API Key:", self.api_key_edit)
        
        # API secret
        self.api_secret_edit = QLineEdit(self.settings.get("connection", {}).get("api_secret", ""))
        self.api_secret_edit.setEchoMode(QLineEdit.Password)
        self.api_secret_edit.textChanged.connect(
            lambda text: self._update_setting("connection", "api_secret", text)
        )
        exchange_form.addRow("API Secret:", self.api_secret_edit)
        
        # Connection timeout
        self.timeout_spin = QSpinBox()
        self.timeout_spin.setRange(5, 120)
        self.timeout_spin.setSuffix(" seconds")
        self.timeout_spin.setValue(self.settings.get("connection", {}).get("connection_timeout", 30))
        self.timeout_spin.valueChanged.connect(
            lambda value: self._update_setting("connection", "connection_timeout", value)
        )
        exchange_form.addRow("Connection Timeout:", self.timeout_spin)
        
        # Add exchange group to layout
        layout.addWidget(exchange_group)
        
        # Rate limits group
        rate_group = QGroupBox("Rate Limiting")
        rate_form = QFormLayout(rate_group)
        
        # Respect rate limits
        self.rate_limits_check = QCheckBox()
        self.rate_limits_check.setChecked(self.settings.get("connection", {}).get("respect_rate_limits", True))
        self.rate_limits_check.stateChanged.connect(
            lambda state: self._update_setting("connection", "respect_rate_limits", state == Qt.Checked)
        )
        rate_form.addRow("Respect Rate Limits:", self.rate_limits_check)
        
        # Request rate
        self.request_rate_spin = QDoubleSpinBox()
        self.request_rate_spin.setRange(0.1, 60.0)
        self.request_rate_spin.setDecimals(1)
        self.request_rate_spin.setSuffix(" req/sec")
        self.request_rate_spin.setValue(self.settings.get("connection", {}).get("request_rate", 1.0))
        self.request_rate_spin.valueChanged.connect(
            lambda value: self._update_setting("connection", "request_rate", value)
        )
        rate_form.addRow("Request Rate:", self.request_rate_spin)
        
        # Add rate group to layout
        layout.addWidget(rate_group)
        
        # Authentication storage group
        auth_group = QGroupBox("Authentication Storage")
        auth_layout = QVBoxLayout(auth_group)
        
        # Store credentials securely
        self.secure_storage_check = QCheckBox("Store credentials securely (system keyring)")
        self.secure_storage_check.setChecked(self.settings.get("connection", {}).get("secure_storage", True))
        self.secure_storage_check.stateChanged.connect(
            lambda state: self._update_setting("connection", "secure_storage", state == Qt.Checked)
        )
        auth_layout.addWidget(self.secure_storage_check)
        
        # Add auth group to layout
        layout.addWidget(auth_group)
        
        # Test connection button
        self.test_api_btn = QPushButton("Test API Connection")
        self.test_api_btn.clicked.connect(self._test_api_connection)
        layout.addWidget(self.test_api_btn)
        
        # Add spacer
        layout.addStretch()
        
        # Add tab
        tabs.addTab(connection_tab, "Connection")
        
    def _test_api_connection(self):
        """Test API connection to exchange"""
        # This would be implemented with actual API connection test
        # For now, just show a message
        if self.testnet_check.isChecked():
            QMessageBox.information(self, "Connection Test", f"Testing connection to {self.exchange_combo.currentText()} testnet...")
        else:
            QMessageBox.information(self, "Connection Test", f"Testing connection to {self.exchange_combo.currentText()}...")
        
    def _add_notifications_tab(self, tabs):
        """Add notifications settings tab"""
        notifications_tab = QWidget()
        layout = QVBoxLayout(notifications_tab)
        
        # General notification settings
        general_group = QGroupBox("Notification Settings")
        general_form = QFormLayout(general_group)
        
        # Enable notifications
        self.enable_notifications_check = QCheckBox()
        self.enable_notifications_check.setChecked(self.settings.get("notifications", {}).get("enable_notifications", True))
        self.enable_notifications_check.stateChanged.connect(
            lambda state: self._update_setting("notifications", "enable_notifications", state == Qt.Checked)
        )
        general_form.addRow("Enable Notifications:", self.enable_notifications_check)
        
        # Trade notifications
        self.trade_notifications_check = QCheckBox()
        self.trade_notifications_check.setChecked(self.settings.get("notifications", {}).get("trade_notifications", True))
        self.trade_notifications_check.stateChanged.connect(
            lambda state: self._update_setting("notifications", "trade_notifications", state == Qt.Checked)
        )
        general_form.addRow("Trade Notifications:", self.trade_notifications_check)
        
        # Error notifications
        self.error_notifications_check = QCheckBox()
        self.error_notifications_check.setChecked(self.settings.get("notifications", {}).get("error_notifications", True))
        self.error_notifications_check.stateChanged.connect(
            lambda state: self._update_setting("notifications", "error_notifications", state == Qt.Checked)
        )
        general_form.addRow("Error Notifications:", self.error_notifications_check)
        
        # Daily summary
        self.daily_summary_check = QCheckBox()
        self.daily_summary_check.setChecked(self.settings.get("notifications", {}).get("daily_summary", True))
        self.daily_summary_check.stateChanged.connect(
            lambda state: self._update_setting("notifications", "daily_summary", state == Qt.Checked)
        )
        general_form.addRow("Daily Summary:", self.daily_summary_check)
        
        # Add general group to layout
        layout.addWidget(general_group)
        
        # Email notifications group
        email_group = QGroupBox("Email Notifications")
        email_form = QFormLayout(email_group)
        
        # Enable email notifications
        self.email_notifications_check = QCheckBox()
        self.email_notifications_check.setChecked(self.settings.get("notifications", {}).get("email_notifications", False))
        self.email_notifications_check.stateChanged.connect(
            lambda state: self._update_setting("notifications", "email_notifications", state == Qt.Checked)
        )
        email_form.addRow("Enable Email Notifications:", self.email_notifications_check)
        
        # Email address
        self.email_edit = QLineEdit(self.settings.get("notifications", {}).get("email_address", ""))
        self.email_edit.textChanged.connect(
            lambda text: self._update_setting("notifications", "email_address", text)
        )
        email_form.addRow("Email Address:", self.email_edit)
        
        # Test email button
        self.test_email_btn = QPushButton("Test Email")
        self.test_email_btn.clicked.connect(self._test_email)
        email_form.addRow("", self.test_email_btn)
        
        # Add email group to layout
        layout.addWidget(email_group)
        
        # Notification thresholds group
        thresholds_group = QGroupBox("Notification Thresholds")
        thresholds_form = QFormLayout(thresholds_group)
        
        # Price change threshold
        self.price_change_spin = QDoubleSpinBox()
        self.price_change_spin.setRange(0.1, 10.0)
        self.price_change_spin.setDecimals(1)
        self.price_change_spin.setSuffix("%")
        self.price_change_spin.setValue(self.settings.get("notifications", {}).get("price_change_threshold", 5.0))
        self.price_change_spin.valueChanged.connect(
            lambda value: self._update_setting("notifications", "price_change_threshold", value)
        )
        thresholds_form.addRow("Price Change Threshold:", self.price_change_spin)
        
        # PnL threshold
        self.pnl_spin = QDoubleSpinBox()
        self.pnl_spin.setRange(0.1, 10.0)
        self.pnl_spin.setDecimals(1)
        self.pnl_spin.setSuffix("%")
        self.pnl_spin.setValue(self.settings.get("notifications", {}).get("pnl_threshold", 3.0))
        self.pnl_spin.valueChanged.connect(
            lambda value: self._update_setting("notifications", "pnl_threshold", value)
        )
        thresholds_form.addRow("PnL Threshold:", self.pnl_spin)
        
        # Add thresholds group to layout
        layout.addWidget(thresholds_group)
        
        # Add spacer
        layout.addStretch()
        
        # Add tab
        tabs.addTab(notifications_tab, "Notifications")
        
    def _test_email(self):
        """Test email notification"""
        email = self.email_edit.text()
        if not email:
            QMessageBox.warning(self, "Test Email", "Please enter an email address.")
            return
            
        QMessageBox.information(self, "Test Email", f"A test email would be sent to {email}.")
        
    def _add_ui_tab(self, tabs):
        """Add UI settings tab"""
        ui_tab = QWidget()
        layout = QVBoxLayout(ui_tab)
        
        # Chart settings group
        chart_group = QGroupBox("Chart Settings")
        chart_form = QFormLayout(chart_group)
        
        # Chart style
        self.chart_style_combo = QComboBox()
        self.chart_style_combo.addItems(["candles", "ohlc", "line", "area"])
        current_style = self.settings.get("ui", {}).get("chart_style", "candles")
        self.chart_style_combo.setCurrentText(current_style)
        self.chart_style_combo.currentTextChanged.connect(
            lambda text: self._update_setting("ui", "chart_style", text)
        )
        chart_form.addRow("Chart Style:", self.chart_style_combo)
        
        # Default indicators
        indicators = ["sma", "ema", "rsi", "macd", "bollinger_bands", "volume", "ichimoku"]
        default_indicators = self.settings.get("ui", {}).get("default_indicators", ["sma", "volume", "rsi"])
        
        indicators_layout = QVBoxLayout()
        self.indicator_checkboxes = {}
        
        for indicator in indicators:
            checkbox = QCheckBox(indicator.replace('_', ' ').title())
            checkbox.setChecked(indicator in default_indicators)
            checkbox.stateChanged.connect(
                lambda state, ind=indicator: self._update_default_indicators(ind, state == Qt.Checked)
            )
            indicators_layout.addWidget(checkbox)
            self.indicator_checkboxes[indicator] = checkbox
            
        chart_form.addRow("Default Indicators:", indicators_layout)
        
        # Add chart group to layout
        layout.addWidget(chart_group)
        
        # Display settings group
        display_group = QGroupBox("Display Settings")
        display_form = QFormLayout(display_group)
        
        # Refresh interval
        self.refresh_spin = QSpinBox()
        self.refresh_spin.setRange(1, 60)
        self.refresh_spin.setSuffix(" seconds")
        self.refresh_spin.setValue(self.settings.get("ui", {}).get("refresh_interval", 5))
        self.refresh_spin.valueChanged.connect(
            lambda value: self._update_setting("ui", "refresh_interval", value)
        )
        display_form.addRow("Refresh Interval:", self.refresh_spin)
        
        # Show orderbook
        self.orderbook_check = QCheckBox()
        self.orderbook_check.setChecked(self.settings.get("ui", {}).get("show_orderbook", True))
        self.orderbook_check.stateChanged.connect(
            lambda state: self._update_setting("ui", "show_orderbook", state == Qt.Checked)
        )
        display_form.addRow("Show Order Book:", self.orderbook_check)
        
        # Dark mode
        self.dark_mode_check = QCheckBox()
        self.dark_mode_check.setChecked(self.settings.get("ui", {}).get("dark_mode", False))
        self.dark_mode_check.stateChanged.connect(
            lambda state: self._update_setting("ui", "dark_mode", state == Qt.Checked)
        )
        display_form.addRow("Dark Mode:", self.dark_mode_check)
        
        # Font size
        self.font_size_combo = QComboBox()
        self.font_size_combo.addItems(["small", "medium", "large"])
        current_font_size = self.settings.get("ui", {}).get("font_size", "medium")
        self.font_size_combo.setCurrentText(current_font_size)
        self.font_size_combo.currentTextChanged.connect(
            lambda text: self._update_setting("ui", "font_size", text)
        )
        display_form.addRow("Font Size:", self.font_size_combo)
        
        # Add display group to layout
        layout.addWidget(display_group)
        
        # Add spacer
        layout.addStretch()
        
        # Add tab
        tabs.addTab(ui_tab, "UI")
        
    def _update_default_indicators(self, indicator, enabled):
        """Update the default indicators"""
        indicators = self.settings.get("ui", {}).get("default_indicators", [])
        
        if enabled and indicator not in indicators:
            indicators.append(indicator)
        elif not enabled and indicator in indicators:
            indicators.remove(indicator)
            
        self._update_setting("ui", "default_indicators", indicators)
        
    def _update_setting(self, section, key, value):
        """Update a setting value and mark settings as dirty"""
        # Ensure section exists
        if section not in self.settings:
            self.settings[section] = {}
            
        # Update value
        self.settings[section][key] = value
        
        # Check if settings changed from original
        if self.original_settings.get(section, {}).get(key) != value:
            self.is_dirty = True
            self.save_button.setEnabled(True)
            self.status_label.setText("Settings have changed. Click 'Save Settings' to apply.")
            self.status_label.setStyleSheet("color: red;")
            
    def _save_settings(self):
        """Save settings to trading system"""
        try:
            # Update trading system config
            if hasattr(self.trading_system, 'config'):
                self.trading_system.config.update(self.settings)
                
            # Save to disk if database available
            if hasattr(self.trading_system, 'db'):
                if hasattr(self.trading_system.db, 'store_settings'):
                    self.trading_system.db.store_settings('main_config', self.settings)
                    
            # Update original settings
            self.original_settings = copy.deepcopy(self.settings)
            self.is_dirty = False
            self.save_button.setEnabled(False)
            
            # Update UI
            self.status_label.setText("Settings saved successfully.")
            self.status_label.setStyleSheet("color: green;")
            
            # Hide status message after 3 seconds
            QTimer.singleShot(3000, lambda: self.status_label.setText(""))
            
            # Emit settings changed signal
            self.settings_changed.emit(self.settings)
            
        except Exception as e:
            logging.error(f"Error saving settings: {e}")
            self.status_label.setText(f"Error saving settings: {str(e)}")
            self.status_label.setStyleSheet("color: red;")
            
    def _reset_to_defaults(self):
        """Reset settings to defaults"""
        reply = QMessageBox.question(self, "Reset Settings", 
                                    "Are you sure you want to reset all settings to defaults?",
                                    QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        
        if reply == QMessageBox.Yes:
            self._load_default_settings()
            self.is_dirty = True
            self.save_button.setEnabled(True)
            self.status_label.setText("Settings reset to defaults. Click 'Save Settings' to apply.")
            self.status_label.setStyleSheet("color: red;")
            
            # Refresh UI to show default values
            self._refresh_ui()
            
    def _refresh_ui(self):
        """Refresh UI to reflect current settings"""
        # This is a simplified version - in a real app, you'd update all widgets
        # For now, just recreate the tab widget
        self.layout().removeWidget(self.findChild(QTabWidget))
        self._setup_ui()
            
    def _import_settings(self):
        """Import settings from file"""
        # Ask for file
        filename, _ = QFileDialog.getOpenFileName(self, "Import Settings", "", "JSON Files (*.json)")
        
        if not filename:
            return
            
        try:
            # Load settings from file
            with open(filename, 'r') as f:
                imported_settings = json.load(f)
                
            # Update settings
            self.settings.update(imported_settings)
            self.is_dirty = True
            self.save_button.setEnabled(True)
            self.status_label.setText("Settings imported. Click 'Save Settings' to apply.")
            self.status_label.setStyleSheet("color: red;")
            
            # Refresh UI to show imported values
            self._refresh_ui()
            
        except Exception as e:
            logging.error(f"Error importing settings: {e}")
            QMessageBox.critical(self, "Import Error", f"Error importing settings: {str(e)}")
            
    def _export_settings(self):
        """Export settings to file"""
        # Ask for file
        filename, _ = QFileDialog.getSaveFileName(self, "Export Settings", "", "JSON Files (*.json)")
        
        if not filename:
            return
            
        try:
            # Save settings to file
            with open(filename, 'w') as f:
                json.dump(self.settings, f, indent=4)
                
            QMessageBox.information(self, "Export Settings", f"Settings exported to {filename}")
            
        except Exception as e:
            logging.error(f"Error exporting settings: {e}")
            QMessageBox.critical(self, "Export Error", f"Error exporting settings: {str(e)}")
            
    def _autosave_settings(self):
        """Auto-save settings if enabled and dirty"""
        if self.is_dirty and self.settings.get("general", {}).get("auto_save", True):
            self._save_settings()
            
    def closeEvent(self, event):
        """Handle close event"""
        if self.is_dirty:
            reply = QMessageBox.question(self, "Unsaved Settings", 
                                        "You have unsaved settings. Do you want to save before closing?",
                                        QMessageBox.Save | QMessageBox.Discard | QMessageBox.Cancel, 
                                        QMessageBox.Save)
            
            if reply == QMessageBox.Save:
                self._save_settings()
            elif reply == QMessageBox.Cancel:
                event.ignore()
                return
                
        event.accept()
