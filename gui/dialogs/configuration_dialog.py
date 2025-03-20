import os
import json
import logging
from typing import Dict, Any, Optional

from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QTabWidget, QWidget,
    QFormLayout, QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox,
    QCheckBox, QPushButton, QLabel, QFileDialog, QMessageBox,
    QDialogButtonBox, QGroupBox, QScrollArea
)
from PyQt5.QtCore import Qt, pyqtSignal, QSettings, QSize


class ConfigurationDialog(QDialog):
    """
    Comprehensive configuration dialog for the trading system.
    Allows users to modify application settings across various categories.
    """
    
    config_updated = pyqtSignal(dict)  # Signal emitted when configuration is updated
    
    def __init__(self, trading_system=None, parent=None):
        super().__init__(parent)
        self.trading_system = trading_system
        self.config = {}
        self.modified = False
        self.field_widgets = {}  # Keep track of field widgets for easier access
        
        # Initialize UI
        self._init_ui()
        
        # Load configuration
        self._load_configuration()
        
    def _init_ui(self):
        """Initialize the dialog UI components."""
        # Set dialog properties
        self.setWindowTitle("Trading System Configuration")
        self.resize(850, 650)
        self.setMinimumSize(700, 500)
        
        # Main layout
        main_layout = QVBoxLayout(self)
        
        # Create tab widget for categories
        self.tab_widget = QTabWidget()
        main_layout.addWidget(self.tab_widget)
        
        # Create tabs for different setting categories
        self._create_general_tab()
        self._create_data_sources_tab()
        self._create_database_tab()
        self._create_strategies_tab()
        self._create_risk_tab()
        self._create_ai_tab()
        self._create_network_tab()
        
        # Buttons
        button_layout = QHBoxLayout()
        
        # Save as Default button
        self.save_default_btn = QPushButton("Save as Default")
        self.save_default_btn.clicked.connect(self._save_as_default)
        button_layout.addWidget(self.save_default_btn)
        
        # Reset to Default button
        self.reset_btn = QPushButton("Reset to Default")
        self.reset_btn.clicked.connect(self._reset_to_default)
        button_layout.addWidget(self.reset_btn)
        
        # Spacer
        button_layout.addStretch()
        
        # Standard dialog buttons
        self.button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        self.button_box.accepted.connect(self.accept)
        self.button_box.rejected.connect(self.reject)
        button_layout.addWidget(self.button_box)
        
        main_layout.addLayout(button_layout)
        
    def _create_general_tab(self):
        """Create general settings tab."""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # General settings group
        general_group = QGroupBox("General Settings")
        form_layout = QFormLayout(general_group)
        
        # Trading mode
        self.trading_mode = QComboBox()
        self.trading_mode.addItems(["Backtest", "Paper Trading", "Live Trading"])
        form_layout.addRow("Trading Mode:", self.trading_mode)
        self.field_widgets["trading_mode"] = self.trading_mode
        
        # Base currency
        self.base_currency = QLineEdit()
        form_layout.addRow("Base Currency:", self.base_currency)
        self.field_widgets["base_currency"] = self.base_currency
        
        # Initial capital
        self.initial_capital = QDoubleSpinBox()
        self.initial_capital.setRange(0, 10000000)
        self.initial_capital.setDecimals(2)
        self.initial_capital.setSingleStep(1000)
        form_layout.addRow("Initial Capital:", self.initial_capital)
        self.field_widgets["initial_capital"] = self.initial_capital
        
        # Auto start
        self.auto_start = QCheckBox()
        form_layout.addRow("Auto Start on Launch:", self.auto_start)
        self.field_widgets["auto_start"] = self.auto_start
        
        # Log level
        self.log_level = QComboBox()
        self.log_level.addItems(["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"])
        form_layout.addRow("Log Level:", self.log_level)
        self.field_widgets["log_level"] = self.log_level
        
        layout.addWidget(general_group)
        
        # UI settings group
        ui_group = QGroupBox("User Interface")
        ui_layout = QFormLayout(ui_group)
        
        # Theme
        self.theme = QComboBox()
        self.theme.addItems(["Light", "Dark", "System"])
        ui_layout.addRow("Theme:", self.theme)
        self.field_widgets["theme"] = self.theme
        
        # Chart update interval
        self.chart_update = QSpinBox()
        self.chart_update.setRange(500, 10000)
        self.chart_update.setSingleStep(500)
        self.chart_update.setSuffix(" ms")
        ui_layout.addRow("Chart Update Interval:", self.chart_update)
        self.field_widgets["chart_update_interval"] = self.chart_update
        
        # Auto refresh dashboard
        self.auto_refresh = QCheckBox()
        ui_layout.addRow("Auto-refresh Dashboard:", self.auto_refresh)
        self.field_widgets["auto_refresh"] = self.auto_refresh
        
        layout.addWidget(ui_group)
        
        # Add stretch at the bottom to push widgets to the top
        layout.addStretch()
        
        # Add tab
        self.tab_widget.addTab(tab, "General")
        
    def _create_data_sources_tab(self):
        """Create data sources tab."""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # Market data group
        market_group = QGroupBox("Market Data")
        market_layout = QFormLayout(market_group)
        
        # Provider
        self.data_provider = QComboBox()
        self.data_provider.addItems(["Binance", "Coinbase", "Alpha Vantage", "Yahoo Finance", "Custom"])
        market_layout.addRow("Data Provider:", self.data_provider)
        self.field_widgets["data_provider"] = self.data_provider
        
        # API key
        self.api_key = QLineEdit()
        self.api_key.setEchoMode(QLineEdit.Password)
        market_layout.addRow("API Key:", self.api_key)
        self.field_widgets["api_key"] = self.api_key
        
        # API secret
        self.api_secret = QLineEdit()
        self.api_secret.setEchoMode(QLineEdit.Password)
        market_layout.addRow("API Secret:", self.api_secret)
        self.field_widgets["api_secret"] = self.api_secret
        
        # Data update interval
        self.data_interval = QSpinBox()
        self.data_interval.setRange(1, 3600)
        self.data_interval.setSuffix(" sec")
        market_layout.addRow("Update Interval:", self.data_interval)
        self.field_widgets["data_update_interval"] = self.data_interval
        
        layout.addWidget(market_group)
        
        # Timeframes group
        timeframes_group = QGroupBox("Trading Timeframes")
        timeframes_layout = QFormLayout(timeframes_group)
        
        # Default timeframe
        self.default_timeframe = QComboBox()
        self.default_timeframe.addItems(["1m", "5m", "15m", "30m", "1h", "4h", "1d"])
        timeframes_layout.addRow("Default Timeframe:", self.default_timeframe)
        self.field_widgets["default_timeframe"] = self.default_timeframe
        
        # Available timeframes
        self.available_timeframes = {}
        for tf in ["1m", "5m", "15m", "30m", "1h", "4h", "1d"]:
            checkbox = QCheckBox()
            checkbox.setChecked(True)
            timeframes_layout.addRow(f"Enable {tf}:", checkbox)
            self.available_timeframes[tf] = checkbox
            self.field_widgets[f"timeframe_{tf}"] = checkbox
        
        layout.addWidget(timeframes_group)
        
        # Add stretch at the bottom
        layout.addStretch()
        
        # Add tab
        self.tab_widget.addTab(tab, "Data Sources")
        
    def _create_database_tab(self):
        """Create database settings tab."""
        tab = QScrollArea()
        tab.setWidgetResizable(True)
        tab_content = QWidget()
        layout = QVBoxLayout(tab_content)
        tab.setWidget(tab_content)
        
        # Database connection group
        db_group = QGroupBox("Database Connection")
        db_layout = QFormLayout(db_group)
        
        # Use mock database
        self.use_mock_db = QCheckBox()
        db_layout.addRow("Use Mock Database:", self.use_mock_db)
        self.field_widgets["use_mock_db"] = self.use_mock_db
        
        # Database type
        self.db_type = QComboBox()
        self.db_type.addItems(["TimescaleDB", "PostgreSQL", "SQLite", "MySQL"])
        db_layout.addRow("Database Type:", self.db_type)
        self.field_widgets["db_type"] = self.db_type
        
        # Database connection parameters
        self.db_host = QLineEdit("localhost")
        db_layout.addRow("Host:", self.db_host)
        self.field_widgets["db_host"] = self.db_host
        
        self.db_port = QSpinBox()
        self.db_port.setRange(1, 65535)
        self.db_port.setValue(5432)
        db_layout.addRow("Port:", self.db_port)
        self.field_widgets["db_port"] = self.db_port
        
        self.db_name = QLineEdit("trading_system")
        db_layout.addRow("Database Name:", self.db_name)
        self.field_widgets["db_name"] = self.db_name
        
        self.db_user = QLineEdit()
        db_layout.addRow("Username:", self.db_user)
        self.field_widgets["db_user"] = self.db_user
        
        self.db_password = QLineEdit()
        self.db_password.setEchoMode(QLineEdit.Password)
        db_layout.addRow("Password:", self.db_password)
        self.field_widgets["db_password"] = self.db_password
        
        # Test connection button
        self.test_db_btn = QPushButton("Test Connection")
        self.test_db_btn.clicked.connect(self._test_db_connection)
        db_layout.addRow("", self.test_db_btn)
        
        layout.addWidget(db_group)
        
        # Data storage group
        storage_group = QGroupBox("Data Storage Settings")
        storage_layout = QFormLayout(storage_group)
        
        # Max storage size
        self.max_storage = QSpinBox()
        self.max_storage.setRange(1, 10000)
        self.max_storage.setSuffix(" GB")
        storage_layout.addRow("Max Storage Size:", self.max_storage)
        self.field_widgets["max_storage_size"] = self.max_storage
        
        # Data retention period
        self.data_retention = QSpinBox()
        self.data_retention.setRange(1, 3650)
        self.data_retention.setSuffix(" days")
        storage_layout.addRow("Data Retention Period:", self.data_retention)
        self.field_widgets["data_retention"] = self.data_retention
        
        # Auto cleanup
        self.auto_cleanup = QCheckBox()
        storage_layout.addRow("Auto Cleanup Old Data:", self.auto_cleanup)
        self.field_widgets["auto_cleanup"] = self.auto_cleanup
        
        layout.addWidget(storage_group)
        
        # Add stretch at the bottom
        layout.addStretch()
        
        # Add tab
        self.tab_widget.addTab(tab, "Database")
        
    def _create_strategies_tab(self):
        """Create strategies settings tab."""
        tab = QScrollArea()
        tab.setWidgetResizable(True)
        tab_content = QWidget()
        layout = QVBoxLayout(tab_content)
        tab.setWidget(tab_content)
        
        # Strategy settings group
        strategy_group = QGroupBox("Strategy Settings")
        strategy_layout = QFormLayout(strategy_group)
        
        # Active strategies
        strategy_list = ["Trend Following", "Mean Reversion", "Breakout", "Multi-Timeframe", "Volume-Based", "AI-Driven"]
        self.active_strategies = {}
        
        for strategy in strategy_list:
            checkbox = QCheckBox()
            strategy_layout.addRow(f"Enable {strategy}:", checkbox)
            self.active_strategies[strategy] = checkbox
            self.field_widgets[f"strategy_{strategy.lower().replace('-', '_').replace(' ', '_')}"] = checkbox
        
        layout.addWidget(strategy_group)
        
        # Strategy weights group
        weights_group = QGroupBox("Strategy Weights")
        weights_layout = QFormLayout(weights_group)
        
        # Strategy weights
        self.strategy_weights = {}
        
        for strategy in strategy_list:
            spin = QDoubleSpinBox()
            spin.setRange(0, 1)
            spin.setDecimals(2)
            spin.setSingleStep(0.05)
            spin.setValue(1.0 / len(strategy_list))  # Equal weights by default
            weights_layout.addRow(f"{strategy} Weight:", spin)
            self.strategy_weights[strategy] = spin
            self.field_widgets[f"weight_{strategy.lower().replace('-', '_').replace(' ', '_')}"] = spin
        
        layout.addWidget(weights_group)
        
        # Strategy optimization group
        optimization_group = QGroupBox("Strategy Optimization")
        optimization_layout = QFormLayout(optimization_group)
        
        # Auto-optimize
        self.auto_optimize = QCheckBox()
        optimization_layout.addRow("Auto-optimize Parameters:", self.auto_optimize)
        self.field_widgets["auto_optimize"] = self.auto_optimize
        
        # Optimization frequency
        self.optimization_frequency = QSpinBox()
        self.optimization_frequency.setRange(1, 10000)
        self.optimization_frequency.setSuffix(" trades")
        optimization_layout.addRow("Optimization Frequency:", self.optimization_frequency)
        self.field_widgets["optimization_frequency"] = self.optimization_frequency
        
        # Performance metric for optimization
        self.optimization_metric = QComboBox()
        self.optimization_metric.addItems(["Sharpe Ratio", "Sortino Ratio", "Total Return", "Win Rate", "Profit Factor"])
        optimization_layout.addRow("Optimization Metric:", self.optimization_metric)
        self.field_widgets["optimization_metric"] = self.optimization_metric
        
        layout.addWidget(optimization_group)
        
        # Add stretch at the bottom
        layout.addStretch()
        
        # Add tab
        self.tab_widget.addTab(tab, "Strategies")
        
    def _create_risk_tab(self):
        """Create risk management tab."""
        tab = QScrollArea()
        tab.setWidgetResizable(True)
        tab_content = QWidget()
        layout = QVBoxLayout(tab_content)
        tab.setWidget(tab_content)
        
        # Position sizing group
        position_group = QGroupBox("Position Sizing")
        position_layout = QFormLayout(position_group)
        
        # Base risk per trade
        self.base_risk = QDoubleSpinBox()
        self.base_risk.setRange(0.001, 0.1)
        self.base_risk.setDecimals(3)
        self.base_risk.setSingleStep(0.001)
        self.base_risk.setValue(0.02)  # Default 2%
        position_layout.addRow("Base Risk Per Trade:", self.base_risk)
        self.field_widgets["base_risk"] = self.base_risk
        
        # Max risk per trade
        self.max_risk = QDoubleSpinBox()
        self.max_risk.setRange(0.001, 0.2)
        self.max_risk.setDecimals(3)
        self.max_risk.setSingleStep(0.001)
        self.max_risk.setValue(0.05)  # Default 5%
        position_layout.addRow("Max Risk Per Trade:", self.max_risk)
        self.field_widgets["max_risk"] = self.max_risk
        
        # Min risk per trade
        self.min_risk = QDoubleSpinBox()
        self.min_risk.setRange(0.001, 0.05)
        self.min_risk.setDecimals(3)
        self.min_risk.setSingleStep(0.001)
        self.min_risk.setValue(0.01)  # Default 1%
        position_layout.addRow("Min Risk Per Trade:", self.min_risk)
        self.field_widgets["min_risk"] = self.min_risk
        
        # Kelly criterion
        self.use_kelly = QCheckBox()
        position_layout.addRow("Use Kelly Criterion:", self.use_kelly)
        self.field_widgets["use_kelly"] = self.use_kelly
        
        # Volatility adjustment
        self.volatility_adjustment = QCheckBox()
        self.volatility_adjustment.setChecked(True)
        position_layout.addRow("Adjust for Volatility:", self.volatility_adjustment)
        self.field_widgets["volatility_adjustment"] = self.volatility_adjustment
        
        layout.addWidget(position_group)
        
        # Risk limits group
        limits_group = QGroupBox("Risk Limits")
        limits_layout = QFormLayout(limits_group)
        
        # Maximum open positions
        self.max_positions = QSpinBox()
        self.max_positions.setRange(1, 100)
        self.max_positions.setValue(5)
        limits_layout.addRow("Max Open Positions:", self.max_positions)
        self.field_widgets["max_positions"] = self.max_positions
        
        # Max risk per sector
        self.max_sector_risk = QDoubleSpinBox()
        self.max_sector_risk.setRange(0.01, 0.5)
        self.max_sector_risk.setDecimals(2)
        self.max_sector_risk.setSingleStep(0.01)
        self.max_sector_risk.setValue(0.2)  # Default 20%
        limits_layout.addRow("Max Risk Per Sector:", self.max_sector_risk)
        self.field_widgets["max_sector_risk"] = self.max_sector_risk
        
        # Daily loss limit
        self.daily_loss_limit = QDoubleSpinBox()
        self.daily_loss_limit.setRange(0.01, 0.1)
        self.daily_loss_limit.setDecimals(2)
        self.daily_loss_limit.setSingleStep(0.01)
        self.daily_loss_limit.setValue(0.03)  # Default 3%
        limits_layout.addRow("Daily Loss Limit:", self.daily_loss_limit)
        self.field_widgets["daily_loss_limit"] = self.daily_loss_limit
        
        # Max drawdown limit
        self.max_drawdown = QDoubleSpinBox()
        self.max_drawdown.setRange(0.05, 0.3)
        self.max_drawdown.setDecimals(2)
        self.max_drawdown.setSingleStep(0.01)
        self.max_drawdown.setValue(0.15)  # Default 15%
        limits_layout.addRow("Max Drawdown Limit:", self.max_drawdown)
        self.field_widgets["max_drawdown"] = self.max_drawdown
        
        # Circuit breakers
        self.circuit_breakers = QCheckBox()
        self.circuit_breakers.setChecked(True)
        limits_layout.addRow("Enable Circuit Breakers:", self.circuit_breakers)
        self.field_widgets["circuit_breakers"] = self.circuit_breakers
        
        layout.addWidget(limits_group)
        
        # Risk actions group
        actions_group = QGroupBox("Risk Actions")
        actions_layout = QFormLayout(actions_group)
        
        # Auto hedging
        self.auto_hedging = QCheckBox()
        actions_layout.addRow("Enable Auto Hedging:", self.auto_hedging)
        self.field_widgets["auto_hedging"] = self.auto_hedging
        
        # Hedging threshold
        self.hedging_threshold = QDoubleSpinBox()
        self.hedging_threshold.setRange(0.01, 0.2)
        self.hedging_threshold.setDecimals(2)
        self.hedging_threshold.setSingleStep(0.01)
        self.hedging_threshold.setValue(0.05)  # Default 5%
        actions_layout.addRow("Hedging Threshold:", self.hedging_threshold)
        self.field_widgets["hedging_threshold"] = self.hedging_threshold
        
        layout.addWidget(actions_group)
        
        # Add stretch at the bottom
        layout.addStretch()
        
        # Add tab
        self.tab_widget.addTab(tab, "Risk Management")
        
    def _create_ai_tab(self):
        """Create AI settings tab."""
        tab = QScrollArea()
        tab.setWidgetResizable(True)
        tab_content = QWidget()
        layout = QVBoxLayout(tab_content)
        tab.setWidget(tab_content)
        
        # AI system group
        ai_group = QGroupBox("AI System Settings")
        ai_layout = QFormLayout(ai_group)
        
        # Enable AI system
        self.enable_ai = QCheckBox()
        ai_layout.addRow("Enable AI System:", self.enable_ai)
        self.field_widgets["enable_ai"] = self.enable_ai
        
        # Enable specific AI components
        ai_components = [
            "Market Making AI", 
            "Portfolio Allocation AI", 
            "Trade Timing AI", 
            "Trade Exit AI", 
            "Order Flow Prediction AI", 
            "Risk Management AI"
        ]
        
        self.ai_components = {}
        for component in ai_components:
            checkbox = QCheckBox()
            ai_layout.addRow(f"Enable {component}:", checkbox)
            self.ai_components[component] = checkbox
            self.field_widgets[f"ai_{component.lower().replace(' ', '_')}"] = checkbox
        
        layout.addWidget(ai_group)
        
        # AI training group
        training_group = QGroupBox("AI Training Settings")
        training_layout = QFormLayout(training_group)
        
        # Learning rate
        self.learning_rate = QDoubleSpinBox()
        self.learning_rate.setRange(0.0001, 0.1)
        self.learning_rate.setDecimals(4)
        self.learning_rate.setSingleStep(0.0001)
        self.learning_rate.setValue(0.001)
        training_layout.addRow("Learning Rate:", self.learning_rate)
        self.field_widgets["learning_rate"] = self.learning_rate
        
        # Batch size
        self.batch_size = QSpinBox()
        self.batch_size.setRange(1, 1024)
        self.batch_size.setValue(32)
        training_layout.addRow("Batch Size:", self.batch_size)
        self.field_widgets["batch_size"] = self.batch_size
        
        # Enable continuous learning
        self.continuous_learning = QCheckBox()
        training_layout.addRow("Continuous Learning:", self.continuous_learning)
        self.field_widgets["continuous_learning"] = self.continuous_learning
        
        # Auto-optimize hyperparameters
        self.auto_optimize_hyperparams = QCheckBox()
        training_layout.addRow("Auto-optimize Hyperparameters:", self.auto_optimize_hyperparams)
        self.field_widgets["auto_optimize_hyperparams"] = self.auto_optimize_hyperparams
        
        layout.addWidget(training_group)
        
        # Models directory
        models_group = QGroupBox("Model Directories")
        models_layout = QFormLayout(models_group)
        
        # Models directory
        self.models_dir = QLineEdit("models")
        models_layout.addRow("Models Directory:", self.models_dir)
        self.field_widgets["models_dir"] = self.models_dir
        
        # Browse button
        browse_btn = QPushButton("Browse...")
        browse_btn.clicked.connect(self._browse_models_dir)
        models_layout.addRow("", browse_btn)
        
        layout.addWidget(models_group)
        
        # Add stretch at the bottom
        layout.addStretch()
        
        # Add tab
        self.tab_widget.addTab(tab, "AI Settings")
        
    def _create_network_tab(self):
        """Create network settings tab."""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # Connection settings group
        conn_group = QGroupBox("Connection Settings")
        conn_layout = QFormLayout(conn_group)
        
        # Use proxy
        self.use_proxy = QCheckBox()
        conn_layout.addRow("Use Proxy:", self.use_proxy)
        self.field_widgets["use_proxy"] = self.use_proxy
        
        # Proxy host
        self.proxy_host = QLineEdit()
        conn_layout.addRow("Proxy Host:", self.proxy_host)
        self.field_widgets["proxy_host"] = self.proxy_host
        
        # Proxy port
        self.proxy_port = QSpinBox()
        self.proxy_port.setRange(1, 65535)
        self.proxy_port.setValue(8080)
        conn_layout.addRow("Proxy Port:", self.proxy_port)
        self.field_widgets["proxy_port"] = self.proxy_port
        
        # Connection timeout
        self.conn_timeout = QSpinBox()
        self.conn_timeout.setRange(1, 300)
        self.conn_timeout.setValue(30)
        self.conn_timeout.setSuffix(" sec")
        conn_layout.addRow("Connection Timeout:", self.conn_timeout)
        self.field_widgets["connection_timeout"] = self.conn_timeout
        
        # Retry attempts
        self.retry_attempts = QSpinBox()
        self.retry_attempts.setRange(0, 10)
        self.retry_attempts.setValue(3)
        conn_layout.addRow("Retry Attempts:", self.retry_attempts)
        self.field_widgets["retry_attempts"] = self.retry_attempts
        
        layout.addWidget(conn_group)
        
        # Webhook settings group
        webhook_group = QGroupBox("Webhook Settings")
        webhook_layout = QFormLayout(webhook_group)
        
        # Enable webhooks
        self.enable_webhooks = QCheckBox()
        webhook_layout.addRow("Enable Webhooks:", self.enable_webhooks)
        self.field_widgets["enable_webhooks"] = self.enable_webhooks
        
        # Webhook URL
        self.webhook_url = QLineEdit()
        webhook_layout.addRow("Webhook URL:", self.webhook_url)
        self.field_widgets["webhook_url"] = self.webhook_url
        
        # Webhook events
        events = ["Trade Execution", "Position Closed", "Strategy Change", "Risk Alert"]
        self.webhook_events = {}
        
        for event in events:
            checkbox = QCheckBox()
            webhook_layout.addRow(f"Send {event} Events:", checkbox)
            self.webhook_events[event] = checkbox
            self.field_widgets[f"webhook_{event.lower().replace(' ', '_')}"] = checkbox
        
        layout.addWidget(webhook_group)
        
        # Add stretch at the bottom
        layout.addStretch()
        
        # Add tab
        self.tab_widget.addTab(tab, "Network")
        
    def _load_configuration(self):
        """Load current configuration values."""
        if self.trading_system is None:
            # Try to load from file directly
            self._load_from_file()
            return
            
        # Load from trading system if available
        try:
            if hasattr(self.trading_system, "config") and self.trading_system.config:
                self.config = self.trading_system.config
                self._populate_fields()
            else:
                self._load_from_file()
        except Exception as e:
            logging.error(f"Error loading configuration: {e}")
            self._load_from_file()
            
    def _load_from_file(self):
        """Load configuration from file."""
        try:
            config_path = os.path.join(os.getcwd(), "config.json")
            if os.path.exists(config_path):
                with open(config_path, "r") as f:
                    self.config = json.load(f)
                self._populate_fields()
            else:
                logging.warning("Config file not found. Using default values.")
        except Exception as e:
            logging.error(f"Error loading config from file: {e}")
            
    def _populate_fields(self):
        """Populate form fields with configuration values."""
        # Extract configuration values and set them to the appropriate widgets
        
        # General settings
        self._set_value("trading_mode", self.config.get("trading", {}).get("mode", "Backtest"))
        self._set_value("base_currency", self.config.get("trading", {}).get("base_currency", "USDT"))
        self._set_value("initial_capital", self.config.get("trading", {}).get("initial_capital", 10000))
        self._set_value("auto_start", self.config.get("general", {}).get("auto_start", False))
        self._set_value("log_level", self.config.get("logging", {}).get("level", "INFO"))
        self._set_value("theme", self.config.get("ui", {}).get("theme", "System"))
        self._set_value("chart_update_interval", self.config.get("ui", {}).get("chart_update_interval", 1000))
        self._set_value("auto_refresh", self.config.get("ui", {}).get("auto_refresh", True))
        
        # Data sources
        self._set_value("data_provider", self.config.get("data", {}).get("provider", "Binance"))
        self._set_value("api_key", self.config.get("exchange", {}).get("api_key", ""))
        self._set_value("api_secret", self.config.get("exchange", {}).get("api_secret", ""))
        self._set_value("data_update_interval", self.config.get("data", {}).get("update_interval", 10))
        self._set_value("default_timeframe", self.config.get("data", {}).get("default_timeframe", "1h"))
        
        # Set available timeframes
        available_tfs = self.config.get("data", {}).get("available_timeframes", ["1m", "5m", "15m", "30m", "1h", "4h", "1d"])
        for tf in self.available_timeframes:
            self._set_value(f"timeframe_{tf}", tf in available_tfs)
        
        # Database settings
        self._set_value("use_mock_db", self.config.get("database", {}).get("use_mock", True))
        self._set_value("db_type", self.config.get("database", {}).get("type", "TimescaleDB"))
        self._set_value("db_host", self.config.get("database", {}).get("host", "localhost"))
        self._set_value("db_port", self.config.get("database", {}).get("port", 5432))
        self._set_value("db_name", self.config.get("database", {}).get("name", "trading_system"))
        self._set_value("db_user", self.config.get("database", {}).get("user", ""))
        self._set_value("db_password", self.config.get("database", {}).get("password", ""))
        self._set_value("max_storage_size", self.config.get("database", {}).get("max_storage_size", 100))
        self._set_value("data_retention", self.config.get("database", {}).get("retention_days", 365))
        self._set_value("auto_cleanup", self.config.get("database", {}).get("auto_cleanup", True))
        
        # Strategy settings
        active_strategies = self.config.get("strategies", {}).get("active_strategies", [])
        for strategy in self.active_strategies:
            strategy_key = strategy.lower().replace('-', '_').replace(' ', '_')
            self._set_value(f"strategy_{strategy_key}", strategy in active_strategies)
            
        # Strategy weights
        weights = self.config.get("strategies", {}).get("weights", {})
        for strategy in self.strategy_weights:
            strategy_key = strategy.lower().replace('-', '_').replace(' ', '_')
            self._set_value(f"weight_{strategy_key}", weights.get(strategy_key, 1.0 / len(self.strategy_weights)))
            
        # Strategy optimization
        self._set_value("auto_optimize", self.config.get("strategies", {}).get("auto_optimize", False))
        self._set_value("optimization_frequency", self.config.get("strategies", {}).get("optimization_frequency", 100))
        self._set_value("optimization_metric", self.config.get("strategies", {}).get("optimization_metric", "Sharpe Ratio"))
        
        # Risk management
        risk_config = self.config.get("risk_limits", {})
        self._set_value("base_risk", risk_config.get("base_risk", 0.02))
        self._set_value("max_risk", risk_config.get("max_risk", 0.05))
        self._set_value("min_risk", risk_config.get("min_risk", 0.01))
        self._set_value("use_kelly", risk_config.get("use_kelly", False))
        self._set_value("volatility_adjustment", risk_config.get("volatility_adjustment", True))
        self._set_value("max_positions", risk_config.get("max_positions", 5))
        self._set_value("max_sector_risk", risk_config.get("max_sector_risk", 0.2))
        self._set_value("daily_loss_limit", risk_config.get("daily_loss_limit", 0.03))
        self._set_value("max_drawdown", risk_config.get("max_drawdown_limit", 0.15))
        self._set_value("circuit_breakers", risk_config.get("circuit_breakers", True))
        self._set_value("auto_hedging", risk_config.get("auto_hedging", False))
        self._set_value("hedging_threshold", risk_config.get("hedging_threshold", 0.05))
        
        # AI settings
        ai_config = self.config.get("ai", {})
        self._set_value("enable_ai", ai_config.get("enabled", False))
        
        # AI components
        ai_components = ai_config.get("components", {})
        for component in self.ai_components:
            component_key = component.lower().replace(' ', '_')
            self._set_value(f"ai_{component_key}", ai_components.get(component_key, False))
            
        # AI training
        training_config = ai_config.get("training", {})
        self._set_value("learning_rate", training_config.get("learning_rate", 0.001))
        self._set_value("batch_size", training_config.get("batch_size", 32))
        self._set_value("continuous_learning", training_config.get("continuous_learning", False))
        self._set_value("auto_optimize_hyperparams", training_config.get("auto_optimize_hyperparams", False))
        self._set_value("models_dir", ai_config.get("models_dir", "models"))
        
        # Network settings
        network_config = self.config.get("network", {})
        self._set_value("use_proxy", network_config.get("use_proxy", False))
        self._set_value("proxy_host", network_config.get("proxy_host", ""))
        self._set_value("proxy_port", network_config.get("proxy_port", 8080))
        self._set_value("connection_timeout", network_config.get("timeout", 30))
        self._set_value("retry_attempts", network_config.get("retry_attempts", 3))
        
        # Webhook settings
        webhook_config = network_config.get("webhooks", {})
        self._set_value("enable_webhooks", webhook_config.get("enabled", False))
        self._set_value("webhook_url", webhook_config.get("url", ""))
        
        # Webhook events
        webhook_events = webhook_config.get("events", {})
        for event in self.webhook_events:
            event_key = event.lower().replace(' ', '_')
            self._set_value(f"webhook_{event_key}", webhook_events.get(event_key, False))
            
    def _set_value(self, field_name, value):
        """Set value to the appropriate widget based on field name."""
        if field_name not in self.field_widgets:
            return
            
        widget = self.field_widgets[field_name]
        
        try:
            if isinstance(widget, QLineEdit):
                widget.setText(str(value))
            elif isinstance(widget, QSpinBox) or isinstance(widget, QDoubleSpinBox):
                widget.setValue(value)
            elif isinstance(widget, QComboBox):
                index = widget.findText(str(value))
                if index >= 0:
                    widget.setCurrentIndex(index)
            elif isinstance(widget, QCheckBox):
                widget.setChecked(bool(value))
        except Exception as e:
            logging.error(f"Error setting value for field {field_name}: {e}")
            
    def _get_value(self, field_name):
        """Get value from a widget based on field name."""
        if field_name not in self.field_widgets:
            return None
            
        widget = self.field_widgets[field_name]
        
        try:
            if isinstance(widget, QLineEdit):
                return widget.text()
            elif isinstance(widget, QSpinBox):
                return widget.value()
            elif isinstance(widget, QDoubleSpinBox):
                return widget.value()
            elif isinstance(widget, QComboBox):
                return widget.currentText()
            elif isinstance(widget, QCheckBox):
                return widget.isChecked()
        except Exception as e:
            logging.error(f"Error getting value for field {field_name}: {e}")
            
        return None
        
    def _browse_models_dir(self):
        """Browse for models directory."""
        current_dir = self.models_dir.text()
        if not os.path.isdir(current_dir):
            current_dir = os.getcwd()
            
        models_dir = QFileDialog.getExistingDirectory(
            self, "Select Models Directory", current_dir
        )
        
        if models_dir:
            self.models_dir.setText(models_dir)
            
    def _test_db_connection(self):
        """Test database connection."""
        # This would actually test the database connection in a real implementation
        # For now, just show a message
        use_mock = self.use_mock_db.isChecked()
        
        if use_mock:
            QMessageBox.information(self, "Database Connection", "Mock database is enabled. No connection test needed.")
            return
            
        db_type = self.db_type.currentText()
        host = self.db_host.text()
        port = self.db_port.value()
        db_name = self.db_name.text()
        user = self.db_user.text()
        password = self.db_password.text()
        
        # In a real implementation, you would test the connection here
        # For demonstration, just show the connection parameters
        QMessageBox.information(
            self, 
            "Database Connection", 
            f"Would attempt to connect to {db_type} database at {host}:{port}/{db_name}"
        )
        
    def _save_as_default(self):
        """Save current settings as default."""
        # Get current configuration
        config = self._get_current_config()
        
        # Save to settings
        settings = QSettings("TradingSystem", "Config")
        settings.setValue("default_config", json.dumps(config))
        
        QMessageBox.information(self, "Default Settings", "Current settings saved as default.")
        
    def _reset_to_default(self):
        """Reset settings to default."""
        # Load default settings
        settings = QSettings("TradingSystem", "Config")
        default_config_json = settings.value("default_config")
        
        if default_config_json:
            try:
                default_config = json.loads(default_config_json)
                self.config = default_config
                self._populate_fields()
                QMessageBox.information(self, "Reset Settings", "Settings reset to default values.")
            except Exception as e:
                logging.error(f"Error loading default config: {e}")
                QMessageBox.warning(self, "Reset Settings", "Could not load default settings.")
        else:
            QMessageBox.information(self, "Reset Settings", "No default settings found.")
            
    def _get_current_config(self) -> Dict[str, Any]:
        """Get current configuration from form fields."""
        config = {
            "general": {
                "auto_start": self._get_value("auto_start"),
            },
            "trading": {
                "mode": self._get_value("trading_mode"),
                "base_currency": self._get_value("base_currency"),
                "initial_capital": self._get_value("initial_capital")
            },
            "logging": {
                "level": self._get_value("log_level")
            },
            "ui": {
                "theme": self._get_value("theme"),
                "chart_update_interval": self._get_value("chart_update_interval"),
                "auto_refresh": self._get_value("auto_refresh")
            },
            "data": {
                "provider": self._get_value("data_provider"),
                "update_interval": self._get_value("data_update_interval"),
                "default_timeframe": self._get_value("default_timeframe"),
                "available_timeframes": [
                    tf for tf in self.available_timeframes
                    if self._get_value(f"timeframe_{tf}")
                ]
            },
            "exchange": {
                "api_key": self._get_value("api_key"),
                "api_secret": self._get_value("api_secret")
            },
            "database": {
                "use_mock": self._get_value("use_mock_db"),
                "type": self._get_value("db_type"),
                "host": self._get_value("db_host"),
                "port": self._get_value("db_port"),
                "name": self._get_value("db_name"),
                "user": self._get_value("db_user"),
                "password": self._get_value("db_password"),
                "max_storage_size": self._get_value("max_storage_size"),
                "retention_days": self._get_value("data_retention"),
                "auto_cleanup": self._get_value("auto_cleanup")
            },
            "strategies": {
                "active_strategies": [
                    strategy for strategy in self.active_strategies
                    if self._get_value(f"strategy_{strategy.lower().replace('-', '_').replace(' ', '_')}")
                ],
                "weights": {
                    strategy.lower().replace('-', '_').replace(' ', '_'): 
                    self._get_value(f"weight_{strategy.lower().replace('-', '_').replace(' ', '_')}")
                    for strategy in self.strategy_weights
                },
                "auto_optimize": self._get_value("auto_optimize"),
                "optimization_frequency": self._get_value("optimization_frequency"),
                "optimization_metric": self._get_value("optimization_metric")
            },
            "risk_limits": {
                "base_risk": self._get_value("base_risk"),
                "max_risk": self._get_value("max_risk"),
                "min_risk": self._get_value("min_risk"),
                "use_kelly": self._get_value("use_kelly"),
                "volatility_adjustment": self._get_value("volatility_adjustment"),
                "max_positions": self._get_value("max_positions"),
                "max_sector_risk": self._get_value("max_sector_risk"),
                "daily_loss_limit": self._get_value("daily_loss_limit"),
                "max_drawdown_limit": self._get_value("max_drawdown"),
                "circuit_breakers": self._get_value("circuit_breakers"),
                "auto_hedging": self._get_value("auto_hedging"),
                "hedging_threshold": self._get_value("hedging_threshold")
            },
            "ai": {
                "enabled": self._get_value("enable_ai"),
                "components": {
                    component.lower().replace(' ', '_'): 
                    self._get_value(f"ai_{component.lower().replace(' ', '_')}")
                    for component in self.ai_components
                },
                "training": {
                    "learning_rate": self._get_value("learning_rate"),
                    "batch_size": self._get_value("batch_size"),
                    "continuous_learning": self._get_value("continuous_learning"),
                    "auto_optimize_hyperparams": self._get_value("auto_optimize_hyperparams")
                },
                "models_dir": self._get_value("models_dir")
            },
            "network": {
                "use_proxy": self._get_value("use_proxy"),
                "proxy_host": self._get_value("proxy_host"),
                "proxy_port": self._get_value("proxy_port"),
                "timeout": self._get_value("connection_timeout"),
                "retry_attempts": self._get_value("retry_attempts"),
                "webhooks": {
                    "enabled": self._get_value("enable_webhooks"),
                    "url": self._get_value("webhook_url"),
                    "events": {
                        event.lower().replace(' ', '_'): 
                        self._get_value(f"webhook_{event.lower().replace(' ', '_')}")
                        for event in self.webhook_events
                    }
                }
            }
        }
        
        return config
        
    def accept(self):
        """Handle dialog acceptance."""
        # Get current configuration
        config = self._get_current_config()
        
        # Validate configuration
        if not self._validate_config(config):
            return
            
        # Update configuration
        self.config = config
        
        # Save configuration to file
        self._save_config_to_file()
        
        # Emit signal
        self.config_updated.emit(self.config)
        
        # Close dialog
        super().accept()
        
    def _validate_config(self, config) -> bool:
        """Validate configuration."""
        # Check base risk <= max risk
        if config["risk_limits"]["base_risk"] > config["risk_limits"]["max_risk"]:
            QMessageBox.warning(
                self,
                "Invalid Configuration",
                "Base risk per trade cannot be greater than max risk per trade."
            )
            return False
            
        # Check min risk <= base risk
        if config["risk_limits"]["min_risk"] > config["risk_limits"]["base_risk"]:
            QMessageBox.warning(
                self,
                "Invalid Configuration",
                "Min risk per trade cannot be greater than base risk per trade."
            )
            return False
            
        # Check that at least one strategy is active
        if not config["strategies"]["active_strategies"]:
            QMessageBox.warning(
                self,
                "Invalid Configuration",
                "At least one strategy must be active."
            )
            return False
            
        # Check that strategy weights sum to approximately 1.0
        strategy_weights = sum(config["strategies"]["weights"].values())
        if abs(strategy_weights - 1.0) > 0.01:
            QMessageBox.warning(
                self,
                "Invalid Configuration",
                f"Strategy weights should sum to 1.0 (current sum: {strategy_weights:.2f})."
            )
            return False
            
        # Check that models directory exists if AI is enabled
        if config["ai"]["enabled"] and not os.path.isdir(config["ai"]["models_dir"]):
            response = QMessageBox.question(
                self,
                "Models Directory",
                f"Models directory '{config['ai']['models_dir']}' does not exist. Create it?",
                QMessageBox.Yes | QMessageBox.No
            )
            
            if response == QMessageBox.Yes:
                try:
                    os.makedirs(config["ai"]["models_dir"], exist_ok=True)
                except Exception as e:
                    QMessageBox.critical(
                        self,
                        "Error",
                        f"Could not create models directory: {e}"
                    )
                    return False
            else:
                return False
                
        return True
        
    def _save_config_to_file(self):
        """Save configuration to file."""
        try:
            config_path = os.path.join(os.getcwd(), "config.json")
            with open(config_path, "w") as f:
                json.dump(self.config, f, indent=4)
                
            logging.info(f"Configuration saved to {config_path}")
            return True
        except Exception as e:
            logging.error(f"Error saving configuration: {e}")
            QMessageBox.critical(
                self,
                "Save Error",
                f"Could not save configuration: {e}"
            )
            return False
            
    def sizeHint(self) -> QSize:
        """Return a sensible size for the dialog."""
        return QSize(850, 650)


if __name__ == "__main__":
    # For standalone testing
    import sys
    from PyQt5.QtWidgets import QApplication
    
    app = QApplication(sys.argv)
    dialog = ConfigurationDialog()
    if dialog.exec_():
        print("Configuration updated:", dialog.config)
