# gui/dialogs.py

import os
import json
import time
import logging
from datetime import datetime, timedelta
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from PyQt5.QtWidgets import (
    QDialog, QDialogButtonBox, QFormLayout, QVBoxLayout, QHBoxLayout, 
    QLabel, QLineEdit, QPushButton, QCheckBox, QComboBox, QSpinBox, 
    QDoubleSpinBox, QTabWidget, QWidget, QGroupBox, QRadioButton,
    QTextEdit, QFileDialog, QMessageBox, QProgressBar, QDateEdit,
    QTimeEdit, QDateTimeEdit, QTableWidget, QTableWidgetItem, QHeaderView,
    QScrollArea, QSizePolicy, QFrame, QGridLayout, QButtonGroup, QSlider,
    QApplication, QStyle, QListWidget, QListWidgetItem, QTreeWidget, QTreeWidgetItem
)
from PyQt5.QtCore import (
    Qt, QSize, QTimer, QDateTime, QDate, QTime, QEvent, 
    pyqtSignal, QObject, QThread, QRegExp, QSettings
)
from PyQt5.QtGui import (
    QIcon, QPalette, QColor, QFont, QPixmap, QRegExpValidator,
    QPainter, QBrush, QPen, QCursor
)

# Try to import the error handling module for consistent error handling
try:
    from core.error_handling import ErrorHandler, ErrorCategory, ErrorSeverity, safe_execute
    HAVE_ERROR_HANDLING = True
except ImportError:
    HAVE_ERROR_HANDLING = False
    logging.warning("Error handling module not available in dialogs. Using basic error handling.")

class BaseDialog(QDialog):
    """
    Base dialog class with common functionality for all application dialogs.
    Provides standardized styling, error handling, and layout management.
    """
    
    def __init__(self, parent=None, title="Dialog", modal=True, 
                 flags=Qt.WindowCloseButtonHint | Qt.WindowTitleHint):
        """
        Initialize the base dialog.
        
        Args:
            parent: Parent widget
            title: Dialog title
            modal: Whether dialog should be modal
            flags: Window flags
        """
        super().__init__(parent, flags)
        self.setWindowTitle(title)
        self.setModal(modal)
        self._setup_appearance()
        
        # Layout setup
        self.main_layout = QVBoxLayout(self)
        self.main_layout.setContentsMargins(15, 15, 15, 15)
        self.main_layout.setSpacing(10)
        
        # Store reference to trading system if available in parent
        self.trading_system = self._get_trading_system()
        
    def _setup_appearance(self):
        """Set up consistent dialog appearance."""
        # Set minimum width
        self.setMinimumWidth(400)
        
        # Set stylesheet for consistent appearance
        self.setStyleSheet("""
            QDialog {
                background-color: #f5f5f5;
            }
            QLabel {
                font-size: 11pt;
            }
            QLabel#title {
                font-size: 14pt;
                font-weight: bold;
                color: #2c3e50;
            }
            QLabel#subtitle {
                font-size: 11pt;
                color: #7f8c8d;
            }
            QPushButton {
                background-color: #3498db;
                color: white;
                padding: 6px 14px;
                border-radius: 4px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #2980b9;
            }
            QPushButton:pressed {
                background-color: #1c6ea4;
            }
            QPushButton:disabled {
                background-color: #bdc3c7;
            }
            QPushButton#danger {
                background-color: #e74c3c;
            }
            QPushButton#danger:hover {
                background-color: #c0392b;
            }
            QPushButton#warning {
                background-color: #f39c12;
            }
            QPushButton#warning:hover {
                background-color: #d35400;
            }
            QPushButton#success {
                background-color: #2ecc71;
            }
            QPushButton#success:hover {
                background-color: #27ae60;
            }
            QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox, QDateEdit, QTimeEdit {
                padding: 5px;
                border: 1px solid #bdc3c7;
                border-radius: 3px;
            }
            QGroupBox {
                font-weight: bold;
                border: 1px solid #bdc3c7;
                border-radius: 5px;
                margin-top: 10px;
                padding-top: 15px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                subcontrol-position: top left;
                padding: 0 5px;
            }
        """)
    
    def _get_trading_system(self):
        """
        Recursively traverse parent widgets to find the trading system reference.
        
        Returns:
            Trading system object or None if not found
        """
        parent = self.parent()
        while parent:
            if hasattr(parent, 'trading_system'):
                return parent.trading_system
            parent = parent.parent()
        return None
    
    def add_title(self, title_text, subtitle_text=None):
        """
        Add title and optional subtitle to the dialog.
        
        Args:
            title_text: Main title text
            subtitle_text: Optional subtitle text
        """
        title_label = QLabel(title_text)
        title_label.setObjectName("title")
        self.main_layout.addWidget(title_label)
        
        if subtitle_text:
            subtitle_label = QLabel(subtitle_text)
            subtitle_label.setObjectName("subtitle")
            self.main_layout.addWidget(subtitle_label)
            
        # Add some spacing
        self.main_layout.addSpacing(10)
    
    def add_button_box(self, standard_buttons=QDialogButtonBox.Ok | QDialogButtonBox.Cancel):
        """
        Add standard button box to the dialog.
        
        Args:
            standard_buttons: Standard button configuration
        
        Returns:
            QDialogButtonBox instance
        """
        button_box = QDialogButtonBox(standard_buttons)
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)
        self.main_layout.addWidget(button_box)
        return button_box
    
    def add_form_layout(self):
        """
        Add form layout to the dialog.
        
        Returns:
            QFormLayout instance
        """
        form_layout = QFormLayout()
        form_layout.setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow)
        form_layout.setLabelAlignment(Qt.AlignRight | Qt.AlignVCenter)
        form_layout.setSpacing(10)
        self.main_layout.addLayout(form_layout)
        return form_layout
    
    def add_separator(self):
        """Add horizontal separator line to the dialog."""
        line = QFrame()
        line.setFrameShape(QFrame.HLine)
        line.setFrameShadow(QFrame.Sunken)
        self.main_layout.addWidget(line)
    
    def show_error(self, title, message):
        """
        Show error message box.
        
        Args:
            title: Error title
            message: Error message
        """
        if HAVE_ERROR_HANDLING:
            ErrorHandler.handle_error(
                Exception(message),
                category=ErrorCategory.SYSTEM,
                severity=ErrorSeverity.ERROR,
                context={"dialog": self.__class__.__name__, "title": title}
            )
        
        QMessageBox.critical(self, title, message)
    
    def show_warning(self, title, message):
        """
        Show warning message box.
        
        Args:
            title: Warning title
            message: Warning message
        """
        QMessageBox.warning(self, title, message)
    
    def show_info(self, title, message):
        """
        Show information message box.
        
        Args:
            title: Info title
            message: Info message
        """
        QMessageBox.information(self, title, message)
    
    def confirm(self, title, message, default_btn=QMessageBox.No):
        """
        Show confirmation dialog.
        
        Args:
            title: Dialog title
            message: Confirmation message
            default_btn: Default button (Yes/No)
        
        Returns:
            bool: True if confirmed, False otherwise
        """
        reply = QMessageBox.question(
            self, title, message, 
            QMessageBox.Yes | QMessageBox.No, 
            default_btn
        )
        return reply == QMessageBox.Yes


class LoginDialog(BaseDialog):
    """Dialog for user authentication."""
    
    def __init__(self, parent=None, api_keys_required=True, remember_credentials=True):
        """
        Initialize login dialog.
        
        Args:
            parent: Parent widget
            api_keys_required: Whether API keys are required
            remember_credentials: Whether to offer remembering credentials
        """
        super().__init__(parent, "Login to Trading System")
        self.api_keys_required = api_keys_required
        self.remember_credentials = remember_credentials
        
        # Setup UI
        self._setup_ui()
        
        # Load saved credentials if available
        self._load_saved_credentials()
    
    def _setup_ui(self):
        """Set up dialog UI components."""
        # Title and subtitle
        self.add_title("Trading System Login", "Please enter your credentials")
        
        # Form layout
        form_layout = self.add_form_layout()
        
        # Username
        self.username_edit = QLineEdit()
        self.username_edit.setPlaceholderText("Username")
        form_layout.addRow("Username:", self.username_edit)
        
        # Password
        self.password_edit = QLineEdit()
        self.password_edit.setEchoMode(QLineEdit.Password)
        self.password_edit.setPlaceholderText("Password")
        form_layout.addRow("Password:", self.password_edit)
        
        # API keys (if required)
        if self.api_keys_required:
            self.add_separator()
            
            # API key label
            api_label = QLabel("Exchange API Keys:")
            api_label.setObjectName("subtitle")
            self.main_layout.addWidget(api_label)
            
            # API keys form
            api_form = QFormLayout()
            
            # API Key
            self.api_key_edit = QLineEdit()
            self.api_key_edit.setPlaceholderText("API Key")
            api_form.addRow("API Key:", self.api_key_edit)
            
            # API Secret
            self.api_secret_edit = QLineEdit()
            self.api_secret_edit.setEchoMode(QLineEdit.Password)
            self.api_secret_edit.setPlaceholderText("API Secret")
            api_form.addRow("API Secret:", self.api_secret_edit)
            
            # Exchange selection
            self.exchange_combo = QComboBox()
            self.exchange_combo.addItems(["Binance", "Coinbase Pro", "Kraken", "Bitfinex", "Bybit"])
            api_form.addRow("Exchange:", self.exchange_combo)
            
            self.main_layout.addLayout(api_form)
        
        # Remember me checkbox
        if self.remember_credentials:
            self.remember_checkbox = QCheckBox("Remember credentials")
            self.main_layout.addWidget(self.remember_checkbox)
        
        # Add button box
        button_box = self.add_button_box()
        
        # Customize buttons
        login_button = button_box.button(QDialogButtonBox.Ok)
        login_button.setText("Login")
        login_button.setDefault(True)
        login_button.setAutoDefault(True)
        
        cancel_button = button_box.button(QDialogButtonBox.Cancel)
        cancel_button.setText("Cancel")
        
        # Add progress indicator
        self.progress_indicator = QLabel("Authenticating...")
        self.progress_indicator.setAlignment(Qt.AlignCenter)
        self.progress_indicator.setVisible(False)
        self.main_layout.addWidget(self.progress_indicator)
        
        # Connect signals
        button_box.accepted.disconnect()  # Disconnect default connection
        button_box.accepted.connect(self._on_login)
    
    def _load_saved_credentials(self):
        """Load saved credentials if available."""
        if not self.remember_credentials:
            return
            
        settings = QSettings("TradingSystem", "Credentials")
        
        username = settings.value("username", "")
        password = settings.value("password", "")
        remember = settings.value("remember", False, type=bool)
        
        self.username_edit.setText(username)
        self.password_edit.setText(password)
        self.remember_checkbox.setChecked(remember)
        
        if self.api_keys_required:
            api_key = settings.value("api_key", "")
            api_secret = settings.value("api_secret", "")
            exchange = settings.value("exchange", "")
            
            self.api_key_edit.setText(api_key)
            self.api_secret_edit.setText(api_secret)
            
            if exchange:
                index = self.exchange_combo.findText(exchange)
                if index >= 0:
                    self.exchange_combo.setCurrentIndex(index)
    
    def _save_credentials(self):
        """Save credentials if remember me is checked."""
        if not self.remember_credentials or not self.remember_checkbox.isChecked():
            return
            
        settings = QSettings("TradingSystem", "Credentials")
        
        settings.setValue("username", self.username_edit.text())
        settings.setValue("password", self.password_edit.text())
        settings.setValue("remember", self.remember_checkbox.isChecked())
        
        if self.api_keys_required:
            settings.setValue("api_key", self.api_key_edit.text())
            settings.setValue("api_secret", self.api_secret_edit.text())
            settings.setValue("exchange", self.exchange_combo.currentText())
    
    def _on_login(self):
        """Handle login button click."""
        # Get credentials
        username = self.username_edit.text()
        password = self.password_edit.text()
        
        # Validate input
        if not username or not password:
            self.show_warning("Incomplete Information", "Please enter both username and password.")
            return
        
        if self.api_keys_required:
            api_key = self.api_key_edit.text()
            api_secret = self.api_secret_edit.text()
            exchange = self.exchange_combo.currentText()
            
            if not api_key or not api_secret:
                self.show_warning("Incomplete Information", "Please enter both API Key and API Secret.")
                return
        
        # Show progress indicator
        self.progress_indicator.setVisible(True)
        QApplication.processEvents()
        
        # Simulate authentication delay
        start_time = time.time()
        
        # Perform authentication
        try:
            # In a real system, this would call the actual authentication service
            success = self._authenticate(username, password)
            
            # Ensure minimum delay for UX
            elapsed = time.time() - start_time
            if elapsed < 1.0:
                time.sleep(1.0 - elapsed)
            
            if success:
                # Save credentials if requested
                self._save_credentials()
                
                # Accept dialog
                self.accept()
            else:
                self.progress_indicator.setVisible(False)
                self.show_error("Authentication Failed", "Invalid username or password.")
                
        except Exception as e:
            self.progress_indicator.setVisible(False)
            self.show_error("Authentication Error", f"An error occurred during authentication: {str(e)}")
    
    def _authenticate(self, username, password):
        """
        Authenticate user with the provided credentials.
        In a real system, this would verify credentials with an authentication service.
        
        Args:
            username: Username
            password: Password
        
        Returns:
            bool: True if authentication succeeded, False otherwise
        """
        # Simulate authentication
        # In a real system, this would verify credentials with an authentication service
        
        # Demo credentials for testing
        if username == "admin" and password == "password":
            return True
            
        # Connect to trading system if available
        if self.trading_system and hasattr(self.trading_system, 'authenticate'):
            return self.trading_system.authenticate(username, password)
            
        # For demo purposes, accept any non-empty credentials
        return bool(username and password)
    
    def get_credentials(self):
        """
        Get entered credentials.
        
        Returns:
            dict: Credential information
        """
        credentials = {
            "username": self.username_edit.text(),
            "password": self.password_edit.text(),
            "remember": self.remember_checkbox.isChecked() if self.remember_credentials else False
        }
        
        if self.api_keys_required:
            credentials.update({
                "api_key": self.api_key_edit.text(),
                "api_secret": self.api_secret_edit.text(),
                "exchange": self.exchange_combo.currentText()
            })
            
        return credentials


class ConfirmTradeDialog(BaseDialog):
    """Dialog for confirming trade execution."""
    
    def __init__(self, parent=None, trade_details=None):
        """
        Initialize trade confirmation dialog.
        
        Args:
            parent: Parent widget
            trade_details: Dictionary with trade details
        """
        super().__init__(parent, "Confirm Trade")
        self.trade_details = trade_details or {}
        
        # Set up UI
        self._setup_ui()
    
    def _setup_ui(self):
        """Set up dialog UI components."""
        # Title based on trade type
        action = self.trade_details.get('action', 'Unknown').capitalize()
        symbol = self.trade_details.get('symbol', 'Unknown')
        
        self.add_title(f"Confirm {action} Order", f"You are about to {action.lower()} {symbol}")
        
        # Trade details
        details_group = QGroupBox("Order Details")
        details_layout = QFormLayout(details_group)
        
        # Symbol
        symbol_label = QLabel(self.trade_details.get('symbol', 'Unknown'))
        symbol_label.setStyleSheet("font-weight: bold; font-size: 12pt;")
        details_layout.addRow("Symbol:", symbol_label)
        
        # Action
        action_text = self.trade_details.get('action', 'unknown').upper()
        action_label = QLabel(action_text)
        action_color = "#27ae60" if action_text == "BUY" else "#e74c3c"
        action_label.setStyleSheet(f"color: {action_color}; font-weight: bold;")
        details_layout.addRow("Action:", action_label)
        
        # Type
        order_type = self.trade_details.get('type', 'market').capitalize()
        details_layout.addRow("Order Type:", QLabel(order_type))
        
        # Price
        price_text = str(self.trade_details.get('price', 'Market Price'))
        details_layout.addRow("Price:", QLabel(price_text))
        
        # Amount
        amount = self.trade_details.get('amount', 0)
        amount_label = QLabel(f"{amount:.8f}")
        details_layout.addRow("Amount:", amount_label)
        
        # Total cost
        if 'price' in self.trade_details and self.trade_details['price'] is not None:
            total = float(self.trade_details['price']) * float(amount)
            details_layout.addRow("Total:", QLabel(f"${total:.2f}"))
        
        # Stop Loss / Take Profit
        if self.trade_details.get('stop_loss'):
            details_layout.addRow("Stop Loss:", QLabel(str(self.trade_details['stop_loss'])))
            
        if self.trade_details.get('take_profit'):
            details_layout.addRow("Take Profit:", QLabel(str(self.trade_details['take_profit'])))
        
        self.main_layout.addWidget(details_group)
        
        # Risk assessment
        if self.trading_system and hasattr(self.trading_system, 'risk_manager'):
            risk_group = QGroupBox("Risk Assessment")
            risk_layout = QFormLayout(risk_group)
            
            # Portfolio percentage
            portfolio_value = 10000  # Placeholder - get from trading system
            trade_value = float(self.trade_details.get('price', 0)) * float(amount)
            portfolio_percentage = (trade_value / portfolio_value) * 100 if portfolio_value else 0
            
            risk_layout.addRow("Portfolio Percentage:", QLabel(f"{portfolio_percentage:.2f}%"))
            
            # Risk level
            risk_level = "Low"
            risk_color = "#27ae60"  # Green
            
            if portfolio_percentage > 10:
                risk_level = "High"
                risk_color = "#e74c3c"  # Red
            elif portfolio_percentage > 5:
                risk_level = "Medium"
                risk_color = "#f39c12"  # Orange
                
            risk_label = QLabel(risk_level)
            risk_label.setStyleSheet(f"color: {risk_color}; font-weight: bold;")
            risk_layout.addRow("Risk Level:", risk_label)
            
            self.main_layout.addWidget(risk_group)
        
        # Add disclaimer
        disclaimer = QLabel("Please verify all details before confirming this trade.")
        disclaimer.setStyleSheet("color: #7f8c8d; font-style: italic;")
        disclaimer.setWordWrap(True)
        self.main_layout.addWidget(disclaimer)
        
        # Add button box
        button_box = self.add_button_box()
        
        # Customize buttons
        confirm_button = button_box.button(QDialogButtonBox.Ok)
        confirm_button.setText("Confirm Trade")
        confirm_button.setObjectName("success" if action_text == "BUY" else "danger")
        
        cancel_button = button_box.button(QDialogButtonBox.Cancel)
        cancel_button.setText("Cancel")
    
    def get_trade_details(self):
        """
        Get trade details.
        
        Returns:
            dict: Trade details
        """
        return self.trade_details


class TradeSettingsDialog(BaseDialog):
    """Dialog for configuring trading settings."""
    
    def __init__(self, parent=None, trading_system=None):
        """
        Initialize trade settings dialog.
        
        Args:
            parent: Parent widget
            trading_system: Trading system reference
        """
        super().__init__(parent, "Trading Settings")
        if trading_system:
            self.trading_system = trading_system
            
        self._setup_ui()
        self._load_current_settings()
    
    def _setup_ui(self):
        """Set up dialog UI components."""
        self.add_title("Trading Settings", "Configure system-wide trading parameters")
        
        # Create tabs
        tab_widget = QTabWidget()
        
        # General Settings Tab
        general_tab = QWidget()
        general_layout = QVBoxLayout(general_tab)
        
        # Trading Mode Group
        mode_group = QGroupBox("Trading Mode")
        mode_layout = QVBoxLayout(mode_group)
        
        # Trading Mode options
        self.live_radio = QRadioButton("Live Trading")
        self.paper_radio = QRadioButton("Paper Trading")
        self.backtest_radio = QRadioButton("Backtesting")
        
        mode_layout.addWidget(self.live_radio)
        mode_layout.addWidget(self.paper_radio)
        mode_layout.addWidget(self.backtest_radio)
        
        general_layout.addWidget(mode_group)
        
        # Default Order Settings
        order_group = QGroupBox("Default Order Settings")
        order_layout = QFormLayout(order_group)
        
        # Default order type
        self.order_type_combo = QComboBox()
        self.order_type_combo.addItems(["Market", "Limit", "Stop", "Stop Limit"])
        order_layout.addRow("Default Order Type:", self.order_type_combo)
        
        # Default order size
        self.order_size_spin = QDoubleSpinBox()
        self.order_size_spin.setRange(0.001, 100)
        self.order_size_spin.setSingleStep(0.01)
        self.order_size_spin.setDecimals(3)
        order_layout.addRow("Default Order Size:", self.order_size_spin)
        
        # Size type (absolute or percentage)
        self.size_type_combo = QComboBox()
        self.size_type_combo.addItems(["Fixed Size", "Percentage of Balance"])
        order_layout.addRow("Size Type:", self.size_type_combo)
        
        general_layout.addWidget(order_group)
        
        # Auto Trading Group
        auto_group = QGroupBox("Automated Trading")
        auto_layout = QFormLayout(auto_group)
        
        # Enable auto trading
        self.auto_trading_check = QCheckBox("Enable automated trading")
        auto_layout.addRow("", self.auto_trading_check)
        
        # Confirmation required
        self.confirm_trades_check = QCheckBox("Require confirmation for automated trades")
        auto_layout.addRow("", self.confirm_trades_check)
        
        general_layout.addWidget(auto_group)
        
        # Add to tab widget
        tab_widget.addTab(general_tab, "General")
        
        # Risk Management Tab
        risk_tab = QWidget()
        risk_layout = QVBoxLayout(risk_tab)
        
        # Position Sizing
        position_group = QGroupBox("Position Sizing")
        position_layout = QFormLayout(position_group)
        
        # Risk per trade
        self.risk_per_trade_spin = QDoubleSpinBox()
        self.risk_per_trade_spin.setRange(0.1, 10)
        self.risk_per_trade_spin.setSingleStep(0.1)
        self.risk_per_trade_spin.setDecimals(1)
        self.risk_per_trade_spin.setSuffix("%")
        position_layout.addRow("Risk Per Trade:", self.risk_per_trade_spin)
        
        # Max risk per asset
        self.max_asset_risk_spin = QDoubleSpinBox()
        self.max_asset_risk_spin.setRange(1, 50)
        self.max_asset_risk_spin.setSingleStep(1)
        self.max_asset_risk_spin.setSuffix("%")
        position_layout.addRow("Max Risk Per Asset:", self.max_asset_risk_spin)
        
        # Max portfolio risk
        self.max_portfolio_risk_spin = QDoubleSpinBox()
        self.max_portfolio_risk_spin.setRange(5, 100)
        self.max_portfolio_risk_spin.setSingleStep(5)
        self.max_portfolio_risk_spin.setSuffix("%")
        position_layout.addRow("Max Portfolio Risk:", self.max_portfolio_risk_spin)
        
        risk_layout.addWidget(position_group)
        
        # Stop Loss / Take Profit
        sltp_group = QGroupBox("Stop Loss / Take Profit")
        sltp_layout = QFormLayout(sltp_group)
        
        # Default SL percentage
        self.default_sl_spin = QDoubleSpinBox()
        self.default_sl_spin.setRange(0.5, 20)
        self.default_sl_spin.setSingleStep(0.5)
        self.default_sl_spin.setDecimals(1)
        self.default_sl_spin.setSuffix("%")
        sltp_layout.addRow("Default Stop Loss:", self.default_sl_spin)
        
        # Default TP percentage
        self.default_tp_spin = QDoubleSpinBox()
        self.default_tp_spin.setRange(0.5, 50)
        self.default_tp_spin.setSingleStep(0.5)
        self.default_tp_spin.setDecimals(1)
        self.default_tp_spin.setSuffix("%")
        sltp_layout.addRow("Default Take Profit:", self.default_tp_spin)
        
        # Automatic SL/TP
        self.auto_sltp_check = QCheckBox("Automatically set Stop Loss and Take Profit")
        sltp_layout.addRow("", self.auto_sltp_check)
        
        risk_layout.addWidget(sltp_group)
        
        # Advanced Risk Controls
        advanced_group = QGroupBox("Advanced Risk Controls")
        advanced_layout = QFormLayout(advanced_group)
        
        # Max drawdown
        self.max_drawdown_spin = QDoubleSpinBox()
        self.max_drawdown_spin.setRange(5, 50)
        self.max_drawdown_spin.setSingleStep(1)
        self.max_drawdown_spin.setSuffix("%")
        advanced_layout.addRow("Max Drawdown Limit:", self.max_drawdown_spin)
        
        # Daily loss limit
        self.daily_loss_spin = QDoubleSpinBox()
        self.daily_loss_spin.setRange(1, 20)
        self.daily_loss_spin.setSingleStep(0.5)
        self.daily_loss_spin.setSuffix("%")
        advanced_layout.addRow("Daily Loss Limit:", self.daily_loss_spin)
        
        # Circuit breaker
        self.circuit_breaker_check = QCheckBox("Enable circuit breaker")
        advanced_layout.addRow("", self.circuit_breaker_check)
        
        risk_layout.addWidget(advanced_group)
        
        # Add to tab widget
        tab_widget.addTab(risk_tab, "Risk Management")
        
        # Exchange Settings Tab
        exchange_tab = QWidget()
        exchange_layout = QVBoxLayout(exchange_tab)
        
        # API Connection
        api_group = QGroupBox("API Connection")
        api_layout = QFormLayout(api_group)
        
        # Exchange selection
        self.exchange_combo = QComboBox()
        self.exchange_combo.addItems(["Binance", "Coinbase Pro", "Kraken", "Bitfinex", "Bybit"])
        api_layout.addRow("Exchange:", self.exchange_combo)
        
        # API Key
        self.api_key_edit = QLineEdit()
        self.api_key_edit.setPlaceholderText("Enter API Key")
        api_layout.addRow("API Key:", self.api_key_edit)
        
        # API Secret
        self.api_secret_edit = QLineEdit()
        self.api_secret_edit.setEchoMode(QLineEdit.Password)
        self.api_secret_edit.setPlaceholderText("Enter API Secret")
        api_layout.addRow("API Secret:", self.api_secret_edit)
        
        # Test API Connection button
        self.test_api_btn = QPushButton("Test API Connection")
        self.test_api_btn.clicked.connect(self._test_api_connection)
        api_layout.addRow("", self.test_api_btn)
        
        exchange_layout.addWidget(api_group)
        
        # Exchange Options
        options_group = QGroupBox("Exchange Options")
        options_layout = QFormLayout(options_group)
        
        # Request rate limit
        self.rate_limit_spin = QSpinBox()
        self.rate_limit_spin.setRange(1, 20)
        self.rate_limit_spin.setSuffix(" requests/second")
        options_layout.addRow("Rate Limit:", self.rate_limit_spin)
        
        # Retry attempts
        self.retry_spin = QSpinBox()
        self.retry_spin.setRange(0, 10)
        options_layout.addRow("Retry Attempts:", self.retry_spin)
        
        # Request timeout
        self.timeout_spin = QSpinBox()
        self.timeout_spin.setRange(1, 60)
        self.timeout_spin.setSuffix(" seconds")
        options_layout.addRow("Request Timeout:", self.timeout_spin)
        
        exchange_layout.addWidget(options_group)
        
        # Add to tab widget
        tab_widget.addTab(exchange_tab, "Exchange")
        
        # AI Settings Tab
        ai_tab = QWidget()
        ai_layout = QVBoxLayout(ai_tab)
        
        # AI Controls
        ai_control_group = QGroupBox("AI Trading Controls")
        ai_control_layout = QFormLayout(ai_control_group)
        
        # Enable AI
        self.enable_ai_check = QCheckBox("Enable AI-assisted trading")
        ai_control_layout.addRow("", self.enable_ai_check)
        
        # Confidence threshold
        self.confidence_spin = QDoubleSpinBox()
        self.confidence_spin.setRange(0.5, 1.0)
        self.confidence_spin.setSingleStep(0.05)
        self.confidence_spin.setDecimals(2)
        ai_control_layout.addRow("Confidence Threshold:", self.confidence_spin)
        
        # AI control level
        self.ai_control_combo = QComboBox()
        self.ai_control_combo.addItems([
            "Advisory Only", 
            "Semi-Autonomous (Confirmation Required)", 
            "Fully Autonomous"
        ])
        ai_control_layout.addRow("AI Control Level:", self.ai_control_combo)
        
        ai_layout.addWidget(ai_control_group)
        
        # Model Settings
        model_group = QGroupBox("Model Settings")
        model_layout = QFormLayout(model_group)
        
        # Available models
        model_types = [
            "Order Flow Prediction", 
            "Market Making", 
            "Trade Timing", 
            "Trade Exit", 
            "Portfolio Allocation"
        ]
        
        # Create checkboxes for each model
        self.model_checkboxes = {}
        for model_type in model_types:
            checkbox = QCheckBox(f"Enable {model_type}")
            checkbox.setChecked(True)
            model_layout.addRow("", checkbox)
            self.model_checkboxes[model_type] = checkbox
        
        # Training frequency
        self.training_combo = QComboBox()
        self.training_combo.addItems([
            "Daily", 
            "Weekly", 
            "Monthly", 
            "Manual Only"
        ])
        model_layout.addRow("Training Frequency:", self.training_combo)
        
        ai_layout.addWidget(model_group)
        
        # Add to tab widget
        tab_widget.addTab(ai_tab, "AI Settings")
        
        # Add tabs to main layout
        self.main_layout.addWidget(tab_widget)
        
        # Add button box
        button_box = self.add_button_box()
        
        # Customize buttons
        save_button = button_box.button(QDialogButtonBox.Ok)
        save_button.setText("Save Settings")
        save_button.setObjectName("success")
        
        cancel_button = button_box.button(QDialogButtonBox.Cancel)
        cancel_button.setText("Cancel")
        
        # Add reset button
        reset_button = QPushButton("Reset to Defaults")
        reset_button.setObjectName("warning")
        reset_button.clicked.connect(self._reset_to_defaults)
        button_box.addButton(reset_button, QDialogButtonBox.ResetRole)
    
    def _load_current_settings(self):
        """Load current settings from trading system or settings storage."""
        # Default values
        self.paper_radio.setChecked(True)
        self.order_type_combo.setCurrentText("Market")
        self.order_size_spin.setValue(0.01)
        self.size_type_combo.setCurrentText("Fixed Size")
        self.auto_trading_check.setChecked(False)
        self.confirm_trades_check.setChecked(True)
        self.risk_per_trade_spin.setValue(2.0)
        self.max_asset_risk_spin.setValue(10.0)
        self.max_portfolio_risk_spin.setValue(20.0)
        self.default_sl_spin.setValue(5.0)
        self.default_tp_spin.setValue(10.0)
        self.auto_sltp_check.setChecked(True)
        self.max_drawdown_spin.setValue(20.0)
        self.daily_loss_spin.setValue(5.0)
        self.circuit_breaker_check.setChecked(True)
        self.exchange_combo.setCurrentText("Binance")
        self.rate_limit_spin.setValue(10)
        self.retry_spin.setValue(3)
        self.timeout_spin.setValue(30)
        self.enable_ai_check.setChecked(True)
        self.confidence_spin.setValue(0.7)
        self.ai_control_combo.setCurrentText("Semi-Autonomous (Confirmation Required)")
        self.training_combo.setCurrentText("Weekly")
        
        # Load API keys from settings (if saved)
        settings = QSettings("TradingSystem", "Settings")
        self.api_key_edit.setText(settings.value("api_key", ""))
        self.api_secret_edit.setText(settings.value("api_secret", ""))
        
        # If trading system is available, load settings from it
        if self.trading_system:
            config = getattr(self.trading_system, 'config', {})
            
            # Load trading mode
            mode = config.get('trading', {}).get('mode', 'paper')
            if mode == 'live':
                self.live_radio.setChecked(True)
            elif mode == 'paper':
                self.paper_radio.setChecked(True)
            elif mode == 'backtest':
                self.backtest_radio.setChecked(True)
                
            # Load order settings
            order_settings = config.get('trading', {}).get('default_order', {})
            if order_settings:
                self.order_type_combo.setCurrentText(order_settings.get('type', 'Market').capitalize())
                self.order_size_spin.setValue(order_settings.get('size', 0.01))
                self.size_type_combo.setCurrentText(
                    "Percentage of Balance" if order_settings.get('size_type') == 'percentage' else "Fixed Size"
                )
            
            # Load auto trading settings
            auto_settings = config.get('trading', {}).get('automated', {})
            if auto_settings:
                self.auto_trading_check.setChecked(auto_settings.get('enabled', False))
                self.confirm_trades_check.setChecked(auto_settings.get('require_confirmation', True))
            
            # Load risk management settings
            risk_settings = config.get('risk', {})
            if risk_settings:
                self.risk_per_trade_spin.setValue(risk_settings.get('risk_per_trade', 2.0) * 100)
                self.max_asset_risk_spin.setValue(risk_settings.get('max_asset_risk', 0.1) * 100)
                self.max_portfolio_risk_spin.setValue(risk_settings.get('max_portfolio_risk', 0.2) * 100)
                self.default_sl_spin.setValue(risk_settings.get('default_stop_loss', 0.05) * 100)
                self.default_tp_spin.setValue(risk_settings.get('default_take_profit', 0.1) * 100)
                self.auto_sltp_check.setChecked(risk_settings.get('auto_sltp', True))
                self.max_drawdown_spin.setValue(risk_settings.get('max_drawdown', 0.2) * 100)
                self.daily_loss_spin.setValue(risk_settings.get('daily_loss_limit', 0.05) * 100)
                self.circuit_breaker_check.setChecked(risk_settings.get('circuit_breaker', True))
            
            # Load exchange settings
            exchange_settings = config.get('exchange', {})
            if exchange_settings:
                exchange_name = exchange_settings.get('name', 'Binance')
                index = self.exchange_combo.findText(exchange_name, Qt.MatchFixedString)
                if index >= 0:
                    self.exchange_combo.setCurrentIndex(index)
                
                self.rate_limit_spin.setValue(exchange_settings.get('rate_limit', 10))
                self.retry_spin.setValue(exchange_settings.get('retry_attempts', 3))
                self.timeout_spin.setValue(exchange_settings.get('timeout', 30))
            
            # Load AI settings
            ai_settings = config.get('ai', {})
            if ai_settings:
                self.enable_ai_check.setChecked(ai_settings.get('enabled', True))
                self.confidence_spin.setValue(ai_settings.get('confidence_threshold', 0.7))
                
                control_level = ai_settings.get('control_level', 'semi')
                if control_level == 'advisory':
                    self.ai_control_combo.setCurrentText("Advisory Only")
                elif control_level == 'semi':
                    self.ai_control_combo.setCurrentText("Semi-Autonomous (Confirmation Required)")
                elif control_level == 'full':
                    self.ai_control_combo.setCurrentText("Fully Autonomous")
                
                model_settings = ai_settings.get('models', {})
                for model_type, checkbox in self.model_checkboxes.items():
                    model_key = model_type.lower().replace(' ', '_')
                    checkbox.setChecked(model_settings.get(model_key, {}).get('enabled', True))
                
                training_freq = ai_settings.get('training_frequency', 'weekly')
                self.training_combo.setCurrentText(training_freq.capitalize())
    
    def _reset_to_defaults(self):
        """Reset all settings to default values."""
        if self.confirm("Reset Settings", "Are you sure you want to reset all settings to defaults?"):
            self._load_current_settings()  # This reloads default settings
    
    def _test_api_connection(self):
        """Test API connection with current credentials."""
        # Get API details
        exchange = self.exchange_combo.currentText()
        api_key = self.api_key_edit.text()
        api_secret = self.api_secret_edit.text()
        
        if not api_key or not api_secret:
            self.show_warning("Missing Credentials", "Please enter both API Key and API Secret.")
            return
        
        # Show testing message
        self.test_api_btn.setText("Testing...")
        self.test_api_btn.setEnabled(False)
        QApplication.processEvents()
        
        # Simulate API test
        time.sleep(1.5)
        
        # Reset button state
        self.test_api_btn.setText("Test API Connection")
        self.test_api_btn.setEnabled(True)
        
        # Show result (success for demo)
        self.show_info("Connection Successful", f"Successfully connected to {exchange} API.")
    
    def get_settings(self):
        """
        Get all settings as a dictionary.
        
        Returns:
            dict: All settings values
        """
        # Determine trading mode
        if self.live_radio.isChecked():
            mode = 'live'
        elif self.paper_radio.isChecked():
            mode = 'paper'
        else:
            mode = 'backtest'
        
        # Determine size type
        size_type = 'percentage' if self.size_type_combo.currentText() == "Percentage of Balance" else 'fixed'
        
        # Determine AI control level
        control_level = self.ai_control_combo.currentText()
        if control_level == "Advisory Only":
            ai_control = 'advisory'
        elif control_level == "Semi-Autonomous (Confirmation Required)":
            ai_control = 'semi'
        else:
            ai_control = 'full'
        
        # Build settings dictionary
        settings = {
            'trading': {
                'mode': mode,
                'default_order': {
                    'type': self.order_type_combo.currentText().lower(),
                    'size': self.order_size_spin.value(),
                    'size_type': size_type
                },
                'automated': {
                    'enabled': self.auto_trading_check.isChecked(),
                    'require_confirmation': self.confirm_trades_check.isChecked()
                }
            },
            'risk': {
                'risk_per_trade': self.risk_per_trade_spin.value() / 100.0,
                'max_asset_risk': self.max_asset_risk_spin.value() / 100.0,
                'max_portfolio_risk': self.max_portfolio_risk_spin.value() / 100.0,
                'default_stop_loss': self.default_sl_spin.value() / 100.0,
                'default_take_profit': self.default_tp_spin.value() / 100.0,
                'auto_sltp': self.auto_sltp_check.isChecked(),
                'max_drawdown': self.max_drawdown_spin.value() / 100.0,
                'daily_loss_limit': self.daily_loss_spin.value() / 100.0,
                'circuit_breaker': self.circuit_breaker_check.isChecked()
            },
            'exchange': {
                'name': self.exchange_combo.currentText(),
                'api_key': self.api_key_edit.text(),
                'api_secret': self.api_secret_edit.text(),
                'rate_limit': self.rate_limit_spin.value(),
                'retry_attempts': self.retry_spin.value(),
                'timeout': self.timeout_spin.value()
            },
            'ai': {
                'enabled': self.enable_ai_check.isChecked(),
                'confidence_threshold': self.confidence_spin.value(),
                'control_level': ai_control,
                'training_frequency': self.training_combo.currentText().lower(),
                'models': {}
            }
        }
        
        # Add model settings
        for model_type, checkbox in self.model_checkboxes.items():
            model_key = model_type.lower().replace(' ', '_')
            settings['ai']['models'][model_key] = {
                'enabled': checkbox.isChecked()
            }
        
        return settings
    
    def accept(self):
        """Handle OK button click."""
        # Save API credentials
        settings = QSettings("TradingSystem", "Settings")
        settings.setValue("api_key", self.api_key_edit.text())
        settings.setValue("api_secret", self.api_secret_edit.text())
        
        # Call parent accept method
        super().accept()


class StrategyConfigDialog(BaseDialog):
    """Dialog for configuring trading strategy parameters."""
    
    def __init__(self, parent=None, strategy_name=None, strategy_params=None):
        """
        Initialize strategy configuration dialog.
        
        Args:
            parent: Parent widget
            strategy_name: Name of the strategy to configure
            strategy_params: Current strategy parameters
        """
        super().__init__(parent, f"Configure {strategy_name or 'Strategy'}")
        self.strategy_name = strategy_name or "Unknown"
        self.strategy_params = strategy_params or {}
        
        self._setup_ui()
        self._load_parameters()
    
    def _setup_ui(self):
        """Set up dialog UI components."""
        self.add_title(f"Configure {self.strategy_name}", "Adjust strategy parameters")
        
        # Parameter form
        self.params_form = self.add_form_layout()
        
        # Parameter widgets
        self.param_widgets = {}
        
        # Add button box
        button_box = self.add_button_box()
        
        # Add Apply button
        apply_button = QPushButton("Apply")
        apply_button.clicked.connect(self._apply_changes)
        button_box.addButton(apply_button, QDialogButtonBox.ApplyRole)
        
        # Add Reset button
        reset_button = QPushButton("Reset")
        reset_button.clicked.connect(self._reset_parameters)
        button_box.addButton(reset_button, QDialogButtonBox.ResetRole)
    
    def _load_parameters(self):
        """Load strategy parameters into form."""
        # Clear existing widgets
        while self.params_form.rowCount() > 0:
            self.params_form.removeRow(0)
        
        self.param_widgets = {}
        
        # If no strategy parameters, show message
        if not self.strategy_params:
            self.params_form.addRow(QLabel("No parameters available for this strategy."))
            return
        
        # Add widgets for each parameter
        for param_name, param_value in self.strategy_params.items():
            # Skip internal parameters
            if param_name.startswith('_'):
                continue
            
            # Create label
            label = QLabel(param_name.replace('_', ' ').title() + ":")
            
            # Create appropriate widget based on parameter type
            if isinstance(param_value, bool):
                widget = QCheckBox()
                widget.setChecked(param_value)
            elif isinstance(param_value, int):
                widget = QSpinBox()
                widget.setRange(-1000000, 1000000)
                widget.setValue(param_value)
            elif isinstance(param_value, float):
                widget = QDoubleSpinBox()
                widget.setRange(-1000000, 1000000)
                widget.setDecimals(5)
                widget.setValue(param_value)
            elif isinstance(param_value, str) and param_value.lower() in ['buy', 'sell', 'long', 'short']:
                widget = QComboBox()
                widget.addItems(['Buy', 'Sell', 'Long', 'Short'])
                widget.setCurrentText(param_value.capitalize())
            else:
                widget = QLineEdit(str(param_value))
            
            # Add to form and store reference
            self.params_form.addRow(label, widget)
            self.param_widgets[param_name] = widget
    
    def _reset_parameters(self):
        """Reset parameters to original values."""
        self._load_parameters()
    
    def _apply_changes(self):
        """Apply parameter changes without closing dialog."""
        new_params = self.get_parameters()
        
        # Update local copy
        self.strategy_params = new_params
        
        # If trading system is available, update strategy parameters
        if self.trading_system and hasattr(self.trading_system, 'update_strategy_parameters'):
            self.trading_system.update_strategy_parameters(self.strategy_name, new_params)
            self.show_info("Parameters Updated", "Strategy parameters have been updated.")
    
    def get_parameters(self):
        """
        Get parameter values from form widgets.
        
        Returns:
            dict: Updated parameter values
        """
        new_params = {}
        
        for param_name, widget in self.param_widgets.items():
            # Get value based on widget type
            if isinstance(widget, QCheckBox):
                value = widget.isChecked()
            elif isinstance(widget, QSpinBox):
                value = widget.value()
            elif isinstance(widget, QDoubleSpinBox):
                value = widget.value()
            elif isinstance(widget, QComboBox):
                value = widget.currentText().lower()
            elif isinstance(widget, QLineEdit):
                # Try to convert to number if possible
                text_value = widget.text()
                try:
                    if '.' in text_value:
                        value = float(text_value)
                    else:
                        value = int(text_value)
                except ValueError:
                    value = text_value
            else:
                value = None
            
            new_params[param_name] = value
        
        return new_params


class BacktestConfigDialog(BaseDialog):
    """Dialog for configuring backtest parameters."""
    
    def __init__(self, parent=None, strategies=None):
        """
        Initialize backtest configuration dialog.
        
        Args:
            parent: Parent widget
            strategies: List of available strategies
        """
        super().__init__(parent, "Backtest Configuration")
        self.strategies = strategies or []
        
        self._setup_ui()
    
    def _setup_ui(self):
        """Set up dialog UI components."""
        self.add_title("Backtest Configuration", "Configure and run strategy backtests")
        
        # Date range
        date_group = QGroupBox("Date Range")
        date_layout = QFormLayout(date_group)
        
        # Start date
        self.start_date = QDateEdit()
        self.start_date.setCalendarPopup(True)
        self.start_date.setDate(QDate.currentDate().addMonths(-3))
        date_layout.addRow("Start Date:", self.start_date)
        
        # End date
        self.end_date = QDateEdit()
        self.end_date.setCalendarPopup(True)
        self.end_date.setDate(QDate.currentDate())
        date_layout.addRow("End Date:", self.end_date)
        
        self.main_layout.addWidget(date_group)
        
        # Strategy selection
        strategy_group = QGroupBox("Strategy Selection")
        strategy_layout = QVBoxLayout(strategy_group)
        
        # Strategy list
        self.strategy_list = QListWidget()
        
        # Add strategies
        for strategy_name in self.strategies:
            item = QListWidgetItem(strategy_name)
            item.setCheckState(Qt.Unchecked)
            self.strategy_list.addItem(item)
        
        strategy_layout.addWidget(self.strategy_list)
        
        # Select all / None buttons
        select_layout = QHBoxLayout()
        
        select_all_btn = QPushButton("Select All")
        select_all_btn.clicked.connect(self._select_all_strategies)
        select_layout.addWidget(select_all_btn)
        
        select_none_btn = QPushButton("Select None")
        select_none_btn.clicked.connect(self._select_no_strategies)
        select_layout.addWidget(select_none_btn)
        
        strategy_layout.addLayout(select_layout)
        
        self.main_layout.addWidget(strategy_group)
        
        # Initial capital
        capital_group = QGroupBox("Capital Settings")
        capital_layout = QFormLayout(capital_group)
        
        # Initial capital
        self.initial_capital_spin = QDoubleSpinBox()
        self.initial_capital_spin.setRange(100, 1000000)
        self.initial_capital_spin.setValue(10000)
        self.initial_capital_spin.setPrefix("$")
        self.initial_capital_spin.setDecimals(2)
        capital_layout.addRow("Initial Capital:", self.initial_capital_spin)
        
        # Commission
        self.commission_spin = QDoubleSpinBox()
        self.commission_spin.setRange(0, 1)
        self.commission_spin.setValue(0.001)
        self.commission_spin.setSingleStep(0.001)
        self.commission_spin.setDecimals(3)
        self.commission_spin.setSuffix("%")
        capital_layout.addRow("Commission Rate:", self.commission_spin)
        
        self.main_layout.addWidget(capital_group)
        
        # Advanced settings
        advanced_group = QGroupBox("Advanced Settings")
        advanced_layout = QFormLayout(advanced_group)
        
        # Slippage
        self.slippage_spin = QDoubleSpinBox()
        self.slippage_spin.setRange(0, 1)
        self.slippage_spin.setValue(0.001)
        self.slippage_spin.setSingleStep(0.001)
        self.slippage_spin.setDecimals(3)
        self.slippage_spin.setSuffix("%")
        advanced_layout.addRow("Slippage:", self.slippage_spin)
        
        # Timeframe
        self.timeframe_combo = QComboBox()
        self.timeframe_combo.addItems(["1m", "5m", "15m", "30m", "1h", "4h", "1d"])
        self.timeframe_combo.setCurrentText("1h")
        advanced_layout.addRow("Timeframe:", self.timeframe_combo)
        
        # Data source
        self.data_source_combo = QComboBox()
        self.data_source_combo.addItems(["Database", "CSV File", "API"])
        advanced_layout.addRow("Data Source:", self.data_source_combo)
        
        # Enable optimization
        self.optimization_check = QCheckBox("Enable parameter optimization")
        advanced_layout.addRow("", self.optimization_check)
        
        self.main_layout.addWidget(advanced_group)
        
        # Add button box
        button_box = self.add_button_box()
        
        # Customize buttons
        run_button = button_box.button(QDialogButtonBox.Ok)
        run_button.setText("Run Backtest")
        
        cancel_button = button_box.button(QDialogButtonBox.Cancel)
        cancel_button.setText("Cancel")
        
        # Add save config button
        save_config_button = QPushButton("Save Configuration")
        save_config_button.clicked.connect(self._save_config)
        button_box.addButton(save_config_button, QDialogButtonBox.ActionRole)
        
        # Add load config button
        load_config_button = QPushButton("Load Configuration")
        load_config_button.clicked.connect(self._load_config)
        button_box.addButton(load_config_button, QDialogButtonBox.ActionRole)
    
    def _select_all_strategies(self):
        """Select all strategies in the list."""
        for i in range(self.strategy_list.count()):
            item = self.strategy_list.item(i)
            item.setCheckState(Qt.Checked)
    
    def _select_no_strategies(self):
        """Deselect all strategies in the list."""
        for i in range(self.strategy_list.count()):
            item = self.strategy_list.item(i)
            item.setCheckState(Qt.Unchecked)
    
    def _save_config(self):
        """Save backtest configuration to file."""
        # Get file path
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Save Backtest Configuration", "", "JSON Files (*.json)"
        )
        
        if not file_path:
            return
        
        # Ensure .json extension
        if not file_path.endswith('.json'):
            file_path += '.json'
        
        # Get configuration
        config = self.get_config()
        
        # Save to file
        try:
            with open(file_path, 'w') as f:
                json.dump(config, f, indent=2)
            
            self.show_info("Configuration Saved", f"Backtest configuration saved to {file_path}")
        except Exception as e:
            self.show_error("Save Error", f"Error saving configuration: {str(e)}")
    
    def _load_config(self):
        """Load backtest configuration from file."""
        # Get file path
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Load Backtest Configuration", "", "JSON Files (*.json)"
        )
        
        if not file_path:
            return
        
        # Load from file
        try:
            with open(file_path, 'r') as f:
                config = json.load(f)
            
            # Apply configuration
            # Start date
            if 'start_date' in config:
                self.start_date.setDate(QDate.fromString(config['start_date'], "yyyy-MM-dd"))
            
            # End date
            if 'end_date' in config:
                self.end_date.setDate(QDate.fromString(config['end_date'], "yyyy-MM-dd"))
            
            # Strategies
            if 'strategies' in config:
                # Clear all first
                self._select_no_strategies()
                
                # Select strategies in config
                for i in range(self.strategy_list.count()):
                    item = self.strategy_list.item(i)
                    if item.text() in config['strategies']:
                        item.setCheckState(Qt.Checked)
            
            # Initial capital
            if 'initial_capital' in config:
                self.initial_capital_spin.setValue(config['initial_capital'])
            
            # Commission
            if 'commission' in config:
                self.commission_spin.setValue(config['commission'])
            
            # Slippage
            if 'slippage' in config:
                self.slippage_spin.setValue(config['slippage'])
            
            # Timeframe
            if 'timeframe' in config:
                self.timeframe_combo.setCurrentText(config['timeframe'])
            
            # Data source
            if 'data_source' in config:
                index = self.data_source_combo.findText(config['data_source'])
                if index >= 0:
                    self.data_source_combo.setCurrentIndex(index)
            
            # Optimization
            if 'optimization' in config:
                self.optimization_check.setChecked(config['optimization'])
            
            self.show_info("Configuration Loaded", f"Backtest configuration loaded from {file_path}")
            
        except Exception as e:
            self.show_error("Load Error", f"Error loading configuration: {str(e)}")
    
    def get_config(self):
        """
        Get backtest configuration.
        
        Returns:
            dict: Backtest configuration
        """
        # Get selected strategies
        selected_strategies = []
        for i in range(self.strategy_list.count()):
            item = self.strategy_list.item(i)
            if item.checkState() == Qt.Checked:
                selected_strategies.append(item.text())
        
        # Build configuration
        config = {
            'start_date': self.start_date.date().toString("yyyy-MM-dd"),
            'end_date': self.end_date.date().toString("yyyy-MM-dd"),
            'strategies': selected_strategies,
            'initial_capital': self.initial_capital_spin.value(),
            'commission': self.commission_spin.value(),
            'slippage': self.slippage_spin.value(),
            'timeframe': self.timeframe_combo.currentText(),
            'data_source': self.data_source_combo.currentText(),
            'optimization': self.optimization_check.isChecked()
        }
        
        return config


class ProgressDialog(BaseDialog):
    """Dialog for showing operation progress with cancel option."""
    
    def __init__(self, parent=None, title="Operation in Progress", operation_name=None):
        """
        Initialize progress dialog.
        
        Args:
            parent: Parent widget
            title: Dialog title
            operation_name: Name of the operation
        """
        super().__init__(parent, title)
        self.operation_name = operation_name or "Operation"
        self.canceled = False
        
        self._setup_ui()
        
        # Configure as modal but with ability to cancel
        self.setModal(True)
        self.setWindowFlags(Qt.Window | Qt.WindowTitleHint | Qt.CustomizeWindowHint)
    
    def _setup_ui(self):
        """Set up dialog UI components."""
        # Make a bit smaller than other dialogs
        self.setMinimumWidth(350)
        
        # Operation name
        self.operation_label = QLabel(self.operation_name)
        self.operation_label.setAlignment(Qt.AlignCenter)
        self.operation_label.setStyleSheet("font-weight: bold; font-size: 12pt;")
        self.main_layout.addWidget(self.operation_label)
        
        # Status label
        self.status_label = QLabel("Initializing...")
        self.status_label.setAlignment(Qt.AlignCenter)
        self.main_layout.addWidget(self.status_label)
        
        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.main_layout.addWidget(self.progress_bar)
        
        # Time remaining
        self.time_label = QLabel("Estimating time remaining...")
        self.time_label.setAlignment(Qt.AlignCenter)
        self.main_layout.addWidget(self.time_label)
        
        # Cancel button
        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.clicked.connect(self._on_cancel)
        self.main_layout.addWidget(self.cancel_button)
        
        # Timer for updating time remaining
        self.start_time = time.time()
        
        # Information display (optional)
        self.info_text = QTextEdit()
        self.info_text.setReadOnly(True)
        self.info_text.setMaximumHeight(100)
        self.info_text.setVisible(False)
        self.main_layout.addWidget(self.info_text)
        
        # Toggle info button
        self.toggle_info_button = QPushButton("Show Details")
        self.toggle_info_button.clicked.connect(self._toggle_info)
        self.main_layout.addWidget(self.toggle_info_button)
    
    def _toggle_info(self):
        """Toggle information display."""
        self.info_text.setVisible(not self.info_text.isVisible())
        self.toggle_info_button.setText("Hide Details" if self.info_text.isVisible() else "Show Details")
        
        # Adjust dialog size
        if self.info_text.isVisible():
            self.setMinimumHeight(300)
        else:
            self.setMinimumHeight(0)
    
    def _on_cancel(self):
        """Handle cancel button click."""
        if self.confirm("Cancel Operation", f"Are you sure you want to cancel {self.operation_name}?"):
            self.canceled = True
            self.status_label.setText("Canceling...")
            self.cancel_button.setEnabled(False)
    
    def set_progress(self, progress, status=None):
        """
        Update progress bar and status.
        
        Args:
            progress: Progress value (0-100)
            status: Status message
        """
        self.progress_bar.setValue(progress)
        
        if status:
            self.status_label.setText(status)
        
        # Update time remaining
        if progress > 0:
            elapsed = time.time() - self.start_time
            estimated_total = elapsed / (progress / 100.0)
            remaining = estimated_total - elapsed
            
            if remaining > 60:
                minutes = int(remaining / 60)
                seconds = int(remaining % 60)
                self.time_label.setText(f"Approximately {minutes}m {seconds}s remaining")
            else:
                self.time_label.setText(f"Approximately {int(remaining)}s remaining")
        
        # Process events to update UI
        QApplication.processEvents()
    
    def add_info(self, info):
        """
        Add information to the info display.
        
        Args:
            info: Information text
        """
        self.info_text.append(info)
        
        # Scroll to bottom
        scrollbar = self.info_text.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())
    
    def is_canceled(self):
        """
        Check if operation was canceled.
        
        Returns:
            bool: True if canceled, False otherwise
        """
        return self.canceled
    
    def complete(self):
        """Complete the operation and close dialog."""
        self.progress_bar.setValue(100)
        self.status_label.setText("Operation completed")
        self.time_label.setText("Completed")
        self.cancel_button.setText("Close")
        self.cancel_button.clicked.disconnect()
        self.cancel_button.clicked.connect(self.accept)


class ChartConfigDialog(BaseDialog):
    """Dialog for configuring chart appearance and indicators."""
    
    def __init__(self, parent=None):
        """
        Initialize chart configuration dialog.
        
        Args:
            parent: Parent widget
        """
        super().__init__(parent, "Chart Configuration")
        
        self._setup_ui()
        self._load_current_config()
    
    def _setup_ui(self):
        """Set up dialog UI components."""
        self.add_title("Chart Configuration", "Customize chart appearance and indicators")
        
        # Create tabs
        tab_widget = QTabWidget()
        
        # Appearance tab
        appearance_tab = QWidget()
        appearance_layout = QVBoxLayout(appearance_tab)
        
        # Chart type
        chart_type_group = QGroupBox("Chart Type")
        chart_type_layout = QVBoxLayout(chart_type_group)
        
        self.chart_type_combo = QComboBox()
        self.chart_type_combo.addItems([
            "Candlestick", 
            "OHLC", 
            "Line", 
            "Area", 
            "Heikin-Ashi", 
            "Renko"
        ])
        chart_type_layout.addWidget(self.chart_type_combo)
        
        appearance_layout.addWidget(chart_type_group)
        
        # Colors
        colors_group = QGroupBox("Colors")
        colors_layout = QFormLayout(colors_group)
        
        # Background color
        self.bg_color_btn = QPushButton()
        self.bg_color_btn.setStyleSheet("background-color: #FFFFFF;")
        self.bg_color_btn.setFixedSize(60, 20)
        self.bg_color_btn.clicked.connect(lambda: self._select_color(self.bg_color_btn))
        colors_layout.addRow("Background:", self.bg_color_btn)
        
        # Grid color
        self.grid_color_btn = QPushButton()
        self.grid_color_btn.setStyleSheet("background-color: #EEEEEE;")
        self.grid_color_btn.setFixedSize(60, 20)
        self.grid_color_btn.clicked.connect(lambda: self._select_color(self.grid_color_btn))
        colors_layout.addRow("Grid:", self.grid_color_btn)
        
        # Up color
        self.up_color_btn = QPushButton()
        self.up_color_btn.setStyleSheet("background-color: #26A69A;")
        self.up_color_btn.setFixedSize(60, 20)
        self.up_color_btn.clicked.connect(lambda: self._select_color(self.up_color_btn))
        colors_layout.addRow("Up Color:", self.up_color_btn)
        
        # Down color
        self.down_color_btn = QPushButton()
        self.down_color_btn.setStyleSheet("background-color: #EF5350;")
        self.down_color_btn.setFixedSize(60, 20)
        self.down_color_btn.clicked.connect(lambda: self._select_color(self.down_color_btn))
        colors_layout.addRow("Down Color:", self.down_color_btn)
        
        appearance_layout.addWidget(colors_group)
        
        # Scale options
        scale_group = QGroupBox("Scale Options")
        scale_layout = QFormLayout(scale_group)
        
        self.log_scale_check = QCheckBox("Logarithmic Scale")
        scale_layout.addRow("", self.log_scale_check)
        
        self.auto_scale_check = QCheckBox("Auto-scale")
        self.auto_scale_check.setChecked(True)
        scale_layout.addRow("", self.auto_scale_check)
        
        appearance_layout.addWidget(scale_group)
        
        tab_widget.addTab(appearance_tab, "Appearance")
        
        # Indicators tab
        indicators_tab = QWidget()
        indicators_layout = QVBoxLayout(indicators_tab)
        
        # Available indicators
        available_group = QGroupBox("Available Indicators")
        available_layout = QVBoxLayout(available_group)
        
        self.indicator_list = QListWidget()
        indicators = [
            "Moving Average", 
            "Exponential Moving Average", 
            "Bollinger Bands", 
            "RSI", 
            "MACD", 
            "Stochastic", 
            "ADX", 
            "Average True Range", 
            "Volume", 
            "OBV", 
            "Ichimoku Cloud"
        ]
        self.indicator_list.addItems(indicators)
        self.indicator_list.setSelectionMode(QListWidget.ExtendedSelection)
        available_layout.addWidget(self.indicator_list)
        
        indicators_layout.addWidget(available_group)
        
        # Active indicators
        active_group = QGroupBox("Active Indicators")
        active_layout = QVBoxLayout(active_group)
        
        self.active_indicators_list = QListWidget()
        active_layout.addWidget(self.active_indicators_list)
        
        # Add/Remove buttons
        button_layout = QHBoxLayout()
        
        add_btn = QPushButton("Add →")
        add_btn.clicked.connect(self._add_indicator)
        button_layout.addWidget(add_btn)
        
        remove_btn = QPushButton("← Remove")
        remove_btn.clicked.connect(self._remove_indicator)
        button_layout.addWidget(remove_btn)
        
        indicators_layout.addLayout(button_layout)
        indicators_layout.addWidget(active_group)
        
        tab_widget.addTab(indicators_tab, "Indicators")
        
        # Overlays tab
        overlays_tab = QWidget()
        overlays_layout = QVBoxLayout(overlays_tab)
        
        # Trading levels
        levels_group = QGroupBox("Trading Levels")
        levels_layout = QFormLayout(levels_group)
        
        # Support/Resistance
        self.support_resistance_check = QCheckBox("Show Support/Resistance")
        levels_layout.addRow("", self.support_resistance_check)
        
        # Pivot Points
        self.pivot_points_check = QCheckBox("Show Pivot Points")
        levels_layout.addRow("", self.pivot_points_check)
        
        # Fibonacci Levels
        self.fibonacci_check = QCheckBox("Show Fibonacci Levels")
        levels_layout.addRow("", self.fibonacci_check)
        
        overlays_layout.addWidget(levels_group)
        
        # Signals overlay
        signals_group = QGroupBox("Trading Signals")
        signals_layout = QFormLayout(signals_group)
        
        # Show signals
        self.show_signals_check = QCheckBox("Show Trading Signals")
        signals_layout.addRow("", self.show_signals_check)
        
        # Signal types
        self.signal_types_combo = QComboBox()
        self.signal_types_combo.addItems([
            "All Signals", 
            "Entry Points Only", 
            "Exit Points Only", 
            "Buy Signals Only", 
            "Sell Signals Only"
        ])
        signals_layout.addRow("Signal Types:", self.signal_types_combo)
        
        overlays_layout.addWidget(signals_group)
        
        # Volume Profile
        volume_profile_group = QGroupBox("Volume Profile")
        volume_profile_layout = QFormLayout(volume_profile_group)
        
        # Show volume profile
        self.volume_profile_check = QCheckBox("Show Volume Profile")
        volume_profile_layout.addRow("", self.volume_profile_check)
        
        # Profile side
        self.volume_profile_side_combo = QComboBox()
        self.volume_profile_side_combo.addItems(["Right", "Left"])
        volume_profile_layout.addRow("Profile Side:", self.volume_profile_side_combo)
        
        overlays_layout.addWidget(volume_profile_group)
        
        tab_widget.addTab(overlays_tab, "Overlays")
        
        # Add tab widget to main layout
        self.main_layout.addWidget(tab_widget)
        
        # Add button box
        button_box = self.add_button_box()
        
        # Customize buttons
        apply_button = button_box.button(QDialogButtonBox.Ok)
        apply_button.setText("Apply Configuration")
        
        cancel_button = button_box.button(QDialogButtonBox.Cancel)
        cancel_button.setText("Cancel")
        
        # Add reset button
        reset_button = QPushButton("Reset to Defaults")
        reset_button.clicked.connect(self._reset_to_defaults)
        button_box.addButton(reset_button, QDialogButtonBox.ResetRole)
    
    def _load_current_config(self):
        """Load current chart configuration."""
        # This would normally load from settings or chart widget
        # For now, we'll set some defaults
        
        # Chart type
        self.chart_type_combo.setCurrentText("Candlestick")
        
        # Colors - already set in setup_ui
        
        # Scale options
        self.log_scale_check.setChecked(False)
        self.auto_scale_check.setChecked(True)
        
        # Active indicators - example
        self.active_indicators_list.addItem("Moving Average")
        self.active_indicators_list.addItem("Volume")
        
        # Overlays
        self.support_resistance_check.setChecked(True)
        self.pivot_points_check.setChecked(False)
        self.fibonacci_check.setChecked(False)
        self.show_signals_check.setChecked(True)
        self.signal_types_combo.setCurrentText("All Signals")
        self.volume_profile_check.setChecked(False)
        self.volume_profile_side_combo.setCurrentText("Right")
    
    def _select_color(self, button):
        """Open color dialog to select color for button."""
        from PyQt5.QtWidgets import QColorDialog
        
        # Get current color
        current_style = button.styleSheet()
        current_color = current_style.split("background-color: ")[1].split(";")[0]
        
        # Open color dialog
        color = QColorDialog.getColor(QColor(current_color), self)
        
        if color.isValid():
            # Set button color
            button.setStyleSheet(f"background-color: {color.name()};")
    
    def _add_indicator(self):
        """Add selected indicators to active list."""
        selected_items = self.indicator_list.selectedItems()
        
        for item in selected_items:
            # Check if already added
            found = False
            for i in range(self.active_indicators_list.count()):
                if self.active_indicators_list.item(i).text() == item.text():
                    found = True
                    break
            
            if not found:
                self.active_indicators_list.addItem(item.text())
    
    def _remove_indicator(self):
        """Remove selected indicators from active list."""
        selected_items = self.active_indicators_list.selectedItems()
        
        for item in selected_items:
            self.active_indicators_list.takeItem(self.active_indicators_list.row(item))
    
    def _reset_to_defaults(self):
        """Reset chart configuration to defaults."""
        if self.confirm("Reset Configuration", "Are you sure you want to reset chart configuration to defaults?"):
            self._load_current_config()
    
    def get_configuration(self):
        """
        Get chart configuration.
        
        Returns:
            dict: Chart configuration
        """
        # Get colors from buttons
        bg_color = self.bg_color_btn.styleSheet().split("background-color: ")[1].split(";")[0]
        grid_color = self.grid_color_btn.styleSheet().split("background-color: ")[1].split(";")[0]
        up_color = self.up_color_btn.styleSheet().split("background-color: ")[1].split(";")[0]
        down_color = self.down_color_btn.styleSheet().split("background-color: ")[1].split(";")[0]
        
        # Get active indicators
        active_indicators = []
        for i in range(self.active_indicators_list.count()):
            active_indicators.append(self.active_indicators_list.item(i).text())
        
        # Build configuration
        config = {
            'appearance': {
                'chart_type': self.chart_type_combo.currentText(),
                'colors': {
                    'background': bg_color,
                    'grid': grid_color,
                    'up': up_color,
                    'down': down_color
                },
                'scale': {
                    'logarithmic': self.log_scale_check.isChecked(),
                    'auto_scale': self.auto_scale_check.isChecked()
                }
            },
            'indicators': active_indicators,
            'overlays': {
                'support_resistance': self.support_resistance_check.isChecked(),
                'pivot_points': self.pivot_points_check.isChecked(),
                'fibonacci': self.fibonacci_check.isChecked(),
                'signals': {
                    'show': self.show_signals_check.isChecked(),
                    'type': self.signal_types_combo.currentText()
                },
                'volume_profile': {
                    'show': self.volume_profile_check.isChecked(),
                    'side': self.volume_profile_side_combo.currentText()
                }
            }
        }
        
        return config


# For standalone testing
if __name__ == "__main__":
    import sys
    from PyQt5.QtWidgets import QApplication
    
    app = QApplication(sys.argv)
    
    # Uncomment to test different dialogs
    
    #dialog = LoginDialog()
    #dialog = ConfirmTradeDialog(trade_details={'symbol': 'BTC/USD', 'action': 'buy', 'price': 50000, 'amount': 0.5})
    #dialog = TradeSettingsDialog()
    #dialog = StrategyConfigDialog(strategy_name="Mean Reversion", strategy_params={'window': 20, 'std_dev': 2.0, 'stop_loss': 0.05})
    #dialog = BacktestConfigDialog(strategies=["Trend Following", "Mean Reversion", "Breakout"])
    dialog = ChartConfigDialog()
    
    if dialog.exec_():
        # Handle dialog acceptance
        print("Dialog accepted")
        # For testing, print returned values
        if hasattr(dialog, 'get_configuration'):
            print(dialog.get_configuration())
        elif hasattr(dialog, 'get_parameters'):
            print(dialog.get_parameters())
        elif hasattr(dialog, 'get_settings'):
            print(dialog.get_settings())
        elif hasattr(dialog, 'get_config'):
            print(dialog.get_config())
        elif hasattr(dialog, 'get_credentials'):
            # Don't print credentials in a real app
            print("Credentials entered (not showing for security)")
        elif hasattr(dialog, 'get_trade_details'):
            print(dialog.get_trade_details())
    else:
        print("Dialog canceled")
    
    sys.exit(0)
