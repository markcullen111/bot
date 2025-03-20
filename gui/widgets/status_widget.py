# gui/widgets/status_widget.py

import time
import logging
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGridLayout, QLabel, QPushButton, 
    QFrame, QProgressBar, QScrollArea, QTabWidget, QGroupBox, 
    QSizePolicy, QSpacerItem, QMenu, QDialog, QTextEdit, QDialogButtonBox
)
from PyQt5.QtCore import Qt, QTimer, pyqtSignal, pyqtSlot, QSize
from PyQt5.QtGui import QColor, QPalette, QIcon, QFont, QPixmap

try:
    # Try to import from the core package first
    from core.error_handling import ErrorCategory, ErrorSeverity, ErrorHandler
    HAVE_ERROR_HANDLING = True
except ImportError:
    try:
        # Try direct import if not in package
        from error_handling import ErrorCategory, ErrorSeverity, ErrorHandler
        HAVE_ERROR_HANDLING = True
    except ImportError:
        HAVE_ERROR_HANDLING = False
        logging.warning("Error handling module not available. Using basic error handling.")

class StatusIndicator(QWidget):
    """
    A colored status indicator with label.
    Shows component status with appropriate colors and icons.
    """
    
    # Status colors
    STATUS_COLORS = {
        "online": QColor(39, 174, 96),  # Green
        "connecting": QColor(241, 196, 15),  # Yellow
        "offline": QColor(231, 76, 60),  # Red
        "warning": QColor(230, 126, 34),  # Orange
        "unknown": QColor(149, 165, 166)  # Gray
    }
    
    clicked = pyqtSignal(str)  # Signal emitted when indicator is clicked
    
    def __init__(self, label: str, status: str = "unknown", parent=None):
        """
        Initialize the status indicator.
        
        Args:
            label: Label for the indicator
            status: Initial status (online, connecting, offline, warning, unknown)
            parent: Parent widget
        """
        super().__init__(parent)
        self.label = label
        self.status = status
        self.details = ""
        self.setupUI()
        
    def setupUI(self):
        """Setup the UI components."""
        layout = QHBoxLayout(self)
        layout.setContentsMargins(2, 2, 2, 2)
        layout.setSpacing(4)
        
        # Status indicator
        self.indicator = QFrame()
        self.indicator.setFixedSize(12, 12)
        self.indicator.setFrameShape(QFrame.Box)
        self.indicator.setFrameShadow(QFrame.Plain)
        self.indicator.setAutoFillBackground(True)
        
        # Set initial color
        self.updateColor()
        
        # Label
        self.label_widget = QLabel(self.label)
        font = QFont()
        font.setBold(True)
        self.label_widget.setFont(font)
        
        # Value
        self.value_widget = QLabel(self.status.capitalize())
        
        # Add to layout
        layout.addWidget(self.indicator)
        layout.addWidget(self.label_widget, 1)  # 1 = stretch factor
        layout.addWidget(self.value_widget)
        
        # Make widget clickable
        self.setCursor(Qt.PointingHandCursor)
        
    def updateColor(self):
        """Update the indicator color based on status."""
        palette = self.indicator.palette()
        color = self.STATUS_COLORS.get(self.status.lower(), self.STATUS_COLORS["unknown"])
        palette.setColor(QPalette.Window, color)
        self.indicator.setPalette(palette)
        
        # Update text as well
        if self.value_widget:
            self.value_widget.setText(self.status.capitalize())
        
    def setStatus(self, status: str, details: str = ""):
        """
        Set the status and update the indicator.
        
        Args:
            status: New status value
            details: Optional details about the status
        """
        self.status = status.lower()
        self.details = details
        self.updateColor()
        
        # Set tooltip with details
        tooltip = f"{self.label}: {self.status.capitalize()}"
        if self.details:
            tooltip += f"\n{self.details}"
        self.setToolTip(tooltip)
        
    def mousePressEvent(self, event):
        """Handle mouse press events to emit the clicked signal."""
        self.clicked.emit(self.label)
        super().mousePressEvent(event)

class ComponentStatusGroup(QGroupBox):
    """
    Group box containing status indicators for various system components.
    """
    
    status_clicked = pyqtSignal(str, str, str)  # Component, status, details
    
    def __init__(self, title: str, components: List[str], parent=None):
        """
        Initialize the component status group.
        
        Args:
            title: Group title
            components: List of component names to display
            parent: Parent widget
        """
        super().__init__(title, parent)
        self.components = components
        self.indicators = {}
        self.setupUI()
        
    def setupUI(self):
        """Setup the UI components."""
        layout = QGridLayout(self)
        layout.setVerticalSpacing(2)
        
        # Create indicators for each component
        for i, component in enumerate(self.components):
            indicator = StatusIndicator(component)
            indicator.clicked.connect(self._onIndicatorClicked)
            layout.addWidget(indicator, i, 0)
            self.indicators[component] = indicator
    
    def updateStatus(self, component: str, status: str, details: str = ""):
        """
        Update the status of a component.
        
        Args:
            component: Component name
            status: New status
            details: Optional details
        """
        if component in self.indicators:
            self.indicators[component].setStatus(status, details)
    
    def _onIndicatorClicked(self, component: str):
        """Handle indicator click events."""
        if component in self.indicators:
            indicator = self.indicators[component]
            self.status_clicked.emit(component, indicator.status, indicator.details)

class MetricDisplay(QFrame):
    """
    Widget for displaying a numeric metric with label and optional trend indicator.
    """
    
    def __init__(self, label: str, value: str = "0", trend: str = "neutral", parent=None):
        """
        Initialize the metric display.
        
        Args:
            label: Metric label
            value: Initial value
            trend: Trend direction (up, down, neutral)
            parent: Parent widget
        """
        super().__init__(parent)
        self.label = label
        self.value = value
        self.trend = trend
        self.setupUI()
        
        # Set frame style
        self.setFrameShape(QFrame.StyledPanel)
        self.setFrameShadow(QFrame.Raised)
        self.setStyleSheet("background-color: #f5f5f5; border-radius: 4px;")
    
    def setupUI(self):
        """Setup the UI components."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        
        # Label
        self.label_widget = QLabel(self.label)
        self.label_widget.setStyleSheet("color: #7f8c8d; font-size: 11px;")
        self.label_widget.setAlignment(Qt.AlignCenter)
        
        # Value
        self.value_widget = QLabel(self.value)
        self.value_widget.setStyleSheet("font-size: 16px; font-weight: bold;")
        self.value_widget.setAlignment(Qt.AlignCenter)
        
        # Trend indicator
        self.trend_widget = QLabel()
        self.updateTrend(self.trend)
        self.trend_widget.setAlignment(Qt.AlignCenter)
        
        # Add to layout
        layout.addWidget(self.label_widget)
        layout.addWidget(self.value_widget)
        layout.addWidget(self.trend_widget)
    
    def setValue(self, value: str, trend: str = None):
        """
        Update the displayed value and trend.
        
        Args:
            value: New value to display
            trend: New trend direction (if None, keep current)
        """
        self.value = value
        self.value_widget.setText(value)
        
        if trend is not None:
            self.updateTrend(trend)
    
    def updateTrend(self, trend: str):
        """
        Update the trend indicator.
        
        Args:
            trend: New trend direction (up, down, neutral)
        """
        self.trend = trend
        
        # Update trend indicator with arrow or symbol
        if trend == "up":
            self.trend_widget.setText("▲")
            self.trend_widget.setStyleSheet("color: #27ae60;")  # Green
        elif trend == "down":
            self.trend_widget.setText("▼")
            self.trend_widget.setStyleSheet("color: #e74c3c;")  # Red
        else:
            self.trend_widget.setText("—")
            self.trend_widget.setStyleSheet("color: #7f8c8d;")  # Gray

class AlertItem(QFrame):
    """
    Widget for displaying an alert message with severity indicator.
    """
    
    dismissed = pyqtSignal(str)  # Signal emitted when alert is dismissed
    
    def __init__(self, alert_id: str, message: str, severity: str = "info", timestamp=None, parent=None):
        """
        Initialize the alert item.
        
        Args:
            alert_id: Unique identifier for the alert
            message: Alert message text
            severity: Alert severity level (info, warning, error, critical)
            timestamp: Alert timestamp (if None, use current time)
            parent: Parent widget
        """
        super().__init__(parent)
        self.alert_id = alert_id
        self.message = message
        self.severity = severity
        self.timestamp = timestamp or datetime.now()
        self.setupUI()
        
        # Set frame style
        self.setFrameShape(QFrame.StyledPanel)
        self.setFrameShadow(QFrame.Raised)
        
        # Set background color based on severity
        if severity == "critical":
            self.setStyleSheet("background-color: #fadbd8;")  # Light red
        elif severity == "error":
            self.setStyleSheet("background-color: #f8c9bb;")  # Light orange-red
        elif severity == "warning":
            self.setStyleSheet("background-color: #fef9e7;")  # Light yellow
        else:
            self.setStyleSheet("background-color: #eaecee;")  # Light gray
    
    def setupUI(self):
        """Setup the UI components."""
        layout = QHBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        
        # Severity indicator
        self.indicator = QLabel()
        self.indicator.setFixedSize(16, 16)
        
        # Set icon based on severity
        if self.severity == "critical":
            self.indicator.setText("🔴")  # Red circle
        elif self.severity == "error":
            self.indicator.setText("⛔")  # No entry
        elif self.severity == "warning":
            self.indicator.setText("⚠️")  # Warning
        else:
            self.indicator.setText("ℹ️")  # Info
            
        # Message
        self.message_widget = QLabel(self.message)
        self.message_widget.setWordWrap(True)
        
        # Timestamp
        time_str = self.timestamp.strftime("%H:%M:%S")
        self.time_widget = QLabel(time_str)
        self.time_widget.setStyleSheet("color: #7f8c8d; font-size: 10px;")
        
        # Dismiss button
        self.dismiss_btn = QPushButton("×")
        self.dismiss_btn.setFixedSize(20, 20)
        self.dismiss_btn.setStyleSheet("background: none; border: none;")
        self.dismiss_btn.clicked.connect(self._onDismiss)
        
        # Add to layout
        layout.addWidget(self.indicator)
        layout.addWidget(self.message_widget, 1)  # 1 = stretch factor
        layout.addWidget(self.time_widget)
        layout.addWidget(self.dismiss_btn)
    
    def _onDismiss(self):
        """Handle dismiss button click."""
        self.dismissed.emit(self.alert_id)

class StatusWidget(QWidget):
    """
    Widget for displaying comprehensive system status information.
    Shows component status, performance metrics, recent alerts, and task status.
    Provides controls for basic system actions.
    """
    
    # Signals
    component_action_requested = pyqtSignal(str, str)  # Component, action
    refresh_requested = pyqtSignal()
    alert_dismissed = pyqtSignal(str)  # Alert ID
    
    def __init__(self, trading_system=None, parent=None):
        """
        Initialize the status widget.
        
        Args:
            trading_system: Reference to the main trading system (optional)
            parent: Parent widget
        """
        super().__init__(parent)
        self.trading_system = trading_system
        self.alerts = {}  # Store active alerts by ID
        self.setupUI()
        
        # Set up update timer for periodic status refresh
        self.update_timer = QTimer(self)
        self.update_timer.timeout.connect(self.refreshStatus)
        self.update_timer.start(2000)  # Update every 2 seconds
        
        # Initial status refresh
        QTimer.singleShot(100, self.refreshStatus)
    
    def setupUI(self):
        """Setup the UI components."""
        main_layout = QVBoxLayout(self)
        
        # Tab widget for different status views
        self.tab_widget = QTabWidget()
        main_layout.addWidget(self.tab_widget)
        
        # Overview tab
        overview_tab = QWidget()
        self._setupOverviewTab(overview_tab)
        self.tab_widget.addTab(overview_tab, "Overview")
        
        # System tab
        system_tab = QWidget()
        self._setupSystemTab(system_tab)
        self.tab_widget.addTab(system_tab, "System")
        
        # Trading tab
        trading_tab = QWidget()
        self._setupTradingTab(trading_tab)
        self.tab_widget.addTab(trading_tab, "Trading")
        
        # Tasks tab
        tasks_tab = QWidget()
        self._setupTasksTab(tasks_tab)
        self.tab_widget.addTab(tasks_tab, "Tasks")
        
        # Bottom controls
        controls_layout = QHBoxLayout()
        
        self.refresh_btn = QPushButton("Refresh")
        self.refresh_btn.clicked.connect(self.refreshStatus)
        
        self.clear_alerts_btn = QPushButton("Clear Alerts")
        self.clear_alerts_btn.clicked.connect(self.clearAlerts)
        
        self.action_btn = QPushButton("Actions")
        self.action_btn.clicked.connect(self._showActionMenu)
        
        controls_layout.addWidget(self.refresh_btn)
        controls_layout.addWidget(self.clear_alerts_btn)
        controls_layout.addWidget(self.action_btn)
        
        main_layout.addLayout(controls_layout)
    
    def _setupOverviewTab(self, tab):
        """
        Setup the overview tab with high-level system status.
        
        Args:
            tab: The tab widget to setup
        """
        layout = QVBoxLayout(tab)
        
        # System status section
        status_layout = QHBoxLayout()
        
        # Core components status
        core_components = ["Trading System", "Database", "Strategies", "Risk Manager"]
        self.core_status = ComponentStatusGroup("Core Components", core_components)
        self.core_status.status_clicked.connect(self._onStatusClicked)
        status_layout.addWidget(self.core_status)
        
        # AI Components status
        ai_components = ["ML Models", "Market Analysis", "Order Flow", "Portfolio AI"]
        self.ai_status = ComponentStatusGroup("AI Components", ai_components)
        self.ai_status.status_clicked.connect(self._onStatusClicked)
        status_layout.addWidget(self.ai_status)
        
        layout.addLayout(status_layout)
        
        # Key metrics section
        metrics_layout = QHBoxLayout()
        metrics_layout.setSpacing(10)
        
        self.metrics = {}
        
        metric_names = [
            "Portfolio Value", "Day P&L", "Open Positions", 
            "Success Rate", "Active Strategies", "System Load"
        ]
        
        for name in metric_names:
            metric = MetricDisplay(name)
            metrics_layout.addWidget(metric)
            self.metrics[name] = metric
        
        layout.addLayout(metrics_layout)
        
        # Alerts section
        alerts_group = QGroupBox("Recent Alerts")
        alerts_layout = QVBoxLayout(alerts_group)
        
        self.alerts_container = QWidget()
        self.alerts_layout = QVBoxLayout(self.alerts_container)
        self.alerts_layout.setSpacing(2)
        self.alerts_layout.addStretch()
        
        alerts_scroll = QScrollArea()
        alerts_scroll.setWidgetResizable(True)
        alerts_scroll.setWidget(self.alerts_container)
        alerts_scroll.setMinimumHeight(150)
        
        alerts_layout.addWidget(alerts_scroll)
        
        layout.addWidget(alerts_group)
    
    def _setupSystemTab(self, tab):
        """
        Setup the system tab with detailed system information.
        
        Args:
            tab: The tab widget to setup
        """
        layout = QVBoxLayout(tab)
        
        # System information section
        info_group = QGroupBox("System Information")
        info_layout = QGridLayout(info_group)
        
        # System info labels (left column)
        labels = [
            "System Version:", "Python Version:", "Operating System:",
            "CPU Usage:", "Memory Usage:", "Disk Space:",
            "Uptime:", "Last Update:", "Thread Count:"
        ]
        
        self.system_info_values = {}
        
        for i, label in enumerate(labels):
            label_widget = QLabel(label)
            value_widget = QLabel("Loading...")
            info_layout.addWidget(label_widget, i, 0)
            info_layout.addWidget(value_widget, i, 1)
            self.system_info_values[label] = value_widget
        
        # Add spacer on the right side to push content left
        info_layout.setColumnStretch(2, 1)
        
        layout.addWidget(info_group)
        
        # Database status section
        db_group = QGroupBox("Database Status")
        db_layout = QGridLayout(db_group)
        
        db_labels = [
            "Connection Type:", "Connection Status:", "Database Size:",
            "Last Sync:", "Tables:", "Market Data Size:",
            "Trade History:", "Portfolio History:", "Cache Status:"
        ]
        
        self.db_info_values = {}
        
        for i, label in enumerate(db_labels):
            label_widget = QLabel(label)
            value_widget = QLabel("Loading...")
            db_layout.addWidget(label_widget, i, 0)
            db_layout.addWidget(value_widget, i, 1)
            self.db_info_values[label] = value_widget
        
        # Add spacer on the right side to push content left
        db_layout.setColumnStretch(2, 1)
        
        layout.addWidget(db_group)
        
        # Add a spacer at the bottom to push content to the top
        layout.addStretch()
    
    def _setupTradingTab(self, tab):
        """
        Setup the trading tab with trading-specific information.
        
        Args:
            tab: The tab widget to setup
        """
        layout = QVBoxLayout(tab)
        
        # Exchange connections
        exchange_group = QGroupBox("Exchange Connections")
        exchange_layout = QGridLayout(exchange_group)
        
        exchange_headers = ["Exchange", "Status", "Balance", "Rate Limit", "Last Update"]
        
        for i, header in enumerate(exchange_headers):
            label = QLabel(header)
            label.setStyleSheet("font-weight: bold;")
            exchange_layout.addWidget(label, 0, i)
        
        # Placeholder for exchange data (will be populated dynamically)
        self.exchange_rows = 0
        
        layout.addWidget(exchange_group)
        
        # Active strategies
        strategies_group = QGroupBox("Active Strategies")
        strategies_layout = QGridLayout(strategies_group)
        
        strategy_headers = ["Strategy", "Status", "P&L", "Trades", "Win Rate"]
        
        for i, header in enumerate(strategy_headers):
            label = QLabel(header)
            label.setStyleSheet("font-weight: bold;")
            strategies_layout.addWidget(label, 0, i)
        
        # Placeholder for strategy data (will be populated dynamically)
        self.strategy_rows = 0
        
        layout.addWidget(strategies_group)
        
        # Current positions
        positions_group = QGroupBox("Open Positions")
        positions_layout = QGridLayout(positions_group)
        
        position_headers = ["Symbol", "Side", "Size", "Entry Price", "Current P&L", "Duration"]
        
        for i, header in enumerate(position_headers):
            label = QLabel(header)
            label.setStyleSheet("font-weight: bold;")
            positions_layout.addWidget(label, 0, i)
        
        # Placeholder for position data (will be populated dynamically)
        self.position_rows = 0
        
        layout.addWidget(positions_group)
    
    def _setupTasksTab(self, tab):
        """
        Setup the tasks tab with information about background tasks.
        
        Args:
            tab: The tab widget to setup
        """
        layout = QVBoxLayout(tab)
        
        # Active tasks
        tasks_group = QGroupBox("Active Tasks")
        tasks_layout = QGridLayout(tasks_group)
        
        task_headers = ["Task", "Status", "Progress", "Runtime", "Priority"]
        
        for i, header in enumerate(task_headers):
            label = QLabel(header)
            label.setStyleSheet("font-weight: bold;")
            tasks_layout.addWidget(label, 0, i)
        
        # Placeholder for task data (will be populated dynamically)
        self.task_rows = 0
        self.task_progresses = {}
        
        layout.addWidget(tasks_group)
        
        # Task statistics
        stats_group = QGroupBox("Task Statistics")
        stats_layout = QGridLayout(stats_group)
        
        stats_labels = [
            "Total Tasks:", "Completed Tasks:", "Failed Tasks:",
            "Cancelled Tasks:", "Average Runtime:", "Workers:",
            "Queue Depth:", "Last Completed:", "Task Types:"
        ]
        
        self.task_stats_values = {}
        
        for i, label in enumerate(stats_labels):
            label_widget = QLabel(label)
            value_widget = QLabel("Loading...")
            stats_layout.addWidget(label_widget, i, 0)
            stats_layout.addWidget(value_widget, i, 1)
            self.task_stats_values[label] = value_widget
        
        # Add spacer on the right side to push content left
        stats_layout.setColumnStretch(2, 1)
        
        layout.addWidget(stats_group)
        
        # Add a spacer at the bottom to push content to the top
        layout.addStretch()
    
    @pyqtSlot()
    def refreshStatus(self):
        """Refresh all status information with current data."""
        if not self.isVisible():
            return  # Skip updates when not visible
        
        try:
            # Update component status
            self._updateComponentStatus()
            
            # Update metrics
            self._updateMetrics()
            
            # Update system information
            if self.tab_widget.currentIndex() == 1:  # System tab
                self._updateSystemInfo()
            
            # Update trading information
            if self.tab_widget.currentIndex() == 2:  # Trading tab
                self._updateTradingInfo()
            
            # Update task information
            if self.tab_widget.currentIndex() == 3:  # Tasks tab
                self._updateTaskInfo()
                
            # Emit refresh signal
            self.refresh_requested.emit()
            
        except Exception as e:
            logging.error(f"Error refreshing status: {e}")
            self.addAlert("refresh_error", f"Error refreshing status: {e}", "error")
    
    def _updateComponentStatus(self):
        """Update the status of all system components."""
        if not self.trading_system:
            return
        
        # Core components
        components = {
            "Trading System": "online",
            "Database": "unknown",
            "Strategies": "unknown",
            "Risk Manager": "unknown"
        }
        
        # Update database status
        if hasattr(self.trading_system, 'db'):
            if self.trading_system.db:
                if hasattr(self.trading_system.db, '_test_connection'):
                    try:
                        if self.trading_system.db._test_connection():
                            components["Database"] = "online"
                        else:
                            components["Database"] = "offline"
                    except:
                        components["Database"] = "offline"
                else:
                    # Assume it's online if it exists
                    components["Database"] = "online"
        
        # Update strategies status
        if hasattr(self.trading_system, 'strategy_system'):
            if self.trading_system.strategy_system:
                if hasattr(self.trading_system.strategy_system, 'active_strategies'):
                    if self.trading_system.strategy_system.active_strategies:
                        components["Strategies"] = "online"
                    else:
                        components["Strategies"] = "warning"
                else:
                    components["Strategies"] = "online"
        
        # Update risk manager status
        if hasattr(self.trading_system, 'risk_manager'):
            if self.trading_system.risk_manager:
                components["Risk Manager"] = "online"
        
        # Update component status indicators
        for component, status in components.items():
            self.core_status.updateStatus(component, status)
        
        # AI components
        ai_components = {
            "ML Models": "unknown",
            "Market Analysis": "unknown",
            "Order Flow": "unknown",
            "Portfolio AI": "unknown"
        }
        
        # Update ML Models status
        if hasattr(self.trading_system, 'ai_engine'):
            ai_components["ML Models"] = "online"
        elif hasattr(self.trading_system, 'model_trainer'):
            ai_components["ML Models"] = "online"
            
        # Update Market Analysis status
        if hasattr(self.trading_system, 'market_analyzer'):
            ai_components["Market Analysis"] = "online"
        elif hasattr(self.trading_system, 'indicators'):
            ai_components["Market Analysis"] = "online"
            
        # Update Order Flow status
        if hasattr(self.trading_system, 'order_flow_predictor'):
            ai_components["Order Flow"] = "online"
            
        # Update Portfolio AI status
        if hasattr(self.trading_system, 'portfolio_allocator'):
            ai_components["Portfolio AI"] = "online"
        
        # Update AI component status indicators
        for component, status in ai_components.items():
            self.ai_status.updateStatus(component, status)
    
    def _updateMetrics(self):
        """Update the displayed metrics with current values."""
        if not self.trading_system:
            return
        
        metrics = {
            "Portfolio Value": "$0.00",
            "Day P&L": "$0.00",
            "Open Positions": "0",
            "Success Rate": "0%",
            "Active Strategies": "0",
            "System Load": "0%"
        }
        
        # Update Portfolio Value
        portfolio_value = 0
        portfolio_trend = "neutral"
        
        if hasattr(self.trading_system, 'get_portfolio_value'):
            try:
                portfolio = self.trading_system.get_portfolio_value()
                if portfolio and isinstance(portfolio, dict) and 'total_value' in portfolio:
                    portfolio_value = portfolio['total_value']
                    # Determine trend based on previous value
                    if hasattr(self, '_last_portfolio_value'):
                        if portfolio_value > self._last_portfolio_value:
                            portfolio_trend = "up"
                        elif portfolio_value < self._last_portfolio_value:
                            portfolio_trend = "down"
                    self._last_portfolio_value = portfolio_value
            except:
                pass
                
        metrics["Portfolio Value"] = f"${portfolio_value:,.2f}"
        
        # Update Day P&L
        day_pnl = 0
        day_pnl_trend = "neutral"
        
        if hasattr(self.trading_system, 'get_daily_pnl'):
            try:
                day_pnl = self.trading_system.get_daily_pnl()
                # Determine trend based on value
                if day_pnl > 0:
                    day_pnl_trend = "up"
                elif day_pnl < 0:
                    day_pnl_trend = "down"
            except:
                pass
                
        metrics["Day P&L"] = f"${day_pnl:,.2f}"
        
        # Update Open Positions
        open_positions = 0
        
        if hasattr(self.trading_system, 'open_positions'):
            try:
                open_positions = len(self.trading_system.open_positions)
            except:
                pass
                
        metrics["Open Positions"] = str(open_positions)
        
        # Update Success Rate
        success_rate = 0
        
        if hasattr(self.trading_system, 'position_sizer'):
            try:
                if hasattr(self.trading_system.position_sizer, 'get_performance_stats'):
                    stats = self.trading_system.position_sizer.get_performance_stats()
                    if stats and 'win_rate' in stats:
                        success_rate = stats['win_rate'] * 100
            except:
                pass
                
        metrics["Success Rate"] = f"{success_rate:.1f}%"
        
        # Update Active Strategies
        active_strategies = 0
        
        if hasattr(self.trading_system, 'strategy_system'):
            try:
                if hasattr(self.trading_system.strategy_system, 'active_strategies'):
                    active_strategies = len(self.trading_system.strategy_system.active_strategies)
            except:
                pass
                
        metrics["Active Strategies"] = str(active_strategies)
        
        # Update System Load
        system_load = 0
        
        if hasattr(self.trading_system, 'thread_manager'):
            try:
                active_tasks = self.trading_system.thread_manager.get_active_tasks()
                if active_tasks:
                    # Calculate system load based on active tasks and worker count
                    worker_count = len(self.trading_system.thread_manager.workers)
                    if worker_count > 0:
                        system_load = min(100, (len(active_tasks) / worker_count) * 100)
            except:
                pass
                
        metrics["System Load"] = f"{system_load:.1f}%"
        
        # Update metric displays
        self.metrics["Portfolio Value"].setValue(metrics["Portfolio Value"], portfolio_trend)
        self.metrics["Day P&L"].setValue(metrics["Day P&L"], day_pnl_trend)
        self.metrics["Open Positions"].setValue(metrics["Open Positions"])
        self.metrics["Success Rate"].setValue(metrics["Success Rate"])
        self.metrics["Active Strategies"].setValue(metrics["Active Strategies"])
        self.metrics["System Load"].setValue(metrics["System Load"])
    
    def _updateSystemInfo(self):
        """Update system information tab."""
        import platform
        import psutil
        from datetime import datetime, timedelta
        
        # System info
        system_info = {
            "System Version:": "1.0.0",  # Replace with actual version
            "Python Version:": platform.python_version(),
            "Operating System:": f"{platform.system()} {platform.release()}",
            "CPU Usage:": f"{psutil.cpu_percent()}%",
            "Memory Usage:": f"{psutil.virtual_memory().percent}%",
            "Disk Space:": f"{psutil.disk_usage('/').percent}% used",
            "Uptime:": "Unknown",
            "Last Update:": "Unknown",
            "Thread Count:": str(threading.active_count())
        }
        
        # Get uptime
        if hasattr(self.trading_system, '_start_time'):
            uptime = datetime.now() - self.trading_system._start_time
            hours, remainder = divmod(uptime.total_seconds(), 3600)
            minutes, seconds = divmod(remainder, 60)
            system_info["Uptime:"] = f"{int(hours)}h {int(minutes)}m {int(seconds)}s"
        
        # Get last update
        if hasattr(self.trading_system, '_last_update'):
            system_info["Last Update:"] = self.trading_system._last_update.strftime("%Y-%m-%d %H:%M:%S")
        
        # Update labels
        for label, value in system_info.items():
            if label in self.system_info_values:
                self.system_info_values[label].setText(value)
        
        # Database info
        db_info = {
            "Connection Type:": "Unknown",
            "Connection Status:": "Unknown",
            "Database Size:": "Unknown",
            "Last Sync:": "Unknown",
            "Tables:": "Unknown",
            "Market Data Size:": "Unknown",
            "Trade History:": "Unknown",
            "Portfolio History:": "Unknown",
            "Cache Status:": "Unknown"
        }
        
        # Get database info if available
        if hasattr(self.trading_system, 'db'):
            db = self.trading_system.db
            
            # Connection type
            if db.__class__.__name__ == "MockDatabaseManager":
                db_info["Connection Type:"] = "Mock Database"
                db_info["Connection Status:"] = "Connected"
            elif db.__class__.__name__ == "DatabaseManager":
                db_info["Connection Type:"] = "TimescaleDB"
                
                # Test connection
                if hasattr(db, '_test_connection'):
                    try:
                        if db._test_connection():
                            db_info["Connection Status:"] = "Connected"
                        else:
                            db_info["Connection Status:"] = "Disconnected"
                    except:
                        db_info["Connection Status:"] = "Error"
                else:
                    db_info["Connection Status:"] = "Unknown"
            
            # Get additional database info (placeholder)
            db_info["Tables:"] = "6 tables"
            db_info["Market Data Size:"] = "2.3 GB"
            db_info["Trade History:"] = "124 trades"
            db_info["Portfolio History:"] = "45 days"
            db_info["Last Sync:"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            db_info["Cache Status:"] = "Optimized"
        
        # Update labels
        for label, value in db_info.items():
            if label in self.db_info_values:
                self.db_info_values[label].setText(value)
    
    def _updateTradingInfo(self):
        """Update trading information tab."""
        if not self.trading_system:
            return
        
        # Exchange information
        exchanges = []
        
        # Get exchange connections
        if hasattr(self.trading_system, 'exchanges'):
            try:
                for exchange_id, exchange in self.trading_system.exchanges.items():
                    # Determine status
                    status = "Unknown"
                    try:
                        # Attempt a simple API call to check status
                        if hasattr(exchange, 'fetchStatus'):
                            exchange_status = exchange.fetchStatus()
                            status = "Online" if exchange_status.get('status') == 'ok' else "Limited"
                        else:
                            status = "Online"  # Assume online if we can't check
                    except:
                        status = "Offline"
                    
                    # Get balance if available
                    balance = "Unknown"
                    try:
                        if hasattr(exchange, 'fetchBalance'):
                            balance_data = exchange.fetchBalance()
                            if balance_data and 'total' in balance_data:
                                # Sum up USDT-equivalent balance
                                usd_balance = sum(float(balance_data['total'].get(currency, 0)) for currency in ['USDT', 'USD', 'BUSD'])
                                balance = f"${usd_balance:,.2f}"
                    except:
                        pass
                    
                    # Get rate limit info
                    rate_limit = "Unknown"
                    try:
                        if hasattr(exchange, 'rateLimit'):
                            rate_limit = f"{exchange.rateLimit} ms"
                        elif hasattr(exchange, 'options') and 'rateLimit' in exchange.options:
                            rate_limit = f"{exchange.options['rateLimit']} ms"
                    except:
                        pass
                    
                    # Get last update time
                    last_update = "Unknown"
                    if hasattr(exchange, 'last_request_time'):
                        try:
                            last_update = datetime.fromtimestamp(exchange.last_request_time / 1000).strftime("%H:%M:%S")
                        except:
                            pass
                    
                    exchanges.append({
                        "exchange": exchange_id,
                        "status": status,
                        "balance": balance,
                        "rate_limit": rate_limit,
                        "last_update": last_update
                    })
            except Exception as e:
                logging.error(f"Error getting exchange info: {e}")
        
        # Update exchange table
        self._updateExchangeTable(exchanges)
        
        # Strategy information
        strategies = []
        
        # Get active strategies
        if hasattr(self.trading_system, 'strategy_system'):
            try:
                if hasattr(self.trading_system.strategy_system, 'active_strategies'):
                    active_strategies = self.trading_system.strategy_system.active_strategies
                    
                    for strategy_name in active_strategies:
                        # Get strategy details
                        status = "Active"
                        pnl = "Unknown"
                        trades = "0"
                        win_rate = "0%"
                        
                        # Get performance data if available
                        if hasattr(self.trading_system.strategy_system, 'strategy_executor'):
                            if hasattr(self.trading_system.strategy_system.strategy_executor, 'performance_history'):
                                perf = self.trading_system.strategy_system.strategy_executor.performance_history.get(strategy_name, {})
                                if perf:
                                    pnl = f"${perf.get('pnl', 0):,.2f}"
                                    trades = str(perf.get('total_trades', 0))
                                    win_rate = f"{perf.get('win_rate', 0) * 100:.1f}%"
                        
                        strategies.append({
                            "strategy": strategy_name,
                            "status": status,
                            "pnl": pnl,
                            "trades": trades,
                            "win_rate": win_rate
                        })
            except Exception as e:
                logging.error(f"Error getting strategy info: {e}")
        
        # Update strategy table
        self._updateStrategyTable(strategies)
        
        # Position information
        positions = []
        
        # Get open positions
        if hasattr(self.trading_system, 'open_positions'):
            try:
                for symbol, position in self.trading_system.open_positions.items():
                    # Calculate position details
                    side = position.get('side', 'unknown')
                    size = position.get('position_size', 0)
                    entry_price = position.get('entry_price', 0)
                    
                    # Calculate current P&L if possible
                    pnl = "Unknown"
                    if hasattr(self.trading_system, 'get_market_data'):
                        try:
                            data = self.trading_system.get_market_data(symbol, limit=1)
                            if not data.empty:
                                current_price = data['close'].iloc[-1]
                                if side.lower() == 'buy' or side.lower() == 'long':
                                    pnl_value = (current_price - entry_price) * size
                                else:
                                    pnl_value = (entry_price - current_price) * size
                                pnl = f"${pnl_value:,.2f}"
                        except:
                            pass
                    
                    # Calculate duration
                    duration = "Unknown"
                    if 'entry_time' in position:
                        try:
                            entry_time = position['entry_time']
                            if isinstance(entry_time, str):
                                entry_time = datetime.strptime(entry_time, "%Y-%m-%d %H:%M:%S")
                            time_diff = datetime.now() - entry_time
                            hours, remainder = divmod(time_diff.total_seconds(), 3600)
                            minutes, _ = divmod(remainder, 60)
                            duration = f"{int(hours)}h {int(minutes)}m"
                        except:
                            pass
                    
                    positions.append({
                        "symbol": symbol,
                        "side": side.capitalize(),
                        "size": f"{size:,.4f}",
                        "entry_price": f"${entry_price:,.2f}",
                        "pnl": pnl,
                        "duration": duration
                    })
            except Exception as e:
                logging.error(f"Error getting position info: {e}")
        
        # Update positions table
        self._updatePositionsTable(positions)
    
    def _updateTaskInfo(self):
        """Update task information tab."""
        if not self.trading_system or not hasattr(self.trading_system, 'thread_manager'):
            return
        
        thread_manager = self.trading_system.thread_manager
        
        # Get active tasks
        tasks = []
        active_tasks = {}
        
        try:
            # Get all tasks with details
            active_tasks = thread_manager.get_active_tasks()
            
            for task_id, task_info in active_tasks.items():
                # Format task ID for display
                display_id = task_id
                if len(display_id) > 20:
                    display_id = display_id[:17] + "..."
                
                # Get status
                status = task_info.get('status', 'unknown').capitalize()
                
                # Get progress
                progress = task_info.get('progress', 0)
                
                # Get execution time
                execution_time = task_info.get('execution_time')
                runtime = "Just started"
                if execution_time:
                    if execution_time < 60:
                        runtime = f"{execution_time:.1f} sec"
                    else:
                        minutes, seconds = divmod(execution_time, 60)
                        runtime = f"{int(minutes)}m {int(seconds)}s"
                
                # Get priority
                priority = task_info.get('priority', 0)
                
                tasks.append({
                    "id": task_id,
                    "display_id": display_id,
                    "status": status,
                    "progress": progress,
                    "runtime": runtime,
                    "priority": str(priority)
                })
        except Exception as e:
            logging.error(f"Error getting task info: {e}")
            self.addAlert("task_error", f"Error getting task information: {e}", "error")
        
        # Update task table
        self._updateTaskTable(tasks)
        
        # Update task statistics
        task_stats = {
            "Total Tasks:": "0",
            "Completed Tasks:": "0",
            "Failed Tasks:": "0",
            "Cancelled Tasks:": "0",
            "Average Runtime:": "Unknown",
            "Workers:": "0",
            "Queue Depth:": "0",
            "Last Completed:": "Never",
            "Task Types:": "None"
        }
        
        try:
            # Get task statistics
            stats = thread_manager.get_stats()
            
            if stats:
                task_stats["Total Tasks:"] = str(stats.get("tasks_submitted", 0))
                task_stats["Completed Tasks:"] = str(stats.get("tasks_completed", 0))
                task_stats["Failed Tasks:"] = str(stats.get("tasks_failed", 0))
                task_stats["Cancelled Tasks:"] = str(stats.get("tasks_cancelled", 0))
            
            # Get worker count
            task_stats["Workers:"] = str(len(thread_manager.workers))
            
            # Get queue depth
            try:
                task_stats["Queue Depth:"] = str(thread_manager.task_queue.qsize())
            except:
                pass
                
            # Calculate average runtime (placeholder)
            task_stats["Average Runtime:"] = "0.5s"
            
            # Get last completed task time (placeholder)
            task_stats["Last Completed:"] = datetime.now().strftime("%H:%M:%S")
            
            # Count task types (placeholder)
            task_stats["Task Types:"] = "3 types"
            
        except Exception as e:
            logging.error(f"Error getting task stats: {e}")
        
        # Update statistics labels
        for label, value in task_stats.items():
            if label in self.task_stats_values:
                self.task_stats_values[label].setText(value)
    
    def _updateExchangeTable(self, exchanges):
        """
        Update the exchange table with current data.
        
        Args:
            exchanges: List of exchange data dictionaries
        """
        # Get the exchange table
        exchange_layout = None
        
        for i in range(self.tab_widget.count()):
            if self.tab_widget.tabText(i) == "Trading":
                trading_tab = self.tab_widget.widget(i)
                for child in trading_tab.children():
                    if isinstance(child, QGroupBox) and child.title() == "Exchange Connections":
                        exchange_layout = child.layout()
                        break
                break
        
        if not exchange_layout:
            return
        
        # Clear existing rows
        for i in range(1, self.exchange_rows + 1):
            for j in range(5):
                item = exchange_layout.itemAtPosition(i, j)
                if item:
                    widget = item.widget()
                    if widget:
                        widget.deleteLater()
        
        # Add new rows
        for i, exchange in enumerate(exchanges):
            row = i + 1
            
            # Exchange name
            name_label = QLabel(exchange["exchange"])
            exchange_layout.addWidget(name_label, row, 0)
            
            # Status
            status_label = QLabel(exchange["status"])
            if exchange["status"] == "Online":
                status_label.setStyleSheet("color: #27ae60;")  # Green
            elif exchange["status"] == "Limited":
                status_label.setStyleSheet("color: #f39c12;")  # Orange
            else:
                status_label.setStyleSheet("color: #e74c3c;")  # Red
            exchange_layout.addWidget(status_label, row, 1)
            
            # Balance
            balance_label = QLabel(exchange["balance"])
            exchange_layout.addWidget(balance_label, row, 2)
            
            # Rate limit
            rate_limit_label = QLabel(exchange["rate_limit"])
            exchange_layout.addWidget(rate_limit_label, row, 3)
            
            # Last update
            update_label = QLabel(exchange["last_update"])
            exchange_layout.addWidget(update_label, row, 4)
        
        self.exchange_rows = len(exchanges)
    
    def _updateStrategyTable(self, strategies):
        """
        Update the strategy table with current data.
        
        Args:
            strategies: List of strategy data dictionaries
        """
        # Get the strategy table
        strategy_layout = None
        
        for i in range(self.tab_widget.count()):
            if self.tab_widget.tabText(i) == "Trading":
                trading_tab = self.tab_widget.widget(i)
                for child in trading_tab.children():
                    if isinstance(child, QGroupBox) and child.title() == "Active Strategies":
                        strategy_layout = child.layout()
                        break
                break
        
        if not strategy_layout:
            return
        
        # Clear existing rows
        for i in range(1, self.strategy_rows + 1):
            for j in range(5):
                item = strategy_layout.itemAtPosition(i, j)
                if item:
                    widget = item.widget()
                    if widget:
                        widget.deleteLater()
        
        # Add new rows
        for i, strategy in enumerate(strategies):
            row = i + 1
            
            # Strategy name
            name_label = QLabel(strategy["strategy"])
            strategy_layout.addWidget(name_label, row, 0)
            
            # Status
            status_label = QLabel(strategy["status"])
            if strategy["status"] == "Active":
                status_label.setStyleSheet("color: #27ae60;")  # Green
            else:
                status_label.setStyleSheet("color: #e74c3c;")  # Red
            strategy_layout.addWidget(status_label, row, 1)
            
            # P&L
            pnl_label = QLabel(strategy["pnl"])
            strategy_layout.addWidget(pnl_label, row, 2)
            
            # Trades
            trades_label = QLabel(strategy["trades"])
            strategy_layout.addWidget(trades_label, row, 3)
            
            # Win rate
            win_rate_label = QLabel(strategy["win_rate"])
            strategy_layout.addWidget(win_rate_label, row, 4)
        
        self.strategy_rows = len(strategies)
    
    def _updatePositionsTable(self, positions):
        """
        Update the positions table with current data.
        
        Args:
            positions: List of position data dictionaries
        """
        # Get the positions table
        positions_layout = None
        
        for i in range(self.tab_widget.count()):
            if self.tab_widget.tabText(i) == "Trading":
                trading_tab = self.tab_widget.widget(i)
                for child in trading_tab.children():
                    if isinstance(child, QGroupBox) and child.title() == "Open Positions":
                        positions_layout = child.layout()
                        break
                break
        
        if not positions_layout:
            return
        
        # Clear existing rows
        for i in range(1, self.position_rows + 1):
            for j in range(6):
                item = positions_layout.itemAtPosition(i, j)
                if item:
                    widget = item.widget()
                    if widget:
                        widget.deleteLater()
        
        # Add new rows
        for i, position in enumerate(positions):
            row = i + 1
            
            # Symbol
            symbol_label = QLabel(position["symbol"])
            positions_layout.addWidget(symbol_label, row, 0)
            
            # Side
            side_label = QLabel(position["side"])
            if position["side"].lower() in ["buy", "long"]:
                side_label.setStyleSheet("color: #27ae60;")  # Green
            else:
                side_label.setStyleSheet("color: #e74c3c;")  # Red
            positions_layout.addWidget(side_label, row, 1)
            
            # Size
            size_label = QLabel(position["size"])
            positions_layout.addWidget(size_label, row, 2)
            
            # Entry price
            entry_label = QLabel(position["entry_price"])
            positions_layout.addWidget(entry_label, row, 3)
            
            # P&L
            pnl_label = QLabel(position["pnl"])
            if position["pnl"].startswith("$-"):
                pnl_label.setStyleSheet("color: #e74c3c;")  # Red
            elif position["pnl"] != "Unknown" and not position["pnl"].startswith("$0.00"):
                pnl_label.setStyleSheet("color: #27ae60;")  # Green
            positions_layout.addWidget(pnl_label, row, 4)
            
            # Duration
            duration_label = QLabel(position["duration"])
            positions_layout.addWidget(duration_label, row, 5)
        
        self.position_rows = len(positions)
    
    def _updateTaskTable(self, tasks):
        """
        Update the task table with current data.
        
        Args:
            tasks: List of task data dictionaries
        """
        # Get the task table
        task_layout = None
        
        for i in range(self.tab_widget.count()):
            if self.tab_widget.tabText(i) == "Tasks":
                tasks_tab = self.tab_widget.widget(i)
                for child in tasks_tab.children():
                    if isinstance(child, QGroupBox) and child.title() == "Active Tasks":
                        task_layout = child.layout()
                        break
                break
        
        if not task_layout:
            return
        
        # Clear existing rows
        for i in range(1, self.task_rows + 1):
            for j in range(5):
                item = task_layout.itemAtPosition(i, j)
                if item:
                    widget = item.widget()
                    if widget:
                        widget.deleteLater()
            
            # Also remove any progress bars
            task_id = f"task_{i-1}"
            if task_id in self.task_progresses:
                self.task_progresses[task_id].deleteLater()
                del self.task_progresses[task_id]
        
        # Add new rows
        for i, task in enumerate(tasks):
            row = i + 1
            task_id = task["id"]
            
            # Task ID
            name_label = QLabel(task["display_id"])
            name_label.setToolTip(task_id)
            task_layout.addWidget(name_label, row, 0)
            
            # Status
            status_label = QLabel(task["status"])
            if task["status"] == "Running":
                status_label.setStyleSheet("color: #27ae60;")  # Green
            elif task["status"] == "Pending":
                status_label.setStyleSheet("color: #f39c12;")  # Orange
            elif task["status"] == "Failed":
                status_label.setStyleSheet("color: #e74c3c;")  # Red
            task_layout.addWidget(status_label, row, 1)
            
            # Progress
            progress_bar = QProgressBar()
            progress_bar.setRange(0, 100)
            progress_bar.setValue(task["progress"])
            progress_bar.setTextVisible(True)
            task_layout.addWidget(progress_bar, row, 2)
            self.task_progresses[task_id] = progress_bar
            
            # Runtime
            runtime_label = QLabel(task["runtime"])
            task_layout.addWidget(runtime_label, row, 3)
            
            # Priority
            priority_label = QLabel(task["priority"])
            task_layout.addWidget(priority_label, row, 4)
        
        self.task_rows = len(tasks)
    
    def addAlert(self, alert_id: str, message: str, severity: str = "info"):
        """
        Add a new alert to the alerts section.
        
        Args:
            alert_id: Unique identifier for the alert
            message: Alert message text
            severity: Alert severity level (info, warning, error, critical)
        """
        # Check if alert already exists
        if alert_id in self.alerts:
            # Update existing alert
            self.alerts[alert_id].deleteLater()
            del self.alerts[alert_id]
        
        # Create new alert
        alert = AlertItem(alert_id, message, severity)
        alert.dismissed.connect(self._onAlertDismissed)
        
        # Add to layout at the top
        self.alerts_layout.insertWidget(0, alert)
        
        # Store reference
        self.alerts[alert_id] = alert
        
        # Limit maximum number of alerts
        if len(self.alerts) > 10:
            # Remove oldest alert
            oldest_alert = None
            for i in range(self.alerts_layout.count() - 1, -1, -1):
                item = self.alerts_layout.itemAt(i)
                if item and item.widget() and isinstance(item.widget(), AlertItem):
                    oldest_alert = item.widget()
                    break
                    
            if oldest_alert:
                alert_id = oldest_alert.alert_id
                oldest_alert.deleteLater()
                if alert_id in self.alerts:
                    del self.alerts[alert_id]
    
    def _onAlertDismissed(self, alert_id: str):
        """
        Handle alert dismissal.
        
        Args:
            alert_id: ID of the dismissed alert
        """
        if alert_id in self.alerts:
            # Remove alert widget
            self.alerts[alert_id].deleteLater()
            del self.alerts[alert_id]
            
            # Emit signal
            self.alert_dismissed.emit(alert_id)
    
    def clearAlerts(self):
        """Clear all alerts."""
        for alert_id, alert in list(self.alerts.items()):
            alert.deleteLater()
            del self.alerts[alert_id]
    
    def _onStatusClicked(self, component: str, status: str, details: str):
        """
        Handle component status indicator click.
        
        Args:
            component: Name of the component
            status: Current status
            details: Additional details
        """
        # Show component details dialog
        dialog = ComponentDetailsDialog(component, status, details, self)
        dialog.exec_()
    
    def _showActionMenu(self):
        """Show the actions menu."""
        menu = QMenu(self)
        
        # Basic actions
        menu.addAction("Refresh Status", self.refreshStatus)
        menu.addAction("Clear Alerts", self.clearAlerts)
        
        # Component actions
        components_menu = menu.addMenu("Components")
        components_menu.addAction("Restart Trading System", lambda: self._requestComponentAction("Trading System", "restart"))
        components_menu.addAction("Reconnect Database", lambda: self._requestComponentAction("Database", "reconnect"))
        components_menu.addAction("Reload Strategies", lambda: self._requestComponentAction("Strategies", "reload"))
        components_menu.addAction("Reset Risk Manager", lambda: self._requestComponentAction("Risk Manager", "reset"))
        
        # AI actions
        ai_menu = menu.addMenu("AI Components")
        ai_menu.addAction("Reload ML Models", lambda: self._requestComponentAction("ML Models", "reload"))
        ai_menu.addAction("Retrain ML Models", lambda: self._requestComponentAction("ML Models", "retrain"))
        
        # Trading actions
        trading_menu = menu.addMenu("Trading")
        trading_menu.addAction("Close All Positions", lambda: self._requestComponentAction("Trading", "close_all"))
        trading_menu.addAction("Disable Trading", lambda: self._requestComponentAction("Trading", "disable"))
        trading_menu.addAction("Enable Trading", lambda: self._requestComponentAction("Trading", "enable"))
        
        # Show menu
        menu.exec_(QCursor.pos())
    
    def _requestComponentAction(self, component: str, action: str):
        """
        Request an action on a component.
        
        Args:
            component: Component name
            action: Action to perform
        """
        self.component_action_requested.emit(component, action)

class ComponentDetailsDialog(QDialog):
    """Dialog showing detailed information about a system component."""
    
    def __init__(self, component: str, status: str, details: str, parent=None):
        super().__init__(parent)
        
        self.component = component
        self.status = status
        self.details = details
        
        self.setWindowTitle(f"{component} Details")
        self.setupUI()
    
    def setupUI(self):
        """Setup the UI components."""
        layout = QVBoxLayout(self)
        
        # Header
        header_layout = QHBoxLayout()
        
        # Status indicator
        status_color = QColor(149, 165, 166)  # Default gray
        if self.status == "online":
            status_color = QColor(39, 174, 96)  # Green
        elif self.status == "warning":
            status_color = QColor(230, 126, 34)  # Orange
        elif self.status == "offline":
            status_color = QColor(231, 76, 60)  # Red
            
        status_indicator = QFrame()
        status_indicator.setFixedSize(16, 16)
        status_indicator.setFrameShape(QFrame.Box)
        status_indicator.setFrameShadow(QFrame.Plain)
        status_indicator.setAutoFillBackground(True)
        
        palette = status_indicator.palette()
        palette.setColor(QPalette.Window, status_color)
        status_indicator.setPalette(palette)
        
        # Component name
        name_label = QLabel(self.component)
        font = QFont()
        font.setBold(True)
        font.setPointSize(14)
        name_label.setFont(font)
        
        # Status text
        status_label = QLabel(self.status.capitalize())
        
        header_layout.addWidget(status_indicator)
        header_layout.addWidget(name_label, 1)
        header_layout.addWidget(status_label)
        
        layout.addLayout(header_layout)
        
        # Details
        if self.details:
            details_label = QLabel(self.details)
            details_label.setWordWrap(True)
            layout.addWidget(details_label)
        
        # Additional component-specific information
        # This would be expanded with actual component details
        
        # Close button
        button_box = QDialogButtonBox(QDialogButtonBox.Close)
        button_box.rejected.connect(self.reject)
        layout.addWidget(button_box)
        
        # Set minimum size
        self.setMinimumSize(400, 200)

# Integrate with error handling system if available
if HAVE_ERROR_HANDLING:
    # Create automatic alert handler
    def error_alert_handler(error, context=None):
        """Error handler that shows alerts in the status widget."""
        if hasattr(error, 'severity') and isinstance(error.severity, ErrorSeverity):
            severity = error.severity.name.lower()
            if severity == "debug" or severity == "info":
                severity = "info"
            elif severity == "warning":
                severity = "warning"
            else:
                severity = "error"
        else:
            severity = "error"
            
        message = str(error)
        if hasattr(error, 'message'):
            message = error.message
            
        # Find all status widgets
        for widget in QApplication.allWidgets():
            if isinstance(widget, StatusWidget):
                widget.addAlert(f"error_{int(time.time())}", message, severity)
                
        return None
        
    # Register handler for different error categories
    ErrorHandler.register_recovery_strategy(ErrorCategory.DATABASE, error_alert_handler)
    ErrorHandler.register_recovery_strategy(ErrorCategory.NETWORK, error_alert_handler)
    ErrorHandler.register_recovery_strategy(ErrorCategory.MODEL_LOADING, error_alert_handler)
    ErrorHandler.register_recovery_strategy(ErrorCategory.STRATEGY, error_alert_handler)
    ErrorHandler.register_recovery_strategy(ErrorCategory.RISK, error_alert_handler)
