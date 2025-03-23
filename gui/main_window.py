#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import json
import logging
from typing import Optional

from PyQt5.QtWidgets import (QMainWindow, QTabWidget, QAction, QMessageBox, 
                             QFileDialog, QDockWidget, QStatusBar, QWidget, QVBoxLayout, 
                             QHBoxLayout, QPushButton, QLabel)
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QIcon

# Import GUI components for tabs and widgets
from gui.tabs.dashboard_tab import DashboardTab
from gui.tabs.trading_tab import TradingTab
from gui.tabs.backtester_tab import BacktesterTab
from gui.tabs.strategy_tab import StrategyTab
from gui.tabs.risk_tab import RiskTab
from gui.tabs.chart_tab import ChartTab
from gui.tabs.analytics_tab import AnalyticsTab
from gui.tabs.settings_tab import SettingsTab
from gui.widgets.status_widget import StatusWidget
from gui.widgets.notification_widget import NotificationWidget
from gui.dialogs.configuration_dialog import ConfigurationDialog

# Import the PortfolioView
from gui.portfolio_view import PortfolioView
from trading_system import TradingSystem
from .dashboard import Dashboard
from .portfolio_view import PortfolioView
from .order_panel import OrderPanel
from .chart_widget import ChartWidget
from .risk_tab import RiskTab
from .settings_dialog import SettingsDialog
from .alert_widget import AlertWidget


class MainWindow(QMainWindow):
    """Main window for the trading system GUI."""
    
    def __init__(self, trading_system: TradingSystem):
        super().__init__()
        self.trading_system = trading_system
        self.logger = logging.getLogger(__name__)
        
        # Initialize UI components
        self.init_ui()
        
        # Setup update timer
        self.update_timer = QTimer()
        self.update_timer.timeout.connect(self.update_data)
        self.update_timer.start(1000)  # Update every second
    
    def init_ui(self):
        """Initialize the user interface."""
        self.setWindowTitle('Trading System')
        self.setGeometry(100, 100, 1200, 800)
        
        # Create central widget and layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)
        
        # Create header
        header_layout = QHBoxLayout()
        self.status_label = QLabel('System Status: Running')
        header_layout.addWidget(self.status_label)
        
        # Add settings button
        settings_btn = QPushButton('Settings')
        settings_btn.clicked.connect(self.show_settings)
        header_layout.addWidget(settings_btn)
        
        layout.addLayout(header_layout)
        
        # Create tab widget
        tabs = QTabWidget()
        
        # Dashboard tab
        self.dashboard = Dashboard(self.trading_system)
        tabs.addTab(self.dashboard, 'Dashboard')
        
        # Portfolio tab
        portfolio_tab = QWidget()
        portfolio_layout = QHBoxLayout(portfolio_tab)
        
        # Left side - Portfolio view
        self.portfolio_view = PortfolioView(self.trading_system)
        portfolio_layout.addWidget(self.portfolio_view)
        
        # Right side - Order panel
        self.order_panel = OrderPanel(self.trading_system)
        portfolio_layout.addWidget(self.order_panel)
        
        tabs.addTab(portfolio_tab, 'Portfolio')
        
        # Charts tab
        charts_tab = QWidget()
        charts_layout = QVBoxLayout(charts_tab)
        self.chart_widget = ChartWidget(self.trading_system)
        charts_layout.addWidget(self.chart_widget)
        
        tabs.addTab(charts_tab, 'Charts')
        
        # Risk Management tab
        self.risk_tab = RiskTab(self.trading_system)
        tabs.addTab(self.risk_tab, 'Risk Management')
        
        layout.addWidget(tabs)
        
        # Add alert widget at the bottom
        self.alert_widget = AlertWidget()
        layout.addWidget(self.alert_widget)
        
        # Set the layout
        central_widget.setLayout(layout)
    
    def update_data(self):
        """Update all UI components with latest data."""
        try:
            # Update portfolio status
            portfolio_status = self.trading_system.update_portfolio_status()
            
            # Update UI components
            self.dashboard.update_data(portfolio_status)
            self.portfolio_view.update_data(portfolio_status)
            self.chart_widget.update_data()
            self.risk_tab.update_data()
            
            # Update status label
            self.status_label.setText(
                f'System Status: Running | '
                f'Account Value: ${portfolio_status["account_value"]:,.2f}'
            )
            
        except Exception as e:
            self.logger.error(f"Error updating UI: {str(e)}")
            self.alert_widget.add_alert(
                "Error updating data", 
                str(e), 
                "error"
            )
    
    def show_settings(self):
        """Show the settings dialog."""
        dialog = SettingsDialog(self.trading_system, self)
        dialog.exec_()
    
    def closeEvent(self, event):
        """Handle application shutdown."""
        reply = QMessageBox.question(
            self, 'Exit',
            "Are you sure you want to exit?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )
        
        if reply == QMessageBox.Yes:
            # Stop the update timer
            self.update_timer.stop()
            
            # Cleanup trading system
            try:
                self.trading_system.cleanup()
                self.logger.info("Trading system shutdown complete")
            except Exception as e:
                self.logger.error(f"Error during shutdown: {str(e)}")
            
            event.accept()
        else:
            event.ignore()


# If you want to run this file directly for testing, uncomment the lines below.
# if __name__ == '__main__':
#     from PyQt5.QtWidgets import QApplication
#     import sys
#     app = QApplication(sys.argv)
#     # You would need a trading_system object to instantiate MainWindow.
#     trading_system = ...  # Initialize your trading system here.
#     window = MainWindow(trading_system)
#     window.show()
#     sys.exit(app.exec_())

