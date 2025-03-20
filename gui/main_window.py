#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import json
import logging

from PyQt5.QtWidgets import (QMainWindow, QTabWidget, QAction, QMessageBox, 
                             QFileDialog, QDockWidget, QStatusBar)
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


class MainWindow(QMainWindow):
    """Main window for the trading application."""

    def __init__(self, trading_system, parent=None):
        super().__init__(parent)
        self.trading_system = trading_system

        # Set window properties
        self.setWindowTitle("AI Crypto Trading Platform")
        self.setMinimumSize(1200, 800)

        # Initialize UI
        self._init_ui()

        # Start update timer
        self.timer = QTimer(self)
        self.timer.timeout.connect(self._update_status)
        self.timer.start(1000)  # Update every second

    def _init_ui(self):
        """Initialize the user interface."""
        # Create central tab widget
        self.tabs = QTabWidget()
        self.setCentralWidget(self.tabs)

        # Create tabs
        self.dashboard_tab = DashboardTab(self.trading_system)
        self.trading_tab = TradingTab(self.trading_system)
        self.backtester_tab = BacktesterTab(self.trading_system)
        self.strategy_tab = StrategyTab(self.trading_system)
        self.risk_tab = RiskTab(self.trading_system)
        self.chart_tab = ChartTab(self.trading_system)
        self.analytics_tab = AnalyticsTab(self.trading_system)
        self.settings_tab = SettingsTab(self.trading_system)
        self.portfolio_view = PortfolioView(self.trading_system)

        # Add tabs to the tab widget
        self.tabs.addTab(self.dashboard_tab, "Dashboard")
        self.tabs.addTab(self.trading_tab, "Trading")
        self.tabs.addTab(self.backtester_tab, "Backtesting")
        self.tabs.addTab(self.strategy_tab, "Strategies")
        self.tabs.addTab(self.risk_tab, "Risk Management")
        self.tabs.addTab(self.chart_tab, "Charts")
        self.tabs.addTab(self.analytics_tab, "Analytics")
        self.tabs.addTab(self.settings_tab, "Settings")
        self.tabs.addTab(self.portfolio_view, "Portfolio")

        # Connect PortfolioView signals
        self.portfolio_view.close_position_requested.connect(self.close_position)
        self.portfolio_view.modify_position_requested.connect(self.modify_position)
        self.portfolio_view.new_order_requested.connect(self.open_order_panel)

        # Create status bar and add status widget
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_widget = StatusWidget(self.trading_system)
        self.status_bar.addPermanentWidget(self.status_widget)

        # Create notification dock widget
        self.notification_dock = QDockWidget("Notifications", self)
        self.notification_dock.setAllowedAreas(Qt.RightDockWidgetArea)
        self.notification_widget = NotificationWidget()
        self.notification_dock.setWidget(self.notification_widget)
        self.addDockWidget(Qt.RightDockWidgetArea, self.notification_dock)

        # Create the menu bar
        self._create_menu_bar()

    def _create_menu_bar(self):
        """Create the main menu bar."""
        # File menu
        file_menu = self.menuBar().addMenu("&File")

        open_config_action = QAction("&Open Configuration", self)
        open_config_action.setShortcut("Ctrl+O")
        open_config_action.triggered.connect(self._open_configuration)
        file_menu.addAction(open_config_action)

        save_config_action = QAction("&Save Configuration", self)
        save_config_action.setShortcut("Ctrl+S")
        save_config_action.triggered.connect(self._save_configuration)
        file_menu.addAction(save_config_action)

        file_menu.addSeparator()

        load_state_action = QAction("Load System &State", self)
        load_state_action.triggered.connect(self._load_system_state)
        file_menu.addAction(load_state_action)

        save_state_action = QAction("Save System S&tate", self)
        save_state_action.triggered.connect(self._save_system_state)
        file_menu.addAction(save_state_action)

        file_menu.addSeparator()

        exit_action = QAction("E&xit", self)
        exit_action.setShortcut("Ctrl+Q")
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)

        # Trading menu
        trading_menu = self.menuBar().addMenu("&Trading")

        self.start_action = QAction("&Start Trading", self)
        self.start_action.triggered.connect(self._start_trading)
        trading_menu.addAction(self.start_action)

        self.stop_action = QAction("S&top Trading", self)
        self.stop_action.triggered.connect(self._stop_trading)
        self.stop_action.setEnabled(False)
        trading_menu.addAction(self.stop_action)

        trading_menu.addSeparator()

        self.enable_real_trading_action = QAction("Enable &Real Trading", self)
        self.enable_real_trading_action.setCheckable(True)
        self.enable_real_trading_action.setChecked(self.trading_system.config["trading"]["use_real_trading"])
        self.enable_real_trading_action.triggered.connect(self._toggle_real_trading)
        trading_menu.addAction(self.enable_real_trading_action)

        # View menu
        view_menu = self.menuBar().addMenu("&View")

        toggle_notifications_action = QAction("&Notifications", self)
        toggle_notifications_action.setCheckable(True)
        toggle_notifications_action.setChecked(True)
        toggle_notifications_action.triggered.connect(
            lambda checked: self.notification_dock.setVisible(checked))
        view_menu.addAction(toggle_notifications_action)

        # Tools menu
        tools_menu = self.menuBar().addMenu("&Tools")

        config_editor_action = QAction("&Configuration Editor", self)
        config_editor_action.triggered.connect(self._open_config_editor)
        tools_menu.addAction(config_editor_action)

        # Help menu
        help_menu = self.menuBar().addMenu("&Help")

        about_action = QAction("&About", self)
        about_action.triggered.connect(self._show_about_dialog)
        help_menu.addAction(about_action)

    def _open_configuration(self):
        """Open configuration file."""
        file_name, _ = QFileDialog.getOpenFileName(
            self, "Open Configuration", "", "JSON Files (*.json)"
        )
        if file_name:
            try:
                self.trading_system.config = self.trading_system._load_config(file_name)
                self.notification_widget.add_notification("Configuration loaded successfully")
                self.enable_real_trading_action.setChecked(
                    self.trading_system.config["trading"]["use_real_trading"]
                )
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to load configuration: {e}")

    def _save_configuration(self):
        """Save configuration file."""
        file_name, _ = QFileDialog.getSaveFileName(
            self, "Save Configuration", "", "JSON Files (*.json)"
        )
        if file_name:
            try:
                with open(file_name, 'w') as f:
                    json.dump(self.trading_system.config, f, indent=4)
                self.notification_widget.add_notification("Configuration saved successfully")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to save configuration: {e}")

    def _load_system_state(self):
        """Load system state."""
        file_name, _ = QFileDialog.getOpenFileName(
            self, "Load System State", "data/", "JSON Files (*.json)"
        )
        if file_name:
            try:
                success = self.trading_system.load_system_state(file_name)
                if success:
                    self.notification_widget.add_notification("System state loaded successfully")
                    self._update_status()
                    self.dashboard_tab.refresh_data()
                    self.trading_tab.refresh_data()
                else:
                    QMessageBox.warning(self, "Warning", "Failed to load system state")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to load system state: {e}")

    def _save_system_state(self):
        """Save system state."""
        file_name, _ = QFileDialog.getSaveFileName(
            self, "Save System State", "data/", "JSON Files (*.json)"
        )
        if file_name:
            try:
                success = self.trading_system.save_system_state(os.path.basename(file_name))
                if success:
                    self.notification_widget.add_notification("System state saved successfully")
                else:
                    QMessageBox.warning(self, "Warning", "Failed to save system state")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to save system state: {e}")

    def _start_trading(self):
        """Start trading."""
        try:
            if not self.trading_system.initialized:
                QMessageBox.warning(self, "Warning", "Trading system not initialized")
                return
            success = self.trading_system.start()
            if success:
                self.notification_widget.add_notification("Trading started")
                self.start_action.setEnabled(False)
                self.stop_action.setEnabled(True)
                self.status_widget.update_status()
            else:
                QMessageBox.warning(self, "Warning", "Failed to start trading")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to start trading: {e}")

    def _stop_trading(self):
        """Stop trading."""
        try:
            success = self.trading_system.stop()
            if success:
                self.notification_widget.add_notification("Trading stopped")
                self.start_action.setEnabled(True)
                self.stop_action.setEnabled(False)
                self.status_widget.update_status()
            else:
                QMessageBox.warning(self, "Warning", "Failed to stop trading")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to stop trading: {e}")

    def _toggle_real_trading(self, checked):
        """Toggle real trading mode."""
        try:
            if checked:
                reply = QMessageBox.question(
                    self, "Enable Real Trading",
                    "Are you sure you want to enable real trading? This will use real funds.",
                    QMessageBox.Yes | QMessageBox.No, QMessageBox.No
                )
                if reply == QMessageBox.No:
                    self.enable_real_trading_action.setChecked(False)
                    return
            self.trading_system.config["trading"]["use_real_trading"] = checked
            status = "enabled" if checked else "disabled"
            self.notification_widget.add_notification(f"Real trading {status}")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to toggle real trading: {e}")

    def _open_config_editor(self):
        """Open configuration editor dialog."""
        dialog = ConfigurationDialog(self.trading_system.config, self)
        if dialog.exec_():
            self.trading_system.config = dialog.get_config()
            self.notification_widget.add_notification("Configuration updated")
            self.enable_real_trading_action.setChecked(
                self.trading_system.config["trading"]["use_real_trading"]
            )

    def _show_about_dialog(self):
        """Show about dialog."""
        QMessageBox.about(
            self, "About AI Crypto Trading Platform",
            """<b>AI Crypto Trading Platform</b><br><br>
            An advanced cryptocurrency trading system with AI capabilities.<br><br>
            Combines technical analysis, machine learning, and reinforcement learning 
            for automated trading."""
        )

    def _update_status(self):
        """Update status display."""
        self.status_widget.update_status()
        self.dashboard_tab.update_status()
        self.trading_tab.update_status()

    def closeEvent(self, event):
        """Handle window close event."""
        if self.trading_system.running:
            reply = QMessageBox.question(
                self, "Exit Confirmation",
                "Trading is still running. Are you sure you want to exit?",
                QMessageBox.Yes | QMessageBox.No, QMessageBox.No
            )
            if reply == QMessageBox.No:
                event.ignore()
                return
        try:
            self.trading_system.save_system_state()
        except Exception as e:
            logging.error(f"Error saving system state: {e}")
        if self.trading_system.running:
            self.trading_system.stop()
        self.trading_system.cleanup()
        event.accept()

    def close_position(self, symbol, price, reason):
        """Handle closing a position."""
        logging.info(f"Closing position: {symbol} at {price} ({reason})")
        self.trading_system.close_position(symbol, price, reason)

    def modify_position(self, symbol, modifications):
        """Handle modifying a position."""
        logging.info(f"Modifying position: {symbol} with {modifications}")
        self.trading_system.update_position(symbol, modifications)

    def open_order_panel(self, order_data):
        """Open order panel with pre-filled data.
        
        This method assumes that self.order_panel and self.tab_order_index are defined.
        """
        # Example implementation – adjust according to your order panel design.
        if hasattr(self, 'order_panel') and hasattr(self, 'tab_order_index'):
            self.tabs.setCurrentIndex(self.tab_order_index)
            self.order_panel.set_order_data(order_data)
        else:
            logging.warning("Order panel or tab_order_index is not defined.")


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

