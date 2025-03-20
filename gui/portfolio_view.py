# portfolio_view.py

import os
import time
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Union

# PyQt imports
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QFrame, QTableWidget, QTableWidgetItem,
    QPushButton, QHeaderView, QSplitter, QComboBox, QGridLayout, QMenu, QAction,
    QMessageBox, QDoubleSpinBox, QDialog, QDialogButtonBox, QFormLayout, QGroupBox,
    QTabWidget, QStackedWidget, QProgressBar, QToolButton, QSizePolicy, QLineEdit,
    QCheckBox, QToolBar, QStatusBar, QScrollArea
)
from PyQt5.QtCore import Qt, QTimer, pyqtSignal, pyqtSlot, QSize, QThread, QDateTime
from PyQt5.QtGui import QColor, QPainter, QBrush, QPen, QFont, QIcon, QPixmap, QPalette

# Chart imports
try:
    import pyqtgraph as pg
    HAVE_PYQTGRAPH = True
except ImportError:
    HAVE_PYQTGRAPH = False
    logging.warning("PyQtGraph not available. Chart features will be disabled.")

# Error handling imports
try:
    from core.error_handling import safe_execute, ErrorCategory, ErrorSeverity, ErrorHandler
    HAVE_ERROR_HANDLING = True
except ImportError:
    HAVE_ERROR_HANDLING = False
    logging.warning("Error handling module not available. Using basic error handling.")

# Define decorator fallback if error handling module isn't available
if not HAVE_ERROR_HANDLING:
    def safe_execute(error_category, default_return=None, severity=None):
        def decorator(func):
            def wrapper(*args, **kwargs):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    logging.error(f"Error in {func.__name__}: {str(e)}")
                    return default_return
            return wrapper
        return decorator

class AssetAllocationChart(QWidget):
    """Pie chart showing portfolio asset allocation."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(300, 300)
        self.allocation_data = {}
        self.color_map = {
            "BTC": QColor(247, 147, 26),  # Bitcoin orange
            "ETH": QColor(114, 137, 218),  # Ethereum blue
            "SOL": QColor(20, 241, 149),  # Solana green
            "AVAX": QColor(232, 65, 66),  # Avalanche red
            "CASH": QColor(200, 200, 200),  # Gray for cash
        }
        # Default colors for other assets
        self.default_colors = [
            QColor(66, 134, 244),   # Blue
            QColor(76, 175, 80),    # Green
            QColor(255, 87, 34),    # Orange
            QColor(156, 39, 176),   # Purple
            QColor(255, 193, 7),    # Amber
            QColor(0, 188, 212),    # Cyan
            QColor(233, 30, 99),    # Pink
        ]
        
    def update_allocation(self, allocation_data: Dict[str, float]) -> None:
        """Update chart with new allocation data."""
        self.allocation_data = allocation_data
        self.update()
        
    def paintEvent(self, event):
        """Custom paint event to draw the pie chart."""
        if not self.allocation_data:
            return
            
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        
        # Calculate chart dimensions
        width = self.width()
        height = self.height()
        chart_size = min(width, height) - 20  # Padding
        chart_rect = QSize(chart_size, chart_size)
        
        # Center the chart
        left = (width - chart_size) // 2
        top = (height - chart_size) // 2
        
        # Draw the pie chart
        total = sum(self.allocation_data.values())
        if total <= 0:
            return
            
        start_angle = 0
        color_index = 0
        
        # Sort items by allocation (largest first)
        sorted_items = sorted(self.allocation_data.items(), key=lambda x: x[1], reverse=True)
        
        # Draw legend
        legend_x = 10
        legend_y = top + 20
        legend_item_height = 20
        painter.setFont(QFont("Arial", 9))
        
        # Draw pie segments and legend
        for asset, value in sorted_items:
            angle = 360 * (value / total)
            
            # Get color for this asset
            if asset in self.color_map:
                color = self.color_map[asset]
            else:
                color = self.default_colors[color_index % len(self.default_colors)]
                color_index += 1
            
            # Draw pie segment
            painter.setBrush(QBrush(color))
            painter.setPen(QPen(Qt.white, 1))
            painter.drawPie(left, top, chart_size, chart_size, int(start_angle * 16), int(angle * 16))
            
            # Draw legend item
            painter.fillRect(legend_x, legend_y, 15, 15, color)
            painter.drawRect(legend_x, legend_y, 15, 15)
            painter.drawText(legend_x + 20, legend_y + 12, f"{asset}: {value*100:.1f}%")
            
            legend_y += legend_item_height
            start_angle += angle

class PortfolioValueChart(QWidget):
    """Line chart showing portfolio value over time."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumHeight(200)
        
        # Initialize layout
        self.layout = QVBoxLayout(self)
        
        if HAVE_PYQTGRAPH:
            # Create pyqtgraph plot
            self.plot_widget = pg.PlotWidget()
            self.plot_widget.setBackground('w')
            self.plot_widget.showGrid(x=True, y=True)
            self.plot_widget.setLabel('left', 'Portfolio Value')
            self.plot_widget.setLabel('bottom', 'Date')
            
            # Add to layout
            self.layout.addWidget(self.plot_widget)
            
            # Prepare chart data
            self.x_data = []
            self.y_data = []
            self.plot = None
        else:
            # Fallback when pyqtgraph isn't available
            self.label = QLabel("PyQtGraph not available. Install it to view charts.")
            self.label.setAlignment(Qt.AlignCenter)
            self.layout.addWidget(self.label)
    
    @safe_execute(ErrorCategory.DATA_PROCESSING, None)
    def update_chart(self, data: pd.DataFrame) -> None:
        """Update chart with new portfolio value data."""
        if not HAVE_PYQTGRAPH or data is None or data.empty:
            return
            
        # Extract data
        if 'time' in data.columns and 'total_value' in data.columns:
            # Sort by time
            data = data.sort_values('time')
            
            # Convert timestamps to matplotlib dates
            self.x_data = pd.to_datetime(data['time']).map(lambda x: x.timestamp())
            self.y_data = data['total_value'].values
            
            # Clear previous plot
            self.plot_widget.clear()
            
            # Create new plot
            pen = pg.mkPen(color=(66, 134, 244), width=2)
            self.plot = self.plot_widget.plot(self.x_data, self.y_data, pen=pen)
            
            # Set axes
            self.plot_widget.getAxis('bottom').setStyle(tickFont=QFont('Arial', 8))
            self.plot_widget.getAxis('left').setStyle(tickFont=QFont('Arial', 8))
            
            # Add date labels
            date_axis = pg.DateAxisItem(orientation='bottom')
            self.plot_widget.setAxisItems({"bottom": date_axis})

class PortfolioMetricsWidget(QWidget):
    """Widget displaying key portfolio metrics."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        # Initialize layout
        self.layout = QGridLayout(self)
        self.layout.setSpacing(10)
        
        # Create metric labels
        self.metrics = {
            "Total Value": self._create_metric_widget("$0.00"),
            "Daily P&L": self._create_metric_widget("$0.00 (0.00%)"),
            "Weekly P&L": self._create_metric_widget("$0.00 (0.00%)"),
            "Monthly P&L": self._create_metric_widget("$0.00 (0.00%)"),
            "Open Positions": self._create_metric_widget("0"),
            "Buying Power": self._create_metric_widget("$0.00"),
            "Margin Used": self._create_metric_widget("0.00%"),
            "Portfolio Beta": self._create_metric_widget("0.00"),
        }
        
        # Add metrics to grid layout
        row, col = 0, 0
        for label, (frame, _) in self.metrics.items():
            self.layout.addWidget(QLabel(f"{label}:"), row, col*2)
            self.layout.addWidget(frame, row, col*2 + 1)
            
            col += 1
            if col >= 4:
                col = 0
                row += 1
    
    def _create_metric_widget(self, initial_text: str) -> Tuple[QFrame, QLabel]:
        """Create a styled frame with value label."""
        frame = QFrame()
        frame.setFrameShape(QFrame.StyledPanel)
        frame.setMinimumWidth(120)
        
        layout = QVBoxLayout(frame)
        layout.setContentsMargins(5, 5, 5, 5)
        
        value_label = QLabel(initial_text)
        value_label.setAlignment(Qt.AlignCenter)
        value_label.setFont(QFont("Arial", 10, QFont.Bold))
        
        layout.addWidget(value_label)
        
        return frame, value_label
    
    @safe_execute(ErrorCategory.DATA_PROCESSING, None)
    def update_metrics(self, portfolio_data: Dict[str, Any]) -> None:
        """Update metrics with new portfolio data."""
        # Update total value
        if 'total_value' in portfolio_data:
            total_value = portfolio_data['total_value']
            self.metrics["Total Value"][1].setText(f"${total_value:,.2f}")
            
            # Set color based on value change
            if 'prev_total_value' in portfolio_data:
                prev_value = portfolio_data['prev_total_value']
                if total_value > prev_value:
                    self.metrics["Total Value"][1].setStyleSheet("color: green;")
                elif total_value < prev_value:
                    self.metrics["Total Value"][1].setStyleSheet("color: red;")
            
        # Update P&L metrics
        for period in ['daily', 'weekly', 'monthly']:
            key = f"{period}_pnl"
            pct_key = f"{period}_pnl_pct"
            
            if key in portfolio_data and pct_key in portfolio_data:
                pnl = portfolio_data[key]
                pnl_pct = portfolio_data[pct_key]
                
                label = self.metrics[f"{period.capitalize()} P&L"][1]
                label.setText(f"${pnl:+,.2f} ({pnl_pct:+.2f}%)")
                
                if pnl > 0:
                    label.setStyleSheet("color: green;")
                elif pnl < 0:
                    label.setStyleSheet("color: red;")
                else:
                    label.setStyleSheet("")
                    
        # Update other metrics
        if 'open_positions' in portfolio_data:
            self.metrics["Open Positions"][1].setText(f"{portfolio_data['open_positions']}")
            
        if 'buying_power' in portfolio_data:
            self.metrics["Buying Power"][1].setText(f"${portfolio_data['buying_power']:,.2f}")
            
        if 'margin_used' in portfolio_data:
            margin_pct = portfolio_data['margin_used']
            self.metrics["Margin Used"][1].setText(f"{margin_pct:.2f}%")
            
            # Set color based on margin usage
            if margin_pct > 80:
                self.metrics["Margin Used"][1].setStyleSheet("color: red; font-weight: bold;")
            elif margin_pct > 50:
                self.metrics["Margin Used"][1].setStyleSheet("color: orange;")
            else:
                self.metrics["Margin Used"][1].setStyleSheet("")
                
        if 'portfolio_beta' in portfolio_data:
            self.metrics["Portfolio Beta"][1].setText(f"{portfolio_data['portfolio_beta']:.2f}")

class PositionsTableWidget(QTableWidget):
    """Table widget displaying open positions with advanced features."""
    
    # Signals
    close_position_requested = pyqtSignal(str, float, str)  # Symbol, price, reason
    modify_position_requested = pyqtSignal(str, dict)  # Symbol, modifications
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        # Configure table
        self.setColumnCount(10)
        self.setHorizontalHeaderLabels([
            "Symbol", "Side", "Size", "Entry Price", "Current Price", 
            "P&L", "P&L %", "Stop Loss", "Take Profit", "Actions"
        ])
        
        # Style table
        self.setAlternatingRowColors(True)
        self.setSelectionBehavior(QTableWidget.SelectRows)
        self.setEditTriggers(QTableWidget.NoEditTriggers)
        self.setShowGrid(True)
        self.setSortingEnabled(True)
        
        # Set column widths
        self.horizontalHeader().setSectionResizeMode(QHeaderView.Interactive)
        self.horizontalHeader().setStretchLastSection(True)
        
        # Default column widths
        self.setColumnWidth(0, 80)   # Symbol
        self.setColumnWidth(1, 70)   # Side
        self.setColumnWidth(2, 80)   # Size
        self.setColumnWidth(3, 100)  # Entry Price
        self.setColumnWidth(4, 100)  # Current Price
        self.setColumnWidth(5, 100)  # P&L
        self.setColumnWidth(6, 80)   # P&L %
        self.setColumnWidth(7, 100)  # Stop Loss
        self.setColumnWidth(8, 100)  # Take Profit
        
        # Enable context menu
        self.setContextMenuPolicy(Qt.CustomContextMenu)
        self.customContextMenuRequested.connect(self._show_context_menu)
    
    @safe_execute(ErrorCategory.DATA_PROCESSING, None)
    def update_positions(self, positions: List[Dict[str, Any]]) -> None:
        """Update table with current positions data."""
        # Save selection
        selected_rows = [item.row() for item in self.selectedItems()]
        selected_symbols = []
        
        if selected_rows:
            selected_symbols = [self.item(row, 0).text() for row in set(selected_rows)]
            
        # Clear table
        self.setRowCount(0)
        
        # Populate table
        for row, position in enumerate(positions):
            self.insertRow(row)
            
            # Symbol
            symbol_item = QTableWidgetItem(position.get('symbol', ''))
            self.setItem(row, 0, symbol_item)
            
            # Side (buy/sell)
            side = position.get('side', '').capitalize()
            side_item = QTableWidgetItem(side)
            side_item.setTextAlignment(Qt.AlignCenter)
            if side.lower() == 'buy':
                side_item.setForeground(QColor('green'))
            elif side.lower() == 'sell':
                side_item.setForeground(QColor('red'))
            self.setItem(row, 1, side_item)
            
            # Size
            size = position.get('position_size', 0)
            size_item = QTableWidgetItem(f"{size:,.6f}")
            size_item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
            self.setItem(row, 2, size_item)
            
            # Entry Price
            entry_price = position.get('entry_price', 0)
            entry_item = QTableWidgetItem(f"${entry_price:,.2f}")
            entry_item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
            self.setItem(row, 3, entry_item)
            
            # Current Price
            current_price = position.get('current_price', 0)
            price_item = QTableWidgetItem(f"${current_price:,.2f}")
            price_item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
            self.setItem(row, 4, price_item)
            
            # P&L
            pnl = position.get('pnl', 0)
            pnl_item = QTableWidgetItem(f"${pnl:+,.2f}")
            pnl_item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
            if pnl > 0:
                pnl_item.setForeground(QColor('green'))
            elif pnl < 0:
                pnl_item.setForeground(QColor('red'))
            self.setItem(row, 5, pnl_item)
            
            # P&L %
            pnl_pct = position.get('pnl_percentage', 0)
            pnl_pct_item = QTableWidgetItem(f"{pnl_pct:+.2f}%")
            pnl_pct_item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
            if pnl_pct > 0:
                pnl_pct_item.setForeground(QColor('green'))
            elif pnl_pct < 0:
                pnl_pct_item.setForeground(QColor('red'))
            self.setItem(row, 6, pnl_pct_item)
            
            # Stop Loss
            stop_loss = position.get('stop_loss', None)
            if stop_loss:
                sl_item = QTableWidgetItem(f"${stop_loss:,.2f}")
            else:
                sl_item = QTableWidgetItem("None")
            sl_item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
            self.setItem(row, 7, sl_item)
            
            # Take Profit
            take_profit = position.get('take_profit', None)
            if take_profit:
                tp_item = QTableWidgetItem(f"${take_profit:,.2f}")
            else:
                tp_item = QTableWidgetItem("None")
            tp_item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
            self.setItem(row, 8, tp_item)
            
            # Actions Button
            actions_widget = QWidget()
            actions_layout = QHBoxLayout(actions_widget)
            actions_layout.setContentsMargins(2, 2, 2, 2)
            
            # Close button
            close_btn = QPushButton("Close")
            close_btn.setToolTip(f"Close {position.get('symbol', '')} position")
            close_btn.clicked.connect(lambda checked, s=position.get('symbol', ''), p=current_price: 
                                     self._handle_close_position(s, p))
            actions_layout.addWidget(close_btn)
            
            # Edit button
            edit_btn = QPushButton("Edit")
            edit_btn.setToolTip(f"Edit {position.get('symbol', '')} position")
            edit_btn.clicked.connect(lambda checked, pos=position: 
                                    self._handle_edit_position(pos))
            actions_layout.addWidget(edit_btn)
            
            self.setCellWidget(row, 9, actions_widget)
            
        # Restore selection
        if selected_symbols:
            for row in range(self.rowCount()):
                symbol = self.item(row, 0).text()
                if symbol in selected_symbols:
                    self.selectRow(row)
    
    def _handle_close_position(self, symbol: str, price: float) -> None:
        """Handle close position button click."""
        reply = QMessageBox.question(
            self, 
            "Close Position", 
            f"Are you sure you want to close {symbol} position at ${price:,.2f}?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )
        
        if reply == QMessageBox.Yes:
            self.close_position_requested.emit(symbol, price, "Manual close")
    
    def _handle_edit_position(self, position: Dict[str, Any]) -> None:
        """Handle edit position button click."""
        symbol = position.get('symbol', '')
        dialog = EditPositionDialog(position, self)
        if dialog.exec_() == QDialog.Accepted:
            modifications = dialog.get_modifications()
            if modifications:
                self.modify_position_requested.emit(symbol, modifications)
    
    def _show_context_menu(self, pos):
        """Show custom context menu for selected position."""
        global_pos = self.mapToGlobal(pos)
        selected_indexes = self.selectedIndexes()
        
        if not selected_indexes:
            return
            
        row = selected_indexes[0].row()
        symbol = self.item(row, 0).text()
        
        menu = QMenu(self)
        
        close_action = menu.addAction(f"Close {symbol} Position")
        close_action.triggered.connect(
            lambda: self._handle_close_position(symbol, float(self.item(row, 4).text().replace('$', '').replace(',', '')))
        )
        
        edit_action = menu.addAction(f"Edit {symbol} Position")
        edit_action.triggered.connect(
            lambda: self._handle_edit_position({
                'symbol': symbol,
                'position_size': float(self.item(row, 2).text().replace(',', '')),
                'entry_price': float(self.item(row, 3).text().replace('$', '').replace(',', '')),
                'stop_loss': None if self.item(row, 7).text() == "None" else float(self.item(row, 7).text().replace('$', '').replace(',', '')),
                'take_profit': None if self.item(row, 8).text() == "None" else float(self.item(row, 8).text().replace('$', '').replace(',', ''))
            })
        )
        
        menu.addSeparator()
        
        # Add technical analysis option
        analyze_action = menu.addAction(f"Technical Analysis for {symbol}")
        analyze_action.triggered.connect(lambda: self._show_technical_analysis(symbol))
        
        # Add stop loss options
        sl_menu = menu.addMenu("Set Stop Loss")
        
        # Predefined stop loss levels
        for pct in [1, 2, 5, 10]:
            action = sl_menu.addAction(f"{pct}% Stop Loss")
            action.triggered.connect(lambda checked, s=symbol, p=pct: self._set_percentage_sl(s, p))
        
        # Add take profit options
        tp_menu = menu.addMenu("Set Take Profit")
        
        # Predefined take profit levels
        for pct in [2, 5, 10, 20]:
            action = tp_menu.addAction(f"{pct}% Take Profit")
            action.triggered.connect(lambda checked, s=symbol, p=pct: self._set_percentage_tp(s, p))
        
        menu.exec_(global_pos)
    
    def _set_percentage_sl(self, symbol: str, percentage: float) -> None:
        """Set stop loss at percentage below entry price."""
        for row in range(self.rowCount()):
            if self.item(row, 0).text() == symbol:
                # Get position details
                side = self.item(row, 1).text().lower()
                entry_price = float(self.item(row, 3).text().replace('$', '').replace(',', ''))
                
                # Calculate stop loss price
                if side == 'buy':
                    stop_price = entry_price * (1 - percentage / 100)
                else:
                    stop_price = entry_price * (1 + percentage / 100)
                
                # Update position
                modifications = {'stop_loss': stop_price}
                self.modify_position_requested.emit(symbol, modifications)
                break
    
    def _set_percentage_tp(self, symbol: str, percentage: float) -> None:
        """Set take profit at percentage above entry price."""
        for row in range(self.rowCount()):
            if self.item(row, 0).text() == symbol:
                # Get position details
                side = self.item(row, 1).text().lower()
                entry_price = float(self.item(row, 3).text().replace('$', '').replace(',', ''))
                
                # Calculate take profit price
                if side == 'buy':
                    take_price = entry_price * (1 + percentage / 100)
                else:
                    take_price = entry_price * (1 - percentage / 100)
                
                # Update position
                modifications = {'take_profit': take_price}
                self.modify_position_requested.emit(symbol, modifications)
                break
    
    def _show_technical_analysis(self, symbol: str) -> None:
        """Show technical analysis for symbol."""
        QMessageBox.information(
            self,
            "Technical Analysis",
            f"Technical analysis for {symbol} would be shown here.\n"
            "This feature will be integrated with the chart system."
        )

class EditPositionDialog(QDialog):
    """Dialog for editing position parameters."""
    
    def __init__(self, position: Dict[str, Any], parent=None):
        super().__init__(parent)
        
        self.position = position
        self.setWindowTitle(f"Edit {position.get('symbol', '')} Position")
        self.setMinimumWidth(350)
        
        # Initialize layout
        self.layout = QVBoxLayout(self)
        
        # Create form
        form_layout = QFormLayout()
        
        # Stop Loss input
        self.sl_enable = QCheckBox("Enable Stop Loss")
        self.sl_enable.setChecked(position.get('stop_loss') is not None)
        
        self.sl_input = QDoubleSpinBox()
        self.sl_input.setRange(0, 1000000)
        self.sl_input.setDecimals(2)
        self.sl_input.setPrefix("$")
        self.sl_input.setSingleStep(1.0)
        if position.get('stop_loss'):
            self.sl_input.setValue(position.get('stop_loss'))
        else:
            # Default to 5% below current price for long, 5% above for short
            price = position.get('current_price', 0)
            side = position.get('side', '').lower()
            if side == 'buy':
                self.sl_input.setValue(price * 0.95)
            else:
                self.sl_input.setValue(price * 1.05)
        
        self.sl_input.setEnabled(self.sl_enable.isChecked())
        self.sl_enable.stateChanged.connect(self.sl_input.setEnabled)
        
        # Add to form
        sl_layout = QHBoxLayout()
        sl_layout.addWidget(self.sl_enable)
        sl_layout.addWidget(self.sl_input)
        form_layout.addRow("Stop Loss:", sl_layout)
        
        # Take Profit input
        self.tp_enable = QCheckBox("Enable Take Profit")
        self.tp_enable.setChecked(position.get('take_profit') is not None)
        
        self.tp_input = QDoubleSpinBox()
        self.tp_input.setRange(0, 1000000)
        self.tp_input.setDecimals(2)
        self.tp_input.setPrefix("$")
        self.tp_input.setSingleStep(1.0)
        if position.get('take_profit'):
            self.tp_input.setValue(position.get('take_profit'))
        else:
            # Default to 10% above current price for long, 10% below for short
            price = position.get('current_price', 0)
            side = position.get('side', '').lower()
            if side == 'buy':
                self.tp_input.setValue(price * 1.1)
            else:
                self.tp_input.setValue(price * 0.9)
        
        self.tp_input.setEnabled(self.tp_enable.isChecked())
        self.tp_enable.stateChanged.connect(self.tp_input.setEnabled)
        
        # Add to form
        tp_layout = QHBoxLayout()
        tp_layout.addWidget(self.tp_enable)
        tp_layout.addWidget(self.tp_input)
        form_layout.addRow("Take Profit:", tp_layout)
        
        # Add form to layout
        self.layout.addLayout(form_layout)
        
        # Add R:R ratio display
        self.rr_label = QLabel("Risk/Reward Ratio: --")
        self.layout.addWidget(self.rr_label)
        
        # Update R:R when values change
        self.sl_input.valueChanged.connect(self._update_rr_ratio)
        self.tp_input.valueChanged.connect(self._update_rr_ratio)
        self.sl_enable.stateChanged.connect(self._update_rr_ratio)
        self.tp_enable.stateChanged.connect(self._update_rr_ratio)
        
        # Update initial R:R
        self._update_rr_ratio()
        
        # Add buttons
        button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)
        self.layout.addWidget(button_box)
    
    def _update_rr_ratio(self) -> None:
        """Update risk/reward ratio display."""
        if not self.sl_enable.isChecked() or not self.tp_enable.isChecked():
            self.rr_label.setText("Risk/Reward Ratio: --")
            return
            
        entry_price = self.position.get('entry_price', 0)
        if entry_price == 0:
            self.rr_label.setText("Risk/Reward Ratio: --")
            return
            
        sl_price = self.sl_input.value()
        tp_price = self.tp_input.value()
        side = self.position.get('side', '').lower()
        
        # Calculate risk and reward
        if side == 'buy':
            risk = abs(entry_price - sl_price)
            reward = abs(tp_price - entry_price)
        else:
            risk = abs(sl_price - entry_price)
            reward = abs(entry_price - tp_price)
            
        # Calculate ratio
        if risk == 0:
            ratio = float('inf')
        else:
            ratio = reward / risk
            
        self.rr_label.setText(f"Risk/Reward Ratio: {ratio:.2f}")
        
        # Color code based on ratio
        if ratio >= 2:
            self.rr_label.setStyleSheet("color: green; font-weight: bold;")
        elif ratio >= 1:
            self.rr_label.setStyleSheet("color: orange;")
        else:
            self.rr_label.setStyleSheet("color: red;")
    
    def get_modifications(self) -> Dict[str, Any]:
        """Get modifications made to the position."""
        modifications = {}
        
        # Stop Loss
        if self.sl_enable.isChecked():
            modifications['stop_loss'] = self.sl_input.value()
        else:
            modifications['stop_loss'] = None
            
        # Take Profit
        if self.tp_enable.isChecked():
            modifications['take_profit'] = self.tp_input.value()
        else:
            modifications['take_profit'] = None
            
        return modifications

class PortfolioView(QWidget):
    """
    Comprehensive portfolio management view with real-time updates.
    
    Features:
    - Portfolio metrics display
    - Asset allocation visualization
    - Position management
    - Portfolio value chart
    - Position details and management
    
    This component integrates with the trading system's thread manager
    and database components for efficient data retrieval.
    """
    
    # Signals
    new_order_requested = pyqtSignal(dict)  # New order request
    close_position_requested = pyqtSignal(str, float, str)  # Symbol, price, reason
    modify_position_requested = pyqtSignal(str, dict)  # Symbol, modifications
    refresh_data_requested = pyqtSignal()  # Manual refresh request
    
    def __init__(self, trading_system, parent=None):
        """
        Initialize the portfolio view.
        
        Args:
            trading_system: The main trading system instance
            parent: Parent widget
        """
        super().__init__(parent)
        
        self.trading_system = trading_system
        self.positions_data = []
        self.portfolio_history = pd.DataFrame()
        self.update_timer = None
        self.last_update_time = None
        
        # Initialize UI
        self._init_ui()
        
        # Set up data update timer
        self.update_timer = QTimer(self)
        self.update_timer.timeout.connect(self.refresh_data)
        self.update_timer.start(5000)  # Update every 5 seconds
        
        # Connect signals
        self.positions_table.close_position_requested.connect(self.close_position_requested)
        self.positions_table.modify_position_requested.connect(self.modify_position_requested)
        
        # Initial data load
        self.refresh_data()
    
    def _init_ui(self):
        """Initialize the UI components."""
        # Main layout
        main_layout = QVBoxLayout(self)
        main_layout.setSpacing(10)
        
        # Top section
        top_layout = QHBoxLayout()
        
        # Portfolio metrics
        self.metrics_widget = PortfolioMetricsWidget()
        top_layout.addWidget(self.metrics_widget)
        
        # Add top section to main layout
        main_layout.addLayout(top_layout)
        
        # Middle section - graph and allocation
        middle_layout = QHBoxLayout()
        
        # Create tabs for charts
        charts_tabs = QTabWidget()
        
        # Portfolio value chart
        self.portfolio_chart = PortfolioValueChart()
        charts_tabs.addTab(self.portfolio_chart, "Portfolio Value")
        
        # Asset allocation chart
        self.allocation_chart = AssetAllocationChart()
        charts_tabs.addTab(self.allocation_chart, "Asset Allocation")
        
        middle_layout.addWidget(charts_tabs)
        
        # Add middle section to main layout
        main_layout.addLayout(middle_layout)
        
        # Bottom section - Positions table
        positions_group = QGroupBox("Open Positions")
        positions_layout = QVBoxLayout(positions_group)
        
        # Positions table
        self.positions_table = PositionsTableWidget()
        positions_layout.addWidget(self.positions_table)
        
        # Controls below table
        controls_layout = QHBoxLayout()
        
        # Refresh button
        refresh_btn = QPushButton("Refresh")
        refresh_btn.clicked.connect(self.refresh_data)
        controls_layout.addWidget(refresh_btn)
        
        # New Order button
        new_order_btn = QPushButton("New Order")
        new_order_btn.clicked.connect(self._open_new_order_dialog)
        controls_layout.addWidget(new_order_btn)
        
        # Close All button
        close_all_btn = QPushButton("Close All Positions")
        close_all_btn.clicked.connect(self._confirm_close_all)
        controls_layout.addWidget(close_all_btn)
        
        # Last update time
        self.update_label = QLabel("Last update: Never")
        self.update_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        controls_layout.addWidget(self.update_label)
        
        positions_layout.addLayout(controls_layout)
        
        # Add positions section to main layout
        main_layout.addWidget(positions_group)
    
    @safe_execute(ErrorCategory.DATA_PROCESSING, None)
    def refresh_data(self) -> None:
        """Refresh portfolio data."""
        # Set update time
        self.last_update_time = datetime.now()
        self.update_label.setText(f"Last update: {self.last_update_time.strftime('%H:%M:%S')}")
        
        # Check if thread manager is available
        if hasattr(self.trading_system, 'thread_manager'):
            # Use thread manager for non-blocking data retrieval
            self.trading_system.thread_manager.submit_task(
                "portfolio_data_refresh",
                self._fetch_data_async,
                priority=1  # High priority
            )
        else:
            # Synchronous data fetching (fallback)
            self._fetch_data_sync()
    
    def _fetch_data_async(self):
        """Fetch portfolio data asynchronously."""
        try:
            # Get open positions
            if hasattr(self.trading_system, 'get_open_positions'):
                positions = self.trading_system.get_open_positions()
            elif hasattr(self.trading_system, 'db'):
                positions = self.trading_system.db.get_open_trades().to_dict('records')
            else:
                positions = []
                
            # Process positions data
            positions_data = self._process_positions_data(positions)
            
            # Get portfolio history
            if hasattr(self.trading_system, 'get_portfolio_history'):
                portfolio_history = self.trading_system.get_portfolio_history(days=30)
            elif hasattr(self.trading_system, 'db'):
                portfolio_history = self.trading_system.db.get_portfolio_history(days=30)
            else:
                # Create sample data if no database available
                portfolio_history = self._create_sample_portfolio_history()
                
            # Calculate portfolio metrics
            portfolio_data = self._calculate_portfolio_metrics(positions_data, portfolio_history)
            
            # Update UI from main thread
            from PyQt5.QtCore import QMetaObject, Q_ARG, Qt
            QMetaObject.invokeMethod(
                self, 
                "_update_ui", 
                Qt.QueuedConnection,
                Q_ARG(list, positions_data),
                Q_ARG(object, portfolio_history),
                Q_ARG(dict, portfolio_data)
            )
            
        except Exception as e:
            logging.error(f"Error fetching portfolio data: {e}")
            # Do not propagate exception to allow timer to continue
    
    def _fetch_data_sync(self):
        """Fetch portfolio data synchronously (fallback method)."""
        try:
            # Get open positions
            if hasattr(self.trading_system, 'get_open_positions'):
                positions = self.trading_system.get_open_positions()
            elif hasattr(self.trading_system, 'db'):
                positions = self.trading_system.db.get_open_trades().to_dict('records')
            else:
                positions = []
                
            # Process positions data
            positions_data = self._process_positions_data(positions)
            
            # Get portfolio history
            if hasattr(self.trading_system, 'get_portfolio_history'):
                portfolio_history = self.trading_system.get_portfolio_history(days=30)
            elif hasattr(self.trading_system, 'db'):
                portfolio_history = self.trading_system.db.get_portfolio_history(days=30)
            else:
                # Create sample data if no database available
                portfolio_history = self._create_sample_portfolio_history()
                
            # Calculate portfolio metrics
            portfolio_data = self._calculate_portfolio_metrics(positions_data, portfolio_history)
            
            # Update UI directly
            self._update_ui(positions_data, portfolio_history, portfolio_data)
            
        except Exception as e:
            logging.error(f"Error fetching portfolio data synchronously: {e}")
            # Show error in the UI
            self.update_label.setText(f"Error updating data: {str(e)}")
    
    @pyqtSlot(list, object, dict)
    def _update_ui(self, positions_data, portfolio_history, portfolio_data):
        """Update UI with portfolio data."""
        # Store data
        self.positions_data = positions_data
        self.portfolio_history = portfolio_history
        
        # Update positions table
        self.positions_table.update_positions(positions_data)
        
        # Update portfolio metrics
        self.metrics_widget.update_metrics(portfolio_data)
        
        # Update asset allocation chart
        self.allocation_chart.update_allocation(portfolio_data.get('allocation', {}))
        
        # Update portfolio value chart
        self.portfolio_chart.update_chart(portfolio_history)
    
    @safe_execute(ErrorCategory.DATA_PROCESSING, [])
    def _process_positions_data(self, positions):
        """
        Process positions data to add additional metrics.
        
        Args:
            positions: List of position data dictionaries
            
        Returns:
            List of enriched position data dictionaries
        """
        processed_positions = []
        
        for position in positions:
            # Skip invalid positions
            if not position.get('symbol') or not position.get('entry_price'):
                continue
                
            # Create a copy to avoid modifying original
            pos = position.copy()
            
            # Ensure we have current price
            if not pos.get('current_price'):
                # Try to get current price from market data
                if hasattr(self.trading_system, 'get_market_data'):
                    market_data = self.trading_system.get_market_data(pos['symbol'], limit=1)
                    if not market_data.empty:
                        pos['current_price'] = market_data['close'].iloc[-1]
                # Fallback to entry price if current price not available
                if not pos.get('current_price'):
                    pos['current_price'] = pos['entry_price']
            
            # Calculate P&L
            entry_price = pos.get('entry_price', 0)
            current_price = pos.get('current_price', 0)
            position_size = pos.get('position_size', 0)
            side = pos.get('side', '').lower()
            
            if side == 'buy':
                pos['pnl'] = (current_price - entry_price) * position_size
                if entry_price != 0:
                    pos['pnl_percentage'] = (current_price / entry_price - 1) * 100
                else:
                    pos['pnl_percentage'] = 0
            elif side == 'sell':
                pos['pnl'] = (entry_price - current_price) * position_size
                if entry_price != 0:
                    pos['pnl_percentage'] = (1 - current_price / entry_price) * 100
                else:
                    pos['pnl_percentage'] = 0
            else:
                pos['pnl'] = 0
                pos['pnl_percentage'] = 0
                
            processed_positions.append(pos)
            
        return processed_positions
    
    @safe_execute(ErrorCategory.DATA_PROCESSING, {})
    def _calculate_portfolio_metrics(self, positions, portfolio_history):
        """
        Calculate portfolio metrics from positions and history.
        
        Args:
            positions: List of position dictionaries
            portfolio_history: DataFrame of portfolio history
            
        Returns:
            Dictionary of portfolio metrics
        """
        metrics = {}
        
        # Calculate total position value
        total_value = 0
        for position in positions:
            position_value = position.get('position_size', 0) * position.get('current_price', 0)
            total_value += position_value
            
        # Get cash value from portfolio history or default
        if not portfolio_history.empty and 'cash_value' in portfolio_history.columns:
            cash_value = portfolio_history['cash_value'].iloc[-1]
        else:
            cash_value = 10000  # Default cash value
            
        # Set total value including cash
        metrics['total_value'] = total_value + cash_value
        metrics['cash_value'] = cash_value
        metrics['crypto_value'] = total_value
        metrics['open_positions'] = len(positions)
        
        # Get previous value for comparison
        if not portfolio_history.empty and len(portfolio_history) > 1:
            metrics['prev_total_value'] = portfolio_history['total_value'].iloc[-2]
        else:
            metrics['prev_total_value'] = metrics['total_value']
            
        # Calculate buying power (assuming 3x leverage for example)
        leverage = 3.0
        metrics['buying_power'] = cash_value * leverage
        
        # Calculate margin used
        max_leverage_value = cash_value * leverage
        if max_leverage_value > 0:
            metrics['margin_used'] = (total_value / max_leverage_value) * 100
        else:
            metrics['margin_used'] = 0
            
        # Calculate portfolio beta (placeholder)
        metrics['portfolio_beta'] = 1.2
        
        # Calculate P&L for different time periods
        if not portfolio_history.empty and 'total_value' in portfolio_history.columns:
            # Sort by time if available
            if 'time' in portfolio_history.columns:
                portfolio_history = portfolio_history.sort_values('time')
                
            # Get current value
            current_value = portfolio_history['total_value'].iloc[-1]
            
            # Daily P&L (last 24h)
            if len(portfolio_history) > 1:
                prev_value = portfolio_history['total_value'].iloc[-2]
                metrics['daily_pnl'] = current_value - prev_value
                metrics['daily_pnl_pct'] = (metrics['daily_pnl'] / prev_value) * 100 if prev_value > 0 else 0
            else:
                metrics['daily_pnl'] = 0
                metrics['daily_pnl_pct'] = 0
                
            # Weekly P&L (last 7 days)
            if len(portfolio_history) > 7:
                prev_value = portfolio_history['total_value'].iloc[-8]
                metrics['weekly_pnl'] = current_value - prev_value
                metrics['weekly_pnl_pct'] = (metrics['weekly_pnl'] / prev_value) * 100 if prev_value > 0 else 0
            else:
                metrics['weekly_pnl'] = metrics['daily_pnl']
                metrics['weekly_pnl_pct'] = metrics['daily_pnl_pct']
                
            # Monthly P&L (last 30 days)
            if len(portfolio_history) > 30:
                prev_value = portfolio_history['total_value'].iloc[-31]
                metrics['monthly_pnl'] = current_value - prev_value
                metrics['monthly_pnl_pct'] = (metrics['monthly_pnl'] / prev_value) * 100 if prev_value > 0 else 0
            else:
                metrics['monthly_pnl'] = metrics['weekly_pnl']
                metrics['monthly_pnl_pct'] = metrics['weekly_pnl_pct']
        else:
            # Default values if no history
            metrics['daily_pnl'] = 0
            metrics['daily_pnl_pct'] = 0
            metrics['weekly_pnl'] = 0
            metrics['weekly_pnl_pct'] = 0
            metrics['monthly_pnl'] = 0
            metrics['monthly_pnl_pct'] = 0
            
        # Calculate asset allocation
        allocation = {}
        
        # Add each position to allocation
        for position in positions:
            symbol = position.get('symbol', '')
            if not symbol:
                continue
                
            # Get base asset (e.g., BTC from BTC/USDT)
            base_asset = symbol.split('/')[0] if '/' in symbol else symbol
            
            position_value = position.get('position_size', 0) * position.get('current_price', 0)
            if base_asset in allocation:
                allocation[base_asset] += position_value
            else:
                allocation[base_asset] = position_value
                
        # Add cash to allocation
        allocation['CASH'] = cash_value
        
        # Convert to percentages
        total_portfolio = sum(allocation.values())
        if total_portfolio > 0:
            for asset in allocation:
                allocation[asset] = allocation[asset] / total_portfolio
                
        metrics['allocation'] = allocation
        
        return metrics
    
    def _create_sample_portfolio_history(self):
        """Create sample portfolio history for demonstration."""
        # Create sample data for last 30 days
        dates = pd.date_range(end=datetime.now(), periods=30, freq='D')
        
        # Generate random-ish portfolio values (starting at 10k with some growth and volatility)
        base_value = 10000
        noise = np.random.normal(0, 200, 30)  # Random noise
        trend = np.linspace(0, 1000, 30)  # Upward trend
        values = base_value + trend + noise.cumsum()  # Cumulative noise for random walk
        
        # Create DataFrame
        df = pd.DataFrame({
            'time': dates,
            'total_value': values,
            'cash_value': np.linspace(base_value, base_value / 2, 30),  # Decreasing cash as we invest
            'crypto_value': values - np.linspace(base_value, base_value / 2, 30)  # Increasing crypto holdings
        })
        
        return df
    
    def _open_new_order_dialog(self):
        """Open dialog for creating a new order."""
        # TODO: Implement new order dialog
        # For now, just emit a sample order for demonstration
        QMessageBox.information(
            self,
            "New Order",
            "New order dialog would appear here.\n"
            "This will integrate with the order panel component."
        )
    
    def _confirm_close_all(self):
        """Confirm and close all positions."""
        if not self.positions_data:
            QMessageBox.information(self, "No Positions", "There are no open positions to close.")
            return
            
        reply = QMessageBox.question(
            self, 
            "Close All Positions", 
            f"Are you sure you want to close all {len(self.positions_data)} open positions?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )
        
        if reply == QMessageBox.Yes:
            for position in self.positions_data:
                symbol = position.get('symbol', '')
                price = position.get('current_price', 0)
                if symbol and price > 0:
                    self.close_position_requested.emit(symbol, price, "Close all")
    
    def closeEvent(self, event):
        """Clean up resources when widget is closed."""
        # Stop the update timer
        if self.update_timer:
            self.update_timer.stop()
        super().closeEvent(event)
