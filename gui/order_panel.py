# gui/order_panel.py

import os
import logging
import json
from typing import Dict, List, Optional, Tuple, Union, Any
from datetime import datetime
from decimal import Decimal, ROUND_DOWN

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGridLayout, QLabel, QPushButton, QLineEdit,
    QComboBox, QDoubleSpinBox, QCheckBox, QTabWidget, QGroupBox, QRadioButton, 
    QSlider, QFrame, QSplitter, QTableWidget, QTableWidgetItem, QHeaderView,
    QStyledItemDelegate, QStyleOptionViewItem, QToolButton, QMenu, QAction,
    QMessageBox, QInputDialog, QFormLayout, QSpinBox, QScrollArea, QSizePolicy
)
from PyQt5.QtCore import Qt, pyqtSignal, QSize, QTimer, QEvent, QObject, QRect, QModelIndex
from PyQt5.QtGui import QColor, QPainter, QFont, QIcon, QPixmap, QPalette, QBrush

# Import system components with error handling
try:
    from core.error_handling import safe_execute, ErrorCategory
    HAVE_ERROR_HANDLING = True
except ImportError:
    HAVE_ERROR_HANDLING = False
    logging.warning("Error handling module not available in OrderPanel. Using basic error handling.")

# Define order types and time-in-force options
ORDER_TYPES = ["Market", "Limit", "Stop", "Stop Limit", "Trailing Stop", "OCO (One Cancels Other)"]
TIME_IN_FORCE = ["GTC (Good Till Canceled)", "IOC (Immediate or Cancel)", "FOK (Fill or Kill)", "Day"]
LEVERAGE_OPTIONS = [1, 2, 3, 5, 10, 20, 50, 100]

class PriceLineEdit(QLineEdit):
    """Enhanced QLineEdit for price input with ticker buttons and validation."""
    
    valueChanged = pyqtSignal(float)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.initUI()
        
    def initUI(self):
        """Initialize UI components."""
        self.setFixedHeight(30)
        
        # Add up/down buttons on the right side
        self.up_down_buttons = QWidget(self)
        self.up_down_buttons.setFixedWidth(20)
        layout = QVBoxLayout(self.up_down_buttons)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        
        self.up_button = QToolButton()
        self.up_button.setArrowType(Qt.UpArrow)
        self.up_button.setFixedSize(20, 15)
        self.up_button.clicked.connect(self.increment)
        
        self.down_button = QToolButton()
        self.down_button.setArrowType(Qt.DownArrow)
        self.down_button.setFixedSize(20, 15)
        self.down_button.clicked.connect(self.decrement)
        
        layout.addWidget(self.up_button)
        layout.addWidget(self.down_button)
        
        # Set validator
        self.setValidator(QDoubleValidator())
        
        self.textChanged.connect(self.onTextChanged)
        
    def resizeEvent(self, event):
        """Handle resize events to position the buttons."""
        super().resizeEvent(event)
        self.up_down_buttons.setGeometry(
            self.width() - 20, 0, 20, self.height()
        )
        
    def increment(self):
        """Increment the price value."""
        try:
            value = float(self.text() or "0")
            # Get appropriate tick size based on price
            tick_size = self.getTickSize(value)
            new_value = value + tick_size
            self.setText(f"{new_value:.8f}".rstrip('0').rstrip('.'))
            self.valueChanged.emit(new_value)
        except ValueError:
            self.setText("0")
            
    def decrement(self):
        """Decrement the price value."""
        try:
            value = float(self.text() or "0")
            # Get appropriate tick size based on price
            tick_size = self.getTickSize(value)
            new_value = max(0, value - tick_size)
            self.setText(f"{new_value:.8f}".rstrip('0').rstrip('.'))
            self.valueChanged.emit(new_value)
        except ValueError:
            self.setText("0")
            
    def getTickSize(self, price):
        """
        Get appropriate tick size based on price.
        Implements dynamic tick sizing similar to major exchanges.
        """
        if price >= 10000:
            return 1.0
        elif price >= 1000:
            return 0.1
        elif price >= 100:
            return 0.01
        elif price >= 10:
            return 0.001
        elif price >= 1:
            return 0.0001
        elif price >= 0.1:
            return 0.00001
        elif price >= 0.01:
            return 0.000001
        elif price >= 0.001:
            return 0.0000001
        else:
            return 0.00000001
            
    def onTextChanged(self, text):
        """Handle text changes."""
        if text:
            try:
                value = float(text)
                self.valueChanged.emit(value)
            except ValueError:
                pass

class OrderSizeWidget(QWidget):
    """
    Custom widget for order size input with multiple sizing options.
    Integrates with risk management system for position sizing suggestions.
    """
    
    sizeChanged = pyqtSignal(float)
    
    def __init__(self, trading_system, parent=None):
        super().__init__(parent)
        self.trading_system = trading_system
        self.risk_percentage = 0.01  # 1% risk per trade default
        self.account_balance = 10000  # Default value until updated
        self.current_price = 0
        self.initUI()
        
    def initUI(self):
        """Initialize UI components."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        
        # Size input row
        size_layout = QHBoxLayout()
        size_layout.addWidget(QLabel("Size:"))
        
        self.size_input = QDoubleSpinBox()
        self.size_input.setDecimals(8)
        self.size_input.setRange(0, 1000000)
        self.size_input.setSingleStep(0.01)
        self.size_input.valueChanged.connect(self.onSizeInputChanged)
        size_layout.addWidget(self.size_input, 1)
        
        # Currency/USD toggle
        self.is_base_currency = True
        self.currency_toggle = QPushButton("BTC")  # Will toggle between BTC and USD
        self.currency_toggle.setFixedWidth(50)
        self.currency_toggle.clicked.connect(self.toggleCurrency)
        size_layout.addWidget(self.currency_toggle)
        
        layout.addLayout(size_layout)
        
        # Quick size buttons
        quick_size_layout = QHBoxLayout()
        
        # Risk-based sizing buttons
        for pct in [1, 2, 5, 10]:
            btn = QPushButton(f"{pct}%")
            btn.setFixedHeight(25)
            btn.setProperty("percentage", pct)
            btn.clicked.connect(self.onRiskButtonClicked)
            quick_size_layout.addWidget(btn)
            
        layout.addLayout(quick_size_layout)
        
        # Advanced sizing options
        advanced_sizing_group = QGroupBox("Advanced Sizing")
        advanced_layout = QFormLayout(advanced_sizing_group)
        
        # Risk percentage slider
        self.risk_slider = QSlider(Qt.Horizontal)
        self.risk_slider.setRange(1, 100)  # 0.1% to 10% risk
        self.risk_slider.setValue(10)  # Default 1% risk
        self.risk_slider.setFixedWidth(150)
        
        risk_layout = QHBoxLayout()
        risk_layout.addWidget(self.risk_slider)
        self.risk_label = QLabel("1.0%")
        risk_layout.addWidget(self.risk_label)
        
        advanced_layout.addRow("Risk:", risk_layout)
        
        # Update risk label when slider changes
        self.risk_slider.valueChanged.connect(self.updateRiskLabel)
        
        # Button to apply advanced sizing
        apply_button = QPushButton("Apply Risk-Based Size")
        apply_button.clicked.connect(self.applyRiskBasedSize)
        advanced_layout.addRow("", apply_button)
        
        layout.addWidget(advanced_sizing_group)
        
        # Add stretch to bottom
        layout.addStretch(1)
        
    def toggleCurrency(self):
        """Toggle between base currency and quote currency (USD)."""
        self.is_base_currency = not self.is_base_currency
        
        if self.is_base_currency:
            self.currency_toggle.setText("BTC")
            # Convert USD value to BTC
            if self.current_price > 0:
                current_value = self.size_input.value()
                self.size_input.setValue(current_value / self.current_price)
        else:
            self.currency_toggle.setText("USD")
            # Convert BTC value to USD
            current_value = self.size_input.value()
            self.size_input.setValue(current_value * self.current_price)
        
    def onSizeInputChanged(self, value):
        """Handle size input changes."""
        self.sizeChanged.emit(value)
        
    def onRiskButtonClicked(self):
        """Handle risk percentage button clicks."""
        btn = self.sender()
        pct = btn.property("percentage")
        self.risk_percentage = pct / 100.0
        
        # Update risk slider
        self.risk_slider.setValue(int(pct * 10))
        
        # Apply the risk-based size
        self.applyRiskBasedSize()
        
    def updateRiskLabel(self, value):
        """Update risk label when slider changes."""
        self.risk_percentage = value / 1000.0  # Convert to percentage (0.1% to 10%)
        self.risk_label.setText(f"{self.risk_percentage * 100:.1f}%")
        
    def applyRiskBasedSize(self):
        """Apply risk-based position sizing."""
        try:
            # Get account balance
            balance = self.getAccountBalance()
            
            # Calculate risk amount
            risk_amount = balance * self.risk_percentage
            
            # Get stop loss distance (default 2%)
            stop_loss_pct = 0.02
            
            # Calculate position size
            if self.current_price > 0 and stop_loss_pct > 0:
                # Position size = Risk amount / (Entry price * Stop loss percentage)
                position_size = risk_amount / (self.current_price * stop_loss_pct)
                
                # Set position size in base currency
                if not self.is_base_currency:
                    position_size = position_size / self.current_price
                    
                # Round to 8 decimal places
                position_size = round(position_size, 8)
                
                # Update size input
                self.size_input.setValue(position_size)
            
        except Exception as e:
            logging.error(f"Error applying risk-based size: {e}")
        
    def getAccountBalance(self):
        """Get account balance."""
        if hasattr(self.trading_system, 'get_portfolio_value'):
            try:
                portfolio = self.trading_system.get_portfolio_value()
                if portfolio and 'total_value' in portfolio:
                    self.account_balance = portfolio['total_value']
            except Exception as e:
                logging.error(f"Error getting portfolio value: {e}")
                
        return self.account_balance
        
    def setSymbol(self, symbol):
        """Update symbol in currency toggle button."""
        if symbol and '/' in symbol:
            base_currency = symbol.split('/')[0]
            self.currency_toggle.setText(base_currency)
        
    def setCurrentPrice(self, price):
        """Set current market price."""
        self.current_price = price
        
    def getCurrentSize(self):
        """Get current size in base currency."""
        size = self.size_input.value()
        
        # Convert to base currency if currently in quote currency
        if not self.is_base_currency and self.current_price > 0:
            size = size / self.current_price
            
        return size

class OrderTypeSelector(QWidget):
    """
    Custom widget for selecting order type with dynamic form changes.
    Supports all major order types with appropriate parameters.
    """
    
    orderTypeChanged = pyqtSignal(str)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.initUI()
        
    def initUI(self):
        """Initialize UI components."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        
        # Order type combo box
        type_layout = QHBoxLayout()
        type_layout.addWidget(QLabel("Order Type:"))
        
        self.type_combo = QComboBox()
        self.type_combo.addItems(ORDER_TYPES)
        self.type_combo.currentTextChanged.connect(self.onOrderTypeChanged)
        type_layout.addWidget(self.type_combo, 1)
        
        layout.addLayout(type_layout)
        
        # Time in force
        tif_layout = QHBoxLayout()
        tif_layout.addWidget(QLabel("Time in Force:"))
        
        self.tif_combo = QComboBox()
        self.tif_combo.addItems(TIME_IN_FORCE)
        tif_layout.addWidget(self.tif_combo, 1)
        
        layout.addLayout(tif_layout)
        
        # Reduce only checkbox
        self.reduce_only = QCheckBox("Reduce Only")
        self.reduce_only.setToolTip("Order will only reduce your position, not increase it")
        layout.addWidget(self.reduce_only)
        
        # Post only checkbox
        self.post_only = QCheckBox("Post Only")
        self.post_only.setToolTip("Order will only be posted as a maker order")
        layout.addWidget(self.post_only)
        
    def onOrderTypeChanged(self, order_type):
        """Handle order type changes."""
        self.orderTypeChanged.emit(order_type)
        
        # Enable/disable time in force based on order type
        self.tif_combo.setEnabled(order_type != "Market")
        
        # Enable/disable post only based on order type
        self.post_only.setEnabled(order_type in ["Limit", "Stop Limit"])
        
    def getOrderType(self):
        """Get selected order type."""
        return self.type_combo.currentText()
        
    def getTimeInForce(self):
        """Get selected time in force."""
        return self.tif_combo.currentText().split(" (")[0]
        
    def isReduceOnly(self):
        """Check if reduce only is enabled."""
        return self.reduce_only.isChecked()
        
    def isPostOnly(self):
        """Check if post only is enabled."""
        return self.post_only.isChecked()

class MarketDepthWidget(QWidget):
    """
    Widget for displaying market depth (order book) information.
    Includes visualization of buy/sell pressure and price levels.
    """
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.asks = []  # List of [price, size] for asks
        self.bids = []  # List of [price, size] for bids
        self.max_quantity = 1
        self.mid_price = 0
        self.initUI()
        
    def initUI(self):
        """Initialize UI components."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        
        # Title
        title_layout = QHBoxLayout()
        title_layout.addWidget(QLabel("<b>Market Depth</b>"))
        
        # Auto-update checkbox
        self.auto_update = QCheckBox("Auto-update")
        self.auto_update.setChecked(True)
        title_layout.addWidget(self.auto_update, 0, Qt.AlignRight)
        
        layout.addLayout(title_layout)
        
        # Order book table
        self.depth_table = QTableWidget(10, 3)
        self.depth_table.setHorizontalHeaderLabels(["Bid Size", "Price", "Ask Size"])
        self.depth_table.verticalHeader().setVisible(False)
        self.depth_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.depth_table.setEditTriggers(QTableWidget.NoEditTriggers)
        
        layout.addWidget(self.depth_table)
        
        # Depth visualization
        self.depth_view = OrderBookVisualization()
        layout.addWidget(self.depth_view, 1)
        
        # Refresh button
        refresh_btn = QPushButton("Refresh")
        refresh_btn.clicked.connect(self.refreshDepth)
        layout.addWidget(refresh_btn)
        
        # Initialize auto-update timer
        self.update_timer = QTimer(self)
        self.update_timer.timeout.connect(self.refreshDepth)
        self.update_timer.start(2000)  # Refresh every 2 seconds
        
    def setOrderBook(self, bids, asks):
        """Set order book data."""
        self.bids = bids
        self.asks = asks
        
        # Find mid price
        if len(bids) > 0 and len(asks) > 0:
            self.mid_price = (bids[0][0] + asks[0][0]) / 2
            
        # Calculate max quantity for visualization scaling
        all_quantities = [qty for _, qty in bids] + [qty for _, qty in asks]
        if all_quantities:
            self.max_quantity = max(all_quantities)
            
        # Update table
        self.updateTable()
        
        # Update visualization
        self.depth_view.setData(bids, asks, self.max_quantity, self.mid_price)
        
    def updateTable(self):
        """Update the order book table."""
        # Sort bids (descending) and asks (ascending)
        bids_sorted = sorted(self.bids, key=lambda x: x[0], reverse=True)
        asks_sorted = sorted(self.asks, key=lambda x: x[0])
        
        # Get up to 10 levels per side
        bids_display = bids_sorted[:10]
        asks_display = asks_sorted[:10]
        
        # Determine how many rows we need
        row_count = max(len(bids_display), len(asks_display))
        self.depth_table.setRowCount(row_count)
        
        # Fill table
        for i in range(row_count):
            # Add bid
            if i < len(bids_display):
                price, quantity = bids_display[i]
                
                # Bid size
                size_item = QTableWidgetItem(f"{quantity:.8f}")
                size_item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
                size_item.setForeground(QColor(0, 170, 0))  # Green
                self.depth_table.setItem(i, 0, size_item)
                
                # Bid price
                price_item = QTableWidgetItem(f"{price:.8f}")
                price_item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
                price_item.setForeground(QColor(0, 170, 0))  # Green
                self.depth_table.setItem(i, 1, price_item)
            else:
                self.depth_table.setItem(i, 0, QTableWidgetItem(""))
                self.depth_table.setItem(i, 1, QTableWidgetItem(""))
                
            # Add ask
            if i < len(asks_display):
                price, quantity = asks_display[i]
                
                # Ask price (use same cell as bid price if it's the first row)
                if i >= len(bids_display):
                    price_item = QTableWidgetItem(f"{price:.8f}")
                    price_item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
                    price_item.setForeground(QColor(170, 0, 0))  # Red
                    self.depth_table.setItem(i, 1, price_item)
                
                # Ask size
                size_item = QTableWidgetItem(f"{quantity:.8f}")
                size_item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
                size_item.setForeground(QColor(170, 0, 0))  # Red
                self.depth_table.setItem(i, 2, size_item)
            else:
                self.depth_table.setItem(i, 2, QTableWidgetItem(""))
        
    def refreshDepth(self):
        """Refresh market depth data."""
        if not self.auto_update.isChecked():
            return
            
        # This would actually fetch new data from the exchange
        # For now, just emit an update signal that the parent can connect to
        self.requestDepthUpdate()
        
    def requestDepthUpdate(self):
        """Request a depth update from parent."""
        # This would be connected to a signal in the parent
        pass

class OrderBookVisualization(QWidget):
    """Custom widget for visualizing the order book with buy/sell pressure."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.bids = []
        self.asks = []
        self.max_quantity = 1
        self.mid_price = 0
        self.setMinimumHeight(100)
        
    def setData(self, bids, asks, max_quantity, mid_price):
        """Set visualization data."""
        self.bids = bids
        self.asks = asks
        self.max_quantity = max_quantity
        self.mid_price = mid_price
        self.update()
        
    def paintEvent(self, event):
        """Paint the order book visualization."""
        if not self.bids and not self.asks:
            return
            
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        
        # Draw background
        painter.fillRect(self.rect(), Qt.white)
        
        # Calculate ranges
        width = self.width()
        height = self.height()
        middle_x = width // 2
        
        # Sort bids and asks
        bids_sorted = sorted(self.bids, key=lambda x: x[0], reverse=True)
        asks_sorted = sorted(self.asks, key=lambda x: x[0])
        
        # Find min and max prices within 5% of mid price
        price_range = self.mid_price * 0.05
        min_price = max(0, self.mid_price - price_range)
        max_price = self.mid_price + price_range
        
        # Filter bids and asks within price range
        bids_in_range = [b for b in bids_sorted if b[0] >= min_price]
        asks_in_range = [a for a in asks_sorted if a[0] <= max_price]
        
        # Draw bids (left side, green)
        for price, quantity in bids_in_range:
            # Calculate bar width based on quantity (normalized)
            bar_width = (quantity / self.max_quantity) * middle_x
            
            # Calculate Y position (price)
            y_pos = height - ((price - min_price) / (max_price - min_price) * height)
            
            # Draw bar
            painter.fillRect(
                middle_x - bar_width, y_pos - 2, 
                bar_width, 4, 
                QColor(0, 170, 0, 200)  # Semi-transparent green
            )
            
        # Draw asks (right side, red)
        for price, quantity in asks_in_range:
            # Calculate bar width based on quantity (normalized)
            bar_width = (quantity / self.max_quantity) * middle_x
            
            # Calculate Y position (price)
            y_pos = height - ((price - min_price) / (max_price - min_price) * height)
            
            # Draw bar
            painter.fillRect(
                middle_x, y_pos - 2, 
                bar_width, 4, 
                QColor(170, 0, 0, 200)  # Semi-transparent red
            )
            
        # Draw mid price line
        mid_y = height - ((self.mid_price - min_price) / (max_price - min_price) * height)
        painter.setPen(Qt.gray)
        painter.drawLine(0, mid_y, width, mid_y)

class OrderHistoryWidget(QWidget):
    """
    Widget for displaying order history with filtering and sorting capabilities.
    """
    
    orderSelected = pyqtSignal(dict)
    
    def __init__(self, trading_system, parent=None):
        super().__init__(parent)
        self.trading_system = trading_system
        self.orders = []
        self.initUI()
        
    def initUI(self):
        """Initialize UI components."""
        layout = QVBoxLayout(self)
        
        # Title and filter
        header_layout = QHBoxLayout()
        header_layout.addWidget(QLabel("<b>Order History</b>"))
        
        # Filter combo
        self.filter_combo = QComboBox()
        self.filter_combo.addItems(["All Orders", "Open Orders", "Filled Orders", "Canceled Orders"])
        self.filter_combo.currentTextChanged.connect(self.filterOrders)
        header_layout.addWidget(self.filter_combo, 1)
        
        # Refresh button
        refresh_btn = QPushButton("Refresh")
        refresh_btn.clicked.connect(self.refreshOrders)
        header_layout.addWidget(refresh_btn)
        
        layout.addLayout(header_layout)
        
        # Orders table
        self.orders_table = QTableWidget()
        self.orders_table.setColumnCount(7)
        self.orders_table.setHorizontalHeaderLabels(["Time", "Symbol", "Type", "Side", "Price", "Amount", "Status"])
        self.orders_table.verticalHeader().setVisible(False)
        self.orders_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        self.orders_table.horizontalHeader().setSectionResizeMode(6, QHeaderView.Stretch)
        self.orders_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.orders_table.setSelectionBehavior(QTableWidget.SelectRows)
        self.orders_table.setSelectionMode(QTableWidget.SingleSelection)
        self.orders_table.setAlternatingRowColors(True)
        self.orders_table.itemDoubleClicked.connect(self.onOrderDoubleClicked)
        
        layout.addWidget(self.orders_table)
        
        # Action buttons
        actions_layout = QHBoxLayout()
        
        # Cancel button
        self.cancel_btn = QPushButton("Cancel Order")
        self.cancel_btn.clicked.connect(self.cancelSelectedOrder)
        actions_layout.addWidget(self.cancel_btn)
        
        # Modify button
        self.modify_btn = QPushButton("Modify Order")
        self.modify_btn.clicked.connect(self.modifySelectedOrder)
        actions_layout.addWidget(self.modify_btn)
        
        layout.addLayout(actions_layout)
        
        # Disable buttons initially
        self.cancel_btn.setEnabled(False)
        self.modify_btn.setEnabled(False)
        
        # Enable buttons when an order is selected
        self.orders_table.itemSelectionChanged.connect(self.updateButtonState)
        
        # Auto-refresh timer
        self.refresh_timer = QTimer(self)
        self.refresh_timer.timeout.connect(self.refreshOrders)
        self.refresh_timer.start(10000)  # Refresh every 10 seconds
        
    def updateButtonState(self):
        """Update button enable state based on selection."""
        selected_row = self.getSelectedRow()
        has_selection = selected_row >= 0
        
        self.cancel_btn.setEnabled(has_selection)
        self.modify_btn.setEnabled(has_selection)
        
        # Further checks for modifying/canceling
        if has_selection:
            status = self.orders_table.item(selected_row, 6).text()
            self.cancel_btn.setEnabled(status in ["Open", "Partially Filled"])
            self.modify_btn.setEnabled(status in ["Open"])
        
    def getSelectedRow(self):
        """Get the selected row index."""
        selected_indexes = self.orders_table.selectedIndexes()
        if selected_indexes:
            return selected_indexes[0].row()
        return -1
        
    def refreshOrders(self):
        """Refresh orders from trading system."""
        try:
            if hasattr(self.trading_system, 'get_orders'):
                orders = self.trading_system.get_orders()
                if orders:
                    self.orders = orders
                    self.filterOrders(self.filter_combo.currentText())
        except Exception as e:
            logging.error(f"Error refreshing orders: {e}")
        
    def filterOrders(self, filter_text):
        """Filter orders based on selection."""
        self.orders_table.clearContents()
        
        filtered_orders = []
        
        if filter_text == "All Orders":
            filtered_orders = self.orders
        elif filter_text == "Open Orders":
            filtered_orders = [o for o in self.orders if o.get('status') in ['open', 'partially_filled']]
        elif filter_text == "Filled Orders":
            filtered_orders = [o for o in self.orders if o.get('status') == 'filled']
        elif filter_text == "Canceled Orders":
            filtered_orders = [o for o in self.orders if o.get('status') == 'canceled']
            
        # Update table
        self.orders_table.setRowCount(len(filtered_orders))
        
        for i, order in enumerate(filtered_orders):
            # Convert timestamp to datetime
            timestamp = order.get('timestamp', 0)
            if isinstance(timestamp, (int, float)):
                dt = datetime.fromtimestamp(timestamp / 1000)
                time_str = dt.strftime("%H:%M:%S")
            else:
                time_str = str(timestamp)
                
            # Format status
            status = order.get('status', '').capitalize()
            if status == 'Partially_filled':
                status = 'Partially Filled'
                
            # Add data to table
            self.orders_table.setItem(i, 0, QTableWidgetItem(time_str))
            self.orders_table.setItem(i, 1, QTableWidgetItem(order.get('symbol', '')))
            self.orders_table.setItem(i, 2, QTableWidgetItem(order.get('type', '')))
            
            # Set color for side
            side_item = QTableWidgetItem(order.get('side', '').capitalize())
            if order.get('side') == 'buy':
                side_item.setForeground(QColor(0, 170, 0))
            else:
                side_item.setForeground(QColor(170, 0, 0))
            self.orders_table.setItem(i, 3, side_item)
            
            # Format price
            price = order.get('price', 0)
            if price == 0 and order.get('type') == 'market':
                price_str = 'Market'
            else:
                price_str = f"{price:.8f}".rstrip('0').rstrip('.')
            self.orders_table.setItem(i, 4, QTableWidgetItem(price_str))
            
            # Format amount
            amount = order.get('amount', 0)
            self.orders_table.setItem(i, 5, QTableWidgetItem(f"{amount:.8f}".rstrip('0').rstrip('.')))
            
            # Set status with color
            status_item = QTableWidgetItem(status)
            if status == 'Open':
                status_item.setForeground(QColor(0, 0, 170))
            elif status == 'Filled':
                status_item.setForeground(QColor(0, 170, 0))
            elif status == 'Canceled':
                status_item.setForeground(QColor(170, 0, 0))
            elif status == 'Partially Filled':
                status_item.setForeground(QColor(170, 85, 0))
                
            self.orders_table.setItem(i, 6, status_item)
            
            # Store order in item data
            self.orders_table.item(i, 0).setData(Qt.UserRole, order)
        
    def onOrderDoubleClicked(self, item):
        """Handle order double click."""
        row = item.row()
        order_item = self.orders_table.item(row, 0)
        if order_item:
            order = order_item.data(Qt.UserRole)
            if order:
                self.orderSelected.emit(order)
        
    def cancelSelectedOrder(self):
        """Cancel the selected order."""
        selected_row = self.getSelectedRow()
        if selected_row < 0:
            return
            
        order_item = self.orders_table.item(selected_row, 0)
        if not order_item:
            return
            
        order = order_item.data(Qt.UserRole)
        if not order:
            return
            
        # Confirm cancellation
        symbol = order.get('symbol', '')
        side = order.get('side', '').capitalize()
        amount = order.get('amount', 0)
        
        reply = QMessageBox.question(
            self, 
            "Cancel Order", 
            f"Are you sure you want to cancel this {side} order for {amount} {symbol}?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )
        
        if reply != QMessageBox.Yes:
            return
            
        # Cancel order
        try:
            if hasattr(self.trading_system, 'cancel_order'):
                success = self.trading_system.cancel_order(order.get('id'))
                
                if success:
                    QMessageBox.information(self, "Success", "Order canceled successfully")
                    self.refreshOrders()
                else:
                    QMessageBox.warning(self, "Failed", "Failed to cancel order")
                    
        except Exception as e:
            logging.error(f"Error canceling order: {e}")
            QMessageBox.critical(self, "Error", f"Error canceling order: {str(e)}")
        
    def modifySelectedOrder(self):
        """Modify the selected order."""
        selected_row = self.getSelectedRow()
        if selected_row < 0:
            return
            
        order_item = self.orders_table.item(selected_row, 0)
        if not order_item:
            return
            
        order = order_item.data(Qt.UserRole)
        if not order:
            return
            
        # Create modify dialog
        dialog = QDialog(self)
        dialog.setWindowTitle("Modify Order")
        dialog.setMinimumWidth(300)
        
        layout = QVBoxLayout(dialog)
        
        # Order info
        symbol = order.get('symbol', '')
        side = order.get('side', '').capitalize()
        amount = order.get('amount', 0)
        
        layout.addWidget(QLabel(f"<b>{side} {amount} {symbol}</b>"))
        
        # Price input
        form_layout = QFormLayout()
        
        price_input = QDoubleSpinBox()
        price_input.setDecimals(8)
        price_input.setRange(0, 1000000)
        price_input.setValue(order.get('price', 0))
        form_layout.addRow("Price:", price_input)
        
        # Amount input
        amount_input = QDoubleSpinBox()
        amount_input.setDecimals(8)
        amount_input.setRange(0, 1000000)
        amount_input.setValue(amount)
        form_layout.addRow("Amount:", amount_input)
        
        layout.addLayout(form_layout)
        
        # Buttons
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        layout.addWidget(buttons)
        
        # Show dialog
        if dialog.exec_() != QDialog.Accepted:
            return
            
        # Get new values
        new_price = price_input.value()
        new_amount = amount_input.value()
        
        # Modify order
        try:
            if hasattr(self.trading_system, 'modify_order'):
                success = self.trading_system.modify_order(
                    order.get('id'),
                    price=new_price,
                    amount=new_amount
                )
                
                if success:
                    QMessageBox.information(self, "Success", "Order modified successfully")
                    self.refreshOrders()
                else:
                    QMessageBox.warning(self, "Failed", "Failed to modify order")
                    
        except Exception as e:
            logging.error(f"Error modifying order: {e}")
            QMessageBox.critical(self, "Error", f"Error modifying order: {str(e)}")

class LeverageSelector(QWidget):
    """Widget for selecting leverage for margin/futures trading."""
    
    leverageChanged = pyqtSignal(int)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.initUI()
        
    def initUI(self):
        """Initialize UI components."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        
        # Title
        layout.addWidget(QLabel("<b>Leverage</b>"))
        
        # Leverage buttons
        leverage_layout = QHBoxLayout()
        
        for lev in LEVERAGE_OPTIONS:
            btn = QPushButton(f"{lev}x")
            btn.setCheckable(True)
            btn.setFixedHeight(30)
            btn.clicked.connect(lambda checked, l=lev: self.setLeverage(l))
            leverage_layout.addWidget(btn)
            
            if lev == 1:
                btn.setChecked(True)
                self.current_button = btn
                
        layout.addLayout(leverage_layout)
        
        # Slider
        self.leverage_slider = QSlider(Qt.Horizontal)
        self.leverage_slider.setRange(1, 100)
        self.leverage_slider.setValue(1)
        self.leverage_slider.valueChanged.connect(self.onSliderChanged)
        layout.addWidget(self.leverage_slider)
        
        # Current leverage label
        self.leverage_label = QLabel("1x")
        self.leverage_label.setAlignment(Qt.AlignCenter)
        self.leverage_label.setStyleSheet("font-size: 14px; font-weight: bold;")
        layout.addWidget(self.leverage_label)
        
    def setLeverage(self, leverage):
        """Set leverage value and update UI."""
        # Find the button that was clicked
        sender = self.sender()
        
        # Uncheck previous button
        if hasattr(self, 'current_button') and self.current_button != sender:
            self.current_button.setChecked(False)
            
        # Set current button
        self.current_button = sender
        
        # Update slider
        self.leverage_slider.setValue(leverage)
        
        # Update label
        self.leverage_label.setText(f"{leverage}x")
        
        # Emit signal
        self.leverageChanged.emit(leverage)
        
    def onSliderChanged(self, value):
        """Handle slider value change."""
        # Update label
        self.leverage_label.setText(f"{value}x")
        
        # Update button states
        for button in self.findChildren(QPushButton):
            button_leverage = int(button.text().rstrip('x'))
            button.setChecked(button_leverage == value)
            
            if button_leverage == value:
                self.current_button = button
                
        # Emit signal
        self.leverageChanged.emit(value)
        
    def getLeverage(self):
        """Get current leverage value."""
        return self.leverage_slider.value()

class OrderPanel(QWidget):
    """
    Main order panel widget for creating and managing trading orders.
    Integrates all sub-components for a complete trading experience.
    """
    
    orderCreated = pyqtSignal(dict)
    
    def __init__(self, trading_system, parent=None):
        super().__init__(parent)
        self.trading_system = trading_system
        self.current_symbol = ""
        self.current_price = 0
        self.current_balance = 0
        self.initUI()
        self.setupConnections()
        
        # Initialize with default symbol
        self.loadSymbols()
        
    def initUI(self):
        """Initialize UI components."""
        main_layout = QVBoxLayout(self)
        
        # Split view
        splitter = QSplitter(Qt.Vertical)
        
        # Order entry section
        order_entry = QWidget()
        entry_layout = QVBoxLayout(order_entry)
        
        # Symbol selection
        symbol_layout = QHBoxLayout()
        symbol_layout.addWidget(QLabel("Symbol:"))
        
        self.symbol_combo = QComboBox()
        symbol_layout.addWidget(self.symbol_combo, 1)
        
        entry_layout.addLayout(symbol_layout)
        
        # Current price
        price_layout = QHBoxLayout()
        price_layout.addWidget(QLabel("Last Price:"))
        
        self.price_label = QLabel("0.00")
        self.price_label.setStyleSheet("font-weight: bold; color: green;")
        price_layout.addWidget(self.price_label)
        
        price_layout.addWidget(QLabel("24h Change:"))
        
        self.change_label = QLabel("0.00%")
        self.change_label.setStyleSheet("font-weight: bold; color: green;")
        price_layout.addWidget(self.change_label)
        
        # Balance
        price_layout.addWidget(QLabel("Balance:"))
        
        self.balance_label = QLabel("0.00")
        price_layout.addWidget(self.balance_label)
        
        entry_layout.addLayout(price_layout)
        
        # Tabs for different order sections
        self.order_tabs = QTabWidget()
        
        # Spot tab
        spot_tab = self.createSpotTab()
        self.order_tabs.addTab(spot_tab, "Spot")
        
        # Advanced tab (conditional orders)
        advanced_tab = self.createAdvancedTab()
        self.order_tabs.addTab(advanced_tab, "Advanced")
        
        # Futures tab
        futures_tab = self.createFuturesTab()
        self.order_tabs.addTab(futures_tab, "Futures")
        
        entry_layout.addWidget(self.order_tabs)
        
        # Market depth visualization
        self.market_depth = MarketDepthWidget()
        entry_layout.addWidget(self.market_depth)
        
        # Add to splitter
        splitter.addWidget(order_entry)
        
        # Order history section
        self.order_history = OrderHistoryWidget(self.trading_system)
        splitter.addWidget(self.order_history)
        
        # Add splitter to main layout
        main_layout.addWidget(splitter)
        
        # Set initial splitter sizes
        splitter.setSizes([600, 400])
        
        # Set dark mode if system is using dark theme
        self.applyStyleSheet()
        
    def applyStyleSheet(self):
        """Apply appropriate style based on system theme."""
        # Check if system is using dark mode
        if self.isDarkMode():
            self.setStyleSheet("""
                QWidget {
                    background-color: #2D2D2D;
                    color: #E0E0E0;
                }
                QLabel {
                    color: #E0E0E0;
                }
                QPushButton {
                    background-color: #505050;
                    color: #E0E0E0;
                    border: 1px solid #707070;
                    border-radius: 4px;
                    padding: 4px 8px;
                }
                QPushButton:hover {
                    background-color: #606060;
                }
                QPushButton:pressed {
                    background-color: #404040;
                }
                QPushButton:checked {
                    background-color: #3070B0;
                }
                QComboBox, QLineEdit, QSpinBox, QDoubleSpinBox {
                    background-color: #404040;
                    color: #E0E0E0;
                    border: 1px solid #606060;
                    border-radius: 4px;
                    padding: 2px 4px;
                }
                QComboBox::drop-down {
                    border: 0px;
                }
                QTableWidget {
                    background-color: #2D2D2D;
                    color: #E0E0E0;
                    gridline-color: #505050;
                }
                QHeaderView::section {
                    background-color: #3C3C3C;
                    color: #E0E0E0;
                    border: 1px solid #505050;
                }
                QTabWidget::pane {
                    border: 1px solid #505050;
                }
                QTabBar::tab {
                    background-color: #404040;
                    color: #E0E0E0;
                    border: 1px solid #505050;
                    padding: 4px 8px;
                }
                QTabBar::tab:selected {
                    background-color: #505050;
                }
                QGroupBox {
                    border: 1px solid #505050;
                    margin-top: 8px;
                }
                QGroupBox::title {
                    subcontrol-origin: margin;
                    left: 10px;
                    padding: 0 3px;
                }
            """)
            
    def isDarkMode(self):
        """Check if system is using dark mode."""
        # Get application palette
        palette = self.palette()
        bg_color = palette.color(QPalette.Window)
        
        # Dark mode if background is dark
        return bg_color.lightness() < 128
        
    def createSpotTab(self):
        """Create spot trading tab."""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # Buy/Sell tabs
        self.spot_action_tabs = QTabWidget()
        
        # Buy tab
        buy_tab = QWidget()
        buy_layout = QVBoxLayout(buy_tab)
        
        # Price input
        buy_layout.addWidget(QLabel("Price:"))
        self.spot_buy_price = PriceLineEdit()
        buy_layout.addWidget(self.spot_buy_price)
        
        # Order type selector
        self.spot_buy_type = OrderTypeSelector()
        buy_layout.addWidget(self.spot_buy_type)
        
        # Size widget
        self.spot_buy_size = OrderSizeWidget(self.trading_system)
        buy_layout.addWidget(self.spot_buy_size)
        
        # Total section
        total_layout = QHBoxLayout()
        total_layout.addWidget(QLabel("Total:"))
        
        self.spot_buy_total = QLabel("0.00")
        total_layout.addWidget(self.spot_buy_total)
        
        buy_layout.addLayout(total_layout)
        
        # Stop loss / Take profit
        sl_tp_layout = QHBoxLayout()
        
        sl_tp_layout.addWidget(QLabel("Stop Loss:"))
        self.spot_buy_sl = PriceLineEdit()
        sl_tp_layout.addWidget(self.spot_buy_sl)
        
        sl_tp_layout.addWidget(QLabel("Take Profit:"))
        self.spot_buy_tp = PriceLineEdit()
        sl_tp_layout.addWidget(self.spot_buy_tp)
        
        buy_layout.addLayout(sl_tp_layout)
        
        # Buy button
        self.spot_buy_button = QPushButton("Buy / Long")
        self.spot_buy_button.setStyleSheet("background-color: #28a745; color: white; font-weight: bold; padding: 8px;")
        self.spot_buy_button.clicked.connect(lambda: self.createOrder("buy", "spot"))
        buy_layout.addWidget(self.spot_buy_button)
        
        self.spot_action_tabs.addTab(buy_tab, "Buy")
        
        # Sell tab
        sell_tab = QWidget()
        sell_layout = QVBoxLayout(sell_tab)
        
        # Price input
        sell_layout.addWidget(QLabel("Price:"))
        self.spot_sell_price = PriceLineEdit()
        sell_layout.addWidget(self.spot_sell_price)
        
        # Order type selector
        self.spot_sell_type = OrderTypeSelector()
        sell_layout.addWidget(self.spot_sell_type)
        
        # Size widget
        self.spot_sell_size = OrderSizeWidget(self.trading_system)
        sell_layout.addWidget(self.spot_sell_size)
        
        # Total section
        sell_total_layout = QHBoxLayout()
        sell_total_layout.addWidget(QLabel("Total:"))
        
        self.spot_sell_total = QLabel("0.00")
        sell_total_layout.addWidget(self.spot_sell_total)
        
        sell_layout.addLayout(sell_total_layout)
        
        # Stop loss / Take profit
        sell_sl_tp_layout = QHBoxLayout()
        
        sell_sl_tp_layout.addWidget(QLabel("Stop Loss:"))
        self.spot_sell_sl = PriceLineEdit()
        sell_sl_tp_layout.addWidget(self.spot_sell_sl)
        
        sell_sl_tp_layout.addWidget(QLabel("Take Profit:"))
        self.spot_sell_tp = PriceLineEdit()
        sell_sl_tp_layout.addWidget(self.spot_sell_tp)
        
        sell_layout.addLayout(sell_sl_tp_layout)
        
        # Sell button
        self.spot_sell_button = QPushButton("Sell / Short")
        self.spot_sell_button.setStyleSheet("background-color: #dc3545; color: white; font-weight: bold; padding: 8px;")
        self.spot_sell_button.clicked.connect(lambda: self.createOrder("sell", "spot"))
        sell_layout.addWidget(self.spot_sell_button)
        
        self.spot_action_tabs.addTab(sell_tab, "Sell")
        
        layout.addWidget(self.spot_action_tabs)
        
        return tab
        
    def createAdvancedTab(self):
        """Create advanced orders tab."""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # Order type
        type_layout = QHBoxLayout()
        type_layout.addWidget(QLabel("Order Type:"))
        
        self.advanced_type_combo = QComboBox()
        self.advanced_type_combo.addItems([
            "OCO (One Cancels Other)", 
            "Trailing Stop", 
            "Scaled Order", 
            "Iceberg Order"
        ])
        self.advanced_type_combo.currentTextChanged.connect(self.updateAdvancedForm)
        type_layout.addWidget(self.advanced_type_combo, 1)
        
        layout.addLayout(type_layout)
        
        # Stacked widget for different forms
        self.advanced_stack = QStackedWidget()
        
        # OCO form
        oco_widget = QWidget()
        oco_layout = QVBoxLayout(oco_widget)
        
        # Side selection
        side_layout = QHBoxLayout()
        
        self.oco_buy_radio = QRadioButton("Buy")
        self.oco_buy_radio.setChecked(True)
        side_layout.addWidget(self.oco_buy_radio)
        
        self.oco_sell_radio = QRadioButton("Sell")
        side_layout.addWidget(self.oco_sell_radio)
        
        oco_layout.addLayout(side_layout)
        
        # Size widget
        self.oco_size = OrderSizeWidget(self.trading_system)
        oco_layout.addWidget(self.oco_size)
        
        # Price limits
        oco_limits_layout = QGridLayout()
        
        oco_limits_layout.addWidget(QLabel("Limit Order:"), 0, 0)
        self.oco_limit_price = PriceLineEdit()
        oco_limits_layout.addWidget(self.oco_limit_price, 0, 1)
        
        oco_limits_layout.addWidget(QLabel("Stop Order:"), 1, 0)
        self.oco_stop_price = PriceLineEdit()
        oco_limits_layout.addWidget(self.oco_stop_price, 1, 1)
        
        oco_limits_layout.addWidget(QLabel("Stop Limit:"), 2, 0)
        self.oco_stop_limit = PriceLineEdit()
        oco_limits_layout.addWidget(self.oco_stop_limit, 2, 1)
        
        oco_layout.addLayout(oco_limits_layout)
        
        # Submit button
        self.oco_submit_btn = QPushButton("Place OCO Order")
        self.oco_submit_btn.clicked.connect(lambda: self.createOrder(
            "buy" if self.oco_buy_radio.isChecked() else "sell", 
            "oco"
        ))
        oco_layout.addWidget(self.oco_submit_btn)
        
        self.advanced_stack.addWidget(oco_widget)
        
        # Trailing Stop form
        trailing_widget = QWidget()
        trailing_layout = QVBoxLayout(trailing_widget)
        
        # Side selection
        ts_side_layout = QHBoxLayout()
        
        self.ts_buy_radio = QRadioButton("Buy")
        self.ts_buy_radio.setChecked(True)
        ts_side_layout.addWidget(self.ts_buy_radio)
        
        self.ts_sell_radio = QRadioButton("Sell")
        ts_side_layout.addWidget(self.ts_sell_radio)
        
        trailing_layout.addLayout(ts_side_layout)
        
        # Size widget
        self.ts_size = OrderSizeWidget(self.trading_system)
        trailing_layout.addWidget(self.ts_size)
        
        # Trailing settings
        ts_settings_layout = QFormLayout()
        
        self.ts_activation_price = PriceLineEdit()
        ts_settings_layout.addRow("Activation Price:", self.ts_activation_price)
        
        self.ts_trail_value = QDoubleSpinBox()
        self.ts_trail_value.setDecimals(2)
        self.ts_trail_value.setRange(0.01, 100)
        self.ts_trail_value.setValue(1.0)
        
        self.ts_trail_type = QComboBox()
        self.ts_trail_type.addItems(["Percent", "Amount"])
        
        trail_layout = QHBoxLayout()
        trail_layout.addWidget(self.ts_trail_value)
        trail_layout.addWidget(self.ts_trail_type)
        
        ts_settings_layout.addRow("Trail Value:", trail_layout)
        
        trailing_layout.addLayout(ts_settings_layout)
        
        # Submit button
        self.ts_submit_btn = QPushButton("Place Trailing Stop Order")
        self.ts_submit_btn.clicked.connect(lambda: self.createOrder(
            "buy" if self.ts_buy_radio.isChecked() else "sell", 
            "trailing_stop"
        ))
        trailing_layout.addWidget(self.ts_submit_btn)
        
        self.advanced_stack.addWidget(trailing_widget)
        
        # Scaled Order form
        scaled_widget = QWidget()
        scaled_layout = QVBoxLayout(scaled_widget)
        
        # Side selection
        scaled_side_layout = QHBoxLayout()
        
        self.scaled_buy_radio = QRadioButton("Buy")
        self.scaled_buy_radio.setChecked(True)
        scaled_side_layout.addWidget(self.scaled_buy_radio)
        
        self.scaled_sell_radio = QRadioButton("Sell")
        scaled_side_layout.addWidget(self.scaled_sell_radio)
        
        scaled_layout.addLayout(scaled_side_layout)
        
        # Size widget
        self.scaled_size = OrderSizeWidget(self.trading_system)
        scaled_layout.addWidget(self.scaled_size)
        
        # Scale settings
        scale_settings_layout = QFormLayout()
        
        self.scale_upper_price = PriceLineEdit()
        scale_settings_layout.addRow("Upper Price:", self.scale_upper_price)
        
        self.scale_lower_price = PriceLineEdit()
        scale_settings_layout.addRow("Lower Price:", self.scale_lower_price)
        
        self.scale_order_count = QSpinBox()
        self.scale_order_count.setRange(2, 50)
        self.scale_order_count.setValue(5)
        scale_settings_layout.addRow("Number of Orders:", self.scale_order_count)
        
        # Distribution type
        self.scale_distribution = QComboBox()
        self.scale_distribution.addItems(["Linear", "Decreasing", "Increasing"])
        scale_settings_layout.addRow("Distribution:", self.scale_distribution)
        
        scaled_layout.addLayout(scale_settings_layout)
        
        # Submit button
        self.scale_submit_btn = QPushButton("Place Scaled Order")
        self.scale_submit_btn.clicked.connect(lambda: self.createOrder(
            "buy" if self.scaled_buy_radio.isChecked() else "sell", 
            "scaled"
        ))
        scaled_layout.addWidget(self.scale_submit_btn)
        
        self.advanced_stack.addWidget(scaled_widget)
        
        # Iceberg Order form
        iceberg_widget = QWidget()
        iceberg_layout = QVBoxLayout(iceberg_widget)
        
        # Side selection
        iceberg_side_layout = QHBoxLayout()
        
        self.iceberg_buy_radio = QRadioButton("Buy")
        self.iceberg_buy_radio.setChecked(True)
        iceberg_side_layout.addWidget(self.iceberg_buy_radio)
        
        self.iceberg_sell_radio = QRadioButton("Sell")
        iceberg_side_layout.addWidget(self.iceberg_sell_radio)
        
        iceberg_layout.addLayout(iceberg_side_layout)
        
        # Size widget
        self.iceberg_size = OrderSizeWidget(self.trading_system)
        iceberg_layout.addWidget(self.iceberg_size)
        
        # Iceberg settings
        iceberg_settings_layout = QFormLayout()
        
        self.iceberg_price = PriceLineEdit()
        iceberg_settings_layout.addRow("Price:", self.iceberg_price)
        
        self.iceberg_visible_size = QDoubleSpinBox()
        self.iceberg_visible_size.setDecimals(8)
        self.iceberg_visible_size.setRange(0.00000001, 1000000)
        self.iceberg_visible_size.setValue(0.1)
        iceberg_settings_layout.addRow("Visible Size:", self.iceberg_visible_size)
        
        self.iceberg_variance = QDoubleSpinBox()
        self.iceberg_variance.setDecimals(2)
        self.iceberg_variance.setRange(0, 100)
        self.iceberg_variance.setValue(0)
        self.iceberg_variance.setSuffix(" %")
        iceberg_settings_layout.addRow("Variance:", self.iceberg_variance)
        
        iceberg_layout.addLayout(iceberg_settings_layout)
        
        # Submit button
        self.iceberg_submit_btn = QPushButton("Place Iceberg Order")
        self.iceberg_submit_btn.clicked.connect(lambda: self.createOrder(
            "buy" if self.iceberg_buy_radio.isChecked() else "sell", 
            "iceberg"
        ))
        iceberg_layout.addWidget(self.iceberg_submit_btn)
        
        self.advanced_stack.addWidget(iceberg_widget)
        
        layout.addWidget(self.advanced_stack)
        
        return tab
        
    def createFuturesTab(self):
        """Create futures tab."""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # Leverage selector
        self.futures_leverage = LeverageSelector()
        layout.addWidget(self.futures_leverage)
        
        # Buy/Sell tabs
        self.futures_action_tabs = QTabWidget()
        
        # Buy tab
        buy_tab = QWidget()
        buy_layout = QVBoxLayout(buy_tab)
        
        # Price input
        buy_layout.addWidget(QLabel("Price:"))
        self.futures_buy_price = PriceLineEdit()
        buy_layout.addWidget(self.futures_buy_price)
        
        # Order type selector
        self.futures_buy_type = OrderTypeSelector()
        buy_layout.addWidget(self.futures_buy_type)
        
        # Size widget
        self.futures_buy_size = OrderSizeWidget(self.trading_system)
        buy_layout.addWidget(self.futures_buy_size)
        
        # Stop loss / Take profit
        sl_tp_layout = QHBoxLayout()
        
        sl_tp_layout.addWidget(QLabel("Stop Loss:"))
        self.futures_buy_sl = PriceLineEdit()
        sl_tp_layout.addWidget(self.futures_buy_sl)
        
        sl_tp_layout.addWidget(QLabel("Take Profit:"))
        self.futures_buy_tp = PriceLineEdit()
        sl_tp_layout.addWidget(self.futures_buy_tp)
        
        buy_layout.addLayout(sl_tp_layout)
        
        # Position mode selection (Hedge vs One-way)
        position_layout = QHBoxLayout()
        position_layout.addWidget(QLabel("Position Mode:"))
        
        self.futures_position_mode = QComboBox()
        self.futures_position_mode.addItems(["One-way", "Hedge"])
        position_layout.addWidget(self.futures_position_mode, 1)
        
        buy_layout.addLayout(position_layout)
        
        # Buy button
        self.futures_buy_button = QPushButton("Buy / Long")
        self.futures_buy_button.setStyleSheet("background-color: #28a745; color: white; font-weight: bold; padding: 8px;")
        self.futures_buy_button.clicked.connect(lambda: self.createOrder("buy", "futures"))
        buy_layout.addWidget(self.futures_buy_button)
        
        self.futures_action_tabs.addTab(buy_tab, "Buy")
        
        # Sell tab
        sell_tab = QWidget()
        sell_layout = QVBoxLayout(sell_tab)
        
        # Price input
        sell_layout.addWidget(QLabel("Price:"))
        self.futures_sell_price = PriceLineEdit()
        sell_layout.addWidget(self.futures_sell_price)
        
        # Order type selector
        self.futures_sell_type = OrderTypeSelector()
        sell_layout.addWidget(self.futures_sell_type)
        
        # Size widget
        self.futures_sell_size = OrderSizeWidget(self.trading_system)
        sell_layout.addWidget(self.futures_sell_size)
        
        # Stop loss / Take profit
        sell_sl_tp_layout = QHBoxLayout()
        
        sell_sl_tp_layout.addWidget(QLabel("Stop Loss:"))
        self.futures_sell_sl = PriceLineEdit()
        sell_sl_tp_layout.addWidget(self.futures_sell_sl)
        
        sell_sl_tp_layout.addWidget(QLabel("Take Profit:"))
        self.futures_sell_tp = PriceLineEdit()
        sell_sl_tp_layout.addWidget(self.futures_sell_tp)
        
        sell_layout.addLayout(sell_sl_tp_layout)
        
        # Position mode selection (Hedge vs One-way)
        sell_position_layout = QHBoxLayout()
        sell_position_layout.addWidget(QLabel("Position Mode:"))
        
        self.futures_sell_position_mode = QComboBox()
        self.futures_sell_position_mode.addItems(["One-way", "Hedge"])
        sell_position_layout.addWidget(self.futures_sell_position_mode, 1)
        
        sell_layout.addLayout(sell_position_layout)
        
        # Sell button
        self.futures_sell_button = QPushButton("Sell / Short")
        self.futures_sell_button.setStyleSheet("background-color: #dc3545; color: white; font-weight: bold; padding: 8px;")
        self.futures_sell_button.clicked.connect(lambda: self.createOrder("sell", "futures"))
        sell_layout.addWidget(self.futures_sell_button)
        
        self.futures_action_tabs.addTab(sell_tab, "Sell")
        
        layout.addWidget(self.futures_action_tabs)
        
        # Risk warning
        risk_warning = QLabel(
            "⚠️ <b>Warning:</b> Futures trading involves significant risk of loss. "
            "Use leverage with extreme caution."
        )
        risk_warning.setStyleSheet("color: #dc3545;")
        risk_warning.setWordWrap(True)
        layout.addWidget(risk_warning)
        
        return tab
        
    def setupConnections(self):
        """Set up signal/slot connections."""
        # Symbol change
        self.symbol_combo.currentTextChanged.connect(self.onSymbolChanged)
        
        # Price updates
        self.spot_buy_price.valueChanged.connect(self.updateBuyTotal)
        self.spot_sell_price.valueChanged.connect(self.updateSellTotal)
        
        self.spot_buy_size.sizeChanged.connect(self.updateBuyTotal)
        self.spot_sell_size.sizeChanged.connect(self.updateSellTotal)
        
        # Order type changes
        self.spot_buy_type.orderTypeChanged.connect(self.onOrderTypeChanged)
        self.spot_sell_type.orderTypeChanged.connect(self.onOrderTypeChanged)
        
        # Advanced tab changes
        self.advanced_type_combo.currentIndexChanged.connect(self.advanced_stack.setCurrentIndex)
        
        # Set market price when selecting market order
        self.spot_buy_type.orderTypeChanged.connect(self.updatePriceForOrderType)
        self.spot_sell_type.orderTypeChanged.connect(self.updatePriceForOrderType)
        
        # Connect order history selection to form
        self.order_history.orderSelected.connect(self.fillOrderForm)
        
        # Start price update timer
        self.price_timer = QTimer(self)
        self.price_timer.timeout.connect(self.updatePrice)
        self.price_timer.start(2000)  # Update every 2 seconds
        
    def loadSymbols(self):
        """Load available symbols from trading system."""
        try:
            # Get symbols from trading system
            if hasattr(self.trading_system, 'get_markets'):
                markets = self.trading_system.get_markets()
                symbols = [m['symbol'] for m in markets] if markets else []
            else:
                # Default symbols
                symbols = [
                    "BTC/USDT", "ETH/USDT", "BNB/USDT", "SOL/USDT", 
                    "XRP/USDT", "ADA/USDT", "DOGE/USDT", "AVAX/USDT"
                ]
                
            # Add symbols to combo box
            self.symbol_combo.clear()
            self.symbol_combo.addItems(symbols)
            
            # Set current symbol
            if symbols:
                self.current_symbol = symbols[0]
                self.onSymbolChanged(self.current_symbol)
                
        except Exception as e:
            logging.error(f"Error loading symbols: {e}")
            # Add default symbols
            default_symbols = ["BTC/USDT", "ETH/USDT"]
            self.symbol_combo.clear()
            self.symbol_combo.addItems(default_symbols)
            
            if default_symbols:
                self.current_symbol = default_symbols[0]
                self.onSymbolChanged(self.current_symbol)
        
    def onSymbolChanged(self, symbol):
        """Handle symbol selection change."""
        if not symbol:
            return
            
        self.current_symbol = symbol
        
        # Update price
        self.updatePrice()
        
        # Update size widget symbol
        self.spot_buy_size.setSymbol(symbol)
        self.spot_sell_size.setSymbol(symbol)
        self.oco_size.setSymbol(symbol)
        self.ts_size.setSymbol(symbol)
        self.scaled_size.setSymbol(symbol)
        self.iceberg_size.setSymbol(symbol)
        self.futures_buy_size.setSymbol(symbol)
        self.futures_sell_size.setSymbol(symbol)
        
        # Update order book
        self.updateOrderBook()
        
        # Refresh open orders for this symbol
        self.order_history.refreshOrders()
        
    def updatePrice(self):
        """Update price for current symbol."""
        try:
            if not self.current_symbol:
                return
                
            # Get price from trading system
            if hasattr(self.trading_system, 'get_ticker'):
                ticker = self.trading_system.get_ticker(self.current_symbol)
                if ticker:
                    price = ticker.get('last', 0)
                    change = ticker.get('percentage', 0)
                    
                    # Update current price
                    self.current_price = price
                    
                    # Update price widgets
                    self.price_label.setText(f"{price:.8f}".rstrip('0').rstrip('.'))
                    
                    # Set color based on change
                    if change >= 0:
                        self.change_label.setText(f"+{change:.2f}%")
                        self.change_label.setStyleSheet("font-weight: bold; color: #28a745;")
                    else:
                        self.change_label.setText(f"{change:.2f}%")
                        self.change_label.setStyleSheet("font-weight: bold; color: #dc3545;")
                    
                    # Update price inputs if empty or market order is selected
                    if not self.spot_buy_price.text() or self.spot_buy_type.getOrderType() == "Market":
                        self.spot_buy_price.setText(f"{price:.8f}".rstrip('0').rstrip('.'))
                        
                    if not self.spot_sell_price.text() or self.spot_sell_type.getOrderType() == "Market":
                        self.spot_sell_price.setText(f"{price:.8f}".rstrip('0').rstrip('.'))
                    
                    # Update size widgets with current price
                    self.spot_buy_size.setCurrentPrice(price)
                    self.spot_sell_size.setCurrentPrice(price)
                    self.oco_size.setCurrentPrice(price)
                    self.ts_size.setCurrentPrice(price)
                    self.scaled_size.setCurrentPrice(price)
                    self.iceberg_size.setCurrentPrice(price)
                    self.futures_buy_size.setCurrentPrice(price)
                    self.futures_sell_size.setCurrentPrice(price)
            
            # Update balance
            if hasattr(self.trading_system, 'get_balance'):
                balance = self.trading_system.get_balance(self.current_symbol.split('/')[1])
                if balance:
                    self.current_balance = balance
                    self.balance_label.setText(f"{balance:.8f}".rstrip('0').rstrip('.'))
                
        except Exception as e:
            logging.error(f"Error updating price: {e}")
        
    def updateOrderBook(self):
        """Update order book visualization."""
        try:
            if not self.current_symbol:
                return
                
            # Get order book from trading system
            if hasattr(self.trading_system, 'get_order_book'):
                order_book = self.trading_system.get_order_book(self.current_symbol)
                if order_book:
                    bids = order_book.get('bids', [])
                    asks = order_book.get('asks', [])
                    
                    # Update depth visualization
                    self.market_depth.setOrderBook(bids, asks)
                    
        except Exception as e:
            logging.error(f"Error updating order book: {e}")
        
    def updateBuyTotal(self, _=None):
        """Update total for buy order."""
        try:
            price = float(self.spot_buy_price.text() or "0")
            size = self.spot_buy_size.getCurrentSize()
            
            total = price * size
            
            self.spot_buy_total.setText(f"{total:.8f}".rstrip('0').rstrip('.'))
            
        except Exception as e:
            logging.error(f"Error updating buy total: {e}")
        
    def updateSellTotal(self, _=None):
        """Update total for sell order."""
        try:
            price = float(self.spot_sell_price.text() or "0")
            size = self.spot_sell_size.getCurrentSize()
            
            total = price * size
            
            self.spot_sell_total.setText(f"{total:.8f}".rstrip('0').rstrip('.'))
            
        except Exception as e:
            logging.error(f"Error updating sell total: {e}")
        
    def onOrderTypeChanged(self, order_type):
        """Handle order type change."""
        # Enable/disable price input based on order type
        sender = self.sender()
        
        if sender == self.spot_buy_type:
            self.spot_buy_price.setEnabled(order_type != "Market")
        elif sender == self.spot_sell_type:
            self.spot_sell_price.setEnabled(order_type != "Market")
        elif sender == self.futures_buy_type:
            self.futures_buy_price.setEnabled(order_type != "Market")
        elif sender == self.futures_sell_type:
            self.futures_sell_price.setEnabled(order_type != "Market")
        
    def updatePriceForOrderType(self, order_type):
        """Update price input for market orders."""
        if order_type == "Market":
            # Use current market price for market orders
            sender = self.sender()
            
            if sender == self.spot_buy_type:
                self.spot_buy_price.setText(f"{self.current_price:.8f}".rstrip('0').rstrip('.'))
            elif sender == self.spot_sell_type:
                self.spot_sell_price.setText(f"{self.current_price:.8f}".rstrip('0').rstrip('.'))
            elif sender == self.futures_buy_type:
                self.futures_buy_price.setText(f"{self.current_price:.8f}".rstrip('0').rstrip('.'))
            elif sender == self.futures_sell_type:
                self.futures_sell_price.setText(f"{self.current_price:.8f}".rstrip('0').rstrip('.'))
        
    def updateAdvancedForm(self, order_type):
        """Update advanced order form based on selected type."""
        index = self.advanced_type_combo.currentIndex()
        self.advanced_stack.setCurrentIndex(index)
        
    def fillOrderForm(self, order):
        """Fill order form with data from selected order."""
        if not order:
            return
            
        # Get order details
        symbol = order.get('symbol', '')
        side = order.get('side', '')
        order_type = order.get('type', '')
        price = order.get('price', 0)
        amount = order.get('amount', 0)
        
        # Set symbol
        index = self.symbol_combo.findText(symbol)
        if index >= 0:
            self.symbol_combo.setCurrentIndex(index)
            
        # Fill appropriate form based on side
        if side == 'buy':
            # Select buy tab
            self.spot_action_tabs.setCurrentIndex(0)
            
            # Set price and size
            self.spot_buy_price.setText(f"{price:.8f}".rstrip('0').rstrip('.'))
            self.spot_buy_size.size_input.setValue(amount)
            
            # Set order type
            index = self.spot_buy_type.type_combo.findText(order_type.capitalize())
            if index >= 0:
                self.spot_buy_type.type_combo.setCurrentIndex(index)
                
        elif side == 'sell':
            # Select sell tab
            self.spot_action_tabs.setCurrentIndex(1)
            
            # Set price and size
            self.spot_sell_price.setText(f"{price:.8f}".rstrip('0').rstrip('.'))
            self.spot_sell_size.size_input.setValue(amount)
            
            # Set order type
            index = self.spot_sell_type.type_combo.findText(order_type.capitalize())
            if index >= 0:
                self.spot_sell_type.type_combo.setCurrentIndex(index)
        
    def validateOrder(self, side, market_type, price, amount):
        """Validate order parameters."""
        errors = []
        
        # Check symbol
        if not self.current_symbol:
            errors.append("No trading pair selected")
            
        # Check price (except for market orders)
        if market_type != "Market" and price <= 0:
            errors.append("Invalid price")
            
        # Check amount
        if amount <= 0:
            errors.append("Invalid amount")
            
        # Check balance for buys
        if side == "buy" and market_type != "futures":
            quote_currency = self.current_symbol.split('/')[1]
            required_balance = price * amount
            
            # Get actual balance
            if hasattr(self.trading_system, 'get_balance'):
                balance = self.trading_system.get_balance(quote_currency) or 0
                
                if required_balance > balance:
                    errors.append(f"Insufficient {quote_currency} balance")
                    
        # Check balance for sells
        if side == "sell" and market_type != "futures":
            base_currency = self.current_symbol.split('/')[0]
            
            # Get actual balance
            if hasattr(self.trading_system, 'get_balance'):
                balance = self.trading_system.get_balance(base_currency) or 0
                
                if amount > balance:
                    errors.append(f"Insufficient {base_currency} balance")
                    
        return errors
        
    def createOrder(self, side, market_type):
        """Create a new order."""
        try:
            # Get order parameters based on market type and side
            if market_type == "spot":
                if side == "buy":
                    order_type = self.spot_buy_type.getOrderType()
                    price = float(self.spot_buy_price.text() or "0")
                    amount = self.spot_buy_size.getCurrentSize()
                    stop_loss = float(self.spot_buy_sl.text() or "0")
                    take_profit = float(self.spot_buy_tp.text() or "0")
                    tif = self.spot_buy_type.getTimeInForce()
                    reduce_only = self.spot_buy_type.isReduceOnly()
                    post_only = self.spot_buy_type.isPostOnly()
                else:
                    order_type = self.spot_sell_type.getOrderType()
                    price = float(self.spot_sell_price.text() or "0")
                    amount = self.spot_sell_size.getCurrentSize()
                    stop_loss = float(self.spot_sell_sl.text() or "0")
                    take_profit = float(self.spot_sell_tp.text() or "0")
                    tif = self.spot_sell_type.getTimeInForce()
                    reduce_only = self.spot_sell_type.isReduceOnly()
                    post_only = self.spot_sell_type.isPostOnly()
                    
            elif market_type == "futures":
                leverage = self.futures_leverage.getLeverage()
                
                if side == "buy":
                    order_type = self.futures_buy_type.getOrderType()
                    price = float(self.futures_buy_price.text() or "0")
                    amount = self.futures_buy_size.getCurrentSize()
                    stop_loss = float(self.futures_buy_sl.text() or "0")
                    take_profit = float(self.futures_buy_tp.text() or "0")
                    tif = self.futures_buy_type.getTimeInForce()
                    reduce_only = self.futures_buy_type.isReduceOnly()
                    post_only = self.futures_buy_type.isPostOnly()
                    position_mode = self.futures_position_mode.currentText()
                else:
                    order_type = self.futures_sell_type.getOrderType()
                    price = float(self.futures_sell_price.text() or "0")
                    amount = self.futures_sell_size.getCurrentSize()
                    stop_loss = float(self.futures_sell_sl.text() or "0")
                    take_profit = float(self.futures_sell_tp.text() or "0")
                    tif = self.futures_sell_type.getTimeInForce()
                    reduce_only = self.futures_sell_type.isReduceOnly()
                    post_only = self.futures_sell_type.isPostOnly()
                    position_mode = self.futures_sell_position_mode.currentText()
                    
            elif market_type == "oco":
                side = "buy" if self.oco_buy_radio.isChecked() else "sell"
                order_type = "OCO"
                amount = self.oco_size.getCurrentSize()
                limit_price = float(self.oco_limit_price.text() or "0")
                stop_price = float(self.oco_stop_price.text() or "0")
                stop_limit_price = float(self.oco_stop_limit.text() or "0")
                
                # Use limit price as main price
                price = limit_price
                
            elif market_type == "trailing_stop":
                side = "buy" if self.ts_buy_radio.isChecked() else "sell"
                order_type = "Trailing Stop"
                amount = self.ts_size.getCurrentSize()
                price = float(self.ts_activation_price.text() or "0")
                trail_value = self.ts_trail_value.value()
                trail_type = self.ts_trail_type.currentText()
                
            elif market_type == "scaled":
                side = "buy" if self.scaled_buy_radio.isChecked() else "sell"
                order_type = "Scaled"
                amount = self.scaled_size.getCurrentSize()
                upper_price = float(self.scale_upper_price.text() or "0")
                lower_price = float(self.scale_lower_price.text() or "0")
                order_count = self.scale_order_count.value()
                distribution = self.scale_distribution.currentText()
                
                # Use upper price as main price
                price = upper_price
                
            elif market_type == "iceberg":
                side = "buy" if self.iceberg_buy_radio.isChecked() else "sell"
                order_type = "Iceberg"
                amount = self.iceberg_size.getCurrentSize()
                price = float(self.iceberg_price.text() or "0")
                visible_size = self.iceberg_visible_size.value()
                variance = self.iceberg_variance.value() / 100.0
                
            else:
                logging.error(f"Unknown market type: {market_type}")
                return
                
            # Validate order
            errors = self.validateOrder(side, market_type, price, amount)
            
            if errors:
                error_msg = "\n".join(errors)
                QMessageBox.warning(self, "Order Validation", f"Cannot create order:\n{error_msg}")
                return
                
            # Create order object
            order = {
                'symbol': self.current_symbol,
                'side': side,
                'type': order_type.lower(),
                'price': price,
                'amount': amount
            }
            
            # Add additional parameters based on order type
            if order_type != "Market":
                order['timeInForce'] = tif
                
                if post_only:
                    order['postOnly'] = True
                    
            if reduce_only:
                order['reduceOnly'] = True
                
            # Add stop loss and take profit if set
            if stop_loss > 0:
                order['stopLoss'] = stop_loss
                
            if take_profit > 0:
                order['takeProfit'] = take_profit
                
            # Add futures specific parameters
            if market_type == "futures":
                order['leverage'] = leverage
                order['positionMode'] = position_mode.lower()
                
            # Add OCO specific parameters
            if market_type == "oco":
                order['limitPrice'] = limit_price
                order['stopPrice'] = stop_price
                order['stopLimitPrice'] = stop_limit_price
                
            # Add trailing stop specific parameters
            if market_type == "trailing_stop":
                order['activationPrice'] = price
                order['trailValue'] = trail_value
                order['trailType'] = trail_type.lower()
                
            # Add scaled order specific parameters
            if market_type == "scaled":
                order['upperPrice'] = upper_price
                order['lowerPrice'] = lower_price
                order['orderCount'] = order_count
                order['distribution'] = distribution.lower()
                
            # Add iceberg specific parameters
            if market_type == "iceberg":
                order['visibleSize'] = visible_size
                order['variance'] = variance
                
            # Create order with trading system
            if hasattr(self.trading_system, 'create_order'):
                # Confirm order
                confirm_msg = f"Create {order_type} {side.upper()} order for {amount} {self.current_symbol}?"
                if order_type != "Market":
                    confirm_msg += f"\nPrice: {price}"
                    
                if stop_loss > 0:
                    confirm_msg += f"\nStop Loss: {stop_loss}"
                    
                if take_profit > 0:
                    confirm_msg += f"\nTake Profit: {take_profit}"
                    
                if market_type == "futures":
                    confirm_msg += f"\nLeverage: {leverage}x"
                    
                reply = QMessageBox.question(
                    self, 
                    "Confirm Order", 
                    confirm_msg,
                    QMessageBox.Yes | QMessageBox.No,
                    QMessageBox.No
                )
                
                if reply != QMessageBox.Yes:
                    return
                    
                # Submit order
                result = self.trading_system.create_order(order)
                
                if result and 'id' in result:
                    # Order created successfully
                    QMessageBox.information(
                        self, 
                        "Order Created", 
                        f"Order created successfully\nOrder ID: {result['id']}"
                    )
                    
                    # Refresh orders
                    self.order_history.refreshOrders()
                    
                    # Clear form if successful
                    self.clearForm(side, market_type)
                    
                    # Emit signal
                    self.orderCreated.emit(result)
                else:
                    # Order creation failed
                    error_msg = result.get('error', 'Unknown error') if result else 'Failed to create order'
                    QMessageBox.warning(self, "Order Failed", error_msg)
            else:
                # Trading system doesn't support order creation
                QMessageBox.warning(
                    self, 
                    "Not Supported", 
                    "Order creation is not supported by the trading system"
                )
                
        except Exception as e:
            logging.error(f"Error creating order: {e}")
            QMessageBox.critical(self, "Error", f"Error creating order: {str(e)}")
        
    def clearForm(self, side, market_type):
        """Clear form after successful order."""
        if market_type == "spot":
            if side == "buy":
                self.spot_buy_size.size_input.setValue(0)
                self.spot_buy_sl.setText("")
                self.spot_buy_tp.setText("")
            else:
                self.spot_sell_size.size_input.setValue(0)
                self.spot_sell_sl.setText("")
                self.spot_sell_tp.setText("")
                
        elif market_type == "futures":
            if side == "buy":
                self.futures_buy_size.size_input.setValue(0)
                self.futures_buy_sl.setText("")
                self.futures_buy_tp.setText("")
            else:
                self.futures_sell_size.size_input.setValue(0)
                self.futures_sell_sl.setText("")
                self.futures_sell_tp.setText("")
                
        elif market_type == "oco":
            self.oco_size.size_input.setValue(0)
            self.oco_limit_price.setText("")
            self.oco_stop_price.setText("")
            self.oco_stop_limit.setText("")
            
        elif market_type == "trailing_stop":
            self.ts_size.size_input.setValue(0)
            self.ts_activation_price.setText("")
            
        elif market_type == "scaled":
            self.scaled_size.size_input.setValue(0)
            self.scale_upper_price.setText("")
            self.scale_lower_price.setText("")
            
        elif market_type == "iceberg":
            self.iceberg_size.size_input.setValue(0)
            self.iceberg_price.setText("")
