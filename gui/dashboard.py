# gui/dashboard.py

import os
import sys
import logging
import datetime
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Union
from collections import deque

# PyQt5 imports
from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QGridLayout, QSplitter, 
                           QLabel, QPushButton, QFrame, QTableWidget, QTableWidgetItem, 
                           QHeaderView, QComboBox, QTabWidget, QProgressBar, QToolButton,
                           QMenu, QAction, QMessageBox, QLineEdit, QSpinBox, QDoubleSpinBox,
                           QCheckBox, QScrollArea, QSizePolicy, QApplication, QStyle)
from PyQt5.QtCore import Qt, QSize, QTimer, QThread, pyqtSignal, QDateTime, QRect, QPoint
from PyQt5.QtGui import QColor, QPainter, QPen, QBrush, QFont, QIcon, QPalette, QPixmap

# Data visualization
import pyqtgraph as pg
from pyqtgraph import PlotWidget, plot, mkPen

# Import error handling
try:
    from error_handling import safe_execute, ErrorCategory, ErrorSeverity
    HAVE_ERROR_HANDLING = True
except ImportError:
    HAVE_ERROR_HANDLING = False
    logging.warning("Error handling module not available. Using simplified error handling.")

class RefreshWorker(QThread):
    """Background worker thread for refreshing dashboard data"""
    
    data_ready = pyqtSignal(dict)
    error_occurred = pyqtSignal(str)
    
    def __init__(self, trading_system, symbols=None, timeframe='1h', parent=None):
        super().__init__(parent)
        self.trading_system = trading_system
        self.symbols = symbols or []
        self.timeframe = timeframe
        self.running = True
        
    def run(self):
        """Refresh data in background thread"""
        try:
            # Collect all required data
            data = {}
            
            # 1. Portfolio data
            try:
                portfolio = self.trading_system.get_portfolio_value()
                data['portfolio'] = portfolio
            except Exception as e:
                logging.error(f"Error fetching portfolio data: {e}")
                data['portfolio'] = None
            
            # 2. Active positions
            try:
                positions = self.trading_system.get_active_positions()
                data['positions'] = positions
            except Exception as e:
                logging.error(f"Error fetching positions: {e}")
                data['positions'] = None
                
            # 3. Recent trades
            try:
                trades = self.trading_system.get_recent_trades(limit=20)
                data['trades'] = trades
            except Exception as e:
                logging.error(f"Error fetching trades: {e}")
                data['trades'] = None
                
            # 4. Market data for each symbol
            data['market_data'] = {}
            for symbol in self.symbols:
                try:
                    market_data = self.trading_system.get_market_data(
                        symbol, timeframe=self.timeframe, limit=100
                    )
                    data['market_data'][symbol] = market_data
                except Exception as e:
                    logging.error(f"Error fetching market data for {symbol}: {e}")
                    data['market_data'][symbol] = None
                    
            # 5. Performance metrics
            try:
                metrics = self.trading_system.get_performance_metrics()
                data['metrics'] = metrics
            except Exception as e:
                logging.error(f"Error fetching performance metrics: {e}")
                data['metrics'] = None
                
            # 6. AI predictions if available
            try:
                if hasattr(self.trading_system, 'ai_engine'):
                    predictions = {}
                    for symbol in self.symbols:
                        symbol_data = data['market_data'][symbol]
                        if symbol_data is not None:
                            # Get AI predictions for the symbol
                            symbol_pred, rl_signals = self.trading_system.ai_engine.get_ai_signals(
                                symbol, symbol_data
                            )
                            predictions[symbol] = {
                                'ai_predictions': symbol_pred,
                                'rl_signals': rl_signals
                            }
                    data['predictions'] = predictions
            except Exception as e:
                logging.error(f"Error fetching AI predictions: {e}")
                data['predictions'] = None
                
            # 7. Risk metrics
            try:
                risk_data = {}
                # Current drawdown
                portfolio_history = self.trading_system.get_portfolio_history(days=30)
                if not portfolio_history.empty:
                    peak = portfolio_history['total_value'].cummax()
                    drawdown = (portfolio_history['total_value'] / peak - 1)
                    current_drawdown = drawdown.iloc[-1]
                    max_drawdown = drawdown.min()
                    risk_data['current_drawdown'] = current_drawdown
                    risk_data['max_drawdown'] = max_drawdown
                    
                # VaR if available
                if hasattr(self.trading_system, 'risk_manager'):
                    var = self.trading_system.risk_manager.get_portfolio_var()
                    risk_data['var'] = var
                    
                data['risk'] = risk_data
            except Exception as e:
                logging.error(f"Error fetching risk metrics: {e}")
                data['risk'] = None
                
            # Emit the complete data set
            self.data_ready.emit(data)
            
        except Exception as e:
            logging.error(f"Error in refresh worker: {e}")
            self.error_occurred.emit(str(e))

class PerformanceChart(QWidget):
    """Widget for displaying portfolio performance chart"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        # Create layout
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        
        # Create plot widget
        self.plot_widget = pg.PlotWidget()
        self.plot_widget.setBackground('w')
        self.plot_widget.setLabel('left', 'Portfolio Value')
        self.plot_widget.setLabel('bottom', 'Date')
        layout.addWidget(self.plot_widget)
        
        # Setup plot items
        self.portfolio_curve = self.plot_widget.plot(
            pen=pg.mkPen(color=(0, 150, 0), width=2)
        )
        self.drawdown_curve = self.plot_widget.plot(
            pen=pg.mkPen(color=(200, 0, 0), width=1)
        )
        
        # Additional configuration
        self.date_axis = pg.DateAxisItem(orientation='bottom')
        self.plot_widget.setAxisItems({'bottom': self.date_axis})
        
        # Add legend
        self.plot_widget.addLegend()
        self.portfolio_curve.setName("Portfolio Value")
        self.drawdown_curve.setName("Drawdown %")
        
    def update_chart(self, portfolio_data):
        """Update chart with new portfolio data"""
        if portfolio_data is None or portfolio_data.empty:
            return
            
        # Convert timestamps to unix timestamps for date axis
        timestamps = [pd.Timestamp(ts).timestamp() for ts in portfolio_data.index]
        
        # Plot portfolio value
        values = portfolio_data['total_value'].values
        self.portfolio_curve.setData(timestamps, values)
        
        # Calculate and plot drawdown
        peak = portfolio_data['total_value'].cummax()
        drawdown = (portfolio_data['total_value'] / peak - 1) * 100  # as percentage
        
        # Create second y-axis for drawdown
        drawdown_axis = self.plot_widget.getAxis('right')
        drawdown_axis.setLabel('Drawdown %')
        drawdown_axis.setGrid(True)
        self.plot_widget.showAxis('right')
        
        # Update drawdown curve
        self.drawdown_curve.setData(timestamps, drawdown.values)
        
        # Auto-scale view
        self.plot_widget.autoRange()

class MarketChart(QWidget):
    """Widget for displaying market data charts with indicators"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        # Create layout
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        
        # Control panel
        control_layout = QHBoxLayout()
        
        # Symbol selector
        self.symbol_selector = QComboBox()
        self.symbol_selector.currentIndexChanged.connect(self.symbol_changed)
        control_layout.addWidget(QLabel("Symbol:"))
        control_layout.addWidget(self.symbol_selector)
        
        # Timeframe selector
        self.timeframe_selector = QComboBox()
        self.timeframe_selector.addItems(['1m', '5m', '15m', '1h', '4h', '1d'])
        self.timeframe_selector.setCurrentText('1h')
        self.timeframe_selector.currentIndexChanged.connect(self.timeframe_changed)
        control_layout.addWidget(QLabel("Timeframe:"))
        control_layout.addWidget(self.timeframe_selector)
        
        # Indicator selector
        self.indicator_selector = QComboBox()
        self.indicator_selector.addItems(['None', 'MA', 'EMA', 'MACD', 'RSI', 'Bollinger Bands'])
        self.indicator_selector.currentIndexChanged.connect(self.indicator_changed)
        control_layout.addWidget(QLabel("Indicator:"))
        control_layout.addWidget(self.indicator_selector)
        
        # Add prediction toggle
        self.show_predictions_cb = QCheckBox("Show AI Predictions")
        self.show_predictions_cb.setChecked(True)
        self.show_predictions_cb.stateChanged.connect(self.toggle_predictions)
        control_layout.addWidget(self.show_predictions_cb)
        
        # Add control panel to layout
        layout.addLayout(control_layout)
        
        # Create tab widget for different chart types
        self.chart_tabs = QTabWidget()
        
        # Candlestick chart
        self.candle_chart = pg.PlotWidget()
        self.candle_chart.setBackground('w')
        self.chart_tabs.addTab(self.candle_chart, "Candlestick")
        
        # Line chart
        self.line_chart = pg.PlotWidget()
        self.line_chart.setBackground('w')
        self.chart_tabs.addTab(self.line_chart, "Line")
        
        # Volume chart
        self.volume_chart = pg.PlotWidget()
        self.volume_chart.setBackground('w')
        self.chart_tabs.addTab(self.volume_chart, "Volume")
        
        # Indicator chart
        self.indicator_chart = pg.PlotWidget()
        self.indicator_chart.setBackground('w')
        self.chart_tabs.addTab(self.indicator_chart, "Indicator")
        
        # Add chart tabs to layout
        layout.addWidget(self.chart_tabs)
        
        # Setup date axis
        self.date_axis = pg.DateAxisItem(orientation='bottom')
        self.candle_chart.setAxisItems({'bottom': self.date_axis})
        self.line_chart.setAxisItems({'bottom': self.date_axis})
        self.volume_chart.setAxisItems({'bottom': self.date_axis})
        self.indicator_chart.setAxisItems({'bottom': self.date_axis})
        
        # Current data
        self.current_symbol = None
        self.current_timeframe = '1h'
        self.current_indicator = 'None'
        self.show_predictions = True
        self.market_data = {}
        self.predictions = {}
        
        # Plot items
        self.candle_plot = None
        self.line_plot = None
        self.volume_plot = None
        self.indicator_plots = []
        self.prediction_plots = []
        
    def symbol_changed(self, index):
        """Handle symbol selection change"""
        symbol = self.symbol_selector.currentText()
        if not symbol:
            return
            
        self.current_symbol = symbol
        self.update_charts()
        
    def timeframe_changed(self, index):
        """Handle timeframe selection change"""
        self.current_timeframe = self.timeframe_selector.currentText()
        # Would trigger a data refresh in a real implementation
        
    def indicator_changed(self, index):
        """Handle indicator selection change"""
        self.current_indicator = self.indicator_selector.currentText()
        self.update_charts()
        
    def toggle_predictions(self, state):
        """Toggle showing AI predictions"""
        self.show_predictions = state == Qt.Checked
        self.update_charts()
        
    def set_symbols(self, symbols):
        """Set available symbols"""
        self.symbol_selector.clear()
        self.symbol_selector.addItems(symbols)
        if symbols:
            self.current_symbol = symbols[0]
            
    def update_data(self, market_data, predictions=None):
        """Update with new market data"""
        self.market_data = market_data
        self.predictions = predictions or {}
        self.update_charts()
        
    def update_charts(self):
        """Update all charts with current data"""
        if not self.current_symbol or self.current_symbol not in self.market_data:
            return
            
        df = self.market_data[self.current_symbol]
        if df is None or df.empty:
            return
            
        # Convert timestamps for x-axis
        timestamps = [pd.Timestamp(ts).timestamp() for ts in df.index]
        
        # Clear previous plots
        self.candle_chart.clear()
        self.line_chart.clear()
        self.volume_chart.clear()
        self.indicator_chart.clear()
        self.indicator_plots = []
        self.prediction_plots = []
        
        # Create candlestick items
        for i in range(len(df)):
            if i >= len(timestamps):
                continue
                
            # Get candle data
            t = timestamps[i]
            open_price = df['open'].iloc[i]
            close_price = df['close'].iloc[i]
            high_price = df['high'].iloc[i]
            low_price = df['low'].iloc[i]
            
            # Determine candle color
            candle_color = 'g' if close_price >= open_price else 'r'
            
            # Draw candle body
            body = pg.BarGraphItem(
                x=[t], height=[abs(close_price - open_price)], 
                width=0.7, brush=candle_color, pen=candle_color
            )
            body.setPos(0, min(open_price, close_price))
            self.candle_chart.addItem(body)
            
            # Draw candle wick
            wick = pg.PlotDataItem(
                x=[t, t], y=[low_price, high_price], 
                pen=pg.mkPen(candle_color, width=1)
            )
            self.candle_chart.addItem(wick)
            
        # Draw line chart
        line_pen = pg.mkPen(color=(0, 0, 255), width=2)
        self.line_chart.plot(timestamps, df['close'].values, pen=line_pen)
        
        # Draw volume chart
        volume_data = df['volume'].values
        self.volume_chart.plot(
            timestamps, volume_data, 
            fillLevel=0, fillBrush=(100, 100, 255, 100),
            pen=pg.mkPen(color=(0, 0, 255), width=1)
        )
        
        # Draw selected indicator
        if self.current_indicator != 'None':
            if self.current_indicator == 'MA':
                # Calculate and plot MA
                ma_period = 20
                ma = df['close'].rolling(window=ma_period).mean()
                self.indicator_chart.plot(
                    timestamps, ma.values, 
                    pen=pg.mkPen(color=(255, 165, 0), width=2), 
                    name=f'MA({ma_period})'
                )
                # Also add to candle chart
                self.candle_chart.plot(
                    timestamps, ma.values, 
                    pen=pg.mkPen(color=(255, 165, 0), width=2)
                )
                
            elif self.current_indicator == 'EMA':
                # Calculate and plot EMA
                ema_period = 20
                ema = df['close'].ewm(span=ema_period, adjust=False).mean()
                self.indicator_chart.plot(
                    timestamps, ema.values, 
                    pen=pg.mkPen(color=(255, 0, 255), width=2), 
                    name=f'EMA({ema_period})'
                )
                # Also add to candle chart
                self.candle_chart.plot(
                    timestamps, ema.values, 
                    pen=pg.mkPen(color=(255, 0, 255), width=2)
                )
                
            elif self.current_indicator == 'MACD':
                # Calculate MACD
                ema12 = df['close'].ewm(span=12, adjust=False).mean()
                ema26 = df['close'].ewm(span=26, adjust=False).mean()
                macd = ema12 - ema26
                signal = macd.ewm(span=9, adjust=False).mean()
                histogram = macd - signal
                
                # Plot MACD and signal line
                self.indicator_chart.plot(
                    timestamps, macd.values, 
                    pen=pg.mkPen(color=(0, 0, 255), width=2), 
                    name='MACD'
                )
                self.indicator_chart.plot(
                    timestamps, signal.values, 
                    pen=pg.mkPen(color=(255, 165, 0), width=2), 
                    name='Signal'
                )
                
                # Plot histogram
                for i in range(len(df)):
                    if i >= len(timestamps):
                        continue
                        
                    t = timestamps[i]
                    h = histogram.iloc[i]
                    color = 'g' if h >= 0 else 'r'
                    
                    bar = pg.BarGraphItem(
                        x=[t], height=[abs(h)], width=0.7, 
                        brush=color, pen=color
                    )
                    if h >= 0:
                        bar.setPos(0, 0)
                    else:
                        bar.setPos(0, h)
                    self.indicator_chart.addItem(bar)
                    
            elif self.current_indicator == 'RSI':
                # Calculate RSI if available in dataframe
                if 'rsi' in df.columns:
                    rsi = df['rsi']
                else:
                    # Calculate RSI
                    delta = df['close'].diff()
                    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
                    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
                    rs = gain / loss.replace(0, np.finfo(float).eps)
                    rsi = 100 - (100 / (1 + rs))
                    
                # Plot RSI
                self.indicator_chart.plot(
                    timestamps, rsi.values, 
                    pen=pg.mkPen(color=(75, 0, 130), width=2), 
                    name='RSI'
                )
                
                # Add overbought/oversold lines
                self.indicator_chart.addLine(y=70, pen=pg.mkPen(color=(200, 0, 0), width=1, style=Qt.DashLine))
                self.indicator_chart.addLine(y=30, pen=pg.mkPen(color=(0, 200, 0), width=1, style=Qt.DashLine))
                
            elif self.current_indicator == 'Bollinger Bands':
                # Calculate Bollinger Bands
                window = 20
                std_dev = 2
                
                if 'sma_20' in df.columns:
                    middle = df['sma_20']
                else:
                    middle = df['close'].rolling(window=window).mean()
                    
                if 'bollinger_upper' in df.columns and 'bollinger_lower' in df.columns:
                    upper = df['bollinger_upper']
                    lower = df['bollinger_lower']
                else:
                    std = df['close'].rolling(window=window).std()
                    upper = middle + (std * std_dev)
                    lower = middle - (std * std_dev)
                
                # Plot Bollinger Bands
                self.indicator_chart.plot(
                    timestamps, middle.values, 
                    pen=pg.mkPen(color=(0, 0, 255), width=2), 
                    name='Middle Band'
                )
                self.indicator_chart.plot(
                    timestamps, upper.values, 
                    pen=pg.mkPen(color=(255, 0, 0), width=1), 
                    name='Upper Band'
                )
                self.indicator_chart.plot(
                    timestamps, lower.values, 
                    pen=pg.mkPen(color=(0, 255, 0), width=1), 
                    name='Lower Band'
                )
                
                # Also add to candle chart
                self.candle_chart.plot(
                    timestamps, middle.values, 
                    pen=pg.mkPen(color=(0, 0, 255), width=2)
                )
                self.candle_chart.plot(
                    timestamps, upper.values, 
                    pen=pg.mkPen(color=(255, 0, 0), width=1)
                )
                self.candle_chart.plot(
                    timestamps, lower.values, 
                    pen=pg.mkPen(color=(0, 255, 0), width=1)
                )
        
        # Draw AI predictions if available and enabled
        if self.show_predictions and self.current_symbol in self.predictions:
            symbol_pred = self.predictions[self.current_symbol]
            
            # Add price prediction if available
            if 'ai_predictions' in symbol_pred and 'order_flow_prediction' in symbol_pred['ai_predictions']:
                # Get prediction and convert to scaled visual indicator
                prediction = symbol_pred['ai_predictions']['order_flow_prediction']
                
                latest_price = df['close'].iloc[-1]
                # Simple scaling for visualization: +/- 2% range based on prediction
                predicted_range = latest_price * 0.02 * prediction
                predicted_price = latest_price + predicted_range
                
                # Add prediction point to line chart
                prediction_point = self.line_chart.plot(
                    [timestamps[-1] + 3600], [predicted_price],  # Add 1h to timestamp
                    pen=None, symbol='o', symbolSize=10,
                    symbolBrush=(255, 0, 0) if prediction < 0 else (0, 255, 0)
                )
                self.prediction_plots.append(prediction_point)
                
                # Add predicted direction arrow
                arrow_color = (0, 255, 0) if prediction > 0 else (255, 0, 0)
                arrow = pg.ArrowItem(
                    angle=90 if prediction > 0 else -90,
                    pen=pg.mkPen(color=arrow_color, width=2),
                    brush=arrow_color
                )
                arrow.setPos(timestamps[-1], latest_price)
                self.candle_chart.addItem(arrow)
                self.prediction_plots.append(arrow)

class PositionTable(QTableWidget):
    """Table widget for displaying active positions"""
    
    close_position_requested = pyqtSignal(str, float)
    modify_position_requested = pyqtSignal(str, float, float)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        # Set up table properties
        self.setColumnCount(7)
        self.setHorizontalHeaderLabels([
            "Symbol", "Side", "Entry Price", "Current Price", 
            "Size", "PnL", "Actions"
        ])
        
        # Configure table appearance
        self.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.verticalHeader().setVisible(False)
        self.setSelectionBehavior(QTableWidget.SelectRows)
        self.setEditTriggers(QTableWidget.NoEditTriggers)
        self.setAlternatingRowColors(True)
        
        # Connect signals
        self.itemClicked.connect(self.handle_item_click)
        
    def update_positions(self, positions):
        """Update the table with current positions"""
        self.clearContents()
        
        if positions is None:
            self.setRowCount(0)
            return
            
        # Convert to DataFrame if it's a dict
        if isinstance(positions, dict):
            positions_list = []
            for symbol, pos in positions.items():
                pos['symbol'] = symbol
                positions_list.append(pos)
            positions = pd.DataFrame(positions_list)
            
        # Set row count
        self.setRowCount(len(positions))
        
        # Fill table
        for i, (_, position) in enumerate(positions.iterrows()):
            # Symbol
            self.setItem(i, 0, QTableWidgetItem(position['symbol']))
            
            # Side
            side_item = QTableWidgetItem(position['side'])
            side_item.setForeground(
                QColor(0, 150, 0) if position['side'].lower() == 'buy' else QColor(200, 0, 0)
            )
            self.setItem(i, 1, side_item)
            
            # Entry price
            self.setItem(i, 2, QTableWidgetItem(f"{position['entry_price']:.4f}"))
            
            # Current price
            current_price = position.get('current_price', position['entry_price'])
            self.setItem(i, 3, QTableWidgetItem(f"{current_price:.4f}"))
            
            # Size
            self.setItem(i, 4, QTableWidgetItem(f"{position['position_size']:.6f}"))
            
            # PnL
            pnl = position.get('unrealized_pnl', 0)
            pnl_pct = position.get('unrealized_pnl_pct', 0)
            pnl_item = QTableWidgetItem(f"${pnl:.2f} ({pnl_pct:.2f}%)")
            pnl_item.setForeground(
                QColor(0, 150, 0) if pnl >= 0 else QColor(200, 0, 0)
            )
            self.setItem(i, 5, pnl_item)
            
            # Actions
            actions_widget = QWidget()
            actions_layout = QHBoxLayout(actions_widget)
            actions_layout.setContentsMargins(2, 2, 2, 2)
            
            # Close button
            close_btn = QPushButton("Close")
            close_btn.setProperty("symbol", position['symbol'])
            close_btn.setProperty("price", current_price)
            close_btn.clicked.connect(self.close_position)
            actions_layout.addWidget(close_btn)
            
            # Modify button
            modify_btn = QPushButton("Modify")
            modify_btn.setProperty("symbol", position['symbol'])
            modify_btn.setProperty("position", position)
            modify_btn.clicked.connect(self.modify_position)
            actions_layout.addWidget(modify_btn)
            
            self.setCellWidget(i, 6, actions_widget)
            
    def close_position(self):
        """Handle position close button click"""
        button = self.sender()
        symbol = button.property("symbol")
        price = button.property("price")
        
        # Confirm close
        reply = QMessageBox.question(
            self, "Close Position", 
            f"Are you sure you want to close {symbol} position?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )
        
        if reply == QMessageBox.Yes:
            self.close_position_requested.emit(symbol, price)
            
    def modify_position(self):
        """Handle position modify button click"""
        button = self.sender()
        symbol = button.property("symbol")
        position = button.property("position")
        
        # Show modification dialog
        from PyQt5.QtWidgets import QDialog, QFormLayout, QDialogButtonBox, QVBoxLayout
        
        dialog = QDialog(self)
        dialog.setWindowTitle(f"Modify {symbol} Position")
        
        layout = QVBoxLayout(dialog)
        form = QFormLayout()
        
        # Stop loss
        stop_loss = QDoubleSpinBox()
        stop_loss.setDecimals(4)
        stop_loss.setRange(0, 1000000)
        current_sl = position.get('stop_loss', 0)
        stop_loss.setValue(current_sl)
        form.addRow("Stop Loss:", stop_loss)
        
        # Take profit
        take_profit = QDoubleSpinBox()
        take_profit.setDecimals(4)
        take_profit.setRange(0, 1000000)
        current_tp = position.get('take_profit', 0)
        take_profit.setValue(current_tp)
        form.addRow("Take Profit:", take_profit)
        
        layout.addLayout(form)
        
        # Dialog buttons
        button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        button_box.accepted.connect(dialog.accept)
        button_box.rejected.connect(dialog.reject)
        layout.addWidget(button_box)
        
        if dialog.exec_():
            # Get values
            new_sl = stop_loss.value()
            new_tp = take_profit.value()
            
            self.modify_position_requested.emit(symbol, new_sl, new_tp)
            
    def handle_item_click(self, item):
        """Handle click on table item"""
        # This could be used to show detailed position information
        pass

class TradeTable(QTableWidget):
    """Table widget for displaying recent trades"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        # Set up table properties
        self.setColumnCount(8)
        self.setHorizontalHeaderLabels([
            "Time", "Symbol", "Side", "Price", "Size", 
            "PnL", "Status", "Strategy"
        ])
        
        # Configure table appearance
        self.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.verticalHeader().setVisible(False)
        self.setSelectionBehavior(QTableWidget.SelectRows)
        self.setEditTriggers(QTableWidget.NoEditTriggers)
        self.setAlternatingRowColors(True)
        
    def update_trades(self, trades):
        """Update the table with recent trades"""
        self.clearContents()
        
        if trades is None:
            self.setRowCount(0)
            return
            
        # Convert to DataFrame if it's a list of dicts
        if isinstance(trades, list):
            trades = pd.DataFrame(trades)
            
        # Set row count
        self.setRowCount(len(trades))
        
        # Fill table
        for i, (_, trade) in enumerate(trades.iterrows()):
            # Time
            time_str = trade.get('time', trade.get('timestamp', ''))
            if isinstance(time_str, (pd.Timestamp, datetime.datetime)):
                time_str = time_str.strftime('%Y-%m-%d %H:%M:%S')
            self.setItem(i, 0, QTableWidgetItem(str(time_str)))
            
            # Symbol
            self.setItem(i, 1, QTableWidgetItem(trade['symbol']))
            
            # Side
            side_item = QTableWidgetItem(trade['action'])
            side_item.setForeground(
                QColor(0, 150, 0) if trade['action'].lower() == 'buy' else QColor(200, 0, 0)
            )
            self.setItem(i, 2, side_item)
            
            # Price
            self.setItem(i, 3, QTableWidgetItem(f"{trade['price']:.4f}"))
            
            # Size
            self.setItem(i, 4, QTableWidgetItem(f"{trade['amount']:.6f}"))
            
            # PnL
            pnl = trade.get('pnl', 0)
            pnl_item = QTableWidgetItem(f"${pnl:.2f}")
            pnl_item.setForeground(
                QColor(0, 150, 0) if pnl >= 0 else QColor(200, 0, 0)
            )
            self.setItem(i, 5, pnl_item)
            
            # Status
            status_item = QTableWidgetItem(trade.get('status', ''))
            if trade.get('status', '').lower() == 'closed':
                status_item.setForeground(QColor(0, 0, 200))
            elif trade.get('status', '').lower() == 'open':
                status_item.setForeground(QColor(0, 150, 0))
            self.setItem(i, 6, status_item)
            
            # Strategy
            self.setItem(i, 7, QTableWidgetItem(trade.get('strategy', '')))

class MetricsPanel(QWidget):
    """Panel for displaying key performance metrics"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        # Create layout
        layout = QVBoxLayout(self)
        
        # Portfolio value section
        portfolio_group = QGroupBox("Portfolio")
        portfolio_layout = QFormLayout(portfolio_group)
        
        self.total_value_label = QLabel("$0.00")
        self.total_value_label.setStyleSheet("font-size: 20px; font-weight: bold;")
        portfolio_layout.addRow("Total Value:", self.total_value_label)
        
        self.cash_value_label = QLabel("$0.00")
        portfolio_layout.addRow("Cash Value:", self.cash_value_label)
        
        self.crypto_value_label = QLabel("$0.00")
        portfolio_layout.addRow("Crypto Value:", self.crypto_value_label)
        
        self.daily_pnl_label = QLabel("$0.00 (0.00%)")
        portfolio_layout.addRow("Daily P&L:", self.daily_pnl_label)
        
        layout.addWidget(portfolio_group)
        
        # Risk metrics section
        risk_group = QGroupBox("Risk Metrics")
        risk_layout = QFormLayout(risk_group)
        
        self.var_label = QLabel("$0.00")
        risk_layout.addRow("Value at Risk (95%):", self.var_label)
        
        self.drawdown_label = QLabel("0.00% / 0.00%")
        risk_layout.addRow("Current/Max Drawdown:", self.drawdown_label)
        
        self.sharpe_label = QLabel("0.00")
        risk_layout.addRow("Sharpe Ratio:", self.sharpe_label)
        
        self.win_rate_label = QLabel("0.00%")
        risk_layout.addRow("Win Rate:", self.win_rate_label)
        
        layout.addWidget(risk_group)
        
        # Performance metrics section
        perf_group = QGroupBox("Performance")
        perf_layout = QFormLayout(perf_group)
        
        self.total_trades_label = QLabel("0")
        perf_layout.addRow("Total Trades:", self.total_trades_label)
        
        self.profit_factor_label = QLabel("0.00")
        perf_layout.addRow("Profit Factor:", self.profit_factor_label)
        
        self.avg_win_label = QLabel("$0.00")
        perf_layout.addRow("Avg. Win:", self.avg_win_label)
        
        self.avg_loss_label = QLabel("$0.00")
        perf_layout.addRow("Avg. Loss:", self.avg_loss_label)
        
        layout.addWidget(perf_group)
        
        # Add stretch to fill any remaining space
        layout.addStretch()
        
    def update_metrics(self, portfolio=None, metrics=None, risk=None):
        """Update displayed metrics"""
        # Update portfolio value
        if portfolio:
            self.total_value_label.setText(f"${portfolio.get('total_value', 0):.2f}")
            self.cash_value_label.setText(f"${portfolio.get('cash_value', 0):.2f}")
            self.crypto_value_label.setText(f"${portfolio.get('crypto_value', 0):.2f}")
            
            # Daily PnL
            daily_pnl = portfolio.get('daily_pnl', 0)
            daily_pnl_pct = portfolio.get('daily_pnl_pct', 0)
            self.daily_pnl_label.setText(f"${daily_pnl:.2f} ({daily_pnl_pct:.2f}%)")
            self.daily_pnl_label.setStyleSheet(
                "color: green;" if daily_pnl >= 0 else "color: red;"
            )
            
        # Update risk metrics
        if risk:
            # VaR
            var = risk.get('var', 0)
            self.var_label.setText(f"${var:.2f}")
            
            # Drawdown
            current_dd = risk.get('current_drawdown', 0) * 100
            max_dd = risk.get('max_drawdown', 0) * 100
            self.drawdown_label.setText(f"{current_dd:.2f}% / {max_dd:.2f}%")
            self.drawdown_label.setStyleSheet("color: red;")
            
        # Update performance metrics
        if metrics:
            # Sharpe ratio
            sharpe = metrics.get('sharpe_ratio', 0)
            self.sharpe_label.setText(f"{sharpe:.2f}")
            
            # Win rate
            win_rate = metrics.get('win_rate', 0) * 100
            self.win_rate_label.setText(f"{win_rate:.2f}%")
            
            # Total trades
            total_trades = metrics.get('total_trades', 0)
            self.total_trades_label.setText(f"{total_trades}")
            
            # Profit factor
            profit_factor = metrics.get('profit_factor', 0)
            self.profit_factor_label.setText(f"{profit_factor:.2f}")
            
            # Average win/loss
            avg_win = metrics.get('avg_win', 0)
            avg_loss = metrics.get('avg_loss', 0)
            self.avg_win_label.setText(f"${avg_win:.2f}")
            self.avg_loss_label.setText(f"${avg_loss:.2f}")

class PredictionPanel(QWidget):
    """Panel for displaying AI predictions and signals"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        # Create layout
        layout = QVBoxLayout(self)
        
        # Create symbol selector
        self.symbol_selector = QComboBox()
        self.symbol_selector.currentIndexChanged.connect(self.update_prediction_display)
        layout.addWidget(QLabel("Symbol:"))
        layout.addWidget(self.symbol_selector)
        
        # Create prediction display
        self.predictions_text = QTextEdit()
        self.predictions_text.setReadOnly(True)
        layout.addWidget(QLabel("AI Predictions:"))
        layout.addWidget(self.predictions_text)
        
        # Store predictions data
        self.predictions = {}
        
    def set_symbols(self, symbols):
        """Set available symbols"""
        current = self.symbol_selector.currentText()
        
        self.symbol_selector.clear()
        self.symbol_selector.addItems(symbols)
        
        # Try to restore previous selection
        index = self.symbol_selector.findText(current)
        if index >= 0:
            self.symbol_selector.setCurrentIndex(index)
        
    def update_predictions(self, predictions):
        """Update with new predictions data"""
        self.predictions = predictions or {}
        self.update_prediction_display()
        
    def update_prediction_display(self):
        """Update prediction display for selected symbol"""
        symbol = self.symbol_selector.currentText()
        if not symbol or symbol not in self.predictions:
            self.predictions_text.setText("No predictions available")
            return
            
        # Get predictions for symbol
        symbol_pred = self.predictions[symbol]
        
        # Format prediction text
        prediction_text = f"<h3>AI Predictions for {symbol}</h3>"
        
        if 'ai_predictions' in symbol_pred:
            ai_pred = symbol_pred['ai_predictions']
            
            # Order flow prediction
            if 'order_flow_prediction' in ai_pred:
                value = ai_pred['order_flow_prediction']
                direction = "Bullish" if value > 0 else "Bearish"
                confidence = abs(value)
                prediction_text += f"<p><b>Order Flow Prediction:</b> {direction} (confidence: {confidence:.2f})</p>"
                
            # Timing quality
            if 'timing_quality' in ai_pred:
                value = ai_pred['timing_quality']
                prediction_text += f"<p><b>Entry Timing Quality:</b> {value:.2f}</p>"
                
        if 'rl_signals' in symbol_pred:
            rl_sig = symbol_pred['rl_signals']
            
            # Trade confidence
            if 'trade' in rl_sig:
                value = rl_sig['trade']
                prediction_text += f"<p><b>Trade Confidence:</b> {value:.2f}</p>"
                
        # Add AI recommendation
        recommendation = self._generate_recommendation(symbol_pred)
        prediction_text += f"<h4>AI Recommendation:</h4>"
        prediction_text += f"<p>{recommendation}</p>"
        
        self.predictions_text.setHtml(prediction_text)
        
    def _generate_recommendation(self, predictions):
        """Generate a human-readable recommendation based on predictions"""
        # Extract signals
        order_flow = predictions.get('ai_predictions', {}).get('order_flow_prediction', 0)
        timing = predictions.get('ai_predictions', {}).get('timing_quality', 0.5)
        confidence = predictions.get('rl_signals', {}).get('trade', 0.5)
        
        # Determine action
        if abs(order_flow) < 0.3:
            action = "Hold/Neutral"
            reason = "market direction is unclear"
        elif order_flow > 0:
            if timing > 0.7 and confidence > 0.6:
                action = "Strong Buy"
                reason = "strong bullish signals with good timing"
            elif timing > 0.5 and confidence > 0.5:
                action = "Buy"
                reason = "positive momentum and reasonable timing"
            else:
                action = "Watch for Buy"
                reason = "positive bias but timing not yet optimal"
        else:  # order_flow < 0
            if timing > 0.7 and confidence > 0.6:
                action = "Strong Sell"
                reason = "strong bearish signals with good timing"
            elif timing > 0.5 and confidence > 0.5:
                action = "Sell"
                reason = "negative momentum and reasonable timing"
            else:
                action = "Watch for Sell"
                reason = "negative bias but timing not yet optimal"
                
        return f"<span style='color: {'green' if 'Buy' in action else 'red' if 'Sell' in action else 'gray'};'><b>{action}</b></span> - {reason.capitalize()}."

class ActionPanel(QWidget):
    """Panel with quick action buttons"""
    
    place_order_requested = pyqtSignal(str, str, float, float)
    close_all_requested = pyqtSignal()
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        # Create layout
        layout = QGridLayout(self)
        
        # Create symbol selector
        self.symbol_selector = QComboBox()
        layout.addWidget(QLabel("Symbol:"), 0, 0)
        layout.addWidget(self.symbol_selector, 0, 1, 1, 2)
        
        # Create buy/sell buttons
        self.buy_btn = QPushButton("Buy")
        self.buy_btn.setStyleSheet("background-color: #d6f5d6;")
        self.buy_btn.clicked.connect(self.place_buy_order)
        layout.addWidget(self.buy_btn, 1, 0)
        
        self.sell_btn = QPushButton("Sell")
        self.sell_btn.setStyleSheet("background-color: #f5d6d6;")
        self.sell_btn.clicked.connect(self.place_sell_order)
        layout.addWidget(self.sell_btn, 1, 1)
        
        # Create amount input
        self.amount_input = QDoubleSpinBox()
        self.amount_input.setRange(0.0001, 1000000)
        self.amount_input.setDecimals(6)
        self.amount_input.setValue(0.01)
        layout.addWidget(QLabel("Amount:"), 2, 0)
        layout.addWidget(self.amount_input, 2, 1, 1, 2)
        
        # Create price input
        self.price_input = QDoubleSpinBox()
        self.price_input.setRange(0.0001, 1000000)
        self.price_input.setDecimals(4)
        self.price_input.setValue(0)
        layout.addWidget(QLabel("Price:"), 3, 0)
        layout.addWidget(self.price_input, 3, 1, 1, 2)
        
        # Market order checkbox
        self.market_order_cb = QCheckBox("Market Order")
        self.market_order_cb.setChecked(True)
        self.market_order_cb.stateChanged.connect(self.toggle_market_order)
        layout.addWidget(self.market_order_cb, 4, 0, 1, 3)
        
        # Emergency buttons
        self.emergency_group = QGroupBox("Emergency Actions")
        emergency_layout = QVBoxLayout(self.emergency_group)
        
        self.close_all_btn = QPushButton("Close All Positions")
        self.close_all_btn.setStyleSheet("background-color: #ff9999;")
        self.close_all_btn.clicked.connect(self.confirm_close_all)
        emergency_layout.addWidget(self.close_all_btn)
        
        self.cancel_all_btn = QPushButton("Cancel All Orders")
        self.cancel_all_btn.setStyleSheet("background-color: #ffcc99;")
        self.cancel_all_btn.clicked.connect(self.cancel_all_orders)
        emergency_layout.addWidget(self.cancel_all_btn)
        
        layout.addWidget(self.emergency_group, 5, 0, 1, 3)
        
        # Add stretch to fill remaining space
        layout.setRowStretch(6, 1)
        
        # Initialize
        self.toggle_market_order(self.market_order_cb.isChecked())
        
    def set_symbols(self, symbols):
        """Set available symbols"""
        self.symbol_selector.clear()
        self.symbol_selector.addItems(symbols)
        
    def update_price(self, symbol, price):
        """Update price for a symbol"""
        if symbol == self.symbol_selector.currentText():
            self.price_input.setValue(price)
            
    def toggle_market_order(self, checked):
        """Toggle between market and limit order"""
        self.price_input.setEnabled(not checked)
        
    def place_buy_order(self):
        """Handle buy button click"""
        symbol = self.symbol_selector.currentText()
        amount = self.amount_input.value()
        price = self.price_input.value() if not self.market_order_cb.isChecked() else 0
        
        self.place_order_requested.emit(symbol, "buy", amount, price)
        
    def place_sell_order(self):
        """Handle sell button click"""
        symbol = self.symbol_selector.currentText()
        amount = self.amount_input.value()
        price = self.price_input.value() if not self.market_order_cb.isChecked() else 0
        
        self.place_order_requested.emit(symbol, "sell", amount, price)
        
    def confirm_close_all(self):
        """Confirm before closing all positions"""
        reply = QMessageBox.question(
            self, "Close All Positions", 
            "Are you sure you want to close ALL positions?\nThis cannot be undone.",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )
        
        if reply == QMessageBox.Yes:
            self.close_all_requested.emit()
            
    def cancel_all_orders(self):
        """Handle cancel all orders button click"""
        reply = QMessageBox.question(
            self, "Cancel All Orders", 
            "Are you sure you want to cancel ALL open orders?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )
        
        if reply == QMessageBox.Yes:
            # This would call the trading system's cancel_all_orders method
            pass

class TradingDashboard(QWidget):
    """
    Main trading dashboard widget that integrates all components.
    
    This dashboard provides a comprehensive view of the trading system
    with real-time updates and interactive controls.
    """
    
    def __init__(self, trading_system, parent=None):
        super().__init__(parent)
        
        # Store reference to trading system
        self.trading_system = trading_system
        
        # Initial state variables
        self.symbols = []
        self.init_symbols()
        
        # Create UI
        self.setup_ui()
        
        # Create refresh worker thread
        self.refresh_worker = None
        self.start_refresh_worker()
        
        # Start periodic refresh timer
        self.refresh_timer = QTimer(self)
        self.refresh_timer.timeout.connect(self.refresh_data)
        self.refresh_timer.start(30000)  # Refresh every 30 seconds
        
        # Initial data refresh
        self.refresh_data()
        
    def init_symbols(self):
        """Initialize tradable symbols"""
        try:
            # Get symbols from trading system
            if hasattr(self.trading_system, 'config') and 'trading' in self.trading_system.config:
                self.symbols = self.trading_system.config['trading'].get('symbols', [])
            else:
                # Default symbols
                self.symbols = ['BTC/USDT', 'ETH/USDT', 'XRP/USDT', 'ADA/USDT', 'SOL/USDT']
        except:
            # Fallback symbols
            self.symbols = ['BTC/USDT', 'ETH/USDT', 'XRP/USDT', 'ADA/USDT', 'SOL/USDT']
            
    def setup_ui(self):
        """Set up the dashboard UI"""
        # Main layout
        main_layout = QVBoxLayout(self)
        
        # Portfolio summary section
        portfolio_layout = QHBoxLayout()
        
        # Portfolio chart
        self.portfolio_chart = PerformanceChart()
        portfolio_layout.addWidget(self.portfolio_chart, 2)
        
        # Key metrics panel
        self.metrics_panel = MetricsPanel()
        portfolio_layout.addWidget(self.metrics_panel, 1)
        
        main_layout.addLayout(portfolio_layout)
        
        # Main content splitter
        self.main_splitter = QSplitter(Qt.Horizontal)
        
        # Left side: Market charts
        self.market_chart = MarketChart()
        self.market_chart.set_symbols(self.symbols)
        self.main_splitter.addWidget(self.market_chart)
        
        # Right side: Trading panels
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        
        # Action panel
        self.action_panel = ActionPanel()
        self.action_panel.set_symbols(self.symbols)
        self.action_panel.place_order_requested.connect(self.place_order)
        self.action_panel.close_all_requested.connect(self.close_all_positions)
        right_layout.addWidget(self.action_panel)
        
        # AI predictions panel
        self.prediction_panel = PredictionPanel()
        self.prediction_panel.set_symbols(self.symbols)
        right_layout.addWidget(self.prediction_panel)
        
        self.main_splitter.addWidget(right_widget)
        
        # Set initial splitter sizes
        self.main_splitter.setSizes([700, 300])
        
        main_layout.addWidget(self.main_splitter)
        
        # Lower section: Positions and trades
        self.lower_tabs = QTabWidget()
        
        # Positions tab
        self.position_table = PositionTable()
        self.position_table.close_position_requested.connect(self.close_position)
        self.position_table.modify_position_requested.connect(self.modify_position)
        self.lower_tabs.addTab(self.position_table, "Active Positions")
        
        # Trades tab
        self.trade_table = TradeTable()
        self.lower_tabs.addTab(self.trade_table, "Recent Trades")
        
        main_layout.addWidget(self.lower_tabs)
        
        # Status bar
        status_layout = QHBoxLayout()
        
        self.status_label = QLabel("Ready")
        status_layout.addWidget(self.status_label)
        
        self.last_update_label = QLabel("Last update: Never")
        status_layout.addWidget(self.last_update_label, alignment=Qt.AlignRight)
        
        main_layout.addLayout(status_layout)
        
    def start_refresh_worker(self):
        """Start the refresh worker thread"""
        if self.refresh_worker and self.refresh_worker.isRunning():
            self.refresh_worker.running = False
            self.refresh_worker.quit()
            self.refresh_worker.wait()
            
        self.refresh_worker = RefreshWorker(
            self.trading_system, symbols=self.symbols, timeframe='1h'
        )
        self.refresh_worker.data_ready.connect(self.update_dashboard)
        self.refresh_worker.error_occurred.connect(self.handle_refresh_error)
        
    def refresh_data(self):
        """Trigger data refresh"""
        if self.refresh_worker and not self.refresh_worker.isRunning():
            self.refresh_worker.start()
            self.status_label.setText("Refreshing data...")
        
    def update_dashboard(self, data):
        """Update dashboard with new data"""
        try:
            # Update portfolio chart
            if 'portfolio' is not None and 'db' in self.trading_system.__dict__:
                try:
                    portfolio_history = self.trading_system.get_portfolio_history(days=30)
                    if not portfolio_history.empty:
                        self.portfolio_chart.update_chart(portfolio_history)
                except Exception as e:
                    logging.error(f"Error updating portfolio chart: {e}")
            
            # Update metrics panel
            self.metrics_panel.update_metrics(
                portfolio=data.get('portfolio'),
                metrics=data.get('metrics'),
                risk=data.get('risk')
            )
            
            # Update market charts
            self.market_chart.update_data(
                data.get('market_data', {}),
                predictions=data.get('predictions')
            )
            
            # Update AI predictions
            self.prediction_panel.update_predictions(data.get('predictions'))
            
            # Update position table
            self.position_table.update_positions(data.get('positions'))
            
            # Update trade table
            self.trade_table.update_trades(data.get('trades'))
            
            # Update action panel prices
            if 'market_data' in data:
                for symbol, market_data in data['market_data'].items():
                    if market_data is not None and not market_data.empty:
                        latest_price = market_data['close'].iloc[-1]
                        self.action_panel.update_price(symbol, latest_price)
            
            # Update status
            self.status_label.setText("Ready")
            self.last_update_label.setText(f"Last update: {datetime.datetime.now().strftime('%H:%M:%S')}")
            
        except Exception as e:
            logging.error(f"Error updating dashboard: {e}")
            self.status_label.setText(f"Error updating dashboard: {str(e)}")
            
    def handle_refresh_error(self, error_message):
        """Handle errors from refresh worker"""
        self.status_label.setText(f"Refresh error: {error_message}")
        logging.error(f"Refresh error: {error_message}")
        
    def place_order(self, symbol, side, amount, price):
        """Place a new order"""
        try:
            order_type = "market" if price == 0 else "limit"
            
            # Call trading system to place order
            order_result = self.trading_system.place_order(
                symbol, side, amount, price, order_type
            )
            
            if order_result and 'id' in order_result:
                QMessageBox.information(
                    self, "Order Placed", 
                    f"{side.capitalize()} order placed successfully.\nOrder ID: {order_result['id']}"
                )
                
                # Refresh data to show new position/order
                self.refresh_data()
            else:
                QMessageBox.warning(
                    self, "Order Failed", 
                    f"Failed to place order. {order_result.get('error', '')}"
                )
                
        except Exception as e:
            logging.error(f"Error placing order: {e}")
            QMessageBox.critical(self, "Order Error", f"Error placing order: {str(e)}")
            
    def close_position(self, symbol, price):
        """Close a specific position"""
        try:
            # Call trading system to close position
            result = self.trading_system.close_position(symbol, price, "user_requested")
            
            if result and result.get('success', False):
                QMessageBox.information(
                    self, "Position Closed", 
                    f"Position for {symbol} closed successfully."
                )
                
                # Refresh data to update positions
                self.refresh_data()
            else:
                QMessageBox.warning(
                    self, "Close Failed", 
                    f"Failed to close position. {result.get('error', '')}"
                )
                
        except Exception as e:
            logging.error(f"Error closing position: {e}")
            QMessageBox.critical(self, "Close Error", f"Error closing position: {str(e)}")
            
    def modify_position(self, symbol, stop_loss, take_profit):
        """Modify position parameters"""
        try:
            # Call trading system to modify position
            result = self.trading_system.modify_position(symbol, stop_loss, take_profit)
            
            if result and result.get('success', False):
                QMessageBox.information(
                    self, "Position Modified", 
                    f"Position for {symbol} modified successfully."
                )
                
                # Refresh data to update positions
                self.refresh_data()
            else:
                QMessageBox.warning(
                    self, "Modify Failed", 
                    f"Failed to modify position. {result.get('error', '')}"
                )
                
        except Exception as e:
            logging.error(f"Error modifying position: {e}")
            QMessageBox.critical(self, "Modify Error", f"Error modifying position: {str(e)}")
            
    def close_all_positions(self):
        """Close all open positions"""
        try:
            # Call trading system to close all positions
            result = self.trading_system.close_all_positions("user_requested")
            
            if result and result.get('success', False):
                closed_count = result.get('closed_count', 0)
                QMessageBox.information(
                    self, "All Positions Closed", 
                    f"Successfully closed {closed_count} position(s)."
                )
                
                # Refresh data to update positions
                self.refresh_data()
            else:
                QMessageBox.warning(
                    self, "Close Failed", 
                    f"Failed to close all positions. {result.get('error', '')}"
                )
                
        except Exception as e:
            logging.error(f"Error closing all positions: {e}")
            QMessageBox.critical(self, "Close Error", f"Error closing all positions: {str(e)}")
            
    def closeEvent(self, event):
        """Handle window close event"""
        # Stop the worker thread
        if self.refresh_worker and self.refresh_worker.isRunning():
            self.refresh_worker.running = False
            self.refresh_worker.quit()
            self.refresh_worker.wait()
            
        # Stop timer
        self.refresh_timer.stop()
        
        # Accept the event
        event.accept()

# Example usage:
if __name__ == "__main__":
    import sys
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Create a sample trading system instance for development/testing
    # In a real app, this would be passed from main.py
    class SampleTradingSystem:
        def __init__(self):
            self.config = {
                'trading': {
                    'symbols': ['BTC/USDT', 'ETH/USDT', 'XRP/USDT', 'ADA/USDT', 'SOL/USDT']
                }
            }
        
        def get_portfolio_value(self):
            return {
                'total_value': 15000.0,
                'cash_value': 5000.0,
                'crypto_value': 10000.0,
                'daily_pnl': 500.0,
                'daily_pnl_pct': 3.45
            }
            
        def get_active_positions(self):
            return pd.DataFrame([
                {'symbol': 'BTC/USDT', 'side': 'buy', 'entry_price': 45000.0, 'current_price': 46500.0, 
                 'position_size': 0.15, 'unrealized_pnl': 225.0, 'unrealized_pnl_pct': 3.33},
                {'symbol': 'ETH/USDT', 'side': 'buy', 'entry_price': 3000.0, 'current_price': 3200.0, 
                 'position_size': 1.5, 'unrealized_pnl': 300.0, 'unrealized_pnl_pct': 6.67},
                {'symbol': 'XRP/USDT', 'side': 'sell', 'entry_price': 0.55, 'current_price': 0.52, 
                 'position_size': 1000.0, 'unrealized_pnl': 30.0, 'unrealized_pnl_pct': 5.45}
            ])
            
        def get_recent_trades(self, limit=20):
            return pd.DataFrame([
                {'time': '2023-10-01 14:30:00', 'symbol': 'BTC/USDT', 'action': 'buy', 'price': 45000.0, 
                 'amount': 0.15, 'pnl': 0, 'status': 'open', 'strategy': 'trend_following'},
                {'time': '2023-10-01 12:45:00', 'symbol': 'ETH/USDT', 'action': 'buy', 'price': 3000.0, 
                 'amount': 1.5, 'pnl': 0, 'status': 'open', 'strategy': 'mean_reversion'},
                {'time': '2023-10-01 10:15:00', 'symbol': 'XRP/USDT', 'action': 'sell', 'price': 0.55, 
                 'amount': 1000.0, 'pnl': 0, 'status': 'open', 'strategy': 'breakout'},
                {'time': '2023-09-30 16:20:00', 'symbol': 'BTC/USDT', 'action': 'sell', 'price': 44800.0, 
                 'amount': 0.1, 'pnl': 250.0, 'status': 'closed', 'strategy': 'trend_following'}
            ])
            
        def get_market_data(self, symbol, timeframe='1h', limit=100):
            # Generate sample market data
            import numpy as np
            
            # Set starting price based on symbol
            if symbol == 'BTC/USDT':
                start_price = 45000.0
            elif symbol == 'ETH/USDT':
                start_price = 3000.0
            elif symbol == 'XRP/USDT':
                start_price = 0.55
            elif symbol == 'ADA/USDT':
                start_price = 1.20
            elif symbol == 'SOL/USDT':
                start_price = 100.0
            else:
                start_price = 100.0
                
            # Generate random walk prices
            np.random.seed(42)  # For reproducible results
            returns = np.random.normal(0, 0.02, limit)
            prices = start_price * (1 + np.cumsum(returns))
            
            # Generate OHLCV data
            dates = pd.date_range(end=pd.Timestamp.now(), periods=limit, freq=timeframe)
            data = []
            
            for i, price in enumerate(prices):
                high_pct = 1 + abs(np.random.normal(0, 0.01))
                low_pct = 1 - abs(np.random.normal(0, 0.01))
                
                high = price * high_pct
                low = price * low_pct
                
                if i > 0:
                    open_price = prices[i-1]
                else:
                    open_price = price * (1 - np.random.normal(0, 0.005))
                    
                volume = np.random.normal(100, 30) * price
                
                data.append({
                    'timestamp': dates[i],
                    'open': open_price,
                    'high': high,
                    'low': low,
                    'close': price,
                    'volume': volume
                })
                
            df = pd.DataFrame(data)
            df.set_index('timestamp', inplace=True)
            
            # Add some indicators
            df['sma_20'] = df['close'].rolling(window=20).mean()
            df['rsi'] = np.random.uniform(30, 70, len(df))  # Fake RSI
            
            # Add Bollinger Bands
            middle = df['close'].rolling(window=20).mean()
            std = df['close'].rolling(window=20).std()
            df['bollinger_upper'] = middle + (std * 2)
            df['bollinger_lower'] = middle - (std * 2)
            
            return df
            
        def get_performance_metrics(self):
            return {
                'sharpe_ratio': 1.8,
                'win_rate': 0.65,
                'profit_factor': 2.3,
                'total_trades': 125,
                'avg_win': 350.0,
                'avg_loss': -180.0
            }
            
        def get_portfolio_history(self, days=30):
            # Generate sample portfolio history
            dates = pd.date_range(end=pd.Timestamp.now(), periods=days, freq='1d')
            
            # Random walk for portfolio value
            np.random.seed(42)  # For reproducible results
            returns = np.random.normal(0.002, 0.02, days)  # Slightly positive drift
            values = 10000 * (1 + np.cumsum(returns))
            
            data = []
            for i, date in enumerate(dates):
                data.append({
                    'time': date,
                    'total_value': values[i],
                    'cash_value': values[i] * 0.3,  # Arbitrary cash ratio
                    'crypto_value': values[i] * 0.7
                })
                
            df = pd.DataFrame(data)
            df.set_index('time', inplace=True)
            return df
            
        def place_order(self, symbol, side, amount, price, order_type):
            # Simulate placing an order
            order_id = f"order_{symbol.replace('/', '_')}_{side}_{int(time.time())}"
            return {'id': order_id, 'success': True}
            
        def close_position(self, symbol, price, reason):
            # Simulate closing a position
            return {'success': True, 'position': symbol, 'exit_price': price}
            
        def modify_position(self, symbol, stop_loss, take_profit):
            # Simulate modifying a position
            return {'success': True, 'position': symbol}
            
        def close_all_positions(self, reason):
            # Simulate closing all positions
            return {'success': True, 'closed_count': 3}
            
        # Add AI engine to simulate predictions
        class AIEngine:
            def __init__(self):
                pass
                
            def get_ai_signals(self, symbol, market_data):
                # Generate random predictions
                order_flow = np.random.uniform(-1, 1)
                timing = np.random.uniform(0, 1)
                
                return {
                    'order_flow_prediction': order_flow,
                    'timing_quality': timing
                }, {
                    'trade': np.random.uniform(0, 1)
                }
                
        ai_engine = AIEngine()
    
    # Create test application
    app = QApplication(sys.argv)
    
    # Create sample trading system
    trading_system = SampleTradingSystem()
    
    # Create dashboard
    dashboard = TradingDashboard(trading_system)
    dashboard.setWindowTitle("Trading Dashboard")
    dashboard.resize(1200, 800)
    dashboard.show()
    
    sys.exit(app.exec_())
